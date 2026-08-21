//! # ternary-constant-cache
//!
//! Constant cache simulation for ternary GPU kernels.
//!
//! Models a fixed-size read-only cache optimized for ternary (base-3) data access.
//! Provides hit-rate tracking, access-pattern analysis (sequential vs random), LRU
//! eviction, and optimal cache-size estimation. CPU-side tool for kernel profiling.
//!
//! ## When to use this
//!
//! Use this crate when you are authoring or tuning a ternary (base-3) GPU kernel and
//! want to know, *without running on hardware*, whether a read-only working set will
//! benefit from the GPU constant cache. It answers practical sizing questions — e.g.
//! "how many cache lines do I need to reach a 95% hit rate?" — and classifies whether
//! your access pattern is sequential, strided, or divergent/random. For divergent
//! patterns (each thread of a warp landing on a different line) it confirms that the
//! constant cache is the wrong resource and shared memory should be used instead.

// Every item on the public API surface must be documented.
#![deny(missing_docs)]

use std::collections::{HashMap, VecDeque};

/// Cache line size in bytes (typical GPU constant cache line = 32 bytes).
pub const CACHE_LINE_SIZE: usize = 32;

/// Number of trits storable per cache line (32 bytes × 8 bits / log2(3) ≈ 161 trits).
pub const TRITS_PER_LINE: usize = 161;

/// Unique identifier for a cache line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CacheLineId(
    /// The raw tag/index value identifying this line (`address / CACHE_LINE_SIZE`).
    pub u64,
);

/// A single cache line holding ternary data.
#[derive(Debug, Clone)]
pub struct CacheLine {
    /// Identifier (derived from the tag) for this line.
    pub id: CacheLineId,
    /// Address tag this line was loaded from (`address / CACHE_LINE_SIZE`).
    pub tag: u64,
    /// Raw payload bytes (zero-initialized placeholder in this simulation).
    pub data: Vec<u8>,
    /// Whether the line currently holds valid data.
    pub valid: bool,
    /// Number of times this line has been touched.
    pub access_count: u64,
    /// Cycle counter of the most recent touch (used for LRU recency).
    pub last_access_cycle: u64,
}

impl CacheLine {
    /// Create a new cache line with the given id and tag, initially invalid and untouched.
    pub fn new(id: CacheLineId, tag: u64) -> Self {
        Self {
            id,
            tag,
            data: vec![0u8; CACHE_LINE_SIZE],
            valid: false,
            access_count: 0,
            last_access_cycle: 0,
        }
    }

    /// Mark this line as accessed.
    pub fn touch(&mut self, cycle: u64) {
        self.access_count += 1;
        self.last_access_cycle = cycle;
        self.valid = true;
    }
}

/// Access pattern classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPattern {
    /// Sequential stride-1 access.
    Sequential,
    /// Access with a fixed stride > 1.
    Strided {
        /// The constant gap (in addresses) between consecutive accesses (always > 1).
        stride: u64,
    },
    /// Random / irregular access.
    Random,
    /// Unknown (not enough data yet).
    Unknown,
}

impl std::fmt::Display for AccessPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AccessPattern::Sequential => write!(f, "Sequential"),
            AccessPattern::Strided { stride } => write!(f, "Strided(stride={})", stride),
            AccessPattern::Random => write!(f, "Random"),
            AccessPattern::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Cache statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total number of simulated [`ConstantCache::access`] calls.
    pub total_accesses: u64,
    /// Accesses that found their line already resident in the cache.
    pub hits: u64,
    /// Accesses that did not find their line resident (compulsory + capacity).
    pub misses: u64,
    /// Lines removed from the cache to make room for new ones.
    pub evictions: u64,
    /// Misses to a line seen for the very first time (unavoidable cold misses).
    pub compulsory_misses: u64,
    /// Misses to a line that was previously loaded but since evicted (preventable).
    pub capacity_misses: u64,
}

impl CacheStats {
    /// Fraction of all accesses that were hits (`0.0` when there have been no accesses).
    pub fn hit_rate(&self) -> f64 {
        if self.total_accesses == 0 {
            return 0.0;
        }
        self.hits as f64 / self.total_accesses as f64
    }

    /// Fraction of all accesses that were misses (`1.0 - hit_rate`).
    pub fn miss_rate(&self) -> f64 {
        1.0 - self.hit_rate()
    }

    /// Reset every counter back to zero.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Constant cache with LRU eviction policy.
#[derive(Debug, Clone)]
pub struct ConstantCache {
    /// Total number of cache lines.
    capacity: usize,
    /// Cache lines indexed by tag.
    lines: HashMap<u64, CacheLine>,
    /// LRU order: front = most recent, back = least recent.
    lru_order: VecDeque<u64>,
    /// Current simulation cycle.
    cycle: u64,
    /// Statistics.
    stats: CacheStats,
    /// Recent access addresses for pattern analysis.
    access_history: Vec<u64>,
    /// Maximum history length for pattern analysis.
    max_history: usize,
    /// Set of tags ever loaded (for compulsory miss tracking).
    ever_loaded: std::collections::HashSet<u64>,
}

impl ConstantCache {
    /// Create a new constant cache with the given number of lines.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            lines: HashMap::new(),
            lru_order: VecDeque::new(),
            cycle: 0,
            stats: CacheStats::default(),
            access_history: Vec::new(),
            max_history: 1024,
            ever_loaded: std::collections::HashSet::new(),
        }
    }

    /// Number of cache lines.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Current number of occupied lines.
    pub fn occupied(&self) -> usize {
        self.lines.len()
    }

    /// Get cache statistics.
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Reset all statistics while keeping the current cache *contents*.
    ///
    /// This clears the counters, the access-pattern history, and the "ever loaded"
    /// set used to classify compulsory vs. capacity misses, so that the next access
    /// window starts from a clean measurement baseline. Note that lines currently
    /// resident stay resident; re-accessing them is still a hit. Use [`flush`] to
    /// also evict the contents.
    ///
    /// [`flush`]: ConstantCache::flush
    pub fn reset_stats(&mut self) {
        self.stats.reset();
        self.access_history.clear();
        self.ever_loaded.clear();
    }

    /// Compute the tag from a ternary data address (address / line_size).
    pub fn tag_for_address(address: u64) -> u64 {
        address / CACHE_LINE_SIZE as u64
    }

    /// Compute the offset within a cache line.
    pub fn offset_for_address(address: u64) -> usize {
        (address % CACHE_LINE_SIZE as u64) as usize
    }

    /// Access the cache at the given address. Returns true on hit.
    pub fn access(&mut self, address: u64) -> bool {
        self.cycle += 1;
        self.stats.total_accesses += 1;
        self.access_history.push(address);
        if self.access_history.len() > self.max_history {
            self.access_history.remove(0);
        }

        let tag = Self::tag_for_address(address);

        if let Some(line) = self.lines.get_mut(&tag) {
            // Cache hit
            line.touch(self.cycle);
            self.stats.hits += 1;
            // Update LRU: move to front
            self.lru_order.retain(|&t| t != tag);
            self.lru_order.push_front(tag);
            true
        } else {
            // Cache miss
            self.stats.misses += 1;
            if !self.ever_loaded.contains(&tag) {
                self.stats.compulsory_misses += 1;
                self.ever_loaded.insert(tag);
            } else {
                self.stats.capacity_misses += 1;
            }
            self.load_line(tag);
            false
        }
    }

    /// Load a cache line, evicting via LRU if necessary.
    fn load_line(&mut self, tag: u64) {
        // A zero-capacity cache never stores anything; every access is a miss.
        if self.capacity == 0 {
            return;
        }
        while self.lines.len() >= self.capacity {
            // Evict the least-recently-used line.
            match self.lru_order.pop_back() {
                Some(lru_tag) => {
                    self.lines.remove(&lru_tag);
                    self.stats.evictions += 1;
                }
                // Defensive: `lru_order` is empty, so nothing left to evict.
                None => break,
            }
        }
        let line_id = CacheLineId(tag);
        let mut line = CacheLine::new(line_id, tag);
        line.touch(self.cycle);
        self.lines.insert(tag, line);
        self.lru_order.push_front(tag);
    }

    /// Preload a line into the cache (warm up).
    pub fn preload(&mut self, address: u64) {
        let tag = Self::tag_for_address(address);
        if !self.lines.contains_key(&tag) {
            self.load_line(tag);
            self.ever_loaded.insert(tag);
        }
    }

    /// Analyze the access pattern from recent history.
    ///
    /// Classification is based on the *dominant* stride — the most common gap between
    /// consecutive accesses. If a single stride accounts for **more than half** of all
    /// consecutive gaps, the pattern is reported as that stride:
    ///
    /// - dominant stride of `1` → [`AccessPattern::Sequential`]
    /// - dominant stride `> 1` → [`AccessPattern::Strided`]
    /// - otherwise (no clear majority, or a non-positive dominant stride) →
    ///   [`AccessPattern::Random`]
    ///
    /// Using a majority rather than requiring *every* stride to match deliberately
    /// tolerates the single large wrap-around jump that occurs when a kernel loops
    /// over a fixed tile (e.g. `0..N` repeated), which is the most common access shape
    /// for ternary kernels. At least 4 accesses are required; before that,
    /// [`AccessPattern::Unknown`] is returned.
    pub fn analyze_access_pattern(&self) -> AccessPattern {
        if self.access_history.len() < 4 {
            return AccessPattern::Unknown;
        }

        let mut strides: Vec<i64> = Vec::with_capacity(self.access_history.len() - 1);
        for i in 1..self.access_history.len() {
            strides.push(self.access_history[i] as i64 - self.access_history[i - 1] as i64);
        }

        // Tally how often each stride value occurs.
        let mut counts: HashMap<i64, usize> = HashMap::new();
        for &s in &strides {
            *counts.entry(s).or_insert(0) += 1;
        }

        let total = strides.len();
        // `strides` is non-empty whenever the history has >= 4 entries.
        let (dominant, dominant_count) = counts
            .into_iter()
            .max_by_key(|&(_, c)| c)
            .expect("at least one stride when history length >= 4");

        // Require a strict majority: one stride must make up more than half the gaps.
        // At most a single value can satisfy this, so the result is deterministic.
        if dominant_count * 2 > total {
            if dominant == 1 {
                return AccessPattern::Sequential;
            }
            if dominant > 1 {
                return AccessPattern::Strided {
                    stride: dominant as u64,
                };
            }
        }

        AccessPattern::Random
    }

    /// Get the current hit rate.
    pub fn hit_rate(&self) -> f64 {
        self.stats.hit_rate()
    }

    /// Get the current miss rate.
    pub fn miss_rate(&self) -> f64 {
        self.stats.miss_rate()
    }

    /// Flush the entire cache.
    pub fn flush(&mut self) {
        self.lines.clear();
        self.lru_order.clear();
    }
}

/// Optimal cache size estimator.
pub struct CacheSizeEstimator;

impl CacheSizeEstimator {
    /// Simulate accesses with varying cache sizes and return (size, hit_rate) pairs.
    pub fn estimate_optimal_size(
        addresses: &[u64],
        min_size: usize,
        max_size: usize,
        step: usize,
    ) -> Vec<(usize, f64)> {
        let mut results = Vec::new();
        for size in (min_size..=max_size).step_by(step.max(1)) {
            let mut cache = ConstantCache::new(size);
            for &addr in addresses {
                cache.access(addr);
            }
            results.push((size, cache.hit_rate()));
        }
        results
    }

    /// Find the minimum cache size that achieves the target hit rate.
    pub fn min_size_for_hit_rate(
        addresses: &[u64],
        target_hit_rate: f64,
        max_size: usize,
    ) -> Option<usize> {
        for size in 1..=max_size {
            let mut cache = ConstantCache::new(size);
            for &addr in addresses {
                cache.access(addr);
            }
            if cache.hit_rate() >= target_hit_rate {
                return Some(size);
            }
        }
        None
    }

    /// Compute the working set size (unique cache lines touched).
    pub fn working_set_size(addresses: &[u64]) -> usize {
        let tags: std::collections::HashSet<u64> = addresses
            .iter()
            .map(|&a| ConstantCache::tag_for_address(a))
            .collect();
        tags.len()
    }
}

/// Miss tracking utility for detailed miss analysis.
///
/// Use this alongside [`ConstantCache`] (or any other cache model) when you want to
/// keep the *addresses* that caused each kind of miss, not just the counts.
#[derive(Debug, Clone, Default)]
pub struct MissTracker {
    /// Addresses of accesses that were the first-ever touch of their line.
    pub compulsory_misses: Vec<u64>,
    /// Addresses of accesses to a line that had been loaded and since evicted.
    pub capacity_misses: Vec<u64>,
    ever_seen: std::collections::HashSet<u64>,
}

impl MissTracker {
    /// Create a new empty miss tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an access and classify any miss.
    pub fn record(&mut self, address: u64, hit: bool) -> MissType {
        if hit {
            return MissType::Hit;
        }
        let tag = ConstantCache::tag_for_address(address);
        if self.ever_seen.contains(&tag) {
            self.capacity_misses.push(address);
            MissType::Capacity
        } else {
            self.ever_seen.insert(tag);
            self.compulsory_misses.push(address);
            MissType::Compulsory
        }
    }

    /// Total miss count.
    pub fn total_misses(&self) -> usize {
        self.compulsory_misses.len() + self.capacity_misses.len()
    }

    /// Reset tracking state.
    pub fn reset(&mut self) {
        self.compulsory_misses.clear();
        self.capacity_misses.clear();
        self.ever_seen.clear();
    }
}

/// Classification of a cache miss.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissType {
    /// The access was a hit — no miss at all.
    Hit,
    /// First-ever access to the line (a cold, unavoidable miss).
    Compulsory,
    /// Re-access to a line that was previously loaded but since evicted.
    Capacity,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_access_high_hit_rate() {
        let mut cache = ConstantCache::new(16);
        // Sequential access: each line holds 32 addresses, so 16 lines hold 512 addresses
        for _ in 0..3 {
            for addr in 0..512u64 {
                cache.access(addr);
            }
        }
        // After warmup, should have very high hit rate
        assert!(cache.hit_rate() > 0.9, "hit rate was {}", cache.hit_rate());
    }

    #[test]
    fn test_random_access_lower_hit_rate() {
        // Use a deterministic "random" pattern
        let mut rng_state: u64 = 42;
        let mut next_random = || -> u64 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            rng_state >> 33
        };

        let mut cache = ConstantCache::new(4);
        for _ in 0..1000 {
            cache.access(next_random() % 256);
        }
        // Random access should have lower hit rate than sequential
        // with only 4 cache lines for 8 possible lines (256/32=8)
        assert!(cache.hit_rate() < 0.7, "hit rate was {}", cache.hit_rate());
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = ConstantCache::new(2);

        // Load tag 0
        cache.access(0);
        assert_eq!(cache.occupied(), 1);

        // Load tag 1 (different line)
        cache.access(CACHE_LINE_SIZE as u64);
        assert_eq!(cache.occupied(), 2);

        // Load tag 2 — should evict tag 0 (LRU)
        cache.access((CACHE_LINE_SIZE * 2) as u64);
        assert_eq!(cache.occupied(), 2);
        assert_eq!(cache.stats.evictions, 1);

        // Access tag 0 again — should be a miss (was evicted)
        let hit = cache.access(0);
        assert!(!hit);
        assert_eq!(cache.stats.evictions, 2);
    }

    #[test]
    fn test_cache_size_vs_hit_rate() {
        // Round-robin over 4 distinct lines, one access per line per round.
        // Working set = 4 lines; 200 rounds => 800 accesses.
        let addresses: Vec<u64> = (0..200).flat_map(|_| [0u64, 32, 64, 96]).collect();

        let results = CacheSizeEstimator::estimate_optimal_size(&addresses, 1, 6, 1);
        // Hand calculation:
        //  - sizes 1,2,3 (< working set): every access misses (thrashing) => 0.0
        //  - sizes >= 4: 4 compulsory misses in round 1, then all hits => 796/800 = 0.995
        assert_eq!(results.len(), 6);
        for (size, rate) in &results[..3] {
            assert_eq!(*rate, 0.0, "size {} should thrash at exactly 0.0", size);
        }
        for (size, rate) in &results[3..] {
            assert!(
                (rate - 0.995).abs() < 1e-12,
                "size {} should reach 0.995, got {}",
                size,
                rate
            );
        }
        // Hit rate must be monotonic non-decreasing as the cache grows.
        for w in results.windows(2) {
            assert!(
                w[1].1 >= w[0].1,
                "hit rate should not decrease as cache grows"
            );
        }
    }

    #[test]
    fn test_miss_tracking() {
        let mut tracker = MissTracker::new();

        // First access to tag 0: compulsory miss
        assert_eq!(tracker.record(0, false), MissType::Compulsory);

        // First access to tag 1: compulsory miss
        assert_eq!(
            tracker.record(CACHE_LINE_SIZE as u64, false),
            MissType::Compulsory
        );

        // Second access to tag 0: if miss, it's capacity
        assert_eq!(tracker.record(0, false), MissType::Capacity);

        // Hit
        assert_eq!(tracker.record(0, true), MissType::Hit);

        assert_eq!(tracker.compulsory_misses.len(), 2);
        assert_eq!(tracker.capacity_misses.len(), 1);
    }

    #[test]
    fn test_access_pattern_sequential() {
        let mut cache = ConstantCache::new(16);
        for addr in 0..100u64 {
            cache.access(addr);
        }
        assert_eq!(cache.analyze_access_pattern(), AccessPattern::Sequential);
    }

    #[test]
    fn test_access_pattern_strided() {
        let mut cache = ConstantCache::new(16);
        for i in 0..50u64 {
            cache.access(i * 4);
        }
        assert_eq!(
            cache.analyze_access_pattern(),
            AccessPattern::Strided { stride: 4 }
        );
    }

    #[test]
    fn test_access_pattern_random() {
        let mut cache = ConstantCache::new(16);
        // Mixed strides → random
        let addrs = [0, 5, 100, 3, 200, 7, 50, 11];
        for &a in &addrs {
            cache.access(a);
        }
        assert_eq!(cache.analyze_access_pattern(), AccessPattern::Random);
    }

    #[test]
    fn test_access_pattern_unknown_too_few() {
        let cache = ConstantCache::new(4);
        assert_eq!(cache.analyze_access_pattern(), AccessPattern::Unknown);
    }

    #[test]
    fn test_access_pattern_cyclic_tile_is_not_random() {
        // A kernel that loops over a fixed tile (0..N repeated) is the most common
        // ternary access shape. It has a single wrap-around jump per iteration, which
        // must NOT cause it to be misreported as Random.
        let mut cache = ConstantCache::new(16);
        for _ in 0..5 {
            for addr in 0..96u64 {
                cache.access(addr);
            }
        }
        assert_eq!(cache.analyze_access_pattern(), AccessPattern::Sequential);
    }

    #[test]
    fn test_preload() {
        let mut cache = ConstantCache::new(4);
        cache.preload(0);
        cache.preload(CACHE_LINE_SIZE as u64);
        assert_eq!(cache.occupied(), 2);
        // Accessing preloaded lines should be hits
        assert!(cache.access(0));
        assert!(cache.access(CACHE_LINE_SIZE as u64));
    }

    #[test]
    fn test_flush() {
        let mut cache = ConstantCache::new(4);
        cache.access(0);
        cache.access(CACHE_LINE_SIZE as u64);
        assert_eq!(cache.occupied(), 2);
        cache.flush();
        assert_eq!(cache.occupied(), 0);
    }

    #[test]
    fn test_working_set_size() {
        let addrs: Vec<u64> = vec![
            0,
            1,
            2,
            CACHE_LINE_SIZE as u64,
            CACHE_LINE_SIZE as u64 + 1,
            CACHE_LINE_SIZE as u64 * 2,
        ];
        assert_eq!(CacheSizeEstimator::working_set_size(&addrs), 3);
    }

    #[test]
    fn test_min_size_for_hit_rate() {
        // Pattern touching 4 unique lines
        let addresses: Vec<u64> = (0..100)
            .flat_map(|i| {
                let base = (i % 4) as u64 * CACHE_LINE_SIZE as u64;
                (base..base + 10).collect::<Vec<_>>()
            })
            .collect();

        let min_size = CacheSizeEstimator::min_size_for_hit_rate(&addresses, 0.9, 16);
        assert!(min_size.is_some());
        assert!(min_size.unwrap() <= 4);
    }

    #[test]
    fn test_cache_line_touch() {
        let mut line = CacheLine::new(CacheLineId(0), 0);
        assert!(!line.valid);
        assert_eq!(line.access_count, 0);
        line.touch(10);
        assert!(line.valid);
        assert_eq!(line.access_count, 1);
        assert_eq!(line.last_access_cycle, 10);
        line.touch(20);
        assert_eq!(line.access_count, 2);
        assert_eq!(line.last_access_cycle, 20);
    }

    #[test]
    fn test_stats_reset() {
        let mut cache = ConstantCache::new(4);
        cache.access(0);
        cache.access(0);
        assert!(cache.stats().total_accesses >= 2);
        cache.reset_stats();
        assert_eq!(cache.stats().total_accesses, 0);
    }

    #[test]
    fn test_compulsory_vs_capacity_misses() {
        let mut cache = ConstantCache::new(2);
        // Touch 3 different tags with only 2 slots
        cache.access(0); // compulsory miss (tag 0)
        cache.access(CACHE_LINE_SIZE as u64); // compulsory miss (tag 1)
        cache.access((CACHE_LINE_SIZE * 2) as u64); // compulsory miss (tag 2), evicts tag 0
        cache.access(0); // capacity miss (tag 0 was evicted)
        assert_eq!(cache.stats.compulsory_misses, 3);
        assert_eq!(cache.stats.capacity_misses, 1);
    }

    #[test]
    fn test_capacity_zero_never_caches() {
        // A zero-capacity cache must never store anything: every access is a miss.
        let mut cache = ConstantCache::new(0);
        assert_eq!(cache.capacity(), 0);
        assert!(!cache.access(0)); // miss, nothing stored
        assert!(!cache.access(0)); // still a miss
        assert!(!cache.access(32)); // different line, still a miss
        assert_eq!(cache.occupied(), 0);
        assert_eq!(cache.stats().evictions, 0);
        assert_eq!(cache.stats().total_accesses, 3);
        assert_eq!(cache.stats().misses, 3);
        assert_eq!(cache.stats().hits, 0);
    }

    #[test]
    fn test_preload_capacity_zero_is_noop() {
        let mut cache = ConstantCache::new(0);
        cache.preload(0);
        cache.preload(32);
        assert_eq!(cache.occupied(), 0);
        assert!(!cache.access(0)); // nothing preloaded
    }

    #[test]
    fn test_single_entry_cache() {
        let mut cache = ConstantCache::new(1);
        assert!(!cache.access(0)); // tag0 miss
        assert!(cache.access(5)); // same tag (5/32 == 0) -> hit
        assert!(!cache.access(32)); // tag1 miss, evicts tag0
        assert!(!cache.access(0)); // tag0 was evicted -> miss
        assert_eq!(cache.occupied(), 1);
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().evictions, 2);
    }

    #[test]
    fn test_capacity_boundary_exact_fit() {
        // Working set exactly equals capacity: no capacity misses after warmup.
        let mut cache = ConstantCache::new(4);
        for _ in 0..50u64 {
            for tag in 0..4u64 {
                cache.access(tag * CACHE_LINE_SIZE as u64);
            }
        }
        // 200 accesses: round 0 has 4 compulsory misses; rounds 1..50 are all hits.
        // hits = 196, compulsory = 4, capacity = 0 => 196/200 = 0.98.
        assert_eq!(cache.stats().total_accesses, 200);
        assert_eq!(cache.stats().compulsory_misses, 4);
        assert_eq!(cache.stats().capacity_misses, 0);
        assert_eq!(cache.stats().hits, 196);
        assert!((cache.hit_rate() - 0.98).abs() < 1e-12);
    }

    #[test]
    fn test_empty_cache_hit_rate() {
        let cache = ConstantCache::new(8);
        assert_eq!(cache.hit_rate(), 0.0); // no accesses -> 0.0, never NaN/panic
        assert_eq!(cache.miss_rate(), 1.0);
        assert_eq!(cache.occupied(), 0);
        assert_eq!(cache.stats().total_accesses, 0);
        assert_eq!(cache.analyze_access_pattern(), AccessPattern::Unknown);
    }

    #[test]
    fn test_reset_stats_resets_miss_classification() {
        // After reset_stats, the compulsory/capacity split must restart from a clean
        // baseline (ever_loaded is cleared), otherwise a re-access to a previously
        // evicted line would be wrongly counted as a capacity miss.
        let mut cache = ConstantCache::new(1);
        cache.access(0); // tag0 compulsory miss
        cache.access(32); // tag1 compulsory miss, evicts tag0
        cache.access(0); // tag0 capacity miss (was evicted)
        assert_eq!(cache.stats().compulsory_misses, 2);
        assert_eq!(cache.stats().capacity_misses, 1);

        cache.reset_stats(); // clears ever_loaded as well
        assert!(cache.access(0)); // tag0 still resident -> HIT, not a miss
        cache.access(32); // tag1 not resident -> must be COMPULSORY now
        assert_eq!(cache.stats().compulsory_misses, 1);
        assert_eq!(cache.stats().capacity_misses, 0);
    }

    #[test]
    fn test_lru_recency_on_reaccess() {
        // Re-accessing a line must promote its recency, so a *different* line is evicted.
        let mut cache = ConstantCache::new(2);
        cache.access(0); // tag0 -> lru [0]
        cache.access(32); // tag1 -> lru [1, 0]
        cache.access(0); // tag0 HIT -> lru [0, 1] (tag1 is now least recent)
        cache.access(64); // tag2 miss, must evict tag1 (not tag0)
        assert!(cache.access(0)); // tag0 still resident -> HIT
        assert!(!cache.access(32)); // tag1 was evicted -> MISS
    }
}
