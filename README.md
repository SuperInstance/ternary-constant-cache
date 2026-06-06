# ternary-constant-cache

Constant cache simulation for ternary GPU kernels — hit-rate tracking, access-pattern analysis, LRU eviction, and optimal cache-size estimation. CPU-side profiling tool; no GPU hardware required.

## Why This Exists

GPU constant caches are small, fast, read-only caches that broadcast uniform data to every thread in a warp. Weight matrices, lookup tables, bias vectors — if every thread reads the same value, constant cache is the fastest path. But the cache is tiny (typically 8–64 KB) and unforgiving: a miss doesn't just stall one thread, it stalls the whole warp.

For ternary kernels, the constant cache holds packed trit data — ~161 trits per 32-byte cache line. Before you write your kernel, you need answers: Will my weights fit? What's the expected hit rate? Is my access pattern sequential or pathological? This crate simulates the cache so you can answer those questions without burning GPU hours.

## The Key Insight

The ternary encoding gives constant cache a surprising density advantage. A 32-byte cache line holds 256 bits, which encodes 161 ternary trits (256 / log₂3 ≈ 161). That's 161 ternary weights per cache line versus 256 binary weights per line in a binary network — but ternary weights have only 3 unique values. A ternary kernel accessing a 10K-trit weight matrix needs only ~63 cache lines. That fits entirely in even a small constant cache with near-100% hit rate after warmup.

The access pattern matters more than the cache size. This crate doesn't just count hits and misses — it *classifies* your access pattern (sequential, strided, or random) and estimates the minimum cache size for a target hit rate. You get actionable data, not just statistics.

## Quick Start

```rust
use ternary_constant_cache::*;

let mut cache = ConstantCache::new(16); // 16 cache lines

// Simulate kernel accesses
for addr in 0..1024u64 {
    cache.access(addr);
}

println!("Hit rate: {:.2}%", cache.hit_rate() * 100.0);
println!("Pattern:  {}", cache.analyze_access_pattern());
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│  ConstantCache (capacity: N lines)              │
│                                                 │
│  ┌──────┐ ┌──────┐ ┌──────┐      ┌──────┐     │
│  │Line 0│ │Line 1│ │Line 2│ ...  │Line N│     │
│  │tag=5 │ │tag=12│ │tag=3 │      │tag=7 │     │
│  └──────┘ └──────┘ └──────┘      └──────┘     │
│                                                 │
│  LRU order: [5] → [12] → [3] → ... → [7]      │
│  Evict from tail (LRU) on miss                  │
│                                                 │
│  Stats: hits, misses, evictions                 │
│         compulsory vs capacity miss breakdown   │
│  History: recent addresses for pattern analysis │
└─────────────────────────────────────────────────┘
```

Three main components:

1. **`ConstantCache`** — The simulator. Set capacity, feed addresses, get hit rates and pattern classification. LRU eviction, compulsory vs. capacity miss tracking.
2. **`CacheSizeEstimator`** — Sweep cache sizes against your access trace. Find the minimum size for a target hit rate. Measure working set.
3. **`MissTracker`** — Independent miss classification: compulsory (first access to a line) vs. capacity (was evicted, reaccessed).

## API Reference

### `ConstantCache`

```rust
let mut cache = ConstantCache::new(16);           // 16-line cache

cache.access(addr);                                // → bool (hit/miss)
cache.hit_rate();                                  // f64 [0, 1]
cache.miss_rate();                                 // f64 [0, 1]
cache.stats();                                     // &CacheStats
cache.reset_stats();                               // clear stats, keep contents
cache.analyze_access_pattern();                    // AccessPattern
cache.preload(addr);                               // warm up a line
cache.flush();                                     // clear all lines
cache.occupied();                                  // current line count
```

### `CacheStats`

```rust
pub struct CacheStats {
    pub total_accesses: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub compulsory_misses: u64,    // first access to a line
    pub capacity_misses: u64,      // reaccess after eviction
}
```

### `AccessPattern`

```rust
pub enum AccessPattern {
    Sequential,                  // stride-1
    Strided { stride: u64 },     // fixed stride > 1
    Random,                      // irregular
    Unknown,                     // not enough history yet
}
```

### `CacheSizeEstimator`

```rust
CacheSizeEstimator::estimate_optimal_size(&addresses, min, max, step)
    // → Vec<(usize, f64)>  — (cache_size, hit_rate) pairs

CacheSizeEstimator::min_size_for_hit_rate(&addresses, target_rate, max_size)
    // → Option<usize>  — smallest cache achieving target

CacheSizeEstimator::working_set_size(&addresses)
    // → usize  — unique cache lines touched
```

### `MissTracker`

```rust
let mut tracker = MissTracker::new();
tracker.record(address, was_hit);   // → MissType { Hit, Compulsory, Capacity }
tracker.compulsory_misses;          // Vec<u64>
tracker.capacity_misses;            // Vec<u64>
tracker.total_misses();
tracker.reset();
```

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `CACHE_LINE_SIZE` | 32 | Bytes per cache line (GPU standard) |
| `TRITS_PER_LINE` | 161 | Ternary trits per line (⌊32 × 8 / log₂3⌋) |

## Real-World Example: Sizing a Weight Cache

```rust
use ternary_constant_cache::*;

// Simulate accessing a 10K-trit weight matrix repeatedly
let weight_trits = 10_000;
let weight_bytes = (weight_trits * 2 + 7) / 8;  // 2 bits per trit, packed
let addresses: Vec<u64> = (0..100)              // 100 iterations
    .flat_map(|_| 0..(weight_bytes as u64))
    .collect();

// Find minimum cache size for 95% hit rate
let min_size = CacheSizeEstimator::min_size_for_hit_rate(&addresses, 0.95, 128);
println!("Need {} cache lines for 95% hit rate", min_size.unwrap());

// What's the working set?
let ws = CacheSizeEstimator::working_set_size(&addresses);
println!("Working set: {} unique lines", ws);

// Sweep all sizes for a full picture
let results = CacheSizeEstimator::estimate_optimal_size(&addresses, 1, 64, 1);
for (size, hr) in &results {
    println!("  {} lines → {:.1}% hit rate", size, hr * 100.0);
}
```

## Ecosystem Connections

| Crate | Relationship |
|-------|-------------|
| `ternary-shared-memory` | Shared memory is the fallback when constant cache misses |
| `ternary-grid-launch` | Launch configs determine how many blocks compete for constant cache |
| `ternary-warp-block` | Warp-level uniform reads are the primary constant cache use case |

## Performance Characteristics

- **Access simulation**: O(1) amortized per access — HashMap lookup, VecDeque LRU update
- **Pattern analysis**: O(H) where H = history length (max 1024)
- **Cache size sweep**: O(S × A) where S = sizes to sweep, A = addresses in trace
- **Memory**: O(C + H) where C = cache capacity, H = max history length
- **Accuracy**: Models a fully-associative LRU cache. Real GPU constant caches are typically 4-way set-associative, so simulated hit rates are an upper bound.

## Open Questions

- **Set-associative modeling**: Current simulation is fully-associative. Real GPU constant caches use set associativity, which produces lower hit rates for certain patterns.
- **Prefetch hints**: Some GPU architectures support constant cache prefetching. Modeling prefetch accuracy would improve sizing estimates.
- **Multi-warp contention**: Multiple warps sharing the constant cache can evict each other's lines. Per-warm simulation would be more accurate.

---

*16 tests · MIT OR Apache-2.0 · Zero dependencies*
