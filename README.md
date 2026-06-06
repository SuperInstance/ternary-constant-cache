# ternary-constant-cache

Constant cache simulation for ternary GPU kernels.

GPU constant caches are small, fast, read-only caches that broadcast data to all threads in a warp simultaneously. For ternary kernels, each 32-byte cache line holds ~161 packed trits — enough for a weight tile, a lookup table, or a set of quantization parameters. This crate simulates that cache so you can profile hit rates, classify access patterns, and find the optimal cache size before burning GPU hours.

The key insight: constant cache is only fast when all threads in a warp access the *same* cache line (broadcast). If threads diverge to different lines, the accesses serialize. This crate helps you determine whether your access pattern benefits from constant cache or should use shared memory instead.

## Why This Exists

Constant cache sits between shared memory (fast, per-block, read-write) and global memory (slow, universal). It's perfect for read-only data that all threads read uniformly — weight matrices in inference, lookup tables for activation functions, quantization parameters.

But the performance story depends entirely on your access pattern:
- **Sequential/strided** → high hit rate → constant cache wins
- **Random/divergent** → thrashing → constant cache is worse than global memory

This crate lets you simulate your access pattern and measure hit rates *without running on hardware*. You get:
1. Hit rate measurement with configurable cache size
2. Access pattern classification (sequential, strided, random)
3. Compulsory vs. capacity miss breakdown
4. Optimal cache size estimation

## Quick Start

```rust
use ternary_constant_cache::*;

// Create a 16-line constant cache (16 × 32 bytes = 512 bytes)
let mut cache = ConstantCache::new(16);

// Simulate kernel accesses (each address is a byte address into ternary data)
for _ in 0..3 {
    for addr in 0..512u64 {
        cache.access(addr);
    }
}

println!("Hit rate: {:.2}%", cache.hit_rate() * 100.0);
println!("Pattern:  {}", cache.analyze_access_pattern());
println!("Misses: {} compulsory, {} capacity",
    cache.stats().compulsory_misses, cache.stats().capacity_misses);
```

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   ConstantCache                           │
│                                                          │
│  access(address) → bool (hit/miss)                       │
│  preload(address) → warm up cache                        │
│  analyze_access_pattern() → Sequential/Strided/Random    │
│  hit_rate() / miss_rate() → f64                          │
│  flush() / reset_stats()                                 │
│                                                          │
│  ┌──────────────────────────────────────┐                │
│  │   Cache Lines (HashMap<tag, Line>)   │                │
│  │   LRU Order (VecDeque<tag>)          │                │
│  │   Stats (hits, misses, evictions)    │                │
│  └──────────────────────────────────────┘                │
├──────────────────────────────────────────────────────────┤
│              CacheSizeEstimator                           │
│  estimate_optimal_size(addrs, min, max, step)            │
│  min_size_for_hit_rate(addrs, target, max)               │
│  working_set_size(addrs) → unique lines                  │
├──────────────────────────────────────────────────────────┤
│                 MissTracker                               │
│  record(address, hit) → MissType                         │
│  compulsory_misses / capacity_misses                     │
└──────────────────────────────────────────────────────────┘
```

### Cache Model

The simulation uses:
- **Set-associative via LRU**: When the cache is full, the least-recently-used line is evicted
- **Tag-based addressing**: `tag = address / CACHE_LINE_SIZE`. Same tag = same line = hit.
- **32-byte cache lines**: Standard GPU constant cache line size
- **~161 trits per line**: `⌊32 × 8 / log₂(3)⌋ = 161` ternary digits per line

### Access Pattern Detection

The cache tracks recent access addresses and classifies the pattern:

```rust
// Sequential: every access is +1 from the previous
for addr in 0..100u64 { cache.access(addr); }
assert_eq!(cache.analyze_access_pattern(), AccessPattern::Sequential);

// Strided: consistent gap between addresses
for i in 0..50u64 { cache.access(i * 4); }
// pattern is Strided { stride: 4 }

// Random: no consistent pattern
cache.access(42); cache.access(7); cache.access(999); cache.access(3);
// pattern is Random
```

Pattern classification needs at least 4 accesses. Before that, it returns `AccessPattern::Unknown`.

## API Reference

### `ConstantCache`

| Method | Description |
|--------|-------------|
| `new(capacity)` | Create cache with N lines |
| `access(address)` | Simulate access, returns true on hit |
| `preload(address)` | Warm up cache without counting as access |
| `analyze_access_pattern()` | Classify recent access pattern |
| `hit_rate()` / `miss_rate()` | Current rates |
| `stats()` | Detailed hit/miss/eviction statistics |
| `flush()` | Clear all cached lines |
| `reset_stats()` | Reset statistics, keep contents |
| `occupied()` | Currently cached lines |
| `capacity()` | Total lines available |

### `CacheStats`

| Field | Description |
|-------|-------------|
| `total_accesses` | Total simulated accesses |
| `hits` / `misses` | Hit and miss counts |
| `evictions` | Lines evicted to make room |
| `compulsory_misses` | First access to a line (unavoidable) |
| `capacity_misses` | Re-access to an evicted line (preventable) |

### `CacheSizeEstimator`

| Method | Description |
|--------|-------------|
| `estimate_optimal_size(addrs, min, max, step)` | Sweep cache sizes, return (size, hit_rate) |
| `min_size_for_hit_rate(addrs, target, max)` | Smallest cache achieving target hit rate |
| `working_set_size(addrs)` | Unique cache lines touched |

### `MissTracker`

| Method | Description |
|--------|-------------|
| `record(address, hit)` | Classify a miss as compulsory or capacity |
| `total_misses()` | Combined miss count |
| `reset()` | Clear tracking state |

### `AccessPattern`

```rust
pub enum AccessPattern {
    Sequential,           // stride-1
    Strided { stride },   // consistent stride > 1
    Random,               // no pattern
    Unknown,              // not enough data
}
```

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `CACHE_LINE_SIZE` | 32 | Bytes per cache line |
| `TRITS_PER_LINE` | 161 | Packed trits per line |

## Real-World Example: Sizing Cache for Ternary Weight Streaming

```rust
use ternary_constant_cache::*;

// Your kernel reads ternary weights sequentially for a convolution layer
// Layer has 3×3×256×256 = 589,824 trits = 36,864 cache lines of 161 trits
let total_trits = 3 * 3 * 256 * 256;
let total_lines = (total_trits + 160) / 161;

// Simulate the access pattern: each warp reads 32 consecutive lines
let addresses: Vec<u64> = (0..1000).flat_map(|_| {
    (0..32).map(|i| (i * 32) as u64) // 32 sequential cache lines
}).collect();

// How much cache do we need?
let working_set = CacheSizeEstimator::working_set_size(&addresses);
println!("Working set: {} cache lines", working_set);

let optimal = CacheSizeEstimator::min_size_for_hit_rate(&addresses, 0.95, 128);
println!("Need {} lines for 95% hit rate", optimal.unwrap_or(0));

// Sweep to find the sweet spot
let results = CacheSizeEstimator::estimate_optimal_size(&addresses, 1, 64, 1);
for (size, rate) in &results {
    println!("Size {:3}: {:.1}% hit rate", size, rate * 100.0);
}
```

## Real-World Example: Detecting Divergent Access

```rust
use ternary_constant_cache::*;

// Simulate divergent warp access (each thread reads a different weight)
let mut cache = ConstantCache::new(8);

// All 32 threads read different cache lines → worst case for constant cache
for i in 0..100u64 {
    for thread in 0..32u64 {
        cache.access(thread * 256 + i); // each thread far apart
    }
}

println!("Pattern: {}", cache.analyze_access_pattern()); // Random
println!("Hit rate: {:.2}%", cache.hit_rate() * 100.0);  // Very low

// Lesson: this access pattern should use shared memory, not constant cache
```

## Ecosystem Connections

- **`ternary-grid-launch`** — Determines cache footprint based on problem size
- **`ternary-shared-memory`** — Alternative to constant cache for non-uniform access patterns
- **`ternary-register-file`** — Data in registers never touches cache; good register allocation reduces cache pressure
- **`ternary-warp-block`** — Warp-wide operations that may benefit from constant-cached lookup tables

## Performance Notes

- **Sequential access hit rate**: Approaches `(cache_lines − 1) / cache_lines` after warmup. With 16 lines, you'll see ~94% hit rate.
- **Random access hit rate**: Approximates `cache_size / working_set`. If your working set is 64 lines and you have 8 cache lines, expect ~12.5%.
- **Optimal sizing**: Use `CacheSizeEstimator::min_size_for_hit_rate` with your target. The function is O(max_size × N) — fast for reasonable values.
- **Trit density**: 161 trits per 32-byte line means constant cache is 5× denser for ternary data than for float32 data (8 floats per line).

## Open Questions

- **Set-associative modeling**: Currently fully associative (any line can go anywhere). Real GPU constant caches may have set-associativity constraints.
- **Warp-level simulation**: Currently models per-address access. A warp-level model would simulate 32 threads accessing simultaneously and check for broadcast conditions.
- **Prefetching**: No prefetch model. Real GPUs may prefetch sequential access patterns, improving hit rates beyond what this simulator predicts.
- **Multi-level cache**: Single level only. Modern GPUs may have L1/L2 hierarchies for constant data.

## License

MIT OR Apache-2.0
