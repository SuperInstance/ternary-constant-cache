# ternary-constant-cache

[![crate](https://img.shields.io/badge/crate-ternary--constant--cache-blue)](https://crates.io)
[![license](https://img.shields.io/badge/license-MIT%2FApache--2.0-green)](./LICENSE)

Constant cache simulation for **ternary GPU kernels** — a CPU-side profiling tool that models fixed-size read-only caches for ternary data with hit-rate tracking, LRU eviction, access-pattern analysis, and optimal cache-size estimation.

## Overview

GPU constant caches are small, fast, read-only caches that serve uniform data to all threads in a warp. For ternary kernels, the cache holds packed trit data with each 32-byte cache line storing ~161 trits. This crate simulates that cache hierarchy to help you:

- **Profile cache behavior** before running on real hardware
- **Classify access patterns** as sequential, strided, or random
- **Size your constant cache** for optimal hit rates
- **Track compulsory vs. capacity misses** independently

## Architecture

```
┌─────────────────────────────────────────────┐
│              ConstantCache                  │
│                                             │
│  ┌──────┐ ┌──────┐ ┌──────┐    ┌──────┐   │
│  │Line 0│ │Line 1│ │Line 2│...│Line N│   │
│  │tag=5 │ │tag=12│ │tag=3 │    │tag=7 │   │
│  └──────┘ └──────┘ └──────┘    └──────┘   │
│                                             │
│  LRU order: [5] → [12] → [3] → ... → [7]  │
│  Evict from tail (LRU) on miss              │
│                                             │
│  Stats: hits, misses, evictions, compulsory │
│         vs capacity miss breakdown          │
└─────────────────────────────────────────────┘
```

## Quick Start

```rust
use ternary_constant_cache::*;

// Create a 16-line constant cache
let mut cache = ConstantCache::new(16);

// Simulate kernel accesses
for addr in 0..1024u64 {
    cache.access(addr);
}

println!("Hit rate: {:.2}%", cache.hit_rate() * 100.0);
println!("Pattern:  {}", cache.analyze_access_pattern());
```

## Access Pattern Analysis

The cache automatically tracks recent access history and classifies the pattern:

```rust
use ternary_constant_cache::*;

let mut cache = ConstantCache::new(16);

// Sequential access → AccessPattern::Sequential
for addr in 0..100u64 { cache.access(addr); }
assert_eq!(cache.analyze_access_pattern(), AccessPattern::Sequential);

// Strided access → AccessPattern::Strided { stride: 4 }
for i in 0..50u64 { cache.access(i * 4); }
// (reset stats first to clear old history)
```

## Cache Size Estimation

Find the optimal cache size for your access pattern:

```rust
use ternary_constant_cache::*;

let addresses: Vec<u64> = (0..200).flat_map(|_| 0..128u64).collect();

// Sweep cache sizes and get hit rates
let results = CacheSizeEstimator::estimate_optimal_size(&addresses, 1, 16, 1);
for (size, hit_rate) in &results {
    println!("Size {}: {:.2}% hit rate", size, hit_rate * 100.0);
}

// Find minimum size for 90% hit rate
let min_size = CacheSizeEstimator::min_size_for_hit_rate(&addresses, 0.9, 32);
println!("Need at least {} lines for 90% hit rate", min_size.unwrap());

// Working set analysis
let working_set = CacheSizeEstimator::working_set_size(&addresses);
println!("Working set: {} unique cache lines", working_set);
```

## Miss Tracking

Detailed miss classification:

```rust
use ternary_constant_cache::*;

let mut tracker = MissTracker::new();

// Record accesses with hit/miss info
tracker.record(0, false);                    // Compulsory miss (first time)
tracker.record(CACHE_LINE_SIZE as u64, false); // Compulsory miss
tracker.record(0, false);                    // Capacity miss (was evicted)
tracker.record(0, true);                     // Hit

assert_eq!(tracker.compulsory_misses.len(), 2);
assert_eq!(tracker.capacity_misses.len(), 1);
```

## Key Types

| Type | Description |
|------|-------------|
| `ConstantCache` | Main cache simulator with LRU eviction |
| `CacheLine` | Individual cache line with tag, data, access tracking |
| `CacheStats` | Cumulative hit/miss/eviction statistics |
| `AccessPattern` | Sequential / Strided / Random classification |
| `CacheSizeEstimator` | Tools for optimal cache sizing |
| `MissTracker` | Compulsory vs capacity miss analysis |

## Constants

- `CACHE_LINE_SIZE = 32` — Bytes per cache line (GPU standard).
- `TRITS_PER_LINE = 161` — Ternary trits per 32-byte line (⌊32 × 8 / log₂(3)⌋).

## Testing

```bash
cargo test
```

16 tests covering sequential/random hit rates, LRU eviction, access pattern detection, cache sizing, miss tracking, preloading, flushing, and working set analysis.

## License

MIT OR Apache-2.0
