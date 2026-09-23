# v0.1 benchmark results

Measured on the v0.1 release candidate (branch `release/v0.1-gate`, library
code identical to what is released) with `hnsw-bench`. Raw reports:
`results/v0.1/*.json`; configs: `configs/v0.1/*.toml`.

## Setup

| | |
| --- | --- |
| CPU | Intel Core i5-8365U, 4 cores / 8 threads, 1.6 GHz base (laptop) |
| Memory | 16 GB |
| OS | Windows 11 Pro 10.0.26200 |
| Toolchain | rustc 1.97.1 (x86_64-pc-windows-msvc, LLVM 22.1.6) |
| Build | `cargo run --release --locked -p hnsw-bench --features hdf5` (opt-level 3; no `target-cpu=native`) |
| Dataset | `sift-128-euclidean.hdf5` from ann-benchmarks.com, SHA-256 `dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984` |
| Metric | squared L2 (`L2Squared`), recall@10 against exact neighbors |
| Graph | `M = 16`, `M0 = 32`, `ef_construction = 128`, seed 42 |
| Queries | first 1000 SIFT queries; 100 warm-up, 900 measured, repeated `query_cycles` times; one search thread, reused search context |
| Memory | `memory_usage_bytes()`: the index's own heap (vectors, links, ids, duplicate lists) including `Vec` capacity slack; excludes the benchmark's copy of the dataset and the allocator's overhead. It is not process RSS. |

Latencies on a laptop are noisy (turbo and thermal limits, background
processes); single runs, no repetitions. These numbers describe this code on
this machine; they are **not** a comparison with other libraries or with the
historical Apple M3 Pro figures, and no speedup is claimed.

Reproduce:

```sh
curl -L https://ann-benchmarks.com/sift-128-euclidean.hdf5 -o data/sift-128-euclidean.hdf5
cargo run --release --locked -p hnsw-bench --features hdf5 -- benchmarks/configs/v0.1/sift-100k.toml
cargo run --release --locked -p hnsw-bench --features hdf5 -- benchmarks/configs/v0.1/sift-1m.toml
cargo run --release --locked -p hnsw-bench --features hdf5 -- benchmarks/configs/v0.1/sift-1m-save.toml
cargo run --release --locked -p hnsw-bench --features hdf5 -- benchmarks/configs/v0.1/sift-1m-load.toml
```

## Small: SIFT-128, first 100,000 base vectors

Ground truth recomputed by brute force for the truncated base set. 4,500 timed
searches per point (900 queries × 5 cycles).

| build | threads | build time | memory (index) | ef_search | recall@10 | QPS (1 thread) | p50 | p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sequential | 1 | 62.2 s | 116.3 MiB | 16 | 0.8668 | 9714 | 0.096 ms | 0.199 ms |
| sequential | 1 | 62.2 s | 116.3 MiB | 32 | 0.9454 | 4714 | 0.210 ms | 0.361 ms |
| sequential | 1 | 62.2 s | 116.3 MiB | 64 | 0.9841 | 2729 | 0.341 ms | 0.633 ms |
| sequential | 1 | 62.2 s | 116.3 MiB | 128 | 0.9938 | 1670 | 0.589 ms | 1.021 ms |
| sequential | 1 | 62.2 s | 116.3 MiB | 256 | 0.9970 | 844 | 1.207 ms | 1.855 ms |
| batched | 8 | 13.2 s | 115.4 MiB | 16 | 0.8703 | 8279 | 0.119 ms | 0.209 ms |
| batched | 8 | 13.2 s | 115.4 MiB | 32 | 0.9489 | 4729 | 0.212 ms | 0.352 ms |
| batched | 8 | 13.2 s | 115.4 MiB | 64 | 0.9857 | 2685 | 0.377 ms | 0.600 ms |
| batched | 8 | 13.2 s | 115.4 MiB | 128 | 0.9972 | 1480 | 0.680 ms | 1.071 ms |
| batched | 8 | 13.2 s | 115.4 MiB | 256 | 0.9997 | 852 | 1.197 ms | 1.909 ms |

## Meaningful: SIFT-128, 1,000,000 base vectors

Ground truth from the dataset. 2,700 timed searches per point (900 × 3).

| build | threads | build time | memory (index) | ef_search | recall@10 | QPS (1 thread) | p50 | p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sequential | 1 | 994.3 s | 1068.2 MiB | 16 | 0.7851 | 4230 | 0.228 ms | 0.445 ms |
| sequential | 1 | 994.3 s | 1068.2 MiB | 32 | 0.8903 | 2801 | 0.342 ms | 0.814 ms |
| sequential | 1 | 994.3 s | 1068.2 MiB | 64 | 0.9593 | 1685 | 0.589 ms | 0.963 ms |
| sequential | 1 | 994.3 s | 1068.2 MiB | 128 | 0.9876 | 947 | 1.065 ms | 1.721 ms |
| sequential | 1 | 994.3 s | 1068.2 MiB | 256 | 0.9964 | 484 | 2.060 ms | 3.597 ms |
| batched | 8 | 260.6 s | 1068.7 MiB | 16 | 0.7893 | 3370 | 0.261 ms | 0.882 ms |
| batched | 8 | 260.6 s | 1068.7 MiB | 32 | 0.8893 | 2243 | 0.386 ms | 1.392 ms |
| batched | 8 | 260.6 s | 1068.7 MiB | 64 | 0.9584 | 1631 | 0.592 ms | 1.273 ms |
| batched | 8 | 260.6 s | 1068.7 MiB | 128 | 0.9869 | 941 | 1.031 ms | 2.215 ms |
| batched | 8 | 260.6 s | 1068.7 MiB | 256 | 0.9964 | 341 | 2.521 ms | 11.406 ms |

Build throughput: 1,006 inserts/s sequential, 3,838 inserts/s batched with 8
threads on 4 physical cores. The batched `ef_search = 256` p99 of 11.4 ms is an
outlier of this single run.

## Snapshot at scale (SIFT1M, batched build)

| step | result |
| --- | --- |
| build (batched, 8 threads) | 248.0 s |
| `save` (write, fsync, atomic rename) | 9.52 s, 703,202,277 bytes |
| `load` (read, both CRC-32s, full structural validation) | 5.40 s |
| recall@10 at `ef_search = 64`, before save → after load | 0.9574 → 0.9574 |
| index memory before save → after load | 1069.0 MiB → 894.8 MiB |

The loaded index holds the same graph with exact-size allocations; the freshly
built one carries `Vec` growth slack, hence the lower memory after load. The
QPS of these single-cycle runs (1224 before, 1637 after) is not a controlled
comparison.

Search and insert latency *during* a save is measured separately in
[`../docs/SNAPSHOT_FORMAT.md`](../docs/SNAPSHOT_FORMAT.md#consistency-and-blocking):
searches are unaffected, inserts wait for the save.
