# hnsw

**Hierarchical Navigable Small World** (HNSW) approximate nearest-neighbor index in Rust over fixed-size `[f32; D]` vectors.

Search walks a layered proximity graph instead of comparing the query with every vector. It is much faster than brute force; recall depends on the build parameters (`M`, `M0`, `ef_construction`) and the per-query effort `ef_search`.

This repository is a fork of [leo0o7/hnsw-rs](https://github.com/leo0o7/hnsw-rs) (MIT). v0.1 is a reliability release: typed errors instead of panics or silent corruption at every input boundary, stable caller-supplied ids, crash-safe versioned snapshots with full validation on load, and CI gates for all of it. See [`CHANGELOG.md`](CHANGELOG.md).

## v0.1 contract

**Supported**

| Operation | API |
| --- | --- |
| Create with validated parameters | `Hnsw::try_new`, `try_new_seeded` (`M` 2–4096, `M0` 1–4096, `ef_construction` 1–65536) |
| Insert with positional ids (0, 1, 2, …) | `try_insert`, `try_insert_with_context`, `try_insert_slice` |
| Insert with caller-supplied `usize` ids | `try_insert_with_id`, `try_insert_with_id_and_context` |
| Parallel build of an empty index / parallel extend | `try_build_parallel(_with_ids)`, `try_extend_parallel(_with_ids)` |
| k-NN search | `HnswSearcher::try_search`, `try_search_with_ef`, `try_search_with_context`, `try_search_slice` |
| Concurrent searches and inserts through `&Hnsw` | all of the above |
| Crash-safe snapshots | `save` / `load`, `write_snapshot` / `read_snapshot` |
| Custom metrics | `Distance<D>` (statically dispatched) |

Every panicking convenience method (`new`, `insert`, `search`, `build_parallel`, …) has a `try_*` counterpart and panics exactly where it would return an error.

**Guarantees**

- **Inputs:** NaN/±∞ components, components large enough to overflow the metric (`L2Squared::component_limit(D)`), wrong runtime lengths, `ef_search = 0`, out-of-range parameters and duplicate ids are rejected with `HnswError`, identically in debug and release builds, and leave the index unchanged. Batches are validated before anything is inserted. `k = 0` and an empty index return no results; `ef_search < k` is widened to `k`.
- **Results** are `(id, distance)` sorted by distance, ties by ascending id.
- **Ids:** an index is positional or explicit, fixed by its first insert; mixing returns `IdModeMismatch`. Explicit ids are reserved before insertion, so concurrent inserts of one id cannot both succeed; batches are all-or-nothing.
- **Determinism:** seeded sequential inserts in a fixed order build the same graph, also across save/load. Parallel builds depend on thread scheduling.
- **Concurrency:** once an insert returns, later searches can find its vector (subject to approximate recall). A search overlapping an insert may or may not see it.
- **Duplicates:** exact copies of a vector are grouped with it and returned together, each under its own id.
- **Snapshots:** format version 1 with magic, version, dimension, metric identity and parameters, graph parameters, sizes and CRC-32s. `save` replaces the file atomically (temporary file, fsync, rename; the previous snapshot survives any failure). `load` bounds allocations by the file size, verifies checksums and validates the whole graph before returning. A snapshot is a single point in time: it **blocks inserts, not searches**, for its duration. Details: [`docs/SNAPSHOT_FORMAT.md`](docs/SNAPSHOT_FORMAT.md).

**Not supported in v0.1**

- delete, update or upsert; filtering; metadata storage
- loading snapshots written before v0.1 (unversioned; rejected with `NotASnapshot`, rebuild them)
- more than 2³² − 1 vectors per snapshot; non-blocking (copy-on-write) snapshots
- a guarantee that every vector of an approximate graph is reachable: backlink pruning can occasionally leave a vector without incoming links, more likely in parallel builds (tests hold parallel builds to ≥ 99.5 % self-retrieval)
- the product-quantized index as a stable API (see *Experimental PQ*)

## Usage

```toml
[dependencies]
hnsw = { git = "https://github.com/DaniAreddu/hnsw", tag = "v0.1.0" }
```

The crate name `hnsw` is already taken on crates.io by an unrelated project, so v0.1 is consumed from git; `cargo publish --dry-run` passes, but publishing needs a different package name.

```rust
use hnsw::{Hnsw, HnswError, HnswSearcher, L2Squared};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // M = 16, M0 = 32, ef_construction = 128, seed = 42
    let index = Hnsw::<2>::try_new_seeded(16, 32, 128, 42, L2Squared)?;
    index.try_insert([0.0, 0.0])?;
    index.try_insert([3.0, 3.0])?;
    index.try_insert([4.0, 4.0])?;

    // two nearest neighbors with ef_search = 32: (id, squared L2 distance)
    let hits = index.try_search_with_ef(&[1.0, 1.0], 2, 32)?;
    assert_eq!(hits, vec![(0, 2.0), (1, 8.0)]);

    // invalid input is an error, never a panic or a corrupted graph
    assert!(matches!(
        index.try_insert([f32::NAN, 1.0]),
        Err(HnswError::NonFiniteComponent { component: 0, .. })
    ));
    Ok(())
}
```

`new_default(M)` is shorthand for `M0 = 2 * M`, `ef_construction = 128` and a random seed. `search(q, k)` uses `ef_search = 32` (`DEFAULT_EF_SEARCH`); pass a different effort per query with `search_with_ef`. For repeated operations, reuse scratch buffers with `search_context()` / `insert_context()` and the `*_with_context` methods.

### Caller-supplied ids and parallel builds

```rust
use hnsw::{Hnsw, HnswError, HnswSearcher, L2Squared};
use std::num::NonZeroUsize;

fn main() -> Result<(), HnswError> {
    let items: Vec<(usize, [f32; 2])> = (0..1_000)
        .map(|i| (100_000 + i, [i as f32, (i % 10) as f32]))
        .collect();
    let mut index = Hnsw::<2>::try_new_seeded(16, 32, 128, 42, L2Squared)?;
    index.try_build_parallel_with_ids(&items, NonZeroUsize::new(4))?;

    assert_eq!(index.try_search_with_ef(&[7.0, 7.0], 1, 64)?, vec![(100_007, 0.0)]);
    assert_eq!(
        index.try_insert_with_id(100_007, [1.0, 1.0]),
        Err(HnswError::DuplicateId { id: 100_007 })
    );
    Ok(())
}
```

### Snapshots

```rust
use hnsw::{Hnsw, HnswSearcher, L2Squared, SnapshotError};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let index = Hnsw::<2>::try_new_seeded(16, 32, 128, 42, L2Squared)?;
    for i in 0..100 {
        index.try_insert([i as f32, 0.0])?;
    }
    let path = std::env::temp_dir().join(format!("readme-{}.hnsw", std::process::id()));
    index.save(&path)?; // atomic replace
    let loaded = Hnsw::<2>::load(&path)?; // validated before it is returned
    assert_eq!(loaded.try_search(&[3.0, 0.5], 1)?, vec![(3, 0.25)]);

    // the dimension and metric are part of the snapshot's identity
    assert!(matches!(
        Hnsw::<3>::load(&path),
        Err(SnapshotError::DimensionMismatch { expected: 3, found: 2 })
    ));
    std::fs::remove_file(path)?;
    Ok(())
}
```

Continuing to insert into a loaded, sequentially built index gives the same graph as never having saved: the seed is stored and the RNG is replayed.

### Custom metrics

`Hnsw<D, DS = L2Squared>` is generic over `Distance<D>`, so distance calls are statically dispatched. A metric must return finite distances (negative values are fine; smaller means closer) for every vector its `validate` accepts, and should override `metric_id` with a stable string so its snapshots stay loadable:

```rust
use hnsw::{Distance, Hnsw, HnswError, HnswSearcher, check_finite};

/// Negative inner product: larger dot products are closer.
#[derive(Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
struct NegDot;

impl<const D: usize> Distance<D> for NegDot {
    fn distance(&self, a: &[f32; D], b: &[f32; D]) -> f32 {
        -a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
    }
    fn validate(&self, v: &[f32; D]) -> Result<(), HnswError> {
        check_finite(v) // plus any bound that keeps the sum finite
    }
    fn metric_id() -> &'static str {
        "example::NegDot"
    }
}

fn main() -> Result<(), HnswError> {
    let index = Hnsw::<2, NegDot>::try_new_seeded(8, 16, 64, 1, NegDot)?;
    index.try_insert([1.0, 0.0])?;
    index.try_insert([3.0, 0.0])?;
    assert_eq!(index.try_search(&[1.0, 0.0], 1)?, vec![(1, -3.0)]);
    Ok(())
}
```

Saving requires `DS: Serialize`, loading `DS: DeserializeOwned`; metric parameters are stored as JSON in the snapshot header.

### Experimental PQ

With `features = ["experimental-pq"]`, `Hnsw<D, L2Squared>::freeze_seeded::<Q>(k, seed)` (or `freeze` / `freeze_with_pq`) converts an index into `hnsw::pq::FrozenPQHnsw`, which keeps the graph and ids but replaces each vector with `Q` product-quantization bytes. It is **outside the v0.1 compatibility contract**: it keeps no original vectors, so there is no exact rescoring (distances are ADC approximations and recall is bounded by the quantization); it cannot be extended; it cannot be saved or loaded. The quantizer lives in the workspace crate [`hnsw-pq`](crates/pq) (vendored from [leo0o7/product-quantization](https://github.com/leo0o7/product-quantization), MIT).

## How it works

Each vector becomes a node. Most nodes live only on layer 0; a few are randomly promoted to sparse upper layers that act as long-range shortcuts. Search starts at the entry point on the top layer, greedily moves closer to the query, and descends; on layer 0 it keeps a frontier of candidates and a bounded set of the best `ef_search` results, stopping when the closest candidate is worse than the worst result. Insertion runs the same search, selects diverse neighbors (the paper's heuristic), links the node and adds backlinks, pruning any neighbor list that exceeds `M`/`M0`.

Dynamic parallel construction inserts concurrently and can extend a non-empty index; batched construction preallocates all nodes of an empty index first and is the faster path when all vectors are available up front.

Vectors are `[f32; D]`, so the dimension is part of the type and storage is flat. Visited sets use epoch markers so an operation never clears a whole array.

## Benchmarks

v0.1 results measured for this release — SIFT-128 100k and 1M, recall@10, p50/p99 latency, build time and memory, with hardware and parameters — are in [`benchmarks/RESULTS-v0.1.md`](benchmarks/RESULTS-v0.1.md). The suite, configs and the figures below are described in [`benchmarks/README.md`](benchmarks/README.md).

The figures below are **historical**: they were produced by the original author on an Apple M3 Pro with the pre-v0.1 code and have not been re-measured.

![Search trade-off](benchmarks/plots/search_tradeoff.svg)
![Construction trade-off](benchmarks/plots/construction_tradeoff.svg)
![Dataset-size scaling](benchmarks/plots/size_scaling.svg)
![Parallel construction](benchmarks/plots/parallel_construction.svg)
![Product-quantization trade-off](benchmarks/plots/pq_tradeoff.svg)

## Development

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace            # also with --release
cargo test -p hnsw                # default features (no PQ)
bash scripts/verify-package.sh    # package, then build a consumer outside the repo
cargo run --release -p hnsw-bench -- benchmarks/configs/smoke/synthetic-16d.toml
cargo +nightly fuzz run load_snapshot fuzz/corpus/load_snapshot   # needs cargo-fuzz
```

CI runs all of these (the fuzzer for 90 s) on every PR into `develop` and `main`; baseline and history in [`docs/BASELINE.md`](docs/BASELINE.md).

## References

- Yu. A. Malkov and D. A. Yashunin, *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*, 2018.
- Redis' HNSW/vector set implementation informed some practical details of construction and pruning.

## License

MIT, see [`LICENSE`](LICENSE) (Copyright (c) 2026 leo0o7). `crates/pq` carries its own MIT license from the upstream project.
