# Changelog

## 0.1.0 — reliability release

First versioned release of this fork of
[leo0o7/hnsw-rs](https://github.com/leo0o7/hnsw-rs). Everything below changed
relative to the unreleased code at `d4232f1`.

### Breaking

- `save` / `load` use **snapshot format version 1** and return `SnapshotError`.
  Unversioned files written by earlier code are rejected with
  `SnapshotError::NotASnapshot`; rebuild those indexes.
- `Hnsw` no longer implements `serde::Serialize` / `Deserialize` (that path
  skipped all validation). Use `save`/`load` or `write_snapshot`/`read_snapshot`.
- The product-quantized index moved behind the off-by-default
  `experimental-pq` feature as `hnsw::pq::FrozenPQHnsw`, outside the v0.1
  compatibility contract. `freeze_with_pq(pq)` now encodes the stored vectors
  itself (the `codes` argument is gone).
- `HnswSearcher`'s required method is now `try_search_with_context`;
  `search_with_context` is provided.
- Invalid vectors and queries (NaN, ±∞, components that overflow the metric)
  now make the panicking methods panic instead of being stored or searched;
  parameters have upper bounds (`M`, `M0` ≤ 4096, `ef_construction` ≤ 65536);
  panic messages are the `HnswError` texts.
- Results with equal distances are ordered by ascending id.

### Added

- `HnswError` and fallible `try_*` counterparts for construction, insertion,
  parallel builds and every search method; `*_slice` variants that check the
  runtime length; `Distance::validate` (overflow bound for `L2Squared`) and
  `Distance::metric_id`.
- Caller-supplied ids: `insert_with_id`, `extend_parallel_with_ids`,
  `build_parallel_with_ids`, `contains_id`, `id_mode`, `IdMode`, with
  duplicate rejection, atomic batch reservation and persistence.
- Crash-safe `save` (temporary file, fsync, atomic rename, directory fsync on
  Unix), validated `load`, `write_snapshot` / `read_snapshot(max_bytes)`,
  `SNAPSHOT_FORMAT_VERSION`. Specification in `docs/SNAPSHOT_FORMAT.md`.
- `ProductQuantizer::fit_seeded` and `Hnsw::freeze_seeded` for reproducible PQ.
- Exported `InsertContext`, `SearchContext`, `MAX_CONNECTIONS`,
  `MAX_EF_CONSTRUCTION`, `DEFAULT_EF_SEARCH`, `check_finite`.
- CI on Linux and Windows (debug and release tests, clippy, rustfmt, MSRV 1.88,
  cargo-deny, rustdoc, benchmark smoke runs, a 90 s fuzz run, packaging with a
  consumer build outside the repository); a cargo-fuzz target and corpus for the
  snapshot loader.

### Fixed

- `Link` equality disagreed with its ordering (`0.0 == -0.0` but ordered,
  `NaN != NaN` but equal, ties unordered).
- Concurrent first inserts into an empty index could commit several unlinked
  nodes, only one of which was reachable.
- Exact duplicate vectors became unreachable (4 of 50 copies found); they are now
  grouped with the node they duplicate and returned together.
- Huge `k`/`ef_search` panicked with a capacity overflow.
- k-means++ initialization tracked chosen vectors by float offset instead of
  vector index, so it could pick the same vector twice.
- `save` truncated the previous snapshot before writing it.
- The library no longer depends on HDF5/CMake (benchmark-only dependencies moved
  to the unpublished `hnsw-bench` crate); the `pq` git submodule became the
  versioned workspace crate `hnsw-pq`, so the library can be packaged.
- The example `bench-config.toml` repeated configurations the runner rejects.

### Known limits

See "Not supported in v0.1" in the README: no delete/update/upsert, no
non-blocking snapshots, no reachability guarantee for every vector of an
approximate graph, PQ is experimental, and the `hnsw` name on crates.io belongs
to another project.
