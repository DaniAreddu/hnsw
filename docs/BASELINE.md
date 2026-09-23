# Baseline (v0.1 reliability work)

Recorded before any v0.1 change, at `d4232f1` (`main` and the newly created
`develop` both pointed there), and again after the CI/baseline PR.

Machine: Intel Core i5-8365U (4 cores / 8 threads), 16 GB RAM, Windows 11 Pro,
rustc 1.97.1 (stable-x86_64-pc-windows-msvc).

## State at `d4232f1`

| Command | Result |
| --- | --- |
| `cargo build --all-targets` (no CMake installed) | **fails**: `hdf5-src` build script: `is cmake not installed?`. The HDF5 dependency belonged to the library package, so every consumer of `hnsw` needed CMake and a from-source HDF5 build. |
| `cargo test -p hnsw -p pq` | ok: 16 + 6 tests, 0 doc tests (debug, ~35–45 s) |
| `cargo fmt --all -- --check` | clean |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean |
| `cargo run --release --bin bench` (default `bench-config.toml`) | **rejected** by `validate_config`: `configs must not contain duplicate graph configurations` |
| CI / branch rules / lockfile | none / none / not committed |

## After the CI/baseline PR

| Command | Result (local) |
| --- | --- |
| `cargo build --workspace` | ok without CMake; `hnsw` depends only on `bincode2`, `pq`, `rand`, `rayon`, `serde` |
| `cargo test --workspace --locked` | ok: hnsw 16, hnsw-bench 6, pq 6 (debug, 77 s wall incl. build) |
| `cargo test --workspace --locked --release` | ok: same tests (36 s wall incl. build) |
| `cargo run --release -p hnsw-bench -- benchmarks/configs/smoke/synthetic-16d.toml` | ok; lowest recall@10 0.93 at ef_search=16 against the 0.88 gate |
| `cargo build --release -p hnsw-bench --features hdf5` (CMake 4.4.3 via `uv tool install cmake`) | ok, 7 min 29 s cold |
| `cargo deny --all-features check` (0.20.2) | advisories ok, bans ok, licenses ok, sources ok |

The CI workflow (`.github/workflows/ci.yml`) runs the same commands on
`ubuntu-24.04` and `windows-2025`. Large datasets (SIFT1M, MNIST) are never
downloaded in CI; `benchmarks/run.sh` remains the manual path for them.
