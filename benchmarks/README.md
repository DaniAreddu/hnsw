# Benchmark Suite

The benchmark suite contains the configs, structured JSON results, plotting script, and generated figures used in the root README.

## Running the suite

Use the benchmark runner as:

```sh
./benchmarks/run.sh <command>
```

Available commands:

- `build` — build the indexes required by the benchmark
- `measure` — run the measurements and write JSON results
- `plot` — regenerate the figures from the existing JSON results
- `all` — run `build`, `measure`, and `plot` in order

Construction mode selection is available for the build stage:

```sh
./benchmarks/run.sh build sequential
./benchmarks/run.sh build parallel
./benchmarks/run.sh build both
```

`build` defaults to `both`; `all` also runs both construction groups.

The runner sets `RUSTFLAGS="-C target-cpu=native"` unless `RUSTFLAGS` is already defined.

## Benchmark setup

These results were measured on an Apple M3 Pro:

- CPU: Apple M3 Pro
- Memory: 18 GB
- OS: macOS 15.6.1
- Rust: rustc 1.95.0
- Distance: squared L2
- Metric: recall@10
- Warmup queries: 100
- Distinct measured queries: 900
- Query cycles: 100
- Timed searches: 90,000 (900 queries × 100 cycles)

Download the datasets:

```sh
mkdir -p data

curl -L https://ann-benchmarks.com/sift-128-euclidean.hdf5 \
  -o data/sift-128-euclidean.hdf5

curl -L https://ann-benchmarks.com/mnist-784-euclidean.hdf5 \
  -o data/mnist-784-euclidean.hdf5
```

## Running an individual config

The benchmark binary reads a TOML config, builds or loads the requested indexes, and optionally writes a JSON report.

```sh
RUSTFLAGS="-C target-cpu=native" \
  cargo run --release -p hnsw-bench -- benchmarks/configs/path/to/config.toml
```

Existing benchmark configs are organized under:

- `configs/build/`: measures construction and saves the indexes reused by measurement configs.
- `configs/measure/`: loads the saved indexes and measures normal or PQ search performance.

## Config file

The benchmark is configured through a TOML file such as `bench-config.toml`.

The fields describe the dataset and how the benchmark should run:

- `dataset_path`: HDF5 file to read
- `dimension`: vector dimension, currently matched in `src/bin/bench.rs`
- `top_k`: number of expected neighbors
- `warmup_queries`: queries excluded from timing
- `query_limit`: number of queries
- `query_cycles`: number of passes
- `base_limit`: optional number of base vectors to load
- `seed`: optional random seed used during construction
- `load_index_prefix`: path prefix for loading indexes
- `save_index_prefix`: path prefix for saving indexes
- `output_json`: optional JSON report path
- `build_repetitions`: number of repetitions per construction point; defaults to one

The `[[configs]]` entries are the HNSW parameter sets to run.
Each one produces or loads a separate index using the same filename pattern:

```toml
[[configs]]
m = 16
m0 = 32
ef_construction = 128
build_mode = "batched"
build_threads = 8
```

`build_mode` is optional and defaults to `sequential`. It also accepts `dynamic` and `batched`.
`build_threads` optionally selects a worker count for the parallel modes; omitting it uses available parallelism.

The top-level `ef_searches` list defines a query-time sweep.
Each value is measured against every graph configuration.

An optional `[quantized]` table enables PQ search and sets `quantizers`, `pq_k`, and the `pq_oracle` diagnostic.

If `load_index_prefix` is set, the benchmark loads matching index files from disk.  
If `save_index_prefix` is set instead, it builds the index from the base dataset and writes it to disk after construction.  
If `output_json` is set, the runner writes one pretty-printed JSON report after all configured runs complete successfully. Console output remains unchanged.

## Outputs

Benchmark reports are written as JSON under `benchmarks/results/`.
Each report includes its configuration and measurements.

Generated figures are under `benchmarks/plots/` and can be regenerated from the existing reports with:

```sh
./benchmarks/run.sh plot
```
