# pq

> [!NOTE]
> This is mostly a learning project. It implements the parts of product quantization needed by the parent HNSW project.

Product quantization splits a vector into smaller subvectors, trains a codebook for each subvector, and stores each vector as a list of centroid IDs.
The resulting codes use one byte per subquantizer when `k <= 256`.

## What this includes

- [x] const-generic vector dimensions and subquantizer counts
- [x] k-means codebook training with k-means++ initialization
- [x] parallel training of subquantizers
- [x] encode and decode support
- [x] asymmetric distance computation (ADC)
- [x] symmetric distance computation (SDC)
- [x] squared L2 distance

The quantizer is trained once. It does not currently support save/load, incremental training, seeded randomness, or custom distance metrics.

## Usage

```rust
use pq::{sdc_distance, ProductQuantizer};

let data = [
    [0.0, 0.0, 10.0, 10.0],
    [1.0, 1.0, 11.0, 11.0],
    [8.0, 8.0, 20.0, 20.0],
    [9.0, 9.0, 21.0, 21.0],
];

// M = 2 subquantizers, D = 4 dimensions, k = 2 centroids per subquantizer.
let mut pq = ProductQuantizer::<2, 4>::new(2);
pq.fit(&data);

let query = [0.5, 0.5, 10.5, 10.5];
let code = pq.encode(&data[0]);
let decoded = pq.decode(&code);

// ADC: keep the query in full precision and compare it with a code.
let adc_table = pq.adc_table(&query);
let adc = pq.adc_distance(&adc_table, &code);

// SDC: compare two encoded vectors.
let other_code = pq.encode(&data[1]);
let sdc_table = pq.sdc_table();
let sdc = sdc_distance(&sdc_table, &code, &other_code);

println!("decoded: {decoded:?}, ADC: {adc}, SDC: {sdc}");
```

`D` must be divisible by `M`, and `k` must be between 1 and 256.

## How it works

Each vector is split into `M` contiguous subvectors. `fit` trains a separate k-means codebook for each one. Encoding chooses the closest centroid in every codebook and returns a `[u8; M]` code.

ADC precomputes distances from a full-precision query to every centroid, then sums the entries selected by a stored code. SDC precomputes distances between centroids and compares two codes without a full-precision query.

Training samples at most `max(10_000, 256 * k)` input vectors, unless the input is smaller. The codebooks are stored in one flat `Vec<f32>`.

## Tests

```sh
cargo test
```
