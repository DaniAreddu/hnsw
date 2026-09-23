#[cfg(feature = "hdf5")]
mod hdf5_file;

use crate::{BenchFile, SyntheticConfig, helpers::compute_ground_truth};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::error::Error;

pub(crate) struct BenchData<const DIM: usize> {
    pub(crate) base_name: String,
    pub(crate) query_name: String,
    pub(crate) base: Vec<[f32; DIM]>,
    pub(crate) queries: Vec<[f32; DIM]>,
    pub(crate) ground_truth: Vec<Vec<usize>>,
    pub(crate) k: usize,
}

pub(crate) fn load_bench_data<const DIM: usize>(
    config: &BenchFile,
) -> Result<BenchData<DIM>, Box<dyn Error>> {
    match (&config.dataset_path, config.synthetic) {
        (_, Some(synthetic)) => Ok(synthetic_data(synthetic, config.top_k)),
        (Some(path), None) => load_hdf5(config, path),
        (None, None) => Err("one of dataset_path or synthetic is required".into()),
    }
}

#[cfg(feature = "hdf5")]
fn load_hdf5<const DIM: usize>(
    config: &BenchFile,
    path: &str,
) -> Result<BenchData<DIM>, Box<dyn Error>> {
    hdf5_file::load(config, path)
}

#[cfg(not(feature = "hdf5"))]
fn load_hdf5<const DIM: usize>(
    _config: &BenchFile,
    path: &str,
) -> Result<BenchData<DIM>, Box<dyn Error>> {
    Err(format!(
        "cannot read '{path}': the bench binary was built without the `hdf5` feature; \
         rebuild with `cargo run --release -p hnsw-bench --features hdf5`"
    )
    .into())
}

/// Uniform vectors in `[-1, 1)^DIM` drawn from a seeded RNG, with exact
/// brute-force ground truth. Intended for smoke checks, not for quality claims.
fn synthetic_data<const DIM: usize>(synthetic: SyntheticConfig, top_k: usize) -> BenchData<DIM> {
    let mut rng = StdRng::seed_from_u64(synthetic.seed);
    let mut vectors = |count: usize| -> Vec<[f32; DIM]> {
        (0..count)
            .map(|_| std::array::from_fn(|_| rng.random_range(-1.0..1.0)))
            .collect()
    };
    let base = vectors(synthetic.base);
    let queries = vectors(synthetic.queries);
    let k = top_k.min(base.len());
    let ground_truth = compute_ground_truth(&base, &queries, k);

    BenchData {
        base_name: "synthetic-base".to_owned(),
        query_name: "synthetic-queries".to_owned(),
        base,
        queries,
        ground_truth,
        k,
    }
}
