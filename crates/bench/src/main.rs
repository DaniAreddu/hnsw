mod dataset;
mod helpers;
mod measurement;
mod report;

use measurement::{precompute_pq, run_benchmark};
use serde::Deserialize;
use std::{env, error::Error, num::NonZeroUsize};

const DEFAULT_CONFIG_PATH: &str = "bench-config.toml";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BenchFile {
    pub(crate) dataset_path: Option<String>,
    pub(crate) synthetic: Option<SyntheticConfig>,
    pub(crate) dimension: usize,
    pub(crate) top_k: usize,
    pub(crate) warmup_queries: usize,
    pub(crate) query_cycles: Option<usize>,
    pub(crate) query_limit: Option<usize>,
    pub(crate) base_limit: Option<usize>,
    pub(crate) load_index_prefix: Option<String>,
    pub(crate) save_index_prefix: Option<String>,
    pub(crate) seed: Option<u64>,
    #[cfg_attr(not(feature = "hdf5"), allow(dead_code))]
    pub(crate) base_datasets: Option<Vec<String>>,
    #[cfg_attr(not(feature = "hdf5"), allow(dead_code))]
    pub(crate) query_datasets: Option<Vec<String>>,
    #[cfg_attr(not(feature = "hdf5"), allow(dead_code))]
    pub(crate) ground_truth_datasets: Option<Vec<String>>,
    pub(crate) ef_searches: Vec<usize>,
    pub(crate) output_json: Option<String>,
    pub(crate) quantized: Option<QuantizedConfig>,
    #[serde(default = "default_build_repetitions")]
    pub(crate) build_repetitions: usize,
    pub(crate) min_recall: Option<f64>,
    pub(crate) configs: Vec<BenchConfig>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum BuildMode {
    #[default]
    Sequential,
    Dynamic,
    Batched,
}

impl BuildMode {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
            Self::Dynamic => "dynamic",
            Self::Batched => "batched",
        }
    }

    pub(crate) fn is_parallel(self) -> bool {
        !matches!(self, Self::Sequential)
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SyntheticConfig {
    pub(crate) base: usize,
    pub(crate) queries: usize,
    pub(crate) seed: u64,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct QuantizedConfig {
    pub(crate) quantizers: usize,
    pub(crate) pq_k: usize,
    #[serde(default)]
    pub(crate) pq_oracle: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct BenchConfig {
    pub(crate) m: usize,
    pub(crate) m0: usize,
    pub(crate) ef_construction: usize,
    #[serde(default)]
    pub(crate) build_mode: BuildMode,
    pub(crate) build_threads: Option<NonZeroUsize>,
}

impl BenchConfig {
    pub(crate) fn index_path(self, prefix: &str, dimension: usize) -> String {
        format!(
            "{prefix}-dim{dimension}-m{}-m0{}-efc{}.bin",
            self.m, self.m0, self.ef_construction
        )
    }
}

impl BenchFile {
    pub(crate) fn dataset_label(&self) -> String {
        match (&self.dataset_path, self.synthetic) {
            (Some(path), _) => path.clone(),
            (None, Some(synthetic)) => format!(
                "synthetic(base={}, queries={}, seed={})",
                synthetic.base, synthetic.queries, synthetic.seed
            ),
            (None, None) => "<none>".to_owned(),
        }
    }

    pub(crate) fn query_cycles(&self) -> usize {
        self.query_cycles.unwrap_or(1).max(1)
    }

    pub(crate) fn warmup_count(&self, query_count: usize) -> usize {
        self.warmup_queries.min(query_count.saturating_sub(1))
    }
}

fn default_build_repetitions() -> usize {
    1
}

fn main() -> Result<(), Box<dyn Error>> {
    let config_path = env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_CONFIG_PATH.to_owned());
    let config: BenchFile = toml::from_str(&std::fs::read_to_string(&config_path)?)?;

    // TODO: find some better way of doing this
    match config.dimension {
        16 => run_dimension::<16>(&config),
        128 => run_dimension::<128>(&config),
        784 => run_dimension::<784>(&config),
        dimension => Err(format!(
            "unsupported dimension {}; add a match arm in crates/bench/src/main.rs",
            dimension
        )
        .into()),
    }
}

fn run_dimension<const DIM: usize>(config: &BenchFile) -> Result<(), Box<dyn Error>> {
    match config.quantized {
        None => run::<DIM, 0>(config),
        Some(quantized) => match quantized.quantizers {
            4 if DIM == 16 => run::<DIM, 4>(config),
            8 if DIM == 16 => run::<DIM, 8>(config),
            32 => run::<DIM, 32>(config),
            64 => run::<DIM, 64>(config),
            128 if DIM == 128 => run::<DIM, 128>(config),
            196 if DIM == 784 => run::<DIM, 196>(config),
            other => Err(unsupported_quantizers(DIM, other).into()),
        },
    }
}

fn unsupported_quantizers(dimension: usize, quantizers: usize) -> String {
    let supported = match dimension {
        16 => "4, 8",
        128 => "32, 64, 128",
        784 => "32, 64, 196",
        _ => "none",
    };
    format!(
        "unsupported quantizers {quantizers} for {dimension}D; supported values are {supported}"
    )
}

fn validate_config(config: &BenchFile) -> Result<(), &'static str> {
    match (&config.dataset_path, config.synthetic) {
        (Some(_), Some(_)) => return Err("set either dataset_path or synthetic, not both"),
        (None, None) => return Err("one of dataset_path or synthetic is required"),
        (None, Some(synthetic)) if synthetic.base == 0 || synthetic.queries == 0 => {
            return Err("synthetic base and queries must be greater than zero");
        }
        _ => {}
    }
    if config
        .min_recall
        .is_some_and(|recall| !(0.0..=1.0).contains(&recall))
    {
        return Err("min_recall must be within 0.0..=1.0");
    }

    if config.configs.is_empty() {
        return Err("configs must contain at least one graph configuration");
    }

    if config.ef_searches.is_empty() || config.ef_searches.contains(&0) {
        return Err("ef_searches must contain at least one positive value");
    }
    for (index, ef_search) in config.ef_searches.iter().enumerate() {
        if config.ef_searches[..index].contains(ef_search) {
            return Err("ef_searches must not contain duplicates");
        }
    }

    if config.build_repetitions == 0 {
        return Err("build_repetitions must be greater than zero");
    }
    if config.build_repetitions > 1
        && (config.load_index_prefix.is_some() || config.save_index_prefix.is_some())
    {
        return Err("build_repetitions cannot be used when loading or saving indexes");
    }

    for (index, params) in config.configs.iter().enumerate() {
        if config.configs[..index].contains(params) {
            return Err("configs must not contain duplicate graph configurations");
        }
        if !params.build_mode.is_parallel() && params.build_threads.is_some() {
            return Err("build_threads requires a dynamic or batched build mode");
        }
        if config.load_index_prefix.is_some() && params.build_mode.is_parallel() {
            return Err("parallel build modes cannot be used when loading an index");
        }
        if config.save_index_prefix.is_some() && params.build_mode == BuildMode::Dynamic {
            return Err("dynamic builds cannot be saved because their id mapping is not persisted");
        }
        if config.save_index_prefix.is_some()
            && config.configs[..index].iter().any(|other| {
                other.m == params.m
                    && other.m0 == params.m0
                    && other.ef_construction == params.ef_construction
            })
        {
            return Err("configs must not write to the same index path");
        }
    }

    Ok(())
}

fn run<const DIM: usize, const Q: usize>(config: &BenchFile) -> Result<(), Box<dyn Error>> {
    validate_config(config)?;
    let quantized = config.quantized;
    let data = dataset::load_bench_data::<DIM>(config)?;
    let pq_data = quantized.map(|quantized| precompute_pq::<DIM, Q>(&data.base, quantized.pq_k));
    report::print_header(config, &data, quantized, pq_data.as_ref());

    let mut runs = config.output_json.as_ref().map(|_| {
        Vec::with_capacity(
            config
                .configs
                .len()
                .saturating_mul(config.build_repetitions)
                .saturating_mul(config.ef_searches.len()),
        )
    });

    let mut lowest_recall: Option<(f64, BenchConfig, usize)> = None;
    for params in config.configs.iter().copied() {
        for repetition in 0..config.build_repetitions {
            let metrics = run_benchmark::<DIM, Q>(
                &data,
                params,
                &config.ef_searches,
                config,
                quantized,
                pq_data.as_ref(),
            )?;
            for (run_index, run) in metrics.into_iter().enumerate() {
                report::print_metrics(
                    params,
                    repetition,
                    run.ef_search,
                    data.k,
                    &run.metrics,
                    run_index == 0,
                );
                if lowest_recall.is_none_or(|(recall, _, _)| run.metrics.query.recall < recall) {
                    lowest_recall = Some((run.metrics.query.recall, params, run.ef_search));
                }
                if let Some(runs) = &mut runs {
                    runs.push(report::run_entry(
                        params,
                        repetition,
                        run.ef_search,
                        &run.metrics,
                    ));
                }
            }
        }
    }

    if let Some(output_json) = config.output_json.as_deref() {
        report::write_json_report(
            output_json,
            config,
            &data,
            quantized,
            pq_data.as_ref(),
            runs.expect("JSON output should initialize report runs"),
        )?;
        println!("wrote benchmark JSON: {output_json}");
    }

    if let (Some(min_recall), Some((recall, params, ef_search))) =
        (config.min_recall, lowest_recall)
        && recall < min_recall
    {
        return Err(format!(
            "recall@{} {recall:.4} for M={} M0={} ef_construction={} ef_search={ef_search} is below min_recall {min_recall:.4}",
            data.k, params.m, params.m0, params.ef_construction
        )
        .into());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{BenchConfig, BenchFile, BuildMode, validate_config};
    use std::num::NonZeroUsize;

    fn config(m: usize) -> BenchConfig {
        BenchConfig {
            m,
            m0: m * 2,
            ef_construction: 200,
            build_mode: BuildMode::Sequential,
            build_threads: None,
        }
    }

    #[test]
    fn accepts_distinct_graph_and_search_configurations() {
        let mut bench = bench_file();
        bench.configs = vec![config(8), config(16)];
        bench.ef_searches = vec![16, 64];
        validate_config(&bench).expect("valid sweep");
    }

    #[test]
    fn rejects_empty_or_zero_search_effort() {
        let mut bench = bench_file();
        bench.configs.clear();
        assert!(validate_config(&bench).is_err());

        bench.configs.push(config(8));
        bench.ef_searches.clear();
        assert!(validate_config(&bench).is_err());

        bench.ef_searches = vec![0];
        assert!(validate_config(&bench).is_err());

        bench.ef_searches = vec![32, 32];
        assert!(validate_config(&bench).is_err());

        bench.ef_searches = vec![32];
        bench.configs.push(config(8));
        assert!(validate_config(&bench).is_err());
    }

    fn bench_file() -> BenchFile {
        toml::from_str(
            r#"
dataset_path = "data/test.hdf5"
dimension = 128
top_k = 10
warmup_queries = 1
ef_searches = [32]

[[configs]]
m = 16
m0 = 32
ef_construction = 128
"#,
        )
        .expect("valid benchmark file")
    }

    #[test]
    fn defaults_to_sequential_construction() {
        let config = bench_file();
        assert_eq!(config.build_repetitions, 1);
        assert_eq!(config.configs[0].build_mode, BuildMode::Sequential);
        assert_eq!(config.configs[0].build_threads, None);
    }

    #[test]
    fn accepts_parallel_build_configurations() {
        let mut config = bench_file();
        let mut dynamic = config.configs[0];
        dynamic.build_mode = BuildMode::Dynamic;
        dynamic.build_threads = NonZeroUsize::new(2);
        let mut batched = dynamic;
        batched.build_mode = BuildMode::Batched;
        config.configs = vec![dynamic, batched];
        config.build_repetitions = 3;

        validate_config(&config).expect("valid build sweep");
        assert_eq!(config.configs[0].build_mode, BuildMode::Dynamic);
        assert_eq!(config.configs[1].build_mode, BuildMode::Batched);
    }

    #[test]
    fn rejects_threads_for_sequential_construction() {
        let mut config = bench_file();
        config.configs[0].build_threads = NonZeroUsize::new(2);
        assert!(validate_config(&config).is_err());
    }
}
