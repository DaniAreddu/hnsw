use super::helpers::{duration_average, percentile, recall_at_k};
use super::{BenchConfig, BenchFile, BuildMode, QuantizedConfig, dataset::BenchData};
use hnsw::{Hnsw, HnswSearcher, L2Squared};
use pq::ProductQuantizer;
use rayon::prelude::*;
use std::{
    error::Error,
    hint::black_box,
    num::NonZeroUsize,
    path::Path,
    time::{Duration, Instant},
};

#[derive(Clone, Debug)]
pub(crate) struct IndexTimings {
    pub(crate) build_time: Option<Duration>,
    pub(crate) insert_qps: Option<f64>,
    pub(crate) load_time: Option<Duration>,
    pub(crate) load_path: Option<String>,
    pub(crate) save_time: Option<Duration>,
    pub(crate) save_path: Option<String>,
    pub(crate) effective_build_threads: Option<usize>,
}

type PreparedIndex<const DIM: usize> = (Hnsw<DIM>, IndexTimings, Option<Vec<usize>>);

#[derive(Debug)]
pub(crate) struct QueryMetrics {
    pub(crate) query_count: usize,
    pub(crate) qps: f64,
    pub(crate) recall: f64,
    pub(crate) avg_latency: Duration,
    pub(crate) p50: Duration,
    pub(crate) p90: Duration,
    pub(crate) p99: Duration,
    pub(crate) max_latency: Duration,
}

#[derive(Debug)]
pub(crate) struct Metrics {
    pub(crate) index: IndexTimings,
    pub(crate) query: QueryMetrics,
    pub(crate) memory_bytes: usize,
    pub(crate) pq_oracle_recall: Option<f64>,
}

pub(crate) struct PqBenchData<const DIM: usize, const Q: usize> {
    pub(crate) pq: ProductQuantizer<Q, DIM>,
    pub(crate) quantized_data: Vec<[u8; Q]>,
    pub(crate) fit_time: Duration,
    pub(crate) encode_time: Duration,
}

pub(crate) fn precompute_pq<const DIM: usize, const Q: usize>(
    base: &[[f32; DIM]],
    pq_k: usize,
) -> PqBenchData<DIM, Q> {
    let mut pq = ProductQuantizer::<Q, DIM>::new(pq_k);

    let fit_start = Instant::now();
    pq.fit(base);
    let fit_time = fit_start.elapsed();

    let encode_start = Instant::now();
    let quantized_data = base.par_iter().map(|vector| pq.encode(vector)).collect();
    let encode_time = encode_start.elapsed();

    PqBenchData {
        pq,
        quantized_data,
        fit_time,
        encode_time,
    }
}

pub(crate) fn run_benchmark<const DIM: usize, const Q: usize>(
    data: &BenchData<DIM>,
    params: BenchConfig,
    ef_searches: &[usize],
    config: &BenchFile,
    quantized: Option<QuantizedConfig>,
    pq_data: Option<&PqBenchData<DIM, Q>>,
) -> Result<Vec<SearchRun>, Box<dyn Error>> {
    let (index, timings, id_mapping) = prepare_index(data, params, config)?;
    let mapped_ground_truth = id_mapping
        .as_deref()
        .map(|mapping| remap_ground_truth(&data.ground_truth, mapping));
    let ground_truth = mapped_ground_truth.as_deref().unwrap_or(&data.ground_truth);

    if let Some(quantized) = quantized {
        let pq_data = pq_data.ok_or("quantized benchmark is missing precomputed PQ data")?;
        let quantized_data = id_mapping.as_deref().map_or_else(
            || pq_data.quantized_data.clone(),
            |mapping| remap_data(&pq_data.quantized_data, mapping),
        );
        let index = index.freeze_with_pq(pq_data.pq.clone(), quantized_data);
        let warmup = config.warmup_count(data.queries.len());
        let pq_oracle_recall = if quantized.pq_oracle {
            let measured_queries = data.queries.len() - warmup;
            let recall_sum: f64 = data
                .queries
                .iter()
                .zip(ground_truth)
                .skip(warmup)
                .map(|(query, expected)| {
                    recall_at_k(expected, &index.brute_force_adc(query, data.k))
                })
                .sum();
            Some(recall_sum / measured_queries as f64)
        } else {
            None
        };

        Ok(measure_search_sweep(
            &index,
            &timings,
            data,
            ground_truth,
            config,
            ef_searches,
            pq_oracle_recall,
        ))
    } else {
        Ok(measure_search_sweep(
            &index,
            &timings,
            data,
            ground_truth,
            config,
            ef_searches,
            None,
        ))
    }
}

fn prepare_index<const DIM: usize>(
    data: &BenchData<DIM>,
    params: BenchConfig,
    config: &BenchFile,
) -> Result<PreparedIndex<DIM>, Box<dyn Error>> {
    let base = &data.base;
    let load_path = config
        .load_index_prefix
        .as_deref()
        .map(|prefix| params.index_path(prefix, DIM));
    let (index, mut timings, id_mapping) = match &load_path {
        Some(path) => {
            let load_start = Instant::now();
            let index = Hnsw::<DIM>::load(path)?;
            let load_time = load_start.elapsed();
            if index.len() != base.len() {
                return Err(format!(
                    "loaded index '{path}' has {} vectors, but benchmark base has {} vectors; use a matching index or remove load_index_prefix",
                    index.len(),
                    base.len()
                )
                .into());
            }
            (
                index,
                IndexTimings {
                    build_time: None,
                    insert_qps: None,
                    load_time: Some(load_time),
                    load_path,
                    save_time: None,
                    save_path: None,
                    effective_build_threads: None,
                },
                None,
            )
        }
        None => {
            let mut index = Hnsw::<DIM>::new_seeded(
                params.m,
                params.m0,
                params.ef_construction,
                config.seed.unwrap_or(42),
                L2Squared,
            );

            let build_start = Instant::now();
            let id_mapping = match params.build_mode {
                BuildMode::Sequential => {
                    let mut insert_ctx = index.insert_context();
                    for &vector in base {
                        index.insert_with_context(vector, &mut insert_ctx);
                    }
                    None
                }
                BuildMode::Dynamic => Some(index.extend_parallel(base, params.build_threads)),
                BuildMode::Batched => {
                    index.build_parallel(base, params.build_threads);
                    None
                }
            };
            let build_time = build_start.elapsed();
            let insert_qps = base.len() as f64 / build_time.as_secs_f64();
            (
                index,
                IndexTimings {
                    build_time: Some(build_time),
                    insert_qps: Some(insert_qps),
                    load_time: None,
                    load_path: None,
                    save_time: None,
                    save_path: None,
                    effective_build_threads: params
                        .build_mode
                        .is_parallel()
                        .then(|| effective_thread_count(params.build_threads)),
                },
                id_mapping,
            )
        }
    };

    if let Some(prefix) = config.save_index_prefix.as_deref() {
        let path = params.index_path(prefix, DIM);
        if let Some(parent) = Path::new(&path)
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)?;
        }
        let save_start = Instant::now();
        index.save(&path)?;
        timings.save_path = Some(path);
        timings.save_time = Some(save_start.elapsed());
    }

    Ok((index, timings, id_mapping))
}

pub(crate) struct SearchRun {
    pub(crate) ef_search: usize,
    pub(crate) metrics: Metrics,
}

fn measure_search_sweep<const DIM: usize, S: HnswSearcher<DIM>>(
    index: &S,
    timings: &IndexTimings,
    data: &BenchData<DIM>,
    ground_truth: &[Vec<usize>],
    config: &BenchFile,
    ef_searches: &[usize],
    pq_oracle_recall: Option<f64>,
) -> Vec<SearchRun> {
    ef_searches
        .iter()
        .map(|&ef_search| SearchRun {
            ef_search,
            metrics: measure_index(
                index,
                timings.clone(),
                data,
                ground_truth,
                config,
                ef_search,
                pq_oracle_recall,
            ),
        })
        .collect()
}

fn measure_index<const DIM: usize, S: HnswSearcher<DIM>>(
    index: &S,
    timings: IndexTimings,
    data: &BenchData<DIM>,
    ground_truth: &[Vec<usize>],
    config: &BenchFile,
    ef_search: usize,
    pq_oracle_recall: Option<f64>,
) -> Metrics {
    let memory_bytes = index.memory_usage_bytes();
    let warmup = config.warmup_count(data.queries.len());
    let mut search_ctx = index.search_context();
    let query = measure_queries(
        &data.queries,
        ground_truth,
        warmup,
        config.query_cycles(),
        |query| index.search_with_context(query, data.k, ef_search, &mut search_ctx),
    );

    Metrics {
        index: timings,
        query,
        memory_bytes,
        pq_oracle_recall,
    }
}

fn remap_ground_truth(ground_truth: &[Vec<usize>], mapping: &[usize]) -> Vec<Vec<usize>> {
    ground_truth
        .iter()
        .map(|neighbors| {
            neighbors
                .iter()
                .map(|&id| {
                    *mapping
                        .get(id)
                        .unwrap_or_else(|| panic!("ground-truth id {id} is out of bounds"))
                })
                .collect()
        })
        .collect()
}

fn remap_data<T: Copy>(data: &[T], mapping: &[usize]) -> Vec<T> {
    assert_eq!(data.len(), mapping.len(), "id mapping length mismatch");
    let mut remapped = data.to_vec();
    for (original_id, &internal_id) in mapping.iter().enumerate() {
        let destination = remapped
            .get_mut(internal_id)
            .unwrap_or_else(|| panic!("internal id {internal_id} is out of bounds"));
        *destination = data[original_id];
    }
    remapped
}

fn effective_thread_count(requested: Option<NonZeroUsize>) -> usize {
    let available =
        std::thread::available_parallelism().expect("unable to get number available of threads");
    requested
        .map_or(available, |requested| requested.min(available))
        .get()
}

fn measure_queries<const DIM: usize>(
    queries: &[[f32; DIM]],
    ground_truth: &[Vec<usize>],
    warmup: usize,
    query_cycles: usize,
    mut search: impl FnMut(&[f32; DIM]) -> Vec<(usize, f32)>,
) -> QueryMetrics {
    for query in queries.iter().take(warmup) {
        let _ = black_box(search(black_box(query)));
    }

    let measured_queries = queries.len() - warmup;
    let total_measured_queries = measured_queries * query_cycles;
    let mut total_search_time = Duration::ZERO;
    let mut recall_sum = 0.0;
    let mut latencies = Vec::with_capacity(total_measured_queries);

    for _ in 0..query_cycles {
        for (query, expected) in queries.iter().zip(ground_truth).skip(warmup) {
            let start = Instant::now();
            let result = search(black_box(query));
            let latency = start.elapsed();

            total_search_time += latency;
            recall_sum += recall_at_k(expected, &result);
            latencies.push(latency);
            black_box(result);
        }
    }

    latencies.sort_unstable();
    QueryMetrics {
        query_count: total_measured_queries,
        qps: total_measured_queries as f64 / total_search_time.as_secs_f64(),
        recall: recall_sum / total_measured_queries as f64,
        avg_latency: duration_average(total_search_time, total_measured_queries),
        p50: percentile(&latencies, 0.50),
        p90: percentile(&latencies, 0.90),
        p99: percentile(&latencies, 0.99),
        max_latency: *latencies.last().unwrap(),
    }
}
