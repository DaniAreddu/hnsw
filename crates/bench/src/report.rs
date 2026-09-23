use super::{
    BenchConfig, BenchFile, QuantizedConfig,
    dataset::BenchData,
    helpers::{mib, ms},
    measurement::{Metrics, PqBenchData},
};
use serde::Serialize;
use std::{error::Error, fs, num::NonZeroUsize, path::Path, time::Duration};

const BENCHMARK_REPORT_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Serialize)]
struct BenchmarkReport {
    schema_version: u32,
    dataset_path: String,
    dimension: usize,
    base_count: usize,
    query_count: usize,
    query_limit: Option<usize>,
    base_limit: Option<usize>,
    top_k: usize,
    effective_k: usize,
    quantization_setting: Option<&'static str>,
    pq: Option<PqReport>,
    runs: Vec<RunReport>,
}

#[derive(Debug, Serialize)]
struct PqReport {
    quantizers: usize,
    pq_k: usize,
    fit_time_s: f64,
    encode_time_s: f64,
}

#[derive(Debug, Serialize)]
pub(crate) struct RunReport {
    m: usize,
    m0: usize,
    ef_construction: usize,
    ef_search: usize,
    measured_query_count: usize,
    memory_bytes: usize,
    memory_mib: f64,
    recall: f64,
    qps: f64,
    avg_latency_ms: f64,
    p50_ms: f64,
    p90_ms: f64,
    p99_ms: f64,
    max_ms: f64,
    build_time_s: Option<f64>,
    insert_qps: Option<f64>,
    load_time_s: Option<f64>,
    load_path: Option<String>,
    save_time_s: Option<f64>,
    save_path: Option<String>,
    build_mode: Option<&'static str>,
    build_repetition: Option<usize>,
    build_threads: Option<usize>,
    effective_build_threads: Option<usize>,
    pq_oracle_recall: Option<f64>,
}

pub(crate) fn run_entry(
    params: BenchConfig,
    repetition: usize,
    ef_search: usize,
    metrics: &Metrics,
) -> RunReport {
    let was_built = metrics.index.build_time.is_some();
    RunReport {
        m: params.m,
        m0: params.m0,
        ef_construction: params.ef_construction,
        ef_search,
        measured_query_count: metrics.query.query_count,
        memory_bytes: metrics.memory_bytes,
        memory_mib: mib(metrics.memory_bytes),
        recall: metrics.query.recall,
        qps: metrics.query.qps,
        avg_latency_ms: ms(metrics.query.avg_latency),
        p50_ms: ms(metrics.query.p50),
        p90_ms: ms(metrics.query.p90),
        p99_ms: ms(metrics.query.p99),
        max_ms: ms(metrics.query.max_latency),
        build_time_s: duration_seconds(metrics.index.build_time),
        insert_qps: metrics.index.insert_qps,
        load_time_s: duration_seconds(metrics.index.load_time),
        load_path: metrics.index.load_path.clone(),
        save_time_s: duration_seconds(metrics.index.save_time),
        save_path: metrics.index.save_path.clone(),
        build_mode: was_built.then(|| params.build_mode.label()),
        build_repetition: was_built.then_some(repetition),
        build_threads: if was_built {
            params.build_threads.map(NonZeroUsize::get)
        } else {
            None
        },
        effective_build_threads: metrics.index.effective_build_threads,
        pq_oracle_recall: metrics.pq_oracle_recall,
    }
}

pub(crate) fn write_json_report<const DIM: usize, const Q: usize>(
    path: &str,
    config: &BenchFile,
    data: &BenchData<DIM>,
    quantized: Option<QuantizedConfig>,
    pq_data: Option<&PqBenchData<DIM, Q>>,
    runs: Vec<RunReport>,
) -> Result<(), Box<dyn Error>> {
    let report = BenchmarkReport {
        schema_version: BENCHMARK_REPORT_SCHEMA_VERSION,
        dataset_path: config.dataset_path.clone(),
        dimension: DIM,
        base_count: data.base.len(),
        query_count: data.queries.len(),
        query_limit: config.query_limit,
        base_limit: config.base_limit,
        top_k: config.top_k,
        effective_k: data.k,
        quantization_setting: quantized.map(|_| "pq"),
        pq: match (quantized, pq_data) {
            (Some(quantized), Some(pq_data)) => Some(PqReport {
                quantizers: quantized.quantizers,
                pq_k: quantized.pq_k,
                fit_time_s: pq_data.fit_time.as_secs_f64(),
                encode_time_s: pq_data.encode_time.as_secs_f64(),
            }),
            _ => None,
        },
        runs,
    };

    let path = Path::new(path);
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }

    let json = serde_json::to_string_pretty(&report)?;
    fs::write(path, format!("{json}\n"))?;
    Ok(())
}

pub(crate) fn print_header<const DIM: usize, const Q: usize>(
    config: &BenchFile,
    data: &BenchData<DIM>,
    quantized: Option<QuantizedConfig>,
    pq_data: Option<&PqBenchData<DIM, Q>>,
) {
    println!("dataset: {}", config.dataset_path);
    println!("base: {} ({} vectors)", data.base_name, data.base.len());
    println!(
        "queries: {} ({} vectors)",
        data.query_name,
        data.queries.len()
    );
    println!("dimension: {DIM}");
    if let Some(quantized) = quantized {
        println!("quantizers: {}", quantized.quantizers);
        println!("pq k: {}", quantized.pq_k);
        if let Some(pq_data) = pq_data {
            println!("pq fit: {:.3}s", pq_data.fit_time.as_secs_f64());
            println!("pq encode: {:.3}s", pq_data.encode_time.as_secs_f64());
        }
    }
    println!("recall metric: recall@{}", data.k);
    println!(
        "warmup queries: {}",
        config.warmup_count(data.queries.len())
    );
    println!("query cycles: {}", config.query_cycles());
    println!();
}

pub(crate) fn print_metrics(
    params: BenchConfig,
    repetition: usize,
    ef_search: usize,
    k: usize,
    metrics: &Metrics,
    print_index_timings: bool,
) {
    let index = &metrics.index;
    let query = &metrics.query;

    println!(
        "M={} M0={} ef_construction={} ef_search={}",
        params.m, params.m0, params.ef_construction, ef_search
    );
    if print_index_timings {
        if index.build_time.is_some() {
            if let Some(effective_threads) = index.effective_build_threads {
                println!(
                    "  build mode: {} (repetition {}, requested threads {}, effective threads {})",
                    params.build_mode.label(),
                    repetition + 1,
                    params
                        .build_threads
                        .map_or_else(|| "auto".to_owned(), |threads| threads.to_string()),
                    effective_threads,
                );
            } else {
                println!(
                    "  build mode: {} (repetition {})",
                    params.build_mode.label(),
                    repetition + 1,
                );
            }
        }
        if let Some(load_time) = index.load_time {
            if let Some(path) = &index.load_path {
                println!("  load: {:.3}s ({path})", load_time.as_secs_f64());
            } else {
                println!("  load: {:.3}s", load_time.as_secs_f64());
            }
        }
        if let (Some(build_time), Some(insert_qps)) = (index.build_time, index.insert_qps) {
            println!(
                "  build: {:.3}s ({:.0} inserts/s)",
                build_time.as_secs_f64(),
                insert_qps,
            );
        }
        if let Some(save_time) = index.save_time {
            if let Some(path) = &index.save_path {
                println!("  save: {:.3}s ({path})", save_time.as_secs_f64());
            } else {
                println!("  save: {:.3}s", save_time.as_secs_f64());
            }
        }
    }
    println!(
        "  memory: {:.2} MiB ({} bytes)",
        mib(metrics.memory_bytes),
        metrics.memory_bytes,
    );
    println!(
        "  search: recall@{k} {:.4}, {:.1} QPS over {} measured queries",
        query.recall, query.qps, query.query_count,
    );
    if let Some(pq_oracle_recall) = metrics.pq_oracle_recall {
        println!(
            "  diagnostic: pq_bruteforce_oracle_recall@{k} {:.4}, frozen_hnsw_adc_recall@{k} {:.4}",
            pq_oracle_recall, query.recall,
        );
    }
    println!(
        "  latency: avg {:.3} ms, p50 {:.3} ms, p90 {:.3} ms, p99 {:.3} ms, max {:.3} ms",
        ms(query.avg_latency),
        ms(query.p50),
        ms(query.p90),
        ms(query.p99),
        ms(query.max_latency),
    );
    println!();
}

fn duration_seconds(duration: Option<Duration>) -> Option<f64> {
    duration.map(|duration| duration.as_secs_f64())
}
