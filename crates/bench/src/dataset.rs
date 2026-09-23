use super::{BenchFile, helpers::compute_ground_truth};
use hdf5::{Dataset, File};
use std::error::Error;

const DEFAULT_BASE_DATASETS: &[&str] = &["train", "base"];
const DEFAULT_QUERY_DATASETS: &[&str] = &["test", "query", "queries"];
const DEFAULT_GROUND_TRUTH_DATASETS: &[&str] = &["neighbors", "knns", "groundtruth"];

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
    let file = File::open(&config.dataset_path)?;
    let base_dataset_names = dataset_names(config.base_datasets.as_deref(), DEFAULT_BASE_DATASETS);
    let query_dataset_names =
        dataset_names(config.query_datasets.as_deref(), DEFAULT_QUERY_DATASETS);
    let ground_truth_dataset_names = dataset_names(
        config.ground_truth_datasets.as_deref(),
        DEFAULT_GROUND_TRUTH_DATASETS,
    );
    let (base_name, base_dataset) = open_dataset(&file, &base_dataset_names)?;
    let (query_name, query_dataset) = open_dataset(&file, &query_dataset_names)?;

    let base = load_vectors::<DIM>(&base_name, &base_dataset, config.base_limit)?;
    let queries = load_vectors::<DIM>(&query_name, &query_dataset, config.query_limit)?;

    if base.is_empty() {
        return Err("base dataset is empty".into());
    }
    if queries.is_empty() {
        return Err("query dataset is empty".into());
    }

    let k = config.top_k.min(base.len());
    let ground_truth = match open_optional_dataset(&file, &ground_truth_dataset_names) {
        Some((name, dataset)) if config.base_limit.is_none() => {
            println!("using ground truth dataset '{name}'");
            load_ground_truth(&name, &dataset, queries.len(), k)?
        }
        Some((name, _)) => {
            println!(
                "ignoring ground truth dataset '{name}' because base_limit is set; computing exact recall for the truncated base set"
            );
            compute_ground_truth(&base, &queries, k)
        }
        None => {
            println!("no ground truth dataset found; computing exact recall with brute force");
            compute_ground_truth(&base, &queries, k)
        }
    };

    Ok(BenchData {
        base_name,
        query_name,
        base,
        queries,
        ground_truth,
        k,
    })
}

fn dataset_names<'a>(configured: Option<&'a [String]>, defaults: &[&'a str]) -> Vec<&'a str> {
    configured
        .map(|names| names.iter().map(String::as_str).collect())
        .unwrap_or_else(|| defaults.to_vec())
}

pub(crate) fn open_dataset(
    file: &File,
    candidates: &[&str],
) -> Result<(String, Dataset), Box<dyn Error>> {
    open_optional_dataset(file, candidates).ok_or_else(|| {
        format!(
            "could not find any dataset named one of: {}",
            candidates.join(", ")
        )
        .into()
    })
}

pub(crate) fn open_optional_dataset(file: &File, candidates: &[&str]) -> Option<(String, Dataset)> {
    candidates.iter().find_map(|name| {
        file.dataset(name)
            .ok()
            .map(|dataset| ((*name).to_owned(), dataset))
    })
}

pub(crate) fn load_vectors<const D: usize>(
    dataset_name: &str,
    dataset: &Dataset,
    limit: Option<usize>,
) -> Result<Vec<[f32; D]>, Box<dyn Error>> {
    let shape = dataset.shape();
    if shape.len() != 2 {
        return Err(format!("dataset '{dataset_name}' must be 2-D, got shape {shape:?}").into());
    }

    let rows = limit.map_or(shape[0], |value| value.min(shape[0]));
    let dims = shape[1];
    if dims != D {
        return Err(format!(
            "dataset '{dataset_name}' has dimension {dims}, but DIM is set to {D}"
        )
        .into());
    }

    let raw = read_f32_values(dataset, dataset_name)?;
    let expected_len = shape[0] * dims;
    if raw.len() != expected_len {
        return Err(format!("dataset '{dataset_name}' has {expected_len} values from shape {shape:?}, but read {} values", raw.len()).into());
    }

    let mut vectors = Vec::with_capacity(rows);
    for row in raw.chunks_exact(dims).take(rows) {
        vectors.push(row.try_into().unwrap());
    }
    Ok(vectors)
}

pub(crate) fn load_ground_truth(
    dataset_name: &str,
    dataset: &Dataset,
    query_count: usize,
    k: usize,
) -> Result<Vec<Vec<usize>>, Box<dyn Error>> {
    let shape = dataset.shape();
    if shape.len() != 2 {
        return Err(format!("dataset '{dataset_name}' must be 2-D, got shape {shape:?}").into());
    }
    if shape[1] < k {
        return Err(format!("dataset '{dataset_name}' has only {} ground-truth neighbors per query, but recall@{k} was requested", shape[1]).into());
    }

    let rows = query_count.min(shape[0]);
    if rows != query_count {
        return Err(format!("dataset '{dataset_name}' contains only {rows} rows, but {query_count} queries were loaded").into());
    }

    if let Ok(raw) = dataset.read_raw::<u64>() {
        return build_ground_truth_from_u64(raw, dataset_name, shape[1], rows, k);
    }
    if let Ok(raw) = dataset.read_raw::<u32>() {
        return Ok(build_ground_truth_from_usize(
            raw.into_iter().map(|value| value as usize).collect(),
            dataset_name,
            shape[1],
            rows,
            k,
        )?);
    }
    if let Ok(raw) = dataset.read_raw::<i64>() {
        let mut ids = Vec::with_capacity(raw.len());
        for value in raw {
            ids.push(usize::try_from(value).map_err(|_| {
                format!("dataset '{dataset_name}' contains a negative ground-truth id: {value}")
            })?);
        }
        return Ok(build_ground_truth_from_usize(
            ids,
            dataset_name,
            shape[1],
            rows,
            k,
        )?);
    }
    if let Ok(raw) = dataset.read_raw::<i32>() {
        let mut ids = Vec::with_capacity(raw.len());
        for value in raw {
            ids.push(usize::try_from(value).map_err(|_| {
                format!("dataset '{dataset_name}' contains a negative ground-truth id: {value}")
            })?);
        }
        return Ok(build_ground_truth_from_usize(
            ids,
            dataset_name,
            shape[1],
            rows,
            k,
        )?);
    }

    Err(format!(
        "dataset '{dataset_name}' could not be read as u64, u32, i64, or i32 ground-truth ids"
    )
    .into())
}

fn build_ground_truth_from_u64(
    raw: Vec<u64>,
    dataset_name: &str,
    width: usize,
    rows: usize,
    k: usize,
) -> Result<Vec<Vec<usize>>, Box<dyn Error>> {
    let mut ids = Vec::with_capacity(raw.len());
    for value in raw {
        ids.push(usize::try_from(value).map_err(|_| {
            format!("dataset '{dataset_name}' contains a ground-truth id that does not fit in usize: {value}")
        })?);
    }
    Ok(build_ground_truth_from_usize(
        ids,
        dataset_name,
        width,
        rows,
        k,
    )?)
}

fn build_ground_truth_from_usize(
    raw: Vec<usize>,
    dataset_name: &str,
    width: usize,
    rows: usize,
    k: usize,
) -> Result<Vec<Vec<usize>>, String> {
    let expected_len = rows * width;
    if raw.len() < expected_len {
        return Err(format!(
            "dataset '{dataset_name}' ended early: expected at least {expected_len} ids, got {}",
            raw.len()
        ));
    }

    let mut ground_truth = Vec::with_capacity(rows);
    for row in raw.chunks_exact(width).take(rows) {
        ground_truth.push(row[..k].to_vec());
    }
    Ok(ground_truth)
}

fn read_f32_values(dataset: &Dataset, dataset_name: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    if let Ok(values) = dataset.read_raw::<f32>() {
        return Ok(values);
    }
    if let Ok(values) = dataset.read_raw::<f64>() {
        return Ok(values.into_iter().map(|value| value as f32).collect());
    }

    Err(format!("dataset '{dataset_name}' could not be read as f32 or f64").into())
}
