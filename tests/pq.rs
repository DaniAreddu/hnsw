//! Experimental PQ index: ids, seeded reproducibility and quality relative to
//! exact search and to its own brute-force ADC oracle.
#![cfg(feature = "experimental-pq")]

use hnsw::{Distance, Hnsw, HnswSearcher, L2Squared, pq::ProductQuantizer};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::collections::HashSet;

const D: usize = 16;

fn clustered(n: usize, seed: u64) -> Vec<[f32; D]> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centers: Vec<[f32; D]> = (0..8)
        .map(|_| std::array::from_fn(|_| rng.random_range(-10.0..10.0)))
        .collect();
    (0..n)
        .map(|i| {
            let c = centers[i % centers.len()];
            std::array::from_fn(|d| c[d] + rng.random_range(-1.0..1.0))
        })
        .collect()
}

fn explicit_index(base: &[[f32; D]]) -> Hnsw<D> {
    let index = Hnsw::<D>::new_seeded(12, 24, 100, 3, L2Squared);
    for (i, &v) in base.iter().enumerate() {
        index.insert_with_id(5_000 + i, v);
    }
    index
}

fn top10(ids: impl Iterator<Item = usize>) -> HashSet<usize> {
    ids.take(10).collect()
}

#[test]
fn frozen_index_reports_ids_and_tracks_its_adc_oracle() {
    let base = clustered(3_000, 1);
    let queries = clustered(50, 2);
    let full_memory = explicit_index(&base).memory_usage_bytes();
    let frozen = explicit_index(&base).freeze_seeded::<4>(64, 9);
    assert!(frozen.memory_usage_bytes() < full_memory);

    let (mut vs_exact, mut vs_oracle) = (0, 0);
    for q in &queries {
        let mut exact: Vec<(f32, usize)> = base
            .iter()
            .enumerate()
            .map(|(i, v)| (L2Squared.distance(v, q), 5_000 + i))
            .collect();
        exact.sort_by(|a, b| a.0.total_cmp(&b.0));
        let exact = top10(exact.into_iter().map(|(_, id)| id));
        // ADC distances tie often, so compare against the oracle's 10th distance
        // rather than its (arbitrary) choice among tied ids
        let oracle_cutoff = frozen.brute_force_adc(q, 10)[9].1;
        let found = frozen.search_with_ef(q, 10, 512);
        assert!(found.iter().all(|&(id, _)| (5_000..8_000).contains(&id)));
        vs_exact += found.iter().filter(|(id, _)| exact.contains(id)).count();
        vs_oracle += found.iter().filter(|&&(_, d)| d <= oracle_cutoff).count();
    }
    let (vs_exact, vs_oracle) = (vs_exact as f64 / 500.0, vs_oracle as f64 / 500.0);
    // quantization bounds recall against exact search; the graph walk should
    // lose little relative to exhaustive ADC over the same codes
    // measured: 0.40 vs exact and 1.0 vs the oracle at ef_search = 512
    assert!(vs_exact >= 0.3, "recall@10 vs exact {vs_exact}");
    assert!(vs_oracle >= 0.98, "recall@10 vs ADC oracle {vs_oracle}");
}

#[test]
fn seeded_freezing_is_reproducible() {
    let base = clustered(2_000, 3);
    let q = clustered(1, 4)[0];
    let a = explicit_index(&base).freeze_seeded::<4>(32, 1);
    let b = explicit_index(&base).freeze_seeded::<4>(32, 1);
    assert_eq!(a.search_with_ef(&q, 10, 64), b.search_with_ef(&q, 10, 64));

    let mut pq = ProductQuantizer::<4, D>::new(32);
    pq.fit_seeded(&base, 1);
    let c = explicit_index(&base).freeze_with_pq(pq);
    assert_eq!(a.search_with_ef(&q, 10, 64), c.search_with_ef(&q, 10, 64));
}

#[test]
fn frozen_index_validates_queries_like_hnsw() {
    let frozen = explicit_index(&clustered(300, 5)).freeze_seeded::<4>(16, 1);
    assert!(frozen.try_search(&[f32::NAN; D], 1).is_err());
    assert!(frozen.try_search_with_ef(&[0.0; D], 1, 0).is_err());
    assert_eq!(frozen.try_search(&[0.0; D], 0).unwrap(), vec![]);
}
