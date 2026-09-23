//! Sequential, batched and dynamic builds against exact (brute-force) search on
//! small seeded datasets.
//!
//! Sequential insertion into a seeded index is deterministic. Parallel builds
//! depend on thread scheduling: the graph (and, for `extend_parallel`, the id
//! each vector receives) can differ between runs, so they are held to recall
//! bounds rather than to exact equality.

use hnsw::{Distance, Hnsw, HnswSearcher, L2Squared};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::{collections::HashSet, num::NonZeroUsize};

const D: usize = 8;
const K: usize = 10;

fn vectors(n: usize, seed: u64) -> Vec<[f32; D]> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| std::array::from_fn(|_| rng.random_range(-1.0..1.0)))
        .collect()
}

/// Exact top-k ids by (distance, id); `ids[i]` is the id of `base[i]`.
fn exact(base: &[[f32; D]], ids: &[usize], q: &[f32; D], k: usize) -> Vec<usize> {
    let mut all: Vec<(f32, usize)> = base
        .iter()
        .zip(ids)
        .map(|(v, &id)| (L2Squared.distance(v, q), id))
        .collect();
    all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    all.into_iter().take(k).map(|(_, id)| id).collect()
}

fn recall(index: &Hnsw<D>, base: &[[f32; D]], ids: &[usize], queries: &[[f32; D]]) -> f64 {
    let mut hits = 0;
    for q in queries {
        let expected: HashSet<usize> = exact(base, ids, q, K).into_iter().collect();
        let found = index.search_with_ef(q, K, 100);
        assert!(
            found.windows(2).all(|w| w[0].1 <= w[1].1),
            "unsorted results"
        );
        hits += found.iter().filter(|(id, _)| expected.contains(id)).count();
    }
    hits as f64 / (queries.len() * K) as f64
}

fn seeded() -> Hnsw<D> {
    Hnsw::<D>::new_seeded(12, 24, 100, 11, L2Squared)
}

#[test]
fn every_build_mode_matches_exact_search() {
    let base = vectors(2000, 1);
    let queries = vectors(200, 2);
    let positions: Vec<usize> = (0..base.len()).collect();

    let sequential = seeded();
    for &v in &base {
        sequential.insert(v);
    }
    let r = recall(&sequential, &base, &positions, &queries);
    assert!(r >= 0.98, "sequential recall@{K} {r}");

    for threads in [1, 2, 4] {
        let mut batched = seeded();
        batched.build_parallel(&base, NonZeroUsize::new(threads));
        let r = recall(&batched, &base, &positions, &queries);
        assert!(r >= 0.98, "batched({threads}) recall@{K} {r}");

        let dynamic = seeded();
        let ids = dynamic.extend_parallel(&base, NonZeroUsize::new(threads));
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, positions, "ids must be a permutation of positions");
        let r = recall(&dynamic, &base, &ids, &queries);
        assert!(r >= 0.98, "dynamic({threads}) recall@{K} {r}");
    }
}

#[test]
fn every_vector_finds_itself() {
    let base = vectors(1500, 3);
    let mut batched = seeded();
    batched.build_parallel(&base, NonZeroUsize::new(4));
    let sequential = seeded();
    for &v in &base {
        sequential.insert(v);
    }
    for index in [&sequential, &batched] {
        for (id, v) in base.iter().enumerate() {
            assert_eq!(index.search_with_ef(v, 1, 64), vec![(id, 0.0)]);
        }
    }
}

#[test]
fn sequential_builds_are_deterministic() {
    let base = vectors(1000, 4);
    let queries = vectors(100, 5);
    let (a, b) = (seeded(), seeded());
    for &v in &base {
        a.insert(v);
        b.insert(v);
    }
    for q in &queries {
        assert_eq!(a.search_with_ef(q, K, 40), b.search_with_ef(q, K, 40));
    }
}

#[test]
fn save_load_then_insert_matches_the_uninterrupted_build() {
    let base = vectors(1200, 6);
    let queries = vectors(100, 7);
    let (head, tail) = base.split_at(700);

    let uninterrupted = seeded();
    let interrupted = seeded();
    for &v in head {
        uninterrupted.insert(v);
        interrupted.insert(v);
    }
    let path = std::env::temp_dir().join(format!("hnsw-resume-{}.bin", std::process::id()));
    interrupted.save(&path).unwrap();
    drop(interrupted);
    let resumed = Hnsw::<D>::load(&path).unwrap();
    std::fs::remove_file(&path).unwrap();

    for &v in tail {
        uninterrupted.insert(v);
        resumed.insert(v);
    }
    for q in &queries {
        assert_eq!(
            resumed.search_with_ef(q, K, 40),
            uninterrupted.search_with_ef(q, K, 40)
        );
    }
}
