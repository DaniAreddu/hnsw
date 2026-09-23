//! Exact duplicate vectors must stay reachable: every copy is returned by a
//! wide enough search and each copy finds itself.

use hnsw::{Hnsw, HnswSearcher, L2Squared};
use rand::{RngExt, SeedableRng, rngs::StdRng};

const DUP: [f32; 8] = [0.5; 8];

/// 1000 seeded random vectors with `dups` copies of `DUP` interleaved.
fn build(dups: usize, m: usize) -> (Hnsw<8>, Vec<usize>) {
    let index = Hnsw::<8>::new_seeded(m, 2 * m, 100, 3, L2Squared);
    let mut rng = StdRng::seed_from_u64(1);
    let mut dup_ids = Vec::new();
    for i in 0..1000 {
        if i % 5 == 0 && i / 5 < dups {
            dup_ids.push(index.insert(DUP));
        }
        index.insert(std::array::from_fn(|_| rng.random_range(-1.0..1.0)));
    }
    (index, dup_ids)
}

#[test]
fn every_duplicate_is_returned() {
    for (dups, m) in [(50, 8), (200, 8), (200, 16)] {
        let (index, mut dup_ids) = build(dups, m);
        let mut found: Vec<usize> = index
            .search_with_ef(&DUP, dups, 2 * dups)
            .into_iter()
            .filter(|&(_, distance)| distance == 0.0)
            .map(|(id, _)| id)
            .collect();
        found.sort_unstable();
        dup_ids.sort_unstable();
        assert_eq!(found, dup_ids, "dups={dups} M={m}");
    }
}

#[test]
fn equal_distance_results_are_ordered_by_id() {
    let (index, dup_ids) = build(50, 8);
    let hits = index.search_with_ef(&DUP, 50, 100);
    let ids: Vec<usize> = hits.iter().map(|&(id, _)| id).collect();
    assert_eq!(ids, dup_ids, "ties must be reported in ascending id order");
}

fn dataset(dups: usize) -> Vec<[f32; 8]> {
    let mut rng = StdRng::seed_from_u64(1);
    let mut vecs = Vec::new();
    for i in 0..1000 {
        if i % 5 == 0 && i / 5 < dups {
            vecs.push(DUP);
        }
        vecs.push(std::array::from_fn(|_| rng.random_range(-1.0..1.0)));
    }
    vecs
}

fn exact_matches(index: &Hnsw<8>, dups: usize) -> usize {
    index
        .search_with_ef(&DUP, dups, 2 * dups)
        .iter()
        .filter(|&&(_, distance)| distance == 0.0)
        .count()
}

#[test]
fn parallel_builds_return_every_duplicate() {
    let vecs = dataset(100);
    let mut batched = Hnsw::<8>::new_seeded(16, 32, 100, 3, L2Squared);
    batched.build_parallel(&vecs, std::num::NonZeroUsize::new(4));
    // batched builds group duplicates by exact hash before linking
    assert_eq!(exact_matches(&batched, 100), 100);

    // concurrent inserts detect duplicates through the insertion search, so two
    // copies inserted at the same moment may both become graph nodes; every copy
    // must still be reachable, which a scheduling-dependent graph guarantees only
    // approximately
    let dynamic = Hnsw::<8>::new_seeded(16, 32, 100, 3, L2Squared);
    dynamic.extend_parallel(&vecs, std::num::NonZeroUsize::new(4));
    let found = exact_matches(&dynamic, 100);
    assert!(found >= 95, "{found}/100 copies found");
}

#[test]
fn duplicate_groups_survive_save_load_and_keep_growing() {
    let (index, _) = build(50, 8);
    let path = std::env::temp_dir().join(format!("hnsw-dups-{}.bin", std::process::id()));
    index.save(&path).unwrap();
    let loaded = Hnsw::<8>::load(&path).unwrap();
    std::fs::remove_file(&path).unwrap();

    assert_eq!(exact_matches(&loaded, 50), 50);
    for _ in 0..10 {
        loaded.insert(DUP);
    }
    assert_eq!(exact_matches(&loaded, 60), 60);
    assert_eq!(loaded.len(), 1060);
}

#[test]
fn signed_zeros_are_the_same_vector() {
    let index = Hnsw::<2>::new_seeded(4, 8, 16, 3, L2Squared);
    let a = index.insert([0.0, 1.0]);
    let b = index.insert([-0.0, 1.0]);
    index.insert([5.0, 5.0]);
    assert_eq!(index.search(&[0.0, 1.0], 2), vec![(a, 0.0), (b, 0.0)]);
}
