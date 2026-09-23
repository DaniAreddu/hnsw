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
