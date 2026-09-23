//! Invalid inputs at the public API must be rejected before they reach the
//! graph, identically in debug and release builds.

use hnsw::{Hnsw, HnswSearcher, L2Squared};
use std::panic::{AssertUnwindSafe, catch_unwind};

fn small_index() -> Hnsw<4> {
    let index = Hnsw::<4>::new_seeded(4, 8, 16, 7, L2Squared);
    for i in 0..10 {
        index.insert([i as f32, 0.0, 0.0, 0.0]);
    }
    index
}

#[test]
fn huge_k_and_ef_search_return_every_vector() {
    let index = small_index();
    let results = index.search_with_ef(&[0.0; 4], usize::MAX, usize::MAX);
    assert_eq!(results.len(), 10);
}

#[test]
fn nan_vector_is_rejected_even_as_first_insert() {
    let index = Hnsw::<4>::new_seeded(4, 8, 16, 7, L2Squared);
    let inserted = catch_unwind(AssertUnwindSafe(|| index.insert([f32::NAN, 0.0, 0.0, 0.0])));
    assert!(inserted.is_err(), "NaN vector was accepted");
    assert_eq!(index.len(), 0);
}

#[test]
fn nan_query_is_rejected() {
    let index = small_index();
    let searched = catch_unwind(AssertUnwindSafe(|| {
        index.search(&[0.0, f32::NAN, 0.0, 0.0], 3)
    }));
    assert!(searched.is_err(), "NaN query returned results");
}

#[test]
fn finite_vectors_whose_distance_overflows_are_rejected() {
    let index = Hnsw::<4>::new_seeded(4, 8, 16, 7, L2Squared);
    let _ = catch_unwind(AssertUnwindSafe(|| {
        index.insert([f32::MAX; 4]);
        index.insert([-f32::MAX; 4]);
    }));
    for (_, distance) in index.search(&[0.0; 4], 2) {
        assert!(distance.is_finite(), "stored vectors overflow the metric");
    }
}
