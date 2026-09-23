//! Behavior of the fallible `try_*` API: every rejected input is reported as a
//! typed error and leaves the index unchanged.

use hnsw::{Distance, Hnsw, HnswError, HnswSearcher, L2Squared};
use std::num::NonZeroUsize;

fn line(n: usize) -> Vec<[f32; 4]> {
    (0..n).map(|i| [i as f32, 0.0, 0.0, 0.0]).collect()
}

fn index_with(n: usize) -> Hnsw<4> {
    let index = Hnsw::<4>::try_new_seeded(4, 8, 16, 7, L2Squared).unwrap();
    for v in line(n) {
        index.try_insert(v).unwrap();
    }
    index
}

fn parameter_name(error: HnswError) -> &'static str {
    match error {
        HnswError::InvalidParameter { name, .. } => name,
        other => panic!("expected InvalidParameter, got {other:?}"),
    }
}

#[test]
fn construction_rejects_out_of_range_parameters() {
    let new = |m, m0, efc| Hnsw::<4>::try_new_seeded(m, m0, efc, 1, L2Squared).err();
    for (m, m0, efc, name) in [
        (0, 8, 16, "M"),
        (1, 8, 16, "M"),
        (4097, 8, 16, "M"),
        (4, 0, 16, "M0"),
        (4, 4097, 16, "M0"),
        (4, 8, 0, "ef_construction"),
        (4, 8, 65_537, "ef_construction"),
    ] {
        let error = new(m, m0, efc).unwrap_or_else(|| panic!("accepted {m}/{m0}/{efc}"));
        assert!(!error.to_string().is_empty());
        assert_eq!(parameter_name(error), name);
    }
    assert!(new(2, 1, 1).is_none());
    assert!(new(4096, 4096, 65_536).is_none());
}

#[test]
fn search_parameter_semantics() {
    let index = index_with(10);
    let q = [0.0; 4];

    let error = index.try_search_with_ef(&q, 3, 0).unwrap_err();
    assert_eq!(parameter_name(error), "ef_search");
    // ef_search is validated even when there is nothing to search
    let empty = index_with(0);
    assert!(empty.try_search_with_ef(&q, 3, 0).is_err());

    assert_eq!(index.try_search(&q, 0).unwrap(), vec![]);
    assert_eq!(empty.try_search(&q, 5).unwrap(), vec![]);

    // ef_search smaller than k is widened to k
    assert_eq!(index.try_search_with_ef(&q, 10, 1).unwrap().len(), 10);
}

#[test]
fn search_results_are_sorted_and_ties_break_by_id() {
    let index = Hnsw::<4>::try_new_seeded(4, 8, 16, 7, L2Squared).unwrap();
    for v in [
        [1.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ] {
        index.try_insert(v).unwrap();
    }
    let results = index.try_search_with_ef(&[0.0; 4], 3, 8).unwrap();
    assert_eq!(results, vec![(0, 1.0), (1, 1.0), (2, 1.0)]);
}

#[test]
fn non_finite_query_and_vector_are_rejected() {
    let index = index_with(3);
    let error = index
        .try_search(&[0.0, f32::INFINITY, 0.0, 0.0], 1)
        .unwrap_err();
    assert!(matches!(
        error,
        HnswError::NonFiniteComponent { component: 1, .. }
    ));

    let error = index.try_insert([0.0, 0.0, f32::NAN, 0.0]).unwrap_err();
    assert!(matches!(
        error,
        HnswError::NonFiniteComponent { component: 2, .. }
    ));
    assert_eq!(index.len(), 3);
}

#[test]
fn runtime_sized_vectors_must_match_the_dimension() {
    let index = index_with(3);
    assert_eq!(
        index.try_insert_slice(&[1.0, 2.0, 3.0]).unwrap_err(),
        HnswError::DimensionMismatch {
            expected: 4,
            found: 3
        }
    );
    assert_eq!(
        index.try_search_slice(&[0.0; 5], 1, 8).unwrap_err(),
        HnswError::DimensionMismatch {
            expected: 4,
            found: 5
        }
    );
    assert_eq!(index.len(), 3);

    let id = index.try_insert_slice(&[9.0, 0.0, 0.0, 0.0]).unwrap();
    assert_eq!(
        index.try_search_slice(&[9.0, 0.0, 0.0, 0.0], 1, 8).unwrap(),
        vec![(id, 0.0)]
    );
}

#[test]
fn l2_component_limit_keeps_distances_finite() {
    let limit = L2Squared::component_limit(4);
    let index = Hnsw::<4>::try_new_seeded(4, 8, 16, 7, L2Squared).unwrap();
    index.try_insert([limit; 4]).unwrap();
    index.try_insert([-limit; 4]).unwrap();
    for (_, distance) in index.try_search_with_ef(&[limit; 4], 2, 8).unwrap() {
        assert!(distance.is_finite());
    }
    assert!(L2Squared.distance(&[limit; 4], &[-limit; 4]).is_finite());

    let above = limit.next_up();
    let error = index.try_insert([0.0, 0.0, 0.0, above]).unwrap_err();
    assert!(matches!(
        error,
        HnswError::ComponentOutOfRange { component: 3, .. }
    ));
    assert!(index.try_search(&[-above, 0.0, 0.0, 0.0], 1).is_err());
    assert_eq!(index.len(), 2);
}

#[test]
fn parallel_builds_validate_the_whole_batch_first() {
    let mut vecs = line(20);
    vecs[5][1] = f32::NAN;

    let mut index = Hnsw::<4>::try_new_seeded(4, 8, 16, 7, L2Squared).unwrap();
    let error = index
        .try_build_parallel(&vecs, NonZeroUsize::new(2))
        .unwrap_err();
    match &error {
        HnswError::InvalidBatchVector { index, error } => {
            assert_eq!(*index, 5);
            assert!(matches!(**error, HnswError::NonFiniteComponent { .. }));
        }
        other => panic!("unexpected {other:?}"),
    }
    assert!(std::error::Error::source(&error).is_some());
    assert_eq!(index.len(), 0);

    let error = index
        .try_extend_parallel(&vecs, NonZeroUsize::new(2))
        .unwrap_err();
    assert!(matches!(
        error,
        HnswError::InvalidBatchVector { index: 5, .. }
    ));
    assert_eq!(index.len(), 0);

    index.try_build_parallel(&[], None).unwrap();
    assert_eq!(index.try_extend_parallel(&[], None).unwrap(), vec![]);
    assert_eq!(index.len(), 0);

    vecs[5][1] = 0.0;
    index
        .try_build_parallel(&vecs, NonZeroUsize::new(2))
        .unwrap();
    assert_eq!(index.len(), 20);
    assert_eq!(
        index.try_build_parallel(&vecs, None).unwrap_err(),
        HnswError::IndexNotEmpty { len: 20 }
    );
}

/// L2 that returns NaN whenever either vector has a negative first component.
#[derive(Clone, Copy, Default)]
struct NanForNegative;

impl Distance<4> for NanForNegative {
    fn distance(&self, a: &[f32; 4], b: &[f32; 4]) -> f32 {
        if a[0] < 0.0 || b[0] < 0.0 {
            f32::NAN
        } else {
            L2Squared.distance(a, b)
        }
    }
}

#[test]
fn non_finite_distances_from_a_custom_metric_are_reported() {
    let index = Hnsw::<4, NanForNegative>::try_new_seeded(4, 8, 16, 7, NanForNegative).unwrap();
    for v in line(5) {
        index.try_insert(v).unwrap();
    }

    let error = index.try_insert([-1.0, 0.0, 0.0, 0.0]).unwrap_err();
    assert!(matches!(error, HnswError::NonFiniteDistance { distance } if distance.is_nan()));
    assert_eq!(index.len(), 5);

    let error = index.try_search(&[-1.0, 0.0, 0.0, 0.0], 1).unwrap_err();
    assert!(matches!(error, HnswError::NonFiniteDistance { .. }));
    assert_eq!(
        index.try_search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap(),
        vec![(1, 0.0)]
    );
}

/// Negative inner product: larger dot products are closer; distances are negative.
#[derive(Clone, Copy, Default)]
struct NegativeDot;

impl Distance<4> for NegativeDot {
    fn distance(&self, a: &[f32; 4], b: &[f32; 4]) -> f32 {
        -a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
    }
}

#[test]
fn negative_distances_are_supported() {
    let index = Hnsw::<4, NegativeDot>::try_new_seeded(4, 8, 16, 7, NegativeDot).unwrap();
    for v in line(10) {
        index.try_insert(v).unwrap();
    }
    let results = index
        .try_search_with_ef(&[1.0, 0.0, 0.0, 0.0], 2, 16)
        .unwrap();
    assert_eq!(results, vec![(9, -9.0), (8, -8.0)]);
}
