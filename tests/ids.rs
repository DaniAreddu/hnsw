//! Caller-supplied ids: uniqueness, mode separation, atomic batches,
//! concurrency and persistence.

use hnsw::{Distance, Hnsw, HnswError, HnswSearcher, IdMode, L2Squared};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::{
    collections::{HashMap, HashSet},
    num::NonZeroUsize,
    sync::Barrier,
};

fn new_index() -> Hnsw<4> {
    Hnsw::<4>::try_new_seeded(8, 16, 64, 5, L2Squared).unwrap()
}

fn vectors(n: usize, seed: u64) -> Vec<[f32; 4]> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| std::array::from_fn(|_| rng.random_range(-1.0..1.0)))
        .collect()
}

fn temp_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("hnsw-ids-{name}-{}.bin", std::process::id()))
}

#[test]
fn searches_report_the_caller_ids() {
    let index = new_index();
    assert_eq!(index.id_mode(), None);
    index.try_insert_with_id(usize::MAX - 1, [0.0; 4]).unwrap();
    index.try_insert_with_id(42, [1.0, 0.0, 0.0, 0.0]).unwrap();
    index.try_insert_with_id(7, [5.0, 0.0, 0.0, 0.0]).unwrap();
    assert_eq!(index.id_mode(), Some(IdMode::Explicit));

    let hits = index.try_search(&[0.9, 0.0, 0.0, 0.0], 3).unwrap();
    let ids: Vec<usize> = hits.iter().map(|&(id, _)| id).collect();
    assert_eq!(ids, vec![42, usize::MAX - 1, 7]);
    assert!(index.contains_id(42) && index.contains_id(7) && !index.contains_id(0));
}

#[test]
fn duplicate_ids_are_rejected_and_change_nothing() {
    let index = new_index();
    index.try_insert_with_id(1, [0.0; 4]).unwrap();
    assert_eq!(
        index.try_insert_with_id(1, [9.0; 4]).unwrap_err(),
        HnswError::DuplicateId { id: 1 }
    );
    assert_eq!(index.len(), 1);
    // the rejected vector was not stored: only id 1 at [0; 4] is found
    assert_eq!(index.try_search(&[9.0; 4], 5).unwrap(), vec![(1, 324.0)]);
}

#[test]
fn positional_and_explicit_ids_cannot_be_mixed() {
    let positional = new_index();
    positional.try_insert([0.0; 4]).unwrap();
    assert_eq!(positional.id_mode(), Some(IdMode::Positional));
    assert!(positional.contains_id(0) && !positional.contains_id(1));
    assert_eq!(
        positional.try_insert_with_id(5, [1.0; 4]).unwrap_err(),
        HnswError::IdModeMismatch {
            index_mode: IdMode::Positional
        }
    );
    assert!(matches!(
        positional.try_extend_parallel_with_ids(&[(9, [1.0; 4])], None),
        Err(HnswError::IdModeMismatch { .. })
    ));

    let explicit = new_index();
    explicit.try_insert_with_id(5, [0.0; 4]).unwrap();
    for error in [
        explicit.try_insert([1.0; 4]).unwrap_err(),
        explicit.try_extend_parallel(&[[1.0; 4]], None).unwrap_err(),
    ] {
        assert_eq!(
            error,
            HnswError::IdModeMismatch {
                index_mode: IdMode::Explicit
            }
        );
    }
    assert_eq!((positional.len(), explicit.len()), (1, 1));
}

#[test]
fn batches_are_all_or_nothing() {
    let index = new_index();
    index.try_insert_with_id(3, [0.0; 4]).unwrap();
    let vecs = vectors(4, 1);

    let repeated = [(10, vecs[0]), (11, vecs[1]), (10, vecs[2])];
    assert_eq!(
        index.try_extend_parallel_with_ids(&repeated, NonZeroUsize::new(2)),
        Err(HnswError::DuplicateId { id: 10 })
    );
    let clashing = [(20, vecs[0]), (3, vecs[1])];
    assert_eq!(
        index.try_extend_parallel_with_ids(&clashing, NonZeroUsize::new(2)),
        Err(HnswError::DuplicateId { id: 3 })
    );
    assert_eq!(index.len(), 1);
    assert!(!index.contains_id(10) && !index.contains_id(20));

    let mut fresh = new_index();
    assert_eq!(
        fresh.try_build_parallel_with_ids(&repeated, None),
        Err(HnswError::DuplicateId { id: 10 })
    );
    assert_eq!(fresh.len(), 0);
    assert_eq!(fresh.id_mode(), Some(IdMode::Explicit));
}

#[test]
fn concurrent_inserts_of_one_id_succeed_exactly_once() {
    const THREADS: usize = 8;
    let index = new_index();
    let barrier = Barrier::new(THREADS);
    let wins: Vec<Vec<usize>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..THREADS)
            .map(|t| {
                let (index, barrier) = (&index, &barrier);
                s.spawn(move || {
                    barrier.wait();
                    (0..50)
                        .filter(|&id| {
                            index
                                .try_insert_with_id(id, [id as f32, t as f32, 0.0, 0.0])
                                .is_ok()
                        })
                        .collect()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut per_id: HashMap<usize, usize> = HashMap::new();
    for id in wins.into_iter().flatten() {
        *per_id.entry(id).or_default() += 1;
    }
    assert_eq!(per_id.len(), 50);
    assert!(per_id.values().all(|&n| n == 1), "an id was inserted twice");
    assert_eq!(index.len(), 50);
}

#[test]
fn overlapping_concurrent_batches_never_share_an_id() {
    let index = new_index();
    let vecs = vectors(400, 2);
    let a: Vec<(usize, [f32; 4])> = (0..200).map(|i| (i, vecs[i])).collect();
    // shares id 150..200 with `a`
    let b: Vec<(usize, [f32; 4])> = (150..350).map(|i| (i, vecs[i])).collect();
    let (ra, rb) = std::thread::scope(|s| {
        let ha = s.spawn(|| index.try_extend_parallel_with_ids(&a, NonZeroUsize::new(2)));
        let hb = s.spawn(|| index.try_extend_parallel_with_ids(&b, NonZeroUsize::new(2)));
        (ha.join().unwrap(), hb.join().unwrap())
    });
    assert!(
        ra.is_err() || rb.is_err(),
        "both overlapping batches succeeded"
    );
    let expected = [&ra, &rb]
        .iter()
        .zip([a.len(), b.len()])
        .filter(|(result, _)| result.is_ok())
        .map(|(_, len)| len)
        .sum::<usize>();
    assert_eq!(index.len(), expected);
}

#[test]
fn parallel_builds_with_ids_match_exact_search() {
    let vecs = vectors(1500, 3);
    let items: Vec<(usize, [f32; 4])> = vecs
        .iter()
        .enumerate()
        .map(|(i, &v)| (1_000 + 7 * i, v))
        .collect();
    let queries = vectors(100, 4);

    let mut built = new_index();
    built.build_parallel_with_ids(&items, NonZeroUsize::new(4));
    let extended = new_index();
    extended.extend_parallel_with_ids(&items, NonZeroUsize::new(4));

    for index in [&built, &extended] {
        let mut hits = 0;
        for q in &queries {
            let mut exact: Vec<(f32, usize)> = items
                .iter()
                .map(|&(id, v)| (L2Squared.distance(&v, q), id))
                .collect();
            exact.sort_by(|x, y| x.0.total_cmp(&y.0));
            let expected: HashSet<usize> = exact.iter().take(10).map(|&(_, id)| id).collect();
            hits += index
                .search_with_ef(q, 10, 100)
                .iter()
                .filter(|(id, _)| expected.contains(id))
                .count();
        }
        let recall = hits as f64 / 1000.0;
        assert!(recall >= 0.98, "recall@10 {recall}");
        // parallel graphs: bounded rather than exact self-retrieval
        let found = items
            .iter()
            .filter(|&&(id, v)| index.search_with_ef(&v, 1, 64) == vec![(id, 0.0)])
            .count();
        assert!(found * 1000 >= items.len() * 995, "self-retrieval {found}");
    }
}

#[test]
fn ids_survive_save_load() {
    let path = temp_path("explicit");
    let index = new_index();
    for (i, v) in vectors(300, 5).into_iter().enumerate() {
        index.insert_with_id(10 * i + 1, v);
    }
    index.save(&path).unwrap();
    let loaded = Hnsw::<4>::load(&path).unwrap();
    std::fs::remove_file(&path).unwrap();

    assert_eq!(loaded.id_mode(), Some(IdMode::Explicit));
    let q = [0.1, 0.2, 0.3, 0.4];
    assert_eq!(loaded.search(&q, 10), index.search(&q, 10));
    assert_eq!(
        loaded.try_insert_with_id(11, [0.0; 4]),
        Err(HnswError::DuplicateId { id: 11 })
    );
    loaded.try_insert_with_id(12, [0.5; 4]).unwrap();
    assert_eq!(loaded.search(&[0.5; 4], 1), vec![(12, 0.0)]);

    let path = temp_path("positional");
    let positional = new_index();
    positional.insert([1.0; 4]);
    positional.save(&path).unwrap();
    let loaded = Hnsw::<4>::load(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!(loaded.id_mode(), Some(IdMode::Positional));
    assert_eq!(loaded.insert([2.0; 4]), 1);
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
fn a_failed_insert_releases_its_id() {
    let index = Hnsw::<4, NanForNegative>::try_new_seeded(8, 16, 64, 5, NanForNegative).unwrap();
    index.try_insert_with_id(1, [1.0; 4]).unwrap();
    assert!(matches!(
        index.try_insert_with_id(2, [-1.0, 0.0, 0.0, 0.0]),
        Err(HnswError::NonFiniteDistance { .. })
    ));
    assert!(!index.contains_id(2));
    index.try_insert_with_id(2, [2.0; 4]).unwrap();
    assert_eq!(index.len(), 2);
}

#[test]
fn duplicate_vectors_keep_distinct_ids() {
    let index = new_index();
    index.insert_with_id(100, [0.5; 4]);
    index.insert_with_id(7, [0.5; 4]);
    index.insert_with_id(55, [2.0; 4]);
    assert_eq!(index.search(&[0.5; 4], 2), vec![(7, 0.0), (100, 0.0)]);
}
