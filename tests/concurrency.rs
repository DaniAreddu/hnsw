//! Concurrent use of one index through `&Hnsw`.

use hnsw::{Hnsw, HnswSearcher, L2Squared};
use std::sync::Barrier;

#[test]
fn concurrent_first_inserts_are_all_reachable() {
    const THREADS: usize = 8;
    for trial in 0..200 {
        let index = Hnsw::<2>::new_seeded(4, 8, 32, trial, L2Squared);
        let barrier = Barrier::new(THREADS);
        std::thread::scope(|s| {
            for t in 0..THREADS {
                let (index, barrier) = (&index, &barrier);
                s.spawn(move || {
                    barrier.wait();
                    index.insert([t as f32, 0.0]);
                });
            }
        });
        assert_eq!(index.len(), THREADS);
        for t in 0..THREADS {
            let hits = index.search_with_ef(&[t as f32, 0.0], 1, 64);
            assert_eq!(
                hits.first().map(|hit| hit.1),
                Some(0.0),
                "trial {trial}: vector {t} is unreachable"
            );
        }
    }
}

#[test]
fn searches_run_concurrently_with_inserts() {
    use rand::{RngExt, SeedableRng, rngs::StdRng};
    use std::sync::atomic::{AtomicBool, Ordering};

    let mut rng = StdRng::seed_from_u64(9);
    let base: Vec<[f32; 4]> = (0..3000)
        .map(|_| std::array::from_fn(|_| rng.random_range(-1.0..1.0)))
        .collect();
    let index = Hnsw::<4>::new_seeded(8, 16, 64, 1, L2Squared);
    let done = AtomicBool::new(false);

    std::thread::scope(|s| {
        for reader in 0..3u64 {
            let (index, done, base) = (&index, &done, &base);
            s.spawn(move || {
                let mut rng = StdRng::seed_from_u64(100 + reader);
                let mut ctx = index.search_context();
                let mut searches = 0;
                while !done.load(Ordering::Acquire) || searches < 100 {
                    let q = base[rng.random_range(0..base.len())];
                    let before = index.len();
                    let hits = index.search_with_context(&q, 5, 32, &mut ctx);
                    let after = index.len();
                    assert!(hits.len() <= 5.min(after));
                    // A vector committed by a concurrent insert becomes reachable only
                    // once that insert has published its backlinks, so only "some
                    // result from a non-empty index" is guaranteed here.
                    assert!(
                        before == 0 || !hits.is_empty(),
                        "no hits from a non-empty index"
                    );
                    assert!(hits.windows(2).all(|w| w[0].1 <= w[1].1));
                    assert!(hits.iter().all(|&(id, _)| id < after));
                    searches += 1;
                }
            });
        }
        let (index, done, base) = (&index, &done, &base);
        s.spawn(move || {
            let (first, rest) = base.split_at(1500);
            for &v in first {
                index.insert(v);
            }
            index.extend_parallel(rest, std::num::NonZeroUsize::new(2));
            done.store(true, Ordering::Release);
        });
    });

    assert_eq!(index.len(), base.len());
    let found = base
        .iter()
        .filter(|v| index.search_with_ef(v, 1, 64)[0].1 == 0.0)
        .count();
    assert!(
        found as f64 >= 0.995 * base.len() as f64,
        "self-recall {found}/{}",
        base.len()
    );
}
