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
