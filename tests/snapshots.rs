//! Snapshot format v1 at the public API: round trips, identity checks,
//! recovery and consistency under concurrent inserts.

use hnsw::{
    Distance, Hnsw, HnswSearcher, IdMode, L2Squared, SNAPSHOT_FORMAT_VERSION, SnapshotError,
};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("hnsw-snap-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn vectors(n: usize, seed: u64) -> Vec<[f32; 4]> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| std::array::from_fn(|_| rng.random_range(-1.0..1.0)))
        .collect()
}

fn snapshot<DS: Distance<4> + Serialize>(index: &Hnsw<4, DS>) -> Vec<u8> {
    let mut bytes = Vec::new();
    index.write_snapshot(&mut bytes).unwrap();
    bytes
}

fn load(bytes: &[u8]) -> Result<Hnsw<4>, SnapshotError> {
    Hnsw::<4>::read_snapshot(bytes, bytes.len() as u64)
}

fn built(n: usize) -> Hnsw<4> {
    let index = Hnsw::<4>::new_seeded(6, 12, 32, 3, L2Squared);
    for v in vectors(n, 1) {
        index.insert(v);
    }
    index
}

#[test]
fn round_trips_preserve_every_index_shape() {
    let queries = vectors(20, 9);
    let check = |a: &Hnsw<4>, b: &Hnsw<4>| {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.id_mode(), b.id_mode());
        for q in &queries {
            assert_eq!(a.search_with_ef(q, 5, 32), b.search_with_ef(q, 5, 32));
        }
    };

    let empty = Hnsw::<4>::new_seeded(6, 12, 32, 3, L2Squared);
    let loaded = load(&snapshot(&empty)).unwrap();
    check(&empty, &loaded);
    assert_eq!(loaded.id_mode(), None);
    loaded.insert_with_id(7, [0.0; 4]);

    check(&built(300), &load(&snapshot(&built(300))).unwrap());

    let explicit = Hnsw::<4>::new_seeded(6, 12, 32, 3, L2Squared);
    for (i, v) in vectors(200, 2).into_iter().enumerate() {
        explicit.insert_with_id(1_000_000 + i, v);
        if i % 10 == 0 {
            explicit.insert_with_id(i, v); // duplicate vector, distinct id
        }
    }
    let loaded = load(&snapshot(&explicit)).unwrap();
    check(&explicit, &loaded);
    assert_eq!(loaded.id_mode(), Some(IdMode::Explicit));
    assert!(loaded.contains_id(1_000_199) && loaded.contains_id(190));

    let mut batched = Hnsw::<4>::new_seeded(6, 12, 32, 3, L2Squared);
    batched.build_parallel(&vectors(500, 4), std::num::NonZeroUsize::new(4));
    check(&batched, &load(&snapshot(&batched)).unwrap());
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct Weighted {
    weights: [f32; 4],
}

impl Distance<4> for Weighted {
    fn distance(&self, a: &[f32; 4], b: &[f32; 4]) -> f32 {
        (0..4)
            .map(|i| self.weights[i] * (a[i] - b[i]).powi(2))
            .sum()
    }
    fn metric_id() -> &'static str {
        "tests::Weighted"
    }
}

#[test]
fn metric_parameters_round_trip_and_identity_is_checked() {
    let metric = Weighted {
        weights: [0.25, 1.0, 2.0, 0.0],
    };
    let index = Hnsw::<4, Weighted>::new_seeded(6, 12, 32, 3, metric);
    for v in vectors(50, 5) {
        index.insert(v);
    }
    let bytes = snapshot(&index);
    let loaded = Hnsw::<4, Weighted>::read_snapshot(&bytes[..], bytes.len() as u64).unwrap();
    let q = [0.1, 0.2, 0.3, 0.4];
    assert_eq!(loaded.search(&q, 5), index.search(&q, 5));
    assert_eq!(snapshot(&loaded), bytes);

    match load(&bytes) {
        Err(SnapshotError::MetricMismatch { expected, found }) => {
            assert_eq!(
                (expected.as_str(), found.as_str()),
                ("hnsw::L2Squared", "tests::Weighted")
            );
        }
        other => panic!("expected MetricMismatch, got {:?}", other.err()),
    }
}

#[test]
fn wrong_dimension_and_version_are_reported() {
    let bytes = snapshot(&built(20));
    let err = Hnsw::<3>::read_snapshot(&bytes[..], bytes.len() as u64)
        .err()
        .unwrap();
    assert!(
        matches!(
            err,
            SnapshotError::DimensionMismatch {
                expected: 3,
                found: 4
            }
        ),
        "{err:?}"
    );

    let mut future = bytes.clone();
    future[8..12].copy_from_slice(&(SNAPSHOT_FORMAT_VERSION + 1).to_le_bytes());
    assert!(matches!(
        load(&future),
        Err(SnapshotError::UnsupportedVersion {
            found: 2,
            supported: 1
        })
    ));
}

#[test]
fn foreign_legacy_and_empty_files_are_not_snapshots() {
    let dir = temp_dir("foreign");
    let path = dir.join("index.bin");
    // the pre-v0.1 layout started with M as a little-endian u64
    let mut legacy = 16u64.to_le_bytes().to_vec();
    legacy.extend_from_slice(&[0; 200]);
    for bytes in [legacy, b"{\"not\": \"an index\"}".to_vec()] {
        fs::write(&path, &bytes).unwrap();
        let err = Hnsw::<4>::load(&path).err().unwrap();
        assert!(matches!(err, SnapshotError::NotASnapshot), "{err:?}");
        assert!(err.to_string().contains("before v0.1"));
    }
    fs::write(&path, b"").unwrap();
    assert!(matches!(
        Hnsw::<4>::load(&path),
        Err(SnapshotError::NotASnapshot)
    ));
    fs::write(&path, b"HNSW").unwrap();
    assert!(matches!(
        Hnsw::<4>::load(&path),
        Err(SnapshotError::Truncated)
    ));
    assert!(matches!(
        Hnsw::<4>::load(dir.join("missing.bin")),
        Err(SnapshotError::Io(_))
    ));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn every_truncation_and_trailing_data_are_rejected() {
    let dir = temp_dir("truncation");
    let path = dir.join("index.bin");
    let bytes = snapshot(&built(30));
    for len in 0..bytes.len() {
        assert!(load(&bytes[..len]).is_err(), "prefix {len} accepted");
    }
    let mut longer = bytes.clone();
    longer.push(0);
    fs::write(&path, &longer).unwrap();
    assert!(matches!(
        Hnsw::<4>::load(&path),
        Err(SnapshotError::TrailingData)
    ));
    fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
    assert!(matches!(
        Hnsw::<4>::load(&path),
        Err(SnapshotError::Truncated)
    ));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn every_single_byte_corruption_is_detected() {
    let bytes = snapshot(&built(30));
    for position in 0..bytes.len() {
        for flip in [0x01, 0x80] {
            let mut corrupted = bytes.clone();
            corrupted[position] ^= flip;
            assert!(
                load(&corrupted).is_err(),
                "flip {flip:#x} at {position} accepted"
            );
        }
    }
}

#[test]
fn a_crash_mid_save_leaves_the_last_good_snapshot_loadable() {
    let dir = temp_dir("crash");
    let path = dir.join("index.bin");
    let good = built(100);
    good.save(&path).unwrap();

    // what an interrupted save leaves behind: a partial temporary file
    let partial = &snapshot(&built(150))[..700];
    fs::write(
        dir.join(format!(".index.bin.{}-99.tmp", std::process::id())),
        partial,
    )
    .unwrap();

    let recovered = Hnsw::<4>::load(&path).unwrap();
    assert_eq!(recovered.len(), 100);
    let q = [0.3, -0.2, 0.1, 0.0];
    assert_eq!(recovered.search(&q, 5), good.search(&q, 5));

    // and saving again still works next to the stray file
    recovered.insert([0.5; 4]);
    recovered.save(&path).unwrap();
    assert_eq!(Hnsw::<4>::load(&path).unwrap().len(), 101);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn snapshots_taken_during_concurrent_inserts_are_consistent() {
    let dir = temp_dir("concurrent");
    let base = vectors(4000, 6);
    let index = Hnsw::<4>::new_seeded(8, 16, 48, 3, L2Squared);
    for &v in &base[..200] {
        index.insert(v);
    }

    let snapshots: Vec<(usize, usize, PathBuf)> = std::thread::scope(|s| {
        let writers: Vec<_> = base[200..]
            .chunks(950)
            .map(|chunk| {
                let index = &index;
                s.spawn(move || {
                    for &v in chunk {
                        index.insert(v);
                    }
                })
            })
            .collect();
        let taken = (0..8)
            .map(|i| {
                let path = dir.join(format!("snap-{i}.bin"));
                let before = index.len();
                index.save(&path).unwrap();
                (before, index.len(), path)
            })
            .collect();
        writers.into_iter().for_each(|w| w.join().unwrap());
        taken
    });

    for (before, after, path) in snapshots {
        let loaded = Hnsw::<4>::load(&path).unwrap();
        assert!((before..=after).contains(&loaded.len()));
        // positional ids of a consistent snapshot are exactly 0..len
        let mut found = 0;
        for (id, v) in base.iter().enumerate().take(loaded.len()).step_by(17) {
            let hit = loaded.search_with_ef(v, 1, 64)[0];
            assert!(hit.0 < loaded.len());
            found += usize::from(hit == (id, 0.0));
        }
        assert!(found > 0);
        loaded.insert([0.25; 4]);
    }
    fs::remove_dir_all(dir).unwrap();
}
