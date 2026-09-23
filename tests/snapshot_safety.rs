//! Snapshot safety at the public API: a failed save must keep the previous
//! snapshot, and a loaded index must never be misinterpreted or panic later.

use hnsw::{Distance, Hnsw, HnswSearcher, L2Squared};
use serde::{Deserialize, Serialize, Serializer};
use std::{
    fs,
    panic::{AssertUnwindSafe, catch_unwind},
    path::PathBuf,
};

fn temp_path(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("hnsw-safety-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir.join("index.bin")
}

fn small_index<DS: Distance<4> + Send + Sync>(dist: DS) -> Hnsw<4, DS> {
    let index = Hnsw::<4, DS>::new_seeded(4, 8, 16, 1, dist);
    for i in 0..10 {
        index.insert([i as f32, (i % 7) as f32, 0.0, 1.0]);
    }
    index
}

/// Scaled L2 whose serialization fails when `fail` is set, standing in for any
/// error in the middle of a save.
#[derive(Clone, Copy, Default, Deserialize)]
struct Scaled {
    factor: f32,
    #[serde(skip)]
    fail: bool,
}

impl Serialize for Scaled {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if self.fail {
            return Err(serde::ser::Error::custom("refusing to serialize"));
        }
        #[derive(Serialize)]
        struct Repr {
            factor: f32,
        }
        Repr {
            factor: self.factor,
        }
        .serialize(serializer)
    }
}

impl Distance<4> for Scaled {
    fn distance(&self, a: &[f32; 4], b: &[f32; 4]) -> f32 {
        self.factor * L2Squared.distance(a, b)
    }
}

#[test]
fn a_failed_save_keeps_the_previous_snapshot() {
    let path = temp_path("failed-save");
    let good = small_index(Scaled {
        factor: 1.0,
        fail: false,
    });
    good.save(&path).unwrap();

    let failing = small_index(Scaled {
        factor: 2.0,
        fail: true,
    });
    assert!(failing.save(&path).is_err());

    let reloaded = Hnsw::<4, Scaled>::load(&path).expect("previous snapshot must survive");
    let q = [3.0, 3.0, 0.0, 1.0];
    assert_eq!(reloaded.search(&q, 5), good.search(&q, 5));
    let leftovers = fs::read_dir(path.parent().unwrap()).unwrap().count();
    assert_eq!(leftovers, 1, "temporary files left behind");
}

/// A different metric with the same (empty) serialized form as L2Squared.
#[derive(Clone, Copy, Default, Serialize, Deserialize)]
struct Manhattan;

impl Distance<4> for Manhattan {
    fn distance(&self, a: &[f32; 4], b: &[f32; 4]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
    }
}

#[test]
fn loading_with_a_different_metric_is_rejected() {
    let path = temp_path("metric");
    small_index(L2Squared).save(&path).unwrap();
    assert!(
        Hnsw::<4, Manhattan>::load(&path).is_err(),
        "an L2 index was silently loaded as a Manhattan index"
    );
}

#[test]
fn corrupted_snapshots_fail_to_load_instead_of_panicking_later() {
    let path = temp_path("corrupt");
    small_index(L2Squared).save(&path).unwrap();
    let original = fs::read(&path).unwrap();

    for position in 0..original.len() {
        let mut bytes = original.clone();
        bytes[position] ^= 0x5a;
        fs::write(&path, &bytes).unwrap();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            if let Ok(index) = Hnsw::<4>::load(&path) {
                let _ = index.try_search(&[1.0, 1.0, 0.0, 1.0], 5);
                let _ = index.try_insert([2.5, 1.0, 0.0, 1.0]);
                return true;
            }
            false
        }));
        match outcome {
            Err(_) => panic!("byte {position}: corrupted snapshot panicked after loading"),
            Ok(loaded) => assert!(!loaded, "byte {position}: corrupted snapshot was accepted"),
        }
    }
}

#[test]
fn truncated_snapshots_are_rejected() {
    let path = temp_path("truncated");
    small_index(L2Squared).save(&path).unwrap();
    let original = fs::read(&path).unwrap();
    for len in 0..original.len() {
        fs::write(&path, &original[..len]).unwrap();
        assert!(
            Hnsw::<4>::load(&path).is_err(),
            "prefix of {len} bytes loaded"
        );
    }
}
