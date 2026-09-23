//! Stable-toolchain counterpart of `fuzz/fuzz_targets/load_snapshot.rs`:
//! replays the checked-in corpus and runs seeded random mutations, with and
//! without recomputed checksums. Nothing may panic.
//!
//! The corpus seeds are regenerated from deterministic builds; if the snapshot
//! format changes, rerun with `HNSW_WRITE_FUZZ_CORPUS=1` to update them.

use hnsw::{Hnsw, HnswSearcher, L2Squared};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::{fs, path::PathBuf};

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fuzz/corpus/load_snapshot")
}

fn snapshot(index: &Hnsw<4>) -> Vec<u8> {
    let mut bytes = Vec::new();
    index.write_snapshot(&mut bytes).unwrap();
    bytes
}

fn line(i: usize) -> [f32; 4] {
    [i as f32 * 0.5, (i % 5) as f32, (i % 3) as f32 - 1.0, 0.25]
}

/// Deterministic seed snapshots (sequential inserts into seeded indexes).
fn seeds() -> Vec<(&'static str, Vec<u8>)> {
    let empty = Hnsw::<4>::new_seeded(4, 8, 16, 1, L2Squared);

    let positional = Hnsw::<4>::new_seeded(4, 8, 16, 2, L2Squared);
    (0..40).for_each(|i| {
        positional.insert(line(i));
    });

    let explicit = Hnsw::<4>::new_seeded(4, 8, 16, 3, L2Squared);
    for i in 0..30 {
        explicit.insert_with_id(1_000 + i, line(i % 20));
    }

    // M = 2 gives the most levels per node
    let deep = Hnsw::<4>::new_seeded(2, 3, 8, 4, L2Squared);
    (0..60).for_each(|i| {
        deep.insert(line(i));
    });

    let mut legacy = 16u64.to_le_bytes().to_vec();
    legacy.extend_from_slice(&32u64.to_le_bytes());
    legacy.extend_from_slice(&[0; 48]);

    vec![
        ("empty.snap", snapshot(&empty)),
        ("positional.snap", snapshot(&positional)),
        ("explicit-duplicates.snap", snapshot(&explicit)),
        ("deep.snap", snapshot(&deep)),
        ("legacy-unversioned.bin", legacy),
    ]
}

fn fix_checksums(bytes: &mut [u8]) {
    if bytes.len() < 20 {
        return;
    }
    let header_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    if header_len < 20 || header_len + 4 > bytes.len() {
        return;
    }
    let crc = crc32fast::hash(&bytes[..header_len - 4]);
    bytes[header_len - 4..header_len].copy_from_slice(&crc.to_le_bytes());
    let end = bytes.len() - 4;
    let crc = crc32fast::hash(&bytes[header_len..end]);
    bytes[end..].copy_from_slice(&crc.to_le_bytes());
}

/// Mirrors the fuzz target; returns whether the input loaded.
fn exercise(bytes: &[u8]) -> bool {
    match Hnsw::<4>::read_snapshot(bytes, bytes.len() as u64) {
        Ok(index) => {
            let _ = index.try_search(&[0.0; 4], 5);
            let _ = index.try_search_with_ef(&[1.0, -1.0, 0.5, 0.0], 10, 64);
            let _ = index.try_insert([0.5; 4]);
            let _ = index.try_insert_with_id(1 << 40, [0.25; 4]);
            let mut out = Vec::new();
            let _ = index.write_snapshot(&mut out);
            true
        }
        Err(_) => false,
    }
}

#[test]
fn corpus_seeds_are_current() {
    let dir = corpus_dir();
    if std::env::var_os("HNSW_WRITE_FUZZ_CORPUS").is_some() {
        fs::create_dir_all(&dir).unwrap();
        for (name, bytes) in seeds() {
            fs::write(dir.join(name), bytes).unwrap();
        }
    }
    for (name, bytes) in seeds() {
        let stored = fs::read(dir.join(name))
            .unwrap_or_else(|_| panic!("missing corpus seed {name}; set HNSW_WRITE_FUZZ_CORPUS=1"));
        assert_eq!(
            stored, bytes,
            "corpus seed {name} is stale; set HNSW_WRITE_FUZZ_CORPUS=1"
        );
        assert_eq!(exercise(&bytes), name.ends_with(".snap"), "{name}");
    }
}

#[test]
fn corpus_replays_without_panicking() {
    let mut replayed = 0;
    for entry in fs::read_dir(corpus_dir()).unwrap() {
        let mut bytes = fs::read(entry.unwrap().path()).unwrap();
        exercise(&bytes);
        fix_checksums(&mut bytes);
        exercise(&bytes);
        replayed += 1;
    }
    assert!(replayed >= 5);
}

fn mutate(rng: &mut StdRng, bytes: &mut Vec<u8>) {
    for _ in 0..rng.random_range(1..=4) {
        if bytes.is_empty() {
            bytes.push(rng.random());
            continue;
        }
        let at = rng.random_range(0..bytes.len());
        match rng.random_range(0..6) {
            0 => bytes[at] ^= 1 << rng.random_range(0..8),
            1 => bytes[at] = rng.random(),
            2 => bytes.insert(at, rng.random()),
            3 => {
                bytes.remove(at);
            }
            4 => bytes.truncate(at),
            _ => {
                // interesting integer values over a 1-, 2-, 4- or 8-byte field
                let width = [1, 2, 4, 8][rng.random_range(0..4)].min(bytes.len() - at);
                let value: u64 = [0, 1, 0xff, 0xffff, u32::MAX as u64, u64::MAX, 1 << 40]
                    [rng.random_range(0..7)];
                bytes[at..at + width].copy_from_slice(&value.to_le_bytes()[..width]);
            }
        }
    }
}

#[test]
fn seeded_mutations_never_panic() {
    let seeds: Vec<Vec<u8>> = seeds().into_iter().map(|(_, bytes)| bytes).collect();
    let mut rng = StdRng::seed_from_u64(0xf022);
    let (mut raw_loaded, mut fixed_loaded) = (0, 0);
    for _ in 0..3_000 {
        let original = seeds[rng.random_range(0..seeds.len())].clone();
        let mut bytes = original.clone();
        mutate(&mut rng, &mut bytes);
        let ok = exercise(&bytes);
        // without recomputed checksums only mutations that changed nothing load
        assert!(
            !ok || bytes == original,
            "changed bytes passed the checksums"
        );
        raw_loaded += usize::from(ok);
        fix_checksums(&mut bytes);
        fixed_loaded += usize::from(exercise(&bytes));
    }
    println!("loaded after mutation: raw {raw_loaded}, with fixed checksums {fixed_loaded}");
}
