use super::*;
use crate::{HnswSearcher, L2Squared};
use std::sync::{Barrier, mpsc};

fn sample() -> Hnsw<4> {
    let index = Hnsw::<4>::new_seeded(4, 8, 16, 1, L2Squared);
    for i in 0..40 {
        index.insert([i as f32, (i % 7) as f32, (i % 3) as f32, 1.0]);
    }
    // a duplicate group
    index.insert([5.0, 5.0, 2.0, 1.0]);
    index.insert([5.0, 5.0, 2.0, 1.0]);
    index
}

fn bytes_of<DS: Distance<4> + Serialize>(index: &Hnsw<4, DS>) -> Vec<u8> {
    let mut bytes = Vec::new();
    index.write_snapshot(&mut bytes).unwrap();
    bytes
}

fn read(bytes: &[u8]) -> Result<Hnsw<4>, SnapshotError> {
    Hnsw::<4>::read_snapshot(bytes, bytes.len() as u64)
}

fn decode(bytes: &[u8]) -> (Header, Decoded<4>) {
    let mut reader = bytes;
    let (header, _) = Header::read(&mut reader).unwrap();
    let decoded = read_payload::<_, 4>(&mut reader, &header).unwrap();
    (header, decoded)
}

/// Writes `header` and `decoded` with correct lengths and checksums, so only
/// the structural validation can reject the result.
fn encode(mut header: Header, decoded: &Decoded<4>) -> Vec<u8> {
    let mut payload = Vec::new();
    for v in &decoded.vectors {
        for x in v {
            payload.extend_from_slice(&x.to_le_bytes());
        }
    }
    for id in &decoded.ids {
        payload.extend_from_slice(&id.to_le_bytes());
    }
    for next in &decoded.dup_next {
        payload.extend_from_slice(&next.to_le_bytes());
    }
    for layers in &decoded.layers {
        payload.push(layers.len() as u8);
        for links in layers {
            payload.extend_from_slice(&(links.len() as u16).to_le_bytes());
            for link in links {
                payload.extend_from_slice(&(link.node_index as u32).to_le_bytes());
                payload.extend_from_slice(&link.distance.to_le_bytes());
            }
        }
    }
    header.node_count = decoded.vectors.len() as u64;
    header.payload_len = payload.len() as u64;
    let mut bytes = header.encode();
    let crc = crc32fast::hash(&payload);
    bytes.extend_from_slice(&payload);
    bytes.extend_from_slice(&crc.to_le_bytes());
    bytes
}

fn mutated(mutate: impl FnOnce(&mut Header, &mut Decoded<4>)) -> Result<Hnsw<4>, SnapshotError> {
    let (mut header, mut decoded) = decode(&bytes_of(&sample()));
    mutate(&mut header, &mut decoded);
    read(&encode(header, &decoded))
}

fn assert_corrupt(result: Result<Hnsw<4>, SnapshotError>, needle: &str) {
    match result {
        Err(SnapshotError::Corrupt(reason)) => {
            assert!(
                reason.contains(needle),
                "{reason:?} does not mention {needle:?}"
            )
        }
        Err(other) => panic!("expected Corrupt({needle:?}), got {other:?}"),
        Ok(_) => panic!("expected Corrupt({needle:?}), got a loaded index"),
    }
}

/// First graph node with a link on layer 0, and that link's position.
fn some_link(decoded: &Decoded<4>) -> (usize, usize) {
    let node = (0..decoded.layers.len())
        .find(|&i| decoded.layers[i].first().is_some_and(|l| l.len() >= 2))
        .unwrap();
    (node, 0)
}

#[test]
fn reencoding_reproduces_the_saved_bytes() {
    let bytes = bytes_of(&sample());
    let (header, decoded) = decode(&bytes);
    assert_eq!(encode(header, &decoded), bytes);
    let loaded = read(&bytes).unwrap();
    assert_eq!(bytes_of(&loaded), bytes, "load then save must be lossless");
}

#[test]
fn structural_corruption_with_valid_checksums_is_rejected() {
    assert_corrupt(
        mutated(|_, d| {
            let (node, link) = some_link(d);
            d.layers[node][0][link].node_index = 10_000;
        }),
        "links to 10000",
    );
    assert_corrupt(
        mutated(|_, d| {
            let (node, link) = some_link(d);
            d.layers[node][0][link].node_index = node;
        }),
        "itself",
    );
    assert_corrupt(
        mutated(|_, d| {
            let (node, _) = some_link(d);
            let first = d.layers[node][0][0];
            d.layers[node][0][1] = first;
        }),
        "twice",
    );
    assert_corrupt(
        mutated(|_, d| {
            let (node, link) = some_link(d);
            d.layers[node][0][link].distance = f32::NAN;
        }),
        "non-finite link distance",
    );
    assert_corrupt(mutated(|_, d| d.vectors[3][2] = f32::INFINITY), "vector 3");
    assert_corrupt(
        mutated(|h, _| h.entry_point = 10_000),
        "entry point is not a graph node",
    );
    assert_corrupt(
        mutated(|h, d| {
            // a graph node that is not on the top layer
            h.entry_point = (0..d.layers.len())
                .find(|&i| d.layers[i].len() == 1)
                .unwrap() as u64;
        }),
        "top layer",
    );
    assert_corrupt(mutated(|h, _| h.max_layer += 1), "top layer");
    assert_corrupt(
        mutated(|_, d| {
            // link to a node that does not exist on layer 1
            let upper = (0..d.layers.len())
                .find(|&i| d.layers[i].len() > 1 && !d.layers[i][1].is_empty())
                .unwrap();
            let flat = (0..d.layers.len())
                .find(|&i| d.layers[i].len() == 1)
                .unwrap();
            d.layers[upper][1][0].node_index = flat;
        }),
        "has no layer 1",
    );
    assert_corrupt(mutated(|_, d| d.ids[5] = 6), "invalid or repeated id");
    assert_corrupt(mutated(|_, d| d.dup_next[0] = 1_000), "outside the index");
    assert_corrupt(
        mutated(|_, d| {
            // point a second graph node at the existing member
            let member = (0..d.layers.len())
                .find(|&i| d.layers[i].is_empty())
                .unwrap();
            let other = (0..d.layers.len())
                .find(|&i| !d.layers[i].is_empty() && d.dup_next[i] == END_OF_LIST)
                .unwrap();
            d.dup_next[other] = member as u64;
        }),
        "",
    );
    assert_corrupt(
        mutated(|_, d| {
            let member = (0..d.layers.len())
                .find(|&i| d.layers[i].is_empty())
                .unwrap();
            d.vectors[member][0] += 1.0;
        }),
        "differs from its graph node",
    );
    assert_corrupt(
        mutated(|_, d| {
            // detach the member from its list
            for next in d.dup_next.iter_mut() {
                *next = END_OF_LIST;
            }
        }),
        "neither linked nor a duplicate",
    );
    assert_corrupt(mutated(|h, _| h.id_mode = 9), "invalid id mode");
}

#[test]
fn oversized_layer_and_link_counts_are_rejected_before_allocating() {
    let err = mutated(|h, _| h.m = 2).err().unwrap();
    assert!(
        matches!(err, SnapshotError::Corrupt(ref r) if r.contains("more than the limit")),
        "{err:?}"
    );
    let err = mutated(|h, _| h.m = 1).err().unwrap();
    assert!(matches!(err, SnapshotError::InvalidConfig(_)), "{err:?}");
    let err = mutated(|_, d| d.layers[0] = vec![Vec::new(); MAX_LAYERS + 1])
        .err()
        .unwrap();
    assert!(
        matches!(err, SnapshotError::Corrupt(ref r) if r.contains("layers")),
        "{err:?}"
    );

    // a header claiming 2^40 nodes in a tiny payload never allocates them
    let (mut header, _) = decode(&bytes_of(&sample()));
    header.node_count = 1 << 40;
    let mut bytes = header.encode();
    bytes.extend_from_slice(&[0; 64]);
    let err = Hnsw::<4>::read_snapshot(&bytes[..], u64::MAX)
        .err()
        .unwrap();
    assert!(
        matches!(err, SnapshotError::Corrupt(ref r) if r.contains("cannot fit")),
        "{err:?}"
    );

    let err = Hnsw::<4>::read_snapshot(&bytes_of(&sample())[..], 100)
        .err()
        .unwrap();
    assert!(matches!(err, SnapshotError::TooLarge { .. }), "{err:?}");
}

#[test]
fn bad_metric_parameters_are_rejected() {
    let (mut header, decoded) = decode(&bytes_of(&sample()));
    header.metric_params = b"{\"not\": ".to_vec();
    let err = read(&encode(header, &decoded)).err().unwrap();
    assert!(matches!(err, SnapshotError::Metric(_)), "{err:?}");
}

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("hnsw-unit-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn every_failed_save_step_keeps_the_previous_snapshot() {
    let dir = temp_dir("steps");
    let path = dir.join("index.bin");
    let old = Hnsw::<4>::new_seeded(4, 8, 16, 1, L2Squared);
    old.insert([1.0; 4]);
    old.save(&path).unwrap();
    let old_bytes = fs::read(&path).unwrap();

    let new = sample();
    let steps = [
        SaveHooks {
            fail_at: Some(SaveStep::CreateTemp),
            ..Default::default()
        },
        SaveHooks {
            fail_at: Some(SaveStep::Write),
            ..Default::default()
        },
        SaveHooks {
            fail_after_bytes: Some(10),
            ..Default::default()
        },
        SaveHooks {
            fail_after_bytes: Some(500),
            ..Default::default()
        },
        SaveHooks {
            fail_at: Some(SaveStep::Sync),
            ..Default::default()
        },
        SaveHooks {
            fail_at: Some(SaveStep::Replace),
            ..Default::default()
        },
    ];
    for hooks in &steps {
        assert!(new.save_with(&path, hooks).is_err());
        assert_eq!(fs::read(&path).unwrap(), old_bytes);
        assert_eq!(Hnsw::<4>::load(&path).unwrap().len(), 1);
        let entries: Vec<_> = fs::read_dir(&dir).unwrap().collect();
        assert_eq!(
            entries.len(),
            1,
            "temporary file left after {:?}",
            hooks.fail_at
        );
    }

    new.save(&path).unwrap();
    assert_eq!(Hnsw::<4>::load(&path).unwrap().len(), new.len());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn saving_blocks_writers_but_not_readers() {
    let dir = temp_dir("blocking");
    let path = dir.join("index.bin");
    let index = sample();
    let (locked_tx, locked_rx) = mpsc::channel::<()>();
    let release = Barrier::new(2);
    let while_locked = || {
        locked_tx.send(()).unwrap();
        release.wait();
    };
    let hooks = SaveHooks {
        while_locked: Some(&while_locked),
        ..Default::default()
    };

    std::thread::scope(|s| {
        let saver = s.spawn(|| index.save_with(&path, &hooks));
        locked_rx.recv().unwrap();

        // the snapshot holds its locks: a search completes...
        assert_eq!(index.search(&[1.0, 1.0, 1.0, 1.0], 1).len(), 1);
        // ...but an insert waits until the snapshot has been written
        let (done_tx, done_rx) = mpsc::channel();
        let index = &index;
        let inserter = s.spawn(move || {
            index.insert([100.0; 4]);
            done_tx.send(()).unwrap();
        });
        assert!(
            done_rx
                .recv_timeout(std::time::Duration::from_millis(200))
                .is_err(),
            "insert completed while a snapshot was being written"
        );
        release.wait();
        saver.join().unwrap().unwrap();
        done_rx.recv().unwrap();
        inserter.join().unwrap();
    });

    // the snapshot is the state before the blocked insert
    assert_eq!(Hnsw::<4>::load(&path).unwrap().len(), index.len() - 1);
    fs::remove_dir_all(dir).unwrap();
}
