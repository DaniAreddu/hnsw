//! Uses only the published API of the packaged `hnsw` crate.

use hnsw::{Hnsw, HnswError, HnswSearcher, IdMode, L2Squared, SnapshotError};

fn vector(i: usize) -> [f32; 4] {
    let x = i as f32;
    [x.sin(), (x * 0.7).cos(), (x * 0.3).sin(), (i % 11) as f32 * 0.1]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let index = Hnsw::<4>::try_new_seeded(8, 16, 64, 7, L2Squared)?;
    for i in 0..1_000 {
        index.try_insert_with_id(10_000 + i, vector(i))?;
    }
    assert_eq!(index.id_mode(), Some(IdMode::Explicit));
    assert_eq!(index.len(), 1_000);

    let hits = index.try_search_with_ef(&vector(123), 5, 64)?;
    assert_eq!(hits[0], (10_123, 0.0));
    assert!(matches!(
        index.try_insert_with_id(10_000, vector(0)),
        Err(HnswError::DuplicateId { id: 10_000 })
    ));
    assert!(matches!(
        index.try_search(&[f32::NAN; 4], 1),
        Err(HnswError::NonFiniteComponent { .. })
    ));

    let dir = std::env::temp_dir().join(format!("hnsw-consumer-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("index.hnsw");
    index.save(&path)?;
    let loaded = Hnsw::<4>::load(&path)?;
    assert_eq!(loaded.try_search_with_ef(&vector(123), 5, 64)?, hits);

    let mut bytes = std::fs::read(&path)?;
    let middle = bytes.len() / 2;
    bytes[middle] ^= 0xff;
    std::fs::write(&path, &bytes)?;
    assert!(matches!(
        Hnsw::<4>::load(&path),
        Err(SnapshotError::PayloadChecksumMismatch)
    ));
    std::fs::remove_dir_all(&dir)?;

    #[cfg(feature = "pq")]
    {
        let frozen = loaded.freeze_seeded::<2>(16, 1);
        let approx = frozen.try_search_with_ef(&vector(123), 5, 64)?;
        assert_eq!(approx.len(), 5);
        assert!(approx.iter().all(|&(id, _)| (10_000..11_000).contains(&id)));
        println!("experimental-pq: ok");
    }

    println!("packaged hnsw consumer: ok");
    Ok(())
}
