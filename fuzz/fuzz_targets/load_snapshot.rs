//! Fuzzes `Hnsw::read_snapshot`: no input may panic, and any index that loads
//! must survive searches and inserts.
//!
//! Each input is tried twice: as is (exercising the framing and checksums) and
//! with both CRC-32 fields recomputed, so mutations reach the structural
//! validation instead of stopping at a checksum mismatch.
//!
//! Run with: `cargo +nightly fuzz run load_snapshot fuzz/corpus/load_snapshot`
#![no_main]

use hnsw::{Hnsw, HnswSearcher};
use libfuzzer_sys::fuzz_target;

/// Recomputes the header and payload checksums if the header length is sane.
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

fn exercise(bytes: &[u8]) {
    if let Ok(index) = Hnsw::<4>::read_snapshot(bytes, bytes.len() as u64) {
        let _ = index.try_search(&[0.0; 4], 5);
        let _ = index.try_search_with_ef(&[1.0, -1.0, 0.5, 0.0], 10, 64);
        let _ = index.try_insert([0.5; 4]);
        let _ = index.try_insert_with_id(1 << 40, [0.25; 4]);
        let mut out = Vec::new();
        let _ = index.write_snapshot(&mut out);
    }
}

fuzz_target!(|data: &[u8]| {
    exercise(data);
    let mut fixed = data.to_vec();
    fix_checksums(&mut fixed);
    exercise(&fixed);
});
