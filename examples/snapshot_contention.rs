//! Measures how `Hnsw::save` affects concurrent searches and inserts.
//!
//! `cargo run --release --example snapshot_contention -- [vectors] [dir]`
//!
//! Builds a seeded random 128-D index, then compares search and insert
//! latencies while idle with those observed while a snapshot is written.

use hnsw::{Hnsw, HnswSearcher, L2Squared};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

const D: usize = 128;

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let n: usize = args.next().map_or(Ok(100_000), |a| a.parse())?;
    let dir = args.next().map_or_else(std::env::temp_dir, Into::into);
    let path = dir.join("hnsw-snapshot-contention.bin");

    let mut rng = StdRng::seed_from_u64(1);
    let mut vector = move || -> [f32; D] { std::array::from_fn(|_| rng.random_range(-1.0..1.0)) };
    let base: Vec<[f32; D]> = (0..n).map(|_| vector()).collect();
    let queries: Vec<[f32; D]> = (0..1_000).map(|_| vector()).collect();
    let extra: Vec<[f32; D]> = (0..100_000).map(|_| vector()).collect();

    let mut index = Hnsw::<D>::new_seeded(16, 32, 100, 7, L2Squared);
    let start = Instant::now();
    index.build_parallel(&base, None);
    println!("built {n} x {D} in {:.1} s", start.elapsed().as_secs_f64());

    let measure = |saving: bool| {
        let stop = AtomicBool::new(false);
        let (searches, inserts, save_time) = std::thread::scope(|s| {
            let reader = s.spawn(|| {
                let mut ctx = index.search_context();
                let mut lat = Vec::new();
                for q in queries.iter().cycle() {
                    if stop.load(Ordering::Acquire) {
                        break;
                    }
                    let t = Instant::now();
                    std::hint::black_box(index.search_with_context(q, 10, 64, &mut ctx));
                    lat.push(t.elapsed());
                }
                lat
            });
            let writer = s.spawn(|| {
                let mut ctx = index.insert_context();
                let mut lat = Vec::new();
                for v in &extra {
                    if stop.load(Ordering::Acquire) {
                        break;
                    }
                    let t = Instant::now();
                    index.insert_with_context(*v, &mut ctx);
                    lat.push(t.elapsed());
                }
                lat
            });
            std::thread::sleep(Duration::from_millis(300));
            let t = Instant::now();
            if saving {
                index.save(&path).expect("save");
            } else {
                std::thread::sleep(Duration::from_millis(1_500));
            }
            let window = t.elapsed();
            std::thread::sleep(Duration::from_millis(300));
            stop.store(true, Ordering::Release);
            (reader.join().unwrap(), writer.join().unwrap(), window)
        });
        let (mut searches, mut inserts) = (searches, inserts);
        searches.sort_unstable();
        inserts.sort_unstable();
        (searches, inserts, save_time)
    };

    println!();
    println!(
        "| phase | window | searches | search p50 | search p99 | search max | inserts | insert p50 | insert max |"
    );
    println!("| --- | --- | --- | --- | --- | --- | --- | --- | --- |");
    for (label, saving) in [("idle", false), ("during save", true)] {
        let (s, i, window) = measure(saving);
        println!(
            "| {label} | {:.0} ms | {} | {:.3} ms | {:.3} ms | {:.1} ms | {} | {:.3} ms | {:.1} ms |",
            ms(window),
            s.len(),
            ms(percentile(&s, 0.5)),
            ms(percentile(&s, 0.99)),
            ms(*s.last().unwrap_or(&Duration::ZERO)),
            i.len(),
            ms(percentile(&i, 0.5)),
            ms(*i.last().unwrap_or(&Duration::ZERO)),
        );
    }
    let size = std::fs::metadata(&path)?.len();
    let t = Instant::now();
    let loaded = Hnsw::<D>::load(&path)?;
    println!(
        "\nsnapshot: {:.1} MiB, {} vectors; load + validation {:.2} s",
        size as f64 / (1 << 20) as f64,
        loaded.len(),
        t.elapsed().as_secs_f64(),
    );
    std::fs::remove_file(&path)?;
    Ok(())
}
