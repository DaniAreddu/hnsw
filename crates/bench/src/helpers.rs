use std::{cmp::Ordering, collections::BinaryHeap, time::Duration};

use hnsw::{Distance, L2Squared};

#[derive(Debug, Clone, Copy, PartialEq)]
struct ScoredId {
    id: usize,
    distance: f32,
}

impl Eq for ScoredId {}

impl PartialOrd for ScoredId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredId {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.id.cmp(&other.id))
    }
}

pub(crate) fn compute_ground_truth<const D: usize>(
    base: &[[f32; D]],
    queries: &[[f32; D]],
    k: usize,
) -> Vec<Vec<usize>> {
    queries
        .iter()
        .map(|query| exact_top_k(base, query, k))
        .collect()
}

fn exact_top_k<const D: usize>(base: &[[f32; D]], query: &[f32; D], k: usize) -> Vec<usize> {
    let mut heap = BinaryHeap::with_capacity(k);

    for (id, vector) in base.iter().enumerate() {
        let candidate = ScoredId {
            id,
            distance: L2Squared.distance(query, vector),
        };

        if heap.len() < k {
            heap.push(candidate);
            continue;
        }

        if heap
            .peek()
            .is_some_and(|worst| worst.distance > candidate.distance)
        {
            heap.pop();
            heap.push(candidate);
        }
    }

    heap.into_sorted_vec()
        .into_iter()
        .map(|item| item.id)
        .collect()
}

pub(crate) fn recall_at_k(expected: &[usize], results: &[(usize, f32)]) -> f64 {
    if expected.is_empty() {
        return 1.0;
    }

    let actual: Vec<usize> = results.iter().map(|(id, _)| *id).collect();
    let hits = expected.iter().filter(|id| actual.contains(id)).count();

    hits as f64 / expected.len() as f64
}

pub(crate) fn percentile(sorted: &[Duration], p: f64) -> Duration {
    let rank = (p.clamp(0.0, 1.0) * sorted.len() as f64).ceil() as usize;
    let index = rank.saturating_sub(1).min(sorted.len() - 1);
    sorted[index]
}

pub(crate) fn duration_average(total: Duration, count: usize) -> Duration {
    Duration::from_secs_f64(total.as_secs_f64() / count as f64)
}

pub(crate) fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

pub(crate) fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}
