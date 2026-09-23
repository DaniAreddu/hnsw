use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashSet},
};

use crate::link::Link;

/// Reusable scratch buffers for searches; see [`crate::HnswSearcher::search_context`].
pub struct SearchContext {
    pub(crate) frontier: BinaryHeap<Reverse<Link>>,
    pub(crate) best: BinaryHeap<Link>,
    pub(crate) visited: VisitedSet,
    /// First non-finite distance seen since the caller last reset it.
    pub(crate) non_finite_distance: Option<f32>,
    results: Vec<Link>,
}

pub(crate) struct SelectContext {
    pub(crate) pq: BinaryHeap<Reverse<Link>>,
    pub(crate) discarded: BinaryHeap<Reverse<Link>>,
    pub(crate) best: Vec<Link>,
    pub(crate) visited: VisitedSet,
}

/// Reusable scratch buffers for insertion; see [`crate::Hnsw::insert_context`].
pub struct InsertContext {
    pub(crate) select_ctx: SelectContext,
    pub(crate) search_ctx: SearchContext,
}

impl SearchContext {
    pub(crate) fn one_off(cap: usize) -> Self {
        Self::with_visited(VisitedSet::hash(cap), cap)
    }

    pub(crate) fn reusable(nodes_cnt: usize) -> Self {
        Self::with_visited(VisitedSet::epoch(nodes_cnt), 32)
    }

    pub(crate) fn reusable_with_capacity(nodes_cnt: usize, cap: usize) -> Self {
        Self::with_visited(VisitedSet::epoch(nodes_cnt), cap)
    }

    fn with_visited(visited: VisitedSet, cap: usize) -> Self {
        assert!(cap > 0, "search context capacity must be > 0");
        Self {
            frontier: BinaryHeap::new(),
            best: BinaryHeap::with_capacity(cap),
            results: Vec::with_capacity(cap),
            visited,
            non_finite_distance: None,
        }
    }

    pub(crate) fn clear(&mut self) {
        self.frontier.clear();
        self.best.clear();
        self.results.clear();
    }

    pub(crate) fn consume_best(&mut self) -> &[Link] {
        while let Some(p) = self.best.pop() {
            self.results.push(p);
        }
        self.results.reverse();
        &self.results
    }
}

impl SelectContext {
    fn one_off(visited_capacity: usize, max_connections: usize) -> Self {
        Self::with_visited(VisitedSet::hash(visited_capacity), max_connections)
    }

    fn reusable(nodes_cnt: usize, max_connections: usize) -> Self {
        Self::with_visited(VisitedSet::epoch(nodes_cnt), max_connections)
    }

    fn with_visited(visited: VisitedSet, max_connections: usize) -> Self {
        Self {
            pq: BinaryHeap::new(),
            discarded: BinaryHeap::new(),
            best: Vec::with_capacity(max_connections),
            visited,
        }
    }

    pub(crate) fn clear(&mut self) {
        self.pq.clear();
        self.discarded.clear();
        self.best.clear();
    }

    pub(crate) fn consume_pq(&mut self) -> Vec<Link> {
        let mut out = Vec::with_capacity(self.pq.len());
        while let Some(Reverse(p)) = self.pq.pop() {
            out.push(p);
        }
        out
    }

    pub(crate) fn consume_best(&mut self) -> Vec<Link> {
        self.best.to_vec()
    }
}

impl InsertContext {
    pub(crate) fn one_off(ef_construction: usize, max_connections: usize) -> Self {
        Self {
            select_ctx: SelectContext::one_off(ef_construction, max_connections),
            search_ctx: SearchContext::one_off(ef_construction),
        }
    }

    pub(crate) fn reusable(
        nodes_cnt: usize,
        ef_construction: usize,
        max_connections: usize,
    ) -> Self {
        Self {
            select_ctx: SelectContext::reusable(nodes_cnt, max_connections),
            search_ctx: SearchContext::reusable_with_capacity(nodes_cnt, ef_construction),
        }
    }
}

pub(crate) enum VisitedSet {
    Epoch(EpochVisitedSet),
    Hash(HashSet<usize>),
}

pub(crate) struct EpochVisitedSet {
    node_epoch: Vec<usize>,
    epoch: usize,
}

impl EpochVisitedSet {
    fn grow_if_needed(&mut self, node: usize) {
        if node >= self.node_epoch.len() {
            self.node_epoch.resize(node + 1, 0);
        }
    }
}

impl VisitedSet {
    fn epoch(nodes_cnt: usize) -> Self {
        Self::Epoch(EpochVisitedSet {
            node_epoch: vec![0; nodes_cnt],
            epoch: 0,
        })
    }

    fn hash(capacity: usize) -> Self {
        Self::Hash(HashSet::with_capacity(capacity))
    }

    pub(crate) fn reset(&mut self) {
        match self {
            Self::Epoch(visited) => {
                visited.epoch = visited.epoch.wrapping_add(1);
                if visited.epoch == 0 {
                    visited.node_epoch.fill(0);
                    visited.epoch = 1;
                }
            }
            Self::Hash(visited) => visited.clear(),
        }
    }

    pub(crate) fn is_visited(&mut self, node: usize) -> bool {
        match self {
            Self::Epoch(visited) => {
                visited.grow_if_needed(node);
                visited.node_epoch[node] == visited.epoch
            }
            Self::Hash(visited) => visited.contains(&node),
        }
    }

    pub(crate) fn mark_visited(&mut self, node: usize) {
        match self {
            Self::Epoch(visited) => {
                visited.grow_if_needed(node);
                visited.node_epoch[node] = visited.epoch;
            }
            Self::Hash(visited) => {
                visited.insert(node);
            }
        }
    }
}
