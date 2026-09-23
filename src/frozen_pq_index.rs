use hnsw_pq::ProductQuantizer;
use rayon::prelude::*;

use crate::{
    Distance, Hnsw, HnswError, HnswSearcher, L2Squared, check_ef_search, context::SearchContext,
    link::Link, node::Node, nodes_heap_usage_bytes, top_k,
};
use std::{cmp::Reverse, mem::size_of};

/// **Experimental** read-only HNSW graph whose vectors are replaced by product-
/// quantization codes (`Q` bytes per vector instead of `4 * D`).
///
/// Built with [`Hnsw::freeze`], [`Hnsw::freeze_seeded`] or [`Hnsw::freeze_with_pq`]
/// from a squared-L2 index, keeping that index's graph and ids.
///
/// What it does **not** do:
/// - it keeps no original vectors, so there is no exact rescoring: reported
///   distances are asymmetric (ADC) approximations of squared L2, and recall is
///   bounded by the quantization (see [`FrozenPQHnsw::brute_force_adc`]);
/// - it cannot be extended with new vectors;
/// - it cannot be saved or loaded.
///
/// Available with the `experimental-pq` feature; not covered by the v0.1
/// compatibility contract.
pub struct FrozenPQHnsw<const D: usize, const Q: usize> {
    entry_point: usize,
    data: Vec<[u8; Q]>,
    nodes: Vec<Node>,
    dup_next: Vec<usize>,
    ids: Vec<usize>,
    max_layer: usize,
    pq: ProductQuantizer<Q, D>,
}

impl<const D: usize, const Q: usize> FrozenPQHnsw<D, Q> {
    fn from_pq(hnsw: Hnsw<D, L2Squared>, pq: ProductQuantizer<Q, D>) -> Self {
        assert!(
            pq.is_trained(),
            "freeze_with_pq needs a trained ProductQuantizer"
        );
        let storage = hnsw.storage.into_inner().unwrap();
        let (entry_point, max_layer) = hnsw.entry.into_inner().unwrap();
        let data = storage.data.par_iter().map(|v| pq.encode(v)).collect();
        Self {
            entry_point,
            data,
            nodes: storage.nodes,
            dup_next: storage.dup_next,
            ids: storage.ids,
            max_layer,
            pq,
        }
    }

    fn from_hnsw(hnsw: Hnsw<D, L2Squared>, k: usize, seed: Option<u64>) -> Self {
        let mut pq: ProductQuantizer<Q, D> = ProductQuantizer::new(k);
        {
            let storage = hnsw.storage.read().unwrap();
            match seed {
                Some(seed) => pq.fit_seeded(&storage.data, seed),
                None => pq.fit(&storage.data),
            }
        }
        Self::from_pq(hnsw, pq)
    }

    fn search_layer_with_context<'a>(
        &self,
        adc_table: &[f32],
        ep: usize,
        lyr: usize,
        ef: usize,
        ctx: &'a mut SearchContext,
    ) -> &'a [Link] {
        ctx.clear();
        let visited = &mut ctx.visited;
        visited.reset();
        let frontier = &mut ctx.frontier;
        let best = &mut ctx.best;

        let ep_link = Link {
            node_index: ep,
            distance: self.pq.adc_distance(adc_table, &self.data[ep]),
        };
        frontier.push(Reverse(ep_link));
        best.push(ep_link);
        visited.mark_visited(ep);

        while let Some(Reverse(candidate)) = frontier.pop() {
            let furthest_dist = best.peek().map_or(f32::INFINITY, |l| l.distance);
            if candidate.distance > furthest_dist {
                break;
            }
            for neigh in self.nodes[candidate.node_index].layers[lyr]
                .read()
                .unwrap()
                .iter()
            {
                if visited.is_visited(neigh.node_index) {
                    continue;
                }
                visited.mark_visited(neigh.node_index);
                let dist = self
                    .pq
                    .adc_distance(adc_table, &self.data[neigh.node_index]);
                if best.len() == ef && best.peek().is_some_and(|furthest| furthest.distance > dist)
                {
                    best.pop();
                }
                if best.len() < ef {
                    let link = Link {
                        node_index: neigh.node_index,
                        distance: dist,
                    };
                    best.push(link);
                    frontier.push(Reverse(link));
                }
            }
        }

        ctx.consume_best()
    }

    /// Exhaustive ADC search over every code: the best any graph search over this
    /// index can do. Returns `(id, approximate distance)` sorted like `search`.
    pub fn brute_force_adc(&self, q: &[f32; D], k: usize) -> Vec<(usize, f32)> {
        let adc = self.pq.adc_table(q);
        let mut distances: Vec<(usize, f32)> = self
            .data
            .iter()
            .zip(&self.ids)
            .map(|(code, &id)| (id, self.pq.adc_distance(&adc, code)))
            .collect();
        distances.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        distances.truncate(k);
        distances
    }
}

impl<const D: usize> Hnsw<D, L2Squared> {
    /// Trains a `Q`-subquantizer PQ with `k` centroids each on the stored vectors
    /// (random seed) and converts the index into a [`FrozenPQHnsw`].
    ///
    /// # Panics
    /// If `D` is not a multiple of `Q`, `k` is 0 or above 256, or the index holds
    /// fewer than `k` vectors.
    pub fn freeze<const Q: usize>(self, k: usize) -> FrozenPQHnsw<D, Q> {
        FrozenPQHnsw::from_hnsw(self, k, None)
    }

    /// [`Hnsw::freeze`] with reproducible PQ training.
    pub fn freeze_seeded<const Q: usize>(self, k: usize, seed: u64) -> FrozenPQHnsw<D, Q> {
        FrozenPQHnsw::from_hnsw(self, k, Some(seed))
    }

    /// Encodes the stored vectors with an already trained quantizer (for example
    /// one trained on a sample) and converts the index into a [`FrozenPQHnsw`].
    ///
    /// # Panics
    /// If `pq` is not trained.
    pub fn freeze_with_pq<const Q: usize>(self, pq: ProductQuantizer<Q, D>) -> FrozenPQHnsw<D, Q> {
        FrozenPQHnsw::from_pq(self, pq)
    }
}

impl<const D: usize, const Q: usize> HnswSearcher<D> for FrozenPQHnsw<D, Q> {
    fn try_search_with_context(
        &self,
        q: &[f32; D],
        k: usize,
        ef_search: usize,
        ctx: &mut SearchContext,
    ) -> Result<Vec<(usize, f32)>, HnswError> {
        check_ef_search(ef_search)?;
        L2Squared.validate(q)?;
        if k == 0 || self.is_empty() {
            return Ok(Vec::new());
        }

        let adc = self.pq.adc_table(q);
        let mut ep = self.entry_point;
        for lyr in (1..=self.max_layer).rev() {
            // a layer search always returns at least its entry point
            ep = self.search_layer_with_context(&adc, ep, lyr, 1, ctx)[0].node_index;
        }

        let results = self.search_layer_with_context(&adc, ep, 0, ef_search.max(k), ctx);
        Ok(top_k(results, k, &self.dup_next, &self.ids))
    }

    fn memory_usage_bytes(&self) -> usize {
        size_of::<Self>()
            + self.data.capacity() * size_of::<[u8; Q]>()
            + nodes_heap_usage_bytes(&self.nodes)
            + self.dup_next.capacity() * size_of::<usize>()
            + self.ids.capacity() * size_of::<usize>()
            + self.pq.heap_usage_bytes()
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}
