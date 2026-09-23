use pq::ProductQuantizer;
use rayon::prelude::*;

use crate::{
    Distance, Hnsw, HnswError, HnswSearcher, L2Squared, check_ef_search, context::SearchContext,
    link::Link, node::Node, nodes_heap_usage_bytes,
};
use std::{cmp::Reverse, mem::size_of};

#[allow(non_snake_case)]
pub struct FrozenPQHnsw<const D: usize, const Q: usize> {
    entry_point: usize,
    data: Vec<[u8; Q]>,
    nodes: Vec<Node>,
    max_layer: usize,
    pq: ProductQuantizer<Q, D>,
}

impl<const D: usize, const Q: usize> FrozenPQHnsw<D, Q> {
    pub(crate) fn from_pq(
        hnsw: Hnsw<D, L2Squared>,
        quantized_data: Vec<[u8; Q]>,
        pq: ProductQuantizer<Q, D>,
    ) -> Self {
        assert_eq!(
            quantized_data.len(),
            hnsw.storage.read().unwrap().nodes.len(),
            "quantized data length must match HNSW index length"
        );
        Self {
            entry_point: hnsw.entry.read().unwrap().0,
            data: quantized_data,
            nodes: std::mem::take(&mut hnsw.storage.write().unwrap().nodes),
            max_layer: hnsw.entry.read().unwrap().1,
            pq,
        }
    }

    pub(crate) fn from_hnsw(hnsw: Hnsw<D, L2Squared>, k: usize) -> Self {
        let mut pq: ProductQuantizer<Q, D> = ProductQuantizer::new(k);
        let hnsw_data = std::mem::take(&mut hnsw.storage.write().unwrap().data);
        pq.fit(&hnsw_data);

        Self::from_pq(
            hnsw,
            hnsw_data.into_par_iter().map(|v| pq.encode(&v)).collect(),
            pq,
        )
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

    pub fn brute_force_adc(&self, q: &[f32; D], k: usize) -> Vec<(usize, f32)> {
        let adc = self.pq.adc_table(q);
        let mut distances: Vec<(usize, f32)> = self
            .data
            .iter()
            .enumerate()
            .map(|(id, code)| (id, self.pq.adc_distance(&adc, code)))
            .collect();
        distances.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        distances.truncate(k);
        distances
    }
}

impl<const D: usize> Hnsw<D, L2Squared> {
    pub fn freeze<const Q: usize>(self, k: usize) -> FrozenPQHnsw<D, Q> {
        FrozenPQHnsw::from_hnsw(self, k)
    }
    pub fn freeze_with_pq<const Q: usize>(
        self,
        pq: ProductQuantizer<Q, D>,
        quantized_data: Vec<[u8; Q]>,
    ) -> FrozenPQHnsw<D, Q> {
        FrozenPQHnsw::from_pq(self, quantized_data, pq)
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
        // take k best from final layer search
        Ok(results[..k.min(results.len())]
            .iter()
            .map(|l| (l.node_index, l.distance))
            .collect())
    }

    fn memory_usage_bytes(&self) -> usize {
        size_of::<Self>()
            + self.data.capacity() * size_of::<[u8; Q]>()
            + nodes_heap_usage_bytes(&self.nodes)
            + self.pq.heap_usage_bytes()
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}
