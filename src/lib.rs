//! Hierarchical Navigable Small World (HNSW) approximate nearest-neighbor index
//! over fixed-size `[f32; D]` vectors.
//!
//! ```
//! use hnsw::{Hnsw, HnswError, HnswSearcher, L2Squared};
//!
//! # fn main() -> Result<(), HnswError> {
//! // M = 16 links per node, M0 = 32 on layer 0, ef_construction = 128, seed = 42
//! let index = Hnsw::<2>::try_new_seeded(16, 32, 128, 42, L2Squared)?;
//! index.try_insert([0.0, 0.0])?;
//! index.try_insert([3.0, 3.0])?;
//! index.try_insert([4.0, 4.0])?;
//!
//! // (id, squared distance), nearest first
//! let hits = index.try_search_with_ef(&[1.0, 1.0], 2, 32)?;
//! assert_eq!(hits, vec![(0, 2.0), (1, 8.0)]);
//!
//! // invalid input is an error, not a panic or a corrupted graph
//! assert!(matches!(
//!     index.try_insert([f32::NAN, 0.0]),
//!     Err(HnswError::NonFiniteComponent { component: 0, .. })
//! ));
//! assert!(index.try_search_with_ef(&[1.0, 1.0], 2, 0).is_err());
//! # Ok(())
//! # }
//! ```
//!
//! Every panicking method (`new`, `insert`, `search`, ...) has a `try_*`
//! counterpart and panics only where that counterpart returns an error.

use rand::{distr::Open01, prelude::*};

use crate::{
    context::SelectContext,
    link::Link,
    node::{Node, nodes_heap_usage_bytes},
};
use std::{
    cmp::Reverse,
    collections::{HashMap, HashSet, hash_map::Entry},
    mem::size_of,
    num::NonZeroUsize,
    sync::{
        Mutex, RwLock,
        atomic::{AtomicU8, Ordering},
    },
};

mod context;
mod disk;
mod dist;
mod error;
#[cfg(feature = "experimental-pq")]
mod frozen_pq_index;
mod link;
mod node;
#[cfg(test)]
mod tests;

pub use context::{InsertContext, SearchContext};
pub use dist::{Distance, L2Squared, check_finite};
pub use error::HnswError;

/// **Experimental** product-quantized index (`experimental-pq` feature).
///
/// Not covered by the v0.1 compatibility contract: [`pq::FrozenPQHnsw`] keeps no
/// original vectors (no exact rescoring), cannot be extended, and cannot be
/// saved or loaded.
#[cfg(feature = "experimental-pq")]
pub mod pq {
    pub use crate::frozen_pq_index::FrozenPQHnsw;
    pub use hnsw_pq::{ProductQuantizer, sdc_distance};
}

/// Largest accepted `M` and `M0`.
pub const MAX_CONNECTIONS: usize = 4096;
/// Largest accepted `ef_construction`.
pub const MAX_EF_CONSTRUCTION: usize = 1 << 16;
/// Search effort used by [`HnswSearcher::search`] and [`HnswSearcher::try_search`].
pub const DEFAULT_EF_SEARCH: usize = 32;

/// An HNSW index over `[f32; D]` vectors with distance metric `DS`.
///
/// # Ids
/// An index is *positional* or *explicit*, fixed by its first insert (see
/// [`IdMode`]). Positional inserts (`insert`, `build_parallel`, ...) get the next
/// insertion position as id. Explicit inserts (`insert_with_id`, ...) use the
/// caller's `usize` id, which must be unique: a repeated id is rejected with
/// [`HnswError::DuplicateId`], also when two threads race on it. Ids are saved
/// with the index. There is no delete, update or upsert.
///
/// # Concurrency
/// Searches and inserts may run concurrently through `&Hnsw`. Once an insert call
/// has returned, its vector is fully linked and later searches can find it
/// (subject to the usual approximate recall); a search that overlaps an insert
/// may or may not see it. `len()` counts vectors as soon
/// as they are stored, which can be slightly before they are reachable.
///
/// # Determinism
/// With a fixed seed, sequential inserts in a fixed order always build the same
/// graph, also across save/load. `build_parallel` and `extend_parallel` depend
/// on thread scheduling and are not reproducible.
///
/// # Duplicates
/// A vector equal to an existing vector found by the insertion search is stored
/// with its own id but shares that vector's graph node, so every copy is
/// returned together. Copies inserted concurrently may instead become separate
/// graph nodes.
#[allow(non_snake_case)]
pub struct Hnsw<const D: usize, DS = L2Squared> {
    M: usize,
    M0: usize,
    pub(crate) ef_construction: usize,
    pub(crate) storage: RwLock<Storage<D>>,
    pub(crate) entry: RwLock<(usize, usize)>,
    pub(crate) update_lock: RwLock<()>,
    ml: f64,
    seed: u64,
    rng: Mutex<StdRng>,
    dist: DS,
    /// `IdMode as u8`, or `IdMode::UNSET` until the first insert.
    id_mode: AtomicU8,
    /// Explicit ids that are stored or reserved by an in-flight insert.
    taken_ids: Mutex<HashSet<usize>>,
}

/// How an index assigns ids, fixed by its first insert.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum IdMode {
    /// Ids are insertion positions `0, 1, 2, ...` (`insert`, `build_parallel`, ...).
    Positional = 1,
    /// Ids are supplied by the caller (`insert_with_id`, ...).
    Explicit = 2,
}

impl IdMode {
    pub(crate) const UNSET: u8 = 0;

    pub(crate) fn from_u8(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::Positional),
            2 => Some(Self::Explicit),
            _ => None,
        }
    }
}

/// Exact duplicates of a graph node are stored as *members* of that node's group:
/// they keep their own position (and id) but have no layers and are never linked.
/// `dup_next` threads each group into a list starting at its graph node, and
/// searches expand every group they reach. Without this, the diversity heuristic
/// keeps at most one link into a set of identical vectors and most copies become
/// unreachable.
#[derive(Debug)]
struct Storage<const D: usize> {
    pub(crate) data: Vec<[f32; D]>,
    pub(crate) nodes: Vec<Node>,
    pub(crate) dup_next: Vec<usize>,
    /// Id reported for each stored vector.
    pub(crate) ids: Vec<usize>,
}

/// `dup_next` value for the end of a duplicate list.
pub(crate) const NO_DUP: usize = usize::MAX;

/// Where a prepared vector goes in the graph.
enum Placement {
    /// Storage was empty: may only be committed as the very first node.
    First,
    /// Linked to neighbors found by the insertion search.
    Linked,
    /// Exact duplicate of this graph node: joins its group.
    DuplicateOf(usize),
}

/// k-nearest-neighbor search over an index.
///
/// Results are `(id, distance)` pairs sorted by ascending distance (ties by id),
/// at most `min(k, len())` of them.
///
/// Query semantics shared by every method:
/// - `ef_search` must be at least 1, otherwise [`HnswError::InvalidParameter`].
///   The effective search width is `max(ef_search, k)`; values larger than the
///   index are allowed and simply visit more of it.
/// - The query must pass the metric's [`Distance::validate`].
/// - `k == 0` and an empty index return an empty result.
///
/// The `try_*` methods return these errors; the other methods panic with the
/// error message instead.
pub trait HnswSearcher<const D: usize> {
    fn search_context(&self) -> SearchContext {
        SearchContext::reusable(self.len())
    }

    fn try_search_with_context(
        &self,
        q: &[f32; D],
        k: usize,
        ef_search: usize,
        ctx: &mut SearchContext,
    ) -> Result<Vec<(usize, f32)>, HnswError>;

    fn try_search_with_ef(
        &self,
        q: &[f32; D],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<(usize, f32)>, HnswError> {
        check_ef_search(ef_search)?;
        let mut ctx = SearchContext::one_off(ef_search.min(self.len()).max(1));
        self.try_search_with_context(q, k, ef_search, &mut ctx)
    }

    fn try_search(&self, q: &[f32; D], k: usize) -> Result<Vec<(usize, f32)>, HnswError> {
        self.try_search_with_ef(q, k, DEFAULT_EF_SEARCH)
    }

    /// Like [`HnswSearcher::try_search_with_ef`] for a runtime-sized query;
    /// returns [`HnswError::DimensionMismatch`] if `q.len() != D`.
    fn try_search_slice(
        &self,
        q: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<(usize, f32)>, HnswError> {
        self.try_search_with_ef(as_array(q)?, k, ef_search)
    }

    /// # Panics
    /// When [`HnswSearcher::try_search`] would return an error.
    fn search(&self, q: &[f32; D], k: usize) -> Vec<(usize, f32)> {
        self.try_search(q, k)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    /// # Panics
    /// When [`HnswSearcher::try_search_with_ef`] would return an error.
    fn search_with_ef(&self, q: &[f32; D], k: usize, ef_search: usize) -> Vec<(usize, f32)> {
        self.try_search_with_ef(q, k, ef_search)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    /// # Panics
    /// When [`HnswSearcher::try_search_with_context`] would return an error.
    fn search_with_context(
        &self,
        q: &[f32; D],
        k: usize,
        ef_search: usize,
        ctx: &mut SearchContext,
    ) -> Vec<(usize, f32)> {
        self.try_search_with_context(q, k, ef_search, ctx)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    fn memory_usage_bytes(&self) -> usize;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// CREATE
impl<const D: usize, DS> Hnsw<D, DS>
where
    DS: Distance<D>,
{
    /// Creates an empty index whose layer assignment is driven by `seed`.
    ///
    /// - `M`: links per node on layers above 0, `2..=MAX_CONNECTIONS`
    /// - `M0`: links per node on layer 0, `1..=MAX_CONNECTIONS`
    /// - `ef_construction`: candidate list size while inserting, `1..=MAX_EF_CONSTRUCTION`
    ///
    /// Sequential insertion into a seeded index is deterministic: the same seed,
    /// parameters and insertion order produce the same graph.
    #[allow(non_snake_case)]
    pub fn try_new_seeded(
        M: usize,
        M0: usize,
        ef_construction: usize,
        seed: u64,
        dist: DS,
    ) -> Result<Self, HnswError> {
        check_graph_params(M, M0, ef_construction)?;
        Ok(Self::new_unchecked(M, M0, ef_construction, seed, dist))
    }

    /// [`Hnsw::try_new_seeded`] with a random seed.
    #[allow(non_snake_case)]
    pub fn try_new(
        M: usize,
        M0: usize,
        ef_construction: usize,
        dist: DS,
    ) -> Result<Self, HnswError> {
        let seed = rand::rng().next_u64();
        Self::try_new_seeded(M, M0, ef_construction, seed, dist)
    }

    /// # Panics
    /// When [`Hnsw::try_new_seeded`] would return an error.
    #[allow(non_snake_case)]
    pub fn new_seeded(M: usize, M0: usize, ef_construction: usize, seed: u64, dist: DS) -> Self {
        Self::try_new_seeded(M, M0, ef_construction, seed, dist)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    #[allow(non_snake_case)]
    fn new_unchecked(M: usize, M0: usize, ef_construction: usize, seed: u64, dist: DS) -> Self {
        let ml = 1.0 / (M as f64).ln();
        Self {
            M,
            M0,
            ef_construction,
            entry: RwLock::new((0, 0)),
            update_lock: RwLock::new(()),
            storage: RwLock::new(Storage {
                data: Vec::new(),
                nodes: Vec::new(),
                dup_next: Vec::new(),
                ids: Vec::new(),
            }),
            ml,
            seed,
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
            dist,
            id_mode: AtomicU8::new(IdMode::UNSET),
            taken_ids: Mutex::new(HashSet::new()),
        }
    }

    /// # Panics
    /// When [`Hnsw::try_new`] would return an error.
    #[allow(non_snake_case)]
    pub fn new(M: usize, M0: usize, ef_construction: usize, dist: DS) -> Self {
        Self::try_new(M, M0, ef_construction, dist).unwrap_or_else(|error| panic!("{error}"))
    }
}

#[allow(non_snake_case)]
pub(crate) fn check_graph_params(
    M: usize,
    M0: usize,
    ef_construction: usize,
) -> Result<(), HnswError> {
    if !(2..=MAX_CONNECTIONS).contains(&M) {
        return Err(HnswError::InvalidParameter {
            name: "M",
            value: M,
            requirement: "between 2 and 4096",
        });
    }
    if !(1..=MAX_CONNECTIONS).contains(&M0) {
        return Err(HnswError::InvalidParameter {
            name: "M0",
            value: M0,
            requirement: "between 1 and 4096",
        });
    }
    if !(1..=MAX_EF_CONSTRUCTION).contains(&ef_construction) {
        return Err(HnswError::InvalidParameter {
            name: "ef_construction",
            value: ef_construction,
            requirement: "between 1 and 65536",
        });
    }
    Ok(())
}

pub(crate) fn check_ef_search(ef_search: usize) -> Result<(), HnswError> {
    if ef_search == 0 {
        return Err(HnswError::InvalidParameter {
            name: "ef_search",
            value: ef_search,
            requirement: "at least 1",
        });
    }
    Ok(())
}

/// The `k` best of a final layer search as `(id, distance)`, ordered by distance
/// and then ascending id. Each graph node is expanded into its duplicate group
/// (at most `k` entries per group, all at the node's distance).
pub(crate) fn top_k(
    candidates: &[Link],
    k: usize,
    dup_next: &[usize],
    ids: &[usize],
) -> Vec<(usize, f32)> {
    let mut hits: Vec<(usize, f32)> = Vec::with_capacity(candidates.len());
    for link in candidates {
        hits.push((ids[link.node_index], link.distance));
        let mut member = dup_next[link.node_index];
        let mut taken = 1;
        while member != NO_DUP && taken < k {
            hits.push((ids[member], link.distance));
            member = dup_next[member];
            taken += 1;
        }
    }
    hits.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    hits.truncate(k);
    hits
}

/// Hash key under which two vectors collide exactly when they compare equal
/// (`-0.0` and `0.0` are normalized; NaN never reaches the index).
fn exact_key<const D: usize>(v: &[f32; D]) -> [u32; D] {
    v.map(|x| if x == 0.0 { 0 } else { x.to_bits() })
}

/// A graph node among the closest insertion-search candidates that equals `vec`.
fn exact_duplicate<const D: usize>(
    storage: &Storage<D>,
    vec: &[f32; D],
    candidates: &[Link],
) -> Option<usize> {
    let closest = candidates.first()?.distance;
    candidates
        .iter()
        .take_while(|link| link.distance == closest)
        .map(|link| link.node_index)
        .find(|&idx| storage.data[idx] == *vec)
}

pub(crate) fn as_array<const D: usize>(v: &[f32]) -> Result<&[f32; D], HnswError> {
    v.try_into().map_err(|_| HnswError::DimensionMismatch {
        expected: D,
        found: v.len(),
    })
}

// INSERT
impl<const D: usize, DS> Hnsw<D, DS>
where
    DS: Distance<D> + Send + Sync,
{
    pub fn insert_context(&self) -> InsertContext {
        InsertContext::reusable(self.len(), self.ef_construction, self.M0)
    }

    /// Inserts `vec` into a positional index and returns its id (its insertion
    /// position).
    ///
    /// Errors: the vector fails [`Distance::validate`], the metric returns a
    /// non-finite distance while the new node's neighbors are searched, or the
    /// index uses explicit ids ([`HnswError::IdModeMismatch`]). On error the index
    /// is unchanged.
    pub fn try_insert(&self, vec: [f32; D]) -> Result<usize, HnswError> {
        let mut ctx = InsertContext::one_off(self.ef_construction, self.M0);
        self.try_insert_with_context(vec, &mut ctx)
    }

    /// Like [`Hnsw::try_insert`] for a runtime-sized vector; returns
    /// [`HnswError::DimensionMismatch`] if `vec.len() != D`.
    pub fn try_insert_slice(&self, vec: &[f32]) -> Result<usize, HnswError> {
        self.try_insert(*as_array(vec)?)
    }

    /// # Panics
    /// When [`Hnsw::try_insert`] would return an error.
    pub fn insert(&self, vec: [f32; D]) -> usize {
        self.try_insert(vec)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    /// [`Hnsw::try_insert`] reusing the scratch buffers in `ctx`.
    pub fn try_insert_with_context(
        &self,
        vec: [f32; D],
        ctx: &mut InsertContext,
    ) -> Result<usize, HnswError> {
        self.dist.validate(&vec)?;
        self.claim_id_mode(IdMode::Positional)?;
        self.try_insert_validated(vec, None, ctx, true)
    }

    /// # Panics
    /// When [`Hnsw::try_insert_with_context`] would return an error.
    pub fn insert_with_context(&self, vec: [f32; D], ctx: &mut InsertContext) -> usize {
        self.try_insert_with_context(vec, ctx)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    /// Inserts `vec` under the caller-supplied `id`; searches report `id` for it.
    ///
    /// The first insert decides whether an index uses explicit ids; mixing with
    /// the positional methods returns [`HnswError::IdModeMismatch`]. An id that is
    /// already present (or reserved by a concurrent insert) returns
    /// [`HnswError::DuplicateId`]. Ids are never reused or replaced: there is no
    /// delete or upsert. On error the index is unchanged.
    pub fn try_insert_with_id(&self, id: usize, vec: [f32; D]) -> Result<(), HnswError> {
        let mut ctx = InsertContext::one_off(self.ef_construction, self.M0);
        self.try_insert_with_id_and_context(id, vec, &mut ctx)
    }

    /// [`Hnsw::try_insert_with_id`] reusing the scratch buffers in `ctx`.
    pub fn try_insert_with_id_and_context(
        &self,
        id: usize,
        vec: [f32; D],
        ctx: &mut InsertContext,
    ) -> Result<(), HnswError> {
        self.dist.validate(&vec)?;
        self.claim_id_mode(IdMode::Explicit)?;
        self.reserve_ids(&[id])?;
        self.try_insert_validated(vec, Some(id), ctx, true)
            .map(|_| ())
            .inspect_err(|_| {
                self.taken_ids.lock().unwrap().remove(&id);
            })
    }

    /// # Panics
    /// When [`Hnsw::try_insert_with_id`] would return an error.
    pub fn insert_with_id(&self, id: usize, vec: [f32; D]) {
        self.try_insert_with_id(id, vec)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    /// Whether `id` is stored in (or currently being inserted into) the index.
    pub fn contains_id(&self, id: usize) -> bool {
        match self.id_mode() {
            None => false,
            Some(IdMode::Positional) => id < self.len(),
            Some(IdMode::Explicit) => self.taken_ids.lock().unwrap().contains(&id),
        }
    }

    /// The id mode fixed by the first insert, or `None` for a fresh index.
    pub fn id_mode(&self) -> Option<IdMode> {
        IdMode::from_u8(self.id_mode.load(Ordering::Acquire))
    }

    /// Builds an empty positional index from `vecs` using up to `threads` worker
    /// threads (`None`: available parallelism). Ids are the positions in `vecs`.
    ///
    /// All vectors are validated before anything is inserted, so on error the
    /// index is unchanged. An empty `vecs` is a no-op. The graph depends on thread
    /// scheduling and is not reproducible even with a seed; use sequential
    /// insertion for a deterministic graph.
    pub fn try_build_parallel(
        &mut self,
        vecs: &[[f32; D]],
        threads: Option<NonZeroUsize>,
    ) -> Result<(), HnswError> {
        self.check_empty()?;
        self.validate_batch(vecs.iter())?;
        if vecs.is_empty() {
            return Ok(());
        }
        self.claim_id_mode(IdMode::Positional)?;
        self.build_parallel_validated(vecs, None, threads);
        Ok(())
    }

    /// # Panics
    /// When [`Hnsw::try_build_parallel`] would return an error.
    pub fn build_parallel(&mut self, vecs: &[[f32; D]], threads: Option<NonZeroUsize>) {
        self.try_build_parallel(vecs, threads)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    /// [`Hnsw::try_build_parallel`] with caller-supplied ids. Duplicate ids within
    /// `items` return [`HnswError::DuplicateId`] before anything is inserted.
    pub fn try_build_parallel_with_ids(
        &mut self,
        items: &[(usize, [f32; D])],
        threads: Option<NonZeroUsize>,
    ) -> Result<(), HnswError> {
        self.check_empty()?;
        self.validate_batch(items.iter().map(|(_, vec)| vec))?;
        if items.is_empty() {
            return Ok(());
        }
        let (ids, vecs): (Vec<usize>, Vec<[f32; D]>) = items.iter().copied().unzip();
        self.claim_id_mode(IdMode::Explicit)?;
        self.reserve_ids(&ids)?;
        self.build_parallel_validated(&vecs, Some(&ids), threads);
        Ok(())
    }

    /// # Panics
    /// When [`Hnsw::try_build_parallel_with_ids`] would return an error.
    pub fn build_parallel_with_ids(
        &mut self,
        items: &[(usize, [f32; D])],
        threads: Option<NonZeroUsize>,
    ) {
        self.try_build_parallel_with_ids(items, threads)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    fn check_empty(&self) -> Result<(), HnswError> {
        match self.len() {
            0 => Ok(()),
            len => Err(HnswError::IndexNotEmpty { len }),
        }
    }

    fn validate_batch<'a>(
        &self,
        vecs: impl Iterator<Item = &'a [f32; D]>,
    ) -> Result<(), HnswError> {
        for (index, vec) in vecs.enumerate() {
            self.dist
                .validate(vec)
                .map_err(|error| HnswError::InvalidBatchVector {
                    index,
                    error: Box::new(error),
                })?;
        }
        Ok(())
    }

    /// Fixes the id mode on first use; errors if the index uses the other one.
    fn claim_id_mode(&self, mode: IdMode) -> Result<(), HnswError> {
        match self.id_mode.compare_exchange(
            IdMode::UNSET,
            mode as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => Ok(()),
            Err(current) if current == mode as u8 => Ok(()),
            Err(current) => Err(HnswError::IdModeMismatch {
                index_mode: IdMode::from_u8(current).expect("id mode is set"),
            }),
        }
    }

    /// Atomically reserves every id in `ids`, or none of them.
    fn reserve_ids(&self, ids: &[usize]) -> Result<(), HnswError> {
        let mut taken = self.taken_ids.lock().unwrap();
        let mut batch = HashSet::with_capacity(ids.len());
        for &id in ids {
            if taken.contains(&id) || !batch.insert(id) {
                return Err(HnswError::DuplicateId { id });
            }
        }
        taken.extend(batch);
        Ok(())
    }

    fn build_parallel_validated(
        &mut self,
        vecs: &[[f32; D]],
        ids: Option<&[usize]>,
        threads: Option<NonZeroUsize>,
    ) {
        let mut nodes = self.preallocate_nodes(vecs, ids);
        // entry is already inserted
        {
            let entry = self.entry.read().unwrap();
            nodes.retain(|&(idx, _)| idx != entry.0);
        }
        if nodes.is_empty() {
            return;
        }

        let nthreads = parallel_thread_count(threads);
        let chunk_sz = nodes.len().div_ceil(nthreads);

        let index = &*self;
        std::thread::scope(|s| {
            for chunk in nodes.chunks(chunk_sz) {
                s.spawn(move || {
                    let mut thread_ctx = index.insert_context();
                    for (idx, lyr) in chunk {
                        index.insert_preallocated(*idx, *lyr, &mut thread_ctx);
                    }
                });
            }
        });
    }

    /// Inserts `vecs` concurrently into a possibly non-empty positional index and
    /// returns the id assigned to each input, in input order.
    ///
    /// All vectors are validated first, so on error nothing is inserted. An
    /// empty `vecs` returns an empty vector. Which id each vector receives, and
    /// the resulting graph, depend on thread scheduling.
    pub fn try_extend_parallel(
        &self,
        vecs: &[[f32; D]],
        threads: Option<NonZeroUsize>,
    ) -> Result<Vec<usize>, HnswError> {
        self.validate_batch(vecs.iter())?;
        if vecs.is_empty() {
            return Ok(Vec::new());
        }
        self.claim_id_mode(IdMode::Positional)?;
        Ok(self.extend_parallel_validated(vecs, None, threads))
    }

    /// # Panics
    /// When [`Hnsw::try_extend_parallel`] would return an error.
    pub fn extend_parallel(&self, vecs: &[[f32; D]], threads: Option<NonZeroUsize>) -> Vec<usize> {
        self.try_extend_parallel(vecs, threads)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    /// Inserts `items` concurrently under their caller-supplied ids.
    ///
    /// All vectors are validated and all ids reserved before anything is
    /// inserted: an id that repeats within `items`, or is already present or being
    /// inserted concurrently, returns [`HnswError::DuplicateId`] and nothing is
    /// inserted. The graph depends on thread scheduling; the id of every vector
    /// does not.
    pub fn try_extend_parallel_with_ids(
        &self,
        items: &[(usize, [f32; D])],
        threads: Option<NonZeroUsize>,
    ) -> Result<(), HnswError> {
        self.validate_batch(items.iter().map(|(_, vec)| vec))?;
        if items.is_empty() {
            return Ok(());
        }
        let (ids, vecs): (Vec<usize>, Vec<[f32; D]>) = items.iter().copied().unzip();
        self.claim_id_mode(IdMode::Explicit)?;
        self.reserve_ids(&ids)?;
        self.extend_parallel_validated(&vecs, Some(&ids), threads);
        Ok(())
    }

    /// # Panics
    /// When [`Hnsw::try_extend_parallel_with_ids`] would return an error.
    pub fn extend_parallel_with_ids(
        &self,
        items: &[(usize, [f32; D])],
        threads: Option<NonZeroUsize>,
    ) {
        self.try_extend_parallel_with_ids(items, threads)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    /// Returns the positional ids assigned (`ids[i]` when `ids` is given).
    fn extend_parallel_validated(
        &self,
        vecs: &[[f32; D]],
        ids: Option<&[usize]>,
        threads: Option<NonZeroUsize>,
    ) -> Vec<usize> {
        let id_of = |i: usize| ids.map(|ids| ids[i]);
        let mut assigned = vec![0; vecs.len()];
        let mut first_ctx = InsertContext::one_off(self.ef_construction, self.M0);
        assigned[0] = self.insert_validated(vecs[0], id_of(0), &mut first_ctx);
        if vecs.len() == 1 {
            return assigned;
        }

        let nthreads = parallel_thread_count(threads);
        let chunk_sz = (vecs.len() - 1).div_ceil(nthreads);
        std::thread::scope(|s| {
            for (chunk_index, out) in assigned[1..].chunks_mut(chunk_sz).enumerate() {
                let start = 1 + chunk_index * chunk_sz;
                s.spawn(move || {
                    let mut thread_ctx = self.insert_context();
                    for (offset, out_id) in out.iter_mut().enumerate() {
                        let i = start + offset;
                        *out_id = self.insert_validated(vecs[i], id_of(i), &mut thread_ctx);
                    }
                });
            }
        });

        assigned
    }

    /// Inserts a vector that already passed `validate` and whose id (if any) is
    /// reserved; non-finite distances are treated as `+inf`, so it cannot fail.
    fn insert_validated(&self, vec: [f32; D], id: Option<usize>, ctx: &mut InsertContext) -> usize {
        self.try_insert_validated(vec, id, ctx, false)
            .expect("non-strict insertion is infallible")
    }

    /// Returns the storage position of the inserted vector.
    fn try_insert_validated(
        &self,
        vec: [f32; D],
        id: Option<usize>,
        ctx: &mut InsertContext,
        strict: bool,
    ) -> Result<usize, HnswError> {
        let _guard = self.update_lock.read().unwrap();
        let (mut node, max_lyr) = self.new_node();

        let idx = loop {
            ctx.search_ctx.non_finite_distance = None;
            let placement = self.prepare_node(vec, &node, max_lyr, ctx);
            if let Some(distance) = ctx.search_ctx.non_finite_distance.take()
                && strict
            {
                return Err(HnswError::NonFiniteDistance { distance });
            }
            // A node prepared against empty storage has no links and may only be
            // committed as the very first node; if another insert won that race,
            // search for neighbors again.
            let is_member = matches!(placement, Placement::DuplicateOf(_));
            match self.commit(vec, id, node, placement) {
                Ok(idx) => break (idx, is_member),
                Err(unlinked) => node = unlinked,
            }
        };
        let (idx, is_member) = idx;
        if !is_member {
            self.publish(idx, max_lyr, &mut ctx.select_ctx);
        }

        Ok(idx)
    }

    fn insert_preallocated(&self, idx: usize, max_lyr: usize, ctx: &mut InsertContext) -> usize {
        {
            let storage = self.storage.read().unwrap();
            assert!(
                idx < storage.data.len(),
                "preallocated node index {} out of bounds for {} nodes",
                idx,
                storage.data.len()
            );

            let node = &storage.nodes[idx];
            let vec = storage.data[idx];
            assert_eq!(
                node.layers.len(),
                max_lyr + 1,
                "preallocated node {} has {} layers, expected {}",
                idx,
                node.layers.len(),
                max_lyr + 1
            );

            self.prepare_node_with_storage(&storage, vec, node, max_lyr, ctx);
        }
        self.publish(idx, max_lyr, &mut ctx.select_ctx);
        idx
    }

    /// Finds neighbors for `node` in the current graph.
    fn prepare_node(
        &self,
        vec: [f32; D],
        node: &Node,
        max_lyr: usize,
        ctx: &mut InsertContext,
    ) -> Placement {
        let storage = self.storage.read().unwrap();
        if storage.data.is_empty() {
            return Placement::First;
        }
        match self.prepare_node_with_storage(&storage, vec, node, max_lyr, ctx) {
            Some(rep) => Placement::DuplicateOf(rep),
            None => Placement::Linked,
        }
    }
    // assumes non empty storage; returns the graph node `vec` duplicates, if found
    fn prepare_node_with_storage(
        &self,
        storage: &Storage<D>,
        vec: [f32; D],
        node: &Node,
        max_lyr: usize,
        ctx: &mut InsertContext,
    ) -> Option<usize> {
        assert!(
            max_lyr < node.layers.len(),
            "node has no layer {} (only {} layers)",
            max_lyr,
            node.layers.len()
        );

        let search_ctx = &mut ctx.search_ctx;
        let select_ctx = &mut ctx.select_ctx;
        let (mut ep, max_layer) = *self.entry.read().unwrap();
        for lyr in ((max_lyr + 1)..=max_layer).rev() {
            ep = self
                .search_layer_with_context(storage, &vec, ep, lyr, 1, search_ctx)
                .first()
                .unwrap_or_else(|| {
                    panic!("ERROR: search_layer@{lyr} returned an empty array (insert)")
                })
                .node_index;
        }

        for lyr in (0..=max_lyr.min(max_layer)).rev() {
            let candidates = self.search_layer_with_context(
                storage,
                &vec,
                ep,
                lyr,
                self.ef_construction,
                search_ctx,
            );
            if lyr == 0
                && let Some(rep) = exact_duplicate(storage, &vec, candidates)
            {
                return Some(rep);
            }
            let selected =
                self.select_neighbors(storage, &vec, lyr, candidates, false, false, select_ctx);

            let next_ep = selected
                .first()
                .expect("neighbor selection returned no nodes")
                .node_index;
            *node.layers[lyr].write().unwrap() = selected;
            ep = next_ep
        }
        None
    }

    /// Appends the vector. A [`Placement::First`] node is given back unless storage
    /// is still empty; a duplicate is stored as a layerless member of its group.
    fn commit(
        &self,
        vec: [f32; D],
        id: Option<usize>,
        node: Node,
        placement: Placement,
    ) -> Result<usize, Node> {
        debug_assert!(
            !node.layers.is_empty(),
            "cannot commit a node with no layers"
        );
        let mut storage = self.storage.write().unwrap();
        let insert_idx = storage.data.len();
        match placement {
            Placement::First if !storage.data.is_empty() => return Err(node),
            Placement::First | Placement::Linked => {
                storage.nodes.push(node);
                storage.dup_next.push(NO_DUP);
            }
            Placement::DuplicateOf(rep) => {
                storage.nodes.push(Node { layers: Vec::new() });
                let next = std::mem::replace(&mut storage.dup_next[rep], insert_idx);
                storage.dup_next.push(next);
            }
        }
        storage.data.push(vec);
        storage.ids.push(id.unwrap_or(insert_idx));
        Ok(insert_idx)
    }

    fn publish(&self, idx: usize, max_lyr: usize, select_ctx: &mut SelectContext) {
        let storage = self.storage.read().unwrap();
        assert!(
            idx < storage.nodes.len(),
            "published node index {} out of bounds for {} nodes",
            idx,
            storage.nodes.len()
        );
        assert!(
            max_lyr < storage.nodes[idx].layers.len(),
            "published node {} has no layer {} (only {} layers)",
            idx,
            max_lyr,
            storage.nodes[idx].layers.len()
        );
        for lyr in 0..=max_lyr {
            let links: Vec<Link> = {
                let links = storage.nodes[idx].layers[lyr].read().unwrap();
                links.iter().copied().collect()
            };
            for fw_link in links {
                let backlink = Link {
                    node_index: idx,
                    distance: fw_link.distance,
                };
                self.add_backlink(&storage, fw_link.node_index, backlink, lyr, select_ctx);
            }
        }

        self.update_entry_point_if_required(idx, max_lyr);
    }

    fn new_node(&self) -> (Node, usize) {
        let max_lyr = self.random_layer();
        let node = Node {
            layers: (0..max_lyr + 1)
                .map(|lyr| {
                    let max_connections = self.max_connections(lyr);
                    RwLock::new(Vec::with_capacity(max_connections))
                })
                .collect(),
        };

        (node, max_lyr)
    }

    fn preallocate_nodes(&self, vecs: &[[f32; D]], ids: Option<&[usize]>) -> Vec<(usize, usize)> {
        let mut storage = self.storage.write().unwrap();
        assert!(
            storage.data.is_empty() && storage.nodes.is_empty(),
            "node preallocation requires empty storage"
        );
        let mut out = Vec::with_capacity(vecs.len());
        let mut first_copy = HashMap::with_capacity(vecs.len());
        for vec in vecs {
            // one level draw per vector keeps the RNG in step with sequential inserts
            let (node, max_lyr) = self.new_node();
            let idx = storage.data.len();
            storage.data.push(*vec);
            storage.ids.push(ids.map_or(idx, |ids| ids[idx]));
            match first_copy.entry(exact_key(vec)) {
                Entry::Occupied(rep) => {
                    let rep = *rep.get();
                    storage.nodes.push(Node { layers: Vec::new() });
                    let next = std::mem::replace(&mut storage.dup_next[rep], idx);
                    storage.dup_next.push(next);
                }
                Entry::Vacant(slot) => {
                    slot.insert(idx);
                    storage.nodes.push(node);
                    storage.dup_next.push(NO_DUP);
                    out.push((idx, max_lyr));
                    self.update_entry_point_if_required(idx, max_lyr);
                }
            }
        }

        out
    }

    fn update_entry_point_if_required(&self, insert_idx: usize, insert_lyr: usize) {
        let mut entry = self.entry.write().unwrap();
        if insert_lyr > entry.1 {
            entry.0 = insert_idx;
            entry.1 = insert_lyr;
        }
    }
}

fn parallel_thread_count(requested: Option<NonZeroUsize>) -> usize {
    let available = std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN);
    requested
        .map_or(available, |requested| requested.min(available))
        .get()
}

// SEARCH
impl<const D: usize, DS> Hnsw<D, DS>
where
    DS: Distance<D>,
{
    fn search_layer_with_context<'a>(
        &self,
        storage: &Storage<D>,
        q: &[f32; D],
        ep: usize,
        lyr: usize,
        ef: usize,
        ctx: &'a mut SearchContext,
    ) -> &'a [Link] {
        debug_assert!(ef > 0, "ef must be > 0");
        debug_assert!(ep < storage.data.len(), "entry point out of bounds");
        debug_assert!(
            lyr < storage.nodes[ep].layers.len(),
            "entry point does not exist in this layer"
        );

        ctx.clear();
        let visited = &mut ctx.visited;
        visited.reset();
        let frontier = &mut ctx.frontier;
        let best = &mut ctx.best;
        let non_finite = &mut ctx.non_finite_distance;

        let ep_link = Link {
            node_index: ep,
            distance: self.checked_distance(q, &storage.data[ep], non_finite),
        };
        frontier.push(Reverse(ep_link));
        best.push(ep_link);
        visited.mark_visited(ep);

        while let Some(Reverse(candidate)) = frontier.pop() {
            let furthest_dist = best.peek().map_or(f32::INFINITY, |l| l.distance);
            if candidate.distance > furthest_dist {
                break;
            }
            for neigh in storage.nodes[candidate.node_index].layers[lyr]
                .read()
                .unwrap()
                .iter()
            {
                if visited.is_visited(neigh.node_index) {
                    continue;
                }
                visited.mark_visited(neigh.node_index);
                let dist = self.checked_distance(q, &storage.data[neigh.node_index], non_finite);
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
}
impl<const D: usize, DS> HnswSearcher<D> for Hnsw<D, DS>
where
    DS: Distance<D>,
{
    fn try_search_with_context(
        &self,
        q: &[f32; D],
        k: usize,
        ef_search: usize,
        ctx: &mut SearchContext,
    ) -> Result<Vec<(usize, f32)>, HnswError> {
        check_ef_search(ef_search)?;
        self.dist.validate(q)?;
        if k == 0 {
            return Ok(Vec::new());
        }

        let storage = self.storage.read().unwrap();
        if storage.data.is_empty() {
            return Ok(Vec::new());
        }
        ctx.non_finite_distance = None;
        let (mut ep, max_layer) = *self.entry.read().unwrap();
        for lyr in (1..=max_layer).rev() {
            // a layer search always returns at least its entry point
            ep = self.search_layer_with_context(&storage, q, ep, lyr, 1, ctx)[0].node_index;
        }

        let results = top_k(
            self.search_layer_with_context(&storage, q, ep, 0, ef_search.max(k), ctx),
            k,
            &storage.dup_next,
            &storage.ids,
        );
        if let Some(distance) = ctx.non_finite_distance {
            return Err(HnswError::NonFiniteDistance { distance });
        }
        Ok(results)
    }

    fn memory_usage_bytes(&self) -> usize {
        let storage = self.storage.read().unwrap();
        size_of::<Self>()
            + storage.data.capacity() * size_of::<[f32; D]>()
            + storage.dup_next.capacity() * size_of::<usize>()
            + storage.ids.capacity() * size_of::<usize>()
            + nodes_heap_usage_bytes(&storage.nodes)
    }

    fn len(&self) -> usize {
        self.storage.read().unwrap().data.len()
    }

    fn is_empty(&self) -> bool {
        self.storage.read().unwrap().data.is_empty()
    }
}

// MISC
impl<const D: usize, DS> Hnsw<D, DS>
where
    DS: Distance<D>,
{
    #[allow(clippy::too_many_arguments)]
    fn select_neighbors(
        &self,
        storage: &Storage<D>,
        qv: &[f32; D],
        lyr: usize,
        candidates: &[Link],
        extend: bool,
        keep_pruned: bool,
        ctx: &mut SelectContext,
    ) -> Vec<Link> {
        ctx.clear();
        let visited = &mut ctx.visited;
        visited.reset();
        let pq = &mut ctx.pq;
        let discarded = &mut ctx.discarded;
        let best = &mut ctx.best;
        let max_connections = self.max_connections(lyr);

        for (node, link, idx) in candidates
            .iter()
            .map(|link| (&storage.nodes[link.node_index], link, link.node_index))
        {
            if visited.is_visited(idx) {
                continue;
            }
            visited.mark_visited(idx);
            pq.push(Reverse(*link));

            if extend {
                let neighs = &node.layers[lyr];
                for (vec, idx) in neighs
                    .read()
                    .unwrap()
                    .iter()
                    .map(|link| (&storage.data[link.node_index], link.node_index))
                {
                    if visited.is_visited(idx) {
                        continue;
                    }
                    visited.mark_visited(idx);
                    pq.push(Reverse(Link {
                        node_index: idx,
                        distance: self.distance(qv, vec),
                    }));
                }
            }
        }

        // no pruning required
        if pq.len() <= max_connections {
            return ctx.consume_pq();
        }

        while let Some((vec, idx)) = pq
            .pop()
            .map(|c| (&storage.data[c.0.node_index], c.0.node_index))
            && best.len() < max_connections
        {
            let mut diverse = true;
            let c_to_q = self.distance(qv, vec);
            for other in best.iter().map(|link| &storage.data[link.node_index]) {
                let c_to_other = self.distance(vec, other);
                if c_to_q >= c_to_other {
                    diverse = false;
                    break;
                }
            }

            if diverse {
                best.push(Link {
                    node_index: idx,
                    distance: c_to_q,
                });
            } else if keep_pruned {
                discarded.push(Reverse(Link {
                    node_index: idx,
                    distance: c_to_q,
                }));
            }
        }

        if keep_pruned {
            while let Some(Reverse(link)) = discarded.pop()
                && best.len() < max_connections
            {
                best.push(link);
            }
        }

        ctx.consume_best()
    }

    fn add_backlink(
        &self,
        storage: &Storage<D>,
        at: usize,
        link: Link,
        lyr: usize,
        ctx: &mut SelectContext,
    ) {
        assert!(at < storage.data.len(), "backlink base index out of bounds",);
        assert!(
            link.node_index < storage.data.len(),
            "backlink connection index out of bounds"
        );
        assert!(
            lyr < storage.nodes[at].layers.len(),
            "node does not exist in this layer"
        );
        assert!(
            lyr < storage.nodes[link.node_index].layers.len(),
            "backlink source node {} does not exist in layer {}",
            link.node_index,
            lyr
        );
        assert!(link.node_index != at, "can't link node to itself");

        let max_connections = self.max_connections(lyr);
        let mut links = storage.nodes[at].layers[lyr].write().unwrap();

        links.push(link);
        if links.len() > max_connections {
            let candidates = std::mem::take(&mut *links);
            let new_links = self.select_neighbors(
                storage,
                &storage.data[at],
                lyr,
                &candidates,
                false,
                false,
                ctx,
            );
            *links = new_links;
        }
    }

    #[inline(always)]
    fn max_connections(&self, lyr: usize) -> usize {
        if lyr == 0 { self.M0 } else { self.M }
    }

    /// Metric distance with non-finite results mapped to `+inf`, keeping the
    /// graph ordering total when a custom metric breaks its contract.
    #[inline(always)]
    fn distance(&self, a: &[f32; D], b: &[f32; D]) -> f32 {
        let distance = self.dist.distance(a, b);
        if distance.is_finite() {
            distance
        } else {
            f32::INFINITY
        }
    }

    /// Like [`Self::distance`], but also records the first non-finite result.
    #[inline(always)]
    fn checked_distance(&self, a: &[f32; D], b: &[f32; D], seen: &mut Option<f32>) -> f32 {
        let distance = self.dist.distance(a, b);
        if distance.is_finite() {
            distance
        } else {
            seen.get_or_insert(distance);
            f32::INFINITY
        }
    }

    #[inline(always)]
    fn random_layer(&self) -> usize {
        let x: f64 = Open01.sample(&mut self.rng.lock().unwrap());
        (-x.ln() * self.ml).floor() as usize
    }
}

impl<const D: usize, DS> Hnsw<D, DS>
where
    DS: Distance<D> + Default,
{
    #[allow(non_snake_case)]
    pub fn new_default(M: usize) -> Self {
        Self::new(M, 2 * M, 128, DS::default())
    }
}
