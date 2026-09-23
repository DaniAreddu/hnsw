//! Versioned, checksummed snapshots (format version 1).
//!
//! Layout (all integers little-endian), specified in `docs/SNAPSHOT_FORMAT.md`:
//!
//! ```text
//! header   magic "HNSWSNAP" | version u32 | header_len u32
//!          dimension u32 | M u32 | M0 u32 | ef_construction u32 | seed u64
//!          id_mode u8 | reserved [u8; 3] = 0 | max_layer u32
//!          node_count u64 | entry_point u64 | payload_len u64
//!          metric_id_len u16 | metric_id | metric_params_len u32 | metric_params (JSON)
//!          header_crc32 u32                      (CRC-32 of every header byte before it)
//! payload  vectors  node_count * dimension * f32
//!          ids      node_count * u64
//!          dup_next node_count * u64            (u64::MAX = end of list)
//!          nodes    per node: layer_count u8, per layer: link_count u16,
//!                   link_count * (target u32, distance f32)
//! trailer  payload_crc32 u32
//! ```
//!
//! Loading reads with allocations bounded by the declared (and checked) sizes,
//! verifies both checksums and then the full graph structure before an index is
//! returned, so a corrupted or hostile file yields an error instead of a later
//! panic or a silently wrong index.

use std::{
    collections::HashSet,
    fmt,
    fs::{self, File, OpenOptions},
    io::{self, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::{
        Mutex, RwLock,
        atomic::{AtomicU8, AtomicU64, Ordering},
    },
};

use rand::{
    SeedableRng,
    distr::{Distribution, Open01},
    rngs::StdRng,
};
use serde::{Serialize, de::DeserializeOwned};

use crate::{
    Hnsw, HnswError, IdMode, NO_DUP, Storage, check_graph_params, dist::Distance, link::Link,
    node::Node,
};

const MAGIC: [u8; 8] = *b"HNSWSNAP";
/// Snapshot format version written by [`Hnsw::save`] and the only one [`Hnsw::load`] reads.
pub const SNAPSHOT_FORMAT_VERSION: u32 = 1;
const FIXED_HEADER_LEN: usize = 72;
const MAX_METRIC_ID_LEN: usize = 256;
const MAX_METRIC_PARAMS_LEN: usize = 64 * 1024;
const MIN_HEADER_LEN: usize = FIXED_HEADER_LEN + 2 + 4 + 4;
const MAX_HEADER_LEN: usize = MIN_HEADER_LEN + MAX_METRIC_ID_LEN + MAX_METRIC_PARAMS_LEN;
/// Levels are drawn as `floor(-ln(x) / ln(M))` with `x` in (0, 1), which stays
/// below 54 for every `M >= 2`; anything above this bound is corruption.
pub(crate) const MAX_LAYERS: usize = 64;
const END_OF_LIST: u64 = u64::MAX;

/// Errors from saving or loading a snapshot.
#[derive(Debug)]
#[non_exhaustive]
pub enum SnapshotError {
    Io(io::Error),
    /// The file does not start with the snapshot magic. Files written before
    /// v0.1 had no header and are not supported: rebuild those indexes.
    NotASnapshot,
    UnsupportedVersion {
        found: u32,
        supported: u32,
    },
    DimensionMismatch {
        expected: usize,
        found: u64,
    },
    MetricMismatch {
        expected: String,
        found: String,
    },
    /// The data ends before the sizes declared in its header.
    Truncated,
    /// A file holds bytes after the end of the snapshot.
    TrailingData,
    /// The declared size exceeds the file size or the caller's limit.
    TooLarge {
        declared: u64,
        limit: u64,
    },
    HeaderChecksumMismatch,
    PayloadChecksumMismatch,
    /// The stored graph parameters are invalid.
    InvalidConfig(HnswError),
    /// The checksums match but the content violates an index invariant.
    Corrupt(String),
    /// The metric's parameters could not be serialized or deserialized.
    Metric(String),
    /// The index cannot be represented in this format version.
    Unsupported(&'static str),
}

impl fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "snapshot I/O error: {error}"),
            Self::NotASnapshot => write!(
                f,
                "not an hnsw snapshot (missing magic); files written before v0.1 are unversioned and cannot be loaded, rebuild the index"
            ),
            Self::UnsupportedVersion { found, supported } => write!(
                f,
                "snapshot format version {found} is not supported (this build reads version {supported})"
            ),
            Self::DimensionMismatch { expected, found } => write!(
                f,
                "snapshot holds {found}-dimensional vectors, but the index type has dimension {expected}"
            ),
            Self::MetricMismatch { expected, found } => write!(
                f,
                "snapshot was written with metric {found:?}, but the index type uses {expected:?}"
            ),
            Self::Truncated => write!(f, "snapshot is truncated"),
            Self::TrailingData => write!(f, "file has data after the end of the snapshot"),
            Self::TooLarge { declared, limit } => write!(
                f,
                "snapshot declares {declared} bytes, more than the {limit} bytes available"
            ),
            Self::HeaderChecksumMismatch => write!(f, "snapshot header checksum mismatch"),
            Self::PayloadChecksumMismatch => write!(f, "snapshot payload checksum mismatch"),
            Self::InvalidConfig(error) => write!(f, "snapshot has invalid parameters: {error}"),
            Self::Corrupt(reason) => write!(f, "corrupt snapshot: {reason}"),
            Self::Metric(reason) => write!(f, "snapshot metric parameters: {reason}"),
            Self::Unsupported(reason) => write!(f, "cannot write snapshot: {reason}"),
        }
    }
}

impl std::error::Error for SnapshotError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::InvalidConfig(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for SnapshotError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

fn corrupt(reason: impl Into<String>) -> SnapshotError {
    SnapshotError::Corrupt(reason.into())
}

/// Step of [`Hnsw::save`] at which a test can inject a failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SaveStep {
    CreateTemp,
    Write,
    Sync,
    Replace,
}

/// Test hooks for [`Hnsw::save_with`]; `SaveHooks::default()` changes nothing.
#[derive(Default)]
pub(crate) struct SaveHooks<'a> {
    pub(crate) fail_at: Option<SaveStep>,
    /// Fail the write once this many bytes have been written.
    pub(crate) fail_after_bytes: Option<u64>,
    /// Called while the snapshot holds its locks, after the header is written.
    pub(crate) while_locked: Option<&'a (dyn Fn() + Sync)>,
}

impl SaveHooks<'_> {
    fn check(&self, step: SaveStep) -> io::Result<()> {
        if self.fail_at == Some(step) {
            return Err(io::Error::other(format!("injected failure at {step:?}")));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Header {
    dimension: u32,
    m: u32,
    m0: u32,
    ef_construction: u32,
    seed: u64,
    id_mode: u8,
    max_layer: u32,
    node_count: u64,
    entry_point: u64,
    payload_len: u64,
    metric_id: String,
    metric_params: Vec<u8>,
}

impl Header {
    fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(MIN_HEADER_LEN + self.metric_params.len());
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&SNAPSHOT_FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes()); // header_len, patched below
        for value in [self.dimension, self.m, self.m0, self.ef_construction] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&self.seed.to_le_bytes());
        bytes.extend_from_slice(&[self.id_mode, 0, 0, 0]);
        bytes.extend_from_slice(&self.max_layer.to_le_bytes());
        for value in [self.node_count, self.entry_point, self.payload_len] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&(self.metric_id.len() as u16).to_le_bytes());
        bytes.extend_from_slice(self.metric_id.as_bytes());
        bytes.extend_from_slice(&(self.metric_params.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.metric_params);
        let header_len = (bytes.len() + 4) as u32;
        bytes[12..16].copy_from_slice(&header_len.to_le_bytes());
        let crc = crc32fast::hash(&bytes);
        bytes.extend_from_slice(&crc.to_le_bytes());
        bytes
    }

    /// Reads and checks magic, version, length and checksum, then parses.
    fn read<R: Read>(reader: &mut R) -> Result<(Self, u64), SnapshotError> {
        let mut start = [0u8; 16];
        let got = read_up_to(reader, &mut start)?;
        if got < MAGIC.len() {
            return Err(if MAGIC.starts_with(&start[..got]) && got > 0 {
                SnapshotError::Truncated
            } else {
                SnapshotError::NotASnapshot
            });
        }
        if start[..8] != MAGIC {
            return Err(SnapshotError::NotASnapshot);
        }
        if got < start.len() {
            return Err(SnapshotError::Truncated);
        }
        let version = u32::from_le_bytes(start[8..12].try_into().unwrap());
        if version != SNAPSHOT_FORMAT_VERSION {
            return Err(SnapshotError::UnsupportedVersion {
                found: version,
                supported: SNAPSHOT_FORMAT_VERSION,
            });
        }
        let header_len = u32::from_le_bytes(start[12..16].try_into().unwrap()) as usize;
        if !(MIN_HEADER_LEN..=MAX_HEADER_LEN).contains(&header_len) {
            return Err(corrupt(format!("header length {header_len}")));
        }
        let mut bytes = vec![0u8; header_len];
        bytes[..16].copy_from_slice(&start);
        read_exact(reader, &mut bytes[16..])?;
        let (body, crc) = bytes.split_at(header_len - 4);
        if crc32fast::hash(body) != u32::from_le_bytes(crc.try_into().unwrap()) {
            return Err(SnapshotError::HeaderChecksumMismatch);
        }

        let mut cursor = Cursor {
            bytes: &body[16..],
            at: 0,
        };
        let dimension = cursor.u32()?;
        let m = cursor.u32()?;
        let m0 = cursor.u32()?;
        let ef_construction = cursor.u32()?;
        let seed = cursor.u64()?;
        let id_mode = cursor.take(4)?;
        if id_mode[1..] != [0, 0, 0] {
            return Err(corrupt("reserved header bytes are not zero"));
        }
        let id_mode = id_mode[0];
        let max_layer = cursor.u32()?;
        let node_count = cursor.u64()?;
        let entry_point = cursor.u64()?;
        let payload_len = cursor.u64()?;
        let id_len = cursor.u16()? as usize;
        if id_len > MAX_METRIC_ID_LEN {
            return Err(corrupt("metric id too long"));
        }
        let metric_id = String::from_utf8(cursor.take(id_len)?.to_vec())
            .map_err(|_| corrupt("metric id is not UTF-8"))?;
        let params_len = cursor.u32()? as usize;
        if params_len > MAX_METRIC_PARAMS_LEN {
            return Err(corrupt("metric parameters too long"));
        }
        let metric_params = cursor.take(params_len)?.to_vec();
        if cursor.at != cursor.bytes.len() {
            return Err(corrupt("header length does not match its fields"));
        }

        let header = Self {
            dimension,
            m,
            m0,
            ef_construction,
            seed,
            id_mode,
            max_layer,
            node_count,
            entry_point,
            payload_len,
            metric_id,
            metric_params,
        };
        Ok((header, header_len as u64))
    }
}

/// Bounds-checked reads from the in-memory header.
struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], SnapshotError> {
        let end = self
            .at
            .checked_add(len)
            .filter(|&end| end <= self.bytes.len())
            .ok_or_else(|| corrupt("header fields exceed the header length"))?;
        let slice = &self.bytes[self.at..end];
        self.at = end;
        Ok(slice)
    }
    fn u16(&mut self) -> Result<u16, SnapshotError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, SnapshotError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, SnapshotError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}

fn read_up_to<R: Read>(reader: &mut R, buf: &mut [u8]) -> Result<usize, SnapshotError> {
    let mut filled = 0;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(SnapshotError::Io(error)),
        }
    }
    Ok(filled)
}

fn read_exact<R: Read>(reader: &mut R, buf: &mut [u8]) -> Result<(), SnapshotError> {
    reader.read_exact(buf).map_err(|error| {
        if error.kind() == io::ErrorKind::UnexpectedEof {
            SnapshotError::Truncated
        } else {
            SnapshotError::Io(error)
        }
    })
}

/// Reads exactly `remaining` payload bytes, hashing them.
struct PayloadReader<'r, R> {
    inner: &'r mut R,
    hasher: crc32fast::Hasher,
    remaining: u64,
}

impl<R: Read> PayloadReader<'_, R> {
    fn fill(&mut self, buf: &mut [u8]) -> Result<(), SnapshotError> {
        if (buf.len() as u64) > self.remaining {
            return Err(corrupt(
                "payload sections exceed the declared payload length",
            ));
        }
        read_exact(self.inner, buf)?;
        self.hasher.update(buf);
        self.remaining -= buf.len() as u64;
        Ok(())
    }
    fn u8(&mut self) -> Result<u8, SnapshotError> {
        let mut b = [0; 1];
        self.fill(&mut b)?;
        Ok(b[0])
    }
    fn u16(&mut self) -> Result<u16, SnapshotError> {
        let mut b = [0; 2];
        self.fill(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }
    fn u64(&mut self) -> Result<u64, SnapshotError> {
        let mut b = [0; 8];
        self.fill(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }
}

/// Counts and hashes written bytes; optionally fails after a byte budget.
struct PayloadWriter<W> {
    inner: W,
    hasher: crc32fast::Hasher,
    written: u64,
    fail_after: Option<u64>,
}

impl<W: Write> Write for PayloadWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let buf = match self.fail_after {
            Some(limit) if self.written + buf.len() as u64 > limit => {
                let allowed = (limit - self.written) as usize;
                if allowed == 0 {
                    return Err(io::Error::other("injected write failure"));
                }
                &buf[..allowed]
            }
            _ => buf,
        };
        let n = self.inner.write(buf)?;
        self.hasher.update(&buf[..n]);
        self.written += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// Graph data read from a snapshot, before structural validation.
struct Decoded<const D: usize> {
    vectors: Vec<[f32; D]>,
    ids: Vec<u64>,
    dup_next: Vec<u64>,
    layers: Vec<Vec<Vec<Link>>>,
}

fn max_connections(header: &Header, lyr: usize) -> usize {
    if lyr == 0 {
        header.m0 as usize
    } else {
        header.m as usize
    }
}

fn read_payload<R: Read, const D: usize>(
    reader: &mut R,
    header: &Header,
) -> Result<Decoded<D>, SnapshotError> {
    let n = header.node_count;
    // Every node takes at least its vector, id, list link and layer count, so
    // the node count is bounded by the checked payload length before anything
    // is allocated.
    let min_node_bytes = 4 * D as u64 + 8 + 8 + 1;
    if n > header.payload_len / min_node_bytes || n > u32::MAX as u64 {
        return Err(corrupt(format!(
            "{n} nodes cannot fit in a payload of {} bytes",
            header.payload_len
        )));
    }
    let n = n as usize;
    let mut payload = PayloadReader {
        inner: reader,
        hasher: crc32fast::Hasher::new(),
        remaining: header.payload_len,
    };

    let mut vectors = Vec::with_capacity(n);
    let mut raw = vec![0u8; 4 * D];
    for _ in 0..n {
        payload.fill(&mut raw)?;
        vectors.push(std::array::from_fn(|i| {
            f32::from_le_bytes(raw[4 * i..4 * i + 4].try_into().unwrap())
        }));
    }
    let ids = (0..n)
        .map(|_| payload.u64())
        .collect::<Result<Vec<_>, _>>()?;
    let dup_next = (0..n)
        .map(|_| payload.u64())
        .collect::<Result<Vec<_>, _>>()?;

    let mut layers = Vec::with_capacity(n);
    let mut link = [0u8; 8];
    for node in 0..n {
        let layer_count = payload.u8()? as usize;
        if layer_count > MAX_LAYERS {
            return Err(corrupt(format!("node {node} has {layer_count} layers")));
        }
        let mut node_layers = Vec::with_capacity(layer_count);
        for lyr in 0..layer_count {
            let count = payload.u16()? as usize;
            if count > max_connections(header, lyr) {
                return Err(corrupt(format!(
                    "node {node} has {count} links on layer {lyr}, more than the limit {}",
                    max_connections(header, lyr)
                )));
            }
            let mut links = Vec::with_capacity(count);
            for _ in 0..count {
                payload.fill(&mut link)?;
                links.push(Link {
                    node_index: u32::from_le_bytes(link[..4].try_into().unwrap()) as usize,
                    distance: f32::from_le_bytes(link[4..].try_into().unwrap()),
                });
            }
            node_layers.push(links);
        }
        layers.push(node_layers);
    }

    if payload.remaining != 0 {
        return Err(corrupt("payload length does not match its sections"));
    }
    let computed = payload.hasher.finalize();
    let mut stored = [0u8; 4];
    read_exact(reader, &mut stored)?;
    if computed != u32::from_le_bytes(stored) {
        return Err(SnapshotError::PayloadChecksumMismatch);
    }
    Ok(Decoded {
        vectors,
        ids,
        dup_next,
        layers,
    })
}

/// Derived state of a validated snapshot.
struct Validated {
    mode: IdMode,
    ids: Vec<usize>,
    taken_ids: HashSet<usize>,
    dup_next: Vec<usize>,
}

/// Checks every invariant searches and inserts rely on.
fn validate<const D: usize, DS: Distance<D>>(
    header: &Header,
    dist: &DS,
    decoded: &Decoded<D>,
) -> Result<Validated, SnapshotError> {
    let n = decoded.vectors.len();

    for (i, v) in decoded.vectors.iter().enumerate() {
        dist.validate(v)
            .map_err(|error| corrupt(format!("vector {i}: {error}")))?;
    }

    // graph nodes have layers; duplicate-group members have none
    let is_graph = |i: usize| !decoded.layers[i].is_empty();
    if n == 0 {
        if header.entry_point != 0 || header.max_layer != 0 {
            return Err(corrupt("empty index with a non-zero entry point"));
        }
    } else {
        let entry = usize::try_from(header.entry_point)
            .ok()
            .filter(|&entry| entry < n && is_graph(entry))
            .ok_or_else(|| corrupt("entry point is not a graph node"))?;
        let top = decoded
            .layers
            .iter()
            .map(|layers| layers.len())
            .max()
            .unwrap_or(0);
        if top != header.max_layer as usize + 1 || decoded.layers[entry].len() != top {
            return Err(corrupt("entry point is not on the top layer"));
        }
    }

    let mut seen = HashSet::new();
    for (node, layers) in decoded.layers.iter().enumerate() {
        for (lyr, links) in layers.iter().enumerate() {
            seen.clear();
            for link in links {
                let target = link.node_index;
                if target >= n || decoded.layers[target].len() <= lyr {
                    return Err(corrupt(format!(
                        "node {node} links to {target}, which has no layer {lyr}"
                    )));
                }
                if target == node {
                    return Err(corrupt(format!("node {node} links to itself")));
                }
                if !seen.insert(target) {
                    return Err(corrupt(format!(
                        "node {node} links to {target} twice on layer {lyr}"
                    )));
                }
                if !link.distance.is_finite() {
                    return Err(corrupt(format!(
                        "node {node} has a non-finite link distance"
                    )));
                }
            }
        }
    }

    // each member belongs to exactly one list headed by a graph node and equals it
    let mut dup_next = Vec::with_capacity(n);
    for &next in &decoded.dup_next {
        dup_next.push(match next {
            END_OF_LIST => NO_DUP,
            next if next < n as u64 => next as usize,
            _ => return Err(corrupt("duplicate list points outside the index")),
        });
    }
    let mut in_list = vec![false; n];
    for head in (0..n).filter(|&i| is_graph(i)) {
        let mut member = dup_next[head];
        while member != NO_DUP {
            if is_graph(member) || in_list[member] {
                return Err(corrupt(format!("invalid duplicate list at node {head}")));
            }
            if decoded.vectors[member] != decoded.vectors[head] {
                return Err(corrupt(format!(
                    "duplicate {member} differs from its graph node {head}"
                )));
            }
            in_list[member] = true;
            member = dup_next[member];
        }
    }
    if let Some(orphan) = (0..n).find(|&i| !is_graph(i) && !in_list[i]) {
        return Err(corrupt(format!(
            "node {orphan} is neither linked nor a duplicate"
        )));
    }

    let mode = match IdMode::from_u8(header.id_mode) {
        Some(mode) => mode,
        None if header.id_mode == IdMode::UNSET && n == 0 => IdMode::Positional,
        None => return Err(corrupt("invalid id mode")),
    };
    let mut ids = Vec::with_capacity(n);
    let mut taken = HashSet::new();
    for (position, &id) in decoded.ids.iter().enumerate() {
        let id = usize::try_from(id).map_err(|_| corrupt("id does not fit in usize"))?;
        let valid = match mode {
            IdMode::Positional => id == position,
            IdMode::Explicit => taken.insert(id),
        };
        if !valid {
            return Err(corrupt(format!(
                "invalid or repeated id {id} at {position}"
            )));
        }
        ids.push(id);
    }
    Ok(Validated {
        mode,
        ids,
        taken_ids: taken,
        dup_next,
    })
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path_for(path: &Path) -> io::Result<PathBuf> {
    let name = path
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "path has no file name"))?;
    let unique = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let temp = format!(
        ".{}.{}-{unique}.tmp",
        name.to_string_lossy(),
        std::process::id()
    );
    Ok(path.with_file_name(temp))
}

#[cfg(unix)]
fn sync_parent_dir(path: &Path) -> io::Result<()> {
    let dir = match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir,
        _ => Path::new("."),
    };
    File::open(dir)?.sync_all()
}

/// Windows cannot open a directory handle through `std`; NTFS journals the
/// rename itself. See `docs/SNAPSHOT_FORMAT.md` for the durability guarantees.
#[cfg(not(unix))]
fn sync_parent_dir(_path: &Path) -> io::Result<()> {
    Ok(())
}

impl<const D: usize, DS> Hnsw<D, DS>
where
    DS: Distance<D> + Serialize,
{
    /// Atomically replaces the snapshot at `path`.
    ///
    /// Writes a temporary file in the same directory, flushes and `fsync`s it,
    /// renames it over `path` and (on Unix) `fsync`s the directory. If any step
    /// fails the temporary file is removed and the previous file at `path`, if
    /// any, is left untouched. A crash can leave a stray `.<name>.<pid>-<n>.tmp`
    /// file next to `path`, but never a partially written `path`.
    ///
    /// Consistency: the snapshot is a single point in time. It waits for
    /// in-flight inserts and blocks new ones until it has been written; searches
    /// keep running concurrently.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), SnapshotError> {
        self.save_with(path.as_ref(), &SaveHooks::default())
    }

    pub(crate) fn save_with(
        &self,
        path: &Path,
        hooks: &SaveHooks<'_>,
    ) -> Result<(), SnapshotError> {
        let temp = temp_path_for(path)?;
        let result = (|| {
            hooks.check(SaveStep::CreateTemp)?;
            let file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp)?;
            let mut writer = BufWriter::with_capacity(1 << 20, file);
            self.write_snapshot_with(&mut writer, hooks)?;
            let file = writer.into_inner().map_err(|error| error.into_error())?;
            hooks.check(SaveStep::Sync)?;
            file.sync_all()?;
            drop(file);
            hooks.check(SaveStep::Replace)?;
            fs::rename(&temp, path)?;
            sync_parent_dir(path)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temp);
        }
        result
    }

    /// Writes a snapshot to `writer` (no atomic replacement; see [`Hnsw::save`]).
    pub fn write_snapshot<W: Write>(&self, writer: W) -> Result<(), SnapshotError> {
        self.write_snapshot_with(writer, &SaveHooks::default())
    }

    fn write_snapshot_with<W: Write>(
        &self,
        writer: W,
        hooks: &SaveHooks<'_>,
    ) -> Result<(), SnapshotError> {
        let metric_id = <DS as Distance<D>>::metric_id();
        if metric_id.len() > MAX_METRIC_ID_LEN {
            return Err(SnapshotError::Unsupported(
                "metric id longer than 256 bytes",
            ));
        }
        let metric_params = serde_json::to_vec(&self.dist)
            .map_err(|error| SnapshotError::Metric(error.to_string()))?;
        if metric_params.len() > MAX_METRIC_PARAMS_LEN {
            return Err(SnapshotError::Unsupported(
                "metric parameters larger than 64 KiB",
            ));
        }

        // blocks new inserts and waits for in-flight ones; searches continue
        let _writers = self.update_lock.write().unwrap();
        let storage = self.storage.read().unwrap();
        let (entry_point, max_layer) = *self.entry.read().unwrap();
        let n = storage.data.len();
        if n > u32::MAX as usize {
            return Err(SnapshotError::Unsupported(
                "format version 1 holds at most 2^32 - 1 vectors",
            ));
        }
        let nodes_len: u64 = storage
            .nodes
            .iter()
            .map(|node| {
                1 + node
                    .layers
                    .iter()
                    .map(|layer| 2 + 8 * layer.read().unwrap().len() as u64)
                    .sum::<u64>()
            })
            .sum();
        let payload_len = n as u64 * (4 * D as u64 + 16) + nodes_len;
        let header = Header {
            dimension: D as u32,
            m: self.M as u32,
            m0: self.M0 as u32,
            ef_construction: self.ef_construction as u32,
            seed: self.seed,
            id_mode: self.id_mode.load(Ordering::Acquire),
            max_layer: max_layer as u32,
            node_count: n as u64,
            entry_point: entry_point as u64,
            payload_len,
            metric_id: metric_id.to_owned(),
            metric_params,
        };

        let mut out = PayloadWriter {
            inner: writer,
            hasher: crc32fast::Hasher::new(),
            written: 0,
            fail_after: hooks.fail_after_bytes,
        };
        hooks.check(SaveStep::Write)?;
        out.write_all(&header.encode())?;
        if let Some(while_locked) = hooks.while_locked {
            while_locked();
        }
        out.hasher = crc32fast::Hasher::new();
        let payload_start = out.written;

        for v in &storage.data {
            for x in v {
                out.write_all(&x.to_le_bytes())?;
            }
        }
        for &id in &storage.ids {
            out.write_all(&(id as u64).to_le_bytes())?;
        }
        for &next in &storage.dup_next {
            let next = if next == NO_DUP {
                END_OF_LIST
            } else {
                next as u64
            };
            out.write_all(&next.to_le_bytes())?;
        }
        for node in &storage.nodes {
            out.write_all(&[node.layers.len() as u8])?;
            for layer in &node.layers {
                let links = layer.read().unwrap();
                out.write_all(&(links.len() as u16).to_le_bytes())?;
                for link in links.iter() {
                    out.write_all(&(link.node_index as u32).to_le_bytes())?;
                    out.write_all(&link.distance.to_le_bytes())?;
                }
            }
        }
        if out.written - payload_start != payload_len {
            return Err(SnapshotError::Io(io::Error::other(
                "snapshot size changed while writing",
            )));
        }
        let crc = out.hasher.clone().finalize();
        out.write_all(&crc.to_le_bytes())?;
        out.flush()?;
        Ok(())
    }
}

impl<const D: usize, DS> Hnsw<D, DS>
where
    DS: Distance<D> + DeserializeOwned,
{
    /// Loads a snapshot written by [`Hnsw::save`].
    ///
    /// The file must be exactly one snapshot of this format version, dimension
    /// and metric (`Distance::metric_id`). Sizes are checked against the file
    /// length before allocating, both checksums are verified, and the graph is
    /// validated (parameters, entry point, levels, link targets and limits, self
    /// and repeated links, duplicate groups, ids, finite values) before the index
    /// is returned. Unversioned files from before v0.1 return
    /// [`SnapshotError::NotASnapshot`].
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, SnapshotError> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        let mut reader = BufReader::with_capacity(1 << 20, file);
        Self::read_snapshot_limited(&mut reader, len, true)
    }

    /// Reads one snapshot from `reader`, rejecting any that declares more than
    /// `max_bytes` bytes. Bytes after the snapshot are not read.
    pub fn read_snapshot<R: Read>(mut reader: R, max_bytes: u64) -> Result<Self, SnapshotError> {
        Self::read_snapshot_limited(&mut reader, max_bytes, false)
    }

    fn read_snapshot_limited<R: Read>(
        reader: &mut R,
        limit: u64,
        exact: bool,
    ) -> Result<Self, SnapshotError> {
        let (header, header_len) = Header::read(reader)?;
        if header.dimension as u64 != D as u64 {
            return Err(SnapshotError::DimensionMismatch {
                expected: D,
                found: header.dimension as u64,
            });
        }
        let expected_metric = <DS as Distance<D>>::metric_id();
        if expected_metric != header.metric_id {
            return Err(SnapshotError::MetricMismatch {
                expected: expected_metric.to_owned(),
                found: header.metric_id,
            });
        }
        let dist = DS::deserialize(&mut serde_json::Deserializer::from_slice(
            &header.metric_params,
        ))
        .map_err(|error| SnapshotError::Metric(error.to_string()))?;
        check_graph_params(
            header.m as usize,
            header.m0 as usize,
            header.ef_construction as usize,
        )
        .map_err(SnapshotError::InvalidConfig)?;
        if header.max_layer as usize >= MAX_LAYERS {
            return Err(corrupt(format!("max layer {}", header.max_layer)));
        }
        let total = header_len
            .checked_add(header.payload_len)
            .and_then(|len| len.checked_add(4))
            .ok_or_else(|| corrupt("declared size overflows"))?;
        if total > limit {
            return Err(if exact {
                SnapshotError::Truncated
            } else {
                SnapshotError::TooLarge {
                    declared: total,
                    limit,
                }
            });
        }
        if exact && total < limit {
            return Err(SnapshotError::TrailingData);
        }

        let decoded = read_payload::<R, D>(reader, &header)?;
        let Validated {
            mode,
            ids,
            taken_ids,
            dup_next,
        } = validate(&header, &dist, &decoded)?;

        let n = decoded.vectors.len();
        // one level draw per stored vector, as when the vectors were inserted
        let mut rng = StdRng::seed_from_u64(header.seed);
        for _ in 0..n {
            let _: f64 = Open01.sample(&mut rng);
        }
        let nodes = decoded
            .layers
            .into_iter()
            .map(|layers| Node {
                layers: layers.into_iter().map(RwLock::new).collect(),
            })
            .collect();
        let mut index = Self::new_unchecked(
            header.m as usize,
            header.m0 as usize,
            header.ef_construction as usize,
            header.seed,
            dist,
        );
        index.storage = RwLock::new(Storage {
            data: decoded.vectors,
            nodes,
            dup_next,
            ids,
        });
        index.entry = RwLock::new((header.entry_point as usize, header.max_layer as usize));
        index.rng = Mutex::new(rng);
        let stored_mode = if n == 0 && header.id_mode == IdMode::UNSET {
            IdMode::UNSET
        } else {
            mode as u8
        };
        index.id_mode = AtomicU8::new(stored_mode);
        index.taken_ids = Mutex::new(taken_ids);
        Ok(index)
    }
}

#[cfg(test)]
mod tests;
