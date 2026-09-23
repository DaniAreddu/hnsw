# Snapshot format and crash safety (v0.1)

`Hnsw::save` / `Hnsw::load` (files) and `Hnsw::write_snapshot` /
`Hnsw::read_snapshot` (streams) use **snapshot format version 1**
(`hnsw::SNAPSHOT_FORMAT_VERSION`). All integers are little-endian.

## Layout

| Section | Field | Type | Notes |
| --- | --- | --- | --- |
| header | magic | `[u8; 8]` | `b"HNSWSNAP"` |
| | format version | `u32` | `1` |
| | header length | `u32` | whole header including its CRC, 82 bytes to 82 + 256 + 64 KiB |
| | dimension | `u32` | must equal `D` of the loading type |
| | `M`, `M0`, `ef_construction` | `u32` × 3 | validated like `Hnsw::try_new` |
| | seed | `u64` | RNG seed; the RNG is replayed by one draw per stored vector |
| | id mode | `u8` | 0 unset (empty index only), 1 positional, 2 explicit |
| | reserved | `[u8; 3]` | must be zero |
| | max layer | `u32` | < 64 |
| | node count | `u64` | < 2³² in version 1 |
| | entry point | `u64` | |
| | payload length | `u64` | |
| | metric id | `u16` length + UTF-8 | `Distance::metric_id()`, ≤ 256 bytes |
| | metric parameters | `u32` length + JSON | `serde_json` of the metric, ≤ 64 KiB |
| | header CRC-32 | `u32` | over every preceding header byte |
| payload | vectors | `node_count × D × f32` | |
| | ids | `node_count × u64` | |
| | duplicate lists | `node_count × u64` | next member of the node's duplicate group, `u64::MAX` = end |
| | nodes | per node | `layer_count: u8` (0 for duplicate-group members), then per layer `link_count: u16` and `link_count × (target: u32, distance: f32)` |
| trailer | payload CRC-32 | `u32` | over the payload bytes |

## What `load` checks before returning an index

1. Magic → otherwise `NotASnapshot`. **Unversioned files written before v0.1
   (bincode, no header) are not supported and always fail here**; rebuild such
   indexes. They are never parsed heuristically.
2. Version → `UnsupportedVersion`; header length bounds; header CRC →
   `HeaderChecksumMismatch`; reserved bytes and field lengths.
3. Dimension → `DimensionMismatch`; metric id → `MetricMismatch` (checked
   before the parameters are parsed; bad parameters → `Metric`); graph
   parameters → `InvalidConfig`; max layer.
4. Declared size against the file length (`Truncated` / `TrailingData`) or the
   caller's `max_bytes` (`TooLarge`). The node count must fit in the payload
   (≥ `4·D + 17` bytes per node), link counts must not exceed `M0`/`M`, layer
   counts must not exceed 64 — all before the corresponding allocation, so a
   hostile file cannot allocate much more than its own size.
5. Payload CRC → `PayloadChecksumMismatch`; exact payload length.
6. Structure → `Corrupt(reason)`: every vector passes `Distance::validate`
   (finite, and within the overflow bound for `L2Squared`); the entry point is a
   graph node on the top layer and the top layer equals the stored max layer;
   every link target exists and has that layer, no self-links, no repeated
   links, finite link distances; every duplicate-group member is in exactly one
   list headed by a graph node and equals its vector, and every layerless node
   is such a member; positional ids equal their position, explicit ids are
   unique.

Tests: `tests/snapshots.rs`, `tests/snapshot_safety.rs`, the in-module tests in
`src/disk/tests.rs` (each structural rule is violated in a file whose checksums
are recomputed, so only the validator can reject it), and the fuzz corpus
replay in `tests/snapshot_fuzz_regression.rs`. `fuzz/` holds a cargo-fuzz
target (`cargo +nightly fuzz run load_snapshot fuzz/corpus/load_snapshot`)
that also recomputes checksums so mutations reach the validator; CI runs it
for 90 seconds.

## Crash-safe replacement

`save(path)`:

1. creates `.<file name>.<pid>-<counter>.tmp` in the **same directory** with
   `create_new`;
2. writes the complete snapshot through a buffered writer and flushes it;
3. `fsync`s the file (`File::sync_all`; `FlushFileBuffers` on Windows);
4. renames it over `path` with `std::fs::rename` (`rename(2)` on Unix, a
   replace-existing rename on Windows);
5. on Unix, `fsync`s the parent directory so the rename itself is durable.

If any step fails, the temporary file is removed and `path` still holds the
previous snapshot (tested by injecting a failure at every step and after 10 and
500 written bytes). A crash (power loss, `kill -9`) between steps leaves either
the old or the new complete file at `path`, plus possibly a stray `.tmp` file,
which `load` never reads and which can be deleted.

Platform limits:

- **Windows:** `std` offers no directory `fsync`; durability of the rename
  relies on NTFS metadata journaling. Replacing fails if another process holds
  `path` open without `FILE_SHARE_DELETE`; `save` then returns the error and the
  old file is kept.
- The temporary file must be on the same filesystem as `path` (it is, being in
  the same directory); network filesystems may not provide atomic rename.
- `write_snapshot` to an arbitrary writer has no atomicity: it is the caller's
  stream.

## Consistency and blocking

A snapshot is a single point in time: `write_snapshot` takes the index's writer
lock, which waits for in-flight inserts and holds back new ones until the
snapshot has been written, and reads storage under a shared lock. **Searches are
not blocked; inserts are blocked for the duration of the save.** Explicit ids
reserved by an insert that has not yet stored its vector are not part of the
snapshot.

Measured with `cargo run --release --example snapshot_contention -- 100000`
(Intel Core i5-8365U 4C/8T, 16 GB, Windows 11, rustc 1.97.1, release build;
random 128-D vectors, M = 16, ef_construction = 100; one search thread at
k = 10, ef_search = 64 and one insert thread running throughout):

| phase | window | searches | search p50 | search p99 | search max | inserts | insert p50 | insert max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| idle | 1500 ms | 1972 | 0.909 ms | 3.765 ms | 32.5 ms | 128 | 11.400 ms | 67.7 ms |
| during save | 1312 ms | 1915 | 0.885 ms | 2.734 ms | 4.4 ms | 41 | 13.145 ms | 1116.0 ms |

The 71 MiB snapshot (100,163 vectors) took 1.3 s to write and 0.53 s to load and
validate. Search latency is unchanged during the save; the longest insert
(1.1 s) is the insert that waited for the snapshot to finish.
