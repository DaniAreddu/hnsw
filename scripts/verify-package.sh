#!/usr/bin/env bash
# Packages hnsw-pq and hnsw exactly as `cargo publish` would, extracts the
# .crate archives into a directory outside the repository, and builds and runs
# a standalone consumer against them, with and without experimental-pq.
#
# usage: scripts/verify-package.sh [work-dir]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${1:-$(mktemp -d)}"
mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd)"
case "$WORK/" in
"$ROOT"/*)
  echo "work dir must be outside the repository: $WORK" >&2
  exit 2
  ;;
esac

cd "$ROOT"
cargo package -p hnsw-pq -p hnsw --locked
version() { cargo pkgid -p "$1" | sed 's/.*[#@]//'; }
HNSW="hnsw-$(version hnsw)"
PQ="hnsw-pq-$(version hnsw-pq)"

rm -rf "$WORK/crates" "$WORK/consumer"
mkdir -p "$WORK/crates"
tar -xzf "target/package/$HNSW.crate" -C "$WORK/crates"
tar -xzf "target/package/$PQ.crate" -C "$WORK/crates"
cp -R scripts/package-consumer "$WORK/consumer"
cat >>"$WORK/consumer/Cargo.toml" <<EOF

[patch.crates-io]
hnsw = { path = "../crates/$HNSW" }
hnsw-pq = { path = "../crates/$PQ" }
EOF

# nothing from the repository may leak into the build
if grep -nE 'path *= *"(\.\./|/|[A-Za-z]:|crates/)' "$WORK/crates/$HNSW/Cargo.toml" "$WORK/crates/$PQ/Cargo.toml"; then
  echo "packaged manifests still contain path dependencies" >&2
  exit 1
fi

cd "$WORK/consumer"
export CARGO_TARGET_DIR="$WORK/target"
cargo run --quiet
cargo run --quiet --features pq
cargo tree --features pq --prefix none | grep -E '^hnsw(-pq)? ' | sort -u
echo "verified packages: $HNSW, $PQ (work dir $WORK)"
