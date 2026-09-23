#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_ROOT="$SCRIPT_DIR/configs"

run_config() {
  local config="$1"
  echo "==> $config"
  (
    cd "$REPO_ROOT"
    RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}" \
      cargo run --release --locked -p hnsw-bench --features hdf5 -- "$config"
  )
}

build_sequential() {
  run_config "$CONFIG_ROOT/build/m-sweep/sift-1m.toml"
  run_config "$CONFIG_ROOT/build/m-sweep/mnist-60k.toml"
  run_config "$CONFIG_ROOT/build/size-sweep/sift-100k.toml"
  run_config "$CONFIG_ROOT/build/size-sweep/sift-250k.toml"
  run_config "$CONFIG_ROOT/build/size-sweep/sift-500k.toml"
  run_config "$CONFIG_ROOT/build/size-sweep/sift-1m.toml"
}

build_parallel() {
  run_config "$CONFIG_ROOT/build/ef-construction-sweep/sift-1m.toml"
  run_config "$CONFIG_ROOT/build/parallel-construction/sift-1m.toml"
  run_config "$CONFIG_ROOT/build/parallel-construction/size-sift-100k.toml"
  run_config "$CONFIG_ROOT/build/parallel-construction/size-sift-250k.toml"
  run_config "$CONFIG_ROOT/build/parallel-construction/size-sift-500k.toml"
  run_config "$CONFIG_ROOT/build/parallel-construction/size-sift-1m.toml"
}

build() {
  case "${1:-both}" in
  sequential)
    build_sequential
    ;;
  parallel)
    build_parallel
    ;;
  both)
    build_sequential
    build_parallel
    ;;
  *)
    printf 'usage: %s build {sequential|parallel|both}\n' "$0" >&2
    exit 2
    ;;
  esac
}

measure() {
  run_config "$CONFIG_ROOT/measure/m-sweep/sift-1m-normal.toml"
  run_config "$CONFIG_ROOT/measure/m-sweep/mnist-60k-normal.toml"
  run_config "$CONFIG_ROOT/measure/size-sweep/sift-100k-normal.toml"
  run_config "$CONFIG_ROOT/measure/size-sweep/sift-250k-normal.toml"
  run_config "$CONFIG_ROOT/measure/size-sweep/sift-500k-normal.toml"
  run_config "$CONFIG_ROOT/measure/size-sweep/sift-1m-normal.toml"
  run_config "$CONFIG_ROOT/measure/pq/m-sweep/sift-1m-pq32.toml"
  run_config "$CONFIG_ROOT/measure/pq/m-sweep/sift-1m-pq64.toml"
  run_config "$CONFIG_ROOT/measure/pq/m-sweep/sift-1m-pq128.toml"
}

plot() {
  if command -v uv >/dev/null 2>&1; then
    uv run "$SCRIPT_DIR/plot.py"
  else
    python3 "$SCRIPT_DIR/plot.py"
  fi
}

stage="${1:-all}"
case "$stage" in
build)
  build "${2:-both}"
  ;;
measure)
  measure
  ;;
plot)
  plot
  ;;
all)
  build both
  measure
  plot
  ;;
*)
  printf 'usage: %s {build [sequential|parallel|both]|measure|plot|all}\n' "$0" >&2
  exit 2
  ;;
esac
