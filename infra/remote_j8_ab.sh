#!/usr/bin/env bash
# J8 certification A/B: two same-box cells isolating the J8 refinements on
# top of the certified J7 baseline —
#   j8-on  : defaults (J8-A arena self-flush + J8-B slab scan-gate, both on)
#   j8-off : LOHALLOC_ARENA_SELF_FLUSH=0 LOHALLOC_SLAB_SCAN_GATE=0
#            (== the shipped J7 behavior, byte-exact modulo trivial always-on
#            RecycleStats counters that don't affect routing)
# each running the classic native timing suite + the realistic workloads +
# the peak-RSS pass.
#
# What J8 is expected to move (ship/no-ship rows):
#   * json-tree / kv-store (all langs)  — J8-B: the futile sibling-scan tax
#     on carve-bound workloads (local: json-tree sibling_steps 115k->120,
#     wall +22%). PRIMARY WIN ROW.
#   * request-loop family (wall + RSS)  — J8-A: tighter arena rotation
#     (local: RSS 7.7->6.8 MB, one fewer mapped chunk, skip_pinned 42->0).
#   * mt-xfree-t4/t8                     — J8-B WATCH-COST: single-class
#     true-sharing on the one hot gain-epoch line (local: ~4% at t8). This
#     is a lohalloc WIN row (Act V) — it must not flip to a loss vs j8-off.
#   * mt-slab / mt-mixed / adv-mixed     — must hold (multi-class false
#     sharing is padded away; read mt-mixed at family level, bimodal rule).
#   * slab family (all langs)            — must hold (J6's territory).
#
# Launch (billable, c9g.4xlarge only — always destroy after):
#   REMOTE_SCRIPT=infra/remote_j8_ab.sh infra/cloud_provision.sh c9g.4xlarge
#   infra/cloud_collect.sh
set -euo pipefail

echo "== Installing toolchain =="
sudo apt-get update -y
sudo apt-get install -y build-essential cmake git pkg-config \
    libjemalloc2 libjemalloc-dev

if [ ! -f /usr/local/lib/libmimalloc.so ]; then
    echo "== Building mimalloc from source =="
    git clone --depth 1 --branch v2.1.7 https://github.com/microsoft/mimalloc.git /tmp/mimalloc
    cmake -S /tmp/mimalloc -B /tmp/mimalloc/build -DCMAKE_BUILD_TYPE=Release
    cmake --build /tmp/mimalloc/build --parallel
    sudo cmake --install /tmp/mimalloc/build
    sudo ldconfig
    rm -rf /tmp/mimalloc
fi

if ! command -v cargo >/dev/null 2>&1; then
    echo "== Installing Rust =="
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
fi
# shellcheck disable=SC1091
source "$HOME/.cargo/env"

if ! command -v hyperfine >/dev/null 2>&1; then
    echo "== Installing hyperfine =="
    cargo install hyperfine --locked
fi

BASE="results/$(date +%Y%m%dT%H%M%S)-j8-ab"
echo "== J8 A/B: 2 cells under $BASE =="

run_cell() {
    local name="$1"; shift
    local RUN_DIR="$BASE/$name"
    echo "== cell $name -> $RUN_DIR =="
    # Classic native timing suite (workload lists already include the
    # realistic request-loop/json-tree/kv-store rows).
    env "$@" make bench-native-host RUN_DIR="$RUN_DIR"
    # Peak-RSS pass over the same matrix (the J8-A axis).
    env "$@" make bench-rss-host RUN_DIR="$RUN_DIR"
    make bench-report RUN_DIR="$RUN_DIR"
}

run_cell "j8-on"
run_cell "j8-off" LOHALLOC_ARENA_SELF_FLUSH=0 LOHALLOC_SLAB_SCAN_GATE=0

echo "Done. $BASE is ready for retrieval (scp)."
