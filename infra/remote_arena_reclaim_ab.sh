#!/usr/bin/env bash
# J7 arena-reclaim certification A/B: two same-box cells — defaults (chunk
# recycling + freeze-demotion lift, LOHALLOC_ARENA_RECLAIM unset) vs
# LOHALLOC_ARENA_RECLAIM=0 (the pre-J7 one-shot-budget arena, byte-exact) —
# each running the classic native timing suite PLUS the realistic workloads
# PLUS the peak-RSS pass. Ship/no-ship rows: request-loop family (wall +
# RSS — the target), rust/arena + cpp/mt-slab-t8 (J4-D's certified killers —
# must not regress vs the =0 cell), cpp/arena + cpp-string, adv-mixed
# (light-arena beneficiaries — must hold), mt-mixed at family level only.
#
# Launch (billable):
#   REMOTE_SCRIPT=infra/remote_arena_reclaim_ab.sh infra/cloud_provision.sh c9g.4xlarge
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

BASE="results/$(date +%Y%m%dT%H%M%S)-arena-reclaim-ab"
echo "== J7 arena-reclaim A/B: 2 cells under $BASE =="

run_cell() {
    local name="$1"; shift
    local RUN_DIR="$BASE/$name"
    echo "== cell $name -> $RUN_DIR =="
    # Classic native timing suite (the workload lists include the realistic
    # request-loop/json-tree/kv-store rows by default).
    env "$@" make bench-native-host RUN_DIR="$RUN_DIR"
    # Peak-RSS pass over the same matrix (the paper-vs-product axis J7
    # exists to move).
    env "$@" make bench-rss-host RUN_DIR="$RUN_DIR"
    make bench-report RUN_DIR="$RUN_DIR"
}

run_cell "reclaim-on"
run_cell "reclaim-off" LOHALLOC_ARENA_RECLAIM=0

echo "Done. $BASE is ready for retrieval (scp)."
