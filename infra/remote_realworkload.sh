#!/usr/bin/env bash
# Real-workload certification run: the request-loop / json-tree / kv-store
# realistic-allocation-pattern benchmarks, measured on BOTH axes — wall time
# (bench-native-host) and peak RSS (bench-rss-host) — across
# lohalloc/system/jemalloc/mimalloc in C, C++, and Rust. This is the
# paper-vs-product experiment: does the learning allocator win realistic
# patterns on speed and/or memory?
#
# Launch via the decoupled hardened flow (survives poller kills; self-terminate
# net guards against leaks):
#   REMOTE_SCRIPT=infra/remote_realworkload.sh bash infra/cloud_provision.sh c9g.4xlarge
#   POLL_MINUTES=15 bash infra/cloud_collect.sh   # re-run until DONE
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

# The three realistic patterns only (no synthetic rows — this run is the
# real-workload verdict). run_native.sh honors WORKLOADS for every lang.
export WORKLOADS="request-loop json-tree kv-store"
RD="results/$(date +%Y%m%dT%H%M%S)-realworkload"

echo "== [1/3] Native timing (wall) — $WORKLOADS =="
make bench-native-host RUN_DIR="$RD"
echo "== [2/3] Peak RSS (memory footprint) — $WORKLOADS =="
make bench-rss-host RUN_DIR="$RD"
echo "== [3/3] Aggregate report + graphs =="
make bench-report RUN_DIR="$RD"

echo "Done. $RD is ready for retrieval (scp)."
