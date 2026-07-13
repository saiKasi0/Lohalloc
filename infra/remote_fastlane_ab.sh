#!/usr/bin/env bash
# J6 fast-lane certification A/B: runs the NATIVE TIMING suite twice on one
# provisioned box — lane ON (shipped default) vs LOHALLOC_FAST_LANE=0 (the
# kill switch reverts to the exact classic pin path) — so every row's delta
# is attributable to the tcache-shaped alloc/free lanes alone, same-box,
# same build. Two timing cells only (~15-20 min each; no criterion, no
# cachegrind, no perf — wall-time verdict; the dref story is already
# measured locally in Docker).
#
# Both cells nest under ONE parent results dir because the collect flow
# pulls only the single newest results/*/ directory.
#
# Launch (billable):
#   REMOTE_SCRIPT=infra/remote_fastlane_ab.sh infra/cloud_provision.sh c9g.4xlarge
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

BASE="results/$(date +%Y%m%dT%H%M%S)-fastlane-ab"
echo "== J6 fast-lane A/B: 2 cells under $BASE =="

# Cell 1: lane ON (shipped default — env unset).
RUN_DIR="$BASE/lane-on"
echo "== cell lane=on -> $RUN_DIR =="
make bench-native-host RUN_DIR="$RUN_DIR"
make bench-report RUN_DIR="$RUN_DIR"

# Cell 2: lane OFF (kill switch; forwarded into every lohalloc triple leg
# by run_native.sh — see its LOHALLOC_FAST_LANE forwarding).
RUN_DIR="$BASE/lane-off"
echo "== cell lane=off -> $RUN_DIR =="
LOHALLOC_FAST_LANE=0 make bench-native-host RUN_DIR="$RUN_DIR"
make bench-report RUN_DIR="$RUN_DIR"

echo "Done. $BASE is ready for retrieval (scp)."
