#!/usr/bin/env bash
# J6-D certification A/B: NATIVE TIMING suite twice on one box —
# new defaults (survivors-only distilled exclusion + fast-lane arm-gate)
# vs LOHALLOC_PIN_EXCLUDE_LEGACY=1 (the pre-D2 candidate-based exclusion,
# i.e. the just-certified 205626 freeze behavior). Isolates the D2
# consistency fix; the D1 arm-gate is safe-by-construction (strictly
# removes dead work) and stays on in both cells. Expected: engagement-
# lottery rows (slab family and possibly the mixed bimodal family)
# stabilize / improve in the defaults cell; everything else within noise.
#
# Launch (billable):
#   REMOTE_SCRIPT=infra/remote_j6d_ab.sh infra/cloud_provision.sh c9g.4xlarge
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

BASE="results/$(date +%Y%m%dT%H%M%S)-j6d-ab"
echo "== J6-D A/B: 2 cells under $BASE =="

# Cell 1: new defaults (survivors-only exclusion).
RUN_DIR="$BASE/exclude-survivors"
echo "== cell exclude=survivors (defaults) -> $RUN_DIR =="
make bench-native-host RUN_DIR="$RUN_DIR"
make bench-report RUN_DIR="$RUN_DIR"

# Cell 2: legacy candidate-based exclusion (the 205626 freeze behavior).
RUN_DIR="$BASE/exclude-legacy"
echo "== cell exclude=legacy -> $RUN_DIR =="
LOHALLOC_PIN_EXCLUDE_LEGACY=1 make bench-native-host RUN_DIR="$RUN_DIR"
make bench-report RUN_DIR="$RUN_DIR"

echo "Done. $BASE is ready for retrieval (scp)."
