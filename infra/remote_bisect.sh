#!/usr/bin/env bash
# J5 Gate 1 bisect: runs the NATIVE TIMING suite (only) 4 times on one
# provisioned box, one cell per (LOHALLOC_STRIPES × LOHALLOC_DEMOTE_FRACTION)
# combination, isolating the two behavioral changes that shipped together in
# the J5 Gate 1 bundle (certified NET REGRESSION, results/20260710T194620):
#
#   (8,  0.0)  ≈ J4-C behavior + CachePadded + clamp — clears the benign suspects
#   (8,  0.10) demotion alone
#   (16, 0.0)  stripes alone
#   (16, 0.10) = J5 Gate 1 as shipped — must reproduce 194620 (plumbing sanity)
#
# One provisioning, four timing cells (~15-20 min each; no criterion, no
# cachegrind, no perf — timing verdict only). Builds are shared across cells:
# both knobs are runtime env vars (see crates/lohalloc-alloc: stripe_mask()'s
# LOHALLOC_STRIPES getenv + the demote_fraction tune key), which is the whole
# point of this script existing instead of four provisionings.
#
# All cells nest under ONE parent results dir because cloud_bench.sh's
# pull_results scp's only the single newest results/*/ directory.
#
# Launch (billable, user-run):
#   REMOTE_SCRIPT=infra/remote_bisect.sh bash infra/cloud_bench.sh c9g.4xlarge
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

BASE="results/$(date +%Y%m%dT%H%M%S)-bisect"
echo "== J5 Gate 1 bisect: stripes x demote_fraction, 4 cells under $BASE =="

for cell in "8 0.0" "8 0.10" "16 0.0" "16 0.10"; do
    read -r s f <<<"$cell"
    RUN_DIR="$BASE/s${s}-f${f}"
    echo "== cell stripes=$s demote_fraction=$f -> $RUN_DIR =="
    # Knobs travel by env: make -> recipe shell -> run_native.sh, which
    # forwards them into the hyperfine command strings of every lohalloc
    # triple leg (train/export/inference) — see run_lohalloc_triple.
    LOHALLOC_STRIPES="$s" LOHALLOC_DEMOTE_FRACTION="$f" \
        make bench-native-host RUN_DIR="$RUN_DIR"
    make bench-report RUN_DIR="$RUN_DIR"
done

echo "Done. $BASE is ready for retrieval (scp)."
