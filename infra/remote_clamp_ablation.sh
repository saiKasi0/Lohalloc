#!/usr/bin/env bash
# clamp_percentile ablation (headerless HELD ON): runs the NATIVE TIMING suite
# once per LOHALLOC_CLAMP_PERCENTILE value on ONE provisioned box, to ISOLATE
# Task B (the per-arm percentile winsorizer) now that Task A (headerless training)
# is a keeper. The combined-A/B run (results/20260711T140456_c9g.4xlarge) showed a
# net win but a REDISTRIBUTION — big arena/slab fixes, a mixed/adv-mixed family
# regression. This sweep answers two questions at once:
#   1. Which clamp_percentile is best for the suite?
#   2. Is the winsorizer the CAUSE of the mixed regression, or is it
#      headerless-dropped dealloc attribution (Part 2)?
#
# Cells (all headerless-on = the shipped default; only the clamp knob varies):
#   off  LOHALLOC_CLAMP_PERCENTILE=0   winsorization DISABLED (headerless still on)
#   p85  LOHALLOC_CLAMP_PERCENTILE=85
#   p90  LOHALLOC_CLAMP_PERCENTILE=90  the current default
#   p95  LOHALLOC_CLAMP_PERCENTILE=95
#
# WHY: clamp_percentile is reward-SHAPING (changes the trained model → inference
# routing → timing). Local M-series numbers are directional; the decision needs a
# certified c9g sweep. All cells share one provisioning; the knob is a runtime env
# var (crates/lohalloc-alloc/src/tune.rs `clamp_percentile` +
# bench/run_native.sh forwarding), so no rebuild between cells. Headerless is left
# at its default-ON (no LOHALLOC_TRAIN_HEADERLESS in any cell), so this sweep does
# NOT re-test Task A — the pre-headerless baseline is the `ctrl` cell of the prior
# combined run (results/20260711T140456_c9g.4xlarge/ctrl), compare against it.
#
# Decision: if a percentile (likely p95/off) recovers the mixed/adv-mixed rows
# WITHOUT losing the arena wins, adopt it as the `clamp_percentile` default. If
# ALL cells keep the mixed rows regressed vs the pre-headerless ctrl, the
# winsorizer is exonerated and the regression is dropped dealloc attribution →
# Part 2 (bilateral-reward recovery) is the fix.
#
# The RETIRED fixed `latency_clamp_ns=1000` clamp's certified result lives in
# COPILOT.md for reference (results/20260711T121357_c9g.4xlarge).
#
# One provisioning, four timing cells (~15-20 min each; no criterion, no
# cachegrind, no perf — timing verdict only). All cells nest under ONE parent
# results dir because cloud_bench.sh's pull_results scp's only the single
# newest results/*/ directory.
#
# Launch (billable, user-run):
#   REMOTE_SCRIPT=infra/remote_clamp_ablation.sh bash infra/cloud_bench.sh c9g.4xlarge
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

# Each cell is "name:ENV=VAL[ ENV=VAL...]" — the env prefix is applied to the
# whole make invocation, then forwarded by run_native.sh into the train legs.
# Headerless is left default-ON (no LOHALLOC_TRAIN_HEADERLESS) in every cell.
#
# BOTH ablations are CERTIFIED (2026-07-11):
# - clamp sweep (results/20260711T163352): p90/p85 beat clamp-off (mean 1.229
#   vs 1.262); winsorizer exonerated for the mixed regression → p90 stays.
# - rt A/B (results/20260711T183928): rt-off WON decisively (mean 1.181 vs
#   1.235; every mt-mixed-t4/t8 row +0.42–0.46 with the ring on) → the ring is
#   now DEFAULT-OFF (LOHALLOC_REWARD_TRACK=1 opts in). rt-off = best certified
#   state: mean 1.181, worst 1.810, mixed family recovered to pre-headerless
#   parity by the Part-3 size-aware context register.
#
# Historical cell configs (note: after the default flip, the ring A/B is
# "rt-on:LOHALLOC_REWARD_TRACK=1" vs "rt-off:" — semantics inverted vs the
# certified run, which predated the flip):
#   CELLS=( "rt-on:LOHALLOC_REWARD_TRACK=1" "rt-off:" )
CELLS=(
    "off:LOHALLOC_CLAMP_PERCENTILE=0"
    "p85:LOHALLOC_CLAMP_PERCENTILE=85"
    "p90:LOHALLOC_CLAMP_PERCENTILE=90"
    "p95:LOHALLOC_CLAMP_PERCENTILE=95"
)

BASE="results/$(date +%Y%m%dT%H%M%S)-clamp-sweep"
echo "== clamp_percentile sweep (headerless on): ${#CELLS[@]} cells under $BASE =="

for cell in "${CELLS[@]}"; do
    name="${cell%%:*}"
    env_prefix="${cell#*:}"
    RUN_DIR="$BASE/$name"
    echo "== cell $name  [${env_prefix:-defaults}] -> $RUN_DIR =="
    # The knobs travel by env: make -> recipe shell -> run_native.sh, which
    # forwards LOHALLOC_TRAIN_HEADERLESS + LOHALLOC_CLAMP_PERCENTILE into the
    # hyperfine command strings of the train + train/export legs of every
    # lohalloc triple (reward shaping only bites while training).
    env $env_prefix make bench-native-host RUN_DIR="$RUN_DIR"
    make bench-report RUN_DIR="$RUN_DIR"
done

echo "Done. $BASE is ready for retrieval (scp)."
