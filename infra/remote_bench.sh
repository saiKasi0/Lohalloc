#!/usr/bin/env bash
# Runs on each provisioned EC2 instance (via SSH from .github/workflows/bench.yml)
# to install the toolchain and run the full Phase 6 suite natively — no
# Docker needed here since the instance already *is* a dedicated Linux box
# for its architecture (x86_64 c6i.large / ARM64 c6g.large).
set -euo pipefail

echo "== Installing toolchain =="
sudo apt-get update -y
sudo apt-get install -y build-essential cmake git pkg-config valgrind \
    libjemalloc2 libjemalloc-dev
# perf (bench-perf) is best-effort: linux-tools is kernel-version-specific and
# the generic metapackage sometimes lags the running kernel on fresh AMIs.
# Never fail the run if it can't install — the probe below gates bench-perf.
sudo apt-get install -y "linux-tools-$(uname -r)" linux-tools-common 2>/dev/null \
    || sudo apt-get install -y linux-tools-generic 2>/dev/null || true

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

# One timestamped run directory shared across every step (each `make` call
# below would otherwise mint its own $(RUN_DIR) and fragment the results).
RUN_DIR="results/$(date +%Y%m%dT%H%M%S)"
echo "== Run directory: $RUN_DIR =="

# EVERY benchmark available — the full suite:
#   make bench          latency_profile + criterion (backend_micro,
#                       hypothesis, inference_overhead, comparison ×4 baselines)
#   make bench-native-host   native C/C++/Rust hyperfine (the 41 vs-jemalloc rows)
#   make bench-cache-host    cachegrind D1/LL miss-rate simulation
#   make bench-perf          real-PMU MT diagnostics (probe-gated; informational)
#   make bench-report        aggregate -> report + graphs
echo "== [1/5] Rust criterion + latency_profile suite (make bench) =="
make bench RUN_DIR="$RUN_DIR"

echo "== [2/5] Native (LD_PRELOAD) cross-allocator timing suite =="
make bench-native-host RUN_DIR="$RUN_DIR"

echo "== [3/5] Native cachegrind cache-miss suite =="
make bench-cache-host RUN_DIR="$RUN_DIR"

# [4/5] Real-PMU MT diagnostics (perf). Cachegrind is a single-thread sim and
# cannot see cross-core coherence / false sharing; perf can. But PMU exposure
# is instance-dependent (Nitro virtualizes it; Graviton non-metal often lacks
# LLC events), so gate on the probe and NEVER fail the run — this is an
# informational, ungated table. Opt out entirely with RUN_PERF=0.
echo "== [4/5] Real-PMU MT diagnostics (perf) =="
if [ "${RUN_PERF:-1}" = 1 ]; then
    # LLC/kernel events need paranoid <= 1; best-effort, ignore if locked down.
    sudo sysctl -w kernel.perf_event_paranoid=1 >/dev/null 2>&1 || true
    if bash bench/probe_perf.sh; then
        echo "== perf PMU sufficient — running bench-perf =="
        make bench-perf RUN_DIR="$RUN_DIR" || echo "WARNING: bench-perf failed — continuing (informational only)"
    else
        echo "== perf PMU insufficient on this instance — skipping bench-perf =="
        echo "   (use a .metal instance for real LLC/coherence events)"
    fi
else
    echo "== RUN_PERF=0 — skipping perf pass =="
fi

echo "== [5/5] Aggregating =="
make bench-report RUN_DIR="$RUN_DIR"

echo "Done. $RUN_DIR is ready for retrieval (scp)."
