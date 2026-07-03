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

echo "== Running Rust criterion + latency_profile suite =="
make bench

echo "== Running native (LD_PRELOAD) cross-allocator timing suite =="
make bench-native-host

echo "== Running native cachegrind cache-miss suite =="
make bench-cache-host

echo "== Aggregating =="
make bench-report

echo "Done. results/ is ready for retrieval (scp)."
