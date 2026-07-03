#!/usr/bin/env bash
# Cross-allocator, cross-language native benchmark runner. Uses `hyperfine`
# to invoke bench/native's C and C++ harness binaries repeatedly under
# different LD_PRELOAD allocators, with warmup and statistical rigor,
# exporting JSON for the Milestone D aggregator.
#
# True LD_PRELOAD interposition only works on Linux — this script still
# runs the "system" (no LD_PRELOAD) baseline on macOS for local iteration,
# but skips the LD_PRELOAD'd allocator rows there (see the plan: Phase 6's
# cross-language comparison is Linux/Docker-only; see docker/Dockerfile.bench).
set -euo pipefail

CACHEGRIND=0
if [ "${1:-}" = "--cachegrind" ]; then
    CACHEGRIND=1
    shift
fi

if [ "${BASH_VERSINFO[0]}" -lt 4 ]; then
    echo "requires bash 4+ (associative arrays) — macOS ships 3.2 by default;" >&2
    echo "install a newer bash (e.g. 'brew install bash') and re-run with it." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NATIVE_DIR="$SCRIPT_DIR/native"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results}"
# Cachegrind simulates every memory access in software (~20-50x slower than
# native execution), so its pass uses a much smaller op count by default.
OPS="${OPS:-$([ "$CACHEGRIND" = 1 ] && echo 2000 || echo 50000)}"

mkdir -p "$RESULTS_DIR"

if [ "$CACHEGRIND" = 1 ]; then
    if ! command -v valgrind >/dev/null 2>&1; then
        echo "valgrind not found — cachegrind mode is Linux-only (see docker/Dockerfile.bench)." >&2
        echo "valgrind does not support modern macOS at all, so this mode cannot run locally there." >&2
        exit 1
    fi
elif ! command -v hyperfine >/dev/null 2>&1; then
    echo "hyperfine not found — install it (e.g. 'brew install hyperfine' or apt-get install hyperfine)" >&2
    exit 1
fi

UNAME="$(uname -s)"
case "$UNAME" in
    Linux) LIBEXT=so; PRELOAD_VAR=LD_PRELOAD ;;
    Darwin) LIBEXT=dylib; PRELOAD_VAR=DYLD_INSERT_LIBRARIES ;;
    *) echo "unsupported platform: $UNAME" >&2; exit 1 ;;
esac

echo "Building native harness..."
make -C "$NATIVE_DIR" >/dev/null

CABI_LIB="$REPO_ROOT/target/release/liblohalloc_cabi.$LIBEXT"
if [ ! -f "$CABI_LIB" ]; then
    echo "Building lohalloc-cabi (release)..."
    (cd "$REPO_ROOT" && cargo build -p lohalloc-cabi --release >/dev/null)
fi

# Allocator name -> preload path (empty = system default). True symbol
# interposition (LD_PRELOAD) only works on Linux — DYLD_INSERT_LIBRARIES on
# macOS does NOT rebind malloc/free (verified: the loaded binary's malloc
# stays bound to libsystem_malloc.dylib regardless), so only the "system"
# baseline is meaningful there. jemalloc/mimalloc paths come from
# docker/Dockerfile.bench's package install.
declare -A ALLOCATORS=(
    [system]=""
)
if [ "$UNAME" = "Linux" ]; then
    ALLOCATORS[lohalloc]="$CABI_LIB"
    for candidate in /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 /usr/lib/aarch64-linux-gnu/libjemalloc.so.2; do
        [ -f "$candidate" ] && ALLOCATORS[jemalloc]="$candidate"
    done
    for candidate in /usr/lib/x86_64-linux-gnu/libmimalloc.so.2 /usr/lib/aarch64-linux-gnu/libmimalloc.so.2 /usr/local/lib/libmimalloc.so; do
        [ -f "$candidate" ] && ALLOCATORS[mimalloc]="$candidate"
    done
else
    echo "NOTE: $UNAME detected — only the 'system' baseline runs meaningfully here;" >&2
    echo "      true interposition (lohalloc/jemalloc/mimalloc via LD_PRELOAD) is Linux-only." >&2
fi

C_WORKLOADS=(slab arena buddy system adv-mixed)
CPP_WORKLOADS=(slab arena buddy system adv-mixed cpp-vector cpp-string)

run_timing() {
    local lang="$1" binary="$2" workload="$3" allocator="$4" preload="$5" mode="$6" extra_env="$7"
    local out="$RESULTS_DIR/native-${lang}-${allocator}-${workload}-${mode}.json"
    echo "==> [timing] $lang/$allocator/$workload ($mode)"
    # LD_PRELOAD/extra_env must be part of the *command string* hyperfine
    # runs, not hyperfine's own environment — `env VAR=x hyperfine ...`
    # would preload our allocator into hyperfine's own (large, Rust) process
    # too, which reproducibly hung indefinitely during its own startup
    # (verified: hyperfine spawned zero child processes for 8+ minutes).
    # hyperfine's default shell wrapper applies env assignments in the
    # command string only to that one child invocation.
    hyperfine \
        --warmup 2 \
        --min-runs 10 \
        --export-json "$out" \
        "env $extra_env $PRELOAD_VAR=$preload $binary $workload $OPS" >/dev/null
}

# Runs `binary` under `valgrind --tool=cachegrind`, parses its stderr
# summary (D1/LLd hit-miss counts), and writes a small JSON file — same
# naming convention as run_timing's --export-json, distinguished by the
# "cachegrind-" prefix, so the Milestone D aggregator can tell them apart.
# `"source":"sim"` (vs `"pmu"`) marks this as simulated, not real hardware
# counters — see the Phase 6 plan's cache-metrics section.
run_cachegrind() {
    local lang="$1" binary="$2" workload="$3" allocator="$4" preload="$5" mode="$6" extra_env="$7"
    local out="$RESULTS_DIR/cachegrind-${lang}-${allocator}-${workload}-${mode}.json"
    local cg_out
    cg_out="$(mktemp)"
    echo "==> [cachegrind] $lang/$allocator/$workload ($mode)"

    local raw
    raw="$(env $extra_env "$PRELOAD_VAR"="$preload" valgrind \
        --tool=cachegrind --cachegrind-out-file="$cg_out" \
        "$binary" "$workload" "$OPS" 2>&1 >/dev/null || true)"
    rm -f "$cg_out"

    extract() {
        # $1 = label (e.g. "D1  misses"). Every cachegrind line is prefixed
        # with valgrind's own "==<PID>==" (a number!), so we must anchor on
        # the metric's own colon and take the number *after* it — anchoring
        # on "the first digit in the line" instead (an earlier bug) grabs
        # the PID from the prefix, not the value.
        echo "$raw" | grep -m1 "$1" | sed -E 's/^.*:[[:space:]]*([0-9,]+).*/\1/' | tr -d ',' | head -1
    }

    local d_refs d1_misses lld_misses ll_misses
    d_refs="$(extract 'D  *refs:')"
    d1_misses="$(extract 'D1  *misses:')"
    lld_misses="$(extract 'LLd misses:')"
    ll_misses="$(extract '^==.*LL misses:')"
    d_refs="${d_refs:-0}"; d1_misses="${d1_misses:-0}"; lld_misses="${lld_misses:-0}"; ll_misses="${ll_misses:-0}"

    cat >"$out" <<EOF
{
  "lang": "$lang",
  "allocator": "$allocator",
  "workload": "$workload",
  "mode": "$mode",
  "ops": $OPS,
  "d_refs": $d_refs,
  "d1_misses": $d1_misses,
  "lld_misses": $lld_misses,
  "ll_misses": $ll_misses,
  "source": "sim"
}
EOF
}

run_one() {
    if [ "$CACHEGRIND" = 1 ]; then
        run_cachegrind "$@"
    else
        run_timing "$@"
    fi
}

for lang_binary in "c:$NATIVE_DIR/build/bench_main_c:${C_WORKLOADS[*]}" "cpp:$NATIVE_DIR/build/bench_main_cpp:${CPP_WORKLOADS[*]}"; do
    IFS=: read -r lang binary workloads_str <<<"$lang_binary"
    read -r -a workloads <<<"$workloads_str"
    for allocator in "${!ALLOCATORS[@]}"; do
        preload="${ALLOCATORS[$allocator]}"
        for workload in "${workloads[@]}"; do
            if [ "$allocator" = "lohalloc" ]; then
                run_one "$lang" "$binary" "$workload" "$allocator" "$preload" "training" ""
                run_one "$lang" "$binary" "$workload" "$allocator" "$preload" "inference" "LOHALLOC_FREEZE_AFTER=1000"
            else
                run_one "$lang" "$binary" "$workload" "$allocator" "$preload" "baseline" ""
            fi
        done
    done
done

echo "Results written to $RESULTS_DIR"
