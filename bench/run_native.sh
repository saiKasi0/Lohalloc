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
PERFMODE=0
RSSMODE=0
case "${1:-}" in
    --cachegrind) CACHEGRIND=1; shift ;;
    --perf)       PERFMODE=1; shift ;;
    --rss)        RSSMODE=1; shift ;;
esac

if [ "${BASH_VERSINFO[0]}" -lt 4 ]; then
    echo "requires bash 4+ (associative arrays) — macOS ships 3.2 by default;" >&2
    echo "install a newer bash (e.g. 'brew install bash') and re-run with it." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NATIVE_DIR="$SCRIPT_DIR/native"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results}"
# Where each per-invocation JSON is written. The Makefile passes RAW_DIR as
# results/<timestamp>/raw so producers write straight into the final run dir
# (the aggregator then reads it in place — no staging, no move). The default
# below is only for a bare `bash run_native.sh` with no RAW_DIR set.
RAW_DIR="${RAW_DIR:-$RESULTS_DIR/raw}"
# Cachegrind simulates every memory access in software (~20-50x slower than
# native execution), so its pass uses a much smaller op count by default.
OPS="${OPS:-$([ "$CACHEGRIND" = 1 ] && echo 2000 || echo 50000)}"

# Trained .lohalloc models live in a temp dir OUTSIDE the run dir so they are
# never mistaken for result JSON by the aggregator. One model per
# (lang, workload): module-base-normalized hashes make a model stable
# across processes *of the same binary*, not across binaries.
MODEL_DIR="$(mktemp -d)"
trap 'rm -rf "$MODEL_DIR"' EXIT

mkdir -p "$RAW_DIR"

if [ "$CACHEGRIND" = 1 ]; then
    if ! command -v valgrind >/dev/null 2>&1; then
        echo "valgrind not found — cachegrind mode is Linux-only (see docker/Dockerfile.bench)." >&2
        echo "valgrind does not support modern macOS at all, so this mode cannot run locally there." >&2
        exit 1
    fi
elif [ "$PERFMODE" = 1 ]; then
    if ! command -v perf >/dev/null 2>&1; then
        echo "perf not found — --perf mode is Linux-only (needs linux-tools + PMU access)." >&2
        echo "Run 'bash bench/probe_perf.sh' on the target host first to confirm the PMU" >&2
        echo "exposes the cache/coherence events (else use a .metal instance)." >&2
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

# Rust picks its allocator at build time (#[global_allocator] via the
# mutually-exclusive alloc-* features), not via LD_PRELOAD — so there is one
# binary per allocator. cargo reuses the same bin name across feature sets,
# hence the copy-out to distinct names. Built here only if missing
# (docker/Dockerfile.bench and `make bench-native-host` pre-build them).
RUST_ALLOCATORS=(system lohalloc jemalloc mimalloc)
for feat in "${RUST_ALLOCATORS[@]}"; do
    bin="$NATIVE_DIR/build/native_workload_$feat"
    if [ ! -x "$bin" ]; then
        echo "Building native_workload (alloc-$feat)..."
        (cd "$REPO_ROOT" && cargo build -p lohalloc-bench --bin native_workload --release --features "alloc-$feat" >/dev/null)
        mkdir -p "$NATIVE_DIR/build"
        cp "$REPO_ROOT/target/release/native_workload" "$bin"
    fi
done

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

# Thread counts swept for the mt-* workloads below; 1 is the scaling
# denominator (S(t) = time(t1)/time(t)), 4/8 are where lock contention on
# the still-serialized Buddy/Arena paths (see crates/lohalloc-alloc's
# module docs) should show up first. Override with MT_THREADS="1 2 4 8" etc.
MT_THREADS="${MT_THREADS:-1 4 8}"
# "interfere" (J5-B2) is the allocator-interference row: fixed application
# compute + occasional alloc, so its lohalloc/jemalloc ratio measures how
# much the allocator slows down a real MT app (expect ~1.0; a big loss =
# false-sharing / cache-pollution finding, the thing cachegrind cannot see).
MT_WORKLOAD_KINDS=(slab mixed xfree interfere)
MT_WORKLOADS=()
for n in $MT_THREADS; do
    for kind in "${MT_WORKLOAD_KINDS[@]}"; do
        MT_WORKLOADS+=("mt-${kind}-t${n}")
    done
done

# perf mode targets the MT rows only — real-hardware PMU beats cachegrind's
# single-thread sim exactly where cross-core coherence / false sharing lives;
# the single-thread rows stay on cachegrind. Default WORKLOADS to the MT set
# (feeds the existing WORKLOADS override below) unless the caller narrowed it.
if [ "$PERFMODE" = 1 ] && [ -z "${WORKLOADS:-}" ]; then
    WORKLOADS="${MT_WORKLOADS[*]}"
fi

C_WORKLOADS=(slab arena buddy system adv-mixed request-loop json-tree kv-store "${MT_WORKLOADS[@]}")
CPP_WORKLOADS=(slab arena buddy system adv-mixed cpp-vector cpp-string request-loop json-tree kv-store "${MT_WORKLOADS[@]}")

# ---- Targeted-subset overrides (diagnosis / sweep drivers) -------------------
# WORKLOADS="buddy adv-mixed"   replaces every language's workload list
#                               (cpp-only names like cpp-string need LANGS=cpp
#                               too — the C binary rejects them).
# LANGS="c cpp"                 which of "c cpp rust" run (default: all).
# ONLY_ALLOCATORS="lohalloc"    restrict the allocator matrix (both the
#                               preload map and the rust build-time list).
# These exist so one-off investigations (e.g. a cachegrind calibration A/B on
# two workloads) and the tune_sweep native driver don't pay for the full
# ~45-minute matrix.
LANGS="${LANGS:-c cpp rust}"
if [ -n "${WORKLOADS:-}" ]; then
    read -r -a C_WORKLOADS <<<"$WORKLOADS"
    read -r -a CPP_WORKLOADS <<<"$WORKLOADS"
fi
if [ -n "${ONLY_ALLOCATORS:-}" ]; then
    for key in "${!ALLOCATORS[@]}"; do
        case " $ONLY_ALLOCATORS " in
            *" $key "*) ;;
            *) unset "ALLOCATORS[$key]" ;;
        esac
    done
    read -r -a RUST_ALLOCATORS <<<"$ONLY_ALLOCATORS"
fi

lang_enabled() {
    case " $LANGS " in
        *" $1 "*) return 0 ;;
        *) return 1 ;;
    esac
}

# For an "mt-<kind>-tN" workload name on Linux, returns "taskset -c 0-(N-1) "
# (trailing space, ready to prepend to a command string) so the workload's N
# threads compete for exactly N CPUs instead of whatever the Docker
# container happens to expose — otherwise a threads=1 vs threads=8 A/B on an
# under-provisioned host measures host scheduling noise, not the
# allocator's lock contention. Empty string for every non-mt workload or on
# macOS (no taskset).
taskset_prefix_for() {
    local workload="$1"
    if [ "$UNAME" = "Linux" ] && [[ "$workload" =~ -t([0-9]+)$ ]]; then
        local n="${BASH_REMATCH[1]}"
        echo "taskset -c 0-$((n - 1)) "
    fi
}

run_timing() {
    local lang="$1" binary="$2" workload="$3" allocator="$4" preload="$5" mode="$6" extra_env="$7"
    local out="$RAW_DIR/native-${lang}-${allocator}-${workload}-${mode}.json"
    echo "==> [timing] $lang/$allocator/$workload ($mode)"
    local pin
    pin="$(taskset_prefix_for "$workload")"
    # Opt-in cold-cache isolation (BENCH_PREPARE=drop-caches): drop the
    # kernel page cache before EVERY timed run so no run inherits warm
    # file-cache state (the model file, the binary, shared libs) from the
    # previous allocator's runs. This deliberately measures COLD launches —
    # the default (no prepare) stays hot steady-state, which is what the
    # regression gate and all historical numbers mean; never mix cold-mode
    # and hot-mode rows in one RUN_DIR. Linux-only (needs a privileged
    # container for /proc/sys/vm/drop_caches — `make bench-native
    # BENCH_PREPARE=drop-caches` adds --privileged); on macOS `purge`
    # requires sudo, so the mode is skipped with a warning rather than
    # prompting mid-benchmark.
    local prepare_args=()
    if [ "${BENCH_PREPARE:-}" = "drop-caches" ]; then
        if [ "$UNAME" = "Linux" ] && [ -w /proc/sys/vm/drop_caches ]; then
            prepare_args=(--prepare "sync; echo 3 > /proc/sys/vm/drop_caches")
        elif [ "$UNAME" = "Linux" ]; then
            echo "ERROR: BENCH_PREPARE=drop-caches but /proc/sys/vm/drop_caches is not writable." >&2
            echo "       Run the container with --privileged (make bench-native BENCH_PREPARE=drop-caches does this)." >&2
            exit 1
        else
            echo "WARN: BENCH_PREPARE=drop-caches is Linux-only (macOS 'purge' needs sudo) — running without prepare." >&2
        fi
    fi
    # LD_PRELOAD/extra_env must be part of the *command string* hyperfine
    # runs, not hyperfine's own environment — `env VAR=x hyperfine ...`
    # would preload our allocator into hyperfine's own (large, Rust) process
    # too, which reproducibly hung indefinitely during its own startup
    # (verified: hyperfine spawned zero child processes for 8+ minutes).
    # hyperfine's default shell wrapper applies env assignments in the
    # command string only to that one child invocation. `taskset` (when
    # present) is just a plain prefix in that same string, same reasoning.
    # --warmup 8 (not hyperfine's more typical 2): a freshly started
    # container's first several mmap-heavy process launches pay a VM-level
    # memory-subsystem warm-up cost specific to a given allocation-size
    # pattern — reproduced directly: the buddy workload's first ~5 process
    # launches after `docker run` took 1.6-1.8s each before settling to a
    # steady ~115-135ms from the 6th launch on (same binary, same model,
    # no code involved — a Docker Desktop VM artifact, not an allocator
    # regression). 2 warmup runs left that cold-start inside the 10 measured
    # runs and inflated the reported mean by ~10x on that one row.
    # ${arr[@]+...} guard: bash 4.0-4.3 treat an empty array as unset under
    # `set -u` and would abort on plain "${prepare_args[@]}".
    hyperfine \
        --warmup 8 \
        --min-runs 10 \
        ${prepare_args[@]+"${prepare_args[@]}"} \
        --export-json "$out" \
        "env $extra_env $PRELOAD_VAR=$preload ${pin}$binary $workload $OPS" >/dev/null
    # ops=1 timing-calibration companion (the same `-cal` pattern the
    # cachegrind/perf passes have used since Step 7.5, now for wall time):
    # measures each mode's FIXED process cost — exec + dynamic linking +
    # model load/PHT build + eager pool mmaps — so the aggregator can
    # publish a startup-immune per-op wall view (`mean_ns_net`). Short rows
    # (W-SYSTEM runs ops/20 iterations in a ~1 ms process) are otherwise
    # dominated by this fixed cost. Fewer warmups than the main pass: the
    # ops=1 process never triggers the buddy-pattern VM warm-up described
    # above. TIMING_CAL=0 skips it.
    if [ "${TIMING_CAL:-1}" = 1 ]; then
        hyperfine \
            --warmup 3 \
            --min-runs 10 \
            ${prepare_args[@]+"${prepare_args[@]}"} \
            --export-json "${out%.json}-cal.json" \
            "env $extra_env $PRELOAD_VAR=$preload ${pin}$binary $workload 1" >/dev/null
    fi
}

# Runs `binary` under `valgrind --tool=cachegrind`, parses its stderr
# summary (D1/LLd hit-miss counts), and writes a small JSON file — same
# naming convention as run_timing's --export-json, distinguished by the
# "cachegrind-" prefix, so the Milestone D aggregator can tell them apart.
# `"source":"sim"` (vs `"pmu"`) marks this as simulated, not real hardware
# counters — see the Phase 6 plan's cache-metrics section.
run_cachegrind() {
    local lang="$1" binary="$2" workload="$3" allocator="$4" preload="$5" mode="$6" extra_env="$7"
    cachegrind_pass "$@" "$OPS" ""
    # ops=1 calibration pass: identical binary, env, preload and (for
    # inference) model file — so its counts are startup + ~1 op. cachegrind
    # counts the WHOLE process, and at the measured OPS=2000 a mode's fixed
    # startup cost (inference: model-file read + PHT build + registry setup)
    # is a material share of the per-op division. The aggregator subtracts
    # this row's counts from the main row's before dividing ->
    # d1_misses_per_op_net, the startup-immune view.
    cachegrind_pass "$@" 1 "-cal"
}

cachegrind_pass() {
    local lang="$1" binary="$2" workload="$3" allocator="$4" preload="$5" mode="$6" extra_env="$7" ops="$8" suffix="$9"
    local out="$RAW_DIR/cachegrind-${lang}-${allocator}-${workload}-${mode}${suffix}.json"
    local cg_out
    cg_out="$(mktemp)"
    echo "==> [cachegrind] $lang/$allocator/$workload ($mode${suffix}, ops=$ops)"

    local pin
    pin="$(taskset_prefix_for "$workload")"
    local raw
    raw="$(env $extra_env "$PRELOAD_VAR"="$preload" ${pin}valgrind \
        --tool=cachegrind --cachegrind-out-file="$cg_out" \
        "$binary" "$workload" "$ops" 2>&1 >/dev/null || true)"
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

    local calibration=false
    [ -n "$suffix" ] && calibration=true
    cat >"$out" <<EOF
{
  "lang": "$lang",
  "allocator": "$allocator",
  "workload": "$workload",
  "mode": "$mode",
  "ops": $ops,
  "calibration": $calibration,
  "d_refs": $d_refs,
  "d1_misses": $d1_misses,
  "lld_misses": $lld_misses,
  "ll_misses": $ll_misses,
  "source": "sim"
}
EOF
}

# Runs `binary` under `perf stat` reading REAL hardware PMU counters (vs
# cachegrind's software simulation). Writes a `perf-`-prefixed JSON with
# `"source":"pmu"` — the field run_cachegrind's `"source":"sim"` was always
# paired against. Same ops=1 `-cal` calibration companion as cachegrind so the
# aggregator can subtract each mode's fixed startup cost before dividing.
run_perf() {
    local lang="$1" binary="$2" workload="$3" allocator="$4" preload="$5" mode="$6" extra_env="$7"
    perf_pass "$@" "$OPS" ""
    perf_pass "$@" 1 "-cal"
}

perf_pass() {
    local lang="$1" binary="$2" workload="$3" allocator="$4" preload="$5" mode="$6" extra_env="$7" ops="$8" suffix="$9"
    local out="$RAW_DIR/perf-${lang}-${allocator}-${workload}-${mode}${suffix}.json"
    echo "==> [perf] $lang/$allocator/$workload ($mode${suffix}, ops=$ops)"

    local pin
    pin="$(taskset_prefix_for "$workload")"

    # perf stat counts the whole target process INCLUDING every worker thread
    # it spawns — exactly the MT total we want. -x, => machine-readable CSV
    # (value,unit,event,run-time,pct,...); an unsupported/unpermitted event
    # prints "<not supported>"/"<not counted>" in the value field, normalized
    # to 0 below. The event set is the arch-portable generic-name set that
    # bench/probe_perf.sh validates.
    # page-faults (software event, always countable) prices the kernel-side
    # touch cost cachegrind's user-space sim cannot see — the arena-row
    # signature is near-parity drefs with a large wall gap, and this counter
    # attributes that gap to fresh-page faults. instructions/cycles give the
    # retired-work view where the PMU exposes them (Nitro virtualized
    # instances often don't; they normalize to 0 exactly like LLC).
    local events="cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,page-faults,instructions,cycles"
    local raw
    raw="$(env $extra_env "$PRELOAD_VAR"="$preload" ${pin}perf stat -x, -e "$events" \
        -- "$binary" "$workload" "$ops" 2>&1 >/dev/null || true)"

    # Pull one event's integer count from the CSV (field 3 == event name;
    # field 1 == value). Non-numeric ("<not supported>", "<not counted>",
    # empty) -> 0 so the row still parses and the miss is visible as a zero.
    pev() {
        local v
        v="$(printf '%s\n' "$raw" | awk -F, -v e="$1" '$3==e {print $1; exit}' | tr -d '[:space:]')"
        case "$v" in
        '' | *[!0-9]*) echo 0 ;;
        *) echo "$v" ;;
        esac
    }

    local cache_references cache_misses l1d_loads l1d_load_misses llc_loads llc_load_misses
    local page_faults instructions cycles
    cache_references="$(pev cache-references)"
    cache_misses="$(pev cache-misses)"
    l1d_loads="$(pev L1-dcache-loads)"
    l1d_load_misses="$(pev L1-dcache-load-misses)"
    llc_loads="$(pev LLC-loads)"
    llc_load_misses="$(pev LLC-load-misses)"
    page_faults="$(pev page-faults)"
    instructions="$(pev instructions)"
    cycles="$(pev cycles)"

    local calibration=false
    [ -n "$suffix" ] && calibration=true
    cat >"$out" <<EOF
{
  "lang": "$lang",
  "allocator": "$allocator",
  "workload": "$workload",
  "mode": "$mode",
  "ops": $ops,
  "calibration": $calibration,
  "cache_references": $cache_references,
  "cache_misses": $cache_misses,
  "l1d_loads": $l1d_loads,
  "l1d_load_misses": $l1d_load_misses,
  "llc_loads": $llc_loads,
  "llc_load_misses": $llc_load_misses,
  "page_faults": $page_faults,
  "instructions": $instructions,
  "cycles": $cycles,
  "source": "pmu"
}
EOF
}

# Peak-RSS pass: run the binary ONCE with LOHALLOC_BENCH_RSS=1 (the binary
# prints `RSS_KIB <n>` to stderr at exit, from getrusage) and record it — the
# memory-footprint axis of the real-workload benchmarks. Not a timing pass, so
# a single run is enough (peak RSS is deterministic-ish; the workloads are
# leak-free by construction, so it's a clean fragmentation signal). Same
# command-string env discipline as the timing/perf passes.
run_rss() {
    local lang="$1" binary="$2" workload="$3" allocator="$4" preload="$5" mode="$6" extra_env="$7"
    local out="$RAW_DIR/rss-${lang}-${allocator}-${workload}-${mode}.json"
    echo "==> [rss] $lang/$allocator/$workload ($mode)"
    local pin
    pin="$(taskset_prefix_for "$workload")"
    local rss
    rss="$(env $extra_env LOHALLOC_BENCH_RSS=1 "$PRELOAD_VAR"="$preload" \
        ${pin}"$binary" "$workload" "$OPS" 2>&1 >/dev/null |
        awk '/^RSS_KIB/{print $2; exit}')"
    case "$rss" in '' | *[!0-9]*) rss=0 ;; esac
    cat >"$out" <<EOF
{
  "lang": "$lang",
  "allocator": "$allocator",
  "workload": "$workload",
  "mode": "$mode",
  "rss_kib": $rss,
  "source": "rss"
}
EOF
}

run_one() {
    if [ "$CACHEGRIND" = 1 ]; then
        run_cachegrind "$@"
    elif [ "$PERFMODE" = 1 ]; then
        run_perf "$@"
    elif [ "$RSSMODE" = 1 ]; then
        run_rss "$@"
    else
        run_timing "$@"
    fi
}

# Ops for the untimed training run. Decoupled from the measured OPS so the
# freeze threshold is always reached: the sparsest workload (`system`) does
# only ops/20 allocations, so at the cachegrind OPS (2000) it would allocate
# just 100 times and never cross FREEZE_AFTER=1000 — the export would fail
# and abort the whole matrix. Training is untimed, so a fixed large count
# here costs nothing and guarantees every workload freezes (system does
# TRAIN_OPS/20 = 2500 >> 1000).
TRAIN_OPS="${TRAIN_OPS:-50000}"
TRAIN_FREEZE_AFTER="${TRAIN_FREEZE_AFTER:-1000}"

# The lohalloc pre/post-training triple, shared by every language:
#   1. "training" (timed)     — the whole run learns online.
#   2. train+export (UNTIMED) — freeze after TRAIN_FREEZE_AFTER allocs, export
#      the model. Earlier versions instead ran "inference" as
#      LOHALLOC_FREEZE_AFTER inside every timed invocation — but hyperfine
#      spawns a fresh process per run, so that *retrained from scratch every
#      time* and never actually measured post-training behavior.
#   3. "inference" (timed)    — every run loads the pre-trained model at
#      startup: pure frozen-path routing, zero training ops. Requires
#      ASLR-stable (module-relative) stack hashes; verify with
#      lohalloc_pht_misses() ~ 0 (LOHALLOC_DEBUG=1 prints it).
# TUNE_FILE (optional): a flat key=value tune config (see
# crates/lohalloc-alloc/src/tune.rs) applied to every leg of the triple via
# LOHALLOC_TUNE — this is how tune_sweep --native ablates the *production*
# LD_PRELOAD path (lohalloc-cabi loads the full config, reward shaping
# included, inside its bootstrap guard; global-allocator Rust builds honor
# only the freeze knobs). Inference gets it too: harmless there (a loaded
# model skips training), and keeping the env identical across legs means
# the calibration/timing deltas measure the config, not the environment.
run_lohalloc_triple() {
    local lang="$1" binary="$2" workload="$3" preload="$4"
    local tune_env=""
    [ -n "${TUNE_FILE:-}" ] && tune_env="LOHALLOC_TUNE=$TUNE_FILE"
    # J5 bisect knobs: forwarded (when set) into the SAME command-string env
    # as LOHALLOC_TUNE, so they reach all three legs — timed training, the
    # untimed train+export, and timed inference. Deliberately including the
    # train/export leg: stripes change training-time latencies → rewards →
    # verdicts, so a bisect cell must train under the config it is benched
    # under (the exported model itself stores only (hash, sc, backend)).
    # Explicit named forwarding (not a generic passthrough) keeps the env
    # injection surface auditable; the values are then self-documented in
    # every raw JSON's stored hyperfine command.
    [ -n "${LOHALLOC_STRIPES:-}" ] && tune_env="$tune_env${tune_env:+ }LOHALLOC_STRIPES=$LOHALLOC_STRIPES"
    [ -n "${LOHALLOC_DEMOTE_FRACTION:-}" ] && tune_env="$tune_env${tune_env:+ }LOHALLOC_DEMOTE_FRACTION=$LOHALLOC_DEMOTE_FRACTION"
    # Task B per-arm spike-winsorization ablation knob (LOHALLOC_CLAMP_PERCENTILE,
    # replaces the retired LOHALLOC_LATENCY_CLAMP_NS): a reward-shaping key, so
    # it must ride the train + train/export legs to change the learned model.
    # Honored by BOTH the C/C++ cabi rows (full config in the bootstrap guard)
    # and the Rust rows (native_workload calls tune::load_from_env at startup —
    # see its main()). `0` disables winsorization; range 0..=100.
    [ -n "${LOHALLOC_CLAMP_PERCENTILE:-}" ] && tune_env="$tune_env${tune_env:+ }LOHALLOC_CLAMP_PERCENTILE=$LOHALLOC_CLAMP_PERCENTILE"
    # Phase 1.6 default-on training-headerless off-switch
    # (LOHALLOC_TRAIN_HEADERLESS=0): a reward-fidelity knob read on the
    # instance's first alloc, so it rides the train + train/export legs (it
    # changes the learned model, hence inference routing). NOT a tune key — a
    # getenv off-switch — but forwarded the same way so the A/B control cell can
    # force the header-based path. Unset = default ON (headerless training).
    [ -n "${LOHALLOC_TRAIN_HEADERLESS:-}" ] && tune_env="$tune_env${tune_env:+ }LOHALLOC_TRAIN_HEADERLESS=$LOHALLOC_TRAIN_HEADERLESS"
    # Part 2 bilateral-reward recovery opt-in (LOHALLOC_REWARD_TRACK=1): a
    # reward-fidelity knob read on the headerless training path, so it rides the
    # train + train/export legs (it changes the learned model). NOT a tune key —
    # a getenv opt-in — forwarded for diagnostics. Unset = default OFF (the
    # certified rt A/B, results/20260711T183928: the ring re-broke the mt-mixed
    # rows the size-aware context register had fixed).
    [ -n "${LOHALLOC_REWARD_TRACK:-}" ] && tune_env="$tune_env${tune_env:+ }LOHALLOC_REWARD_TRACK=$LOHALLOC_REWARD_TRACK"
    # Roadmap-D deep-context escalation gate (LOHALLOC_ESCALATE_VARIANCE, a
    # tune key): reward/model-shaping, so it rides the train + train/export
    # legs. `0` disables deep context entirely (training + freeze).
    [ -n "${LOHALLOC_ESCALATE_VARIANCE:-}" ] && tune_env="$tune_env${tune_env:+ }LOHALLOC_ESCALATE_VARIANCE=$LOHALLOC_ESCALATE_VARIANCE"
    # Servability-aware training opt-in (LOHALLOC_SERVABLE_TRAINING=1):
    # reward/model-shaping (training must not recommend arms inference cannot
    # serve — the headerless-Buddy free-rider fix), rides the train +
    # train/export legs. Unset = default OFF (certified 231127: masking costs
    # the suite more than the free-rider trap does).
    [ -n "${LOHALLOC_SERVABLE_TRAINING:-}" ] && tune_env="$tune_env${tune_env:+ }LOHALLOC_SERVABLE_TRAINING=$LOHALLOC_SERVABLE_TRAINING"
    # Phase-1 slab-diet item A off-switch (LOHALLOC_SC_SHORTCUT=0): the
    # unanimous-size-class inference shortcut, read ONCE at model-publish time
    # (inference leg only — training has a null frozen table, so the gate never
    # fires there and forwarding it to the train legs is a self-documenting
    # no-op). Outcome-preserving, so this A/B measures pure per-op speed. Unset
    # = default ON.
    [ -n "${LOHALLOC_SC_SHORTCUT:-}" ] && tune_env="$tune_env${tune_env:+ }LOHALLOC_SC_SHORTCUT=$LOHALLOC_SC_SHORTCUT"
    # mt-xfree fix off-switch (LOHALLOC_SLAB_OWNER_FREE=0): return cross-thread
    # slab frees to the owning stripe (default on) vs the freeing thread's stripe
    # (=0). A runtime free-path behavior read on both training and inference (it
    # bites whenever a headerless-slab magazine flush spans stripes), so forward
    # it to every leg. Unset = default ON.
    [ -n "${LOHALLOC_SLAB_OWNER_FREE:-}" ] && tune_env="$tune_env${tune_env:+ }LOHALLOC_SLAB_OWNER_FREE=$LOHALLOC_SLAB_OWNER_FREE"
    # J6 fast-lane off-switch (LOHALLOC_FAST_LANE=0): the tcache-shaped
    # frozen alloc/free lanes (default on; armed at model publish, so it
    # affects the inference legs and any post-freeze training tail).
    [ -n "${LOHALLOC_FAST_LANE:-}" ] && tune_env="$tune_env${tune_env:+ }LOHALLOC_FAST_LANE=$LOHALLOC_FAST_LANE"
    # J6-D2 legacy-exclusion switch (LOHALLOC_PIN_EXCLUDE_LEGACY=1): restore
    # the pre-D2 candidate-based distilled exclusion. Freeze-time behavior,
    # so it must reach the train+export leg (it does — tune_env reaches all
    # three legs of the triple).
    [ -n "${LOHALLOC_PIN_EXCLUDE_LEGACY:-}" ] && tune_env="$tune_env${tune_env:+ }LOHALLOC_PIN_EXCLUDE_LEGACY=$LOHALLOC_PIN_EXCLUDE_LEGACY"
    run_one "$lang" "$binary" "$workload" "lohalloc" "$preload" "training" "$tune_env"
    local model="$MODEL_DIR/model-${lang}-${workload}.lohalloc"
    echo "==> [train+export] $lang/$workload -> $model (train ops=$TRAIN_OPS)"
    env $tune_env LOHALLOC_FREEZE_AFTER="$TRAIN_FREEZE_AFTER" LOHALLOC_EXPORT_MODEL="$model" \
        "$PRELOAD_VAR=$preload" "$binary" "$workload" "$TRAIN_OPS" >/dev/null
    if [ ! -f "$model" ]; then
        echo "FATAL: model export failed for $lang/$workload" >&2
        exit 1
    fi
    run_one "$lang" "$binary" "$workload" "lohalloc" "$preload" "inference" "LOHALLOC_MODEL=$model${tune_env:+ $tune_env}"
}

for lang_binary in "c:$NATIVE_DIR/build/bench_main_c:${C_WORKLOADS[*]}" "cpp:$NATIVE_DIR/build/bench_main_cpp:${CPP_WORKLOADS[*]}"; do
    IFS=: read -r lang binary workloads_str <<<"$lang_binary"
    lang_enabled "$lang" || continue
    read -r -a workloads <<<"$workloads_str"
    for allocator in "${!ALLOCATORS[@]}"; do
        preload="${ALLOCATORS[$allocator]}"
        for workload in "${workloads[@]}"; do
            if [ "$allocator" = "lohalloc" ]; then
                run_lohalloc_triple "$lang" "$binary" "$workload" "$preload"
            else
                run_one "$lang" "$binary" "$workload" "$allocator" "$preload" "baseline" ""
            fi
        done
    done
done

# ---- Rust: allocator chosen at build time, no preload -----------------------
# One binary per allocator (native_workload_<alloc>), so these rows run
# meaningfully on macOS too — nothing here depends on LD_PRELOAD.
RUST_WORKLOADS=(slab arena buddy system adv-mixed request-loop json-tree kv-store "${MT_WORKLOADS[@]}")
if [ -n "${WORKLOADS:-}" ]; then
    read -r -a RUST_WORKLOADS <<<"$WORKLOADS"
fi
lang_enabled rust && for allocator in "${RUST_ALLOCATORS[@]}"; do
    binary="$NATIVE_DIR/build/native_workload_$allocator"
    if [ ! -x "$binary" ]; then
        echo "NOTE: $binary missing — skipping rust/$allocator rows" >&2
        continue
    fi
    for workload in "${RUST_WORKLOADS[@]}"; do
        if [ "$allocator" = "lohalloc" ]; then
            run_lohalloc_triple "rust" "$binary" "$workload" ""
        else
            run_one "rust" "$binary" "$workload" "$allocator" "" "baseline" ""
        fi
    done
done

echo "Raw results written to $RAW_DIR (run 'make bench-report RUN_DIR=$(dirname "$RAW_DIR")' to build the report + graphs)"
