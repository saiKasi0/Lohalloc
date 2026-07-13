//! Phase 6 results aggregator.
//!
//! Reads the raw per-invocation JSON that the benchmark producers wrote into
//! `<run-dir>/raw/` (the producers write there *directly* — the `RUN_DIR`
//! make variable hands every step the same timestamped directory, so there
//! is no staging area and nothing is moved), normalizes them into one table,
//! and writes the report + graphs back into the same run directory:
//!
//! ```text
//! results/<timestamp>/          <- RUN_DIR, chosen once by the Makefile
//!   raw/                        <- per-invocation JSON, written here by producers
//!   graphs/                     <- PNGs from bench/graphs/plot_report.py (best-effort)
//!   bench-report.json
//!   bench-report.md
//! ```
//!
//! Invoke as `aggregate --run-dir <dir>` (`--results-dir` is a backward-compat
//! alias). The three raw input kinds are:
//!
//! - `native-<lang>-<allocator>-<workload>-<mode>.json` — hyperfine
//!   `--export-json` output from `bench/run_native.sh`'s timing pass.
//! - `cachegrind-<lang>-<allocator>-<workload>-<mode>.json` — this
//!   workspace's own schema, from `bench/run_native.sh --cachegrind`.
//! - `rust_*.json` — `src/bin/latency_profile.rs`'s `LatencyReport` schema
//!   (always `lang: "rust"`, `allocator` implied to be `lohalloc`).
//!
//! Exits non-zero if any regression-gate row fails: Lohalloc-inference mean
//! latency must not exceed jemalloc's mean by more than 5%, for whichever
//! (lang, workload) pairs have both present in the results directory. This
//! is intentionally the *only* hard gate — everything else in the report is
//! informational, since not every environment has every baseline allocator
//! or a PMU (cachegrind counters are always present when run at all; real
//! PMU counters are best-effort and rarely available in CI/cloud VMs).
//!
//! # Miss *rates* vs misses *per op* — read both, trust per-op
//!
//! Cachegrind rows carry two views of the same counts: `d1_miss_rate`
//! (misses / total data refs) and `d1_misses_per_op` (misses / workload
//! ops). The rate's denominator is *everything the process did*, so any
//! change that removes work shrinks the denominator and inflates the rate
//! without a single extra miss — e.g. freezing removes the per-op
//! state-Mutex and bandit-update bookkeeping (~2.8× fewer data refs per op
//! on slab). Real example from `results/20260704T181553`: rust/mt-slab-t4
//! *inference* has FEWER absolute misses than training (8,987 vs 12,015)
//! yet a HIGHER rate (1.75% vs 1.02%). Per-op misses are
//! denominator-immune; use them for any training-vs-inference or
//! before-vs-after layout comparison, and keep the rate only for
//! same-mode cross-allocator comparisons where per-op work is comparable.
//!
//! ```sh
//! cargo run -p lohalloc-bench --bin aggregate --release -- --results-dir results
//! ```

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// One normalized row, regardless of which source format it came from.
#[derive(Debug, Clone, Serialize)]
struct Row {
    source: &'static str, // "native-timing" | "cachegrind" | "perf" | "rust-latency"
    lang: String,
    allocator: String,
    workload: String,
    mode: String,
    /// Mean wall-clock time per invocation, nanoseconds (native-timing only).
    mean_ns: Option<f64>,
    /// Run-to-run standard deviation of the wall-clock time, nanoseconds
    /// (native-timing only; from hyperfine's own `stddev`). This is the
    /// noise floor for any comparison between two rows of the same
    /// (lang, workload) — a delta smaller than the two rows' combined
    /// stddev is not signal. Docker-on-macOS VM runs routinely show
    /// stddevs large enough to flip the sign of small deltas.
    stddev_ns: Option<f64>,
    /// Workload throughput in million ops / second (native-timing only;
    /// ops recovered from the hyperfine command tail — see
    /// [`ops_from_command`]). Includes process startup in the denominator,
    /// which is fine for cross-allocator comparison: every allocator runs
    /// the identical binary at the identical op count.
    throughput_mops: Option<f64>,
    /// Mean wall of the ops=1 timing-calibration companion, nanoseconds
    /// (native-timing only; requires a `native-…-cal.json` from
    /// `run_native.sh`'s TIMING_CAL pass). This is the mode's FIXED process
    /// cost: exec + linking + model load + eager pool mmaps.
    startup_ns: Option<f64>,
    /// `mean_ns` with the mode's fixed startup subtracted (clamped at 0) —
    /// the startup-immune wall view. Informational: the regression gate
    /// keeps judging the raw `mean_ns` (its historical meaning).
    mean_ns_net: Option<f64>,
    /// Page faults per workload op, startup-subtracted (perf only; the
    /// kernel-side touch cost neither cachegrind nor drefs can see).
    page_faults_per_op_net: Option<f64>,
    /// Retired instructions per workload op, startup-subtracted (perf
    /// only; 0/absent where the virtualized PMU hides the counter).
    instructions_per_op_net: Option<f64>,
    /// D1 (L1 data cache) miss rate, 0.0-1.0 (cachegrind only).
    d1_miss_rate: Option<f64>,
    /// Last-level cache miss rate, 0.0-1.0 (cachegrind only).
    ll_miss_rate: Option<f64>,
    /// D1 misses per workload op (cachegrind only). Denominator-immune —
    /// see the module doc's "Miss rates vs misses per op" section.
    d1_misses_per_op: Option<f64>,
    /// LL misses per workload op (cachegrind only).
    ll_misses_per_op: Option<f64>,
    /// D1 misses per op with the mode's fixed startup cost subtracted
    /// (cachegrind only; requires an ops=1 `-cal` calibration row from
    /// `run_native.sh --cachegrind`). Gross per-op still charges inference
    /// its one-time model-load/PHT-build startup against only OPS ops —
    /// net is the startup-immune view; clamped at 0.
    d1_misses_per_op_net: Option<f64>,
    /// LL misses per op, startup-subtracted (see `d1_misses_per_op_net`).
    ll_misses_per_op_net: Option<f64>,
    /// p50 alloc latency, nanoseconds (rust-latency only).
    alloc_p50_ns: Option<f64>,
    /// Mean alloc latency, nanoseconds (rust-latency only).
    alloc_mean_ns: Option<f64>,
    /// p99 alloc latency, nanoseconds (rust-latency only).
    alloc_p99_ns: Option<f64>,
    /// Mean dealloc latency, nanoseconds (rust-latency only).
    dealloc_mean_ns: Option<f64>,
    /// Per-op alloc throughput in million allocs / second
    /// (= 1e3 / alloc_mean_ns; rust-latency only). Unlike
    /// `throughput_mops` this excludes process startup — it is the pure
    /// hot-path rate.
    rust_throughput_mops: Option<f64>,
    /// Measured `Instant` tick floor on the machine that produced this
    /// rust-latency row (ns). Apple Silicon: ~42ns; x86 TSC: ~1-30ns.
    clock_tick_ns: Option<u64>,
    /// True when this rust-latency row's percentiles sit under ~3× the
    /// tick floor — those values are quantization buckets, not latencies,
    /// and are annotated (`*`) rather than compared in the report.
    quantized: Option<bool>,
    /// Peak RSS in KiB (rss pass only — from `run_native.sh --rss`, the
    /// process high-water resident set). The memory-footprint axis of the
    /// real-workload benchmarks; `None` for every other source.
    rss_kib: Option<u64>,
}

/// A `Row` with every metric `None` — each loader branch fills in only its
/// own source's fields.
fn empty_row(
    source: &'static str,
    lang: String,
    allocator: String,
    workload: String,
    mode: String,
) -> Row {
    Row {
        source,
        lang,
        allocator,
        workload,
        mode,
        mean_ns: None,
        stddev_ns: None,
        throughput_mops: None,
        startup_ns: None,
        mean_ns_net: None,
        page_faults_per_op_net: None,
        instructions_per_op_net: None,
        d1_miss_rate: None,
        ll_miss_rate: None,
        d1_misses_per_op: None,
        ll_misses_per_op: None,
        d1_misses_per_op_net: None,
        ll_misses_per_op_net: None,
        alloc_p50_ns: None,
        alloc_mean_ns: None,
        alloc_p99_ns: None,
        dealloc_mean_ns: None,
        rust_throughput_mops: None,
        clock_tick_ns: None,
        quantized: None,
        rss_kib: None,
    }
}

#[derive(Deserialize)]
struct HyperfineExport {
    results: Vec<HyperfineResult>,
}

#[derive(Deserialize)]
struct HyperfineResult {
    mean: f64, // seconds
    /// Run-to-run standard deviation in seconds. hyperfine has always
    /// exported it, but keep it optional so hand-assembled raw files parse.
    #[serde(default)]
    stddev: Option<f64>,
    /// The benchmarked command line — hyperfine always exports it. Every
    /// command in this harness ends `... <workload> <OPS>`, so the ops
    /// count is recoverable from the trailing token (no sidecar files, no
    /// filename-format changes).
    #[serde(default)]
    command: Option<String>,
    /// Every individual run's wall time in seconds — hyperfine always
    /// exports the full sample array. This is what powers the real
    /// hypothesis tests (Mann-Whitney U): mean±stddev alone can't say
    /// whether a 6% delta on a noisy Docker VM is signal.
    #[serde(default)]
    times: Vec<f64>,
}

/// Recover the workload op count from a hyperfine command string's trailing
/// token. Returns `None` (never panics) for commands that don't end in a
/// number — throughput is then simply omitted for that row.
fn ops_from_command(command: &str) -> Option<u64> {
    command.split_whitespace().next_back()?.parse().ok()
}

#[derive(Deserialize)]
struct CachegrindResult {
    lang: String,
    allocator: String,
    workload: String,
    mode: String,
    /// Optional so pre-Step-7.5 raw files (which always wrote it anyway)
    /// and hand-assembled directories both keep parsing.
    #[serde(default)]
    ops: Option<u64>,
    /// `true` on the ops=1 startup-calibration companion row (`-cal`
    /// files). Calibration rows never become report rows themselves — they
    /// are subtracted from their main row to produce the `_net` per-op
    /// metrics. Default `false` so pre-calibration raw files keep parsing.
    #[serde(default)]
    calibration: bool,
    d_refs: u64,
    d1_misses: u64,
    ll_misses: u64,
}

/// Real-hardware PMU counters from `bench/run_native.sh --perf` (`perf-`
/// prefixed raw files, `"source":"pmu"`). Unlike `CachegrindResult`'s
/// deterministic simulation these are noisy real-silicon counts, MT rows
/// only, and never gated — they exist to attribute an MT regression to actual
/// L1/LLC misses (cross-core coherence traffic cachegrind's single-thread sim
/// cannot see). Uses the L1-dcache-load / LLC-load counters as the L1/last-
/// level miss view; `cache_references`/`cache_misses` are also in the raw file
/// (aggregate all-levels) but not surfaced here.
#[derive(Deserialize)]
struct PerfResult {
    lang: String,
    allocator: String,
    workload: String,
    mode: String,
    #[serde(default)]
    ops: Option<u64>,
    /// ops=1 startup-calibration companion (`-cal` files) — subtracted from
    /// its main row for the `_net` per-op view, exactly like cachegrind.
    #[serde(default)]
    calibration: bool,
    l1d_loads: u64,
    l1d_load_misses: u64,
    llc_loads: u64,
    llc_load_misses: u64,
    /// Software page-fault count (kernel-side touch cost). `default` so
    /// raw files from before the counter was collected keep parsing.
    #[serde(default)]
    page_faults: u64,
    /// Retired instructions (0 where the virtualized PMU hides it).
    #[serde(default)]
    instructions: u64,
}

/// Peak-RSS pass row (`rss-*.json` from `run_native.sh --rss`) — the
/// memory-footprint axis of the real-workload benchmarks.
#[derive(Deserialize)]
struct RssResult {
    lang: String,
    allocator: String,
    workload: String,
    mode: String,
    rss_kib: u64,
}

#[derive(Deserialize)]
struct RustLatencyResult {
    workload: String,
    mode: String,
    alloc_p99_ns: f64,
    /// Optional for forward/backward compatibility with older raw files;
    /// `latency_profile` has always written them.
    #[serde(default)]
    alloc_p50_ns: Option<f64>,
    #[serde(default)]
    alloc_mean_ns: Option<f64>,
    #[serde(default)]
    dealloc_mean_ns: Option<f64>,
    /// Tick floor + quantization verdict stamped by `latency_profile`
    /// (see `lohalloc_bench::clockinfo`). Optional: pre-Ladder-4 raw
    /// files don't carry them.
    #[serde(default)]
    clock_tick_ns: Option<u64>,
    #[serde(default)]
    quantized: Option<bool>,
}

/// Every allocator the native pipeline can produce. Filename parsing
/// validates against this closed set so a future dashed allocator name (or
/// a typo in `run_native.sh`) fails loudly instead of silently leaking into
/// the workload segment (the Ladder 4 label-integrity audit found the
/// convention sound but unenforced).
const KNOWN_ALLOCATORS: [&str; 4] = ["lohalloc", "jemalloc", "mimalloc", "system"];
/// Every run mode the native pipeline can produce (same enforcement).
const KNOWN_MODES: [&str; 3] = ["training", "inference", "baseline"];

/// Parse `native-<lang>-<allocator>-<workload>-<mode>.json` (workload may
/// itself contain hyphens, e.g. "adv-mixed" — allocator/lang/mode never do,
/// so we split from both ends and take whatever's left as the workload).
/// The allocator and mode segments are validated against the known sets:
/// an unknown token returns `None` (the caller logs and skips the file) —
/// mis-attribution is never silent. Note this also protects workloads: a
/// hypothetical workload named `foo-baseline` would need the *mode* slot to
/// hold a non-mode token to mis-parse, which this rejects.
fn parse_native_filename(stem: &str) -> Option<(String, String, String, String)> {
    let rest = stem.strip_prefix("native-")?;
    let parts: Vec<&str> = rest.split('-').collect();
    if parts.len() < 4 {
        return None;
    }
    let lang = parts[0].to_string();
    let allocator = parts[1].to_string();
    let mode = parts[parts.len() - 1].to_string();
    let workload = parts[2..parts.len() - 1].join("-");
    if !KNOWN_ALLOCATORS.contains(&allocator.as_str()) {
        eprintln!(
            "native result filename {stem}: unknown allocator token '{allocator}' \
             (known: {KNOWN_ALLOCATORS:?}) — refusing to guess"
        );
        return None;
    }
    if !KNOWN_MODES.contains(&mode.as_str()) {
        eprintln!(
            "native result filename {stem}: unknown mode token '{mode}' \
             (known: {KNOWN_MODES:?}) — refusing to guess"
        );
        return None;
    }
    Some((lang, allocator, workload, mode))
}

/// Per-run wall-time sample arrays for native-timing rows, keyed by
/// (lang, allocator, workload, mode). Kept OUT of `Row` deliberately:
/// `Row` serializes into bench-report.json (which the graph scripts read),
/// and ~1700 samples × ~130 rows would bloat it ~50× for data only the
/// in-process hypothesis tests need.
type SampleMap = BTreeMap<(String, String, String, String), Vec<f64>>;

/// ops=1 perf calibration counts keyed by (lang, allocator, workload, mode):
/// `(l1d_load_misses, llc_load_misses, page_faults, instructions)`, subtracted
/// from the main row for the startup-immune `_net` view.
type PerfCalMap = BTreeMap<(String, String, String, String), (u64, u64, u64, u64)>;

fn load_rows(dir: &Path) -> (Vec<Row>, SampleMap) {
    let mut samples: SampleMap = BTreeMap::new();
    let mut rows = Vec::new();
    // Cachegrind rows are two-phase: main rows and their ops=1 calibration
    // companions are collected here, then paired by (lang, allocator,
    // workload, mode) after the directory scan to fill the `_net` fields.
    let mut cg_main: Vec<CachegrindResult> = Vec::new();
    let mut cg_cal: BTreeMap<(String, String, String, String), (u64, u64)> = BTreeMap::new();
    // perf (real-PMU) rows are two-phase exactly like cachegrind: main rows and
    // their ops=1 `-cal` companions, paired after the scan for the `_net` view.
    // Cal tuple: (l1d_load_misses, llc_load_misses, page_faults, instructions).
    let mut perf_main: Vec<PerfResult> = Vec::new();
    let mut perf_cal: PerfCalMap = BTreeMap::new();
    // Timing rows are two-phase the same way: `native-…-cal.json` (ops=1
    // hyperfine companion) carries each mode's fixed startup wall, paired
    // after the scan into `startup_ns`/`mean_ns_net`.
    let mut timing_cal: BTreeMap<(String, String, String, String), f64> = BTreeMap::new();
    let Ok(entries) = fs::read_dir(dir) else {
        eprintln!("results directory {dir:?} not found or unreadable");
        return (rows, samples);
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };

        if stem.starts_with("native-") {
            // `-cal` companions reuse the main filename with a suffix; strip
            // it BEFORE the segment parse (the bare stem's trailing segment
            // would otherwise land in the mode slot and be rejected).
            let (core_stem, is_cal) = match stem.strip_suffix("-cal") {
                Some(core) => (core, true),
                None => (stem, false),
            };
            let Some((lang, allocator, workload, mode)) = parse_native_filename(core_stem) else {
                eprintln!("skipping unparseable native result filename: {stem}");
                continue;
            };
            if is_cal {
                match serde_json::from_str::<HyperfineExport>(&text) {
                    Ok(export) => {
                        if let Some(r) = export.results.first() {
                            timing_cal.insert((lang, allocator, workload, mode), r.mean * 1e9);
                        }
                    }
                    Err(e) => eprintln!("failed to parse {path:?} as hyperfine export: {e}"),
                }
                continue;
            }
            match serde_json::from_str::<HyperfineExport>(&text) {
                Ok(export) => {
                    if let Some(r) = export.results.first() {
                        let mut row = empty_row(
                            "native-timing",
                            lang.clone(),
                            allocator.clone(),
                            workload.clone(),
                            mode.clone(),
                        );
                        row.mean_ns = Some(r.mean * 1e9);
                        row.stddev_ns = r.stddev.map(|s| s * 1e9);
                        row.throughput_mops = r
                            .command
                            .as_deref()
                            .and_then(ops_from_command)
                            .filter(|&ops| ops > 0 && r.mean > 0.0)
                            .map(|ops| ops as f64 / r.mean / 1e6);
                        if !r.times.is_empty() {
                            samples.insert((lang, allocator, workload, mode), r.times.clone());
                        }
                        rows.push(row);
                    }
                }
                Err(e) => eprintln!("failed to parse {path:?} as hyperfine export: {e}"),
            }
        } else if stem.starts_with("cachegrind-") {
            match serde_json::from_str::<CachegrindResult>(&text) {
                Ok(cg) if cg.calibration => {
                    cg_cal.insert(
                        (cg.lang, cg.allocator, cg.workload, cg.mode),
                        (cg.d1_misses, cg.ll_misses),
                    );
                }
                Ok(cg) => cg_main.push(cg),
                Err(e) => eprintln!("failed to parse {path:?} as cachegrind result: {e}"),
            }
        } else if stem.starts_with("perf-") {
            match serde_json::from_str::<PerfResult>(&text) {
                Ok(p) if p.calibration => {
                    perf_cal.insert(
                        (p.lang, p.allocator, p.workload, p.mode),
                        (
                            p.l1d_load_misses,
                            p.llc_load_misses,
                            p.page_faults,
                            p.instructions,
                        ),
                    );
                }
                Ok(p) => perf_main.push(p),
                Err(e) => eprintln!("failed to parse {path:?} as perf result: {e}"),
            }
        } else if stem.starts_with("rss-") {
            match serde_json::from_str::<RssResult>(&text) {
                Ok(r) => {
                    let mut row = empty_row("rss", r.lang, r.allocator, r.workload, r.mode);
                    row.rss_kib = Some(r.rss_kib);
                    rows.push(row);
                }
                Err(e) => eprintln!("failed to parse {path:?} as rss result: {e}"),
            }
        } else if stem.starts_with("rust_") || stem.starts_with("rust-") {
            match serde_json::from_str::<RustLatencyResult>(&text) {
                Ok(r) => {
                    let mut row = empty_row(
                        "rust-latency",
                        "rust".to_string(),
                        "lohalloc".to_string(),
                        r.workload,
                        r.mode,
                    );
                    row.alloc_p99_ns = Some(r.alloc_p99_ns);
                    row.alloc_p50_ns = r.alloc_p50_ns;
                    row.alloc_mean_ns = r.alloc_mean_ns;
                    row.dealloc_mean_ns = r.dealloc_mean_ns;
                    row.rust_throughput_mops =
                        r.alloc_mean_ns.filter(|&m| m > 0.0).map(|m| 1e3 / m);
                    row.clock_tick_ns = r.clock_tick_ns;
                    row.quantized = r.quantized;
                    rows.push(row);
                }
                Err(e) => eprintln!("failed to parse {path:?} as rust latency result: {e}"),
            }
        }
        // Unrecognized JSON files (e.g. criterion's own estimates.json, if
        // pointed at that directory) are silently skipped.
    }

    for cg in cg_main {
        let cal = cg_cal
            .get(&(
                cg.lang.clone(),
                cg.allocator.clone(),
                cg.workload.clone(),
                cg.mode.clone(),
            ))
            .copied();
        rows.push(cachegrind_row(cg, cal));
    }
    for p in perf_main {
        let cal = perf_cal
            .get(&(
                p.lang.clone(),
                p.allocator.clone(),
                p.workload.clone(),
                p.mode.clone(),
            ))
            .copied();
        rows.push(perf_row(p, cal));
    }
    // Pair each timing row with its ops=1 companion for the startup-immune
    // wall view. A missing companion (TIMING_CAL=0, or pre-calibration raw
    // dirs) just leaves the new fields None — nothing downstream requires
    // them.
    for row in rows.iter_mut().filter(|r| r.source == "native-timing") {
        let key = (
            row.lang.clone(),
            row.allocator.clone(),
            row.workload.clone(),
            row.mode.clone(),
        );
        if let (Some(mean), Some(&startup)) = (row.mean_ns, timing_cal.get(&key)) {
            row.startup_ns = Some(startup);
            row.mean_ns_net = Some((mean - startup).max(0.0));
        }
    }
    (rows, samples)
}

/// Significance threshold for every hypothesis test in the report. 0.01
/// (not 0.05): with ~40 gate rows per run, α=0.05 would produce ~2 false
/// verdicts per run by chance alone.
const ALPHA: f64 = 0.01;

/// Two-sided Mann-Whitney U test (normal approximation, tie-corrected,
/// continuity-corrected): p-value for "these two sample sets come from the
/// same distribution". Chosen over Welch's t because hyperfine timing
/// samples are right-skewed (occasional scheduler stalls) and MW-U is
/// rank-based, so outliers can't dominate. Self-contained on purpose — no
/// stats crate for one test.
///
/// Returns `None` when either side has < 8 samples (the normal
/// approximation is unreliable below that; hyperfine's `--min-runs 10`
/// means real rows always qualify). All-identical samples return
/// `Some(1.0)` (no evidence of difference, by construction).
fn mann_whitney_p(a: &[f64], b: &[f64]) -> Option<f64> {
    let (n1, n2) = (a.len(), b.len());
    if n1 < 8 || n2 < 8 {
        return None;
    }
    // Rank the pooled samples, averaging ranks across ties.
    let mut pooled: Vec<(f64, usize)> = a
        .iter()
        .map(|&v| (v, 0usize))
        .chain(b.iter().map(|&v| (v, 1usize)))
        .collect();
    pooled.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));

    let n = n1 + n2;
    let mut rank_sum_a = 0.0f64;
    let mut tie_term = 0.0f64; // sum over tie groups of (t^3 - t)
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && pooled[j + 1].0 == pooled[i].0 {
            j += 1;
        }
        let group = (j - i + 1) as f64;
        // Average rank of positions i..=j (1-based ranks).
        let avg_rank = (i + 1 + j + 1) as f64 / 2.0;
        for entry in &pooled[i..=j] {
            if entry.1 == 0 {
                rank_sum_a += avg_rank;
            }
        }
        if group > 1.0 {
            tie_term += group * group * group - group;
        }
        i = j + 1;
    }

    let (n1f, n2f, nf) = (n1 as f64, n2 as f64, n as f64);
    let u_a = rank_sum_a - n1f * (n1f + 1.0) / 2.0;
    let mu = n1f * n2f / 2.0;
    let var = n1f * n2f / 12.0 * ((nf + 1.0) - tie_term / (nf * (nf - 1.0)));
    if var <= 0.0 {
        return Some(1.0); // every value tied — no evidence of difference
    }
    // Continuity correction: shrink |U - mu| by 0.5 (clamped at 0).
    let z = ((u_a - mu).abs() - 0.5).max(0.0) / var.sqrt();
    Some(2.0 * (1.0 - std_normal_cdf(z)))
}

/// Render a p-value for the report. The normal approximation underflows to
/// exactly 0 for |z| ≳ 6 — print that as an inequality, not a fake "0".
fn fmt_p(p: Option<f64>) -> String {
    match p {
        None => "n/a".to_string(),
        Some(p) if p < 1e-15 => "<1e-15".to_string(),
        Some(p) => format!("{p:.2e}"),
    }
}

/// Standard normal CDF via the Abramowitz–Stegun 7.1.26 erf approximation
/// (|error| < 1.5e-7 — far below anything a p-threshold at 0.01 can see).
fn std_normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * (x.abs() / std::f64::consts::SQRT_2));
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let erf = 1.0 - poly * (-(x * x) / 2.0).exp();
    if x >= 0.0 {
        0.5 * (1.0 + erf)
    } else {
        0.5 * (1.0 - erf)
    }
}

/// Turn one main cachegrind result (plus its optional ops=1 calibration
/// counts `(d1_misses, ll_misses)`) into a report row. Net per-op metrics
/// subtract the calibration counts — the mode's fixed startup cost — before
/// dividing, clamped at 0 (a cal run nominally does 1 op more than nothing,
/// and cachegrind's simulation is deterministic enough that main < cal only
/// happens on effectively-zero workloads).
fn cachegrind_row(cg: CachegrindResult, cal: Option<(u64, u64)>) -> Row {
    let mut row = empty_row("cachegrind", cg.lang, cg.allocator, cg.workload, cg.mode);
    if cg.d_refs > 0 {
        row.d1_miss_rate = Some(cg.d1_misses as f64 / cg.d_refs as f64);
        row.ll_miss_rate = Some(cg.ll_misses as f64 / cg.d_refs as f64);
    }
    if let Some(ops) = cg.ops.filter(|&o| o > 0) {
        row.d1_misses_per_op = Some(cg.d1_misses as f64 / ops as f64);
        row.ll_misses_per_op = Some(cg.ll_misses as f64 / ops as f64);
        if let Some((cal_d1, cal_ll)) = cal {
            row.d1_misses_per_op_net =
                Some(cg.d1_misses.saturating_sub(cal_d1) as f64 / ops as f64);
            row.ll_misses_per_op_net =
                Some(cg.ll_misses.saturating_sub(cal_ll) as f64 / ops as f64);
        }
    }
    row
}

/// Turn one main perf result (plus its optional ops=1 calibration L1d/LLC
/// miss counts) into a report row. Reuses the shared `d1_*`/`ll_*` metric
/// fields — L1-dcache maps to the D1 (L1 data) view and LLC to the last-level
/// view — so the perf table renders the same shape as cachegrind's; the
/// `source == "perf"` tag is what distinguishes them. Rates use each level's
/// own load count as the denominator (L1d misses / L1d loads; LLC misses /
/// LLC loads). Net per-op subtracts the mode's ops=1 startup cost, clamped at
/// 0, identical to `cachegrind_row`.
fn perf_row(p: PerfResult, cal: Option<(u64, u64, u64, u64)>) -> Row {
    let mut row = empty_row("perf", p.lang, p.allocator, p.workload, p.mode);
    if p.l1d_loads > 0 {
        row.d1_miss_rate = Some(p.l1d_load_misses as f64 / p.l1d_loads as f64);
    }
    if p.llc_loads > 0 {
        row.ll_miss_rate = Some(p.llc_load_misses as f64 / p.llc_loads as f64);
    }
    if let Some(ops) = p.ops.filter(|&o| o > 0) {
        row.d1_misses_per_op = Some(p.l1d_load_misses as f64 / ops as f64);
        row.ll_misses_per_op = Some(p.llc_load_misses as f64 / ops as f64);
        if let Some((cal_l1d, cal_llc, cal_pf, cal_insn)) = cal {
            row.d1_misses_per_op_net =
                Some(p.l1d_load_misses.saturating_sub(cal_l1d) as f64 / ops as f64);
            row.ll_misses_per_op_net =
                Some(p.llc_load_misses.saturating_sub(cal_llc) as f64 / ops as f64);
            row.page_faults_per_op_net =
                Some(p.page_faults.saturating_sub(cal_pf) as f64 / ops as f64);
            // instructions=0 means the virtualized PMU hid the counter — leave
            // the net field None rather than publish a spurious 0/op.
            if p.instructions > 0 {
                row.instructions_per_op_net =
                    Some(p.instructions.saturating_sub(cal_insn) as f64 / ops as f64);
            }
        }
    }
    row
}

/// The regression gate: for every (lang, workload) where both a
/// lohalloc/inference row and a jemalloc/baseline row exist in
/// `native-timing`, lohalloc's mean must not exceed jemalloc's by >5% —
/// AND (when per-run samples are available) the loss must be statistically
/// significant (Mann-Whitney, p < [`ALPHA`]). The significance requirement
/// stops Docker-VM noise from flipping a borderline row's verdict between
/// runs; rows without samples (pre-Ladder-4 raw files) keep the pure
/// ratio semantics.
fn evaluate_gate(rows: &[Row], samples: &SampleMap) -> Vec<(String, bool, String)> {
    let mut lohalloc_inf: BTreeMap<(String, String), f64> = BTreeMap::new();
    let mut jemalloc_base: BTreeMap<(String, String), (f64, String)> = BTreeMap::new();

    for row in rows {
        if row.source != "native-timing" {
            continue;
        }
        let Some(mean) = row.mean_ns else { continue };
        let key = (row.lang.clone(), row.workload.clone());
        if row.allocator == "lohalloc" && row.mode == "inference" {
            lohalloc_inf.insert(key, mean);
        } else if row.allocator == "jemalloc" {
            jemalloc_base.insert(key, (mean, row.mode.clone()));
        }
    }

    // W-SYSTEM is excluded — mmap-bound for every allocator, not a
    // meaningful comparison point.
    let mut results = Vec::new();
    for (key, lohalloc_mean) in &lohalloc_inf {
        if key.1 == "system" {
            continue;
        }
        if let Some((jemalloc_mean, jemalloc_mode)) = jemalloc_base.get(key) {
            let ratio = lohalloc_mean / jemalloc_mean;
            let p = match (
                samples.get(&(
                    key.0.clone(),
                    "lohalloc".to_string(),
                    key.1.clone(),
                    "inference".to_string(),
                )),
                samples.get(&(
                    key.0.clone(),
                    "jemalloc".to_string(),
                    key.1.clone(),
                    jemalloc_mode.clone(),
                )),
            ) {
                (Some(a), Some(b)) => mann_whitney_p(a, b),
                _ => None,
            };
            // Fail only on a loss that is real: over threshold AND
            // significant (or unmeasurable — old runs without samples keep
            // the strict ratio behavior rather than silently passing).
            let over = ratio > 1.05;
            let significant = p.map_or(true, |p| p < ALPHA);
            let pass = !(over && significant);
            let p_str = fmt_p(p);
            let noise_note = if over && !significant {
                " [over threshold but not significant — treated as noise]"
            } else {
                ""
            };
            results.push((
                format!("{}/{}: lohalloc-inference vs jemalloc", key.0, key.1),
                pass,
                format!(
                    "lohalloc={lohalloc_mean:.0}ns jemalloc={jemalloc_mean:.0}ns ratio={ratio:.3} (max 1.05) p={p_str}{noise_note}"
                ),
            ));
        }
    }
    results
}

/// One H-B comparison row: lohalloc-inference vs one baseline allocator on
/// one (lang, workload), with the Mann-Whitney verdict when samples exist.
struct HbComparison {
    lang: String,
    workload: String,
    baseline: String,
    ratio: f64,
    p_value: Option<f64>,
}

impl HbComparison {
    /// "wins" / "loses" / "no significant difference" / "ratio-only".
    /// A significant verdict needs p < ALPHA; direction comes from the
    /// ratio. Without samples only the ratio is reported (no verdict).
    fn verdict(&self) -> &'static str {
        match self.p_value {
            Some(p) if p < ALPHA && self.ratio < 1.0 => "wins",
            Some(p) if p < ALPHA => "loses",
            Some(_) => "no significant difference",
            None => "ratio-only",
        }
    }
}

/// H-B (trained lohalloc beats other allocators): lohalloc-inference vs
/// EACH of jemalloc / mimalloc / system per (lang, workload) — the gate
/// only tests jemalloc; this is the full-hypothesis view.
fn hb_comparisons(rows: &[Row], samples: &SampleMap) -> Vec<HbComparison> {
    let mut lohalloc_inf: BTreeMap<(String, String), f64> = BTreeMap::new();
    let mut baselines: BTreeMap<(String, String, String), (f64, String)> = BTreeMap::new();
    for row in rows {
        if row.source != "native-timing" {
            continue;
        }
        let Some(mean) = row.mean_ns else { continue };
        if row.allocator == "lohalloc" && row.mode == "inference" {
            lohalloc_inf.insert((row.lang.clone(), row.workload.clone()), mean);
        } else if row.allocator != "lohalloc" {
            baselines.insert(
                (
                    row.lang.clone(),
                    row.workload.clone(),
                    row.allocator.clone(),
                ),
                (mean, row.mode.clone()),
            );
        }
    }
    let mut out = Vec::new();
    for ((lang, workload, baseline), (base_mean, base_mode)) in &baselines {
        if workload == "system" {
            continue; // mmap-bound for everyone; same exclusion as the gate
        }
        let Some(lo_mean) = lohalloc_inf.get(&(lang.clone(), workload.clone())) else {
            continue;
        };
        let p = match (
            samples.get(&(
                lang.clone(),
                "lohalloc".to_string(),
                workload.clone(),
                "inference".to_string(),
            )),
            samples.get(&(
                lang.clone(),
                baseline.clone(),
                workload.clone(),
                base_mode.clone(),
            )),
        ) {
            (Some(a), Some(b)) => mann_whitney_p(a, b),
            _ => None,
        };
        out.push(HbComparison {
            lang: lang.clone(),
            workload: workload.clone(),
            baseline: baseline.clone(),
            ratio: lo_mean / base_mean,
            p_value: p,
        });
    }
    out
}

/// One (lang, workload) lohalloc training↔inference comparison from the
/// native-timing rows — the H-A hypothesis, per row. When per-run samples
/// exist, `p_value` carries the Mann-Whitney verdict (the real test);
/// `within_noise`/`combined_stddev_ns` remain as the fallback heuristic
/// for pre-Ladder-4 runs whose raw files lack `times`.
struct TrainVsInference {
    lang: String,
    workload: String,
    train_mean_ns: f64,
    inf_mean_ns: f64,
    /// Mann-Whitney two-sided p for training vs inference samples.
    /// `None` when either side lacks a sample array.
    p_value: Option<f64>,
    /// `None` when either row predates stddev capture — then no noise
    /// verdict is possible and the delta is reported bare.
    combined_stddev_ns: Option<f64>,
    within_noise: bool,
}

impl TrainVsInference {
    /// H-A verdict for this row: `Some(true)` = inference significantly
    /// faster, `Some(false)` = significantly slower, `None` =
    /// inconclusive (not significant, or no samples to test).
    fn ha_verdict(&self) -> Option<bool> {
        match self.p_value {
            Some(p) if p < ALPHA => Some(self.inf_mean_ns < self.train_mean_ns),
            _ => None,
        }
    }
}

fn train_vs_inference(rows: &[Row], samples: &SampleMap) -> Vec<TrainVsInference> {
    let mut train: BTreeMap<(String, String), (f64, Option<f64>)> = BTreeMap::new();
    let mut inf: BTreeMap<(String, String), (f64, Option<f64>)> = BTreeMap::new();
    for row in rows {
        if row.source != "native-timing" || row.allocator != "lohalloc" {
            continue;
        }
        let Some(mean) = row.mean_ns else { continue };
        let key = (row.lang.clone(), row.workload.clone());
        match row.mode.as_str() {
            "training" => {
                train.insert(key, (mean, row.stddev_ns));
            }
            "inference" => {
                inf.insert(key, (mean, row.stddev_ns));
            }
            _ => {}
        }
    }

    let mut out = Vec::new();
    for (key, &(train_mean_ns, train_sd)) in &train {
        let Some(&(inf_mean_ns, inf_sd)) = inf.get(key) else {
            continue;
        };
        let p_value = match (
            samples.get(&(
                key.0.clone(),
                "lohalloc".to_string(),
                key.1.clone(),
                "training".to_string(),
            )),
            samples.get(&(
                key.0.clone(),
                "lohalloc".to_string(),
                key.1.clone(),
                "inference".to_string(),
            )),
        ) {
            (Some(a), Some(b)) => mann_whitney_p(a, b),
            _ => None,
        };
        let combined_stddev_ns = match (train_sd, inf_sd) {
            (Some(a), Some(b)) => Some(a + b),
            _ => None,
        };
        let within_noise =
            combined_stddev_ns.is_some_and(|sd| (train_mean_ns - inf_mean_ns).abs() <= sd);
        out.push(TrainVsInference {
            lang: key.0.clone(),
            workload: key.1.clone(),
            train_mean_ns,
            inf_mean_ns,
            p_value,
            combined_stddev_ns,
            within_noise,
        });
    }
    out
}

/// The `## Hypothesis verdicts` roll-up: the two claims this whole pipeline
/// exists to test, answered per-row-with-statistics and summarized. H-A =
/// inference beats training (same binary, same workload); H-B = trained
/// lohalloc beats each baseline allocator. "significant" = Mann-Whitney
/// p < [`ALPHA`] over hyperfine's per-run samples.
fn write_hypothesis_verdicts(out: &mut String, tvi: &[TrainVsInference], hb: &[HbComparison]) {
    out.push_str("## Hypothesis verdicts\n\n");
    out.push_str(&format!(
        "Statistical test: two-sided Mann-Whitney U over hyperfine's per-run \
         samples, significance at p < {ALPHA}. Rows without samples \
         (pre-Ladder-4 raw files) are counted as inconclusive.\n\n"
    ));

    // H-A roll-up.
    let ha_holds: Vec<&TrainVsInference> = tvi
        .iter()
        .filter(|c| c.ha_verdict() == Some(true))
        .collect();
    let ha_fails: Vec<&TrainVsInference> = tvi
        .iter()
        .filter(|c| c.ha_verdict() == Some(false))
        .collect();
    let ha_inconclusive = tvi.len() - ha_holds.len() - ha_fails.len();
    if tvi.is_empty() {
        out.push_str("**H-A (inference faster than training):** no paired rows in this run.\n\n");
    } else {
        out.push_str(&format!(
            "**H-A (inference faster than training):** holds significantly in \
             {}/{} rows, inconclusive in {}, fails in {}.\n",
            ha_holds.len(),
            tvi.len(),
            ha_inconclusive,
            ha_fails.len(),
        ));
        for c in &ha_fails {
            out.push_str(&format!(
                "- H-A FAIL: {}/{} — inference {:.1}ms vs training {:.1}ms (p={})\n",
                c.lang,
                c.workload,
                c.inf_mean_ns / 1e6,
                c.train_mean_ns / 1e6,
                fmt_p(c.p_value),
            ));
        }
        out.push('\n');
    }

    // H-B roll-up, per baseline.
    if hb.is_empty() {
        out.push_str(
            "**H-B (trained lohalloc beats baselines):** no comparable rows in this run.\n\n",
        );
    } else {
        for baseline in ["jemalloc", "mimalloc", "system"] {
            let rows_for: Vec<&HbComparison> =
                hb.iter().filter(|c| c.baseline == baseline).collect();
            if rows_for.is_empty() {
                continue;
            }
            let wins = rows_for.iter().filter(|c| c.verdict() == "wins").count();
            let losses = rows_for.iter().filter(|c| c.verdict() == "loses").count();
            let rest = rows_for.len() - wins - losses;
            out.push_str(&format!(
                "**H-B vs {baseline}:** lohalloc-inference wins significantly in \
                 {wins}/{} rows, loses in {losses}, inconclusive/ratio-only in {rest}.\n",
                rows_for.len(),
            ));
        }
        out.push('\n');
        out.push_str("Per-row H-B detail:\n\n");
        out.push_str("| lang | workload | baseline | lohalloc/baseline ratio | p | verdict |\n|---|---|---|---|---|---|\n");
        for c in hb {
            out.push_str(&format!(
                "| {} | {} | {} | {:.3} | {} | {} |\n",
                c.lang,
                c.workload,
                c.baseline,
                c.ratio,
                fmt_p(c.p_value),
                c.verdict(),
            ));
        }
        out.push('\n');
    }
}

fn write_markdown(
    rows: &[Row],
    gate: &[(String, bool, String)],
    samples: &SampleMap,
    path: &Path,
) -> std::io::Result<()> {
    let mut out = String::new();
    out.push_str("# Phase 6 Bench Report\n\n");

    let tvi = train_vs_inference(rows, samples);
    let hb = hb_comparisons(rows, samples);
    write_hypothesis_verdicts(&mut out, &tvi, &hb);

    out.push_str("## Regression gate\n\n");
    if gate.is_empty() {
        out.push_str("_No (lang, workload) pair had both a lohalloc-inference and a jemalloc baseline result — gate is informational-only for this run._\n\n");
    } else {
        for (name, pass, detail) in gate {
            let mark = if *pass { "PASS" } else { "FAIL" };
            out.push_str(&format!("- **{mark}** {name} — {detail}\n"));
        }
        out.push('\n');
    }

    out.push_str("## Native timing (hyperfine)\n\n");
    out.push_str(
        "| lang | allocator | workload | mode | mean ± stddev (ns) | throughput (Mops/s) |\n|---|---|---|---|---|---|\n",
    );
    for row in rows.iter().filter(|r| r.source == "native-timing") {
        let mean = row.mean_ns.unwrap_or(0.0);
        let mean_col = match row.stddev_ns {
            Some(sd) => format!("{mean:.0} ± {sd:.0}"),
            None => format!("{mean:.0}"),
        };
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} |\n",
            row.lang,
            row.allocator,
            row.workload,
            row.mode,
            mean_col,
            row.throughput_mops
                .map_or_else(|| "-".to_string(), |t| format!("{t:.2}")),
        ));
    }

    if !tvi.is_empty() {
        out.push_str("\n### lohalloc training vs inference (native)\n\n");
        out.push_str(
            "Verdicts come from a two-sided Mann-Whitney U over the per-run \
             samples when available (`p < 0.01` = significant); rows without \
             samples fall back to the combined-stddev heuristic (`~ within \
             noise`). Only significant `SLOWER` rows are worth \
             investigating.\n\n",
        );
        out.push_str(
            "| lang | workload | training (ns) | inference (ns) | inf/train | verdict |\n|---|---|---|---|---|---|\n",
        );
        for c in &tvi {
            let ratio = c.inf_mean_ns / c.train_mean_ns;
            let verdict = match c.p_value {
                Some(p) if p < ALPHA && ratio > 1.0 => format!("SLOWER (p={})", fmt_p(Some(p))),
                Some(p) if p < ALPHA => format!("faster (p={})", fmt_p(Some(p))),
                Some(p) => format!("~ no significant difference (p={p:.2})"),
                // Pre-Ladder-4 raw files: no samples — stddev heuristic.
                None if c.within_noise => "~ within noise".to_string(),
                None => {
                    let suffix = match c.combined_stddev_ns {
                        Some(sd) if sd > 0.0 => format!(
                            " (delta {:.1}x combined stddev)",
                            (c.inf_mean_ns - c.train_mean_ns).abs() / sd
                        ),
                        _ => " (no stddev — unverified)".to_string(),
                    };
                    if ratio > 1.0 {
                        format!("SLOWER{suffix}")
                    } else {
                        format!("faster{suffix}")
                    }
                }
            };
            out.push_str(&format!(
                "| {} | {} | {:.0} | {:.0} | {ratio:.3} | {verdict} |\n",
                c.lang, c.workload, c.train_mean_ns, c.inf_mean_ns,
            ));
        }
    }

    // Peak RSS (memory footprint) — the second axis of the real-workload
    // benchmarks. For each (lang, workload): lohalloc-inference peak RSS vs
    // each baseline allocator, ratio <1 = lohalloc uses less memory (the
    // self-resetting-arena win the timing axis can't show).
    {
        let mut rss: BTreeMap<(String, String, String), u64> = BTreeMap::new();
        for row in rows.iter().filter(|r| r.source == "rss") {
            let Some(kib) = row.rss_kib else { continue };
            let alloc = if row.allocator == "lohalloc" {
                if row.mode == "inference" {
                    "lohalloc"
                } else {
                    continue; // training/other lohalloc rows: skip
                }
            } else {
                row.allocator.as_str()
            };
            rss.insert(
                (row.lang.clone(), row.workload.clone(), alloc.to_string()),
                kib,
            );
        }
        if !rss.is_empty() {
            out.push_str("\n## Peak RSS (memory footprint, KiB — lower is better)\n\n");
            out.push_str(
                "Ratio = lohalloc-inference / competitor; **<1 means lohalloc uses less \
                 memory** (the self-resetting-arena advantage the timing axis can't show).\n\n",
            );
            out.push_str(
                "| lang | workload | lohalloc | system | jemalloc | mimalloc | vs-sys | vs-je | vs-mi |\n\
                 |---|---|---|---|---|---|---|---|---|\n",
            );
            let mut seen: std::collections::BTreeSet<(String, String)> =
                std::collections::BTreeSet::new();
            for (l, w, _) in rss.keys() {
                seen.insert((l.clone(), w.clone()));
            }
            for (l, w) in seen {
                let get = |a: &str| rss.get(&(l.clone(), w.clone(), a.to_string())).copied();
                let lo = get("lohalloc");
                let cell = |v: Option<u64>| v.map_or_else(|| "-".to_string(), |x| x.to_string());
                let ratio = |b: Option<u64>| match (lo, b) {
                    (Some(a), Some(c)) if c > 0 => format!("{:.2}", a as f64 / c as f64),
                    _ => "-".to_string(),
                };
                out.push_str(&format!(
                    "| {l} | {w} | {} | {} | {} | {} | {} | {} | {} |\n",
                    cell(lo),
                    cell(get("system")),
                    cell(get("jemalloc")),
                    cell(get("mimalloc")),
                    ratio(get("system")),
                    ratio(get("jemalloc")),
                    ratio(get("mimalloc")),
                ));
            }
        }
    }

    out.push_str("\n## Cache metrics (cachegrind, simulated)\n\n");
    out.push_str(
        "Rates dilute when a mode does more per-op bookkeeping (training) — \
         per-op misses are the denominator-immune view; see the module doc. \
         `net` columns additionally subtract the mode's fixed startup cost \
         (measured by an ops=1 calibration run) before dividing — gross \
         per-op charges inference its one-time model-load/PHT-build against \
         only OPS ops, which inflates it at small op counts.\n\n",
    );
    out.push_str("| lang | allocator | workload | mode | D1 miss rate | LL miss rate | D1 misses/op | LL misses/op | D1/op net | LL/op net |\n|---|---|---|---|---|---|---|---|---|---|\n");
    let fmt2 = |v: Option<f64>| v.map_or_else(|| "-".to_string(), |v| format!("{v:.2}"));
    for row in rows.iter().filter(|r| r.source == "cachegrind") {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {:.4} | {:.4} | {} | {} | {} | {} |\n",
            row.lang,
            row.allocator,
            row.workload,
            row.mode,
            row.d1_miss_rate.unwrap_or(0.0),
            row.ll_miss_rate.unwrap_or(0.0),
            fmt2(row.d1_misses_per_op),
            fmt2(row.ll_misses_per_op),
            fmt2(row.d1_misses_per_op_net),
            fmt2(row.ll_misses_per_op_net),
        ));
    }

    if rows.iter().any(|r| r.source == "perf") {
        out.push_str("\n## MT PMU metrics (perf, real hardware)\n\n");
        out.push_str(
            "Real hardware PMU counters (vs cachegrind's single-thread \
             simulation) — the only view that sees cross-core coherence / \
             false sharing, the mechanism behind MT regressions. MT rows only. \
             `net` subtracts an ops=1 startup-calibration run. INFORMATIONAL: \
             perf counts are run-to-run noisy and never gate — the headline is \
             `LLC/op net` (last-level misses per op, coherence-dominated).\n\n",
        );
        out.push_str("| lang | allocator | workload | mode | L1d miss rate | LLC miss rate | L1d miss/op | LLC miss/op | L1d/op net | LLC/op net |\n|---|---|---|---|---|---|---|---|---|---|\n");
        for row in rows.iter().filter(|r| r.source == "perf") {
            out.push_str(&format!(
                "| {} | {} | {} | {} | {:.4} | {:.4} | {} | {} | {} | {} |\n",
                row.lang,
                row.allocator,
                row.workload,
                row.mode,
                row.d1_miss_rate.unwrap_or(0.0),
                row.ll_miss_rate.unwrap_or(0.0),
                fmt2(row.d1_misses_per_op),
                fmt2(row.ll_misses_per_op),
                fmt2(row.d1_misses_per_op_net),
                fmt2(row.ll_misses_per_op_net),
            ));
        }
    }

    out.push_str("\n## Rust per-op latency (hdrhistogram)\n\n");
    let any_quantized = rows
        .iter()
        .any(|r| r.source == "rust-latency" && r.quantized == Some(true));
    if any_quantized {
        let tick = rows
            .iter()
            .filter(|r| r.source == "rust-latency")
            .find_map(|r| r.clock_tick_ns)
            .unwrap_or(0);
        out.push_str(&format!(
            "`*` = value is at/below ~3× the measuring clock's tick floor \
             ({tick}ns on the machine that produced these rows) — a \
             quantization bucket, not a latency; do not compare starred \
             values across rows (use the throughput column, which is \
             tick-immune, instead).\n\n"
        ));
    }
    out.push_str("| workload | mode | alloc p50 (ns) | alloc mean (ns) | alloc p99 (ns) | dealloc mean (ns) | throughput (Mops/s) |\n|---|---|---|---|---|---|---|\n");
    for row in rows.iter().filter(|r| r.source == "rust-latency") {
        // Star any percentile that sits under the quantization threshold for
        // this row's measured tick (row-level `quantized` alone would star
        // the whole row — per-value is more honest: p50 may be a bucket
        // while p99 is a real measurement).
        let tick = row.clock_tick_ns.unwrap_or(1);
        let star = |v: f64| -> &'static str {
            if lohalloc_bench::clockinfo::is_quantized(v as u64, tick) {
                "*"
            } else {
                ""
            }
        };
        let fmt =
            |v: Option<f64>| v.map_or_else(|| "-".to_string(), |x| format!("{x:.0}{}", star(x)));
        out.push_str(&format!(
            "| {} | {} | {} | {} | {:.0}{} | {} | {} |\n",
            row.workload,
            row.mode,
            fmt(row.alloc_p50_ns),
            fmt(row.alloc_mean_ns),
            row.alloc_p99_ns.unwrap_or(0.0),
            star(row.alloc_p99_ns.unwrap_or(0.0)),
            fmt(row.dealloc_mean_ns),
            row.rust_throughput_mops
                .map_or_else(|| "-".to_string(), |t| format!("{t:.2}")),
        ));
    }

    fs::write(path, out)
}

/// Best-effort: render the report's JSON into PNG graphs via the Python
/// plotter (which manages its own venv). Never fatal — a machine without
/// python3 (some CI/cloud images) still gets the full report, just no charts.
fn generate_graphs(report_json: &Path, graphs_dir: &Path) {
    if env::var_os("LOHALLOC_NO_GRAPHS").is_some() {
        eprintln!("LOHALLOC_NO_GRAPHS set — skipping graph generation");
        return;
    }
    // bench/graphs/generate.sh lives next to this crate, at the repo root.
    // CARGO_MANIFEST_DIR = crates/lohalloc-bench, so go up two levels.
    let script = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/graphs/generate.sh")
        .canonicalize();
    let Ok(script) = script else {
        eprintln!("graph script not found — skipping graphs");
        return;
    };
    match std::process::Command::new("bash")
        .arg(&script)
        .arg(report_json)
        .arg(graphs_dir)
        .status()
    {
        Ok(s) if s.success() => println!("graphs written to {graphs_dir:?}"),
        Ok(s) => eprintln!("graph generation exited with {s} — report is still complete"),
        Err(e) => eprintln!("could not run graph generator ({e}) — report is still complete"),
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    // `--run-dir` is the run directory the producers already wrote into
    // (`<run-dir>/raw/`); the report + graphs are written back into it. No
    // staging dir, no timestamp minting, no move — the Makefile (`RUN_DIR`)
    // owns the single timestamp for the whole pipeline. `--results-dir` is
    // accepted as a backward-compatible alias.
    let mut run_dir = PathBuf::from("results");
    // The regression gate exits non-zero on failure only when `--gate` is
    // passed (CI). By default aggregate is a reporting tool: it always writes
    // the report + graphs and exits 0, printing the gate status for info —
    // so an interactive `make bench-all` never "fails" just because lohalloc
    // is (still) slower than jemalloc on some workload.
    let mut enforce_gate = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--run-dir" | "--results-dir" if i + 1 < args.len() => {
                run_dir = PathBuf::from(&args[i + 1]);
                i += 1;
            }
            "--gate" => enforce_gate = true,
            _ => {}
        }
        i += 1;
    }

    // Raw per-invocation JSON lives in <run-dir>/raw/; fall back to <run-dir>
    // itself for a flat layout (e.g. a hand-assembled directory of files).
    let raw_dir = run_dir.join("raw");
    let source = if raw_dir.is_dir() {
        raw_dir
    } else {
        run_dir.clone()
    };

    let (rows, samples) = load_rows(&source);
    eprintln!("loaded {} result rows from {:?}", rows.len(), source);

    let gate = evaluate_gate(&rows, &samples);
    let any_gate_failed = gate.iter().any(|(_, pass, _)| !pass);

    if let Err(e) = fs::create_dir_all(&run_dir) {
        eprintln!("failed to create {run_dir:?}: {e}");
    }
    let graphs_dir = run_dir.join("graphs");
    let json_path = run_dir.join("bench-report.json");
    let md_path = run_dir.join("bench-report.md");

    let report = serde_json::json!({
        "rows": rows,
        "gate": gate.iter().map(|(name, pass, detail)| serde_json::json!({
            "name": name, "pass": pass, "detail": detail,
        })).collect::<Vec<_>>(),
    });
    if let Err(e) = fs::write(&json_path, serde_json::to_string_pretty(&report).unwrap()) {
        eprintln!("failed to write {json_path:?}: {e}");
    }
    if let Err(e) = write_markdown(&rows, &gate, &samples, &md_path) {
        eprintln!("failed to write {md_path:?}: {e}");
    }

    println!("wrote {json_path:?} and {md_path:?}");
    generate_graphs(&json_path, &graphs_dir);

    if any_gate_failed {
        if enforce_gate {
            eprintln!("REGRESSION GATE FAILED — see {}", md_path.display());
            std::process::exit(1);
        }
        eprintln!(
            "note: regression gate would FAIL (informational; pass --gate to enforce) — see {}",
            md_path.display()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ops_from_command_parses_trailing_number() {
        assert_eq!(
            ops_from_command(
                "env LOHALLOC_MODEL=/tmp/m.lohalloc LD_PRELOAD=x.so build/bench_main_c adv-mixed 50000"
            ),
            Some(50000)
        );
        // MT workload names carry a -tN suffix but ops is still the tail.
        assert_eq!(
            ops_from_command("taskset -c 0-3 build/bench_main_c mt-mixed-t4 8000"),
            Some(8000)
        );
    }

    #[test]
    fn ops_from_command_rejects_non_numeric_tail() {
        assert_eq!(ops_from_command("build/bench_main_c slab"), None);
        assert_eq!(ops_from_command(""), None);
        assert_eq!(ops_from_command("   "), None);
        // Negative / non-integer tails must not parse as u64.
        assert_eq!(ops_from_command("cmd workload -5"), None);
        assert_eq!(ops_from_command("cmd workload 1.5"), None);
    }

    #[test]
    fn parse_native_filename_handles_dashed_workloads() {
        assert_eq!(
            parse_native_filename("native-c-lohalloc-mt-slab-t4-training"),
            Some((
                "c".to_string(),
                "lohalloc".to_string(),
                "mt-slab-t4".to_string(),
                "training".to_string()
            ))
        );
        assert_eq!(
            parse_native_filename("native-cpp-jemalloc-adv-mixed-baseline"),
            Some((
                "cpp".to_string(),
                "jemalloc".to_string(),
                "adv-mixed".to_string(),
                "baseline".to_string()
            ))
        );
    }

    #[test]
    fn parse_native_filename_rejects_unknown_tokens_loudly() {
        // Unknown allocator (e.g. a future dashed name whose first segment
        // isn't in the known set) must be rejected, not guessed at.
        assert_eq!(
            parse_native_filename("native-c-tcmalloc-slab-training"),
            None
        );
        // Unknown mode: a workload accidentally named to end in a non-mode
        // token would land here — refuse rather than absorb it silently.
        assert_eq!(parse_native_filename("native-c-lohalloc-slab-warmup"), None);
        // A workload whose LAST segment collides with a real mode still
        // parses correctly (mode slot holds the true mode)...
        assert_eq!(
            parse_native_filename("native-c-lohalloc-foo-baseline-training"),
            Some((
                "c".to_string(),
                "lohalloc".to_string(),
                "foo-baseline".to_string(),
                "training".to_string()
            ))
        );
        // ...and too-short names never panic.
        assert_eq!(parse_native_filename("native-c-lohalloc"), None);
    }

    #[test]
    fn mann_whitney_identical_samples_is_not_significant() {
        let a: Vec<f64> = (0..20).map(|i| 1.0 + (i % 5) as f64 * 0.01).collect();
        let p = mann_whitney_p(&a, &a).unwrap();
        assert!(p > 0.9, "identical samples must give p≈1, got {p}");
    }

    #[test]
    fn mann_whitney_clearly_shifted_samples_is_significant() {
        // Two tight distributions 2x apart — unambiguous.
        let a: Vec<f64> = (0..20).map(|i| 1.0 + (i % 7) as f64 * 0.001).collect();
        let b: Vec<f64> = (0..20).map(|i| 2.0 + (i % 7) as f64 * 0.001).collect();
        let p = mann_whitney_p(&a, &b).unwrap();
        assert!(
            p < 1e-6,
            "2x-shifted samples must be wildly significant, got {p}"
        );
        // Symmetric.
        let p2 = mann_whitney_p(&b, &a).unwrap();
        assert!((p - p2).abs() < 1e-12);
    }

    #[test]
    fn mann_whitney_overlapping_noise_is_not_significant() {
        // Same distribution, interleaved differently — heavy overlap.
        let a: Vec<f64> = (0..15).map(|i| ((i * 7919) % 100) as f64).collect();
        let b: Vec<f64> = (0..15).map(|i| ((i * 6113 + 13) % 100) as f64).collect();
        let p = mann_whitney_p(&a, &b).unwrap();
        assert!(
            p > ALPHA,
            "overlapping samples must not be significant, got {p}"
        );
    }

    #[test]
    fn mann_whitney_handles_ties_and_small_samples() {
        // All-tied values: no evidence of difference by construction.
        let a = vec![5.0; 12];
        let b = vec![5.0; 12];
        assert_eq!(mann_whitney_p(&a, &b), Some(1.0));
        // Below the normal-approximation floor -> None, never a bogus p.
        assert_eq!(mann_whitney_p(&a[..4], &b), None);
        assert_eq!(mann_whitney_p(&a, &b[..7]), None);
    }

    #[test]
    fn gate_requires_significance_to_fail() {
        // lohalloc 1.10x slower than jemalloc BUT the samples heavily
        // overlap -> not significant -> must NOT fail (noise cannot flip
        // the gate). Same ratio with tight, separated samples -> FAIL.
        let mk = |allocator: &str, mode: &str, mean: f64| {
            let mut r = empty_row(
                "native-timing",
                "c".into(),
                allocator.into(),
                "slab".into(),
                mode.into(),
            );
            r.mean_ns = Some(mean);
            r
        };
        let rows = vec![
            mk("lohalloc", "inference", 1.10e6),
            mk("jemalloc", "baseline", 1.0e6),
        ];

        let key_lo = (
            "c".to_string(),
            "lohalloc".to_string(),
            "slab".to_string(),
            "inference".to_string(),
        );
        let key_je = (
            "c".to_string(),
            "jemalloc".to_string(),
            "slab".to_string(),
            "baseline".to_string(),
        );

        // Overlapping noisy samples (seconds): both spread 0.8..1.4ms.
        let noisy_lo: Vec<f64> = (0..30)
            .map(|i| 0.0008 + ((i * 37) % 60) as f64 * 1e-5)
            .collect();
        let noisy_je: Vec<f64> = (0..30)
            .map(|i| 0.0008 + ((i * 23 + 7) % 60) as f64 * 1e-5)
            .collect();
        let mut samples: SampleMap = BTreeMap::new();
        samples.insert(key_lo.clone(), noisy_lo);
        samples.insert(key_je.clone(), noisy_je);
        let gate = evaluate_gate(&rows, &samples);
        assert_eq!(gate.len(), 1);
        assert!(
            gate[0].1,
            "over-threshold but insignificant must PASS: {}",
            gate[0].2
        );
        assert!(gate[0].2.contains("not significant"));

        // Tight separated samples: lohalloc clearly slower.
        let tight_lo: Vec<f64> = (0..30).map(|i| 0.00110 + (i % 5) as f64 * 1e-7).collect();
        let tight_je: Vec<f64> = (0..30).map(|i| 0.00100 + (i % 5) as f64 * 1e-7).collect();
        samples.insert(key_lo, tight_lo);
        samples.insert(key_je, tight_je);
        let gate = evaluate_gate(&rows, &samples);
        assert!(
            !gate[0].1,
            "significant over-threshold loss must FAIL: {}",
            gate[0].2
        );

        // No samples at all: strict ratio semantics (old runs) -> FAIL.
        let gate = evaluate_gate(&rows, &BTreeMap::new());
        assert!(!gate[0].1, "sample-less over-threshold must keep failing");
    }

    #[test]
    fn hypothesis_verdicts_roll_up_correctly() {
        let tvi = vec![
            TrainVsInference {
                lang: "c".into(),
                workload: "slab".into(),
                train_mean_ns: 6.0e6,
                inf_mean_ns: 1.5e6,
                p_value: Some(1e-9), // significantly faster
                combined_stddev_ns: None,
                within_noise: false,
            },
            TrainVsInference {
                lang: "cpp".into(),
                workload: "adv-mixed".into(),
                train_mean_ns: 20.0e6,
                inf_mean_ns: 26.0e6,
                p_value: Some(1e-5), // significantly SLOWER -> H-A fail
                combined_stddev_ns: None,
                within_noise: false,
            },
            TrainVsInference {
                lang: "rust".into(),
                workload: "buddy".into(),
                train_mean_ns: 90.0e6,
                inf_mean_ns: 88.0e6,
                p_value: Some(0.4), // inconclusive
                combined_stddev_ns: None,
                within_noise: true,
            },
        ];
        let hb = vec![
            HbComparison {
                lang: "c".into(),
                workload: "slab".into(),
                baseline: "jemalloc".into(),
                ratio: 0.9,
                p_value: Some(1e-4),
            },
            HbComparison {
                lang: "c".into(),
                workload: "buddy".into(),
                baseline: "jemalloc".into(),
                ratio: 1.8,
                p_value: Some(1e-8),
            },
            HbComparison {
                lang: "c".into(),
                workload: "arena".into(),
                baseline: "mimalloc".into(),
                ratio: 1.02,
                p_value: Some(0.5),
            },
        ];
        assert_eq!(tvi[0].ha_verdict(), Some(true));
        assert_eq!(tvi[1].ha_verdict(), Some(false));
        assert_eq!(tvi[2].ha_verdict(), None);
        assert_eq!(hb[0].verdict(), "wins");
        assert_eq!(hb[1].verdict(), "loses");
        assert_eq!(hb[2].verdict(), "no significant difference");

        let mut md = String::new();
        write_hypothesis_verdicts(&mut md, &tvi, &hb);
        assert!(
            md.contains("holds significantly in 1/3 rows, inconclusive in 1, fails in 1"),
            "{md}"
        );
        assert!(md.contains("H-A FAIL: cpp/adv-mixed"), "{md}");
        assert!(md.contains("**H-B vs jemalloc:** lohalloc-inference wins significantly in 1/2 rows, loses in 1"), "{md}");
        assert!(md.contains("**H-B vs mimalloc:**"), "{md}");
    }

    #[test]
    fn quantized_percentiles_get_starred_in_markdown() {
        // One quantized row (tick 42, p50=42 is a bucket) and one clean row.
        let mut q = empty_row(
            "rust-latency",
            "rust".into(),
            "lohalloc".into(),
            "slab".into(),
            "inference".into(),
        );
        q.alloc_p50_ns = Some(42.0);
        q.alloc_p99_ns = Some(4_000.0);
        q.alloc_mean_ns = Some(26.0);
        q.clock_tick_ns = Some(42);
        q.quantized = Some(true);
        let mut clean = q.clone();
        clean.workload = "buddy".into();
        clean.alloc_p50_ns = Some(541.0);
        clean.alloc_mean_ns = Some(1_078.0);
        clean.quantized = Some(false);

        let dir = std::env::temp_dir().join("lohalloc-aggregate-quantized-test");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("report.md");
        write_markdown(&[q, clean], &[], &BTreeMap::new(), &path).unwrap();
        let md = fs::read_to_string(&path).unwrap();

        assert!(md.contains("tick floor"), "footnote missing:\n{md}");
        assert!(md.contains("| 42* |"), "quantized p50 not starred:\n{md}");
        assert!(md.contains("| 26* |"), "quantized mean not starred:\n{md}");
        assert!(!md.contains("4000*"), "real p99 must not be starred:\n{md}");
        assert!(!md.contains("541*"), "clean row must not be starred:\n{md}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn throughput_math_from_hyperfine_row() {
        // 50_000 ops in 2ms = 25 Mops/s.
        let ops = ops_from_command("bin slab 50000").unwrap();
        let mean_seconds = 0.002;
        let mops = ops as f64 / mean_seconds / 1e6;
        assert!((mops - 25.0).abs() < 1e-9);
    }

    #[test]
    fn per_op_misses_are_denominator_immune() {
        // The exact scenario from the module doc: training does ~2.8x more
        // refs. Rates invert (training looks better); per-op misses tell
        // the truth (inference has fewer misses per op).
        let ops = 2000u64;
        let (train_refs, train_misses) = (1_181_790u64, 12_015u64);
        let (inf_refs, inf_misses) = (512_268u64, 8_987u64);

        let train_rate = train_misses as f64 / train_refs as f64;
        let inf_rate = inf_misses as f64 / inf_refs as f64;
        assert!(
            inf_rate > train_rate,
            "rate view: inference 'worse' ({inf_rate:.4} > {train_rate:.4})"
        );

        let train_per_op = train_misses as f64 / ops as f64;
        let inf_per_op = inf_misses as f64 / ops as f64;
        assert!(
            inf_per_op < train_per_op,
            "per-op view: inference actually better ({inf_per_op:.2} < {train_per_op:.2})"
        );
    }

    fn native_row(
        lang: &str,
        workload: &str,
        mode: &str,
        mean_ns: f64,
        stddev_ns: Option<f64>,
    ) -> Row {
        let mut row = empty_row(
            "native-timing",
            lang.to_string(),
            "lohalloc".to_string(),
            workload.to_string(),
            mode.to_string(),
        );
        row.mean_ns = Some(mean_ns);
        row.stddev_ns = stddev_ns;
        row
    }

    #[test]
    fn stddev_parses_from_hyperfine_export() {
        let json = r#"{"results":[{"mean":0.002,"stddev":0.0001,"command":"bin slab 50000"}]}"#;
        let export: HyperfineExport = serde_json::from_str(json).unwrap();
        assert_eq!(export.results[0].stddev, Some(0.0001));
        // Old / hand-assembled files without stddev still parse.
        let json = r#"{"results":[{"mean":0.002}]}"#;
        let export: HyperfineExport = serde_json::from_str(json).unwrap();
        assert_eq!(export.results[0].stddev, None);
    }

    #[test]
    fn train_vs_inference_flags_within_noise_deltas() {
        // Delta 5ms, combined stddev 3+4=7ms -> within noise even though
        // inference is nominally slower.
        let rows = vec![
            native_row("cpp", "adv-mixed", "training", 100e6, Some(3e6)),
            native_row("cpp", "adv-mixed", "inference", 105e6, Some(4e6)),
            // Delta 40ms >> combined stddev 2ms -> real slowdown.
            native_row("cpp", "cpp-string", "training", 60e6, Some(1e6)),
            native_row("cpp", "cpp-string", "inference", 100e6, Some(1e6)),
            // Missing stddev on one side -> no noise verdict possible.
            native_row("c", "slab", "training", 7e6, None),
            native_row("c", "slab", "inference", 2e6, Some(0.1e6)),
        ];
        let tvi = train_vs_inference(&rows, &BTreeMap::new());
        assert_eq!(tvi.len(), 3);

        let by_key = |lang: &str, wl: &str| {
            tvi.iter()
                .find(|c| c.lang == lang && c.workload == wl)
                .unwrap()
        };
        assert!(by_key("cpp", "adv-mixed").within_noise);
        let string_row = by_key("cpp", "cpp-string");
        assert!(!string_row.within_noise);
        assert!(string_row.inf_mean_ns > string_row.train_mean_ns);
        let slab_row = by_key("c", "slab");
        assert!(slab_row.combined_stddev_ns.is_none());
        assert!(!slab_row.within_noise);
    }

    #[test]
    fn train_vs_inference_ignores_baselines_and_unpaired_rows() {
        let mut jemalloc = native_row("c", "slab", "baseline", 1e6, Some(0.1e6));
        jemalloc.allocator = "jemalloc".to_string();
        let rows = vec![
            jemalloc,
            // training with no matching inference row: skipped.
            native_row("c", "buddy", "training", 95e6, Some(1e6)),
        ];
        assert!(train_vs_inference(&rows, &BTreeMap::new()).is_empty());
    }

    fn cg(ops: u64, d1: u64, ll: u64) -> CachegrindResult {
        CachegrindResult {
            lang: "c".into(),
            allocator: "lohalloc".into(),
            workload: "buddy".into(),
            mode: "inference".into(),
            ops: Some(ops),
            calibration: false,
            d_refs: 1_000_000,
            d1_misses: d1,
            ll_misses: ll,
        }
    }

    #[test]
    fn cachegrind_net_subtracts_startup_calibration() {
        // 16_000 gross misses over 2000 ops = 8.0/op, of which 6_000 are
        // one-time startup (model load + PHT build) -> net 5.0/op.
        let row = cachegrind_row(cg(2000, 16_000, 2_000), Some((6_000, 1_000)));
        assert_eq!(row.d1_misses_per_op, Some(8.0));
        assert_eq!(row.d1_misses_per_op_net, Some(5.0));
        assert_eq!(row.ll_misses_per_op_net, Some(0.5));
    }

    #[test]
    fn cachegrind_net_absent_without_calibration_row() {
        // Pre-calibration raw dirs keep working: gross only, no net.
        let row = cachegrind_row(cg(2000, 16_000, 2_000), None);
        assert_eq!(row.d1_misses_per_op, Some(8.0));
        assert_eq!(row.d1_misses_per_op_net, None);
        assert_eq!(row.ll_misses_per_op_net, None);
    }

    #[test]
    fn cachegrind_net_clamps_at_zero() {
        // A near-empty workload (system at tiny ops) can measure main < cal
        // within simulation jitter — clamp, never go negative.
        let row = cachegrind_row(cg(100, 500, 50), Some((600, 80)));
        assert_eq!(row.d1_misses_per_op_net, Some(0.0));
        assert_eq!(row.ll_misses_per_op_net, Some(0.0));
    }

    #[test]
    fn calibration_flag_parses_and_defaults_false() {
        let with_flag = r#"{"lang":"c","allocator":"lohalloc","workload":"buddy",
            "mode":"inference","ops":1,"calibration":true,
            "d_refs":10,"d1_misses":5,"ll_misses":1}"#;
        let cg: CachegrindResult = serde_json::from_str(with_flag).unwrap();
        assert!(cg.calibration);
        let without = r#"{"lang":"c","allocator":"lohalloc","workload":"buddy",
            "mode":"inference","ops":2000,
            "d_refs":10,"d1_misses":5,"ll_misses":1}"#;
        let cg: CachegrindResult = serde_json::from_str(without).unwrap();
        assert!(!cg.calibration);
    }

    #[test]
    fn rust_throughput_from_mean_latency() {
        // alloc_mean_ns = 40ns -> 25 M allocs/s.
        let mean_ns = 40.0f64;
        assert!((1e3 / mean_ns - 25.0).abs() < 1e-9);
    }

    fn perf(ops: u64, l1d_m: u64, llc_m: u64) -> PerfResult {
        perf_full(ops, l1d_m, llc_m, 0, 0)
    }

    fn perf_full(
        ops: u64,
        l1d_m: u64,
        llc_m: u64,
        page_faults: u64,
        instructions: u64,
    ) -> PerfResult {
        PerfResult {
            lang: "rust".into(),
            allocator: "lohalloc".into(),
            workload: "mt-interfere-t8".into(),
            mode: "inference".into(),
            ops: Some(ops),
            calibration: false,
            l1d_loads: 1_000_000,
            l1d_load_misses: l1d_m,
            llc_loads: 100_000,
            llc_load_misses: llc_m,
            page_faults,
            instructions,
        }
    }

    #[test]
    fn perf_row_maps_l1d_llc_and_nets_startup() {
        // 16_000 gross LLC misses over 2000 ops = 8.0/op, of which 6_000 are
        // one-time startup (thread spawn + model load) -> net 5.0/op. Mirrors
        // the cachegrind net test but on the real-PMU path, reusing the shared
        // d1/ll Row fields (L1-dcache -> D1 view, LLC -> LL view).
        let row = perf_row(perf(2000, 4_000, 16_000), Some((1_000, 6_000, 0, 0)));
        assert_eq!(row.source, "perf");
        assert_eq!(row.d1_miss_rate, Some(4_000.0 / 1_000_000.0));
        assert_eq!(row.ll_miss_rate, Some(16_000.0 / 100_000.0));
        assert_eq!(row.ll_misses_per_op, Some(8.0));
        assert_eq!(row.ll_misses_per_op_net, Some(5.0));
        assert_eq!(row.d1_misses_per_op_net, Some(1.5));
    }

    #[test]
    fn perf_row_clamps_net_and_survives_missing_calibration() {
        // main < cal (noisy real counters) clamps at 0, never negative.
        let clamped = perf_row(perf(100, 50, 500), Some((80, 600, 0, 0)));
        assert_eq!(clamped.ll_misses_per_op_net, Some(0.0));
        // No calibration companion -> gross only, no net (informational rows
        // from a pre-calibration or partial run still render).
        let no_cal = perf_row(perf(2000, 4_000, 16_000), None);
        assert_eq!(no_cal.ll_misses_per_op, Some(8.0));
        assert_eq!(no_cal.ll_misses_per_op_net, None);
    }

    #[test]
    fn perf_row_nets_page_faults_and_instructions() {
        // 12_000 gross faults over 2000 ops = 6.0/op, 2_000 one-time startup
        // -> net 5.0/op. Instructions likewise (60_000 gross, 20_000 startup).
        let row = perf_row(
            perf_full(2000, 4_000, 16_000, 12_000, 60_000),
            Some((1_000, 6_000, 2_000, 20_000)),
        );
        assert_eq!(row.page_faults_per_op_net, Some(5.0));
        assert_eq!(row.instructions_per_op_net, Some(20.0));
    }

    #[test]
    fn perf_row_hides_instructions_when_pmu_reports_zero() {
        // Virtualized Nitro hosts often expose 0 for instructions/cycles —
        // publish None, not a spurious 0/op. Page faults (a software event)
        // still net normally.
        let row = perf_row(
            perf_full(2000, 4_000, 16_000, 12_000, 0),
            Some((1_000, 6_000, 2_000, 0)),
        );
        assert_eq!(row.instructions_per_op_net, None);
        assert_eq!(row.page_faults_per_op_net, Some(5.0));
    }

    #[test]
    fn perf_result_parses_source_pmu_shape() {
        // The exact JSON perf_pass writes (extra fields like cache_references
        // are ignored by serde), calibration defaults false. page_faults /
        // instructions default 0 for raw files from before those counters.
        let json = r#"{"lang":"rust","allocator":"lohalloc","workload":"mt-xfree-t4",
            "mode":"inference","ops":50000,"calibration":false,
            "cache_references":9,"cache_misses":3,
            "l1d_loads":100,"l1d_load_misses":10,
            "llc_loads":20,"llc_load_misses":2,"source":"pmu"}"#;
        let p: PerfResult = serde_json::from_str(json).unwrap();
        assert!(!p.calibration);
        assert_eq!(p.llc_load_misses, 2);
        assert_eq!(p.page_faults, 0);
        assert_eq!(p.instructions, 0);
    }

    fn hyperfine_json(mean_s: f64, ops: u64) -> String {
        format!(
            r#"{{"results":[{{"mean":{mean_s},"stddev":0.0,
                "command":"env BIN slab {ops}",
                "times":[{mean_s},{mean_s},{mean_s},{mean_s},{mean_s},{mean_s},{mean_s},{mean_s},{mean_s},{mean_s}]}}]}}"#
        )
    }

    #[test]
    fn timing_cal_companion_populates_startup_and_net() {
        // A `native-…-cal.json` (ops=1) pairs with its main row to yield the
        // startup-immune wall view without becoming a report row itself.
        let dir = std::env::temp_dir().join(format!("lohalloc-agg-timing-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        // Main: 300 µs for 50000 ops. Companion: 100 µs fixed startup.
        fs::write(
            dir.join("native-c-lohalloc-slab-inference.json"),
            hyperfine_json(300e-6, 50000),
        )
        .unwrap();
        fs::write(
            dir.join("native-c-lohalloc-slab-inference-cal.json"),
            hyperfine_json(100e-6, 1),
        )
        .unwrap();

        let (rows, _) = load_rows(&dir);
        let _ = fs::remove_dir_all(&dir);
        // Exactly one report row — the `-cal` companion is consumed, not
        // emitted as its own row.
        let timing: Vec<&Row> = rows
            .iter()
            .filter(|r| r.source == "native-timing")
            .collect();
        assert_eq!(
            timing.len(),
            1,
            "cal companion must not become a report row"
        );
        let row = timing[0];
        assert_eq!(row.workload, "slab");
        assert!((row.mean_ns.unwrap() - 300_000.0).abs() < 1.0);
        assert!((row.startup_ns.unwrap() - 100_000.0).abs() < 1.0);
        assert!((row.mean_ns_net.unwrap() - 200_000.0).abs() < 1.0);
    }

    #[test]
    fn rss_pass_row_parses_with_kib() {
        let dir = std::env::temp_dir().join(format!("lohalloc-agg-rss-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("rss-c-lohalloc-request-loop-inference.json"),
            r#"{"lang":"c","allocator":"lohalloc","workload":"request-loop","mode":"inference","rss_kib":2048,"source":"rss"}"#,
        )
        .unwrap();
        let (rows, _) = load_rows(&dir);
        let _ = fs::remove_dir_all(&dir);
        let rss: Vec<&Row> = rows.iter().filter(|r| r.source == "rss").collect();
        assert_eq!(rss.len(), 1);
        assert_eq!(rss[0].workload, "request-loop");
        assert_eq!(rss[0].rss_kib, Some(2048));
    }
}
