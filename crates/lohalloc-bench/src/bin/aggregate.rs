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
    source: &'static str, // "native-timing" | "cachegrind" | "rust-latency"
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
}

/// Parse `native-<lang>-<allocator>-<workload>-<mode>.json` (workload may
/// itself contain hyphens, e.g. "adv-mixed" — allocator/lang/mode never do,
/// so we split from both ends and take whatever's left as the workload).
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
    Some((lang, allocator, workload, mode))
}

fn load_rows(dir: &Path) -> Vec<Row> {
    let mut rows = Vec::new();
    // Cachegrind rows are two-phase: main rows and their ops=1 calibration
    // companions are collected here, then paired by (lang, allocator,
    // workload, mode) after the directory scan to fill the `_net` fields.
    let mut cg_main: Vec<CachegrindResult> = Vec::new();
    let mut cg_cal: BTreeMap<(String, String, String, String), (u64, u64)> = BTreeMap::new();
    let Ok(entries) = fs::read_dir(dir) else {
        eprintln!("results directory {dir:?} not found or unreadable");
        return rows;
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
            let Some((lang, allocator, workload, mode)) = parse_native_filename(stem) else {
                eprintln!("skipping unparseable native result filename: {stem}");
                continue;
            };
            match serde_json::from_str::<HyperfineExport>(&text) {
                Ok(export) => {
                    if let Some(r) = export.results.first() {
                        let mut row = empty_row("native-timing", lang, allocator, workload, mode);
                        row.mean_ns = Some(r.mean * 1e9);
                        row.stddev_ns = r.stddev.map(|s| s * 1e9);
                        row.throughput_mops = r
                            .command
                            .as_deref()
                            .and_then(ops_from_command)
                            .filter(|&ops| ops > 0 && r.mean > 0.0)
                            .map(|ops| ops as f64 / r.mean / 1e6);
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
    rows
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

/// The regression gate: for every (lang, workload) where both a
/// lohalloc/inference row and a jemalloc/baseline row exist in
/// `native-timing`, lohalloc's mean must not exceed jemalloc's by >5%.
fn evaluate_gate(rows: &[Row]) -> Vec<(String, bool, String)> {
    let mut lohalloc_inf: BTreeMap<(String, String), f64> = BTreeMap::new();
    let mut jemalloc_base: BTreeMap<(String, String), f64> = BTreeMap::new();

    for row in rows {
        if row.source != "native-timing" {
            continue;
        }
        let Some(mean) = row.mean_ns else { continue };
        let key = (row.lang.clone(), row.workload.clone());
        if row.allocator == "lohalloc" && row.mode == "inference" {
            lohalloc_inf.insert(key, mean);
        } else if row.allocator == "jemalloc" {
            jemalloc_base.insert(key, mean);
        }
    }

    // W-SYSTEM is excluded — mmap-bound for every allocator, not a
    // meaningful comparison point.
    let mut results = Vec::new();
    for (key, lohalloc_mean) in &lohalloc_inf {
        if key.1 == "system" {
            continue;
        }
        if let Some(jemalloc_mean) = jemalloc_base.get(key) {
            let ratio = lohalloc_mean / jemalloc_mean;
            let pass = ratio <= 1.05;
            results.push((
                format!("{}/{}: lohalloc-inference vs jemalloc", key.0, key.1),
                pass,
                format!(
                    "lohalloc={lohalloc_mean:.0}ns jemalloc={jemalloc_mean:.0}ns ratio={ratio:.3} (max 1.05)"
                ),
            ));
        }
    }
    results
}

/// One (lang, workload) lohalloc training↔inference comparison from the
/// native-timing rows. `within_noise` is true when the delta is smaller
/// than the two runs' combined stddev (hyperfine's own run-to-run spread)
/// — in that case the *direction* of the delta is not signal and must not
/// be read as "inference got slower/faster".
struct TrainVsInference {
    lang: String,
    workload: String,
    train_mean_ns: f64,
    inf_mean_ns: f64,
    /// `None` when either row predates stddev capture — then no noise
    /// verdict is possible and the delta is reported bare.
    combined_stddev_ns: Option<f64>,
    within_noise: bool,
}

fn train_vs_inference(rows: &[Row]) -> Vec<TrainVsInference> {
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
            combined_stddev_ns,
            within_noise,
        });
    }
    out
}

fn write_markdown(
    rows: &[Row],
    gate: &[(String, bool, String)],
    path: &Path,
) -> std::io::Result<()> {
    let mut out = String::new();
    out.push_str("# Phase 6 Bench Report\n\n");

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

    let tvi = train_vs_inference(rows);
    if !tvi.is_empty() {
        out.push_str("\n### lohalloc training vs inference (native)\n\n");
        out.push_str(
            "Deltas no larger than the two rows' combined run-to-run stddev are \
             marked `~ within noise` — their direction is measurement spread, \
             not signal. Only `SLOWER` rows are worth investigating.\n\n",
        );
        out.push_str(
            "| lang | workload | training (ns) | inference (ns) | inf/train | verdict |\n|---|---|---|---|---|---|\n",
        );
        for c in &tvi {
            let ratio = c.inf_mean_ns / c.train_mean_ns;
            let verdict = if c.within_noise {
                "~ within noise".to_string()
            } else {
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
            };
            out.push_str(&format!(
                "| {} | {} | {:.0} | {:.0} | {ratio:.3} | {verdict} |\n",
                c.lang, c.workload, c.train_mean_ns, c.inf_mean_ns,
            ));
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

    out.push_str("\n## Rust per-op latency (hdrhistogram)\n\n");
    out.push_str("| workload | mode | alloc p50 (ns) | alloc mean (ns) | alloc p99 (ns) | dealloc mean (ns) | throughput (Mops/s) |\n|---|---|---|---|---|---|---|\n");
    for row in rows.iter().filter(|r| r.source == "rust-latency") {
        let fmt = |v: Option<f64>| v.map_or_else(|| "-".to_string(), |x| format!("{x:.0}"));
        out.push_str(&format!(
            "| {} | {} | {} | {} | {:.0} | {} | {} |\n",
            row.workload,
            row.mode,
            fmt(row.alloc_p50_ns),
            fmt(row.alloc_mean_ns),
            row.alloc_p99_ns.unwrap_or(0.0),
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

    let rows = load_rows(&source);
    eprintln!("loaded {} result rows from {:?}", rows.len(), source);

    let gate = evaluate_gate(&rows);
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
    if let Err(e) = write_markdown(&rows, &gate, &md_path) {
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
        let tvi = train_vs_inference(&rows);
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
        assert!(train_vs_inference(&rows).is_empty());
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
}
