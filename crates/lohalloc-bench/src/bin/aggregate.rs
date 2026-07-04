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
        throughput_mops: None,
        d1_miss_rate: None,
        ll_miss_rate: None,
        d1_misses_per_op: None,
        ll_misses_per_op: None,
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
                Ok(cg) => {
                    let mut row =
                        empty_row("cachegrind", cg.lang, cg.allocator, cg.workload, cg.mode);
                    if cg.d_refs > 0 {
                        row.d1_miss_rate = Some(cg.d1_misses as f64 / cg.d_refs as f64);
                        row.ll_miss_rate = Some(cg.ll_misses as f64 / cg.d_refs as f64);
                    }
                    if let Some(ops) = cg.ops.filter(|&o| o > 0) {
                        row.d1_misses_per_op = Some(cg.d1_misses as f64 / ops as f64);
                        row.ll_misses_per_op = Some(cg.ll_misses as f64 / ops as f64);
                    }
                    rows.push(row);
                }
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
    rows
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
        "| lang | allocator | workload | mode | mean (ns) | throughput (Mops/s) |\n|---|---|---|---|---|---|\n",
    );
    for row in rows.iter().filter(|r| r.source == "native-timing") {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {:.0} | {} |\n",
            row.lang,
            row.allocator,
            row.workload,
            row.mode,
            row.mean_ns.unwrap_or(0.0),
            row.throughput_mops
                .map_or_else(|| "-".to_string(), |t| format!("{t:.2}")),
        ));
    }

    out.push_str("\n## Cache metrics (cachegrind, simulated)\n\n");
    out.push_str(
        "Rates dilute when a mode does more per-op bookkeeping (training) — \
         per-op misses are the denominator-immune view; see the module doc.\n\n",
    );
    out.push_str("| lang | allocator | workload | mode | D1 miss rate | LL miss rate | D1 misses/op | LL misses/op |\n|---|---|---|---|---|---|---|---|\n");
    for row in rows.iter().filter(|r| r.source == "cachegrind") {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {:.4} | {:.4} | {} | {} |\n",
            row.lang,
            row.allocator,
            row.workload,
            row.mode,
            row.d1_miss_rate.unwrap_or(0.0),
            row.ll_miss_rate.unwrap_or(0.0),
            row.d1_misses_per_op
                .map_or_else(|| "-".to_string(), |v| format!("{v:.2}")),
            row.ll_misses_per_op
                .map_or_else(|| "-".to_string(), |v| format!("{v:.2}")),
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

    #[test]
    fn rust_throughput_from_mean_latency() {
        // alloc_mean_ns = 40ns -> 25 M allocs/s.
        let mean_ns = 40.0f64;
        assert!((1e3 / mean_ns - 25.0).abs() < 1e-9);
    }
}
