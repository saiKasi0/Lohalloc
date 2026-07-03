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
    /// D1 (L1 data cache) miss rate, 0.0-1.0 (cachegrind only).
    d1_miss_rate: Option<f64>,
    /// Last-level cache miss rate, 0.0-1.0 (cachegrind only).
    ll_miss_rate: Option<f64>,
    /// p99 alloc latency, nanoseconds (rust-latency only).
    alloc_p99_ns: Option<f64>,
}

#[derive(Deserialize)]
struct HyperfineExport {
    results: Vec<HyperfineResult>,
}

#[derive(Deserialize)]
struct HyperfineResult {
    mean: f64, // seconds
}

#[derive(Deserialize)]
struct CachegrindResult {
    lang: String,
    allocator: String,
    workload: String,
    mode: String,
    d_refs: u64,
    d1_misses: u64,
    ll_misses: u64,
}

#[derive(Deserialize)]
struct RustLatencyResult {
    workload: String,
    mode: String,
    alloc_p99_ns: f64,
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
                        rows.push(Row {
                            source: "native-timing",
                            lang,
                            allocator,
                            workload,
                            mode,
                            mean_ns: Some(r.mean * 1e9),
                            d1_miss_rate: None,
                            ll_miss_rate: None,
                            alloc_p99_ns: None,
                        });
                    }
                }
                Err(e) => eprintln!("failed to parse {path:?} as hyperfine export: {e}"),
            }
        } else if stem.starts_with("cachegrind-") {
            match serde_json::from_str::<CachegrindResult>(&text) {
                Ok(cg) => {
                    let d1_rate = if cg.d_refs > 0 {
                        Some(cg.d1_misses as f64 / cg.d_refs as f64)
                    } else {
                        None
                    };
                    let ll_rate = if cg.d_refs > 0 {
                        Some(cg.ll_misses as f64 / cg.d_refs as f64)
                    } else {
                        None
                    };
                    rows.push(Row {
                        source: "cachegrind",
                        lang: cg.lang,
                        allocator: cg.allocator,
                        workload: cg.workload,
                        mode: cg.mode,
                        mean_ns: None,
                        d1_miss_rate: d1_rate,
                        ll_miss_rate: ll_rate,
                        alloc_p99_ns: None,
                    });
                }
                Err(e) => eprintln!("failed to parse {path:?} as cachegrind result: {e}"),
            }
        } else if stem.starts_with("rust_") || stem.starts_with("rust-") {
            match serde_json::from_str::<RustLatencyResult>(&text) {
                Ok(r) => rows.push(Row {
                    source: "rust-latency",
                    lang: "rust".to_string(),
                    allocator: "lohalloc".to_string(),
                    workload: r.workload,
                    mode: r.mode,
                    mean_ns: None,
                    d1_miss_rate: None,
                    ll_miss_rate: None,
                    alloc_p99_ns: Some(r.alloc_p99_ns),
                }),
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
    out.push_str("| lang | allocator | workload | mode | mean (ns) |\n|---|---|---|---|---|\n");
    for row in rows.iter().filter(|r| r.source == "native-timing") {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {:.0} |\n",
            row.lang,
            row.allocator,
            row.workload,
            row.mode,
            row.mean_ns.unwrap_or(0.0)
        ));
    }

    out.push_str("\n## Cache metrics (cachegrind, simulated)\n\n");
    out.push_str("| lang | allocator | workload | mode | D1 miss rate | LL miss rate |\n|---|---|---|---|---|---|\n");
    for row in rows.iter().filter(|r| r.source == "cachegrind") {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {:.4} | {:.4} |\n",
            row.lang,
            row.allocator,
            row.workload,
            row.mode,
            row.d1_miss_rate.unwrap_or(0.0),
            row.ll_miss_rate.unwrap_or(0.0)
        ));
    }

    out.push_str("\n## Rust per-op latency (hdrhistogram)\n\n");
    out.push_str("| workload | mode | alloc p99 (ns) |\n|---|---|---|\n");
    for row in rows.iter().filter(|r| r.source == "rust-latency") {
        out.push_str(&format!(
            "| {} | {} | {:.0} |\n",
            row.workload,
            row.mode,
            row.alloc_p99_ns.unwrap_or(0.0)
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
