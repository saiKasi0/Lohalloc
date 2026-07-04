//! Step 8 ablation sweep: run the training→inference pipeline once per
//! tune-config point and rank the results — the "front-loaded, automated
//! ablation" harness that turns `focus` presets from guesses into measured
//! winners.
//!
//! ```sh
//! cargo run -p lohalloc-bench --bin tune_sweep --release -- \
//!     --grid bench/tune-grid.json --workloads slab,buddy,adv-mixed \
//!     --ops 100000 --out results/tune-sweep
//! # or via the Makefile: make bench-tune
//! ```
//!
//! # How it works
//!
//! The training config is a process-wide `OnceLock` (`lohalloc_alloc::tune`),
//! so distinct config points **cannot** coexist in one process — each grid
//! point therefore runs as a child `latency_profile` process with
//! `LOHALLOC_TUNE=<generated key=value file>` in its environment (JSON stays
//! at this harness layer; the allocator only ever parses the flat file).
//! Per (config, workload) the sweep runs `--mode training` and
//! `--mode inference` and collects the `LatencyReport` metrics: mean / p50 /
//! p99 alloc latency, *measured* wall-clock throughput (Mops/s), and peak
//! RSS (the memory-side check on `frag_weight` configs). A pure-defaults
//! baseline point is always injected (point 0) and every row reports its
//! Δ% against it.
//!
//! # Grid format
//!
//! A JSON object mapping tune keys to candidate value lists; the sweep runs
//! the full cartesian product:
//!
//! ```json
//! { "focus": ["latency", "throughput"], "ucb_c": [1.0, 2.0] }
//! ```
//!
//! Values may be strings or numbers; they are written verbatim as
//! `key=value` lines (see `tune.rs` for the key table).
//!
//! # Ranking
//!
//! One ranked table per `focus` value found in the grid (configs without a
//! `focus` key rank in the "latency" group, the default): `throughput`
//! groups rank by inference throughput (descending), everything else by
//! inference p99 (ascending) — each group is judged by the metric it claims
//! to optimize. Same-session methodology applies: all children run
//! back-to-back on this machine, so ratios *within* one sweep are
//! comparable; absolute numbers across sweeps are not.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct LatencyReport {
    alloc_p50_ns: u64,
    alloc_p99_ns: u64,
    alloc_mean_ns: f64,
    /// Wall-clock throughput of the measured phase (Mops/s) — the real
    /// number a throughput focus is judged by, unlike the synthetic
    /// `1e3/mean` derived from per-op timer sums. `default` so reports
    /// from older `latency_profile` builds still parse (rank as 0).
    #[serde(default)]
    measured_mops: f64,
    /// Peak RSS in bytes — the memory-side view for judging `frag_weight`
    /// configs.
    #[serde(default)]
    peak_rss_bytes: u64,
}

/// One grid point: ordered `(key, value)` pairs (BTreeMap for a stable,
/// readable config label).
type ConfigPoint = BTreeMap<String, String>;

/// One mode's collected metrics.
#[derive(Clone, Copy, Serialize)]
struct Metrics {
    p50: u64,
    p99: u64,
    mean: f64,
    /// Measured wall-clock Mops/s (0.0 when the child predates the field).
    mops: f64,
    rss_bytes: u64,
}

#[derive(Serialize)]
struct RunResult {
    label: String,
    focus: String,
    workload: String,
    training: Option<Metrics>,
    inference: Option<Metrics>,
}

impl RunResult {
    fn is_defaults(&self) -> bool {
        self.label == "defaults"
    }
}

struct Args {
    grid: PathBuf,
    workloads: Vec<String>,
    ops: usize,
    out: PathBuf,
    /// Additionally ablate the *production* LD_PRELOAD path: per config
    /// point, drive `bench/run_native.sh`'s lohalloc triple (C bench_main
    /// under lohalloc-cabi) with `TUNE_FILE=<config>` — in Docker on
    /// non-Linux hosts, directly on Linux.
    native: bool,
    /// Hyperfine op count for the native triple (its own scale, distinct
    /// from the in-process `--ops`).
    native_ops: usize,
}

fn parse_args() -> Args {
    let mut parsed = Args {
        grid: PathBuf::from("bench/tune-grid.json"),
        workloads: vec![
            "slab".to_string(),
            "buddy".to_string(),
            "adv-mixed".to_string(),
        ],
        ops: 100_000,
        out: PathBuf::from("results/tune-sweep"),
        native: false,
        native_ops: 50_000,
    };
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--grid" => parsed.grid = args.next().map(PathBuf::from).unwrap_or(parsed.grid),
            "--workloads" => {
                if let Some(list) = args.next() {
                    parsed.workloads = list.split(',').map(|s| s.trim().to_string()).collect();
                }
            }
            "--ops" => {
                parsed.ops = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(parsed.ops)
            }
            "--out" => parsed.out = args.next().map(PathBuf::from).unwrap_or(parsed.out),
            "--native" => parsed.native = true,
            "--native-ops" => {
                parsed.native_ops = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(parsed.native_ops)
            }
            other => eprintln!("tune_sweep: ignoring unknown arg {other}"),
        }
    }
    parsed
}

/// Expand a `{key: [v1, v2, ...]}` grid into its cartesian product. Keys
/// iterate in JSON-object order via `serde_json::Map` (preserve_order is
/// off, so alphabetical — deterministic either way).
fn expand_grid(grid: &serde_json::Map<String, serde_json::Value>) -> Vec<ConfigPoint> {
    let mut points: Vec<ConfigPoint> = vec![ConfigPoint::new()];
    for (key, values) in grid {
        let Some(list) = values.as_array() else {
            eprintln!("tune_sweep: grid key {key} is not an array — skipping");
            continue;
        };
        let mut next = Vec::with_capacity(points.len() * list.len());
        for point in &points {
            for v in list {
                let mut p = point.clone();
                // Strings render without JSON quotes; numbers/bools as-is.
                let rendered = match v {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                p.insert(key.clone(), rendered);
                next.push(p);
            }
        }
        points = next;
    }
    points
}

/// Always sweep a pure-defaults baseline as point 0 — every other config's
/// report row shows its Δ% against this, so "is this tune actually better
/// than not tuning?" is answered inside every sweep, never by
/// cross-referencing older runs (absolute numbers don't compare across
/// sessions).
fn with_defaults_baseline(mut points: Vec<ConfigPoint>) -> Vec<ConfigPoint> {
    if !points.iter().any(|p| p.is_empty()) {
        points.insert(0, ConfigPoint::new());
    }
    points
}

/// Render a config point as the flat `key=value` file `tune.rs` parses.
fn tune_file_body(point: &ConfigPoint) -> String {
    let mut body = String::from("# generated by tune_sweep\n");
    for (k, v) in point {
        body.push_str(&format!("{k}={v}\n"));
    }
    body
}

/// Short human-readable label for one config point.
fn label_for(point: &ConfigPoint) -> String {
    if point.is_empty() {
        return "defaults".to_string();
    }
    point
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Run one `latency_profile` child and parse its report.
fn run_child(
    tune_path: &std::path::Path,
    workload: &str,
    mode: &str,
    ops: usize,
    out_json: &std::path::Path,
) -> Option<Metrics> {
    // The sweep and the profiler are built from the same workspace/profile,
    // so the sibling binary sits next to our own executable.
    let profiler = env::current_exe().ok()?.with_file_name("latency_profile");
    let status = Command::new(&profiler)
        .env("LOHALLOC_TUNE", tune_path)
        .args([
            "--workload",
            workload,
            "--mode",
            mode,
            "--ops",
            &ops.to_string(),
        ])
        .arg("--out")
        .arg(out_json)
        .status()
        .map_err(|e| eprintln!("tune_sweep: failed to spawn {profiler:?}: {e}"))
        .ok()?;
    if !status.success() {
        eprintln!("tune_sweep: {workload}/{mode} exited with {status}");
        return None;
    }
    let report: LatencyReport = serde_json::from_str(&fs::read_to_string(out_json).ok()?).ok()?;
    Some(Metrics {
        p50: report.alloc_p50_ns,
        p99: report.alloc_p99_ns,
        mean: report.alloc_mean_ns,
        mops: report.measured_mops,
        rss_bytes: report.peak_rss_bytes,
    })
}

fn fmt_metrics(m: Option<Metrics>) -> String {
    match m {
        Some(m) => format!(
            "p50={}ns p99={}ns mean={:.0}ns thpt={:.2}Mops/s rss={:.1}MiB",
            m.p50,
            m.p99,
            m.mean,
            m.mops,
            m.rss_bytes as f64 / (1024.0 * 1024.0),
        ),
        None => "(failed)".to_string(),
    }
}

/// `Δ% vs defaults` for one metric: negative = better for
/// smaller-is-better metrics (p99), positive = better for throughput.
fn delta_pct(value: f64, baseline: f64) -> Option<f64> {
    (baseline > 0.0).then(|| (value - baseline) / baseline * 100.0)
}

/// `(mean_ns, stddev_ns)` from one hyperfine export.
#[derive(Clone, Copy, Serialize)]
struct NativeTiming {
    mean_ns: f64,
    stddev_ns: f64,
}

#[derive(Serialize)]
struct NativeRunResult {
    label: String,
    focus: String,
    workload: String,
    training: Option<NativeTiming>,
    inference: Option<NativeTiming>,
}

#[derive(Deserialize)]
struct HyperfineExport {
    results: Vec<HyperfineTiming>,
}

#[derive(Deserialize)]
struct HyperfineTiming {
    mean: f64,
    #[serde(default)]
    stddev: Option<f64>,
}

fn read_native_timing(path: &std::path::Path) -> Option<NativeTiming> {
    let export: HyperfineExport = serde_json::from_str(&fs::read_to_string(path).ok()?).ok()?;
    let r = export.results.first()?;
    Some(NativeTiming {
        mean_ns: r.mean * 1e9,
        stddev_ns: r.stddev.unwrap_or(0.0) * 1e9,
    })
}

/// Drive `bench/run_native.sh`'s lohalloc triple for one config point —
/// the production-path (LD_PRELOAD via lohalloc-cabi) ablation. LD_PRELOAD
/// interposition is Linux-only, so on any other host the script runs inside
/// the prebuilt `lohalloc-bench` Docker image (`make bench-image`), with the
/// sweep's out dir mounted at /sweep; on Linux it runs directly. Returns
/// the per-workload raw dir on success.
fn run_native_point(
    point_idx: usize,
    out_dir: &std::path::Path,
    workloads: &[String],
    native_ops: usize,
) -> Option<PathBuf> {
    let out_abs = fs::canonicalize(out_dir).ok()?;
    let raw_rel = format!("native-raw-{point_idx}");
    let workloads_str = workloads.join(" ");
    let status = if cfg!(target_os = "linux") {
        let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        Command::new("bash")
            .current_dir(&repo_root)
            .arg("bench/run_native.sh")
            .env("RAW_DIR", out_abs.join(&raw_rel))
            .env(
                "TUNE_FILE",
                out_abs.join(format!("config-{point_idx}.tune")),
            )
            .env("WORKLOADS", &workloads_str)
            .env("LANGS", "c")
            .env("ONLY_ALLOCATORS", "lohalloc")
            .env("OPS", native_ops.to_string())
            .status()
    } else {
        Command::new("docker")
            .args(["run", "--rm"])
            .args(["-e", &format!("RAW_DIR=/sweep/{raw_rel}")])
            .args(["-e", &format!("TUNE_FILE=/sweep/config-{point_idx}.tune")])
            .args(["-e", &format!("WORKLOADS={workloads_str}")])
            .args(["-e", "LANGS=c"])
            .args(["-e", "ONLY_ALLOCATORS=lohalloc"])
            .args(["-e", &format!("OPS={native_ops}")])
            .args(["-v", &format!("{}:/sweep", out_abs.display())])
            .args(["--entrypoint", "bash"])
            .args(["lohalloc-bench", "bench/run_native.sh"])
            .status()
    };
    match status {
        Ok(s) if s.success() => Some(out_abs.join(raw_rel)),
        Ok(s) => {
            eprintln!("tune_sweep: native point {point_idx} exited with {s}");
            None
        }
        Err(e) => {
            eprintln!(
                "tune_sweep: cannot run native point {point_idx} ({e}) — \
                 is Docker running / the lohalloc-bench image built (make bench-image)?"
            );
            None
        }
    }
}

fn fmt_native(m: Option<NativeTiming>) -> String {
    match m {
        Some(t) => format!("{:.2}ms ± {:.2}ms", t.mean_ns / 1e6, t.stddev_ns / 1e6),
        None => "(failed)".to_string(),
    }
}

fn main() {
    let args = parse_args();
    let (workloads, ops, out_dir) = (&args.workloads, args.ops, &args.out);

    let grid_text = match fs::read_to_string(&args.grid) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("tune_sweep: cannot read grid {:?}: {e}", args.grid);
            std::process::exit(2);
        }
    };
    let grid: serde_json::Map<String, serde_json::Value> = match serde_json::from_str(&grid_text) {
        Ok(g) => g,
        Err(e) => {
            eprintln!(
                "tune_sweep: grid {:?} is not a JSON object of arrays: {e}",
                args.grid
            );
            std::process::exit(2);
        }
    };

    let points = with_defaults_baseline(expand_grid(&grid));
    if let Err(e) = fs::create_dir_all(out_dir) {
        eprintln!("tune_sweep: cannot create {out_dir:?}: {e}");
        std::process::exit(2);
    }
    eprintln!(
        "tune_sweep: {} config points x {} workloads x 2 modes (ops={ops}{})",
        points.len(),
        workloads.len(),
        if args.native {
            ", + native triples"
        } else {
            ""
        }
    );

    let mut results: Vec<RunResult> = Vec::new();
    let mut native_results: Vec<NativeRunResult> = Vec::new();
    for (i, point) in points.iter().enumerate() {
        let label = label_for(point);
        let focus = point
            .get("focus")
            .cloned()
            .unwrap_or_else(|| "latency".to_string());
        let tune_path = out_dir.join(format!("config-{i}.tune"));
        if let Err(e) = fs::write(&tune_path, tune_file_body(point)) {
            eprintln!("tune_sweep: cannot write {tune_path:?}: {e}");
            continue;
        }
        for workload in &workloads[..] {
            eprintln!("==> [{}/{}] {label} / {workload}", i + 1, points.len());
            let json = |mode: &str| out_dir.join(format!("raw-{i}-{workload}-{mode}.json"));
            let training = run_child(&tune_path, workload, "training", ops, &json("training"));
            let inference = run_child(&tune_path, workload, "inference", ops, &json("inference"));
            results.push(RunResult {
                label: label.clone(),
                focus: focus.clone(),
                workload: workload.clone(),
                training,
                inference,
            });
        }
        if args.native {
            eprintln!("==> [{}/{}] {label} / native triple", i + 1, points.len());
            let raw = run_native_point(i, out_dir, workloads, args.native_ops);
            for workload in &workloads[..] {
                let timing = |mode: &str| {
                    raw.as_ref().and_then(|dir| {
                        read_native_timing(
                            &dir.join(format!("native-c-lohalloc-{workload}-{mode}.json")),
                        )
                    })
                };
                native_results.push(NativeRunResult {
                    label: label.clone(),
                    focus: focus.clone(),
                    workload: workload.clone(),
                    training: timing("training"),
                    inference: timing("inference"),
                });
            }
        }
    }

    // Per-workload defaults (inference) metrics — the Δ% baseline for
    // every other row, across all focus groups.
    let defaults_by_workload: BTreeMap<&str, Metrics> = results
        .iter()
        .filter(|r| r.is_defaults())
        .filter_map(|r| Some((r.workload.as_str(), r.inference?)))
        .collect();

    // Ranked report: one section per focus group, ranked per workload by
    // the metric that focus claims to optimize.
    let mut report = String::from("# tune_sweep report\n\n");
    report.push_str(
        "Δ columns compare each config's inference run against the same-sweep \
         `defaults` baseline (negative p99 Δ / positive throughput Δ = better \
         than not tuning).\n\n",
    );
    let mut focuses: Vec<String> = results.iter().map(|r| r.focus.clone()).collect();
    focuses.sort();
    focuses.dedup();
    for focus in &focuses {
        report.push_str(&format!("## focus group: {focus}\n\n"));
        for workload in &workloads[..] {
            let mut group: Vec<&RunResult> = results
                .iter()
                .filter(|r| &r.focus == focus && &r.workload == workload)
                .collect();
            if group.is_empty() {
                continue;
            }
            // throughput groups rank by measured inference Mops/s desc;
            // everything else by inference p99 asc. Failed runs sink.
            group.sort_by(|a, b| match (a.inference, b.inference) {
                (Some(ma), Some(mb)) => {
                    if focus == "throughput" {
                        mb.mops
                            .partial_cmp(&ma.mops)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    } else {
                        ma.p99.cmp(&mb.p99)
                    }
                }
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => std::cmp::Ordering::Equal,
            });
            report.push_str(&format!("### {workload}\n\n"));
            let baseline = defaults_by_workload.get(workload.as_str());
            for (rank, r) in group.iter().enumerate() {
                let delta = match (r.inference, baseline) {
                    (Some(m), Some(base)) if !r.is_defaults() => {
                        let p99_d = delta_pct(m.p99 as f64, base.p99 as f64);
                        let mops_d = delta_pct(m.mops, base.mops);
                        match (p99_d, mops_d) {
                            (Some(p), Some(t)) => {
                                format!(" (vs defaults: p99 {p:+.1}%, thpt {t:+.1}%)")
                            }
                            (Some(p), None) => format!(" (vs defaults: p99 {p:+.1}%)"),
                            _ => String::new(),
                        }
                    }
                    _ => String::new(),
                };
                report.push_str(&format!(
                    "{}. `{}`{delta}\n   - inference: {}\n   - training:  {}\n",
                    rank + 1,
                    r.label,
                    fmt_metrics(r.inference),
                    fmt_metrics(r.training),
                ));
            }
            report.push('\n');
        }
    }

    // Native (LD_PRELOAD production-path) section: wall time is the metric
    // for *every* focus here — rank by inference mean ascending, Δ% vs the
    // defaults point's inference mean.
    if !native_results.is_empty() {
        let native_defaults: BTreeMap<&str, NativeTiming> = native_results
            .iter()
            .filter(|r| r.label == "defaults")
            .filter_map(|r| Some((r.workload.as_str(), r.inference?)))
            .collect();
        report.push_str("## native LD_PRELOAD triples (C bench_main via lohalloc-cabi)\n\n");
        report.push_str(
            "Whole-process hyperfine wall time (the production interposition \
             path — full tune config applies, reward shaping included). \
             Ranked by inference mean regardless of focus.\n\n",
        );
        for workload in &workloads[..] {
            let mut group: Vec<&NativeRunResult> = native_results
                .iter()
                .filter(|r| &r.workload == workload)
                .collect();
            if group.is_empty() {
                continue;
            }
            group.sort_by(|a, b| match (a.inference, b.inference) {
                (Some(ma), Some(mb)) => ma
                    .mean_ns
                    .partial_cmp(&mb.mean_ns)
                    .unwrap_or(std::cmp::Ordering::Equal),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => std::cmp::Ordering::Equal,
            });
            report.push_str(&format!("### {workload}\n\n"));
            let baseline = native_defaults.get(workload.as_str());
            for (rank, r) in group.iter().enumerate() {
                let delta = match (r.inference, baseline) {
                    (Some(m), Some(base)) if r.label != "defaults" => {
                        delta_pct(m.mean_ns, base.mean_ns)
                            .map(|d| format!(" (vs defaults: mean {d:+.1}%)"))
                            .unwrap_or_default()
                    }
                    _ => String::new(),
                };
                report.push_str(&format!(
                    "{}. `{}`{delta}\n   - inference: {}\n   - training:  {}\n",
                    rank + 1,
                    r.label,
                    fmt_native(r.inference),
                    fmt_native(r.training),
                ));
            }
            report.push('\n');
        }
    }

    let report_path = out_dir.join("tune-report.md");
    if let Err(e) = fs::write(&report_path, &report) {
        eprintln!("tune_sweep: cannot write {report_path:?}: {e}");
        std::process::exit(2);
    }
    // Structured twin of the markdown. plot_tune.py reads the in-process
    // list under "inprocess" (and tolerates the pre-C4 bare-list shape).
    let json_path = out_dir.join("tune-report.json");
    let combined = serde_json::json!({
        "inprocess": results,
        "native": native_results,
    });
    match serde_json::to_string_pretty(&combined) {
        Ok(json) => {
            if let Err(e) = fs::write(&json_path, json) {
                eprintln!("tune_sweep: cannot write {json_path:?}: {e}");
            }
        }
        Err(e) => eprintln!("tune_sweep: cannot serialize results: {e}"),
    }
    println!("{report}");
    println!("wrote {}", report_path.display());
    generate_graphs(&json_path, &out_dir.join("graphs"));
}

/// Best-effort graph rendering via the shared venv wrapper — mirrors
/// `aggregate::generate_graphs`, but hands generate.sh the tune plotter.
/// Never fatal; `LOHALLOC_NO_GRAPHS=1` skips (CI images without python3).
fn generate_graphs(report_json: &std::path::Path, graphs_dir: &std::path::Path) {
    if env::var_os("LOHALLOC_NO_GRAPHS").is_some() {
        eprintln!("LOHALLOC_NO_GRAPHS set — skipping graph generation");
        return;
    }
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/graphs/generate.sh")
        .canonicalize();
    let Ok(script) = script else {
        eprintln!("graph script not found — skipping graphs");
        return;
    };
    match Command::new("bash")
        .arg(&script)
        .arg(report_json)
        .arg(graphs_dir)
        .arg("plot_tune.py")
        .status()
    {
        Ok(s) if s.success() => println!("graphs written to {graphs_dir:?}"),
        Ok(s) => eprintln!("graph generation exited with {s} — report is still complete"),
        Err(e) => eprintln!("could not run graph generator ({e}) — report is still complete"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(json: &str) -> serde_json::Map<String, serde_json::Value> {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn expand_grid_is_cartesian() {
        let points = expand_grid(&grid(
            r#"{"focus":["latency","throughput"],"ucb_c":[1.0,2.0]}"#,
        ));
        assert_eq!(points.len(), 4);
        // Every combination appears exactly once.
        let labels: Vec<String> = points.iter().map(label_for).collect();
        assert!(labels.contains(&"focus=latency ucb_c=1.0".to_string()));
        assert!(labels.contains(&"focus=throughput ucb_c=2.0".to_string()));
    }

    #[test]
    fn expand_empty_grid_is_single_defaults_point() {
        let points = expand_grid(&grid("{}"));
        assert_eq!(points.len(), 1);
        assert_eq!(label_for(&points[0]), "defaults");
    }

    #[test]
    fn defaults_baseline_is_always_injected_once() {
        // Non-empty grid without an empty point: defaults prepended.
        let points = with_defaults_baseline(expand_grid(&grid(r#"{"ucb_c":[1.0,2.0]}"#)));
        assert_eq!(points.len(), 3);
        assert!(points[0].is_empty(), "defaults must be point 0");
        // Empty grid already IS the defaults point: not duplicated.
        let points = with_defaults_baseline(expand_grid(&grid("{}")));
        assert_eq!(points.len(), 1);
        assert!(points[0].is_empty());
    }

    #[test]
    fn delta_pct_math_and_guard() {
        // 120 vs 100 baseline = +20%.
        assert!((delta_pct(120.0, 100.0).unwrap() - 20.0).abs() < 1e-9);
        // 80 vs 100 = -20%.
        assert!((delta_pct(80.0, 100.0).unwrap() + 20.0).abs() < 1e-9);
        // Zero/absent baseline (old child without measured_mops): no delta.
        assert_eq!(delta_pct(80.0, 0.0), None);
    }

    #[test]
    fn latency_report_parses_without_new_fields() {
        // Reports from pre-C2 latency_profile builds keep working.
        let old = r#"{"alloc_p50_ns":10,"alloc_p99_ns":20,"alloc_mean_ns":12.0}"#;
        let r: LatencyReport = serde_json::from_str(old).unwrap();
        assert_eq!(r.measured_mops, 0.0);
        assert_eq!(r.peak_rss_bytes, 0);
    }

    #[test]
    fn tune_file_body_is_flat_key_value() {
        let points = expand_grid(&grid(r#"{"focus":["throughput"],"frag_weight":[0.1]}"#));
        let body = tune_file_body(&points[0]);
        assert!(body.contains("focus=throughput\n"));
        assert!(body.contains("frag_weight=0.1\n"));
        assert!(
            !body.contains('"'),
            "no JSON quoting may leak into the tune file"
        );
    }
}
