//! Per-op latency profiler (hdrhistogram) for Phase 6 hypothesis validation.
//!
//! ```sh
//! cargo run -p lohalloc-bench --bin latency_profile --release -- \
//!     --workload slab --mode forced:slab --ops 100000 --out results/rust_slab_forced-slab.json
//! ```
//!
//! `--mode` is one of `training`, `inference` (brief warm-up then freeze
//! before the measured run), `baseline` (freeze an empty bandit — pure
//! size-based fallback), or `forced:<slab|buddy|system|arena>` (hand-built
//! `.lohalloc` model forcing every allocation at this call site to that
//! backend, regardless of what the size-based fallback would pick).

use std::env;
use std::fs;
use std::path::PathBuf;

use hdrhistogram::Histogram;
use serde::Serialize;

use lohalloc_bench::forced::lohalloc_forced_single;
use lohalloc_bench::workloads::{self, hashes, AllocDriver, HarnessDriver, TimingDriver};
use lohalloc_core::Backend;

#[derive(Serialize)]
struct LatencyReport {
    lang: &'static str,
    workload: String,
    mode: String,
    arch: String,
    ops: usize,
    alloc_p50_ns: u64,
    alloc_p95_ns: u64,
    alloc_p99_ns: u64,
    alloc_p999_ns: u64,
    alloc_mean_ns: f64,
    dealloc_p50_ns: u64,
    dealloc_p99_ns: u64,
    dealloc_mean_ns: f64,
}

struct Args {
    workload: String,
    mode: String,
    ops: usize,
    out: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut workload = None;
    let mut mode = "training".to_string();
    let mut ops = 50_000usize;
    let mut out = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--workload" => workload = args.next(),
            "--mode" => mode = args.next().unwrap_or(mode),
            "--ops" => ops = args.next().and_then(|s| s.parse().ok()).unwrap_or(ops),
            "--out" => out = args.next().map(PathBuf::from),
            other => eprintln!("latency_profile: ignoring unknown arg {other}"),
        }
    }
    Args {
        workload: workload
            .expect("--workload is required (slab|arena|buddy|system|adv-mixed|adv-exhaust)"),
        mode,
        ops,
        out,
    }
}

fn hash_for(workload: &str) -> u64 {
    match workload {
        "slab" => hashes::W_SLAB,
        "arena" => hashes::W_ARENA,
        "buddy" => hashes::W_BUDDY,
        "system" => hashes::W_SYSTEM,
        "adv-mixed" => hashes::W_ADV_MIXED,
        "adv-exhaust" => hashes::W_ADV_EXHAUST,
        other => panic!("unknown --workload {other}"),
    }
}

/// A representative request size for `--mode forced:<backend>`, matching
/// what the workload generator actually allocates — needed to build the
/// right `combine_hash_size_class` key (see `lohalloc_bench::forced`).
fn size_for_workload(workload: &str) -> usize {
    match workload {
        "slab" | "arena" | "adv-exhaust" => workloads::SMALL_FIXED_REQUEST,
        "buddy" => workloads::BUDDY_SIZES[0],
        "system" => workloads::SYSTEM_SIZES[0],
        "adv-mixed" => workloads::SMALL_FIXED_REQUEST,
        other => panic!("unknown --workload {other}"),
    }
}

fn run_workload<D: AllocDriver>(workload: &str, driver: &D, hash: u64, ops: usize) {
    match workload {
        "slab" => workloads::workload_slab_churn(driver, hash, ops),
        "arena" => workloads::workload_arena_bursts(driver, hash, (ops / 500).max(1), 500),
        "buddy" => workloads::workload_buddy_interleaved(driver, hash, ops),
        "system" => workloads::workload_system_large(driver, hash, ops),
        "adv-mixed" => workloads::workload_adversarial_mixed(driver, hash, ops),
        "adv-exhaust" => {
            // Never freed by design (models unbounded long-lived growth);
            // the pointers are dropped (not the memory) when this process
            // exits shortly after printing its report.
            let _ptrs = workloads::workload_exhaust_no_free(driver, hash, ops);
        }
        other => panic!("unknown --workload {other}"),
    }
}

fn percentiles(hist: &Histogram<u64>) -> (u64, u64, u64, u64, f64) {
    (
        hist.value_at_quantile(0.50),
        hist.value_at_quantile(0.95),
        hist.value_at_quantile(0.99),
        hist.value_at_quantile(0.999),
        hist.mean(),
    )
}

fn main() {
    let args = parse_args();
    let hash = hash_for(&args.workload);

    let harness;
    let alloc_samples: Vec<u64>;
    let dealloc_samples: Vec<u64>;

    if let Some(backend_name) = args.mode.strip_prefix("forced:") {
        let backend = match backend_name {
            "slab" => Backend::Slab,
            "buddy" => Backend::Buddy,
            "system" => Backend::System,
            "arena" => Backend::Arena,
            other => panic!("unknown forced backend {other}"),
        };
        harness = HarnessDriver {
            alloc: lohalloc_forced_single(hash, size_for_workload(&args.workload), backend),
        };
        let timing = TimingDriver::new(&harness);
        run_workload(&args.workload, &timing, hash, args.ops);
        alloc_samples = timing.alloc_samples();
        dealloc_samples = timing.dealloc_samples();
    } else {
        harness = HarnessDriver::new();
        match args.mode.as_str() {
            "training" => {}
            "inference" => {
                run_workload(&args.workload, &harness, hash, args.ops.min(500));
                harness.alloc.freeze();
            }
            "baseline" => {
                harness.alloc.freeze(); // empty table -> pure size-based fallback
            }
            other => panic!(
                "unknown --mode {other} (expected training|inference|baseline|forced:<backend>)"
            ),
        }
        let timing = TimingDriver::new(&harness);
        run_workload(&args.workload, &timing, hash, args.ops);
        alloc_samples = timing.alloc_samples();
        dealloc_samples = timing.dealloc_samples();
    }

    let mut alloc_hist = Histogram::<u64>::new_with_bounds(1, 60_000_000_000, 3).unwrap();
    let mut dealloc_hist = Histogram::<u64>::new_with_bounds(1, 60_000_000_000, 3).unwrap();
    for &ns in &alloc_samples {
        let _ = alloc_hist.record(ns.max(1));
    }
    for &ns in &dealloc_samples {
        let _ = dealloc_hist.record(ns.max(1));
    }

    let (a50, a95, a99, a999, amean) = percentiles(&alloc_hist);
    let (d50, _d95, d99, _d999, dmean) = percentiles(&dealloc_hist);

    let report = LatencyReport {
        lang: "rust",
        workload: args.workload.clone(),
        mode: args.mode.clone(),
        arch: env::consts::ARCH.to_string(),
        ops: alloc_samples.len(),
        alloc_p50_ns: a50,
        alloc_p95_ns: a95,
        alloc_p99_ns: a99,
        alloc_p999_ns: a999,
        alloc_mean_ns: amean,
        dealloc_p50_ns: d50,
        dealloc_p99_ns: d99,
        dealloc_mean_ns: dmean,
    };

    let json = serde_json::to_string_pretty(&report).unwrap();
    match args.out {
        Some(path) => {
            if let Some(parent) = path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            fs::write(&path, &json).unwrap_or_else(|e| panic!("failed to write {path:?}: {e}"));
            eprintln!("wrote {}", path.display());
        }
        None => println!("{json}"),
    }
}
