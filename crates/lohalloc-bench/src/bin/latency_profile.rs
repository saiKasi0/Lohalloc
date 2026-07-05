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
    /// Measured wall-clock throughput of the measured phase (million
    /// alloc ops / second): total elapsed around the workload run ÷ alloc
    /// count. Unlike `1e3 / alloc_mean_ns` (per-op timer sums, which
    /// exclude the workload's own bookkeeping between ops) this is what a
    /// throughput-focused tune config should actually be judged by.
    measured_mops: f64,
    /// Peak RSS of this process in bytes (`getrusage`; macOS reports
    /// bytes, Linux KiB — normalized here). The memory-side metric for
    /// judging `frag_weight`/throughput configs: a config that wins
    /// latency by fragmenting pays here.
    peak_rss_bytes: u64,
    /// Measured effective tick of `Instant` on this machine (ns) — see
    /// `lohalloc_bench::clockinfo`. Per-op percentiles below ~3× this are
    /// quantization buckets, not latencies (Apple Silicon: ~42ns).
    clock_tick_ns: u64,
    /// True when any reported alloc percentile sits under 3× the tick —
    /// the aggregator/report must flag those values rather than compare
    /// them. `measured_mops` (outer wall-clock over the whole run) is
    /// unaffected either way.
    quantized: bool,
}

/// Peak RSS in bytes, cross-platform (`ru_maxrss` is KiB on Linux, bytes
/// on macOS). 0 if the syscall fails — the report stays usable.
fn peak_rss_bytes() -> u64 {
    // SAFETY: getrusage writes into the zeroed struct we hand it; no
    // pointers retained.
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut ru) != 0 {
            return 0;
        }
        let raw = ru.ru_maxrss.max(0) as u64;
        if cfg!(target_os = "macos") {
            raw
        } else {
            raw * 1024
        }
    }
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

/// Render the resolved [`TrainingConfig`] in the same flat `key=value`
/// format tune files use — one line per knob, deterministic order.
fn dump_config(cfg: &lohalloc_alloc::tune::TrainingConfig) -> String {
    let freeze_mode = match cfg.freeze_mode {
        lohalloc_alloc::tune::FreezeMode::Ops => "ops",
        lohalloc_alloc::tune::FreezeMode::Converged => "converged",
    };
    let mut out = String::new();
    out.push_str(&format!("ucb_c={}\n", cfg.ucb_c));
    out.push_str(&format!("hysteresis={}\n", cfg.hysteresis));
    for (i, name) in ["slab", "buddy", "system", "arena"].iter().enumerate() {
        out.push_str(&format!("baseline_{name}={}\n", cfg.baseline_rewards[i]));
    }
    out.push_str(&format!("t_ref_ns={}\n", cfg.t_ref_ns));
    out.push_str(&format!("frag_weight={}\n", cfg.frag_weight));
    out.push_str(&format!("freeze_mode={freeze_mode}\n"));
    out.push_str(&format!("converge_stable_n={}\n", cfg.converge_stable_n));
    out.push_str(&format!("reward_batch={}\n", cfg.reward_batch));
    // Omitted when unset so the dump is itself a parseable tune file
    // (tune.rs has no "none" spelling for freeze_after).
    if let Some(v) = cfg.freeze_after {
        out.push_str(&format!("freeze_after={v}\n"));
    }
    out
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
    // Install the tune config (LOHALLOC_TUNE + LOHALLOC_* env overrides) so
    // sweep-driven runs (`tune_sweep`) shape the private instance's training
    // — safe to do plainly in an ordinary binary (no interposition).
    let cfg = lohalloc_alloc::tune::load_from_env();
    // `--dump-config`: print the *resolved* config as the same flat
    // key=value format tune files use, then exit. This is the e2e probe for
    // the defaults -> LOHALLOC_TUNE file -> LOHALLOC_<KEY> env precedence
    // chain across a real process boundary (tests/tune_e2e.rs).
    if env::args().any(|a| a == "--dump-config") {
        print!("{}", dump_config(cfg));
        return;
    }
    let args = parse_args();
    let hash = hash_for(&args.workload);

    let harness;
    let alloc_samples: Vec<u64>;
    let dealloc_samples: Vec<u64>;
    let measured_elapsed: std::time::Duration;

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
        let t0 = std::time::Instant::now();
        run_workload(&args.workload, &timing, hash, args.ops);
        measured_elapsed = t0.elapsed();
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
        let t0 = std::time::Instant::now();
        run_workload(&args.workload, &timing, hash, args.ops);
        measured_elapsed = t0.elapsed();
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

    let clock_tick_ns = lohalloc_bench::clockinfo::instant_tick_ns();
    let quantized = [a50, a95, a99]
        .iter()
        .any(|&v| lohalloc_bench::clockinfo::is_quantized(v, clock_tick_ns));
    if quantized {
        eprintln!(
            "latency_profile: WARNING — clock tick floor is {clock_tick_ns}ns and some \
             percentiles sit below 3 ticks; those values are quantization buckets, not \
             latencies (compare measured_mops instead)"
        );
    }

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
        measured_mops: {
            let secs = measured_elapsed.as_secs_f64();
            if secs > 0.0 {
                alloc_samples.len() as f64 / secs / 1e6
            } else {
                0.0
            }
        },
        peak_rss_bytes: peak_rss_bytes(),
        clock_tick_ns,
        quantized,
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
