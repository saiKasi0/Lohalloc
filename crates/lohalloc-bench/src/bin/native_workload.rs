//! Native-matrix Rust workload driver — the Rust counterpart of
//! `bench/native/bench_main.c`, with the exact same CLI contract so
//! `bench/run_native.sh` can drive all three languages uniformly:
//!
//! ```sh
//! native_workload <slab|arena|buddy|system|adv-mixed> [ops]   # ops default 50000
//! native_workload <mt-slab-tN|mt-mixed-tN|mt-xfree-tN> [ops]  # N threads
//! ```
//!
//! Unlike C/C++ (which swap allocators via `LD_PRELOAD`), Rust selects its
//! allocator at build time via this crate's mutually-exclusive `alloc-*`
//! features (`src/global_alloc.rs`) — the idiomatic way a Rust user adopts
//! an allocator. The bench script therefore builds this bin once per
//! feature and picks the matching binary per matrix row, with no preload.
//!
//! Under `alloc-lohalloc` the bin honors the same env-var contract as
//! `lohalloc-cabi`, so the script's training / train+export / inference
//! triple works identically across languages:
//!
//! - `LOHALLOC_FREEZE_AFTER=<n>` — freeze after n *workload* allocations
//!   (counted here at the driver level, not per process malloc like the
//!   cabi — close enough for the same intent, and deterministic).
//! - `LOHALLOC_EXPORT_MODEL=<path>` — after the workload (freezing first if
//!   still training), write the `.lohalloc` model to `<path>`.
//! - `LOHALLOC_MODEL=<path>` — load a model before the workload runs; the
//!   whole run is pure Inference.
//! - `LOHALLOC_DEBUG` — print the frozen-table miss count at the end.

#[cfg(feature = "alloc-lohalloc")]
use std::alloc::Layout;
use std::process::ExitCode;

use lohalloc_bench::workloads::{self, AllocDriver, GlobalDriver};

const DEFAULT_OPS: usize = 50_000;

/// Parses "mt-<kind>-t<N>" into (`kind`, thread count). Splits on the last
/// '-' so a `kind` could itself contain hyphens (not currently needed).
/// Mirrors `bench/native/workloads.c`'s `parse_mt_workload` exactly, so the
/// same workload names work identically across languages.
fn parse_mt_workload(workload: &str) -> Option<(&str, usize)> {
    let rest = workload.strip_prefix("mt-")?;
    let (kind, t_part) = rest.rsplit_once('-')?;
    let threads: usize = t_part.strip_prefix('t')?.parse().ok()?;
    (!kind.is_empty() && threads > 0).then_some((kind, threads))
}

/// Dispatch one workload run, mirroring `bench_main.c`'s scaling rules
/// (arena → ops/500 bursts of 500; system → ops/20 large pairs). The hash
/// argument is ignored by `GlobalDriver`-style drivers — under
/// `alloc-lohalloc` the real stack walker provides call-site identity.
fn run<D: AllocDriver + Sync>(driver: &D, workload: &str, ops: usize) -> bool {
    if let Some((kind, threads)) = parse_mt_workload(workload) {
        match kind {
            "slab" => workloads::workload_mt_slab_churn(driver, threads, ops),
            "mixed" => workloads::workload_mt_adversarial_mixed(driver, threads, ops),
            "xfree" => workloads::workload_mt_xfree(driver, threads, ops),
            _ => return false,
        }
        return true;
    }
    match workload {
        "slab" => workloads::workload_slab_churn(driver, 0, ops),
        "arena" => workloads::workload_arena_bursts(driver, 0, (ops / 500).max(1), 500),
        "buddy" => workloads::workload_buddy_interleaved(driver, 0, ops),
        "system" => workloads::workload_system_large(driver, 0, (ops / 20).max(1)),
        "adv-mixed" => workloads::workload_adversarial_mixed(driver, 0, ops),
        _ => return false,
    }
    true
}

/// Sentinel `remaining` value meaning "no freeze scheduled" (already
/// frozen, or `LOHALLOC_FREEZE_AFTER` was unset) — chosen so a fresh driver
/// with freezing disabled never even attempts the `fetch_update` CAS loop
/// below once it settles.
#[cfg(feature = "alloc-lohalloc")]
const FREEZE_DISABLED: u64 = u64::MAX;

/// How often (in workload allocations) `freeze_mode=converged` polls
/// `Lohalloc::is_converged()` — same cadence rationale as `lohalloc-cabi`'s
/// `CONVERGENCE_POLL_EVERY` (the poll takes the state Mutex and walks every
/// Signature, so it must not run per-op).
#[cfg(feature = "alloc-lohalloc")]
const CONVERGENCE_POLL_EVERY: u64 = 256;

/// Driver wrapper that freezes the global Lohalloc after N workload
/// allocations (`LOHALLOC_FREEZE_AFTER` semantics) and/or at detected
/// bandit convergence (`freeze_mode=converged`, polled every
/// `CONVERGENCE_POLL_EVERY` allocations) — for `#[global_allocator]`
/// builds. `remaining` is an `AtomicU64` (not the original
/// `Cell<Option<u64>>`) because the mt-* workloads call `alloc`
/// concurrently from multiple threads; a `Cell` is `!Sync` and, worse,
/// a plain load-then-store race here could let two threads both observe
/// "1 remaining" and both call `freeze_global_lohalloc()` (which panics on
/// a double-freeze). `fetch_update` makes "decrement, and tell me if I was
/// the one who hit zero" a single atomic operation. With both triggers
/// live (converged mode with an op-count hard cap) either can fire first,
/// so the actual freeze call is guarded by the `froze` swap — exactly one
/// thread on exactly one path ever calls it.
#[cfg(feature = "alloc-lohalloc")]
struct FreezeAfterDriver {
    inner: GlobalDriver,
    remaining: core::sync::atomic::AtomicU64,
    converged_mode: bool,
    polls: core::sync::atomic::AtomicU64,
    froze: core::sync::atomic::AtomicBool,
}

#[cfg(feature = "alloc-lohalloc")]
impl FreezeAfterDriver {
    fn freeze_once(&self) {
        use core::sync::atomic::Ordering;
        if !self.froze.swap(true, Ordering::AcqRel) {
            lohalloc_bench::global_alloc::freeze_global_lohalloc();
        }
    }
}

#[cfg(feature = "alloc-lohalloc")]
impl AllocDriver for FreezeAfterDriver {
    unsafe fn alloc(&self, layout: Layout, hash: u64) -> *mut u8 {
        use core::sync::atomic::Ordering;
        let ptr = unsafe { self.inner.alloc(layout, hash) };
        let prev = self
            .remaining
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                if n == FREEZE_DISABLED || n == 0 {
                    None // Already disabled/frozen — no-op, no CAS retry storm.
                } else {
                    Some(n - 1)
                }
            });
        if prev == Ok(1) {
            // `prev` is the value *before* this thread's decrement — 1 means
            // this thread (and only this thread; the CAS guarantees
            // uniqueness) just took the count from 1 to 0.
            self.freeze_once();
        } else if self.converged_mode && !self.froze.load(Ordering::Acquire) {
            let n = self.polls.fetch_add(1, Ordering::Relaxed) + 1;
            if n % CONVERGENCE_POLL_EVERY == 0
                && lohalloc_bench::global_alloc::global_lohalloc_is_converged()
            {
                self.freeze_once();
            }
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout, hash: u64) {
        unsafe { self.inner.dealloc(ptr, layout, hash) };
    }
}

#[cfg(feature = "alloc-lohalloc")]
fn run_lohalloc(workload: &str, ops: usize) -> bool {
    use lohalloc_bench::global_alloc as ga;

    // Model load first: a loaded model means the whole run is Inference and
    // any FREEZE_AFTER is ignored (freezing an already-frozen allocator
    // panics by design).
    let mut model_loaded = false;
    if let Ok(path) = std::env::var("LOHALLOC_MODEL") {
        if !path.is_empty() {
            match std::fs::read(&path) {
                Ok(bytes) if ga::load_global_lohalloc(&bytes) => model_loaded = true,
                Ok(_) => eprintln!(
                    "native_workload: LOHALLOC_MODEL={path} is malformed — staying in training"
                ),
                Err(e) => eprintln!("native_workload: could not read LOHALLOC_MODEL={path}: {e}"),
            }
        }
    }

    // Read the tune config for our OWN freeze-policy decision. We use the
    // *uncached* loader, not `config()`/`load_from_env()`: because Lohalloc
    // is this binary's `#[global_allocator]`, the language runtime already
    // allocated (and thus locked `tune::config()` to defaults) before main
    // ran — so the freeze knobs below must come from a fresh env/file read,
    // not the poisoned OnceLock. (The reward-shaping knobs genuinely cannot
    // apply in a global-allocator build for the same reason; that's the
    // documented limitation — the sweep tunes via `latency_profile`'s
    // private instance, and production tuning is the cabi/LD_PRELOAD path
    // where the config loads before the first allocation.)
    let cfg = lohalloc_alloc::tune::load_config_uncached();

    let freeze_after = if model_loaded {
        None
    } else {
        std::env::var("LOHALLOC_FREEZE_AFTER")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .filter(|&n| n > 0)
            .or(cfg.freeze_after)
    };
    let converged_mode =
        !model_loaded && cfg.freeze_mode == lohalloc_alloc::tune::FreezeMode::Converged;

    let driver = FreezeAfterDriver {
        inner: GlobalDriver,
        remaining: core::sync::atomic::AtomicU64::new(freeze_after.unwrap_or(FREEZE_DISABLED)),
        converged_mode,
        polls: core::sync::atomic::AtomicU64::new(0),
        froze: core::sync::atomic::AtomicBool::new(false),
    };
    let ok = run(&driver, workload, ops);
    // Capture whether the run itself ended frozen — BEFORE the export block
    // below, which freezes on demand and would otherwise mask a
    // convergence/op-count freeze that fired mid-run.
    let froze_during_run = ga::global_lohalloc_is_inference();

    if let Ok(path) = std::env::var("LOHALLOC_EXPORT_MODEL") {
        if !path.is_empty() {
            if !ga::global_lohalloc_is_inference() {
                ga::freeze_global_lohalloc();
            }
            match ga::export_global_lohalloc() {
                Some(bytes) => {
                    let tmp = format!("{path}.tmp");
                    if let Err(e) =
                        std::fs::write(&tmp, &bytes).and_then(|()| std::fs::rename(&tmp, &path))
                    {
                        eprintln!("native_workload: failed to write model to {path}: {e}");
                    }
                }
                None => eprintln!("native_workload: export requested but allocator not frozen"),
            }
        }
    }

    if std::env::var("LOHALLOC_DEBUG").is_ok() {
        eprintln!(
            "native_workload: pht_misses={} (model_loaded={model_loaded} froze_during_run={froze_during_run})",
            ga::global_lohalloc_pht_misses()
        );
    }
    ok
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let Some(workload) = args.get(1) else {
        eprintln!(
            "usage: {} <slab|arena|buddy|system|adv-mixed|mt-slab-tN|mt-mixed-tN|mt-xfree-tN> [ops]",
            args.first()
                .map(String::as_str)
                .unwrap_or("native_workload")
        );
        return ExitCode::from(2);
    };
    let ops = args
        .get(2)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_OPS);

    #[cfg(feature = "alloc-lohalloc")]
    let ok = run_lohalloc(workload, ops);
    #[cfg(not(feature = "alloc-lohalloc"))]
    let ok = run(&GlobalDriver, workload, ops);

    if ok {
        ExitCode::SUCCESS
    } else {
        eprintln!("unknown workload '{workload}'");
        ExitCode::from(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_well_formed_mt_names() {
        assert_eq!(parse_mt_workload("mt-slab-t4"), Some(("slab", 4)));
        assert_eq!(parse_mt_workload("mt-mixed-t1"), Some(("mixed", 1)));
        assert_eq!(parse_mt_workload("mt-xfree-t16"), Some(("xfree", 16)));
    }

    #[test]
    fn rejects_non_mt_and_malformed_names() {
        assert_eq!(parse_mt_workload("slab"), None);
        assert_eq!(parse_mt_workload("adv-mixed"), None);
        assert_eq!(parse_mt_workload("mt-slab"), None); // no "-tN" suffix
        assert_eq!(parse_mt_workload("mt-slab-t0"), None); // zero threads
        assert_eq!(parse_mt_workload("mt-slab-tX"), None); // non-numeric
        assert_eq!(parse_mt_workload("mt--t4"), None); // empty kind
    }
}
