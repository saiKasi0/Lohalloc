//! Native-matrix Rust workload driver — the Rust counterpart of
//! `bench/native/bench_main.c`, with the exact same CLI contract so
//! `bench/run_native.sh` can drive all three languages uniformly:
//!
//! ```sh
//! native_workload <slab|arena|buddy|system|adv-mixed> [ops]   # ops default 50000
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

/// Dispatch one workload run, mirroring `bench_main.c`'s scaling rules
/// (arena → ops/500 bursts of 500; system → ops/20 large pairs). The hash
/// argument is ignored by `GlobalDriver`-style drivers — under
/// `alloc-lohalloc` the real stack walker provides call-site identity.
fn run<D: AllocDriver>(driver: &D, workload: &str, ops: usize) -> bool {
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

/// Driver wrapper that freezes the global Lohalloc after N workload
/// allocations — the `LOHALLOC_FREEZE_AFTER` semantics for
/// `#[global_allocator]` builds (single-threaded by construction: the
/// workloads all run on the main thread).
#[cfg(feature = "alloc-lohalloc")]
struct FreezeAfterDriver {
    inner: GlobalDriver,
    remaining: core::cell::Cell<Option<u64>>,
}

#[cfg(feature = "alloc-lohalloc")]
impl AllocDriver for FreezeAfterDriver {
    unsafe fn alloc(&self, layout: Layout, hash: u64) -> *mut u8 {
        let ptr = unsafe { self.inner.alloc(layout, hash) };
        if let Some(n) = self.remaining.get() {
            if n <= 1 {
                self.remaining.set(None);
                lohalloc_bench::global_alloc::freeze_global_lohalloc();
            } else {
                self.remaining.set(Some(n - 1));
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

    let freeze_after = if model_loaded {
        None
    } else {
        std::env::var("LOHALLOC_FREEZE_AFTER")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .filter(|&n| n > 0)
    };

    let driver = FreezeAfterDriver {
        inner: GlobalDriver,
        remaining: core::cell::Cell::new(freeze_after),
    };
    let ok = run(&driver, workload, ops);

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
            "native_workload: pht_misses={} (model_loaded={model_loaded})",
            ga::global_lohalloc_pht_misses()
        );
    }
    ok
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let Some(workload) = args.get(1) else {
        eprintln!(
            "usage: {} <slab|arena|buddy|system|adv-mixed> [ops]",
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
