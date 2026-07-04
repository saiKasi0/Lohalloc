//! Mutually-exclusive global-allocator selection for the `comparison` bench.
//!
//! Enable exactly one of `alloc-lohalloc` / `alloc-jemalloc` / `alloc-mimalloc`
//! when running the `comparison` bench so it measures a single allocator's
//! effect on the shared workload generators (`crate::workloads`, driven via
//! [`GlobalDriver`](crate::workloads::GlobalDriver)). With no feature enabled,
//! the platform default allocator serves `comparison` (the `alloc-system`
//! baseline) — and every other bench/test target in this crate, which drive
//! a private `Lohalloc` instance directly via [`HarnessDriver`
//! ](crate::workloads::HarnessDriver) and never touch the process's global
//! allocator regardless of these features.

#[cfg(all(feature = "alloc-lohalloc", feature = "alloc-jemalloc"))]
compile_error!("enable only one of alloc-lohalloc / alloc-jemalloc / alloc-mimalloc");
#[cfg(all(feature = "alloc-lohalloc", feature = "alloc-mimalloc"))]
compile_error!("enable only one of alloc-lohalloc / alloc-jemalloc / alloc-mimalloc");
#[cfg(all(feature = "alloc-jemalloc", feature = "alloc-mimalloc"))]
compile_error!("enable only one of alloc-lohalloc / alloc-jemalloc / alloc-mimalloc");

#[cfg(feature = "alloc-lohalloc")]
#[global_allocator]
static ALLOC: lohalloc_alloc::Lohalloc = lohalloc_alloc::Lohalloc::new();

#[cfg(feature = "alloc-jemalloc")]
#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(feature = "alloc-mimalloc")]
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Freeze the process-wide Lohalloc allocator (only meaningful under
/// `alloc-lohalloc`) so `comparison` can also measure Inference-mode
/// overhead as a real `#[global_allocator]`, not just via the harness path.
#[cfg(feature = "alloc-lohalloc")]
pub fn freeze_global_lohalloc() {
    ALLOC.freeze();
}

/// Load a `.lohalloc` model into the process-wide Lohalloc allocator,
/// starting it directly in Inference mode. Mirrors `lohalloc-cabi`'s
/// `LOHALLOC_MODEL` behavior for the `native_workload` bin's
/// `#[global_allocator]` builds.
#[cfg(feature = "alloc-lohalloc")]
pub fn load_global_lohalloc(bytes: &[u8]) -> bool {
    ALLOC.load(bytes)
}

/// Export the process-wide Lohalloc allocator's frozen routing table
/// (`None` if still training — call [`freeze_global_lohalloc`] first).
#[cfg(feature = "alloc-lohalloc")]
pub fn export_global_lohalloc() -> Option<Vec<u8>> {
    ALLOC.export()
}

/// Whether the process-wide Lohalloc allocator is frozen (Inference mode).
#[cfg(feature = "alloc-lohalloc")]
pub fn global_lohalloc_is_inference() -> bool {
    ALLOC.is_inference()
}

/// Whether the process-wide Lohalloc allocator's training has converged —
/// the `freeze_mode=converged` poll (see `Lohalloc::is_converged`).
#[cfg(feature = "alloc-lohalloc")]
pub fn global_lohalloc_is_converged() -> bool {
    ALLOC.is_converged()
}

/// Process-wide count of frozen-table lookup misses — see
/// `Lohalloc::pht_miss_count`. ~0 on a model-loaded run proves the model's
/// (ASLR-normalized) keys matched this process's call sites.
#[cfg(feature = "alloc-lohalloc")]
pub fn global_lohalloc_pht_misses() -> u64 {
    lohalloc_alloc::Lohalloc::pht_miss_count()
}

/// Human-readable label for whichever allocator this build selected. Used to
/// tag criterion baselines and latency-profile output so results are
/// self-describing.
pub fn active_allocator_label() -> &'static str {
    #[cfg(feature = "alloc-lohalloc")]
    {
        return "lohalloc";
    }
    #[cfg(feature = "alloc-jemalloc")]
    {
        return "jemalloc";
    }
    #[cfg(feature = "alloc-mimalloc")]
    {
        return "mimalloc";
    }
    #[cfg(not(any(
        feature = "alloc-lohalloc",
        feature = "alloc-jemalloc",
        feature = "alloc-mimalloc"
    )))]
    {
        "system"
    }
}
