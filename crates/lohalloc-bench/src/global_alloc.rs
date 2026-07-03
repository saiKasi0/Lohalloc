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
