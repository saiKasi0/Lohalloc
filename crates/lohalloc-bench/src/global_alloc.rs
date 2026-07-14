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

/// Ladder 6 pin-cache counters `(misses, hits, negative)`. Misses are
/// always counted (cold path); hits/negative read `0` unless
/// `lohalloc-alloc` was built with `route-metrics`.
#[cfg(feature = "alloc-lohalloc")]
pub fn global_lohalloc_pin_counters() -> (u64, u64, u64) {
    (
        lohalloc_alloc::Lohalloc::pin_miss_count(),
        lohalloc_alloc::Lohalloc::pin_hit_count(),
        lohalloc_alloc::Lohalloc::pin_negative_count(),
    )
}

/// Frozen-path per-backend service count (0 without `route-metrics`).
#[cfg(feature = "alloc-lohalloc")]
pub fn global_lohalloc_route_count(backend: lohalloc_core::Backend) -> u64 {
    lohalloc_alloc::Lohalloc::route_count(backend)
}

/// Frozen-path fallthrough count (0 without `route-metrics`).
#[cfg(feature = "alloc-lohalloc")]
pub fn global_lohalloc_fallthroughs() -> u64 {
    lohalloc_alloc::Lohalloc::fallthrough_count()
}

/// J5-bisect slab central-refill counters
/// `(central_refills, sibling_steps, sibling_hits)` — the sibling-scan
/// instrumentation (all 0 without `route-metrics`).
#[cfg(feature = "alloc-lohalloc")]
pub fn global_lohalloc_slab_refill_counts() -> (u64, u64, u64) {
    lohalloc_alloc::Lohalloc::slab_refill_counts()
}

/// The latched active stripe count (reflects `LOHALLOC_STRIPES` when set).
#[cfg(feature = "alloc-lohalloc")]
pub fn global_lohalloc_active_stripes() -> usize {
    lohalloc_alloc::Lohalloc::active_stripes()
}

/// J8-B firing-rate probe: `(fast_lane_alloc_hits, fast_lane_free_hits)`
/// (0 without `route-metrics`).
#[cfg(feature = "alloc-lohalloc")]
pub fn global_lohalloc_fast_lane_counts() -> (u64, u64) {
    (
        lohalloc_alloc::Lohalloc::fast_lane_count(),
        lohalloc_alloc::Lohalloc::fast_lane_free_count(),
    )
}

/// J8-A diagnostics: per-chunk `(used_bytes, carved, freed, spans)`.
#[cfg(feature = "alloc-lohalloc")]
pub fn global_lohalloc_arena_chunk_debug() -> Vec<(usize, usize, usize, usize)> {
    ALLOC.arena_chunk_debug()
}

/// J8-A diagnostics: arena recycle-scan outcomes
/// `(recycles, grows, skip_pinned, skip_unfreed, skip_uncarved)` — the
/// rotation-breadth probe (why did the arena map fresh chunks instead of
/// recycling quiescent ones?).
#[cfg(feature = "alloc-lohalloc")]
pub fn global_lohalloc_arena_recycle_stats() -> (usize, usize, usize, usize, usize) {
    let s = ALLOC.arena_recycle_stats();
    (
        s.recycles,
        s.grows,
        s.skip_pinned,
        s.skip_unfreed,
        s.skip_uncarved,
    )
}

/// Live mapped-footprint counters `(slab_regions, buddy_regions, arena_used,
/// arena_capacity)` — the RSS-attribution view (J8 retention diagnostics):
/// which backend is holding how much backing memory at the point of the call.
#[cfg(feature = "alloc-lohalloc")]
pub fn global_lohalloc_backend_counters() -> (usize, usize, usize, usize) {
    let c = ALLOC.backend_counters();
    (
        c.slab_region_count,
        c.buddy_region_count,
        c.arena_used,
        c.arena_capacity,
    )
}

/// Whether the global instance latched each backend's header-free fast path
/// on `load()` `(slab, buddy, arena)`. Diagnostic for the touch-cost story:
/// these only latch if the backend was untouched at model-load time.
#[cfg(feature = "alloc-lohalloc")]
pub fn global_lohalloc_headerless() -> (bool, bool, bool) {
    (
        ALLOC.is_slab_headerless(),
        ALLOC.is_buddy_headerless(),
        ALLOC.is_arena_headerless(),
    )
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
