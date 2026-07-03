//! The Layer-1 (execution-plane) routing-distribution hypothesis matrix.
//!
//! Each row is a data-driven assertion: force `hash` (at representative
//! request size `size`) to route to `forced_backend`, drive the matching
//! workload generator (see `tests/routing_validation.rs` for the hash →
//! generator mapping) for N allocations, and check that at least
//! `min_fraction` of them were actually served by `expected_backend` (read
//! via `Lohalloc::backend_for_ptr`).
//!
//! `forced_backend != expected_backend` rows test the size-guard fallthrough
//! chain in `Lohalloc::try_backend`/`route_by_size` — e.g. forcing Slab onto
//! a >16 KiB request must fall through to Buddy, never corrupt state or
//! panic.
//!
//! `size` matters beyond picking a valid request: Phase 6's v2 wire format
//! keys the frozen table on `combine_hash_size_class(hash,
//! size_class_for(size))` (see `crate::forced` and
//! `lohalloc_alloc::perfect_hash`'s module doc), so forcing a hash affects
//! only requests whose size falls in the *same* `size_class_for` bucket as
//! `size`. The W-BUDDY workload's four sizes (32 KiB–256 KiB, see
//! `workloads::BUDDY_SIZES`) span two such buckets; only one is explicitly
//! forced here; the other falls back to Inference's size-based default,
//! which happens to agree with `expected_backend` in every row below (Buddy
//! for medium sizes) — so the observed fraction is unaffected either way.

use lohalloc_core::Backend;

use crate::workloads::{hashes, BUDDY_SIZES, SMALL_FIXED_REQUEST, SYSTEM_SIZES};

#[derive(Debug, Clone, Copy)]
pub struct RoutingHypothesis {
    pub id: &'static str,
    pub description: &'static str,
    pub hash: u64,
    pub size: usize,
    pub forced_backend: Backend,
    pub expected_backend: Backend,
    pub min_fraction: f64,
}

pub const ROUTING_HYPOTHESES: &[RoutingHypothesis] = &[
    RoutingHypothesis {
        id: "W-SLAB/forced-slab",
        description: "Forcing Slab on the slab-favorable workload: every allocation actually served by Slab.",
        hash: hashes::W_SLAB,
        size: SMALL_FIXED_REQUEST,
        forced_backend: Backend::Slab,
        expected_backend: Backend::Slab,
        min_fraction: 1.0,
    },
    RoutingHypothesis {
        id: "W-ARENA/forced-arena",
        description: "Forcing Arena on the arena-favorable workload (with per-burst reset): every allocation served by Arena.",
        hash: hashes::W_ARENA,
        size: SMALL_FIXED_REQUEST,
        forced_backend: Backend::Arena,
        expected_backend: Backend::Arena,
        min_fraction: 1.0,
    },
    RoutingHypothesis {
        id: "W-BUDDY/forced-buddy",
        description: "Forcing Buddy on the buddy-favorable workload: every allocation served by Buddy.",
        hash: hashes::W_BUDDY,
        size: BUDDY_SIZES[0],
        forced_backend: Backend::Buddy,
        expected_backend: Backend::Buddy,
        min_fraction: 1.0,
    },
    RoutingHypothesis {
        id: "W-SYSTEM/forced-system",
        description: "Forcing System on the system-favorable workload: every allocation served by System.",
        hash: hashes::W_SYSTEM,
        size: SYSTEM_SIZES[0],
        forced_backend: Backend::System,
        expected_backend: Backend::System,
        min_fraction: 1.0,
    },
    RoutingHypothesis {
        id: "W-BUDDY/forced-slab-falls-through",
        description: "Forcing Slab on the buddy-sized workload: Slab structurally cannot serve >16 KiB, so every allocation falls through to Buddy.",
        hash: hashes::W_BUDDY,
        size: BUDDY_SIZES[0],
        forced_backend: Backend::Slab,
        expected_backend: Backend::Buddy,
        min_fraction: 1.0,
    },
    RoutingHypothesis {
        id: "W-SYSTEM/forced-slab-falls-through",
        description: "Forcing Slab on the system-sized workload: falls through Slab and Buddy (both structurally too small) to System.",
        hash: hashes::W_SYSTEM,
        size: SYSTEM_SIZES[0],
        forced_backend: Backend::Slab,
        expected_backend: Backend::System,
        min_fraction: 1.0,
    },
    RoutingHypothesis {
        id: "W-SYSTEM/forced-arena-falls-through",
        description: "Forcing Arena on the system-sized workload: exceeds the 1 MiB arena capacity, falls through to System.",
        hash: hashes::W_SYSTEM,
        size: SYSTEM_SIZES[0],
        forced_backend: Backend::Arena,
        expected_backend: Backend::System,
        min_fraction: 1.0,
    },
    RoutingHypothesis {
        id: "W-COMBO-SA/slab-site",
        description: "Slab+Arena combo: the slab-favorable site, forced to Slab, is served by Slab.",
        hash: hashes::W_COMBO_SA_SLAB,
        size: SMALL_FIXED_REQUEST,
        forced_backend: Backend::Slab,
        expected_backend: Backend::Slab,
        min_fraction: 1.0,
    },
    RoutingHypothesis {
        id: "W-COMBO-SA/arena-site",
        description: "Slab+Arena combo: the arena-favorable site, forced to Arena, is served by Arena.",
        hash: hashes::W_COMBO_SA_ARENA,
        size: SMALL_FIXED_REQUEST,
        forced_backend: Backend::Arena,
        expected_backend: Backend::Arena,
        min_fraction: 1.0,
    },
    RoutingHypothesis {
        id: "W-COMBO-BA/buddy-site",
        description: "Buddy+small combo: the buddy-favorable site, forced to Buddy, is served by Buddy.",
        hash: hashes::W_COMBO_BA_BUDDY,
        size: BUDDY_SIZES[0],
        forced_backend: Backend::Buddy,
        expected_backend: Backend::Buddy,
        min_fraction: 1.0,
    },
    RoutingHypothesis {
        id: "W-COMBO-BA/small-site",
        description: "Buddy+small combo: the slab-favorable site, forced to Slab, is served by Slab.",
        hash: hashes::W_COMBO_BA_SMALL,
        size: SMALL_FIXED_REQUEST,
        forced_backend: Backend::Slab,
        expected_backend: Backend::Slab,
        min_fraction: 1.0,
    },
];
