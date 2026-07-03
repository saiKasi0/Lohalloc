//! Layer 1 (execution-plane) forced routing.
//!
//! Hand-builds a `.lohalloc` model that routes specific `(call-site hash,
//! request size)` combinations to specific backends, then loads it into a
//! fresh `Lohalloc` instance so it starts directly in Inference mode. This
//! validates each backend on its favorable workload independent of whether
//! the bandit's learning is any good — a prerequisite check before trusting
//! trained-routing hypotheses (see `crate::hypotheses` and
//! `tests/decision_plane.rs`).
//!
//! Since Phase 6's v2 wire format keys the frozen table on
//! `combine_hash_size_class(hash, size_class_for(size))` rather than the
//! raw hash alone (see `lohalloc_alloc::perfect_hash`'s module doc), forcing
//! a hash to a backend requires a representative `size` so the right
//! combined key gets built — every request whose size falls in the same
//! `size_class_for` bucket is affected, not just that exact size.

use lohalloc_alloc::perfect_hash::PerfectHashTable;
use lohalloc_alloc::state::{combine_hash_size_class, size_class_for};
use lohalloc_alloc::Lohalloc;
use lohalloc_core::Backend;

/// Serialize a `.lohalloc` model forcing each `(hash, size, backend)`
/// triple: every allocation at call site `hash` whose request size falls in
/// the same `size_class_for` bucket as `size` is routed to `backend`.
/// Combinations not covered fall back to Inference mode's default
/// size-based routing (see `state::default_backend_for_size`).
pub fn forced_model_bytes(triples: &[(u64, usize, Backend)]) -> Vec<u8> {
    let entries: Vec<(u64, u8, Backend)> = triples
        .iter()
        .map(|&(hash, size, backend)| {
            let size_class = size_class_for(size);
            (
                combine_hash_size_class(hash, size_class),
                size_class,
                backend,
            )
        })
        .collect();
    PerfectHashTable::from_entries(entries).serialize()
}

/// Build a fresh `Lohalloc` instance pre-loaded with a forced-routing model.
/// Panics if the model fails to load (a bug in this helper, not the system
/// under test).
pub fn lohalloc_forced(triples: &[(u64, usize, Backend)]) -> Lohalloc {
    let alloc = Lohalloc::new();
    let bytes = forced_model_bytes(triples);
    assert!(alloc.load(&bytes), "forced model load failed");
    alloc
}

/// Convenience: a model forcing a single `(hash, size)` combination to a
/// single backend.
pub fn lohalloc_forced_single(hash: u64, size: usize, backend: Backend) -> Lohalloc {
    lohalloc_forced(&[(hash, size, backend)])
}
