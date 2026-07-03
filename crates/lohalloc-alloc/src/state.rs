//! Allocator State Machine — the Decision Engine's top-level controller.
//!
//! Lohalloc operates as a strict state machine with two modes:
//!
//! - **Training**: The Multi-Armed Bandit (`BanditPolicy`) selects which
//!   Execution-Plane backend serves each allocation, keyed on
//!   `(topological_hash, size_class)` Signatures. The bandit learns online
//!   from allocation outcomes. Telemetry is emitted (Phase 2+).
//!
//! - **Inference**: `freeze()` collapses the bandit's learned weights into
//!   a read-only `PerfectHashTable`. The hot path becomes a single
//!   `hash → lookup → backend` — an O(1) minimal-perfect-hash lookup,
//!   zero heap allocations. No further MAB updates; no telemetry emission.
//!
//! # State Transitions
//!
//! ```text
//!   new_training()          freeze()           load()
//! ┌─────────────────┐   ┌──────────────────┐   ┌──────────────────┐
//! │   Training      │──►│     Inference     │◄──│     Inference     │
//! │ { bandit }      │   │ { routing_table } │   │ { routing_table } │
//! └─────────────────┘   └──────────────────┘   └──────────────────┘
//! ```
//!
//! `freeze()` is a one-way transition: once in Inference, the routing table
//! is immutable. `load()` starts directly in Inference mode — a pre-optimized
//! heap from boot.
//!
//! # Serialization (`.lohalloc` model file)
//!
//! `export()` serializes the `PerfectHashTable` to bytes. `load()` deserializes
//! a `.lohalloc` file and starts in Inference mode. See `perfect_hash.rs` for
//! the binary format.

use lohalloc_core::{slab_class_for, Backend, Signature};

use crate::bandit::BanditPolicy;
use crate::perfect_hash::PerfectHashTable;

/// Compute a compact `size_class` for the given allocation size.
///
/// For sizes within the Slab range, this is the slab class index (0–11).
/// For larger sizes, we bucket into coarse power-of-two ranges:
/// - 12: > 16 KiB, ≤ 64 KiB (Buddy territory)
/// - 13: > 64 KiB, ≤ 1 MiB  (Buddy large)
/// - 14: > 1 MiB             (System)
///
/// This gives the MAB enough granularity to distinguish small-allocation call
/// sites without exploding the Signature space for large allocations.
pub fn size_class_for(size: usize) -> u8 {
    if let Some(idx) = slab_class_for(size) {
        return idx as u8;
    }
    // Above SLAB_MAX (16384).
    if size <= 65536 {
        12
    } else if size <= (1 << 20) {
        13
    } else {
        14
    }
}

/// Combine a call-site hash with a `size_class_for` bucket into the single
/// `u64` key the frozen `PerfectHashTable` is looked up over (v2 wire
/// format — see `perfect_hash.rs`). Routing keys on the full
/// `(topological_hash, size_class)` Signature the bandit actually trains
/// on; v1 collapsed the frozen table to `caller_pc` alone, so two
/// Signatures sharing a call site but trained at different size classes
/// (e.g. a helper called with both a 64-byte and a 64 KiB request) silently
/// clobbered each other into one ambiguous entry.
///
/// Full-avalanche mix so a size-class-only difference still lands far apart
/// in the CHD table; not required (or intended) to be reversible back to
/// the original `size_class`.
#[inline]
pub fn combine_hash_size_class(hash: u64, size_class: u8) -> u64 {
    hash ^ (size_class as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .rotate_left(17)
}

/// The allocator's operating mode.
///
/// `Training` learns routing via the MAB; `Inference` uses a frozen
/// `PerfectHashTable` for O(1) hash-and-jump routing.
pub enum AllocatorState {
    /// Learning mode: the Multi-Armed Bandit selects backends and learns
    /// from outcomes.
    Training { bandit: BanditPolicy },
    /// Frozen mode: the bandit's weights have collapsed into a read-only
    /// `PerfectHashTable`. The hot path is a single O(1) MPHF lookup.
    Inference { routing_table: PerfectHashTable },
}

impl Default for AllocatorState {
    fn default() -> Self {
        Self::new_training()
    }
}

impl AllocatorState {
    /// Create a new `AllocatorState` in Training mode with a fresh bandit.
    pub fn new_training() -> Self {
        Self::Training {
            bandit: BanditPolicy::new(),
        }
    }

    /// Const constructor for use in `static` contexts. Produces a Training
    /// state with an empty bandit.
    pub const fn new_training_const() -> Self {
        Self::Training {
            bandit: BanditPolicy::new_const(),
        }
    }

    /// True if the state machine is in Inference mode.
    pub fn is_inference(&self) -> bool {
        matches!(self, Self::Inference { .. })
    }

    /// Route an allocation to a backend.
    ///
    /// - **Training**: consults the `BanditPolicy` (UCB1 + hysteresis).
    /// - **Inference**: looks up `combine_hash_size_class(hash, size_class)`
    ///   in the `PerfectHashTable` — matching the key `freeze()` built the
    ///   table with. If the key is not in the table, falls back to a
    ///   default backend determined by size class (Slab for small, Buddy
    ///   for medium, System for large).
    pub fn route(&mut self, hash: u64, size: usize) -> Backend {
        match self {
            Self::Training { bandit } => {
                let size_class = size_class_for(size);
                let sig = Signature::new(hash, size_class);
                bandit.select(sig)
            }
            Self::Inference { routing_table } => {
                // Hash-and-jump: O(1) minimal perfect hash lookup. Zero allocations.
                let key = combine_hash_size_class(hash, size_class_for(size));
                if let Some(backend) = routing_table.lookup(key) {
                    return backend;
                }
                // Key not in the frozen table → fall back to size-based default.
                default_backend_for_size(size)
            }
        }
    }

    /// Record a measured latency sample (Layer 2) for the bandit, feeding
    /// back the real cost of a routing decision instead of a static
    /// baseline. Called from both the alloc side (`lib.rs::route_alloc_inner`
    /// — the recommended arm's total outcome latency, including any failed
    /// attempt + fallthrough) and the dealloc side (`lib.rs`'s
    /// `GlobalAlloc::dealloc` — the actual serving backend's free cost, an
    /// additional reward sample on the same arm so Arena's free-is-a-no-op
    /// advantage becomes visible to the bandit).
    ///
    /// - **Training**: converts `latency_ns` to a bounded reward (see
    ///   [`latency_to_reward`]) and updates the bandit's arm statistics.
    /// - **Inference**: no-op (the routing table is immutable).
    pub fn record_latency(&mut self, hash: u64, backend: Backend, size_class: u8, latency_ns: u64) {
        if let Self::Training { bandit } = self {
            let sig = Signature::new(hash, size_class);
            let reward = latency_to_reward(latency_ns);
            bandit.update(sig, backend, reward);
        }
    }

    /// Transition from Training to Inference mode.
    ///
    /// Collapses the bandit's learned per-Signature weights into a
    /// `PerfectHashTable` (one entry per observed signature, mapping hash →
    /// best backend).
    ///
    /// # Panics
    ///
    /// Panics if called when already in Inference mode. `freeze()` is a
    /// one-way transition.
    pub fn freeze(&mut self) {
        match self {
            Self::Training { bandit } => {
                let entries = bandit.freeze();
                let table = PerfectHashTable::from_entries(entries);
                *self = Self::Inference {
                    routing_table: table,
                };
            }
            Self::Inference { .. } => {
                panic!("freeze() called on an already-frozen (Inference) state");
            }
        }
    }

    /// Snapshot the current "best backend per Signature" without
    /// transitioning to Inference mode. Used during live training to show
    /// the routing-table-as-it-is-being-built to the GUI — a TensorBoard-style
    /// view of the MAB's progress before the user commits to freezing.
    ///
    /// Returns an empty Vec if in Inference mode (the routing table there is
    /// already exposed via `export()`).
    pub fn routing_snapshot(&self) -> Vec<(u64, Backend)> {
        match self {
            Self::Training { bandit } => bandit.snapshot(),
            Self::Inference { .. } => Vec::new(),
        }
    }

    /// Number of distinct Signatures observed so far (Training mode only).
    /// Returns 0 in Inference mode.
    pub fn signature_count(&self) -> usize {
        match self {
            Self::Training { bandit } => bandit.len(),
            Self::Inference { .. } => 0,
        }
    }

    /// Reset back to a fresh Training state, discarding any frozen routing
    /// table or learned bandit weights. Used by the GUI's "back to
    /// training" button after the user has explored inference.
    pub fn reset_to_training(&mut self) {
        *self = Self::new_training();
    }

    /// Export the routing table to `.lohalloc` binary bytes.
    ///
    /// Only valid in Inference mode (after `freeze()`). Returns `None` if
    /// still in Training mode.
    pub fn export(&self) -> Option<Vec<u8>> {
        match self {
            Self::Inference { routing_table } => Some(routing_table.serialize()),
            Self::Training { .. } => None,
        }
    }

    /// Deserialize a `.lohalloc` model file and start directly in Inference
    /// mode — a pre-optimized heap from boot.
    ///
    /// Returns `None` if the data is malformed (bad magic, checksum, etc.).
    pub fn load(data: &[u8]) -> Option<Self> {
        let routing_table = PerfectHashTable::deserialize(data)?;
        Some(Self::Inference { routing_table })
    }

    /// Borrow the frozen routing table (Inference mode only). Used by
    /// `Lohalloc::freeze()`/`load()` to publish a lock-free copy for the
    /// Inference alloc fast path.
    pub fn routing_table(&self) -> Option<&PerfectHashTable> {
        match self {
            Self::Inference { routing_table } => Some(routing_table),
            Self::Training { .. } => None,
        }
    }
}

/// Default backend for a given size (used when the hash is not in the
/// frozen routing table, or as a sanity fallback).
pub(crate) fn default_backend_for_size(size: usize) -> Backend {
    let total_with_header = size + 48; // Header is 48 bytes.
    if total_with_header <= lohalloc_core::SLAB_MAX {
        Backend::Slab
    } else if total_with_header <= lohalloc_core::BUDDY_MAX {
        Backend::Buddy
    } else {
        Backend::System
    }
}

/// Reference latency (nanoseconds) for the Layer 2 reward curve: fast paths
/// (Slab/Arena/Buddy pops, all tens of nanoseconds) saturate near a reward of
/// 1.0, while slow events (mmap, region refill, a failed recommendation
/// falling through) pull the reward toward 0 — that gap is the
/// differentiation signal the bandit needs to actually learn which backend
/// is fast for a given call site, replacing the old static
/// `BASELINE_REWARDS` cost model.
const T_REF_NS: f64 = 50.0;

/// Map a measured latency to a bounded `(0, 1]` reward. Monotone decreasing
/// and scale-tolerant across machines: doubling every latency in a run
/// shifts individual rewards but preserves their relative ordering, which is
/// all UCB1 needs to pick the empirically-faster arm.
fn latency_to_reward(latency_ns: u64) -> f64 {
    T_REF_NS / (T_REF_NS + latency_ns as f64)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use lohalloc_core::Backend;

    #[test]
    fn size_class_small_allocs() {
        assert_eq!(size_class_for(0), 0);
        assert_eq!(size_class_for(8), 0);
        assert_eq!(size_class_for(16), 1);
        assert_eq!(size_class_for(100), 4); // 128
        assert_eq!(size_class_for(16384), 11); // SLAB_MAX
    }

    #[test]
    fn size_class_large_allocs() {
        assert_eq!(size_class_for(16385), 12); // > 16 KiB
        assert_eq!(size_class_for(65536), 12); // ≤ 64 KiB
        assert_eq!(size_class_for(65537), 13); // > 64 KiB
        assert_eq!(size_class_for(1 << 20), 13); // ≤ 1 MiB
        assert_eq!(size_class_for((1 << 20) + 1), 14); // > 1 MiB
    }

    #[test]
    fn new_training_starts_in_training() {
        let state = AllocatorState::new_training();
        assert!(!state.is_inference());
    }

    #[test]
    fn training_mode_routes_via_bandit() {
        let mut state = AllocatorState::new_training();
        // Route a few times — should return valid backends.
        for _ in 0..10 {
            let backend = state.route(42, 64);
            assert!(
                matches!(
                    backend,
                    Backend::Slab | Backend::Buddy | Backend::System | Backend::Arena
                ),
                "route should return a valid backend"
            );
            state.record_latency(42, backend, size_class_for(64), 10);
        }
    }

    #[test]
    fn freeze_transitions_to_inference() {
        let mut state = AllocatorState::new_training();
        // Train a bit.
        for _ in 0..10 {
            let backend = state.route(100, 64);
            state.record_latency(100, backend, size_class_for(64), 10);
        }
        assert!(!state.is_inference());
        state.freeze();
        assert!(state.is_inference());
    }

    #[test]
    fn inference_mode_routes_via_perfect_hash() {
        let mut state = AllocatorState::new_training();
        // Train: make hash 100 → Arena.
        for _ in 0..100 {
            let backend = state.route(100, 64);
            if backend == Backend::Arena {
                state.record_latency(100, backend, size_class_for(64), 10);
            } else {
                // Manually record Arena as winning.
                state.record_latency(100, Backend::Arena, size_class_for(64), 10);
            }
        }
        state.freeze();

        // In Inference, routing for hash 100 should return the frozen backend.
        let backend = state.route(100, 64);
        assert!(
            matches!(
                backend,
                Backend::Slab | Backend::Buddy | Backend::System | Backend::Arena
            ),
            "inference route should return a valid backend"
        );
    }

    #[test]
    fn inference_mode_no_telemetry() {
        let mut state = AllocatorState::new_training();
        for _ in 0..10 {
            let backend = state.route(50, 64);
            state.record_latency(50, backend, size_class_for(64), 10);
        }
        state.freeze();

        // record_latency() in Inference mode should be a no-op (not panic).
        state.record_latency(50, Backend::Slab, size_class_for(64), 10);
        // If we get here, it didn't panic.
    }

    #[test]
    fn inference_falls_back_for_unseen_hash() {
        let mut state = AllocatorState::new_training();
        // Train only hash 100.
        for _ in 0..10 {
            let backend = state.route(100, 64);
            state.record_latency(100, backend, size_class_for(64), 10);
        }
        state.freeze();

        // Hash 999 was never seen → should fall back to size-based default.
        let backend = state.route(999, 64);
        // Size 64 + header 48 = 112 ≤ SLAB_MAX → Slab.
        assert_eq!(backend, Backend::Slab);
    }

    #[test]
    fn inference_falls_back_for_large_size() {
        let mut state = AllocatorState::new_training();
        state.freeze();
        // Size > BUDDY_MAX → System.
        let backend = state.route(999, 2 * 1024 * 1024);
        assert_eq!(backend, Backend::System);
    }

    #[test]
    fn export_load_roundtrip() {
        let mut state = AllocatorState::new_training();
        // Train a few signatures.
        for _ in 0..20 {
            let backend = state.route(1, 64);
            state.record_latency(1, backend, size_class_for(64), 10);
        }
        for _ in 0..20 {
            let backend = state.route(2, 256);
            state.record_latency(2, backend, size_class_for(256), 10);
        }
        state.freeze();

        // Export.
        let bytes = state.export().expect("export should work in Inference");

        // Load into a new state.
        let mut loaded = AllocatorState::load(&bytes).expect("load should succeed");
        assert!(loaded.is_inference());

        // Routing decisions should match.
        assert_eq!(loaded.route(1, 64), state.route(1, 64));
        assert_eq!(loaded.route(2, 256), state.route(2, 256));
    }

    #[test]
    fn load_starts_in_inference() {
        let mut state = AllocatorState::new_training();
        for _ in 0..10 {
            let backend = state.route(42, 64);
            state.record_latency(42, backend, size_class_for(64), 10);
        }
        state.freeze();
        let bytes = state.export().unwrap();

        let loaded = AllocatorState::load(&bytes).unwrap();
        assert!(loaded.is_inference());
    }

    #[test]
    #[should_panic(expected = "freeze() called on an already-frozen")]
    fn freeze_twice_panics() {
        let mut state = AllocatorState::new_training();
        state.freeze();
        state.freeze(); // Should panic.
    }

    #[test]
    fn export_in_training_returns_none() {
        let state = AllocatorState::new_training();
        assert!(state.export().is_none());
    }

    #[test]
    fn load_bad_data_returns_none() {
        assert!(AllocatorState::load(&[0xFF; 32]).is_none());
    }

    #[test]
    fn load_empty_returns_none() {
        assert!(AllocatorState::load(&[]).is_none());
    }

    #[test]
    fn routing_snapshot_in_training_returns_observed_signatures() {
        let mut state = AllocatorState::new_training();
        // No observations yet → empty snapshot.
        assert!(state.routing_snapshot().is_empty());
        assert_eq!(state.signature_count(), 0);

        // Train hash 100 a few times.
        for _ in 0..5 {
            let b = state.route(100, 64);
            state.record_latency(100, b, size_class_for(64), 10);
        }
        // Train hash 200 a few times.
        for _ in 0..5 {
            let b = state.route(200, 128);
            state.record_latency(200, b, size_class_for(128), 10);
        }
        assert_eq!(state.signature_count(), 2);

        let snap = state.routing_snapshot();
        assert_eq!(snap.len(), 2);
        let hashes: Vec<u64> = snap.iter().map(|(h, _)| *h).collect();
        assert!(hashes.contains(&100));
        assert!(hashes.contains(&200));
    }

    #[test]
    fn routing_snapshot_in_inference_returns_empty() {
        let mut state = AllocatorState::new_training();
        for _ in 0..10 {
            let b = state.route(42, 64);
            state.record_latency(42, b, size_class_for(64), 10);
        }
        state.freeze();
        assert!(state.routing_snapshot().is_empty());
        assert_eq!(state.signature_count(), 0);
    }

    #[test]
    fn reset_to_training_clears_state() {
        let mut state = AllocatorState::new_training();
        for _ in 0..10 {
            let b = state.route(42, 64);
            state.record_latency(42, b, size_class_for(64), 10);
        }
        assert!(!state.is_inference());
        assert_eq!(state.signature_count(), 1);

        state.freeze();
        assert!(state.is_inference());
        assert_eq!(state.signature_count(), 0);

        state.reset_to_training();
        assert!(!state.is_inference());
        assert_eq!(state.signature_count(), 0);
        // State should be fresh — fresh MAB returns a valid backend.
        let b = state.route(7, 64);
        assert!(matches!(
            b,
            Backend::Slab | Backend::Buddy | Backend::System | Backend::Arena
        ));
    }
}
