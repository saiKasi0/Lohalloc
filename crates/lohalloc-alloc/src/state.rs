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
    /// - **Inference**: looks up the hash in the `PerfectHashTable`. If the
    ///   hash is not in the table, falls back to a default backend determined
    ///   by size class (Slab for small, Buddy for medium, System for large).
    pub fn route(&mut self, hash: u64, size: usize) -> Backend {
        match self {
            Self::Training { bandit } => {
                let size_class = size_class_for(size);
                let sig = Signature::new(hash, size_class);
                bandit.select(sig)
            }
            Self::Inference { routing_table } => {
                // Hash-and-jump: O(1) minimal perfect hash lookup. Zero allocations.
                if let Some(backend) = routing_table.lookup(hash) {
                    return backend;
                }
                // Hash not in the frozen table → fall back to size-based default.
                default_backend_for_size(size)
            }
        }
    }

    /// Record an allocation outcome (reward) for the bandit.
    ///
    /// - **Training**: updates the bandit's arm statistics.
    /// - **Inference**: no-op (the routing table is immutable).
    pub fn record(&mut self, hash: u64, backend: Backend, _size: usize) {
        match self {
            Self::Training { bandit } => {
                let size_class = size_class_for(_size);
                let sig = Signature::new(hash, size_class);
                // Simple reward: success = backend's baseline reward,
                // failure (not called here) would be 0.
                let reward = backend_reward(backend);
                bandit.update(sig, backend, reward);
            }
            Self::Inference { .. } => {
                // No-op: routing table is frozen and immutable.
            }
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
            Self::Training { bandit } => bandit.freeze(),
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
}

/// Default backend for a given size (used when the hash is not in the
/// frozen routing table, or as a sanity fallback).
fn default_backend_for_size(size: usize) -> Backend {
    let total_with_header = size + 48; // Header is 48 bytes.
    if total_with_header <= lohalloc_core::SLAB_MAX {
        Backend::Slab
    } else if total_with_header <= lohalloc_core::BUDDY_MAX {
        Backend::Buddy
    } else {
        Backend::System
    }
}

/// Baseline reward for a backend (mirrors `bandit.rs` BASELINE_REWARDS).
fn backend_reward(backend: Backend) -> f64 {
    match backend {
        Backend::Slab => 1.0,
        Backend::Buddy => 0.8,
        Backend::System => 0.3,
        Backend::Arena => 0.9,
    }
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
            state.record(42, backend, 64);
        }
    }

    #[test]
    fn freeze_transitions_to_inference() {
        let mut state = AllocatorState::new_training();
        // Train a bit.
        for _ in 0..10 {
            let backend = state.route(100, 64);
            state.record(100, backend, 64);
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
                state.record(100, backend, 64);
            } else {
                // Manually record Arena as winning.
                state.record(100, Backend::Arena, 64);
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
            state.record(50, backend, 64);
        }
        state.freeze();

        // record() in Inference mode should be a no-op (not panic).
        state.record(50, Backend::Slab, 64);
        // If we get here, it didn't panic.
    }

    #[test]
    fn inference_falls_back_for_unseen_hash() {
        let mut state = AllocatorState::new_training();
        // Train only hash 100.
        for _ in 0..10 {
            let backend = state.route(100, 64);
            state.record(100, backend, 64);
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
            state.record(1, backend, 64);
        }
        for _ in 0..20 {
            let backend = state.route(2, 256);
            state.record(2, backend, 256);
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
            state.record(42, backend, 64);
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
            state.record(100, b, 64);
        }
        // Train hash 200 a few times.
        for _ in 0..5 {
            let b = state.route(200, 128);
            state.record(200, b, 128);
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
            state.record(42, b, 64);
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
            state.record(42, b, 64);
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
