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

use std::collections::{BTreeMap, BTreeSet};

use lohalloc_core::{slab_class_for, Backend, Signature};

use crate::bandit::BanditPolicy;
use crate::perfect_hash::{FrozenRouting, PerfectHashTable};

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

/// Phase-1 context routing: derive the FINE routing key from a coarse
/// combined key and the allocation-history context (see `lib.rs`'s `AHR`).
/// Same shape as [`combine_hash_size_class`] with a different odd multiplier
/// and rotation so ctx-mixing and size-class-mixing never produce structured
/// collisions; `ctx + 1` so ctx `0` still lands away from the coarse key.
/// Both frozen sections (coarse + fine) live in ONE perfect hash table —
/// this key derivation is what keeps them distinct.
#[inline]
pub fn combine_key_ctx(key: u64, ctx: u8) -> u64 {
    key ^ (ctx as u64 + 1)
        .wrapping_mul(0xD6E8_FEB8_6659_FD93)
        .rotate_left(23)
}

/// Roadmap-D: the DEEP fine routing key — same derivation as
/// [`combine_key_ctx`] over a salted coarse key, so a shallow entry at ctx
/// `x` and a deep entry at deep-ctx `x` for the same site can never alias in
/// the shared MPHF even though both ctx values live in the same `u8` space.
/// (Freeze also makes the two families mutually exclusive per site — the
/// salt is defense in depth and keeps `model_dump` diffs unambiguous.)
#[inline]
pub fn combine_key_ctx_deep(key: u64, deep_ctx: u8) -> u64 {
    combine_key_ctx(key ^ 0xA5A5_5A5A_C33C_3CC3, deep_ctx)
}

/// One in-flight reward batch for a (Signature, Backend) arm — raw latency
/// and fragmentation sums awaiting enough samples to convert to a reward.
#[derive(Clone, Copy, Default)]
struct PendingBatch {
    latency_sum: u64,
    frag_sum: f32,
    count: u32,
}

/// Number of log2 buckets in a [`LatencyHist`]. Bucket `i` covers
/// `[2^i, 2^(i+1)) ns`, so 24 buckets span ~1 ns … ~16 ms — every real
/// alloc/free latency plus the cold mmap/segment spikes the winsorizer trims.
const LATENCY_HIST_BUCKETS: usize = 24;

/// Task B: a fixed log-bucketed latency histogram for one `(Signature,
/// Backend)` arm, the per-arm estimator that replaced the Phase-1.5 global
/// `latency_clamp_ns` constant. A bump arena self-clamps low (its cold spikes
/// are noise), a contended slab mix self-clamps high (its tail is signal) —
/// the split a single constant could not express. Allocation-free, O(1)
/// `record`, O(buckets) `quantile` — a deterministic histogram was chosen
/// over P²'s float markers precisely so the trained model is reproducible.
#[derive(Default)]
struct LatencyHist {
    buckets: [u32; LATENCY_HIST_BUCKETS],
    total: u64,
}

impl LatencyHist {
    /// `floor(log2(max(ns, 1)))`, clamped to the top bucket.
    #[inline]
    fn bucket_of(ns: u64) -> usize {
        let lz = ns.max(1).leading_zeros(); // 0..=63
        ((63 - lz) as usize).min(LATENCY_HIST_BUCKETS - 1)
    }

    #[inline]
    fn record(&mut self, ns: u64) {
        let b = Self::bucket_of(ns);
        self.buckets[b] = self.buckets[b].saturating_add(1);
        self.total = self.total.saturating_add(1);
    }

    /// Winsorization threshold at percentile `pct` (1..=100): the upper edge
    /// `2^(i+1)` of the bucket that contains the `pct`-th percentile sample.
    /// Conservative — a sample at or below this passes through unchanged; a
    /// higher sample is clamped down to it. Returns `u64::MAX` (no clamp) when
    /// the arm has no samples yet.
    fn quantile(&self, pct: u8) -> u64 {
        if self.total == 0 {
            return u64::MAX;
        }
        let pct = pct.min(100) as u64;
        // 1-based rank of the target sample: ceil(total * pct / 100), >= 1.
        let target = (self.total * pct).div_ceil(100).max(1);
        let mut cum: u64 = 0;
        for (i, &c) in self.buckets.iter().enumerate() {
            cum += c as u64;
            if cum >= target {
                return 1u64 << (i + 1);
            }
        }
        u64::MAX
    }
}

/// Per-(Signature, Backend) reward batching (see [`AllocatorState::record_latency`]).
///
/// # Why rewards are batched (the ARM clock-floor fix)
///
/// `clock::now_ns()` is `Instant`-backed, and on Apple Silicon (and this
/// project's local Docker, which passes the same 24 MHz ARM generic timer
/// through) a single measured interval quantizes to multiples of ~41.7ns.
/// A true ~25ns Slab/Arena op therefore reads as **either 0ns or 42ns** —
/// with the default `t_ref_ns = 50` that's a reward of 1.000 vs 0.543, a
/// ~2× swing for the identical operation, injected straight into UCB1.
/// The bandit was learning fast-backend rankings from timer noise.
///
/// Averaging `reward_batch` (default 16) raw latencies before converting
/// to a reward makes the quantization error average out (each sample's
/// tick-rounding is ~uniform in [0, tick)), recovering a sub-tick-accurate
/// mean; x86 (~ns TSC) is unaffected either way. `reward_batch = 1`
/// reproduces the old per-op behavior bit-for-bit.
///
/// Batches are keyed by (Signature, Backend) because a signature's samples
/// can span backends (fallthrough on alloc, dealloc-side attribution) and
/// a reward must only ever credit the arm that produced its latencies.
/// At `freeze()` any partial batches (< reward_batch samples) are simply
/// dropped — at most `reward_batch - 1` samples per arm, noise-level.
#[derive(Default)]
pub struct PendingRewards {
    batches: BTreeMap<(Signature, u8), PendingBatch>,
    /// Phase-1 shadow fine batches, keyed `(Signature, backend, ctx)`.
    /// DELIBERATELY separate from `batches`: folding ctx into the coarse
    /// key would split coarse batches 16 ways and change their flush
    /// cadence — coarse training must stay byte-identical to the
    /// pre-context policy. Fed from both sides: alloc (live AHR ctx) and,
    /// since Phase 1.5, dealloc (the alloc-time ctx from the `Header`).
    fine_batches: BTreeMap<(Signature, u8, u8), PendingBatch>,
    /// Roadmap-D deep fine batches, keyed `(Signature, backend, deep_ctx)`.
    /// Alloc-side only, and empty whenever `escalate_variance == 0` (see
    /// [`AllocatorState::record_fine_deep_latency`]).
    fine_deep_batches: BTreeMap<(Signature, u8, u8), PendingBatch>,
    /// Task B: per-arm latency histograms keyed `(Signature, backend)` —
    /// **without** ctx, so a coarse `record_latency` sample and a fine
    /// `record_fine_latency` sample for the same arm feed the SAME
    /// distribution (they measure the same backend's latency). Unlike
    /// `batches`/`fine_batches` this **persists across flushes** — the running
    /// quantile is what winsorizes each new sample. Bounded by observed
    /// signatures (same order as the bandit's `stats`), fixed-size buckets, so
    /// no unbounded growth; dropped whole at `freeze()` (Training→Inference),
    /// never carried into the frozen table.
    hists: BTreeMap<(Signature, u8), LatencyHist>,
}

impl PendingRewards {
    pub const fn new() -> Self {
        Self {
            batches: BTreeMap::new(),
            fine_batches: BTreeMap::new(),
            fine_deep_batches: BTreeMap::new(),
            hists: BTreeMap::new(),
        }
    }
}

/// The allocator's operating mode.
///
/// `Training` learns routing via the MAB; `Inference` uses a frozen
/// `PerfectHashTable` for O(1) hash-and-jump routing.
pub enum AllocatorState {
    /// Learning mode: the Multi-Armed Bandit selects backends and learns
    /// from outcomes (rewards arrive in batches — see [`PendingRewards`]).
    Training {
        bandit: BanditPolicy,
        pending: PendingRewards,
    },
    /// Frozen mode: the bandit's weights have collapsed into a read-only
    /// `PerfectHashTable`s. The hot path is a single O(1) MPHF lookup
    /// (against `routing.main`); `routing.distilled` carries the Ladder-6
    /// 1-frame-keyed pinnable subset consumed by the Inference pin cache.
    Inference { routing: FrozenRouting },
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
            pending: PendingRewards::new(),
        }
    }

    /// Const constructor for use in `static` contexts. Produces a Training
    /// state with an empty bandit.
    pub const fn new_training_const() -> Self {
        Self::Training {
            bandit: BanditPolicy::new_const(),
            pending: PendingRewards::new(),
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
        self.route_with_frame(hash, 0, size, 0)
    }

    /// [`route`](Self::route) plus the J2/Ladder-6 one-frame hash for this
    /// call site (`0` = unknown, e.g. the replay path). Training retains it
    /// per Signature so `freeze()` can distill unambiguous sites into the
    /// 1-frame pin table; Inference ignores it entirely.
    ///
    /// `unservable_mask`: bit-per-`Backend` arms training must not select
    /// because this instance's inference path cannot serve them — see
    /// `BanditPolicy::select_with_frame`. Ignored in Inference (the frozen
    /// table already passed the freeze-time servability clamp).
    pub fn route_with_frame(
        &mut self,
        hash: u64,
        one_frame: u64,
        size: usize,
        unservable_mask: u8,
    ) -> Backend {
        match self {
            Self::Training { bandit, .. } => {
                let size_class = size_class_for(size);
                let sig = Signature::new(hash, size_class);
                bandit.select_with_frame(sig, one_frame, crate::tune::config(), unservable_mask)
            }
            Self::Inference { routing } => {
                // Hash-and-jump: O(1) minimal perfect hash lookup. Zero allocations.
                let key = combine_hash_size_class(hash, size_class_for(size));
                if let Some(backend) = routing.main.lookup(key) {
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
    /// - **Training**: converts `latency_ns` (plus, when `frag_weight > 0`,
    ///   the allocation's internal-fragmentation percentage) to a shaped
    ///   reward (see [`shaped_reward`]) and updates the bandit's arm
    ///   statistics. `total_bytes` is the backend-facing allocation size
    ///   (request + header padding) the fragmentation math needs; it is
    ///   only inspected when the configured `frag_weight` is nonzero, so
    ///   the default config pays nothing for it.
    /// - **Inference**: no-op (the routing table is immutable).
    pub fn record_latency(
        &mut self,
        hash: u64,
        backend: Backend,
        size_class: u8,
        latency_ns: u64,
        total_bytes: usize,
    ) {
        if let Self::Training { bandit, pending } = self {
            let sig = Signature::new(hash, size_class);
            let cfg = crate::tune::config();
            // Task B per-arm spike winsorization (see `winsorize_latency`):
            // bound a single cold refill spike's influence on its batch — a
            // batch reward is credited with weight `count`, so an unclamped
            // 20µs mmap spike poisons 16 pulls at once. The threshold is this
            // arm's own running `clamp_percentile` quantile, not a global
            // constant.
            let latency_ns = {
                let hist = pending.hists.entry((sig, backend as u8)).or_default();
                winsorize_latency(hist, latency_ns, cfg)
            };
            let frag_pct = if cfg.frag_weight > 0.0 {
                crate::frag_pct_for(backend, total_bytes)
            } else {
                0.0
            };
            // Accumulate into this arm's pending batch; only convert to a
            // reward and update the bandit once `reward_batch` samples have
            // landed (averaging out the clock's tick quantization — see
            // `PendingRewards`). `reward_batch == 1` flushes every call, so
            // the reward is `shaped_reward(latency_ns, frag_pct, cfg)`
            // exactly as before batching existed.
            let batch = pending.batches.entry((sig, backend as u8)).or_default();
            batch.latency_sum += latency_ns;
            batch.frag_sum += frag_pct;
            batch.count += 1;
            if batch.count >= cfg.reward_batch {
                let mean_latency = batch.latency_sum / batch.count as u64;
                let mean_frag = batch.frag_sum / batch.count as f32;
                let reward = shaped_reward(mean_latency, mean_frag, cfg);
                // Credit the de-quantized reward once per pull in the batch
                // so `mean_reward = sum/pulls` stays a true average (see
                // `BanditPolicy::update_weighted`). With reward_batch == 1
                // this is a plain weight-1 update — bit-identical to the
                // pre-batching per-op path.
                bandit.update_weighted(sig, backend, reward, batch.count);
                pending.batches.remove(&(sig, backend as u8));
            }
        }
    }

    /// Phase-1 shadow fine reward: the counterpart of
    /// [`record_latency`](Self::record_latency), additionally keyed on the
    /// allocation-history context. Called from the alloc path with the ctx
    /// read from the live AHR, and (Phase 1.5) from the dealloc path with
    /// the *alloc-time* ctx carried in the allocation's `Header` — the
    /// history at alloc time is what routing would have seen, so both
    /// halves of an allocation's lifecycle credit the same fine arm.
    /// Batching mirrors `record_latency` exactly; flushes land in the
    /// bandit's passive fine map (`update_fine`), never touching coarse
    /// routing.
    pub fn record_fine_latency(
        &mut self,
        hash: u64,
        backend: Backend,
        size_class: u8,
        ctx: u8,
        latency_ns: u64,
        total_bytes: usize,
    ) {
        if let Self::Training { bandit, pending } = self {
            let sig = Signature::new(hash, size_class);
            let cfg = crate::tune::config();
            // Same per-arm winsorization as `record_latency`, sharing the arm's
            // histogram (keyed without ctx) so both alloc- and dealloc-side
            // samples update one distribution.
            let latency_ns = {
                let hist = pending.hists.entry((sig, backend as u8)).or_default();
                winsorize_latency(hist, latency_ns, cfg)
            };
            let frag_pct = if cfg.frag_weight > 0.0 {
                crate::frag_pct_for(backend, total_bytes)
            } else {
                0.0
            };
            let batch = pending
                .fine_batches
                .entry((sig, backend as u8, ctx))
                .or_default();
            batch.latency_sum += latency_ns;
            batch.frag_sum += frag_pct;
            batch.count += 1;
            if batch.count >= cfg.reward_batch {
                let mean_latency = batch.latency_sum / batch.count as u64;
                let mean_frag = batch.frag_sum / batch.count as f32;
                let reward = shaped_reward(mean_latency, mean_frag, cfg);
                bandit.update_fine(sig, ctx, backend, reward, batch.count);
                pending.fine_batches.remove(&(sig, backend as u8, ctx));
            }
        }
    }

    /// Roadmap-D deep counterpart of
    /// [`record_fine_latency`](Self::record_fine_latency): the same batching,
    /// keyed on the 8-event **folded** deep context, feeding the bandit's
    /// passive `fine_deep` map. No-op when `escalate_variance == 0` (the
    /// disable switch removes the training-time upsert cost too, so the
    /// off-cell of an A/B is byte-identical to pre-D training). Alloc-side
    /// only — the `Header` carries just the shallow ctx, and the Part-2 rt
    /// verdict showed dealloc-side samples are where the noise lives anyway.
    /// Winsorization: the sample is clamped at the arm's existing histogram
    /// quantile WITHOUT re-recording it (the coarse/shallow record paths in
    /// the same lock section already recorded this very sample; a third
    /// record would over-count it in the arm's distribution).
    pub fn record_fine_deep_latency(
        &mut self,
        hash: u64,
        backend: Backend,
        size_class: u8,
        deep_ctx: u8,
        latency_ns: u64,
        total_bytes: usize,
    ) {
        if let Self::Training { bandit, pending } = self {
            let cfg = crate::tune::config();
            if cfg.escalate_variance <= 0.0 {
                return;
            }
            let sig = Signature::new(hash, size_class);
            // Peek-clamp against the arm's histogram (same warmup rule as
            // `winsorize_latency`, minus the record).
            let latency_ns = if cfg.clamp_percentile > 0 {
                match pending.hists.get(&(sig, backend as u8)) {
                    Some(h) if h.total >= cfg.reward_batch as u64 => {
                        latency_ns.min(h.quantile(cfg.clamp_percentile))
                    }
                    _ => latency_ns,
                }
            } else {
                latency_ns
            };
            let frag_pct = if cfg.frag_weight > 0.0 {
                crate::frag_pct_for(backend, total_bytes)
            } else {
                0.0
            };
            let batch = pending
                .fine_deep_batches
                .entry((sig, backend as u8, deep_ctx))
                .or_default();
            batch.latency_sum += latency_ns;
            batch.frag_sum += frag_pct;
            batch.count += 1;
            if batch.count >= cfg.reward_batch {
                let mean_latency = batch.latency_sum / batch.count as u64;
                let mean_frag = batch.frag_sum / batch.count as f32;
                let reward = shaped_reward(mean_latency, mean_frag, cfg);
                bandit.update_fine_deep(sig, deep_ctx, backend, reward, batch.count);
                pending
                    .fine_deep_batches
                    .remove(&(sig, backend as u8, deep_ctx));
            }
        }
    }

    /// True when this state's routing has stabilized enough to freeze —
    /// the `freeze_mode=converged` trigger, forwarded to
    /// [`BanditPolicy::is_converged`] with the configured thresholds.
    /// Inference mode is trivially "converged" (already frozen).
    pub fn is_converged(&self) -> bool {
        match self {
            Self::Training { bandit, .. } => {
                let cfg = crate::tune::config();
                bandit.is_converged(cfg.converge_stable_n)
            }
            Self::Inference { .. } => true,
        }
    }

    /// Transition from Training to Inference mode.
    ///
    /// Collapses the bandit's learned per-Signature weights into a
    /// `PerfectHashTable` (one entry per observed signature, mapping hash →
    /// best backend).
    ///
    /// `demote_arena` is the J4-C bistability fix, made **volume-selective**
    /// by J5-A: when the caller observed **zero arena resets during
    /// training** (`Lohalloc` passes `arena_epoch == 0`), an Arena verdict is
    /// demoted to the size-appropriate Slab/Buddy backend in *both* frozen
    /// tables — but only for a **heavy** signature, one whose training volume
    /// (`total_pulls`) is ≥ [`ARENA_DEMOTE_VOLUME_FRACTION`] of that table's
    /// total. Rationale: a bump arena only reclaims via `reset()`; in a
    /// deployment that never resets (LD_PRELOAD / global-allocator, the whole
    /// native bench matrix), a *dominant* burst site fills the arena once and
    /// then falls through on every allocation *forever* (measured: 68,940
    /// fallthroughs / 200k ops) — and whether it froze to Arena or Slab was
    /// cold-start UCB noise swinging gate rows 1.3×↔3.5× (J4-C). But J4-C's
    /// *blanket* demotion also flattened sites that used Arena lightly and
    /// beneficially (adv-mixed routes only ~117 allocs there — it can never
    /// fill a chunk, so its bump-speed win is free), and J4-D proved keeping
    /// heavy sites is even worse (worst 3.98×). The volume split takes the
    /// good half of both: light Arena kept, heavy Arena demoted,
    /// deterministically. Workloads that genuinely reset (GUI/replay) keep
    /// everything.
    ///
    /// `demote_fraction` is the heavy/light threshold. Production passes the
    /// `demote_fraction` tune key (default [`ARENA_DEMOTE_VOLUME_FRACTION`]);
    /// it's a parameter (not a `tune::config()` read here) so tests can pin
    /// it without racing the process-global config. `0.0` demotes every
    /// Arena verdict (J4-C blanket); `> 1.0` never demotes.
    ///
    /// # Panics
    ///
    /// Panics if called when already in Inference mode. `freeze()` is a
    /// one-way transition.
    pub fn freeze(&mut self, demote_arena: bool, demote_fraction: f64, escalate_variance: f64) {
        match self {
            Self::Training { bandit, .. } => {
                // Same per-size-class clamp for both tables: a distilled
                // entry must never license a backend the main table would
                // have clamped away (the pin cache serves straight from
                // distilled with no further checks). `total_all` is computed
                // per table — main counts every signature, distilled only
                // the unambiguous groups — so each table's fractions are
                // self-consistent. Returns the per-entry `heavy` verdict
                // alongside so the fine layer can inherit its parent's
                // Arena-demotion decision.
                let clamp = |entries: Vec<(u64, u8, lohalloc_core::Backend, u32)>| {
                    let total_all: u64 = entries.iter().map(|&(_, _, _, p)| p as u64).sum();
                    entries
                        .into_iter()
                        .map(|(key, size_class, backend, pulls)| {
                            let heavy = total_all > 0
                                && (pulls as f64 / total_all as f64) >= demote_fraction;
                            (
                                key,
                                size_class,
                                clamp_backend_for_size_class(
                                    size_class,
                                    backend,
                                    demote_arena && heavy,
                                ),
                                heavy,
                            )
                        })
                        .collect::<Vec<_>>()
                };
                let main_clamped = clamp(bandit.freeze());

                // Phase-1 hierarchical freeze: the shadow fine map emits
                // context-keyed overrides, compared against the FINAL
                // (clamped) coarse verdicts; a fine Arena verdict inherits
                // its parent's heavy/demotion decision; every fine entry
                // passes the same unservable-verdict clamp.
                let final_map: BTreeMap<u64, lohalloc_core::Backend> = main_clamped
                    .iter()
                    .map(|&(key, _, backend, _)| (key, backend))
                    .collect();
                let heavy_map: BTreeMap<u64, bool> = main_clamped
                    .iter()
                    .map(|&(key, _, _, heavy)| (key, heavy))
                    .collect();
                let fine = bandit.freeze_fine(&final_map, escalate_variance);
                // One pass: clamp each fine entry with its parent's heavy
                // verdict; drop any whose clamped verdict collapses back to
                // the parent's final backend (it would only buy an extra
                // probe). A coarse key is flagged only if at least one of
                // its fine entries SURVIVES — a flag with no fine siblings
                // is one wasted probe per alloc at that site, forever.
                let mut fine_entries: Vec<(u64, u8, lohalloc_core::Backend, u8)> = Vec::new();
                let mut surviving_parents: BTreeSet<u64> = BTreeSet::new();
                for &(fine_key, coarse_key, size_class, backend) in &fine.entries {
                    let parent_heavy = heavy_map.get(&coarse_key).copied().unwrap_or(true);
                    let clamped = clamp_backend_for_size_class(
                        size_class,
                        backend,
                        demote_arena && parent_heavy,
                    );
                    if final_map.get(&coarse_key) == Some(&clamped) {
                        continue;
                    }
                    fine_entries.push((fine_key, size_class, clamped, 0u8));
                    surviving_parents.insert(coarse_key);
                }
                // Roadmap-D deep entries: same clamp/collapse rules. Deep and
                // shallow parents are disjoint by bandit construction (deep
                // is only emitted where no shallow entry was), so each site
                // carries at most one of the two flags.
                let mut deep_surviving_parents: BTreeSet<u64> = BTreeSet::new();
                for &(deep_key, coarse_key, size_class, backend) in &fine.deep_entries {
                    let parent_heavy = heavy_map.get(&coarse_key).copied().unwrap_or(true);
                    let clamped = clamp_backend_for_size_class(
                        size_class,
                        backend,
                        demote_arena && parent_heavy,
                    );
                    if final_map.get(&coarse_key) == Some(&clamped) {
                        continue;
                    }
                    fine_entries.push((deep_key, size_class, clamped, 0u8));
                    deep_surviving_parents.insert(coarse_key);
                }

                let mut main_quads: Vec<(u64, u8, lohalloc_core::Backend, u8)> = main_clamped
                    .into_iter()
                    .map(|(key, size_class, backend, _)| {
                        let flags = if surviving_parents.contains(&key) {
                            crate::perfect_hash::FLAG_HAS_CONTEXT
                        } else if deep_surviving_parents.contains(&key) {
                            crate::perfect_hash::FLAG_DEEP_CONTEXT
                        } else {
                            0
                        };
                        (key, size_class, backend, flags)
                    })
                    .collect();
                main_quads.extend(fine_entries);
                let main = PerfectHashTable::from_entries_flagged(main_quads);

                // Distilled: exclude context-routed sites — the pin cache
                // would serve them without the ctx probe.
                let excluded: BTreeSet<u64> = fine
                    .flagged_frames
                    .iter()
                    .map(|&(one_frame, sc)| combine_hash_size_class(one_frame, sc))
                    .collect();
                let distilled_entries: Vec<(u64, u8, lohalloc_core::Backend)> =
                    clamp(bandit.distill())
                        .into_iter()
                        .filter(|(key, _, _, _)| !excluded.contains(key))
                        .map(|(key, size_class, backend, _)| (key, size_class, backend))
                        .collect();
                let distilled = PerfectHashTable::from_entries(distilled_entries);
                *self = Self::Inference {
                    routing: FrozenRouting { main, distilled },
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
            Self::Training { bandit, .. } => bandit.snapshot(),
            Self::Inference { .. } => Vec::new(),
        }
    }

    /// Diagnostic passthrough to [`BanditPolicy::fine_snapshot`] (empty in
    /// Inference mode). Test/route-metrics introspection only.
    #[cfg(any(feature = "route-metrics", test))]
    pub fn fine_snapshot(&self) -> Vec<crate::bandit::FineSnapshotRow> {
        match self {
            Self::Training { bandit, .. } => bandit.fine_snapshot(),
            Self::Inference { .. } => Vec::new(),
        }
    }

    /// Diagnostic passthrough to [`BanditPolicy::variance_snapshot`] — the
    /// paradigm-investigation aliasing meter (empty in Inference mode).
    /// Test/route-metrics introspection only.
    #[cfg(any(feature = "route-metrics", test))]
    pub fn variance_snapshot(&self) -> Vec<(u64, u8, u32, Backend, f64)> {
        match self {
            Self::Training { bandit, .. } => bandit.variance_snapshot(),
            Self::Inference { .. } => Vec::new(),
        }
    }

    /// Number of distinct Signatures observed so far (Training mode only).
    /// Returns 0 in Inference mode.
    pub fn signature_count(&self) -> usize {
        match self {
            Self::Training { bandit, .. } => bandit.len(),
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
            Self::Inference { routing } => Some(routing.serialize()),
            Self::Training { .. } => None,
        }
    }

    /// Deserialize a `.lohalloc` model file and start directly in Inference
    /// mode — a pre-optimized heap from boot.
    ///
    /// Returns `None` if the data is malformed (bad magic, non-v3 version,
    /// checksum, etc.).
    pub fn load(data: &[u8]) -> Option<Self> {
        let routing = FrozenRouting::deserialize(data)?;
        Some(Self::Inference { routing })
    }

    /// Borrow the frozen main routing table (Inference mode only) —
    /// introspection (e.g. `model_dump`). The lock-free hot-path copy is
    /// published from [`Self::routing`].
    pub fn routing_table(&self) -> Option<&PerfectHashTable> {
        self.routing().map(|r| &r.main)
    }

    /// Borrow the frozen distilled (1-frame pinnable) table (Inference mode
    /// only) — introspection counterpart of [`Self::routing_table`].
    pub fn distilled_table(&self) -> Option<&PerfectHashTable> {
        self.routing().map(|r| &r.distilled)
    }

    /// Borrow the complete frozen decision plane (Inference mode only).
    /// Used by `Lohalloc::freeze()`/`load()` to publish a lock-free copy
    /// for the Inference alloc fast path.
    pub fn routing(&self) -> Option<&FrozenRouting> {
        match self {
            Self::Inference { routing } => Some(routing),
            Self::Training { .. } => None,
        }
    }
}

/// Default backend for a given size (used when the hash is not in the
/// frozen routing table, or as a sanity fallback).
pub(crate) fn default_backend_for_size(size: usize) -> Backend {
    let total_with_header = size + 48; // Header is 48 bytes.
    let size_class = size_class_for(total_with_header);
    default_backend_for_size_class(size_class)
}

/// The size-appropriate backend for a `size_class_for` bucket: Slab for the
/// 12 slab classes (0–11), Buddy for the two Buddy-range buckets (12, 13),
/// System only for the genuinely-large bucket (14, > 1 MiB). Shared by
/// `default_backend_for_size` (raw-size fallback) and
/// `clamp_backend_for_size_class` (freeze-time sanity check).
pub(crate) fn default_backend_for_size_class(size_class: u8) -> Backend {
    if size_class <= 11 {
        Backend::Slab
    } else if size_class <= 13 {
        Backend::Buddy
    } else {
        Backend::System
    }
}

/// Freeze-time sanity clamp: a Signature whose size class fits Slab or Buddy
/// must never freeze to System. Without this, a bandit that locked onto
/// System from early noise (a low `LOHALLOC_FREEZE_AFTER` cutoff, or a few
/// unlucky early samples) produces a frozen entry that always succeeds
/// (`try_backend(System)` never fails, so there is no fallthrough
/// self-correction the way there is for a bad Slab/Arena recommendation) —
/// every allocation at that Signature becomes an mmap/munmap instead of a
/// Slab/Buddy pop, observed as a >5x inference-slower-than-training
/// regression on cpp/buddy. This is a strict narrowing to what the
/// fallthrough chain would already do for an out-of-range recommendation;
/// it never changes behavior for a Signature that legitimately belongs on
/// System (size_class 14).
///
/// `demote_arena` (J4-C) extends the same idea to Arena for reset-free
/// deployments: an Arena verdict frozen from pre-exhaustion samples becomes a
/// permanent fallthrough storm once the un-reset arena fills (see
/// [`AllocatorState::freeze`]'s doc for the measured bistability). Demotion
/// targets what the fallthrough chain would pick anyway — Slab for slab-range
/// classes, Buddy for buddy-range — just without paying the doomed Arena
/// attempt per allocation.
///
/// J5 generalizes the same lesson to **unservable** verdicts: a backend that
/// cannot serve the size class in the deployment that will load this model
/// is a *per-allocation* doomed attempt + fallthrough forever once frozen:
/// - Slab above `SLAB_MAX` (sc > 11);
/// - Buddy above `BUDDY_MAX` (sc > 13), and — the subtle one — **below the
///   headerless order-map floor** (sc ≤ 10, requests ≤ 8 KiB, which round
///   below `buddy::MIN_HEADERLESS_ORDER`'s 16 KiB and are refused by
///   `buddy_block_headerless_via_magazine`). Training runs header-based
///   where Buddy CAN serve those sizes, so the bandit legitimately learns
///   the verdict — and every `load()`-booted (headerless) inference process
///   then fails it per-op. Measured on adv-mixed as ONE flipped sc-10 entry
///   swinging fallthrough 1.6k↔15.4k/200k across identically-built models.
/// - Arena above its 1 MiB chunk (sc > 13).
///
/// Like the System clamp, each is a strict narrowing to what the fallthrough
/// chain would do anyway, minus the doomed attempt per op.
fn clamp_backend_for_size_class(size_class: u8, backend: Backend, demote_arena: bool) -> Backend {
    if backend == Backend::System && size_class <= 13 {
        return default_backend_for_size_class(size_class);
    }
    let unservable = match backend {
        Backend::Slab => size_class > 11,
        Backend::Buddy => size_class > 13 || size_class <= 10,
        Backend::Arena => size_class > 13,
        Backend::System => false, // serves any size (clamped above when dominated)
    };
    if unservable {
        return default_backend_for_size_class(size_class);
    }
    if demote_arena && backend == Backend::Arena {
        return default_backend_for_size_class(size_class);
    }
    backend
}

/// Map a measured latency (and optionally an internal-fragmentation
/// percentage) to the bandit's reward under the given config.
///
/// The latency term `t_ref / (t_ref + latency)` is monotone decreasing and
/// scale-tolerant across machines: doubling every latency in a run shifts
/// individual rewards but preserves their relative ordering, which is all
/// UCB1 needs to pick the empirically-faster arm. `t_ref_ns` (default 50,
/// the former hard-coded `T_REF_NS`) sets the curve's knee: fast paths
/// (Slab/Arena/Buddy pops, tens of ns) saturate near 1.0 while slow events
/// (mmap, region refill, failed-recommendation fallthrough) pull toward 0.
/// A *small* `t_ref` punishes tail latencies hard (latency focus); a
/// *large* one flattens the curve toward mean-cost optimization
/// (throughput focus) — this is what the `focus` presets in `tune.rs`
/// tune.
///
/// The fragmentation term subtracts `frag_weight` per 100% internal
/// fragmentation (`frag_pct` from [`crate::frag_pct_for`]), letting a
/// throughput/memory-density-focused config prefer a backend that packs
/// tighter when raw latencies are close. `frag_weight = 0` (the default)
/// reproduces the pre-Step-8 reward bit-for-bit.
pub(crate) fn shaped_reward(
    latency_ns: u64,
    frag_pct: f32,
    cfg: &crate::tune::TrainingConfig,
) -> f64 {
    cfg.t_ref_ns / (cfg.t_ref_ns + latency_ns as f64)
        - cfg.frag_weight * f64::from(frag_pct) / 100.0
}

/// Task B per-arm spike winsorization (see `TrainingConfig::clamp_percentile`):
/// cap one raw latency sample at a running high percentile of THIS arm's own
/// distribution before it enters a reward batch. A bump arena self-clamps low
/// (cold spikes are noise) and a contended slab mix self-clamps high (its tail
/// is signal) — the split the fixed Phase-1.5 clamp could not express. The
/// histogram is updated with the current sample **first** (so it is
/// represented in its own quantile), then the sample is capped at the arm's
/// `clamp_percentile` quantile. **Warmup:** below `reward_batch` samples the
/// arm has no reliable quantile, so the sample passes through unclamped.
/// `clamp_percentile == 0` disables winsorization entirely (pre-1.5 behavior).
#[inline]
fn winsorize_latency(
    hist: &mut LatencyHist,
    latency_ns: u64,
    cfg: &crate::tune::TrainingConfig,
) -> u64 {
    if cfg.clamp_percentile == 0 {
        return latency_ns;
    }
    hist.record(latency_ns);
    if hist.total < cfg.reward_batch as u64 {
        return latency_ns; // warmup: no reliable quantile yet
    }
    latency_ns.min(hist.quantile(cfg.clamp_percentile))
}

/// J5-A: the volume threshold for freeze-time Arena demotion. A signature
/// whose `total_pulls` is at least this fraction of the table's total is
/// "heavy": in a reset-free deployment it will dominate arena traffic,
/// exhaust the cap, and storm — demote it. Below the threshold it is
/// "light": it can never meaningfully fill the arena, so its bump-speed
/// win is kept. Calibration from the certified A/Bs: the storm rows'
/// burst signatures carry the majority of training pulls (rust/arena's
/// site is ~all 1000 pre-freeze pulls), while adv-mixed's kept arena
/// signatures sit far below 10% each (~117 arena allocs / 200k mixed ops).
/// The gap between the two cases is orders of magnitude, so the exact
/// value is not delicate — 0.10 sits comfortably between them.
///
/// Since J5's bisect knobs this const is the *default* of the
/// `demote_fraction` tune key (`LOHALLOC_DEMOTE_FRACTION`), which is what
/// production `Lohalloc::freeze()` actually passes to
/// [`AllocatorState::freeze`].
pub(crate) const ARENA_DEMOTE_VOLUME_FRACTION: f64 = 0.10;

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
    fn shaped_reward_default_matches_pre_step8_curve() {
        // frag_weight = 0 (the default) must reproduce the historical
        // `T_REF_NS / (T_REF_NS + latency)` reward bit-for-bit — including
        // total insensitivity to the frag_pct argument.
        let cfg = crate::tune::TrainingConfig::default();
        for latency in [0u64, 10, 50, 1000, 1_000_000] {
            let legacy = 50.0 / (50.0 + latency as f64);
            assert_eq!(shaped_reward(latency, 0.0, &cfg), legacy);
            assert_eq!(
                shaped_reward(latency, 87.5, &cfg),
                legacy,
                "frag_pct must be inert at frag_weight = 0"
            );
        }
    }

    #[test]
    fn shaped_reward_frag_weight_penalizes_fragmentation() {
        let cfg = crate::tune::TrainingConfig {
            frag_weight: 0.1,
            ..crate::tune::TrainingConfig::default()
        };
        let tight = shaped_reward(100, 0.0, &cfg);
        let wasteful = shaped_reward(100, 50.0, &cfg);
        // Same latency, 50% internal fragmentation -> reward drops by
        // exactly frag_weight * 0.5.
        assert!((tight - wasteful - 0.05).abs() < 1e-12);
        assert!(wasteful < tight);
    }

    #[test]
    fn shaped_reward_t_ref_sets_curve_knee() {
        // A larger t_ref flattens the curve: the same slow op is punished
        // less (throughput focus tolerates occasional slow ops in exchange
        // for better mean behavior); a small t_ref punishes it harder.
        let latency = 500u64;
        let sharp = crate::tune::TrainingConfig {
            t_ref_ns: 50.0,
            ..crate::tune::TrainingConfig::default()
        };
        let flat = crate::tune::TrainingConfig {
            t_ref_ns: 200.0,
            ..crate::tune::TrainingConfig::default()
        };
        assert!(shaped_reward(latency, 0.0, &flat) > shaped_reward(latency, 0.0, &sharp));
    }

    #[test]
    fn batched_reward_de_quantizes_toward_true_mean() {
        // The A2 fix's premise, as pure arithmetic. A backend whose true
        // per-op cost is 21ns, measured on a 42ns-tick clock: each sample
        // reads either 0ns or 42ns (1:1, so the sample mean is the true
        // 21ns). `shaped_reward` is nonlinear in latency, so the per-op
        // average of the two quantized rewards is BIASED away from the
        // truth, while feeding the averaged latency into `shaped_reward`
        // once (what batching does) lands exactly on it.
        let cfg = crate::tune::TrainingConfig::default(); // t_ref_ns = 50
        let ground_truth = shaped_reward(21, 0.0, &cfg); // 50/71
        let per_op_avg = (shaped_reward(0, 0.0, &cfg) + shaped_reward(42, 0.0, &cfg)) / 2.0;
        let batched = shaped_reward(42 / 2, 0.0, &cfg);

        assert_eq!(batched, ground_truth, "batched == shaped(mean latency)");
        assert!(
            (batched - ground_truth).abs() < (per_op_avg - ground_truth).abs(),
            "batched ({batched:.4}) must be closer to the true reward \
             ({ground_truth:.4}) than the per-op quantized average \
             ({per_op_avg:.4})"
        );
    }

    #[test]
    fn batching_preserves_decisive_ranking_through_bandit() {
        // Feed the bandit batched (de-quantized) rewards directly via the
        // weighted-update path and confirm a decisively-faster arm wins.
        // Uses an explicit config (the process-wide OnceLock can't be
        // safely mutated under parallel tests — same reason `select_with`
        // exists).
        use crate::bandit::BanditPolicy;
        let cfg = crate::tune::TrainingConfig::default();
        let sig = Signature::new(0xABC, 4);
        let mut bandit = BanditPolicy::new();
        // Mirror `record_latency`'s real loop: every selected arm gets a
        // reward, batched by `reward_batch`. Slab is decisively fastest
        // (~5ns); every other backend is ~500ns. Accumulate per-arm and
        // flush in weighted batches, exactly as production does.
        let batch = cfg.reward_batch;
        let mut pending: [(u64, u32); 4] = [(0, 0); 4];
        for _ in 0..(batch as usize * 60) {
            let backend = bandit.select_with(sig, &cfg);
            let latency = if backend == Backend::Slab { 5 } else { 500 };
            let e = &mut pending[backend as usize];
            e.0 += latency;
            e.1 += 1;
            if e.1 >= batch {
                let reward = shaped_reward(e.0 / e.1 as u64, 0.0, &cfg);
                bandit.update_weighted(sig, backend, reward, e.1);
                *e = (0, 0);
            }
        }
        let mut slab_wins = 0;
        for _ in 0..50 {
            if bandit.select_with(sig, &cfg) == Backend::Slab {
                slab_wins += 1;
            }
        }
        assert!(
            slab_wins > 45,
            "de-quantized rewards must let the decisively-faster arm win \
             ({slab_wins}/50 for Slab)"
        );
    }

    #[test]
    fn per_arm_winsorization_trims_spike_but_not_a_steady_arm() {
        // Task B: winsorization is now per-arm against the arm's OWN running
        // quantile (default p90), not a global constant. Two properties:
        //   (1) A near-instant arm (Arena) with one huge cold spike must have
        //       that spike trimmed to its own low quantile, so the spike does
        //       not poison the batch and Arena still outscores a steady slow
        //       arm (Slab at 200ns) — the regime the Phase-1 finding showed
        //       the myopic reward model getting backwards.
        //   (2) A steady arm whose samples all sit in one low bucket is
        //       unaffected (its p90 == its steady cost, nothing to trim).
        // reward_batch default is 16, which is also the warmup floor: the
        // first 16 samples pass unclamped, so we drive 32 samples per arm
        // (two full batches) — the spike lands after warmup and is trimmed.
        let mut state = AllocatorState::new_training();
        let sc = size_class_for(64);
        // Arena: 31 near-instant samples (bucket 0-ish) + 1 giant 1ms spike.
        for _ in 0..31 {
            state.record_fine_latency(7, Backend::Arena, sc, 3, 1, 64);
        }
        state.record_fine_latency(7, Backend::Arena, sc, 3, 1_000_000, 64);
        // Slab: 32 steady 200ns samples.
        for _ in 0..32 {
            state.record_fine_latency(7, Backend::Slab, sc, 3, 200, 64);
        }
        let rows = state.fine_snapshot();
        let row = rows
            .iter()
            .find(|r| r.0 == 7 && r.2 == 3)
            .expect("fine row for (hash 7, ctx 3)");
        let (slab_i, arena_i) = (Backend::Slab as usize, Backend::Arena as usize);
        assert_eq!(row.3[arena_i], 32, "both Arena batches must have flushed");
        assert_eq!(row.3[slab_i], 32, "both Slab batches must have flushed");
        // The Arena arm — one giant spike trimmed to its own low p90 — must
        // still beat the steady 200ns Slab arm.
        assert!(
            row.4[arena_i] > row.4[slab_i],
            "a mostly-instant Arena arm with a winsorized spike ({}) must \
             outscore a steady 200ns Slab arm ({})",
            row.4[arena_i],
            row.4[slab_i]
        );

        // Direct unit check of the estimator: a steady arm's p90 sits at its
        // own bucket (unaffected), and a spike above it is trimmed down.
        let mut hist = LatencyHist::default();
        for _ in 0..100 {
            hist.record(200);
        }
        let cfg = crate::tune::TrainingConfig::default();
        assert_eq!(
            winsorize_latency(&mut hist, 200, &cfg),
            200,
            "a sample at the steady cost is not clamped"
        );
        // A 1ms spike into a distribution of ~200ns samples is trimmed to the
        // steady bucket's upper edge (256), far below 1_000_000.
        let trimmed = winsorize_latency(&mut hist, 1_000_000, &cfg);
        assert!(
            trimmed <= 256,
            "spike must be trimmed to the arm's p90 bucket edge (got {trimmed})"
        );
        // percentile 0 disables winsorization entirely.
        let cfg_off = crate::tune::TrainingConfig {
            clamp_percentile: 0,
            ..cfg
        };
        assert_eq!(
            winsorize_latency(&mut hist, 1_000_000, &cfg_off),
            1_000_000,
            "clamp_percentile=0 disables winsorization"
        );
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
    fn converged_freeze_chain_fires_on_a_decisive_workload() {
        // Drives the exact production convergence path — route() ->
        // record_latency() -> is_converged() — deterministically: a single
        // call site where Slab is ~2000x faster than any other backend. The
        // huge reward gap makes the UCB intervals separate (see
        // `BanditPolicy::is_converged`), so `AllocatorState::is_converged`
        // (which `Lohalloc::is_converged` and the cabi/native converged-mode
        // freeze poll forward to) must flip true. This is the deterministic
        // counterpart to the near-tied real microbenchmarks (slab/arena),
        // where Slab≈Arena legitimately never separates and convergence
        // correctly never fires.
        let mut state = AllocatorState::new_training();
        let hash = 0xC0FFEE;
        let size = 64usize;
        let sc = size_class_for(size);
        assert!(!state.is_converged(), "empty bandit is not converged");

        for _ in 0..3000 {
            let backend = state.route(hash, size);
            // Slab wins decisively; everything else is punished hard.
            let latency = if backend == Backend::Slab { 1 } else { 200_000 };
            state.record_latency(hash, backend, sc, latency, size);
        }
        assert!(
            state.is_converged(),
            "a decisively Slab-favorable workload must converge"
        );
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
            state.record_latency(42, backend, size_class_for(64), 10, 64);
        }
    }

    #[test]
    fn freeze_transitions_to_inference() {
        let mut state = AllocatorState::new_training();
        // Train a bit.
        for _ in 0..10 {
            let backend = state.route(100, 64);
            state.record_latency(100, backend, size_class_for(64), 10, 64);
        }
        assert!(!state.is_inference());
        state.freeze(false, ARENA_DEMOTE_VOLUME_FRACTION, 0.0);
        assert!(state.is_inference());
    }

    #[test]
    fn inference_mode_routes_via_perfect_hash() {
        let mut state = AllocatorState::new_training();
        // Train: make hash 100 → Arena.
        for _ in 0..100 {
            let backend = state.route(100, 64);
            if backend == Backend::Arena {
                state.record_latency(100, backend, size_class_for(64), 10, 64);
            } else {
                // Manually record Arena as winning.
                state.record_latency(100, Backend::Arena, size_class_for(64), 10, 64);
            }
        }
        state.freeze(false, ARENA_DEMOTE_VOLUME_FRACTION, 0.0);

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
            state.record_latency(50, backend, size_class_for(64), 10, 64);
        }
        state.freeze(false, ARENA_DEMOTE_VOLUME_FRACTION, 0.0);

        // record_latency() in Inference mode should be a no-op (not panic).
        state.record_latency(50, Backend::Slab, size_class_for(64), 10, 64);
        // If we get here, it didn't panic.
    }

    #[test]
    fn inference_falls_back_for_unseen_hash() {
        let mut state = AllocatorState::new_training();
        // Train only hash 100.
        for _ in 0..10 {
            let backend = state.route(100, 64);
            state.record_latency(100, backend, size_class_for(64), 10, 64);
        }
        state.freeze(false, ARENA_DEMOTE_VOLUME_FRACTION, 0.0);

        // Hash 999 was never seen → should fall back to size-based default.
        let backend = state.route(999, 64);
        // Size 64 + header 48 = 112 ≤ SLAB_MAX → Slab.
        assert_eq!(backend, Backend::Slab);
    }

    #[test]
    fn inference_falls_back_for_large_size() {
        let mut state = AllocatorState::new_training();
        state.freeze(false, ARENA_DEMOTE_VOLUME_FRACTION, 0.0);
        // Size > BUDDY_MAX → System.
        let backend = state.route(999, 2 * 1024 * 1024);
        assert_eq!(backend, Backend::System);
    }

    #[test]
    fn freeze_clamps_system_lock_for_buddy_range_size() {
        // 20_000 bytes → size_class 12 (> 16 KiB, <= 64 KiB: Buddy range).
        let size = 20_000usize;
        let sc = size_class_for(size);
        assert_eq!(sc, 12);

        let mut state = AllocatorState::new_training();
        for _ in 0..20 {
            let _ = state.route(200, size);
            // Force every observation's reward onto System regardless of
            // what was actually recommended — same pattern as
            // `inference_mode_routes_via_perfect_hash`'s forced-Arena case.
            state.record_latency(200, Backend::System, sc, 10, size);
        }
        state.freeze(false, ARENA_DEMOTE_VOLUME_FRACTION, 0.0);

        let backend = state.route(200, size);
        assert_ne!(
            backend,
            Backend::System,
            "a Signature whose size class fits Buddy must never freeze to System"
        );
        assert_eq!(backend, Backend::Buddy);
    }

    #[test]
    fn clamp_is_a_noop_for_genuinely_large_sizes() {
        // size_class 14 (> 1 MiB) legitimately belongs on System — the
        // clamp must never touch it.
        let size = 2 * 1024 * 1024usize;
        let sc = size_class_for(size);
        assert_eq!(sc, 14);

        let mut state = AllocatorState::new_training();
        for _ in 0..20 {
            let _ = state.route(201, size);
            state.record_latency(201, Backend::System, sc, 10, size);
        }
        state.freeze(false, ARENA_DEMOTE_VOLUME_FRACTION, 0.0);

        assert_eq!(state.route(201, size), Backend::System);
    }

    /// Train a state whose winning arm for `(hash, size)` is Arena — the
    /// forced-reward pattern from `freeze_clamps_system_lock_for_buddy_range_size`.
    fn train_arena_locked(hash: u64, size: usize) -> AllocatorState {
        let sc = size_class_for(size);
        let mut state = AllocatorState::new_training();
        for _ in 0..20 {
            let _ = state.route(hash, size);
            state.record_latency(hash, Backend::Arena, sc, 10, size);
        }
        state
    }

    #[test]
    fn freeze_demotes_arena_when_no_reset_observed() {
        // J4-C: in a reset-free deployment (`demote_arena = true`), an
        // Arena-locked Signature must freeze to the size-appropriate
        // backend instead — Slab for slab-range, Buddy for buddy-range.
        // This is what kills the freeze-at-1000 arena/slab coin flip.
        let mut small = train_arena_locked(300, 64); // sc 0-11 → Slab
        small.freeze(true, ARENA_DEMOTE_VOLUME_FRACTION, 0.0);
        assert_eq!(small.route(300, 64), Backend::Slab);

        let mut mid = train_arena_locked(301, 20_000); // sc 12 → Buddy
        mid.freeze(true, ARENA_DEMOTE_VOLUME_FRACTION, 0.0);
        assert_eq!(mid.route(301, 20_000), Backend::Buddy);
    }

    #[test]
    fn freeze_keeps_arena_when_resets_observed() {
        // A deployment that resets its arena (GUI/replay) keeps Arena
        // verdicts — demotion is strictly opt-in via the flag.
        let mut state = train_arena_locked(302, 64);
        state.freeze(false, ARENA_DEMOTE_VOLUME_FRACTION, 0.0);
        assert_eq!(state.route(302, 64), Backend::Arena);
    }

    #[test]
    fn demote_fraction_zero_is_blanket_demotion() {
        // Bisect knob semantics: 0.0 demotes EVERY Arena verdict (the J4-C
        // blanket behavior) — any signature with pulls > 0 satisfies
        // `pulls/total >= 0.0`.
        let mut state = train_arena_locked(303, 64);
        state.freeze(true, 0.0, 0.0);
        assert_eq!(state.route(303, 64), Backend::Slab);
    }

    #[test]
    fn demote_fraction_above_one_never_demotes() {
        // Bisect knob semantics: > 1.0 keeps every Arena verdict even in a
        // reset-free deployment — no signature can exceed 100% of volume.
        // (train_arena_locked's single signature IS 100% of its table, the
        // heaviest case possible.)
        let mut state = train_arena_locked(304, 64);
        state.freeze(true, 2.0, 0.0);
        assert_eq!(state.route(304, 64), Backend::Arena);
    }

    #[test]
    fn hierarchical_freeze_emits_flagged_coarse_plus_fine_and_probes_by_ctx() {
        // Phase-1 end-to-end at the state level: coarse training settles on
        // Slab; the shadow fine map observes Arena decisively winning under
        // ctx 7 (fed through the real record_fine_latency batching). After
        // freeze, the frozen routing must serve Slab everywhere EXCEPT
        // ctx 7, where the fine entry overrides to Arena.
        let hash = 4242u64;
        let size = 64usize;
        let sc = size_class_for(size);
        let mut state = AllocatorState::new_training();
        // Coarse: Slab fast (high reward), everything else slow.
        for _ in 0..200 {
            let b = state.route(hash, size);
            let lat = if b == Backend::Slab { 1 } else { 20_000 };
            state.record_latency(hash, b, sc, lat, size);
        }
        // Fine, ctx 7: Arena decisively faster than Slab. Enough samples to
        // clear both the reward_batch flush (16) and FINE_MIN_PULLS (32).
        for _ in 0..64 {
            state.record_fine_latency(hash, Backend::Arena, sc, 7, 1, size);
            state.record_fine_latency(hash, Backend::Slab, sc, 7, 20_000, size);
        }
        // demote_arena=false (resets observed): fine Arena verdicts survive.
        state.freeze(false, ARENA_DEMOTE_VOLUME_FRACTION, 0.0);

        let routing = state.routing().expect("inference");
        let key = combine_hash_size_class(hash, sc);
        // Unrelated ctx → coarse verdict.
        assert_eq!(routing.route_main(key, Some((3, 0))), Some(Backend::Slab));
        // No register maintained → coarse verdict (degraded, correct).
        assert_eq!(routing.route_main(key, None), Some(Backend::Slab));
        // ctx 7 → the fine override.
        assert_eq!(routing.route_main(key, Some((7, 0))), Some(Backend::Arena));
        // The coarse entry itself is flagged.
        assert_eq!(
            routing.main.lookup_with_flags(key).map(|(_, f)| f),
            Some(crate::perfect_hash::FLAG_HAS_CONTEXT)
        );
        assert!(routing.has_context_entries());
    }

    #[test]
    fn hierarchical_freeze_demotes_fine_arena_with_heavy_parent() {
        // A fine Arena override on a HEAVY signature in a reset-free
        // deployment must be demoted exactly like its parent would be —
        // otherwise the ctx probe would re-open the J4 fallthrough storm
        // through the fine path.
        let hash = 4343u64;
        let size = 64usize;
        let sc = size_class_for(size);
        let mut state = AllocatorState::new_training();
        for _ in 0..200 {
            let b = state.route(hash, size);
            let lat = if b == Backend::Slab { 1 } else { 20_000 };
            state.record_latency(hash, b, sc, lat, size);
        }
        for _ in 0..64 {
            state.record_fine_latency(hash, Backend::Arena, sc, 7, 1, size);
            state.record_fine_latency(hash, Backend::Slab, sc, 7, 20_000, size);
        }
        // demote_arena=true + fraction 0.0 → every signature is heavy →
        // the fine Arena verdict clamps to Slab == parent → dropped, and
        // with no surviving fine sibling the coarse entry must NOT be
        // flagged (a flag with no siblings is a wasted probe forever).
        state.freeze(true, 0.0, 0.0);
        let routing = state.routing().expect("inference");
        let key = combine_hash_size_class(hash, sc);
        assert_eq!(routing.route_main(key, Some((7, 0))), Some(Backend::Slab));
        assert_eq!(routing.main.lookup_with_flags(key).map(|(_, f)| f), Some(0));
        assert!(!routing.has_context_entries());
    }

    #[test]
    fn arena_demotion_applies_to_distilled_table_too() {
        // The pin cache serves straight from the distilled table with no
        // further checks, so a demoted main entry with an un-demoted
        // distilled sibling would re-open the fallthrough storm through
        // the pin path. Train with a consistent one_frame so distill()
        // emits the site, then check the distilled section.
        let size = 64usize;
        let sc = size_class_for(size);
        let mut state = AllocatorState::new_training();
        for _ in 0..20 {
            let _ = state.route_with_frame(400, 41, size, 0);
            state.record_latency(400, Backend::Arena, sc, 10, size);
        }
        state.freeze(true, ARENA_DEMOTE_VOLUME_FRACTION, 0.0);
        let distilled = state.distilled_table().expect("inference mode");
        let key = combine_hash_size_class(41, sc);
        match distilled.lookup(key) {
            Some(backend) => assert_eq!(
                backend,
                Backend::Slab,
                "distilled entry must be demoted identically to main"
            ),
            None => {
                // Site not distilled (ambiguity rules) — acceptable: no
                // pin-path entry means no un-demoted verdict can leak.
            }
        }
    }

    #[test]
    fn clamp_fixes_unservable_verdicts() {
        // J5: a frozen verdict whose backend cannot serve the size class is
        // a per-op doomed attempt + fallthrough — clamp to the
        // size-appropriate backend. (Slab caps at sc 11; Buddy and Arena at
        // sc 13.) A servable verdict passes through untouched.
        // Slab above SLAB_MAX → Buddy (sc 12/13), System (sc 14).
        assert_eq!(
            clamp_backend_for_size_class(12, Backend::Slab, false),
            Backend::Buddy
        );
        assert_eq!(
            clamp_backend_for_size_class(13, Backend::Slab, false),
            Backend::Buddy
        );
        assert_eq!(
            clamp_backend_for_size_class(14, Backend::Slab, false),
            Backend::System
        );
        // Buddy / Arena above BUDDY_MAX / chunk size → System.
        assert_eq!(
            clamp_backend_for_size_class(14, Backend::Buddy, false),
            Backend::System
        );
        assert_eq!(
            clamp_backend_for_size_class(14, Backend::Arena, false),
            Backend::System
        );
        // Buddy below the headerless order-map floor (sc <= 10: requests
        // <= 8 KiB round below MIN_HEADERLESS_ORDER's 16 KiB and are refused
        // per-op in every load()-booted deployment) → Slab.
        assert_eq!(
            clamp_backend_for_size_class(0, Backend::Buddy, false),
            Backend::Slab
        );
        assert_eq!(
            clamp_backend_for_size_class(10, Backend::Buddy, false),
            Backend::Slab
        );
        // Servable verdicts are untouched (no demote flag).
        assert_eq!(
            clamp_backend_for_size_class(11, Backend::Slab, false),
            Backend::Slab
        );
        assert_eq!(
            clamp_backend_for_size_class(11, Backend::Buddy, false),
            Backend::Buddy,
            "sc 11 requests (8-16 KiB) round UP to the 16 KiB order — servable"
        );
        assert_eq!(
            clamp_backend_for_size_class(13, Backend::Buddy, false),
            Backend::Buddy
        );
        assert_eq!(
            clamp_backend_for_size_class(5, Backend::Arena, false),
            Backend::Arena
        );
    }

    #[test]
    fn volume_selective_demotion_keeps_light_arena() {
        // J5-A: only a HEAVY arena signature (>= ARENA_DEMOTE_VOLUME_FRACTION
        // of the table's training volume) is demoted; a light one keeps its
        // bump-speed win — the adv-mixed case J4-C's blanket demotion
        // flattened (c/adv-mixed 1.10→1.43).
        let size = 64usize;
        let sc = size_class_for(size);
        let mut state = AllocatorState::new_training();
        // Light arena signature: the standard 20-iteration lock-in...
        for _ in 0..20 {
            let _ = state.route(500, size);
            state.record_latency(500, Backend::Arena, sc, 10, size);
        }
        // ...dwarfed by a heavy slab signature (50× the volume), pushing the
        // arena signature's fraction well below the 10% threshold.
        for _ in 0..1000 {
            let _ = state.route(501, 256);
            state.record_latency(501, Backend::Slab, size_class_for(256), 10, 256);
        }
        state.freeze(true, ARENA_DEMOTE_VOLUME_FRACTION, 0.0);
        assert_eq!(
            state.route(500, size),
            Backend::Arena,
            "light arena signature must be KEPT below the volume threshold"
        );
    }

    #[test]
    fn volume_selective_demotion_demotes_heavy_arena_among_light() {
        // The split is per-signature: the heavy burst site is demoted even
        // while a light arena site in the same table is kept.
        let size = 64usize;
        let sc = size_class_for(size);
        let mut state = AllocatorState::new_training();
        // Heavy arena burst site (dominates training volume).
        for _ in 0..1000 {
            let _ = state.route(510, size);
            state.record_latency(510, Backend::Arena, sc, 10, size);
        }
        // Light arena site.
        for _ in 0..20 {
            let _ = state.route(511, size);
            state.record_latency(511, Backend::Arena, sc, 10, size);
        }
        state.freeze(true, ARENA_DEMOTE_VOLUME_FRACTION, 0.0);
        assert_eq!(
            state.route(510, size),
            Backend::Slab,
            "heavy arena signature must be demoted to the size-appropriate backend"
        );
        assert_eq!(
            state.route(511, size),
            Backend::Arena,
            "light arena signature in the same table must be kept"
        );
    }

    #[test]
    fn volume_selective_demotion_distilled_parity_for_light_arena() {
        // A light KEPT arena verdict must be kept in the distilled table too
        // — the pin cache serves distilled with no further checks, so a
        // distilled entry demoted differently from main would change routing
        // through the pin path.
        let size = 64usize;
        let sc = size_class_for(size);
        let mut state = AllocatorState::new_training();
        // Light arena site with a stable one-frame (so distill() emits it).
        for _ in 0..20 {
            let _ = state.route_with_frame(520, 61, size, 0);
            state.record_latency(520, Backend::Arena, sc, 10, size);
        }
        // Heavy slab site on a different one-frame, dominating volume.
        for _ in 0..1000 {
            let _ = state.route_with_frame(521, 62, 256, 0);
            state.record_latency(521, Backend::Slab, size_class_for(256), 10, 256);
        }
        state.freeze(true, ARENA_DEMOTE_VOLUME_FRACTION, 0.0);
        let distilled = state.distilled_table().expect("inference mode");
        let key = combine_hash_size_class(61, sc);
        match distilled.lookup(key) {
            Some(backend) => assert_eq!(
                backend,
                Backend::Arena,
                "light arena distilled entry must be kept, matching main"
            ),
            None => {
                // Site not distilled (ambiguity rules) — acceptable: no
                // pin-path entry means no divergent verdict can leak.
            }
        }
    }

    #[test]
    fn export_load_roundtrip() {
        let mut state = AllocatorState::new_training();
        // Train a few signatures.
        for _ in 0..20 {
            let backend = state.route(1, 64);
            state.record_latency(1, backend, size_class_for(64), 10, 64);
        }
        for _ in 0..20 {
            let backend = state.route(2, 256);
            state.record_latency(2, backend, size_class_for(256), 10, 256);
        }
        state.freeze(false, ARENA_DEMOTE_VOLUME_FRACTION, 0.0);

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
            state.record_latency(42, backend, size_class_for(64), 10, 64);
        }
        state.freeze(false, ARENA_DEMOTE_VOLUME_FRACTION, 0.0);
        let bytes = state.export().unwrap();

        let loaded = AllocatorState::load(&bytes).unwrap();
        assert!(loaded.is_inference());
    }

    #[test]
    #[should_panic(expected = "freeze() called on an already-frozen")]
    fn freeze_twice_panics() {
        let mut state = AllocatorState::new_training();
        state.freeze(false, ARENA_DEMOTE_VOLUME_FRACTION, 0.0);
        state.freeze(false, ARENA_DEMOTE_VOLUME_FRACTION, 0.0); // Should panic.
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
            state.record_latency(100, b, size_class_for(64), 10, 64);
        }
        // Train hash 200 a few times.
        for _ in 0..5 {
            let b = state.route(200, 128);
            state.record_latency(200, b, size_class_for(128), 10, 128);
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
            state.record_latency(42, b, size_class_for(64), 10, 64);
        }
        state.freeze(false, ARENA_DEMOTE_VOLUME_FRACTION, 0.0);
        assert!(state.routing_snapshot().is_empty());
        assert_eq!(state.signature_count(), 0);
    }

    #[test]
    fn reset_to_training_clears_state() {
        let mut state = AllocatorState::new_training();
        for _ in 0..10 {
            let b = state.route(42, 64);
            state.record_latency(42, b, size_class_for(64), 10, 64);
        }
        assert!(!state.is_inference());
        assert_eq!(state.signature_count(), 1);

        state.freeze(false, ARENA_DEMOTE_VOLUME_FRACTION, 0.0);
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
