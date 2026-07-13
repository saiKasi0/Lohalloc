//! Multi-Armed Bandit policy — the Decision Engine's training-mode router.
//!
//! Uses the **UCB1** (Upper Confidence Bound) algorithm to select which
//! Execution-Plane backend should serve each allocation Signature
//! `(`topological_hash, size_class)`. UCB1 balances exploration (trying
//! underused arms) with exploitation (preferring arms with high average
//! reward), giving us an online learning policy that converges to the best
//! backend for each call site without requiring a separate training pass.
//!
//! # Hysteresis / Dampening
//!
//! A naive UCB1 policy can oscillate ("jitter") between near-equally-good
//! arms under a steady mixed workload. To prevent this, each Signature
//! tracks its `last_choice`. When UCB1 would select a *different* arm than
//! `last_choice`, we apply a `HYSTERESIS_PENALTY` to the new arm's score.
//! The policy only switches if the new arm's penalized score still exceeds
//! the current arm's unpenalized score. This dampens rapid flipping while
//! still allowing the policy to adapt to genuine workload shifts.
//!
//! # Reward Model
//!
//! Phase 6 feeds the bandit **measured latency**, not a static cost model:
//! `lib.rs::route_alloc_inner`/`GlobalAlloc::dealloc` time the actual
//! alloc/dealloc outcome and convert it to a reward via
//! `state::latency_to_reward` (`update()` below receives that reward).
//! `BASELINE_REWARDS` below survives only as each arm's *initial prior* — one
//! virtual pull seeded at `SignatureStats::new()` so a never-seen Signature's
//! first `select()` still has a sensible cold-start ordering:
//!
//! | Backend | Prior reward | Rationale |
//! |---------|--------------|-----------|
//! | Slab    | 1.0          | Fastest: O(1) free-list pop |
//! | Arena   | 0.9          | Fastest for clusters: bump pointer; but no per-alloc free |
//! | Buddy   | 0.8          | Good for medium: split/coalesce overhead |
//! | System  | 0.3          | Slowest: full `mmap` syscall |
//!
//! Every subsequent pull's reward comes from `update()` alone — `select()`
//! no longer injects the prior a second time on every pull (that used to
//! double-count and drown out real signal; see git history for the bug this
//! replaced). The bandit adjusts purely from measured outcomes via UCB1
//! exploration: if Arena consistently serves a Signature fast, its empirical
//! mean reward rises and it gets selected more often; if it starts failing
//! (e.g. exhausted) the measured latency includes the fallthrough cost and
//! its reward collapses.
//!
//! # `#![no_std]` / Zero-Allocation Hot Path
//!
//! **Training mode** uses `BTreeMap<Signature, ArmStats>` which requires
//! `alloc`. Training mode is *not* the zero-allocation hot path — it's the
//! learning phase. Once `freeze()` collapses the bandit into a
//! `PerfectHashTable`, the hot path touches only the read-only table (an
//! O(1) minimal perfect hash lookup), which is zero-allocation.

use std::collections::{BTreeMap, BTreeSet};

use lohalloc_core::{Backend, Signature};

use crate::tune::TrainingConfig;

/// One diagnostic row of the shadow fine map:
/// `(caller_pc, size_class, ctx, pulls[4], means[4])`.
pub type FineSnapshotRow = (u64, u8, u8, [u32; 4], [f64; 4]);

/// Minimum observations of a fine `(Signature, ctx)` arm before its verdict
/// may override the coarse entry at freeze time. Below this, a divergent
/// mean is more likely cold-start noise than a real contextual effect.
pub(crate) const FINE_MIN_PULLS: u32 = 32;

/// Minimum mean-reward margin (fine best arm vs. the coarse verdict's arm,
/// both measured *within the same ctx*) required to emit a fine entry.
/// Guards table bloat: a fine entry that ties the coarse verdict buys
/// nothing but an extra inference probe.
pub(crate) const FINE_MARGIN: f64 = 0.05;

/// Roadmap-D refinement (i): a deep `(Signature, deep_ctx)` may emit an
/// override only when the ctx carries at least this fraction of the site's
/// total training traffic. The absolute `FINE_MIN_PULLS` alone let the
/// certified D A/B's `c/slab` regression through: the deep register splinters
/// a heavily-trained site across many folded contexts, and thin slices of a
/// big site each clear 32 pulls while representing noise-level shares of the
/// traffic. A genuine regime split concentrates in a few dominant contexts
/// (each ≥ 25% in the context_gap workloads); 5% sits well below that and
/// well above the splinter shares (~1/256).
pub(crate) const DEEP_CTX_MIN_FRACTION: f64 = 0.05;

// The former hard-coded constants (`EXPLORATION_C = 2.0`,
// `HYSTERESIS_PENALTY = 0.15`, `BASELINE_REWARDS = [1.0, 0.8, 0.3, 0.9]`)
// now live in `tune::TrainingConfig` (`ucb_c`, `hysteresis`,
// `baseline_rewards`) with the same values as defaults — see `tune.rs`.
// Index order for `baseline_rewards`: [Slab=0, Buddy=1, System=2, Arena=3].

/// Per-arm statistics tracked by the bandit for one Signature.
#[derive(Clone, Debug)]
struct ArmStats {
    /// Sum of all rewards received by this arm.
    sum_reward: f64,
    /// Number of times this arm has been pulled (selected).
    pulls: u32,
    /// Roadmap-D Welford accumulator over the **batch-mean rewards** fed to
    /// [`BanditPolicy::update_weighted`] — each flushed batch counts as ONE
    /// observation regardless of its weight, because the batch means are the
    /// actual reward signal driving UCB and their spread is the regime
    /// spread (within-batch noise is already averaged out by the flush).
    /// Deliberately separate from `sum_reward`/`pulls`, which stay
    /// bit-identical to the pre-D weighted bookkeeping: the variance is a
    /// freeze-time *gate* (see `freeze_fine`'s escalation), never a routing
    /// input. Baseline priors (`new`) are not observations — `obs` starts 0.
    obs: u32,
    obs_mean: f64,
    obs_m2: f64,
}

impl ArmStats {
    fn new(reward: f64) -> Self {
        Self {
            sum_reward: reward,
            pulls: 1,
            obs: 0,
            obs_mean: 0.0,
            obs_m2: 0.0,
        }
    }

    /// Mean reward (average reward per pull).
    fn mean_reward(&self) -> f64 {
        if self.pulls == 0 {
            return 0.0;
        }
        self.sum_reward / (self.pulls as f64)
    }

    /// One Welford step over a batch-mean reward observation.
    #[inline]
    fn observe(&mut self, reward: f64) {
        self.obs += 1;
        let delta = reward - self.obs_mean;
        self.obs_mean += delta / self.obs as f64;
        self.obs_m2 += delta * (reward - self.obs_mean);
    }

    /// Population variance of the observed batch-mean rewards (`0.0` until
    /// two observations exist — a single sample has no spread).
    fn reward_variance(&self) -> f64 {
        if self.obs < 2 {
            return 0.0;
        }
        self.obs_m2 / self.obs as f64
    }
}

/// Per-Signature state: one set of arm stats + hysteresis tracking.
#[derive(Clone, Debug)]
struct SignatureStats {
    /// Stats for each of the 4 backends (indexed by `Backend as usize`).
    arms: [ArmStats; 4],
    /// Total pulls across all arms for this signature.
    total_pulls: u32,
    /// The last backend selected for this signature (for hysteresis).
    last_choice: Option<Backend>,
    /// How many consecutive `select()` calls have picked `last_choice`
    /// without switching. Drives `freeze_mode=converged` (see
    /// [`BanditPolicy::is_converged`]); costs one increment-or-reset per
    /// selection.
    stable_count: u32,
    /// The J2 one-frame (leaf-return-address) topological hash for this
    /// Signature, captured at `select` time (`0` = unknown, e.g. the replay
    /// path which only supplies a full stack hash). Two Signatures that share
    /// a leaf call site but sit under different 3-frame contexts carry the
    /// *same* `one_frame`; `freeze` distills the sites where every such
    /// context agrees on one backend into the cheap 1-frame routing table
    /// (see [`BanditPolicy::freeze`]).
    one_frame: u64,
}

impl SignatureStats {
    fn new(baselines: &[f64; 4]) -> Self {
        Self {
            arms: [
                ArmStats::new(baselines[0]),
                ArmStats::new(baselines[1]),
                ArmStats::new(baselines[2]),
                ArmStats::new(baselines[3]),
            ],
            total_pulls: 4, // Each arm starts with 1 pull.
            last_choice: None,
            stable_count: 0,
            one_frame: 0,
        }
    }
}

/// Passive per-arm shadow statistics for one `(Signature, ctx)` — no
/// hysteresis, no `last_choice`: the fine map never routes during training
/// (routing stays purely coarse, byte-identical to the pre-context policy);
/// it only *observes* which backend the coarse policy used under which
/// history context and what it cost. `pulls` here counts weighted reward
/// observations, not selections.
#[derive(Clone, Debug, Default)]
struct FineStats {
    sum_reward: [f64; 4],
    pulls: [u32; 4],
}

impl FineStats {
    fn mean(&self, arm: usize) -> f64 {
        if self.pulls[arm] == 0 {
            return 0.0;
        }
        self.sum_reward[arm] / self.pulls[arm] as f64
    }
}

/// The freeze-time output of the shadow fine map (see
/// [`BanditPolicy::freeze_fine`]).
pub struct FineFreeze {
    /// `(fine_key, coarse_key, size_class, backend)` — `fine_key` is
    /// `combine_key_ctx(coarse_key, ctx)`; `coarse_key` rides along so the
    /// caller can apply the parent's Arena-demotion (heavy) verdict to the
    /// fine entry.
    pub entries: Vec<(u64, u64, u8, Backend)>,
    /// Coarse combined keys that received at least one fine entry — their
    /// main-table entries get `FLAG_HAS_CONTEXT`.
    pub flagged: BTreeSet<u64>,
    /// `coarse_key → (one_frame, size_class)` for every Signature that
    /// received a fine/deep CANDIDATE — the raw material for the distilled
    /// exclusion set. J6-D2: the caller (`state::freeze`) must exclude only
    /// the groups whose coarse key actually SURVIVES its collapse filter
    /// (i.e. whose main entry really carries a context flag at runtime):
    /// the pin cache serves distilled verdicts with no further checks, so a
    /// pinned context-routed site would silently bypass the ctx probe — but
    /// a site whose candidates all collapsed routes coarse-only anyway, and
    /// excluding it costs pin/lane engagement for zero behavioral
    /// difference (measured: a ~coin-flip per training roll on the C slab
    /// row, from noisy free-rider Buddy candidates that always clamp back
    /// to the parent verdict). Keyed by coarse key so the caller can do
    /// that survivor filtering; shallow and deep candidates both land here.
    pub flagged_frames: BTreeMap<u64, (u64, u8)>,
    /// Roadmap-D: `(deep_fine_key, coarse_key, size_class, backend)` —
    /// `deep_fine_key` is `combine_key_ctx_deep(coarse_key, deep_ctx)`.
    /// Emitted only for variance-escalated sites (see `freeze_fine`).
    pub deep_entries: Vec<(u64, u64, u8, Backend)>,
    /// Coarse combined keys that received at least one deep entry — their
    /// main-table entries get `FLAG_DEEP_CONTEXT`. Disjoint from `flagged`
    /// by construction (a shallow-explained site is never escalated).
    pub deep_flagged: BTreeSet<u64>,
}

/// The Multi-Armed Bandit policy. Owns per-Signature statistics and selects
/// the best backend for each allocation during Training mode.
///
/// After training, call [`freeze`](Self::freeze) to collapse the policy into
/// a flat `(hash, backend)` mapping for the `PerfectHashTable`.
pub struct BanditPolicy {
    stats: BTreeMap<Signature, SignatureStats>,
    /// Phase-1 shadow fine map: per-`(Signature, ctx)` observation
    /// statistics (see [`FineStats`]). Fed alloc-side only — the dealloc
    /// reward path has no context (the history register at *alloc* time is
    /// what routing would see; it is gone by dealloc time). On-policy
    /// caveat: fine arms only observe backends the coarse policy actually
    /// chose under that ctx, so coverage of non-chosen arms comes from
    /// UCB1's residual exploration.
    fine: BTreeMap<(Signature, u8), FineStats>,
    /// Roadmap-D deep shadow fine map, keyed on the **8-event folded**
    /// context (`lib.rs::ahr_deep`) instead of the shallow 3-event one.
    /// Same passive contract as `fine`; only consulted at freeze time, and
    /// only for sites the variance gate escalates (see `freeze_fine`).
    /// Alloc-side only (the `Header` carries just the shallow ctx). Empty
    /// whenever `escalate_variance == 0` — `state::record_fine_deep_latency`
    /// gates the upserts, so disabling the feature also removes its
    /// training cost.
    fine_deep: BTreeMap<(Signature, u8), FineStats>,
}

impl Default for BanditPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl BanditPolicy {
    /// Create a fresh bandit with no observed signatures.
    pub fn new() -> Self {
        Self {
            stats: BTreeMap::new(),
            fine: BTreeMap::new(),
            fine_deep: BTreeMap::new(),
        }
    }

    /// Const constructor for use in `static` contexts. Produces an empty
    /// bandit (no observed signatures).
    pub const fn new_const() -> Self {
        Self {
            stats: BTreeMap::new(),
            fine: BTreeMap::new(),
            fine_deep: BTreeMap::new(),
        }
    }

    /// Select the best backend for the given Signature using UCB1 with
    /// hysteresis, reading knobs from the process-wide
    /// [`tune::config`](crate::tune::config). If the signature is unseen,
    /// it is initialized with the configured baseline rewards.
    pub fn select(&mut self, sig: Signature) -> Backend {
        self.select_with(sig, crate::tune::config())
    }

    /// [`select`](Self::select) with an explicit config — the testable
    /// core (unit tests can't safely mutate the process-wide `OnceLock`
    /// under parallel execution). Records no J2 one-frame hash (`0`), so
    /// sites selected only through this path are never distilled.
    pub fn select_with(&mut self, sig: Signature, cfg: &TrainingConfig) -> Backend {
        self.select_with_frame(sig, 0, cfg, 0)
    }

    /// [`select_with`](Self::select_with) plus the J2 one-frame topological
    /// hash for this call site (`one_frame == 0` means "unknown", identical
    /// to `select_with`). The production alloc path
    /// (`state::AllocatorState::route`) supplies it so `freeze` can build the
    /// distilled 1-frame table; the replay path passes `0`.
    ///
    /// `unservable_mask` (paradigm-investigation fix, "servability-aware
    /// training"): a bit per `Backend as usize` marking arms this instance's
    /// **inference path cannot serve** for this request — currently the
    /// headerless Buddy below `MIN_HEADERLESS_ORDER` (the caller computes
    /// it). Masked arms are never selected, so they can never free-ride on
    /// their fallthrough target's speed: before this, a headerless-training
    /// "Buddy" pull at a small size always fell through to warm Slab and
    /// the cheap outcome was attributed to Buddy, making the un-servable
    /// arm read as the best one — the freeze-time unservable clamp (J5)
    /// then silently rewrote the verdict. Train what you can serve. `0` =
    /// no masking (replay path and every pre-existing caller).
    pub fn select_with_frame(
        &mut self,
        sig: Signature,
        one_frame: u64,
        cfg: &TrainingConfig,
        unservable_mask: u8,
    ) -> Backend {
        let entry = self
            .stats
            .entry(sig)
            .or_insert_with(|| SignatureStats::new(&cfg.baseline_rewards));
        // A given Signature always maps to one leaf call site, so `one_frame`
        // is constant for it; only overwrite when we actually have one, so a
        // later replay-style `0` never erases a real hash.
        if one_frame != 0 {
            entry.one_frame = one_frame;
        }
        let total = entry.total_pulls as f64;

        let mut best = Backend::Slab;
        let mut best_score = f64::MIN;

        for (i, arm) in entry.arms.iter().enumerate() {
            // Servability mask: an arm inference cannot serve is not a
            // candidate (see the method doc). Never masks everything —
            // Slab/System are always servable.
            if unservable_mask & (1 << i) != 0 {
                continue;
            }
            let backend = backend_from_index(i);
            // UCB1: mean_reward + C * sqrt(ln(N) / n_i)
            let exploration = if arm.pulls > 0 {
                cfg.ucb_c * (total.ln() / (arm.pulls as f64)).sqrt()
            } else {
                f64::INFINITY // Unpulled arms have infinite exploration value.
            };
            let mut score = arm.mean_reward() + exploration;

            // Hysteresis: penalize arms that differ from the last choice.
            if let Some(last) = entry.last_choice {
                if backend != last {
                    score -= cfg.hysteresis;
                }
            }

            if score > best_score {
                best_score = score;
                best = backend;
            }
        }

        // Update pull counts, last_choice, and the convergence streak. The
        // reward for this pull arrives later via `update()` (once the real
        // outcome is measured) — `select()` only accounts for the pull
        // itself, it does not inject any reward.
        let arm = &mut entry.arms[best as usize];
        arm.pulls += 1;
        entry.total_pulls += 1;
        entry.stable_count = if entry.last_choice == Some(best) {
            entry.stable_count.saturating_add(1)
        } else {
            0
        };
        entry.last_choice = Some(best);

        best
    }

    /// Record a reward for a (Signature, Backend) pair after an allocation
    /// completes. This updates the arm's statistics, allowing the bandit to
    /// learn from allocation outcomes.
    pub fn update(&mut self, sig: Signature, backend: Backend, reward: f64) {
        self.update_weighted(sig, backend, reward, 1);
    }

    /// Like [`update`](Self::update) but credits `reward` as `weight`
    /// separate observations. This is what keeps `mean_reward =
    /// sum_reward / pulls` a true average under reward batching: `select()`
    /// increments `pulls` once *per op*, but a batched reward flush arrives
    /// only once per `reward_batch` ops (see `state::record_latency`).
    /// Crediting the flushed reward with `weight = batch count` makes the
    /// running mean identical to having recorded that (de-quantized)
    /// reward on each of those pulls — without it, `sum_reward` would grow
    /// ~1/batch as fast as `pulls` and every mean would collapse toward
    /// zero. `weight = 1` is the ordinary per-op path.
    pub fn update_weighted(&mut self, sig: Signature, backend: Backend, reward: f64, weight: u32) {
        let entry = self
            .stats
            .entry(sig)
            .or_insert_with(|| SignatureStats::new(&crate::tune::config().baseline_rewards));
        let arm = &mut entry.arms[backend as usize];
        arm.sum_reward += reward * weight as f64;
        // Note: pulls was already incremented in select() (once per op, so
        // `weight` of them since the last flush). We don't double-count.
        // Roadmap-D: each flushed batch mean is one variance observation
        // (see `ArmStats::observe` — a gate for freeze-time context
        // escalation, never a routing input).
        arm.observe(reward);
    }

    /// Phase-1 shadow fine update: record a weighted reward observation for
    /// `(sig, ctx, backend)`. Purely passive — never consulted by
    /// `select()`, so training routing stays byte-identical; the fine map
    /// only matters at [`freeze_fine`](Self::freeze_fine) time. Unlike
    /// `update_weighted`, `pulls` is counted here (there is no `select()`
    /// counting fine pulls).
    pub fn update_fine(
        &mut self,
        sig: Signature,
        ctx: u8,
        backend: Backend,
        reward: f64,
        weight: u32,
    ) {
        let entry = self.fine.entry((sig, ctx)).or_default();
        let i = backend as usize;
        entry.sum_reward[i] += reward * weight as f64;
        entry.pulls[i] += weight;
    }

    /// Roadmap-D deep counterpart of [`update_fine`](Self::update_fine),
    /// keyed on the 8-event folded context. Same passive contract.
    pub fn update_fine_deep(
        &mut self,
        sig: Signature,
        deep_ctx: u8,
        backend: Backend,
        reward: f64,
        weight: u32,
    ) {
        let entry = self.fine_deep.entry((sig, deep_ctx)).or_default();
        let i = backend as usize;
        entry.sum_reward[i] += reward * weight as f64;
        entry.pulls[i] += weight;
    }

    /// True once every observed Signature's routing has stabilized — the
    /// trigger for `freeze_mode=converged` (checked by the embedding layer
    /// every few ops; the bandit never freezes itself).
    ///
    /// A Signature counts as converged when BOTH hold:
    /// 1. its last `stable_n` consecutive selections all picked the same
    ///    arm (`stable_count >= stable_n`), and
    /// 2. every other arm's mean reward plus a **unit confidence radius**
    ///    `sqrt(ln(total) / pulls_i)` stays below the winner's mean. A
    ///    streak alone can be hysteresis-induced luck; separation means
    ///    more selections cannot plausibly flip the mean ranking that
    ///    `freeze()` will lock in.
    ///
    /// The radius is deliberately the UCB bonus **without** the
    /// exploration constant: UCB1's steady state re-pulls a suboptimal
    /// arm whenever `ucb_c * radius` climbs back above its reward gap Δ,
    /// so with the *full* bonus, separation is structurally unreachable —
    /// arms hover at the re-exploration boundary forever. Dividing out
    /// `ucb_c` makes the criterion exactly right: equilibrium pull counts
    /// (`n_i ≈ ucb_c²·ln N / Δ²`) put the unit radius at ≈ Δ/ucb_c, which
    /// is strictly inside any *real* gap whenever `ucb_c > 1` (default 2),
    /// while a zero gap (genuinely tied arms) never separates.
    ///
    /// An empty bandit is *not* converged (nothing was learned — freezing
    /// would lock in pure size-based fallback).
    pub fn is_converged(&self, stable_n: u32) -> bool {
        if self.stats.is_empty() {
            return false;
        }
        self.stats.values().all(|entry| {
            if entry.stable_count < stable_n {
                return false;
            }
            let best = Self::best_arm_index(entry);
            let total = entry.total_pulls as f64;
            let best_mean = entry.arms[best].mean_reward();
            entry.arms.iter().enumerate().all(|(i, arm)| {
                if i == best {
                    return true;
                }
                if arm.pulls == 0 {
                    return false; // never-pulled arm: interval unbounded
                }
                let radius = (total.ln() / (arm.pulls as f64)).sqrt();
                arm.mean_reward() + radius < best_mean
            })
        })
    }

    /// Collapse the bandit into a flat list of `(combined_key, size_class,
    /// best_backend)` triples, one per observed Signature — the input to
    /// `PerfectHashTable` via `state::AllocatorState::freeze()`.
    /// `combined_key` folds `size_class` into the hash
    /// (`state::combine_hash_size_class`) so two Signatures that share a
    /// call site but differ by size class get distinct frozen entries
    /// instead of silently clobbering each other (the v1 bug this
    /// replaced — see `perfect_hash::PerfectHashTable`'s wire-format v2).
    ///
    /// The "best" backend for a Signature is the one with the highest mean
    /// reward (most reliable, not just most-pulled).
    pub fn freeze(&self) -> Vec<(u64, u8, Backend, u32)> {
        // The 4th element is the Signature's training volume (`total_pulls`)
        // — J5-A's volume-selective Arena demotion needs it at freeze time
        // to tell a heavy burst site (demote: it would exhaust the arena and
        // storm) from a light one (keep: bump-speed win, never fills). See
        // `AllocatorState::freeze`.
        self.stats
            .iter()
            .map(|(sig, stats)| {
                let best = backend_from_index(Self::best_arm_index(stats));
                let key = crate::state::combine_hash_size_class(sig.caller_pc, sig.size_class);
                (key, sig.size_class, best, stats.total_pulls)
            })
            .collect()
    }

    /// Phase-1 hierarchical freeze, fine layer: for every observed
    /// `(Signature, ctx)`, decide whether the context's evidence justifies a
    /// fine entry that OVERRIDES the coarse verdict at inference time.
    ///
    /// `coarse_final` maps each coarse combined key to its FINAL (clamped,
    /// possibly Arena-demoted) verdict — the comparison must run against
    /// what inference will actually serve, not the raw bandit argmax, or a
    /// fine entry could be emitted that merely re-states a pre-clamp
    /// verdict.
    ///
    /// Emission criteria (all must hold — see [`FINE_MIN_PULLS`] /
    /// [`FINE_MARGIN`]):
    /// 1. the fine best arm has ≥ `FINE_MIN_PULLS` weighted observations,
    /// 2. the coarse verdict's arm was ALSO observed within this ctx
    ///    (otherwise the means aren't comparable),
    /// 3. the fine best differs from the coarse verdict,
    /// 4. its mean beats the coarse verdict's *within-ctx* mean by
    ///    ≥ `FINE_MARGIN`.
    ///
    /// A coarse key gaining any fine entry is flagged (`FLAG_HAS_CONTEXT`),
    /// and its `(one_frame, size_class)` group is reported for exclusion
    /// from the distilled table (the pin cache must never serve a
    /// context-routed site).
    ///
    /// # Roadmap-D deep escalation (`escalate_variance > 0`)
    ///
    /// After the shallow pass, sites the shallow context could NOT explain
    /// are considered for **deep** (8-event folded) context: a deep fine
    /// entry is emitted, under the same significance rules, iff additionally
    /// 1. the coarse key gained **no** shallow entry (mutual exclusivity —
    ///    a site is routed at exactly one context depth, so the two entry
    ///    families can never collide in the shared MPHF), and
    /// 2. the parent Signature's **frozen-verdict arm** has batch-reward
    ///    variance ≥ `escalate_variance` (see [`ArmStats::reward_variance`])
    ///    — a low-variance arm has nothing left for context to explain, so
    ///    stable sites never pay a deep probe or a table entry (the TAGE
    ///    "spend history where it pays" guard, and what keeps a spurious
    ///    margin from sampling noise out of the table).
    ///
    /// `escalate_variance == 0` disables the deep pass entirely.
    pub fn freeze_fine(
        &self,
        coarse_final: &BTreeMap<u64, Backend>,
        escalate_variance: f64,
    ) -> FineFreeze {
        let mut out = FineFreeze {
            entries: Vec::new(),
            flagged: BTreeSet::new(),
            flagged_frames: BTreeMap::new(),
            deep_entries: Vec::new(),
            deep_flagged: BTreeSet::new(),
        };
        for ((sig, ctx), fine) in self.fine.iter() {
            let coarse_key = crate::state::combine_hash_size_class(sig.caller_pc, sig.size_class);
            let Some(&coarse_backend) = coarse_final.get(&coarse_key) else {
                continue; // parent didn't freeze (shouldn't happen) — skip
            };
            let Some((best_backend, _)) = Self::fine_override(fine, coarse_backend) else {
                continue;
            };
            let fine_key = crate::state::combine_key_ctx(coarse_key, *ctx);
            out.entries
                .push((fine_key, coarse_key, sig.size_class, best_backend));
            out.flagged.insert(coarse_key);
            if let Some(stats) = self.stats.get(sig) {
                if stats.one_frame != 0 {
                    out.flagged_frames
                        .insert(coarse_key, (stats.one_frame, sig.size_class));
                }
            }
        }
        if escalate_variance > 0.0 {
            // D-iii precompute: distinct size classes observed per call site.
            // A site trained at a single size class gets NO deep escalation —
            // its verdict-arm variance is spike/amortization noise by
            // construction (the certified `c/slab` regression: magazine-refill
            // phase aligns with a folded ctx and reads as a per-ctx "win"
            // whose inference cost — splitting one slab site across two
            // backends' working sets — the per-op reward can't see). Deep
            // context exists for multi-size mixed workloads.
            let mut sizes_per_site: BTreeMap<u64, BTreeSet<u8>> = BTreeMap::new();
            for s in self.stats.keys() {
                sizes_per_site
                    .entry(s.caller_pc)
                    .or_default()
                    .insert(s.size_class);
            }
            // D-ii precompute: each Signature's best SHALLOW margin (best arm
            // mean − coarse arm mean, within a shallow ctx where the coarse
            // arm was observed), whether or not it qualified for emission.
            // A deep override must EXCEED this — if 8 events of history show
            // no more separation than 3 did, the deep margin is depth-flavored
            // noise, not evidence that longer history explains the variance.
            let mut shallow_best_margin: BTreeMap<Signature, f64> = BTreeMap::new();
            for ((sig, _ctx), fine) in self.fine.iter() {
                let coarse_key =
                    crate::state::combine_hash_size_class(sig.caller_pc, sig.size_class);
                let Some(&coarse_backend) = coarse_final.get(&coarse_key) else {
                    continue;
                };
                let coarse_arm = coarse_backend as usize;
                if fine.pulls[coarse_arm] == 0 {
                    continue; // not comparable in this ctx
                }
                let best = (0..4).map(|a| fine.mean(a)).fold(f64::MIN, f64::max);
                let margin = best - fine.mean(coarse_arm);
                let entry = shallow_best_margin.entry(*sig).or_insert(0.0);
                if margin > *entry {
                    *entry = margin;
                }
            }
            for ((sig, deep_ctx), fine) in self.fine_deep.iter() {
                let coarse_key =
                    crate::state::combine_hash_size_class(sig.caller_pc, sig.size_class);
                if out.flagged.contains(&coarse_key) {
                    continue; // shallow already explains this site
                }
                let Some(&coarse_backend) = coarse_final.get(&coarse_key) else {
                    continue;
                };
                // Variance gate: only the frozen verdict's arm spread counts
                // (that arm's reward is what inference will actually live
                // with; the un-chosen arms' spread is exploration noise).
                let Some(stats) = self.stats.get(sig) else {
                    continue;
                };
                let verdict_arm = Self::best_arm_index(stats);
                if stats.arms[verdict_arm].reward_variance() < escalate_variance {
                    continue;
                }
                // D-iii: single-size-class sites are exempt (see precompute).
                if sizes_per_site
                    .get(&sig.caller_pc)
                    .map(|s| s.len())
                    .unwrap_or(0)
                    < 2
                {
                    continue;
                }
                // D-i: the ctx must carry a real share of the site's traffic
                // — thin splinters of a big site cannot emit.
                let ctx_pulls: u32 = fine.pulls.iter().sum();
                if stats.total_pulls == 0
                    || (ctx_pulls as f64 / stats.total_pulls as f64) < DEEP_CTX_MIN_FRACTION
                {
                    continue;
                }
                let Some((best_backend, deep_margin)) = Self::fine_override(fine, coarse_backend)
                else {
                    continue;
                };
                // D-ii: depth must add separation beyond what shallow saw.
                if deep_margin <= shallow_best_margin.get(sig).copied().unwrap_or(0.0) {
                    continue;
                }
                let deep_key = crate::state::combine_key_ctx_deep(coarse_key, *deep_ctx);
                out.deep_entries
                    .push((deep_key, coarse_key, sig.size_class, best_backend));
                out.deep_flagged.insert(coarse_key);
                if stats.one_frame != 0 {
                    out.flagged_frames
                        .insert(coarse_key, (stats.one_frame, sig.size_class));
                }
            }
        }
        out
    }

    /// Shared fine-emission significance test (shallow and deep passes):
    /// returns the overriding backend **and its margin** iff the ctx's best
    /// arm has enough pulls, the coarse arm was observed in-ctx (comparable
    /// means), the best differs from the coarse verdict, and it wins by
    /// `FINE_MARGIN`. The margin feeds the deep pass's dominance gate (D-ii).
    fn fine_override(fine: &FineStats, coarse_backend: Backend) -> Option<(Backend, f64)> {
        let best = (0..4)
            .max_by(|&a, &b| {
                fine.mean(a)
                    .partial_cmp(&fine.mean(b))
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .unwrap_or(0);
        let best_backend = backend_from_index(best);
        let coarse_arm = coarse_backend as usize;
        let margin = fine.mean(best) - fine.mean(coarse_arm);
        if fine.pulls[best] < FINE_MIN_PULLS
            || fine.pulls[coarse_arm] == 0
            || best_backend == coarse_backend
            || margin < FINE_MARGIN
        {
            return None;
        }
        Some((best_backend, margin))
    }

    /// J2 distillation: the subset of call sites that route *unambiguously*
    /// at one frame of context. Grouping every observed Signature by its
    /// `(one_frame, size_class)`, a group qualifies only when **every**
    /// 3-frame context in it resolves to the same best backend — then a
    /// single distilled entry keyed `combine_hash_size_class(one_frame,
    /// size_class)` can stand in for all of them, letting Inference route
    /// that site from just the leaf return address (the pin-hot-sites inline
    /// cache) without the full 3-frame walk or the main-table lookup.
    ///
    /// Sites with `one_frame == 0` (unknown — the replay path) and ambiguous
    /// groups (a shared wrapper whose callers genuinely want different
    /// backends) are omitted; those still route through the full 3-frame main
    /// table, so distillation never changes a routing decision, it only makes
    /// the unambiguous ones cheaper to reach. Returned as the same
    /// `(combined_key, size_class, backend)` triples `freeze` uses.
    pub fn distill(&self) -> Vec<(u64, u8, Backend, u32)> {
        // (one_frame, size_class) -> (agreed backend so far — None once a
        // conflicting backend proves the group ambiguous — and the group's
        // summed training volume). Summing `total_pulls` across the group
        // means a hot 3-frame site distilled together with cold siblings
        // still reads as heavy to J5-A's volume-selective Arena demotion —
        // the pin cache serves this table with no further checks, so its
        // demotion decision must be at least as conservative as main's.
        let mut groups: BTreeMap<(u64, u8), (Option<Backend>, u64)> = BTreeMap::new();
        for (sig, stats) in self.stats.iter() {
            if stats.one_frame == 0 {
                continue; // unknown leaf hash → cannot distill
            }
            let best = backend_from_index(Self::best_arm_index(stats));
            let entry = groups
                .entry((stats.one_frame, sig.size_class))
                .or_insert((Some(best), 0));
            if entry.0 != Some(best) {
                entry.0 = None; // conflicting context → ambiguous
            }
            entry.1 += stats.total_pulls as u64;
        }
        groups
            .into_iter()
            .filter_map(|((one_frame, size_class), (agreed, pulls))| {
                agreed.map(|backend| {
                    let key = crate::state::combine_hash_size_class(one_frame, size_class);
                    (key, size_class, backend, pulls.min(u32::MAX as u64) as u32)
                })
            })
            .collect()
    }

    /// Collapse the bandit into a flat `(caller_pc, best_backend)` snapshot
    /// for the GUI's live "training progress" view
    /// (`state::AllocatorState::routing_snapshot`), *not* the frozen
    /// routing table. Unlike `freeze()`, this keys purely on the call site
    /// — matching the `stack_hash` telemetry records carry — so the GUI can
    /// correlate live per-hash activity with routing state. If a call site
    /// was trained at more than one size class, the displayed backend is
    /// whichever Signature happens to iterate last for that `caller_pc`
    /// (informational only; the authoritative per-size-class routing lives
    /// in the frozen table built by `freeze()`).
    pub fn snapshot(&self) -> Vec<(u64, Backend)> {
        self.stats
            .iter()
            .map(|(sig, stats)| {
                (
                    sig.caller_pc,
                    backend_from_index(Self::best_arm_index(stats)),
                )
            })
            .collect()
    }

    /// The arm index with the highest mean reward for one Signature's
    /// stats. Shared by `freeze()` and `snapshot()`.
    fn best_arm_index(stats: &SignatureStats) -> usize {
        stats
            .arms
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.mean_reward()
                    .partial_cmp(&b.mean_reward())
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Diagnostic snapshot of the shadow fine map:
    /// `(caller_pc, size_class, ctx, pulls[4], means[4])` per observed
    /// `(Signature, ctx)`. Test/route-metrics introspection only.
    #[cfg(any(feature = "route-metrics", test))]
    pub fn fine_snapshot(&self) -> Vec<FineSnapshotRow> {
        self.fine
            .iter()
            .map(|((sig, ctx), f)| {
                (
                    sig.caller_pc,
                    sig.size_class,
                    *ctx,
                    f.pulls,
                    [f.mean(0), f.mean(1), f.mean(2), f.mean(3)],
                )
            })
            .collect()
    }

    /// Diagnostic aliasing meter (paradigm investigation): one row per
    /// Signature — `(caller_pc, size_class, total_pulls, verdict_backend,
    /// verdict_arm_variance)`. A spatially-aliased Signature (two hidden
    /// call paths or recursion depths folded into one hash) whose regimes
    /// want different backends shows up here as HIGH verdict-arm variance —
    /// the Roadmap-D Welford accumulator read as a capability probe instead
    /// of an escalation gate. Test/route-metrics introspection only.
    #[cfg(any(feature = "route-metrics", test))]
    pub fn variance_snapshot(&self) -> Vec<(u64, u8, u32, Backend, f64)> {
        self.stats
            .iter()
            .map(|(sig, stats)| {
                let verdict = Self::best_arm_index(stats);
                (
                    sig.caller_pc,
                    sig.size_class,
                    stats.total_pulls,
                    backend_from_index(verdict),
                    stats.arms[verdict].reward_variance(),
                )
            })
            .collect()
    }

    /// Number of distinct signatures observed.
    pub fn len(&self) -> usize {
        self.stats.len()
    }

    /// True if no signatures have been observed.
    pub fn is_empty(&self) -> bool {
        self.stats.is_empty()
    }
}

/// Map an array index to a `Backend`. The order matches `BASELINE_REWARDS`.
fn backend_from_index(i: usize) -> Backend {
    match i {
        0 => Backend::Slab,
        1 => Backend::Buddy,
        2 => Backend::System,
        3 => Backend::Arena,
        _ => Backend::Slab,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perfect_hash::PerfectHashTable;

    fn sig(hash: u64) -> Signature {
        Signature::new(hash, 0)
    }

    #[test]
    fn bandit_new_signature_starts_equal() {
        let mut bandit = BanditPolicy::new();
        // Unseen signature — should initialize and return a valid backend.
        let backend = bandit.select(sig(42));
        assert!(
            matches!(
                backend,
                Backend::Slab | Backend::Buddy | Backend::System | Backend::Arena
            ),
            "select should return a valid backend"
        );
        // After one select, the signature should be tracked.
        assert_eq!(bandit.len(), 1);
    }

    #[test]
    fn distill_keeps_unambiguous_omits_ambiguous_and_unknown() {
        let cfg = crate::tune::config();
        let mut bandit = BanditPolicy::new();
        const LEAF_A: u64 = 0xAAAA_0001;
        const LEAF_B: u64 = 0xBBBB_0002;

        // Two distinct 3-frame contexts sharing leaf LEAF_A, both converging
        // on Buddy → the (LEAF_A, sc0) group is unambiguous → distilled.
        for h in [0x1001u64, 0x1002] {
            let s = Signature::new(h, 0);
            bandit.select_with_frame(s, LEAF_A, cfg, 0);
            bandit.update(s, Backend::Buddy, 5.0);
        }
        // Two contexts sharing leaf LEAF_B but wanting different backends →
        // ambiguous group → omitted from the distilled table.
        let sc = Signature::new(0x2001, 0);
        bandit.select_with_frame(sc, LEAF_B, cfg, 0);
        bandit.update(sc, Backend::Slab, 5.0);
        let sd = Signature::new(0x2002, 0);
        bandit.select_with_frame(sd, LEAF_B, cfg, 0);
        bandit.update(sd, Backend::Arena, 5.0);
        // A site with no known leaf hash (replay path) → never distilled.
        let se = Signature::new(0x3001, 0);
        bandit.select(se);
        bandit.update(se, Backend::Buddy, 5.0);

        let distilled = bandit.distill();
        assert_eq!(distilled.len(), 1, "only the unambiguous group distills");
        let (key, size_class, backend, pulls) = distilled[0];
        assert_eq!(key, crate::state::combine_hash_size_class(LEAF_A, 0));
        assert_eq!(size_class, 0);
        assert_eq!(backend, Backend::Buddy);
        assert!(
            pulls > 0,
            "distilled entry must carry the group's summed training volume"
        );
    }

    #[test]
    fn bandit_converges_to_dominant_arm() {
        let mut bandit = BanditPolicy::new();
        let s = sig(100);

        // Simulate a workload where Arena always succeeds (high reward)
        // and other backends always fail (reward 0). UCB1's exploration term
        // never fully vanishes (it grows with ln(N)), so give it enough
        // rounds that Arena's accumulated mean reward dominates the residual
        // exploration bonus on the weaker arms.
        for _ in 0..2000 {
            let backend = bandit.select(s);
            if backend == Backend::Arena {
                bandit.update(s, backend, 1.0); // Arena wins
            } else {
                bandit.update(s, backend, 0.0); // others lose
            }
        }

        // After 200 rounds, Arena should be the dominant choice.
        let mut arena_count = 0;
        for _ in 0..50 {
            let backend = bandit.select(s);
            if backend == Backend::Arena {
                arena_count += 1;
            }
            bandit.update(
                s,
                backend,
                if backend == Backend::Arena { 1.0 } else { 0.0 },
            );
        }
        // Arena should be selected most of the time (> 80%).
        assert!(
            arena_count > 40,
            "Arena should dominate after training, got {arena_count}/50"
        );
    }

    #[test]
    fn bandit_hysteresis_prevents_jitter() {
        let mut bandit = BanditPolicy::new();
        let s = sig(200);

        // Feed a steady mixed workload where Slab and Arena are roughly equal.
        // The hysteresis should prevent rapid flipping.
        let mut choices = Vec::new();
        for _ in 0..100 {
            let backend = bandit.select(s);
            // Give both Slab and Arena a moderate reward (near-equal).
            let reward = match backend {
                Backend::Slab => 0.5,
                Backend::Arena => 0.5,
                _ => 0.1, // discourage others
            };
            bandit.update(s, backend, reward);
            choices.push(backend);
        }

        // Count how many times the choice changed.
        let mut switches = 0;
        for i in 1..choices.len() {
            if choices[i] != choices[i - 1] {
                switches += 1;
            }
        }
        // With hysteresis, switches should be bounded well below a random
        // 50/50 coin-flip baseline. UCB1's exploration term never fully
        // decays (it grows with ln(N) for an arm that hasn't been pulled
        // recently), so for two arms with a permanently-tied reward, some
        // residual switching is expected and structural, not a hysteresis
        // failure — the threshold reflects that residual, not zero churn.
        assert!(
            switches < 40,
            "hysteresis should limit switches, got {switches}"
        );
    }

    #[test]
    fn bandit_update_adjusts_weights() {
        let mut bandit = BanditPolicy::new();
        let s = sig(300);

        // Pull a few times.
        let b1 = bandit.select(s);
        bandit.update(s, b1, 0.9);

        let b2 = bandit.select(s);
        bandit.update(s, b2, 0.1);

        // The arm that got 0.9 should have higher mean reward than 0.1. Each
        // arm's sum_reward still includes its initial baseline prior (one
        // virtual pull, seeded in `SignatureStats::new`) on top of the
        // explicit `update()` reward — `select()` itself no longer adds
        // anything.
        let stats = bandit.stats.get(&s).unwrap();
        let arm_high = &stats.arms[b1 as usize];
        let arm_low = &stats.arms[b2 as usize];
        assert!(arm_high.sum_reward > 0.0);
        assert!(arm_low.sum_reward > 0.0);
    }

    #[test]
    fn bandit_snapshot_collapses_to_best() {
        let mut bandit = BanditPolicy::new();
        let s = sig(400);

        // Train heavily toward Arena.
        for _ in 0..100 {
            let backend = bandit.select(s);
            bandit.update(
                s,
                backend,
                if backend == Backend::Arena { 1.0 } else { 0.0 },
            );
        }

        let snapshot = bandit.snapshot();
        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot[0].0, 400);
        assert_eq!(
            snapshot[0].1,
            Backend::Arena,
            "snapshot best should be Arena"
        );
    }

    #[test]
    fn bandit_freeze_distinguishes_size_classes_at_same_call_site() {
        // The v1 bug this replaces: freeze() used to key purely on
        // `caller_pc`, so a helper called at two different size classes
        // (e.g. once with a 64-byte request, once with a 64 KiB request)
        // collapsed into one ambiguous frozen entry. v2 keys on
        // `combine_hash_size_class(caller_pc, size_class)`, so they must
        // now freeze to two distinct entries.
        let mut bandit = BanditPolicy::new();
        let small = Signature::new(777, 0); // size_class 0 (e.g. 8 bytes)
        let large = Signature::new(777, 12); // size_class 12 (e.g. 32 KiB), same call site

        for _ in 0..100 {
            let b = bandit.select(small);
            bandit.update(small, b, if b == Backend::Slab { 1.0 } else { 0.0 });
        }
        for _ in 0..100 {
            let b = bandit.select(large);
            bandit.update(large, b, if b == Backend::Buddy { 1.0 } else { 0.0 });
        }

        let frozen = bandit.freeze();
        assert_eq!(
            frozen.len(),
            2,
            "same call site at two size classes must freeze to two distinct entries"
        );

        let small_key = crate::state::combine_hash_size_class(777, 0);
        let large_key = crate::state::combine_hash_size_class(777, 12);
        assert_ne!(small_key, large_key, "combined keys must differ");

        let map: std::collections::HashMap<u64, Backend> = frozen
            .into_iter()
            .map(|(k, _sc, b, _pulls)| (k, b))
            .collect();
        assert_eq!(map.get(&small_key), Some(&Backend::Slab));
        assert_eq!(map.get(&large_key), Some(&Backend::Buddy));

        // `snapshot()` (the GUI's live pre-freeze view) still emits one row
        // per Signature — it does not itself deduplicate. But its rows key
        // on the raw `caller_pc` alone, so feeding them into a
        // `PerfectHashTable` (as v1 effectively did) collides: both rows
        // share caller_pc=777, and `from_entries`'s last-wins dedup
        // collapses them to a single, ambiguous entry — this is the v1 bug
        // `freeze()`'s combined keys fix.
        let snapshot = bandit.snapshot();
        assert_eq!(snapshot.len(), 2, "snapshot emits one row per Signature");
        let snapshot_table =
            PerfectHashTable::from_entries(snapshot.into_iter().map(|(h, b)| (h, 0, b)).collect());
        assert_eq!(
            snapshot_table.len(),
            1,
            "raw caller_pc collides across size classes once fed into a PerfectHashTable"
        );
    }

    #[test]
    fn bandit_snapshot_multiple_signatures() {
        let mut bandit = BanditPolicy::new();

        // Signature A → Slab
        for _ in 0..50 {
            let b = bandit.select(sig(1));
            bandit.update(sig(1), b, if b == Backend::Slab { 1.0 } else { 0.0 });
        }
        // Signature B → Buddy
        for _ in 0..50 {
            let b = bandit.select(sig(2));
            bandit.update(sig(2), b, if b == Backend::Buddy { 1.0 } else { 0.0 });
        }

        let snapshot = bandit.snapshot();
        assert_eq!(snapshot.len(), 2);

        let map: std::collections::HashMap<u64, Backend> = snapshot.into_iter().collect();
        assert_eq!(map.get(&1), Some(&Backend::Slab));
        assert_eq!(map.get(&2), Some(&Backend::Buddy));
    }

    #[test]
    fn bandit_empty_is_empty() {
        let bandit = BanditPolicy::new();
        assert!(bandit.is_empty());
        assert_eq!(bandit.len(), 0);
    }

    #[test]
    fn freeze_fine_emits_only_significant_divergent_contexts() {
        let mut bandit = BanditPolicy::new();
        let s = sig(900);

        // Coarse: Slab wins overall.
        for _ in 0..100 {
            let b = bandit.select(s);
            bandit.update(s, b, if b == Backend::Slab { 1.0 } else { 0.1 });
        }
        // ctx 7: Arena decisively better, BOTH arms observed, enough pulls.
        bandit.update_fine(s, 7, Backend::Arena, 0.9, FINE_MIN_PULLS);
        bandit.update_fine(s, 7, Backend::Slab, 0.2, FINE_MIN_PULLS);
        // ctx 3: Arena better but UNDER the pull threshold → not emitted.
        bandit.update_fine(s, 3, Backend::Arena, 0.9, FINE_MIN_PULLS - 1);
        bandit.update_fine(s, 3, Backend::Slab, 0.2, FINE_MIN_PULLS);
        // ctx 5: Arena "better" but within the margin → not emitted.
        bandit.update_fine(s, 5, Backend::Arena, 0.50, FINE_MIN_PULLS);
        bandit.update_fine(s, 5, Backend::Slab, 0.49, FINE_MIN_PULLS);
        // ctx 9: coarse arm never observed in-ctx → not comparable → skip.
        bandit.update_fine(s, 9, Backend::Arena, 0.9, FINE_MIN_PULLS);

        let coarse_key = crate::state::combine_hash_size_class(900, 0);
        let final_map: BTreeMap<u64, Backend> = [(coarse_key, Backend::Slab)].into_iter().collect();
        let fine = bandit.freeze_fine(&final_map, 0.0);

        assert_eq!(fine.entries.len(), 1, "only ctx 7 qualifies");
        let (fine_key, parent, sc, backend) = fine.entries[0];
        assert_eq!(parent, coarse_key);
        assert_eq!(sc, 0);
        assert_eq!(backend, Backend::Arena);
        assert_eq!(fine_key, crate::state::combine_key_ctx(coarse_key, 7));
        assert!(fine.flagged.contains(&coarse_key));
    }

    #[test]
    fn arm_variance_tracks_batch_mean_spread() {
        // Roadmap-D: `update_weighted` runs one Welford observation per
        // flushed batch mean, independent of `weight`. A regime-split arm
        // (rewards alternating 0.1 / 0.5) must read high variance; a steady
        // arm (all 0.3) must read ~zero — the escalation gate's signal.
        let mut bandit = BanditPolicy::new();
        let split = sig(910);
        let steady = sig(911);
        for i in 0..40 {
            let r = if i % 2 == 0 { 0.1 } else { 0.5 };
            bandit.update_weighted(split, Backend::Slab, r, 16);
            bandit.update_weighted(steady, Backend::Slab, 0.3, 16);
        }
        let v_split = bandit.stats[&split].arms[Backend::Slab as usize].reward_variance();
        let v_steady = bandit.stats[&steady].arms[Backend::Slab as usize].reward_variance();
        // Alternating 0.1/0.5 → population variance = 0.04 exactly.
        assert!(
            (v_split - 0.04).abs() < 1e-12,
            "split arm variance {v_split} != 0.04"
        );
        assert!(v_steady < 1e-12, "steady arm variance {v_steady} != 0");
        // Weighted mean bookkeeping unchanged by the observation step:
        // baseline 1.0 + 16 × (20×0.1 + 20×0.5) = 193.
        let sum = bandit.stats[&split].arms[Backend::Slab as usize].sum_reward;
        assert!(
            (sum - 193.0).abs() < 1e-9,
            "sum_reward still weighted: {sum}"
        );
    }

    #[test]
    fn freeze_fine_escalates_deep_only_for_high_variance_unexplained_sites() {
        // Roadmap-D gates, all four cases:
        //  A: high-variance verdict arm + divergent deep ctx + NO shallow
        //     override  → deep entry emitted, FLAG_DEEP parent.
        //  B: LOW variance, same divergent deep evidence → NOT emitted (the
        //     table-bloat guard vs. spurious margins).
        //  C: high variance but the SHALLOW map already explains the site →
        //     deep skipped (mutual exclusivity).
        //  D: escalate_variance == 0 → no deep pass at all.
        let mk = |hash: u64, variance_split: bool| {
            let mut bandit = BanditPolicy::new();
            let s = sig(hash);
            for i in 0..100 {
                let b = bandit.select(s);
                let base = if b == Backend::Slab { 1.0 } else { 0.1 };
                // variance_split arms alternate around the base.
                let r = if variance_split && i % 2 == 0 {
                    base - 0.4
                } else {
                    base
                };
                bandit.update(s, b, r);
            }
            // D-iii: make the call site multi-size (a second size class at
            // the same caller_pc) so the single-size exemption doesn't gate
            // the cases below — the exemption has its own dedicated test.
            bandit.update_weighted(Signature::new(hash, 1), Backend::Slab, 0.5, 1);
            (bandit, s)
        };

        // Case A: high variance, deep divergence, no shallow entry.
        let (mut bandit, s) = mk(920, true);
        bandit.update_fine_deep(s, 42, Backend::Arena, 0.9, FINE_MIN_PULLS);
        bandit.update_fine_deep(s, 42, Backend::Slab, 0.2, FINE_MIN_PULLS);
        let coarse_key = crate::state::combine_hash_size_class(920, 0);
        let final_map: BTreeMap<u64, Backend> = [(coarse_key, Backend::Slab)].into_iter().collect();
        let out = bandit.freeze_fine(&final_map, 0.01);
        assert_eq!(out.deep_entries.len(), 1, "case A must escalate");
        let (deep_key, parent, _, backend) = out.deep_entries[0];
        assert_eq!(parent, coarse_key);
        assert_eq!(backend, Backend::Arena);
        assert_eq!(deep_key, crate::state::combine_key_ctx_deep(coarse_key, 42));
        assert!(out.deep_flagged.contains(&coarse_key));
        assert!(!out.flagged.contains(&coarse_key), "no shallow flag");
        // Case D on the same evidence: disabled → nothing.
        let out_off = bandit.freeze_fine(&final_map, 0.0);
        assert!(out_off.deep_entries.is_empty(), "case D: 0 disables");

        // Case B: same deep evidence, but a STEADY (low-variance) verdict arm.
        let (mut bandit, s) = mk(921, false);
        bandit.update_fine_deep(s, 42, Backend::Arena, 0.9, FINE_MIN_PULLS);
        bandit.update_fine_deep(s, 42, Backend::Slab, 0.2, FINE_MIN_PULLS);
        let coarse_key = crate::state::combine_hash_size_class(921, 0);
        let final_map: BTreeMap<u64, Backend> = [(coarse_key, Backend::Slab)].into_iter().collect();
        let out = bandit.freeze_fine(&final_map, 0.01);
        assert!(
            out.deep_entries.is_empty(),
            "case B: low-variance site must not escalate"
        );

        // Case C: high variance but shallow already explains it.
        let (mut bandit, s) = mk(922, true);
        bandit.update_fine(s, 7, Backend::Arena, 0.9, FINE_MIN_PULLS);
        bandit.update_fine(s, 7, Backend::Slab, 0.2, FINE_MIN_PULLS);
        bandit.update_fine_deep(s, 42, Backend::Arena, 0.9, FINE_MIN_PULLS);
        bandit.update_fine_deep(s, 42, Backend::Slab, 0.2, FINE_MIN_PULLS);
        let coarse_key = crate::state::combine_hash_size_class(922, 0);
        let final_map: BTreeMap<u64, Backend> = [(coarse_key, Backend::Slab)].into_iter().collect();
        let out = bandit.freeze_fine(&final_map, 0.01);
        assert_eq!(out.entries.len(), 1, "shallow entry emitted");
        assert!(
            out.deep_entries.is_empty(),
            "case C: shallow-explained site must not also escalate"
        );
    }

    /// Shared fixture for the D-refinement gate tests: a high-variance,
    /// MULTI-size site with deep evidence that passes every original gate.
    /// Each refinement test then violates exactly one refinement gate.
    fn escalation_fixture(hash: u64) -> (BanditPolicy, Signature, u64, BTreeMap<u64, Backend>) {
        let mut bandit = BanditPolicy::new();
        let s = sig(hash);
        for i in 0..100 {
            let b = bandit.select(s);
            let base = if b == Backend::Slab { 1.0 } else { 0.1 };
            let r = if i % 2 == 0 { base - 0.4 } else { base };
            bandit.update(s, b, r); // high-variance verdict arm
        }
        bandit.update_weighted(Signature::new(hash, 1), Backend::Slab, 0.5, 1); // multi-size
        let coarse_key = crate::state::combine_hash_size_class(hash, 0);
        let final_map: BTreeMap<u64, Backend> = [(coarse_key, Backend::Slab)].into_iter().collect();
        (bandit, s, coarse_key, final_map)
    }

    #[test]
    fn deep_escalation_requires_ctx_traffic_fraction() {
        // D-i: a splintered site — deep evidence spread across many contexts,
        // each clearing FINE_MIN_PULLS but carrying a thin share of a large
        // site's traffic — must not emit; a DOMINANT ctx must. Same total
        // deep evidence per ctx, only the site's total_pulls differ.
        let (mut bandit, s, _key, final_map) = escalation_fixture(930);
        // Inflate the site's training volume far beyond the ctx's share
        // (total_pulls ≈ 100 fixture + 3000), keeping the verdict arm
        // HIGH-variance throughout (alternating rewards, like the fixture)
        // so only the fraction gate distinguishes the two cases below.
        for i in 0..3000 {
            let b = bandit.select(s);
            let base = if b == Backend::Slab { 1.0 } else { 0.1 };
            bandit.update(s, b, if i % 2 == 0 { base - 0.4 } else { base });
        }
        // Deep ctx with 64+64 pulls: 128 / ~3100 ≈ 4% < DEEP_CTX_MIN_FRACTION.
        bandit.update_fine_deep(s, 42, Backend::Arena, 0.9, 2 * FINE_MIN_PULLS);
        bandit.update_fine_deep(s, 42, Backend::Slab, 0.2, 2 * FINE_MIN_PULLS);
        let out = bandit.freeze_fine(&final_map, 0.01);
        assert!(
            out.deep_entries.is_empty(),
            "a thin-slice ctx (~4% of traffic) must not emit"
        );
        // Grow the ctx's evidence to a dominant share (~50%) → emits.
        bandit.update_fine_deep(s, 42, Backend::Arena, 0.9, 700);
        bandit.update_fine_deep(s, 42, Backend::Slab, 0.2, 700);
        let out = bandit.freeze_fine(&final_map, 0.01);
        assert_eq!(out.deep_entries.len(), 1, "a dominant ctx must emit");
    }

    #[test]
    fn deep_escalation_requires_margin_beyond_shallow() {
        // D-ii: if the SHALLOW map already shows as much separation as the
        // deep map (even without qualifying for an override), depth added no
        // information — the deep margin is depth-flavored noise. Deep margin
        // 0.5 vs shallow best margin 0.6 → blocked; shrink shallow's margin
        // below 0.5 → emitted.
        let (mut bandit, s, _key, final_map) = escalation_fixture(931);
        // Deep ctx: Arena 0.7 vs Slab 0.2 → margin 0.5, all gates pass.
        bandit.update_fine_deep(s, 42, Backend::Arena, 0.7, FINE_MIN_PULLS);
        bandit.update_fine_deep(s, 42, Backend::Slab, 0.2, FINE_MIN_PULLS);
        // Shallow ctx shows a LARGER separation (0.9 vs 0.2 = 0.7) but under
        // the pull threshold, so it emitted nothing (site stays unflagged).
        bandit.update_fine(s, 7, Backend::Arena, 0.9, FINE_MIN_PULLS - 1);
        bandit.update_fine(s, 7, Backend::Slab, 0.2, FINE_MIN_PULLS - 1);
        let out = bandit.freeze_fine(&final_map, 0.01);
        assert!(out.entries.is_empty(), "shallow under-pulled: no override");
        assert!(
            out.deep_entries.is_empty(),
            "deep margin (0.5) <= shallow best margin (0.7): depth adds nothing"
        );

        // Fresh site where shallow shows only 0.1 separation → deep's 0.5 wins.
        let (mut bandit, s, _key, final_map) = escalation_fixture(932);
        bandit.update_fine_deep(s, 42, Backend::Arena, 0.7, FINE_MIN_PULLS);
        bandit.update_fine_deep(s, 42, Backend::Slab, 0.2, FINE_MIN_PULLS);
        bandit.update_fine(s, 7, Backend::Arena, 0.3, FINE_MIN_PULLS - 1);
        bandit.update_fine(s, 7, Backend::Slab, 0.2, FINE_MIN_PULLS - 1);
        let out = bandit.freeze_fine(&final_map, 0.01);
        assert_eq!(
            out.deep_entries.len(),
            1,
            "deep margin (0.5) > shallow best (0.1): depth explains the variance"
        );
    }

    #[test]
    fn deep_escalation_exempts_single_size_class_sites() {
        // D-iii: the certified c/slab regression — a site trained at exactly
        // ONE size class must never escalate (its variance is spike /
        // amortization noise by construction), even with otherwise-perfect
        // deep evidence. The same evidence with a second size class at the
        // site → emits.
        let mut bandit = BanditPolicy::new();
        let s = sig(940);
        for i in 0..100 {
            let b = bandit.select(s);
            let base = if b == Backend::Slab { 1.0 } else { 0.1 };
            bandit.update(s, b, if i % 2 == 0 { base - 0.4 } else { base });
        }
        bandit.update_fine_deep(s, 42, Backend::Arena, 0.9, FINE_MIN_PULLS);
        bandit.update_fine_deep(s, 42, Backend::Slab, 0.2, FINE_MIN_PULLS);
        let coarse_key = crate::state::combine_hash_size_class(940, 0);
        let final_map: BTreeMap<u64, Backend> = [(coarse_key, Backend::Slab)].into_iter().collect();
        let out = bandit.freeze_fine(&final_map, 0.01);
        assert!(
            out.deep_entries.is_empty(),
            "single-size-class site must be exempt from escalation"
        );
        // Add a second size class at the same call site → eligible.
        bandit.update_weighted(Signature::new(940, 3), Backend::Slab, 0.5, 1);
        let out = bandit.freeze_fine(&final_map, 0.01);
        assert_eq!(out.deep_entries.len(), 1, "multi-size site escalates");
    }

    #[test]
    fn config_plumbs_through_ucb_c_zero_is_greedy() {
        // With ucb_c = 0 (and no hysteresis) the policy is purely greedy:
        // after one arm's mean reward dominates, exploration never pulls
        // the others again — observable as a perfect selection streak. The
        // default ucb_c = 2.0 provably keeps exploring (ln N grows), so a
        // long unbroken streak also proves the config value is actually
        // read, not the old hard-coded constant.
        let cfg = TrainingConfig {
            ucb_c: 0.0,
            hysteresis: 0.0,
            ..TrainingConfig::default()
        };
        let mut bandit = BanditPolicy::new();
        let s = sig(500);

        // Establish Arena as dominant with a few shaped pulls.
        for _ in 0..8 {
            let b = bandit.select_with(s, &cfg);
            bandit.update(s, b, if b == Backend::Arena { 1.0 } else { 0.0 });
        }
        // Greedy phase: every selection must now be Arena, no exceptions.
        for i in 0..200 {
            let b = bandit.select_with(s, &cfg);
            assert_eq!(b, Backend::Arena, "greedy run broke at pull {i}");
            bandit.update(s, b, 1.0);
        }
    }

    #[test]
    fn converges_on_stable_workload_not_on_adversarial() {
        let cfg = TrainingConfig::default();
        let stable_n = 32;

        // Stable: Arena always wins by a wide margin -> must converge.
        let mut stable = BanditPolicy::new();
        let s = sig(600);
        for _ in 0..3000 {
            let b = stable.select_with(s, &cfg);
            stable.update(s, b, if b == Backend::Arena { 1.0 } else { 0.05 });
        }
        assert!(
            stable.is_converged(stable_n),
            "a decisively-won workload must report convergence"
        );

        // Adversarial: two arms permanently tied -> UCB intervals never
        // separate, must NOT converge no matter how long it runs.
        let mut tied = BanditPolicy::new();
        let t = sig(601);
        for _ in 0..3000 {
            let b = tied.select_with(t, &cfg);
            let r = match b {
                Backend::Slab | Backend::Arena => 0.5,
                _ => 0.5,
            };
            tied.update(t, b, r);
        }
        assert!(
            !tied.is_converged(stable_n),
            "permanently-tied arms must never report convergence"
        );
    }

    #[test]
    fn empty_bandit_is_not_converged() {
        let bandit = BanditPolicy::new();
        assert!(
            !bandit.is_converged(1),
            "freezing an empty bandit would lock in pure fallback routing"
        );
    }
}
