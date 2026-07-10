//! Step-0 oracle-gap eval: how much would a *context-aware* Decision Engine
//! be worth?
//!
//! Motivation (2026-07-10 investigation): the Decision Engine is saturated
//! on the standard suite — every workload there has a static per-site
//! optimum, which the tabular `(site, size_class)` bandit learns perfectly
//! (`pht_misses ≈ 0`). The J4/J5 arena-demotion saga showed the real gap:
//! sites whose optimum *changes at runtime* (by phase, allocator state, or
//! data) force hand-coded freeze-time heuristics. Before building any
//! context machinery (allocation-history registers, gshare-style contextual
//! keys, TAGE-style adaptive depth), this bench measures the ceiling.
//!
//! Per adversarial workload (`workloads::workload_phase_lifetime` /
//! `workload_arena_fill_churn` / `workload_data_dependent_lifetime` — each
//! parameterized over TWO call-site hashes and using ONE slab class,
//! [`MID_SLAB_REQUEST`]), every variant runs the **identical op sequence**;
//! only the hash-to-backend mapping differs. The full 2×2 combo grid is
//! measured — no assumed per-regime optima (the first version of this bench
//! hand-assigned them and was promptly falsified):
//!
//! - `combo_slab_slab` / `combo_arena_arena`: BOTH regimes on one backend =
//!   what a real single call site looks like to a static frozen table. The
//!   better of these = **best static verdict**.
//! - `combo_slab_arena` / `combo_arena_slab`: each regime routed
//!   independently = what a context-aware model could express. The best of
//!   all four = **oracle**.
//! - `trained`: the real UCB1 bandit on one hash — where the current
//!   learner actually lands (informational; timing-noisy by nature).
//!
//! **Gap = min(static combos) / min(all combos)**, per workload. If a mixed
//! combo is the overall winner, context has measurable value.
//!
//! Decision rule (Step 0 of the context-awareness ladder):
//! - gap < ~5%  → context-awareness is dead for these shapes; stop.
//! - gap 5-15%  → refine the workload shapes before deciding.
//! - gap > ~15-20% → proceed to Phase 1 (AHR + gshare contextual keys +
//!   hierarchical coarse/fine freeze; see the 2026-07-10 /investigate
//!   report — traps: C5 memo invalidation `topology.rs`, PHT pad bytes
//!   `perfect_hash.rs`, no variance in `ArmStats`).
//!
//! Arena-budget arithmetic (why [`MID_SLAB_REQUEST`] and these op counts):
//! the reset-free regimes only present a real tradeoff if the bump arena's
//! budget (32 MiB floor, CPU-scaled to ~60-64 MiB on 15-16-core hosts,
//! 128 MiB ceiling — `arena::scaled_max_chunks`) binds mid-run. At 8 KiB,
//! an all-arena run of 10k ops routes ~80 MiB (exhausts → storm) while a
//! split run routes ≤ ~40 MiB to arena (survives). On very wide hosts
//! (≥32 cores, 128 MiB cap) raise `OPS` accordingly.
//!
//! The gap is a same-machine ratio, so local (macOS M-series) numbers are
//! acceptable for the go/no-go; re-measure on the c9g gate box only if it
//! gates a build decision.

// NOTE: every `iter_batched` here uses `BatchSize::PerIteration`, never
// `SmallInput` — each setup mints a fresh `Lohalloc`, and a batch of them
// alive at once OOMs on wide-stripe hosts (see benches/hypothesis.rs).
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use lohalloc_bench::forced::lohalloc_forced;
use lohalloc_bench::workloads::{self, hashes, HarnessDriver, MID_SLAB_REQUEST};
use lohalloc_core::Backend;

/// Ops per iteration — see the arena-budget arithmetic in the module doc.
const OPS: usize = 10_000;

/// fill-churn runs more ops: only its (reset-free) churn HALF spends arena
/// budget un-reclaimed — the burst half's `phase_end` resets reclaim as they
/// go — so the churn half alone must exceed the ~60-64 MiB budget
/// (24k/2 × 8 KiB = 96 MiB).
const FILL_OPS: usize = 24_000;

/// A forced instance mapping regime A's hash to `a` and regime B's to `b`.
fn combo(hash_a: u64, a: Backend, hash_b: u64, b: Backend) -> HarnessDriver {
    HarnessDriver::with_alloc(lohalloc_forced(&[
        (hash_a, MID_SLAB_REQUEST, a),
        (hash_b, MID_SLAB_REQUEST, b),
    ]))
}

/// One benchmark group for one adversarial workload: the full 2×2
/// regime→backend combo grid plus the real trained bandit.
fn gap_group<F>(c: &mut Criterion, name: &str, hash_a: u64, hash_b: u64, run: F)
where
    F: Fn(&HarnessDriver, u64, u64) + Copy,
{
    let mut group = c.benchmark_group(name);
    for (a, b) in [
        (Backend::Slab, Backend::Slab),
        (Backend::Arena, Backend::Arena),
        (Backend::Slab, Backend::Arena),
        (Backend::Arena, Backend::Slab),
    ] {
        let label = format!("combo_{a:?}_{b:?}").to_lowercase();
        group.bench_function(label, |bch| {
            bch.iter_batched(
                || combo(hash_a, a, hash_b, b),
                // Every variant passes BOTH hashes — the mapping above is
                // what differs. The single-backend combos are semantically
                // identical to a one-hash run (both hashes hit the same
                // verdict), keeping all rows on the same code path.
                |h| run(&h, hash_a, hash_b),
                BatchSize::PerIteration,
            );
        });
    }
    group.bench_function("trained", |bch| {
        bch.iter_batched(
            HarnessDriver::new,
            // The trained row models the real production view: ONE call
            // site, so one hash for both regimes.
            |h| run(&h, hash_a, hash_a),
            BatchSize::PerIteration,
        );
    });
    group.finish();
}

fn bench_phase_lifetime(c: &mut Criterion) {
    gap_group(
        c,
        "context_gap_phase_lifetime",
        hashes::W_PHASE_A,
        hashes::W_PHASE_B,
        |h, a, b| workloads::workload_phase_lifetime(h, a, b, OPS),
    );
}

fn bench_arena_fill_churn(c: &mut Criterion) {
    gap_group(
        c,
        "context_gap_arena_fill_churn",
        hashes::W_FILL_A,
        hashes::W_FILL_B,
        |h, a, b| workloads::workload_arena_fill_churn(h, a, b, FILL_OPS),
    );
}

fn bench_data_dependent_lifetime(c: &mut Criterion) {
    gap_group(
        c,
        "context_gap_data_lifetime",
        hashes::W_DATA_A,
        hashes::W_DATA_B,
        |h, a, b| workloads::workload_data_dependent_lifetime(h, a, b, OPS),
    );
}

criterion_group!(
    benches,
    bench_phase_lifetime,
    bench_arena_fill_churn,
    bench_data_dependent_lifetime
);
criterion_main!(benches);
