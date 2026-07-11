//! Calling-paradigm oracle-gap eval (investigation 2026-07-11): what does
//! the 3-frame walk's spatial blindness (recursion depth, deep branching)
//! actually COST, and does the temporal context machinery compensate?
//!
//! The capability probes (`lohalloc-alloc` `paradigm_p1..p5` tests) proved
//! the structure: recursion depths ≥2 and call paths diverging at frame ≥3
//! fold into ONE Signature. This bench prices that: each workload runs the
//! **identical op sequence** in two textual shapes —
//!
//! - **aliased**: the hidden variable (recursion depth / deep caller) is
//!   invisible to the walk — both regimes share one Signature. This is what
//!   real code with deep call graphs looks like to the model.
//! - **split**: the same op sequence through textually distinct call paths
//!   the walk CAN distinguish — the "if the model could see it" world, i.e.
//!   the spatial-oracle upper bound.
//!
//! Rows per workload (real stack walks throughout — allocations go through
//! `GlobalAlloc::alloc`, never the hash-injecting replay path):
//!
//! - `aliased_blanket_{slab,arena}`: forced single-verdict models on the
//!   harvested (runtime) aliased hash — what any static frozen table can
//!   express for the aliased shape. Best of these = **best static**.
//! - `aliased_trained_frozen`: the real learner (current defaults: size-aware
//!   AHR ctx + refined deep escalation) on the aliased shape — does temporal
//!   context recover what spatial blindness lost?
//! - `split_combo_{a}_{b}`: the 2×2 forced grid on the split shape's two
//!   harvested hashes. Best = **spatial oracle**.
//! - `split_trained_frozen`: the learner when the walk can see the paths.
//!
//! **Spatial-blindness cost = min(aliased_blanket_*) / min(split_combo_*).**
//! **Context compensation = aliased_trained_frozen vs min(split_combo_*).**
//!
//! Hashes are harvested at runtime via `routing_snapshot()` — real walked
//! hashes are ASLR-normalized and stable within a binary, so a model built
//! from a probe run routes the timed run's identical call sites. The
//! harvest asserts the probe facts again (aliased = exactly 1 hash, split =
//! exactly 2), so a toolchain-induced walk change fails loudly here too.
//!
//! Same-machine ratios; local M-series numbers are decision-grade for the
//! go/no-go (memory trap (e)).

use core::alloc::{GlobalAlloc, Layout};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use lohalloc_alloc::Lohalloc;
use lohalloc_bench::forced::lohalloc_forced;
use lohalloc_bench::workloads::MID_SLAB_REQUEST;
use lohalloc_core::Backend;

/// Ops per iteration — same arena-budget arithmetic as `context_gap`: at
/// 8 KiB, an all-arena reset-free run of 10k ops routes ~80 MiB, exceeding
/// the ~60-64 MiB scaled budget, so a wrong blanket-Arena verdict pays the
/// fallthrough storm it would pay in production.
const OPS: usize = 10_000;

/// Recursion depths for the two regimes. Both ≥3 — past the collapse
/// threshold in BOTH observed inlining worlds (the window consumes one more
/// recursion frame when the allocator fully inlines into the caller), so the
/// two regimes are INDISTINGUISHABLE in the aliased shape (probe P2's
/// collapse; the harvest assert re-checks per binary).
const BURST_DEPTH: usize = 3;
const CHURN_DEPTH: usize = 8;

// ---------------------------------------------------------------------------
// W-RECURSE: the hidden variable is recursion depth.
// ---------------------------------------------------------------------------

/// Allocate at the bottom of `depth` recursion frames. The trailing
/// `black_box` defeats sibling-call optimization of the recursive frame.
#[inline(never)]
fn rec_alloc(a: &Lohalloc, layout: Layout, depth: usize) -> *mut u8 {
    if depth == 0 {
        let p = unsafe { a.alloc(layout) };
        assert!(!p.is_null(), "recursive alloc failed");
        p
    } else {
        let p = rec_alloc(a, layout, depth - 1);
        let p = core::hint::black_box(p); // data-dependent anchor: un-hoistable
        p
    }
}

/// Textual copy of [`rec_alloc`]: a distinct function, so a distinct leaf/
/// recursive call-site PC set — the split shape routes one regime here.
#[inline(never)]
fn rec_alloc_b(a: &Lohalloc, layout: Layout, depth: usize) -> *mut u8 {
    if depth == 0 {
        let p = unsafe { a.alloc(layout) };
        assert!(!p.is_null(), "recursive alloc failed");
        p
    } else {
        let p = rec_alloc_b(a, layout, depth - 1);
        let p = core::hint::black_box(p); // data-dependent anchor: un-hoistable
        p
    }
}

type RecPath = fn(&Lohalloc, Layout, usize) -> *mut u8;

/// The shared op sequence: alternating phases of depth-`CHURN_DEPTH`
/// alloc/free churn (slab-friendly) and depth-`BURST_DEPTH` burst-hold
/// (arena-friendly), reset-free — `workload_phase_lifetime`'s shape with
/// the regime decided by recursion depth instead of call-site hash.
fn run_recurse(a: &Lohalloc, burst: RecPath, churn: RecPath, ops: usize) {
    let layout = Layout::from_size_align(MID_SLAB_REQUEST, 16).unwrap();
    let phase_len = (ops / 8).max(1);
    let mut done = 0;
    let mut phase = 0usize;
    while done < ops {
        let n = phase_len.min(ops - done);
        if phase % 2 == 0 {
            for _ in 0..n {
                let p = churn(a, layout, core::hint::black_box(CHURN_DEPTH));
                unsafe { a.dealloc(p, layout) };
            }
        } else {
            let mut held = Vec::with_capacity(n);
            for _ in 0..n {
                held.push(burst(a, layout, core::hint::black_box(BURST_DEPTH)));
            }
            for p in held {
                unsafe { a.dealloc(p, layout) };
            }
        }
        done += n;
        phase += 1;
    }
}

// ---------------------------------------------------------------------------
// W-DEEPBRANCH: the hidden variable is the caller ≥3 frames up.
// ---------------------------------------------------------------------------

#[inline(never)]
fn db_helper(a: &Lohalloc, layout: Layout) -> *mut u8 {
    let p = unsafe { a.alloc(layout) };
    let p = core::hint::black_box(p); // defeat sibling-call TCO: keep this frame
    assert!(!p.is_null(), "branch alloc failed");
    p
}
// Shared 2-wrapper spine: fills frames 1–2, so the paths below diverge only
// at frame ≥3 (outside the window) — the aliased shape.
#[inline(never)]
fn db_w1(a: &Lohalloc, layout: Layout) -> *mut u8 {
    let p = db_helper(a, layout);
    let p = core::hint::black_box(p); // defeat sibling-call TCO: keep this frame
    assert!(!p.is_null());
    p
}
#[inline(never)]
fn db_w2(a: &Lohalloc, layout: Layout) -> *mut u8 {
    let p = db_w1(a, layout);
    let p = core::hint::black_box(p); // defeat sibling-call TCO: keep this frame
    assert!(!p.is_null());
    p
}
#[inline(never)]
fn db_path_burst(a: &Lohalloc, layout: Layout) -> *mut u8 {
    let p = db_w2(a, layout);
    let p = core::hint::black_box(p); // defeat sibling-call TCO: keep this frame
    assert!(!p.is_null());
    p
}
#[inline(never)]
fn db_path_churn(a: &Lohalloc, layout: Layout) -> *mut u8 {
    let p = db_w2(a, layout);
    let p = core::hint::black_box(p); // defeat sibling-call TCO: keep this frame
    assert!(!p.is_null());
    p
}
// Split shape: each path owns frames 1–2 (divergence inside the window).
#[inline(never)]
fn db_s_burst_mid(a: &Lohalloc, layout: Layout) -> *mut u8 {
    let p = db_helper(a, layout);
    let p = core::hint::black_box(p); // defeat sibling-call TCO: keep this frame
    assert!(!p.is_null());
    p
}
#[inline(never)]
fn db_split_burst(a: &Lohalloc, layout: Layout) -> *mut u8 {
    let p = db_s_burst_mid(a, layout);
    let p = core::hint::black_box(p); // defeat sibling-call TCO: keep this frame
    assert!(!p.is_null());
    p
}
#[inline(never)]
fn db_s_churn_mid(a: &Lohalloc, layout: Layout) -> *mut u8 {
    let p = db_helper(a, layout);
    let p = core::hint::black_box(p); // defeat sibling-call TCO: keep this frame
    assert!(!p.is_null());
    p
}
#[inline(never)]
fn db_split_churn(a: &Lohalloc, layout: Layout) -> *mut u8 {
    let p = db_s_churn_mid(a, layout);
    let p = core::hint::black_box(p); // defeat sibling-call TCO: keep this frame
    assert!(!p.is_null());
    p
}

type BranchPath = fn(&Lohalloc, Layout) -> *mut u8;

fn run_branch(a: &Lohalloc, burst: BranchPath, churn: BranchPath, ops: usize) {
    let layout = Layout::from_size_align(MID_SLAB_REQUEST, 16).unwrap();
    let phase_len = (ops / 8).max(1);
    let mut done = 0;
    let mut phase = 0usize;
    while done < ops {
        let n = phase_len.min(ops - done);
        if phase % 2 == 0 {
            for _ in 0..n {
                let p = churn(a, layout);
                unsafe { a.dealloc(p, layout) };
            }
        } else {
            let mut held = Vec::with_capacity(n);
            for _ in 0..n {
                held.push(burst(a, layout));
            }
            for p in held {
                unsafe { a.dealloc(p, layout) };
            }
        }
        done += n;
        phase += 1;
    }
}

// ---------------------------------------------------------------------------
// Harvest + grid plumbing.
// ---------------------------------------------------------------------------

/// Run `probe` on a fresh training instance and return the distinct real
/// walked hashes it produced (sentinel excluded), asserting the expected
/// count — the probe facts re-checked at bench time.
fn harvest(probe: impl Fn(&Lohalloc), expect: usize, what: &str) -> Vec<u64> {
    let a = Box::new(Lohalloc::new());
    probe(&a);
    let mut hashes: Vec<u64> = a
        .routing_snapshot()
        .into_iter()
        .map(|(h, _)| h)
        .filter(|&h| h != 0)
        .collect();
    hashes.sort_unstable();
    hashes.dedup();
    assert_eq!(
        hashes.len(),
        expect,
        "{what}: expected {expect} distinct walked hashes, got {:?}",
        hashes
    );
    hashes
}

/// Fresh instance with every harvested hash forced to one backend.
#[inline(never)]
fn forced_blanket(hashes: &[u64], backend: Backend) -> Box<Lohalloc> {
    let triples: Vec<(u64, usize, Backend)> = hashes
        .iter()
        .map(|&h| (h, MID_SLAB_REQUEST, backend))
        .collect();
    Box::new(lohalloc_forced(&triples))
}

/// Fresh instance mapping the burst-path hash to `burst` and the churn-path
/// hash to `churn`.
#[inline(never)]
fn forced_combo(h_burst: u64, burst: Backend, h_churn: u64, churn: Backend) -> Box<Lohalloc> {
    Box::new(lohalloc_forced(&[
        (h_burst, MID_SLAB_REQUEST, burst),
        (h_churn, MID_SLAB_REQUEST, churn),
    ]))
}

/// Train on the workload, mark the training/measurement cluster boundary,
/// freeze — the `context_gap` `trained_frozen` protocol.
fn trained_frozen(run: impl Fn(&Lohalloc)) -> Box<Lohalloc> {
    let a = Box::new(Lohalloc::new());
    run(&a);
    a.reset_arena();
    a.freeze();
    a
}

/// One workload's full grid. `run_aliased`/`run_split` execute the identical
/// op sequence; `probe_burst`/`probe_churn` drive ONLY that regime through
/// the split shape (to identify which harvested hash is which).
#[allow(clippy::too_many_arguments)]
fn gap_group(
    c: &mut Criterion,
    name: &str,
    run_aliased: impl Fn(&Lohalloc) + Copy,
    run_split: impl Fn(&Lohalloc) + Copy,
    probe_burst: impl Fn(&Lohalloc),
    probe_churn: impl Fn(&Lohalloc),
) {
    let aliased_hashes = harvest(run_aliased, 1, name);
    let h_burst = harvest(probe_burst, 1, name)[0];
    let h_churn = harvest(probe_churn, 1, name)[0];
    assert_ne!(h_burst, h_churn, "{name}: split paths must be distinct");

    let mut group = c.benchmark_group(name);
    for backend in [Backend::Slab, Backend::Arena] {
        let label = format!("aliased_blanket_{backend:?}").to_lowercase();
        let hashes = aliased_hashes.clone();
        group.bench_function(&label, |bch| {
            bch.iter_batched(
                || forced_blanket(&hashes, backend),
                |a| run_aliased(&a),
                BatchSize::PerIteration,
            );
        });
    }
    group.bench_function("aliased_trained_frozen", |bch| {
        bch.iter_batched(
            || trained_frozen(run_aliased),
            |a| run_aliased(&a),
            BatchSize::PerIteration,
        );
    });
    for (b, ch) in [
        (Backend::Slab, Backend::Slab),
        (Backend::Arena, Backend::Arena),
        (Backend::Arena, Backend::Slab),
        (Backend::Slab, Backend::Arena),
    ] {
        let label = format!("split_combo_{b:?}_{ch:?}").to_lowercase();
        group.bench_function(&label, |bch| {
            bch.iter_batched(
                || forced_combo(h_burst, b, h_churn, ch),
                |a| run_split(&a),
                BatchSize::PerIteration,
            );
        });
    }
    group.bench_function("split_trained_frozen", |bch| {
        bch.iter_batched(
            || trained_frozen(run_split),
            |a| run_split(&a),
            BatchSize::PerIteration,
        );
    });
    group.finish();
}

fn bench_recurse(c: &mut Criterion) {
    gap_group(
        c,
        "paradigm_gap_recurse",
        |a| run_recurse(a, rec_alloc, rec_alloc, OPS),
        |a| run_recurse(a, rec_alloc, rec_alloc_b, OPS),
        |a| {
            // Burst regime only, through the split shape's burst path.
            let layout = Layout::from_size_align(MID_SLAB_REQUEST, 16).unwrap();
            let held: Vec<*mut u8> = (0..64)
                .map(|_| rec_alloc(a, layout, core::hint::black_box(BURST_DEPTH)))
                .collect();
            for p in held {
                unsafe { a.dealloc(p, layout) };
            }
        },
        |a| {
            let layout = Layout::from_size_align(MID_SLAB_REQUEST, 16).unwrap();
            for _ in 0..64 {
                let p = rec_alloc_b(a, layout, core::hint::black_box(CHURN_DEPTH));
                unsafe { a.dealloc(p, layout) };
            }
        },
    );
}

fn bench_deepbranch(c: &mut Criterion) {
    gap_group(
        c,
        "paradigm_gap_deepbranch",
        |a| run_branch(a, db_path_burst, db_path_churn, OPS),
        |a| run_branch(a, db_split_burst, db_split_churn, OPS),
        |a| {
            let layout = Layout::from_size_align(MID_SLAB_REQUEST, 16).unwrap();
            let held: Vec<*mut u8> = (0..64).map(|_| db_split_burst(a, layout)).collect();
            for p in held {
                unsafe { a.dealloc(p, layout) };
            }
        },
        |a| {
            let layout = Layout::from_size_align(MID_SLAB_REQUEST, 16).unwrap();
            for _ in 0..64 {
                let p = db_split_churn(a, layout);
                unsafe { a.dealloc(p, layout) };
            }
        },
    );
}

criterion_group!(benches, bench_recurse, bench_deepbranch);
criterion_main!(benches);
