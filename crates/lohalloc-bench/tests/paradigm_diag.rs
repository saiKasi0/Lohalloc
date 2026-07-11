//! Paradigm-investigation diagnostic: what does the learner actually FREEZE
//! on the aliased recursion workload, and why? Prints the coarse verdict,
//! the verdict-arm variance (the aliasing meter), and the fine-ctx rows.
//! `#[ignore]`d (route-metrics introspection, informational — run with
//! `cargo test -p lohalloc-bench --release --features route-metrics
//!  --test paradigm_diag -- --ignored --nocapture`).

#![cfg(feature = "route-metrics")]

use core::alloc::{GlobalAlloc, Layout};
use lohalloc_alloc::Lohalloc;
use lohalloc_bench::workloads::MID_SLAB_REQUEST;

#[inline(never)]
fn rec_alloc(a: &Lohalloc, layout: Layout, depth: usize) -> *mut u8 {
    if depth == 0 {
        let p = unsafe { a.alloc(layout) };
        assert!(!p.is_null());
        p
    } else {
        let p = rec_alloc(a, layout, depth - 1);
        let p = core::hint::black_box(p); // data-dependent anchor: un-hoistable
        p
    }
}

fn run_aliased(a: &Lohalloc, ops: usize) {
    let layout = Layout::from_size_align(MID_SLAB_REQUEST, 16).unwrap();
    let phase_len = (ops / 8).max(1);
    let (mut done, mut phase) = (0, 0usize);
    while done < ops {
        let n = phase_len.min(ops - done);
        if phase % 2 == 0 {
            for _ in 0..n {
                let p = rec_alloc(a, layout, core::hint::black_box(8));
                unsafe { a.dealloc(p, layout) };
            }
        } else {
            let held: Vec<*mut u8> = (0..n)
                .map(|_| rec_alloc(a, layout, core::hint::black_box(3)))
                .collect();
            for p in held {
                unsafe { a.dealloc(p, layout) };
            }
        }
        done += n;
        phase += 1;
    }
}

#[test]
#[ignore]
fn diag_aliased_recurse_training_state() {
    let a = Box::new(Lohalloc::new());
    run_aliased(&a, 10_000);

    eprintln!("== variance_snapshot (aliasing meter) ==");
    for (h, sc, pulls, verdict, var) in a.variance_snapshot() {
        eprintln!("  hash={h:#x} sc={sc} pulls={pulls} verdict={verdict:?} variance={var:.5}");
    }
    eprintln!("== fine_snapshot (shallow ctx rows, pulls>0 arms) ==");
    for (h, sc, ctx, pulls, means) in a.fine_snapshot() {
        let total: u32 = pulls.iter().sum();
        if total > 0 {
            eprintln!("  hash={h:#x} sc={sc} ctx={ctx:#04x} pulls={pulls:?} means={means:?}");
        }
    }
    eprintln!("== frozen verdicts (after reset_arena + freeze) ==");
    a.reset_arena();
    a.freeze();
    // Frozen-run routing: the service counters only count the frozen fast
    // path, and this test owns its process, so they read clean here.
    run_aliased(&a, 10_000);
    for b in [
        lohalloc_core::Backend::Slab,
        lohalloc_core::Backend::Buddy,
        lohalloc_core::Backend::System,
        lohalloc_core::Backend::Arena,
    ] {
        eprintln!("  served {b:?} = {}", Lohalloc::route_count(b));
    }
    eprintln!("  fallthroughs = {}", Lohalloc::fallthrough_count());
}

#[test]
#[ignore]
fn diag_same_site_depth_hashes() {
    // Decisive: ONE textual call site, runtime-alternating depth. If the
    // hashes differ, recursion depth reaches the walk window in this
    // binary despite the 3-frame collapse expectation.
    let a = Box::new(Lohalloc::new());
    let layout = Layout::from_size_align(MID_SLAB_REQUEST, 16).unwrap();
    for &d in [2usize, 3, 4, 5, 6, 8, 2, 64].iter() {
        let before: Vec<u64> = a.routing_snapshot().iter().map(|&(h, _)| h).collect();
        let p = rec_alloc(&a, layout, core::hint::black_box(d));
        unsafe { a.dealloc(p, layout) };
        let after: Vec<u64> = a.routing_snapshot().iter().map(|&(h, _)| h).collect();
        let new: Vec<u64> = after
            .iter()
            .filter(|h| !before.contains(h))
            .copied()
            .collect();
        eprintln!("depth {d}: new signatures {new:?} (total {})", after.len());
    }
}
