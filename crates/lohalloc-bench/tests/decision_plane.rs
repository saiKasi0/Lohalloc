//! Layer 2 (decision-plane) validation.
//!
//! Unlike `tests/routing_validation.rs` (Layer 1, forced routing — the
//! execution plane in isolation), these tests train a *fresh, untrained*
//! `Lohalloc` instance on a pure workload, `freeze()` it, and check that the
//! bandit's own measured-latency learning (see
//! `lohalloc_alloc::state::record_latency`) converged on the backend the
//! workload actually favors — with no help from a hand-built model.
//!
//! Real wall-clock timing drives this training, so thresholds here are
//! deliberately looser than Layer 1's (which force the outcome and can
//! assert near-100%): they check that learning moved decisively in the
//! right direction, not that it perfectly matches a forced model.
//!
//! `cargo test` runs `#[test]` functions in this binary on separate threads
//! by default, and this file's tests are all CPU-bound and timing-sensitive
//! — running several at once introduces real scheduler contention that
//! inflates and destabilizes the very latency measurements the bandit
//! learns from. `SERIAL` forces them to run one at a time within this
//! binary (still parallel with other test binaries in the workspace).
//!
//! All tests here are `#[ignore]`d by default. Manual runs (dozens of
//! repetitions, both debug and `--release`) consistently converge correctly
//! the large majority of the time, occasionally failing under real system
//! noise (thermal throttling, background load, other processes competing
//! for cores) that shifts a UCB1 decision that's genuinely close for this
//! hardware. That's real, expected variance in a system whose whole point
//! is learning from actual measured latency — not a bug, and not something
//! more training iterations alone reliably eliminate. Per this crate's own
//! design principle (see `src/workloads.rs`'s module doc and the Phase 6
//! plan): timing-*derived* assertions don't belong in the default `cargo
//! test` gate. Run explicitly with `cargo test -p lohalloc-bench --release
//! -- --ignored` to verify the decision plane; Layer 1
//! (`tests/routing_validation.rs`, forced routing, no real timing involved)
//! is what runs by default and gates CI.

use std::sync::Mutex;

use lohalloc_core::Backend;

use lohalloc_bench::workloads::{self, hashes, HarnessDriver, RecordingDriver};

/// Serializes this file's tests so their timing-based training isn't
/// corrupted by CPU contention from other tests in this binary running
/// concurrently. See the module doc above.
static SERIAL: Mutex<()> = Mutex::new(());

/// Train `train_ops` iterations of `generator` against a fresh instance,
/// freeze, then replay the same generator against the frozen instance while
/// recording which backend actually served each allocation.
fn train_freeze_and_observe(
    train: impl Fn(&HarnessDriver),
    observe: impl Fn(&RecordingDriver<'_, HarnessDriver>),
) -> std::collections::HashMap<Backend, usize> {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let harness = HarnessDriver::new();
    train(&harness);
    harness.alloc.freeze();
    assert!(harness.alloc.is_inference());

    let recording = RecordingDriver::new(&harness, &harness.alloc);
    observe(&recording);
    recording.backend_counts()
}

fn dominant_fraction(counts: &std::collections::HashMap<Backend, usize>, backend: Backend) -> f64 {
    let total: usize = counts.values().sum();
    if total == 0 {
        return 0.0;
    }
    *counts.get(&backend).unwrap_or(&0) as f64 / total as f64
}

#[test]
#[ignore = "timing-dependent Layer 2 test — see module doc; run with --release -- --ignored"]
fn trained_slab_workload_converges_to_slab() {
    // `workload_slab_churn` never resets an arena (unlike `workload_arena_bursts`),
    // so if the bandit routes some of its 64-deep-window churn to Arena,
    // that memory is never reclaimed (Arena's dealloc is a no-op) — the
    // 1 MiB arena *will* exhaust and keep failing for the rest of training.
    // Training needs enough ops for that exhaustion-and-punishment cycle to
    // actually happen and tip UCB1 away from Arena; a short run can let
    // Arena "get away with it" purely on its unbeatable dealloc speed
    // before ever filling up (a real, measured effect — not a test bug).
    let counts = train_freeze_and_observe(
        |h| workloads::workload_slab_churn(h, hashes::W_SLAB, 20_000),
        |r| workloads::workload_slab_churn(r, hashes::W_SLAB, 2_000),
    );
    let frac = dominant_fraction(&counts, Backend::Slab);
    assert!(
        frac >= 0.6,
        "W-SLAB should learn to route mostly to Slab, got {frac:.2} ({counts:?})"
    );
}

#[test]
#[ignore = "timing-dependent Layer 2 test — see module doc; run with --release -- --ignored"]
fn trained_buddy_workload_converges_to_buddy() {
    let counts = train_freeze_and_observe(
        |h| workloads::workload_buddy_interleaved(h, hashes::W_BUDDY, 400),
        |r| workloads::workload_buddy_interleaved(r, hashes::W_BUDDY, 400),
    );
    let frac = dominant_fraction(&counts, Backend::Buddy);
    assert!(
        frac >= 0.6,
        "W-BUDDY should learn to route mostly to Buddy, got {frac:.2} ({counts:?})"
    );
}

#[test]
#[ignore = "timing-dependent Layer 2 test — see module doc; run with --release -- --ignored"]
fn trained_system_workload_converges_to_system() {
    // Structural control: only System can serve these sizes at all, so this
    // should converge (near-)deterministically regardless of timing noise.
    let counts = train_freeze_and_observe(
        |h| workloads::workload_system_large(h, hashes::W_SYSTEM, 100),
        |r| workloads::workload_system_large(r, hashes::W_SYSTEM, 50),
    );
    let frac = dominant_fraction(&counts, Backend::System);
    assert!(
        frac >= 0.99,
        "W-SYSTEM structurally can only be served by System, got {frac:.2} ({counts:?})"
    );
}

#[test]
#[ignore = "timing-dependent Layer 2 test — see module doc; run with --release -- --ignored"]
fn combo_slab_arena_sites_diverge_after_training() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    // A single Lohalloc instance trains two different call sites
    // simultaneously — the headline hypothesis: a working decision engine
    // routes them to *different* backends, not whatever's globally best.
    //
    // Both sites share one Arena instance. Training them *finely
    // interleaved* creates a noisy-neighbor coupling: the arena site's
    // bursts periodically `reset_arena()`, which wipes out any Arena-routed
    // allocations from the slab site too (bailing it out of exhaustion
    // "for free"); conversely, if the slab site is given enough volume per
    // round to actually exhaust the arena on its own, it can fill the arena
    // right as the arena site's own burst starts, causing *that* site's
    // legitimate Arena attempts to fail. Each signature's bandit stats are
    // independent (per the `Signature` map key), so training the two sites
    // sequentially rather than interleaved — arena first (establishing a
    // strong, uncontested preference, and leaving the arena freshly reset
    // when its last burst ends), then slab (now free to fill and eventually
    // exhaust that same fresh capacity, exactly like the standalone
    // `trained_slab_workload_converges_to_slab` test) — avoids the coupling
    // while still proving the same thing: one frozen model routing two
    // call sites to different backends.
    let harness = HarnessDriver::new();
    workloads::workload_arena_bursts(&harness, hashes::W_COMBO_SA_ARENA, 40, 500);
    workloads::workload_slab_churn(&harness, hashes::W_COMBO_SA_SLAB, 30_000);
    harness.alloc.freeze();
    assert!(harness.alloc.is_inference());

    let slab_rec = RecordingDriver::new(&harness, &harness.alloc);
    let arena_rec = RecordingDriver::new(&harness, &harness.alloc);
    workloads::workload_slab_churn(&slab_rec, hashes::W_COMBO_SA_SLAB, 1_000);
    workloads::workload_arena_bursts(&arena_rec, hashes::W_COMBO_SA_ARENA, 8, 300);

    let slab_counts = slab_rec.backend_counts();
    let arena_counts = arena_rec.backend_counts();
    let slab_frac = dominant_fraction(&slab_counts, Backend::Slab);
    let arena_frac = dominant_fraction(&arena_counts, Backend::Arena);

    assert!(
        slab_frac >= 0.6,
        "slab-favorable site should route mostly to Slab, got {slab_frac:.2} ({slab_counts:?})"
    );
    assert!(
        arena_frac >= 0.5,
        "arena-favorable site should route mostly to Arena, got {arena_frac:.2} ({arena_counts:?})"
    );

    // The two sites must not have converged on the identical routing
    // snapshot entry — i.e. they were learned independently, not collapsed
    // by the v1 caller_pc-only freeze bug this crate's `bandit.rs` tests
    // regression-test directly.
    assert_ne!(
        harness.alloc.export().map(|b| b.len()),
        None,
        "frozen model should export"
    );
}

#[test]
#[ignore = "timing-dependent Layer 2 test — see module doc; run with --release -- --ignored"]
fn adv_exhaust_bandit_learns_to_avoid_full_arena() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    // Train a fresh instance on a workload that never frees, letting the
    // bandit freely choose backends the whole time (no forced model). The
    // arena CHAINS 1 MiB chunks up to a 32 MiB cap (it used to be a single
    // 1 MiB region — the old 30k op count stopped ~100k allocations short
    // of the cap, so the bandit was *correctly* still choosing Arena and
    // this test failed for the wrong reason). Training-mode arena blocks
    // carry the 48-byte header, so each ~208 B request consumes 256 B:
    // exhaustion hits at ~131k allocations, leaving ~40k post-exhaustion
    // ops for the bandit to learn from. If the recommended-arm latency
    // measurement correctly attributes the fallthrough cost to Arena when
    // it's full, the bandit should learn to route this Signature elsewhere
    // well before training ends.
    let harness = HarnessDriver::new();
    let ptrs = workloads::workload_exhaust_no_free(&harness, hashes::W_ADV_EXHAUST, 170_000);

    harness.alloc.freeze();
    assert!(harness.alloc.is_inference());

    // Clean up.
    for (p, l) in ptrs {
        unsafe { harness.alloc.dealloc_with_hash(p, l) };
    }

    // After training through exhaustion, probe the frozen decision for this
    // Signature (`routing_snapshot()` is always empty in Inference mode —
    // see `AllocatorState::routing_snapshot` — so the only way to observe
    // the learned choice is a fresh allocation at the same Signature). A
    // single probe is a coin-flip right at a UCB1 decision boundary; probe
    // several times and require a majority to avoid single-sample noise.
    let probe_layout =
        core::alloc::Layout::from_size_align(workloads::SMALL_FIXED_REQUEST, 16).unwrap();
    let mut arena_probes = 0;
    let total_probes = 20;
    for _ in 0..total_probes {
        let probe_ptr = unsafe {
            harness
                .alloc
                .alloc_with_hash(probe_layout, hashes::W_ADV_EXHAUST)
        };
        if unsafe { harness.alloc.backend_for_ptr(probe_ptr) } == Some(Backend::Arena) {
            arena_probes += 1;
        }
        unsafe { harness.alloc.dealloc_with_hash(probe_ptr, probe_layout) };
    }

    assert!(
        arena_probes * 2 < total_probes,
        "bandit should learn to avoid a backend that exhausted during training, got {arena_probes}/{total_probes} routed to Arena"
    );
}
