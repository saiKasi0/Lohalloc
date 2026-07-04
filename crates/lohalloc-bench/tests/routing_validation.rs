//! Layer 1 (execution-plane) routing validation.
//!
//! Forces each hash in `lohalloc_bench::hypotheses::ROUTING_HYPOTHESES` to a
//! specific backend via a hand-built `.lohalloc` model, drives the matching
//! workload generator, and asserts the observed routing distribution meets
//! the hypothesis's threshold — including deliberately-wrong forcings that
//! must fall through the size-guard chain (Slab can't serve >16 KiB, etc.)
//! without corrupting state.

use lohalloc_alloc::Lohalloc;
use lohalloc_core::Backend;

use lohalloc_bench::forced::forced_model_bytes;
use lohalloc_bench::hypotheses::ROUTING_HYPOTHESES;
use lohalloc_bench::workloads::{
    self, hashes, AllocDriver, HarnessDriver, RecordingDriver, BUDDY_SIZES, SMALL_FIXED_REQUEST,
};

/// Drive the workload generator matching `hash` for a fixed op count chosen
/// to comfortably exercise the backend in question (and, for W-ADV-EXHAUST,
/// to overflow the 1 MiB arena).
fn run_generator_for_hash<D: AllocDriver>(driver: &D, hash: u64) {
    match hash {
        h if h == hashes::W_SLAB
            || h == hashes::W_COMBO_SA_SLAB
            || h == hashes::W_COMBO_BA_SMALL =>
        {
            workloads::workload_slab_churn(driver, hash, 20_000);
        }
        h if h == hashes::W_ARENA || h == hashes::W_COMBO_SA_ARENA => {
            workloads::workload_arena_bursts(driver, hash, 20, 500);
        }
        h if h == hashes::W_BUDDY || h == hashes::W_COMBO_BA_BUDDY => {
            workloads::workload_buddy_interleaved(driver, hash, 2_000);
        }
        h if h == hashes::W_SYSTEM => {
            workloads::workload_system_large(driver, hash, 200);
        }
        _ => panic!("no generator registered for hash {hash:#x}"),
    }
}

#[test]
fn layer1_forced_routing_matches_hypotheses() {
    for hyp in ROUTING_HYPOTHESES {
        let model = forced_model_bytes(&[(hyp.hash, hyp.size, hyp.forced_backend)]);
        let alloc = Lohalloc::new();
        assert!(alloc.load(&model), "{}: model load failed", hyp.id);
        let harness = HarnessDriver { alloc };
        let recording = RecordingDriver::new(&harness, &harness.alloc);

        run_generator_for_hash(&recording, hyp.hash);

        let fraction = recording.fraction_served_by(hyp.expected_backend);
        assert!(
            fraction >= hyp.min_fraction,
            "{}: expected >= {:.2} fraction served by {:?}, got {:.4} ({})",
            hyp.id,
            hyp.min_fraction,
            hyp.expected_backend,
            fraction,
            hyp.description
        );
    }
}

#[test]
fn combo_slab_arena_diverges_per_site() {
    let model = forced_model_bytes(&[
        (hashes::W_COMBO_SA_SLAB, SMALL_FIXED_REQUEST, Backend::Slab),
        (
            hashes::W_COMBO_SA_ARENA,
            SMALL_FIXED_REQUEST,
            Backend::Arena,
        ),
    ]);
    let alloc = Lohalloc::new();
    assert!(alloc.load(&model), "combo model load failed");
    let harness = HarnessDriver { alloc };

    let slab_rec = RecordingDriver::new(&harness, &harness.alloc);
    let arena_rec = RecordingDriver::new(&harness, &harness.alloc);
    for _ in 0..10 {
        workloads::workload_slab_churn(&slab_rec, hashes::W_COMBO_SA_SLAB, 500);
        workloads::workload_arena_bursts(&arena_rec, hashes::W_COMBO_SA_ARENA, 1, 300);
    }

    assert!(
        slab_rec.fraction_served_by(Backend::Slab) >= 0.99,
        "slab site should be served by Slab: {:?}",
        slab_rec.backend_counts()
    );
    assert!(
        arena_rec.fraction_served_by(Backend::Arena) >= 0.99,
        "arena site should be served by Arena: {:?}",
        arena_rec.backend_counts()
    );
}

#[test]
fn combo_buddy_small_diverges_per_site() {
    let model = forced_model_bytes(&[
        (hashes::W_COMBO_BA_BUDDY, BUDDY_SIZES[0], Backend::Buddy),
        (hashes::W_COMBO_BA_SMALL, SMALL_FIXED_REQUEST, Backend::Slab),
    ]);
    let alloc = Lohalloc::new();
    assert!(alloc.load(&model), "combo model load failed");
    let harness = HarnessDriver { alloc };

    let buddy_rec = RecordingDriver::new(&harness, &harness.alloc);
    let small_rec = RecordingDriver::new(&harness, &harness.alloc);
    for _ in 0..5 {
        workloads::workload_buddy_interleaved(&buddy_rec, hashes::W_COMBO_BA_BUDDY, 50);
        workloads::workload_slab_churn(&small_rec, hashes::W_COMBO_BA_SMALL, 200);
    }

    assert!(
        buddy_rec.fraction_served_by(Backend::Buddy) >= 0.99,
        "buddy site should be served by Buddy: {:?}",
        buddy_rec.backend_counts()
    );
    assert!(
        small_rec.fraction_served_by(Backend::Slab) >= 0.99,
        "small site should be served by Slab: {:?}",
        small_rec.backend_counts()
    );
}

#[test]
fn adv_exhaust_forces_fallthrough_once_arena_full() {
    let model = forced_model_bytes(&[(hashes::W_ADV_EXHAUST, SMALL_FIXED_REQUEST, Backend::Arena)]);
    let alloc = Lohalloc::new();
    assert!(alloc.load(&model), "exhaust model load failed");
    let harness = HarnessDriver { alloc };
    let recording = RecordingDriver::new(&harness, &harness.alloc);

    // The arena now CHAINS 1 MiB chunks up to a 32 MiB cap (it used to be a
    // single 1 MiB region), so exhausting it takes > 32 MiB of live,
    // never-freed traffic: 140_000 × (208B + 48B header) ≈ 36 MiB.
    let ptrs = workloads::workload_exhaust_no_free(&recording, hashes::W_ADV_EXHAUST, 140_000);

    let arena_frac = recording.fraction_served_by(Backend::Arena);
    assert!(
        arena_frac > 0.0 && arena_frac < 1.0,
        "expected partial arena exhaustion (some Arena, some fallthrough), got {arena_frac} ({:?})",
        recording.backend_counts()
    );

    // Clean up every allocation regardless of which backend actually served
    // it — dealloc routes correctly per-header.
    for (p, l) in ptrs {
        unsafe { harness.alloc.dealloc_with_hash(p, l) };
    }
}

#[test]
fn adv_mixed_smoke_spans_multiple_backends() {
    // No forced model: freeze an untrained (empty) bandit so every
    // allocation falls back to pure size-based routing
    // (`state::default_backend_for_size`). This is a Layer-1 sanity check
    // that erratic sizes naturally span backends; the trained-vs-size-based
    // performance hypothesis for this workload is Layer 2.
    let alloc = Lohalloc::new();
    alloc.freeze();
    assert!(alloc.is_inference());
    let harness = HarnessDriver { alloc };
    let recording = RecordingDriver::new(&harness, &harness.alloc);

    workloads::workload_adversarial_mixed(&recording, hashes::W_ADV_MIXED, 5_000);

    let counts = recording.backend_counts();
    assert!(
        counts.len() >= 2,
        "expected sizes to span multiple backends, got {counts:?}"
    );
}
