//! Whole-workload hypothesis benchmark: trained routing vs. forced-best vs.
//! forced-worst, for each pure backend-favorable workload (Layer 1 for the
//! forced variants; "trained" measures the current — still placeholder —
//! bandit policy, see `crates/lohalloc-alloc/src/bandit.rs`. Once Layer 2
//! lands, "trained" becomes a real decision-plane measurement without any
//! change needed here).

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use lohalloc_bench::forced::lohalloc_forced_single;
use lohalloc_bench::workloads::{self, hashes, HarnessDriver, BUDDY_SIZES, SMALL_FIXED_REQUEST};
use lohalloc_core::Backend;

fn bench_slab(c: &mut Criterion) {
    let mut group = c.benchmark_group("hypothesis_slab");
    group.bench_function("trained", |b| {
        b.iter_batched(
            HarnessDriver::new,
            |h| workloads::workload_slab_churn(&h, hashes::W_SLAB, 2000),
            BatchSize::SmallInput,
        );
    });
    group.bench_function("forced_best_slab", |b| {
        b.iter_batched(
            || {
                HarnessDriver::with_alloc(lohalloc_forced_single(
                    hashes::W_SLAB,
                    SMALL_FIXED_REQUEST,
                    Backend::Slab,
                ))
            },
            |h| workloads::workload_slab_churn(&h, hashes::W_SLAB, 2000),
            BatchSize::SmallInput,
        );
    });
    group.bench_function("forced_worst_system", |b| {
        b.iter_batched(
            || {
                HarnessDriver::with_alloc(lohalloc_forced_single(
                    hashes::W_SLAB,
                    SMALL_FIXED_REQUEST,
                    Backend::System,
                ))
            },
            |h| workloads::workload_slab_churn(&h, hashes::W_SLAB, 2000),
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_arena(c: &mut Criterion) {
    let mut group = c.benchmark_group("hypothesis_arena");
    group.bench_function("trained", |b| {
        b.iter_batched(
            HarnessDriver::new,
            |h| workloads::workload_arena_bursts(&h, hashes::W_ARENA, 5, 300),
            BatchSize::SmallInput,
        );
    });
    group.bench_function("forced_best_arena", |b| {
        b.iter_batched(
            || {
                HarnessDriver::with_alloc(lohalloc_forced_single(
                    hashes::W_ARENA,
                    SMALL_FIXED_REQUEST,
                    Backend::Arena,
                ))
            },
            |h| workloads::workload_arena_bursts(&h, hashes::W_ARENA, 5, 300),
            BatchSize::SmallInput,
        );
    });
    group.bench_function("forced_worst_system", |b| {
        b.iter_batched(
            || {
                HarnessDriver::with_alloc(lohalloc_forced_single(
                    hashes::W_ARENA,
                    SMALL_FIXED_REQUEST,
                    Backend::System,
                ))
            },
            |h| workloads::workload_arena_bursts(&h, hashes::W_ARENA, 5, 300),
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_buddy(c: &mut Criterion) {
    let mut group = c.benchmark_group("hypothesis_buddy");
    group.bench_function("trained", |b| {
        b.iter_batched(
            HarnessDriver::new,
            |h| workloads::workload_buddy_interleaved(&h, hashes::W_BUDDY, 200),
            BatchSize::SmallInput,
        );
    });
    group.bench_function("forced_best_buddy", |b| {
        b.iter_batched(
            || {
                HarnessDriver::with_alloc(lohalloc_forced_single(
                    hashes::W_BUDDY,
                    BUDDY_SIZES[0],
                    Backend::Buddy,
                ))
            },
            |h| workloads::workload_buddy_interleaved(&h, hashes::W_BUDDY, 200),
            BatchSize::SmallInput,
        );
    });
    group.bench_function("forced_worst_system", |b| {
        b.iter_batched(
            || {
                HarnessDriver::with_alloc(lohalloc_forced_single(
                    hashes::W_BUDDY,
                    BUDDY_SIZES[0],
                    Backend::System,
                ))
            },
            |h| workloads::workload_buddy_interleaved(&h, hashes::W_BUDDY, 200),
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(benches, bench_slab, bench_arena, bench_buddy);
criterion_main!(benches);
