//! Cross-allocator comparison: runs the shared workload shapes against
//! whichever allocator this build's `#[global_allocator]` selects (see
//! `crate::global_alloc` — `alloc-lohalloc` / `alloc-jemalloc` /
//! `alloc-mimalloc`, or the platform default with no feature enabled).
//!
//! Run once per allocator with `--save-baseline <name>` so criterion can
//! diff them:
//!
//! ```sh
//! cargo bench -p lohalloc-bench --bench comparison --no-default-features --features alloc-lohalloc -- --save-baseline lohalloc
//! cargo bench -p lohalloc-bench --bench comparison --no-default-features --features alloc-jemalloc -- --save-baseline jemalloc
//! cargo bench -p lohalloc-bench --bench comparison --no-default-features --features alloc-mimalloc -- --save-baseline mimalloc
//! cargo bench -p lohalloc-bench --bench comparison -- --save-baseline system
//! ```

use criterion::{criterion_group, criterion_main, Criterion};
use lohalloc_bench::global_alloc::active_allocator_label;
use lohalloc_bench::workloads::{self, hashes, GlobalDriver};

fn comparison(c: &mut Criterion) {
    let label = active_allocator_label();
    let driver = GlobalDriver;
    let mut group = c.benchmark_group(format!("comparison_{label}"));

    group.bench_function("w_slab", |b| {
        b.iter(|| workloads::workload_slab_churn(&driver, hashes::W_SLAB, 2000));
    });
    group.bench_function("w_arena", |b| {
        b.iter(|| workloads::workload_arena_bursts(&driver, hashes::W_ARENA, 5, 300));
    });
    group.bench_function("w_buddy", |b| {
        b.iter(|| workloads::workload_buddy_interleaved(&driver, hashes::W_BUDDY, 200));
    });
    group.bench_function("w_adv_mixed", |b| {
        b.iter(|| workloads::workload_adversarial_mixed(&driver, hashes::W_ADV_MIXED, 1000));
    });

    group.finish();
}

criterion_group!(benches, comparison);
criterion_main!(benches);
