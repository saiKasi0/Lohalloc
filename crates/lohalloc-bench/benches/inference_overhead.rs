//! W-INFER-OVERHEAD: training vs. frozen-trained vs. frozen-empty-table.
//!
//! Hypothesis: Inference-mode routing (a single MPHF lookup) should be no
//! slower than Training mode (UCB1 + hysteresis over a `BTreeMap`), and
//! should be close to the empty-table fallback (pure size-based routing) —
//! the MPHF lookup itself is close to free.

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use lohalloc_bench::workloads::{self, hashes, HarnessDriver};

fn inference_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference_overhead");

    group.bench_function("training", |b| {
        b.iter_batched(
            HarnessDriver::new,
            |h| workloads::workload_slab_churn(&h, hashes::W_SLAB, 2000),
            BatchSize::SmallInput,
        );
    });

    group.bench_function("frozen_trained", |b| {
        b.iter_batched(
            || {
                let h = HarnessDriver::new();
                workloads::workload_slab_churn(&h, hashes::W_SLAB, 200);
                h.alloc.freeze();
                h
            },
            |h| workloads::workload_slab_churn(&h, hashes::W_SLAB, 2000),
            BatchSize::SmallInput,
        );
    });

    group.bench_function("frozen_empty_table", |b| {
        b.iter_batched(
            || {
                let h = HarnessDriver::new();
                h.alloc.freeze(); // empty bandit -> empty table -> pure size-based fallback
                h
            },
            |h| workloads::workload_slab_churn(&h, hashes::W_SLAB, 2000),
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, inference_overhead);
criterion_main!(benches);
