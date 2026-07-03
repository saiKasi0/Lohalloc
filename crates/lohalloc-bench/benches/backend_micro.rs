//! Per-backend forced micro-benchmark: one alloc+dealloc pair per size,
//! each backend forced via a hand-built `.lohalloc` model (Layer 1).
//! Establishes the baseline per-op cost of each backend in isolation before
//! comparing whole-workload trends in `hypothesis.rs`.

use core::alloc::Layout;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use lohalloc_bench::forced::lohalloc_forced_single;
use lohalloc_core::Backend;

const HASH: u64 = 0xB4C4_0001;

fn backend_micro(c: &mut Criterion) {
    let mut group = c.benchmark_group("backend_micro");
    let cases: &[(Backend, usize)] = &[
        (Backend::Slab, 208),  // -> 256B slab class
        (Backend::Buddy, 208), // buddy forced on a slab-sized req
        (Backend::Arena, 208),
        (Backend::System, 208),
        (Backend::Buddy, 65536 - 48), // mid-size buddy territory
        (Backend::System, 2 * 1024 * 1024 - 48), // large, system-only territory
    ];
    for &(backend, size) in cases {
        let alloc = lohalloc_forced_single(HASH, size, backend);
        let layout = Layout::from_size_align(size, 16).unwrap();
        group.bench_with_input(
            BenchmarkId::new(format!("{backend:?}"), size),
            &size,
            |b, _| {
                b.iter(|| unsafe {
                    let ptr = alloc.alloc_with_hash(layout, HASH);
                    alloc.dealloc_with_hash(ptr, layout);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, backend_micro);
criterion_main!(benches);
