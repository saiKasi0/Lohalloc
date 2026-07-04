// Backend-pure and adversarial workload generators shared by the C and C++
// harness binaries. Mirrors crates/lohalloc-bench/src/workloads.rs so the
// same hypothesis matrix (see crates/lohalloc-bench/src/hypotheses.rs)
// applies across languages. Portable across every allocator under test
// (system/jemalloc/mimalloc/lohalloc) -- unlike the Rust harness, there is
// no allocator-specific reset/arena API available here, so "arena-favorable"
// is expressed as a plain burst-then-free-all pattern any allocator can
// serve, while only Lohalloc gets the full benefit of its dedicated Arena
// backend + reset.
#ifndef LOHALLOC_BENCH_WORKLOADS_H
#define LOHALLOC_BENCH_WORKLOADS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Bytes of Lohalloc header overhead at the default 16-byte alignment (see
// crates/lohalloc-alloc/src/lib.rs::header_pad). A request of
// `class - HEADER_PAD` bytes lands exactly on `class` under Lohalloc; other
// allocators ignore this, so the same sizes are still meaningful workload
// shapes for them.
#define HEADER_PAD 48
#define SMALL_FIXED_REQUEST (256 - HEADER_PAD) /* 208 */

extern const size_t BUDDY_SIZES[4]; /* 32 KiB, 64 KiB, 128 KiB, 256 KiB */
extern const size_t SYSTEM_SIZES[2]; /* 2 MiB, 8 MiB */

// W-SLAB: tight churn of a single fixed size, bounded 64-deep live window.
void workload_slab_churn(size_t ops);

// W-ARENA (portable variant): bursts of short-lived allocations, all freed
// at the end of each burst.
void workload_arena_bursts(size_t num_bursts, size_t burst_size);

// W-BUDDY: variable medium sizes (32 KiB-256 KiB) with interleaved
// alloc/free.
void workload_buddy_interleaved(size_t ops);

// W-SYSTEM: large (2 MiB / 8 MiB) allocations, immediately freed.
void workload_system_large(size_t ops);

// W-ADV-MIXED: erratic sizes (1 B-64 KiB) via a deterministic xorshift-style
// PRNG, pseudo-random lifetimes. Adversarial -- no single backend dominates.
void workload_adversarial_mixed(size_t ops);

// W-MT-SLAB: `threads` independent threads, each running a same-thread copy
// of W-SLAB's churn (own live window, own allocs+frees). Isolates
// magazine/tcache scaling under concurrent but non-cross-thread traffic --
// no thread ever touches another thread's blocks.
void workload_mt_slab_churn(size_t ops, int threads);

// W-MT-MIXED: `threads` independent threads, each running a same-thread copy
// of W-ADV-MIXED. Isolates medium/buddy-range lock contention under
// concurrent traffic (same shape as W-MT-SLAB, adversarial size mix).
void workload_mt_adversarial_mixed(size_t ops, int threads);

// W-MT-XFREE: `threads` threads paired into producer/consumer roles over a
// bounded mailbox ring -- the producer allocates, the consumer frees a
// different thread's allocation. The hard cross-thread-free case: every
// freed block must migrate back through whatever thread-local structures
// the allocator uses on the alloc side.
void workload_mt_xfree(size_t ops, int threads);

// Dispatches any workload name this file knows about, including the
// "mt-<kind>-t<N>" multithreaded names (kind is "slab", "mixed", or
// "xfree"; N is the thread count) -- shared by the C and C++ drivers so the
// name-parsing logic lives in exactly one place. Returns 1 on a recognized
// name, 0 (with nothing run) otherwise.
int dispatch_workload(const char *workload, size_t ops);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // LOHALLOC_BENCH_WORKLOADS_H
