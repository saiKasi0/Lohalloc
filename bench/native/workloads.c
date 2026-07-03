#include "workloads.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

const size_t BUDDY_SIZES[4] = {32 * 1024, 64 * 1024, 128 * 1024, 256 * 1024};
const size_t SYSTEM_SIZES[2] = {2 * 1024 * 1024, 8 * 1024 * 1024};

// Marked so the compiler cannot inline these into bench_main -- keeps each
// workload a distinct, stable call site, mirroring the Rust
// `#[inline(never)]` generators.
#if defined(__GNUC__) || defined(__clang__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif

NOINLINE void workload_slab_churn(size_t ops) {
    void *window[64];
    size_t head = 0, count = 0;
    for (size_t i = 0; i < ops; i++) {
        void *p = malloc(SMALL_FIXED_REQUEST);
        if (count == 64) {
            free(window[head]);
            window[head] = p;
            head = (head + 1) % 64;
        } else {
            window[(head + count) % 64] = p;
            count++;
        }
    }
    for (size_t i = 0; i < count; i++) {
        free(window[(head + i) % 64]);
    }
}

NOINLINE void workload_arena_bursts(size_t num_bursts, size_t burst_size) {
    void **ptrs = malloc(burst_size * sizeof(void *));
    for (size_t b = 0; b < num_bursts; b++) {
        for (size_t i = 0; i < burst_size; i++) {
            ptrs[i] = malloc(SMALL_FIXED_REQUEST);
        }
        for (size_t i = 0; i < burst_size; i++) {
            free(ptrs[i]);
        }
    }
    free(ptrs);
}

NOINLINE void workload_buddy_interleaved(size_t ops) {
    void **live = malloc(ops * sizeof(void *));
    size_t live_count = 0;
    for (size_t i = 0; i < ops; i++) {
        size_t size = BUDDY_SIZES[i % 4];
        live[live_count++] = malloc(size);
        if (i % 2 == 1 && live_count > 0) {
            free(live[0]);
            memmove(live, live + 1, (live_count - 1) * sizeof(void *));
            live_count--;
        }
    }
    for (size_t i = 0; i < live_count; i++) {
        free(live[i]);
    }
    free(live);
}

NOINLINE void workload_system_large(size_t ops) {
    for (size_t i = 0; i < ops; i++) {
        size_t size = SYSTEM_SIZES[i % 2];
        void *p = malloc(size);
        // Touch the allocation through a volatile pointer: without this,
        // GCC/Clang's builtin-malloc semantics let them prove `p` unused
        // and DELETE the whole malloc/free pair at -O2 — this workload
        // measured literally nothing for every allocator (caught when the
        // Phase 6 train+export step saw zero interposed mallocs). The
        // other workloads escape their pointers through arrays/later
        // frees, so only this trivial pair was elidable. Mirrored in the
        // Rust generator (`workloads.rs::workload_system_large`) to keep
        // the cross-language shapes identical.
        if (p) {
            ((volatile char *)p)[0] = (char)i;
        }
        free(p);
    }
}

NOINLINE void workload_adversarial_mixed(size_t ops) {
    void **live = malloc(ops * sizeof(void *));
    size_t live_count = 0;
    uint64_t state = 0x9E3779B97F4A7C15ULL;
    for (size_t i = 0; i < ops; i++) {
        state = state * 6364136223846793005ULL + 1ULL;
        size_t size = 1 + ((state >> 33) % (64 * 1024));
        live[live_count++] = malloc(size);
        if (live_count > 32 && (state & 1) == 0) {
            size_t idx = (size_t)(state >> 1) % live_count;
            free(live[idx]);
            live[idx] = live[live_count - 1];
            live_count--;
        }
    }
    for (size_t i = 0; i < live_count; i++) {
        free(live[i]);
    }
    free(live);
}
