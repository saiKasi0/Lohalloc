// Native (C) benchmark driver: runs one workload, once, then exits.
// Outer timing/statistics (warmup, repetitions, percentiles) are handled by
// `hyperfine` invoking this binary repeatedly under different
// LD_PRELOAD/allocator configurations -- see bench/run_native.sh.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "workloads.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <slab|arena|buddy|system|adv-mixed> [ops]\n", argv[0]);
        return 2;
    }
    const char *workload = argv[1];
    size_t ops = argc >= 3 ? (size_t)strtoull(argv[2], NULL, 10) : 50000;

    if (strcmp(workload, "slab") == 0) {
        workload_slab_churn(ops);
    } else if (strcmp(workload, "arena") == 0) {
        workload_arena_bursts(ops / 500 > 0 ? ops / 500 : 1, 500);
    } else if (strcmp(workload, "buddy") == 0) {
        workload_buddy_interleaved(ops);
    } else if (strcmp(workload, "system") == 0) {
        workload_system_large(ops / 20 > 0 ? ops / 20 : 1);
    } else if (strcmp(workload, "adv-mixed") == 0) {
        workload_adversarial_mixed(ops);
    } else {
        fprintf(stderr, "unknown workload '%s'\n", workload);
        return 2;
    }
    return 0;
}
