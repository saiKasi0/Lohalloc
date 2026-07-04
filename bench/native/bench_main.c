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
        fprintf(stderr,
                "usage: %s <slab|arena|buddy|system|adv-mixed|mt-slab-tN|mt-mixed-tN|mt-xfree-tN> [ops]\n",
                argv[0]);
        return 2;
    }
    const char *workload = argv[1];
    size_t ops = argc >= 3 ? (size_t)strtoull(argv[2], NULL, 10) : 50000;

    if (!dispatch_workload(workload, ops)) {
        fprintf(stderr, "unknown workload '%s'\n", workload);
        return 2;
    }
    return 0;
}
