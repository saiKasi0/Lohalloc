// Native (C) benchmark driver: runs one workload, once, then exits.
// Outer timing/statistics (warmup, repetitions, percentiles) are handled by
// `hyperfine` invoking this binary repeatedly under different
// LD_PRELOAD/allocator configurations -- see bench/run_native.sh.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

#include "workloads.h"

// When LOHALLOC_BENCH_RSS is set, print peak RSS (high-water RSS over the whole
// run) as one `RSS_KIB <n>` line to stderr at exit — the memory-footprint axis
// the run_native.sh --rss pass captures. Kept out of the timing path (timing
// runs never set the env), so it never perturbs the wall measurement.
static void maybe_print_rss(void) {
    if (!getenv("LOHALLOC_BENCH_RSS")) return;
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return;
    long kib = ru.ru_maxrss; /* Linux: KiB; macOS: bytes */
#ifdef __APPLE__
    kib /= 1024;
#endif
    fprintf(stderr, "RSS_KIB %ld\n", kib);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s <slab|arena|buddy|system|adv-mixed|request-loop|json-tree|kv-store|mt-slab-tN|mt-mixed-tN|mt-xfree-tN|mt-interfere-tN> [ops]\n",
                argv[0]);
        return 2;
    }
    const char *workload = argv[1];
    size_t ops = argc >= 3 ? (size_t)strtoull(argv[2], NULL, 10) : 50000;

    if (!dispatch_workload(workload, ops)) {
        fprintf(stderr, "unknown workload '%s'\n", workload);
        return 2;
    }
    maybe_print_rss();
    return 0;
}
