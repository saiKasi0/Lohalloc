#include "workloads.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
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

// ---- Realistic application patterns ----------------------------------
// Dependency-free, deterministic (seeded LCG, same mixer as
// `workload_adversarial_mixed`) allocation SHAPES that mirror how real
// software allocates — the paper-vs-product benchmarks. Mirrored in
// `crates/lohalloc-bench/src/workloads.rs` so the C/C++/Rust sequences match.

// W-REQUEST-LOOP: N "requests", each allocating a burst of small request
// structs (16-256 B) + a couple medium response buffers (4-64 KiB), touched
// then freed ALL AT ONCE at request end — the per-request-arena pattern (HTTP
// handler / RPC dispatch). `ops` = total small allocations (~20 per request).
NOINLINE void workload_request_loop(size_t ops) {
    enum { SMALL_PER_REQ = 20, MED_PER_REQ = 2 };
    void *req[SMALL_PER_REQ + MED_PER_REQ];
    uint64_t state = 0xD1B54A32D192ED03ULL;
    size_t num_requests = ops / SMALL_PER_REQ;
    if (num_requests == 0) num_requests = 1;
    for (size_t r = 0; r < num_requests; r++) {
        size_t n = 0;
        for (size_t i = 0; i < SMALL_PER_REQ; i++) {
            state = state * 6364136223846793005ULL + 1ULL;
            size_t size = 16 + ((state >> 33) % (256 - 16)); /* 16..256 */
            void *p = malloc(size);
            if (p) ((volatile char *)p)[0] = (char)r;
            req[n++] = p;
        }
        for (size_t i = 0; i < MED_PER_REQ; i++) {
            state = state * 6364136223846793005ULL + 1ULL;
            size_t size = 4 * 1024 + ((state >> 33) % (60 * 1024)); /* 4K..64K */
            void *p = malloc(size);
            if (p) ((volatile char *)p)[0] = (char)r;
            req[n++] = p;
        }
        for (size_t i = 0; i < n; i++) {
            free(req[i]);
        }
    }
}

struct json_node {
    char *key;
    struct json_node **kids;
    size_t nkids;
    size_t cap;
};

// W-JSON-TREE: build a nested document tree of `ops` node structs, each with a
// variable-length string key (4-64 B) and a growable child-pointer array
// (realloc), linked under an earlier node; walk it, then free the whole tree.
// Mixed small+variable sizes + realloc growth + whole-tree burst free.
NOINLINE void workload_json_tree(size_t ops) {
    size_t n = ops;
    if (n == 0) n = 1;
    struct json_node **nodes = malloc(n * sizeof(struct json_node *));
    uint64_t state = 0x2545F4914F6CDD1DULL;
    for (size_t i = 0; i < n; i++) {
        struct json_node *node = malloc(sizeof(struct json_node));
        node->kids = NULL;
        node->nkids = 0;
        node->cap = 0;
        state = state * 6364136223846793005ULL + 1ULL;
        size_t klen = 4 + ((state >> 33) % 60); /* 4..64 */
        node->key = malloc(klen);
        if (node->key) ((volatile char *)node->key)[0] = (char)i;
        nodes[i] = node;
        if (i > 0) {
            state = state * 6364136223846793005ULL + 1ULL;
            size_t parent = (size_t)(state >> 1) % i;
            struct json_node *pp = nodes[parent];
            if (pp->nkids == pp->cap) {
                size_t ncap = pp->cap == 0 ? 4 : pp->cap * 2;
                pp->kids = realloc(pp->kids, ncap * sizeof(struct json_node *));
                pp->cap = ncap;
            }
            pp->kids[pp->nkids++] = node;
        }
    }
    volatile char sink = 0;
    for (size_t i = 0; i < n; i++) {
        if (nodes[i]->key) sink = (char)(sink + nodes[i]->key[0]);
    }
    (void)sink;
    for (size_t i = 0; i < n; i++) {
        free(nodes[i]->key);
        free(nodes[i]->kids);
        free(nodes[i]);
    }
    free(nodes);
}

struct kv_entry {
    int used;
    uint64_t key;
    void *val;
};

// W-KV-STORE: open-addressing hash table (fixed capacity, linear probe) with
// variable-size values (8-512 B). Random insert/overwrite/delete/lookup churn
// over `ops` operations — long-lived allocations + steady-state churn (the
// cache/store pattern). Evict-on-collision keeps it leak-free so RSS is a fair
// fragmentation signal.
NOINLINE void workload_kv_store(size_t ops) {
    const size_t CAP = 1u << 14; /* 16384 slots */
    struct kv_entry *tab = calloc(CAP, sizeof(struct kv_entry));
    uint64_t state = 0x9E6C63D0676A9A99ULL;
    for (size_t i = 0; i < ops; i++) {
        state = state * 6364136223846793005ULL + 1ULL;
        uint64_t key = state >> 12;
        size_t slot = (size_t)((key * 0x9E3779B97F4A7C15ULL) >> 50) & (CAP - 1);
        size_t probes = 0;
        while (tab[slot].used && tab[slot].key != key && probes < 32) {
            slot = (slot + 1) & (CAP - 1);
            probes++;
        }
        state = state * 6364136223846793005ULL + 1ULL;
        int op = (int)((state >> 40) % 3); /* 0=insert/update 1=delete 2=lookup */
        if (op == 0) {
            state = state * 6364136223846793005ULL + 1ULL;
            size_t vlen = 8 + ((state >> 33) % (512 - 8));
            if (tab[slot].used) free(tab[slot].val); /* overwrite or evict */
            void *v = malloc(vlen);
            if (v) ((volatile char *)v)[0] = (char)i;
            tab[slot].used = 1;
            tab[slot].key = key;
            tab[slot].val = v;
        } else if (op == 1) {
            if (tab[slot].used && tab[slot].key == key) {
                free(tab[slot].val);
                tab[slot].used = 0;
                tab[slot].val = NULL;
            }
        } else {
            if (tab[slot].used && tab[slot].key == key && tab[slot].val) {
                volatile char c = ((char *)tab[slot].val)[0];
                (void)c;
            }
        }
    }
    for (size_t i = 0; i < CAP; i++) {
        if (tab[i].used) free(tab[i].val);
    }
    free(tab);
}

// ---- Multithreaded workloads -----------------------------------------

struct mt_ops_arg {
    size_t ops;
};

static void *mt_slab_worker(void *arg) {
    workload_slab_churn(((struct mt_ops_arg *)arg)->ops);
    return NULL;
}

NOINLINE void workload_mt_slab_churn(size_t ops, int threads) {
    if (threads < 1) threads = 1;
    size_t per_thread = ops / (size_t)threads;
    if (per_thread == 0) per_thread = 1;

    pthread_t *tids = malloc(sizeof(pthread_t) * (size_t)threads);
    struct mt_ops_arg *args = malloc(sizeof(struct mt_ops_arg) * (size_t)threads);
    for (int i = 0; i < threads; i++) {
        args[i].ops = per_thread;
        pthread_create(&tids[i], NULL, mt_slab_worker, &args[i]);
    }
    for (int i = 0; i < threads; i++) {
        pthread_join(tids[i], NULL);
    }
    free(tids);
    free(args);
}

static void *mt_mixed_worker(void *arg) {
    workload_adversarial_mixed(((struct mt_ops_arg *)arg)->ops);
    return NULL;
}

NOINLINE void workload_mt_adversarial_mixed(size_t ops, int threads) {
    if (threads < 1) threads = 1;
    size_t per_thread = ops / (size_t)threads;
    if (per_thread == 0) per_thread = 1;

    pthread_t *tids = malloc(sizeof(pthread_t) * (size_t)threads);
    struct mt_ops_arg *args = malloc(sizeof(struct mt_ops_arg) * (size_t)threads);
    for (int i = 0; i < threads; i++) {
        args[i].ops = per_thread;
        pthread_create(&tids[i], NULL, mt_mixed_worker, &args[i]);
    }
    for (int i = 0; i < threads; i++) {
        pthread_join(tids[i], NULL);
    }
    free(tids);
    free(args);
}

// Bounded mailbox ring pairing one producer thread with one consumer
// thread: the producer allocates and enqueues, the consumer dequeues and
// frees -- every freed pointer crossed a thread boundary from its alloc.
#define MT_XFREE_RING_CAP 256

struct mt_xfree_mailbox {
    void *ring[MT_XFREE_RING_CAP];
    size_t head;
    size_t tail;
    size_t count;
    size_t to_produce;
    int done;
    pthread_mutex_t mu;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
};

static void mt_xfree_mailbox_init(struct mt_xfree_mailbox *mb, size_t ops) {
    mb->head = 0;
    mb->tail = 0;
    mb->count = 0;
    mb->to_produce = ops;
    mb->done = 0;
    pthread_mutex_init(&mb->mu, NULL);
    pthread_cond_init(&mb->not_empty, NULL);
    pthread_cond_init(&mb->not_full, NULL);
}

static void mt_xfree_mailbox_destroy(struct mt_xfree_mailbox *mb) {
    pthread_mutex_destroy(&mb->mu);
    pthread_cond_destroy(&mb->not_empty);
    pthread_cond_destroy(&mb->not_full);
}

static void *mt_xfree_producer(void *arg) {
    struct mt_xfree_mailbox *mb = arg;
    for (size_t i = 0; i < mb->to_produce; i++) {
        void *p = malloc(SMALL_FIXED_REQUEST);
        pthread_mutex_lock(&mb->mu);
        while (mb->count == MT_XFREE_RING_CAP) {
            pthread_cond_wait(&mb->not_full, &mb->mu);
        }
        mb->ring[mb->tail] = p;
        mb->tail = (mb->tail + 1) % MT_XFREE_RING_CAP;
        mb->count++;
        pthread_cond_signal(&mb->not_empty);
        pthread_mutex_unlock(&mb->mu);
    }
    pthread_mutex_lock(&mb->mu);
    mb->done = 1;
    pthread_cond_broadcast(&mb->not_empty);
    pthread_mutex_unlock(&mb->mu);
    return NULL;
}

static void *mt_xfree_consumer(void *arg) {
    struct mt_xfree_mailbox *mb = arg;
    for (;;) {
        pthread_mutex_lock(&mb->mu);
        while (mb->count == 0 && !mb->done) {
            pthread_cond_wait(&mb->not_empty, &mb->mu);
        }
        if (mb->count == 0 && mb->done) {
            pthread_mutex_unlock(&mb->mu);
            break;
        }
        void *p = mb->ring[mb->head];
        mb->head = (mb->head + 1) % MT_XFREE_RING_CAP;
        mb->count--;
        pthread_cond_signal(&mb->not_full);
        pthread_mutex_unlock(&mb->mu);
        free(p);
    }
    return NULL;
}

NOINLINE void workload_mt_xfree(size_t ops, int threads) {
    int pairs = threads / 2;
    if (pairs < 1) pairs = 1;
    size_t ops_per_pair = ops / (size_t)pairs;
    if (ops_per_pair == 0) ops_per_pair = 1;

    struct mt_xfree_mailbox *mbs = malloc(sizeof(struct mt_xfree_mailbox) * (size_t)pairs);
    pthread_t *producers = malloc(sizeof(pthread_t) * (size_t)pairs);
    pthread_t *consumers = malloc(sizeof(pthread_t) * (size_t)pairs);

    for (int i = 0; i < pairs; i++) {
        mt_xfree_mailbox_init(&mbs[i], ops_per_pair);
        pthread_create(&consumers[i], NULL, mt_xfree_consumer, &mbs[i]);
        pthread_create(&producers[i], NULL, mt_xfree_producer, &mbs[i]);
    }
    for (int i = 0; i < pairs; i++) {
        pthread_join(producers[i], NULL);
        pthread_join(consumers[i], NULL);
        mt_xfree_mailbox_destroy(&mbs[i]);
    }
    free(mbs);
    free(producers);
    free(consumers);
}

// W-MT-INTERFERE (J5-B2): fixed cache-resident application compute with only
// occasional allocation -- the allocator-interference benchmark. The compute
// (an FNV-1a pass over a rotating 512-byte window of a 4 KiB thread-local
// buffer) dominates; one SMALL_FIXED_REQUEST malloc/free happens every 8
// iterations. hyperfine's wall-time delta between allocators IS the
// interference signal (an ideal allocator scores ~1.0 vs any other).
// Deterministic: fixed iteration count, volatile sink so the kernel can't be
// elided. Mirrors crates/lohalloc-bench/src/workloads.rs::workload_mt_interfere.
#define INTERFERE_BUF_BYTES 4096
#define INTERFERE_WINDOW_BYTES 512
#define INTERFERE_ALLOC_EVERY 8

struct mt_interfere_arg {
    size_t ops;
    int thread_index;
};

static void *mt_interfere_worker(void *arg) {
    struct mt_interfere_arg *a = arg;
    unsigned char buf[INTERFERE_BUF_BYTES];
    memset(buf, 0, sizeof(buf));
    uint64_t seed = 0x9E3779B97F4A7C15ull ^ (uint64_t)a->thread_index;
    volatile uint64_t sink = 0;
    void *held = NULL;
    for (size_t i = 0; i < a->ops; i++) {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        buf[(size_t)seed & (INTERFERE_BUF_BYTES - 1)] = (unsigned char)seed;
        size_t start = (i * INTERFERE_WINDOW_BYTES) & (INTERFERE_BUF_BYTES - 1);
        uint64_t h = 0xcbf29ce484222325ull;
        for (size_t j = 0; j < INTERFERE_WINDOW_BYTES; j++) {
            h = (h ^ buf[start + j]) * 0x1000001b3ull;
        }
        sink = sink ^ h;
        if (i % INTERFERE_ALLOC_EVERY == 0) {
            if (held != NULL) {
                free(held);
            }
            held = malloc(SMALL_FIXED_REQUEST);
            if (held != NULL) {
                *(volatile unsigned char *)held = (unsigned char)seed;
            }
        }
    }
    if (held != NULL) {
        free(held);
    }
    (void)sink;
    return NULL;
}

NOINLINE void workload_mt_interfere(size_t ops, int threads) {
    if (threads < 1) threads = 1;
    size_t per_thread = ops / (size_t)threads;
    if (per_thread == 0) per_thread = 1;

    pthread_t *tids = malloc(sizeof(pthread_t) * (size_t)threads);
    struct mt_interfere_arg *args = malloc(sizeof(struct mt_interfere_arg) * (size_t)threads);
    for (int i = 0; i < threads; i++) {
        args[i].ops = per_thread;
        args[i].thread_index = i;
        pthread_create(&tids[i], NULL, mt_interfere_worker, &args[i]);
    }
    for (int i = 0; i < threads; i++) {
        pthread_join(tids[i], NULL);
    }
    free(tids);
    free(args);
}

// Parses "mt-<kind>-t<N>" into `kind_out` (nul-terminated) and `*threads_out`.
// Splits on the *last* '-' in the string so a kind name could itself contain
// hyphens (not currently needed, but costs nothing). Returns 1 on a
// well-formed name with a parsed thread count > 0, 0 otherwise.
static int parse_mt_workload(const char *workload, char *kind_out, size_t kind_cap, int *threads_out) {
    if (strncmp(workload, "mt-", 3) != 0) {
        return 0;
    }
    const char *rest = workload + 3; // "<kind>-t<N>"
    const char *dash_t = strrchr(rest, '-');
    if (dash_t == NULL || dash_t[1] != 't') {
        return 0;
    }
    size_t kind_len = (size_t)(dash_t - rest);
    if (kind_len == 0 || kind_len >= kind_cap) {
        return 0;
    }
    memcpy(kind_out, rest, kind_len);
    kind_out[kind_len] = '\0';
    *threads_out = atoi(dash_t + 2);
    return *threads_out > 0;
}

int dispatch_workload(const char *workload, size_t ops) {
    char kind[16];
    int threads;
    if (parse_mt_workload(workload, kind, sizeof(kind), &threads)) {
        if (strcmp(kind, "slab") == 0) {
            workload_mt_slab_churn(ops, threads);
        } else if (strcmp(kind, "mixed") == 0) {
            workload_mt_adversarial_mixed(ops, threads);
        } else if (strcmp(kind, "xfree") == 0) {
            workload_mt_xfree(ops, threads);
        } else if (strcmp(kind, "interfere") == 0) {
            workload_mt_interfere(ops, threads);
        } else {
            fprintf(stderr, "unknown mt workload kind '%s'\n", kind);
            return 0;
        }
        return 1;
    }
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
    } else if (strcmp(workload, "request-loop") == 0) {
        workload_request_loop(ops);
    } else if (strcmp(workload, "json-tree") == 0) {
        workload_json_tree(ops);
    } else if (strcmp(workload, "kv-store") == 0) {
        workload_kv_store(ops);
    } else {
        return 0;
    }
    return 1;
}
