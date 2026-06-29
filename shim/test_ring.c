/* Unit tests for liblohalloc_obs.
 *
 * Compiled with -DLOHALLOC_OBS_TESTING to expose the test-only API in
 * lohalloc_obs.h. Run via `make test` (or directly: `./test_ring`).
 *
 * Test cases:
 *   1. test_ring_basic_push_pop   — push 10, drain, check count
 *   2. test_ring_overflow         — push > capacity, check drop counter
 *   3. test_json_format_alloc     — alloc record has all fields incl. backend
 *   4. test_json_format_free      — free record omits backend
 *   5. test_json_format_backends  — every backend byte encodes correctly
 *   6. test_record_size_pin       — sizeof == 72 (wire-format contract)
 *   7. test_emit_sink_no_crash    — public sink is callable under stress
 */

#include "lohalloc_obs.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- helpers ----------------------------------------------------------- */

static LohallocTelemetryRecord make_record(uint64_t ts, uint8_t op, size_t size,
                                           uint64_t hash, uint32_t tid, uint64_t ptr,
                                           uint64_t lat, float frag, uint8_t backend) {
    LohallocTelemetryRecord r;
    memset(&r, 0, sizeof(r));
    r.timestamp = ts;
    r.op = op;
    r.size = size;
    r.stack_hash = hash;
    r.thread_id = tid;
    r.result_ptr = ptr;
    r.latency_ns = lat;
    r.fragmentation_pct = frag;
    r.backend = backend;
    return r;
}

/* Tiny substring search — avoids depending on <regex.h> just for tests. */
static int contains(const char *hay, const char *needle) {
    return strstr(hay, needle) != NULL;
}

#define TEST(name) static void name(void)

#define RUN(name) do {                     \
    printf("  - %-32s ", #name);           \
    fflush(stdout);                         \
    name();                                 \
    printf("ok\n");                         \
} while (0)

/* ---- tests ------------------------------------------------------------- */

TEST(test_record_size_pin) {
    assert(sizeof(LohallocTelemetryRecord) == 72);
    assert(_Alignof(LohallocTelemetryRecord) == 8);
    /* Field offsets, mirrored from observer.rs::record_field_offsets_are_stable. */
    assert(offsetof(LohallocTelemetryRecord, timestamp) == 0);
    assert(offsetof(LohallocTelemetryRecord, op) == 8);
    assert(offsetof(LohallocTelemetryRecord, size) == 16);
    assert(offsetof(LohallocTelemetryRecord, stack_hash) == 24);
    assert(offsetof(LohallocTelemetryRecord, thread_id) == 32);
    assert(offsetof(LohallocTelemetryRecord, result_ptr) == 40);
    assert(offsetof(LohallocTelemetryRecord, latency_ns) == 48);
    assert(offsetof(LohallocTelemetryRecord, fragmentation_pct) == 56);
    assert(offsetof(LohallocTelemetryRecord, backend) == 60);
}

TEST(test_json_format_alloc) {
    LohallocTelemetryRecord r = make_record(
        12345, 0 /*Alloc*/, 64, 9876543210ULL, 7, 0x1000ULL, 100, 0.0f, 0 /*Slab*/);
    char buf[512];
    int n = lohalloc_obs_format_record(buf, (int)sizeof(buf), &r);
    assert(n > 0);

    /* Every field must appear. */
    assert(contains(buf, "\"timestamp\":12345"));
    assert(contains(buf, "\"op\":\"alloc\""));
    assert(contains(buf, "\"size\":64"));
    assert(contains(buf, "\"stack_hash\":9876543210"));
    assert(contains(buf, "\"thread_id\":7"));
    assert(contains(buf, "\"result_ptr\":\"0x1000\""));
    assert(contains(buf, "\"latency_ns\":100"));
    assert(contains(buf, "\"fragmentation_pct\":0"));
    assert(contains(buf, "\"backend\":\"Slab\""));
}

TEST(test_json_format_free) {
    /* Free: backend = 0xFF (Unknown) must be omitted, matching the Rust
     * `Option::is_none` skip behavior. */
    LohallocTelemetryRecord r = make_record(
        99, 1 /*Free*/, 32, 0xdeadbeefULL, 1, 0x2000ULL, 50, 0.0f, 0xFF);
    char buf[512];
    int n = lohalloc_obs_format_record(buf, (int)sizeof(buf), &r);
    assert(n > 0);

    assert(contains(buf, "\"op\":\"free\""));
    assert(contains(buf, "\"size\":32"));
    assert(!contains(buf, "\"backend\"")); /* must be omitted */
}

TEST(test_json_format_backends) {
    struct { uint8_t code; const char *name; } cases[] = {
        { 0, "Slab"   },
        { 1, "Buddy"  },
        { 2, "System" },
        { 3, "Arena"  },
    };
    for (size_t i = 0; i < sizeof(cases)/sizeof(cases[0]); ++i) {
        LohallocTelemetryRecord r = make_record(
            1, 0, 16, 1, 0, 0x10, 1, 0.0f, cases[i].code);
        char buf[256];
        int n = lohalloc_obs_format_record(buf, (int)sizeof(buf), &r);
        assert(n > 0);
        char needle[64];
        snprintf(needle, sizeof(needle), "\"backend\":\"%s\"", cases[i].name);
        assert(contains(buf, needle));
    }
}

TEST(test_json_format_unknown_backend_omitted) {
    /* Any backend byte outside 0..3 must be omitted, not serialized as
     * a bogus string. This matches the Rust `Option::is_none` path. */
    LohallocTelemetryRecord r = make_record(
        1, 0, 16, 1, 0, 0x10, 1, 0.0f, 42 /*out of range*/);
    char buf[256];
    int n = lohalloc_obs_format_record(buf, (int)sizeof(buf), &r);
    assert(n > 0);
    assert(!contains(buf, "\"backend\""));
}

TEST(test_json_format_ptr_hex_lowercase) {
    LohallocTelemetryRecord r = make_record(
        1, 0, 8, 1, 0, 0xDEADBEEFCAFE1234ULL, 1, 0.0f, 0);
    char buf[512];
    int n = lohalloc_obs_format_record(buf, (int)sizeof(buf), &r);
    assert(n > 0);
    /* Lowercase hex, no padding. */
    assert(contains(buf, "\"result_ptr\":\"0xdeadbeefcafe1234\""));
}

TEST(test_ring_basic_push_pop) {
    lohalloc_obs_test_reset();

    LohallocTelemetryRecord batch[10];
    for (int i = 0; i < 10; ++i) {
        batch[i] = make_record((uint64_t)i, 0, (size_t)(i + 1), (uint64_t)i, 0,
                               0x100ULL + (uint64_t)i, 1, 0.0f, 0);
    }
    lohalloc_obs_test_push(batch, 10);

    assert(lohalloc_obs_test_pending() == 10);

    char buf[8192];
    int n = lohalloc_obs_test_drain_to_buffer(buf, (int)sizeof(buf));
    assert(n == 10);
    /* Drains to a JSON array: starts with '[' and ends with ']'. */
    assert(buf[0] == '[');
    assert(buf[n - 1] == ']' || strrchr(buf, ']') != NULL);
    assert(lohalloc_obs_test_pending() == 0);
}

TEST(test_ring_overflow) {
    lohalloc_obs_test_reset();

    /* Push past capacity. Capacity is 65,536; push 100,000 and expect at
     * least 100,000 - 65,536 = 34,464 drops. */
    const int total = 100000;
    LohallocTelemetryRecord batch[1024];
    for (int i = 0; i < 1024; ++i) {
        batch[i] = make_record(0, 0, 8, 0, 0, 0, 0, 0.0f, 0);
    }
    for (int sent = 0; sent < total; sent += 1024) {
        int n = (total - sent >= 1024) ? 1024 : (total - sent);
        lohalloc_obs_test_push(batch, n);
    }

    unsigned long long pending = lohalloc_obs_test_pending();
    unsigned long long dropped = lohalloc_obs_test_dropped();

    /* Pending ≤ capacity; dropped + pending == total. */
    assert(pending <= 65536);
    assert(dropped + pending == (unsigned long long)total);
    /* Most importantly: at least one record was dropped — otherwise the
     * test isn't actually exercising the overflow path. */
    assert(dropped > 0);
}

TEST(test_emit_sink_no_crash) {
    /* Hammer the public sink. This will lazily start the consumer thread,
     * which tries to POST to 127.0.0.1:3000 (nothing listening → silent
     * failure, batch retried). We don't care about the HTTP result here,
     * only that no signal is raised and the process keeps running. */
    for (int i = 0; i < 1000; ++i) {
        LohallocTelemetryRecord r = make_record(
            (uint64_t)i, (uint8_t)(i & 1), 64, (uint64_t)i, 0, 0x100ULL,
            1, 0.0f, (uint8_t)(i & 3));
        lohalloc_telemetry_emit(&r);
    }
    /* If we got here, no crash. The pending count is at least 0 (the
     * consumer may have raced ahead). */
    (void)lohalloc_obs_test_pending();
}

/* ---- main -------------------------------------------------------------- */

int main(void) {
    printf("liblohalloc_obs tests:\n");
    RUN(test_record_size_pin);
    RUN(test_json_format_alloc);
    RUN(test_json_format_free);
    RUN(test_json_format_backends);
    RUN(test_json_format_unknown_backend_omitted);
    RUN(test_json_format_ptr_hex_lowercase);
    RUN(test_ring_basic_push_pop);
    RUN(test_ring_overflow);
    RUN(test_emit_sink_no_crash);
    printf("all tests passed\n");
    return 0;
}
