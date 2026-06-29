#ifndef LOHALLOC_OBS_H
#define LOHALLOC_OBS_H

#include <stdint.h>
#include <stddef.h>

/* MUST match crates/lohalloc-alloc/src/observer.rs TelemetryCRecord exactly.
 *
 * 72 bytes total, 8-byte aligned. Pinned by `record_size_is_stable` and
 * `record_field_offsets_are_stable` tests on the Rust side; the shim has a
 * matching `_Static_assert` in lohalloc_obs.c.
 */
typedef struct {
    uint64_t timestamp;
    uint8_t  op;           /* 0 = Alloc, 1 = Free */
    uint8_t  _pad0[7];
    size_t   size;
    uint64_t stack_hash;
    uint32_t thread_id;
    uint8_t  _pad1[4];
    uint64_t result_ptr;
    uint64_t latency_ns;
    float    fragmentation_pct;
    uint8_t  backend;      /* 0=Slab 1=Buddy 2=System 3=Arena 0xFF=Unknown */
    uint8_t  _pad2[7];
} LohallocTelemetryRecord;

/* The sink function the Rust allocator calls (extern "C").
 *
 * Hot path: must not allocate, must not block. The shim copies the record
 * into a bounded ring (65,536 entries) and signals a background consumer.
 * If the ring is full the record is silently dropped and a counter is
 * incremented — the allocator never blocks on telemetry.
 *
 * Safe to call from any thread concurrently. The first call lazily spawns
 * the consumer pthread and registers an atexit handler.
 */
extern void lohalloc_telemetry_emit(const LohallocTelemetryRecord *rec);

/* Override the server port. Default is 3000. The env var LOHALLOC_OBS_PORT
 * (if set at process startup) is also honored — whichever runs first wins,
 * with explicit calls taking precedence over the env var.
 *
 * Safe to call before the first emit; calls after that have no effect.
 */
void lohalloc_obs_set_port(int port);

#ifdef LOHALLOC_OBS_TESTING

/* Test-only API. Compiled in only when test_ring.c is built with
 * -DLOHALLOC_OBS_TESTING. NOT part of the production wire contract. */

/* Format a single record into `buf` as one JSON object (no trailing
 * newline). Returns bytes written excluding the trailing NUL, or -1 if
 * `buf` is too small. */
int lohalloc_obs_format_record(char *buf, int cap, const LohallocTelemetryRecord *rec);

/* Push records directly into the ring (bypassing the public sink). Useful
 * for tests that want to exercise the consumer without spinning up the
 * full atexit machinery. */
void lohalloc_obs_test_push(const LohallocTelemetryRecord *recs, int count);

/* Drain the ring into `buf` as a JSON array. Updates the ring tail.
 * Returns the number of records drained, or -1 on overflow. */
int lohalloc_obs_test_drain_to_buffer(char *buf, int cap);

/* Reset the ring to empty and the dropped counter to 0. */
void lohalloc_obs_test_reset(void);

/* Number of records that have been silently dropped (ring was full). */
unsigned long long lohalloc_obs_test_dropped(void);

/* Number of records currently buffered in the ring. */
unsigned long long lohalloc_obs_test_pending(void);

#endif /* LOHALLOC_OBS_TESTING */

#endif /* LOHALLOC_OBS_H */
