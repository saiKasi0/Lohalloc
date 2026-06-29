//! LD_PRELOAD / DYLD_INSERT_LIBRARIES shim that bridges Lohalloc's C-ABI
//! telemetry sink to the Lohalloc Axum server.
//!
//! Architecture
//! ============
//!
//! ```text
//!   ┌──────────────────────────────────┐
//!   │ Rust allocator (hot path)        │
//!   │   emit_alloc() / emit_free()     │
//!   └────────────────┬─────────────────┘
//!                    │ C-ABI call (extern "C")
//!                    ▼
//!   ┌──────────────────────────────────┐
//!   │ lohalloc_telemetry_emit()        │  (this file)
//!   │   - copies 72 bytes into ring    │
//!   │   - signals cond on 0→1          │
//!   │   - drops on full (counter++)    │
//!   └────────────────┬─────────────────┘
//!                    │
//!                    ▼
//!   ┌──────────────────────────────────┐
//!   │ SPSC-ish ring (mutex-protected,  │
//!   │  cap = 65,536 entries)           │
//!   └────────────────┬─────────────────┘
//!                    │ pthread_cond signal
//!                    ▼
//!   ┌──────────────────────────────────┐
//!   │ Consumer thread                  │
//!   │   - batches up to 256 records    │
//!   │   - JSON-encodes                 │
//!   │   - POSTs to 127.0.0.1:<port>    │
//!   │     /api/telemetry               │
//!   └──────────────────────────────────┘
//! ```
//!
//! The producer side is held under a single global mutex. Per the task spec,
//! contention is acceptable for a demo — the lock is held for one 72-byte
//! `memcpy` plus a head increment, and the allocator's hot path is dominated
//! by `route_alloc` / `route_free` work, not by this sink.
//!
//! On the very first emit we lazily spawn the consumer pthread and register
//! an `atexit` handler. `atexit` sets `stopping`, signals the cond var, and
//! joins the consumer; the consumer drains anything still buffered.

#include "lohalloc_obs.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <fcntl.h>
#include <errno.h>

/* Compile-time pin: the Rust `TelemetryCRecord` is 72 bytes (pinned by
   `record_size_is_stable`). If this trips, the shim is out of sync with the
   allocator — fix one, then the other. */
_Static_assert(sizeof(LohallocTelemetryRecord) == 72,
               "LohallocTelemetryRecord must be 72 bytes (see observer.rs)");
_Static_assert(_Alignof(LohallocTelemetryRecord) == 8,
               "LohallocTelemetryRecord must be 8-byte aligned");

/* Ring buffer geometry. 65,536 entries × 72 bytes = 4.5 MiB. Power-of-two so
   the head/tail wrap math is one AND with `RING_MASK`. */
#define RING_CAPACITY  65536u
#define RING_MASK      (RING_CAPACITY - 1u)
#define BATCH_SIZE     256
#define HTTP_BUF_SIZE  (64 * 1024)

/* Forward decls for helpers shared with the test-only API. */
static void consumer_wake(void);

/* ------------------------------------------------------------------
 * Global state
 * ------------------------------------------------------------------ */

static LohallocTelemetryRecord g_ring[RING_CAPACITY];
static uint64_t g_head;          /* producer index (write next slot) */
static uint64_t g_tail;          /* consumer index (read next slot) */
static _Atomic unsigned long long g_dropped;
static _Atomic int g_initialized;
static _Atomic int g_stopping;

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_cond = PTHREAD_COND_INITIALIZER;

static int            g_port = 3000;
static pthread_t      g_consumer;
static int            g_consumer_started;

/* ------------------------------------------------------------------
 * JSON encoder (no library)
 * ------------------------------------------------------------------ */

/* Append `n` bytes from `src` to `buf`, advancing `*pos`. Returns 0 on
   success, -1 if the buffer would overflow. */
static int buf_append(char *buf, int cap, int *pos, const char *src, int n) {
    if (*pos + n > cap) return -1;
    memcpy(buf + *pos, src, (size_t)n);
    *pos += n;
    return 0;
}

/* Append a uint64 as decimal digits. */
static int buf_append_u64(char *buf, int cap, int *pos, uint64_t v) {
    char tmp[32];
    int n = 0;
    if (v == 0) {
        tmp[n++] = '0';
    } else {
        while (v > 0) {
            tmp[n++] = (char)('0' + (v % 10));
            v /= 10;
        }
    }
    /* Reverse into place. */
    for (int i = 0; i < n / 2; ++i) {
        char t = tmp[i];
        tmp[i] = tmp[n - 1 - i];
        tmp[n - 1 - i] = t;
    }
    return buf_append(buf, cap, pos, tmp, n);
}

/* Append a uint64 as `0x<lowercase-hex>`. */
static int buf_append_ptr_hex(char *buf, int cap, int *pos, uint64_t v) {
    static const char hex[] = "0123456789abcdef";
    char tmp[18]; /* "0x" + up to 16 hex digits */
    int n = 0;
    tmp[n++] = '0';
    tmp[n++] = 'x';
    if (v == 0) {
        tmp[n++] = '0';
    } else {
        char digits[16];
        int dc = 0;
        while (v > 0) {
            digits[dc++] = hex[v & 0xF];
            v >>= 4;
        }
        for (int i = dc - 1; i >= 0; --i) tmp[n++] = digits[i];
    }
    return buf_append(buf, cap, pos, tmp, n);
}

/* Append a float with a fixed `%g`-style format: trim trailing zeros, fall
   back to scientific notation only if `g` would produce something weird
   (we keep it simple: %.9g-equivalent up to 9 significant digits). */
static int buf_append_f32(char *buf, int cap, int *pos, float f) {
    char tmp[32];
    int n = snprintf(tmp, sizeof(tmp), "%.9g", (double)f);
    if (n < 0) return -1;
    if (n >= (int)sizeof(tmp)) n = (int)sizeof(tmp) - 1;
    return buf_append(buf, cap, pos, tmp, n);
}

static const char *backend_name(uint8_t b) {
    switch (b) {
        case 0:  return "Slab";
        case 1:  return "Buddy";
        case 2:  return "System";
        case 3:  return "Arena";
        default: return NULL; /* 0xFF or anything else → omit (matches Rust `Option::is_none`) */
    }
}

static const char *op_name(uint8_t op) {
    return op == 0 ? "alloc" : "free";
}

/* Format a single record as one JSON object (no trailing newline). Writes
   up to `cap - 1` bytes plus a NUL. Returns bytes written excluding NUL, or
   -1 if the buffer is too small. */
static int format_record_impl(char *buf, int cap, const LohallocTelemetryRecord *rec) {
    int pos = 0;
    /* Opening brace. */
    if (buf_append(buf, cap, &pos, "{", 1) < 0) return -1;

    /* timestamp */
    if (buf_append(buf, cap, &pos, "\"timestamp\":", 12) < 0) return -1;
    if (buf_append_u64(buf, cap, &pos, rec->timestamp) < 0) return -1;

    /* op */
    if (buf_append(buf, cap, &pos, ",\"op\":\"", 7) < 0) return -1;
    {
        const char *name = op_name(rec->op);
        int len = (int)strlen(name);
        if (buf_append(buf, cap, &pos, name, len) < 0) return -1;
    }
    if (buf_append(buf, cap, &pos, "\"", 1) < 0) return -1;

    /* size */
    if (buf_append(buf, cap, &pos, ",\"size\":", 8) < 0) return -1;
    if (buf_append_u64(buf, cap, &pos, (uint64_t)rec->size) < 0) return -1;

    /* stack_hash */
    if (buf_append(buf, cap, &pos, ",\"stack_hash\":", 14) < 0) return -1;
    if (buf_append_u64(buf, cap, &pos, rec->stack_hash) < 0) return -1;

    /* thread_id */
    if (buf_append(buf, cap, &pos, ",\"thread_id\":", 13) < 0) return -1;
    if (buf_append_u64(buf, cap, &pos, (uint64_t)rec->thread_id) < 0) return -1;

    /* result_ptr */
    if (buf_append(buf, cap, &pos, ",\"result_ptr\":\"", 15) < 0) return -1;
    if (buf_append_ptr_hex(buf, cap, &pos, rec->result_ptr) < 0) return -1;
    if (buf_append(buf, cap, &pos, "\"", 1) < 0) return -1;

    /* latency_ns */
    if (buf_append(buf, cap, &pos, ",\"latency_ns\":", 14) < 0) return -1;
    if (buf_append_u64(buf, cap, &pos, rec->latency_ns) < 0) return -1;

    /* fragmentation_pct */
    if (buf_append(buf, cap, &pos, ",\"fragmentation_pct\":", 21) < 0) return -1;
    if (buf_append_f32(buf, cap, &pos, rec->fragmentation_pct) < 0) return -1;

    /* backend (only for allocs where backend is a known variant; matches
       the Rust TelemetryRecord which uses `Option<Backend>` with
       `skip_serializing_if = "Option::is_none"`). Free records and unknown
       backends are omitted. */
    const char *bn = backend_name(rec->backend);
    if (bn != NULL) {
        if (buf_append(buf, cap, &pos, ",\"backend\":\"", 12) < 0) return -1;
        int blen = (int)strlen(bn);
        if (buf_append(buf, cap, &pos, bn, blen) < 0) return -1;
        if (buf_append(buf, cap, &pos, "\"", 1) < 0) return -1;
    }

    /* Closing brace. */
    if (buf_append(buf, cap, &pos, "}", 1) < 0) return -1;

    if (pos < cap) buf[pos] = '\0';
    return pos;
}

/* ------------------------------------------------------------------
 * HTTP POST
 * ------------------------------------------------------------------ */

/* Open a TCP connection to 127.0.0.1:<port>. Returns the socket fd, or -1
   on failure. `*out_errno` receives the relevant `errno`. */
static int http_connect(int port, int *out_errno) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { if (out_errno) *out_errno = errno; return -1; }

    /* Reasonable timeouts so a stalled server can't wedge the consumer. */
    struct timeval tv = { .tv_sec = 2, .tv_usec = 0 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port   = htons((uint16_t)port);
    inet_pton(AF_INET, "127.0.0.1", &sa.sin_addr);

    if (connect(fd, (struct sockaddr *)&sa, sizeof(sa)) < 0) {
        if (out_errno) *out_errno = errno;
        close(fd);
        return -1;
    }
    return fd;
}

/* Send `len` bytes from `buf` over `fd`. Returns 0 on success, -1 on
   failure. Loops until all bytes are written (short-write safe). */
static int send_all(int fd, const char *buf, size_t len) {
    size_t off = 0;
    while (off < len) {
        ssize_t n = send(fd, buf + off, len - off, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) return -1;
        off += (size_t)n;
    }
    return 0;
}

/* Read until the peer closes (or we time out). Discards the body — we only
   care that the server accepted the request. */
static void drain_response(int fd) {
    char scratch[1024];
    for (;;) {
        ssize_t n = recv(fd, scratch, sizeof(scratch), 0);
        if (n <= 0) return;
    }
}

/* POST a JSON body to 127.0.0.1:<port>/api/telemetry. Retries once on
   connection failure. Returns 0 on success, -1 on permanent failure (any
   non-recoverable error including the second failure). */
static int http_post_telemetry(int port, const char *body, size_t body_len) {
    char header[512];
    int hdr_len = snprintf(header, sizeof(header),
        "POST /api/telemetry HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "\r\n",
        body_len);
    if (hdr_len <= 0 || (size_t)hdr_len >= sizeof(header)) return -1;

    for (int attempt = 0; attempt < 2; ++attempt) {
        int err = 0;
        int fd = http_connect(port, &err);
        if (fd < 0) continue; /* try once more */

        int ok = 1;
        if (send_all(fd, header, (size_t)hdr_len) < 0) ok = 0;
        if (ok && send_all(fd, body, body_len) < 0) ok = 0;
        if (ok) drain_response(fd);
        close(fd);
        if (ok) return 0;
        /* fall through and retry */
    }
    return -1;
}

/* ------------------------------------------------------------------
 * Consumer thread
 * ------------------------------------------------------------------ */

/* Build a JSON array from `n` records into `body` (cap = HTTP_BUF_SIZE).
   Returns the byte length written, or -1 on overflow. */
static int build_batch_json(char *body, int cap,
                            const LohallocTelemetryRecord *batch, int n) {
    int pos = 0;
    if (buf_append(body, cap, &pos, "[", 1) < 0) return -1;
    for (int i = 0; i < n; ++i) {
        if (i > 0) {
            if (buf_append(body, cap, &pos, ",", 1) < 0) return -1;
        }
        int written = format_record_impl(body + pos, cap - pos, &batch[i]);
        if (written < 0) return -1;
        pos += written;
    }
    if (buf_append(body, cap, &pos, "]", 1) < 0) return -1;
    if (pos < cap) body[pos] = '\0';
    return pos;
}

/* Snapshot the current ring contents up to BATCH_SIZE records into `batch`.
   Updates `*tail_out` to the new tail position. The caller must hold the
   mutex. Returns the number of records copied. */
static int snapshot_batch(LohallocTelemetryRecord *batch, uint64_t *tail_out) {
    uint64_t head = g_head;
    uint64_t tail = g_tail;
    if (head <= tail) return 0;
    uint64_t avail = head - tail;
    int n = (int)(avail > BATCH_SIZE ? BATCH_SIZE : avail);
    for (int i = 0; i < n; ++i) {
        batch[i] = g_ring[(tail + (uint64_t)i) & RING_MASK];
    }
    *tail_out = tail + (uint64_t)n;
    return n;
}

static void *consumer_main(void *arg) {
    (void)arg;
    LohallocTelemetryRecord *batch =
        (LohallocTelemetryRecord *)malloc(sizeof(LohallocTelemetryRecord) * BATCH_SIZE);
    char *body = (char *)malloc(HTTP_BUF_SIZE);
    if (!batch || !body) {
        free(batch);
        free(body);
        return NULL;
    }

    for (;;) {
        pthread_mutex_lock(&g_lock);
        while (g_head == g_tail && !atomic_load(&g_stopping)) {
            pthread_cond_wait(&g_cond, &g_lock);
        }
        uint64_t new_tail = g_tail;
        int n = snapshot_batch(batch, &new_tail);
        pthread_mutex_unlock(&g_lock);

        if (n > 0) {
            int body_len = build_batch_json(body, HTTP_BUF_SIZE, batch, n);
            if (body_len > 0) {
                /* Best-effort: ignore failure, the ring's tail only advances
                   on the next iteration so a transient failure will retry. */
                if (http_post_telemetry(g_port, body, (size_t)body_len) == 0) {
                    pthread_mutex_lock(&g_lock);
                    g_tail = new_tail;
                    pthread_mutex_unlock(&g_lock);
                }
                /* On failure we leave g_tail unchanged so the same batch is
                   retried on the next wakeup. */
            }
        }

        if (atomic_load(&g_stopping) && g_head == g_tail) break;
    }

    free(batch);
    free(body);
    return NULL;
}

/* ------------------------------------------------------------------
 * Initialization (lazy, on first emit)
 * ------------------------------------------------------------------ */

static void read_env_port(void) {
    const char *p = getenv("LOHALLOC_OBS_PORT");
    if (!p || !*p) return;
    char *end = NULL;
    long v = strtol(p, &end, 10);
    if (end != p && v >= 1 && v <= 65535) {
        g_port = (int)v;
    }
}

static void atexit_flush(void) {
    /* Signal the consumer to drain and exit. */
    atomic_store(&g_stopping, 1);
    pthread_mutex_lock(&g_lock);
    pthread_cond_broadcast(&g_cond);
    pthread_mutex_unlock(&g_lock);

    if (g_consumer_started) {
        pthread_join(g_consumer, NULL);
        g_consumer_started = 0;
    }
}

static void init_once(void) {
    /* Only the first caller proceeds; others spin until init is done. The
       atomic CAS provides both the "exactly once" guarantee and a memory
       barrier so all subsequent calls see fully-published state. */
    int expected = 0;
    if (!atomic_compare_exchange_strong(&g_initialized, &expected, 1)) {
        /* Another thread is initializing. Spin briefly. */
        while (atomic_load(&g_initialized) != 2) { /* spin */ }
        return;
    }

    read_env_port();
    atexit(atexit_flush);

    if (pthread_create(&g_consumer, NULL, consumer_main, NULL) == 0) {
        g_consumer_started = 1;
    }

    atomic_store(&g_initialized, 2);
}

/* ------------------------------------------------------------------
 * Public sink
 * ------------------------------------------------------------------ */

void lohalloc_telemetry_emit(const LohallocTelemetryRecord *rec) {
    if (!rec) return;

    /* Lazy init on first call. */
    if (atomic_load(&g_initialized) == 0) {
        init_once();
    }

    pthread_mutex_lock(&g_lock);
    uint64_t head = g_head;
    uint64_t tail = g_tail;
    if (head - tail >= RING_CAPACITY) {
        /* Ring is full — drop and account. */
        atomic_fetch_add(&g_dropped, 1);
        pthread_mutex_unlock(&g_lock);
        return;
    }
    int was_empty = (head == tail);
    g_ring[head & RING_MASK] = *rec;
    g_head = head + 1;
    pthread_mutex_unlock(&g_lock);

    if (was_empty) consumer_wake();
}

/* Public port override. */
void lohalloc_obs_set_port(int port) {
    if (port < 1 || port > 65535) return;
    g_port = port;
}

/* ------------------------------------------------------------------
 * Internal: wake the consumer
 * ------------------------------------------------------------------ */

static void consumer_wake(void) {
    pthread_mutex_lock(&g_lock);
    pthread_cond_signal(&g_cond);
    pthread_mutex_unlock(&g_lock);
}

/* ------------------------------------------------------------------
 * Test-only API (compiled when LOHALLOC_OBS_TESTING is defined)
 * ------------------------------------------------------------------ */

#ifdef LOHALLOC_OBS_TESTING

int lohalloc_obs_format_record(char *buf, int cap, const LohallocTelemetryRecord *rec) {
    if (!buf || cap <= 0 || !rec) return -1;
    return format_record_impl(buf, cap, rec);
}

void lohalloc_obs_test_push(const LohallocTelemetryRecord *recs, int count) {
    if (!recs || count <= 0) return;
    pthread_mutex_lock(&g_lock);
    for (int i = 0; i < count; ++i) {
        uint64_t head = g_head;
        uint64_t tail = g_tail;
        if (head - tail >= RING_CAPACITY) {
            atomic_fetch_add(&g_dropped, 1);
            continue;
        }
        g_ring[head & RING_MASK] = recs[i];
        g_head = head + 1;
    }
    pthread_mutex_unlock(&g_lock);
}

int lohalloc_obs_test_drain_to_buffer(char *buf, int cap) {
    if (!buf || cap < 3) return -1;
    pthread_mutex_lock(&g_lock);
    uint64_t head = g_head;
    uint64_t tail = g_tail;
    int n = (int)(head > tail ? (head - tail) : 0);
    int pos = 0;
    buf[pos++] = '[';
    for (int i = 0; i < n; ++i) {
        if (i > 0 && pos < cap) buf[pos++] = ',';
        int w = format_record_impl(buf + pos, cap - pos, &g_ring[(tail + (uint64_t)i) & RING_MASK]);
        if (w < 0) { pthread_mutex_unlock(&g_lock); return -1; }
        pos += w;
    }
    if (pos < cap) buf[pos++] = ']';
    if (pos < cap) buf[pos] = '\0';
    g_tail = head; /* consumed */
    pthread_mutex_unlock(&g_lock);
    return n;
}

void lohalloc_obs_test_reset(void) {
    pthread_mutex_lock(&g_lock);
    g_head = 0;
    g_tail = 0;
    atomic_store(&g_dropped, 0);
    pthread_mutex_unlock(&g_lock);
}

unsigned long long lohalloc_obs_test_dropped(void) {
    return atomic_load(&g_dropped);
}

unsigned long long lohalloc_obs_test_pending(void) {
    pthread_mutex_lock(&g_lock);
    uint64_t n = g_head - g_tail;
    pthread_mutex_unlock(&g_lock);
    return (unsigned long long)n;
}

#endif /* LOHALLOC_OBS_TESTING */
