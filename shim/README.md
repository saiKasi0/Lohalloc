# liblohalloc_obs — Lohalloc C-ABI telemetry shim

A small C shared library that bridges Lohalloc's `telemetry-observer`
C-ABI sink to the Lohalloc Axum server (`POST /api/telemetry`).

## What it does

```
┌──────────────────────────────────┐
│ Rust allocator (hot path)        │
│   emit_alloc() / emit_free()     │
└────────────────┬─────────────────┘
                 │ extern "C" call
                 ▼
┌──────────────────────────────────┐
│ lohalloc_telemetry_emit()        │  ← this shim
│   - copies 72 bytes into ring    │
│   - signals cond on 0→1          │
│   - drops on full (counter++)    │
└────────────────┬─────────────────┘
                 │ pthread_cond signal
                 ▼
┌──────────────────────────────────┐
│ Consumer thread                  │
│   - batches up to 256 records    │
│   - JSON-encodes                 │
│   - POSTs to 127.0.0.1:<port>    │
│     /api/telemetry               │
└──────────────────────────────────┘
```

The producer side holds a single global mutex around a 65,536-entry ring.
Per the task spec the producer can be MPSC (the allocator may call from
multiple threads), so we keep the lock on the producer side too; contention
is fine for a demo. The hot path is dominated by `route_alloc` work, not
this sink.

On the very first emit we lazily spawn the consumer pthread and register an
`atexit` handler that signals shutdown, drains the ring one last time, and
joins the consumer.

## Build

```bash
cd shim
make            # produces build/liblohalloc_obs.dylib (mac) or build/liblohalloc_obs.so (Linux)
make test       # runs build/test_ring (unit tests for the buffer + JSON encoder)
```

Both targets produce **zero compiler warnings** with `-Wall -Wextra`.

## Use with the Lohalloc example binary

The `lohalloc-example` binary now ships with built-in shim sink support.
Build it with `--features install-shim-sink` and preload the shim:

```bash
# macOS
DYLD_INSERT_LIBRARIES=$PWD/build/liblohalloc_obs.dylib \
  cargo run -p lohalloc-example --release --features install-shim-sink -- --duration-secs 60

# Linux
LD_PRELOAD=$PWD/build/liblohalloc_obs.so \
  cargo run -p lohalloc-example --release --features install-shim-sink -- --duration-secs 60
```

The `install-shim-sink` feature makes the binary `dlsym` the shim's
`lohalloc_telemetry_emit` symbol at startup and install it as the
allocator's observer sink. The `telemetry-observer` feature on
`lohalloc-alloc` (pulled in automatically) enables the emit calls on the
hot path.

For the dedicated training-demo binary that always installs the sink:

```bash
# macOS
DYLD_INSERT_LIBRARIES=$PWD/build/liblohalloc_obs.dylib \
  cargo run -p lohalloc-demo --features install-shim-sink --release

# Linux
LD_PRELOAD=$PWD/build/liblohalloc_obs.so \
  cargo run -p lohalloc-demo --features install-shim-sink --release
```

## Configuration

| Knob                | Default     | How                                          |
|---------------------|-------------|----------------------------------------------|
| Server port         | `3000`      | `LOHALLOC_OBS_PORT=4000 ./your_app`          |
| Server port (C)     | `3000`      | `lohalloc_obs_set_port(4000);` before first emit |

The environment variable is read **once**, on first emit. Subsequent
changes to `$LOHALLOC_OBS_PORT` are ignored — use `lohalloc_obs_set_port`
from C if you need dynamic reconfiguration (e.g. for tests).

## Threading model

- **Producer** (allocator thread, possibly many): grabs `g_lock`, copies
  one 72-byte record into the next ring slot, increments `g_head`,
  signals `g_cond` if the ring just transitioned empty → non-empty,
  releases the lock.
- **Consumer** (one pthread): waits on `g_cond`, snapshots up to 256
  records, releases the lock, builds a JSON array, POSTs over a
  short-lived TCP connection, then advances `g_tail`.

The ring uses an unsigned 64-bit head/tail pair; the consumer treats
`g_head - g_tail` as the pending count. Because capacity is a power of
two (65,536), the slot index is `head & 0xFFFF`. The pair will not
overflow in any realistic timeframe (~584 years at 1 Gemit/s).

## Drop semantics

If the ring is full when `lohalloc_telemetry_emit` is called, the record
is **silently dropped** and an atomic `dropped` counter is incremented.
The allocator never blocks on telemetry. `atexit` joins the consumer
thread, so by the time the process exits the ring has been drained (or
its contents have been lost — the drop counter is not currently
persisted; the demo binary inspects it via `dlsym` if needed).

## HTTP details

We speak raw HTTP/1.1 over POSIX sockets — no libcurl dependency.

```
POST /api/telemetry HTTP/1.1
Host: 127.0.0.1
Content-Type: application/json
Content-Length: <n>
Connection: close

[<json-array>]
```

`Connection: close` lets us read until EOF to discard the response body
without managing chunked transfer encoding. On connection failure we
retry once; on the second failure we leave the ring's tail unchanged
so the batch is re-attempted on the next wakeup.

## Wire format

Each record is encoded as:

```json
{"timestamp":<u64>,"op":"<alloc|free>","size":<usize>,"stack_hash":<u64>,"thread_id":<u32>,"result_ptr":"0x<hex>","latency_ns":<u64>,"fragmentation_pct":<float>,"backend":"<Slab|Buddy|System|Arena>"}
```

The `backend` field is **omitted** for free records (and for any record
whose backend byte is `0xFF`), matching the Rust `TelemetryRecord` whose
`backend: Option<Backend>` is `skip_serializing_if = "Option::is_none"`.

The consumer batches up to 256 records into a JSON array:

```json
[
  {"timestamp":...,"op":"alloc",...,"backend":"Slab"},
  {"timestamp":...,"op":"free",...}
]
```

This matches the server's `POST /api/telemetry` schema, which accepts
both a single object and an array.

## File layout

```
shim/
  Makefile             — build rules (auto-picks .dylib vs .so, outputs to build/)
  lohalloc_obs.h       — public header (C-ABI mirror of TelemetryCRecord)
  lohalloc_obs.c       — sink + ring + consumer + JSON encoder + HTTP POST
  test_ring.c          — unit tests (compiled with -DLOHALLOC_OBS_TESTING)
  build/               — compiled artifacts (gitignored):
    liblohalloc_obs.{so,dylib}  — the preloadable shim library
    test_ring                  — the test binary
    *.o                        — object files
  README.md            — this file
```

This directory is **not** part of the Cargo workspace — it's a
standalone C project. Build it independently with `make`.

## Testing

```bash
make test
```

Runs `test_ring`, which covers:

- `sizeof(LohallocTelemetryRecord) == 72` (the wire-format contract)
- JSON encoding for an alloc record (all fields present, `backend` set)
- JSON encoding for a free record (`backend` omitted)
- JSON encoding for all four backends (Slab/Buddy/System/Arena)
- Ring push/drain round-trip
- Ring overflow (drops 100 records beyond capacity)
- Public sink call path (smoke test)
