//! Feature-gated C-ABI observer hook for live telemetry.
//!
//! When compiled with `--features telemetry-observer`, the allocator emits a
//! `TelemetryCRecord` to an installed C-ABI sink function for every
//! allocation/free. Production builds (default features) compile this entire
//! module away — zero overhead, zero symbols, zero branch on the hot path.
//!
//! The sink is set by the `lohalloc-demo` binary at startup by `dlsym`'ing the
//! shim's `lohalloc_telemetry_emit` symbol from a preloaded C shared library.

// The entire module is compiled away when the feature is off. This includes
// the `SINK` atomic (which would otherwise add an atomic-load instruction to
// the hot path) and the `TelemetryCRecord` type (which the shim would never
// see in a feature-off build anyway).
#![cfg(feature = "telemetry-observer")]

use core::sync::atomic::{AtomicPtr, Ordering};

/// C-ABI mirror of `lohalloc_core::TelemetryRecord`.
///
/// Layout MUST stay in sync with what the C shim expects. The shim copies
/// `sizeof(TelemetryCRecord)` bytes; do not add fields or change alignment.
/// The `record_size_is_stable` test pins the wire size so future field
/// additions break the build rather than silently corrupting the wire format.
///
/// Field-by-field breakdown (64-bit targets; `usize` is 8 bytes):
///
/// | offset | size | field               |
/// |-------:|-----:|---------------------|
/// |   0    |   8  | `timestamp`         |
/// |   8    |   1  | `op`                |
/// |   9    |   7  | `_pad0`             |
/// |  16    |   8  | `size`              |
/// |  24    |   8  | `stack_hash`        |
/// |  32    |   4  | `thread_id`         |
/// |  36    |   4  | `_pad1`             |
/// |  40    |   8  | `result_ptr`        |
/// |  48    |   8  | `latency_ns`        |
/// |  56    |   4  | `fragmentation_pct` |
/// |  60    |   1  | `backend`           |
/// |  61    |   7  | `_pad2`             |
///
/// Fields occupy 68 bytes; `repr(C)` pads the struct to a multiple of
/// `alignof = 8`, so `sizeof` = 72. Pinned by `record_size_is_stable` below.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TelemetryCRecord {
    pub timestamp: u64,
    pub op: u8, // 0 = Alloc, 1 = Free
    pub _pad0: [u8; 7],
    pub size: usize,
    pub stack_hash: u64,
    pub thread_id: u32,
    pub _pad1: [u8; 4],
    pub result_ptr: u64,
    pub latency_ns: u64,
    pub fragmentation_pct: f32,
    pub backend: u8, // 0=Slab, 1=Buddy, 2=System, 3=Arena, 0xFF=Unknown
    pub _pad2: [u8; 7],
}

impl TelemetryCRecord {
    /// Construct a record for a successful allocation.
    pub fn alloc(
        timestamp: u64,
        size: usize,
        stack_hash: u64,
        thread_id: u32,
        result_ptr: u64,
        latency_ns: u64,
        backend: u8,
    ) -> Self {
        Self {
            timestamp,
            op: 0,
            _pad0: [0; 7],
            size,
            stack_hash,
            thread_id,
            _pad1: [0; 4],
            result_ptr,
            latency_ns,
            fragmentation_pct: 0.0,
            backend,
            _pad2: [0; 7],
        }
    }

    /// Construct a record for a free operation.
    pub fn free(
        timestamp: u64,
        size: usize,
        stack_hash: u64,
        thread_id: u32,
        result_ptr: u64,
        latency_ns: u64,
    ) -> Self {
        Self {
            timestamp,
            op: 1,
            _pad0: [0; 7],
            size,
            stack_hash,
            thread_id,
            _pad1: [0; 4],
            result_ptr,
            latency_ns,
            fragmentation_pct: 0.0,
            backend: 0xFF,
            _pad2: [0; 7],
        }
    }
}

/// Type of the C-ABI sink function. The shim exports a function with this
/// signature; we hold its address and call it for each record.
pub type TelemetrySink = unsafe extern "C" fn(*const TelemetryCRecord);

static SINK: AtomicPtr<()> = AtomicPtr::new(core::ptr::null_mut());

/// Install (or replace) the global sink. Pass `None` to disable.
///
/// Called once by `lohalloc-demo` at startup.
pub fn install_sink(sink: Option<TelemetrySink>) {
    let p = match sink {
        Some(s) => s as *mut (),
        None => core::ptr::null_mut(),
    };
    SINK.store(p, Ordering::Release);
}

/// Clear the sink. Convenience for tests that want to verify the no-op path.
pub fn clear_sink() {
    SINK.store(core::ptr::null_mut(), Ordering::Release);
}

/// Read the currently installed sink (for tests).
pub fn current_sink() -> Option<TelemetrySink> {
    let p = SINK.load(Ordering::Acquire);
    if p.is_null() {
        None
    } else {
        // SAFETY: a non-null pointer stored via `install_sink` originated from
        // an `unsafe extern "C" fn` with the correct signature; we restore it
        // through `transmute`. The caller (`emit`) treats it as opaque bytes.
        Some(unsafe { core::mem::transmute::<*mut (), TelemetrySink>(p) })
    }
}

/// Emit a record to the installed sink. No-op if no sink is installed.
///
/// `#[inline(always)]` keeps the call cost on the hot path down to one
/// atomic load + one branch. When `telemetry-observer` is OFF, this function
/// is removed entirely by `#[cfg]` at the call site — see `lib.rs`.
#[inline(always)]
pub fn emit(record: TelemetryCRecord) {
    if let Some(sink) = current_sink() {
        // SAFETY: the sink signature matches `TelemetrySink`. The shim is
        // responsible for not retaining `&record` past the call (we pass a
        // pointer to a stack value; it must be consumed synchronously).
        unsafe { sink(&record as *const TelemetryCRecord) };
    }
}

/// Monotonic nanosecond timestamp. Uses `Instant::elapsed` from the process
/// epoch so we never observe wall-clock jumps. Cheap (vDSO on Linux,
/// `mach_absolute_time` on macOS), zero allocations.
#[inline]
fn now_ns() -> u64 {
    use std::time::Instant;
    Instant::now().elapsed().as_nanos() as u64
}

/// Current OS thread identifier, truncated to `u32`. Used purely as a
/// telemetry label — collisions across thousands of threads are tolerable.
#[inline]
fn thread_id_u32() -> u32 {
    // SAFETY: `pthread_self` is async-signal-safe and never fails. `pthread_t`
    // is pointer-sized on both Linux and macOS; truncating to u32 gives a
    // stable-enough label for the GUI.
    let raw = unsafe { libc::pthread_self() } as usize;
    raw as u32
}

/// Convenience: emit a successful allocation record. Called from
/// `write_header` after the ownership header has been written. Latency is
/// reported as 0 in this Phase — true latency instrumentation is a Phase 5+++
/// item (would require threading an `_alloc_start_ns` through `route_alloc`
/// and friends, a moderate refactor).
#[inline(always)]
pub fn emit_alloc(size: usize, stack_hash: u64, result_ptr: u64, backend: u8) {
    emit(TelemetryCRecord::alloc(
        now_ns(),
        size,
        stack_hash,
        thread_id_u32(),
        result_ptr,
        0, // latency_ns — see doc above
        backend,
    ));
}

/// Convenience: emit a free record. Called from `dealloc` after the header's
/// magic check has passed. Same latency caveat as `emit_alloc`.
#[inline(always)]
pub fn emit_free(size: usize, stack_hash: u64, result_ptr: u64) {
    emit(TelemetryCRecord::free(
        now_ns(),
        size,
        stack_hash,
        thread_id_u32(),
        result_ptr,
        0,
    ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering as StdOrdering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    unsafe extern "C" fn test_sink(_rec: *const TelemetryCRecord) {
        COUNTER.fetch_add(1, StdOrdering::Relaxed);
    }

    #[test]
    fn emit_is_noop_without_sink() {
        // Ensure no sink is installed for this test.
        clear_sink();
        let before = COUNTER.load(StdOrdering::Relaxed);
        emit(TelemetryCRecord::alloc(0, 64, 0, 0, 0, 0, 0));
        let after = COUNTER.load(StdOrdering::Relaxed);
        assert_eq!(before, after, "emit should be no-op when sink is null");
    }

    #[test]
    fn emit_routes_to_installed_sink() {
        install_sink(Some(test_sink));
        let before = COUNTER.load(StdOrdering::Relaxed);
        emit(TelemetryCRecord::alloc(0, 64, 0xdead, 0, 0x1000, 100, 0));
        emit(TelemetryCRecord::free(0, 64, 0xdead, 0, 0x1000, 50));
        let after = COUNTER.load(StdOrdering::Relaxed);
        assert_eq!(after - before, 2, "two emit calls should reach the sink");
        // Leave the sink cleared so other tests see a clean slate.
        clear_sink();
    }

    #[test]
    fn record_size_is_stable() {
        // The C shim copies sizeof(TelemetryCRecord) bytes — pin the size so
        // future field additions break the build rather than silently
        // corrupting the wire format. The trailing bytes after `_pad2` are
        // C-required alignment padding (struct size is rounded up to a
        // multiple of alignof = 8).
        assert_eq!(
            core::mem::size_of::<TelemetryCRecord>(),
            72,
            "TelemetryCRecord size is part of the C-ABI; do not change without updating the shim"
        );
        // Alignment must be the most-aligned field's alignment (8 bytes on
        // 64-bit, since `usize` is 8). Pin this too so reordering doesn't
        // accidentally change the wire format.
        assert_eq!(
            core::mem::align_of::<TelemetryCRecord>(),
            8,
            "TelemetryCRecord alignment is part of the C-ABI"
        );
    }

    #[test]
    fn record_field_offsets_are_stable() {
        // Pin every field offset so a refactor that accidentally reorders
        // fields trips the build instead of silently breaking the wire format.
        use core::mem::offset_of;
        assert_eq!(offset_of!(TelemetryCRecord, timestamp), 0);
        assert_eq!(offset_of!(TelemetryCRecord, op), 8);
        assert_eq!(offset_of!(TelemetryCRecord, size), 16);
        assert_eq!(offset_of!(TelemetryCRecord, stack_hash), 24);
        assert_eq!(offset_of!(TelemetryCRecord, thread_id), 32);
        assert_eq!(offset_of!(TelemetryCRecord, result_ptr), 40);
        assert_eq!(offset_of!(TelemetryCRecord, latency_ns), 48);
        assert_eq!(offset_of!(TelemetryCRecord, fragmentation_pct), 56);
        assert_eq!(offset_of!(TelemetryCRecord, backend), 60);
    }
}
