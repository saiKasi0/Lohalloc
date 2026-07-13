//! J6 Phase A: the single hot-path TLS block.
//!
//! Before this module, the allocator's per-op thread-local state was spread
//! across five separate `thread_local!` statics (`IN_ALLOC`, `AHR`,
//! `PIN_CACHE`, the magazine's `MAG`, `THREAD_STRIPE`/`ARENA_SPAN`) plus a
//! sixth in `lohalloc-cabi` (`IN_ALLOC_FN`). Each static is its own TLS
//! variable, so under the general-dynamic TLS model a `LD_PRELOAD`ed cdylib
//! paid one `__tls_get_addr` call *per variable per op* — the measured
//! C/C++ interposition tax from the beat-system investigation. glibc's
//! tcache does the whole malloc fast path inside one TLS block.
//!
//! This module merges them into one `HotTls` struct behind a single
//! `thread_local!`. Every accessor references the same variable, so once the
//! `#[inline]` helpers land in a common caller, LLVM computes the TLS base
//! address once per function instead of once per variable — the (c)
//! "TLS merge" item from that investigation, realized on stable Rust
//! (the nightly-only part was `-Z tls-model=initial-exec`, which this does
//! not need).
//!
//! # Layout (deliberate, `#[repr(C)]`)
//!
//! The per-op scalars — re-entrancy depth, cabi export depth, history
//! register, thread stripe, magazine owner + the 12 count bytes — pack into
//! the first cache line, so the common alloc/dealloc touches one line plus
//! the specific magazine slot / pin entry it needs. The bulk arrays
//! (magazine slots ~3 KiB, pin cache 2 KiB) sit behind.
//!
//! # TLS discipline (unchanged, load-bearing)
//!
//! Everything is a dtor-free `Cell` in a `const`-initialized
//! `thread_local!` — no `Drop`, no interior heap, so first touch never
//! allocates and thread teardown never re-enters the allocator (see
//! `magazine.rs`'s module doc for the invariant's history). The training-only
//! `REWARD_TRACK` table (12 KiB) and the `telemetry-observer`-gated
//! `ALLOC_START_NS` deliberately stay *outside* this block: neither is on
//! the frozen hot path, and folding them in would only bloat the block every
//! thread touches.

use core::cell::Cell;

use crate::magazine::{MAG_CLASSES, MAG_MAX_CAP};
use crate::perfect_hash;

/// Slots in the per-thread direct-mapped pin cache. Indexed by the raw leaf
/// return address ALONE — one slot serves *every* size class of a site via
/// the per-sc verdict array below. 64 × 32 B = 2 KiB.
///
/// The first cut keyed slots on `(ret0, size_class)`; a mixed workload
/// spraying ~15 size classes from one site then held ~15 colliding keys
/// that ping-pong-evicted each other, so *every* allocation took the miss
/// path (one-frame derivation + cold distilled lookup + a shared-cacheline
/// miss-counter bump) — measured +24-41% on the adv-mixed/mt-mixed rows.
/// Per-site slots make a site's verdicts coreside: adv-mixed occupies 1-2
/// slots total and the miss path is genuinely once per (site, sc).
pub(crate) const PIN_ENTRIES: usize = 64;

/// Per-slot verdict-array width. `state::size_class_for` yields 0..=14
/// (12 Slab classes, 2 Buddy, 1 System), so 16 covers every class; an
/// out-of-range sc (impossible today) bypasses the cache entirely.
pub(crate) const PIN_SC_SLOTS: usize = 16;

/// Verdict byte: this size class has not been probed against the distilled
/// table yet.
pub(crate) const PIN_UNKNOWN: u8 = 0xFE;

/// Verdict byte: probed, and the (site, size_class) is NOT pinnable — the
/// negative cache that keeps non-distilled sites from re-paying the
/// one-frame derivation + distilled lookup on every allocation. Values 0–3
/// are `Backend as u8`.
pub(crate) const PIN_NOT_PINNED: u8 = 0xFF;

/// One pin-cache slot: a call site (raw leaf return address) plus one
/// verdict byte per size class. `ret0 == 0` marks an empty slot
/// (`walk_leaf` rejects zero return addresses before the cache is ever
/// probed). `table` tags the `FrozenRouting` snapshot the verdicts were
/// derived from: a probe under a different published pointer is a miss,
/// which makes `reset_to_training()` / re-`load()` invalidation and
/// multi-instance isolation automatic — no flush protocol, entries from a
/// stale table or another `Lohalloc` instance simply never match. (Pointer
/// equality after a free-reuse of the same address is impossible here:
/// frozen tables are deliberately leaked, never freed — see
/// `frozen_table`'s doc.)
pub(crate) struct PinEntry {
    pub(crate) table: Cell<*const perfect_hash::FrozenRouting>,
    pub(crate) ret0: Cell<usize>,
    pub(crate) states: [Cell<u8>; PIN_SC_SLOTS],
}

/// This thread's private arena bump window (Ladder 5 span carve — see
/// `Lohalloc::arena_alloc_fast`). Validity is (owner, epoch)-checked on
/// every use, so instance mixing and `reset_arena()` both simply discard
/// the span.
pub(crate) struct ArenaSpan {
    pub(crate) owner: Cell<u64>,
    pub(crate) epoch: Cell<u64>,
    pub(crate) cursor: Cell<usize>,
    pub(crate) end: Cell<usize>,
}

/// The merged per-thread hot block. Field order is layout (`repr(C)`):
/// scalars first (one cache line), bulk arrays after.
#[repr(C)]
pub(crate) struct HotTls {
    /// Re-entrancy depth. >0 means we are already inside `alloc`/`dealloc`
    /// on this thread — any further allocation must bypass to `mmap`
    /// directly. (Formerly the standalone `IN_ALLOC` static.)
    pub(crate) in_alloc: Cell<usize>,
    /// `lohalloc-cabi`'s export-entry depth (formerly its own
    /// `IN_ALLOC_FN` TLS — see the pub accessors at the bottom). Tracks how
    /// deep this thread is inside the exported `malloc`/`free` family,
    /// which is a *different* question from `in_alloc` (a nested `malloc`
    /// triggered by allocator internals arrives with `in_alloc > 0` but
    /// export depth 0 — deadlock #4's mechanism).
    pub(crate) export_depth: Cell<usize>,
    /// The allocation-history register: 2-bit event codes, LSB = most
    /// recent. (Formerly the standalone `AHR` static.)
    pub(crate) ahr: Cell<u64>,
    /// This thread's central-backend stripe (Ladder 4 C3/C4), assigned
    /// round-robin on first use. `usize::MAX` = unassigned. Process-wide
    /// (not per-instance): a stripe index is just a load-spreading hint on
    /// the alloc side — frees always resolve ownership through the
    /// per-instance registry, so instance mixing is harmless here.
    pub(crate) thread_stripe: Cell<usize>,
    /// Magazine owner id (see `magazine.rs`'s ownership doc — a SIGSEGV
    /// taught us this). `0` = unassigned.
    pub(crate) mag_owner: Cell<u64>,
    /// J6 Phase C last-segment cache: the most recent headerless-slab free's
    /// segment base and its class, tagged with the owning instance's
    /// magazine id. Lets the free fast lane skip the shared
    /// `SegmentRegistry` probe when consecutive frees land in the same
    /// segment (the common churn case). Filled ONLY from verified registry
    /// hits (`Lohalloc::seg_cache_fill`), so a hit implies the pointer lies
    /// inside a registered headerless slab segment of that instance —
    /// segments are never unmapped, and `seg_owner` (a magazine id) rolls
    /// on every reload/instance switch, exactly like the magazine itself.
    pub(crate) seg_base: Cell<usize>,
    /// Class of `seg_base`'s segment (slab segments are single-class).
    pub(crate) seg_class: Cell<u8>,
    /// Instance tag for `seg_base` (0 = empty; magazine ids are never 0).
    pub(crate) seg_owner: Cell<u64>,
    /// Per-class magazine fill counts (12 bytes — completes the first
    /// cache line together with the scalars above).
    pub(crate) mag_counts: [Cell<u8>; MAG_CLASSES],
    /// The private arena bump window (owner/epoch-validated).
    pub(crate) span: ArenaSpan,
    /// Magazine block stacks, one per slab class (~3 KiB).
    pub(crate) mag_slots: [[Cell<*mut u8>; MAG_MAX_CAP]; MAG_CLASSES],
    /// The Ladder 6 inference pin cache (2 KiB).
    pub(crate) pin: [PinEntry; PIN_ENTRIES],
}

thread_local! {
    /// The one hot-path TLS variable. Const-init + no `Drop` anywhere in
    /// the struct keeps std's `thread_local!` on its zero-check fast path.
    static HOT: HotTls = const {
        HotTls {
            in_alloc: Cell::new(0),
            export_depth: Cell::new(0),
            ahr: Cell::new(0),
            thread_stripe: Cell::new(usize::MAX),
            mag_owner: Cell::new(0),
            seg_base: Cell::new(0),
            seg_class: Cell::new(0),
            seg_owner: Cell::new(0),
            mag_counts: [const { Cell::new(0) }; MAG_CLASSES],
            span: ArenaSpan {
                owner: Cell::new(0),
                epoch: Cell::new(0),
                cursor: Cell::new(0),
                end: Cell::new(0),
            },
            mag_slots: [const { [const { Cell::new(core::ptr::null_mut()) }; MAG_MAX_CAP] };
                MAG_CLASSES],
            pin: [const {
                PinEntry {
                    table: Cell::new(core::ptr::null()),
                    ret0: Cell::new(0),
                    states: [const { Cell::new(PIN_UNKNOWN) }; PIN_SC_SLOTS],
                }
            }; PIN_ENTRIES],
        }
    };
}

/// Run `f` against this thread's hot block. `#[inline(always)]` so every
/// use in a common caller resolves the same TLS variable — the whole point
/// of the merge.
#[inline(always)]
pub(crate) fn with<R>(f: impl FnOnce(&HotTls) -> R) -> R {
    HOT.with(f)
}

// ---------------------------------------------------------------------------
// Re-entrancy depth accessors (the former `IN_ALLOC` API shape).
// ---------------------------------------------------------------------------

#[inline(always)]
pub(crate) fn in_alloc_get() -> usize {
    with(|h| h.in_alloc.get())
}

#[inline(always)]
pub(crate) fn in_alloc_set(v: usize) {
    with(|h| h.in_alloc.set(v))
}

// ---------------------------------------------------------------------------
// cabi export-depth accessors — `pub` so `lohalloc-cabi` shares this block
// instead of declaring its own TLS variable (one fewer `__tls_get_addr`
// per exported call). `#[inline]` so the access compiles into the cabi
// object code directly.
// ---------------------------------------------------------------------------

/// Read this thread's cabi export depth (see `HotTls::export_depth`).
#[inline]
pub fn export_depth_get() -> usize {
    with(|h| h.export_depth.get())
}

/// Set this thread's cabi export depth (see `HotTls::export_depth`).
#[inline]
pub fn export_depth_set(v: usize) {
    with(|h| h.export_depth.set(v))
}
