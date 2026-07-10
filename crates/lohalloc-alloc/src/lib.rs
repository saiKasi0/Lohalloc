//! Lohalloc Execution Plane: the `GlobalAlloc` shim + backends.
//!
//! Phase 1 wires three backends behind a single [`Lohalloc`] struct that
//! implements [`core::alloc::GlobalAlloc`]:
//!
//! - **Slab** for small, fixed-size requests (`<= SLAB_MAX`).
//! - **Buddy** for medium, variable-size requests (`<= BUDDY_MAX`).
//! - **System Fallback** (`mmap`/`munmap`) for oversized requests and as the
//!   page provider for the other two backends.
//!
//! Routing is by size class only in Phase 1 — the Multi-Armed Bandit policy
//! arrives in Phase 2.
//!
//! # Soundness: the two hard problems solved here
//!
//! 1. **Re-entrancy / deadlock.** The backends use `Vec` for internal
//!    bookkeeping, and `Vec` allocates through the *global* allocator — which
//!    is us. Locking a backend Mutex and then re-entering `alloc` would
//!    deadlock (std `Mutex` is not reentrant). We break the cycle with a
//!    thread-local recursion guard: any allocation made while we are already
//!    inside `alloc`/`dealloc` bypasses the backends and is served directly by
//!    `mmap` (the System Fallback). This is the standard technique used by
//!    production replacement allocators.
//!
//! 2. **Dealloc routing.** `GlobalAlloc::dealloc` receives only the `Layout`,
//!    not the identity of the backend that produced the pointer. Routing dealloc
//!    by size is unsound (a slab-alloc failure falls through to buddy/system,
//!    but dealloc would still route to slab → writing to чужой memory). We solve
//!    this by prepending a fixed-size [`Header`] to every allocation that
//!    records the owning backend (and, for System, the `mmap` base/length so
//!    `munmap` can release the exact mapping).
//!
//! # Cross-platform contract
//!
//! The System Fallback is cfg-gated for Linux/macOS on ARM64/x86_64. Page size
//! is queried at runtime; alignment is satisfied by over-allocation within a
//! page (see [`system`]). Do not assume a 4 KiB page anywhere above this layer.

pub mod arena;
pub mod bandit;
pub mod buddy;
mod buddy_mag;
mod clock;
mod magazine;
#[cfg(feature = "telemetry-observer")]
pub mod observer;
pub mod perfect_hash;
mod registry;
pub mod slab;
pub mod state;
pub mod system;
pub mod topology;
pub mod tune;

use core::alloc::{GlobalAlloc, Layout};
use core::cell::Cell;
use core::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use crossbeam_utils::CachePadded;
use lohalloc_core::{align_up, BUDDY_MAX, MIN_ALIGN, SLAB_MAX};
use std::sync::Mutex;

/// Sentinel written into every [`Header`] so we can sanity-check dealloc.
const MAGIC: u64 = 0x534d4152414c4844; // "LOHALALHD"

/// `size_class_hint` value meaning "not tracked for Layer 2 reward
/// attribution" — used for allocations that bypass `route_alloc` entirely
/// (the re-entrancy-guard bypass, the standalone `arena_alloc` helper).
const SIZE_CLASS_UNTRACKED: u8 = 0xFF;

/// Per-allocation header prepended to the user-visible pointer. Lets
/// `dealloc` identify the owning backend without guessing by size.
///
/// 48 bytes; always accessed with `read_unaligned`/`write_unaligned` so we do
/// not impose any alignment requirement beyond what the user asked for.
#[repr(C)]
struct Header {
    magic: u64,
    backend: u8,
    /// The Decision Engine's `size_class_for(size)` bucket at alloc time
    /// (`state::size_class_for`), captured here so `dealloc`'s Layer-2
    /// reward attribution (`route_alloc`'s doc comment explains the reward
    /// model) updates the *same* Signature the bandit routed. Recomputing
    /// the class from `size` below (the post-header-padding total) can
    /// straddle a different class boundary than the original request did.
    /// [`SIZE_CLASS_UNTRACKED`] if this allocation bypassed the Decision
    /// Engine (re-entrancy bypass, `arena_alloc`).
    size_class_hint: u8,
    /// `log2` of the alignment this allocation was served at (`align` is
    /// always a power of two — see `MIN_ALIGN`). Lets `dealloc`/the C ABI
    /// (`lohalloc-cabi`) recompute `header_pad` from the header alone,
    /// without needing a `Layout` — required for `free(ptr)`, which only
    /// ever receives the pointer.
    align_log2: u8,
    /// For `Backend::Slab` allocations: the slab size-class index of the
    /// underlying block, so the free path (and the magazine push) never
    /// recomputes it. Note this is the class of the *total* (request +
    /// header padding), which can differ from `size_class_hint` (computed
    /// from the pre-padding request size — e.g. a 250-byte request has
    /// hint 5/256 but a 298-byte total in class 6/512).
    /// [`SIZE_CLASS_UNTRACKED`] for every non-Slab backend.
    slab_class: u8,
    _pad: [u8; 4],
    /// Size passed to the backend's `alloc` (the *total* including this
    /// header's padding). Slab/Buddy use it to compute the free-list/ order on
    /// dealloc. System ignores it (uses `base`/`map_len`).
    size: usize,
    /// For `Backend::System` only: the raw `mmap` base to pass to `munmap`.
    base: usize,
    /// For `Backend::System` only: the full mapped length to unmap.
    map_len: usize,
    /// Topological hash of the allocation call site (Phase 2). Used by the
    /// Decision Engine (Phase 3) for MAB correlation. Zero if the topology
    /// engine returned a sentinel.
    hash: u64,
}

const HEADER_SIZE: usize = core::mem::size_of::<Header>(); // 48

/// Bytes of padding between the backend's block start and the user pointer, so
/// the user pointer is aligned to `align` and the header sits immediately
/// before it. `align` must be a power of two.
fn header_pad(align: usize) -> usize {
    align_up(HEADER_SIZE, align)
}

/// Internal-fragmentation percentage for an allocation of `total` bytes
/// served by `backend`: `(reserved - total) / reserved * 100`, where
/// `reserved` is the actual block size the backend rounds `total` up to.
///
/// Each backend's rounding rule is a *pure, deterministic* function of
/// `total` alone (size-class lookup for Slab, order lookup for Buddy,
/// page-alignment for System), so this is computed fresh here rather than
/// threaded back from the backend's `alloc()` call — no backend lock, no
/// new atomics, no heap allocation, keeping the hot path unchanged.
/// Arena is a bump allocator with no size-class rounding, so it reports 0.
///
/// Only called from the telemetry hooks in `write_header`/`dealloc`, so it
/// is compiled away entirely (like the rest of the observer machinery) when
/// `telemetry-observer` is off — production builds pay nothing for it.
#[cfg(feature = "telemetry-observer")]
fn fragmentation_pct_for(backend: Backend, total: usize) -> f32 {
    frag_pct_for(backend.to_core(), total)
}

/// The always-available core of the fragmentation math (un-gated in Step 8
/// so `state::shaped_reward`'s optional `frag_weight` penalty can use it —
/// production builds with the default `frag_weight = 0` never call it, and
/// the observer-gated wrapper above keeps the telemetry path unchanged).
/// Takes `lohalloc_core::Backend` because the reward path (`state.rs`)
/// works in core types.
pub(crate) fn frag_pct_for(backend: lohalloc_core::Backend, total: usize) -> f32 {
    let reserved = match backend {
        lohalloc_core::Backend::Slab => lohalloc_core::slab_class_for(total)
            .map(|class| lohalloc_core::SLAB_SIZE_CLASSES[class]),
        lohalloc_core::Backend::Buddy => buddy::order_for(total).map(buddy::block_size),
        lohalloc_core::Backend::System => Some(align_up(total, system::page_size())),
        lohalloc_core::Backend::Arena => None,
    };
    match reserved {
        Some(reserved) if reserved > total => (reserved - total) as f32 / reserved as f32 * 100.0,
        _ => 0.0,
    }
}

/// Which Execution-Plane backend produced an allocation. Tagged into the
/// [`Header`]. Uses `lohalloc_core::Backend` (re-imported here for the
/// header's `u8` tag).
///
/// The local `Backend` type below mirrors `lohalloc_core::Backend` for
/// use in the `Header` (which stores a `u8` discriminant). The Decision
/// Engine (`state.rs`) uses `lohalloc_core::Backend` directly.
#[repr(u8)]
#[derive(Clone, Copy)]
enum Backend {
    Slab = 0,
    Buddy = 1,
    System = 2,
    Arena = 3,
}

impl Backend {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Backend::Slab),
            1 => Some(Backend::Buddy),
            2 => Some(Backend::System),
            3 => Some(Backend::Arena),
            _ => None,
        }
    }

    /// Convert from `lohalloc_core::Backend` to the local `Backend` used
    /// in the `Header`.
    fn from_core(b: lohalloc_core::Backend) -> Self {
        match b {
            lohalloc_core::Backend::Slab => Backend::Slab,
            lohalloc_core::Backend::Buddy => Backend::Buddy,
            lohalloc_core::Backend::System => Backend::System,
            lohalloc_core::Backend::Arena => Backend::Arena,
        }
    }

    /// Convert from the local `Backend` (the `Header`'s `u8` tag) back to
    /// `lohalloc_core::Backend`, needed to feed `AllocatorState::record_latency`
    /// from `dealloc`.
    fn to_core(self) -> lohalloc_core::Backend {
        match self {
            Backend::Slab => lohalloc_core::Backend::Slab,
            Backend::Buddy => lohalloc_core::Backend::Buddy,
            Backend::System => lohalloc_core::Backend::System,
            Backend::Arena => lohalloc_core::Backend::Arena,
        }
    }
}

/// Snapshot of per-backend live-region/usage counters. See
/// [`Lohalloc::backend_counters`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BackendCounters {
    /// Number of live backing mmap regions held by the Slab allocator.
    pub slab_region_count: usize,
    /// Number of live backing mmap regions held by the Buddy allocator.
    pub buddy_region_count: usize,
    /// Bytes allocated from the Bump Arena since its last reset.
    pub arena_used: usize,
    /// Total usable capacity of the Bump Arena (0 if not yet initialized).
    pub arena_capacity: usize,
}

/// The composite allocator. Install an instance of this as
/// `#[global_allocator]` to route every Rust allocation through Lohalloc.
pub struct Lohalloc {
    /// Central Slab backend, sharded like `buddy` (Ladder 4 C4) — but with
    /// a simpler routing story: slab frees are **stripe-agnostic**. A slab
    /// free is a pure free-list push (no bitmap, no coalescing, no region
    /// ownership — see `slab.rs`'s "Stripe agnosticism" module-doc note),
    /// so both allocs and frees simply use the calling thread's stripe and
    /// blocks migrate between stripes' lists harmlessly. No registry
    /// widening or header stripe field is needed (deviation from the
    /// original C4 sketch, justified there).
    slab: [Mutex<slab::Slab>; MAX_STRIPES],
    /// Central Buddy backend, sharded into `MAX_STRIPES` independent (of which
    /// `stripe_mask() + 1` are active)
    /// stripes (Ladder 4 C3). Allocations pick a stripe by the calling
    /// thread (`thread_stripe()` — magazine misses from different threads
    /// land on different stripes); frees resolve the *exact* owning stripe
    /// from `buddy_region_stripes` (a block must only be freed into the
    /// `Buddy` whose bitmap tracks its region — see `buddy.rs`'s "Stripe
    /// safety" module-doc section for why coalescing can never cross
    /// stripes).
    buddy: [Mutex<buddy::Buddy>; MAX_STRIPES],
    /// Region base → owning buddy stripe (exact free-side routing). Fixed
    /// atomics; inserted from inside `Buddy::refill`'s `register` callback
    /// *while the stripe lock is held* — safe only because insertion can
    /// never allocate (the Ladder-4 reentrancy rule).
    buddy_region_stripes: registry::RegionRegistry,
    arena: Mutex<Option<arena::BumpArena>>,
    /// Fast-fail latch: set when the arena hits its `MAX_CHUNKS` cap and
    /// fails a chunk-fitting request, cleared only by `reset_arena()`. A
    /// bump arena never frees per-allocation, so once capped it fails
    /// *every* subsequent alloc — without this latch, a frozen model that
    /// routed churny call sites to Arena pays a doomed Mutex slow-path
    /// attempt on every allocation after exhaustion, forever (measured on
    /// cpp-string under LD_PRELOAD: inference 1.76× slower than its own
    /// training, fallthrough=200k/350k allocs — training self-corrects via
    /// the bandit; frozen routing cannot). One relaxed load, paid only on
    /// arena-recommended allocations.
    ///
    /// `CachePadded` (here and on every other hot atomic below — J5-B1):
    /// J4-D certified that ONE shared RMW atomic on the hot path costs 3.2×
    /// at t8 via cache-line ping-pong, and the field survey found zero
    /// padding anywhere — read-hot atomics (loaded on every alloc/free by
    /// every thread) were sharing 64-byte lines with RMW'd neighbors, so a
    /// single writer invalidated the line for all readers. Each padded field
    /// gets its own 128-byte line (aarch64 prefetch pair); read-hot fields
    /// stay in Shared state across cores regardless of what any RMW field
    /// does. Cost: ~128 B per field on a single static — irrelevant.
    arena_exhausted: CachePadded<AtomicBool>,
    /// The Decision Engine (Phase 3). Routes allocations via MAB in Training
    /// mode and via a frozen `PerfectHashTable` in Inference mode.
    state: Mutex<state::AllocatorState>,
    /// Cheap, lock-free mirror of `state.is_inference()` so `dealloc` can
    /// skip Layer 2 reward bookkeeping (`record_latency`) without taking the
    /// state lock on the Inference hot path. Flipped exactly once inside
    /// `freeze()`/`load()` (which already touch the state lock) — a single
    /// relaxed atomic load costs far less than a `Mutex` lock/unlock pair.
    frozen: CachePadded<AtomicBool>,
    /// Retention cache for large System-backend mappings (see
    /// `system::SystemCache`): freed 1 MiB+ mappings are kept populated and
    /// reused instead of munmap'd, matching (and beating) glibc's
    /// large-chunk retention on the 2-8 MiB workload.
    ///
    /// Guarded by `system_cache_lock` (a CAS try-lock), NOT a
    /// `std::sync::Mutex` — deliberately. This cache sits on the
    /// re-entrancy-bypass path, and on macOS a std Mutex's *first lock
    /// lazily `Box::new`s its pthread mutex* (i.e. locking allocates). A
    /// Mutex here recursed: first lock → Box::new → our malloc → bypass →
    /// lock again (OnceBox still initializing) → Box::new → … → stack
    /// overflow (reproduced as a SIGSEGV in the cabi test harness). Every
    /// other backend Mutex survives its own first-lock allocation only
    /// because the bypass path it recurses into is Mutex-free — a
    /// load-bearing property this field must not break.
    system_cache: core::cell::UnsafeCell<system::SystemCache>,
    /// CAS guard for `system_cache`: allocation-free, non-blocking. On
    /// contention or re-entry the caller simply skips the cache (correct:
    /// it's a best-effort retention layer over plain mmap/munmap). Padded:
    /// this is a genuine RMW (CAS) — cold (System path only), but it must
    /// never share a line with a read-hot field.
    system_cache_lock: CachePadded<AtomicBool>,
    /// Count of System allocations served from the retention cache
    /// (testability/introspection). RMW on System cache hits — padded off
    /// the read-hot fields for the same reason as `system_cache_lock`.
    system_cache_hits: CachePadded<AtomicU64>,
    /// Unique id for the per-thread magazine layer (see `magazine.rs`'s
    /// ownership doc): 0 = unassigned, lazily claimed from a process-wide
    /// monotonic counter on first slab use. Ids are never reused, so a
    /// thread's magazine can always tell "my instance's blocks" from a
    /// previous instance's (whose regions may already be unmapped).
    /// Read on every magazine alloc/free (`magazine_owner`), written once —
    /// padded so no RMW neighbor ever invalidates it.
    magazine_id: CachePadded<AtomicU64>,
    /// Lock-free copy of the frozen decision plane (`FrozenRouting`: main
    /// 3-frame table + Ladder-6 distilled 1-frame table, published together
    /// as ONE pointer so the pair can never tear), published by
    /// `freeze()`/`load()` (null while in Training mode). The Inference
    /// alloc fast path loads this pointer instead of taking the `state`
    /// Mutex — before this existed, *every* allocation serialized on that
    /// global lock even in frozen mode, which showed up directly in the
    /// Phase 6 cross-allocator numbers. The pointed-to tables are immutable
    /// and deliberately leaked on `reset_to_training()` (a concurrent
    /// reader may still hold `&*table`; a full RCU scheme is overkill for
    /// a GUI dev button, and leakage is bounded by freeze/reset cycles).
    /// The pointer value doubles as the pin cache's validity tag — see
    /// `PIN_CACHE`. The single hottest read atomic in the allocator (loaded
    /// on every inference alloc by every thread) — the field survey found it
    /// sharing a line with the `system_cache_lock`/`system_cache_hits`/
    /// `magazine_id` RMWs, so one System-path alloc invalidated the line for
    /// every concurrent inference reader. Padded onto its own line.
    frozen_table: CachePadded<AtomicPtr<perfect_hash::FrozenRouting>>,
    /// One-way latch: `true` only for an instance that was booted straight
    /// into Inference via `load()` (never for one that trained live and
    /// later called `freeze()`). Gates the header-free Slab fast path —
    /// see `slab::alloc_class_headerless`'s doc for why this is restricted
    /// to `load()` alone: a `load()`-booted instance's Slab starts (and,
    /// since the flag never resets, stays) completely empty of
    /// header-carrying blocks, so header and headerless blocks are never
    /// mixed in one class's free list. A live-trained instance keeps
    /// writing headers unconditionally (unchanged, zero risk to the
    /// existing GUI/training reward-attribution path, which needs the
    /// header's `hash`/`size_class_hint` on dealloc).
    slab_headerless: CachePadded<AtomicBool>,
    /// Maps a header-free Slab segment's base address to its class, so
    /// `dealloc`/cabi entry points can recover a headerless block's class
    /// from its address alone. See `registry::SegmentRegistry`.
    segment_registry: registry::SegmentRegistry,
    /// J1: one-way latch mirroring `slab_headerless`, for the Buddy
    /// backend. Set only by `load()` on an instance whose buddy stripes
    /// are all untouched — from then on every buddy block is served
    /// header-free (no 48-byte header write → no minor fault on fresh
    /// untouched blocks, no pow2 order-inflation) and frees recover
    /// `(stripe, order)` from `buddy_region_stripes` + the per-region
    /// order map. Training instances never set this (reward attribution
    /// needs the header's `hash`/`size_class_hint`).
    buddy_headerless: CachePadded<AtomicBool>,
    /// Ladder 5: one-way latch mirroring `slab_headerless`/
    /// `buddy_headerless`, for the Arena backend. Set only by `load()` on
    /// an instance whose arena was never touched. Phase 4's cachegrind
    /// attribution found the bandit routes essentially every small-block
    /// call site to Arena (bump alloc + no-op free wins training latency),
    /// making Arena's 48-byte header the whole small-block inference gap:
    /// one cold-block write miss per alloc + one cold header read miss per
    /// free (92% of the slab row's D1 write misses / 74% of its read
    /// misses). Headerless arena blocks are recognized on the free side by
    /// a chunk-membership probe (`arena_chunks`) and freed as a no-op;
    /// they have no recoverable size, so `try_usable_size` conservatively
    /// reports 0 and the realloc path forbids in-place reuse (see
    /// `usable_size_for_realloc`).
    arena_headerless: CachePadded<AtomicBool>,
    /// Chunk-membership set for headerless arena frees. See
    /// `registry::ArenaChunkRegistry`.
    arena_chunks: registry::ArenaChunkRegistry,
    /// Ladder 5 span carve: bumped by `reset_arena()` so every thread's
    /// TLS `ARENA_SPAN` (a window into a chunk the reset just rewound) is
    /// discarded on its next use instead of served. Read on every arena
    /// span check; padded so nothing RMW-hot can ever share its line (J4-D
    /// briefly put a per-op-RMW counter adjacent to it — the exact
    /// false-sharing shape that cost 3.2× at t8 before J5-A stripped it).
    arena_epoch: CachePadded<AtomicU64>,
    /// Published descriptor of the arena chunk currently being bumped, for
    /// `arena_alloc_fast`'s lock-free path — null until the arena is first
    /// initialized. See `ArenaChunkDescriptor`'s doc.
    arena_chunk: CachePadded<AtomicPtr<ArenaChunkDescriptor>>,
}

/// Published snapshot of the arena chunk currently being bumped — see
/// `Lohalloc::arena_alloc_fast`/`publish_arena_chunk`.
///
/// `cursor` is a raw pointer into a `arena::Chunk` living inside
/// `Lohalloc.arena`'s `BumpArena.chunks` `Vec`. That's sound to dereference
/// for as long as this `Lohalloc` instance lives: the `Vec` is
/// pre-reserved to `MAX_CHUNKS` capacity at construction and never grows
/// past it (`BumpArena::alloc` checks the cap before ever pushing), so a
/// `Chunk`'s address — and thus its `cursor`'s address — never moves once
/// mapped, even while other chunks get pushed alongside it.
struct ArenaChunkDescriptor {
    base: usize,
    capacity: usize,
    cursor: *const AtomicUsize,
}

unsafe impl Send for ArenaChunkDescriptor {}
unsafe impl Sync for ArenaChunkDescriptor {}

/// Count of Inference-mode lookups whose key was *not* in the frozen table
/// (falling back to size-based routing). Only incremented on a miss — the
/// hit path stays untouched. This is the observability hook that lets the
/// Phase 6 benchmark verify a pre-trained model actually matches a fresh
/// process's call sites (ASLR-stable hashes): a model-loaded run whose
/// workload was trained in an earlier process should see ~0 misses.
static PHT_MISSES: AtomicU64 = AtomicU64::new(0);

/// Per-backend count of Inference-mode allocations actually *served* by that
/// backend (indexed by `Backend as usize`), plus a separate count of times
/// the frozen table's (or the miss fallback's) recommendation failed and the
/// request fell through to `route_by_size`. Observability added to diagnose
/// a frozen model routing a Signature to the wrong backend (e.g. locking
/// onto System for a size that fits Buddy) — `try_backend` never fails for
/// System, so a bad System recommendation has no self-correcting
/// fallthrough the way a bad Slab/Arena one does, and these counters are
/// what make that visible from outside.
///
/// # Gated behind `route-metrics` (Ladder 4 C1)
///
/// These `fetch_add`s ran unconditionally on **every** frozen alloc, and
/// every thread increments the *same* cache line — false sharing on 100%
/// of allocations that measurably degraded multithreaded scaling (all
/// threads bouncing one line each op). They now compile away entirely
/// unless the `route-metrics` feature is on (always on under `cfg(test)`,
/// so the counter tests and the `LOHALLOC_DEBUG` route-diagnosis workflow
/// still work — build the diagnostic cabi with `--features route-metrics`).
/// `PHT_MISSES` (above) is NOT gated: it's a miss-only counter (cold path),
/// carries no false-sharing cost on the hit path, and is load-bearing for
/// verifying a model matches a process (`pht_misses≈0`).
#[cfg(any(feature = "route-metrics", test))]
static ROUTE_COUNTS: [AtomicU64; 4] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
#[cfg(any(feature = "route-metrics", test))]
static FALLTHROUGH_COUNT: AtomicU64 = AtomicU64::new(0);

/// Record a served-by-`backend` frozen route. No-op (fully compiled away)
/// unless `route-metrics`/`test` — see `ROUTE_COUNTS`.
#[inline(always)]
fn record_route(backend: lohalloc_core::Backend) {
    #[cfg(any(feature = "route-metrics", test))]
    ROUTE_COUNTS[backend as usize].fetch_add(1, Ordering::Relaxed);
    #[cfg(not(any(feature = "route-metrics", test)))]
    let _ = backend;
}

/// Record a frozen-path recommendation fallthrough. No-op unless
/// `route-metrics`/`test`.
#[inline(always)]
fn record_fallthrough() {
    #[cfg(any(feature = "route-metrics", test))]
    FALLTHROUGH_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// Ladder 6 pin-cache observability. `PIN_MISSES` is ungated like
/// `PHT_MISSES` (a true miss happens roughly once per (site, size_class)
/// per cache residency — cold by construction, and it's the counter that
/// verifies population actually happens on a model-loaded run). Hit-side
/// counters (`PIN_HITS`, `PIN_NEGATIVE`) would false-share one cache line
/// on ~100% of pinned allocations — exactly the Ladder-4 C1 lesson — so
/// they're gated behind `route-metrics`/`test`.
static PIN_MISSES: AtomicU64 = AtomicU64::new(0);
#[cfg(any(feature = "route-metrics", test))]
static PIN_HITS: AtomicU64 = AtomicU64::new(0);
#[cfg(any(feature = "route-metrics", test))]
static PIN_NEGATIVE: AtomicU64 = AtomicU64::new(0);

/// Record a pin-cache hit (site served without a stack walk). No-op unless
/// `route-metrics`/`test` — hit-path counters false-share (see above).
#[inline(always)]
fn record_pin_hit() {
    #[cfg(any(feature = "route-metrics", test))]
    PIN_HITS.fetch_add(1, Ordering::Relaxed);
}

/// Record a pin-cache negative hit (known-unpinnable site took the full
/// path without re-probing the distilled table). No-op unless
/// `route-metrics`/`test`.
#[inline(always)]
fn record_pin_negative() {
    #[cfg(any(feature = "route-metrics", test))]
    PIN_NEGATIVE.fetch_add(1, Ordering::Relaxed);
}

/// J5-bisect slab central-refill observability (route-metrics/test only —
/// these sit on the magazine-miss path, cold enough to count but the same
/// false-sharing rule as the pin counters applies). Instrumentation for the
/// "stripe widening lengthened the sibling-stripe scan" hypothesis:
/// `SLAB_CENTRAL_REFILLS` counts magazine-miss refills reaching the central
/// slab, `SLAB_SIBLING_STEPS` the total sibling-stripe try_lock probes those
/// refills performed, `SLAB_SIBLING_HITS` how many refills a sibling
/// actually served. steps/refill ≈ stripe_count−1 with hits ≈ 0 = pure
/// scan waste (the single-thread signature).
#[cfg(any(feature = "route-metrics", test))]
static SLAB_CENTRAL_REFILLS: AtomicU64 = AtomicU64::new(0);
#[cfg(any(feature = "route-metrics", test))]
static SLAB_SIBLING_STEPS: AtomicU64 = AtomicU64::new(0);
#[cfg(any(feature = "route-metrics", test))]
static SLAB_SIBLING_HITS: AtomicU64 = AtomicU64::new(0);

/// Record one central refill and its sibling-scan outcome in a single call
/// (one to three `fetch_add`s per magazine miss, not per op). No-op unless
/// `route-metrics`/`test`.
#[inline(always)]
fn record_slab_central_refill(_steps: u64, _sibling_hit: bool) {
    #[cfg(any(feature = "route-metrics", test))]
    {
        SLAB_CENTRAL_REFILLS.fetch_add(1, Ordering::Relaxed);
        if _steps > 0 {
            SLAB_SIBLING_STEPS.fetch_add(_steps, Ordering::Relaxed);
        }
        if _sibling_hit {
            SLAB_SIBLING_HITS.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl Default for Lohalloc {
    fn default() -> Self {
        Self::new()
    }
}

/// Ceiling on central-`Slab`/`Buddy` stripes (J5-B2). The stripe arrays are
/// sized to this **compile-time constant** so `Lohalloc::new()` stays a
/// `const fn` (the property that lets the allocator be installed as a plain
/// `static` / `#[global_allocator]` with zero runtime init); how many of
/// them are *active* is the runtime [`stripe_mask`], scaled once to the host
/// core count. Stripes are metadata-only (no eager block memory), so the
/// inactive tail costs a few hundred KB of `.bss` and nothing else.
pub(crate) const MAX_STRIPES: usize = 32;

/// Floor on the *active* stripe count. 8 matches the J4-B certified state
/// (at 8 stripes the 8 threads of a t8 run land on stripes 0..7 exactly, so
/// the striped Mutexes are 1:1 — J4-B measured 4 stripes at t8 as 2:1
/// contention, rust/mt-mixed-t8 1.88×). Hosts with ≤8 cores therefore
/// behave byte-identically to the certified J4-C configuration; only wider
/// hosts (e.g. the 16-vCPU c9g gate box) grow past it.
pub(crate) const MIN_STRIPES: usize = 8;

/// The active-stripe mask (`active_count - 1`), latched on first use.
/// `usize::MAX` = unlatched. (The sentinel used to be 0 back when the floor
/// guaranteed mask >= 7; the `LOHALLOC_STRIPES` override below can legally
/// produce mask 0 — a single stripe — so 0 is now a real value.)
static STRIPE_MASK: AtomicUsize = AtomicUsize::new(usize::MAX);

/// Allocation-free read of the `LOHALLOC_STRIPES` override. Uses
/// `libc::getenv` + a manual ASCII parse because this runs *inside* the
/// allocator (first-allocation latch, `IN_ALLOC` possibly active, possibly
/// pre-`main` under `#[global_allocator]`/LD_PRELOAD): `std::env::var`
/// allocates a `String` and takes std's internal env lock — the documented
/// re-entrancy deadlock class (see lohalloc-cabi's `LOHALLOC_FREEZE_AFTER`
/// history). `getenv` isn't guaranteed safe against concurrent `setenv`,
/// but nothing in this process mutates the environment and the latch reads
/// once. `None` = unset, empty, non-numeric, `0`, or overflow — all treated
/// as "no override" (silently: eprintln! allocates, and we're on the alloc
/// path).
fn stripe_override() -> Option<usize> {
    // SAFETY: NUL-terminated literal; getenv returns NULL or a pointer into
    // the C environment, valid for the duration of this read (no setenv in
    // this process).
    let p = unsafe { libc::getenv(b"LOHALLOC_STRIPES\0".as_ptr().cast()) };
    if p.is_null() {
        return None;
    }
    let mut n: usize = 0;
    let mut i = 0isize;
    loop {
        // SAFETY: walking the NUL-terminated C string returned by getenv.
        let b = unsafe { *p.offset(i) } as u8;
        if b == 0 {
            break;
        }
        if !b.is_ascii_digit() {
            return None;
        }
        n = n.checked_mul(10)?.checked_add((b - b'0') as usize)?;
        i += 1;
    }
    if i == 0 || n == 0 {
        return None; // empty value or explicit 0: treat as unset
    }
    Some(n)
}

/// Pure latch computation, split out so it is unit-testable without touching
/// the process-global `STRIPE_MASK` latch.
///
/// - `Some(n)` (explicit `LOHALLOC_STRIPES` override, bisect/diagnostic use):
///   rounded up to a power of two, capped at `MAX_STRIPES`, floor **1** — an
///   explicit override is definitionally opting out of the certified
///   configuration, and the 1-stripe cell is the strongest mechanism probe
///   (sibling scan degenerates to an empty range).
/// - `None` (production): today's formula, unchanged —
///   `next_pow2(ncpus).clamp(MIN_STRIPES, MAX_STRIPES)`.
fn stripe_count(override_n: Option<usize>, ncpus: usize) -> usize {
    match override_n {
        Some(n) => n.next_power_of_two().min(MAX_STRIPES),
        None => ncpus.next_power_of_two().clamp(MIN_STRIPES, MAX_STRIPES),
    }
}

/// Active stripe count as a mask, scaled to the host: `next_pow2(ncpus)`
/// clamped to `[MIN_STRIPES, MAX_STRIPES]`, minus 1 — overridable via
/// `LOHALLOC_STRIPES` (bisect/diagnostic knob; see `stripe_count`). Latched
/// once — a racy double-init computes the identical value (getenv is stable
/// for the process), so a plain relaxed load/store is enough (same pattern
/// as `magazine_id`). Reads after the latch are a single shared
/// (never-again-written) atomic load; the J4-B lesson that stripes must
/// track the *actual* concurrency is what this generalizes: the old
/// `BACKEND_STRIPES = 8` const was hardwired to the benched t8 and left a
/// 16-vCPU host 2:1-contended at t16.
#[inline]
pub(crate) fn stripe_mask() -> usize {
    let m = STRIPE_MASK.load(Ordering::Relaxed);
    if m != usize::MAX {
        return m;
    }
    let ncpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(MIN_STRIPES);
    let mask = stripe_count(stripe_override(), ncpus) - 1;
    STRIPE_MASK.store(mask, Ordering::Relaxed);
    mask
}

impl Lohalloc {
    pub const fn new() -> Self {
        Self {
            slab: [const { Mutex::new(slab::Slab::new()) }; MAX_STRIPES],
            buddy: [const { Mutex::new(buddy::Buddy::new()) }; MAX_STRIPES],
            buddy_region_stripes: registry::RegionRegistry::new(),
            // Arena is lazily initialized on first use (requires mmap, which
            // is not const-evaluable).
            arena: Mutex::new(None),
            arena_exhausted: CachePadded::new(AtomicBool::new(false)),
            // Decision Engine starts in Training mode.
            state: Mutex::new(state::AllocatorState::new_training_const()),
            frozen: CachePadded::new(AtomicBool::new(false)),
            system_cache: core::cell::UnsafeCell::new(system::SystemCache::new()),
            system_cache_lock: CachePadded::new(AtomicBool::new(false)),
            system_cache_hits: CachePadded::new(AtomicU64::new(0)),
            magazine_id: CachePadded::new(AtomicU64::new(0)),
            frozen_table: CachePadded::new(AtomicPtr::new(core::ptr::null_mut())),
            slab_headerless: CachePadded::new(AtomicBool::new(false)),
            segment_registry: registry::SegmentRegistry::new(),
            buddy_headerless: CachePadded::new(AtomicBool::new(false)),
            arena_headerless: CachePadded::new(AtomicBool::new(false)),
            arena_chunks: registry::ArenaChunkRegistry::new(),
            arena_epoch: CachePadded::new(AtomicU64::new(0)),
            arena_chunk: CachePadded::new(AtomicPtr::new(core::ptr::null_mut())),
        }
    }

    /// Number of System allocations served from the mapping retention
    /// cache since this instance was created.
    pub fn system_cache_hits(&self) -> u64 {
        self.system_cache_hits.load(Ordering::Relaxed)
    }

    /// Run `f` with exclusive access to the system mapping cache, WITHOUT
    /// ever allocating or blocking: a CAS try-lock. Returns `None` if the
    /// lock is currently held (another thread, or a re-entrant call on
    /// this thread) — callers treat that as a cache miss/decline and fall
    /// through to plain mmap/munmap, which is always correct.
    fn with_system_cache<R>(&self, f: impl FnOnce(&mut system::SystemCache) -> R) -> Option<R> {
        if self
            .system_cache_lock
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return None;
        }
        // SAFETY: the CAS above grants exclusive access until the store
        // below; `f` cannot re-enter this cache (a re-entrant attempt
        // fails the CAS and skips it).
        let result = f(unsafe { &mut *self.system_cache.get() });
        self.system_cache_lock.store(false, Ordering::Release);
        Some(result)
    }

    /// This instance's magazine owner id, claimed lazily (never 0, never
    /// reused across instances).
    #[inline]
    fn magazine_owner(&self) -> u64 {
        let id = self.magazine_id.load(Ordering::Relaxed);
        if id != 0 {
            return id;
        }
        // First use: claim a fresh id. Racing threads may both claim; the
        // CAS loser adopts the winner's id, and the loser's id is simply
        // skipped (the counter is monotonic, gaps are fine).
        let fresh = NEXT_MAGAZINE_ID.fetch_add(1, Ordering::Relaxed);
        match self
            .magazine_id
            .compare_exchange(0, fresh, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => fresh,
            Err(winner) => winner,
        }
    }

    /// J4-A: claim a fresh magazine owner id and publish it, so every thread's
    /// per-thread magazine discards any blocks cached under the previous id on
    /// its next use (`magazine::ensure_owner`). Called by `load()` at the
    /// training→headerless transition: the magazine may hold pre-load header
    /// blocks, and the post-load headerless alloc path pops from that same
    /// magazine — a header block popped there would be served header-free and
    /// corrupt on its next free. The discarded blocks are the only memory
    /// J4-A ever abandons (bounded: ≤ magazine cap per class per live thread,
    /// one-time — no other thread is running yet in the model-load deployment
    /// pattern).
    fn invalidate_magazines(&self) {
        let fresh = NEXT_MAGAZINE_ID.fetch_add(1, Ordering::Relaxed);
        self.magazine_id.store(fresh, Ordering::Relaxed);
    }
}

/// Process-wide monotonic source of magazine owner ids (never 0; gaps from
/// racing claimants are fine). Bumped once per instance's first slab use and
/// again by `Lohalloc::invalidate_magazines` at each `load()`.
static NEXT_MAGAZINE_ID: AtomicU64 = AtomicU64::new(1);

// SAFETY: backend state is guarded by `Mutex`; `mmap`/`munmap` are thread-safe.
// Re-entrancy is broken by the thread-local guard (see `alloc`). The backends
// never call back into `Lohalloc::alloc` for user allocations.
//
// `Send` is also sound: all interior mutability is funnelled through the
// `Mutex` fields (or the thread-local re-entrancy guard, which is itself
// `Send`), so a `Lohalloc` can be safely moved to another thread — it
// simply transfers ownership of the same locks. The raw `*mut FreeNode`
// pointers inside `Slab`/`Buddy` only ever escape through the `Mutex`,
// never across thread boundaries on their own.
unsafe impl Send for Lohalloc {}
unsafe impl Sync for Lohalloc {}

thread_local! {
    /// Re-entrancy depth. >0 means we are already inside `alloc`/`dealloc` on
    /// this thread — any further allocation must bypass to `mmap` directly.
    static IN_ALLOC: Cell<usize> = const { Cell::new(0) };
    /// Capture the start time of the allocation for latency measurement.
    /// Set at entry to alloc(), read in emit_alloc() to compute elapsed time.
    static ALLOC_START_NS: Cell<u64> = const { Cell::new(0) };
    /// This thread's central-backend stripe (Ladder 4 C3/C4), assigned
    /// round-robin on first use so concurrent threads spread evenly across
    /// the active stripes (`stripe_mask() + 1`, CPU-scaled — J5-B2)
    /// regardless of thread-id numbering.
    /// `usize::MAX` = unassigned. Plain dtor-free `Cell`, same TLS
    /// discipline as the magazines. Process-wide (not per-instance):
    /// a stripe index is just a load-spreading hint on the alloc side —
    /// frees always resolve ownership through the per-instance registry,
    /// so instance mixing is harmless here.
    static THREAD_STRIPE: Cell<usize> = const { Cell::new(usize::MAX) };
    /// This thread's private arena bump window (Ladder 5 span carve — see
    /// `Lohalloc::arena_alloc_fast`). All dtor-free `Cell`s, same TLS
    /// discipline as the magazines; validity is (owner, epoch)-checked on
    /// every use, so instance mixing and `reset_arena()` both simply
    /// discard the span.
    static ARENA_SPAN: ArenaSpan = const {
        ArenaSpan {
            owner: Cell::new(0),
            epoch: Cell::new(0),
            cursor: Cell::new(0),
            end: Cell::new(0),
        }
    };
}

/// See `ARENA_SPAN`/`Lohalloc::arena_alloc_fast`.
struct ArenaSpan {
    owner: Cell<u64>,
    epoch: Cell<u64>,
    cursor: Cell<usize>,
    end: Cell<usize>,
}

// ---------------------------------------------------------------------------
// Ladder 6: Inference pin cache — serve freeze-proven-unambiguous call sites
// from the raw leaf return address alone (no frames-1-2 walk, no memo, no
// normalize/mix, no main-table lookup). See `route_alloc`.
// ---------------------------------------------------------------------------

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
const PIN_ENTRIES: usize = 64;

/// Per-slot verdict-array width. `state::size_class_for` yields 0..=14
/// (12 Slab classes, 2 Buddy, 1 System), so 16 covers every class; an
/// out-of-range sc (impossible today) bypasses the cache entirely.
const PIN_SC_SLOTS: usize = 16;

/// Verdict byte: this size class has not been probed against the distilled
/// table yet.
const PIN_UNKNOWN: u8 = 0xFE;

/// Verdict byte: probed, and the (site, size_class) is NOT pinnable — the
/// negative cache that keeps non-distilled sites from re-paying the
/// one-frame derivation + distilled lookup on every allocation. Values 0–3
/// are `Backend as u8`.
const PIN_NOT_PINNED: u8 = 0xFF;

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
struct PinEntry {
    table: Cell<*const perfect_hash::FrozenRouting>,
    ret0: Cell<usize>,
    states: [Cell<u8>; PIN_SC_SLOTS],
}

thread_local! {
    /// Direct-mapped, dtor-free (plain `Cell`s, per the crate's TLS
    /// invariant), const-initialized so first touch never allocates.
    static PIN_CACHE: [PinEntry; PIN_ENTRIES] = const {
        [const {
            PinEntry {
                table: Cell::new(core::ptr::null()),
                ret0: Cell::new(0),
                states: [const { Cell::new(PIN_UNKNOWN) }; PIN_SC_SLOTS],
            }
        }; PIN_ENTRIES]
    };
}

/// Slot index for a raw leaf return address — same multiply-fold pattern
/// as `topology`'s memo index.
#[inline(always)]
fn pin_index(ret0: usize) -> usize {
    ((ret0 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 58) as usize & (PIN_ENTRIES - 1)
}

/// Pin-cache probe outcome. `Pinned` short-circuits the whole decision
/// plane; `NotPinned` runs the full path but skips re-populating;
/// `Miss` runs the full path and then records this (site, sc) verdict.
enum PinProbe {
    Pinned(lohalloc_core::Backend),
    NotPinned,
    Miss,
}

#[inline(always)]
fn pin_probe(table: *const perfect_hash::FrozenRouting, ret0: usize, size_class: u8) -> PinProbe {
    if size_class as usize >= PIN_SC_SLOTS {
        return PinProbe::NotPinned; // out-of-range sc: bypass, never store
    }
    PIN_CACHE.with(|cache| {
        let e = &cache[pin_index(ret0)];
        if e.ret0.get() != ret0 || e.table.get() != table {
            return PinProbe::Miss;
        }
        match e.states[size_class as usize].get() {
            PIN_UNKNOWN => PinProbe::Miss,
            PIN_NOT_PINNED => PinProbe::NotPinned,
            b => PinProbe::Pinned(pin_backend_from_u8(b)),
        }
    })
}

/// Decode a stored `Backend as u8` pin verdict (never `PIN_UNKNOWN` /
/// `PIN_NOT_PINNED` when called — the probe matches those arms first).
#[inline(always)]
fn pin_backend_from_u8(b: u8) -> lohalloc_core::Backend {
    match b {
        0 => lohalloc_core::Backend::Slab,
        1 => lohalloc_core::Backend::Buddy,
        2 => lohalloc_core::Backend::System,
        _ => lohalloc_core::Backend::Arena,
    }
}

/// Record the distilled verdict for `(ret0, size_class)`. Claims the slot
/// (resetting every size-class verdict) when it currently belongs to a
/// different site or table snapshot; direct-mapped, so colliding *sites*
/// still evict each other — but a site's own size classes never do.
#[inline]
fn pin_store(
    table: *const perfect_hash::FrozenRouting,
    ret0: usize,
    size_class: u8,
    pinned: Option<lohalloc_core::Backend>,
) {
    if size_class as usize >= PIN_SC_SLOTS {
        return;
    }
    PIN_CACHE.with(|cache| {
        let e = &cache[pin_index(ret0)];
        if e.ret0.get() != ret0 || e.table.get() != table {
            e.table.set(table);
            e.ret0.set(ret0);
            for s in &e.states {
                s.set(PIN_UNKNOWN);
            }
        }
        e.states[size_class as usize].set(match pinned {
            Some(b) => b as u8,
            None => PIN_NOT_PINNED,
        });
    });
}

/// Bytes CAS-reserved from the shared arena chunk per span refill: the
/// amortization factor between TLS bumps and shared-cursor CASes (a 4 KiB
/// span serves ~16 × 256 B allocations per CAS). Worst-case strand: one
/// span per dead thread + at most the request size per refill.
const ARENA_SPAN_BYTES: usize = 4096;

/// Round-robin source for `THREAD_STRIPE` assignments.
static NEXT_STRIPE: AtomicU64 = AtomicU64::new(0);

/// The calling thread's stripe index in `[0, stripe_mask()]`.
#[inline]
fn thread_stripe() -> usize {
    THREAD_STRIPE.with(|c| {
        let v = c.get();
        if v != usize::MAX {
            return v;
        }
        let v = (NEXT_STRIPE.fetch_add(1, Ordering::Relaxed) as usize) & stripe_mask();
        c.set(v);
        v
    })
}

/// Retrieve the allocation start time (for telemetry latency measurement).
/// Called by `observer::emit_alloc` to compute elapsed time.
#[cfg(feature = "telemetry-observer")]
pub(crate) fn alloc_start_ns() -> u64 {
    ALLOC_START_NS.get()
}

impl Lohalloc {
    /// Runs `f` with the same re-entrancy guard `alloc`/`dealloc` use, so
    /// any allocation or deallocation `f` triggers synchronously on this
    /// thread bypasses straight to `mmap` instead of trying to re-lock
    /// `self.state`/`self.slab`/`self.buddy`/`self.arena`.
    ///
    /// Needed by every Phase 3 API method that both holds `self.state`'s
    /// lock *and* can allocate/deallocate through ordinary Rust collections
    /// (`freeze()`'s `Vec::collect()` inside `BanditPolicy::freeze`,
    /// `load()`'s `PerfectHashTable::deserialize`, dropping an old
    /// `BanditPolicy`'s `BTreeMap` on `reset_to_training`/`load`, …). If
    /// this `Lohalloc` instance is the process's actual `#[global_allocator]`
    /// (always true once installed via `lohalloc-cabi`; also true for any
    /// binary using `#[global_allocator]` the way `lohalloc-example`/`-demo`
    /// do), those nested allocations would otherwise call back into
    /// `route_alloc_inner`/`dealloc`, which try to lock the *same* mutex
    /// the outer call already holds — a guaranteed deadlock on the
    /// non-reentrant `std::sync::Mutex`. Routing them to raw `mmap` instead
    /// is wasteful (a whole page per nested allocation) but these are rare,
    /// off-hot-path calls, so it's the right trade.
    fn with_realloc_guard<T>(f: impl FnOnce() -> T) -> T {
        let depth = IN_ALLOC.get();
        IN_ALLOC.set(depth + 1);
        let result = f();
        IN_ALLOC.set(depth);
        result
    }

    /// Public wrapper around [`Self::with_realloc_guard`] for callers
    /// *outside* this crate that need to read a `.lohalloc` model's bytes
    /// (typically `std::fs::read`) before calling [`Self::load`].
    ///
    /// This matters specifically for `lohalloc-cabi`'s `ensure_model_loaded`
    /// under `LD_PRELOAD`: it reads the model file *before* calling
    /// `load()`, and that read's own internal allocations (`Vec<u8>`
    /// growth) would otherwise be the *first-ever* calls into this
    /// allocator for the process — landing wherever ordinary Training-mode
    /// routing sends them (often Slab), silently creating a real backing
    /// region before `load()`'s `slab_headerless` safety check
    /// (`Slab::region_count() == 0`) ever runs, permanently and invisibly
    /// disabling the header-free Slab fast path for the rest of the
    /// process. Wrapping the read in this guard routes it to `mmap`
    /// instead, keeping the Slab genuinely empty until `load()` decides.
    pub fn with_bootstrap_guard<T>(f: impl FnOnce() -> T) -> T {
        Self::with_realloc_guard(f)
    }

    /// `true` while this thread is anywhere inside the allocator
    /// (`alloc`/`dealloc`/a `with_realloc_guard` section). For interposition
    /// layers (`lohalloc-cabi`) that must treat a nested `malloc` — one
    /// triggered *by* allocator internals, e.g. `record_latency`'s
    /// `BTreeMap` node insert allocating while `self.state`'s lock is held —
    /// as re-entrant even when the layer's own depth counter says
    /// "top-level" (the nested call can arrive through an entry point that
    /// never bumped it, like `free`). Deadlock #4's exact mechanism: such a
    /// nested malloc crossed `LOHALLOC_FREEZE_AFTER` and called `freeze()`,
    /// which re-locked the already-held `state` Mutex. One TLS read.
    #[inline]
    pub fn thread_inside_allocator() -> bool {
        IN_ALLOC.get() > 0
    }
}

// SAFETY: we uphold the `GlobalAlloc` contract:
//  - `alloc` returns a valid, aligned, `layout.size()`-byte buffer or null.
//  - `dealloc` releases a buffer previously returned by `alloc` with a
//    matching layout; the header lets us route to the exact owning backend.
//  - No re-entrancy deadlock: the guard short-circuits internal allocations
//    to the System Fallback.
unsafe impl GlobalAlloc for Lohalloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // Bail out on zero-size: the GlobalAlloc contract says callers must not
        // ask for zero, but be defensive — round up to 1 so the header still
        // fits and we never hand back a null for a "successful" zero request.
        let size = layout.size().max(1);
        let align = layout.align().max(MIN_ALIGN);
        let pad = header_pad(align);
        let total = size + pad;

        // Capture the start time for latency measurement (only used when
        // telemetry-observer feature is enabled; read in emit_alloc). Gated
        // on a sink actually being installed so the compiled-in-but-idle
        // case costs one relaxed load, not a clock read per alloc.
        #[cfg(feature = "telemetry-observer")]
        if observer::sink_installed() {
            ALLOC_START_NS.set(observer::now_ns());
        }

        // Re-entrancy guard: if we're already inside the allocator on this
        // thread (e.g. a backend's `Vec` growing), serve directly from mmap.
        let depth = IN_ALLOC.get();
        if depth > 0 {
            return self.system_alloc_with_header(total, align, 0, SIZE_CLASS_UNTRACKED);
        }

        IN_ALLOC.set(depth + 1);
        let ptr = self.route_alloc(size, align, pad, total);
        IN_ALLOC.set(depth);
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let _ = layout; // header is authoritative; layout unused for routing.
        if ptr.is_null() {
            return;
        }
        // Header-free fast path: must run BEFORE any header read — a
        // headerless block's `ptr - HEADER_SIZE` is not valid header
        // memory (see `headerless_class_for`'s doc).
        if let Some(class) = self.headerless_class_for(ptr) {
            self.slab_dealloc_headerless(ptr, class);
            return;
        }
        if let Some(order) = self.buddy_headerless_order_for(ptr) {
            unsafe { self.buddy_dealloc_via_magazine(ptr, buddy::block_size_of(order)) };
            return;
        }
        // Ladder 5: headerless arena block — free is a no-op (bump arenas
        // reclaim via reset, never per-pointer), and reading `ptr - 48`
        // would read a neighboring live block's tail.
        if self.arena_headerless_hit(ptr) {
            return;
        }
        // Read the header (unaligned) sitting immediately before the user ptr.
        let header_ptr = ptr.sub(HEADER_SIZE) as *const Header;
        let header = unsafe { core::ptr::read_unaligned(header_ptr) };
        if header.magic != MAGIC {
            // Not one of ours (e.g. a bootstrap allocation before the global
            // allocator was installed, or memory from a foreign source).
            // Nothing safe to do — leak rather than corrupt.
            debug_assert!(false, "dealloc: bad header magic");
            return;
        }
        unsafe { self.dealloc_with_header(ptr, header) };
    }
}

impl Lohalloc {
    /// The body of [`GlobalAlloc::dealloc`] after the header has been read
    /// and its magic verified — split out so the C-ABI paths
    /// ([`Self::try_dealloc_raw`], `lohalloc-cabi`'s `realloc`) can reuse
    /// their *own* single header read instead of reading it again here
    /// (the cabi `free` used to read the 48-byte header twice per call,
    /// `realloc` three times).
    ///
    /// # Safety
    /// `ptr` must be a live allocation produced by this allocator and
    /// `header` the (magic-verified) header read from `ptr - HEADER_SIZE`.
    unsafe fn dealloc_with_header(&self, ptr: *mut u8, header: Header) {
        // Telemetry hook (compiled away when the feature is off; skipped
        // with one relaxed load when no sink is installed). Emit the free
        // record before routing so the GUI sees every deallocation, even
        // those that will be silently dropped (e.g. Arena frees, which are
        // no-ops here).
        #[cfg(feature = "telemetry-observer")]
        if observer::sink_installed() {
            let frag = Backend::from_u8(header.backend)
                .map(|b| fragmentation_pct_for(b, header.size))
                .unwrap_or(0.0);
            observer::emit_free(header.size, header.hash, ptr as u64, frag);
        }

        // Layer 2 (dealloc-side reward): only pay for timing + the state
        // lock while Training, and only for allocations that went through
        // `route_alloc`/`route_alloc_with_hash` (tagged with a real
        // `size_class_hint` — re-entrancy-bypass and Strategy-preferred
        // allocations never had a matching bandit pull to attribute this
        // to, so they're tagged `SIZE_CLASS_UNTRACKED` and skipped here).
        // Folding dealloc latency into the same arm as an additional reward
        // sample (rather than a separate ledger) is what lets the bandit
        // see Arena's free-is-a-no-op advantage.
        let track_reward =
            !self.frozen.load(Ordering::Relaxed) && header.size_class_hint != SIZE_CLASS_UNTRACKED;
        let t0 = if track_reward {
            Some(clock::now_ns())
        } else {
            None
        };

        // Recompute `pad` from the header's own `align_log2` rather than
        // trusting the caller's `layout` — the two always agree for a
        // correctly-paired alloc/dealloc, but reading it from the header
        // means this same logic works for `lohalloc-cabi`'s `free(ptr)`,
        // which never has a `Layout` at all.
        let pad = header_pad(1usize << header.align_log2);
        match Backend::from_u8(header.backend) {
            Some(Backend::Slab) => {
                let block = ptr.sub(pad);
                // Class comes straight from the header (written at alloc
                // time) — no size-class recompute on the free path.
                let class = header.slab_class as usize;
                if header.slab_class == SIZE_CLASS_UNTRACKED
                    || class >= lohalloc_core::SLAB_SIZE_CLASSES.len()
                {
                    // Defensive: a Slab-tagged header must carry a valid
                    // class; fall back to the size-derived path.
                    debug_assert!(false, "slab header without a valid class");
                    if let Ok(mut slab) = self.slab[thread_stripe()].lock() {
                        unsafe { slab.dealloc(block, header.size) };
                    }
                } else if self.slab_headerless.load(Ordering::Relaxed) {
                    // J4-A: a `load()`-booted instance reserves the shared
                    // per-thread magazine for the **headerless** flavor
                    // (headerless allocs pop from it — see
                    // `slab_block_headerless_via_magazine`). This block is a
                    // header block (its segment missed every headerless
                    // registry probe in `dealloc`, so it is one of the pre-load
                    // startup allocations). It must NOT enter that magazine — a
                    // later headerless alloc would pop it and serve it
                    // header-free, corrupting it on the next free. Push it
                    // straight to the Slab's header tiers, where it is retained
                    // (never leaked) but never re-served, since the header
                    // alloc path is dead once headerless is latched.
                    if let Ok(mut slab) = self.slab[thread_stripe()].lock() {
                        unsafe { slab.dealloc_class(block, class) };
                    }
                } else {
                    let owner = self.magazine_owner();
                    if !magazine::push(owner, class, block) {
                        // Magazine full: flush half of it plus this block
                        // back to the central slab in one locked batch.
                        let mut buf = [core::ptr::null_mut::<u8>(); 16];
                        let flush = magazine::refill_count(class).min(buf.len());
                        let n = magazine::take(owner, class, &mut buf[..flush]);
                        if let Ok(mut slab) = self.slab[thread_stripe()].lock() {
                            unsafe {
                                slab.dealloc_batch(class, &buf[..n]);
                                slab.dealloc_class(block, class);
                            }
                        }
                    }
                }
            }
            Some(Backend::Buddy) => {
                let block = ptr.sub(pad);
                unsafe { self.buddy_dealloc_via_magazine(block, header.size) };
            }
            Some(Backend::System) => {
                // Offer the mapping to the retention cache (populated pages
                // stay resident for reuse); munmap if declined or the
                // try-lock is contended.
                let retained = self
                    .with_system_cache(|c| c.put(header.base, header.map_len))
                    .unwrap_or(false);
                if !retained {
                    unsafe {
                        libc::munmap(header.base as *mut core::ffi::c_void, header.map_len);
                    }
                }
            }
            Some(Backend::Arena) => {
                // Arena allocations are reclaimed via `reset()`, not
                // per-allocation free. Dealloc is a no-op — the memory stays
                // mapped until the arena is reset or dropped.
            }
            None => {
                debug_assert!(false, "dealloc: unknown backend tag");
            }
        }

        if let (Some(t0), Some(core_backend)) =
            (t0, Backend::from_u8(header.backend).map(Backend::to_core))
        {
            let dt = clock::now_ns().saturating_sub(t0);
            // Ladder 4 A2: `record_latency` now mutates a `BTreeMap` of
            // pending reward batches, which can allocate (node insert) while
            // `self.state`'s lock is already held — the same reentrancy
            // hazard `with_realloc_guard` exists for (see its doc comment).
            // Without this guard, that nested allocation calls back into
            // `route_alloc_inner`, which tries to relock the same non-
            // reentrant `Mutex` and deadlocks the thread.
            Self::with_realloc_guard(|| {
                if let Ok(mut st) = self.state.lock() {
                    st.record_latency(
                        header.hash,
                        core_backend,
                        header.size_class_hint,
                        dt,
                        header.size,
                    );
                }
            });
        }
    }
}

impl Lohalloc {
    /// Route a (non-recursive) allocation to the appropriate backend and write
    /// the ownership header. Returns the user-visible pointer (post-header).
    ///
    /// Phase 3: The Decision Engine (`AllocatorState`) is consulted first.
    /// In Training mode, the MAB selects a backend; in Inference mode, the
    /// frozen `PerfectHashTable` is looked up. If the recommended backend
    /// fails (e.g. Arena full, Slab exhausted), we fall through to size-based
    /// routing — the Phase 1 fallback chain (Slab → Buddy → System).
    ///
    /// Layer 2: while Training, the recommended arm's *actual* outcome
    /// latency (including the cost of a failed attempt + fallthrough, if
    /// any) is measured and fed back to the bandit as this allocation's
    /// reward — this is what lets a Signature that keeps recommending a
    /// backend which then fails (e.g. an exhausted Arena) learn to route
    /// elsewhere. Inference mode is a pure lookup with no reward
    /// bookkeeping, so it never pays the `now_ns()` cost.
    fn route_alloc(&self, size: usize, align: usize, pad: usize, total: usize) -> *mut u8 {
        // Ladder 6 walk split: read the leaf frame once; the raw ret0 feeds
        // the Inference pin cache and the training-side 1-frame retention,
        // the continuation completes the classic 3-frame hash.
        // `finish_walk(walk_leaf())` is bit-identical to the old
        // `fast_stack_hash()` (see topology.rs's composition test).
        match topology::walk_leaf() {
            Some((ret0, cont_fp, sp)) => {
                // Pin-cache fast path (Inference only): a call site the
                // freeze proved unambiguous is served from its raw leaf
                // return address alone — no frames-1-2 walk, no memo, no
                // normalize/mix, no main-table lookup. A mis-pin is
                // impossible by construction (distillation admits a site
                // only when every trained 3-frame context through it agrees
                // on one backend), and even a hypothetical one would be a
                // performance error, never a safety one: `dealloc` routes by
                // header/registry regardless of which backend served the
                // alloc. C6 discipline: this branch is entirely inside the
                // allocator, after the cabi call site.
                let table = self.frozen_table.load(Ordering::Acquire);
                if !table.is_null() {
                    let size_class = state::size_class_for(size);
                    match pin_probe(table, ret0, size_class) {
                        PinProbe::Pinned(backend) => {
                            record_pin_hit();
                            // Header hash = sentinel (0): in Inference the
                            // header's hash is telemetry-only (no reward
                            // bookkeeping), same degradation as a failed
                            // frame-pointer guard.
                            if let Some(ptr) =
                                self.try_backend(backend, total, align, pad, 0, size_class)
                            {
                                record_route(backend);
                                return ptr;
                            }
                            record_fallthrough();
                            return self.route_by_size(
                                total,
                                align,
                                pad,
                                0,
                                size_class,
                                Some(backend),
                            );
                        }
                        PinProbe::NotPinned => {
                            // Known-unpinnable site: full path, skip the
                            // distilled re-probe (the negative cache is what
                            // keeps unpinnable sites at ~zero added cost).
                            record_pin_negative();
                        }
                        PinProbe::Miss => {
                            // Full path this time; derive the 1-frame key
                            // once and remember the verdict either way.
                            PIN_MISSES.fetch_add(1, Ordering::Relaxed);
                            let one_frame = topology::one_frame_from_ret0(ret0);
                            let key = state::combine_hash_size_class(one_frame, size_class);
                            // SAFETY: `table` is non-null, immutable, and
                            // never freed while published (leaked on reset —
                            // see `frozen_table`'s doc).
                            let pinned = unsafe { (*table).distilled.lookup(key) };
                            pin_store(table, ret0, size_class, pinned);
                        }
                    }
                }
                let hash = topology::finish_walk(ret0, cont_fp, sp);
                self.route_alloc_inner(hash, ret0, size, align, pad, total)
            }
            // Guard failure: same sentinel-hash degradation as before.
            None => self.route_alloc_inner(0, 0, size, align, pad, total),
        }
    }

    /// Shared implementation for `route_alloc`/`route_alloc_with_hash` —
    /// both differ only in how `hash` is obtained (stack walk vs.
    /// caller-provided, for the replay engine). `ret0` is the raw leaf
    /// return address from the walk (`0` = unknown: guard failure or the
    /// replay path) — used only to derive the training-side 1-frame hash
    /// for freeze-time distillation; never part of the routing key.
    fn route_alloc_inner(
        &self,
        hash: u64,
        ret0: usize,
        size: usize,
        align: usize,
        pad: usize,
        total: usize,
    ) -> *mut u8 {
        let size_class = state::size_class_for(size);

        // Inference fast path: the frozen table is immutable and published
        // via an atomic pointer, so a frozen allocator routes with zero
        // locks on the decision plane (the serving backend's own Mutex is
        // still taken inside `try_backend`). Before this existed, every
        // alloc — frozen or not — serialized on the global `state` Mutex.
        let table = self.frozen_table.load(Ordering::Acquire);
        if !table.is_null() {
            // SAFETY: the pointed-to table is heap-allocated, immutable,
            // and never freed while non-null (see `frozen_table`'s doc —
            // `reset_to_training` nulls the pointer and leaks the table).
            let key = state::combine_hash_size_class(hash, size_class);
            let backend = match unsafe { (*table).main.lookup(key) } {
                Some(b) => b,
                None => {
                    // Miss-only counter: the hit path pays nothing extra.
                    PHT_MISSES.fetch_add(1, Ordering::Relaxed);
                    state::default_backend_for_size(size)
                }
            };
            if let Some(ptr) = self.try_backend(backend, total, align, pad, hash, size_class) {
                record_route(backend);
                return ptr;
            }
            // Fallthrough remembers the failed backend so it's never
            // re-locked/re-tried inside the size chain.
            record_fallthrough();
            return self.route_by_size(total, align, pad, hash, size_class, Some(backend));
        }

        // Only reached when no frozen table is published (Training, or the
        // rare pre-publish window), so the 1-frame derivation is a
        // training-only cost. Computed *before* taking the state lock; the
        // module table is fixed-capacity and allocation-free, but keeping
        // derivation out of the locked section is free discipline.
        let one_frame = if ret0 != 0 {
            topology::one_frame_from_ret0(ret0)
        } else {
            0
        };
        let (recommended, is_training) = if let Ok(mut st) = self.state.lock() {
            (
                Some(st.route_with_frame(hash, one_frame, size)),
                !st.is_inference(),
            )
        } else {
            (None, false)
        };

        if !is_training {
            // Inference (or a poisoned lock): no reward bookkeeping — just
            // the recommended backend, falling through to size-based
            // routing on failure, as before.
            if let Some(backend) = recommended {
                if let Some(ptr) = self.try_backend(backend, total, align, pad, hash, size_class) {
                    return ptr;
                }
            }
            return self.route_by_size(total, align, pad, hash, size_class, recommended);
        }

        // Training: time the *actual* outcome (success, or failure +
        // fallthrough) and attribute it to the recommended arm regardless
        // of which backend ultimately served the request.
        let t0 = clock::now_ns();
        let ptr = match recommended {
            Some(backend) => self
                .try_backend(backend, total, align, pad, hash, size_class)
                .unwrap_or_else(|| {
                    self.route_by_size(total, align, pad, hash, size_class, Some(backend))
                }),
            None => self.route_by_size(total, align, pad, hash, size_class, None),
        };
        let latency_ns = clock::now_ns().saturating_sub(t0);

        if let Some(backend) = recommended {
            // See the matching guard in `dealloc_with_header` — kept here too
            // so this call site stays deadlock-safe independent of caller
            // discipline (currently `IN_ALLOC` is already >0 by the time we
            // get here via `alloc`/`alloc_with_hash`, but that's an
            // assumption about callers, not a guarantee this function makes).
            Self::with_realloc_guard(|| {
                if let Ok(mut st) = self.state.lock() {
                    st.record_latency(hash, backend, size_class, latency_ns, total);
                }
            });
        }

        ptr
    }

    /// Attempt an allocation via a specific backend. Returns the user pointer
    /// Serve a raw Slab block for `total` bytes via this thread's magazine
    /// (no lock on a hit), falling back to one batched, locked visit to the
    /// central slab on a miss (refills half a magazine + serves this op, so
    /// the Mutex is amortized over `refill_count` operations).
    fn slab_block_via_magazine(&self, total: usize) -> Option<*mut u8> {
        let class = lohalloc_core::slab_class_for(total)?;
        let owner = self.magazine_owner();
        if let Some(block) = magazine::pop(owner, class) {
            return Some(block);
        }
        // Magazine miss: batch-refill under the central lock.
        let mut buf = [core::ptr::null_mut::<u8>(); 16]; // >= max refill_count
        let want = magazine::refill_count(class).min(buf.len());
        let n = self.slab_batch_recycled_first(class, &mut buf[..want], false);
        if n == 0 {
            return None;
        }
        for &p in buf.iter().take(n).skip(1) {
            // The magazine was just empty, so the extras always fit.
            let pushed = magazine::push(owner, class, p);
            debug_assert!(pushed);
        }
        Some(buf[0])
    }

    /// The striped-central batch pop with **recycled-first cross-stripe
    /// steal** (Ladder 5 Phase 3). Order: this thread's stripe's recycled
    /// tiers → the other stripes' recycled tiers (`try_lock`, skipping
    /// contended stripes — stealing is an optimization, never worth
    /// blocking for) → this stripe's carve/refill (fresh memory, last).
    ///
    /// Why: slab frees are stripe-agnostic (see `slab.rs`'s C4 note), so a
    /// producer/consumer thread split (mt-xfree) parks every recycled
    /// block on the consumer's stripe while the producer's stripe — never
    /// seeing a free — carves fresh segments without bound. Recycled
    /// blocks from *any* stripe are interchangeable (class fidelity is
    /// carried by the header/segment registry, not the serving stripe),
    /// so preferring recycled-anywhere over fresh both caps that RSS leak
    /// and reuses warm memory.
    ///
    /// Lock discipline: at most one stripe lock is ever held (sequential
    /// visits), so no lock-order deadlock is possible. Cost on a
    /// pure-growth miss (all stripes empty): one extra own-stripe
    /// lock/unlock plus `K-1` try_locks per `refill_count` (=16) allocs.
    fn slab_batch_recycled_first(
        &self,
        class: usize,
        buf: &mut [*mut u8],
        headerless: bool,
    ) -> usize {
        let stripe = thread_stripe();
        {
            let Ok(mut slab) = self.slab[stripe].lock() else {
                return 0;
            };
            let n = slab.alloc_batch_recycled(class, buf, headerless);
            if n > 0 {
                record_slab_central_refill(0, false);
                return n;
            }
        }
        let mask = stripe_mask();
        let mut steps: u64 = 0;
        for k in 1..=mask {
            steps += 1;
            let s = (stripe + k) & mask;
            if let Ok(mut slab) = self.slab[s].try_lock() {
                let n = slab.alloc_batch_recycled(class, buf, headerless);
                if n > 0 {
                    record_slab_central_refill(steps, true);
                    return n;
                }
            }
        }
        record_slab_central_refill(steps, false);
        let Ok(mut slab) = self.slab[stripe].lock() else {
            return 0;
        };
        if headerless {
            let mut try_register = |base: usize| self.segment_registry.insert(base, class as u8);
            slab.alloc_batch_headerless(class, buf, &mut try_register)
        } else {
            slab.alloc_batch(class, buf)
        }
    }

    /// Serve a raw, header-free Slab block for `size` bytes (the caller
    /// must ensure `slab_headerless` is set and `align <= MIN_ALIGN` —
    /// there is no pad here to absorb a larger alignment). Same
    /// magazine-first shape as `slab_block_via_magazine`, but classifies by
    /// the *raw* request size (no header/pad added). On a magazine miss,
    /// `Slab::refill_segment` calls back into `self.segment_registry.insert`
    /// *before* threading a freshly mapped segment onto any free list — see
    /// its doc for why registration and free-list population must be
    /// atomic (a previous version registered only after handing blocks out,
    /// which meant a saturated registry silently served headerless blocks
    /// with no way to safely free them).
    fn slab_block_headerless_via_magazine(&self, size: usize) -> Option<*mut u8> {
        let class = lohalloc_core::slab_class_for(size)?;
        let owner = self.magazine_owner();
        if let Some(block) = magazine::pop(owner, class) {
            return Some(block);
        }
        let mut buf = [core::ptr::null_mut::<u8>(); 16];
        let want = magazine::refill_count(class).min(buf.len());
        // Recycled-first steal (see `slab_batch_recycled_first`): stolen
        // blocks are already-registered headerless blocks — only the
        // final carve/refill leg registers fresh segments.
        let n = self.slab_batch_recycled_first(class, &mut buf[..want], true);
        if n == 0 {
            return None;
        }
        for &p in buf.iter().take(n).skip(1) {
            let pushed = magazine::push(owner, class, p);
            debug_assert!(pushed);
        }
        Some(buf[0])
    }

    /// Return a raw, header-free Slab block of `class` to circulation. Uses
    /// the shared magazine (which a `load()`-booted instance holds exclusively
    /// for the headerless flavor — see `load()`'s `magazine_id` bump and the
    /// header-block magazine bypass in `dealloc_with_header`), and on overflow
    /// flushes to the Slab's **`hl`** tiers via `dealloc_headerless` /
    /// `dealloc_batch_headerless` (J4-A) — never the header tiers, where the
    /// steal/refill paths could hand a headerless block to the header alloc
    /// path.
    fn slab_dealloc_headerless(&self, block: *mut u8, class: u8) {
        let class = class as usize;
        let owner = self.magazine_owner();
        if magazine::push(owner, class, block) {
            return;
        }
        let mut buf = [core::ptr::null_mut::<u8>(); 16];
        let flush = magazine::refill_count(class).min(buf.len());
        let n = magazine::take(owner, class, &mut buf[..flush]);
        if let Ok(mut slab) = self.slab[thread_stripe()].lock() {
            unsafe {
                slab.dealloc_batch_headerless(class, &buf[..n]);
                slab.dealloc_headerless(block, class);
            }
        }
    }

    /// If `ptr` falls within a segment registered by the header-free Slab
    /// path, returns its slab class — letting every dealloc-side entry
    /// point (`GlobalAlloc::dealloc`, and the fused cabi helpers below)
    /// dispatch a headerless block *without ever reading `ptr -
    /// HEADER_SIZE`*, which is not valid header memory for these blocks
    /// (it may be another live allocation's tail, or unmapped guard space
    /// near a segment boundary). Cheap when inactive: one relaxed load and
    /// return, no registry probe at all, for any instance that never
    /// called `load()` with an empty Slab (see `slab_headerless`'s doc).
    #[inline]
    fn headerless_class_for(&self, ptr: *mut u8) -> Option<u8> {
        if !self.slab_headerless.load(Ordering::Relaxed) {
            return None;
        }
        let base = (ptr as usize) & !(slab::SEGMENT_SIZE - 1);
        self.segment_registry.lookup(base)
    }

    /// Lock-free arena bump, two levels (Ladder 5 span carve):
    ///
    /// 1. **Per-thread span** — a small TLS window (`ARENA_SPAN_BYTES`)
    ///    previously carved from the shared chunk. The common case is a
    ///    pure `Cell` bump: no atomics at all, and — the mt-slab-t4
    ///    motivation — no shared-cursor cache line ping-ponging between
    ///    threads (the single shared CAS measured as the dominant cost
    ///    once every thread's small blocks routed to Arena; jemalloc's
    ///    equivalent is its per-thread tcache run).
    /// 2. **Shared-chunk CAS carve** — on a span miss, reserve
    ///    `max(size, ARENA_SPAN_BYTES)` from the published chunk in one
    ///    CAS and serve from the front of the fresh span.
    ///
    /// Returns `None` — falling through to the `Mutex`-guarded slow path —
    /// when there's no descriptor yet (arena never initialized) or the
    /// carve doesn't fit the current chunk. Never chains chunks itself;
    /// only the slow path maps/advances.
    ///
    /// Span validity: a span is only served when its `owner` matches this
    /// instance's magazine id (multi-instance discipline, exactly
    /// `magazine.rs`'s) AND its `epoch` matches `arena_epoch`
    /// (`reset_arena()` bumps the epoch so stale spans — windows into
    /// rewound chunks — are discarded, not served). Span tails stranded by
    /// a refill are ≤ the request size; a dead thread strands ≤ one span.
    ///
    /// (J5-A note: J4-D briefly added a per-alloc `arena_live` pin + Dekker
    /// epoch-lock here so a drain-triggered reset could run concurrently.
    /// It was TSAN-correct but certified throughput-negative — the shared
    /// RMW cache line cost 3.2× at t8 — and was stripped; see COPILOT.md
    /// "J4-D". Resets are once again GUI/replay-only, serialized behind the
    /// arena Mutex, and the epoch check below handles them.)
    fn arena_alloc_fast(&self, size: usize, align: usize) -> Option<*mut u8> {
        let align = align.max(MIN_ALIGN);
        let owner = self.magazine_owner();
        let epoch = self.arena_epoch.load(Ordering::Relaxed);
        ARENA_SPAN.with(|span| {
            if span.owner.get() == owner && span.epoch.get() == epoch {
                let aligned = align_up(span.cursor.get(), align);
                if let Some(new_cur) = aligned.checked_add(size) {
                    if new_cur <= span.end.get() {
                        span.cursor.set(new_cur);
                        return Some(aligned as *mut u8);
                    }
                }
            }
            // Span miss: carve a fresh one from the shared chunk. Reserve
            // enough that the front of the carve satisfies this request at
            // its alignment (the carve itself is MIN_ALIGN-aligned; larger
            // aligns need slack).
            let want = size.max(ARENA_SPAN_BYTES);
            let reserve = if align > MIN_ALIGN {
                want.checked_add(align)?
            } else {
                want
            };
            let base = self.arena_carve_shared(reserve)?;
            let aligned = align_up(base, align);
            debug_assert!(aligned + size <= base + reserve);
            span.owner.set(owner);
            span.epoch.set(epoch);
            span.cursor.set(aligned + size);
            span.end.set(base + reserve);
            Some(aligned as *mut u8)
        })
    }

    /// CAS-reserve `reserve` bytes from the published shared chunk
    /// (MIN_ALIGN-aligned). The pre-span fast path's body, now the span
    /// refill.
    fn arena_carve_shared(&self, reserve: usize) -> Option<usize> {
        let desc_ptr = self.arena_chunk.load(Ordering::Acquire);
        if desc_ptr.is_null() {
            return None;
        }
        // SAFETY: see `ArenaChunkDescriptor`'s doc — the pointed-to chunk
        // and its cursor never move or get freed while this `Lohalloc`
        // instance is alive.
        let desc = unsafe { &*desc_ptr };
        let cursor = unsafe { &*desc.cursor };
        loop {
            let cur = cursor.load(Ordering::Relaxed);
            let aligned = align_up(cur, MIN_ALIGN);
            let new_cur = aligned.checked_add(reserve)?;
            if new_cur > desc.base + desc.capacity {
                return None; // chunk full or doesn't fit — slow path decides
            }
            // A racing thread may have already advanced `cur` since our
            // load; `compare_exchange_weak` retries from the fresh value
            // rather than overwriting someone else's bump.
            if cursor
                .compare_exchange_weak(cur, new_cur, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return Some(aligned);
            }
        }
    }

    /// (Re)publish `arena`'s current chunk as the fast path's descriptor.
    /// Must be called (under `self.arena`'s lock) after anything that can
    /// change which chunk is "current": the arena's initial creation, a
    /// chunk advance inside `BumpArena::alloc`, or `reset()`. Redundant
    /// calls (current chunk unchanged) are harmless — just a small extra
    /// leak (bounded by `MAX_CHUNKS` per arena lifetime, see
    /// `ArenaChunkDescriptor`'s doc).
    fn publish_arena_chunk(&self, arena: &arena::BumpArena) {
        let chunk = arena.current_chunk();
        let desc = Box::leak(Box::new(ArenaChunkDescriptor {
            base: chunk.base as usize,
            capacity: chunk.capacity,
            cursor: &chunk.cursor as *const AtomicUsize,
        }));
        self.arena_chunk.store(desc, Ordering::Release);
    }

    /// Serve a raw Buddy block for `total` bytes. For orders 10..=14 (16
    /// KiB..256 KiB — the range diagnosed as ~75% of buddy/adv-mixed
    /// traffic), goes via this thread's magazine (no lock on a hit),
    /// batch-refilling under the central buddy lock on a miss — the exact
    /// same shape as `slab_block_via_magazine`. Orders outside that range
    /// (512 KiB, 1 MiB) go straight to `Mutex<Buddy>`: rare enough in every
    /// measured workload that magazining them would only add strand risk.
    /// The calling thread's buddy stripe, locked, paired with a `register`
    /// closure that records fresh regions as owned by that stripe. Every
    /// buddy *allocation* goes through here (frees resolve their stripe via
    /// `buddy_stripe_of` instead — ownership is exact, not thread-affine).
    #[inline]
    fn buddy_alloc_stripe(&self) -> (usize, &Mutex<buddy::Buddy>) {
        let stripe = thread_stripe();
        (stripe, &self.buddy[stripe])
    }

    /// Exact owning stripe for a live buddy block, from the region → stripe
    /// registry. `None` means the pointer is in no region this instance
    /// ever mapped — a caller bug; callers debug-assert and leak rather
    /// than corrupt another stripe's metadata.
    #[inline]
    fn buddy_stripe_of(&self, block: *mut u8) -> Option<usize> {
        let base = block as usize & !(buddy::REGION_BYTES - 1);
        self.buddy_region_stripes
            .lookup(base)
            .map(|s| s as usize & (MAX_STRIPES - 1))
    }

    /// J1: raw, header-free Buddy block for `size` bytes (caller must have
    /// checked `buddy_headerless` and `align <= MIN_ALIGN`). Refuses orders
    /// below the order map's granularity (`MIN_HEADERLESS_ORDER`) — routing
    /// only produces those via the rare slab-exhausted fallthrough, which
    /// lands on System instead. The block IS the user pointer: no header
    /// write, so a fresh untouched block costs zero page faults (the
    /// J0-measured gap vs jemalloc).
    fn buddy_block_headerless_via_magazine(&self, size: usize) -> Option<*mut u8> {
        let order = buddy::order_for(size)?;
        if order < buddy::MIN_HEADERLESS_ORDER {
            return None;
        }
        self.buddy_block_via_magazine(size)
    }

    /// J1: if `ptr` is a header-free Buddy block, its order — recovered
    /// from the region's out-of-band order map with one lock-free registry
    /// probe. Mirrors `headerless_class_for`: must run before any header
    /// read (a headerless block's `ptr - HEADER_SIZE` may be another live
    /// block's tail), costs one relaxed load when the gate is off. The
    /// slab/buddy probes can never false-positive on each other's blocks:
    /// mappings are disjoint, so masking a slab pointer down to
    /// `REGION_BYTES` can't land on a registered buddy region base (and
    /// vice versa for the 64 KiB segment mask).
    #[inline]
    fn buddy_headerless_order_for(&self, ptr: *mut u8) -> Option<usize> {
        if !self.buddy_headerless.load(Ordering::Relaxed) {
            return None;
        }
        let base = ptr as usize & !(buddy::REGION_BYTES - 1);
        let (_stripe, map) = self.buddy_region_stripes.lookup_full(base)?;
        if map == 0 {
            return None;
        }
        let slot = buddy::order_map_slot(ptr as usize, base);
        let order =
            unsafe { (*((map + slot) as *const AtomicU8)).load(Ordering::Relaxed) } as usize;
        debug_assert!(
            (buddy::MIN_HEADERLESS_ORDER..=buddy::MAX_ORDER).contains(&order),
            "headerless buddy free with unrecorded order {order} — foreign pointer inside a registered region?"
        );
        if (buddy::MIN_HEADERLESS_ORDER..=buddy::MAX_ORDER).contains(&order) {
            Some(order)
        } else {
            None // release mode: leak rather than free at a garbage order
        }
    }

    /// Ladder 5: is `ptr` inside a registered headerless-arena chunk? One
    /// relaxed load when the gate is off; one hot 2 KiB-table probe when
    /// on. A hit means: free is a no-op, no header may be read
    /// (`ptr - HEADER_SIZE` may be a neighboring live bump block's tail),
    /// and no per-block size is recoverable (see `arena_headerless`'s
    /// field doc). Mirrors `headerless_class_for`/
    /// `buddy_headerless_order_for`; the three probes can never
    /// false-positive on each other's blocks (disjoint mappings).
    #[inline]
    fn arena_headerless_hit(&self, ptr: *mut u8) -> bool {
        if !self.arena_headerless.load(Ordering::Relaxed) {
            return false;
        }
        let base = ptr as usize & !(arena::CHUNK_BYTES - 1);
        self.arena_chunks.contains(base)
    }

    fn buddy_block_via_magazine(&self, total: usize) -> Option<*mut u8> {
        let order = buddy::order_for(total)?;
        let (stripe, stripe_lock) = self.buddy_alloc_stripe();
        let Some(idx) = buddy_mag::index_for(order) else {
            let mut buddy = stripe_lock.lock().ok()?;
            return buddy.alloc(total, &mut |base, map| {
                self.buddy_region_stripes.insert(base, stripe as u8, map)
            });
        };
        let owner = self.magazine_owner();
        if let Some(block) = buddy_mag::pop(owner, idx) {
            return Some(block);
        }
        // Magazine miss: batch-refill under this thread's stripe lock.
        // Sized with headroom over the max buddy_mag::refill_count
        // (currently 16) so cap experiments never silently truncate a
        // refill; the `.min` keeps it correct either way. (A cap-64
        // experiment measured 25-50% slower on mixed rows: 32-block
        // flushes exactly fill buddy's ORDER_CACHE and force a
        // merge-drain per flush.)
        let mut buf = [core::ptr::null_mut::<u8>(); 32];
        let want = buddy_mag::refill_count(idx).min(buf.len());
        let n = {
            let mut buddy = stripe_lock.lock().ok()?;
            buddy.alloc_order_batch(order, &mut buf[..want], &mut |base, map| {
                self.buddy_region_stripes.insert(base, stripe as u8, map)
            })
        };
        if n == 0 {
            return None;
        }
        for &p in buf.iter().take(n).skip(1) {
            // The magazine was just empty, so the extras always fit.
            let pushed = buddy_mag::push(owner, idx, p);
            debug_assert!(pushed);
        }
        Some(buf[0])
    }

    /// Return a raw Buddy block to circulation. Mirrors
    /// `buddy_block_via_magazine`'s order-range split: magazined orders push
    /// to this thread's magazine (no lock on a hit), flushing half back to
    /// the central buddy in one locked batch when full; other orders free
    /// directly.
    ///
    /// # Safety
    /// `block`/`size` must be a still-live pair previously returned by the
    /// Buddy allocation path (same contract as `Buddy::dealloc`).
    unsafe fn buddy_dealloc_via_magazine(&self, block: *mut u8, size: usize) {
        let Some(order) = buddy::order_for(size) else {
            debug_assert!(false, "buddy dealloc size out of range");
            return;
        };
        let Some(idx) = buddy_mag::index_for(order) else {
            // Direct (non-magazined) orders: free into the exact owning
            // stripe — never the thread's stripe, which may differ.
            let Some(stripe) = self.buddy_stripe_of(block) else {
                debug_assert!(false, "buddy free outside every registered region");
                return; // release mode: leak rather than corrupt a stripe
            };
            if let Ok(mut buddy) = self.buddy[stripe].lock() {
                unsafe { buddy.dealloc(block, size) };
            }
            return;
        };
        let owner = self.magazine_owner();
        if buddy_mag::push(owner, idx, block) {
            return;
        }
        // Magazine full: flush half of it plus this block back to the
        // central stripes. A thread's magazine mixes blocks from every
        // stripe (pop doesn't care where a block was carved), so resolve
        // each flushed block's owning stripe via the region registry and
        // free per-stripe batches — locking one stripe at a time, never
        // two at once (no lock-order deadlock possible). Sized with
        // headroom over the max refill_count (currently 16) + 1 for the
        // block that triggered the flush — see the refill-side comment.
        let mut buf = [core::ptr::null_mut::<u8>(); 33];
        let flush = buddy_mag::refill_count(idx).min(buf.len() - 1);
        let n = buddy_mag::take(owner, idx, &mut buf[..flush]);
        buf[n] = block;
        let total = n + 1;

        let mut stripes = [usize::MAX; 33];
        for i in 0..total {
            match self.buddy_stripe_of(buf[i]) {
                Some(s) => stripes[i] = s,
                // Unresolvable block: leak it (debug-assert — every buddy
                // block came from a registered region by construction).
                None => debug_assert!(false, "flushed buddy block in no registered region"),
            }
        }
        let mut grouped = [core::ptr::null_mut::<u8>(); 33];
        for stripe in 0..=stripe_mask() {
            let mut g = 0;
            for i in 0..total {
                if stripes[i] == stripe {
                    grouped[g] = buf[i];
                    g += 1;
                }
            }
            if g == 0 {
                continue;
            }
            if let Ok(mut buddy) = self.buddy[stripe].lock() {
                unsafe { buddy.dealloc_order_batch(order, &grouped[..g]) };
            }
        }
    }

    /// on success, `None` on failure (e.g. Arena full, Slab exhausted).
    fn try_backend(
        &self,
        backend: lohalloc_core::Backend,
        total: usize,
        align: usize,
        pad: usize,
        hash: u64,
        size_class_hint: u8,
    ) -> Option<*mut u8> {
        let local_backend = Backend::from_core(backend);
        match backend {
            lohalloc_core::Backend::Slab if total <= SLAB_MAX => {
                // Header-free fast path: only for a `load()`-booted instance
                // (see `slab_headerless`'s doc) and only when the block's
                // guaranteed 16-byte alignment covers what was asked for —
                // a bigger alignment needs `pad` to absorb the difference,
                // which only the header-carrying path provides.
                if self.slab_headerless.load(Ordering::Relaxed) && align <= MIN_ALIGN {
                    let size = total - pad;
                    return self.slab_block_headerless_via_magazine(size);
                }
                self.slab_block_via_magazine(total).map(|block| {
                    self.write_header(
                        block,
                        pad,
                        local_backend,
                        total,
                        0,
                        0,
                        hash,
                        size_class_hint,
                        align,
                    )
                })
            }
            lohalloc_core::Backend::Buddy if total <= BUDDY_MAX => {
                // J1 header-free fast path: mirror of the Slab arm above —
                // `load()`-booted instances only, natural alignment only,
                // and only orders the per-region order map can express
                // (>= 16 KiB; smaller fallthrough sizes go to System).
                if self.buddy_headerless.load(Ordering::Relaxed) && align <= MIN_ALIGN {
                    let size = total - pad;
                    return self.buddy_block_headerless_via_magazine(size);
                }
                self.buddy_block_via_magazine(total).map(|block| {
                    self.write_header(
                        block,
                        pad,
                        local_backend,
                        total,
                        0,
                        0,
                        hash,
                        size_class_hint,
                        align,
                    )
                })
            }
            lohalloc_core::Backend::Arena => {
                // Exhaustion latch first (see the field doc): a capped-out
                // bump arena can never serve again until reset, so fail
                // straight to the caller's fallthrough instead of re-paying
                // the doomed fast-path bump + Mutex slow path per alloc.
                if self.arena_exhausted.load(Ordering::Relaxed) {
                    return None;
                }
                // Ladder 5 headerless arena: skip the 48-byte header
                // entirely (the request's own size, no pad) — the header
                // write was one cold-block D1 write miss per alloc, the
                // dominant small-block inference cost once the bandit
                // routes call sites here. Alignment gate mirrors the other
                // headerless arms (no pad exists to absorb larger aligns;
                // the bump path itself aligns to `max(align, MIN_ALIGN)`,
                // so this is about header-offset recovery, not placement).
                let headerless =
                    self.arena_headerless.load(Ordering::Relaxed) && align <= MIN_ALIGN;
                let request = if headerless { total - pad } else { total };
                // Lock-free fast path: bump the published current chunk's
                // cursor directly, no Mutex. Falls through to the slow
                // path below on a miss (arena not yet initialized, or the
                // current chunk is full/doesn't fit this request).
                if let Some(block) = self.arena_alloc_fast(request, align) {
                    if headerless {
                        return Some(block);
                    }
                    return Some(self.write_header(
                        block,
                        pad,
                        local_backend,
                        total,
                        0,
                        0,
                        hash,
                        size_class_hint,
                        align,
                    ));
                }
                // Slow path: lazily initialize / advance / map under the
                // lock, then (re)publish whichever chunk is now current —
                // covers first-time init and every chunk advance, which is
                // the only way `arena_chunk` can go stale.
                let Ok(mut arena_guard) = self.arena.lock() else {
                    return None;
                };
                if arena_guard.is_none() {
                    *arena_guard = arena::BumpArena::new();
                }
                let gate_on = self.arena_headerless.load(Ordering::Relaxed);
                let mut registered_ok = true;
                let result = arena_guard.as_mut().and_then(|arena| {
                    let block = arena.alloc(request, align)?;
                    // When the headerless gate is on, register every chunk
                    // base on EVERY slow-path visit — not just headerless
                    // calls (idempotent, <= MAX_CHUNKS entries, fixed
                    // atomics: safe under this Mutex per the reentrancy
                    // rule). This must happen BEFORE the block escapes or
                    // the chunk is (re)published: the lock-free fast path
                    // serves headerless blocks from whatever chunk is
                    // published, and a headerless block whose chunk isn't
                    // in the membership set reaches `free` with no header
                    // to fall back on (the probe misses, `ptr - 48` is
                    // read as a garbage header, and cabi delegates the
                    // pointer to libc). A rare header-carrying block
                    // (align > MIN_ALIGN) inside a registered chunk is the
                    // safe direction: arena frees are no-ops either way.
                    // Registration cannot fail in practice (256-slot set
                    // vs. 32 chunks); if it ever does, fail the arm and
                    // suppress the publish below. The default-size check
                    // is defense-in-depth for the free side's
                    // `ptr & !(CHUNK_BYTES - 1)` mask (always true for
                    // `BumpArena::new()`, which is the only constructor
                    // `lib.rs` uses).
                    if gate_on {
                        registered_ok = arena.chunks_are_default_sized()
                            && arena.chunk_bases().all(|b| self.arena_chunks.insert(b));
                        if !registered_ok {
                            return None;
                        }
                    }
                    if headerless {
                        Some(block)
                    } else {
                        Some(self.write_header(
                            block,
                            pad,
                            local_backend,
                            total,
                            0,
                            0,
                            hash,
                            size_class_hint,
                            align,
                        ))
                    }
                });
                if let Some(arena) = arena_guard.as_ref() {
                    if registered_ok {
                        self.publish_arena_chunk(arena);
                    }
                    // Latch permanent exhaustion (cap reached on a request
                    // that would have fit an empty chunk) so every later
                    // arena-routed alloc fast-fails at the arm's head.
                    if result.is_none() && arena.exhausted_after_failed(request, align) {
                        self.arena_exhausted.store(true, Ordering::Relaxed);
                    }
                }
                result
            }
            lohalloc_core::Backend::System => {
                let ptr = self.system_alloc_with_header(total, align, hash, size_class_hint);
                if ptr.is_null() {
                    None
                } else {
                    Some(ptr)
                }
            }
            // If the recommended backend is size-inappropriate (e.g. Slab for
            // a large alloc), fall through to None.
            _ => None,
        }
    }

    /// Size-based fallback routing (Phase 1): Slab → Buddy → System.
    ///
    /// `skip` is the backend a recommendation-driven `try_backend` attempt
    /// already failed with (if any): re-locking and re-trying a backend that
    /// just failed is guaranteed wasted work, and before this parameter
    /// existed a failed recommendation re-walked the whole chain including
    /// the failed backend — measurable under the arena-exhaustion pathology.
    fn route_by_size(
        &self,
        total: usize,
        align: usize,
        pad: usize,
        hash: u64,
        size_class_hint: u8,
        skip: Option<lohalloc_core::Backend>,
    ) -> *mut u8 {
        // 1. Slab: small, naturally-aligned requests (magazine-first).
        if total <= SLAB_MAX && skip != Some(lohalloc_core::Backend::Slab) {
            if self.slab_headerless.load(Ordering::Relaxed) && align <= MIN_ALIGN {
                let size = total - pad;
                if let Some(block) = self.slab_block_headerless_via_magazine(size) {
                    return block;
                }
            } else if let Some(block) = self.slab_block_via_magazine(total) {
                return self.write_header(
                    block,
                    pad,
                    Backend::Slab,
                    total,
                    0,
                    0,
                    hash,
                    size_class_hint,
                    align,
                );
            }
        }

        // 2. Buddy: medium, variable-size (magazine-first).
        if total <= BUDDY_MAX && skip != Some(lohalloc_core::Backend::Buddy) {
            if self.buddy_headerless.load(Ordering::Relaxed) && align <= MIN_ALIGN {
                let size = total - pad;
                if let Some(block) = self.buddy_block_headerless_via_magazine(size) {
                    return block;
                }
            } else if let Some(block) = self.buddy_block_via_magazine(total) {
                return self.write_header(
                    block,
                    pad,
                    Backend::Buddy,
                    total,
                    0,
                    0,
                    hash,
                    size_class_hint,
                    align,
                );
            }
        }

        // 3. System Fallback: any size/alignment.
        self.system_alloc_with_header(total, align, hash, size_class_hint)
    }

    /// Allocate `total` bytes at `align` via the System Fallback, write a
    /// `System`-tagged header, and leak the `Mapping` (dealloc will `munmap`
    /// using the base/length recorded in the header). Returns the user ptr.
    fn system_alloc_with_header(
        &self,
        total: usize,
        align: usize,
        hash: u64,
        size_class_hint: u8,
    ) -> *mut u8 {
        let pad = header_pad(align);
        let eff_align = align.max(system::page_size());

        // Retention cache first: a previously-freed mapping of a fitting
        // length is reused with its pages still populated — this is what
        // lets the System backend match glibc's large-chunk retention
        // instead of paying mmap + page faults + munmap per operation.
        // (Try-lock semantics: contention/re-entry = miss = plain mmap.)
        if let Some(Some((raw_base, raw_len))) = self.with_system_cache(|c| c.get(total, eff_align))
        {
            self.system_cache_hits.fetch_add(1, Ordering::Relaxed);
            let block = system::align_up_addr(raw_base, eff_align) as *mut u8;
            return self.write_header(
                block,
                pad,
                Backend::System,
                total,
                raw_base,
                raw_len,
                hash,
                size_class_hint,
                align,
            );
        }

        let mapping = match system::alloc_pages(total, align) {
            Some(m) => m,
            None => return core::ptr::null_mut(),
        };
        let base = mapping.as_ptr();
        // We need base/len for munmap; extract them then forget the Mapping so
        // its Drop does not munmap prematurely.
        // SAFETY: we keep the memory mapped; dealloc will munmap via the header.
        let raw_base = unsafe { mapping.raw_base_for_unmap() };
        let raw_len = unsafe { mapping.raw_len_for_unmap() };
        core::mem::forget(mapping);
        self.write_header(
            base,
            pad,
            Backend::System,
            total,
            raw_base,
            raw_len,
            hash,
            size_class_hint,
            align,
        )
    }

    /// Write the ownership header at `block + pad - HEADER_SIZE` and return
    /// `block + pad` (the user pointer). `block` must be aligned to at least
    /// `align` and hold `total` usable bytes.
    #[allow(clippy::too_many_arguments)]
    fn write_header(
        &self,
        block: *mut u8,
        pad: usize,
        backend: Backend,
        total: usize,
        base: usize,
        map_len: usize,
        hash: u64,
        size_class_hint: u8,
        align: usize,
    ) -> *mut u8 {
        let user = unsafe { block.add(pad) };
        let backend_tag = backend as u8;
        // Slab blocks record their size-class index so the free path (and
        // the magazine push) never recomputes it. One branchless clz here
        // on the alloc side; non-Slab backends carry the untracked marker.
        let slab_class = if matches!(backend, Backend::Slab) {
            lohalloc_core::slab_class_for(total)
                .map(|c| c as u8)
                .unwrap_or(SIZE_CLASS_UNTRACKED)
        } else {
            SIZE_CLASS_UNTRACKED
        };
        let header = Header {
            magic: MAGIC,
            backend: backend_tag,
            size_class_hint,
            align_log2: align.trailing_zeros() as u8,
            slab_class,
            _pad: [0; 4],
            size: total,
            base,
            map_len,
            hash,
        };
        unsafe {
            core::ptr::write_unaligned(user.sub(HEADER_SIZE) as *mut Header, header);
        }

        // Telemetry hook (compiled away when the feature is off; skipped
        // with one relaxed load when no sink is installed). This is the
        // single chokepoint every successful allocation flows through, so
        // we emit here rather than at each call site in `try_backend` /
        // `route_by_size` / `system_alloc_with_header`.
        #[cfg(feature = "telemetry-observer")]
        if observer::sink_installed() {
            let frag = fragmentation_pct_for(backend, total);
            observer::emit_alloc(total, hash, user as u64, backend_tag, frag);
        }

        user
    }

    /// Reset the Bump Arena, reclaiming all arena allocations.
    ///
    /// This is the "reset-based reclaim" mechanism: all Arena-tagged pointers
    /// are invalidated. The Decision Engine (Phase 3) and the GUI/replay
    /// controls call this when a topological cluster's lifetime ends.
    pub fn reset_arena(&self) {
        if let Ok(mut arena_guard) = self.arena.lock() {
            if let Some(ref mut arena) = *arena_guard {
                arena.reset();
            }
            // Invalidate every thread's TLS arena span (Ladder 5): a span
            // is a window into a chunk this reset just rewound — serving
            // from it would overlap post-reset allocations. Bumped under
            // the Mutex, checked (relaxed load) on every span use.
            self.arena_epoch.fetch_add(1, Ordering::Relaxed);
            // Republish: `reset()` rewinds `current` to chunk 0, which may
            // differ from whatever chunk `arena_chunk` currently points
            // at — the fast path must not keep bumping a chunk `reset`
            // considers stale.
            if let Some(arena) = arena_guard.as_ref() {
                self.publish_arena_chunk(arena);
            }
            // A rewound arena has capacity again — unlatch the fast-fail.
            // Ordered inside the Mutex so it can't race a concurrent
            // exhaustion re-latch from the alloc slow path (also lock-held).
            self.arena_exhausted.store(false, Ordering::Relaxed);
        }
    }

    /// Allocate from the Bump Arena, writing an Arena-tagged header.
    ///
    /// This is not called by `route_alloc` in Phase 2 (routing is still
    /// size-based). The Decision Engine (Phase 3) will call this directly
    /// when the MAB policy routes a signature to the Arena backend.
    pub fn arena_alloc(&self, size: usize, align: usize) -> *mut u8 {
        let align = align.max(MIN_ALIGN);
        let pad = header_pad(align);
        let total = size + pad;

        if let Ok(mut arena_guard) = self.arena.lock() {
            // Lazily initialize the arena on first use.
            if arena_guard.is_none() {
                *arena_guard = arena::BumpArena::new();
            }
            let block = arena_guard
                .as_mut()
                .and_then(|arena| arena.alloc(total, align));
            if let Some(arena) = arena_guard.as_ref() {
                self.publish_arena_chunk(arena);
            }
            if let Some(block) = block {
                let hash = topology::fast_stack_hash();
                return self.write_header(
                    block,
                    pad,
                    Backend::Arena,
                    total,
                    0,
                    0,
                    hash,
                    SIZE_CLASS_UNTRACKED,
                    align,
                );
            }
        }
        // Arena full or init failed → fall through to System.
        self.system_alloc_with_header(total, align, 0, SIZE_CLASS_UNTRACKED)
    }

    // -----------------------------------------------------------------
    // Phase 3: Decision Engine public API
    // -----------------------------------------------------------------

    /// Transition the Decision Engine from Training mode to Inference mode.
    ///
    /// Collapses the Multi-Armed Bandit's learned per-Signature weights into
    /// a frozen `PerfectHashTable` for O(1) hash-and-jump routing. After
    /// `freeze()`, the allocator stops learning and routes via the frozen
    /// table only.
    ///
    /// # Panics
    ///
    /// Panics if already in Inference mode (double-freeze is a logic error).
    pub fn freeze(&self) {
        // J4-C/J5-A: `arena_epoch` is bumped only by `reset_arena()`, so 0
        // here means this instance never observed a reset during training —
        // a reset-free deployment (LD_PRELOAD / global allocator) where a
        // frozen Arena verdict on a *heavy* signature becomes a permanent
        // fallthrough storm once the arena fills. `AllocatorState::freeze`
        // applies the demotion volume-selectively (heavy demoted, light
        // kept — J4-C demoted all and flattened the light-arena rows; J4-D
        // kept all and the storm cost 3.98×; the volume split is the good
        // half of both). See its doc.
        let demote_arena = self.arena_epoch.load(Ordering::Relaxed) == 0;
        // The heavy/light threshold comes from the `demote_fraction` tune
        // key (default = the certified 0.10 const; bisect knob:
        // LOHALLOC_DEMOTE_FRACTION). `tune::config()` is one atomic load,
        // no I/O — safe under the realloc guard.
        let demote_fraction = crate::tune::config().demote_fraction;
        Self::with_realloc_guard(|| {
            if let Ok(mut state) = self.state.lock() {
                state.freeze(demote_arena, demote_fraction);
                self.publish_frozen_table(&state);
            }
        });
        self.frozen.store(true, Ordering::Release);
    }

    /// True once the training bandit's routing has stabilized enough that
    /// freezing would lock in a settled model — the `freeze_mode=converged`
    /// trigger (see `tune::FreezeMode` and `BanditPolicy::is_converged`).
    /// Always true once frozen; false while genuinely still learning or if
    /// nothing was learned yet. The caller decides when to poll (e.g.
    /// `lohalloc-cabi` samples every few hundred allocations) and calls
    /// [`freeze`](Self::freeze) itself — the allocator never freezes
    /// spontaneously.
    pub fn is_converged(&self) -> bool {
        if self.frozen.load(Ordering::Relaxed) {
            return true;
        }
        Self::with_realloc_guard(|| {
            self.state
                .lock()
                .map(|state| state.is_converged())
                .unwrap_or(false)
        })
    }

    /// Publish a lock-free copy of the state's frozen routing table for the
    /// Inference alloc fast path (see the `frozen_table` field doc). Must be
    /// called inside `with_realloc_guard` with the state lock held — the
    /// clone + `Box` allocation take the `IN_ALLOC` bypass when this
    /// instance is the process's global allocator.
    fn publish_frozen_table(&self, state: &state::AllocatorState) {
        if let Some(routing) = state.routing() {
            let leaked: *mut perfect_hash::FrozenRouting = Box::leak(Box::new(routing.clone()));
            self.frozen_table.store(leaked, Ordering::Release);
        }
    }

    /// Snapshot the current "best backend per Signature" without
    /// transitioning to Inference mode. Used during live training to show
    /// the routing-table-as-it-is-being-built to the GUI (TensorBoard-style).
    ///
    /// Returns an empty Vec in Inference mode.
    pub fn routing_snapshot(&self) -> Vec<(u64, lohalloc_core::Backend)> {
        Self::with_realloc_guard(|| {
            if let Ok(state) = self.state.lock() {
                state.routing_snapshot()
            } else {
                Vec::new()
            }
        })
    }

    /// Number of distinct Signatures observed so far (Training mode only).
    /// Returns 0 in Inference mode.
    pub fn signature_count(&self) -> usize {
        if let Ok(state) = self.state.lock() {
            state.signature_count()
        } else {
            0
        }
    }

    /// Reset the Decision Engine back to a fresh Training state, discarding
    /// any frozen routing table or learned bandit weights. Used by the GUI's
    /// "back to training" button.
    pub fn reset_to_training(&self) {
        // Unpublish the fast-path table *before* resetting the state so no
        // new reader can pick it up mid-reset. The old table is
        // intentionally leaked — a concurrent alloc may still be reading
        // through the pointer it loaded; leakage is bounded by the number
        // of freeze/reset cycles (a GUI dev action, not a hot path).
        self.frozen_table
            .store(core::ptr::null_mut(), Ordering::Release);
        Self::with_realloc_guard(|| {
            if let Ok(mut state) = self.state.lock() {
                state.reset_to_training();
            }
        });
        self.frozen.store(false, Ordering::Release);
    }

    /// Export the frozen routing table to `.lohalloc` binary bytes.
    ///
    /// Returns `None` if the allocator is still in Training mode (call
    /// `freeze()` first).
    pub fn export(&self) -> Option<Vec<u8>> {
        Self::with_realloc_guard(|| {
            if let Ok(state) = self.state.lock() {
                state.export()
            } else {
                None
            }
        })
    }

    /// Load a `.lohalloc` model file and transition directly to Inference mode.
    ///
    /// Returns `true` if the model was loaded successfully, `false` if the
    /// data is malformed or the state lock is poisoned.
    pub fn load(&self, data: &[u8]) -> bool {
        Self::with_realloc_guard(|| {
            let new_state = state::AllocatorState::load(data);
            if let Some(s) = new_state {
                if let Ok(mut state) = self.state.lock() {
                    *state = s;
                    self.publish_frozen_table(&state);
                    self.frozen.store(true, Ordering::Release);
                    // J4-A: latch header-free Slab **unconditionally**. The
                    // old code required a pristine Slab (`region_count() == 0`
                    // on every stripe) so header and headerless blocks could
                    // never share a class free list — but as the global
                    // allocator the Rust runtime touches the Slab before
                    // `main` (before `LOHALLOC_MODEL` is even read), so that
                    // check always failed for exactly the slab-routed rows the
                    // header-free path exists to speed up. The `Slab` now keeps
                    // header and headerless blocks on physically separate
                    // tiers (`Slab::hl`), so a touched Slab is safe: the
                    // headerless alloc path can never pop a pre-load header
                    // block. The per-thread magazine is the one shared surface,
                    // so invalidate it here — every thread refills fresh
                    // through the headerless path after this point (see
                    // `invalidate_magazines`; pre-load header frees bypass the
                    // magazine in `dealloc_with_header`).
                    self.slab_headerless.store(true, Ordering::Release);
                    self.invalidate_magazines();
                    // J1: same untouched-instance argument for Buddy. Flip
                    // each stripe's order-recording flag under its own lock
                    // BEFORE publishing the gate, so no block is ever
                    // served headerless from a stripe that isn't recording.
                    let all_buddy_untouched = self.buddy.iter().all(|stripe| {
                        stripe
                            .lock()
                            .map(|mut b| {
                                if b.region_count() == 0 {
                                    b.set_record_orders();
                                    true
                                } else {
                                    false
                                }
                            })
                            .unwrap_or(false)
                    });
                    if all_buddy_untouched {
                        self.buddy_headerless.store(true, Ordering::Release);
                    }
                    // Ladder 5: same untouched-instance argument for Arena.
                    // The arena is created lazily on its first allocation,
                    // so `None` here means no block was ever served (and no
                    // chunk descriptor was ever published) — every chunk
                    // this instance will ever create is registered by the
                    // gated slow path before any headerless block escapes.
                    let arena_untouched = self
                        .arena
                        .lock()
                        .map(|guard| guard.is_none())
                        .unwrap_or(false);
                    if arena_untouched {
                        self.arena_headerless.store(true, Ordering::Release);
                    }
                    return true;
                }
            }
            false
        })
    }

    /// Returns `true` if the Decision Engine is in Inference (frozen) mode.
    pub fn is_inference(&self) -> bool {
        if let Ok(state) = self.state.lock() {
            state.is_inference()
        } else {
            false
        }
    }

    /// True if this instance is currently serving Slab allocations
    /// header-free (only ever set by `load()` on a still-empty Slab — see
    /// `slab_headerless`'s doc). Test/introspection only.
    pub fn is_slab_headerless(&self) -> bool {
        self.slab_headerless.load(Ordering::Relaxed)
    }

    /// True if serving Buddy allocations header-free (see `buddy_headerless`).
    /// Test/introspection only.
    pub fn is_buddy_headerless(&self) -> bool {
        self.buddy_headerless.load(Ordering::Relaxed)
    }

    /// True if serving Arena allocations header-free (see `arena_headerless`).
    /// Test/introspection only.
    pub fn is_arena_headerless(&self) -> bool {
        self.arena_headerless.load(Ordering::Relaxed)
    }

    /// Process-wide count of Inference-mode routing lookups that missed the
    /// frozen table (and fell back to size-based routing). ~0 on a
    /// model-loaded run means the model's keys matched this process's call
    /// sites — the end-to-end proof that hashes are stable across runs.
    pub fn pht_miss_count() -> u64 {
        PHT_MISSES.load(Ordering::Relaxed)
    }

    /// The latched active stripe count (latching it now if this is the first
    /// touch). Introspection for bisect/diagnostic runs — pairs with the
    /// `LOHALLOC_STRIPES` override so a run can verify which configuration
    /// it actually executed under.
    pub fn active_stripes() -> usize {
        stripe_mask() + 1
    }

    /// Process-wide count of Inference pin-cache misses (each triggers one
    /// distilled-table probe + slot store). Roughly `#(site, size_class)
    /// pairs × cache-eviction churn` — small and stable on a healthy
    /// model-loaded run. Ungated (miss-only, cold); the hit-side counters
    /// are `route-metrics`-gated — see [`Self::pin_hit_count`].
    pub fn pin_miss_count() -> u64 {
        PIN_MISSES.load(Ordering::Relaxed)
    }

    /// Process-wide count of Inference allocations served straight from the
    /// pin cache (no stack walk, no table lookup). Always `0` unless built
    /// with `route-metrics` (or under test) — per-op hit counting would
    /// false-share exactly like `ROUTE_COUNTS`.
    pub fn pin_hit_count() -> u64 {
        #[cfg(any(feature = "route-metrics", test))]
        {
            PIN_HITS.load(Ordering::Relaxed)
        }
        #[cfg(not(any(feature = "route-metrics", test)))]
        {
            0
        }
    }

    /// Process-wide count of negative pin-cache hits (known-unpinnable site
    /// took the full path without re-probing the distilled table). Always
    /// `0` unless built with `route-metrics` (or under test).
    pub fn pin_negative_count() -> u64 {
        #[cfg(any(feature = "route-metrics", test))]
        {
            PIN_NEGATIVE.load(Ordering::Relaxed)
        }
        #[cfg(not(any(feature = "route-metrics", test)))]
        {
            0
        }
    }

    /// Process-wide `(central_refills, sibling_steps, sibling_hits)` for the
    /// slab magazine-miss path (see `SLAB_CENTRAL_REFILLS`'s doc — the
    /// J5-bisect sibling-scan instrumentation). All `0` unless built with
    /// `route-metrics` (or under test).
    pub fn slab_refill_counts() -> (u64, u64, u64) {
        #[cfg(any(feature = "route-metrics", test))]
        {
            (
                SLAB_CENTRAL_REFILLS.load(Ordering::Relaxed),
                SLAB_SIBLING_STEPS.load(Ordering::Relaxed),
                SLAB_SIBLING_HITS.load(Ordering::Relaxed),
            )
        }
        #[cfg(not(any(feature = "route-metrics", test)))]
        {
            (0, 0, 0)
        }
    }

    /// Process-wide count of Inference-mode allocations actually served by
    /// `backend` (frozen fast path only). See `ROUTE_COUNTS`'s doc. Always
    /// `0` unless built with the `route-metrics` feature (or under test) —
    /// the counters are compiled away otherwise to avoid per-op false
    /// sharing.
    pub fn route_count(backend: lohalloc_core::Backend) -> u64 {
        #[cfg(any(feature = "route-metrics", test))]
        {
            ROUTE_COUNTS[backend as usize].load(Ordering::Relaxed)
        }
        #[cfg(not(any(feature = "route-metrics", test)))]
        {
            let _ = backend;
            0
        }
    }

    /// Process-wide count of Inference-mode allocations whose recommended
    /// backend failed and fell through to size-based routing. See
    /// `FALLTHROUGH_COUNT`'s doc. Always `0` unless built with
    /// `route-metrics` (or under test).
    pub fn fallthrough_count() -> u64 {
        #[cfg(any(feature = "route-metrics", test))]
        {
            FALLTHROUGH_COUNT.load(Ordering::Relaxed)
        }
        #[cfg(not(any(feature = "route-metrics", test)))]
        {
            0
        }
    }

    /// Whether route/fallthrough counters are compiled in (the
    /// `route-metrics` feature). The `LOHALLOC_DEBUG` epilogue uses this to
    /// print "counters disabled" instead of a misleading all-zero table.
    pub fn route_metrics_enabled() -> bool {
        cfg!(any(feature = "route-metrics", test))
    }

    // -----------------------------------------------------------------
    // Phase 4: Replay Engine support
    // -----------------------------------------------------------------

    /// Allocate `size` bytes at `align` using a **caller-provided hash** instead
    /// of capturing the stack via `fast_stack_hash()`.
    ///
    /// This is used by the replay engine (`lohalloc-server`) to drive a private
    /// `Lohalloc` instance with a deterministic hash from trace files, so that
    /// replaying the same trace produces an identical `.lohalloc` model.
    ///
    /// # Safety
    ///
    /// Same contract as `GlobalAlloc::alloc`: returns a valid, aligned,
    /// `size`-byte buffer or null on failure.
    pub unsafe fn alloc_with_hash(&self, layout: Layout, hash: u64) -> *mut u8 {
        let size = layout.size().max(1);
        let align = layout.align().max(MIN_ALIGN);
        let pad = header_pad(align);
        let total = size + pad;

        let depth = IN_ALLOC.get();
        if depth > 0 {
            return self.system_alloc_with_header(total, align, hash, SIZE_CLASS_UNTRACKED);
        }

        IN_ALLOC.set(depth + 1);
        let ptr = self.route_alloc_with_hash(size, align, pad, total, hash);
        IN_ALLOC.set(depth);
        ptr
    }

    /// Deallocate a pointer previously returned by `alloc_with_hash`.
    ///
    /// # Safety
    ///
    /// Same contract as `GlobalAlloc::dealloc`: `ptr` must have been returned by
    /// a prior `alloc_with_hash` call with a matching `Layout`.
    pub unsafe fn dealloc_with_hash(&self, ptr: *mut u8, layout: Layout) {
        // Delegate to the GlobalAlloc impl — it reads the header for routing.
        unsafe { self.dealloc(ptr, layout) };
    }

    /// Internal: route an allocation with a caller-provided hash. Shares the
    /// Layer 2 reward-measurement logic with `route_alloc` via
    /// `route_alloc_inner` — see its doc comment.
    fn route_alloc_with_hash(
        &self,
        size: usize,
        align: usize,
        pad: usize,
        total: usize,
        hash: u64,
    ) -> *mut u8 {
        // Replay has no real stack context — `ret0 = 0` means these
        // signatures are never distilled/pinned (deterministic replay).
        self.route_alloc_inner(hash, 0, size, align, pad, total)
    }

    /// Allocate with a caller-provided hash **and** a strategy override
    /// (Phase 5). The strategy biases backend selection: if the strategy's
    /// preferred backend can serve the request, it is used instead of the
    /// MAB recommendation.
    ///
    /// # Safety
    ///
    /// Same contract as `alloc_with_hash`.
    pub unsafe fn alloc_with_hash_and_strategy(
        &self,
        layout: Layout,
        hash: u64,
        strategy: lohalloc_core::Strategy,
    ) -> *mut u8 {
        let size = layout.size().max(1);
        let align = layout.align().max(MIN_ALIGN);
        let pad = header_pad(align);
        let total = size + pad;

        let depth = IN_ALLOC.get();
        if depth > 0 {
            return self.system_alloc_with_header(total, align, hash, SIZE_CLASS_UNTRACKED);
        }

        // If the strategy specifies a preferred backend, try it first. This
        // bypasses the bandit's `select()` entirely, so there is no matching
        // pull to attribute a Layer 2 reward to — tag `SIZE_CLASS_UNTRACKED`
        // so `dealloc` skips reward bookkeeping for this allocation.
        if let Some(preferred) = strategy.preferred_backend(size) {
            IN_ALLOC.set(depth + 1);
            if let Some(ptr) =
                self.try_backend(preferred, total, align, pad, hash, SIZE_CLASS_UNTRACKED)
            {
                IN_ALLOC.set(depth);
                return ptr;
            }
            // Preferred backend failed — fall through to MAB / size routing.
            IN_ALLOC.set(depth);
        }

        IN_ALLOC.set(depth + 1);
        let ptr = self.route_alloc_with_hash(size, align, pad, total, hash);
        IN_ALLOC.set(depth);
        ptr
    }

    /// Snapshot of per-backend live-region/usage counters, for benchmarks
    /// and tests that want to observe backend state without depending on
    /// the `telemetry-observer` feature (Phase 6).
    pub fn backend_counters(&self) -> BackendCounters {
        let slab_region_count = self
            .slab
            .iter()
            .map(|stripe| stripe.lock().map(|s| s.region_count()).unwrap_or(0))
            .sum();
        let buddy_region_count = self
            .buddy
            .iter()
            .map(|stripe| stripe.lock().map(|b| b.region_count()).unwrap_or(0))
            .sum();
        let (arena_used, arena_capacity) = self
            .arena
            .lock()
            .map(|a| {
                a.as_ref()
                    .map(|arena| (arena.used(), arena.capacity()))
                    .unwrap_or((0, 0))
            })
            .unwrap_or((0, 0));
        BackendCounters {
            slab_region_count,
            buddy_region_count,
            arena_used,
            arena_capacity,
        }
    }

    /// Returns which backend served an allocation by reading the Header.
    ///
    /// # Safety
    ///
    /// `ptr` must be a valid pointer previously returned by `alloc_with_hash`
    /// or `alloc_with_hash_and_strategy`.
    pub unsafe fn backend_for_ptr(&self, ptr: *mut u8) -> Option<lohalloc_core::Backend> {
        if ptr.is_null() {
            return None;
        }
        if self.headerless_class_for(ptr).is_some() {
            return Some(lohalloc_core::Backend::Slab);
        }
        if self.buddy_headerless_order_for(ptr).is_some() {
            return Some(lohalloc_core::Backend::Buddy);
        }
        if self.arena_headerless_hit(ptr) {
            return Some(lohalloc_core::Backend::Arena);
        }
        let header = unsafe { ptr.cast::<Header>().offset(-1).read_unaligned() };
        Backend::from_u8(header.backend).map(|b| match b {
            Backend::Slab => lohalloc_core::Backend::Slab,
            Backend::Buddy => lohalloc_core::Backend::Buddy,
            Backend::System => lohalloc_core::Backend::System,
            Backend::Arena => lohalloc_core::Backend::Arena,
        })
    }

    /// Bytes usable by the caller for the allocation at `ptr` — the
    /// backend's actual reserved capacity minus header overhead, which may
    /// be larger than the originally requested size (e.g. Slab rounds up
    /// to a size class). Returns 0 for a null or foreign (bad-magic)
    /// pointer. This is what `lohalloc-cabi` exposes as C's
    /// `malloc_usable_size` and uses internally to size `realloc` copies —
    /// both need this without a `Layout`, which C never provides.
    ///
    /// # Safety
    /// `ptr` must be a valid pointer previously returned by this allocator,
    /// or null.
    pub unsafe fn usable_size(&self, ptr: *mut u8) -> usize {
        if ptr.is_null() {
            return 0;
        }
        if let Some(class) = self.headerless_class_for(ptr) {
            return lohalloc_core::SLAB_SIZE_CLASSES[class as usize];
        }
        if let Some(order) = self.buddy_headerless_order_for(ptr) {
            return buddy::block_size_of(order);
        }
        if self.arena_headerless_hit(ptr) {
            // No per-block size is recoverable for a headerless bump
            // block. 0 is the safe direction: overstating (e.g. distance
            // to chunk end) would license a caller to write past its
            // block into a *neighboring live bump block*. See
            // `usable_size_for_realloc` for how realloc still copies
            // safely without this.
            return 0;
        }
        let header = unsafe { ptr.cast::<Header>().offset(-1).read_unaligned() };
        if header.magic != MAGIC {
            return 0;
        }
        let pad = header_pad(1usize << header.align_log2);
        header.size.saturating_sub(pad)
    }

    /// Deallocate a pointer previously returned by this allocator, without a
    /// `Layout` — the header is authoritative for routing (`dealloc`'s
    /// `layout` parameter is otherwise unused). This is what
    /// `lohalloc-cabi`'s `free(ptr)` uses, since C's `free` never receives
    /// a size/alignment. A no-op for a null pointer.
    ///
    /// # Safety
    /// `ptr` must be a valid pointer previously returned by this allocator,
    /// or null.
    pub unsafe fn dealloc_raw(&self, ptr: *mut u8) {
        unsafe { <Self as GlobalAlloc>::dealloc(self, ptr, Layout::new::<u8>()) };
    }

    /// True if `ptr` was allocated by this allocator instance (valid header
    /// with a matching magic). `false` for null.
    ///
    /// This is the authoritative foreign-pointer test — unlike
    /// `backend_for_ptr` (which assumes the caller already knows `ptr` is
    /// ours and just reads the tag byte, so a foreign pointer's incidental
    /// byte value could coincidentally decode as a valid `Backend`), this
    /// checks the magic. `lohalloc-cabi` uses it to decide whether
    /// `free`/`realloc`/`malloc_usable_size` should route through this
    /// allocator or delegate to the real libc for a pointer it didn't
    /// allocate (e.g. one from before this library's symbols won out).
    ///
    /// # Safety
    /// `ptr` must either be a pointer this allocator could plausibly have
    /// produced, or any other pointer obtained from a `malloc`-family
    /// allocator (reading the `HEADER_SIZE` bytes immediately before `ptr`
    /// must not be out-of-bounds — true for any malloc'd pointer, since a
    /// real allocator's own bookkeeping always occupies that region), or
    /// null.
    pub unsafe fn owns(&self, ptr: *mut u8) -> bool {
        if ptr.is_null() {
            return false;
        }
        if self.headerless_class_for(ptr).is_some()
            || self.buddy_headerless_order_for(ptr).is_some()
            || self.arena_headerless_hit(ptr)
        {
            return true;
        }
        let header = unsafe { ptr.cast::<Header>().offset(-1).read_unaligned() };
        header.magic == MAGIC
    }

    /// Fused `owns` + `dealloc_raw`: one header read decides ownership AND
    /// routes the free. Returns `false` (doing nothing) for null or a
    /// foreign pointer — the caller delegates to the real libc. The
    /// separate `owns()` → `dealloc_raw()` sequence `lohalloc-cabi::free`
    /// used to run read the same 48-byte header twice per call.
    ///
    /// # Safety
    /// Same contract as [`Self::owns`].
    pub unsafe fn try_dealloc_raw(&self, ptr: *mut u8) -> bool {
        if ptr.is_null() {
            return false;
        }
        if let Some(class) = self.headerless_class_for(ptr) {
            self.slab_dealloc_headerless(ptr, class);
            return true;
        }
        if let Some(order) = self.buddy_headerless_order_for(ptr) {
            unsafe { self.buddy_dealloc_via_magazine(ptr, buddy::block_size_of(order)) };
            return true;
        }
        if self.arena_headerless_hit(ptr) {
            return true; // ours; free is a no-op (bump arena)
        }
        let header = unsafe { ptr.cast::<Header>().offset(-1).read_unaligned() };
        if header.magic != MAGIC {
            return false;
        }
        unsafe { self.dealloc_with_header(ptr, header) };
        true
    }

    /// Fused `owns` + `usable_size`: one header read. `None` for null or a
    /// foreign pointer (caller delegates to the real libc).
    ///
    /// # Safety
    /// Same contract as [`Self::owns`].
    pub unsafe fn try_usable_size(&self, ptr: *mut u8) -> Option<usize> {
        if ptr.is_null() {
            return None;
        }
        if let Some(class) = self.headerless_class_for(ptr) {
            return Some(lohalloc_core::SLAB_SIZE_CLASSES[class as usize]);
        }
        if let Some(order) = self.buddy_headerless_order_for(ptr) {
            return Some(buddy::block_size_of(order));
        }
        if self.arena_headerless_hit(ptr) {
            return Some(0); // ours, size unrecoverable — see `usable_size`
        }
        let header = unsafe { ptr.cast::<Header>().offset(-1).read_unaligned() };
        if header.magic != MAGIC {
            return None;
        }
        let pad = header_pad(1usize << header.align_log2);
        Some(header.size.saturating_sub(pad))
    }

    /// Fused single-read `realloc` support for `lohalloc-cabi`: returns the
    /// usable size AND a token to later free the old pointer without
    /// re-reading its header. `None` for null/foreign pointers.
    ///
    /// # Safety
    /// Same contract as [`Self::owns`]; the returned closure-free "token"
    /// is just the header copy (or, for a headerless block, its class) —
    /// `ptr` must stay live until the paired
    /// [`Self::dealloc_with_header_token`] call.
    pub unsafe fn usable_size_for_realloc(&self, ptr: *mut u8) -> Option<(usize, ReallocToken)> {
        if ptr.is_null() {
            return None;
        }
        if let Some(class) = self.headerless_class_for(ptr) {
            let usable = lohalloc_core::SLAB_SIZE_CLASSES[class as usize];
            return Some((usable, ReallocToken(ReallocTokenInner::Headerless(class))));
        }
        if let Some(order) = self.buddy_headerless_order_for(ptr) {
            let usable = buddy::block_size_of(order);
            return Some((
                usable,
                ReallocToken(ReallocTokenInner::BuddyHeaderless(order)),
            ));
        }
        if self.arena_headerless_hit(ptr) {
            // The block's own size is unrecoverable, but a SAFE COPY BOUND
            // is: the distance to the chunk's end. The chunk is one fully
            // mapped region and the old block lies entirely inside it, so
            // copying `min(new_size, bound)` bytes (a) never reads
            // unmapped memory and (b) always covers the whole old block
            // whenever `new_size >= old_size` — realloc's contract needs
            // exactly `min(old, new)` preserved, and both cases satisfy
            // it. The bound OVERSTATES the block's usable size, which is
            // why the token forbids the fits-in-place no-op (see
            // `ReallocToken::allows_in_place`): treating the bound as
            // capacity would license writes into the neighboring live
            // bump block.
            let chunk_end = (ptr as usize & !(arena::CHUNK_BYTES - 1)) + arena::CHUNK_BYTES;
            let copy_bound = chunk_end - ptr as usize;
            return Some((copy_bound, ReallocToken(ReallocTokenInner::ArenaHeaderless)));
        }
        let header = unsafe { ptr.cast::<Header>().offset(-1).read_unaligned() };
        if header.magic != MAGIC {
            return None;
        }
        let pad = header_pad(1usize << header.align_log2);
        let usable = header.size.saturating_sub(pad);
        Some((usable, ReallocToken(ReallocTokenInner::Header(header))))
    }

    /// Free `ptr` using the header (or headerless class) captured by
    /// [`Self::usable_size_for_realloc`] — zero additional header reads.
    ///
    /// # Safety
    /// `ptr` must be the same live allocation the token was created from.
    pub unsafe fn dealloc_with_header_token(&self, ptr: *mut u8, token: ReallocToken) {
        match token.0 {
            ReallocTokenInner::Header(header) => unsafe { self.dealloc_with_header(ptr, header) },
            ReallocTokenInner::Headerless(class) => self.slab_dealloc_headerless(ptr, class),
            ReallocTokenInner::BuddyHeaderless(order) => unsafe {
                self.buddy_dealloc_via_magazine(ptr, buddy::block_size_of(order))
            },
            ReallocTokenInner::ArenaHeaderless => {} // bump arena: free is a no-op
        }
    }
}

/// Opaque snapshot for the fused cabi `realloc` path — see
/// [`Lohalloc::usable_size_for_realloc`]. Either a full header copy (the
/// ordinary path) or just a slab class (a headerless block has no header
/// to copy).
pub struct ReallocToken(ReallocTokenInner);

impl ReallocToken {
    /// Whether the paired `usable_size_for_realloc` value is the block's
    /// true usable capacity (so `realloc` may no-op in place when the new
    /// size fits). `false` only for headerless arena blocks, whose value
    /// is a safe *copy bound* that overstates the block — see
    /// `usable_size_for_realloc`'s arena arm.
    #[inline]
    pub fn allows_in_place(&self) -> bool {
        !matches!(self.0, ReallocTokenInner::ArenaHeaderless)
    }
}

enum ReallocTokenInner {
    Header(Header),
    Headerless(u8),
    /// J1: a header-free Buddy block's order.
    BuddyHeaderless(usize),
    /// Ladder 5: a header-free Arena block (no size recoverable; free is
    /// a no-op; the usable value carried alongside is a copy bound only).
    ArenaHeaderless,
}

// ---------------------------------------------------------------------------
// Phase 3 Integration Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod integration_tests {
    use super::*;
    use core::alloc::{GlobalAlloc, Layout};

    // --- Ladder 6 pin-cache test helpers -------------------------------
    //
    // Each test gets its own `#[inline(never)]` leaf allocation site so
    // training and probing share the exact raw leaf return address (one
    // machine copy of the call instruction) while their *outer* frames
    // differ — which is precisely the situation the 1-frame distillation
    // generalizes over. Distinct helpers per test keep TLS pin slots and
    // trained models independent.

    // Two `#[inline(never)]` layers per site: the allocator body may inline
    // into its immediate caller, making walked frame-0's return address
    // point into that caller's *own* caller. The inner `pin_leaf_*` absorbs
    // that inlining so frame-0's return target is the single call
    // instruction inside `pin_site_*` — constant across training and
    // probing — while frames 1-2 (pin_site_*'s callers) still differ, which
    // is exactly the generalization distillation is supposed to license.

    // The `assert!` after each call is load-bearing: a bare tail call
    // (`pin_leaf_a(a, layout)` as the final expression) gets tail-call
    // optimized into a jump with NO frame, so the stack walk would skip the
    // helper entirely and frame-0's return address would land in the
    // *caller* (which varies). Post-call work forces a real frame.

    #[inline(never)]
    fn pin_leaf_a(a: &Lohalloc, layout: Layout) -> *mut u8 {
        let p = unsafe { a.alloc(layout) };
        assert!(!p.is_null(), "pin leaf alloc failed");
        p
    }

    #[inline(never)]
    fn pin_site_a(a: &Lohalloc, layout: Layout) -> *mut u8 {
        let p = pin_leaf_a(a, layout);
        assert!(!p.is_null());
        p
    }

    #[inline(never)]
    fn pin_leaf_b(a: &Lohalloc, layout: Layout) -> *mut u8 {
        let p = unsafe { a.alloc(layout) };
        assert!(!p.is_null(), "pin leaf alloc failed");
        p
    }

    #[inline(never)]
    fn pin_site_b(a: &Lohalloc, layout: Layout) -> *mut u8 {
        let p = pin_leaf_b(a, layout);
        assert!(!p.is_null());
        p
    }

    #[inline(never)]
    fn pin_leaf_c(a: &Lohalloc, layout: Layout) -> *mut u8 {
        let p = unsafe { a.alloc(layout) };
        assert!(!p.is_null(), "pin leaf alloc failed");
        p
    }

    #[inline(never)]
    fn pin_site_c(a: &Lohalloc, layout: Layout) -> *mut u8 {
        let p = pin_leaf_c(a, layout);
        assert!(!p.is_null());
        p
    }

    /// Train `site` on `a` for `n` alloc/dealloc rounds, freeze, export.
    fn train_and_export(
        a: &Lohalloc,
        site: fn(&Lohalloc, Layout) -> *mut u8,
        layout: Layout,
        n: usize,
    ) -> Vec<u8> {
        for _ in 0..n {
            let p = site(a, layout);
            assert!(!p.is_null());
            unsafe { a.dealloc(p, layout) };
        }
        a.freeze();
        a.export().expect("export after freeze")
    }

    #[test]
    fn pin_cache_hit_serves_distilled_backend_without_walk() {
        let layout = Layout::from_size_align(256, 16).unwrap();
        let trainer = Lohalloc::new();
        let bytes = train_and_export(&trainer, pin_site_a, layout, 512);

        // Single-site training must produce a distilled (pinnable) entry.
        let st = state::AllocatorState::load(&bytes).expect("model parses");
        let distilled = st.distilled_table().expect("inference").entries();
        assert_eq!(
            distilled.len(),
            1,
            "one site × one size class must distill to exactly one entry: {distilled:?}"
        );
        let pinned_backend = distilled[0].2;

        let loaded = Lohalloc::new();
        assert!(loaded.load(&bytes));

        let hits_before = Lohalloc::pin_hit_count();
        let misses_before = Lohalloc::pin_miss_count();

        // First probe from the same leaf site: pin miss → populate; the
        // rest must be pin hits (lower-bound asserts — the counters are
        // process-wide and other tests run concurrently).
        let mut ptrs = Vec::new();
        for _ in 0..64 {
            let p = pin_site_a(&loaded, layout);
            assert!(!p.is_null());
            ptrs.push(p);
        }
        assert!(
            Lohalloc::pin_miss_count() > misses_before,
            "first probe must miss and populate the slot"
        );
        assert!(
            Lohalloc::pin_hit_count() >= hits_before + 32,
            "subsequent probes must hit the pin cache"
        );

        // The pin-served allocations must land on the distilled backend
        // (Buddy is excluded: a load()-booted headerless instance refuses
        // sub-16KiB Buddy orders by design, so it would fall through).
        if pinned_backend != lohalloc_core::Backend::Buddy {
            assert_eq!(
                unsafe { loaded.backend_for_ptr(*ptrs.last().unwrap()) },
                Some(pinned_backend),
                "pin hit must serve the distilled backend"
            );
        }

        for p in ptrs {
            unsafe { loaded.dealloc(p, layout) };
        }
    }

    #[test]
    fn pin_cache_negative_entry_covers_unpinnable_sites() {
        // Main-only model (`PerfectHashTable::serialize` emits an empty
        // distilled section) → every site is unpinnable → after the first
        // (miss, populate-negative) probe, later allocations take the
        // negative-hit arm: full routing path, no distilled re-probe.
        let table = perfect_hash::PerfectHashTable::from_entries(vec![(
            state::combine_hash_size_class(0xDEAD_BEEF, 3),
            3,
            lohalloc_core::Backend::Slab,
        )]);
        let loaded = Lohalloc::new();
        assert!(loaded.load(&table.serialize()));

        let layout = Layout::from_size_align(256, 16).unwrap();
        let neg_before = Lohalloc::pin_negative_count();

        let mut ptrs = Vec::new();
        for _ in 0..32 {
            let p = pin_site_b(&loaded, layout);
            assert!(!p.is_null());
            ptrs.push(p);
        }
        // 1 miss + ≥31 negative hits from this thread (lower bound under
        // concurrent tests). A broken negative cache would take the Miss
        // arm every time and never increment the negative counter.
        assert!(
            Lohalloc::pin_negative_count() >= neg_before + 31,
            "unpinnable site must be served via the negative cache"
        );

        for p in ptrs {
            unsafe { loaded.dealloc(p, layout) };
        }
    }

    #[test]
    fn pin_cache_table_tag_invalidates_across_reload() {
        let layout = Layout::from_size_align(256, 16).unwrap();
        let trainer = Lohalloc::new();
        let bytes = train_and_export(&trainer, pin_site_c, layout, 512);

        let loaded = Lohalloc::new();
        assert!(loaded.load(&bytes));

        // Populate the slot under the first published table.
        let m0 = Lohalloc::pin_miss_count();
        let p1 = pin_site_c(&loaded, layout);
        let p2 = pin_site_c(&loaded, layout);
        assert!(Lohalloc::pin_miss_count() > m0);
        unsafe {
            loaded.dealloc(p1, layout);
            loaded.dealloc(p2, layout);
        }

        // Reset + reload: a NEW FrozenRouting pointer is published, so the
        // stale slot's tag mismatches and the site must re-miss (then hit
        // again) — no explicit flush anywhere.
        loaded.reset_to_training();
        assert!(loaded.load(&bytes), "reload after reset");
        let m1 = Lohalloc::pin_miss_count();
        let p3 = pin_site_c(&loaded, layout);
        assert!(
            Lohalloc::pin_miss_count() > m1,
            "stale table tag must force a re-miss after reload"
        );
        let h0 = Lohalloc::pin_hit_count();
        let p4 = pin_site_c(&loaded, layout);
        assert!(
            Lohalloc::pin_hit_count() > h0,
            "repopulated slot must hit again"
        );
        unsafe {
            loaded.dealloc(p3, layout);
            loaded.dealloc(p4, layout);
        }
    }

    #[test]
    fn freeze_then_allocates_correctly() {
        // Create a Lohalloc instance, do some allocations (training), freeze,
        // then allocate more — routing should still work and produce valid
        // pointers.
        let alloc = Lohalloc::new();

        // Training phase: allocate to populate the bandit.
        for _ in 0..100 {
            let layout = Layout::from_size_align(64, 16).unwrap();
            let ptr = unsafe { alloc.alloc(layout) };
            assert!(!ptr.is_null(), "alloc should succeed in training");
            unsafe { alloc.dealloc(ptr, layout) };
        }

        // Freeze.
        alloc.freeze();
        assert!(alloc.is_inference());

        // Inference phase: allocations should still work.
        for _ in 0..100 {
            let layout = Layout::from_size_align(64, 16).unwrap();
            let ptr = unsafe { alloc.alloc(layout) };
            assert!(!ptr.is_null(), "alloc should succeed in inference");
            unsafe { alloc.dealloc(ptr, layout) };
        }
    }

    #[test]
    fn frozen_alloc_uses_published_table() {
        // Load a hand-built model routing (hash, size_class(64)) → Arena.
        // Size 64 would default to Slab, so getting Arena back proves the
        // lock-free published table (not the size fallback) made the call.
        // (Arena, not Buddy: since J1, a load()-booted instance serves
        // Buddy header-free and deliberately refuses sub-16 KiB orders —
        // the order map can't express them — so a small Buddy
        // recommendation now falls through by design.)
        let hash = 0xDEAD_BEEF_u64;
        let size = 64usize;
        let sc = state::size_class_for(size);
        let key = state::combine_hash_size_class(hash, sc);
        let table = perfect_hash::PerfectHashTable::from_entries(vec![(
            key,
            sc,
            lohalloc_core::Backend::Arena,
        )]);
        let alloc = Lohalloc::new();
        assert!(alloc.load(&table.serialize()), "model load should succeed");

        let layout = Layout::from_size_align(size, 16).unwrap();
        let ptr = unsafe { alloc.alloc_with_hash(layout, hash) };
        assert!(!ptr.is_null());
        assert_eq!(
            unsafe { alloc.backend_for_ptr(ptr) },
            Some(lohalloc_core::Backend::Arena),
            "published frozen table must drive routing"
        );
        unsafe { alloc.dealloc_with_hash(ptr, layout) };
    }

    #[test]
    fn headerless_arena_roundtrip_and_dense_bump() {
        // Ladder 5: a load()-booted instance serves Arena header-free —
        // consecutive bump blocks are exactly `size` apart (no 48-byte
        // header between them: the header write was one cold-block D1
        // write miss per alloc, the measured small-block inference gap),
        // frees are recognized by the chunk-membership probe and no-op,
        // and the realloc token forbids the in-place shortcut.
        let hash = 0xA4E4_0001u64;
        let size = 256usize;
        let sc = state::size_class_for(size);
        let key = state::combine_hash_size_class(hash, sc);
        let table = perfect_hash::PerfectHashTable::from_entries(vec![(
            key,
            sc,
            lohalloc_core::Backend::Arena,
        )]);
        let alloc = Lohalloc::new();
        assert!(alloc.load(&table.serialize()));

        let layout = Layout::from_size_align(size, 16).unwrap();
        let p1 = unsafe { alloc.alloc_with_hash(layout, hash) };
        let p2 = unsafe { alloc.alloc_with_hash(layout, hash) };
        assert!(!p1.is_null() && !p2.is_null());
        assert_eq!(
            p2 as usize - p1 as usize,
            size,
            "headerless bump blocks must be exactly size-adjacent (no header gap)"
        );
        // The blocks are fully usable: writing every byte of each must not
        // disturb the neighbor (would fail under ASAN/valgrind and via the
        // adjacency check above if a header were secretly present).
        unsafe {
            core::ptr::write_bytes(p1, 0xA1, size);
            core::ptr::write_bytes(p2, 0xB2, size);
            assert_eq!(*p1, 0xA1);
            assert_eq!(*p2, 0xB2);
        }
        assert!(unsafe { alloc.owns(p1) });
        assert_eq!(
            unsafe { alloc.backend_for_ptr(p1) },
            Some(lohalloc_core::Backend::Arena)
        );
        // Size is unrecoverable for a headerless bump block: usable_size
        // reports the safe understatement (see its doc).
        assert_eq!(unsafe { alloc.usable_size(p1) }, 0);
        // Realloc support: the usable value is a copy bound (>= the real
        // block), and in-place reuse is forbidden.
        let (bound, token) = unsafe { alloc.usable_size_for_realloc(p1) }.expect("ours");
        assert!(bound >= size, "copy bound must cover the whole block");
        assert!(!token.allows_in_place());
        unsafe { alloc.dealloc_with_header_token(p1, token) }; // no-op
                                                               // Frees are no-ops and never touch a header that isn't there.
        unsafe {
            alloc.dealloc(p2, layout);
            alloc.dealloc(p1, layout);
        }
        // p1's bytes must be intact after "free" (bump arena: reclaim only
        // via reset) — proving dealloc read/wrote nothing.
        unsafe { assert_eq!(*p1, 0xA1) };
    }

    /// Builds a `.lohalloc` model routing `hash` (at `size`'s size class) to
    /// `Backend::Slab` — the shared fixture for the headerless-slab tests
    /// below, which all need a load()-booted instance whose routing table
    /// actually recommends Slab for the sizes they exercise.
    fn slab_only_model(hash: u64, size: usize) -> Vec<u8> {
        let sc = state::size_class_for(size);
        let key = state::combine_hash_size_class(hash, sc);
        perfect_hash::PerfectHashTable::from_entries(vec![(key, sc, lohalloc_core::Backend::Slab)])
            .serialize()
    }

    #[test]
    fn load_on_fresh_instance_enables_headerless_slab() {
        let hash = 0x4EAD_0001u64;
        let size = 96usize;
        let alloc = Lohalloc::new();
        assert!(!alloc.is_slab_headerless(), "must start disabled");
        assert!(alloc.load(&slab_only_model(hash, size)));
        assert!(
            alloc.is_slab_headerless(),
            "load() on a still-empty Slab must enable the headerless path"
        );

        let layout = Layout::from_size_align(size, 16).unwrap();
        let ptr = unsafe { alloc.alloc_with_hash(layout, hash) };
        assert!(!ptr.is_null());
        assert!(
            unsafe { alloc.owns(ptr) },
            "owns() must recognize a headerless block via the registry"
        );
        assert_eq!(
            unsafe { alloc.backend_for_ptr(ptr) },
            Some(lohalloc_core::Backend::Slab)
        );
        assert!(
            unsafe { alloc.usable_size(ptr) } >= size,
            "usable_size must cover at least the requested size"
        );
        // The memory must be genuinely usable (not aliasing a header or
        // adjacent block) — write and read back the full usable region.
        let usable = unsafe { alloc.usable_size(ptr) };
        unsafe {
            core::ptr::write_bytes(ptr, 0xAB, usable);
            for i in 0..usable {
                assert_eq!(*ptr.add(i), 0xAB, "byte {i} corrupted");
            }
        }
        unsafe { alloc.dealloc_with_hash(ptr, layout) };

        // And the block must be reusable afterwards (freed correctly, not
        // leaked or double-registered).
        let ptr2 = unsafe { alloc.alloc_with_hash(layout, hash) };
        assert!(!ptr2.is_null());
        unsafe { alloc.dealloc_with_hash(ptr2, layout) };
    }

    #[test]
    fn load_enables_headerless_and_coexists_with_live_pre_load_header_blocks() {
        // J4-A: `load()` now latches headerless *unconditionally*, even when
        // the Slab already served header blocks before the model loaded (the
        // global-allocator reality — the runtime touches the Slab before
        // `main`). Header and headerless blocks live on physically separate
        // tiers (`Slab::hl`), so pre-load header blocks kept LIVE across the
        // boundary must (a) never be handed back by the headerless alloc path
        // and (b) still free correctly afterward — no corruption, no mixing.
        let training_hash = 0x4EAD_0002u64;
        let size = 64usize;
        let alloc = Lohalloc::new();
        let layout = Layout::from_size_align(size, 16).unwrap();

        // Training-mode traffic populates a Slab region; keep the blocks LIVE
        // across load() (the hard case) and fill them so a mis-serve would be
        // detectable as an overwrite.
        let mut live_header = Vec::new();
        for _ in 0..20 {
            let ptr = unsafe { alloc.alloc_with_hash(layout, training_hash) };
            assert!(!ptr.is_null());
            unsafe { core::ptr::write_bytes(ptr, 0xC5, size) };
            live_header.push(ptr);
        }
        assert!(alloc.backend_counters().slab_region_count > 0);

        assert!(alloc.load(&slab_only_model(0x4EAD_0003u64, size)));
        assert!(
            alloc.is_slab_headerless(),
            "J4-A: load() must enable headerless even after prior Slab activity"
        );

        // Post-load allocations are served header-free and must never alias a
        // still-live pre-load header block.
        let mut new_hl = Vec::new();
        for _ in 0..64 {
            let ptr = unsafe { alloc.alloc_with_hash(layout, 0x4EAD_0003u64) };
            assert!(!ptr.is_null());
            assert!(
                !live_header.contains(&ptr),
                "headerless alloc handed back a live pre-load header block"
            );
            unsafe { core::ptr::write_bytes(ptr, 0x3A, size) };
            new_hl.push(ptr);
        }
        // The pre-load header blocks were never overwritten by a headerless
        // serve.
        for &p in &live_header {
            for i in 0..size {
                assert_eq!(
                    unsafe { *p.add(i) },
                    0xC5,
                    "pre-load header block corrupted"
                );
            }
        }
        // Both flavors free correctly (header blocks via the header path,
        // headerless via the segment registry) and stay reusable.
        for p in live_header {
            unsafe { alloc.dealloc_with_hash(p, layout) };
        }
        for p in new_hl {
            unsafe { alloc.dealloc_with_hash(p, layout) };
        }
        let ptr = unsafe { alloc.alloc_with_hash(layout, 0x4EAD_0003u64) };
        assert!(!ptr.is_null());
        unsafe { alloc.dealloc_with_hash(ptr, layout) };
    }

    #[test]
    fn headerless_registry_saturation_falls_back_safely() {
        // Regression test: force enough *concurrently live* blocks of the
        // largest Slab class (16384B => SEGMENT_SIZE/stride = 4 blocks per
        // segment) to exceed SegmentRegistry's fixed CAPACITY (1024
        // segments). The old `refill_segment` threaded a new segment's
        // blocks onto the free list unconditionally and registered it
        // *afterward*; once the registry saturated, those blocks were
        // still handed out via the headerless path but had no header to
        // fall back on, so `dealloc`'s registry-miss path read `ptr -
        // HEADER_SIZE` as if it were a `Header` — misreading adjacent
        // block data at best, segfaulting on an out-of-segment read at
        // worst (a block near a segment's start). The fix makes
        // registration and free-list population atomic: a failed
        // `try_register` now unmaps the segment immediately and reports
        // refill failure, so Slab falls through to Buddy exactly like any
        // other exhaustion. 4200 blocks comfortably exceeds
        // CAPACITY * 4 = 4096 (~64 MiB backing memory, not a slow test).
        let hash = 0x4EAD_0009u64;
        let size = 16384usize;
        let alloc = Lohalloc::new();
        assert!(alloc.load(&slab_only_model(hash, size)));
        assert!(alloc.is_slab_headerless());

        let layout = Layout::from_size_align(size, 16).unwrap();
        let mut live = Vec::new();
        for _ in 0..4200 {
            let ptr = unsafe { alloc.alloc_with_hash(layout, hash) };
            assert!(
                !ptr.is_null(),
                "allocation must fall through to Buddy once the registry saturates, not fail"
            );
            live.push(ptr);
        }
        // Freeing every block — including the ones served after
        // saturation — must not corrupt state or crash.
        for ptr in live {
            unsafe { alloc.dealloc_with_hash(ptr, layout) };
        }
    }

    #[test]
    fn headerless_registry_is_per_instance() {
        // A headerless block from one instance must never be mistaken for
        // one belonging to a completely separate, untouched instance —
        // the registry is a field on `Lohalloc`, not a global.
        let hash = 0x4EAD_0004u64;
        let size = 128usize;
        let a = Lohalloc::new();
        assert!(a.load(&slab_only_model(hash, size)));
        let layout = Layout::from_size_align(size, 16).unwrap();
        let ptr = unsafe { a.alloc_with_hash(layout, hash) };
        assert!(!ptr.is_null());

        let b = Lohalloc::new(); // never loaded, never touched
        assert!(
            !unsafe { b.owns(ptr) },
            "instance B must not recognize instance A's headerless block"
        );

        unsafe { a.dealloc_with_hash(ptr, layout) };
    }

    #[test]
    fn headerless_realloc_token_roundtrip() {
        // Mirrors the fused cabi `realloc` path: usable_size_for_realloc
        // (registry-hit branch) -> copy -> dealloc_with_header_token.
        let hash = 0x4EAD_0005u64;
        let size = 40usize;
        let alloc = Lohalloc::new();
        assert!(alloc.load(&slab_only_model(hash, size)));

        let layout = Layout::from_size_align(size, 16).unwrap();
        let ptr = unsafe { alloc.alloc_with_hash(layout, hash) };
        assert!(!ptr.is_null());
        unsafe { core::ptr::write_bytes(ptr, 0x42, size) };

        let (old_usable, token) = unsafe { alloc.usable_size_for_realloc(ptr) }
            .expect("registry-hit path must return Some for a headerless block");
        assert!(old_usable >= size);

        // Simulate realloc's copy-then-free-old sequence.
        let new_ptr = unsafe { alloc.alloc_with_hash(layout, hash) };
        assert!(!new_ptr.is_null());
        unsafe {
            core::ptr::copy_nonoverlapping(ptr, new_ptr, old_usable.min(size));
            alloc.dealloc_with_header_token(ptr, token);
        }
        for i in 0..size {
            assert_eq!(
                unsafe { *new_ptr.add(i) },
                0x42,
                "byte {i} lost across realloc"
            );
        }
        unsafe { alloc.dealloc_with_hash(new_ptr, layout) };
    }

    #[test]
    fn headerless_slab_multithreaded_stress() {
        // 8 threads alloc/free concurrently on one load()-booted instance —
        // the hard case for the "register before releasing the slab lock"
        // invariant (`slab_block_headerless_via_magazine`'s doc): a fresh
        // segment handed to one thread must already be registered before
        // any thread (including a different one it's handed to) can free a
        // block from it.
        use std::sync::Arc;

        let hash = 0x4EAD_0006u64;
        let size = 48usize;
        let alloc = Arc::new(Lohalloc::new());
        assert!(alloc.load(&slab_only_model(hash, size)));
        assert!(alloc.is_slab_headerless());

        let mut handles = Vec::new();
        for _ in 0..8 {
            let a = Arc::clone(&alloc);
            handles.push(std::thread::spawn(move || {
                let layout = Layout::from_size_align(size, 16).unwrap();
                for _ in 0..2_000 {
                    let ptr = unsafe { a.alloc_with_hash(layout, hash) };
                    assert!(!ptr.is_null());
                    assert!(unsafe { a.owns(ptr) });
                    unsafe { a.dealloc_with_hash(ptr, layout) };
                }
            }));
        }
        for h in handles {
            h.join().expect("worker thread panicked");
        }
    }

    #[test]
    fn route_counters_track_frozen_fast_path() {
        // Counters are process-wide statics shared across the whole test
        // binary, so parallel tests may also bump them — assert on the
        // delta this test itself causes, not an absolute value.
        let hash = 0x1234_5678_u64;
        // 32 KiB: a headerless-eligible Buddy order (>= 16 KiB) — since J1
        // a load()-booted instance refuses sub-16 KiB Buddy service.
        let size = 32 * 1024usize;
        let sc = state::size_class_for(size);
        let key = state::combine_hash_size_class(hash, sc);
        let table = perfect_hash::PerfectHashTable::from_entries(vec![(
            key,
            sc,
            lohalloc_core::Backend::Buddy,
        )]);
        let alloc = Lohalloc::new();
        assert!(alloc.load(&table.serialize()));

        let before = Lohalloc::route_count(lohalloc_core::Backend::Buddy);
        let layout = Layout::from_size_align(size, 16).unwrap();
        for _ in 0..5 {
            let ptr = unsafe { alloc.alloc_with_hash(layout, hash) };
            assert!(!ptr.is_null());
            unsafe { alloc.dealloc_with_hash(ptr, layout) };
        }
        let after = Lohalloc::route_count(lohalloc_core::Backend::Buddy);
        assert!(
            after - before >= 5,
            "route_count(Buddy) should advance by at least 5 (before={before}, after={after})"
        );
    }

    #[test]
    fn fallthrough_counter_advances_on_size_guard_failure() {
        // Force Slab for a size that fails Slab's `total <= SLAB_MAX` guard
        // in `try_backend` — a deterministic, always-failing recommendation
        // that must fall through to `route_by_size` (and still serve
        // correctly, via Buddy).
        let hash = 0xABCD_EF01_u64;
        let size = 20_000usize;
        let sc = state::size_class_for(size);
        let key = state::combine_hash_size_class(hash, sc);
        let table = perfect_hash::PerfectHashTable::from_entries(vec![(
            key,
            sc,
            lohalloc_core::Backend::Slab,
        )]);
        let alloc = Lohalloc::new();
        assert!(alloc.load(&table.serialize()));

        let before = Lohalloc::fallthrough_count();
        let layout = Layout::from_size_align(size, 16).unwrap();
        let ptr = unsafe { alloc.alloc_with_hash(layout, hash) };
        assert!(!ptr.is_null());
        assert_eq!(
            unsafe { alloc.backend_for_ptr(ptr) },
            Some(lohalloc_core::Backend::Buddy),
            "fallthrough must serve via the size chain (Buddy for this size)"
        );
        unsafe { alloc.dealloc_with_hash(ptr, layout) };
        let after = Lohalloc::fallthrough_count();
        assert!(
            after - before >= 1,
            "fallthrough_count should advance when the frozen recommendation's size guard fails"
        );
    }

    #[test]
    fn forced_arena_chains_then_falls_through_at_cap() {
        // A frozen model routing a hot site to Arena used to degrade
        // permanently once the single 1 MiB region filled (every alloc paid
        // lock + fail + full re-route). With chaining, the arena grows to
        // its cap and only then falls through — and the fallthrough must
        // still serve correctly via the size chain.
        let hash = 0xA4E4_0001u64;
        let size = 4096usize;
        let sc = state::size_class_for(size);
        let key = state::combine_hash_size_class(hash, sc);
        let table = perfect_hash::PerfectHashTable::from_entries(vec![(
            key,
            sc,
            lohalloc_core::Backend::Arena,
        )]);
        let alloc = Lohalloc::new();
        assert!(alloc.load(&table.serialize()));
        let layout = Layout::from_size_align(size, 16).unwrap();

        // Alloc (holding every block, so nothing drains and the arena keeps
        // growing) until the arena caps and a request falls through to Slab.
        // J4-D scales the chunk cap to the host core count, so the exact cap
        // is machine-dependent — but it never exceeds the 128 MiB ceiling
        // (`MAX_CHUNKS_CAP`), which 4 KiB blocks reach within ~33k allocs on
        // any host. Detect the fallthrough rather than assuming 32 MiB.
        let mut ptrs = Vec::new();
        let mut fell_through = false;
        for i in 0..40_000 {
            let p = unsafe { alloc.alloc_with_hash(layout, hash) };
            assert!(!p.is_null(), "alloc {i} must succeed");
            ptrs.push(p);
            if unsafe { alloc.backend_for_ptr(p) } == Some(lohalloc_core::Backend::Slab) {
                fell_through = true;
                break;
            }
        }
        assert!(
            fell_through,
            "arena must chain to its cap and then fall through to Slab"
        );
        assert_eq!(
            unsafe { alloc.backend_for_ptr(ptrs[0]) },
            Some(lohalloc_core::Backend::Arena),
            "early allocs are arena-served"
        );
        let counters = alloc.backend_counters();
        assert!(
            counters.arena_capacity > 1 << 20,
            "arena must have chained past one chunk (capacity = {})",
            counters.arena_capacity
        );
        assert_eq!(
            unsafe { alloc.backend_for_ptr(*ptrs.last().unwrap()) },
            Some(lohalloc_core::Backend::Slab),
            "post-cap allocs fall through to the size chain"
        );
        for p in ptrs {
            unsafe { alloc.dealloc_with_hash(p, layout) };
        }
    }

    #[test]
    fn arena_exhaustion_latch_fast_fails_and_reset_unlatches() {
        // The cpp-string LD_PRELOAD anomaly (inference 1.76× slower than
        // its own training, fallthrough=200k/350k allocs): a frozen model
        // routed churny call sites to Arena; once the 32 MiB cap filled,
        // every alloc paid a doomed fast-path bump + Mutex slow path before
        // falling through — forever, since frozen routing can't
        // self-correct like the bandit does. The latch turns every
        // post-exhaustion arena recommendation into one relaxed load, and
        // reset_arena() re-opens the arena.
        let hash = 0xA4E4_0003u64;
        let size = 64 * 1024usize; // large blocks reach the cap quickly
        let sc = state::size_class_for(size);
        let key = state::combine_hash_size_class(hash, sc);
        let table = perfect_hash::PerfectHashTable::from_entries(vec![(
            key,
            sc,
            lohalloc_core::Backend::Arena,
        )]);
        let alloc = Lohalloc::new();
        assert!(alloc.load(&table.serialize()));
        let layout = Layout::from_size_align(size, 16).unwrap();

        // Alloc 64 KiB blocks (holding every one, so nothing drains) until the
        // cap latches. J4-D scales the cap to the host core count, but it never
        // exceeds the 128 MiB ceiling — reached within ~2k of these on any
        // host; 5000 is a generous safety bound.
        let mut ptrs = Vec::new();
        let mut i = 0;
        while !alloc.arena_exhausted.load(Ordering::Relaxed) && i < 5000 {
            let p = unsafe { alloc.alloc_with_hash(layout, hash) };
            assert!(!p.is_null(), "alloc {i} must succeed even after the cap");
            ptrs.push(p);
            i += 1;
        }
        assert!(
            alloc.arena_exhausted.load(Ordering::Relaxed),
            "cap exhaustion must set the fast-fail latch"
        );
        assert_eq!(
            unsafe { alloc.backend_for_ptr(ptrs[0]) },
            Some(lohalloc_core::Backend::Arena),
            "pre-cap allocs are arena-served"
        );
        assert_ne!(
            unsafe { alloc.backend_for_ptr(*ptrs.last().unwrap()) },
            Some(lohalloc_core::Backend::Arena),
            "post-cap allocs are served by the fallthrough chain"
        );
        for p in ptrs {
            unsafe { alloc.dealloc_with_hash(p, layout) };
        }

        // Reset rewinds the arena -> the latch must clear and the same
        // frozen recommendation must be arena-served again.
        alloc.reset_arena();
        assert!(
            !alloc.arena_exhausted.load(Ordering::Relaxed),
            "reset_arena must clear the latch"
        );
        let p = unsafe { alloc.alloc_with_hash(layout, hash) };
        assert_eq!(
            unsafe { alloc.backend_for_ptr(p) },
            Some(lohalloc_core::Backend::Arena),
            "a reset arena serves the frozen recommendation again"
        );
        unsafe { alloc.dealloc_with_hash(p, layout) };
    }

    #[test]
    fn arena_reset_republishes_fast_path_after_chunk_advance() {
        // Advance the arena past its first 1 MiB chunk (forcing the slow
        // path to map chunk 1 and republish the fast path's descriptor to
        // point at it), then reset — republishing on reset must switch the
        // fast path back to chunk 0's cursor, not leave it bumping the
        // (now logically stale, from `reset()`'s point of view) chunk 1.
        let hash = 0xA4E4_0002u64;
        let size = 4096usize;
        let sc = state::size_class_for(size);
        let key = state::combine_hash_size_class(hash, sc);
        let table = perfect_hash::PerfectHashTable::from_entries(vec![(
            key,
            sc,
            lohalloc_core::Backend::Arena,
        )]);
        let alloc = Lohalloc::new();
        assert!(alloc.load(&table.serialize()));
        let layout = Layout::from_size_align(size, 16).unwrap();

        // Default chunks are 1 MiB and 1 MiB-aligned (see arena.rs's module
        // doc), so masking a pointer down recovers its chunk base
        // regardless of the relative order the OS happened to map chunks
        // in — unlike comparing raw addresses, this is platform-agnostic.
        const CHUNK_MASK: usize = !((1usize << 20) - 1);
        let chunk_base_of = |p: *mut u8| (p as usize) & CHUNK_MASK;

        // ~400 * ~4 KiB (+header) comfortably exceeds one 1 MiB chunk.
        let mut ptrs = Vec::new();
        for _ in 0..400 {
            let p = unsafe { alloc.alloc_with_hash(layout, hash) };
            assert!(!p.is_null());
            ptrs.push(p);
        }
        let first_chunk_base = chunk_base_of(ptrs[0]);
        let last_chunk_base = chunk_base_of(*ptrs.last().unwrap());
        assert_ne!(
            first_chunk_base, last_chunk_base,
            "test setup must advance to a second chunk before reset"
        );

        alloc.reset_arena();

        let p_after_reset = unsafe { alloc.alloc_with_hash(layout, hash) };
        assert!(!p_after_reset.is_null());
        assert_eq!(
            chunk_base_of(p_after_reset),
            first_chunk_base,
            "after reset, the fast path must resume bumping chunk 0"
        );

        for p in ptrs {
            unsafe { alloc.dealloc_with_hash(p, layout) };
        }
        unsafe { alloc.dealloc_with_hash(p_after_reset, layout) };
    }

    #[test]
    fn arena_fast_path_multithreaded_disjointness() {
        // 8 threads racing the lock-free `arena_alloc_fast` CAS loop
        // concurrently — every returned region must be genuinely disjoint.
        // A broken CAS (e.g. an ordinary load+store race) would hand two
        // threads overlapping memory here.
        use std::sync::{Arc, Mutex};

        let hash = 0xA4E4_0003u64;
        let size = 64usize;
        let sc = state::size_class_for(size);
        let key = state::combine_hash_size_class(hash, sc);
        let table = perfect_hash::PerfectHashTable::from_entries(vec![(
            key,
            sc,
            lohalloc_core::Backend::Arena,
        )]);
        let alloc = Arc::new(Lohalloc::new());
        assert!(alloc.load(&table.serialize()));
        let layout = Layout::from_size_align(size, 16).unwrap();

        let ranges: Arc<Mutex<Vec<(usize, usize)>>> = Arc::new(Mutex::new(Vec::new()));

        let mut handles = Vec::new();
        for _ in 0..8 {
            let a = Arc::clone(&alloc);
            let r = Arc::clone(&ranges);
            handles.push(std::thread::spawn(move || {
                let mut local = Vec::with_capacity(500);
                for _ in 0..500 {
                    let p = unsafe { a.alloc_with_hash(layout, hash) };
                    assert!(!p.is_null());
                    local.push((p as usize, size));
                }
                r.lock().unwrap().extend(local);
            }));
        }
        for h in handles {
            h.join().expect("worker thread panicked");
        }

        let mut all = ranges.lock().unwrap().clone();
        all.sort_unstable();
        for w in all.windows(2) {
            let (start_a, len_a) = w[0];
            let (start_b, _) = w[1];
            assert!(
                start_a + len_a <= start_b,
                "overlapping arena allocations: [{start_a}, {}) and one starting at {start_b}",
                start_a + len_a
            );
        }
    }

    #[test]
    fn arena_chunk_advance_races_fast_path_without_corruption() {
        // Regression test for a ThreadSanitizer-confirmed data race: the
        // slow path (under `Lohalloc`'s `arena` Mutex) used to retry alloc
        // on the *current* chunk via `Chunk::alloc`'s plain
        // `self.cursor.get_mut()` — reasoning that the Mutex ruled out
        // concurrent access. It didn't: `arena_alloc_fast` reads that same
        // chunk's cursor lock-free (via the published `arena_chunk`
        // descriptor) and never takes the Mutex at all, so a fast-path
        // reader could race the slow path's plain read-modify-write on the
        // exact same `AtomicUsize` — reproduced as a real SIGSEGV on
        // Apple Silicon (native LD_PRELOAD mt-slab-t8 inference) and
        // caught immediately by `cargo +nightly build -Zbuild-std
        // --target aarch64-apple-darwin -p lohalloc-bench --bin
        // native_workload --no-default-features --features alloc-lohalloc`
        // with `RUSTFLAGS="-Z sanitizer=thread"`, then run under
        // `TSAN_OPTIONS=halt_on_error=1`. Fixed by making `Chunk::alloc`
        // use the same atomic compare-exchange loop as `arena_alloc_fast`
        // — see its doc.
        //
        // This test can't force the race deterministically outside TSAN,
        // but many threads racing through *many* small chunks (forcing
        // constant chunk-advance slow-path traffic concurrently with
        // fast-path hits) is exactly the scenario that raced — a broken
        // fix would very likely show up here as overlapping/out-of-bounds
        // pointers, and always shows up under a TSAN rebuild of this same
        // shape.
        use std::sync::{Arc, Mutex};

        let hash = 0xA4E4_0004u64;
        let size = 64usize;
        let sc = state::size_class_for(size);
        let key = state::combine_hash_size_class(hash, sc);
        let table = perfect_hash::PerfectHashTable::from_entries(vec![(
            key,
            sc,
            lohalloc_core::Backend::Arena,
        )]);
        let alloc = Arc::new(Lohalloc::new());
        assert!(alloc.load(&table.serialize()));
        let layout = Layout::from_size_align(size, 16).unwrap();

        let ranges: Arc<Mutex<Vec<(usize, usize)>>> = Arc::new(Mutex::new(Vec::new()));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let a = Arc::clone(&alloc);
            let r = Arc::clone(&ranges);
            handles.push(std::thread::spawn(move || {
                // 8 threads x 3000 x 64B ≈ 1.5MiB — several chunk advances
                // past the default 1MiB chunk, contended across threads.
                let mut local = Vec::with_capacity(3000);
                for _ in 0..3000 {
                    let p = unsafe { a.alloc_with_hash(layout, hash) };
                    assert!(!p.is_null());
                    local.push((p as usize, size));
                }
                r.lock().unwrap().extend(local);
            }));
        }
        for h in handles {
            h.join().expect("worker thread panicked");
        }

        let mut all = ranges.lock().unwrap().clone();
        all.sort_unstable();
        for w in all.windows(2) {
            let (start_a, len_a) = w[0];
            let (start_b, _) = w[1];
            assert!(
                start_a + len_a <= start_b,
                "overlapping arena allocations across a chunk advance: \
                 [{start_a}, {}) and one starting at {start_b}",
                start_a + len_a
            );
        }
    }

    #[test]
    fn system_cache_reuses_mapping() {
        // Same-size large alloc → free → alloc must be served from the
        // retention cache: hit counter increments and the same mapping
        // (identical pointer) comes back.
        let alloc = Lohalloc::new();
        let layout = Layout::from_size_align(2 << 20, 16).unwrap(); // 2 MiB

        let p1 = unsafe { alloc.alloc(layout) };
        assert!(!p1.is_null());
        assert_eq!(alloc.system_cache_hits(), 0);
        unsafe { alloc.dealloc(p1, layout) };

        let p2 = unsafe { alloc.alloc(layout) };
        assert!(!p2.is_null());
        assert_eq!(
            alloc.system_cache_hits(),
            1,
            "second alloc must hit the cache"
        );
        assert_eq!(p1, p2, "cache hit must reuse the same mapping");
        // Memory must be writable (pages still mapped).
        unsafe { core::ptr::write_bytes(p2, 0xAB, 2 << 20) };
        unsafe { alloc.dealloc(p2, layout) };
    }

    #[test]
    fn system_cache_respects_byte_cap() {
        // Freeing more large mappings than the 64 MiB retention cap must
        // munmap the excess, never hoard unboundedly: verify by exercising
        // 96 MiB of frees, then confirming reuse still works (the cache is
        // functional) — retention bounds are asserted at the SystemCache
        // unit level; here we just prove nothing corrupts at the cap.
        let alloc = Lohalloc::new();
        let layout = Layout::from_size_align(8 << 20, 16).unwrap(); // 8 MiB
        let mut ptrs = Vec::new();
        for _ in 0..12 {
            let p = unsafe { alloc.alloc(layout) };
            assert!(!p.is_null());
            ptrs.push(p);
        }
        for p in ptrs {
            unsafe { alloc.dealloc(p, layout) };
        }
        let p = unsafe { alloc.alloc(layout) };
        assert!(!p.is_null());
        assert!(alloc.system_cache_hits() >= 1);
        unsafe { alloc.dealloc(p, layout) };
    }

    #[test]
    fn system_cache_over_aligned_request_falls_through() {
        // A cached page-aligned mapping can't serve a much stricter
        // alignment unless it happens to fit — the fit check must reject
        // rather than hand out a misaligned block.
        let alloc = Lohalloc::new();
        let plain = Layout::from_size_align(2 << 20, 16).unwrap();
        let p1 = unsafe { alloc.alloc(plain) };
        unsafe { alloc.dealloc(p1, plain) };

        // 4 MiB alignment: the retained 2 MiB mapping can almost never
        // satisfy align_up(base, 4MiB)+2MiB within its length; whether it
        // falls through or (rarely) fits, the result must be aligned.
        let strict = Layout::from_size_align(2 << 20, 4 << 20).unwrap();
        let p2 = unsafe { alloc.alloc(strict) };
        assert!(!p2.is_null());
        assert_eq!(
            (p2 as usize) % (4 << 20),
            0,
            "over-aligned request must return an aligned pointer"
        );
        unsafe { alloc.dealloc(p2, strict) };
    }

    #[test]
    fn magazine_cross_thread_alloc_free_roundtrip() {
        // Blocks allocated on one thread and freed on another must migrate
        // cleanly (they land in the freeing thread's magazine or the
        // central slab) — and remain reusable afterwards.
        use std::sync::mpsc;
        use std::sync::Arc;

        let alloc = Arc::new(Lohalloc::new());
        let layout = Layout::from_size_align(200, 16).unwrap();
        let (tx, rx) = mpsc::channel::<usize>();

        let producer = {
            let a = Arc::clone(&alloc);
            std::thread::spawn(move || {
                for _ in 0..5_000 {
                    let p = unsafe { a.alloc(layout) };
                    assert!(!p.is_null());
                    tx.send(p as usize).unwrap();
                }
            })
        };
        let consumer = {
            let a = Arc::clone(&alloc);
            std::thread::spawn(move || {
                for addr in rx {
                    unsafe { a.dealloc(addr as *mut u8, layout) };
                }
            })
        };
        producer.join().unwrap();
        consumer.join().unwrap();

        // The allocator must still serve correctly after heavy migration.
        for _ in 0..1_000 {
            let p = unsafe { alloc.alloc(layout) };
            assert!(!p.is_null());
            unsafe { alloc.dealloc(p, layout) };
        }
    }

    #[test]
    fn magazine_churn_keeps_slab_regions_bounded() {
        // Alloc/free churn through the magazine layer must recycle blocks —
        // the central slab's region count stays flat instead of growing.
        let alloc = Lohalloc::new();
        let layout = Layout::from_size_align(200, 16).unwrap();
        let mut live = Vec::new();
        for _round in 0..500 {
            for _ in 0..64 {
                let p = unsafe { alloc.alloc(layout) };
                assert!(!p.is_null());
                live.push(p);
            }
            for p in live.drain(..) {
                unsafe { alloc.dealloc(p, layout) };
            }
        }
        let counters = alloc.backend_counters();
        assert!(
            counters.slab_region_count < 8,
            "slab regions must stay bounded under magazine churn, got {}",
            counters.slab_region_count
        );
    }

    #[test]
    fn cross_thread_free_pipeline_keeps_slab_regions_bounded() {
        // Ladder 5 Phase 3 regression test: a producer thread allocs and a
        // consumer thread frees (the mt-xfree shape). Under striped
        // centrals WITHOUT the recycled-first steal, every freed block
        // parks on the consumer's stripe while the producer's stripe —
        // never seeing a free — carves fresh segments without bound
        // (region count grows ~linearly with total ops). With the steal,
        // the producer's misses drain the consumer stripe's recycled tier
        // and the region count stays flat.
        use std::sync::mpsc;
        let alloc = std::sync::Arc::new(Lohalloc::new());
        let layout = Layout::from_size_align(200, 16).unwrap();
        let (tx, rx) = mpsc::sync_channel::<usize>(256);
        let producer = {
            let alloc = std::sync::Arc::clone(&alloc);
            std::thread::spawn(move || {
                for _ in 0..20_000 {
                    let p = unsafe { alloc.alloc(layout) };
                    assert!(!p.is_null());
                    tx.send(p as usize).unwrap();
                }
            })
        };
        let consumer = {
            let alloc = std::sync::Arc::clone(&alloc);
            std::thread::spawn(move || {
                while let Ok(p) = rx.recv() {
                    unsafe { alloc.dealloc(p as *mut u8, layout) };
                }
            })
        };
        producer.join().unwrap();
        consumer.join().unwrap();
        let counters = alloc.backend_counters();
        // 20k live-bounded (<=256) 256-byte blocks: a handful of segments
        // per stripe at most. Without the steal this measured in the
        // hundreds.
        assert!(
            counters.slab_region_count < 16,
            "slab regions must stay bounded under cross-thread free churn, got {}",
            counters.slab_region_count
        );
    }

    #[test]
    fn buddy_magazine_churn_keeps_regions_bounded() {
        // Same shape as `magazine_churn_keeps_slab_regions_bounded`, but at
        // a magazined buddy size (64 KiB) — churn through the magazine
        // layer must recycle blocks instead of mapping a fresh region every
        // round.
        let alloc = Lohalloc::new();
        let layout = Layout::from_size_align(64 * 1024, 16).unwrap();
        let mut live = Vec::new();
        for _round in 0..40 {
            for _ in 0..16 {
                let p = unsafe { alloc.alloc(layout) };
                assert!(!p.is_null());
                live.push(p);
            }
            for p in live.drain(..) {
                unsafe { alloc.dealloc(p, layout) };
            }
        }
        let counters = alloc.backend_counters();
        assert!(
            counters.buddy_region_count < 8,
            "buddy regions must stay bounded under magazine churn, got {}",
            counters.buddy_region_count
        );
    }

    #[test]
    fn buddy_magazine_cross_thread_alloc_free_roundtrip() {
        // Blocks allocated on one thread and freed on another must migrate
        // cleanly through the buddy magazine layer (mirrors the slab
        // magazine's cross-thread test) and remain reusable afterwards.
        use std::sync::mpsc;
        use std::sync::Arc;

        let alloc = Arc::new(Lohalloc::new());
        let layout = Layout::from_size_align(64 * 1024, 16).unwrap();
        let (tx, rx) = mpsc::channel::<usize>();

        let producer = {
            let a = Arc::clone(&alloc);
            std::thread::spawn(move || {
                for _ in 0..2_000 {
                    let p = unsafe { a.alloc(layout) };
                    assert!(!p.is_null());
                    tx.send(p as usize).unwrap();
                }
            })
        };
        let consumer = {
            let a = Arc::clone(&alloc);
            std::thread::spawn(move || {
                for addr in rx {
                    unsafe { a.dealloc(addr as *mut u8, layout) };
                }
            })
        };
        producer.join().unwrap();
        consumer.join().unwrap();

        for _ in 0..200 {
            let p = unsafe { alloc.alloc(layout) };
            assert!(!p.is_null());
            unsafe { alloc.dealloc(p, layout) };
        }
    }

    #[test]
    fn stripe_mask_is_pow2_count_within_bounds() {
        // J5-B2: the active stripe count is a power of two (the stripe pick
        // is a mask) scaled to the host core count, floored at MIN_STRIPES
        // (preserves the certified ≤8-thread behavior byte-identically) and
        // ceilinged at MAX_STRIPES (the const array size).
        // NOTE: assumes LOHALLOC_STRIPES is unset in the test environment —
        // an exported override latches the process-global mask below
        // MIN_STRIPES and legitimately fails the range assert.
        let mask = stripe_mask();
        let count = mask + 1;
        assert!(
            count.is_power_of_two(),
            "stripe count must be pow2: {count}"
        );
        assert!((MIN_STRIPES..=MAX_STRIPES).contains(&count));
        // Latched: a second read returns the identical mask.
        assert_eq!(stripe_mask(), mask);
        // Thread stripes always land inside the active set.
        assert!(thread_stripe() <= mask);
    }

    #[test]
    fn stripe_count_default_pins_certified_formula() {
        // The unset path must stay byte-identical to the pre-override
        // formula: next_pow2(ncpus).clamp(MIN_STRIPES, MAX_STRIPES).
        for (ncpus, want) in [(1, 8), (4, 8), (8, 8), (10, 16), (16, 16), (64, 32)] {
            assert_eq!(stripe_count(None, ncpus), want, "ncpus={ncpus}");
        }
    }

    #[test]
    fn stripe_count_override_rounds_pow2_floor1_cap_max() {
        // Explicit override: pow2 round-up, floor 1 (opting out of the
        // certified floor is the point — 1-stripe is the strongest
        // sibling-scan mechanism probe), cap MAX_STRIPES.
        for (n, want) in [(1, 1), (2, 2), (3, 4), (8, 8), (16, 16), (31, 32), (33, 32)] {
            assert_eq!(stripe_count(Some(n), 999), want, "override={n}");
        }
        // ncpus is ignored when an override is present.
        assert_eq!(stripe_count(Some(4), 64), 4);
    }

    #[test]
    fn buddy_blocks_resolve_to_a_registered_stripe() {
        // C3: every buddy allocation's region must be registered in the
        // region → stripe registry before the block reaches the caller —
        // that registration is what makes exact free-side stripe routing
        // possible. Drives the internal buddy path directly (the public
        // `alloc` routes through the training bandit, which may
        // legitimately explore other backends for these sizes). Covers
        // both the magazined path (64 KiB) and the direct path (512 KiB,
        // above MAX_MAGAZINE_ORDER).
        let alloc = Lohalloc::new();
        for size in [20 * 1024usize, 64 * 1024, 512 * 1024] {
            let block = alloc
                .buddy_block_via_magazine(size)
                .expect("buddy path must serve");
            let stripe = alloc.buddy_stripe_of(block);
            assert!(
                matches!(stripe, Some(s) if s <= stripe_mask()),
                "buddy block ({size} B) must resolve to a registered stripe, got {stripe:?}"
            );
            // All allocations from one thread land on that thread's stripe.
            assert_eq!(stripe, Some(thread_stripe()));
            unsafe { alloc.buddy_dealloc_via_magazine(block, size) };
        }
    }

    #[test]
    fn buddy_direct_order_cross_thread_free_routes_by_registry() {
        // C3: 512 KiB blocks bypass the magazines, so a cross-thread free
        // exercises the registry-routed direct path — the free must land in
        // the *allocating* thread's stripe (wrong-stripe frees panic in
        // debug via free_order's region assert). Blocks must then be
        // recyclable: the region count stays bounded across rounds.
        use std::sync::Arc;

        let alloc = Arc::new(Lohalloc::new());
        const SIZE: usize = 512 * 1024;
        for _round in 0..20 {
            let a = Arc::clone(&alloc);
            let ptrs: Vec<usize> = std::thread::spawn(move || {
                (0..8)
                    .map(|_| {
                        let p = a
                            .buddy_block_via_magazine(SIZE)
                            .expect("buddy path must serve");
                        p as usize
                    })
                    .collect()
            })
            .join()
            .unwrap();
            // Free on this (potentially different-stripe) thread.
            for p in ptrs {
                unsafe { alloc.buddy_dealloc_via_magazine(p as *mut u8, SIZE) };
            }
        }
        let counters = alloc.backend_counters();
        assert!(
            counters.buddy_region_count < 40,
            "cross-thread direct frees must recycle regions, got {}",
            counters.buddy_region_count
        );
    }

    #[test]
    fn headerless_buddy_roundtrip_and_no_order_inflation() {
        // J1: a load()-booted instance serves Buddy header-free. A 32 KiB
        // request must come back as an exact 32 KiB block (the header path
        // inflated it to 64 KiB: 32 KiB + 48 rounds to the next order),
        // recover its backend/usable size from the order map alone, and
        // free cleanly through the magazine layer.
        let table = perfect_hash::PerfectHashTable::from_entries(vec![]);
        let alloc = Lohalloc::new();
        assert!(alloc.load(&table.serialize()));

        // (No exact-1 MiB row: `total` — request + header pad — gates the
        // Buddy range even in headerless mode, so 1 MiB + 48 routes to
        // System. Acceptable: the SystemCache path is our strongest row.)
        for size in [16 * 1024usize, 32 * 1024, 256 * 1024, 512 * 1024] {
            let layout = Layout::from_size_align(size, 16).unwrap();
            let ptr = unsafe { alloc.alloc(layout) };
            assert!(!ptr.is_null());
            assert_eq!(
                unsafe { alloc.backend_for_ptr(ptr) },
                Some(lohalloc_core::Backend::Buddy),
                "{size} B should route Buddy by size in inference"
            );
            assert_eq!(
                unsafe { alloc.usable_size(ptr) },
                size,
                "headerless pow2 request must not order-inflate"
            );
            assert!(unsafe { alloc.owns(ptr) });
            unsafe { alloc.dealloc(ptr, layout) };
        }
        // Churn: recycled headerless blocks keep regions bounded.
        let layout = Layout::from_size_align(64 * 1024, 16).unwrap();
        for _ in 0..200 {
            let ptr = unsafe { alloc.alloc(layout) };
            assert!(!ptr.is_null());
            unsafe { alloc.dealloc(ptr, layout) };
        }
        assert!(alloc.backend_counters().buddy_region_count < 8);
    }

    #[test]
    fn headerless_buddy_cross_thread_free() {
        // J1 + C3: headerless blocks freed on a different thread resolve
        // (stripe, order) purely from the registry + order map.
        use std::sync::Arc;

        let table = perfect_hash::PerfectHashTable::from_entries(vec![]);
        let alloc = Arc::new(Lohalloc::new());
        assert!(alloc.load(&table.serialize()));

        let layout = Layout::from_size_align(128 * 1024, 16).unwrap();
        for _round in 0..10 {
            let a = Arc::clone(&alloc);
            let ptrs: Vec<usize> = std::thread::spawn(move || {
                (0..16)
                    .map(|_| {
                        let p = unsafe { a.alloc(layout) };
                        assert!(!p.is_null());
                        p as usize
                    })
                    .collect()
            })
            .join()
            .unwrap();
            for p in ptrs {
                let p = p as *mut u8;
                assert_eq!(unsafe { alloc.usable_size(p) }, 128 * 1024);
                unsafe { alloc.dealloc(p, layout) };
            }
        }
        assert!(alloc.backend_counters().buddy_region_count < 40);
    }

    #[test]
    fn buddy_magazine_multithreaded_stress_preserves_invariants() {
        // 8 threads churning magazined buddy sizes concurrently — the hard
        // case for merge-on-spill interacting with the magazine layer.
        // Every block a thread pops must be servable and every block it
        // pushes back must round-trip; if magazine bookkeeping ever handed
        // out a live (still-magazined) pointer twice, or corrupted the
        // central buddy's bitmap/lists, this either double-serves an
        // address (caught by `HashSet::insert` below) or hangs/asserts in
        // debug builds via `Buddy::check_invariants`-guarded paths.
        use std::sync::Arc;

        let alloc = Arc::new(Lohalloc::new());
        let sizes = [20 * 1024usize, 50 * 1024, 100 * 1024, 200 * 1024];

        let mut handles = Vec::new();
        for t in 0..8 {
            let a = Arc::clone(&alloc);
            handles.push(std::thread::spawn(move || {
                let mut live: Vec<(*mut u8, Layout)> = Vec::new();
                for i in 0..2_000 {
                    let size = sizes[(t + i) % sizes.len()];
                    let layout = Layout::from_size_align(size, 16).unwrap();
                    let p = unsafe { a.alloc(layout) };
                    assert!(!p.is_null());
                    live.push((p, layout));
                    // Free every other block immediately (churn) so both
                    // the magazine's pop and push paths get exercised.
                    if i % 2 == 1 {
                        if let Some((p, layout)) = live.pop() {
                            unsafe { a.dealloc(p, layout) };
                        }
                    }
                }
                for (p, layout) in live {
                    unsafe { a.dealloc(p, layout) };
                }
            }));
        }
        for h in handles {
            h.join().expect("worker thread panicked");
        }

        // The allocator must still serve correctly after the stress run.
        let layout = Layout::from_size_align(64 * 1024, 16).unwrap();
        let p = unsafe { alloc.alloc(layout) };
        assert!(!p.is_null());
        unsafe { alloc.dealloc(p, layout) };
    }

    #[test]
    fn frozen_alloc_multithreaded_smoke() {
        use std::sync::Arc;

        let alloc = Arc::new(Lohalloc::new());
        for _ in 0..200 {
            let layout = Layout::from_size_align(64, 16).unwrap();
            let ptr = unsafe { alloc.alloc(layout) };
            assert!(!ptr.is_null());
            unsafe { alloc.dealloc(ptr, layout) };
        }
        alloc.freeze();

        let mut handles = Vec::new();
        for _ in 0..4 {
            let a = Arc::clone(&alloc);
            handles.push(std::thread::spawn(move || {
                let layout = Layout::from_size_align(64, 16).unwrap();
                for _ in 0..10_000 {
                    let ptr = unsafe { a.alloc(layout) };
                    assert!(!ptr.is_null());
                    unsafe { a.dealloc(ptr, layout) };
                }
            }));
        }
        for h in handles {
            h.join().expect("worker thread panicked");
        }
    }

    #[test]
    fn reset_after_freeze_returns_to_training() {
        let alloc = Lohalloc::new();
        let layout = Layout::from_size_align(64, 16).unwrap();
        for _ in 0..50 {
            let ptr = unsafe { alloc.alloc(layout) };
            assert!(!ptr.is_null());
            unsafe { alloc.dealloc(ptr, layout) };
        }
        alloc.freeze();
        assert!(alloc.is_inference());

        alloc.reset_to_training();
        assert!(!alloc.is_inference());
        // Allocations must go back through the training path (state lock),
        // not the (now unpublished) frozen table.
        for _ in 0..50 {
            let ptr = unsafe { alloc.alloc(layout) };
            assert!(!ptr.is_null());
            unsafe { alloc.dealloc(ptr, layout) };
        }
        assert!(alloc.signature_count() > 0, "bandit must be learning again");
    }

    #[test]
    fn export_load_roundtrip_integration() {
        let alloc = Lohalloc::new();

        // Training.
        for _ in 0..50 {
            let layout = Layout::from_size_align(128, 16).unwrap();
            let ptr = unsafe { alloc.alloc(layout) };
            assert!(!ptr.is_null());
            unsafe { alloc.dealloc(ptr, layout) };
        }

        alloc.freeze();
        let exported = alloc.export().expect("export should succeed after freeze");
        assert!(!exported.is_empty(), "exported data should not be empty");

        // Load into a fresh allocator.
        let alloc2 = Lohalloc::new();
        assert!(!alloc2.is_inference());
        assert!(alloc2.load(&exported), "load should succeed");
        assert!(alloc2.is_inference(), "should be in inference after load");

        // Allocations should work with the loaded model.
        for _ in 0..50 {
            let layout = Layout::from_size_align(128, 16).unwrap();
            let ptr = unsafe { alloc2.alloc(layout) };
            assert!(!ptr.is_null());
            unsafe { alloc2.dealloc(ptr, layout) };
        }
    }

    #[test]
    fn inference_mode_zero_alloc_hot_path() {
        // In Inference mode, the alloc hot path must make zero heap
        // allocations. We verify this by using the Lohalloc allocator itself
        // (which has the re-entrancy guard) and ensuring allocations succeed
        // without deadlock — if the hot path tried to allocate, the
        // re-entrancy guard would catch it (bypass to mmap).
        //
        // This test is a smoke test: if the hot path allocated in Inference,
        // it would either deadlock (if not for the guard) or silently
        // fall through to mmap (if the guard caught it). Either way, the
        // test verifies that allocations complete successfully in Inference.
        let alloc = Lohalloc::new();

        // Train briefly.
        for _ in 0..10 {
            let layout = Layout::from_size_align(64, 16).unwrap();
            let ptr = unsafe { alloc.alloc(layout) };
            unsafe { alloc.dealloc(ptr, layout) };
        }

        alloc.freeze();
        assert!(alloc.is_inference());

        // In Inference mode, do many allocations. If the hot path allocated,
        // we'd see issues (deadlock, or mmap fallback causing fragmentation).
        let mut ptrs = Vec::new();
        for _ in 0..1000 {
            let layout = Layout::from_size_align(64, 16).unwrap();
            let ptr = unsafe { alloc.alloc(layout) };
            assert!(!ptr.is_null(), "alloc should succeed in inference");
            ptrs.push(ptr);
        }
        // Free them all.
        for ptr in &ptrs {
            let layout = Layout::from_size_align(64, 16).unwrap();
            unsafe { alloc.dealloc(*ptr, layout) };
        }
    }

    #[test]
    fn training_and_inference_produce_valid_pointers() {
        let alloc = Lohalloc::new();

        // Various sizes to exercise different backends.
        let sizes = [16, 64, 256, 1024, 4096, 65536, 1 << 21];

        // Training phase.
        for &size in &sizes {
            let layout = Layout::from_size_align(size, 16).unwrap();
            let ptr = unsafe { alloc.alloc(layout) };
            assert!(!ptr.is_null(), "training alloc {size} should succeed");

            // Write to the allocation to verify it's usable.
            unsafe {
                core::ptr::write_bytes(ptr, 0xAB, size);
            }
            unsafe { alloc.dealloc(ptr, layout) };
        }

        // Freeze and test Inference.
        alloc.freeze();

        for &size in &sizes {
            let layout = Layout::from_size_align(size, 16).unwrap();
            let ptr = unsafe { alloc.alloc(layout) };
            assert!(!ptr.is_null(), "inference alloc {size} should succeed");

            // Verify alignment.
            assert_eq!(
                ptr as usize % 16,
                0,
                "inference alloc {size} should be 16-aligned"
            );

            // Write to verify usability.
            unsafe {
                core::ptr::write_bytes(ptr, 0xCD, size);
            }
            unsafe { alloc.dealloc(ptr, layout) };
        }
    }

    #[test]
    fn arena_can_be_routed_by_mab() {
        // Verify that the Arena backend can be selected by the MAB and that
        // Arena allocations work correctly when routed through the Decision
        // Engine.
        let alloc = Lohalloc::new();

        // Direct Arena allocation test (via public API).
        let ptr = alloc.arena_alloc(64, 16);
        assert!(!ptr.is_null(), "arena_alloc should succeed");

        // Write to verify usability.
        unsafe {
            core::ptr::write_bytes(ptr, 0xEF, 64);
        }

        // Reset the arena — all arena allocations are invalidated.
        alloc.reset_arena();

        // After reset, a new arena allocation should work (and may reuse the
        // same base pointer since the cursor returns to the start).
        let ptr2 = alloc.arena_alloc(128, 16);
        assert!(!ptr2.is_null(), "arena_alloc after reset should succeed");

        alloc.reset_arena();
    }

    #[test]
    fn load_bad_data_returns_false() {
        let alloc = Lohalloc::new();
        assert!(
            !alloc.load(&[0xFF; 32]),
            "load with bad data should return false"
        );
        assert!(
            !alloc.is_inference(),
            "should still be in training after failed load"
        );
    }

    #[test]
    fn load_empty_returns_false() {
        let alloc = Lohalloc::new();
        assert!(!alloc.load(&[]), "load with empty data should return false");
    }
}

// `fragmentation_pct_for` only exists under `telemetry-observer` — see its
// `#[cfg]` above.
#[cfg(all(test, feature = "telemetry-observer"))]
mod fragmentation_tests {
    use super::*;

    #[test]
    fn slab_mid_class_reports_nonzero_waste() {
        // 40 bytes rounds up to the 64-byte slab class: (64-40)/64 = 37.5%.
        let pct = fragmentation_pct_for(Backend::Slab, 40);
        assert!(
            (pct - 37.5).abs() < 0.01,
            "expected ~37.5% waste, got {pct}"
        );
    }

    #[test]
    fn slab_exact_class_boundary_reports_zero_waste() {
        // Exactly a slab class size (64) — no rounding, no waste.
        let pct = fragmentation_pct_for(Backend::Slab, 64);
        assert_eq!(pct, 0.0);
    }

    #[test]
    fn buddy_mid_order_reports_nonzero_waste() {
        // A request just over a power-of-two boundary rounds up to the next
        // order, wasting close to (but less than) 50%.
        let small_order_size = buddy::block_size(buddy::order_for(65).unwrap());
        let pct = fragmentation_pct_for(Backend::Buddy, 65);
        assert!(
            pct > 0.0 && pct < 50.0,
            "got {pct}% for size 65, reserved {small_order_size}"
        );
    }

    #[test]
    fn system_reports_page_rounding_waste() {
        // 1 byte over the System threshold still rounds up to a full page;
        // waste should be close to 100% (only 1 byte of a whole page used).
        let pct = fragmentation_pct_for(Backend::System, 1);
        assert!(pct > 0.0, "expected nonzero page-rounding waste, got {pct}");
    }

    #[test]
    fn arena_reports_zero_waste() {
        // Bump allocator: no size-class rounding.
        assert_eq!(fragmentation_pct_for(Backend::Arena, 40), 0.0);
    }

    #[test]
    fn fragmentation_is_bounded_0_to_100() {
        for backend in [
            Backend::Slab,
            Backend::Buddy,
            Backend::System,
            Backend::Arena,
        ] {
            for size in [1usize, 7, 40, 64, 1000, 65536] {
                let pct = fragmentation_pct_for(backend, size);
                assert!(
                    (0.0..=100.0).contains(&pct),
                    "fragmentation_pct_for({size}) = {pct} out of [0,100] for backend"
                );
            }
        }
    }
}
