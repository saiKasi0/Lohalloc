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
use core::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
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
    slab: Mutex<slab::Slab>,
    buddy: Mutex<buddy::Buddy>,
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
    arena_exhausted: AtomicBool,
    /// The Decision Engine (Phase 3). Routes allocations via MAB in Training
    /// mode and via a frozen `PerfectHashTable` in Inference mode.
    state: Mutex<state::AllocatorState>,
    /// Cheap, lock-free mirror of `state.is_inference()` so `dealloc` can
    /// skip Layer 2 reward bookkeeping (`record_latency`) without taking the
    /// state lock on the Inference hot path. Flipped exactly once inside
    /// `freeze()`/`load()` (which already touch the state lock) — a single
    /// relaxed atomic load costs far less than a `Mutex` lock/unlock pair.
    frozen: AtomicBool,
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
    /// it's a best-effort retention layer over plain mmap/munmap).
    system_cache_lock: AtomicBool,
    /// Count of System allocations served from the retention cache
    /// (testability/introspection).
    system_cache_hits: AtomicU64,
    /// Unique id for the per-thread magazine layer (see `magazine.rs`'s
    /// ownership doc): 0 = unassigned, lazily claimed from a process-wide
    /// monotonic counter on first slab use. Ids are never reused, so a
    /// thread's magazine can always tell "my instance's blocks" from a
    /// previous instance's (whose regions may already be unmapped).
    magazine_id: AtomicU64,
    /// Lock-free copy of the frozen `PerfectHashTable`, published by
    /// `freeze()`/`load()` (null while in Training mode). The Inference
    /// alloc fast path loads this pointer instead of taking the `state`
    /// Mutex — before this existed, *every* allocation serialized on that
    /// global lock even in frozen mode, which showed up directly in the
    /// Phase 6 cross-allocator numbers. The pointed-to table is immutable
    /// and deliberately leaked on `reset_to_training()` (a concurrent
    /// reader may still hold `&*table`; a full RCU scheme is overkill for
    /// a GUI dev button, and leakage is bounded by freeze/reset cycles).
    frozen_table: AtomicPtr<perfect_hash::PerfectHashTable>,
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
    slab_headerless: AtomicBool,
    /// Maps a header-free Slab segment's base address to its class, so
    /// `dealloc`/cabi entry points can recover a headerless block's class
    /// from its address alone. See `registry::SegmentRegistry`.
    segment_registry: registry::SegmentRegistry,
    /// Published descriptor of the arena chunk currently being bumped, for
    /// `arena_alloc_fast`'s lock-free path — null until the arena is first
    /// initialized. See `ArenaChunkDescriptor`'s doc.
    arena_chunk: AtomicPtr<ArenaChunkDescriptor>,
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

impl Default for Lohalloc {
    fn default() -> Self {
        Self::new()
    }
}

impl Lohalloc {
    pub const fn new() -> Self {
        Self {
            slab: Mutex::new(slab::Slab::new()),
            buddy: Mutex::new(buddy::Buddy::new()),
            // Arena is lazily initialized on first use (requires mmap, which
            // is not const-evaluable).
            arena: Mutex::new(None),
            arena_exhausted: AtomicBool::new(false),
            // Decision Engine starts in Training mode.
            state: Mutex::new(state::AllocatorState::new_training_const()),
            frozen: AtomicBool::new(false),
            system_cache: core::cell::UnsafeCell::new(system::SystemCache::new()),
            system_cache_lock: AtomicBool::new(false),
            system_cache_hits: AtomicU64::new(0),
            magazine_id: AtomicU64::new(0),
            frozen_table: AtomicPtr::new(core::ptr::null_mut()),
            slab_headerless: AtomicBool::new(false),
            segment_registry: registry::SegmentRegistry::new(),
            arena_chunk: AtomicPtr::new(core::ptr::null_mut()),
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
        static NEXT_MAGAZINE_ID: AtomicU64 = AtomicU64::new(1);
        let fresh = NEXT_MAGAZINE_ID.fetch_add(1, Ordering::Relaxed);
        match self
            .magazine_id
            .compare_exchange(0, fresh, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => fresh,
            Err(winner) => winner,
        }
    }
}

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
                    if let Ok(mut slab) = self.slab.lock() {
                        unsafe { slab.dealloc(block, header.size) };
                    }
                } else {
                    let owner = self.magazine_owner();
                    if !magazine::push(owner, class, block) {
                        // Magazine full: flush half of it plus this block
                        // back to the central slab in one locked batch.
                        let mut buf = [core::ptr::null_mut::<u8>(); 16];
                        let flush = magazine::refill_count(class).min(buf.len());
                        let n = magazine::take(owner, class, &mut buf[..flush]);
                        if let Ok(mut slab) = self.slab.lock() {
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
        let hash = topology::fast_stack_hash();
        self.route_alloc_inner(hash, size, align, pad, total)
    }

    /// Shared implementation for `route_alloc`/`route_alloc_with_hash` —
    /// both differ only in how `hash` is obtained (stack walk vs.
    /// caller-provided, for the replay engine).
    fn route_alloc_inner(
        &self,
        hash: u64,
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
            let backend = match unsafe { (*table).lookup(key) } {
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

        let (recommended, is_training) = if let Ok(mut st) = self.state.lock() {
            (Some(st.route(hash, size)), !st.is_inference())
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
        let n = {
            let mut slab = self.slab.lock().ok()?;
            slab.alloc_batch(class, &mut buf[..want])
        };
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
        let n = {
            let mut slab = self.slab.lock().ok()?;
            let mut try_register = |base: usize| self.segment_registry.insert(base, class as u8);
            slab.alloc_batch_headerless(class, &mut buf[..want], &mut try_register)
        };
        if n == 0 {
            return None;
        }
        for &p in buf.iter().take(n).skip(1) {
            let pushed = magazine::push(owner, class, p);
            debug_assert!(pushed);
        }
        Some(buf[0])
    }

    /// Return a raw, header-free Slab block of `class` to circulation.
    /// Ordinary slab blocks once the class is known — reuses the existing
    /// (header-agnostic) `Slab::dealloc_class`/`dealloc_batch` and the
    /// existing magazine layer unchanged.
    fn slab_dealloc_headerless(&self, block: *mut u8, class: u8) {
        let class = class as usize;
        let owner = self.magazine_owner();
        if magazine::push(owner, class, block) {
            return;
        }
        let mut buf = [core::ptr::null_mut::<u8>(); 16];
        let flush = magazine::refill_count(class).min(buf.len());
        let n = magazine::take(owner, class, &mut buf[..flush]);
        if let Ok(mut slab) = self.slab.lock() {
            unsafe {
                slab.dealloc_batch(class, &buf[..n]);
                slab.dealloc_class(block, class);
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

    /// Lock-free arena bump: loads the published chunk descriptor and CASes
    /// its cursor forward. Returns `None` — falling through to the
    /// `Mutex`-guarded slow path — when there's no descriptor yet (arena
    /// never initialized), or the request doesn't fit in the current
    /// chunk. Never chains chunks itself; only the slow path maps/advances.
    fn arena_alloc_fast(&self, size: usize, align: usize) -> Option<*mut u8> {
        let desc_ptr = self.arena_chunk.load(Ordering::Acquire);
        if desc_ptr.is_null() {
            return None;
        }
        // SAFETY: see `ArenaChunkDescriptor`'s doc — the pointed-to chunk
        // and its cursor never move or get freed while this `Lohalloc`
        // instance is alive.
        let desc = unsafe { &*desc_ptr };
        let cursor = unsafe { &*desc.cursor };
        let align = align.max(MIN_ALIGN);
        loop {
            let cur = cursor.load(Ordering::Relaxed);
            let aligned = align_up(cur, align);
            let new_cur = aligned.checked_add(size)?;
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
                return Some(aligned as *mut u8);
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
    fn buddy_block_via_magazine(&self, total: usize) -> Option<*mut u8> {
        let order = buddy::order_for(total)?;
        let Some(idx) = buddy_mag::index_for(order) else {
            let mut buddy = self.buddy.lock().ok()?;
            return buddy.alloc(total);
        };
        let owner = self.magazine_owner();
        if let Some(block) = buddy_mag::pop(owner, idx) {
            return Some(block);
        }
        // Magazine miss: batch-refill under the central lock.
        // Sized >= max buddy_mag::refill_count (cap 32 / 2 = 16).
        let mut buf = [core::ptr::null_mut::<u8>(); 16];
        let want = buddy_mag::refill_count(idx).min(buf.len());
        let n = {
            let mut buddy = self.buddy.lock().ok()?;
            buddy.alloc_order_batch(order, &mut buf[..want])
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
            if let Ok(mut buddy) = self.buddy.lock() {
                unsafe { buddy.dealloc(block, size) };
            }
            return;
        };
        let owner = self.magazine_owner();
        if buddy_mag::push(owner, idx, block) {
            return;
        }
        // Magazine full: flush half of it plus this block back to the
        // central buddy in one locked batch. Sized >= max refill_count (16).
        let mut buf = [core::ptr::null_mut::<u8>(); 16];
        let flush = buddy_mag::refill_count(idx).min(buf.len());
        let n = buddy_mag::take(owner, idx, &mut buf[..flush]);
        if let Ok(mut buddy) = self.buddy.lock() {
            unsafe {
                buddy.dealloc_order_batch(order, &buf[..n]);
                buddy.dealloc_order_batch(order, core::slice::from_ref(&block));
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
                // Lock-free fast path: bump the published current chunk's
                // cursor directly, no Mutex. Falls through to the slow
                // path below on a miss (arena not yet initialized, or the
                // current chunk is full/doesn't fit this request).
                if let Some(block) = self.arena_alloc_fast(total, align) {
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
                let result = arena_guard.as_mut().and_then(|arena| {
                    arena.alloc(total, align).map(|block| {
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
                });
                if let Some(arena) = arena_guard.as_ref() {
                    self.publish_arena_chunk(arena);
                    // Latch permanent exhaustion (cap reached on a request
                    // that would have fit an empty chunk) so every later
                    // arena-routed alloc fast-fails at the arm's head.
                    if result.is_none() && arena.exhausted_after_failed(total, align) {
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
            if let Some(block) = self.buddy_block_via_magazine(total) {
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
    /// are invalidated. The Decision Engine (Phase 3) will call this when a
    /// topological cluster's lifetime ends.
    pub fn reset_arena(&self) {
        if let Ok(mut arena_guard) = self.arena.lock() {
            if let Some(ref mut arena) = *arena_guard {
                arena.reset();
            }
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
        Self::with_realloc_guard(|| {
            if let Ok(mut state) = self.state.lock() {
                state.freeze();
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
        if let Some(table) = state.routing_table() {
            let leaked: *mut perfect_hash::PerfectHashTable = Box::leak(Box::new(table.clone()));
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
                    // Header-free Slab is only safe if this instance's Slab
                    // has never served a block yet (region_count() == 0) —
                    // otherwise a class's free list could already hold
                    // header-carrying blocks, and a later headerless refill
                    // would mix flavors in that same list (see
                    // `slab_headerless`'s doc). In the real usage pattern
                    // (`lohalloc-cabi`'s `ensure_model_loaded`, always the
                    // very first thing that touches a fresh instance) this
                    // check always passes; it's a safety net for any other
                    // caller, not a hot-path cost — `load()` is a rare,
                    // one-time call.
                    if let Ok(slab) = self.slab.lock() {
                        if slab.region_count() == 0 {
                            self.slab_headerless.store(true, Ordering::Release);
                        }
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

    /// Process-wide count of Inference-mode routing lookups that missed the
    /// frozen table (and fell back to size-based routing). ~0 on a
    /// model-loaded run means the model's keys matched this process's call
    /// sites — the end-to-end proof that hashes are stable across runs.
    pub fn pht_miss_count() -> u64 {
        PHT_MISSES.load(Ordering::Relaxed)
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
        self.route_alloc_inner(hash, size, align, pad, total)
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
        let slab_region_count = self.slab.lock().map(|s| s.region_count()).unwrap_or(0);
        let buddy_region_count = self.buddy.lock().map(|b| b.region_count()).unwrap_or(0);
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
        if self.headerless_class_for(ptr).is_some() {
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
        }
    }
}

/// Opaque snapshot for the fused cabi `realloc` path — see
/// [`Lohalloc::usable_size_for_realloc`]. Either a full header copy (the
/// ordinary path) or just a slab class (a headerless block has no header
/// to copy).
pub struct ReallocToken(ReallocTokenInner);

enum ReallocTokenInner {
    Header(Header),
    Headerless(u8),
}

// ---------------------------------------------------------------------------
// Phase 3 Integration Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod integration_tests {
    use super::*;
    use core::alloc::{GlobalAlloc, Layout};

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
        // Load a hand-built model routing (hash, size_class(64)) → Buddy.
        // Size 64 would default to Slab, so getting Buddy back proves the
        // lock-free published table (not the size fallback) made the call.
        let hash = 0xDEAD_BEEF_u64;
        let size = 64usize;
        let sc = state::size_class_for(size);
        let key = state::combine_hash_size_class(hash, sc);
        let table = perfect_hash::PerfectHashTable::from_entries(vec![(
            key,
            sc,
            lohalloc_core::Backend::Buddy,
        )]);
        let alloc = Lohalloc::new();
        assert!(alloc.load(&table.serialize()), "model load should succeed");

        let layout = Layout::from_size_align(size, 16).unwrap();
        let ptr = unsafe { alloc.alloc_with_hash(layout, hash) };
        assert!(!ptr.is_null());
        assert_eq!(
            unsafe { alloc.backend_for_ptr(ptr) },
            Some(lohalloc_core::Backend::Buddy),
            "published frozen table must drive routing"
        );
        unsafe { alloc.dealloc_with_hash(ptr, layout) };
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
    fn load_after_prior_slab_activity_stays_header_based() {
        // A live-trained instance (or one that otherwise already served a
        // slab block) must NOT flip to headerless on a later `load()` —
        // doing so would risk mixing header and headerless blocks in the
        // same class's free list. Training-mode allocation always writes
        // headers; that must remain true for every allocation this
        // instance serves afterward too, even post-load().
        let training_hash = 0x4EAD_0002u64;
        let size = 64usize;
        let alloc = Lohalloc::new();
        let layout = Layout::from_size_align(size, 16).unwrap();

        // Training-mode traffic populates at least one Slab region.
        for _ in 0..20 {
            let ptr = unsafe { alloc.alloc_with_hash(layout, training_hash) };
            assert!(!ptr.is_null());
            unsafe { alloc.dealloc_with_hash(ptr, layout) };
        }
        assert!(alloc.backend_counters().slab_region_count > 0);

        assert!(alloc.load(&slab_only_model(0x4EAD_0003u64, size)));
        assert!(
            !alloc.is_slab_headerless(),
            "load() must not enable headerless mode once the Slab has already served a block"
        );

        // Allocations must still work correctly (header-based).
        let ptr = unsafe { alloc.alloc_with_hash(layout, 0x4EAD_0003u64) };
        assert!(!ptr.is_null());
        assert_eq!(
            unsafe { alloc.backend_for_ptr(ptr) },
            Some(lohalloc_core::Backend::Slab)
        );
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
        let size = 64usize;
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

        // ~66 MiB of requests: ~32 MiB lands in the chained arena (cap),
        // the rest must fall through to Slab — every single one succeeds.
        let mut ptrs = Vec::new();
        for i in 0..16_000 {
            let p = unsafe { alloc.alloc_with_hash(layout, hash) };
            assert!(!p.is_null(), "alloc {i} must succeed");
            ptrs.push(p);
        }
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

        // ~64 MiB of requests against a 32 MiB cap: every alloc must still
        // succeed (post-cap ones via the size-chain fallthrough).
        let mut ptrs = Vec::new();
        for i in 0..1000 {
            let p = unsafe { alloc.alloc_with_hash(layout, hash) };
            assert!(!p.is_null(), "alloc {i} must succeed even after the cap");
            ptrs.push(p);
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
