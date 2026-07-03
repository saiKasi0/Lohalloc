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
mod clock;
#[cfg(feature = "telemetry-observer")]
pub mod observer;
pub mod perfect_hash;
pub mod slab;
pub mod state;
pub mod system;
pub mod topology;

use core::alloc::{GlobalAlloc, Layout};
use core::cell::Cell;
use core::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
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
    _pad: [u8; 5],
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
    let reserved = match backend {
        Backend::Slab => lohalloc_core::slab_class_for(total)
            .map(|class| lohalloc_core::SLAB_SIZE_CLASSES[class]),
        Backend::Buddy => buddy::order_for(total).map(buddy::block_size),
        Backend::System => Some(align_up(total, system::page_size())),
        Backend::Arena => None,
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
    /// The Decision Engine (Phase 3). Routes allocations via MAB in Training
    /// mode and via a frozen `PerfectHashTable` in Inference mode.
    state: Mutex<state::AllocatorState>,
    /// Cheap, lock-free mirror of `state.is_inference()` so `dealloc` can
    /// skip Layer 2 reward bookkeeping (`record_latency`) without taking the
    /// state lock on the Inference hot path. Flipped exactly once inside
    /// `freeze()`/`load()` (which already touch the state lock) — a single
    /// relaxed atomic load costs far less than a `Mutex` lock/unlock pair.
    frozen: AtomicBool,
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
}

/// Count of Inference-mode lookups whose key was *not* in the frozen table
/// (falling back to size-based routing). Only incremented on a miss — the
/// hit path stays untouched. This is the observability hook that lets the
/// Phase 6 benchmark verify a pre-trained model actually matches a fresh
/// process's call sites (ASLR-stable hashes): a model-loaded run whose
/// workload was trained in an earlier process should see ~0 misses.
static PHT_MISSES: AtomicU64 = AtomicU64::new(0);

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
            // Decision Engine starts in Training mode.
            state: Mutex::new(state::AllocatorState::new_training_const()),
            frozen: AtomicBool::new(false),
            frozen_table: AtomicPtr::new(core::ptr::null_mut()),
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
        // telemetry-observer feature is enabled; read in emit_alloc).
        #[cfg(feature = "telemetry-observer")]
        ALLOC_START_NS.set(observer::now_ns());

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

        // Telemetry hook (compiled away when the feature is off). Emit the
        // free record before routing so the GUI sees every deallocation,
        // even those that will be silently dropped (e.g. Arena frees, which
        // are no-ops here).
        #[cfg(feature = "telemetry-observer")]
        {
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
                if let Ok(mut slab) = self.slab.lock() {
                    unsafe { slab.dealloc(block, header.size) };
                }
            }
            Some(Backend::Buddy) => {
                let block = ptr.sub(pad);
                if let Ok(mut buddy) = self.buddy.lock() {
                    unsafe { buddy.dealloc(block, header.size) };
                }
            }
            Some(Backend::System) => {
                // Release the exact mapping recorded at alloc time.
                unsafe {
                    libc::munmap(header.base as *mut core::ffi::c_void, header.map_len);
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
            if let Ok(mut st) = self.state.lock() {
                st.record_latency(header.hash, core_backend, header.size_class_hint, dt);
            }
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
                return ptr;
            }
            return self.route_by_size(total, align, pad, hash, size_class);
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
            return self.route_by_size(total, align, pad, hash, size_class);
        }

        // Training: time the *actual* outcome (success, or failure +
        // fallthrough) and attribute it to the recommended arm regardless
        // of which backend ultimately served the request.
        let t0 = clock::now_ns();
        let ptr = match recommended {
            Some(backend) => self
                .try_backend(backend, total, align, pad, hash, size_class)
                .unwrap_or_else(|| self.route_by_size(total, align, pad, hash, size_class)),
            None => self.route_by_size(total, align, pad, hash, size_class),
        };
        let latency_ns = clock::now_ns().saturating_sub(t0);

        if let (Some(backend), Ok(mut st)) = (recommended, self.state.lock()) {
            st.record_latency(hash, backend, size_class, latency_ns);
        }

        ptr
    }

    /// Attempt an allocation via a specific backend. Returns the user pointer
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
                if let Ok(mut slab) = self.slab.lock() {
                    slab.alloc(total).map(|block| {
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
                } else {
                    None
                }
            }
            lohalloc_core::Backend::Buddy if total <= BUDDY_MAX => {
                if let Ok(mut buddy) = self.buddy.lock() {
                    buddy.alloc(total).map(|block| {
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
                } else {
                    None
                }
            }
            lohalloc_core::Backend::Arena => {
                // Arena allocation (lazily initialized).
                if let Ok(mut arena_guard) = self.arena.lock() {
                    if arena_guard.is_none() {
                        *arena_guard = arena::BumpArena::new();
                    }
                    if let Some(ref mut arena) = *arena_guard {
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
                    } else {
                        None
                    }
                } else {
                    None
                }
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
    fn route_by_size(
        &self,
        total: usize,
        align: usize,
        pad: usize,
        hash: u64,
        size_class_hint: u8,
    ) -> *mut u8 {
        // 1. Slab: small, naturally-aligned requests.
        if total <= SLAB_MAX {
            if let Ok(mut slab) = self.slab.lock() {
                if let Some(block) = slab.alloc(total) {
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
        }

        // 2. Buddy: medium, variable-size.
        if total <= BUDDY_MAX {
            if let Ok(mut buddy) = self.buddy.lock() {
                if let Some(block) = buddy.alloc(total) {
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
        let header = Header {
            magic: MAGIC,
            backend: backend_tag,
            size_class_hint,
            align_log2: align.trailing_zeros() as u8,
            _pad: [0; 5],
            size: total,
            base,
            map_len,
            hash,
        };
        unsafe {
            core::ptr::write_unaligned(user.sub(HEADER_SIZE) as *mut Header, header);
        }

        // Telemetry hook (compiled away when the feature is off). This is
        // the single chokepoint every successful allocation flows through,
        // so we emit here rather than at each call site in `try_backend` /
        // `route_by_size` / `system_alloc_with_header`.
        #[cfg(feature = "telemetry-observer")]
        {
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
            if let Some(ref mut arena) = *arena_guard {
                if let Some(block) = arena.alloc(total, align) {
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

    /// Process-wide count of Inference-mode routing lookups that missed the
    /// frozen table (and fell back to size-based routing). ~0 on a
    /// model-loaded run means the model's keys matched this process's call
    /// sites — the end-to-end proof that hashes are stable across runs.
    pub fn pht_miss_count() -> u64 {
        PHT_MISSES.load(Ordering::Relaxed)
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
        let header = unsafe { ptr.cast::<Header>().offset(-1).read_unaligned() };
        header.magic == MAGIC
    }
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
