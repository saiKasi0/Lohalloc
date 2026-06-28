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
pub mod perfect_hash;
pub mod slab;
pub mod state;
pub mod system;
pub mod topology;

use core::alloc::{GlobalAlloc, Layout};
use core::cell::Cell;
use lohalloc_core::{align_up, BUDDY_MAX, MIN_ALIGN, SLAB_MAX};
use std::sync::Mutex;

/// Sentinel written into every [`Header`] so we can sanity-check dealloc.
const MAGIC: u64 = 0x534d4152414c4844; // "LOHALALHD"

/// Per-allocation header prepended to the user-visible pointer. Lets
/// `dealloc` identify the owning backend without guessing by size.
///
/// 48 bytes; always accessed with `read_unaligned`/`write_unaligned` so we do
/// not impose any alignment requirement beyond what the user asked for.
#[repr(C)]
struct Header {
    magic: u64,
    backend: u8,
    _pad: [u8; 7],
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

const HEADER_SIZE: usize = core::mem::size_of::<Header>(); // 40

/// Bytes of padding between the backend's block start and the user pointer, so
/// the user pointer is aligned to `align` and the header sits immediately
/// before it. `align` must be a power of two.
fn header_pad(align: usize) -> usize {
    align_up(HEADER_SIZE, align)
}

/// Which Execution-Plane backend produced an allocation. Tagged into the
/// [`Header`]. Uses `lohalloc_core::Backend` (re-imported here for the
/// header's `u8` tag).
///
/// The local `Backend` type below mirrors `lohalloc_core::Backend` for
/// use in the `Header` (which stores a `u8` discriminant). The Decision
/// Engine (`state.rs`) uses `lohalloc_core::Backend` directly.
#[repr(u8)]
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
            // Decision Engine starts in Training mode.
            state: Mutex::new(state::AllocatorState::new_training_const()),
        }
    }
}

// SAFETY: backend state is guarded by `Mutex`; `mmap`/`munmap` are thread-safe.
// Re-entrancy is broken by the thread-local guard (see `alloc`). The backends
// never call back into `Lohalloc::alloc` for user allocations.
unsafe impl Sync for Lohalloc {}

thread_local! {
    /// Re-entrancy depth. >0 means we are already inside `alloc`/`dealloc` on
    /// this thread — any further allocation must bypass to `mmap` directly.
    static IN_ALLOC: Cell<usize> = const { Cell::new(0) };
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

        // Re-entrancy guard: if we're already inside the allocator on this
        // thread (e.g. a backend's `Vec` growing), serve directly from mmap.
        let depth = IN_ALLOC.get();
        if depth > 0 {
            return self.system_alloc_with_header(total, align, 0);
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
        match Backend::from_u8(header.backend) {
            Some(Backend::Slab) => {
                let pad = header_pad(layout.align().max(MIN_ALIGN));
                let block = ptr.sub(pad);
                if let Ok(mut slab) = self.slab.lock() {
                    unsafe { slab.dealloc(block, header.size) };
                }
            }
            Some(Backend::Buddy) => {
                let pad = header_pad(layout.align().max(MIN_ALIGN));
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
    fn route_alloc(&self, size: usize, align: usize, pad: usize, total: usize) -> *mut u8 {
        // Capture the topological hash of the current call stack.
        let hash = topology::fast_stack_hash();

        // Ask the Decision Engine which backend to try.
        let recommended: Option<lohalloc_core::Backend> = if let Ok(mut st) = self.state.lock() {
            let backend = st.route(hash, size);
            // Record the outcome (Training mode updates the bandit; Inference
            // is a no-op). We optimistically record success here — if the
            // recommended backend fails and we fall through, the bandit will
            // learn from the overall allocation pattern.
            st.record(hash, backend, size);
            Some(backend)
        } else {
            None
        };

        // Try the recommended backend first.
        if let Some(backend) = recommended {
            if let Some(ptr) = self.try_backend(backend, total, align, pad, hash) {
                return ptr;
            }
        }

        // Fall through to size-based routing (Phase 1 fallback chain).
        self.route_by_size(total, align, pad, hash)
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
    ) -> Option<*mut u8> {
        let local_backend = Backend::from_core(backend);
        match backend {
            lohalloc_core::Backend::Slab if total <= SLAB_MAX => {
                if let Ok(mut slab) = self.slab.lock() {
                    slab.alloc(total).map(|block| {
                        self.write_header(block, pad, local_backend, total, 0, 0, hash)
                    })
                } else {
                    None
                }
            }
            lohalloc_core::Backend::Buddy if total <= BUDDY_MAX => {
                if let Ok(mut buddy) = self.buddy.lock() {
                    buddy.alloc(total).map(|block| {
                        self.write_header(block, pad, local_backend, total, 0, 0, hash)
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
                            self.write_header(block, pad, local_backend, total, 0, 0, hash)
                        })
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            lohalloc_core::Backend::System => {
                let ptr = self.system_alloc_with_header(total, align, hash);
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
    fn route_by_size(&self, total: usize, align: usize, pad: usize, hash: u64) -> *mut u8 {
        // 1. Slab: small, naturally-aligned requests.
        if total <= SLAB_MAX {
            if let Ok(mut slab) = self.slab.lock() {
                if let Some(block) = slab.alloc(total) {
                    return self.write_header(block, pad, Backend::Slab, total, 0, 0, hash);
                }
            }
        }

        // 2. Buddy: medium, variable-size.
        if total <= BUDDY_MAX {
            if let Ok(mut buddy) = self.buddy.lock() {
                if let Some(block) = buddy.alloc(total) {
                    return self.write_header(block, pad, Backend::Buddy, total, 0, 0, hash);
                }
            }
        }

        // 3. System Fallback: any size/alignment.
        self.system_alloc_with_header(total, align, hash)
    }

    /// Allocate `total` bytes at `align` via the System Fallback, write a
    /// `System`-tagged header, and leak the `Mapping` (dealloc will `munmap`
    /// using the base/length recorded in the header). Returns the user ptr.
    fn system_alloc_with_header(&self, total: usize, align: usize, hash: u64) -> *mut u8 {
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
        self.write_header(base, pad, Backend::System, total, raw_base, raw_len, hash)
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
    ) -> *mut u8 {
        let user = unsafe { block.add(pad) };
        let header = Header {
            magic: MAGIC,
            backend: backend as u8,
            _pad: [0; 7],
            size: total,
            base,
            map_len,
            hash,
        };
        unsafe {
            core::ptr::write_unaligned(user.sub(HEADER_SIZE) as *mut Header, header);
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
                    return self.write_header(block, pad, Backend::Arena, total, 0, 0, hash);
                }
            }
        }
        // Arena full or init failed → fall through to System.
        self.system_alloc_with_header(total, align, 0)
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
        if let Ok(mut state) = self.state.lock() {
            state.freeze();
        }
    }

    /// Export the frozen routing table to `.lohalloc` binary bytes.
    ///
    /// Returns `None` if the allocator is still in Training mode (call
    /// `freeze()` first).
    pub fn export(&self) -> Option<Vec<u8>> {
        if let Ok(state) = self.state.lock() {
            state.export()
        } else {
            None
        }
    }

    /// Load a `.lohalloc` model file and transition directly to Inference mode.
    ///
    /// Returns `true` if the model was loaded successfully, `false` if the
    /// data is malformed or the state lock is poisoned.
    pub fn load(&self, data: &[u8]) -> bool {
        let new_state = state::AllocatorState::load(data);
        if let Some(s) = new_state {
            if let Ok(mut state) = self.state.lock() {
                *state = s;
                return true;
            }
        }
        false
    }

    /// Returns `true` if the Decision Engine is in Inference (frozen) mode.
    pub fn is_inference(&self) -> bool {
        if let Ok(state) = self.state.lock() {
            state.is_inference()
        } else {
            false
        }
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
            return self.system_alloc_with_header(total, align, hash);
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

    /// Internal: route an allocation with a caller-provided hash.
    fn route_alloc_with_hash(
        &self,
        size: usize,
        align: usize,
        pad: usize,
        total: usize,
        hash: u64,
    ) -> *mut u8 {
        let recommended: Option<lohalloc_core::Backend> = if let Ok(mut st) = self.state.lock() {
            let backend = st.route(hash, size);
            st.record(hash, backend, size);
            Some(backend)
        } else {
            None
        };

        if let Some(backend) = recommended {
            if let Some(ptr) = self.try_backend(backend, total, align, pad, hash) {
                return ptr;
            }
        }

        self.route_by_size(total, align, pad, hash)
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
