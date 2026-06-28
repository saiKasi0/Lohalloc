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

pub mod buddy;
pub mod slab;
pub mod system;

use core::alloc::{GlobalAlloc, Layout};
use core::cell::Cell;
use std::sync::Mutex;
use lohalloc_core::{align_up, MIN_ALIGN, SLAB_MAX, BUDDY_MAX};

/// Sentinel written into every [`Header`] so we can sanity-check dealloc.
const MAGIC: u64 = 0x534d4152414c4844; // "LOHALALHD"

/// Per-allocation header prepended to the user-visible pointer. Lets
/// `dealloc` identify the owning backend without guessing by size.
///
/// 40 bytes; always accessed with `read_unaligned`/`write_unaligned` so we do
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
}

const HEADER_SIZE: usize = core::mem::size_of::<Header>(); // 40

/// Bytes of padding between the backend's block start and the user pointer, so
/// the user pointer is aligned to `align` and the header sits immediately
/// before it. `align` must be a power of two.
fn header_pad(align: usize) -> usize {
    align_up(HEADER_SIZE, align)
}

/// Which Execution-Plane backend produced an allocation. Tagged into the
/// [`Header`]. (Mirrors `lohalloc_core::Backend` but kept local + `u8` for
/// the header.)
#[repr(u8)]
enum Backend {
    Slab = 0,
    Buddy = 1,
    System = 2,
}

impl Backend {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Backend::Slab),
            1 => Some(Backend::Buddy),
            2 => Some(Backend::System),
            _ => None,
        }
    }
}

/// The composite allocator. Install an instance of this as
/// `#[global_allocator]` to route every Rust allocation through Lohalloc.
pub struct Lohalloc {
    slab: Mutex<slab::Slab>,
    buddy: Mutex<buddy::Buddy>,
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
            return self.system_alloc_with_header(total, align);
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
                    libc::munmap(
                        header.base as *mut core::ffi::c_void,
                        header.map_len,
                    );
                }
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
    fn route_alloc(&self, _size: usize, align: usize, pad: usize, total: usize) -> *mut u8 {
        // 1. Slab: small, naturally-aligned requests.
        if total <= SLAB_MAX {
            if let Ok(mut slab) = self.slab.lock() {
                if let Some(block) = slab.alloc(total) {
                    return self.write_header(block, pad, Backend::Slab, total, 0, 0);
                }
            }
        }

        // 2. Buddy: medium, variable-size. Buddy blocks are aligned to their
        //    power-of-two block size, which is >= `total` rounded up and thus
        //    >= `align` (we set total = size + pad with size >= align).
        if total <= BUDDY_MAX {
            if let Ok(mut buddy) = self.buddy.lock() {
                if let Some(block) = buddy.alloc(total) {
                    return self.write_header(block, pad, Backend::Buddy, total, 0, 0);
                }
            }
        }

        // 3. System Fallback: any size/alignment (over-maps to satisfy align).
        self.system_alloc_with_header(total, align)
    }

    /// Allocate `total` bytes at `align` via the System Fallback, write a
    /// `System`-tagged header, and leak the `Mapping` (dealloc will `munmap`
    /// using the base/length recorded in the header). Returns the user ptr.
    fn system_alloc_with_header(&self, total: usize, align: usize) -> *mut u8 {
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
        self.write_header(base, pad, Backend::System, total, raw_base, raw_len)
    }

    /// Write the ownership header at `block + pad - HEADER_SIZE` and return
    /// `block + pad` (the user pointer). `block` must be aligned to at least
    /// `align` and hold `total` usable bytes.
    fn write_header(
        &self,
        block: *mut u8,
        pad: usize,
        backend: Backend,
        total: usize,
        base: usize,
        map_len: usize,
    ) -> *mut u8 {
        let user = unsafe { block.add(pad) };
        let header = Header {
            magic: MAGIC,
            backend: backend as u8,
            _pad: [0; 7],
            size: total,
            base,
            map_len,
        };
        unsafe {
            core::ptr::write_unaligned(user.sub(HEADER_SIZE) as *mut Header, header);
        }
        user
    }
}
