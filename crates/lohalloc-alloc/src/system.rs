//! System Fallback backend.
//!
//! Page-backed allocation via `mmap`/`munmap`. This is the leaf backend: it
//! serves oversized requests directly and provides backing pages to the Slab
//! and Buddy sub-allocators.
//!
//! # Cross-platform contract
//!
//! Supported targets: `linux` and `macos` on both `x86_64` and `aarch64`.
//! The two OSes differ in `mmap` flag naming and (critically) page size —
//! Apple Silicon uses 16 KiB pages, x86 and most Linux aarch64 use 4 KiB,
//! and some Linux aarch64 kernels use 64 KiB. We therefore **query the page
//! size at runtime** via `sysconf(_SC_PAGESIZE)` and never hard-code it.
//!
//! # Alignment
//!
//! `mmap` always returns page-aligned addresses, which satisfy any alignment
//! request up to the page size. For alignments larger than the page size we
//! over-map and adjust the returned pointer (recording the original base so
//! `munmap` can release the whole mapping). This is the cross-architecture-
//! safe pattern that avoids alignment-related bus errors on ARM64.

use core::ffi::c_void;
use lohalloc_core::{align_up, is_aligned};

/// A raw, owned mapping returned by `mmap`. `base` is the address `munmap`
/// must be called on; `ptr` is the aligned, user-visible address (may differ
/// from `base` for over-aligned requests).
#[derive(Debug)]
pub struct Mapping {
    /// Address returned by `mmap` — what we pass to `munmap`.
    base: *mut u8,
    /// Aligned pointer handed back to the caller.
    ptr: *mut u8,
    /// Total byte length of the mapping (`base..base+len`).
    len: usize,
    /// Usable length from `ptr` (i.e. at least `requested`).
    usable: usize,
}

unsafe impl Send for Mapping {}
unsafe impl Sync for Mapping {}

impl Mapping {
    /// The aligned, user-visible pointer. At least `usable()` bytes long.
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Usable byte count from the returned pointer.
    pub fn usable(&self) -> usize {
        self.usable
    }

    /// Raw address `munmap` must receive (the original `mmap` base). Crate-
    /// private: only the `GlobalAlloc` shim (same crate) uses this to release
    /// a mapping whose ownership it took via `mem::forget`.
    pub(crate) unsafe fn raw_base_for_unmap(&self) -> usize {
        self.base as usize
    }

    /// Length `munmap` must receive (the original mapped length). Crate-private.
    pub(crate) unsafe fn raw_len_for_unmap(&self) -> usize {
        self.len
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        unsafe {
            let rc = libc::munmap(self.base as *mut c_void, self.len);
            // `munmap` failures are a programming error (bad mapping). Surface
            // via debug_assert in debug builds; in release we swallow to keep
            // the drop path infallible (a destructor cannot panic safely here).
            debug_assert_eq!(rc, 0, "munmap failed");
        }
    }
}

/// Runtime page size of the host. Queried once via `sysconf(_SC_PAGESIZE)` and
/// cached. Always a power of two.
pub fn page_size() -> usize {
    static mut PAGE_SIZE: usize = 0;
    // SAFETY: a race here is benign — `sysconf` is pure and idempotent, and any
    // thread computes the same value. We use the relaxed read/write model.
    unsafe {
        let ps = PAGE_SIZE;
        if ps != 0 {
            return ps;
        }
        // `sysconf(_SC_PAGESIZE)` is available on both Linux and macOS.
        let v = libc::sysconf(libc::_SC_PAGESIZE);
        // sysconf returns -1 on error; fall back to 4096 (x86/typical Linux).
        let v = if v <= 0 { 4096 } else { v as usize };
        debug_assert!(v.is_power_of_two());
        PAGE_SIZE = v;
        v
    }
}

/// Round `n` up to a whole number of pages.
fn round_to_pages(n: usize) -> usize {
    align_up(n, page_size())
}

/// Allocate a private, anonymous, read/write mapping of at least `size` bytes,
/// aligned to at least `align` (a power of two).
///
/// Returns a [`Mapping`] that releases the mapping on drop.
pub fn alloc_pages(size: usize, align: usize) -> Option<Mapping> {
    if size == 0 {
        return None;
    }
    let page = page_size();
    let align = align.max(page).max(1);

    // If the requested alignment is <= page size, a plain page-aligned mapping
    // already satisfies it. Otherwise we over-allocate by `align` and trim the
    // returned pointer — but we keep `base` so `munmap` releases everything.
    let need = round_to_pages(size);
    let map_len = if align <= page {
        need
    } else {
        round_to_pages(size + align)
    };

    let base = unsafe {
        libc::mmap(
            core::ptr::null_mut(),
            map_len,
            libc::PROT_READ | libc::PROT_WRITE,
            mmap_flags(),
            -1, // anonymous
            0,
        )
    };

    if base == libc::MAP_FAILED || base.is_null() {
        return None;
    }

    let base_addr = base as usize;
    let ptr_addr = align_up(base_addr, align);
    let ptr = ptr_addr as *mut u8;
    let usable = (base_addr + map_len) - ptr_addr;

    debug_assert!(is_aligned(ptr_addr, align));
    debug_assert!(usable >= size);

    Some(Mapping {
        base: base as *mut u8,
        ptr,
        len: map_len,
        usable,
    })
}

/// `mmap` flags for a private anonymous mapping.
///
/// On Linux the flag is `MAP_PRIVATE | MAP_ANONYMOUS`.
/// On macOS the same names exist in libc. Both are exposed by the `libc` crate.
fn mmap_flags() -> libc::c_int {
    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_size_is_pow2() {
        let ps = page_size();
        assert!(ps.is_power_of_two());
        assert!(ps >= 1024);
    }

    #[test]
    fn alloc_and_drop_roundtrip() {
        let m = alloc_pages(4096, 4096).expect("mmap");
        assert!(is_aligned(m.as_ptr() as usize, page_size()));
        assert!(m.usable() >= 4096);
        // Write to every byte to confirm the mapping is writable & the right length.
        unsafe {
            core::ptr::write_bytes(m.as_ptr(), 0xAB, m.usable());
        }
        // Drop releases the mapping; we rely on the process staying clean.
        drop(m);
    }

    #[test]
    fn over_aligned_mapping() {
        // Request alignment larger than a page — the trim path must engage.
        let big_align = page_size() * 4;
        let m = alloc_pages(123, big_align).expect("mmap over-aligned");
        assert!(is_aligned(m.as_ptr() as usize, big_align));
        assert!(m.usable() >= 123);
    }

    #[test]
    fn small_alloc_succeeds() {
        let m = alloc_pages(1, 1).expect("mmap 1 byte");
        assert!(m.usable() >= 1);
    }
}
