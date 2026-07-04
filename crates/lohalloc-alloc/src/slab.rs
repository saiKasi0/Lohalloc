//! Slab sub-allocator (fixed-size blocks).
//!
//! One free-list per size class. Blocks are carved out of page-backed regions
//! obtained from the System Fallback. On `alloc` we pop a block from the front
//! of the free list; on `dealloc` we push it back. When a list is empty we
//! map a fresh page region, slice it into blocks, and thread them onto the list.
//!
//! Each freed block is linked through its own first bytes (intrusive free
//! list), so we track no per-block metadata of our own — the block size is
//! recoverable from the size class the caller returns at dealloc time. This is
//! a Phase 1 implementation: simple, correct, alignment-clean. Phase 2 adds
//! the telemetry hooks on top without changing the free-list structure.

use crate::system;
use lohalloc_core::{align_up, slab_class_for, SLAB_SIZE_CLASSES};

/// Node of the intrusive free list. Lives at the head of each free block.
/// Blocks are always at least `SLAB_SIZE_CLASSES[0]` bytes (8), so the pointer
/// always fits.
#[repr(C)]
struct FreeNode {
    next: Option<*mut FreeNode>,
}

/// One slab allocator: a per-class free-list array plus the list of backing
/// regions (so they are freed on drop and not leaked).
pub struct Slab {
    /// `free_heads[i]` is the head of the free list for `SLAB_SIZE_CLASSES[i]`.
    free_heads: [Option<*mut FreeNode>; NUM_CLASSES],
    /// Owning handles to the page regions we carved blocks from. Held alive so
    /// the memory stays mapped for the lifetime of the allocator.
    regions: Vec<system::Mapping>,
}

/// Number of slab size classes (compile-time constant so `new()` can be const).
const NUM_CLASSES: usize = SLAB_SIZE_CLASSES.len();

impl Default for Slab {
    fn default() -> Self {
        Self::new()
    }
}

impl Slab {
    pub const fn new() -> Self {
        Self {
            free_heads: [None; NUM_CLASSES],
            regions: Vec::new(),
        }
    }

    /// Allocate a block of `size` bytes (rounded up to the nearest slab class),
    /// aligned to at least the class size (which is a power of two >= 8, so all
    /// slab blocks satisfy `MIN_ALIGN`). Returns `None` if `size` exceeds the
    /// largest slab class — the shim routes those to Buddy/System instead.
    ///
    /// # Safety contract for the caller
    /// The returned pointer is valid until `dealloc` is called with the same
    /// pointer and the same `size`. Reading/writing beyond `class_size` bytes
    /// is UB.
    pub fn alloc(&mut self, size: usize) -> Option<*mut u8> {
        let class = slab_class_for(size)?;
        self.alloc_class(class)
    }

    /// Allocate one block of an already-computed `class` — the class-aware
    /// fast path used by the per-thread magazine layer (`lib.rs` computes
    /// the class once per op instead of rescanning here and at dealloc).
    pub fn alloc_class(&mut self, class: usize) -> Option<*mut u8> {
        debug_assert!(class < NUM_CLASSES);
        loop {
            if let Some(block) = self.free_heads[class].take() {
                // Pop the front of the free list. The block's `next` field
                // holds the new head.
                let node = unsafe { &mut *block };
                self.free_heads[class] = node.next;
                return Some(block as *mut u8);
            }
            // Refill: map a region and slice it into blocks of the class
            // size, then loop to pop (refill either added blocks or failed).
            self.refill(class, SLAB_SIZE_CLASSES[class])?;
        }
    }

    /// Pop up to `out.len()` blocks of `class` in one locked visit — the
    /// magazine refill path. Attempts one region refill if the list runs
    /// dry mid-batch. Returns how many blocks were written to `out`.
    pub fn alloc_batch(&mut self, class: usize, out: &mut [*mut u8]) -> usize {
        debug_assert!(class < NUM_CLASSES);
        let mut n = 0;
        let mut refilled = false;
        while n < out.len() {
            match self.free_heads[class].take() {
                Some(block) => {
                    let node = unsafe { &mut *block };
                    self.free_heads[class] = node.next;
                    out[n] = block as *mut u8;
                    n += 1;
                }
                None => {
                    if refilled || self.refill(class, SLAB_SIZE_CLASSES[class]).is_none() {
                        break;
                    }
                    refilled = true;
                }
            }
        }
        n
    }

    /// Free a block previously returned by `alloc` with the same `size`. After
    /// this the pointer must not be used.
    ///
    /// # Safety
    /// `ptr` must have been returned by `alloc` with a size that rounds to the
    /// same class, and not double-freed.
    pub unsafe fn dealloc(&mut self, ptr: *mut u8, size: usize) {
        let class = match slab_class_for(size) {
            Some(c) => c,
            None => {
                debug_assert!(false, "dealloc size out of slab range");
                return;
            }
        };
        unsafe { self.dealloc_class(ptr, class) };
    }

    /// Free a block of an already-known `class` — see [`Self::alloc_class`].
    ///
    /// # Safety
    /// `ptr` must be a block of exactly this class, not double-freed.
    pub unsafe fn dealloc_class(&mut self, ptr: *mut u8, class: usize) {
        debug_assert!(class < NUM_CLASSES);
        // Push to front of free list.
        let node = ptr as *mut FreeNode;
        unsafe {
            (*node).next = self.free_heads[class].take();
        }
        self.free_heads[class] = Some(node);
    }

    /// Push a whole batch of `class` blocks back in one locked visit — the
    /// magazine flush path.
    ///
    /// # Safety
    /// Every pointer must be a block of exactly this class, not double-freed.
    pub unsafe fn dealloc_batch(&mut self, class: usize, blocks: &[*mut u8]) {
        for &b in blocks {
            unsafe { self.dealloc_class(b, class) };
        }
    }

    /// Map a fresh region sized to hold several `class_size` blocks and thread
    /// them all onto the free list for `class`.
    fn refill(&mut self, class: usize, class_size: usize) -> Option<()> {
        // Grab ~64 KiB of blocks per refill (at least one page). Align block
        // size up to MIN_ALIGN so every block satisfies the global alignment
        // contract even if the class itself is small.
        let stride = align_up(class_size, lohalloc_core::MIN_ALIGN);
        let region_bytes = (page_or(stride * 16, system::page_size())).max(system::page_size());
        let region = system::alloc_pages(region_bytes, stride.max(system::page_size()))?;

        let base = region.as_ptr();
        let usable = region.usable();
        let n = usable / stride;

        // Thread every block onto the free list. (The pre-loop version of
        // `alloc` skipped block 0 here on the assumption its recursion
        // would pop it; the loop-based `alloc_class`/`alloc_batch` pop
        // strictly from the list, so all `n` blocks must be threaded or
        // block 0 leaks once per region.)
        let mut next = self.free_heads[class];
        for i in (0..n).rev() {
            let block = unsafe { base.add(i * stride) } as *mut FreeNode;
            unsafe {
                (*block).next = next;
            }
            next = Some(block);
        }
        self.free_heads[class] = next;
        self.regions.push(region);
        Some(())
    }

    /// Number of currently-live backing regions (useful for leak accounting in
    /// tests — does not grow without bound under a fixed live set).
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }
}

fn page_or(a: usize, b: usize) -> usize {
    if a > b {
        a
    } else {
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_small() {
        let mut s = Slab::new();
        let p = s.alloc(7).expect("alloc 7");
        assert!(!p.is_null());
        assert!(is_aligned_to(p, lohalloc_core::MIN_ALIGN));
        unsafe { core::ptr::write_bytes(p, 0xCD, 8) };
        unsafe { s.dealloc(p, 7) };
    }

    #[test]
    fn reuses_blocks_under_fixed_live_set() {
        // Allocating and freeing the same number of blocks repeatedly must not
        // grow the region count unboundedly.
        let mut s = Slab::new();
        let mut live = Vec::new();
        for _round in 0..1000 {
            for _ in 0..8 {
                live.push(s.alloc(64).expect("alloc"));
            }
            for p in live.drain(..) {
                unsafe { s.dealloc(p, 64) };
            }
        }
        // After 1000 rounds with a bounded live set, we should have mapped only
        // a handful of regions (one or two refills at most for class 64).
        assert!(s.region_count() < 8, "region_count = {}", s.region_count());
    }

    #[test]
    fn alloc_too_large_is_none() {
        let mut s = Slab::new();
        // Above SLAB_MAX.
        assert!(s.alloc(1 << 17).is_none());
    }

    fn is_aligned_to(ptr: *mut u8, align: usize) -> bool {
        (ptr as usize) & (align - 1) == 0
    }
}
