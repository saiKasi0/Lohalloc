//! Buddy sub-allocator (variable-size, power-of-two blocks).
//!
//! A classic binary-buddy allocator backed by page regions from the System
//! Fallback. Each region is a power-of-two number of pages; we recursively split
//! it into halves down to `MIN_BLOCK` (the minimum buddy block, which equals
//! `MIN_ALIGN` to keep every block aligned). Free blocks of the same order are
//! linked on a per-order free list. On `dealloc` a block's buddy is computed by
//! XOR-ing the appropriate bit; if the buddy is free the pair coalesces up to
//! the next order, and so on.
//!
//! This is a Phase 1 implementation: single-threaded (`&mut self`), correct,
//! and alignment-clean. Phase 2 adds telemetry hooks without changing the
//! split/coalesce structure.

use crate::system;
use lohalloc_core::{round_up_pow2, MIN_ALIGN};

/// Minimum buddy block size (bytes). Must be a power of two and >= `MIN_ALIGN`
/// so every block satisfies the global alignment contract. 16 keeps us SIMD-
/// friendly and matches the slab floor.
const MIN_BLOCK: usize = MIN_ALIGN;

/// Maximum order index. Order `o` holds blocks of size `MIN_BLOCK << o`.
/// We cap at `BUDDY_MAX`, which is 1 MiB — requests above that go to the System
/// Fallback directly (whole-page `mmap`).
const MAX_ORDER: usize = {
    let mut o = 0;
    let mut s = MIN_BLOCK;
    while s < lohalloc_core::BUDDY_MAX {
        s <<= 1;
        o += 1;
    }
    o
};

/// Intrusive free-list node living at the head of each free block.
#[repr(C)]
struct FreeNode {
    next: Option<*mut FreeNode>,
}

/// A buddy region: a contiguous, page-backed run of `region_bytes` bytes whose
/// base is aligned to `region_bytes` (so buddy arithmetic works cleanly).
struct Region {
    base: *mut u8,
    bytes: usize,
    _mapping: system::Mapping,
}

unsafe impl Send for Region {}

/// Number of buddy orders (compile-time constant so `new()` can be const).
const NUM_ORDERS: usize = MAX_ORDER + 1;

/// One buddy allocator instance.
pub struct Buddy {
    /// `free_lists[o]` is the head of the free list for blocks of order `o`.
    free_lists: [Option<*mut FreeNode>; NUM_ORDERS],
    /// Owning regions. Held alive so the memory stays mapped.
    regions: Vec<Region>,
}

impl Default for Buddy {
    fn default() -> Self {
        Self::new()
    }
}

impl Buddy {
    pub const fn new() -> Self {
        Self {
            free_lists: [None; NUM_ORDERS],
            regions: Vec::new(),
        }
    }

    /// Allocate `size` bytes, aligned to at least `MIN_ALIGN`. Returns `None`
    /// if `size` exceeds `BUDDY_MAX` (the shim routes those to the System
    /// backend).
    ///
    /// # Safety contract for the caller
    /// The returned pointer is valid until `dealloc` is called with the same
    /// pointer and the same `size`. Reading/writing beyond the rounded block
    /// size is UB.
    pub fn alloc(&mut self, size: usize) -> Option<*mut u8> {
        if size == 0 || size > lohalloc_core::BUDDY_MAX {
            return None;
        }
        let order = order_for(size)?;
        let block = self.alloc_order(order)?;
        Some(block)
    }

    /// Free a block previously returned by `alloc` with the same `size`.
    ///
    /// # Safety
    /// `ptr` must have been returned by `alloc` with a size rounding to the same
    /// order, and not double-freed.
    pub unsafe fn dealloc(&mut self, ptr: *mut u8, size: usize) {
        let order = match order_for(size) {
            Some(o) => o,
            None => {
                debug_assert!(false, "buddy dealloc size out of range");
                return;
            }
        };
        unsafe { self.free_order(ptr, order) };
    }

    /// Number of backing regions (leak accounting in tests).
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    // ---- internals --------------------------------------------------------

    /// Allocate a block of the given order, splitting larger blocks as needed.
    fn alloc_order(&mut self, order: usize) -> Option<*mut u8> {
        debug_assert!(order <= MAX_ORDER);

        // Find the smallest order >= `order` with a free block.
        let mut o = order;
        while o <= MAX_ORDER && self.free_lists[o].is_none() {
            o += 1;
        }
        if o > MAX_ORDER {
            // Nothing big enough: map a fresh region.
            self.refill(order)?;
            return self.alloc_order(order);
        }

        // Pop a block from order `o`.
        let block = self.free_lists[o].take().expect("just checked non-None");
        let next = unsafe { (*block).next };
        self.free_lists[o] = next;

        // Split down to `order`, pushing the buddy of each split onto the
        // appropriate free list.
        let b = block as *mut u8;
        while o > order {
            o -= 1;
            let buddy = b as usize + (block_size(o));
            unsafe { self.push_free(buddy as *mut u8, o) };
        }
        Some(b)
    }

    /// Free a block of `order`, attempting to coalesce with its buddy up the
    /// order ladder.
    ///
    /// # Safety
    /// `ptr` must point to a block of the given order that was previously
    /// allocated and not yet freed.
    unsafe fn free_order(&mut self, ptr: *mut u8, order: usize) {
        let mut p = ptr;
        let mut o = order;
        while o < MAX_ORDER {
            let (base, buddy) = buddy_pair(p, o);
            let bs = block_size(o);

            // Stop if the block isn't aligned to its order's block size (shouldn't
            // happen for well-formed inputs, but prevents corrupting free lists).
            if (p as usize) & (bs - 1) != 0 {
                break;
            }

            // Critical: do NOT coalesce across region boundaries. The buddy
            // computed by XOR-ing the block-size bit may land in a *different*
            // mmap region if `base + 2*bs` exceeds the current region. We
            // must check that both halves of the pair are within the same
            // region. Without this, `remove_free` would remove a node from
            // another region's free list, corrupting it and creating cycles.
            let pair_base = base as usize;
            let pair_top = pair_base + 2 * bs;
            if !self.same_region_range(pair_base, pair_top) {
                break;
            }

            // Try to remove the buddy from its free list. If it's free, coalesce.
            if let Some(removed) = unsafe { self.remove_free(buddy, o) } {
                debug_assert_eq!(removed, buddy, "remove_free returned wrong node");
                p = base;
                o += 1;
            } else {
                break;
            }
        }
        unsafe { self.push_free(p, o) };
    }

    /// Check whether `[range_base, range_top)` fits entirely within a single
    /// backing region. Used by coalescing to prevent buddy arithmetic from
    /// crossing mmap boundaries.
    fn same_region_range(&self, range_base: usize, range_top: usize) -> bool {
        self.regions.iter().any(|r| {
            let rb = r.base as usize;
            let rt = rb + r.bytes;
            range_base >= rb && range_top <= rt
        })
    }

    /// Push a block onto the free list for `order`.
    ///
    /// # Safety
    /// `ptr` must point to a block of the given order that is not currently on
    /// any free list.
    unsafe fn push_free(&mut self, ptr: *mut u8, order: usize) {
        let node = ptr as *mut FreeNode;
        unsafe {
            (*node).next = self.free_lists[order].take();
        }
        self.free_lists[order] = Some(node);
    }

    /// Remove and return a specific block from the free list for `order`, if
    /// present (used by coalescing to take the buddy out of circulation).
    ///
    /// # Safety
    /// `ptr` must point to a block of the given order.
    unsafe fn remove_free(&mut self, ptr: *mut u8, order: usize) -> Option<*mut u8> {
        let target = ptr as *mut FreeNode;
        let head = self.free_lists[order].take();
        let mut cur = head;
        let mut prev: Option<*mut FreeNode> = None;
        let mut result: Option<*mut u8> = None;
        let mut removed = false;
        while let Some(node) = cur {
            let next = unsafe { (*node).next };
            if node == target {
                result = Some(node as *mut u8);
                unsafe {
                    (*node).next = None;
                }
                cur = next;
                removed = true;
                break;
            }
            prev = Some(node);
            cur = next;
        }
        if removed {
            if let Some(prev) = prev {
                unsafe { (*prev).next = cur };
                self.free_lists[order] = head;
            } else {
                self.free_lists[order] = cur;
            }
        } else {
            self.free_lists[order] = head;
        }
        result
    }

    /// Map a fresh region large enough to satisfy order `order` and place its
    /// single top-order block on the free list. The region size is rounded up
    /// to a power of two and to a whole number of pages, and the mapping is
    /// aligned to that size so buddy arithmetic is exact.
    fn refill(&mut self, order: usize) -> Option<()> {
        let block = block_size(order);
        // Round up to the next power of two >= one page and >= the block.
        let region_bytes = {
            let page = system::page_size();
            let need = block.max(page);
            round_up_pow2(need)
        };
        let region = system::alloc_pages(region_bytes, region_bytes)?;
        let base = region.as_ptr();
        self.regions.push(Region {
            base,
            bytes: region_bytes,
            _mapping: region,
        });
        unsafe { self.push_free(base, order_index(region_bytes)) };
        Some(())
    }
}

impl Drop for Buddy {
    fn drop(&mut self) {
        // Regions release their mappings via their own Drop; nothing to do
        // for the free-list nodes (they live inside the mapped memory).
    }
}

// ---- free functions --------------------------------------------------------

/// Size of a block at `order` (bytes).
fn block_size(order: usize) -> usize {
    MIN_BLOCK << order
}

/// Order needed to satisfy a request of `size` bytes (ceil to power of two,
/// clamp to `MAX_ORDER`).
fn order_for(size: usize) -> Option<usize> {
    if size == 0 || size > lohalloc_core::BUDDY_MAX {
        return None;
    }
    let bs = round_up_pow2(size).max(MIN_BLOCK);
    let mut o = 0;
    let mut s = MIN_BLOCK;
    while s < bs {
        s <<= 1;
        o += 1;
    }
    if o > MAX_ORDER {
        return None;
    }
    Some(o)
}

/// Order whose block size equals `bytes` (assuming `bytes` is a power of two >=
/// `MIN_BLOCK`).
fn order_index(bytes: usize) -> usize {
    let mut o = 0;
    let mut s = MIN_BLOCK;
    while s < bytes {
        s <<= 1;
        o += 1;
    }
    debug_assert_eq!(s, bytes, "order_index given non power-of-two");
    o
}

/// Given a pointer and order, return `(pair_base, buddy)` where `pair_base` is
/// the lower address of the buddy pair (the base of the coalesced block) and
/// `buddy` is the partner block's pointer (the block we test for freeness). The
/// buddy address is computed by toggling the bit corresponding to the block
/// size — the classic buddy arithmetic.
fn buddy_pair(ptr: *mut u8, order: usize) -> (*mut u8, *mut u8) {
    let bs = block_size(order);
    let addr = ptr as usize;
    // Align down to block size to get the block's base.
    let block_base = addr & !(bs - 1);
    let buddy = block_base ^ bs;
    // The coalesced pair always starts at the lower address.
    let pair_base = block_base.min(buddy);
    (pair_base as *mut u8, buddy as *mut u8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lohalloc_core::is_aligned;

    #[test]
    fn order_for_basic() {
        assert_eq!(order_for(1), Some(0)); // rounds to MIN_BLOCK
        assert_eq!(order_for(MIN_BLOCK), Some(0));
        assert_eq!(order_for(MIN_BLOCK + 1), Some(1));
        assert_eq!(order_for(0), None);
        assert_eq!(order_for(lohalloc_core::BUDDY_MAX + 1), None);
    }

    #[test]
    fn alloc_and_free_roundtrip() {
        let mut b = Buddy::new();
        let p = b.alloc(100).expect("alloc 100");
        assert!(!p.is_null());
        assert!(is_aligned(p as usize, MIN_ALIGN));
        unsafe { core::ptr::write_bytes(p, 0x7E, 100) };
        unsafe { b.dealloc(p, 100) };
    }

    #[test]
    fn split_then_coalesce() {
        // Alloc two blocks of the same order, free them both, then verify we
        // can satisfy a single block of the next order without a new region.
        let mut b = Buddy::new();
        let before = b.region_count();
        let p1 = b.alloc(64).expect("alloc 64 #1");
        let p2 = b.alloc(64).expect("alloc 64 #2");
        assert!(b.region_count() > before, "expected a refill");
        unsafe { b.dealloc(p1, 64) };
        unsafe { b.dealloc(p2, 64) };
        // Coalesced block should now satisfy a 128-byte request without mapping.
        let before2 = b.region_count();
        let p3 = b.alloc(128).expect("alloc 128 after coalesce");
        assert_eq!(
            b.region_count(),
            before2,
            "coalesce should have reused memory, no new region"
        );
        unsafe { b.dealloc(p3, 128) };
    }

    #[test]
    fn neighbors_do_not_overlap() {
        let mut b = Buddy::new();
        let p1 = b.alloc(64).expect("a1");
        let p2 = b.alloc(64).expect("a2");
        let a1 = p1 as usize;
        let a2 = p2 as usize;
        let bs = round_up_pow2(64).max(MIN_BLOCK);
        // Either they are adjacent (no overlap) or separated; either way ranges
        // [a1, a1+bs) and [a2, a2+bs) must not intersect.
        let lo = a1.min(a2);
        let hi = a1.max(a2);
        assert!(hi - lo >= bs, "blocks overlap: {a1:x} {a2:x} bs={bs}");
        unsafe { b.dealloc(p1, 64) };
        unsafe { b.dealloc(p2, 64) };
    }

    #[test]
    fn min_block_alignment() {
        let mut b = Buddy::new();
        let p = b.alloc(1).expect("alloc 1");
        assert!(is_aligned(p as usize, MIN_BLOCK));
        unsafe { b.dealloc(p, 1) };
    }

    #[test]
    fn rejects_oversize() {
        let mut b = Buddy::new();
        assert!(b.alloc(0).is_none());
        assert!(b.alloc(lohalloc_core::BUDDY_MAX + 1).is_none());
    }

    #[test]
    fn bounded_regions_under_fixed_live_set() {
        let mut b = Buddy::new();
        let mut live = Vec::new();
        for _ in 0..5 {
            for _ in 0..4 {
                live.push(b.alloc(64).expect("alloc"));
            }
            for p in live.drain(..) {
                unsafe { b.dealloc(p, 64) };
            }
        }
        assert!(b.region_count() < 32, "region_count = {}", b.region_count());
    }

    #[test]
    fn heavy_churn_no_cycle() {
        // Heavy alloc/free churn with interleaved sizes. Previously triggered
        // the remove_free cycle guard. Now coalescing stops at region boundaries.
        let mut b = Buddy::new();
        let mut pool: Vec<(*mut u8, usize)> = Vec::new();
        let sizes = [16, 32, 64, 128, 256, 512, 1024];
        for _round in 0..100 {
            for &sz in &sizes {
                if let Some(p) = b.alloc(sz) {
                    pool.push((p, sz));
                }
            }
            let half = pool.len() / 2;
            for (p, sz) in pool.drain(..half) {
                unsafe { b.dealloc(p, sz) };
            }
        }
        for (p, sz) in pool.drain(..) {
            unsafe { b.dealloc(p, sz) };
        }
        assert!(b.region_count() < 64, "region_count = {}", b.region_count());
    }
}
