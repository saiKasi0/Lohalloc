//! Buddy sub-allocator (variable-size, power-of-two blocks).
//!
//! A classic binary-buddy allocator backed by fixed-size regions from the
//! System Fallback. Every region is exactly `REGION_BYTES` (4 MiB), aligned
//! to `REGION_BYTES`, and carved into `REGION_BYTES / BUDDY_MAX` top-order
//! blocks on arrival — so one `mmap` serves 4×1 MiB, 64×64 KiB, etc.,
//! instead of the original one-mmap-per-top-order-block scheme that caused
//! visible syscall storms on buddy-heavy workloads. Blocks are recursively
//! split down to `MIN_BLOCK` (= `MIN_ALIGN`); free blocks of the same order
//! are linked on a per-order free list. On `dealloc` a block's buddy is
//! computed by XOR-ing the appropriate bit; if the buddy is free the pair
//! coalesces up to the next order, and so on.
//!
//! Because regions are `REGION_BYTES`-aligned and coalescing is capped at
//! `MAX_ORDER` (1 MiB blocks), a buddy pair can never span two regions: the
//! largest pair is 1 MiB aligned to 1 MiB, which always sits inside one
//! 4 MiB-aligned 4 MiB region. That invariant replaced the old per-step
//! `same_region_range` linear scan over all regions.
//!
//! ## Known limitation: header order-inflation
//!
//! The shim prepends a 48-byte `Header` to every allocation, so a request
//! for an exact power of two (e.g. 32 KiB) arrives here as 32 KiB + 48 and
//! rounds up to the *next* order (64 KiB block) — 2× internal fragmentation
//! for pow2-sized requests, the worst case. This is inherent to inline
//! headers + pow2 buddy blocks; fixing it would require out-of-band
//! metadata (a different allocator design). The multi-block regions above
//! remove the *syscall* cost of that inflation; the memory cost remains and
//! is documented in the Phase 6 bench notes.

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

/// Fixed size (and alignment) of every backing region: 4 MiB = 4 top-order
/// (1 MiB) blocks per `mmap`. Must be a power-of-two multiple of
/// `BUDDY_MAX` so (a) whole top-order blocks tile it exactly and (b) the
/// region-alignment argument in `free_order`'s coalescing invariant holds.
const REGION_BYTES: usize = 4 * lohalloc_core::BUDDY_MAX;

/// log2(MIN_BLOCK), so a block index inside a region is
/// `(addr - base) >> (MIN_BLOCK_SHIFT + order)`.
const MIN_BLOCK_SHIFT: usize = MIN_BLOCK.trailing_zeros() as usize;

/// `ORDER_BIT_OFFSET[o]` is the first bitmap bit belonging to order `o`;
/// the entry at `NUM_ORDERS` is the total bit count. Order `o` owns one bit
/// per possible block position: `REGION_BYTES >> (MIN_BLOCK_SHIFT + o)`.
/// Total ≈ 2 × (REGION_BYTES / MIN_BLOCK) bits = 64 KiB per 4 MiB region
/// (1.5% overhead).
const ORDER_BIT_OFFSET: [usize; NUM_ORDERS + 1] = {
    let mut table = [0usize; NUM_ORDERS + 1];
    let mut o = 0;
    while o < NUM_ORDERS {
        table[o + 1] = table[o] + (REGION_BYTES >> (MIN_BLOCK_SHIFT + o));
        o += 1;
    }
    table
};

/// `u64` words needed for one region's free bitmap.
const BITMAP_WORDS: usize = ORDER_BIT_OFFSET[NUM_ORDERS].div_ceil(64);

/// Intrusive doubly-linked free-list node living at the head of each free
/// block. Doubly-linked (unlike the original singly-linked version) so
/// coalescing can unlink a buddy in O(1) instead of walking the whole
/// order's list to find it — under churn that walk was O(free blocks) per
/// free. `prev`/`next` are null-sentinel raw pointers rather than
/// `Option<*mut _>` because `Option<*mut T>` has no niche: two of them
/// would be 32 bytes, overflowing the 16-byte (`MIN_BLOCK`) minimum block.
#[repr(C)]
struct FreeNode {
    prev: *mut FreeNode,
    next: *mut FreeNode,
}

// A FreeNode must fit in the smallest block the buddy can hand out.
const _: () = assert!(core::mem::size_of::<FreeNode>() <= MIN_BLOCK);

/// A buddy region: a contiguous, page-backed run of `REGION_BYTES` bytes
/// whose base is aligned to `REGION_BYTES` (so buddy arithmetic works
/// cleanly), plus a free bitmap with one bit per (order, block position).
/// The bitmap is what makes "is my buddy free?" O(1): coalescing tests the
/// buddy's bit instead of searching the free list for its node.
struct Region {
    base: *mut u8,
    bytes: usize,
    /// One bit per (order, block position); see `ORDER_BIT_OFFSET`. Heap
    /// allocation happens inside `refill` while the buddy lock is held —
    /// under a global-allocator install that nested allocation takes the
    /// existing `IN_ALLOC` bypass (one extra mmap per 4 MiB region).
    bitmap: Vec<u64>,
    _mapping: system::Mapping,
}

unsafe impl Send for Region {}

impl Region {
    /// Bitmap position for the block at `addr` (must lie in this region)
    /// at `order`.
    #[inline]
    fn bit_pos(&self, addr: usize, order: usize) -> usize {
        debug_assert!(order < NUM_ORDERS);
        let idx = (addr - self.base as usize) >> (MIN_BLOCK_SHIFT + order);
        ORDER_BIT_OFFSET[order] + idx
    }

    #[inline]
    fn set_free_bit(&mut self, addr: usize, order: usize) {
        let pos = self.bit_pos(addr, order);
        self.bitmap[pos / 64] |= 1u64 << (pos % 64);
    }

    #[inline]
    fn clear_free_bit(&mut self, addr: usize, order: usize) {
        let pos = self.bit_pos(addr, order);
        self.bitmap[pos / 64] &= !(1u64 << (pos % 64));
    }

    #[inline]
    fn is_free_bit(&self, addr: usize, order: usize) -> bool {
        let pos = self.bit_pos(addr, order);
        (self.bitmap[pos / 64] >> (pos % 64)) & 1 == 1
    }
}

/// Number of buddy orders (compile-time constant so `new()` can be const).
const NUM_ORDERS: usize = MAX_ORDER + 1;

/// One buddy allocator instance.
pub struct Buddy {
    /// `free_lists[o]` is the head of the doubly-linked free list for
    /// blocks of order `o` (null = empty).
    free_lists: [*mut FreeNode; NUM_ORDERS],
    /// Owning regions, sorted by base address. Held alive so the memory
    /// stays mapped.
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
            free_lists: [core::ptr::null_mut(); NUM_ORDERS],
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
        while o <= MAX_ORDER && self.free_lists[o].is_null() {
            o += 1;
        }
        if o > MAX_ORDER {
            // Nothing big enough: map a fresh region.
            self.refill()?;
            return self.alloc_order(order);
        }

        // Pop the head block from order `o` and clear its free bit.
        let block = self.free_lists[o];
        unsafe { self.unlink(block, o) };
        let region_idx = self
            .region_index_containing(block as usize)
            .expect("free-list node must lie inside a region");
        self.regions[region_idx].clear_free_bit(block as usize, o);

        // Split down to `order`, pushing the buddy of each split onto the
        // appropriate free list. Every split buddy lives in the same region
        // as `block`, so `region_idx` stays valid throughout.
        let b = block as *mut u8;
        while o > order {
            o -= 1;
            let buddy = b as usize + (block_size(o));
            unsafe { self.push_free(buddy as *mut u8, o, region_idx) };
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
        // Region-boundary safety is structural rather than checked per
        // coalesce step: every region is REGION_BYTES-sized and
        // REGION_BYTES-aligned, and coalescing is capped at `o < MAX_ORDER`,
        // so the largest possible pair is one BUDDY_MAX block aligned to
        // BUDDY_MAX — which always lies entirely inside the single
        // REGION_BYTES-aligned region the freed block came from. Two
        // adjacent regions can't interfere either: `buddy_pair`'s XOR never
        // flips a bit at or above the REGION_BYTES boundary for
        // `2 * block_size(o) <= REGION_BYTES`. The old implementation
        // re-verified this per step with a linear scan over all regions
        // (`same_region_range`) — O(regions) per coalesce level. The same
        // invariant means one region lookup here covers every buddy the
        // coalescing loop will ever test.
        let Some(region_idx) = self.region_index_containing(ptr as usize) else {
            debug_assert!(
                false,
                "free_order called with pointer outside every buddy region"
            );
            return; // release mode: leak rather than corrupt the lists
        };
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

            // O(1): test the buddy's free bit; if free, unlink it and merge.
            if unsafe { self.remove_free(buddy, o, region_idx) } {
                p = base;
                o += 1;
            } else {
                break;
            }
        }
        unsafe { self.push_free(p, o, region_idx) };
    }

    /// Push a block onto the head of the free list for `order` and set its
    /// free bit.
    ///
    /// # Safety
    /// `ptr` must point to a block of the given order, inside
    /// `self.regions[region_idx]`, that is not currently on any free list.
    unsafe fn push_free(&mut self, ptr: *mut u8, order: usize, region_idx: usize) {
        let node = ptr as *mut FreeNode;
        let head = self.free_lists[order];
        unsafe {
            (*node).prev = core::ptr::null_mut();
            (*node).next = head;
            if !head.is_null() {
                (*head).prev = node;
            }
        }
        self.free_lists[order] = node;
        self.regions[region_idx].set_free_bit(ptr as usize, order);
    }

    /// Unlink `node` from the free list for `order` (does not touch bits).
    ///
    /// # Safety
    /// `node` must currently be linked on `free_lists[order]`.
    unsafe fn unlink(&mut self, node: *mut FreeNode, order: usize) {
        unsafe {
            let prev = (*node).prev;
            let next = (*node).next;
            if prev.is_null() {
                self.free_lists[order] = next;
            } else {
                (*prev).next = next;
            }
            if !next.is_null() {
                (*next).prev = prev;
            }
        }
    }

    /// If the block at `ptr`/`order` is free, unlink it from its free list,
    /// clear its bit, and return `true` — all O(1) via the region bitmap
    /// (the old singly-linked version walked the whole order's list to find
    /// the node).
    ///
    /// # Safety
    /// `ptr` must point to a block-aligned address of the given order
    /// inside `self.regions[region_idx]`.
    unsafe fn remove_free(&mut self, ptr: *mut u8, order: usize, region_idx: usize) -> bool {
        if !self.regions[region_idx].is_free_bit(ptr as usize, order) {
            return false;
        }
        self.regions[region_idx].clear_free_bit(ptr as usize, order);
        unsafe { self.unlink(ptr as *mut FreeNode, order) };
        true
    }

    /// Map a fresh `REGION_BYTES` region (aligned to `REGION_BYTES` so buddy
    /// arithmetic is exact) and carve it into top-order blocks on the free
    /// list. One `mmap` now serves `REGION_BYTES / BUDDY_MAX` top-order
    /// blocks — the previous scheme mapped one region *per* top-order block,
    /// which made buddy-heavy workloads syscall-bound (hyperfine showed
    /// 24 ms outliers from mmap storms).
    fn refill(&mut self) -> Option<()> {
        let region = system::alloc_pages(REGION_BYTES, REGION_BYTES)?;
        let base = region.as_ptr();
        // Keep `regions` sorted by base so lookups can binary-search. Refill
        // is rare now (once per 4 MiB), so the O(n) insert shift is noise.
        let idx = self
            .regions
            .partition_point(|r| (r.base as usize) < base as usize);
        self.regions.insert(
            idx,
            Region {
                base,
                bytes: REGION_BYTES,
                bitmap: vec![0u64; BITMAP_WORDS],
                _mapping: region,
            },
        );
        let top = block_size(MAX_ORDER);
        let mut off = 0;
        while off < REGION_BYTES {
            unsafe { self.push_free(base.add(off), MAX_ORDER, idx) };
            off += top;
        }
        Some(())
    }

    /// Binary-search the sorted `regions` for the index of the one
    /// containing `addr`.
    fn region_index_containing(&self, addr: usize) -> Option<usize> {
        let idx = self
            .regions
            .partition_point(|r| (r.base as usize) + r.bytes <= addr);
        self.regions
            .get(idx)
            .filter(|r| (r.base as usize) <= addr && addr < (r.base as usize) + r.bytes)
            .map(|_| idx)
    }
}

#[cfg(test)]
impl Buddy {
    /// Test-only consistency check: every free-list node's `prev` links are
    /// coherent, every node's bit is set in its region's bitmap, and the
    /// total number of set bits equals the total number of list nodes (so
    /// no bit is set without a node and vice versa).
    fn check_invariants(&self) {
        let mut list_nodes = 0usize;
        for (o, &head) in self.free_lists.iter().enumerate() {
            let mut node = head;
            let mut prev: *mut FreeNode = core::ptr::null_mut();
            while !node.is_null() {
                unsafe {
                    assert_eq!((*node).prev, prev, "prev link broken at order {o}");
                }
                let idx = self
                    .region_index_containing(node as usize)
                    .expect("free node outside every region");
                assert!(
                    self.regions[idx].is_free_bit(node as usize, o),
                    "free-list node at order {o} has no bitmap bit"
                );
                list_nodes += 1;
                prev = node;
                node = unsafe { (*node).next };
            }
        }
        let set_bits: usize = self
            .regions
            .iter()
            .map(|r| {
                r.bitmap
                    .iter()
                    .map(|w| w.count_ones() as usize)
                    .sum::<usize>()
            })
            .sum();
        assert_eq!(set_bits, list_nodes, "bitmap bits != free-list nodes");
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
///
/// `pub(crate)` so `lib.rs` can compute internal-fragmentation percentages
/// (actual block size vs. requested size) without touching `Buddy`'s
/// internal free-list state — see `fragmentation_pct_for`.
pub(crate) fn block_size(order: usize) -> usize {
    MIN_BLOCK << order
}

/// Order needed to satisfy a request of `size` bytes (ceil to power of two,
/// clamp to `MAX_ORDER`).
pub(crate) fn order_for(size: usize) -> Option<usize> {
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
        // A handful of 64-byte blocks fits comfortably in one 4 MiB region.
        assert_eq!(b.region_count(), 1, "region_count = {}", b.region_count());
    }

    #[test]
    fn refill_batches_blocks() {
        // One 4 MiB region holds REGION_BYTES / BUDDY_MAX top-order blocks;
        // only the (N+1)-th live top-order block forces a second mmap.
        let per_region = REGION_BYTES / lohalloc_core::BUDDY_MAX;
        let top = lohalloc_core::BUDDY_MAX;
        let mut b = Buddy::new();
        let mut live = Vec::new();
        for _ in 0..per_region {
            live.push(b.alloc(top).expect("top-order alloc"));
        }
        assert_eq!(
            b.region_count(),
            1,
            "first {per_region} blocks share one region"
        );
        live.push(b.alloc(top).expect("overflow top-order alloc"));
        assert_eq!(b.region_count(), 2, "next block needs a second region");
        for p in live.drain(..) {
            unsafe { b.dealloc(p, top) };
        }
    }

    #[test]
    fn coalesce_across_top_blocks_stops_at_max_order() {
        // Free two adjacent top-order blocks: coalescing must stop at
        // MAX_ORDER (they stay two 1 MiB free blocks, never a 2 MiB one),
        // and a subsequent top-order alloc must reuse without a refill.
        let top = lohalloc_core::BUDDY_MAX;
        let mut b = Buddy::new();
        let p1 = b.alloc(top).expect("top #1");
        let p2 = b.alloc(top).expect("top #2");
        assert_eq!(b.region_count(), 1);
        unsafe {
            b.dealloc(p1, top);
            b.dealloc(p2, top);
        }
        let p3 = b.alloc(top).expect("top after frees");
        assert_eq!(b.region_count(), 1, "reuse must not map a new region");
        unsafe { b.dealloc(p3, top) };
    }

    #[test]
    fn remove_middle_of_free_list() {
        // Force `remove_free` to unlink a node that is neither head nor
        // tail of its order's free list during coalescing, then verify the
        // list survives by allocating through it again.
        let mut b = Buddy::new();
        // Six 64-byte blocks: freeing them in a scrambled order populates
        // the order-2 free list in non-address order, so a later coalesce
        // has to unlink from the middle.
        let ps: Vec<_> = (0..6)
            .map(|i| b.alloc(64).unwrap_or_else(|| panic!("a{i}")))
            .collect();
        unsafe {
            b.dealloc(ps[1], 64);
            b.dealloc(ps[4], 64);
            b.dealloc(ps[2], 64); // may coalesce with ps[1]/ps[3]'s buddies
            b.dealloc(ps[0], 64);
            b.dealloc(ps[5], 64);
            b.dealloc(ps[3], 64);
        }
        // Everything freed and (partially) coalesced: allocate a spread of
        // sizes to walk the surviving lists.
        let q1 = b.alloc(64).expect("post-churn 64");
        let q2 = b.alloc(128).expect("post-churn 128");
        let q3 = b.alloc(64).expect("post-churn 64 #2");
        unsafe {
            b.dealloc(q1, 64);
            b.dealloc(q2, 128);
            b.dealloc(q3, 64);
        }
    }

    #[test]
    fn bitmap_matches_free_lists() {
        // Drive a mixed alloc/free pattern and assert after every phase
        // that the region bitmaps and the doubly-linked free lists agree
        // exactly (same membership, coherent prev links).
        let mut b = Buddy::new();
        b.check_invariants(); // empty allocator is trivially consistent
        let sizes = [16, 64, 200, 1024, 8192, 65536];
        let mut live = Vec::new();
        for &sz in &sizes {
            live.push((b.alloc(sz).expect("alloc"), sz));
            b.check_invariants();
        }
        // Free every other one, then the rest — exercises both plain frees
        // and multi-level coalesces.
        for (i, (p, sz)) in live.iter().enumerate() {
            if i % 2 == 0 {
                unsafe { b.dealloc(*p, *sz) };
                b.check_invariants();
            }
        }
        for (i, (p, sz)) in live.iter().enumerate() {
            if i % 2 == 1 {
                unsafe { b.dealloc(*p, *sz) };
                b.check_invariants();
            }
        }
    }

    #[test]
    fn heavy_churn_no_cycle() {
        // Heavy alloc/free churn with interleaved sizes. Previously triggered
        // the remove_free cycle guard. Now coalescing stops at region boundaries.
        let mut b = Buddy::new();
        let mut pool: Vec<(*mut u8, usize)> = Vec::new();
        let sizes = [16, 32, 64, 128, 256, 512, 1024];
        for round in 0..100 {
            for &sz in &sizes {
                if let Some(p) = b.alloc(sz) {
                    pool.push((p, sz));
                }
            }
            let half = pool.len() / 2;
            for (p, sz) in pool.drain(..half) {
                unsafe { b.dealloc(p, sz) };
            }
            if round % 10 == 0 {
                b.check_invariants();
            }
        }
        for (p, sz) in pool.drain(..) {
            unsafe { b.dealloc(p, sz) };
        }
        // All sizes in this churn are <= 1024 bytes; the whole run fits in
        // a couple of 4 MiB regions at most.
        assert!(b.region_count() <= 2, "region_count = {}", b.region_count());
    }
}
