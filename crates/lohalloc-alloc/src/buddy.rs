//! Buddy sub-allocator (variable-size, power-of-two blocks).
//!
//! A binary-buddy allocator backed by fixed-size regions from the System
//! Fallback. Every region is exactly `REGION_BYTES` (4 MiB), aligned to
//! `REGION_BYTES`, and carved into `REGION_BYTES / BUDDY_MAX` top-order
//! blocks on arrival — so one `mmap` serves 4×1 MiB, 64×64 KiB, etc.,
//! instead of the original one-mmap-per-top-order-block scheme that caused
//! visible syscall storms on buddy-heavy workloads. Blocks are recursively
//! split down to `MIN_BLOCK` (= `MIN_ALIGN`).
//!
//! ## Three-tier free tracking + merge-on-spill coalescing
//!
//! The per-region bitmap (one bit per (order, block)) is the source of
//! truth for freeness. On top sit two access tiers: a per-order
//! out-of-line pointer cache (the hot tier — freeing/allocating through it
//! never reads or writes the block's own cold memory) and the intrusive
//! doubly-linked lists (overflow tier). Frees do **no** merging on the hot
//! path; merging happens *only* incrementally, in [`Buddy::merge_drain_row`],
//! when a cache row fills (amortized: one bounded drain of half the row per
//! `ORDER_CACHE/2` frees at that order). The eager-merge-then-resplit
//! metadata thrash of a naive design (measured: 3.75% LL miss rate vs
//! jemalloc's 0.46% on the churn benchmark) is gone, and under steady churn
//! frees at an order feed the next alloc at that order directly.
//!
//! An allocation that misses every order does **not** trigger any kind of
//! full-table merge sweep before `refill()` — two designs were tried and
//! rejected for the same reason (both regressed the 50k-op buddy benchmark
//! from ~120 ms to 11-12 **s**): sweeping every region's whole bitmap, and
//! sweeping every order's cache row. Both looked bounded in isolation (one
//! per allocation miss) but the buddy benchmark's live set genuinely grows
//! over the run (net allocation, not just churn — freeing is FIFO against
//! cycling size classes, so most misses are real growth a merge can't
//! satisfy anyway), so "call it on every miss" means "call it on a large
//! fraction of all 50k operations" — that IS the hot path, not a rare
//! last resort. Do not reintroduce a merge sweep on the allocation-miss
//! path; the incremental drain above is the only merge trigger.
//!
//! Because regions are `REGION_BYTES`-aligned, every block size divides
//! `REGION_BYTES`, and merging is capped at `MAX_ORDER` (1 MiB blocks), a
//! buddy pair never spans two regions — `addr ^ block_size(o)` computed on
//! the absolute address stays inside the block's own region.
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
    /// One bit per (order, block position); see `ORDER_BIT_OFFSET`. Heap
    /// allocation happens inside `refill` while the buddy lock is held —
    /// under a global-allocator install that nested allocation takes the
    /// existing `IN_ALLOC` bypass (one extra mmap per 4 MiB region).
    /// (No `bytes` field: every region is exactly `REGION_BYTES`.)
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

    /// Is the block at (`addr`, `order`) free? Used by the merge path to
    /// test a buddy's freeness in O(1) (the alternative — searching the
    /// free tiers — is exactly what the bitmap exists to avoid).
    #[inline]
    fn is_free_bit(&self, addr: usize, order: usize) -> bool {
        let pos = self.bit_pos(addr, order);
        (self.bitmap[pos / 64] >> (pos % 64)) & 1 == 1
    }
}

/// Number of buddy orders (compile-time constant so `new()` can be const).
const NUM_ORDERS: usize = MAX_ORDER + 1;

/// Per-order out-of-line pointer-stack capacity (see `cache` field).
const ORDER_CACHE: usize = 32;

/// One buddy allocator instance.
pub struct Buddy {
    /// `free_lists[o]` is the head of the doubly-linked free list for
    /// blocks of order `o` (null = empty). The *overflow* tier — the hot
    /// path goes through `cache` and never touches block memory.
    free_lists: [*mut FreeNode; NUM_ORDERS],
    /// Per-order out-of-line pointer stacks: the hot free/alloc tier.
    /// Freed block addresses are recorded here (plus their bitmap bit)
    /// WITHOUT writing a `FreeNode` into the freed block itself — under
    /// churn those blocks are cache-cold, and touching them on every
    /// free/alloc was a measured dominant source of buddy's 3.75% LL miss
    /// rate (jemalloc: 0.46%). Only cache overflow spills to the intrusive
    /// lists (touching blocks once, amortized).
    cache: [[*mut u8; ORDER_CACHE]; NUM_ORDERS],
    /// Live entry count per `cache` row.
    cache_len: [u8; NUM_ORDERS],
    /// Owning regions, sorted by base address. Held alive so the memory
    /// stays mapped.
    regions: Vec<Region>,
    /// Last-hit index into `regions` — churn benches touch 1-2 regions, so
    /// this makes the per-op region lookup one compare instead of a binary
    /// search. Verified against the masked base on every use, so a stale
    /// value is a miss, never a wrong answer.
    last_region: usize,
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
            cache: [[core::ptr::null_mut(); ORDER_CACHE]; NUM_ORDERS],
            cache_len: [0; NUM_ORDERS],
            regions: Vec::new(),
            last_region: 0,
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

    /// Allocate a block of the given order, splitting larger blocks as
    /// needed. Free blocks are found in this priority: per-order pointer
    /// cache (no block-memory touch) → intrusive list → lazy coalesce of
    /// everything freed so far → fresh region.
    fn alloc_order(&mut self, order: usize) -> Option<*mut u8> {
        debug_assert!(order <= MAX_ORDER);

        loop {
            // Find the smallest order >= `order` with a free block.
            let mut o = order;
            while o <= MAX_ORDER && self.cache_len[o] == 0 && self.free_lists[o].is_null() {
                o += 1;
            }
            if o > MAX_ORDER {
                // Nothing big enough anywhere. Do NOT sweep every order's
                // cache here: under a workload with a genuinely growing
                // live set (net allocation, not just churn), most misses
                // are real growth that a sweep can't satisfy anyway, yet
                // the sweep's cost scales with total cached entries across
                // every order — calling it on every such miss measured as
                // quadratic overall (50k-op buddy benchmark: ~120 ms with
                // no emergency sweep vs 11+ s with one). Incremental
                // merging already happens for free in `stash_free`/
                // `free_order` whenever a row hits `ORDER_CACHE` capacity
                // (see `merge_drain_row`), which is what keeps steady-churn
                // workloads (free-then-realloc at the same order) merged
                // as far as they can go — a miss here just means genuine
                // growth, so map more memory.
                self.refill()?;
                continue;
            }

            // Pop a block at order `o`: cache first (block memory untouched),
            // intrusive list otherwise.
            let block: *mut u8 = if self.cache_len[o] > 0 {
                let n = self.cache_len[o] - 1;
                self.cache_len[o] = n;
                self.cache[o][n as usize]
            } else {
                let head = self.free_lists[o];
                unsafe { self.unlink(head, o) };
                head as *mut u8
            };
            let region_idx = self
                .region_index_of(block as usize)
                .expect("free block must lie inside a region");
            self.regions[region_idx].clear_free_bit(block as usize, o);

            // Split down to `order`. Split buddies go through the cache
            // (they're brand-new metadata — no reason to touch their
            // memory), spilling to the intrusive list only on overflow.
            // Every split buddy lives in the same region as `block`.
            let b = block as usize;
            while o > order {
                o -= 1;
                let buddy = (b + block_size(o)) as *mut u8;
                self.stash_free(buddy, o, region_idx);
            }
            return Some(b as *mut u8);
        }
    }

    /// Free a block of `order`. Deferred (lazy) coalescing: record the
    /// block as free (bitmap bit + pointer cache) and return — no merge
    /// walk, no writes into the freed block's cold memory on the common
    /// path. Merging happens only when the cache row overflows (see
    /// [`Self::merge_drain_row`]) — under steady churn, frees at an order
    /// feed the very next alloc at that order with zero split/merge
    /// traffic (the eager version merged on every free and re-split on the
    /// next alloc: pure metadata thrash).
    ///
    /// # Safety
    /// `ptr` must point to a block of the given order that was previously
    /// allocated and not yet freed.
    unsafe fn free_order(&mut self, ptr: *mut u8, order: usize) {
        let Some(region_idx) = self.region_index_of(ptr as usize) else {
            debug_assert!(
                false,
                "free_order called with pointer outside every buddy region"
            );
            return; // release mode: leak rather than corrupt the lists
        };
        // On a full row, merge-drain half of it BEFORE marking `ptr` free:
        // draining runs `merge_up`, and a bit-set block that is in no tier
        // yet would get `unlink`ed as a phantom list node (reading garbage
        // "FreeNode" bytes out of the just-freed block). Amortized: one
        // bounded drain per ORDER_CACHE/2 frees at this order.
        if (self.cache_len[order] as usize) == ORDER_CACHE {
            self.merge_drain_row(order, ORDER_CACHE / 2);
        }
        self.regions[region_idx].set_free_bit(ptr as usize, order);
        let n = self.cache_len[order] as usize;
        debug_assert!(n < ORDER_CACHE);
        self.cache[order][n] = ptr;
        self.cache_len[order] = n as u8 + 1;
    }

    /// Record a (bit already known-clear) block as free in the cache tier,
    /// spilling to the intrusive list on overflow. Sets the bitmap bit.
    fn stash_free(&mut self, ptr: *mut u8, order: usize, region_idx: usize) {
        self.regions[region_idx].set_free_bit(ptr as usize, order);
        if (self.cache_len[order] as usize) < ORDER_CACHE {
            self.cache[order][self.cache_len[order] as usize] = ptr;
            self.cache_len[order] += 1;
        } else {
            // Cache full: the *new* block spills straight to the list (one
            // touch) rather than evicting cache entries.
            unsafe { self.push_list(ptr, order) };
        }
    }

    /// Remove a known-free block of `order` from whichever access tier it
    /// occupies: scan the (hot, ≤`ORDER_CACHE`-entry) cache row first,
    /// swap-removing on a hit; otherwise it must be on the intrusive list
    /// — O(1) unlink.
    ///
    /// # Safety
    /// `ptr`'s bitmap bit must be set, i.e. the block is free and resident
    /// in exactly one of the two tiers at `order`.
    unsafe fn remove_free(&mut self, ptr: *mut u8, order: usize) {
        let len = self.cache_len[order] as usize;
        for i in 0..len {
            if self.cache[order][i] == ptr {
                self.cache[order][i] = self.cache[order][len - 1];
                self.cache_len[order] -= 1;
                return;
            }
        }
        unsafe { self.unlink(ptr as *mut FreeNode, order) };
    }

    /// Merge an in-hand block (bitmap bit already cleared, resident in no
    /// tier) upward while its buddy is free, removing each absorbed buddy
    /// from its tier. Returns the final (address, order) — still in hand;
    /// the caller inserts it.
    ///
    /// Region-boundary safety: regions are `REGION_BYTES`-aligned, every
    /// block size divides `REGION_BYTES`, and merging is capped below
    /// `MAX_ORDER`, so `addr ^ block_size(o)` computed on the absolute
    /// address never leaves the block's own region.
    ///
    /// # Safety
    /// `addr` must be a block of `order` inside `regions[region_idx]`,
    /// with its bit cleared and membership in no tier.
    unsafe fn merge_up(
        &mut self,
        mut addr: usize,
        mut order: usize,
        region_idx: usize,
    ) -> (usize, usize) {
        while order < MAX_ORDER {
            let buddy = addr ^ block_size(order);
            if !self.regions[region_idx].is_free_bit(buddy, order) {
                break;
            }
            unsafe { self.remove_free(buddy as *mut u8, order) };
            self.regions[region_idx].clear_free_bit(buddy, order);
            addr &= !(block_size(order + 1) - 1); // min(addr, buddy)
            order += 1;
        }
        (addr, order)
    }

    /// Drain up to `count` entries from the cache row at `order`, merging
    /// each upward and re-inserting the result: merged blocks land in their
    /// new (higher) order's tier via `stash_free`; unmerged blocks go to
    /// this order's intrusive list — never back into the row being drained.
    /// Returns `true` if any merge happened.
    ///
    /// This is the *only* place buddy merging happens — bounded per call
    /// by `count` (at most `ORDER_CACHE`), never by total free mass, and
    /// triggered only by cache-row overflow (see the module doc for why an
    /// allocation-miss-time sweep was tried and rejected instead).
    fn merge_drain_row(&mut self, order: usize, count: usize) -> bool {
        let mut merged_any = false;
        for _ in 0..count {
            let len = self.cache_len[order] as usize;
            if len == 0 {
                break;
            }
            let ptr = self.cache[order][len - 1];
            self.cache_len[order] = (len - 1) as u8;
            let Some(region_idx) = self.region_index_of(ptr as usize) else {
                debug_assert!(false, "cached free block outside every region");
                continue;
            };
            self.regions[region_idx].clear_free_bit(ptr as usize, order);
            let (addr, o) = unsafe { self.merge_up(ptr as usize, order, region_idx) };
            if o == order {
                // No merge: park on the overflow list (one cold-block
                // touch, amortized over the drain trigger).
                self.regions[region_idx].set_free_bit(addr, o);
                unsafe { self.push_list(addr as *mut u8, o) };
            } else {
                merged_any = true;
                self.stash_free(addr as *mut u8, o, region_idx);
            }
        }
        merged_any
    }

    /// Push a block onto the head of the intrusive list for `order`. Does
    /// NOT touch bitmap bits — callers manage bits (the list and cache are
    /// access tiers over the bitmap, which is the source of truth). This is
    /// the only place block memory is written on the free side, and it only
    /// runs on merge-drain spill (cache overflow).
    ///
    /// # Safety
    /// `ptr` must point to a free block of the given order that is not
    /// currently on any list or cache row.
    unsafe fn push_list(&mut self, ptr: *mut u8, order: usize) {
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

    /// Map a fresh `REGION_BYTES` region (aligned to `REGION_BYTES` so buddy
    /// arithmetic is exact) and carve it into top-order blocks on the cache
    /// tier. One `mmap` serves `REGION_BYTES / BUDDY_MAX` top-order blocks
    /// — the original scheme mapped one region *per* top-order block, which
    /// made buddy-heavy workloads syscall-bound (hyperfine showed 24 ms
    /// outliers from mmap storms).
    fn refill(&mut self) -> Option<()> {
        let region = system::alloc_pages(REGION_BYTES, REGION_BYTES)?;
        let base = region.as_ptr();
        // Keep `regions` sorted by base so lookups can binary-search. Refill
        // is rare (once per 4 MiB), so the O(n) insert shift is noise.
        let idx = self
            .regions
            .partition_point(|r| (r.base as usize) < base as usize);
        self.regions.insert(
            idx,
            Region {
                base,
                bitmap: vec![0u64; BITMAP_WORDS],
                _mapping: region,
            },
        );
        // The insert may have shifted whatever `last_region` pointed at;
        // pointing it at the new region is both correct (lookups verify the
        // base) and what the next operations will want.
        self.last_region = idx;
        let top = block_size(MAX_ORDER);
        let mut off = 0;
        while off < REGION_BYTES {
            self.stash_free(unsafe { base.add(off) }, MAX_ORDER, idx);
            off += top;
        }
        Some(())
    }

    /// Index of the region containing `addr`: mask to the region base
    /// (regions are `REGION_BYTES`-aligned and -sized, so `addr & !(RB-1)`
    /// IS the base), check the last-hit cache, then exact-match binary
    /// search. The old containment binary search ran on every free.
    fn region_index_of(&mut self, addr: usize) -> Option<usize> {
        let masked = addr & !(REGION_BYTES - 1);
        if let Some(r) = self.regions.get(self.last_region) {
            if r.base as usize == masked {
                return Some(self.last_region);
            }
        }
        let idx = self.regions.partition_point(|r| (r.base as usize) < masked);
        if self.regions.get(idx).map(|r| r.base as usize) == Some(masked) {
            self.last_region = idx;
            Some(idx)
        } else {
            None
        }
    }
}

#[cfg(test)]
impl Buddy {
    /// Test-only read-only region lookup (mirrors `region_index_of` without
    /// mutating the last-hit cache).
    fn region_index_for_test(&self, addr: usize) -> Option<usize> {
        let masked = addr & !(REGION_BYTES - 1);
        self.regions.iter().position(|r| r.base as usize == masked)
    }

    /// Test-only consistency check: every free-list node's `prev` links are
    /// coherent, every free entry (cache tier + list tier) has its bitmap
    /// bit set, and the total number of set bits equals the total number of
    /// free entries — no bit without an entry, no entry without a bit, no
    /// block in both tiers.
    fn check_invariants(&self) {
        let mut free_entries = 0usize;
        // List tier.
        for (o, &head) in self.free_lists.iter().enumerate() {
            let mut node = head;
            let mut prev: *mut FreeNode = core::ptr::null_mut();
            while !node.is_null() {
                unsafe {
                    assert_eq!((*node).prev, prev, "prev link broken at order {o}");
                }
                let idx = self
                    .region_index_for_test(node as usize)
                    .expect("free node outside every region");
                assert!(
                    self.regions[idx].is_free_bit(node as usize, o),
                    "free-list node at order {o} has no bitmap bit"
                );
                free_entries += 1;
                prev = node;
                node = unsafe { (*node).next };
            }
        }
        // Cache tier.
        for o in 0..NUM_ORDERS {
            for i in 0..self.cache_len[o] as usize {
                let p = self.cache[o][i] as usize;
                let idx = self
                    .region_index_for_test(p)
                    .expect("cached free block outside every region");
                assert!(
                    self.regions[idx].is_free_bit(p, o),
                    "cached free block at order {o} has no bitmap bit"
                );
                free_entries += 1;
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
        assert_eq!(set_bits, free_entries, "bitmap bits != free entries");
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

// NOTE: the classic pointer-space `buddy_pair(ptr, order)` helper is gone —
// `merge_up` computes a block's buddy directly as `addr ^ block_size(order)`.

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
    fn lazy_coalesce_rebuilds_top_block_on_demand() {
        // Fill one region with small blocks, free them all (no merging
        // happens on free anymore), then ask for a top-order block: the
        // allocation-miss path must coalesce everything back together and
        // serve it WITHOUT mapping a new region.
        let top = lohalloc_core::BUDDY_MAX;
        let mut b = Buddy::new();
        let mut small = Vec::new();
        // 64 KiB blocks: REGION/64KiB = 64 of them fill the region exactly.
        let n = REGION_BYTES / (64 * 1024);
        for i in 0..n {
            small.push(b.alloc(64 * 1024).unwrap_or_else(|| panic!("small {i}")));
        }
        assert_eq!(b.region_count(), 1);
        for p in small.drain(..) {
            unsafe { b.dealloc(p, 64 * 1024) };
        }
        b.check_invariants();
        // 64 frees at the same order overflow the 32-entry cache row
        // partway through, so incremental merge-drain has already
        // coalesced most of this region by now (lazy, not batched) — the
        // top-order request finds a fully-merged block waiting, or at
        // worst finishes the job; either way, no new region.
        let big = b.alloc(top).expect("coalesced top-order block");
        assert_eq!(
            b.region_count(),
            1,
            "existing free blocks must merge into the request, not map a new region"
        );
        b.check_invariants();
        unsafe { b.dealloc(big, top) };
        b.check_invariants();
    }

    #[test]
    fn cache_overflow_spills_and_survives() {
        // Free more blocks of one order than the pointer cache holds — the
        // overflow spills to the intrusive list; everything must still be
        // allocatable and invariant-clean.
        let mut b = Buddy::new();
        let mut ptrs = Vec::new();
        let count = ORDER_CACHE * 2 + 7;
        for i in 0..count {
            ptrs.push(b.alloc(4096).unwrap_or_else(|| panic!("alloc {i}")));
        }
        for p in ptrs.drain(..) {
            unsafe { b.dealloc(p, 4096) };
        }
        b.check_invariants();
        for i in 0..count {
            ptrs.push(b.alloc(4096).unwrap_or_else(|| panic!("realloc {i}")));
        }
        b.check_invariants();
        for p in ptrs {
            unsafe { b.dealloc(p, 4096) };
        }
        b.check_invariants();
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
