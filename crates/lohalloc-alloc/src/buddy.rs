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
//! never reads or writes the block's own cold memory) and an overflow
//! tier. Since Ladder 5 J3-OOB the overflow tier is **also out-of-band**
//! for orders >= [`OOB_MIN_ORDER`] (16 KiB — all routed buddy traffic):
//! per-region u16 slot-link arrays in the metadata carve plus per-order
//! lists of regions-with-free-blocks, so *no tier ever touches block
//! memory* at those orders. Phase P measured why this matters: on a
//! workload that never writes its allocations (adv-mixed), the old
//! intrusive `FreeNode` spill was the allocator's only touch of user
//! pages — each one a cold cache/dTLB miss, a page fault on first touch,
//! and a 2 MiB residency grant under THP `[always]` (2.1 GiB RSS vs
//! jemalloc's 21 MB at 100k ops, 15× wall). Orders below `OOB_MIN_ORDER`
//! (reachable only via the slab-exhausted fallthrough and direct test use)
//! keep the intrusive doubly-linked lists. Frees do **no** merging on the
//! hot path; merging happens *only* incrementally, in
//! [`Buddy::merge_drain_row`], when a cache row fills (amortized: one
//! bounded drain of half the row per `ORDER_CACHE/2` frees at that order). The eager-merge-then-resplit
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
//! ## Stripe safety under central sharding (Ladder 4 C3)
//!
//! `lib.rs` shards the central backend into `[Mutex<Buddy>; K]` stripes.
//! Each `Buddy` instance is completely unaware of this — no code in this
//! file changed behavior for it — because the region-boundary argument
//! above is also a stripe-boundary argument: a region is mapped by exactly
//! one stripe (registered region-base → stripe in `RegionRegistry` *before*
//! any block from it is handed out; see `refill`'s `register` callback),
//! all of a block's split-buddies and merge-partners live inside its own
//! region, and coalescing never crosses a region boundary. Therefore
//! coalescing can never need state from another stripe, and routing every
//! free to the region's owning stripe (which `lib.rs` does via the
//! registry) is sufficient for full correctness — there is no cross-stripe
//! metadata, ever, and merge-on-spill is untouched.
//!
//! ## Header order-inflation (training only, since J1)
//!
//! The shim prepends a 48-byte `Header` to every *training-mode*
//! allocation, so a request for an exact power of two (e.g. 32 KiB)
//! arrives here as 32 KiB + 48 and rounds up to the *next* order (64 KiB
//! block) — 2× internal fragmentation for pow2-sized requests. Ladder 4 J1
//! removed this in **inference**: a `load()`-booted instance serves Buddy
//! header-free (see `record_orders`/the per-region order map, and
//! `lib.rs::buddy_headerless`), so pow2 requests get exact-order blocks
//! and fresh untouched blocks are never written to at all (the header
//! write was one minor page fault per fresh block — measured as ~18× more
//! faults than jemalloc on adv-mixed before J1). Training keeps headers
//! because dealloc-side reward attribution needs `hash`/`size_class_hint`.
//! One residual: the routing gate compares `total` (request + pad), so an
//! exact-1 MiB request still routes to System even headerless — accepted,
//! the SystemCache path outperforms there anyway.

use crate::system;
use lohalloc_core::{round_up_pow2, MIN_ALIGN};

/// Minimum buddy block size (bytes). Must be a power of two and >= `MIN_ALIGN`
/// so every block satisfies the global alignment contract. 16 keeps us SIMD-
/// friendly and matches the slab floor.
const MIN_BLOCK: usize = MIN_ALIGN;

/// Maximum order index. Order `o` holds blocks of size `MIN_BLOCK << o`.
/// We cap at `BUDDY_MAX`, which is 1 MiB — requests above that go to the System
/// Fallback directly (whole-page `mmap`).
pub(crate) const MAX_ORDER: usize = {
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
pub(crate) const REGION_BYTES: usize = 4 * lohalloc_core::BUDDY_MAX;

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

/// J1 headerless mode: granularity of the per-region **order map** — one
/// byte per `MIN_HEADERLESS_ORDER`-sized (16 KiB) slot recording the buddy
/// order of the live block starting there. Blocks below this order are
/// never served headerless (routing sends ≤ `SLAB_MAX` to the Slab; the
/// rare slab-exhausted fallthrough of a smaller size is rejected by the
/// headerless gate in `lib.rs` and lands on System instead).
pub(crate) const MIN_HEADERLESS_ORDER: usize = 10;

/// Order-map slots per region (256 for 4 MiB / 16 KiB).
const ORDER_MAP_SLOTS: usize = REGION_BYTES >> (MIN_BLOCK_SHIFT + MIN_HEADERLESS_ORDER);

/// `u64` words the order map occupies beside the bitmap in each region's
/// metadata carve.
const ORDER_MAP_WORDS: usize = ORDER_MAP_SLOTS.div_ceil(8);

/// Ladder 5 J3-OOB: out-of-band free-list links for blocks of order >=
/// [`OOB_MIN_ORDER`]. One `u16` prev + one `u16` next per order-map granule
/// (16 KiB), stored in the region's metadata carve — so linking/unlinking a
/// free block never reads or writes the block itself (the intrusive
/// `FreeNode` touch was Phase P's measured adv-mixed killer: cold-block
/// cache/dTLB misses per list op, and each 16-byte write faulting a page —
/// a 2 MiB page under THP `[always]`). Values use a `slot + 1` encoding
/// (`0` = none) so fresh, kernel-zeroed metadata is a valid empty state
/// with no init pass. A granule belongs to at most one free block at one
/// order at a time (the bitmap is the source of truth and a free block
/// exists at exactly one order), so one shared link array serves every OOB
/// order simultaneously.
const LINK_WORDS: usize = (ORDER_MAP_SLOTS * 2 * core::mem::size_of::<u16>()).div_ceil(8);

/// One region's full out-of-band metadata: free bitmap + order map +
/// free-list links (J3-OOB).
const METADATA_WORDS: usize = BITMAP_WORDS + ORDER_MAP_WORDS + LINK_WORDS;

/// Smallest order whose free blocks are tracked out-of-band (J3-OOB): the
/// order-map granule (16 KiB). Routing only sends sub-16 KiB sizes to Buddy
/// via the rare slab-exhausted fallthrough (plus direct unit-test use), so
/// orders below this keep the intrusive in-block `FreeNode` overflow lists —
/// tracking them out-of-band would need `MIN_BLOCK`-granularity links
/// (256 K slots, ~25% metadata overhead per region) for traffic that
/// essentially never occurs.
const OOB_MIN_ORDER: usize = MIN_HEADERLESS_ORDER;

/// Number of out-of-band-tracked orders (`OOB_MIN_ORDER..=MAX_ORDER`).
const OOB_ORDERS: usize = NUM_ORDERS - OOB_MIN_ORDER;

/// Order-map slot index for a block address inside its region.
#[inline]
pub(crate) fn order_map_slot(addr: usize, region_base: usize) -> usize {
    (addr - region_base) >> (MIN_BLOCK_SHIFT + MIN_HEADERLESS_ORDER)
}

/// Block size in bytes for a buddy order (`MIN_BLOCK << order`).
#[inline]
pub(crate) fn block_size_of(order: usize) -> usize {
    block_size(order)
}

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

/// Number of regions' bitmaps backed by one batch mmap (see
/// `Buddy::alloc_bitmap`). Each region's bitmap is `BITMAP_WORDS * 8`
/// bytes (a few tens of KiB); a workload whose live set spans many
/// `REGION_BYTES` (4 MiB) regions — not a hypothetical, a 2000-op run of
/// the buddy benchmark reaches ~59 regions — used to cost one separate
/// `mmap`/TLB entry per region just for its bitmap. Batching them cuts
/// that by `BITMAP_BATCH`×: 16 regions' worth (~1 MiB) per batch mmap.
const BITMAP_BATCH: usize = 16;

/// Backing memory for up to `BITMAP_BATCH` regions' bitmaps, allocated in
/// one `mmap` rather than one per region. `base` is a stable address for
/// the lifetime of the owning `Buddy` (a batch is never freed or grown
/// once mapped — only new batches are appended), so handing out raw
/// pointers into it to `Region`s is sound even though `regions: Vec<Region>`
/// may itself move individual `Region` structs on reallocation: only the
/// small struct moves, never this batch's own backing memory.
struct BitmapBatch {
    base: *mut u64,
    /// How many of `BITMAP_BATCH` bitmap-sized slots have been handed out.
    used: usize,
    _mapping: system::Mapping,
}

unsafe impl Send for BitmapBatch {}

/// A buddy region: a contiguous, page-backed run of `REGION_BYTES` bytes
/// whose base is aligned to `REGION_BYTES` (so buddy arithmetic works
/// cleanly), plus a free bitmap with one bit per (order, block position).
/// The bitmap is what makes "is my buddy free?" O(1): coalescing tests the
/// buddy's bit instead of searching the free list for its node.
struct Region {
    base: *mut u8,
    /// `BITMAP_WORDS` words carved from a `BitmapBatch` (see
    /// `Buddy::alloc_bitmap`) — one bit per (order, block position), see
    /// `ORDER_BIT_OFFSET`. Always zero-initialized (fresh `mmap` memory is
    /// kernel-zeroed) at first use. (No `bytes` field: every region is
    /// exactly `REGION_BYTES`.)
    bitmap: *mut u64,
    /// J1: `ORDER_MAP_SLOTS` order bytes carved immediately after the
    /// bitmap (same metadata block, same lifetime guarantees). Written at
    /// block hand-out when `record_orders` is on; read lock-free by
    /// `lib.rs`'s headerless free dispatch via the address published in
    /// `RegionRegistry` (each byte accessed as an `AtomicU8` — the
    /// cross-thread hand-off is ordered by pointer publication, the atomic
    /// keeps it TSAN-clean and UB-free).
    order_map: *mut u8,
    /// J3-OOB: `ORDER_MAP_SLOTS` u16 `prev` entries followed by
    /// `ORDER_MAP_SLOTS` u16 `next` entries (see `LINK_WORDS`), carved
    /// after the order map in the same metadata block. `slot + 1`
    /// encoding, `0` = none.
    links: *mut u16,
    /// J3-OOB: head/tail of this region's free-slot list per OOB order
    /// (`oob_head[order - OOB_MIN_ORDER]`, `slot + 1` encoding, `0` =
    /// empty). The tail exists so the drain can park unmergeable blocks at
    /// the cold end instead of re-examining them on the next drain.
    oob_head: [u16; OOB_ORDERS],
    oob_tail: [u16; OOB_ORDERS],
    /// J3-OOB: prev/next *region base addresses* on the owning `Buddy`'s
    /// per-order region list (`0` = none). Base addresses, not `regions`
    /// indices — `refill`'s sorted insert shifts indices, but bases are
    /// stable for the region's lifetime.
    rl_prev: [usize; OOB_ORDERS],
    rl_next: [usize; OOB_ORDERS],
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
        unsafe { *self.bitmap.add(pos / 64) |= 1u64 << (pos % 64) };
    }

    #[inline]
    fn clear_free_bit(&mut self, addr: usize, order: usize) {
        let pos = self.bit_pos(addr, order);
        unsafe { *self.bitmap.add(pos / 64) &= !(1u64 << (pos % 64)) };
    }

    /// Is the block at (`addr`, `order`) free? Used by the merge path to
    /// test a buddy's freeness in O(1) (the alternative — searching the
    /// free tiers — is exactly what the bitmap exists to avoid).
    #[inline]
    fn is_free_bit(&self, addr: usize, order: usize) -> bool {
        let pos = self.bit_pos(addr, order);
        unsafe { (*self.bitmap.add(pos / 64) >> (pos % 64)) & 1 == 1 }
    }

    /// The full bitmap as a slice, for `check_invariants`' bit-counting
    /// cross-check only.
    #[cfg(test)]
    fn bitmap_words(&self) -> &[u64] {
        unsafe { core::slice::from_raw_parts(self.bitmap, BITMAP_WORDS) }
    }

    // J3-OOB link-array accessors (`slot + 1` encoding, `0` = none). The
    // arrays live in the metadata carve, never in block memory.
    #[inline]
    fn link_prev(&self, slot: usize) -> u16 {
        debug_assert!(slot < ORDER_MAP_SLOTS);
        unsafe { *self.links.add(slot) }
    }
    #[inline]
    fn set_link_prev(&mut self, slot: usize, v: u16) {
        debug_assert!(slot < ORDER_MAP_SLOTS);
        unsafe { *self.links.add(slot) = v };
    }
    #[inline]
    fn link_next(&self, slot: usize) -> u16 {
        debug_assert!(slot < ORDER_MAP_SLOTS);
        unsafe { *self.links.add(ORDER_MAP_SLOTS + slot) }
    }
    #[inline]
    fn set_link_next(&mut self, slot: usize, v: u16) {
        debug_assert!(slot < ORDER_MAP_SLOTS);
        unsafe { *self.links.add(ORDER_MAP_SLOTS + slot) = v };
    }
}

/// Number of buddy orders (compile-time constant so `new()` can be const).
const NUM_ORDERS: usize = MAX_ORDER + 1;

/// Per-order out-of-line pointer-stack capacity (see `cache` field) —
/// sub-OOB orders only since the Ladder 5 single-tier rework.
const ORDER_CACHE: usize = 32;

/// J3-OOB merge-drain trigger: when an OOB order's list length crosses
/// this, `free_order` drains half of it through `merge_up` (mirrors the
/// old cache-row-overflow trigger, same amortization).
const OOB_DRAIN_AT: u32 = 32;

/// One buddy allocator instance.
pub struct Buddy {
    /// `free_lists[o]` is the head of the doubly-linked **intrusive** free
    /// list for blocks of order `o` (null = empty) — the overflow tier for
    /// orders below [`OOB_MIN_ORDER`] only (J3-OOB: orders >= 16 KiB
    /// overflow to the out-of-band region lists instead and never touch
    /// block memory). The hot path goes through `cache` either way.
    free_lists: [*mut FreeNode; NUM_ORDERS],
    /// J3-OOB: head *region base* of the per-order list of regions holding
    /// at least one out-of-band-tracked free block at that order (`0` =
    /// none). Together with each region's `oob_head`/`links`, this replaces
    /// the intrusive overflow list for orders >= `OOB_MIN_ORDER`: alloc
    /// pops head region → head slot; free pushes a slot and links the
    /// region in on its empty→nonempty edge.
    oob_region_heads: [usize; OOB_ORDERS],
    /// J3-OOB: exact count of blocks on the out-of-band lists per order
    /// (invariant checking / introspection only — NOT the drain trigger,
    /// see `oob_pending`).
    oob_len: [u32; OOB_ORDERS],
    /// J3-OOB merge-drain trigger: pushes at this order since the last
    /// drain. The drain must key on *recent free pressure*, never on the
    /// standing list length: with a growing live set most free blocks'
    /// buddies are live (unmergeable), so the standing mass never shrinks
    /// — a length-based trigger (the first version of this code) fired a
    /// full 16-block drain on EVERY free once the list crossed the
    /// threshold, endlessly re-examining the same unmergeable blocks
    /// (measured: 76% of adv-mixed's instructions in the buddy free
    /// machinery, ~10 list pops per op). Push-count triggering restores
    /// the old cache-row amortization exactly: each freed block is
    /// examined for merging ~once, and tail-parked mass is only ever
    /// revisited by allocation pops.
    oob_pending: [u32; OOB_ORDERS],
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
    /// Backing storage for regions' bitmaps, batched `BITMAP_BATCH` at a
    /// time (see `BitmapBatch`'s doc) instead of one `mmap` per region.
    bitmap_batches: Vec<BitmapBatch>,
    /// J1: when set (headerless inference, flipped once by `lib.rs` inside
    /// `load()` under this stripe's lock), every block hand-out records
    /// its order in the region's order map so `free(ptr)` can recover it
    /// without a header. One branch per alloc when off.
    record_orders: bool,
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
            oob_region_heads: [0; OOB_ORDERS],
            oob_len: [0; OOB_ORDERS],
            oob_pending: [0; OOB_ORDERS],
            cache: [[core::ptr::null_mut(); ORDER_CACHE]; NUM_ORDERS],
            cache_len: [0; NUM_ORDERS],
            regions: Vec::new(),
            last_region: 0,
            bitmap_batches: Vec::new(),
            record_orders: false,
        }
    }

    /// Enable order recording for headerless serving (J1). Called by
    /// `lib.rs`'s `load()` under this stripe's lock, only on an untouched
    /// stripe (`region_count() == 0`), so every region this stripe will
    /// ever carve records orders from its first block.
    pub(crate) fn set_record_orders(&mut self) {
        debug_assert!(
            self.regions.is_empty(),
            "record_orders must be enabled before any region exists"
        );
        self.record_orders = true;
    }

    /// Allocate `size` bytes, aligned to at least `MIN_ALIGN`. Returns `None`
    /// if `size` exceeds `BUDDY_MAX` (the shim routes those to the System
    /// backend).
    ///
    /// `register` is called with each fresh region's base *before* any block
    /// from that region can be returned (see `refill`); returning `false`
    /// discards the region and fails the allocation. Striped callers
    /// (Ladder 4 C3) use it to record region → stripe ownership; standalone
    /// use passes `&mut |_| true`.
    ///
    /// # Safety contract for the caller
    /// The returned pointer is valid until `dealloc` is called with the same
    /// pointer and the same `size`. Reading/writing beyond the rounded block
    /// size is UB.
    pub fn alloc(
        &mut self,
        size: usize,
        register: &mut dyn FnMut(usize, usize) -> bool,
    ) -> Option<*mut u8> {
        if size == 0 || size > lohalloc_core::BUDDY_MAX {
            return None;
        }
        let order = order_for(size)?;
        let block = self.alloc_order(order, register)?;
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

    /// Allocate up to `out.len()` blocks of `order`, for `buddy_mag`'s
    /// batch refill (one lock acquisition amortized over the whole batch,
    /// same shape as `Slab::alloc_batch`). Stops early — returning fewer
    /// than requested — the first time `alloc_order` returns `None` (only
    /// possible on genuine System exhaustion; a healthy `refill()` never
    /// partially fails). Returns the number of pointers written to `out`.
    pub(crate) fn alloc_order_batch(
        &mut self,
        order: usize,
        out: &mut [*mut u8],
        register: &mut dyn FnMut(usize, usize) -> bool,
    ) -> usize {
        debug_assert!(order <= MAX_ORDER);
        let mut n = 0;
        while n < out.len() {
            match self.alloc_order(order, register) {
                Some(block) => {
                    out[n] = block;
                    n += 1;
                }
                None => break,
            }
        }
        n
    }

    /// Free a batch of same-`order` blocks in one lock acquisition, for
    /// `buddy_mag`'s flush path. Each call is exactly `free_order` — no
    /// batch-specific logic — so the existing merge-on-spill trigger
    /// (cache row overflow) fires normally per block; there is no separate
    /// "batch merge" path to keep in sync with the eager-sweep prohibition
    /// documented at the top of this file.
    ///
    /// # Safety
    /// Every pointer in `blocks` must have been returned by `alloc`/
    /// `alloc_order` at this same `order`, and not double-freed.
    pub(crate) unsafe fn dealloc_order_batch(&mut self, order: usize, blocks: &[*mut u8]) {
        debug_assert!(order <= MAX_ORDER);
        for &ptr in blocks {
            unsafe { self.free_order(ptr, order) };
        }
    }

    // ---- internals --------------------------------------------------------

    /// Allocate a block of the given order, splitting larger blocks as
    /// needed. Free blocks are found in this priority: per-order pointer
    /// cache (no block-memory touch) → intrusive list → lazy coalesce of
    /// everything freed so far → fresh region.
    fn alloc_order(
        &mut self,
        order: usize,
        register: &mut dyn FnMut(usize, usize) -> bool,
    ) -> Option<*mut u8> {
        debug_assert!(order <= MAX_ORDER);

        loop {
            // Find the smallest order >= `order` with a free block in any
            // tier: pointer cache, OOB region lists (orders >=
            // OOB_MIN_ORDER), or intrusive list (orders below).
            let mut o = order;
            while o <= MAX_ORDER
                && self.cache_len[o] == 0
                && self.free_lists[o].is_null()
                && (o < OOB_MIN_ORDER || self.oob_region_heads[o - OOB_MIN_ORDER] == 0)
            {
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
                self.refill(register)?;
                continue;
            }

            // Pop a block at order `o`: cache first (block memory
            // untouched), then the out-of-band tier (also untouched,
            // J3-OOB), intrusive list last (orders below OOB_MIN_ORDER
            // only).
            let block: *mut u8 = if self.cache_len[o] > 0 {
                let n = self.cache_len[o] - 1;
                self.cache_len[o] = n;
                self.cache[o][n as usize]
            } else if o >= OOB_MIN_ORDER && self.oob_region_heads[o - OOB_MIN_ORDER] != 0 {
                self.oob_pop(o).expect("nonempty OOB region list must pop")
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
            // J1: record the handed-out block's order so a headerless
            // `free(ptr)` can recover it from the address alone. Atomic
            // store (Relaxed) — see `Region::order_map`'s doc.
            if self.record_orders && order >= MIN_HEADERLESS_ORDER {
                let region = &self.regions[region_idx];
                let slot = order_map_slot(b, region.base as usize);
                unsafe {
                    (*(region.order_map.add(slot) as *const core::sync::atomic::AtomicU8))
                        .store(order as u8, core::sync::atomic::Ordering::Relaxed);
                }
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
        if order >= OOB_MIN_ORDER {
            // Single-tier OOB path: push (metadata only), then drain when
            // the order's list crosses the trigger. Push-first is safe
            // here (unlike the cache path below): a pushed block IS in
            // its tier, so the drain can never unlink a phantom — and it
            // means the just-freed block is immediately merge-eligible.
            self.regions[region_idx].set_free_bit(ptr as usize, order);
            self.oob_push(ptr, order, region_idx);
            let oi = order - OOB_MIN_ORDER;
            self.oob_pending[oi] += 1;
            if self.oob_pending[oi] >= OOB_DRAIN_AT {
                self.oob_pending[oi] = 0;
                self.oob_drain(order, (OOB_DRAIN_AT / 2) as usize);
            }
            return;
        }
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
        if order >= OOB_MIN_ORDER {
            // J3-OOB single tier: metadata writes only, the block itself
            // stays untouched, and the cache rows stay empty at these
            // orders (which is what keeps `remove_free`'s cache scan free
            // during merges — `cache_len` is 0, the loop never runs).
            self.oob_push(ptr, order, region_idx);
        } else if (self.cache_len[order] as usize) < ORDER_CACHE {
            self.cache[order][self.cache_len[order] as usize] = ptr;
            self.cache_len[order] += 1;
        } else {
            // Sub-OOB orders (slab-exhausted fallthrough / direct test
            // use): the old intrusive spill, one block touch.
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
        if order >= OOB_MIN_ORDER {
            let region_idx = self
                .region_index_of(ptr as usize)
                .expect("free block outside every region");
            self.oob_remove(ptr, order, region_idx);
        } else {
            unsafe { self.unlink(ptr as *mut FreeNode, order) };
        }
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
                // No merge: park on the overflow tier — out-of-band
                // (no block touch) at OOB orders, intrusive below.
                self.regions[region_idx].set_free_bit(addr, o);
                if o >= OOB_MIN_ORDER {
                    self.oob_push(addr as *mut u8, o, region_idx);
                } else {
                    unsafe { self.push_list(addr as *mut u8, o) };
                }
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

    // ---- J3-OOB out-of-band free tracking (orders >= OOB_MIN_ORDER) -------
    //
    // These five methods replace `push_list`/`unlink` for OOB orders. All
    // state lives in region metadata (link arrays, per-region heads) and in
    // `Buddy` itself (per-order region-list heads) — free blocks are never
    // read or written. All are plain field mutations on fixed-size,
    // pre-mapped storage: no allocation can occur here (safe under the
    // stripe lock per the standing reentrancy rule).

    /// Push a free block onto its region's out-of-band slot list for
    /// `order`, linking the region into the per-order region list on the
    /// empty→nonempty edge. Does not touch bitmap bits (callers manage
    /// bits, exactly like `push_list`).
    fn oob_push(&mut self, ptr: *mut u8, order: usize, region_idx: usize) {
        debug_assert!((OOB_MIN_ORDER..NUM_ORDERS).contains(&order));
        let oi = order - OOB_MIN_ORDER;
        let base = self.regions[region_idx].base as usize;
        let slot = order_map_slot(ptr as usize, base);
        let enc = (slot + 1) as u16;
        let head = {
            let r = &mut self.regions[region_idx];
            let head = r.oob_head[oi];
            r.set_link_prev(slot, 0);
            r.set_link_next(slot, head);
            if head != 0 {
                r.set_link_prev((head - 1) as usize, enc);
            } else {
                r.oob_tail[oi] = enc;
            }
            r.oob_head[oi] = enc;
            head
        };
        self.oob_len[oi] += 1;
        if head == 0 {
            self.region_list_link(region_idx, oi);
        }
    }

    /// Like [`Self::oob_push`], but appends at the **tail**: the drain
    /// parks unmergeable blocks here so the next drain examines fresh
    /// frees (at the head) instead of re-scanning the same unmergeable
    /// mass — the single-list equivalent of the old "park on the overflow
    /// list, never back into the row being drained" rule.
    fn oob_push_tail(&mut self, ptr: *mut u8, order: usize, region_idx: usize) {
        debug_assert!((OOB_MIN_ORDER..NUM_ORDERS).contains(&order));
        let oi = order - OOB_MIN_ORDER;
        let base = self.regions[region_idx].base as usize;
        let slot = order_map_slot(ptr as usize, base);
        let enc = (slot + 1) as u16;
        let was_empty = {
            let r = &mut self.regions[region_idx];
            let tail = r.oob_tail[oi];
            r.set_link_next(slot, 0);
            r.set_link_prev(slot, tail);
            if tail != 0 {
                r.set_link_next((tail - 1) as usize, enc);
            } else {
                r.oob_head[oi] = enc;
            }
            r.oob_tail[oi] = enc;
            tail == 0
        };
        self.oob_len[oi] += 1;
        if was_empty {
            self.region_list_link(region_idx, oi);
        }
    }

    /// Unlink a known-free block from its region's out-of-band slot list
    /// for `order`, unlinking the region from the per-order region list on
    /// the nonempty→empty edge. Does not touch bitmap bits.
    fn oob_remove(&mut self, ptr: *mut u8, order: usize, region_idx: usize) {
        debug_assert!((OOB_MIN_ORDER..NUM_ORDERS).contains(&order));
        let oi = order - OOB_MIN_ORDER;
        let base = self.regions[region_idx].base as usize;
        let slot = order_map_slot(ptr as usize, base);
        let now_empty = {
            let r = &mut self.regions[region_idx];
            let p = r.link_prev(slot);
            let n = r.link_next(slot);
            if p == 0 {
                debug_assert_eq!(
                    r.oob_head[oi],
                    (slot + 1) as u16,
                    "oob unlink of a non-listed slot"
                );
                r.oob_head[oi] = n;
            } else {
                r.set_link_next((p - 1) as usize, n);
            }
            if n == 0 {
                r.oob_tail[oi] = p;
            } else {
                r.set_link_prev((n - 1) as usize, p);
            }
            r.oob_head[oi] == 0
        };
        debug_assert!(self.oob_len[oi] > 0);
        self.oob_len[oi] -= 1;
        if now_empty {
            self.region_list_unlink(region_idx, oi);
        }
    }

    /// Pop any free block at `order` from the out-of-band tier: head
    /// region → head slot. Returns `None` when no region holds a free
    /// block at this order. Does not touch bitmap bits.
    fn oob_pop(&mut self, order: usize) -> Option<*mut u8> {
        debug_assert!((OOB_MIN_ORDER..NUM_ORDERS).contains(&order));
        let oi = order - OOB_MIN_ORDER;
        let head_base = self.oob_region_heads[oi];
        if head_base == 0 {
            return None;
        }
        let region_idx = self
            .region_index_of(head_base)
            .expect("oob region-list head must be a mapped region");
        let enc = self.regions[region_idx].oob_head[oi];
        debug_assert!(enc != 0, "region on the order list with an empty slot list");
        let slot = (enc - 1) as usize;
        let ptr = (head_base + (slot << (MIN_BLOCK_SHIFT + OOB_MIN_ORDER))) as *mut u8;
        self.oob_remove(ptr, order, region_idx);
        Some(ptr)
    }

    /// Drain up to `count` blocks from `order`'s out-of-band lists,
    /// merging each upward and re-inserting the result — the J3-OOB
    /// counterpart of [`Self::merge_drain_row`], with identical
    /// amortization (bounded per call, triggered by [`OOB_DRAIN_AT`]).
    /// Merged blocks land at their new order via `stash_free` (still
    /// out-of-band at OOB orders); unmerged blocks are parked at the
    /// **tail** so the next drain examines fresh frees instead of
    /// re-scanning the same unmergeable mass. `merge_up`'s buddy unlinks
    /// are O(1) here: at OOB orders the slot links are the only tier
    /// (`remove_free`'s cache scan sees `cache_len == 0`).
    fn oob_drain(&mut self, order: usize, count: usize) {
        debug_assert!((OOB_MIN_ORDER..NUM_ORDERS).contains(&order));
        for _ in 0..count {
            let Some(ptr) = self.oob_pop(order) else {
                break;
            };
            let Some(region_idx) = self.region_index_of(ptr as usize) else {
                debug_assert!(false, "oob free block outside every region");
                continue;
            };
            self.regions[region_idx].clear_free_bit(ptr as usize, order);
            let (addr, o) = unsafe { self.merge_up(ptr as usize, order, region_idx) };
            if o == order {
                self.regions[region_idx].set_free_bit(addr, o);
                self.oob_push_tail(addr as *mut u8, o, region_idx);
            } else {
                self.stash_free(addr as *mut u8, o, region_idx);
            }
        }
    }

    /// Link `regions[region_idx]` at the head of the per-order region list
    /// (called on its slot list's empty→nonempty edge).
    fn region_list_link(&mut self, region_idx: usize, oi: usize) {
        let base = self.regions[region_idx].base as usize;
        let head_base = self.oob_region_heads[oi];
        {
            let r = &mut self.regions[region_idx];
            r.rl_prev[oi] = 0;
            r.rl_next[oi] = head_base;
        }
        if head_base != 0 {
            let hidx = self
                .region_index_of(head_base)
                .expect("region-list head must be a mapped region");
            self.regions[hidx].rl_prev[oi] = base;
        }
        self.oob_region_heads[oi] = base;
    }

    /// Unlink `regions[region_idx]` from the per-order region list (called
    /// on its slot list's nonempty→empty edge).
    fn region_list_unlink(&mut self, region_idx: usize, oi: usize) {
        let (base, prev_base, next_base) = {
            let r = &self.regions[region_idx];
            (r.base as usize, r.rl_prev[oi], r.rl_next[oi])
        };
        if prev_base == 0 {
            debug_assert_eq!(self.oob_region_heads[oi], base, "region-list head mismatch");
            self.oob_region_heads[oi] = next_base;
        } else {
            let pidx = self
                .region_index_of(prev_base)
                .expect("region-list prev must be a mapped region");
            self.regions[pidx].rl_next[oi] = next_base;
        }
        if next_base != 0 {
            let nidx = self
                .region_index_of(next_base)
                .expect("region-list next must be a mapped region");
            self.regions[nidx].rl_prev[oi] = prev_base;
        }
    }

    /// Map a fresh `REGION_BYTES` region (aligned to `REGION_BYTES` so buddy
    /// arithmetic is exact) and carve it into top-order blocks on the cache
    /// tier. One `mmap` serves `REGION_BYTES / BUDDY_MAX` top-order blocks
    /// — the original scheme mapped one region *per* top-order block, which
    /// made buddy-heavy workloads syscall-bound (hyperfine showed 24 ms
    /// outliers from mmap storms).
    /// Hands out `BITMAP_WORDS` zeroed words for a freshly refilled region,
    /// allocating a new batch `mmap` only every `BITMAP_BATCH` regions
    /// instead of on every single one (see `BitmapBatch`'s doc).
    /// Returns `(bitmap, order_map)` — one region's metadata carve. The
    /// order map lives immediately after the bitmap words (J1); both come
    /// zeroed from the batch `mmap`.
    fn alloc_bitmap(&mut self) -> Option<(*mut u64, *mut u8, *mut u16)> {
        let need_new_batch = match self.bitmap_batches.last() {
            Some(b) => b.used >= BITMAP_BATCH,
            None => true,
        };
        if need_new_batch {
            let bytes = METADATA_WORDS * BITMAP_BATCH * 8;
            let mapping = system::alloc_pages(bytes, 8)?;
            let base = mapping.as_ptr() as *mut u64;
            self.bitmap_batches.push(BitmapBatch {
                base,
                used: 0,
                _mapping: mapping,
            });
        }
        // Just pushed if empty/exhausted, so this is always Some.
        let batch = self.bitmap_batches.last_mut().unwrap();
        let ptr = unsafe { batch.base.add(batch.used * METADATA_WORDS) };
        batch.used += 1;
        let order_map = unsafe { ptr.add(BITMAP_WORDS) } as *mut u8;
        // J3-OOB: link arrays right after the order map — u64-word aligned
        // start, so the u16 accesses are trivially aligned. Kernel-zeroed,
        // and `0` is the links' "none" encoding: no init pass needed.
        let links = unsafe { ptr.add(BITMAP_WORDS + ORDER_MAP_WORDS) } as *mut u16;
        Some((ptr, order_map, links))
    }

    fn refill(&mut self, register: &mut dyn FnMut(usize, usize) -> bool) -> Option<()> {
        let region = system::alloc_pages(REGION_BYTES, REGION_BYTES)?;
        let base = region.as_ptr();
        let (bitmap, order_map, links) = self.alloc_bitmap()?;
        // Register the region BEFORE carving it into blocks: once a block
        // has been handed out, its eventual free must be able to resolve
        // the region (striped callers: to the owning stripe; headerless
        // callers additionally: to this order map), so a region that can't
        // be registered must never contribute a block. Dropping `region`
        // here unmaps it — the refill fails cleanly and the caller falls
        // through to the System backend. (The metadata carve above is not
        // reclaimed on failure — a few hundred bytes inside a shared batch
        // mmap, bounded by registry capacity.)
        if !register(base as usize, order_map as usize) {
            return None;
        }
        // Keep `regions` sorted by base so lookups can binary-search. Refill
        // is rare (once per 4 MiB), so the O(n) insert shift is noise.
        let idx = self
            .regions
            .partition_point(|r| (r.base as usize) < base as usize);
        self.regions.insert(
            idx,
            Region {
                base,
                bitmap,
                order_map,
                links,
                oob_head: [0; OOB_ORDERS],
                oob_tail: [0; OOB_ORDERS],
                rl_prev: [0; OOB_ORDERS],
                rl_next: [0; OOB_ORDERS],
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
    /// Test-only `alloc` without region registration (standalone `Buddy`
    /// use — no stripes, so every region trivially "registers").
    fn alloc_t(&mut self, size: usize) -> Option<*mut u8> {
        self.alloc(size, &mut |_, _| true)
    }

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
        // J3-OOB tier: walk every per-order region list and each region's
        // slot list, checking link coherence and bitmap agreement. Also
        // assert the intrusive lists stay empty at OOB orders — nothing
        // may push there anymore.
        for o in OOB_MIN_ORDER..NUM_ORDERS {
            assert!(
                self.free_lists[o].is_null(),
                "intrusive list non-empty at OOB order {o}"
            );
            assert_eq!(
                self.cache_len[o], 0,
                "cache row non-empty at OOB order {o} (single-tier invariant)"
            );
            let oi = o - OOB_MIN_ORDER;
            let mut oob_walked = 0usize;
            let mut rbase = self.oob_region_heads[oi];
            let mut prev_rbase = 0usize;
            while rbase != 0 {
                let ridx = self
                    .region_index_for_test(rbase)
                    .expect("oob region-list entry outside every region");
                let r = &self.regions[ridx];
                assert_eq!(
                    r.rl_prev[oi], prev_rbase,
                    "region-list prev broken at order {o}"
                );
                let mut enc = r.oob_head[oi];
                assert!(enc != 0, "region on order-{o} list with empty slot list");
                let mut prev_enc = 0u16;
                let mut last_enc = 0u16;
                while enc != 0 {
                    let slot = (enc - 1) as usize;
                    assert_eq!(r.link_prev(slot), prev_enc, "slot prev broken at order {o}");
                    let addr = rbase + (slot << (MIN_BLOCK_SHIFT + OOB_MIN_ORDER));
                    assert!(
                        r.is_free_bit(addr, o),
                        "oob free slot at order {o} has no bitmap bit"
                    );
                    free_entries += 1;
                    oob_walked += 1;
                    prev_enc = enc;
                    last_enc = enc;
                    enc = r.link_next(slot);
                }
                assert_eq!(r.oob_tail[oi], last_enc, "tail mismatch at order {o}");
                prev_rbase = rbase;
                rbase = r.rl_next[oi];
            }
            assert_eq!(
                self.oob_len[oi] as usize, oob_walked,
                "oob_len counter out of sync at order {o}"
            );
        }
        let set_bits: usize = self
            .regions
            .iter()
            .map(|r| {
                r.bitmap_words()
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
        let p = b.alloc_t(100).expect("alloc 100");
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
        let p1 = b.alloc_t(64).expect("alloc 64 #1");
        let p2 = b.alloc_t(64).expect("alloc 64 #2");
        assert!(b.region_count() > before, "expected a refill");
        unsafe { b.dealloc(p1, 64) };
        unsafe { b.dealloc(p2, 64) };
        // Coalesced block should now satisfy a 128-byte request without mapping.
        let before2 = b.region_count();
        let p3 = b.alloc_t(128).expect("alloc 128 after coalesce");
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
        let p1 = b.alloc_t(64).expect("a1");
        let p2 = b.alloc_t(64).expect("a2");
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
        let p = b.alloc_t(1).expect("alloc 1");
        assert!(is_aligned(p as usize, MIN_BLOCK));
        unsafe { b.dealloc(p, 1) };
    }

    #[test]
    fn rejects_oversize() {
        let mut b = Buddy::new();
        assert!(b.alloc_t(0).is_none());
        assert!(b.alloc_t(lohalloc_core::BUDDY_MAX + 1).is_none());
    }

    #[test]
    fn bounded_regions_under_fixed_live_set() {
        let mut b = Buddy::new();
        let mut live = Vec::new();
        for _ in 0..5 {
            for _ in 0..4 {
                live.push(b.alloc_t(64).expect("alloc"));
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
            live.push(b.alloc_t(top).expect("top-order alloc"));
        }
        assert_eq!(
            b.region_count(),
            1,
            "first {per_region} blocks share one region"
        );
        live.push(b.alloc_t(top).expect("overflow top-order alloc"));
        assert_eq!(b.region_count(), 2, "next block needs a second region");
        for p in live.drain(..) {
            unsafe { b.dealloc(p, top) };
        }
    }

    #[test]
    fn bitmap_batches_span_correctly_across_many_regions() {
        // Regression coverage for the bitmap-batching change: BITMAP_BATCH
        // regions share one batch mmap, so forcing more regions than one
        // batch holds exercises the batch-rollover path in `alloc_bitmap`.
        // A batch-offset bug would show up as `check_invariants` seeing
        // bitmap bits from the wrong region (aliased across the batch).
        let per_region = REGION_BYTES / lohalloc_core::BUDDY_MAX;
        let top = lohalloc_core::BUDDY_MAX;
        let mut b = Buddy::new();
        let target_regions = BITMAP_BATCH * 2 + 3; // spans 3 separate batches
        let mut live = Vec::new();
        for _ in 0..(target_regions * per_region) {
            live.push(b.alloc_t(top).expect("top-order alloc"));
        }
        assert_eq!(b.region_count(), target_regions);
        b.check_invariants();
        for p in live.drain(..) {
            unsafe { b.dealloc(p, top) };
        }
        b.check_invariants();
    }

    #[test]
    fn coalesce_across_top_blocks_stops_at_max_order() {
        // Free two adjacent top-order blocks: coalescing must stop at
        // MAX_ORDER (they stay two 1 MiB free blocks, never a 2 MiB one),
        // and a subsequent top-order alloc must reuse without a refill.
        let top = lohalloc_core::BUDDY_MAX;
        let mut b = Buddy::new();
        let p1 = b.alloc_t(top).expect("top #1");
        let p2 = b.alloc_t(top).expect("top #2");
        assert_eq!(b.region_count(), 1);
        unsafe {
            b.dealloc(p1, top);
            b.dealloc(p2, top);
        }
        let p3 = b.alloc_t(top).expect("top after frees");
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
            .map(|i| b.alloc_t(64).unwrap_or_else(|| panic!("a{i}")))
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
        let q1 = b.alloc_t(64).expect("post-churn 64");
        let q2 = b.alloc_t(128).expect("post-churn 128");
        let q3 = b.alloc_t(64).expect("post-churn 64 #2");
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
            live.push((b.alloc_t(sz).expect("alloc"), sz));
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
            small.push(b.alloc_t(64 * 1024).unwrap_or_else(|| panic!("small {i}")));
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
        let big = b.alloc_t(top).expect("coalesced top-order block");
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
            ptrs.push(b.alloc_t(4096).unwrap_or_else(|| panic!("alloc {i}")));
        }
        for p in ptrs.drain(..) {
            unsafe { b.dealloc(p, 4096) };
        }
        b.check_invariants();
        for i in 0..count {
            ptrs.push(b.alloc_t(4096).unwrap_or_else(|| panic!("realloc {i}")));
        }
        b.check_invariants();
        for p in ptrs {
            unsafe { b.dealloc(p, 4096) };
        }
        b.check_invariants();
    }

    #[test]
    fn batch_alloc_dealloc_roundtrip_preserves_invariants() {
        // `alloc_order_batch`/`dealloc_order_batch` (buddy_mag's refill/
        // flush primitives) must behave exactly like the same number of
        // individual `alloc_order`/`free_order` calls — same blocks
        // reachable, invariants intact — since batching is purely a lock-
        // amortization shape, not a different algorithm.
        let mut b = Buddy::new();
        let order = order_for(64 * 1024).unwrap();

        let mut buf = [core::ptr::null_mut::<u8>(); 6];
        let n = b.alloc_order_batch(order, &mut buf, &mut |_, _| true);
        assert_eq!(n, 6, "a fresh Buddy should satisfy a small batch in full");
        b.check_invariants();

        // All six pointers must be distinct and individually freeable.
        let mut seen = std::collections::HashSet::new();
        for &p in &buf {
            assert!(
                seen.insert(p as usize),
                "batch alloc returned a duplicate pointer"
            );
        }

        unsafe { b.dealloc_order_batch(order, &buf) };
        b.check_invariants();

        // The freed batch must be fully reusable afterwards.
        let mut buf2 = [core::ptr::null_mut::<u8>(); 6];
        let n2 = b.alloc_order_batch(order, &mut buf2, &mut |_, _| true);
        assert_eq!(n2, 6);
        b.check_invariants();
        unsafe { b.dealloc_order_batch(order, &buf2) };
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
                if let Some(p) = b.alloc_t(sz) {
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

    /// J3-OOB: cache-row overflow at an OOB order must spill to the
    /// out-of-band tier (never the intrusive list), survive the invariant
    /// walk, and be fully reusable — no fresh regions on re-allocation.
    #[test]
    fn oob_spill_and_reuse_roundtrip() {
        const SZ: usize = 16 * 1024; // order 10, the OOB floor
        let mut b = Buddy::new();
        // 130 live order-10 blocks (spans >1 region: 256 per region).
        let blocks: Vec<*mut u8> = (0..130).map(|_| b.alloc_t(SZ).expect("alloc")).collect();
        let regions_before = b.region_count();
        // Free every other block: no two freed blocks are buddies, so
        // merge_drain can never promote them — the frees pile up at order
        // 10 until the 32-entry cache row overflows into the OOB tier.
        for p in blocks.iter().step_by(2) {
            unsafe { b.dealloc(*p, SZ) };
        }
        assert!(
            b.free_lists[OOB_MIN_ORDER].is_null(),
            "intrusive list must stay empty at OOB orders"
        );
        assert!(
            b.oob_region_heads.iter().any(|&h| h != 0),
            "expected OOB spill after 65 same-order frees"
        );
        b.check_invariants();
        // Re-allocate the same count: every block must come from the freed
        // pool (cache + OOB tiers), mapping zero new regions.
        let again: Vec<*mut u8> = (0..65).map(|_| b.alloc_t(SZ).expect("realloc")).collect();
        assert_eq!(
            b.region_count(),
            regions_before,
            "reuse must not map regions"
        );
        b.check_invariants();
        for p in again {
            unsafe { b.dealloc(p, SZ) };
        }
        for p in blocks.iter().skip(1).step_by(2) {
            unsafe { b.dealloc(*p, SZ) };
        }
        b.check_invariants();
    }

    /// J3-OOB: freeing both buddies must still coalesce upward when the
    /// drain runs, with the OOB lists staying coherent through the
    /// remove_free → oob_remove path.
    #[test]
    fn oob_merge_up_through_drain() {
        const SZ: usize = 16 * 1024;
        let mut b = Buddy::new();
        let blocks: Vec<*mut u8> = (0..128).map(|_| b.alloc_t(SZ).expect("alloc")).collect();
        // Free ALL of them: buddies pair up, so overflow drains merge
        // entire runs up toward MAX_ORDER, exercising oob_remove (buddy
        // unlink) and oob_push (merged re-insert) heavily.
        for p in &blocks {
            unsafe { b.dealloc(*p, SZ) };
        }
        b.check_invariants();
        // A top-order allocation must now be satisfiable from merged mass
        // without a new region.
        let regions_before = b.region_count();
        let big = b.alloc_t(block_size(MAX_ORDER)).expect("top-order alloc");
        assert_eq!(b.region_count(), regions_before, "merge must enable reuse");
        unsafe { b.dealloc(big, block_size(MAX_ORDER)) };
        b.check_invariants();
    }
}
