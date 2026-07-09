//! Slab sub-allocator (fixed-size blocks).
//!
//! Per size class: a no-touch pointer-stack cache over an intrusive
//! overflow free list, plus a bump-carve cursor into the class's active
//! fresh region (Ladder 5 J3-slab — see the [`Slab`] struct doc for the
//! tier design and the Phase P measurement behind it). Blocks are carved
//! out of page-backed regions obtained from the System Fallback; the block
//! size is recoverable from the size class the caller returns at dealloc
//! time (header `slab_class` or the segment registry), so no per-block
//! metadata of our own.
//!
//! # Stripe agnosticism (Ladder 4 C4)
//!
//! `lib.rs` shards the central backend into `[Mutex<Slab>; K]` stripes,
//! with both allocs and frees using the calling thread's stripe. Unlike
//! Buddy (whose frees must reach the exact `Buddy` tracking the block's
//! region bitmap — see `buddy.rs`'s stripe-safety note), a `Slab` free is
//! a pure intrusive-list push with no region lookup, no bitmap, and no
//! coalescing, so a block carved by stripe A and freed into stripe B's
//! list is fully usable from B — regions stay mapped for the whole
//! instance's lifetime regardless of which stripe's `Vec` holds them, and
//! block↔class fidelity is carried by the caller (header `slab_class` or
//! the segment registry), not by the serving stripe. That's why C4 needs
//! no region→stripe registry: cross-stripe migration is harmless by
//! construction. Consequence to keep true: `Slab` must never gain
//! per-region free accounting (e.g. per-region block counts for region
//! reclamation) without revisiting the striped routing in `lib.rs`.

use crate::system;
use lohalloc_core::{align_up, slab_class_for, SLAB_SIZE_CLASSES};

/// Node of the intrusive free list. Lives at the head of each free block.
/// Blocks are always at least `SLAB_SIZE_CLASSES[0]` bytes (8), so the pointer
/// always fits.
#[repr(C)]
struct FreeNode {
    next: Option<*mut FreeNode>,
}

/// Ladder 5 J3-slab: per-class central pointer-stack capacity (the no-touch
/// hot tier in front of the intrusive lists — same design language as
/// `buddy.rs`'s `ORDER_CACHE`). Sized to absorb a whole magazine flush
/// (≤ 16 blocks) plus a whole magazine refill without ping-ponging into
/// the intrusive overflow under steady churn.
const CENTRAL_CACHE: usize = 64;

/// One slab allocator: a per-class free-list array plus the list of backing
/// regions (so they are freed on drop and not leaked).
///
/// # Free/carve tiers (Ladder 5 J3-slab: never touch block memory on the
/// common path)
///
/// Phase P measured that on workloads that never write their allocations,
/// every intrusive free-list op is a cold-block cache/dTLB miss and every
/// first touch of a fresh page is a fault (a 2 MiB residency grant under
/// THP `[always]`) — jemalloc pays none of that because its metadata is
/// fully out-of-band. Three changes mirror `buddy.rs`'s J3-OOB here:
///
/// 1. **Cursor carve**: a fresh region is *not* threaded block-by-block
///    onto the free list at map time (the old `refill` wrote a `FreeNode`
///    into every block upfront — touching every page of the region).
///    Instead `carve_cursor/carve_end[class]` bump-carve it
///    arithmetically; a fresh block's memory is first touched by the
///    *application*, if ever.
/// 2. **Central pointer cache**: freed blocks land in a per-class pointer
///    stack (`cache`, no block writes); only overflow beyond
///    [`CENTRAL_CACHE`] spills to the intrusive lists.
/// 3. Pop order is cache → intrusive → carve → refill: recycled blocks
///    are preferred over fresh carves (RSS-friendly), and the intrusive
///    tier — the only one that touches block memory — is reached only
///    when a class's recycled mass exceeds the cache, which steady churn
///    never does.
pub struct Slab {
    /// Header-carrying blocks: the training / GUI path and every pre-`load()`
    /// startup allocation. Served by `alloc_class`/`alloc_batch` (refilled by
    /// `refill`), freed by `dealloc_class`/`dealloc_batch`.
    header: Tiers,
    /// Header-free blocks (Ladder 5 `slab_headerless`; J4-A): carved from
    /// registered `SEGMENT_SIZE` segments (`refill_segment`) and recovered by
    /// address on free. Served by `alloc_class_headerless`/
    /// `alloc_batch_headerless`, freed by `dealloc_headerless`/
    /// `dealloc_batch_headerless`. **Physically disjoint** from `header`: this
    /// is what lets a `load()`-booted instance serve headerless while pre-load
    /// header blocks still sit safely on `header`. The headerless alloc path
    /// can never pop a header block — which would corrupt on free (its segment
    /// isn't registry-registered, so `dealloc` would read `ptr - HEADER_SIZE`
    /// as a header the user has since overwritten).
    hl: Tiers,
    /// Owning handles to the page regions we carved blocks from (both
    /// flavors). Held alive so the memory stays mapped for the lifetime of the
    /// allocator.
    regions: Vec<system::Mapping>,
}

/// One per-class set of no-touch recycle tiers plus the bump-carve cursor —
/// the state that was inlined into `Slab` before J4-A split header and
/// header-free blocks into two physically separate flavors (see `Slab::hl`).
/// All the block-memory-touching / arithmetic pop+push logic lives here once
/// and is shared by both flavors; `Slab` owns the region mappings and the
/// refill paths that feed each flavor's carve cursor.
struct Tiers {
    /// Intrusive overflow free-list head per class — the only tier that
    /// reads/writes block memory; reached only past `CENTRAL_CACHE` recycled
    /// blocks per class.
    free_heads: [Option<*mut FreeNode>; NUM_CLASSES],
    /// Hot no-touch tier: per-class stacks of recycled block pointers.
    cache: [[*mut u8; CENTRAL_CACHE]; NUM_CLASSES],
    cache_len: [u16; NUM_CLASSES],
    /// Bump cursor into this flavor's active fresh region (`cursor == end`
    /// = exhausted). Strides are `MIN_ALIGN`-aligned at refill time.
    carve_cursor: [usize; NUM_CLASSES],
    carve_end: [usize; NUM_CLASSES],
}

impl Tiers {
    const fn new() -> Self {
        Self {
            free_heads: [None; NUM_CLASSES],
            cache: [[core::ptr::null_mut(); CENTRAL_CACHE]; NUM_CLASSES],
            cache_len: [0; NUM_CLASSES],
            carve_cursor: [0; NUM_CLASSES],
            carve_end: [0; NUM_CLASSES],
        }
    }

    /// Pop one block from the no-touch pointer cache (recycled). `None` = the
    /// cache is empty for this class.
    #[inline]
    fn pop_untouched(&mut self, class: usize) -> Option<*mut u8> {
        let n = self.cache_len[class];
        if n > 0 {
            self.cache_len[class] = n - 1;
            return Some(self.cache[class][(n - 1) as usize]);
        }
        None
    }

    /// Bump-carve one fresh block, if this flavor's active region has any.
    #[inline]
    fn pop_carve(&mut self, class: usize) -> Option<*mut u8> {
        let cur = self.carve_cursor[class];
        if cur == self.carve_end[class] {
            return None;
        }
        let stride = align_up(SLAB_SIZE_CLASSES[class], lohalloc_core::MIN_ALIGN);
        self.carve_cursor[class] = cur + stride;
        Some(cur as *mut u8)
    }

    /// Pop one recycled block from the intrusive overflow list (the only pop
    /// that touches block memory).
    #[inline]
    fn pop_intrusive(&mut self, class: usize) -> Option<*mut u8> {
        let block = self.free_heads[class].take()?;
        let node = unsafe { &mut *block };
        self.free_heads[class] = node.next;
        Some(block as *mut u8)
    }

    /// Recycled-only pop (cache → intrusive), never the carve cursor — the
    /// per-flavor core of the cross-stripe steal primitive.
    #[inline]
    fn pop_recycled(&mut self, class: usize) -> Option<*mut u8> {
        self.pop_untouched(class)
            .or_else(|| self.pop_intrusive(class))
    }

    /// One pop across all tiers in preference order (cache → intrusive →
    /// carve). Recycled-before-fresh keeps RSS bounded; the intrusive tier
    /// sits before the carve so overflow mass drains instead of stranding.
    #[inline]
    fn pop_any(&mut self, class: usize) -> Option<*mut u8> {
        self.pop_recycled(class).or_else(|| self.pop_carve(class))
    }

    /// Return a block of `class` to this flavor's tiers: no-touch cache first,
    /// intrusive overflow only past `CENTRAL_CACHE`.
    ///
    /// # Safety
    /// `ptr` must be a block of exactly this class, from this flavor, not
    /// double-freed.
    unsafe fn dealloc_class(&mut self, ptr: *mut u8, class: usize) {
        let n = self.cache_len[class];
        if (n as usize) < CENTRAL_CACHE {
            self.cache[class][n as usize] = ptr;
            self.cache_len[class] = n + 1;
            return;
        }
        let node = ptr as *mut FreeNode;
        unsafe {
            (*node).next = self.free_heads[class].take();
        }
        self.free_heads[class] = Some(node);
    }

    /// Point this flavor's carve cursor at a fresh `[base, end)` region. Only
    /// valid when the class's carve tier is already exhausted (the callers
    /// assert this, so no partially-carved region is abandoned).
    #[inline]
    fn set_carve(&mut self, class: usize, base: usize, end: usize) {
        debug_assert_eq!(self.carve_cursor[class], self.carve_end[class]);
        self.carve_cursor[class] = base;
        self.carve_end[class] = end;
    }
}

/// Number of slab size classes (compile-time constant so `new()` can be const).
const NUM_CLASSES: usize = SLAB_SIZE_CLASSES.len();

/// Fixed region size used for header-free segments (see
/// `alloc_class_headerless`/`alloc_batch_headerless`): every headerless
/// region is exactly this many bytes, aligned to this boundary, so `ptr &
/// !(SEGMENT_SIZE - 1)` always recovers a real segment base for any live
/// pointer inside it — the property `registry::SegmentRegistry`'s
/// mask-based lookup depends on. Well above every slab class's stride
/// (largest is 16 KiB), so even the largest class gets several blocks per
/// segment.
pub const SEGMENT_SIZE: usize = 64 * 1024;

impl Default for Slab {
    fn default() -> Self {
        Self::new()
    }
}

impl Slab {
    pub const fn new() -> Self {
        Self {
            header: Tiers::new(),
            hl: Tiers::new(),
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
            if let Some(block) = self.header.pop_any(class) {
                return Some(block);
            }
            // Refill: map a region and point the class's carve cursor at it
            // (no block threading — J3-slab), then loop to carve from it.
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
            match self.header.pop_any(class) {
                Some(block) => {
                    out[n] = block;
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

    /// Pop up to `out.len()` **recycled** blocks (cache + intrusive tiers
    /// only — never the carve cursor, never a refill) from the requested
    /// flavor's tiers. Ladder 5 Phase 3: this is the cross-stripe steal
    /// primitive. Under striped centrals, a producer/consumer split
    /// (mt-xfree) migrates every freed block to the consumer's stripe while
    /// the producer's stripe carves fresh segments forever — recycled mass
    /// piles up unreachable. `lib.rs`'s miss path now tries its own stripe's
    /// recycled tier, then *steals* from the other stripes' recycled tiers
    /// (try_lock, one stripe at a time), and only then carves fresh —
    /// recycled-anywhere beats fresh-anywhere.
    ///
    /// J4-A: `headerless` selects which flavor's recycled mass to drain. A
    /// headerless refill must never be handed a header block (it would corrupt
    /// on free — see `Slab::hl`), and a header refill must never be handed a
    /// headerless one, so the steal stays strictly within one flavor.
    pub fn alloc_batch_recycled(
        &mut self,
        class: usize,
        out: &mut [*mut u8],
        headerless: bool,
    ) -> usize {
        debug_assert!(class < NUM_CLASSES);
        let tiers = if headerless {
            &mut self.hl
        } else {
            &mut self.header
        };
        let mut n = 0;
        while n < out.len() {
            match tiers.pop_recycled(class) {
                Some(block) => {
                    out[n] = block;
                    n += 1;
                }
                None => break,
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

    /// Free a **header-flavored** block of an already-known `class` — see
    /// [`Self::alloc_class`].
    ///
    /// # Safety
    /// `ptr` must be a header block of exactly this class, not double-freed.
    pub unsafe fn dealloc_class(&mut self, ptr: *mut u8, class: usize) {
        debug_assert!(class < NUM_CLASSES);
        unsafe { self.header.dealloc_class(ptr, class) };
    }

    /// Push a whole batch of **header-flavored** `class` blocks back in one
    /// locked visit — the magazine flush path.
    ///
    /// # Safety
    /// Every pointer must be a header block of exactly this class, not
    /// double-freed.
    pub unsafe fn dealloc_batch(&mut self, class: usize, blocks: &[*mut u8]) {
        for &b in blocks {
            unsafe { self.header.dealloc_class(b, class) };
        }
    }

    /// Free a **header-free** block of `class` (J4-A): returns it to the `hl`
    /// tiers, never the header tiers. Used only by `lib.rs`'s
    /// `slab_dealloc_headerless` — a headerless block on the header tiers
    /// could be handed to the header alloc/steal path and served with a stale
    /// offset, corrupting on the next free (see `Slab::hl`).
    ///
    /// # Safety
    /// `ptr` must be a headerless block of exactly this class, not
    /// double-freed.
    pub unsafe fn dealloc_headerless(&mut self, ptr: *mut u8, class: usize) {
        debug_assert!(class < NUM_CLASSES);
        unsafe { self.hl.dealloc_class(ptr, class) };
    }

    /// Batch counterpart of [`Self::dealloc_headerless`] — the headerless
    /// magazine flush path.
    ///
    /// # Safety
    /// Every pointer must be a headerless block of exactly this class, not
    /// double-freed.
    pub unsafe fn dealloc_batch_headerless(&mut self, class: usize, blocks: &[*mut u8]) {
        for &b in blocks {
            unsafe { self.hl.dealloc_class(b, class) };
        }
    }

    /// Like `alloc_class`, but every refill this call triggers maps an
    /// exact `SEGMENT_SIZE`-aligned/sized region (instead of the usual
    /// stride-based sizing) and reports that region's base back to the
    /// caller on the call that mapped it — the caller must register the
    /// base in the ownership registry (`registry::SegmentRegistry`) before
    /// any block from it can be validly recovered on the dealloc side.
    ///
    /// Blocks from a headerless-mode segment live on the **`hl` flavor's**
    /// tiers (J4-A), physically disjoint from the header tiers, so the two
    /// flavors are never mixed within one class's free list even after a
    /// `load()` onto a Slab that already served header blocks at startup — the
    /// property that made the old whole-instance `slab_headerless`
    /// pristine-check unnecessary. `try_register(base)` must attempt to record
    /// `base` in the caller's segment ownership registry and report whether it
    /// succeeded — see `refill_segment`'s doc for why a failed registration
    /// must prevent this segment's blocks from ever reaching the free list.
    pub fn alloc_class_headerless(
        &mut self,
        class: usize,
        try_register: &mut dyn FnMut(usize) -> bool,
    ) -> Option<(*mut u8, Option<usize>)> {
        debug_assert!(class < NUM_CLASSES);
        if let Some(block) = self.hl.pop_any(class) {
            return Some((block, None));
        }
        let new_base = self.refill_segment(class, try_register)?;
        let block = self.hl.pop_carve(class)?;
        Some((block, Some(new_base)))
    }

    /// Batch counterpart of `alloc_class_headerless`, for the magazine
    /// refill path: pops up to `out.len()` blocks, refilling with
    /// `SEGMENT_SIZE`-aligned regions as needed via `try_register` (see
    /// `refill_segment`'s doc). Returns how many blocks were written to
    /// `out` — fewer than requested (possibly 0) if a refill's
    /// registration fails and no more free blocks remain, exactly as if
    /// the underlying `mmap` had failed; the caller falls through to the
    /// next backend in the routing chain, same as any other Slab
    /// exhaustion.
    pub fn alloc_batch_headerless(
        &mut self,
        class: usize,
        out: &mut [*mut u8],
        try_register: &mut dyn FnMut(usize) -> bool,
    ) -> usize {
        debug_assert!(class < NUM_CLASSES);
        let mut n = 0;
        while n < out.len() {
            match self.hl.pop_any(class) {
                Some(block) => {
                    out[n] = block;
                    n += 1;
                }
                None => match self.refill_segment(class, try_register) {
                    Some(_) => {}
                    None => break,
                },
            }
        }
        n
    }

    /// Maps one fresh `SEGMENT_SIZE`-aligned/sized region and threads all
    /// its blocks onto `class`'s free list. Returns the region's base.
    ///
    /// Calls `try_register(base)` **before** threading any block onto the
    /// free list or retaining the mapping: a headerless block that reaches
    /// `dealloc` without its segment being registry-registered has no
    /// header to fall back on (it was carved with none, by design) — the
    /// old code registered *after* handing blocks out via a separately
    /// collected `new_regions` list, so a saturated registry (a real,
    /// reachable condition under sustained headerless churn — the fixed
    /// `SegmentRegistry` capacity trades this off against the stack-
    /// overflow risk of a larger embedded table, see `registry.rs`) still
    /// silently served blocks nobody could safely free: `dealloc` read
    /// `ptr - HEADER_SIZE` as a `Header` on the registry miss and either
    /// mis-happened to look like a bad-magic block (silently leaked) or,
    /// worse, read before the segment's mapped range entirely (blocks near
    /// a segment's start) and segfaulted. If `try_register` fails, the
    /// mapping is dropped immediately (unmapped, never pushed to
    /// `self.regions`, never threaded onto any free list) and this refill
    /// reports failure — identical to an `mmap` failure from the caller's
    /// perspective, so the normal Slab-exhaustion fallthrough handles it.
    fn refill_segment(
        &mut self,
        class: usize,
        try_register: &mut dyn FnMut(usize) -> bool,
    ) -> Option<usize> {
        let region = system::alloc_pages(SEGMENT_SIZE, SEGMENT_SIZE)?;
        let base = region.as_ptr();
        if !try_register(base as usize) {
            // `region` drops here, munmapping it — never carved, so no
            // headerless block is ever handed out from it.
            return None;
        }
        let stride = align_up(SLAB_SIZE_CLASSES[class], lohalloc_core::MIN_ALIGN);
        let n = SEGMENT_SIZE / stride;

        // J3-slab: point the **headerless** flavor's carve cursor at the fresh
        // segment instead of threading a FreeNode into every block (which
        // touched — and faulted — every page of the segment at map time). Only
        // called when the hl carve tier is empty, so no partially-carved
        // segment is ever abandoned.
        self.hl
            .set_carve(class, base as usize, base as usize + n * stride);
        self.regions.push(region);
        Some(base as usize)
    }

    /// Map a fresh region sized to hold several `class_size` blocks and
    /// point the class's carve cursor at it (J3-slab: no block threading —
    /// see the struct doc).
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

        // Only called when the header flavor's tiers (cache, intrusive,
        // carve) are all empty for this class, so no partially-carved region
        // is abandoned.
        self.header
            .set_carve(class, base as usize, base as usize + n * stride);
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

    /// J4-A core soundness: after the Slab has served and freed header blocks
    /// (the pre-`load()` startup state), the headerless alloc path must never
    /// hand one of them back — a header block served header-free corrupts on
    /// its next free (its segment isn't registry-registered, so `dealloc`
    /// reads `ptr - HEADER_SIZE` as a header the user overwrote).
    #[test]
    fn headerless_path_never_returns_a_header_block() {
        let mut s = Slab::new();
        let class = 0;
        let mut header_blocks = Vec::new();
        for _ in 0..8 {
            header_blocks.push(s.alloc_class(class).expect("header alloc"));
        }
        for &b in &header_blocks {
            unsafe { s.dealloc_class(b, class) };
        }
        let mut reg = |_base: usize| true;
        for _ in 0..8 {
            let (block, _) = s
                .alloc_class_headerless(class, &mut reg)
                .expect("headerless alloc");
            assert!(
                !header_blocks.contains(&block),
                "headerless path returned a header block {block:?}"
            );
        }
    }

    /// The two flavors recycle strictly within their own tiers: a header free
    /// is only re-served by the header path, a headerless free only by the
    /// headerless path — they never cross.
    #[test]
    fn header_and_headerless_frees_stay_on_their_own_tiers() {
        let mut s = Slab::new();
        let class = 1;
        let mut reg = |_base: usize| true;
        let h = s.alloc_class(class).expect("header alloc");
        let (l, _) = s
            .alloc_class_headerless(class, &mut reg)
            .expect("headerless alloc");
        assert_ne!(h, l);
        unsafe { s.dealloc_class(h, class) };
        unsafe { s.dealloc_headerless(l, class) };
        let h2 = s.alloc_class(class).expect("header realloc");
        let (l2, _) = s
            .alloc_class_headerless(class, &mut reg)
            .expect("headerless realloc");
        assert_eq!(h2, h, "header path did not recycle its own block");
        assert_eq!(l2, l, "headerless path did not recycle its own block");
    }

    /// A headerless block round-trips on the `hl` cache tier (recycled, not a
    /// fresh segment) and is safe to write across its whole class stride.
    #[test]
    fn headerless_round_trip_reuses_hl_tiers() {
        let mut s = Slab::new();
        let class = 2;
        let mut reg = |_base: usize| true;
        let (a, _) = s
            .alloc_class_headerless(class, &mut reg)
            .expect("headerless alloc");
        unsafe { core::ptr::write_bytes(a, 0xAB, SLAB_SIZE_CLASSES[class]) };
        unsafe { s.dealloc_headerless(a, class) };
        let (b, new_base) = s
            .alloc_class_headerless(class, &mut reg)
            .expect("headerless realloc");
        assert_eq!(a, b, "freed headerless block was not recycled");
        assert!(
            new_base.is_none(),
            "a recycled headerless alloc must not map a fresh segment"
        );
    }

    fn is_aligned_to(ptr: *mut u8, align: usize) -> bool {
        (ptr as usize) & (align - 1) == 0
    }
}
