//! Insert-only, lock-free segment ownership registry for header-free slab
//! serving (see `lib.rs`'s `slab_headerless` and
//! `slab::Slab::alloc_class_headerless`/`alloc_batch_headerless`).
//!
//! Maps a `slab::SEGMENT_SIZE`-aligned slab segment base to the slab class
//! it was carved for, so `dealloc`/cabi `free`/`realloc`/`malloc_usable_size`
//! can recover a headerless block's class from `ptr &
//! !(SEGMENT_SIZE - 1)` alone — no header read, no lock. Fixed capacity,
//! open-addressed, insert-only (no deletion, so a lookup either finds an
//! exact match or hits an empty slot and can stop scanning — no ABA, no
//! reclamation problem to solve). Insertion only ever happens under
//! `Lohalloc`'s slab lock (once per fresh segment mapping, i.e. rarely);
//! lookup is fully lock-free (plain atomic loads), since it runs on every
//! headerless dealloc/realloc/usable-size call.
//!
//! Saturation (all `CAPACITY` slots taken) is a safe, silent degrade: a
//! segment that can't be registered simply never round-trips through the
//! headerless path — its blocks still work correctly via the ordinary
//! header-based dealloc route (a registry miss falls through there).

use core::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

/// Fixed slot count. 1024 entries at `slab::SEGMENT_SIZE` (64 KiB) each
/// cover 64 MiB of live headerless slab memory before saturation — ample
/// for the (bounded-live-set, region-count-capped) workloads this crate
/// benchmarks against. Embedded directly in `Lohalloc` (itself often a
/// `const`-initialized static, and sometimes stack-placed — e.g. a test's
/// `let alloc = Lohalloc::new();`) as fixed arrays rather than lazily
/// heap-allocated, so there is no first-use allocation or lock to reason
/// about — the same const-array-repeat pattern `magazine.rs`/`buddy_mag.rs`
/// use for their per-thread slot arrays. Deliberately kept small (~9 KiB,
/// not the ~72 KiB an 8192-entry table would cost): a large embedded array
/// here reproduced as a real stack-overflow SIGSEGV under `cargo test`'s
/// parallel harness, which runs each test on its own (comparatively small)
/// thread stack and had several `Lohalloc::new()` locals alive across
/// nested/concurrent test calls.
const CAPACITY: usize = 1024;

#[allow(clippy::declare_interior_mutable_const)]
const EMPTY_KEY: AtomicUsize = AtomicUsize::new(0);
#[allow(clippy::declare_interior_mutable_const)]
const EMPTY_VAL: AtomicU8 = AtomicU8::new(0);

/// See the module doc.
pub struct SegmentRegistry {
    /// `0` means empty; any registered segment base is guaranteed nonzero
    /// (real mmap addresses are never 0) and `SEGMENT_SIZE`-aligned.
    keys: [AtomicUsize; CAPACITY],
    vals: [AtomicU8; CAPACITY],
}

impl Default for SegmentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl SegmentRegistry {
    pub const fn new() -> Self {
        Self {
            keys: [EMPTY_KEY; CAPACITY],
            vals: [EMPTY_VAL; CAPACITY],
        }
    }

    /// Base slot for `base`: a plain multiplicative mix, taking the high
    /// bits post-multiply for the index. `base` always has many trailing
    /// zero bits (>= 16, since segments are 64 KiB-aligned), which a
    /// naive low-bits mask would collide on directly — the mix spreads
    /// those bits across the whole word before masking.
    #[inline]
    fn slot_for(base: usize) -> usize {
        (((base as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 48) as usize & (CAPACITY - 1)
    }

    /// Register `base` (a fresh segment's start address — must be nonzero
    /// and `SEGMENT_SIZE`-aligned) as carved for `class`. Called once per
    /// fresh segment mapping, always under the slab lock (single-writer in
    /// practice; the CAS below is defense-in-depth, not load-bearing for
    /// correctness under that lock).
    ///
    /// # Publish order (load-bearing — a SIGSEGV on ARM64 taught us this)
    ///
    /// `vals[idx]` is written **before** `keys[idx]` is published via the
    /// `Release` CAS, never after. A concurrent `lookup` only synchronizes
    /// on `keys[idx]` (`Acquire`) — release/acquire on one atomic makes
    /// everything sequenced *before* the release visible to the acquire,
    /// but says nothing about a write to a *different* atomic (`vals`)
    /// sequenced *after* it. Publishing key-then-val left a window on weak
    /// memory-ordering hardware (reproduced on Apple Silicon; masked on
    /// x86-64's stronger TSO, which is why this only ever crashed on
    /// macOS/ARM64) where a lookup could observe the new key but still
    /// read the slot's old/zero `val` — silently handing a headerless
    /// dealloc the wrong slab class, corrupting free-list linkage a few
    /// operations later. Fixed by writing the payload first so the
    /// key's `Release` store is guaranteed to flush it.
    ///
    /// Returns `true` if `base` is registered (either just now, or already
    /// present), `false` if the table is saturated. **The caller must not
    /// serve any headerless (no-header) block from `base` on a `false`
    /// return** — those blocks would have no header to fall back on, so an
    /// unregistered segment must never be threaded onto a headerless free
    /// list (see `slab_block_headerless_via_magazine`'s doc, which unmaps
    /// the segment instead when this returns `false`).
    #[must_use]
    pub fn insert(&self, base: usize, class: u8) -> bool {
        debug_assert!(base != 0, "segment base must be nonzero");
        let mut idx = Self::slot_for(base);
        for _ in 0..CAPACITY {
            let existing = self.keys[idx].load(Ordering::Acquire);
            if existing == base {
                return true; // already registered
            }
            if existing == 0 {
                // Payload before publish (see the ordering note above) —
                // single-writer per the module doc, so no other writer can
                // claim this slot with a different key between here and
                // the CAS below.
                self.vals[idx].store(class, Ordering::Relaxed);
                match self.keys[idx].compare_exchange(0, base, Ordering::Release, Ordering::Relaxed)
                {
                    Ok(_) => return true,
                    Err(actual) if actual == base => return true,
                    Err(_) => {} // lost a theoretical race; fall through and advance
                }
            }
            idx = (idx + 1) & (CAPACITY - 1);
        }
        false // Saturated.
    }

    /// Look up `base` (already masked down by the caller via `ptr &
    /// !(SEGMENT_SIZE - 1)`). Returns the registered class, or `None` if
    /// `base` was never registered (foreign pointer, a header-based block,
    /// or a saturation-skipped segment) — the miss path.
    #[inline]
    pub fn lookup(&self, base: usize) -> Option<u8> {
        let mut idx = Self::slot_for(base);
        for _ in 0..CAPACITY {
            let k = self.keys[idx].load(Ordering::Acquire);
            if k == 0 {
                return None;
            }
            if k == base {
                return Some(self.vals[idx].load(Ordering::Acquire));
            }
            idx = (idx + 1) & (CAPACITY - 1);
        }
        None
    }
}

/// Insert-only, lock-free **buddy region → stripe** registry (Ladder 4 C3).
///
/// Same table, same soundness story as [`SegmentRegistry`] (see its module
/// and method docs — publish ordering, saturation semantics, single-writer
/// insert discipline), but keyed on `buddy::REGION_BYTES`-aligned region
/// bases and valued with the index of the `[Mutex<Buddy>; K]` stripe that
/// owns the region. Frees resolve their owning stripe from
/// `ptr & !(REGION_BYTES - 1)` alone — O(1), no lock, exact (never "best
/// effort": a block must only ever be freed into the stripe whose `Buddy`
/// tracks its region's bitmap).
///
/// Capacity is the shared `CAPACITY` (1024): at 4 MiB per region that
/// covers 4 GiB of live buddy memory before saturation. Saturation is
/// handled *at allocation time* — `Buddy::refill` registers a fresh region
/// **before** carving it and fails the refill (unmapping the region) if
/// registration fails, so no block from an unregistered region can ever
/// reach a caller. A free-side lookup miss therefore indicates a foreign
/// pointer bug, not saturation, and is debug-asserted.
///
/// Reentrancy note (the Ladder-4 standing rule): `insert` is called while
/// a stripe's `Mutex<Buddy>` is held. That is safe **only** because this
/// table is fixed-size embedded atomics — it can never allocate. Do not
/// replace it with a growable structure.
///
/// Besides the stripe, each entry carries the region's **order-map base**
/// (Ladder 4 J1): the per-region out-of-band array holding each live
/// block's buddy order, which lets a headerless `free(ptr)` recover
/// `(stripe, order)` from the address alone with one probe and **no stripe
/// lock**. Both payload words are written before the key is published via
/// the `Release` CAS — the exact ordering rule `SegmentRegistry::insert`'s
/// doc derives from the ARM64 key-before-val SIGSEGV.
pub struct RegionRegistry {
    /// `0` = empty; region bases are nonzero and `REGION_BYTES`-aligned.
    keys: [AtomicUsize; CAPACITY],
    /// Owning `[Mutex<Buddy>]` stripe index.
    stripes: [AtomicU8; CAPACITY],
    /// Address of the region's order map (`REGION_BYTES >> 14` order
    /// bytes, one per 16 KiB slot), or 0 for none.
    maps: [AtomicUsize; CAPACITY],
}

impl Default for RegionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl RegionRegistry {
    pub const fn new() -> Self {
        Self {
            keys: [EMPTY_KEY; CAPACITY],
            stripes: [EMPTY_VAL; CAPACITY],
            maps: [EMPTY_KEY; CAPACITY],
        }
    }

    /// Register `base` (a fresh `REGION_BYTES`-aligned region base) as
    /// owned by `stripe`, with its order map at `map`. Returns `false` on
    /// saturation — the caller must then discard the region (see the type
    /// doc). Same single-writer + payload-before-publish discipline as
    /// [`SegmentRegistry::insert`].
    #[must_use]
    pub fn insert(&self, base: usize, stripe: u8, map: usize) -> bool {
        debug_assert!(base != 0, "region base must be nonzero");
        let mut idx = SegmentRegistry::slot_for(base);
        for _ in 0..CAPACITY {
            let existing = self.keys[idx].load(Ordering::Acquire);
            if existing == base {
                return true; // already registered
            }
            if existing == 0 {
                // Payloads BEFORE the key publish (see the type doc).
                self.stripes[idx].store(stripe, Ordering::Relaxed);
                self.maps[idx].store(map, Ordering::Relaxed);
                match self.keys[idx].compare_exchange(0, base, Ordering::Release, Ordering::Relaxed)
                {
                    Ok(_) => return true,
                    Err(actual) if actual == base => return true,
                    Err(_) => {} // lost a theoretical race; advance
                }
            }
            idx = (idx + 1) & (CAPACITY - 1);
        }
        false // Saturated.
    }

    /// Owning stripe for `base` (already masked to the region base by the
    /// caller), or `None` for a pointer in no registered region.
    #[inline]
    pub fn lookup(&self, base: usize) -> Option<u8> {
        self.lookup_full(base).map(|(stripe, _)| stripe)
    }

    /// `(stripe, order_map_base)` for `base`, or `None` on a miss.
    #[inline]
    pub fn lookup_full(&self, base: usize) -> Option<(u8, usize)> {
        let mut idx = SegmentRegistry::slot_for(base);
        for _ in 0..CAPACITY {
            let k = self.keys[idx].load(Ordering::Acquire);
            if k == 0 {
                return None;
            }
            if k == base {
                return Some((
                    self.stripes[idx].load(Ordering::Acquire),
                    self.maps[idx].load(Ordering::Acquire),
                ));
            }
            idx = (idx + 1) & (CAPACITY - 1);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_registry_roundtrips_stripe_and_map() {
        let r = RegionRegistry::new();
        assert!(r.insert(0x40_0000, 2, 0xDEAD_0000));
        assert_eq!(r.lookup(0x40_0000), Some(2));
        assert_eq!(r.lookup_full(0x40_0000), Some((2, 0xDEAD_0000)));
        assert_eq!(r.lookup_full(0x80_0000), None);
        // Duplicate insert is a no-op keeping the original payloads.
        assert!(r.insert(0x40_0000, 3, 0xBEEF_0000));
        assert_eq!(r.lookup_full(0x40_0000), Some((2, 0xDEAD_0000)));
    }

    #[test]
    fn region_registry_saturation_reports_failure() {
        let r = RegionRegistry::new();
        for i in 0..CAPACITY {
            assert!(r.insert(0x1000_0000 + i * 0x40_0000, (i % 4) as u8, i));
        }
        assert!(!r.insert(0x1000_0000 + CAPACITY * 0x40_0000, 0, 0));
    }

    #[test]
    fn insert_then_lookup_hits() {
        let r = SegmentRegistry::new();
        assert!(r.insert(0x1_0000, 3));
        assert_eq!(r.lookup(0x1_0000), Some(3));
    }

    #[test]
    fn lookup_miss_returns_none() {
        let r = SegmentRegistry::new();
        assert!(r.insert(0x1_0000, 3));
        assert_eq!(r.lookup(0x2_0000), None);
    }

    #[test]
    fn empty_registry_all_misses() {
        let r = SegmentRegistry::new();
        assert_eq!(r.lookup(0x1_0000), None);
        assert_eq!(r.lookup(0), None); // never registered, must not panic
    }

    #[test]
    fn duplicate_insert_is_a_noop() {
        let r = SegmentRegistry::new();
        assert!(r.insert(0x1_0000, 3));
        assert!(r.insert(0x1_0000, 7)); // must not overwrite, still succeeds
        assert_eq!(r.lookup(0x1_0000), Some(3));
    }

    #[test]
    fn many_distinct_segments_all_resolve() {
        let r = SegmentRegistry::new();
        // Well under CAPACITY so no saturation; bases spaced by
        // SEGMENT_SIZE (64 KiB) as real segments would be.
        for i in 0..500usize {
            let base = 0x1000_0000 + i * 0x1_0000;
            assert!(r.insert(base, (i % 12) as u8));
        }
        for i in 0..500usize {
            let base = 0x1000_0000 + i * 0x1_0000;
            assert_eq!(r.lookup(base), Some((i % 12) as u8), "segment {i}");
        }
    }

    #[test]
    fn saturation_reports_failure_not_panic() {
        // Fill the table completely, then a further insert must not panic
        // (the loop must terminate after CAPACITY probes even with no
        // empty slot left) and must report failure so the caller knows not
        // to serve headerless blocks from that segment.
        let r = SegmentRegistry::new();
        for i in 0..CAPACITY {
            let base = 0x1000_0000 + i * 0x1_0000;
            assert!(r.insert(base, 1));
        }
        // One more, past capacity — must return false, not panic or hang.
        assert!(!r.insert(0x1000_0000 + CAPACITY * 0x1_0000, 1));
        // A key that was never inserted (and can't be, table is full):
        // lookup must still terminate and report a miss (or, since the
        // table is full, could in principle find a false match only if
        // hash collision landed exactly there — vanishingly unlikely with
        // this key spacing, and irrelevant to what's being tested: that
        // the loop terminates).
        let _ = r.lookup(0xDEAD_0000);
    }

    #[test]
    fn concurrent_readers_never_observe_a_visible_key_with_wrong_val() {
        // Regression test for the ARM64 SIGSEGV: many reader threads poll
        // `lookup` while one writer thread inserts fresh, distinct entries.
        // A reader that sees a key it doesn't recognize yet is fine (None);
        // one that sees the key MUST see the correct val — the whole bug
        // was a reader observing the key before the val had propagated.
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;

        let registry = Arc::new(SegmentRegistry::new());
        let done = Arc::new(AtomicBool::new(false));
        const N: usize = 700; // well under CAPACITY, no saturation noise

        let readers: Vec<_> = (0..4)
            .map(|_| {
                let registry = Arc::clone(&registry);
                let done = Arc::clone(&done);
                std::thread::spawn(move || {
                    while !done.load(Ordering::Relaxed) {
                        for i in 0..N {
                            let base = 0x2000_0000 + i * 0x1_0000;
                            if let Some(val) = registry.lookup(base) {
                                assert_eq!(
                                    val,
                                    (i % 251) as u8,
                                    "segment {i} visible with wrong class -> stale/zero val race"
                                );
                            }
                        }
                    }
                })
            })
            .collect();

        for i in 0..N {
            let base = 0x2000_0000 + i * 0x1_0000;
            assert!(registry.insert(base, (i % 251) as u8));
        }
        done.store(true, Ordering::Relaxed);
        for r in readers {
            r.join().unwrap();
        }
        // Final sanity: every entry resolves correctly once insertion is done.
        for i in 0..N {
            let base = 0x2000_0000 + i * 0x1_0000;
            assert_eq!(registry.lookup(base), Some((i % 251) as u8));
        }
    }
}
