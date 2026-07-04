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
            match self.keys[idx].compare_exchange(0, base, Ordering::Release, Ordering::Relaxed) {
                Ok(_) => {
                    self.vals[idx].store(class, Ordering::Release);
                    return true;
                }
                Err(existing) if existing == base => return true, // already registered
                Err(_) => idx = (idx + 1) & (CAPACITY - 1),
            }
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
