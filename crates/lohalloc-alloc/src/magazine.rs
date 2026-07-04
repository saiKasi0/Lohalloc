//! Per-thread magazines (tcache) for the Slab backend.
//!
//! jemalloc/mimalloc serve small allocations from per-thread caches with no
//! locks or atomics on the fast path; Lohalloc's slab previously took the
//! whole-`Slab` `Mutex` on **every** alloc and free (an uncontended
//! lock/unlock pair per op — the single biggest small-alloc cost after the
//! stack walk). A magazine is a fixed-capacity per-thread, per-class stack
//! of ready blocks: alloc = TLS pop, free = TLS push; only magazine
//! *misses* (empty on alloc, full on free) touch the central slab, in
//! batches of half a magazine, amortizing the lock to ~1/(cap/2) ops.
//!
//! # Placement
//!
//! Magazines live **inside** the Slab backend path, after routing — never in
//! front of `route_alloc`. The Decision Engine routes per call site
//! (Slab vs Arena vs …); a cache in front of routing would bypass that
//! decision and break per-site semantics.
//!
//! # TLS discipline (load-bearing)
//!
//! Everything here is a dtor-free `Cell` in a `const`-initialized
//! `thread_local!`. The crate invariant (see `lohalloc-cabi`'s module doc)
//! is that alloc-path TLS must have **no destructors**: a TLS dtor could
//! run during thread teardown and re-enter the allocator or deallocate,
//! deadlocking or corrupting state.
//!
//! # Thread exit = bounded strand, by design
//!
//! When a thread dies, the blocks in its magazine are stranded (never
//! returned to the central free lists). This is deliberate: per-class caps
//! bound the strand to ≲200 KiB per thread, the central slab never unmaps
//! regions anyway (so this leaks *reuse*, not *memory mappings*), and the
//! alternative (a pthread-key flush destructor) reintroduces exactly the
//! TLS-dtor hazard the invariant above forbids. Revisit only if long-lived
//! processes with heavy thread churn become a target.

use core::cell::Cell;

use lohalloc_core::SLAB_SIZE_CLASSES;

/// One magazine per slab size class.
pub const NUM_CLASSES: usize = SLAB_SIZE_CLASSES.len();

/// Slot-array dimension (the largest per-class cap).
const MAX_CAP: usize = 32;

/// Per-class capacity: generous for small classes, shrinking with block
/// size so the worst-case strandable bytes per thread stay bounded:
/// 32×(8+16+32+64+128+256) + 16×(512+1024) + 8×(2048+4096) + 4×(8192+16384)
/// ≈ 210 KiB.
const CLASS_CAPS: [u8; NUM_CLASSES] = [32, 32, 32, 32, 32, 32, 16, 16, 8, 8, 4, 4];

/// How many blocks a refill asks the central slab for (half the cap), and
/// how many a flush returns. Half-full hysteresis avoids ping-ponging a
/// magazine that sits right at a boundary.
pub fn refill_count(class: usize) -> usize {
    (CLASS_CAPS[class] as usize / 2).max(1)
}

// `[CONST_ITEM; N]` array-repeat works for non-Copy types (promoted
// constants) — this is what keeps the whole struct const-constructible on
// the workspace MSRV without inline-const blocks. The
// `declare_interior_mutable_const` lint exists to catch consts mistaken
// for shared state; here each use-site deliberately *instantiates a fresh
// Cell* (that's what array-repeat init needs), so the lint is a false
// positive by construction.
#[allow(clippy::declare_interior_mutable_const)]
const NULL_SLOT: Cell<*mut u8> = Cell::new(core::ptr::null_mut());
#[allow(clippy::declare_interior_mutable_const)]
const EMPTY_ROW: [Cell<*mut u8>; MAX_CAP] = [NULL_SLOT; MAX_CAP];
#[allow(clippy::declare_interior_mutable_const)]
const ZERO_COUNT: Cell<u8> = Cell::new(0);

/// The per-thread magazine set: a fixed stack of ready block pointers per
/// slab class. All plain `Cell`s — no `Drop`, no interior heap.
///
/// # Ownership (load-bearing — a SIGSEGV taught us this)
///
/// This TLS is process-wide, but the crate deliberately supports **multiple
/// `Lohalloc` instances** (the replay engine and the forced-routing tests
/// create private ones). A magazine holding instance A's blocks must never
/// serve instance B: B's central slab doesn't own those regions, and once A
/// is dropped its regions are unmapped — popping A's block from B and
/// writing a header into it is a use-after-munmap (reproduced as a SIGSEGV
/// in `routing_validation`). Every operation therefore carries the calling
/// instance's unique `owner` id; on mismatch the magazine **discards** its
/// stale contents (counts zeroed, pointers never dereferenced) and adopts
/// the new owner. Discarding strands at most ~200 KiB of the old instance's
/// blocks per (thread × instance switch) — a bounded leak, never
/// corruption; the production configuration (one static allocator) never
/// switches at all.
struct Magazine {
    owner: Cell<u64>,
    counts: [Cell<u8>; NUM_CLASSES],
    slots: [[Cell<*mut u8>; MAX_CAP]; NUM_CLASSES],
}

std::thread_local! {
    static MAG: Magazine = const {
        Magazine {
            owner: Cell::new(0),
            counts: [ZERO_COUNT; NUM_CLASSES],
            slots: [EMPTY_ROW; NUM_CLASSES],
        }
    };
}

impl Magazine {
    /// Ensure this magazine belongs to `owner`, discarding stale contents
    /// from a previous instance if not. `owner` ids are unique and never
    /// reused (monotonic counter), so a stale match is impossible.
    #[inline]
    fn ensure_owner(&self, owner: u64) {
        debug_assert!(owner != 0, "owner id 0 is reserved for 'unassigned'");
        if self.owner.get() != owner {
            for c in &self.counts {
                c.set(0);
            }
            self.owner.set(owner);
        }
    }
}

/// Pop a ready block for `class` from this thread's magazine (owned by
/// `owner`). Returns `None` only when the magazine is empty for this owner.
#[inline]
pub fn pop(owner: u64, class: usize) -> Option<*mut u8> {
    debug_assert!(class < NUM_CLASSES);
    MAG.with(|m| {
        m.ensure_owner(owner);
        let n = m.counts[class].get();
        if n == 0 {
            return None;
        }
        let n = n - 1;
        m.counts[class].set(n);
        Some(m.slots[class][n as usize].get())
    })
}

/// Push a freed block for `class` onto this thread's magazine (owned by
/// `owner`). Returns `false` (without storing) when the magazine is at its
/// cap — the caller must flush to the central slab.
#[inline]
pub fn push(owner: u64, class: usize, block: *mut u8) -> bool {
    debug_assert!(class < NUM_CLASSES);
    debug_assert!(!block.is_null());
    MAG.with(|m| {
        m.ensure_owner(owner);
        let n = m.counts[class].get();
        if n >= CLASS_CAPS[class] {
            return false;
        }
        m.slots[class][n as usize].set(block);
        m.counts[class].set(n + 1);
        true
    })
}

/// Take up to `out.len()` blocks out of this thread's magazine for `class`
/// (used by the flush path to hand half the magazine back to the central
/// slab in one locked batch). Returns how many were written to `out`.
#[inline]
pub fn take(owner: u64, class: usize, out: &mut [*mut u8]) -> usize {
    debug_assert!(class < NUM_CLASSES);
    MAG.with(|m| {
        m.ensure_owner(owner);
        let have = m.counts[class].get() as usize;
        let take = have.min(out.len());
        for (i, slot) in out.iter_mut().enumerate().take(take) {
            *slot = m.slots[class][have - 1 - i].get();
        }
        m.counts[class].set((have - take) as u8);
        take
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicU64, Ordering};

    // NOTE: tests here manufacture fake pointers (small non-null integers
    // cast to pointers). The magazine never dereferences its slots — it is
    // a pure pointer stack — so this is sound for unit-testing the
    // push/pop/take bookkeeping.

    fn fake(v: usize) -> *mut u8 {
        v as *mut u8
    }

    /// A fresh, never-before-used owner id per test — the ownership check
    /// itself guarantees isolation between tests sharing a test thread.
    fn fresh_owner() -> u64 {
        static NEXT: AtomicU64 = AtomicU64::new(0x1000_0000);
        NEXT.fetch_add(1, Ordering::Relaxed)
    }

    #[test]
    fn pop_empty_returns_none() {
        let o = fresh_owner();
        assert_eq!(pop(o, 3), None);
    }

    #[test]
    fn push_pop_lifo_roundtrip() {
        let o = fresh_owner();
        assert!(push(o, 2, fake(0x10)));
        assert!(push(o, 2, fake(0x20)));
        assert!(push(o, 2, fake(0x30)));
        assert_eq!(pop(o, 2), Some(fake(0x30)));
        assert_eq!(pop(o, 2), Some(fake(0x20)));
        assert_eq!(pop(o, 2), Some(fake(0x10)));
        assert_eq!(pop(o, 2), None);
    }

    #[test]
    fn push_refuses_at_cap() {
        let o = fresh_owner();
        let class = 11; // cap 4
        let cap = CLASS_CAPS[class] as usize;
        for i in 0..cap {
            assert!(push(o, class, fake(0x100 + i)), "push {i} below cap");
        }
        assert!(!push(o, class, fake(0x999)), "push at cap must refuse");
        // Still exactly `cap` blocks, LIFO order preserved.
        for i in (0..cap).rev() {
            assert_eq!(pop(o, class), Some(fake(0x100 + i)));
        }
    }

    #[test]
    fn take_drains_most_recent_first() {
        let o = fresh_owner();
        for i in 0..6 {
            assert!(push(o, 0, fake(0x1000 + i)));
        }
        let mut buf = [core::ptr::null_mut(); 4];
        let n = take(o, 0, &mut buf);
        assert_eq!(n, 4);
        assert_eq!(buf[0], fake(0x1005)); // top of stack first
        assert_eq!(buf[3], fake(0x1002));
        // Two left underneath.
        assert_eq!(pop(o, 0), Some(fake(0x1001)));
        assert_eq!(pop(o, 0), Some(fake(0x1000)));
        assert_eq!(pop(o, 0), None);
    }

    #[test]
    fn take_handles_short_magazine() {
        let o = fresh_owner();
        assert!(push(o, 5, fake(0xA0)));
        let mut buf = [core::ptr::null_mut(); 8];
        assert_eq!(take(o, 5, &mut buf), 1);
        assert_eq!(buf[0], fake(0xA0));
        assert_eq!(take(o, 5, &mut buf), 0);
    }

    #[test]
    fn owner_switch_discards_stale_blocks() {
        // Instance A's cached blocks must be invisible to instance B (they
        // may point into A's — possibly already unmapped — regions), and a
        // later A' (new id) must not see them either: discard-on-switch.
        let a = fresh_owner();
        assert!(push(a, 4, fake(0xAAA0)));
        assert!(push(a, 7, fake(0xAAA1)));
        let b = fresh_owner();
        assert_eq!(pop(b, 4), None, "B must never see A's blocks");
        assert_eq!(pop(b, 7), None);
        // And A's blocks are gone for good (discarded, not resurrected).
        assert_eq!(pop(a, 4), None);
    }

    #[test]
    fn refill_count_is_half_cap_min_one() {
        assert_eq!(refill_count(0), 16);
        assert_eq!(refill_count(7), 8);
        assert_eq!(refill_count(11), 2);
    }
}
