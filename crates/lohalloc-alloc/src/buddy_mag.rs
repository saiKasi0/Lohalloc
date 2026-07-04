//! Per-thread magazines (tcache) fronting the Buddy backend, for the medium
//! size range that dominates buddy-heavy and mixed-size workloads.
//!
//! Mirrors `magazine.rs`'s design exactly (see its module doc for the full
//! rationale — TLS discipline, ownership, thread-exit strand) but over
//! buddy *orders* instead of slab classes: orders 10..=14 (16 KiB..256 KiB
//! block sizes), the range diagnosed as ~75% of adv-mixed's allocation
//! traffic and the whole of the buddy micro-benchmark. 512 KiB/1 MiB
//! (orders 15, 16) stay direct-to-`Mutex<Buddy>` — they're rare enough in
//! every measured workload that a magazine row for them would only add
//! strand risk for no measured benefit.
//!
//! # Interaction with merge-on-spill coalescing
//!
//! A block sitting in a magazine is, from the central `Buddy`'s point of
//! view, still allocated (its bitmap bit is clear) — magazines are a layer
//! *above* `Buddy`, not a second free-block registry inside it. A flush
//! (`Buddy::dealloc_order_batch`) returns blocks through the exact same
//! `free_order` path a direct free would, so `merge_drain_row`'s
//! cache-row-overflow trigger fires exactly as before; magazines change
//! only *how often* the central buddy is touched, never *what* it does
//! when touched. This is load-bearing: it means the buddy sweep-on-miss
//! regression this crate already suffered once (see `buddy.rs`'s module
//! doc) cannot resurface here — nothing here ever asks `Buddy` to merge
//! anything itself.
//!
//! # Placement
//!
//! Like the slab magazines, these sit *inside* the Buddy arm of
//! `try_backend`/`route_by_size`/dealloc — after the Decision Engine has
//! already chosen Buddy for this call site, never in front of routing.

use core::cell::Cell;

/// Buddy orders fronted by a magazine: 16 KiB (order 10) through 256 KiB
/// (order 14). Kept in sync with `buddy::order_for`'s order numbering —
/// `MIN_BLOCK = 16`, so order `o` holds `16 << o` byte blocks.
pub const MIN_MAGAZINE_ORDER: usize = 10;
pub const MAX_MAGAZINE_ORDER: usize = 14;
const NUM_ORDERS: usize = MAX_MAGAZINE_ORDER - MIN_MAGAZINE_ORDER + 1;

/// Slot-array dimension (the largest per-order cap).
const MAX_CAP: usize = 8;

/// Per-order capacity, largest for the smallest (most common) blocks:
/// 8×16KiB + 8×32KiB + 4×64KiB + 4×128KiB + 2×256KiB ≈ 1.4 MiB worst-case
/// strand per thread — bounded, same rationale as `magazine.rs`'s ~210 KiB.
const ORDER_CAPS: [u8; NUM_ORDERS] = [8, 8, 4, 4, 2];

/// Map a buddy order to its magazine row index, or `None` if `order` is
/// outside the magazined range (falls direct to `Mutex<Buddy>`).
#[inline]
pub fn index_for(order: usize) -> Option<usize> {
    if (MIN_MAGAZINE_ORDER..=MAX_MAGAZINE_ORDER).contains(&order) {
        Some(order - MIN_MAGAZINE_ORDER)
    } else {
        None
    }
}

/// How many blocks a refill asks the central buddy for (half the cap), and
/// how many a flush returns. `idx` is a magazine row index (from
/// `index_for`), not a raw buddy order.
pub fn refill_count(idx: usize) -> usize {
    (ORDER_CAPS[idx] as usize / 2).max(1)
}

// See `magazine.rs`'s identical comment: array-repeat init over a
// non-`Copy` `Cell` needs a `const` item to instantiate fresh per slot, so
// `declare_interior_mutable_const` is a false positive here.
#[allow(clippy::declare_interior_mutable_const)]
const NULL_SLOT: Cell<*mut u8> = Cell::new(core::ptr::null_mut());
#[allow(clippy::declare_interior_mutable_const)]
const EMPTY_ROW: [Cell<*mut u8>; MAX_CAP] = [NULL_SLOT; MAX_CAP];
#[allow(clippy::declare_interior_mutable_const)]
const ZERO_COUNT: Cell<u8> = Cell::new(0);

/// The per-thread magazine set: a fixed stack of ready block pointers per
/// magazined buddy order. All plain `Cell`s — no `Drop`, no interior heap.
/// Ownership/ discard-on-instance-switch semantics are identical to
/// `magazine::Magazine` — see that type's doc for the SIGSEGV history this
/// protects against.
struct Magazine {
    owner: Cell<u64>,
    counts: [Cell<u8>; NUM_ORDERS],
    slots: [[Cell<*mut u8>; MAX_CAP]; NUM_ORDERS],
}

std::thread_local! {
    static MAG: Magazine = const {
        Magazine {
            owner: Cell::new(0),
            counts: [ZERO_COUNT; NUM_ORDERS],
            slots: [EMPTY_ROW; NUM_ORDERS],
        }
    };
}

impl Magazine {
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

/// Pop a ready block for magazine row `idx` from this thread's magazine
/// (owned by `owner`). Returns `None` only when empty for this owner.
#[inline]
pub fn pop(owner: u64, idx: usize) -> Option<*mut u8> {
    debug_assert!(idx < NUM_ORDERS);
    MAG.with(|m| {
        m.ensure_owner(owner);
        let n = m.counts[idx].get();
        if n == 0 {
            return None;
        }
        let n = n - 1;
        m.counts[idx].set(n);
        Some(m.slots[idx][n as usize].get())
    })
}

/// Push a freed block for magazine row `idx` onto this thread's magazine
/// (owned by `owner`). Returns `false` (without storing) when at cap — the
/// caller must flush to the central buddy.
#[inline]
pub fn push(owner: u64, idx: usize, block: *mut u8) -> bool {
    debug_assert!(idx < NUM_ORDERS);
    debug_assert!(!block.is_null());
    MAG.with(|m| {
        m.ensure_owner(owner);
        let n = m.counts[idx].get();
        if n >= ORDER_CAPS[idx] {
            return false;
        }
        m.slots[idx][n as usize].set(block);
        m.counts[idx].set(n + 1);
        true
    })
}

/// Take up to `out.len()` blocks out of this thread's magazine for row
/// `idx` (the flush path handing half the magazine back to the central
/// buddy in one locked batch). Returns how many were written to `out`.
#[inline]
pub fn take(owner: u64, idx: usize, out: &mut [*mut u8]) -> usize {
    debug_assert!(idx < NUM_ORDERS);
    MAG.with(|m| {
        m.ensure_owner(owner);
        let have = m.counts[idx].get() as usize;
        let take = have.min(out.len());
        for (i, slot) in out.iter_mut().enumerate().take(take) {
            *slot = m.slots[idx][have - 1 - i].get();
        }
        m.counts[idx].set((have - take) as u8);
        take
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicU64, Ordering};

    // NOTE: as in `magazine.rs`'s tests, these manufacture fake pointers —
    // the magazine never dereferences its slots, so this is sound for
    // unit-testing the push/pop/take bookkeeping alone.

    fn fake(v: usize) -> *mut u8 {
        v as *mut u8
    }

    fn fresh_owner() -> u64 {
        static NEXT: AtomicU64 = AtomicU64::new(0x2000_0000);
        NEXT.fetch_add(1, Ordering::Relaxed)
    }

    #[test]
    fn index_for_covers_16k_to_256k() {
        assert_eq!(index_for(9), None); // 8 KiB — below range
        assert_eq!(index_for(10), Some(0)); // 16 KiB
        assert_eq!(index_for(11), Some(1)); // 32 KiB
        assert_eq!(index_for(12), Some(2)); // 64 KiB
        assert_eq!(index_for(13), Some(3)); // 128 KiB
        assert_eq!(index_for(14), Some(4)); // 256 KiB
        assert_eq!(index_for(15), None); // 512 KiB — above range
        assert_eq!(index_for(16), None); // 1 MiB — above range
    }

    #[test]
    fn pop_empty_returns_none() {
        let o = fresh_owner();
        assert_eq!(pop(o, 2), None);
    }

    #[test]
    fn push_pop_lifo_roundtrip() {
        let o = fresh_owner();
        assert!(push(o, 0, fake(0x10)));
        assert!(push(o, 0, fake(0x20)));
        assert!(push(o, 0, fake(0x30)));
        assert_eq!(pop(o, 0), Some(fake(0x30)));
        assert_eq!(pop(o, 0), Some(fake(0x20)));
        assert_eq!(pop(o, 0), Some(fake(0x10)));
        assert_eq!(pop(o, 0), None);
    }

    #[test]
    fn push_refuses_at_cap() {
        let o = fresh_owner();
        let idx = 4; // cap 2 (256 KiB row)
        let cap = ORDER_CAPS[idx] as usize;
        for i in 0..cap {
            assert!(push(o, idx, fake(0x100 + i)), "push {i} below cap");
        }
        assert!(!push(o, idx, fake(0x999)), "push at cap must refuse");
        for i in (0..cap).rev() {
            assert_eq!(pop(o, idx), Some(fake(0x100 + i)));
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
        assert_eq!(buf[0], fake(0x1005));
        assert_eq!(buf[3], fake(0x1002));
        assert_eq!(pop(o, 0), Some(fake(0x1001)));
        assert_eq!(pop(o, 0), Some(fake(0x1000)));
        assert_eq!(pop(o, 0), None);
    }

    #[test]
    fn owner_switch_discards_stale_blocks() {
        let a = fresh_owner();
        assert!(push(a, 1, fake(0xAAA0)));
        assert!(push(a, 3, fake(0xAAA1)));
        let b = fresh_owner();
        assert_eq!(pop(b, 1), None, "B must never see A's blocks");
        assert_eq!(pop(b, 3), None);
        assert_eq!(pop(a, 1), None);
    }

    #[test]
    fn refill_count_is_half_cap_min_one() {
        assert_eq!(refill_count(0), 4); // cap 8
        assert_eq!(refill_count(2), 2); // cap 4
        assert_eq!(refill_count(4), 1); // cap 2
    }
}
