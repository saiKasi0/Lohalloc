//! Bump Arena sub-allocator — dense topological clusters.
//!
//! A bump-pointer allocator backed by a single `mmap` region. Allocations
//! advance a cursor forward, aligned to `max(align, MIN_ALIGN)`. There is no
//! per-allocation `free`; memory is reclaimed via [`BumpArena::reset`], which
//! moves the cursor back to the arena base. This is the "reset-based reclaim"
//! model from the v3 spec: the Decision Engine (Phase 3) routes entire
//! topological clusters (temporary, bulk allocations with a shared lifetime)
//! to a Bump Arena and resets it when the cluster is done.
//!
//! # Why no per-allocation free?
//!
//! Bump allocation is the fastest possible allocation strategy: a single
//! pointer increment. Adding per-block free lists would destroy this
//! property. Instead, the arena is reset wholesale. This works because
//! topological clusters identified by the stack hash tend to have bursty,
//! correlated lifetimes (e.g. all allocations within a single request handler).

use crate::system;
use lohalloc_core::{align_up, MIN_ALIGN};

/// Default arena size: 1 MiB. Large enough for most clusters, small enough
/// that we don't waste too much memory if a cluster is small.
const DEFAULT_ARENA_SIZE: usize = 1 << 20; // 1 MiB

/// A bump-pointer allocator backed by a single `mmap` region.
///
/// Allocations are O(1) — just a cursor increment. Deallocation is a no-op;
/// memory is reclaimed via [`reset`](Self::reset).
pub struct BumpArena {
    /// The backing mmap region. Kept alive for the arena's lifetime so the
    /// memory stays mapped. Never directly read — its `Drop` releases the
    /// mapping when the arena is dropped.
    #[allow(dead_code)]
    mapping: system::Mapping,
    /// The base address of the arena (aligned start).
    base: *mut u8,
    /// Total usable capacity from `base`.
    capacity: usize,
    /// Current cursor (next free byte). Always >= `base` and <= `base + capacity`.
    cursor: usize,
}

unsafe impl Send for BumpArena {}

impl BumpArena {
    /// Create a new arena with the default capacity (1 MiB).
    pub fn new() -> Option<Self> {
        Self::with_capacity(DEFAULT_ARENA_SIZE)
    }

    /// Create a new arena with `capacity` bytes. The actual mapping is
    /// rounded up to a whole number of pages and aligned to at least
    /// `MIN_ALIGN`.
    pub fn with_capacity(capacity: usize) -> Option<Self> {
        let align = MIN_ALIGN.max(system::page_size());
        let mapping = system::alloc_pages(capacity, align)?;
        let base = mapping.as_ptr();
        let usable = mapping.usable();
        Some(Self {
            mapping,
            base,
            capacity: usable,
            cursor: base as usize,
        })
    }

    /// Allocate `size` bytes aligned to at least `max(align, MIN_ALIGN)`.
    ///
    /// Returns a pointer to the allocated block, or `None` if the arena is
    /// full.
    ///
    /// # Safety contract for the caller
    /// The returned pointer is valid until `reset` is called. Reading/writing
    /// beyond `size` bytes is UB.
    pub fn alloc(&mut self, size: usize, align: usize) -> Option<*mut u8> {
        if size == 0 {
            return None;
        }

        let align = align.max(MIN_ALIGN);
        let current = self.cursor;
        let aligned = align_up(current, align);

        // Check for overflow / arena full.
        if aligned.checked_add(size)? > (self.base as usize) + self.capacity {
            return None;
        }

        self.cursor = aligned + size;
        Some(aligned as *mut u8)
    }

    /// Reset the arena: move the cursor back to the base. All prior
    /// allocations are invalidated.
    pub fn reset(&mut self) {
        self.cursor = self.base as usize;
    }

    /// Total usable capacity (bytes).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bytes allocated since the last `reset`.
    pub fn used(&self) -> usize {
        self.cursor - (self.base as usize)
    }

    /// Bytes still available.
    pub fn remaining(&self) -> usize {
        self.capacity - self.used()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lohalloc_core::is_aligned;

    #[test]
    fn alloc_and_reset() {
        let mut arena = BumpArena::new().expect("arena");
        let p1 = arena.alloc(64, 16).expect("alloc 1");
        let p2 = arena.alloc(128, 16).expect("alloc 2");
        assert!(!p1.is_null());
        assert!(!p2.is_null());
        assert!(p2 as usize > p1 as usize);

        let used_before = arena.used();
        assert!(used_before >= 64 + 128);

        arena.reset();
        assert_eq!(arena.used(), 0);

        // After reset, the next allocation should start from the base again.
        let p3 = arena.alloc(64, 16).expect("alloc after reset");
        assert_eq!(p3 as usize, arena.base as usize);
    }

    #[test]
    fn alignment_respected() {
        let mut arena = BumpArena::new().expect("arena");
        let p = arena.alloc(100, 32).expect("alloc align 32");
        assert!(is_aligned(p as usize, 32));
    }

    #[test]
    fn arena_full_returns_none() {
        let mut arena = BumpArena::new().expect("arena");
        // Fill the arena completely by allocating until it's full.
        let mut total = 0;
        while arena.alloc(256, 16).is_some() {
            total += 1;
            // Safety valve: don't loop forever if something is wrong.
            if total > 10000 {
                break;
            }
        }
        // Arena is now full; the next allocation should fail.
        let p = arena.alloc(256, 16);
        assert!(p.is_none(), "arena should be full after {} allocs", total);

        // After reset, we should be able to allocate again.
        arena.reset();
        assert!(
            arena.alloc(256, 16).is_some(),
            "arena should work after reset"
        );
    }

    #[test]
    fn used_tracking() {
        let mut arena = BumpArena::with_capacity(4096).expect("arena");
        assert_eq!(arena.used(), 0);

        let _ = arena.alloc(100, 16).expect("alloc");
        // used() should be >= 100 (may include alignment padding).
        assert!(arena.used() >= 100);

        let _ = arena.alloc(200, 16).expect("alloc");
        assert!(arena.used() >= 300);

        arena.reset();
        assert_eq!(arena.used(), 0);
    }

    #[test]
    fn zero_size_alloc_returns_none() {
        let mut arena = BumpArena::new().expect("arena");
        assert!(arena.alloc(0, 16).is_none());
    }

    #[test]
    fn sequential_pointers_advance() {
        let mut arena = BumpArena::new().expect("arena");
        let p1 = arena.alloc(32, 16).expect("a1");
        let p2 = arena.alloc(32, 16).expect("a2");
        let p3 = arena.alloc(32, 16).expect("a3");
        // Pointers should be monotonically increasing.
        assert!(p2 as usize > p1 as usize);
        assert!(p3 as usize > p2 as usize);
    }
}
