//! Bump Arena sub-allocator — dense topological clusters.
//!
//! A bump-pointer allocator backed by a **chain** of `mmap` chunks.
//! Allocations advance a cursor forward within the current chunk, aligned to
//! `max(align, MIN_ALIGN)`; when the chunk fills, the arena advances to the
//! next chunk in the chain (recycling one rewound by `reset`, or mapping a
//! fresh one) up to the per-arena `max_chunks` cap (see
//! [`scaled_max_chunks`]). There is no per-allocation `free`; memory
//! is reclaimed via [`BumpArena::reset`], which rewinds every chunk's cursor.
//! This is the "reset-based reclaim" model from the v3 spec: the Decision
//! Engine (Phase 3) routes entire topological clusters (temporary, bulk
//! allocations with a shared lifetime) to a Bump Arena and resets it when the
//! cluster is done.
//!
//! # Why chaining (and why a cap)
//!
//! The original arena was a single fixed 1 MiB region that returned `None`
//! forever once full. Under a frozen (Inference) routing model that maps a
//! hot call site to Arena, that meant **every** subsequent allocation at that
//! site paid lock + failed-attempt + full size-chain re-route — a permanent
//! per-op penalty measured as a prime contributor to the adversarial-mixed
//! benchmark's 6-10× gap. Chaining keeps the bump fast path intact while
//! letting the arena grow; the per-arena `max_chunks` cap (32 MiB floor,
//! CPU-scaled higher, see [`scaled_max_chunks`]) bounds memory for workloads
//! that never reset, after which allocation falls back to the size-based
//! chain exactly as before.
//!
//! Chunks are mapped aligned to their (power-of-two-rounded) size, so the
//! default 1 MiB chunks are 1 MiB-aligned — `ptr & !(CHUNK - 1)` recovers a
//! chunk base, which keeps the door open for per-chunk live-count recycling
//! later without another layout change.
//!
//! # Why no per-allocation free?
//!
//! Bump allocation is the fastest possible allocation strategy: a single
//! pointer increment. Adding per-block free lists would destroy this
//! property. Instead, the arena is reset wholesale. This works because
//! topological clusters identified by the stack hash tend to have bursty,
//! correlated lifetimes (e.g. all allocations within a single request
//! handler). Dealloc for Arena-tagged blocks stays a no-op in `lib.rs`.

use core::sync::atomic::{AtomicUsize, Ordering};

use crate::system;
use lohalloc_core::{align_up, round_up_pow2, MIN_ALIGN};

/// Default per-chunk size: 1 MiB. Large enough for most clusters, small
/// enough that we don't waste too much memory if a cluster is small.
const DEFAULT_CHUNK_SIZE: usize = 1 << 20; // 1 MiB

/// Ladder 5 headerless Arena: the chunk size/alignment the dealloc side's
/// mask-probe depends on (`ptr & !(CHUNK_BYTES - 1)` recovers a
/// default-sized chunk's base — chunks are mapped aligned to their size,
/// see the module doc). Only default-sized chunks are ever registered for
/// headerless serving; `with_capacity` arenas (tests) keep headers.
pub(crate) const CHUNK_BYTES: usize = DEFAULT_CHUNK_SIZE;

/// Chunk-cap floor (and the fixed cap for `with_capacity` test arenas):
/// 32 × 1 MiB = 32 MiB. At the cap, `alloc` returns `None` and the caller
/// falls through to size-based routing — the pre-chaining behavior.
const BASE_MAX_CHUNKS: usize = 32;

/// Extra chained chunks permitted **per available CPU**. Each concurrent
/// thread bumps its own 4 KiB TLS span (see `lib.rs`'s `arena_alloc_fast`),
/// so more cores means more live spans in flight before J4-D's drain-reset
/// can reclaim; a bigger arena avoids premature exhaustion on many-core
/// hosts (the 16-vCPU c9g gate) without penalizing small machines. 4 MiB of
/// headroom per core.
const PER_CPU_CHUNKS: usize = 4;

/// Absolute ceiling regardless of core count (128 × 1 MiB = 128 MiB), so a
/// very wide host can't balloon a never-resetting accumulator unbounded.
const MAX_CHUNKS_CAP: usize = 128;

/// The CPU-scaled chunk cap for a production arena (`BumpArena::new`).
/// `clamp(BASE + PER_CPU * ncpus, BASE..=CAP)`; a failed
/// `available_parallelism` (returns `Err`) degrades to `ncpus == 0`, i.e.
/// exactly the old fixed `BASE_MAX_CHUNKS` (32). Called once per arena at
/// construction (a cheap syscall, allocation-free) — never on the hot path.
fn scaled_max_chunks() -> usize {
    let ncpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    (BASE_MAX_CHUNKS + PER_CPU_CHUNKS * ncpus).clamp(BASE_MAX_CHUNKS, MAX_CHUNKS_CAP)
}

/// One mmap-backed bump chunk.
///
/// `cursor` is an `AtomicUsize` (not a plain `usize`) so `lib.rs`'s
/// lock-free fast path can bump it directly via CAS, without taking
/// `Lohalloc`'s `arena` `Mutex` at all — see that module's
/// `arena_alloc_fast`. The slow path here (under the Mutex, single-writer)
/// still uses the same atomic via a plain load/store, which is just as
/// cheap as a bare field access and keeps exactly one source of truth for
/// "how full is this chunk" between the two paths.
pub(crate) struct Chunk {
    /// The backing mapping. Kept alive so the memory stays mapped; its
    /// `Drop` releases the mapping when the arena is dropped.
    #[allow(dead_code)]
    mapping: system::Mapping,
    /// Aligned start of the chunk.
    pub(crate) base: *mut u8,
    /// Usable bytes from `base`.
    pub(crate) capacity: usize,
    /// Next free byte. `base <= cursor <= base + capacity`.
    pub(crate) cursor: AtomicUsize,
}

unsafe impl Send for Chunk {}
unsafe impl Sync for Chunk {}

impl Chunk {
    fn new(size: usize) -> Option<Self> {
        // Align the mapping to the pow2-rounded chunk size (>= page) so the
        // default 1 MiB chunks are 1 MiB-aligned (see module doc).
        let align = round_up_pow2(size).max(system::page_size()).max(MIN_ALIGN);
        let mapping = system::alloc_pages(size, align)?;
        let base = mapping.as_ptr();
        // Clamp to the requested size: over-aligned mappings report up to
        // ~2× `size` usable (alloc_pages over-maps then trims the pointer,
        // not the tail), which would silently double the MAX_CHUNKS byte
        // cap. The clamp keeps "32 chunks × 1 MiB" meaning exactly 32 MiB.
        let capacity = mapping.usable().min(size);
        Some(Self {
            mapping,
            base,
            capacity,
            cursor: AtomicUsize::new(base as usize),
        })
    }

    /// Slow-path bump, called with `Lohalloc`'s `arena` Mutex held against
    /// concurrent *slow-path* callers — but **not** against
    /// `Lohalloc::arena_alloc_fast`, which reads this same chunk's `cursor`
    /// lock-free via the published `arena_chunk` descriptor and never
    /// touches the Mutex at all. Retrying alloc on the *current* chunk here
    /// (before deciding to advance) can therefore race a concurrent
    /// fast-path reader on the exact same `AtomicUsize` — a real,
    /// ThreadSanitizer-confirmed data race (an earlier version used
    /// `self.cursor.get_mut()`, a plain, non-atomic read-modify-write,
    /// reasoning that the Mutex ruled out concurrent access; it only ruled
    /// out concurrent *slow-path* access). Every access must therefore go
    /// through the same atomic compare-exchange loop `arena_alloc_fast`
    /// uses, matching it exactly — bump-once-under-lock is not a
    /// correctness-relevant hot path, so paying a CAS here is free.
    fn alloc(&mut self, size: usize, align: usize) -> Option<*mut u8> {
        let align = align.max(MIN_ALIGN);
        loop {
            let cur = self.cursor.load(Ordering::Relaxed);
            let aligned = align_up(cur, align);
            let new_cur = aligned.checked_add(size)?;
            if new_cur > (self.base as usize) + self.capacity {
                return None;
            }
            if self
                .cursor
                .compare_exchange_weak(cur, new_cur, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return Some(aligned as *mut u8);
            }
        }
    }

    fn used(&self) -> usize {
        self.cursor.load(Ordering::Relaxed) - self.base as usize
    }
}

/// A bump-pointer allocator backed by a chain of `mmap` chunks.
///
/// Allocations are O(1) — a cursor increment, with an amortized chunk
/// advance. Deallocation is a no-op; memory is reclaimed via
/// [`reset`](Self::reset).
pub struct BumpArena {
    /// All mapped chunks. `chunks[current]` is the one being bumped; chunks
    /// before it are full (until `reset` rewinds everything). The Vec only
    /// grows (≤ `max_chunks` entries); its own heap growth is served through
    /// the caller's re-entrancy bypass when this arena lives inside the
    /// process's global allocator.
    chunks: Vec<Chunk>,
    /// Index of the chunk currently being bumped.
    current: usize,
    /// Size for newly mapped chunks (`DEFAULT_CHUNK_SIZE` unless constructed
    /// via `with_capacity`, which tests use to exercise the cap cheaply).
    chunk_size: usize,
    /// Chunk cap for this arena. Production arenas (`new`) scale it to the
    /// host core count (see [`scaled_max_chunks`]); `with_capacity` test
    /// arenas pin it to [`BASE_MAX_CHUNKS`] so cap-exhaustion tests stay
    /// deterministic regardless of the machine they run on.
    max_chunks: usize,
}

unsafe impl Send for BumpArena {}

impl BumpArena {
    /// Create a new arena with the default chunk size (1 MiB) and a
    /// CPU-scaled chunk cap (see [`scaled_max_chunks`]).
    pub fn new() -> Option<Self> {
        Self::with_cap(DEFAULT_CHUNK_SIZE, scaled_max_chunks())
    }

    /// Create a new arena whose chunks are `chunk_size` bytes (rounded up to
    /// whole pages), pinned to the fixed [`BASE_MAX_CHUNKS`] cap. Tests use
    /// this to exercise cap exhaustion cheaply and deterministically.
    pub fn with_capacity(chunk_size: usize) -> Option<Self> {
        Self::with_cap(chunk_size, BASE_MAX_CHUNKS)
    }

    /// Shared constructor: `chunk_size` bytes per chunk, capped at
    /// `max_chunks` chained chunks. The first chunk is mapped eagerly.
    fn with_cap(chunk_size: usize, max_chunks: usize) -> Option<Self> {
        let first = Chunk::new(chunk_size)?;
        let mut chunks = Vec::with_capacity(max_chunks);
        chunks.push(first);
        Some(Self {
            chunks,
            current: 0,
            chunk_size,
            max_chunks,
        })
    }

    /// Allocate `size` bytes aligned to at least `max(align, MIN_ALIGN)`.
    ///
    /// Returns `None` only when `size` can never fit in a chunk or the
    /// `max_chunks` cap is exhausted.
    ///
    /// # Safety contract for the caller
    /// The returned pointer is valid until `reset` is called. Reading/writing
    /// beyond `size` bytes is UB.
    pub fn alloc(&mut self, size: usize, align: usize) -> Option<*mut u8> {
        if size == 0 {
            return None;
        }
        let align = align.max(MIN_ALIGN);

        // Fast path: bump the current chunk.
        if let Some(p) = self.chunks[self.current].alloc(size, align) {
            return Some(p);
        }

        // A request that can't fit even in an empty chunk must fail rather
        // than chain fresh chunks forever.
        if size.checked_add(align)? > self.chunks[self.current].capacity.max(self.chunk_size) {
            return None;
        }

        // Advance: reuse the next already-mapped chunk (rewound by `reset`)
        // or map a new one below the cap.
        loop {
            if self.current + 1 < self.chunks.len() {
                self.current += 1;
            } else if self.chunks.len() < self.max_chunks {
                let chunk = Chunk::new(self.chunk_size)?;
                self.chunks.push(chunk);
                self.current += 1;
            } else {
                return None; // cap reached — caller falls through
            }
            if let Some(p) = self.chunks[self.current].alloc(size, align) {
                return Some(p);
            }
        }
    }

    /// Reset the arena: rewind every chunk's cursor and start bumping from
    /// the first chunk again. All prior allocations are invalidated. Chunks
    /// stay mapped (recycled on the next fill cycle).
    ///
    /// The rewind stores are **atomic** (`store`, not `get_mut`): J4-D fires
    /// this from `lib.rs`'s drain-triggered auto-reset, which can run
    /// concurrently with the lock-free fast path reading these same cursors
    /// via the published `ArenaChunkDescriptor`. `lib.rs`'s epoch-lock
    /// protocol proves no fast-path *carve* CAS overlaps a rewind, but a
    /// non-atomic `get_mut` write racing the descriptor's atomic loads would
    /// still be UB (and a ThreadSanitizer report); an atomic store is
    /// well-defined and just as cheap under `&mut self`.
    pub fn reset(&mut self) {
        for chunk in &mut self.chunks {
            let base = chunk.base as usize;
            chunk.cursor.store(base, Ordering::Relaxed);
        }
        self.current = 0;
    }

    /// The chunk currently being bumped — `lib.rs` reads this to (re)publish
    /// the lock-free fast path's descriptor after taking the slow path
    /// (initial arena creation, a chunk advance, or a `reset`).
    pub(crate) fn current_chunk(&self) -> &Chunk {
        &self.chunks[self.current]
    }

    /// After a failed [`alloc`](Self::alloc): `true` when the failure means
    /// the arena is *permanently* out of memory for chunk-fitting requests
    /// (the `max_chunks` cap is reached and this request would have fit an
    /// empty chunk), as opposed to a one-off oversized request that could
    /// never fit any chunk. `lib.rs` uses this to set its `arena_exhausted`
    /// fast-fail flag — a bump arena never recovers from cap exhaustion
    /// except via [`reset`](Self::reset). Deliberately conservative about
    /// tail space: a smaller later request might still squeeze into the last
    /// chunk's remaining bytes, but that transient (< one chunk) is not
    /// worth re-attempting the Mutex slow path on every allocation forever.
    pub(crate) fn exhausted_after_failed(&self, size: usize, align: usize) -> bool {
        self.chunks.len() == self.max_chunks
            && size.saturating_add(align.max(MIN_ALIGN)) <= self.chunk_size
    }

    /// This arena's chunk cap (CPU-scaled for `new`, fixed for
    /// `with_capacity`). Exposed for tests asserting the scaling.
    #[cfg(test)]
    pub(crate) fn max_chunks(&self) -> usize {
        self.max_chunks
    }

    /// Base address of every currently mapped chunk, for the headerless
    /// chunk registry (`lib.rs` registers them — idempotently, ≤
    /// `max_chunks` entries — on the chunk-creating slow path, before
    /// the current chunk is (re)published to the lock-free fast path).
    pub(crate) fn chunk_bases(&self) -> impl Iterator<Item = usize> + '_ {
        self.chunks.iter().map(|c| c.base as usize)
    }

    /// Whether every chunk is exactly [`CHUNK_BYTES`] (default-sized) —
    /// the mask-probe precondition for headerless serving.
    pub(crate) fn chunks_are_default_sized(&self) -> bool {
        self.chunk_size == DEFAULT_CHUNK_SIZE
            && self.chunks.iter().all(|c| c.capacity <= CHUNK_BYTES)
    }

    /// Total usable capacity across all currently mapped chunks (bytes).
    pub fn capacity(&self) -> usize {
        self.chunks.iter().map(|c| c.capacity).sum()
    }

    /// Bytes allocated since the last `reset`, across all chunks.
    pub fn used(&self) -> usize {
        self.chunks.iter().map(Chunk::used).sum()
    }

    /// Bytes still available in mapped chunks (does not count unmapped
    /// potential growth up to the cap).
    pub fn remaining(&self) -> usize {
        self.capacity() - self.used()
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
        assert_eq!(p3 as usize, arena.chunks[0].base as usize);
    }

    #[test]
    fn alignment_respected() {
        let mut arena = BumpArena::new().expect("arena");
        let p = arena.alloc(100, 32).expect("alloc align 32");
        assert!(is_aligned(p as usize, 32));
    }

    #[test]
    fn chains_past_one_chunk() {
        // >1 chunk of total traffic must keep succeeding (the original
        // single-region arena failed here forever). Derive the alloc count
        // from the *actual* chunk capacity — `with_capacity(4096)` maps a
        // whole page, which is 16 KiB on Apple Silicon (never assume 4 KiB).
        let mut arena = BumpArena::with_capacity(4096).expect("arena");
        let first_chunk_cap = arena.chunks[0].capacity;
        let allocs = first_chunk_cap / 256 + 8; // guaranteed to spill over
        for i in 0..allocs {
            assert!(
                arena.alloc(256, 16).is_some(),
                "alloc {i}/{allocs} must succeed while below the cap"
            );
        }
        assert!(arena.chunks.len() > 1, "expected multiple chunks");
    }

    #[test]
    fn arena_full_returns_none_at_cap() {
        // Small chunks so the BASE_MAX_CHUNKS cap is reachable quickly:
        // 32 chunks × 4 KiB = 128 KiB total. `with_capacity` pins the cap to
        // BASE_MAX_CHUNKS so this is deterministic on any host.
        let mut arena = BumpArena::with_capacity(4096).expect("arena");
        assert_eq!(arena.max_chunks(), BASE_MAX_CHUNKS);
        let mut total = 0;
        while arena.alloc(256, 16).is_some() {
            total += 1;
            if total > 10_000 {
                break;
            }
        }
        assert_eq!(
            arena.chunks.len(),
            BASE_MAX_CHUNKS,
            "should have chained to the cap"
        );
        assert!(
            arena.alloc(256, 16).is_none(),
            "cap reached after {total} allocs — further allocs must fail"
        );

        // Reset recycles every chunk: allocation works again without any
        // new mapping.
        let mapped = arena.chunks.len();
        arena.reset();
        assert_eq!(arena.used(), 0);
        assert!(arena.alloc(256, 16).is_some(), "works after reset");
        assert_eq!(arena.chunks.len(), mapped, "reset must not map chunks");
    }

    #[test]
    fn oversized_request_fails_without_chaining() {
        let mut arena = BumpArena::with_capacity(4096).expect("arena");
        let before = arena.chunks.len();
        assert!(arena.alloc(1 << 20, 16).is_none());
        assert_eq!(
            arena.chunks.len(),
            before,
            "an unservable request must not burn chunks"
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

    #[test]
    fn cpu_scaled_cap_is_at_least_base_and_within_ceiling() {
        // A production arena's cap floors at BASE_MAX_CHUNKS and never
        // exceeds MAX_CHUNKS_CAP, regardless of the host core count. It must
        // also match the standalone scaler exactly (single source of truth).
        let arena = BumpArena::new().expect("arena");
        let cap = arena.max_chunks();
        assert!(cap >= BASE_MAX_CHUNKS, "cap {cap} below floor");
        assert!(cap <= MAX_CHUNKS_CAP, "cap {cap} above ceiling");
        assert_eq!(cap, scaled_max_chunks(), "arena cap must match scaler");
    }

    #[test]
    fn scaled_max_chunks_is_monotone_and_clamped() {
        // The scaling formula, exercised directly (available_parallelism is
        // fixed on a given host, so drive the math via the same clamp). More
        // cores never shrinks the cap; the ceiling always holds.
        let f = |n: usize| {
            (BASE_MAX_CHUNKS + PER_CPU_CHUNKS * n).clamp(BASE_MAX_CHUNKS, MAX_CHUNKS_CAP)
        };
        assert_eq!(f(0), BASE_MAX_CHUNKS, "Err/zero-cpu degrades to the floor");
        assert!(f(1) >= f(0));
        assert!(f(16) >= f(8));
        assert!(f(8) >= f(1));
        assert_eq!(
            f(1_000_000),
            MAX_CHUNKS_CAP,
            "huge core count clamps to ceiling"
        );
    }

    #[test]
    fn default_chunks_are_chunk_aligned() {
        // 1 MiB chunks must be 1 MiB-aligned so `ptr & !(1MiB-1)` recovers
        // the chunk base (future live-count recycling relies on this).
        let arena = BumpArena::new().expect("arena");
        assert!(is_aligned(
            arena.chunks[0].base as usize,
            DEFAULT_CHUNK_SIZE
        ));
    }
}
