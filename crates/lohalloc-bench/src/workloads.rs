//! Backend-pure and adversarial workload generators.
//!
//! Each generator is written once against [`AllocDriver`] and reused by
//! routing-validation tests (`tests/routing_validation.rs`), criterion
//! benches (`benches/*.rs`), and the latency profiler
//! (`src/bin/latency_profile.rs`). Request sizes are chosen relative to
//! [`HEADER_PAD`] so they land exactly on backend size-class boundaries —
//! see `crates/lohalloc-alloc/src/lib.rs::header_pad` for why every
//! allocation carries 48 bytes of header overhead at the default 16-byte
//! alignment.

use core::alloc::Layout;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};

use lohalloc_alloc::Lohalloc;
use lohalloc_core::Backend;

/// Bytes of header padding prepended to every Lohalloc allocation at the
/// default 16-byte alignment. A user request of `class - HEADER_PAD` bytes
/// lands exactly on `class` after the header is added.
pub const HEADER_PAD: usize = 48;

/// Request size that lands exactly on the 256-byte Slab class. The
/// canonical "small, fixed-size, high-churn" request used by W-SLAB and
/// W-ARENA.
pub const SMALL_FIXED_REQUEST: usize = 256 - HEADER_PAD; // 208

/// Variable medium sizes (32 KiB .. 256 KiB), all safely above `SLAB_MAX`
/// (16 KiB) and below `BUDDY_MAX` (1 MiB) even after the header is added, so
/// only Buddy or System can structurally serve them.
pub const BUDDY_SIZES: [usize; 4] = [32 * 1024, 64 * 1024, 128 * 1024, 256 * 1024];

/// Sizes above `BUDDY_MAX`; only the System backend can serve these.
pub const SYSTEM_SIZES: [usize; 2] = [2 * 1024 * 1024, 8 * 1024 * 1024];

/// Request size landing exactly on the 8 KiB Slab class — used by the
/// Step-0 context workloads. Deliberately mid-slab rather than 256 B: the
/// bump arena's budget (32 MiB floor, CPU-scaled; see
/// `arena::scaled_max_chunks`) must *bind within bench-scale op counts* for
/// a reset-free regime to have a real arena-vs-slab tradeoff — at 8 KiB the
/// arena exhausts after ~4-8k routed allocations, at 256 B only after
/// ~250k (which made arena trivially dominate every regime in the first
/// version of this eval).
pub const MID_SLAB_REQUEST: usize = 8192 - HEADER_PAD; // 8144

/// Deterministic synthetic hashes identifying each workload's call site.
/// Used with [`HarnessDriver`] (via `alloc_with_hash`) so routing outcomes
/// are assertable regardless of inlining or call depth — see
/// `crate::forced` and `tests/routing_validation.rs`.
pub mod hashes {
    pub const W_SLAB: u64 = 0x5AA5_0001;
    pub const W_ARENA: u64 = 0x5AA5_0002;
    pub const W_BUDDY: u64 = 0x5AA5_0003;
    pub const W_SYSTEM: u64 = 0x5AA5_0004;
    pub const W_COMBO_SA_SLAB: u64 = 0x5AA5_0005;
    pub const W_COMBO_SA_ARENA: u64 = 0x5AA5_0006;
    pub const W_COMBO_BA_BUDDY: u64 = 0x5AA5_0007;
    pub const W_COMBO_BA_SMALL: u64 = 0x5AA5_0008;
    pub const W_ADV_MIXED: u64 = 0x5AA5_0009;
    pub const W_ADV_EXHAUST: u64 = 0x5AA5_000A;
    // Step-0 oracle-gap workloads (see `benches/context_gap.rs`): each has an
    // `_A`/`_B` pair. The *real*/static variants pass the same hash for both
    // params (one logical call site — what production topology hashing would
    // see); the *oracle* variant passes both, giving each behavioral regime
    // its own identity so a static frozen model can express the per-regime
    // optimum. The timing delta between those two runs is exactly the
    // headroom a context-aware Decision Engine could capture.
    pub const W_PHASE_A: u64 = 0x5AA5_000B;
    pub const W_PHASE_B: u64 = 0x5AA5_000C;
    pub const W_FILL_A: u64 = 0x5AA5_000D;
    pub const W_FILL_B: u64 = 0x5AA5_000E;
    pub const W_DATA_A: u64 = 0x5AA5_000F;
    pub const W_DATA_B: u64 = 0x5AA5_0010;
}

/// A minimal seam so the same workload generator can drive either a private
/// `Lohalloc` instance with a deterministic synthetic hash ([`HarnessDriver`],
/// used for routing/forced-model validation) or the process's real global
/// allocator ([`GlobalDriver`], used for cross-allocator comparison
/// benches).
pub trait AllocDriver {
    /// # Safety
    /// Same contract as `GlobalAlloc::alloc`.
    unsafe fn alloc(&self, layout: Layout, hash: u64) -> *mut u8;
    /// # Safety
    /// `ptr` must have been returned by a prior `alloc` call on this driver
    /// with a matching `layout`.
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout, hash: u64);
    /// Called between bursts in workloads that model a cluster lifetime
    /// ending (e.g. W-ARENA). No-op for drivers with no such concept.
    fn phase_end(&self) {}
}

/// Drives a private `Lohalloc` instance via the replay-engine API
/// (`alloc_with_hash`/`dealloc_with_hash`), using a caller-supplied
/// deterministic hash instead of the real stack walk. This is what makes
/// routing outcomes (`backend_for_ptr`) and forced-model tests
/// (`crate::forced`) assertable: the hash for a given workload/site is
/// always the same value regardless of inlining or call depth.
pub struct HarnessDriver {
    /// **Boxed on purpose** — this is the fix for the criterion-bench O3
    /// compile explosion. `Lohalloc` is a large by-value struct (`MAX_STRIPES
    /// = 32` × two `Mutex<{Slab,Buddy}>` arrays, tens of KB). Held inline it
    /// becomes a giant stack `alloca` with thousands of uses, and LLVM's
    /// `InstCombine` `visitAllocSite` pass is superlinear in an alloca's use
    /// count — after 8→32 that made a single bench crate take 20+ min / OOM
    /// to compile at opt-level>0 (confirmed by sampling rustc: 100% in
    /// `InstCombinerImpl::visitAllocSite`). Behind a `Box` the optimizer only
    /// ever sees an 8-byte pointer — exactly like production, where `Lohalloc`
    /// lives in a `static`, never on the stack. `Box` derefs transparently, so
    /// `driver.alloc.freeze()` etc. compile unchanged; the one-time heap alloc
    /// happens in untimed setup.
    pub alloc: Box<Lohalloc>,
}

impl HarnessDriver {
    pub fn new() -> Self {
        Self::with_alloc(Lohalloc::new())
    }

    /// Wraps a pre-configured (e.g. forced-model) instance on the heap. Every
    /// `HarnessDriver { alloc: ... }` struct literal became this call when the
    /// field became boxed.
    pub fn with_alloc(alloc: Lohalloc) -> Self {
        Self {
            alloc: Box::new(alloc),
        }
    }
}

impl Default for HarnessDriver {
    fn default() -> Self {
        Self::new()
    }
}

impl AllocDriver for HarnessDriver {
    unsafe fn alloc(&self, layout: Layout, hash: u64) -> *mut u8 {
        unsafe { self.alloc.alloc_with_hash(layout, hash) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout, _hash: u64) {
        unsafe { self.alloc.dealloc_with_hash(ptr, layout) }
    }
    fn phase_end(&self) {
        self.alloc.reset_arena();
    }
}

/// Drives the process's real global allocator via `std::alloc`. Used only by
/// the `comparison` bench, which swaps in Lohalloc/jemalloc/mimalloc/system
/// as `#[global_allocator]` per build (see `crate::global_alloc`) and wants
/// to measure each one's effect on the same workload shapes. The `hash`
/// parameter is unused here — the real call-site topology (when the
/// allocator is Lohalloc) is whatever the process's actual stack walker
/// sees from the generator function's call stack.
pub struct GlobalDriver;

impl AllocDriver for GlobalDriver {
    unsafe fn alloc(&self, layout: Layout, _hash: u64) -> *mut u8 {
        unsafe { std::alloc::alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout, _hash: u64) {
        unsafe { std::alloc::dealloc(ptr, layout) }
    }
}

/// W-SLAB: tight churn of a single fixed size, bounded 64-deep live window.
/// Slab-favorable — small, reused, high-frequency alloc/free.
#[inline(never)]
pub fn workload_slab_churn<D: AllocDriver>(driver: &D, hash: u64, ops: usize) {
    let layout = Layout::from_size_align(SMALL_FIXED_REQUEST, 16).unwrap();
    let mut window: VecDeque<*mut u8> = VecDeque::with_capacity(64);
    for _ in 0..ops {
        let p = unsafe { driver.alloc(layout, hash) };
        window.push_back(p);
        if window.len() > 64 {
            let old = window.pop_front().unwrap();
            unsafe { driver.dealloc(old, layout, hash) };
        }
    }
    for p in window {
        unsafe { driver.dealloc(p, layout, hash) };
    }
}

/// W-ARENA: bursts of short-lived allocations dropped en masse, with
/// `phase_end` (arena reset) between bursts. Arena-favorable — dense,
/// same-lifetime clusters; per-allocation free is never on the critical
/// path.
#[inline(never)]
pub fn workload_arena_bursts<D: AllocDriver>(
    driver: &D,
    hash: u64,
    num_bursts: usize,
    burst_size: usize,
) {
    let layout = Layout::from_size_align(SMALL_FIXED_REQUEST, 16).unwrap();
    for _ in 0..num_bursts {
        let mut ptrs = Vec::with_capacity(burst_size);
        for _ in 0..burst_size {
            ptrs.push(unsafe { driver.alloc(layout, hash) });
        }
        for p in ptrs {
            unsafe { driver.dealloc(p, layout, hash) };
        }
        driver.phase_end();
    }
}

/// W-BUDDY: variable medium sizes (32 KiB–256 KiB) with interleaved
/// alloc/free, exercising split/coalesce pressure. Buddy-favorable — too
/// large for Slab, too frequently freed for Arena's no-op-dealloc model to
/// help.
#[inline(never)]
pub fn workload_buddy_interleaved<D: AllocDriver>(driver: &D, hash: u64, ops: usize) {
    let mut live: Vec<(*mut u8, Layout)> = Vec::new();
    for i in 0..ops {
        let size = BUDDY_SIZES[i % BUDDY_SIZES.len()];
        let layout = Layout::from_size_align(size, 16).unwrap();
        let p = unsafe { driver.alloc(layout, hash) };
        live.push((p, layout));
        if i % 2 == 1 && !live.is_empty() {
            let (op, ol) = live.remove(0);
            unsafe { driver.dealloc(op, ol, hash) };
        }
    }
    for (p, l) in live {
        unsafe { driver.dealloc(p, l, hash) };
    }
}

/// W-SYSTEM: large (2 MiB / 8 MiB) allocations, immediately freed. Control
/// case — only the System backend can structurally serve these, so every
/// forced/trained mode should converge identically.
#[inline(never)]
pub fn workload_system_large<D: AllocDriver>(driver: &D, hash: u64, ops: usize) {
    for i in 0..ops {
        let size = SYSTEM_SIZES[i % SYSTEM_SIZES.len()];
        let layout = Layout::from_size_align(size, 16).unwrap();
        let p = unsafe { driver.alloc(layout, hash) };
        // Touch the allocation — mirrors the volatile write in the C
        // generator (`bench/native/workloads.c`), where it is load-bearing:
        // a bare malloc/free pair with an unused pointer gets *deleted
        // entirely* by GCC/Clang at -O2. Kept identical here so the
        // cross-language workload shapes (and their page-fault behavior)
        // match exactly.
        if !p.is_null() {
            unsafe { p.write_volatile(i as u8) };
        }
        unsafe { driver.dealloc(p, layout, hash) };
    }
}

/// W-ADV-MIXED: single call site, erratic sizes (1 B–64 KiB) via a
/// deterministic xorshift-style PRNG, pseudo-random lifetimes. Adversarial —
/// no single backend dominates.
#[inline(never)]
pub fn workload_adversarial_mixed<D: AllocDriver>(driver: &D, hash: u64, ops: usize) {
    let mut live: Vec<(*mut u8, Layout)> = Vec::new();
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    for _ in 0..ops {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let size = 1 + ((state >> 33) as usize % (64 * 1024));
        let layout = Layout::from_size_align(size, 16).unwrap();
        let p = unsafe { driver.alloc(layout, hash) };
        live.push((p, layout));
        if live.len() > 32 && (state & 1) == 0 {
            let idx = (state >> 1) as usize % live.len();
            let (op, ol) = live.swap_remove(idx);
            unsafe { driver.dealloc(op, ol, hash) };
        }
    }
    for (p, l) in live {
        unsafe { driver.dealloc(p, l, hash) };
    }
}

// ---------------------------------------------------------------------------
// Step-0 adversarial-context workloads (oracle-gap eval)
//
// Unlike every workload above, the per-site optimum in these CHANGES AT
// RUNTIME — by phase, by allocator state, or by data — which a static
// `(site, size_class)` frozen verdict cannot express. Each takes TWO hash
// params; see the `hashes` module doc for the real-vs-oracle convention.
// All three use ONE request size ([`MID_SLAB_REQUEST`], a single slab
// class), so `size_class` cannot separate the regimes either — only context
// could. Deliberately **reset-free** (no `phase_end`) except where noted:
// the production LD_PRELOAD/global-allocator deployments the gate measures
// never call `reset_arena`, so the bump arena is a finite budget and
// spending it well IS the contextual decision (the J4/J5 story).
// ---------------------------------------------------------------------------

/// W-PHASE: alternating temporal regimes at one call site, reset-free.
/// Even phases are tight alloc/free churn; odd phases allocate a burst,
/// hold it, and drop it en masse. `ops/8` per phase. A static verdict
/// either burns the whole arena budget on churn (exhausts mid-run → storm)
/// or forfeits bump speed everywhere; a per-regime split can spend the
/// budget on one regime only.
#[inline(never)]
pub fn workload_phase_lifetime<D: AllocDriver>(driver: &D, hash_a: u64, hash_b: u64, ops: usize) {
    let layout = Layout::from_size_align(MID_SLAB_REQUEST, 16).unwrap();
    let phase_len = (ops / 8).max(1);
    let mut done = 0;
    let mut phase = 0usize;
    while done < ops {
        let n = phase_len.min(ops - done);
        if phase % 2 == 0 {
            for _ in 0..n {
                let p = unsafe { driver.alloc(layout, hash_a) };
                unsafe { driver.dealloc(p, layout, hash_a) };
            }
        } else {
            let mut held = Vec::with_capacity(n);
            for _ in 0..n {
                held.push(unsafe { driver.alloc(layout, hash_b) });
            }
            for p in held {
                unsafe { driver.dealloc(p, layout, hash_b) };
            }
        }
        done += n;
        phase += 1;
    }
}

/// W-FILL: one regime *shift* plus the arena-exhaustion state trap (the J4
/// saga as a workload). First half: arena-friendly bursts WITH cluster
/// resets (`phase_end` — the one place resets are modeled, an app that
/// resets during a setup phase). Second half: reset-free steady churn — a
/// bump arena's no-op dealloc means churn *fills it monotonically* until
/// exhaustion, so a static Arena verdict pays a permanent fallthrough storm
/// there while a static Slab verdict forfeits the burst-half bump speed.
/// The optimum flips with allocator state, invisible to a stateless bandit.
#[inline(never)]
pub fn workload_arena_fill_churn<D: AllocDriver>(driver: &D, hash_a: u64, hash_b: u64, ops: usize) {
    let layout = Layout::from_size_align(MID_SLAB_REQUEST, 16).unwrap();
    let half = ops / 2;
    let bursts = 8;
    let burst = (half / bursts).max(1);
    for _ in 0..bursts {
        let mut ptrs = Vec::with_capacity(burst);
        for _ in 0..burst {
            ptrs.push(unsafe { driver.alloc(layout, hash_a) });
        }
        for p in ptrs {
            unsafe { driver.dealloc(p, layout, hash_a) };
        }
        driver.phase_end();
    }
    let mut window: VecDeque<*mut u8> = VecDeque::with_capacity(64);
    for _ in 0..(ops - half) {
        let p = unsafe { driver.alloc(layout, hash_b) };
        window.push_back(p);
        if window.len() > 64 {
            let old = window.pop_front().unwrap();
            unsafe { driver.dealloc(old, layout, hash_b) };
        }
    }
    for p in window {
        unsafe { driver.dealloc(p, layout, hash_b) };
    }
}

/// W-DATA: per-op data-dependent lifetime at one call site, reset-free —
/// the finest context granularity (the allocation-history-register case).
/// A deterministic xorshift "data stream" decides each op: ~70% short-lived
/// (alloc+free immediately), ~30% held and released en masse every 4096
/// ops. Static routing sees one blended optimum; a per-stream split routes
/// each *individual op* by the same data bit the workload itself branches
/// on.
#[inline(never)]
pub fn workload_data_dependent_lifetime<D: AllocDriver>(
    driver: &D,
    hash_a: u64,
    hash_b: u64,
    ops: usize,
) {
    let layout = Layout::from_size_align(MID_SLAB_REQUEST, 16).unwrap();
    let mut state: u64 = 0xA5A5_5A5A_DEAD_BEEF;
    let mut held: Vec<*mut u8> = Vec::with_capacity(4096);
    for i in 0..ops {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        if state % 10 < 7 {
            let p = unsafe { driver.alloc(layout, hash_a) };
            unsafe { driver.dealloc(p, layout, hash_a) };
        } else {
            held.push(unsafe { driver.alloc(layout, hash_b) });
        }
        if i % 4096 == 4095 {
            for p in held.drain(..) {
                unsafe { driver.dealloc(p, layout, hash_b) };
            }
        }
    }
    for p in held {
        unsafe { driver.dealloc(p, layout, hash_b) };
    }
}

/// W-MT-SLAB: `threads` independent threads, each running a same-thread
/// copy of [`workload_slab_churn`] (own live window, own allocs+frees).
/// Isolates magazine/tcache scaling under concurrent but non-cross-thread
/// traffic — no thread ever touches another thread's blocks. Mirrors
/// `bench/native/workloads.c::workload_mt_slab_churn`.
pub fn workload_mt_slab_churn<D: AllocDriver + Sync>(driver: &D, threads: usize, ops: usize) {
    let threads = threads.max(1);
    let per_thread = (ops / threads).max(1);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| workload_slab_churn(driver, 0, per_thread));
        }
    });
}

/// W-MT-MIXED: `threads` independent threads, each running a same-thread
/// copy of [`workload_adversarial_mixed`]. Isolates medium/buddy-range lock
/// contention under concurrent traffic. Mirrors
/// `bench/native/workloads.c::workload_mt_adversarial_mixed`.
pub fn workload_mt_adversarial_mixed<D: AllocDriver + Sync>(
    driver: &D,
    threads: usize,
    ops: usize,
) {
    let threads = threads.max(1);
    let per_thread = (ops / threads).max(1);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| workload_adversarial_mixed(driver, 0, per_thread));
        }
    });
}

/// W-MT-XFREE: `threads` threads paired into producer/consumer roles over a
/// bounded channel — the producer allocates, the consumer frees a
/// *different* thread's allocation. The hard cross-thread-free case: every
/// freed block must migrate back through whatever thread-local structures
/// the allocator uses on the alloc side. Mirrors
/// `bench/native/workloads.c::workload_mt_xfree`'s mailbox-ring design (a
/// bounded `sync_channel` is the Rust-idiomatic equivalent).
pub fn workload_mt_xfree<D: AllocDriver + Sync>(driver: &D, threads: usize, ops: usize) {
    let pairs = (threads / 2).max(1);
    let ops_per_pair = (ops / pairs).max(1);
    let layout = Layout::from_size_align(SMALL_FIXED_REQUEST, 16).unwrap();
    std::thread::scope(|scope| {
        for _ in 0..pairs {
            let (tx, rx) = std::sync::mpsc::sync_channel::<usize>(256);
            scope.spawn(move || {
                for _ in 0..ops_per_pair {
                    let p = unsafe { driver.alloc(layout, 0) };
                    tx.send(p as usize).expect("consumer thread panicked");
                }
            });
            scope.spawn(move || {
                for addr in rx {
                    unsafe { driver.dealloc(addr as *mut u8, layout, 0) };
                }
            });
        }
    });
}

/// W-MT-INTERFERE (J5-B2): `threads` threads doing a FIXED amount of
/// cache-resident application compute with only *occasional* allocation —
/// the allocator-interference benchmark. Every prior workload is a pure
/// alloc/free loop, which measures allocator throughput but not what the
/// allocator does to a real MT application (shared-cache-line traffic,
/// lock stalls, cache pollution bleeding into application work). Here the
/// compute dominates (an FNV-1a pass over a 4 KiB thread-local buffer per
/// iteration) and one 208-byte alloc/free happens every
/// `INTERFERE_ALLOC_EVERY` iterations, so hyperfine's wall-time delta
/// between allocators IS the interference signal — an ideal allocator
/// scores ~1.0 vs any other. Deterministic: fixed iteration count, no
/// wall-clock dependence, buffer contents fold into a volatile sink so the
/// kernel can't be optimized away. Mirrors
/// `bench/native/workloads.c::workload_mt_interfere`.
pub fn workload_mt_interfere<D: AllocDriver + Sync>(driver: &D, threads: usize, ops: usize) {
    let threads = threads.max(1);
    let per_thread = (ops / threads).max(1);
    let layout = Layout::from_size_align(SMALL_FIXED_REQUEST, 16).unwrap();
    std::thread::scope(|scope| {
        for t in 0..threads {
            scope.spawn(move || {
                // 4 KiB thread-local working set: L1-resident, so the only
                // cross-core cache traffic is whatever the ALLOCATOR causes.
                let mut buf = [0u8; INTERFERE_BUF_BYTES];
                let mut seed = 0x9E37_79B9_7F4A_7C15u64 ^ (t as u64);
                let mut held: *mut u8 = core::ptr::null_mut();
                for i in 0..per_thread {
                    // xorshift-fed refresh of a few buffer slots, then a
                    // full FNV-1a pass — the "application work".
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    buf[(seed as usize) & (INTERFERE_BUF_BYTES - 1)] = seed as u8;
                    // FNV a rotating 512-byte window (whole buffer covered
                    // every 8 iterations) — ~0.5 µs of compute per iteration,
                    // enough to dominate the occasional alloc without making
                    // hyperfine rows minutes long.
                    let start = (i * INTERFERE_WINDOW_BYTES) & (INTERFERE_BUF_BYTES - 1);
                    let mut h = 0xcbf2_9ce4_8422_2325u64;
                    for &b in &buf[start..start + INTERFERE_WINDOW_BYTES] {
                        h = (h ^ b as u64).wrapping_mul(0x1_0000_01b3);
                    }
                    // Volatile sink: the compute must not be elided.
                    unsafe { core::ptr::write_volatile(&mut seed, seed ^ h) };
                    // Occasional allocation: hold one block across the gap
                    // (so the allocator's memory placement matters), free it
                    // and take a fresh one every INTERFERE_ALLOC_EVERY iters.
                    if i % INTERFERE_ALLOC_EVERY == 0 {
                        if !held.is_null() {
                            unsafe { driver.dealloc(held, layout, 0) };
                        }
                        held = unsafe { driver.alloc(layout, 0) };
                        // Touch the block so its cache line is genuinely used.
                        unsafe { held.write(seed as u8) };
                    }
                }
                if !held.is_null() {
                    unsafe { driver.dealloc(held, layout, 0) };
                }
            });
        }
    });
}

/// Thread-local working-set size for [`workload_mt_interfere`]: 4 KiB is
/// comfortably L1-resident so the compute kernel itself never causes
/// cross-core traffic.
pub const INTERFERE_BUF_BYTES: usize = 4096;
/// One alloc/free per this many compute iterations in
/// [`workload_mt_interfere`] — allocation is *occasional*, the compute
/// dominates (mirrors `bench/native/workloads.c`).
pub const INTERFERE_ALLOC_EVERY: usize = 8;
/// Bytes FNV-hashed per iteration (a rotating window over the buffer; must
/// divide [`INTERFERE_BUF_BYTES`] so windows never wrap).
pub const INTERFERE_WINDOW_BYTES: usize = 512;

/// W-ADV-EXHAUST: long-lived small allocations, never freed by the
/// generator — returned to the caller so it can free them after asserting
/// on routing/exhaustion behavior. Used with a forced-Arena model to
/// observe the bounded cost of arena exhaustion + fallthrough.
#[inline(never)]
pub fn workload_exhaust_no_free<D: AllocDriver>(
    driver: &D,
    hash: u64,
    ops: usize,
) -> Vec<(*mut u8, Layout)> {
    let layout = Layout::from_size_align(SMALL_FIXED_REQUEST, 16).unwrap();
    let mut live = Vec::with_capacity(ops);
    for _ in 0..ops {
        let p = unsafe { driver.alloc(layout, hash) };
        live.push((p, layout));
    }
    live
}

/// W-COMBO-SA: W-SLAB at one call site interleaved (coarse-grained) with
/// W-ARENA at another. The headline combination hypothesis — a working
/// decision engine should route the two sites to different backends.
#[inline(never)]
pub fn workload_combo_slab_arena<D: AllocDriver>(
    driver: &D,
    slab_hash: u64,
    arena_hash: u64,
    rounds: usize,
) {
    for _ in 0..rounds {
        workload_slab_churn(driver, slab_hash, 200);
        workload_arena_bursts(driver, arena_hash, 1, 500);
    }
}

/// W-COMBO-BA: W-BUDDY at one call site interleaved with small Slab-favorable
/// bursts at another.
#[inline(never)]
pub fn workload_combo_buddy_small<D: AllocDriver>(
    driver: &D,
    buddy_hash: u64,
    small_hash: u64,
    rounds: usize,
) {
    for _ in 0..rounds {
        workload_buddy_interleaved(driver, buddy_hash, 50);
        workload_slab_churn(driver, small_hash, 200);
    }
}

/// Wraps another [`AllocDriver`] and records which backend served each
/// allocation (read via `backend_for_ptr` immediately after `alloc`, before
/// the caller frees it — a System-backed pointer becomes invalid the instant
/// it's `munmap`'d, so inspection must happen at alloc time, not after the
/// workload generator has already freed everything). Used by
/// routing-validation tests to check hypothesis distributions without
/// modifying the workload generators themselves.
pub struct RecordingDriver<'a, D: AllocDriver> {
    inner: &'a D,
    lohalloc: &'a Lohalloc,
    counts: RefCell<HashMap<Backend, usize>>,
    total: Cell<usize>,
}

impl<'a, D: AllocDriver> RecordingDriver<'a, D> {
    /// `lohalloc` must be the same instance `inner` allocates through (so
    /// `backend_for_ptr` reads a valid header) — for [`HarnessDriver`] that's
    /// `&harness.alloc`.
    pub fn new(inner: &'a D, lohalloc: &'a Lohalloc) -> Self {
        Self {
            inner,
            lohalloc,
            counts: RefCell::new(HashMap::new()),
            total: Cell::new(0),
        }
    }

    pub fn fraction_served_by(&self, backend: Backend) -> f64 {
        let total = self.total.get();
        if total == 0 {
            return 0.0;
        }
        let counts = self.counts.borrow();
        *counts.get(&backend).unwrap_or(&0) as f64 / total as f64
    }

    pub fn backend_counts(&self) -> HashMap<Backend, usize> {
        self.counts.borrow().clone()
    }
}

impl<'a, D: AllocDriver> AllocDriver for RecordingDriver<'a, D> {
    unsafe fn alloc(&self, layout: Layout, hash: u64) -> *mut u8 {
        let ptr = unsafe { self.inner.alloc(layout, hash) };
        if !ptr.is_null() {
            if let Some(backend) = unsafe { self.lohalloc.backend_for_ptr(ptr) } {
                *self.counts.borrow_mut().entry(backend).or_insert(0) += 1;
            }
            self.total.set(self.total.get() + 1);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout, hash: u64) {
        unsafe { self.inner.dealloc(ptr, layout, hash) }
    }

    fn phase_end(&self) {
        self.inner.phase_end();
    }
}

/// Wraps another [`AllocDriver`] and records per-call wall-clock latency
/// (nanoseconds) for `alloc` and `dealloc` separately. Used by
/// `src/bin/latency_profile.rs` to build hdrhistogram percentile reports —
/// alloc and dealloc are timed independently rather than stitched into a
/// single per-pair latency, since backends free asynchronously relative to
/// the matching alloc (e.g. Arena's dealloc is a no-op).
pub struct TimingDriver<'a, D: AllocDriver> {
    inner: &'a D,
    alloc_ns: RefCell<Vec<u64>>,
    dealloc_ns: RefCell<Vec<u64>>,
}

impl<'a, D: AllocDriver> TimingDriver<'a, D> {
    pub fn new(inner: &'a D) -> Self {
        Self {
            inner,
            alloc_ns: RefCell::new(Vec::new()),
            dealloc_ns: RefCell::new(Vec::new()),
        }
    }

    pub fn alloc_samples(&self) -> Vec<u64> {
        self.alloc_ns.borrow().clone()
    }

    pub fn dealloc_samples(&self) -> Vec<u64> {
        self.dealloc_ns.borrow().clone()
    }
}

impl<'a, D: AllocDriver> AllocDriver for TimingDriver<'a, D> {
    unsafe fn alloc(&self, layout: Layout, hash: u64) -> *mut u8 {
        let t0 = std::time::Instant::now();
        let ptr = unsafe { self.inner.alloc(layout, hash) };
        self.alloc_ns
            .borrow_mut()
            .push(t0.elapsed().as_nanos() as u64);
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout, hash: u64) {
        let t0 = std::time::Instant::now();
        unsafe { self.inner.dealloc(ptr, layout, hash) };
        self.dealloc_ns
            .borrow_mut()
            .push(t0.elapsed().as_nanos() as u64);
    }

    fn phase_end(&self) {
        self.inner.phase_end();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forced::lohalloc_forced;

    /// Completion smoke: every Step-0 workload runs to completion against a
    /// fresh training-mode instance without crashing or leaking (all pointers
    /// are freed by construction; a buddy/slab invariant violation would
    /// abort or hang here).
    #[test]
    fn context_workloads_complete() {
        let h = HarnessDriver::new();
        workload_phase_lifetime(&h, hashes::W_PHASE_A, hashes::W_PHASE_A, 2000);
        workload_arena_fill_churn(&h, hashes::W_FILL_A, hashes::W_FILL_A, 2000);
        workload_data_dependent_lifetime(&h, hashes::W_DATA_A, hashes::W_DATA_A, 2000);
    }

    /// The oracle convention works end-to-end: a forced model with distinct
    /// per-regime hashes routes each regime to its own backend — this is
    /// what `benches/context_gap.rs`'s `oracle_per_phase` rows rely on.
    /// (Everything goes through the Box'd `HarnessDriver` — a bare by-value
    /// `Lohalloc` overflows the 2 MB test-thread stack; see the field doc.)
    #[test]
    fn oracle_per_phase_hashes_route_independently() {
        let h = HarnessDriver::with_alloc(lohalloc_forced(&[
            (hashes::W_PHASE_A, MID_SLAB_REQUEST, Backend::Slab),
            (hashes::W_PHASE_B, MID_SLAB_REQUEST, Backend::Arena),
        ]));
        let layout = Layout::from_size_align(MID_SLAB_REQUEST, 16).unwrap();
        let pa = unsafe { h.alloc.alloc_with_hash(layout, hashes::W_PHASE_A) };
        let pb = unsafe { h.alloc.alloc_with_hash(layout, hashes::W_PHASE_B) };
        assert_eq!(unsafe { h.alloc.backend_for_ptr(pa) }, Some(Backend::Slab));
        assert_eq!(unsafe { h.alloc.backend_for_ptr(pb) }, Some(Backend::Arena));
        unsafe {
            h.alloc.dealloc_with_hash(pa, layout);
            h.alloc.dealloc_with_hash(pb, layout);
        }
        // And the full oracle workload completes on that model.
        workload_phase_lifetime(&h, hashes::W_PHASE_A, hashes::W_PHASE_B, 2000);
    }
}

#[cfg(all(test, feature = "route-metrics"))]
mod ctx_diag {
    use super::*;
    use lohalloc_core::Backend;

    /// Diagnostic (run explicitly with --ignored --nocapture): what does the
    /// hierarchical freeze actually emit after a phase_lifetime training
    /// pass, and where does the frozen run route?
    #[test]
    #[ignore]
    fn dump_trained_frozen_phase_model() {
        let h = HarnessDriver::new();
        workload_phase_lifetime(&h, hashes::W_PHASE_A, hashes::W_PHASE_A, 10_000);
        eprintln!("== fine map before freeze (hash, sc, ctx, pulls[S,B,Sy,A], means):");
        let mut snap = h.alloc.fine_snapshot();
        snap.sort_by_key(|&(_, _, ctx, _, _)| ctx);
        for (hash, sc, ctx, pulls, means) in snap {
            eprintln!(
                "  {hash:#x} sc={sc} ctx={ctx:2} ({ctx:04b}) pulls={pulls:?} means=[{:.3},{:.3},{:.3},{:.3}]",
                means[0], means[1], means[2], means[3]
            );
        }
        h.alloc.reset_arena();
        h.alloc.freeze();
        let bytes = h.alloc.export().expect("frozen");
        let main = lohalloc_alloc::perfect_hash::PerfectHashTable::deserialize(&bytes)
            .expect("valid model");
        eprintln!("== main entries (hash, sc, backend, flags):");
        for (hash, sc, backend, flags) in main.entries_flagged() {
            eprintln!("  {hash:#018x} sc={sc} {backend:?} flags={flags}");
        }
        // Frozen re-run: where do ops actually land?
        workload_phase_lifetime(&h, hashes::W_PHASE_A, hashes::W_PHASE_A, 10_000);
        eprintln!(
            "== frozen re-run served: slab={} buddy={} system={} arena={} fallthrough={} pht_misses={}",
            lohalloc_alloc::Lohalloc::route_count(Backend::Slab),
            lohalloc_alloc::Lohalloc::route_count(Backend::Buddy),
            lohalloc_alloc::Lohalloc::route_count(Backend::System),
            lohalloc_alloc::Lohalloc::route_count(Backend::Arena),
            lohalloc_alloc::Lohalloc::fallthrough_count(),
            lohalloc_alloc::Lohalloc::pht_miss_count(),
        );
    }
}
