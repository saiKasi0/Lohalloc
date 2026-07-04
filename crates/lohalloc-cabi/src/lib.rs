//! C ABI (malloc family) over [`Lohalloc`], letting a C/C++ program use
//! Lohalloc as its allocator via `LD_PRELOAD` (Linux — see the Phase 6 plan;
//! macOS malloc-zone interposition is out of scope). Exports the standard
//! malloc-family symbols with `#[no_mangle] extern "C"` so the dynamic
//! linker's symbol interposition makes them win over glibc's for any
//! process this library is preloaded into.
//!
//! # Foreign pointers
//!
//! Every pointer this library hands out has a Lohalloc `Header` immediately
//! before it. If `free`/`realloc`/`malloc_usable_size` is called with a
//! pointer this library did **not** allocate (e.g. one handed out by glibc
//! before this library's symbols won out, or before `LOHALLOC_FREEZE_AFTER`
//! bookkeeping was installed), [`Lohalloc::owns`] returns `false` and we
//! delegate to the real libc via `dlsym(RTLD_NEXT, ...)`, lazily resolved
//! and cached once per process — the standard technique for a safe
//! `LD_PRELOAD` interposer. If libc's symbol can't be resolved (should not
//! happen under a normal dynamic link), we leak rather than risk corrupting
//! an allocator we don't understand.
//!
//! # Re-entrancy / init order
//!
//! `Lohalloc`'s thread-local recursion guard (`IN_ALLOC`) and its
//! `Mutex`-guarded backends never themselves call back into these exported
//! symbols, so there's no interposition re-entrancy hazard. Its
//! thread-locals use dtor-free `Cell`s, so no allocation happens during
//! thread teardown either.
//!
//! # Fork / thread safety
//!
//! Not a specific goal for Phase 6 (single-process, single-run benchmarks).
//! `Lohalloc`'s backends are already `Mutex`-guarded, so concurrent
//! malloc/free from multiple threads is sound; `fork()` while another
//! thread holds a backend lock is a known general hazard for *any*
//! allocator and isn't specifically addressed here.

use core::alloc::{GlobalAlloc, Layout};
use core::ffi::c_void;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::os::raw::{c_char, c_int};
use std::sync::OnceLock;

use lohalloc_alloc::Lohalloc;

/// The process-wide allocator instance every exported symbol routes
/// through.
static ALLOC: Lohalloc = Lohalloc::new();

/// Minimum alignment every allocation satisfies, mirroring
/// `lohalloc_alloc`'s internal `MIN_ALIGN` (16 bytes, SIMD-friendly).
const MIN_ALIGN: usize = 16;

// ---------------------------------------------------------------------------
// Foreign-pointer delegation: dlsym(RTLD_NEXT, ...), resolved once.
// ---------------------------------------------------------------------------

type FreeFn = unsafe extern "C" fn(*mut c_void);
type ReallocFn = unsafe extern "C" fn(*mut c_void, usize) -> *mut c_void;
type MallocUsableSizeFn = unsafe extern "C" fn(*mut c_void) -> usize;

/// Resolve `name`'s address in the next library in the dynamic-link search
/// order (i.e. the real libc, since we're preloaded ahead of it).
///
/// # Safety
/// `F` must exactly match the real signature of the symbol named `name`.
unsafe fn resolve_next<F: Copy>(name: &[u8]) -> Option<F> {
    debug_assert_eq!(*name.last().unwrap(), 0, "name must be nul-terminated");
    let sym = unsafe { libc::dlsym(libc::RTLD_NEXT, name.as_ptr() as *const c_char) };
    if sym.is_null() {
        return None;
    }
    // SAFETY: `sym` is non-null and `F` is documented (by the caller) to
    // match the real symbol's signature; both are pointer-sized.
    Some(unsafe { core::mem::transmute_copy::<*mut c_void, F>(&sym) })
}

fn libc_free() -> Option<FreeFn> {
    static CELL: OnceLock<Option<FreeFn>> = OnceLock::new();
    *CELL.get_or_init(|| unsafe { resolve_next(b"free\0") })
}

fn libc_realloc() -> Option<ReallocFn> {
    static CELL: OnceLock<Option<ReallocFn>> = OnceLock::new();
    *CELL.get_or_init(|| unsafe { resolve_next(b"realloc\0") })
}

fn libc_malloc_usable_size() -> Option<MallocUsableSizeFn> {
    static CELL: OnceLock<Option<MallocUsableSizeFn>> = OnceLock::new();
    *CELL.get_or_init(|| unsafe { resolve_next(b"malloc_usable_size\0") })
}

// ---------------------------------------------------------------------------
// Env-var control surface for unmodified C/C++ programs:
//
// - LOHALLOC_FREEZE_AFTER=<n>     auto-freeze after n successful mallocs.
// - LOHALLOC_EXPORT_MODEL=<path>  write the frozen `.lohalloc` model to
//                                 <path> the moment the freeze happens
//                                 (auto or explicit). Enables the Phase 6
//                                 "train once, measure inference in a fresh
//                                 process" benchmark methodology.
// - LOHALLOC_MODEL=<path>         load a `.lohalloc` model before the first
//                                 top-level malloc — the process starts
//                                 directly in Inference mode with zero
//                                 training. Only meaningful now that stack
//                                 hashes are module-base-normalized
//                                 (ASLR-stable); the model must come from
//                                 the same binary/architecture.
// ---------------------------------------------------------------------------

static MALLOC_COUNT: AtomicU64 = AtomicU64::new(0);

/// Set once the auto-freeze (or an explicit freeze/model load) has happened.
/// Lets `maybe_auto_freeze` skip the global `MALLOC_COUNT.fetch_add` for the
/// rest of the run — before this flag existed, an "inference" benchmark run
/// (`LOHALLOC_FREEZE_AFTER=1000`, 50k+ ops) kept paying a contended global
/// RMW on every single malloc long after the freeze had fired.
static FREEZE_FIRED: AtomicBool = AtomicBool::new(false);

std::thread_local! {
    /// Re-entrancy depth for this thread's exported allocation calls. See
    /// `with_freeze_check` for why this exists.
    static IN_ALLOC_FN: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

fn freeze_after_threshold() -> Option<u64> {
    static CELL: OnceLock<Option<u64>> = OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("LOHALLOC_FREEZE_AFTER")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .filter(|&n| n > 0)
            // The env var wins; the tune file's `freeze_after` key is the
            // fallback (`ensure_model_loaded` has already populated the
            // config by the time any top-level malloc gets here).
            .or(lohalloc_alloc::tune::config().freeze_after)
    })
}

/// How often (in successful top-level mallocs) the `freeze_mode=converged`
/// path polls `Lohalloc::is_converged()`. Polling takes the `state` Mutex
/// and walks every Signature's arms, so it must not run per-op; every 256
/// ops keeps the freeze point within noise of the true convergence point
/// while costing ~nothing between polls.
const CONVERGENCE_POLL_EVERY: u64 = 256;

/// Called after every successful "real" (non-delegated) allocation. Freezes
/// exactly once — at a fixed op count (`freeze_mode=ops`, the default) or
/// at detected bandit convergence (`freeze_mode=converged`, with the op
/// count as a hard cap if also set) — then becomes a single relaxed load
/// for the rest of the process lifetime.
#[inline]
fn maybe_auto_freeze() {
    if FREEZE_FIRED.load(Ordering::Relaxed) {
        return;
    }
    let threshold = freeze_after_threshold();
    let converged_mode =
        lohalloc_alloc::tune::config().freeze_mode == lohalloc_alloc::tune::FreezeMode::Converged;
    if threshold.is_none() && !converged_mode {
        return;
    }
    let count = MALLOC_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    // `>=` + `swap` (not `== threshold`): several threads can cross the
    // threshold simultaneously; exactly one wins the swap and freezes.
    let ops_hit = threshold.is_some_and(|t| count >= t);
    let converged = converged_mode && count % CONVERGENCE_POLL_EVERY == 0 && ALLOC.is_converged();
    if (ops_hit || converged) && !FREEZE_FIRED.swap(true, Ordering::Relaxed) {
        ALLOC.freeze();
        // Safe here: we are inside `with_freeze_check`'s depth bump, so any
        // allocation the export triggers (env read, Vec, file I/O buffers)
        // re-enters `malloc` at depth > 0 and skips this whole layer.
        maybe_export_model();
    }
}

/// If `LOHALLOC_EXPORT_MODEL` is set, write the (just-frozen) routing table
/// there — tmp-file + rename so a crash mid-write never leaves a truncated
/// model. Failures warn on stderr and continue; the benchmark script is the
/// layer that treats a missing model file as fatal.
///
/// Must only be called with `IN_ALLOC_FN` depth > 0 (i.e. from inside
/// `with_freeze_check` or another depth bump): `std::env::var`, `format!`,
/// and `std::fs` all allocate, and those nested mallocs must skip the
/// top-level init/freeze layer (see `with_freeze_check`'s doc).
fn maybe_export_model() {
    let Ok(path) = std::env::var("LOHALLOC_EXPORT_MODEL") else {
        return;
    };
    if path.is_empty() {
        return;
    }
    match ALLOC.export() {
        Some(bytes) => {
            let tmp = format!("{path}.tmp");
            if let Err(e) = std::fs::write(&tmp, &bytes).and_then(|()| std::fs::rename(&tmp, &path))
            {
                eprintln!("lohalloc: failed to write model to {path}: {e}");
            }
        }
        None => eprintln!("lohalloc: LOHALLOC_EXPORT_MODEL set but allocator is not frozen"),
    }
}

/// If `LOHALLOC_MODEL` is set, load that `.lohalloc` model exactly once,
/// before the first top-level allocation is served. On success the process
/// runs pure Inference from its very first malloc (and `FREEZE_FIRED` is
/// set so the `LOHALLOC_FREEZE_AFTER` counter is never touched). On any
/// failure: warn once on stderr and stay in Training.
///
/// Re-entrancy: only ever called at `IN_ALLOC_FN` depth 0, from
/// `with_freeze_check`, *after* the depth bump — so the nested allocations
/// its own body triggers (env read, `fs::read`'s Vec, the deserialized
/// table) re-enter `malloc` at depth > 0 and can never re-enter this
/// non-reentrant `OnceLock` on the same thread. This is the same structure
/// that fixed the documented `LOHALLOC_FREEZE_AFTER` env-var deadlock.
fn ensure_model_loaded() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        // The whole body runs under the bootstrap guard: `env::var`'s own
        // first-ever internal setup and `fs::read`'s `Vec<u8>` growth would
        // otherwise be the *first-ever* calls into `ALLOC.alloc()` for this
        // process, landing wherever ordinary Training-mode routing sends
        // them (often Slab) — silently populating the Slab with a real
        // region *before* `load()` ever runs, which permanently (and
        // invisibly) disables `load()`'s header-free fast path (its
        // `Slab::region_count() == 0` safety check would then see a
        // nonzero count and correctly, but unhelpfully, decline). See
        // `Lohalloc::with_bootstrap_guard`'s doc.
        Lohalloc::with_bootstrap_guard(|| {
            // Install the training config (LOHALLOC_TUNE file + LOHALLOC_*
            // env overrides) before any training traffic — this is the one
            // guarded bootstrap point where its env/file reads are safe
            // (see `tune::load_from_env`'s re-entrancy contract). Cheap
            // when unconfigured: two absent env-var reads.
            lohalloc_alloc::tune::load_from_env();
            let Ok(path) = std::env::var("LOHALLOC_MODEL") else {
                return;
            };
            if path.is_empty() {
                return;
            }
            match std::fs::read(&path) {
                Ok(bytes) => {
                    if ALLOC.load(&bytes) {
                        FREEZE_FIRED.store(true, Ordering::Relaxed);
                        // Debug probe: report at exit how many Inference
                        // lookups missed the loaded model — ~0 proves the
                        // model's (ASLR-normalized) keys matched this fresh
                        // process's call sites end to end.
                        if std::env::var("LOHALLOC_DEBUG").is_ok() {
                            unsafe {
                                libc::atexit(report_pht_misses_at_exit);
                            }
                        }
                    } else {
                        eprintln!(
                            "lohalloc: LOHALLOC_MODEL={path} is malformed — staying in training"
                        );
                    }
                }
                Err(e) => {
                    eprintln!(
                        "lohalloc: could not read LOHALLOC_MODEL={path}: {e} — staying in training"
                    );
                }
            }
        });
    });
}

/// atexit hook installed by `ensure_model_loaded` under `LOHALLOC_DEBUG`.
extern "C" fn report_pht_misses_at_exit() {
    use lohalloc_core::Backend;
    eprintln!(
        "lohalloc: pht_misses={} (model-loaded run; ~0 means the model matched this process)",
        Lohalloc::pht_miss_count()
    );
    eprintln!(
        "lohalloc: routes slab={} buddy={} system={} arena={} fallthrough={}",
        Lohalloc::route_count(Backend::Slab),
        Lohalloc::route_count(Backend::Buddy),
        Lohalloc::route_count(Backend::System),
        Lohalloc::route_count(Backend::Arena),
        Lohalloc::fallthrough_count(),
    );
    eprintln!("lohalloc: slab_headerless={}", ALLOC.is_slab_headerless());
}

/// Runs `f` (an `ALLOC.alloc(...)` call), then — only for a true top-level,
/// non-nested call on this thread — checks whether to auto-freeze.
///
/// `maybe_auto_freeze`'s first call ever reads `LOHALLOC_FREEZE_AFTER` via
/// `std::env::var`, which is guarded by std's own internal (non-reentrant)
/// `Once`. If `ALLOC.alloc(layout)` itself needs to allocate for its own
/// bookkeeping (e.g. `BanditPolicy::select`'s `BTreeMap` inserting a new
/// node the first time a Signature is seen), that nested allocation also
/// flows through this same exported `malloc` — self-referential symbol
/// binding inside this cdylib means std's *own* internal Vec/BTreeMap
/// machinery calls our `malloc`, not a separate "real" allocator. Without
/// this guard, that nested call would try to initialize the *same*
/// `Once` a second time on the *same* thread before the first
/// initialization finishes — `Once` is not reentrant, so it deadlocks.
/// Verified the hard way under real `LD_PRELOAD` on Linux (not reproduced
/// by calling exported symbols directly via `dlsym`, which doesn't trigger
/// std's own internal allocations the same way).
fn with_freeze_check(f: impl FnOnce() -> *mut c_void) -> *mut c_void {
    let depth = IN_ALLOC_FN.with(|c| c.get());
    IN_ALLOC_FN.with(|c| c.set(depth + 1));
    // Model load must happen before the first top-level allocation is
    // served, and — like `maybe_auto_freeze` below — only while the depth
    // bump is active, so its own nested allocations skip this layer.
    if depth == 0 {
        ensure_model_loaded();
    }
    let ptr = f();
    // Guard must still be held (depth+1) through `maybe_auto_freeze()`
    // itself, not just around `f()` — its first-ever call reads
    // LOHALLOC_FREEZE_AFTER via `std::env::var`, which can trigger its own
    // nested allocation (std's internal env-access lock initializing).
    // Resetting the depth before this call would make that nested
    // allocation look like a fresh top-level call, re-entering the exact
    // same non-reentrant `Once` this guard exists to protect.
    if !ptr.is_null() && depth == 0 {
        maybe_auto_freeze();
    }
    IN_ALLOC_FN.with(|c| c.set(depth));
    ptr
}

// ---------------------------------------------------------------------------
// The malloc family
// ---------------------------------------------------------------------------

/// # Safety
/// Same contract as C's `malloc`.
#[no_mangle]
pub unsafe extern "C" fn malloc(size: usize) -> *mut c_void {
    let Ok(layout) = Layout::from_size_align(size.max(1), MIN_ALIGN) else {
        return core::ptr::null_mut();
    };
    with_freeze_check(|| unsafe { ALLOC.alloc(layout) as *mut c_void })
}

/// # Safety
/// Same contract as C's `free`. A no-op for a null pointer. Delegates to
/// the real libc `free` for a pointer this allocator didn't allocate (see
/// the module doc's "Foreign pointers" section).
#[no_mangle]
pub unsafe extern "C" fn free(ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }
    let p = ptr as *mut u8;
    // Fused ownership-check + free: one 48-byte header read instead of the
    // two the separate `owns()` → `dealloc_raw()` sequence paid.
    if !unsafe { ALLOC.try_dealloc_raw(p) } {
        if let Some(real_free) = libc_free() {
            unsafe { real_free(ptr) };
        }
        // No real `free` resolvable: leak rather than risk corrupting an
        // allocator we don't understand (should not happen under a normal
        // dynamic link).
    }
}

/// # Safety
/// Same contract as C's `calloc`.
#[no_mangle]
pub unsafe extern "C" fn calloc(nmemb: usize, size: usize) -> *mut c_void {
    let Some(total) = nmemb.checked_mul(size) else {
        return core::ptr::null_mut();
    };
    let ptr = unsafe { malloc(total) };
    if !ptr.is_null() && total > 0 {
        // Recycled slab/buddy/arena/system memory is dirty — calloc must
        // always zero, unlike malloc.
        unsafe { core::ptr::write_bytes(ptr as *mut u8, 0, total) };
    }
    ptr
}

/// # Safety
/// Same contract as C's `realloc`. `ptr == NULL` behaves like `malloc`;
/// `new_size == 0` behaves like `free` (returning `NULL`) — both glibc
/// conventions. For a foreign `ptr`, delegates to the real libc `realloc`.
#[no_mangle]
pub unsafe extern "C" fn realloc(ptr: *mut c_void, new_size: usize) -> *mut c_void {
    if ptr.is_null() {
        return unsafe { malloc(new_size) };
    }
    if new_size == 0 {
        unsafe { free(ptr) };
        return core::ptr::null_mut();
    }

    let p = ptr as *mut u8;
    // One header read serves the ownership check, the usable-size query,
    // AND (via the token) the final free — this path used to read the same
    // header three times.
    let Some((old_usable, token)) = (unsafe { ALLOC.usable_size_for_realloc(p) }) else {
        return match libc_realloc() {
            Some(real_realloc) => unsafe { real_realloc(ptr, new_size) },
            None => core::ptr::null_mut(),
        };
    };

    let new_ptr = unsafe { malloc(new_size) };
    if new_ptr.is_null() {
        return core::ptr::null_mut();
    }
    let copy_len = old_usable.min(new_size);
    unsafe {
        core::ptr::copy_nonoverlapping(p, new_ptr as *mut u8, copy_len);
        ALLOC.dealloc_with_header_token(p, token);
    }
    new_ptr
}

/// # Safety
/// Same contract as C's `reallocarray` (glibc extension): `realloc(ptr,
/// nmemb * size)`, but fails (returning `NULL`, `errno = ENOMEM`) on
/// overflow instead of silently wrapping.
#[no_mangle]
pub unsafe extern "C" fn reallocarray(ptr: *mut c_void, nmemb: usize, size: usize) -> *mut c_void {
    match nmemb.checked_mul(size) {
        Some(total) => unsafe { realloc(ptr, total) },
        None => {
            set_errno(libc::ENOMEM);
            core::ptr::null_mut()
        }
    }
}

/// # Safety
/// Same contract as POSIX `posix_memalign`.
#[no_mangle]
pub unsafe extern "C" fn posix_memalign(
    memptr: *mut *mut c_void,
    alignment: usize,
    size: usize,
) -> c_int {
    if memptr.is_null() {
        return libc::EINVAL;
    }
    let ptr_size = core::mem::size_of::<*mut c_void>();
    if !alignment.is_power_of_two() || alignment < ptr_size {
        return libc::EINVAL;
    }
    let Ok(layout) = Layout::from_size_align(size.max(1), alignment) else {
        return libc::EINVAL;
    };
    let ptr = with_freeze_check(|| unsafe { ALLOC.alloc(layout) as *mut c_void });
    if ptr.is_null() {
        return libc::ENOMEM;
    }
    unsafe { *memptr = ptr };
    0
}

/// # Safety
/// Same contract as C11 `aligned_alloc`. Returns `NULL` if `alignment`
/// isn't a power of two (this implementation does not require `size` to be
/// a multiple of `alignment`, matching glibc's current behavior rather than
/// the stricter historical C11 wording).
#[no_mangle]
pub unsafe extern "C" fn aligned_alloc(alignment: usize, size: usize) -> *mut c_void {
    if !alignment.is_power_of_two() {
        return core::ptr::null_mut();
    }
    let Ok(layout) = Layout::from_size_align(size.max(1), alignment.max(MIN_ALIGN)) else {
        return core::ptr::null_mut();
    };
    with_freeze_check(|| unsafe { ALLOC.alloc(layout) as *mut c_void })
}

/// # Safety
/// Same contract as the legacy (obsolete but still linked by some
/// benchmarks) `memalign`.
#[no_mangle]
pub unsafe extern "C" fn memalign(alignment: usize, size: usize) -> *mut c_void {
    unsafe { aligned_alloc(alignment, size) }
}

/// # Safety
/// Same contract as the legacy `valloc`: page-aligned allocation of `size`
/// bytes. Page size is queried at runtime (`lohalloc_alloc::system::page_size`)
/// — never hard-coded, since it varies across targets (e.g. 16 KiB on Apple
/// Silicon, 4 KiB on most x86/Linux-aarch64).
#[no_mangle]
pub unsafe extern "C" fn valloc(size: usize) -> *mut c_void {
    unsafe { aligned_alloc(lohalloc_alloc::system::page_size(), size) }
}

/// # Safety
/// Same contract as glibc's `malloc_usable_size` extension. Returns 0 for a
/// null or foreign pointer.
#[no_mangle]
pub unsafe extern "C" fn malloc_usable_size(ptr: *mut c_void) -> usize {
    if ptr.is_null() {
        return 0;
    }
    let p = ptr as *mut u8;
    // Fused ownership-check + size query: one header read, not two.
    match unsafe { ALLOC.try_usable_size(p) } {
        Some(usable) => usable,
        None => libc_malloc_usable_size()
            .map(|real| unsafe { real(ptr) })
            .unwrap_or(0),
    }
}

/// Sets `errno`. The libc symbol differs by platform: glibc/Linux exposes
/// `__errno_location`, macOS/BSD expose `__error`. This crate targets Linux
/// (`LD_PRELOAD`) but stays buildable on macOS for local dev iteration.
#[inline]
fn set_errno(value: c_int) {
    #[cfg(target_os = "linux")]
    unsafe {
        *libc::__errno_location() = value;
    }
    #[cfg(target_os = "macos")]
    unsafe {
        *libc::__error() = value;
    }
}

// ---------------------------------------------------------------------------
// Control surface: let a driving harness (or the process itself) freeze
// explicitly instead of relying on LOHALLOC_FREEZE_AFTER.
// ---------------------------------------------------------------------------

/// Freeze the process-wide allocator: collapse Training-mode learning into
/// the frozen Inference routing table (and export it if
/// `LOHALLOC_EXPORT_MODEL` is set).
#[no_mangle]
pub extern "C" fn lohalloc_freeze() {
    FREEZE_FIRED.store(true, Ordering::Relaxed);
    // Bump the same depth guard `with_freeze_check` uses: the export path
    // reads env vars and does file I/O, whose nested allocations must not
    // look like fresh top-level mallocs (they'd re-enter the non-reentrant
    // init `OnceLock`s on this same thread).
    let depth = IN_ALLOC_FN.with(|c| c.get());
    IN_ALLOC_FN.with(|c| c.set(depth + 1));
    ALLOC.freeze();
    maybe_export_model();
    IN_ALLOC_FN.with(|c| c.set(depth));
}

/// True (`1`) if the process-wide allocator is currently in Inference mode.
#[no_mangle]
pub extern "C" fn lohalloc_is_inference() -> c_int {
    ALLOC.is_inference() as c_int
}

/// Process-wide count of Inference-mode routing lookups that missed the
/// frozen table (fell back to size-based routing). ~0 on a model-loaded run
/// proves the model's keys matched this process's call sites — the
/// end-to-end check that hashes are stable across runs (ASLR-normalized).
#[no_mangle]
pub extern "C" fn lohalloc_pht_misses() -> u64 {
    Lohalloc::pht_miss_count()
}

/// True (`1`) if the process-wide allocator is currently serving Slab
/// allocations header-free (only possible after `LOHALLOC_MODEL`/
/// `lohalloc_load_model` on a still-empty Slab — see `Lohalloc::load()`'s
/// doc). Debug/introspection only.
#[no_mangle]
pub extern "C" fn lohalloc_slab_headerless() -> c_int {
    ALLOC.is_slab_headerless() as c_int
}

/// Load a `.lohalloc` model file, starting the process-wide allocator
/// directly in Inference mode. Returns `1` on success, `0` on failure
/// (missing file, malformed model, or the model already being loaded).
///
/// # Safety
/// `path` must be a valid, nul-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn lohalloc_load_model(path: *const c_char) -> c_int {
    if path.is_null() {
        return 0;
    }
    let c_str = unsafe { std::ffi::CStr::from_ptr(path) };
    let Ok(path_str) = c_str.to_str() else {
        return 0;
    };
    let Ok(bytes) = std::fs::read(path_str) else {
        return 0;
    };
    ALLOC.load(&bytes) as c_int
}
