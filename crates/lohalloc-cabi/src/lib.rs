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
// LOHALLOC_FREEZE_AFTER: let an unmodified C/C++ program exercise Inference
// mode by auto-freezing after N successful mallocs.
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
    })
}

/// Called after every successful "real" (non-delegated) allocation. Freezes
/// exactly once when the configured threshold is crossed, then becomes a
/// single relaxed load for the rest of the process lifetime.
#[inline]
fn maybe_auto_freeze() {
    if FREEZE_FIRED.load(Ordering::Relaxed) {
        return;
    }
    let Some(threshold) = freeze_after_threshold() else {
        return;
    };
    let count = MALLOC_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    // `>=` + `swap` (not `== threshold`): several threads can cross the
    // threshold simultaneously; exactly one wins the swap and freezes.
    if count >= threshold && !FREEZE_FIRED.swap(true, Ordering::Relaxed) {
        ALLOC.freeze();
    }
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
    if unsafe { ALLOC.owns(p) } {
        unsafe { ALLOC.dealloc_raw(p) };
    } else if let Some(real_free) = libc_free() {
        unsafe { real_free(ptr) };
    }
    // No real `free` resolvable: leak rather than risk corrupting an
    // allocator we don't understand (should not happen under a normal
    // dynamic link).
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
    if !unsafe { ALLOC.owns(p) } {
        return match libc_realloc() {
            Some(real_realloc) => unsafe { real_realloc(ptr, new_size) },
            None => core::ptr::null_mut(),
        };
    }

    let old_usable = unsafe { ALLOC.usable_size(p) };
    let new_ptr = unsafe { malloc(new_size) };
    if new_ptr.is_null() {
        return core::ptr::null_mut();
    }
    let copy_len = old_usable.min(new_size);
    unsafe {
        core::ptr::copy_nonoverlapping(p, new_ptr as *mut u8, copy_len);
        ALLOC.dealloc_raw(p);
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
    if unsafe { ALLOC.owns(p) } {
        unsafe { ALLOC.usable_size(p) }
    } else {
        libc_malloc_usable_size()
            .map(|real| unsafe { real(ptr) })
            .unwrap_or(0)
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
/// the frozen Inference routing table.
#[no_mangle]
pub extern "C" fn lohalloc_freeze() {
    FREEZE_FIRED.store(true, Ordering::Relaxed);
    ALLOC.freeze();
}

/// True (`1`) if the process-wide allocator is currently in Inference mode.
#[no_mangle]
pub extern "C" fn lohalloc_is_inference() -> c_int {
    ALLOC.is_inference() as c_int
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
