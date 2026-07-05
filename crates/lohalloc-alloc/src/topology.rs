//! Topology Engine — inline-asm stack walker + bitwise hash.
//!
//! Replaces the Phase 1 `caller_pc` capture with a 3-frame topological hash
//! of the call stack. The hash identifies allocation call sites by their
//! position in the call tree rather than a single PC, enabling the Decision
//! Engine (Phase 3) to route by logical topology.
//!
//! # Two-Layered Frame-Pointer Defense
//!
//! 1. **Build-system enforcement** (`.cargo/config.toml`):
//!    `rustflags = ["-C", "force-frame-pointers=yes"]` guarantees the compiler
//!    never repurposes `rbp` (x86_64) or `x29` (ARM64) as a general-purpose
//!    register.
//!
//! 2. **Hot-path heuristic guard** (this module): Before dereferencing any
//!    frame pointer, validate it mathematically (no memory access):
//!    - Alignment: `fp & 0xF == 0`
//!    - Direction: `fp > sp`
//!    - Proximity: `fp - sp <= 8_000_000`
//!
//! A failed guard returns a sentinel hash (`0`) that routes to the
//! System/Buddy fallback — never a segfault. This catches pre-compiled
//! `.so` files built without frame pointers.
//!
//! # ASLR-stable hashing (module-base PC normalization)
//!
//! Raw return addresses are absolute virtual addresses, which shift every
//! run under ASLR/PIE — a hash folded from them identifies a call site only
//! within one process lifetime, making exported `.lohalloc` models useless
//! in any later process. Before mixing, each collected PC is therefore
//! normalized to `module_ident ^ (pc - module_base)`: a lazily-built,
//! fixed-capacity table of loaded-module ranges (Linux: `dl_iterate_phdr`;
//! macOS: dyld image APIs) is binary-searched per PC, and the module's
//! identity is an FNV-1a hash of its basename — stable across runs and
//! independent of load order. PCs outside every known module (JIT pages,
//! vdso, or beyond the 64-module cap) fall back to the raw address:
//! correctness is unaffected, those sites just degrade to per-run hashes.
//! Models remain per-binary/per-architecture — offsets are only meaningful
//! against the same module contents.
//!
//! # Hot-path cost & allocation discipline
//!
//! The stack walk + hash itself uses only `core` features (inline asm,
//! integer math) and makes no heap allocations. A per-thread direct-mapped
//! memo (`HASH_MEMO`, 64 entries, C5) keyed on the raw walked PC triple
//! serves repeat call sites without touching the module table or the mix —
//! those only run on a memo miss. The table itself adds one `OnceLock`
//! acquire-load plus, per PC, a ~6-level binary search over ≤64 entries. Table construction runs at most once, is allocation-free
//! (`dl_iterate_phdr` fills a fixed array via callback; the dyld APIs are
//! straight reads of mapped memory), and any incidental allocation would
//! anyway hit the caller's `IN_ALLOC` bypass since `fast_stack_hash` runs
//! inside `alloc`'s re-entrancy guard. Caveat (documented, accepted):
//! `dl_iterate_phdr` takes the loader lock, so a first-ever malloc racing a
//! concurrent `dlopen` on another thread could in theory contend; the
//! escape hatch would be eager init from an `.init_array` constructor.

use std::sync::OnceLock;

/// Sentinel hash returned when the frame pointer fails the heuristic guard
/// or when no valid instruction pointers could be collected. Routes to the
/// System/Buddy fallback.
pub const SENTINEL_HASH: u64 = 0;

// ---------------------------------------------------------------------------
// Module table: loaded-image ranges for ASLR-stable PC normalization.
// ---------------------------------------------------------------------------

/// Fixed capacity of the module table — allocation-free by construction.
/// Typical benchmark/server processes load well under 64 images; anything
/// past the cap silently degrades to raw (per-run) PCs.
const MAX_MODULES: usize = 64;

/// One loaded module's address range plus a load-order-independent identity.
#[derive(Clone, Copy)]
struct ModuleRange {
    /// Lowest mapped address of the module's PT_LOAD/segment span.
    base: usize,
    /// One past the highest mapped address.
    end: usize,
    /// FNV-1a hash of the module path's basename (`""` for the main
    /// executable on Linux — a fixed, documented identity). Stable across
    /// runs and across load-order changes, unlike the base address.
    ident: u64,
}

const EMPTY_RANGE: ModuleRange = ModuleRange {
    base: 0,
    end: 0,
    ident: 0,
};

/// Fixed-capacity, sorted-by-base table of loaded modules (~1.5 KiB).
struct ModuleTable {
    entries: [ModuleRange; MAX_MODULES],
    len: usize,
}

impl ModuleTable {
    fn entries(&self) -> &[ModuleRange] {
        &self.entries[..self.len]
    }

    /// In-place insertion sort by base — allocation-free, runs once at
    /// table build over ≤64 entries.
    fn sort(&mut self) {
        for i in 1..self.len {
            let mut j = i;
            while j > 0 && self.entries[j - 1].base > self.entries[j].base {
                self.entries.swap(j - 1, j);
                j -= 1;
            }
        }
    }
}

static MODULE_TABLE: OnceLock<ModuleTable> = OnceLock::new();

/// The process's module table, built lazily on first use. Construction is
/// allocation-free; see the module doc for the loader-lock caveat.
fn module_table() -> &'static ModuleTable {
    MODULE_TABLE.get_or_init(build_module_table)
}

/// FNV-1a over the basename of a nul-terminated C path string. The empty
/// string hashes to the FNV offset basis — the fixed identity used for the
/// main executable on Linux (its `dlpi_name` is `""`).
///
/// # Safety
/// `path` must be null or point to a nul-terminated C string.
unsafe fn fnv1a_basename(path: *const core::ffi::c_char) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    if path.is_null() {
        return FNV_OFFSET;
    }
    // Find the byte after the last '/', then hash to the nul.
    let mut start = path;
    let mut p = path;
    unsafe {
        while *p != 0 {
            if *p as u8 == b'/' {
                start = p.add(1);
            }
            p = p.add(1);
        }
        let mut hash = FNV_OFFSET;
        let mut q = start;
        while *q != 0 {
            hash ^= *q as u8 as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
            q = q.add(1);
        }
        hash
    }
}

#[cfg(target_os = "linux")]
fn build_module_table() -> ModuleTable {
    unsafe extern "C" fn callback(
        info: *mut libc::dl_phdr_info,
        _size: libc::size_t,
        data: *mut core::ffi::c_void,
    ) -> core::ffi::c_int {
        // SAFETY: `data` is the &mut ModuleTable we passed below; `info` is
        // valid for the duration of the callback per dl_iterate_phdr's
        // contract.
        let table = unsafe { &mut *(data as *mut ModuleTable) };
        if table.len >= MAX_MODULES {
            return 0; // cap reached: remaining modules degrade to raw PCs
        }
        let info = unsafe { &*info };
        let phdrs =
            unsafe { core::slice::from_raw_parts(info.dlpi_phdr, info.dlpi_phnum as usize) };
        let mut lo = usize::MAX;
        let mut hi = 0usize;
        for ph in phdrs {
            if ph.p_type == libc::PT_LOAD {
                let start = info.dlpi_addr as usize + ph.p_vaddr as usize;
                let end = start + ph.p_memsz as usize;
                lo = lo.min(start);
                hi = hi.max(end);
            }
        }
        if lo < hi {
            let ident = unsafe { fnv1a_basename(info.dlpi_name) };
            table.entries[table.len] = ModuleRange {
                base: lo,
                end: hi,
                ident,
            };
            table.len += 1;
        }
        0
    }

    let mut table = ModuleTable {
        entries: [EMPTY_RANGE; MAX_MODULES],
        len: 0,
    };
    unsafe {
        libc::dl_iterate_phdr(
            Some(callback),
            &mut table as *mut ModuleTable as *mut core::ffi::c_void,
        );
    }
    table.sort();
    table
}

#[cfg(target_os = "macos")]
fn build_module_table() -> ModuleTable {
    use core::ffi::{c_char, c_uint};

    // Raw dyld APIs — all straight reads of already-mapped memory, no
    // allocation. Declared here rather than via the libc crate to avoid
    // depending on which of them libc happens to expose.
    extern "C" {
        fn _dyld_image_count() -> c_uint;
        fn _dyld_get_image_header(image_index: c_uint) -> *const MachHeader64;
        fn _dyld_get_image_vmaddr_slide(image_index: c_uint) -> isize;
        fn _dyld_get_image_name(image_index: c_uint) -> *const c_char;
    }

    #[repr(C)]
    struct MachHeader64 {
        magic: u32,
        cputype: i32,
        cpusubtype: i32,
        filetype: u32,
        ncmds: u32,
        sizeofcmds: u32,
        flags: u32,
        reserved: u32,
    }

    #[repr(C)]
    struct SegmentCommand64 {
        cmd: u32,
        cmdsize: u32,
        segname: [u8; 16],
        vmaddr: u64,
        vmsize: u64,
        fileoff: u64,
        filesize: u64,
        maxprot: i32,
        initprot: i32,
        nsects: u32,
        flags: u32,
    }

    const LC_SEGMENT_64: u32 = 0x19;

    let mut table = ModuleTable {
        entries: [EMPTY_RANGE; MAX_MODULES],
        len: 0,
    };

    unsafe {
        let count = _dyld_image_count();
        for i in 0..count {
            if table.len >= MAX_MODULES {
                break;
            }
            let header = _dyld_get_image_header(i);
            if header.is_null() {
                continue;
            }
            let slide = _dyld_get_image_vmaddr_slide(i) as usize;
            // Walk the load commands directly after the 64-bit mach header.
            let mut cmd_ptr = (header as *const u8).add(core::mem::size_of::<MachHeader64>());
            let mut lo = usize::MAX;
            let mut hi = 0usize;
            for _ in 0..(*header).ncmds {
                let cmd = &*(cmd_ptr as *const SegmentCommand64);
                if cmd.cmd == LC_SEGMENT_64 && cmd.initprot != 0 && cmd.vmsize > 0 {
                    // initprot == 0 filters out __PAGEZERO, whose 4 GiB
                    // span would swallow every low address.
                    let start = cmd.vmaddr as usize + slide;
                    let end = start + cmd.vmsize as usize;
                    lo = lo.min(start);
                    hi = hi.max(end);
                }
                cmd_ptr = cmd_ptr.add(cmd.cmdsize as usize);
            }
            if lo < hi {
                let ident = fnv1a_basename(_dyld_get_image_name(i));
                table.entries[table.len] = ModuleRange {
                    base: lo,
                    end: hi,
                    ident,
                };
                table.len += 1;
            }
        }
    }
    table.sort();
    table
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn build_module_table() -> ModuleTable {
    // Unknown platform: empty table → every PC passes through raw
    // (per-run hashes, exactly the pre-normalization behavior).
    ModuleTable {
        entries: [EMPTY_RANGE; MAX_MODULES],
        len: 0,
    }
}

std::thread_local! {
    /// Last-hit module cache: `(base, end, ident)` of the module the
    /// previous normalized PC fell in. Nearly all walked PCs live in one
    /// hot module (the workload binary), so this turns ~3 binary searches
    /// per allocation into ~3 range compares — measured ~15-30ns off every
    /// alloc. A plain dtor-free `Cell` (per the crate's TLS invariant: no
    /// destructors on the alloc path); lossless, since a miss just falls
    /// through to the authoritative binary search.
    static LAST_MODULE: core::cell::Cell<(usize, usize, u64)> =
        const { core::cell::Cell::new((0, 0, 0)) };
}

/// Test-only: clear this thread's last-hit module cache, so unit tests that
/// probe `normalize_pc_with` with *different fake tables* don't see stale
/// entries cached by an earlier test on the same test thread. (In production
/// the cache is always consistent — every call uses the single immutable
/// `MODULE_TABLE`.)
#[cfg(test)]
fn reset_module_cache() {
    LAST_MODULE.set((0, 0, 0));
}

/// Normalize one PC against a sorted module table:
/// `module_ident ^ (pc - module_base)` when the PC falls inside a known
/// module, raw passthrough otherwise. Same (module, offset) → same value in
/// every run regardless of where ASLR placed the module.
#[inline]
fn normalize_pc_with(pc: u64, entries: &[ModuleRange]) -> u64 {
    let addr = pc as usize;

    // Fast path: same module as the previous PC (the overwhelmingly common
    // case — all of a call stack's frames usually live in one binary).
    let (c_base, c_end, c_ident) = LAST_MODULE.get();
    if addr >= c_base && addr < c_end {
        return c_ident ^ (addr - c_base) as u64;
    }

    // Slow path: binary search the sorted table; refresh the cache on hit.
    let idx = entries.partition_point(|m| m.base <= addr);
    if idx > 0 {
        let m = &entries[idx - 1];
        if addr < m.end {
            LAST_MODULE.set((m.base, m.end, m.ident));
            return m.ident ^ (addr - m.base) as u64;
        }
    }
    pc
}

/// Maximum number of frames to walk.
const NUM_FRAMES: usize = 3;

// ---------------------------------------------------------------------------
// C5: TLS stack-hash memo — skip the frame-1/2 walk + normalize + mix when
// this exact call site was hashed recently on this thread.
// ---------------------------------------------------------------------------

/// Entries in the per-thread direct-mapped hash memo. 64 × 24 bytes =
/// 1.5 KiB per thread.
const MEMO_ENTRIES: usize = 64;

/// One memo slot: raw walked PC triple `(ret0, ret1, ret2)` → final hash.
///
/// The key is the *complete* raw input of the normalize+mix pipeline, so a
/// hit is sound by construction: the module table is immutable after first
/// build and `normalize_pc_with`/`mix_hash` are pure, so the same raw
/// triple always maps to the same hash within a process. (The originally
/// planned `(fp, ret0)` key aliased in practice — the debug cross-check
/// caught `lohalloc-demo`'s churn workload producing two different
/// 3-frame topologies behind one `(fp, ret0)` pair within milliseconds:
/// a helper called from two sites at identical stack depth shares both the
/// leaf frame address and the leaf return address. Keying on the full
/// triple trades away skipping the frame walk — 3 guarded derefs, cheap —
/// and keeps skipping what actually costs: the module-table lookups and
/// the mix.)
///
/// Unwalked frames key as `0`. `ret0 == 0` marks an empty slot (a zero
/// first return address is rejected before the memo is consulted).
/// `SENTINEL_HASH` results are never stored, so a hit is always a real
/// hash.
struct MemoEntry {
    ret0: core::cell::Cell<usize>,
    ret1: core::cell::Cell<usize>,
    ret2: core::cell::Cell<usize>,
    hash: core::cell::Cell<u64>,
}

std::thread_local! {
    /// Direct-mapped, dtor-free (plain `Cell`s, per the crate's TLS
    /// invariant: no destructors on the alloc path), const-initialized so
    /// first touch never allocates.
    static HASH_MEMO: [MemoEntry; MEMO_ENTRIES] = const {
        [const {
            MemoEntry {
                ret0: core::cell::Cell::new(0),
                ret1: core::cell::Cell::new(0),
                ret2: core::cell::Cell::new(0),
                hash: core::cell::Cell::new(0),
            }
        }; MEMO_ENTRIES]
    };
}

#[cfg(debug_assertions)]
std::thread_local! {
    /// Debug-only sample counter for the 1/256 memo-vs-fresh-walk
    /// cross-check.
    static MEMO_SAMPLE: core::cell::Cell<u32> = const { core::cell::Cell::new(0) };
}

/// Slot index for a raw PC triple: fold + one multiply, top bits.
#[inline(always)]
fn memo_index(ret0: usize, ret1: usize, ret2: usize) -> usize {
    let folded = ret0 ^ ret1.rotate_left(21) ^ ret2.rotate_left(42);
    ((folded as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 58) as usize & (MEMO_ENTRIES - 1)
}

/// Maximum distance from the current stack pointer to a valid frame pointer
/// (8 MB — the typical thread stack size on Linux).
const MAX_FP_DISTANCE: usize = 8_000_000;

/// Capture a topological hash of the current call stack.
///
/// Walks exactly 3 frames up the call stack via inline assembly, collects the
/// return addresses (instruction pointers), and folds them into a `u64` using
/// a bitwise XOR-shift mixing function.
///
/// A per-thread direct-mapped memo keyed on the raw walked PC triple (C5)
/// short-circuits repeat call sites: the guarded frame walk still runs, but
/// PC normalization (module table) and the mix are skipped on a hit.
/// Hashes are bit-identical either way.
///
/// # Returns
///
/// - A non-zero `u64` hash for a valid call stack.
/// - `SENTINEL_HASH` (0) if the frame pointer fails the heuristic guard or
///   no valid IPs could be collected.
///
/// # Safety
///
/// This function is safe to call from any context. The inline assembly reads
/// register values and follows the frame pointer chain, but the heuristic
/// guard ensures we never dereference an invalid pointer.
#[inline(always)]
pub fn fast_stack_hash() -> u64 {
    let (sp, fp) = read_sp_fp();

    // Heuristic guard on the initial frame pointer.
    if !is_valid_fp(fp, sp) {
        return SENTINEL_HASH;
    }

    // Raw 3-frame walk: guarded frame-pointer derefs and integer math only —
    // no module-table work yet. Raw (per-run) PCs are collected first so
    // they can key the memo; normalization to the ASLR-stable form and the
    // mix only run on a memo miss.
    let mut raw: [usize; NUM_FRAMES] = [0; NUM_FRAMES];
    let mut current_fp = fp;
    let mut collected = 0usize;

    for slot in raw.iter_mut() {
        // Re-validate the frame pointer before each dereference.
        if !is_valid_fp(current_fp, sp) {
            break;
        }

        let (next_fp, return_addr) = match read_frame(current_fp) {
            Some(frame) => frame,
            None => break,
        };

        if return_addr == 0 {
            break;
        }

        *slot = return_addr;
        collected += 1;

        if next_fp == 0 || next_fp == current_fp {
            break;
        }
        current_fp = next_fp;
    }

    if collected == 0 {
        return SENTINEL_HASH;
    }

    // C5 memo probe: a hit skips the module-table normalization and the
    // mix. Sound by construction — see `MemoEntry`.
    let idx = memo_index(raw[0], raw[1], raw[2]);
    let cached = HASH_MEMO.with(|memo| {
        let e = &memo[idx];
        if e.ret0.get() == raw[0] && e.ret1.get() == raw[1] && e.ret2.get() == raw[2] {
            Some(e.hash.get())
        } else {
            None
        }
    });
    if let Some(hash) = cached {
        // 1/256 sampled cross-check: with the full-triple key this must
        // always pass (normalize+mix are pure); a failure means the memo
        // store/probe itself regressed.
        #[cfg(debug_assertions)]
        {
            let n = MEMO_SAMPLE.with(|c| {
                let n = c.get().wrapping_add(1);
                c.set(n);
                n
            });
            if n & 0xFF == 0 {
                debug_assert_eq!(
                    hash,
                    normalize_and_mix(&raw, collected),
                    "C5 memo corrupted: cached hash diverged from fresh normalize+mix"
                );
            }
        }
        return hash;
    }

    let hash = normalize_and_mix(&raw, collected);
    if hash != SENTINEL_HASH {
        HASH_MEMO.with(|memo| {
            let e = &memo[idx];
            e.ret0.set(raw[0]);
            e.ret1.set(raw[1]);
            e.ret2.set(raw[2]);
            e.hash.set(hash);
        });
    }
    hash
}

/// One-frame topological hash: the leaf return address only, normalized and
/// mixed exactly as [`fast_stack_hash`] mixes its frame-0 slot
/// (`mix_hash(&[normalize(ret0)])`). This is J2's distilled key: call sites
/// that route identically across *every* 3-frame context they appear in are
/// keyed on this cheaper 1-frame hash, so Inference can skip walking frames
/// 1–2 for them.
///
/// Returns `SENTINEL_HASH` (0) if the frame-pointer guard fails or no leaf
/// return address is readable — same degradation contract as
/// `fast_stack_hash` (a sentinel routes to the size fallback, never a
/// misroute). Not memoized: it is already just one guarded deref + one
/// normalize + one mix, cheaper than a memo probe.
#[inline(always)]
pub fn one_frame_hash() -> u64 {
    let (sp, fp) = read_sp_fp();
    if !is_valid_fp(fp, sp) {
        return SENTINEL_HASH;
    }
    let ret0 = match read_frame(fp) {
        Some((_next_fp, ret)) if ret != 0 => ret,
        _ => return SENTINEL_HASH,
    };
    // Same module table + normalization + mixer the 3-frame path uses, so a
    // 1-frame hash computed here at Inference is bit-identical to the one the
    // training path stored for this site (see `AllocatorState`/`freeze`).
    let modules = module_table().entries();
    mix_hash(&[normalize_pc_with(ret0 as u64, modules)])
}

/// ASLR-stable normalization + mix over an already-walked raw PC array.
/// Bit-identical to the pre-C5 single-pass implementation — exported
/// `.lohalloc` models key on these exact hash values, so any change here
/// silently invalidates saved models.
#[inline]
fn normalize_and_mix(raw: &[usize; NUM_FRAMES], collected: usize) -> u64 {
    // One OnceLock acquire-load per call after first init; see module doc.
    let modules = module_table().entries();

    let mut ips: [u64; NUM_FRAMES] = [0; NUM_FRAMES];
    // ASLR-stable: hash the (module identity, offset) form of each PC,
    // not its absolute (per-run) virtual address.
    for (ip, &pc) in ips.iter_mut().zip(&raw[..collected]) {
        *ip = normalize_pc_with(pc as u64, modules);
    }

    mix_hash(&ips[..collected])
}

/// Read the current stack pointer and frame pointer via inline assembly.
///
/// Returns `(sp, fp)`:
/// - **x86_64**: `sp = rsp`, `fp = rbp`
/// - **ARM64**: `sp = sp`, `fp = x29`
#[inline(always)]
fn read_sp_fp() -> (usize, usize) {
    let sp: usize;
    let fp: usize;

    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::asm!(
            "mov {sp}, rsp",
            "mov {fp}, rbp",
            sp = out(reg) sp,
            fp = out(reg) fp,
            options(nostack, nomem, preserves_flags),
        );
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!(
            "mov {sp}, sp",
            "mov {fp}, x29",
            sp = out(reg) sp,
            fp = out(reg) fp,
            options(nostack, nomem, preserves_flags),
        );
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = (sp, fp);
        return (0, 0);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        (sp, fp)
    }
}

/// Read a frame from the frame pointer: returns `(next_fp, return_addr)`.
///
/// - **x86_64**: `next_fp = [fp + 0]`, `return_addr = [fp + 8]`
/// - **ARM64**: `next_fp = [fp + 0]`, `return_addr = [fp + 8]`
///
/// Returns `None` if the read would be unsafe (null pointer).
///
/// # Safety
///
/// The caller must have validated `fp` via `is_valid_fp` before calling.
#[inline(always)]
fn read_frame(fp: usize) -> Option<(usize, usize)> {
    if fp == 0 {
        return None;
    }

    // SAFETY: The caller has validated fp via is_valid_fp (alignment,
    // direction, proximity). We trust the frame pointer chain here.
    // If the memory is unmapped, we'd get a segfault — but the heuristic
    // guard's proximity check (within 8 MB of SP) makes this extremely
    // unlikely for valid thread stacks.
    unsafe {
        let next_fp = core::ptr::read_unaligned(fp as *const usize);
        let return_addr = core::ptr::read_unaligned((fp as *const usize).add(1));
        Some((next_fp, return_addr))
    }
}

/// Heuristic guard: validate a frame pointer mathematically (no memory
/// access) before dereferencing it.
///
/// Returns `true` if the frame pointer passes all three checks:
/// 1. **Alignment**: `fp & 0xF == 0` (16-byte aligned)
/// 2. **Direction**: `fp > sp` (stack grows downward; FP must be above SP)
/// 3. **Proximity**: `fp - sp <= MAX_FP_DISTANCE` (within 8 MB of SP)
#[inline(always)]
fn is_valid_fp(fp: usize, sp: usize) -> bool {
    // Alignment: stack frames are 16-byte aligned.
    if fp & 0xF != 0 {
        return false;
    }
    // Direction: frame pointer must be at a higher address than the stack
    // pointer (stack grows downward).
    if fp <= sp {
        return false;
    }
    // Proximity: must be within 8 MB of the current SP.
    if fp - sp > MAX_FP_DISTANCE {
        return false;
    }
    true
}

/// Fold instruction pointers into a `u64` hash using XOR-shift mixing.
///
/// This is a deterministic, zero-allocation hash function. Identical 3-IP
/// tuples always produce identical hashes. The mixing uses the classic
/// XOR-shift pattern to distribute bits evenly.
#[inline]
fn mix_hash(ips: &[u64]) -> u64 {
    let mut hash: u64 = 0;

    for (i, &ip) in ips.iter().enumerate() {
        // Rotate by a prime-based offset per frame index to differentiate
        // frame positions.
        let rotated = ip.rotate_left(((i as u32) * 17) % 64);
        hash ^= rotated;
        // XOR-shift mix: propagate bits both ways.
        hash ^= hash << 13;
        hash ^= hash >> 7;
        hash ^= hash << 17;
    }

    // Ensure we never return the sentinel for a valid collected set.
    if hash == SENTINEL_HASH {
        hash = 1;
    }

    hash
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heuristic_guard_rejects_misaligned_fp() {
        let sp: usize = 0x1000;
        let fp: usize = 0x1001; // misaligned (not 16-byte aligned)
        assert!(!is_valid_fp(fp, sp));
    }

    #[test]
    fn heuristic_guard_rejects_fp_below_sp() {
        let sp: usize = 0x2000;
        let fp: usize = 0x1000; // below SP
        assert!(!is_valid_fp(fp, sp));
    }

    #[test]
    fn heuristic_guard_rejects_fp_too_far() {
        let sp: usize = 0x1000;
        let fp: usize = sp + MAX_FP_DISTANCE + 1; // > 8 MB
        assert!(!is_valid_fp(fp, sp));
    }

    #[test]
    fn heuristic_guard_accepts_valid_fp() {
        let sp: usize = 0x1000;
        let fp: usize = 0x1010; // aligned, above SP, close
        assert!(is_valid_fp(fp, sp));
    }

    #[test]
    fn hash_is_deterministic() {
        // Calling fast_stack_hash twice from the same call site should
        // produce identical hashes (same 3-frame topology).
        let h1 = fast_stack_hash();
        let h2 = fast_stack_hash();
        assert_eq!(h1, h2, "hash should be deterministic for same call site");
    }

    #[test]
    fn hash_is_nonzero_for_valid_stack() {
        // On a real call stack with frame pointers enforced, the hash should
        // be non-zero (not the sentinel).
        let h = fast_stack_hash();
        assert_ne!(h, SENTINEL_HASH, "hash should be non-zero for valid stack");
    }

    #[test]
    fn memo_hit_equals_full_walk() {
        // C5: the first call from a site takes the full-walk (miss) path and
        // populates the memo; later calls from the same site take the memo
        // hit path. Both must yield the identical hash — run enough
        // iterations that the debug-mode 1/256 sampled cross-check also
        // fires at least once (it panics on an aliased entry).
        let first = fast_stack_hash();
        for _ in 0..1024 {
            assert_eq!(
                fast_stack_hash(),
                first,
                "memo hit must be bit-identical to the full walk"
            );
        }
    }

    #[test]
    fn one_frame_hash_is_deterministic_and_nonzero() {
        // J2: the 1-frame distilled key must be stable for a fixed call site
        // (so a training-time key matches the Inference-time probe) and
        // non-sentinel on a real stack (so distilled entries are reachable).
        let h1 = one_frame_hash();
        let h2 = one_frame_hash();
        assert_eq!(h1, h2, "1-frame hash must be deterministic for a site");
        assert_ne!(h1, SENTINEL_HASH, "1-frame hash should be non-sentinel");
    }

    #[test]
    fn one_frame_differs_from_three_frame() {
        // The distilled key is a *different* value than the full 3-frame hash
        // for the same site (it folds only the leaf return address), so the
        // two tables never alias keys.
        assert_ne!(one_frame_hash(), fast_stack_hash());
    }

    #[test]
    fn memo_index_stays_in_range() {
        for &(r0, r1, r2) in &[
            (0usize, 0usize, 0usize),
            (0x7fff_dead_bee0, 0x1000_4242, 0),
            (usize::MAX, usize::MAX, usize::MAX),
            (0x10, 0x7fff_ffff_ffff, 0x4000_0000),
        ] {
            assert!(memo_index(r0, r1, r2) < MEMO_ENTRIES);
        }
    }

    #[test]
    fn mix_hash_is_deterministic() {
        let ips = [0x1000u64, 0x2000, 0x3000];
        let h1 = mix_hash(&ips);
        let h2 = mix_hash(&ips);
        assert_eq!(h1, h2);
    }

    #[test]
    fn mix_hash_different_ips_different_hash() {
        let ips_a = [0x1000u64, 0x2000, 0x3000];
        let ips_b = [0x1000u64, 0x2000, 0x4000]; // different 3rd frame
        let h_a = mix_hash(&ips_a);
        let h_b = mix_hash(&ips_b);
        assert_ne!(h_a, h_b, "different IPs should produce different hashes");
    }

    #[test]
    fn mix_hash_never_returns_sentinel() {
        let ips = [0u64; 3];
        let h = mix_hash(&ips);
        assert_ne!(h, SENTINEL_HASH, "mix_hash should avoid sentinel value");
    }

    #[test]
    fn sentinel_hash_is_zero() {
        assert_eq!(SENTINEL_HASH, 0);
    }

    // ---- PC normalization (ASLR stability) --------------------------------

    fn fake_table(bases: &[(usize, usize, u64)]) -> Vec<ModuleRange> {
        let mut v: Vec<ModuleRange> = bases
            .iter()
            .map(|&(base, end, ident)| ModuleRange { base, end, ident })
            .collect();
        v.sort_by_key(|m| m.base);
        v
    }

    #[test]
    fn normalize_same_offset_different_bases_identical() {
        reset_module_cache();
        // The ASLR property: the same module (same ident) loaded at two
        // different bases yields the same normalized value for the same
        // in-module offset.
        let ident = 0xABCD_EF01_2345_6789u64;
        let run_a = fake_table(&[(0x5555_0000_0000, 0x5555_0010_0000, ident)]);
        let run_b = fake_table(&[(0x7FFF_0000_0000, 0x7FFF_0010_0000, ident)]);
        let offset = 0x4_2000u64;
        let pc_a = 0x5555_0000_0000u64 + offset;
        let pc_b = 0x7FFF_0000_0000u64 + offset;
        assert_eq!(
            normalize_pc_with(pc_a, &run_a),
            normalize_pc_with(pc_b, &run_b),
            "same (module, offset) must normalize identically across bases"
        );
    }

    #[test]
    fn normalize_different_modules_differ() {
        reset_module_cache();
        // Same offset in two *different* modules must not collide (their
        // idents differ).
        let table = fake_table(&[
            (0x1000_0000, 0x2000_0000, 0x1111),
            (0x3000_0000, 0x4000_0000, 0x2222),
        ]);
        let a = normalize_pc_with(0x1000_4000, &table);
        let b = normalize_pc_with(0x3000_4000, &table);
        assert_ne!(a, b, "different modules at same offset must differ");
    }

    #[test]
    fn normalize_outside_all_modules_passes_through() {
        reset_module_cache();
        let table = fake_table(&[(0x1000_0000, 0x2000_0000, 0x1111)]);
        // Below the first module, in a gap, and with an empty table.
        assert_eq!(normalize_pc_with(0x0FFF_FFFF, &table), 0x0FFF_FFFF);
        assert_eq!(normalize_pc_with(0x2000_0000, &table), 0x2000_0000);
        assert_eq!(normalize_pc_with(0xDEAD_BEEF, &[]), 0xDEAD_BEEF);
    }

    #[test]
    fn normalize_picks_containing_module_among_many() {
        reset_module_cache();
        let table = fake_table(&[
            (0x1000, 0x2000, 0xA),
            (0x2000, 0x3000, 0xB),
            (0x5000, 0x6000, 0xC),
        ]);
        assert_eq!(normalize_pc_with(0x2800, &table), 0xB ^ 0x800);
        assert_eq!(normalize_pc_with(0x5000, &table), 0xC); // offset 0
        assert_eq!(normalize_pc_with(0x4000, &table), 0x4000); // gap
    }

    #[test]
    fn normalize_cache_hit_equals_miss() {
        reset_module_cache();
        let table = fake_table(&[
            (0x1000_0000, 0x2000_0000, 0x1111),
            (0x3000_0000, 0x4000_0000, 0x2222),
        ]);
        // First lookup = cold (binary search, populates cache); second
        // lookup of the same PC = last-hit fast path. Must be identical.
        let cold = normalize_pc_with(0x1000_4000, &table);
        let hot = normalize_pc_with(0x1000_4000, &table);
        assert_eq!(cold, hot, "cache hit must equal binary-search result");
        // Different PC in the SAME cached module: fast path again — must
        // equal a from-scratch (cache-reset) computation.
        let hot2 = normalize_pc_with(0x1000_8000, &table);
        reset_module_cache();
        let cold2 = normalize_pc_with(0x1000_8000, &table);
        assert_eq!(hot2, cold2);
        // Switching modules evicts and still returns the right value.
        let other = normalize_pc_with(0x3000_4000, &table);
        assert_eq!(other, 0x2222 ^ 0x4000);
    }

    #[test]
    fn real_module_table_covers_our_own_code() {
        // The running test binary's code must be inside some module range,
        // so PCs from our own frames get normalized (not passed through).
        let table = module_table();
        assert!(
            table.len > 0,
            "module table should see at least the test binary"
        );
        let our_pc = real_module_table_covers_our_own_code as fn() as usize as u64;
        let normalized = normalize_pc_with(our_pc, table.entries());
        assert_ne!(
            normalized, our_pc,
            "a PC inside our own binary must be module-normalized"
        );
    }

    #[test]
    fn fnv_basename_ignores_directories() {
        // Nul-terminated byte strings (not c"" literals — those would bump
        // the workspace MSRV past its declared 1.74).
        let a = b"/usr/lib/libfoo.so\0".as_ptr() as *const core::ffi::c_char;
        let b = b"/opt/other/path/libfoo.so\0".as_ptr() as *const core::ffi::c_char;
        let bare = b"libfoo.so\0".as_ptr() as *const core::ffi::c_char;
        let other = b"libbar.so\0".as_ptr() as *const core::ffi::c_char;
        unsafe {
            assert_eq!(fnv1a_basename(a), fnv1a_basename(b));
            assert_eq!(fnv1a_basename(a), fnv1a_basename(bare));
            assert_ne!(fnv1a_basename(a), fnv1a_basename(other));
        }
    }
}
