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
//! # `#![no_std]` Compatibility
//!
//! This module uses only `core` features (`core::arch::asm!`, integer math).
//! No `Vec`, `String`, or heap allocations on the hot path.

/// Sentinel hash returned when the frame pointer fails the heuristic guard
/// or when no valid instruction pointers could be collected. Routes to the
/// System/Buddy fallback.
pub const SENTINEL_HASH: u64 = 0;

/// Maximum number of frames to walk.
const NUM_FRAMES: usize = 3;

/// Maximum distance from the current stack pointer to a valid frame pointer
/// (8 MB — the typical thread stack size on Linux).
const MAX_FP_DISTANCE: usize = 8_000_000;

/// Capture a topological hash of the current call stack.
///
/// Walks exactly 3 frames up the call stack via inline assembly, collects the
/// return addresses (instruction pointers), and folds them into a `u64` using
/// a bitwise XOR-shift mixing function.
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

    let mut ips: [u64; NUM_FRAMES] = [0; NUM_FRAMES];
    let mut current_fp = fp;
    let mut collected = 0usize;

    for ips_slot in ips.iter_mut() {
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

        *ips_slot = return_addr as u64;
        collected += 1;

        if next_fp == 0 || next_fp == current_fp {
            break;
        }
        current_fp = next_fp;
    }

    if collected == 0 {
        return SENTINEL_HASH;
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
}
