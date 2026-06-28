//! Shared types and helpers for Lohalloc.
//!
//! This crate is deliberately platform-agnostic: it holds the size-class
//! table, alignment math, the request "Signature" tuple used by the Decision
//! Engine, and the trace-record struct used by Phase 3 replay. Allocation
//! backends and platform-specific `mmap` glue live in `lohalloc-alloc`.

#![forbid(unsafe_code)]

/// A request signature: the key the Decision Engine maps to a backend.
///
/// Defined here as the tuple `(caller_pc, size_class)`. `caller_pc` is the
/// program-counter of the allocation call site (captured by the Observer in
/// Phase 2); `size_class` is a compact index into the size-class table. The
/// bandit weights one of these per call site.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Signature {
    pub caller_pc: u64,
    pub size_class: u8,
}

impl Signature {
    pub const fn new(caller_pc: u64, size_class: u8) -> Self {
        Self {
            caller_pc,
            size_class,
        }
    }
}

/// Which Execution-Plane worker handled a request. Stored in trace records
/// and surfaced in the GUI's Policy Matrix.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum Backend {
    Slab = 0,
    Buddy = 1,
    System = 2,
    Arena = 3,
}

impl Backend {
    pub const ALL: [Backend; 4] = [
        Backend::Slab,
        Backend::Buddy,
        Backend::System,
        Backend::Arena,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Backend::Slab => "slab",
            Backend::Buddy => "buddy",
            Backend::System => "system",
            Backend::Arena => "arena",
        }
    }
}

/// The canonical size-class table used by the Slab allocator.
///
/// Each entry is a fixed block size (bytes). A request of `n` bytes is served
/// by the smallest class `>= n`. Keep these powers of two for buddy-friendliness
/// and simple alignment; the table is small and indexed by `u8`.
pub const SLAB_SIZE_CLASSES: [usize; 12] =
    [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384];

/// Largest allocation served by the Slab allocator. Above this we fall back to
/// the Buddy allocator; above `BUDDY_MAX` we go straight to the System backend.
pub const SLAB_MAX: usize = 16384;

/// Largest allocation the Buddy allocator will accept. Beyond this, the request
/// is served directly by the System Fallback (whole-page(s) `mmap`). Tuned so a
/// single buddy region stays within a modest number of pages.
pub const BUDDY_MAX: usize = 1 << 20; // 1 MiB

/// Minimum alignment guaranteed by every backend. We target 16 bytes so SIMD
/// types are naturally aligned; larger alignment requests are honoured by the
/// System Fallback's page-aligned mapping.
pub const MIN_ALIGN: usize = 16;

/// Returns the index into [`SLAB_SIZE_CLASSES`] for `size`, or `None` if `size`
/// exceeds the largest slab class.
pub fn slab_class_for(size: usize) -> Option<usize> {
    if size == 0 {
        return Some(0);
    }
    SLAB_SIZE_CLASSES
        .iter()
        .position(|&c| c >= size)
        .filter(|&i| SLAB_SIZE_CLASSES[i] <= SLAB_MAX)
}

/// Rounds `size` up to the next power of two (minimum 1). Buddy allocator and
/// alignment math both use this.
pub fn round_up_pow2(size: usize) -> usize {
    if size <= 1 {
        return 1;
    }
    // Next power of two via the classic bit-twiddle. `leading_zeros` is the
    // portable, branchless way; works for all our target architectures.
    1usize << (usize::BITS - (size - 1).leading_zeros())
}

/// Rounds `n` up to the next multiple of `align`. `align` must be a power of
/// two — checked at runtime by the caller (the alloc backends always pass
/// power-of-two alignments).
pub fn align_up(n: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two() && align != 0);
    (n + align - 1) & !(align - 1)
}

/// True if `ptr` is aligned to `align`.
pub fn is_aligned(ptr: usize, align: usize) -> bool {
    debug_assert!(align.is_power_of_two() && align != 0);
    ptr & (align - 1) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slab_class_lookup() {
        assert_eq!(slab_class_for(0), Some(0));
        assert_eq!(slab_class_for(1), Some(0)); // 8
        assert_eq!(slab_class_for(8), Some(0));
        assert_eq!(slab_class_for(9), Some(1)); // 16
        assert_eq!(slab_class_for(100), Some(4)); // 128 (index 4)
        assert_eq!(slab_class_for(16384), Some(11));
        assert_eq!(slab_class_for(16385), None);
    }

    #[test]
    fn pow2_rounding() {
        assert_eq!(round_up_pow2(0), 1);
        assert_eq!(round_up_pow2(1), 1);
        assert_eq!(round_up_pow2(2), 2);
        assert_eq!(round_up_pow2(3), 4);
        assert_eq!(round_up_pow2(513), 1024);
    }

    #[test]
    fn alignment_math() {
        assert_eq!(align_up(5, 8), 8);
        assert_eq!(align_up(8, 8), 8);
        assert_eq!(align_up(17, 16), 32);
        assert!(is_aligned(align_up(0x1001, 4096), 4096));
        assert!(!is_aligned(0x1001, 4096));
    }
}
