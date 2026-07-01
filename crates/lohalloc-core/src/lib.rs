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
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
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

/// Strategy override for the Decision Engine (Phase 5).
///
/// When set, the replay engine biases backend selection toward the strategy's
/// preferred backend(s). This is a layer on top of the MAB — it doesn't
/// replace the bandit but overrides its recommendation when the preferred
/// backend can serve the request.
///
/// Real strategy-driven policy tuning (latency-based reward) arrives in
/// Phase 6. For Phase 5, the override is a simple preference filter.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Strategy {
    /// MAB-driven routing — no override (default).
    #[default]
    Default,
    /// Prefer fast backends (Slab for small, Arena for clusters).
    LatencyPriority,
    /// Prefer high-throughput backends (Buddy/Arena for bulk).
    ThroughputPriority,
}

impl Strategy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Strategy::Default => "default",
            Strategy::LatencyPriority => "latency_priority",
            Strategy::ThroughputPriority => "throughput_priority",
        }
    }

    /// Parse a `Strategy` from its snake_case string form.
    pub fn parse_strategy(s: &str) -> Option<Self> {
        match s.trim() {
            "default" => Some(Strategy::Default),
            "latency_priority" => Some(Strategy::LatencyPriority),
            "throughput_priority" => Some(Strategy::ThroughputPriority),
            _ => None,
        }
    }

    /// Returns the preferred backend for a given size under this strategy,
    /// or `None` if the strategy doesn't override (i.e., `Default`).
    pub fn preferred_backend(self, size: usize) -> Option<Backend> {
        match self {
            Strategy::Default => None,
            Strategy::LatencyPriority => {
                if size <= 16384 {
                    Some(Backend::Slab)
                } else {
                    Some(Backend::Buddy)
                }
            }
            Strategy::ThroughputPriority => {
                if size <= 16384 {
                    Some(Backend::Arena)
                } else {
                    Some(Backend::Buddy)
                }
            }
        }
    }
}

/// Allocation operation kind. Used in trace records and the uploaded trace
/// format (`{"op": "alloc|free", ...}`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum AllocOp {
    Alloc,
    Free,
}

impl AllocOp {
    pub const fn as_str(self) -> &'static str {
        match self {
            AllocOp::Alloc => "alloc",
            AllocOp::Free => "free",
        }
    }

    /// Parse an `AllocOp` from its lowercase string form. Used by the
    /// CSV/JSON trace parsers in `lohalloc-server`.
    pub fn parse_op(s: &str) -> Option<Self> {
        match s.trim() {
            "alloc" => Some(AllocOp::Alloc),
            "free" => Some(AllocOp::Free),
            _ => None,
        }
    }
}

/// A single telemetry record emitted by the allocator or the replay engine.
///
/// Matches the **Performance Trace Format** JSON schema documented in
/// `COPILOT.md`:
///
/// ```json
/// {
///   "timestamp": "u64",
///   "op": "alloc | free",
///   "size": "usize",
///   "stack_hash": "u64",
///   "thread_id": "u32",
///   "result_ptr": "0xAddr",
///   "latency_ns": "u64",
///   "fragmentation_pct": "f32",
///   "backend": "slab | buddy | system | arena"
/// }
/// ```
///
/// `result_ptr` is serialized as a hexadecimal string (`"0x..."`) for
/// human-readability in the GUI; deserialize accepts the same format.
///
/// `backend` (added in Phase 5) indicates which Execution-Plane worker
/// served the allocation. It is omitted from serialization when the backend
/// is unknown (`None`), ensuring backward compatibility with Phase 4
/// consumers.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TelemetryRecord {
    pub timestamp: u64,
    pub op: AllocOp,
    pub size: usize,
    pub stack_hash: u64,
    pub thread_id: u32,
    /// Pointer result serialized as `"0x<hex>"` over JSON.
    #[cfg_attr(
        feature = "serde",
        serde(serialize_with = "serialize_ptr", deserialize_with = "deserialize_ptr")
    )]
    pub result_ptr: u64,
    pub latency_ns: u64,
    pub fragmentation_pct: f32,
    /// Which backend served this allocation (Phase 5). `None` for free ops
    /// or when backend info is unavailable. Serialized only when `Some`.
    #[cfg_attr(
        feature = "serde",
        serde(skip_serializing_if = "Option::is_none", default)
    )]
    pub backend: Option<Backend>,
}

#[cfg(feature = "serde")]
fn serialize_ptr<S: serde::Serializer>(ptr: &u64, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(&format!("0x{ptr:x}"))
}

#[cfg(feature = "serde")]
fn deserialize_ptr<'de, D: serde::Deserializer<'de>>(d: D) -> Result<u64, D::Error> {
    use serde::Deserialize;
    let s = String::deserialize(d)?;
    let s = s.trim();
    let hex = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .unwrap_or(s);
    u64::from_str_radix(hex, 16).map_err(serde::de::Error::custom)
}

/// A single entry in an uploaded trace file (JSON or CSV). This is the
/// input format accepted by `POST /api/upload-trace` and
/// `replay_trace_json`.
///
/// ```json
/// {"op": "alloc", "size": 64, "stack_hash": 1234567890}
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TraceOp {
    pub op: AllocOp,
    pub size: usize,
    pub stack_hash: u64,
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
