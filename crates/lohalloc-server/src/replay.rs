//! Replay engine — drives a private `Lohalloc` allocator instance through a
//! trace of allocation operations, records telemetry, and produces a frozen
//! `.lohalloc` model file.
//!
//! # Input Formats
//!
//! **JSON** (primary):
//! ```json
//! [
//!   {"op": "alloc", "size": 64, "stack_hash": 1234567890},
//!   {"op": "free", "size": 64, "stack_hash": 1234567890}
//! ]
//! ```
//!
//! **CSV** (secondary, hand-rolled parser — no `csv` crate dependency):
//! ```text
//! op,size,stack_hash
//! alloc,64,1234567890
//! free,64,1234567890
//! ```
//!
//! # Determinism
//!
//! Feeding the same trace twice (from the same fresh allocator state) produces
//! identical routing decisions — the MAB's UCB1 formula is deterministic
//! given the same reward sequence, and `freeze()` collapses weights
//! deterministically.
//!
//! # Private Allocator
//!
//! The replay engine instantiates its **own** `Lohalloc` (not the global
//! allocator) via `Lohalloc::new()`. This prevents replayed allocations from
//! corrupting the server process's own heap (the server uses the global
//! allocator for Axum/Tokio bookkeeping).

use core::alloc::Layout;
use lohalloc_alloc::Lohalloc;
use lohalloc_core::{AllocOp, Strategy, TelemetryRecord, TraceOp};
use std::time::Instant;

use crate::telemetry::TelemetrySender;

/// Errors produced by the replay engine.
#[derive(Debug)]
pub enum ReplayError {
    /// JSON parse failure (malformed JSON or schema mismatch).
    JsonParse(String),
    /// CSV parse failure.
    CsvParse(String),
    /// File I/O error.
    Io(std::io::Error),
    /// Allocation failed during replay (null pointer from the allocator).
    AllocFailed { op_index: usize },
}

impl std::fmt::Display for ReplayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReplayError::JsonParse(msg) => write!(f, "JSON parse error: {msg}"),
            ReplayError::CsvParse(msg) => write!(f, "CSV parse error: {msg}"),
            ReplayError::Io(e) => write!(f, "I/O error: {e}"),
            ReplayError::AllocFailed { op_index } => {
                write!(f, "allocation failed at trace op index {op_index}")
            }
        }
    }
}

impl std::error::Error for ReplayError {}

impl From<std::io::Error> for ReplayError {
    fn from(e: std::io::Error) -> Self {
        ReplayError::Io(e)
    }
}

/// Result of a replay run: the frozen `.lohalloc` model bytes plus a count
/// of operations executed and telemetry records emitted.
pub struct ReplayResult {
    /// Serialized `.lohalloc` routing table (from `freeze()` + `export()`).
    pub lohalloc_bytes: Vec<u8>,
    /// Number of trace operations executed.
    pub ops_executed: usize,
    /// Number of telemetry records emitted.
    pub records_emitted: usize,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Replay a JSON trace string through a fresh private `Lohalloc` instance,
/// freeze the trained bandit, and return the serialized `.lohalloc` model.
///
/// The `sender` (if provided) receives a `TelemetryRecord` for each
/// allocation/free operation, enabling WebSocket streaming.
pub fn replay_trace_json(
    json: &str,
    sender: Option<&TelemetrySender>,
) -> Result<ReplayResult, ReplayError> {
    replay_trace_json_with_strategy(json, sender, Strategy::Default)
}

/// Replay a JSON trace with a strategy override (Phase 5).
pub fn replay_trace_json_with_strategy(
    json: &str,
    sender: Option<&TelemetrySender>,
    strategy: Strategy,
) -> Result<ReplayResult, ReplayError> {
    let ops = parse_json_trace(json)?;
    Ok(replay_ops(&ops, sender, strategy))
}

/// Replay a CSV trace string through a fresh private `Lohalloc` instance.
///
/// Expected format:
/// ```text
/// op,size,stack_hash
/// alloc,64,1234567890
/// free,64,1234567890
/// ```
///
/// A header row is optional — if the first line starts with `op` it is
/// treated as a header and skipped.
pub fn replay_trace_csv(
    csv: &str,
    sender: Option<&TelemetrySender>,
) -> Result<ReplayResult, ReplayError> {
    replay_trace_csv_with_strategy(csv, sender, Strategy::Default)
}

/// Replay a CSV trace with a strategy override (Phase 5).
pub fn replay_trace_csv_with_strategy(
    csv: &str,
    sender: Option<&TelemetrySender>,
    strategy: Strategy,
) -> Result<ReplayResult, ReplayError> {
    let ops = parse_csv_trace(csv)?;
    Ok(replay_ops(&ops, sender, strategy))
}

/// Replay a trace file (auto-detected JSON or CSV by extension) from disk.
pub fn replay_trace_file(
    path: &str,
    sender: Option<&TelemetrySender>,
) -> Result<ReplayResult, ReplayError> {
    replay_trace_file_with_strategy(path, sender, Strategy::Default)
}

/// Replay a trace file with a strategy override (Phase 5).
pub fn replay_trace_file_with_strategy(
    path: &str,
    sender: Option<&TelemetrySender>,
    strategy: Strategy,
) -> Result<ReplayResult, ReplayError> {
    let content = std::fs::read_to_string(path)?;
    if path.ends_with(".csv") {
        replay_trace_csv_with_strategy(&content, sender, strategy)
    } else {
        replay_trace_json_with_strategy(&content, sender, strategy)
    }
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Parse a JSON trace array into `TraceOp`s.
pub fn parse_json_trace(json: &str) -> Result<Vec<TraceOp>, ReplayError> {
    serde_json::from_str::<Vec<TraceOp>>(json).map_err(|e| ReplayError::JsonParse(e.to_string()))
}

/// Parse a CSV trace string into `TraceOp`s.
///
/// Format: `op,size,stack_hash` per line. A header row starting with `op` is
/// skipped. Empty lines are ignored.
pub fn parse_csv_trace(csv: &str) -> Result<Vec<TraceOp>, ReplayError> {
    let mut ops = Vec::new();
    for (line_no, line) in csv.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Skip header row.
        if line_no == 0 && line.to_lowercase().starts_with("op") {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() != 3 {
            return Err(ReplayError::CsvParse(format!(
                "line {}: expected 3 columns (op,size,stack_hash), got {}",
                line_no + 1,
                parts.len()
            )));
        }
        let op = AllocOp::parse_op(parts[0]).ok_or_else(|| {
            ReplayError::CsvParse(format!(
                "line {}: unknown op '{}', expected 'alloc' or 'free'",
                line_no + 1,
                parts[0]
            ))
        })?;
        let size: usize = parts[1].trim().parse().map_err(|e| {
            ReplayError::CsvParse(format!("line {}: invalid size: {}", line_no + 1, e))
        })?;
        let stack_hash: u64 = parts[2].trim().parse().map_err(|e| {
            ReplayError::CsvParse(format!("line {}: invalid stack_hash: {}", line_no + 1, e))
        })?;
        ops.push(TraceOp {
            op,
            size,
            stack_hash,
        });
    }
    Ok(ops)
}

// ---------------------------------------------------------------------------
// Core replay logic
// ---------------------------------------------------------------------------

/// Drive a fresh `Lohalloc` instance through `ops`, emit telemetry to
/// `sender`, freeze, and export the `.lohalloc` model.
fn replay_ops(
    ops: &[TraceOp],
    sender: Option<&TelemetrySender>,
    strategy: Strategy,
) -> ReplayResult {
    let alloc = Lohalloc::new();
    let mut records_emitted = 0usize;
    let mut ops_executed = 0usize;

    // Track (ptr, size) pairs for free operations.
    let mut live: Vec<(*mut u8, usize)> = Vec::new();

    for (i, op) in ops.iter().enumerate() {
        let start = Instant::now();
        let (result_ptr, latency_ns, timestamp, backend) = match op.op {
            AllocOp::Alloc => {
                let layout =
                    Layout::from_size_align(op.size.max(1), 16).expect("invalid layout in replay");
                // Use the trace's stack_hash for deterministic routing,
                // plus the strategy override (Phase 5).
                let ptr = if strategy == Strategy::Default {
                    unsafe { alloc.alloc_with_hash(layout, op.stack_hash) }
                } else {
                    unsafe { alloc.alloc_with_hash_and_strategy(layout, op.stack_hash, strategy) }
                };
                let backend = if !ptr.is_null() {
                    unsafe { alloc.backend_for_ptr(ptr) }
                } else {
                    None
                };
                if ptr.is_null() {
                    // We still emit a telemetry record for the failure.
                    let latency = start.elapsed().as_nanos() as u64;
                    if let Some(s) = sender {
                        s.send(TelemetryRecord {
                            timestamp: i as u64,
                            op: AllocOp::Alloc,
                            size: op.size,
                            stack_hash: op.stack_hash,
                            thread_id: 0,
                            result_ptr: 0,
                            latency_ns: latency,
                            fragmentation_pct: 0.0,
                            backend: None,
                        });
                        records_emitted += 1;
                    }
                    continue;
                }
                live.push((ptr, op.size));
                (
                    ptr as u64,
                    start.elapsed().as_nanos() as u64,
                    i as u64,
                    backend,
                )
            }
            AllocOp::Free => {
                // Free the most recent live allocation of matching size (LIFO).
                let idx = live.iter().rposition(|(_, s)| *s == op.size);
                let (ptr, backend) = if let Some(idx) = idx {
                    let (ptr, size) = live.swap_remove(idx);
                    let layout = Layout::from_size_align(size.max(1), 16)
                        .expect("invalid layout in replay free");
                    let backend = unsafe { alloc.backend_for_ptr(ptr) };
                    unsafe { alloc.dealloc_with_hash(ptr, layout) };
                    (ptr as u64, backend)
                } else {
                    (0, None) // No matching live allocation — no-op.
                };
                (ptr, start.elapsed().as_nanos() as u64, i as u64, backend)
            }
        };

        ops_executed += 1;

        if let Some(s) = sender {
            s.send(TelemetryRecord {
                timestamp,
                op: op.op,
                size: op.size,
                stack_hash: op.stack_hash,
                thread_id: 0,
                result_ptr,
                latency_ns,
                // Fragmentation estimate is a placeholder for Phase 4 —
                // real fragmentation tracking arrives with the Observer.
                fragmentation_pct: 0.0,
                backend,
            });
            records_emitted += 1;
        }
    }

    // Freeze the trained bandit and export the `.lohalloc` model.
    alloc.freeze();
    let lohalloc_bytes = alloc.export().unwrap_or_default();

    ReplayResult {
        lohalloc_bytes,
        ops_executed,
        records_emitted,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::telemetry_channel;

    #[test]
    fn parse_simple_json() {
        let json = r#"[
            {"op": "alloc", "size": 64, "stack_hash": 100},
            {"op": "free", "size": 64, "stack_hash": 100}
        ]"#;
        let ops = parse_json_trace(json).unwrap();
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].op, AllocOp::Alloc);
        assert_eq!(ops[0].size, 64);
        assert_eq!(ops[1].op, AllocOp::Free);
    }

    #[test]
    fn parse_malformed_json() {
        assert!(parse_json_trace("not json").is_err());
        assert!(parse_json_trace("[{\"op\":\"bogus\"}]").is_err());
        assert!(parse_json_trace("[]").is_ok()); // empty is valid
    }

    #[test]
    fn parse_csv_with_header() {
        let csv = "op,size,stack_hash\nalloc,64,100\nfree,64,100\n";
        let ops = parse_csv_trace(csv).unwrap();
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].op, AllocOp::Alloc);
        assert_eq!(ops[1].op, AllocOp::Free);
    }

    #[test]
    fn parse_csv_without_header() {
        let csv = "alloc,32,50\nfree,32,50\n";
        let ops = parse_csv_trace(csv).unwrap();
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].size, 32);
    }

    #[test]
    fn parse_csv_malformed() {
        assert!(parse_csv_trace("alloc,64").is_err()); // too few columns
        assert!(parse_csv_trace("bogus,64,100").is_err()); // bad op
        assert!(parse_csv_trace("alloc,notanumber,100").is_err()); // bad size
    }

    #[test]
    fn parse_csv_empty_lines_skipped() {
        let csv = "op,size,stack_hash\n\nalloc,64,100\n\n\nfree,64,100\n";
        let ops = parse_csv_trace(csv).unwrap();
        assert_eq!(ops.len(), 2);
    }

    #[test]
    fn replay_produces_lohalloc_bytes() {
        let json = r#"[
            {"op": "alloc", "size": 64, "stack_hash": 100},
            {"op": "alloc", "size": 128, "stack_hash": 200},
            {"op": "free", "size": 64, "stack_hash": 100},
            {"op": "free", "size": 128, "stack_hash": 200}
        ]"#;
        let result = replay_trace_json(json, None).unwrap();
        assert!(result.ops_executed > 0);
        // .lohalloc bytes should be non-empty (magic + version + count + checksum).
        assert!(!result.lohalloc_bytes.is_empty());
        // Minimum .lohalloc size: 8 (magic) + 4 (version) + 4 (count) + 8 (checksum) = 24
        assert!(result.lohalloc_bytes.len() >= 24);
    }

    #[test]
    fn replay_empty_trace() {
        let result = replay_trace_json("[]", None).unwrap();
        assert_eq!(result.ops_executed, 0);
        // Even with zero ops, freeze() + export() produces a valid (empty) table.
        assert!(!result.lohalloc_bytes.is_empty());
    }

    #[test]
    fn replay_determinism_same_trace_same_model() {
        let json = r#"[
            {"op": "alloc", "size": 64, "stack_hash": 100},
            {"op": "alloc", "size": 64, "stack_hash": 100},
            {"op": "alloc", "size": 128, "stack_hash": 200},
            {"op": "free", "size": 64, "stack_hash": 100}
        ]"#;
        let r1 = replay_trace_json(json, None).unwrap();
        let r2 = replay_trace_json(json, None).unwrap();
        // Same trace → same frozen model (deterministic MAB + freeze).
        assert_eq!(r1.lohalloc_bytes, r2.lohalloc_bytes);
        assert_eq!(r1.ops_executed, r2.ops_executed);
    }

    #[test]
    fn replay_emits_telemetry_records() {
        let (tx, rx) = telemetry_channel();
        let json = r#"[
            {"op": "alloc", "size": 64, "stack_hash": 100},
            {"op": "free", "size": 64, "stack_hash": 100}
        ]"#;
        let result = replay_trace_json(json, Some(&tx)).unwrap();
        assert_eq!(result.records_emitted, 2);
        let records = rx.drain();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].op, AllocOp::Alloc);
        assert_eq!(records[1].op, AllocOp::Free);
        // Latency should be non-zero (we measured *something*).
        // (Could theoretically be 0 on very fast machines, so just check the field exists.)
        let _ = records[0].latency_ns;
    }

    #[test]
    fn replay_lohalloc_roundtrip() {
        // Replay a trace, get the .lohalloc model, load it into a fresh
        // allocator, and verify it's in Inference mode.
        let json = r#"[
            {"op": "alloc", "size": 64, "stack_hash": 100},
            {"op": "alloc", "size": 64, "stack_hash": 100},
            {"op": "free", "size": 64, "stack_hash": 100}
        ]"#;
        let result = replay_trace_json(json, None).unwrap();

        let fresh = Lohalloc::new();
        assert!(!fresh.is_inference());
        assert!(fresh.load(&result.lohalloc_bytes));
        assert!(fresh.is_inference());
    }

    #[test]
    fn replay_csv_via_file_format() {
        let csv = "op,size,stack_hash\nalloc,64,100\nfree,64,100\n";
        let result = replay_trace_csv(csv, None).unwrap();
        assert_eq!(result.ops_executed, 2);
        assert!(!result.lohalloc_bytes.is_empty());
    }

    #[test]
    fn replay_large_trace() {
        // 1000 alloc/free pairs — exercises the allocator under churn.
        let mut json_ops = Vec::new();
        for i in 0..500 {
            json_ops.push(format!(
                r#"{{"op": "alloc", "size": {}, "stack_hash": {}}}"#,
                64 + (i % 4) * 64,
                100 + i
            ));
            json_ops.push(format!(
                r#"{{"op": "free", "size": {}, "stack_hash": {}}}"#,
                64 + (i % 4) * 64,
                100 + i
            ));
        }
        let json = format!("[{}]", json_ops.join(","));
        let result = replay_trace_json(&json, None).unwrap();
        assert_eq!(result.ops_executed, 1000);
    }

    #[test]
    fn replay_mixed_sizes() {
        let json = r#"[
            {"op": "alloc", "size": 16, "stack_hash": 1},
            {"op": "alloc", "size": 256, "stack_hash": 2},
            {"op": "alloc", "size": 4096, "stack_hash": 3},
            {"op": "alloc", "size": 1048576, "stack_hash": 4},
            {"op": "free", "size": 16, "stack_hash": 1},
            {"op": "free", "size": 256, "stack_hash": 2},
            {"op": "free", "size": 4096, "stack_hash": 3},
            {"op": "free", "size": 1048576, "stack_hash": 4}
        ]"#;
        let result = replay_trace_json(json, None).unwrap();
        assert_eq!(result.ops_executed, 8);
    }
}
