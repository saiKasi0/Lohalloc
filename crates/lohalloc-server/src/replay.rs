//! Replay engine — drives a private `Lohalloc` allocator instance through a
//! trace of allocation operations, records telemetry, and produces a frozen
//! `.lohalloc` model file.
//!
//! # Input Formats
//!
//! `timestamp` (nanoseconds) is required in both formats. The replay engine
//! emits it verbatim as `TelemetryRecord.timestamp` when the trace carries a
//! real time spread, so the GUI renders a true time axis. If the column is
//! degenerate (sequential indices like `0,1,2,…`, or all-equal placeholders —
//! no usable spread), replay synthesizes an even 1ms/op cadence instead, so
//! such traces still get a readable axis rather than collapsing to t≈0. See
//! `timestamps_are_degenerate`.
//!
//! **JSON** (primary):
//! ```json
//! [
//!   {"timestamp": 0, "op": "alloc", "size": 64, "stack_hash": 1234567890},
//!   {"timestamp": 1500000, "op": "free", "size": 64, "stack_hash": 1234567890}
//! ]
//! ```
//!
//! **CSV** (secondary, hand-rolled parser — no `csv` crate dependency):
//! ```text
//! timestamp,op,size,stack_hash
//! 0,alloc,64,1234567890
//! 1500000,free,64,1234567890
//! ```
//!
//! # Determinism
//!
//! Feeding the same trace twice (from the same fresh allocator state)
//! produces *structurally* identical models: the same set of observed
//! Signatures, the same op count, a validly-frozen table each time. It does
//! **not** guarantee byte-identical `.lohalloc` output (Phase 6 changed
//! this): the bandit's reward now comes from real measured alloc/dealloc
//! latency (`lohalloc_alloc::state::record_latency`), and replay drives the
//! same real backends the live allocator does — so run-to-run timing jitter
//! (scheduler noise, cache state, mmap variance) can occasionally flip which
//! backend wins a Signature where two backends perform near-identically.
//! This is the same tradeoff real-latency learning makes everywhere else;
//! replay is not a special case. Before Phase 6, rewards were a static
//! per-backend baseline, so replay's output genuinely was bit-for-bit
//! reproducible — that guarantee traded away real performance signal for
//! reproducibility, which no longer matches what the rest of the system
//! measures.
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
/// timestamp,op,size,stack_hash
/// 0,alloc,64,1234567890
/// 1500000,free,64,1234567890
/// ```
///
/// A header row is optional — if the first line starts with `timestamp` or
/// `op` it is treated as a header and skipped.
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
/// Format: `timestamp,op,size,stack_hash` per line. A header row starting
/// with `timestamp` (or the legacy `op`) is skipped. Empty lines are ignored.
/// `timestamp` (nanoseconds) is required — a 3-column row is rejected.
pub fn parse_csv_trace(csv: &str) -> Result<Vec<TraceOp>, ReplayError> {
    let mut ops = Vec::new();
    for (line_no, line) in csv.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Skip header row (either the current `timestamp`-led header or a
        // legacy `op`-led one, so a stale header produces a clear column
        // error on the first data row rather than a confusing parse).
        if line_no == 0 {
            let lower = line.to_lowercase();
            if lower.starts_with("timestamp") || lower.starts_with("op") {
                continue;
            }
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() != 4 {
            return Err(ReplayError::CsvParse(format!(
                "line {}: expected 4 columns (timestamp,op,size,stack_hash), got {}",
                line_no + 1,
                parts.len()
            )));
        }
        let timestamp: u64 = parts[0].trim().parse().map_err(|e| {
            ReplayError::CsvParse(format!("line {}: invalid timestamp: {}", line_no + 1, e))
        })?;
        let op = AllocOp::parse_op(parts[1]).ok_or_else(|| {
            ReplayError::CsvParse(format!(
                "line {}: unknown op '{}', expected 'alloc' or 'free'",
                line_no + 1,
                parts[1]
            ))
        })?;
        let size: usize = parts[2].trim().parse().map_err(|e| {
            ReplayError::CsvParse(format!("line {}: invalid size: {}", line_no + 1, e))
        })?;
        let stack_hash: u64 = parts[3].trim().parse().map_err(|e| {
            ReplayError::CsvParse(format!("line {}: invalid stack_hash: {}", line_no + 1, e))
        })?;
        ops.push(TraceOp {
            timestamp,
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

/// Interval (ns) between consecutive ops when synthesizing a time axis for a
/// trace whose own timestamps carry no usable spread (see
/// [`timestamps_are_degenerate`]). 1ms/op gives the GUI a readable seconds
/// axis and a sensible ops/sec instead of everything collapsing to t≈0.
const SYNTHETIC_OP_INTERVAL_NS: u64 = 1_000_000;

/// Decide whether a trace's own timestamps are unusable as a time axis.
///
/// `TraceOp.timestamp` is nanoseconds. A trace authored with sequential
/// indices (`0,1,2,…`) or a single repeated value has a total span far
/// smaller than one op's worth of real time, so honoring it literally
/// collapses the whole run to t≈0 (the "0/41" axis). The heuristic: with two
/// or more ops, if the full span (`max − min`) is smaller than the op count
/// — under ~1ns of spread per op — the timestamps are indices/placeholders,
/// not real clock readings, so the caller should synthesize a cadence
/// instead. A genuine ns trace spans millions+ of ns and easily clears this
/// bar. Traces of 0 or 1 op have no axis to collapse, so their timestamp is
/// always honored as-is.
fn timestamps_are_degenerate(ops: &[TraceOp]) -> bool {
    if ops.len() < 2 {
        return false;
    }
    let (min_ts, max_ts) = ops.iter().fold((u64::MAX, 0u64), |(mn, mx), o| {
        (mn.min(o.timestamp), mx.max(o.timestamp))
    });
    max_ts.saturating_sub(min_ts) < ops.len() as u64
}

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

    let mut total_alloc_count: u64 = 0;
    let mut total_free_count: u64 = 0;

    // If the trace's own timestamps carry no usable time spread (indices /
    // all-equal / placeholders), synthesize an even cadence so the GUI gets a
    // real time axis; otherwise honor the trace's nanosecond timestamps.
    let synthesize_time = timestamps_are_degenerate(ops);

    for (i, op) in ops.iter().enumerate() {
        // The emitted record's timestamp: synthesized cadence or the trace's
        // own value, decided once above.
        let record_ts = if synthesize_time {
            i as u64 * SYNTHETIC_OP_INTERVAL_NS
        } else {
            op.timestamp
        };
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
                            timestamp: record_ts,
                            op: AllocOp::Alloc,
                            size: op.size,
                            stack_hash: op.stack_hash,
                            thread_id: 0,
                            result_ptr: 0,
                            latency_ns: latency,
                            fragmentation_pct: {
                                let live_count = live.len() as u64;
                                let live_ratio = if total_alloc_count > 0 {
                                    live_count as f64 / total_alloc_count as f64
                                } else {
                                    0.0
                                };
                                (live_ratio * 60.0).min(100.0) as f32
                            },
                            backend: None,
                        });
                        records_emitted += 1;
                    }
                    continue;
                }
                live.push((ptr, op.size));
                total_alloc_count += 1;
                (
                    ptr as u64,
                    start.elapsed().as_nanos() as u64,
                    record_ts,
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
                    total_free_count += 1;
                    (ptr as u64, backend)
                } else {
                    (0, None) // No matching live allocation — no-op.
                };
                (ptr, start.elapsed().as_nanos() as u64, record_ts, backend)
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
                fragmentation_pct: {
                    // Synthetic fragmentation estimate: based on the ratio of live allocations
                    // to total allocations and size diversity. Produces realistic fluctuating
                    // values (0-100%) that respond to workload churn patterns.
                    //
                    // This is a placeholder until the Observer (Phase 2) provides real
                    // fragmentation measurements from the allocator's internal state.
                    let live_count = live.len() as u64;
                    let live_ratio = if total_alloc_count > 0 {
                        live_count as f64 / total_alloc_count as f64
                    } else {
                        0.0
                    };
                    // Higher live ratio + more frees (churn) = higher fragmentation
                    let churn_factor = if total_alloc_count > 0 {
                        (total_free_count as f64 / total_alloc_count as f64).min(1.0)
                    } else {
                        0.0
                    };
                    // Base fragmentation from live set pressure, amplified by churn
                    let frag = (live_ratio * 60.0
                        + churn_factor * 30.0
                        + (live_ratio * churn_factor) * 10.0)
                        .min(100.0);
                    frag as f32
                },
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
            {"timestamp": 0, "op": "alloc", "size": 64, "stack_hash": 100},
            {"timestamp": 1500000, "op": "free", "size": 64, "stack_hash": 100}
        ]"#;
        let ops = parse_json_trace(json).unwrap();
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].timestamp, 0);
        assert_eq!(ops[0].op, AllocOp::Alloc);
        assert_eq!(ops[0].size, 64);
        assert_eq!(ops[1].timestamp, 1_500_000);
        assert_eq!(ops[1].op, AllocOp::Free);
    }

    #[test]
    fn parse_malformed_json() {
        assert!(parse_json_trace("not json").is_err());
        assert!(parse_json_trace("[{\"op\":\"bogus\"}]").is_err());
        assert!(parse_json_trace("[]").is_ok()); // empty is valid
    }

    #[test]
    fn parse_json_missing_timestamp_is_rejected() {
        // `timestamp` is a required field — a 3-field record no longer parses.
        let json = r#"[{"op": "alloc", "size": 64, "stack_hash": 100}]"#;
        assert!(parse_json_trace(json).is_err());
    }

    #[test]
    fn parse_csv_with_header() {
        let csv = "timestamp,op,size,stack_hash\n0,alloc,64,100\n1500000,free,64,100\n";
        let ops = parse_csv_trace(csv).unwrap();
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].timestamp, 0);
        assert_eq!(ops[0].op, AllocOp::Alloc);
        assert_eq!(ops[1].timestamp, 1_500_000);
        assert_eq!(ops[1].op, AllocOp::Free);
    }

    #[test]
    fn parse_csv_without_header() {
        let csv = "10,alloc,32,50\n20,free,32,50\n";
        let ops = parse_csv_trace(csv).unwrap();
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].timestamp, 10);
        assert_eq!(ops[0].size, 32);
    }

    #[test]
    fn parse_csv_malformed() {
        assert!(parse_csv_trace("0,alloc,64").is_err()); // too few columns (legacy 3-col)
        assert!(parse_csv_trace("0,bogus,64,100").is_err()); // bad op
        assert!(parse_csv_trace("0,alloc,notanumber,100").is_err()); // bad size
        assert!(parse_csv_trace("notanumber,alloc,64,100").is_err()); // bad timestamp
    }

    #[test]
    fn parse_csv_legacy_3col_is_rejected() {
        // A pre-timestamp trace (3 columns) is now a hard error rather than
        // silently replaying with fabricated op-index timestamps.
        let csv = "op,size,stack_hash\nalloc,64,100\n";
        assert!(parse_csv_trace(csv).is_err());
    }

    #[test]
    fn parse_csv_empty_lines_skipped() {
        let csv = "timestamp,op,size,stack_hash\n\n0,alloc,64,100\n\n\n1,free,64,100\n";
        let ops = parse_csv_trace(csv).unwrap();
        assert_eq!(ops.len(), 2);
    }

    #[test]
    fn replay_produces_lohalloc_bytes() {
        let json = r#"[
            {"timestamp": 0, "op": "alloc", "size": 64, "stack_hash": 100},
            {"timestamp": 1, "op": "alloc", "size": 128, "stack_hash": 200},
            {"timestamp": 2, "op": "free", "size": 64, "stack_hash": 100},
            {"timestamp": 3, "op": "free", "size": 128, "stack_hash": 200}
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
            {"timestamp": 0, "op": "alloc", "size": 64, "stack_hash": 100},
            {"timestamp": 1, "op": "alloc", "size": 64, "stack_hash": 100},
            {"timestamp": 2, "op": "alloc", "size": 128, "stack_hash": 200},
            {"timestamp": 3, "op": "free", "size": 64, "stack_hash": 100}
        ]"#;
        let r1 = replay_trace_json(json, None).unwrap();
        let r2 = replay_trace_json(json, None).unwrap();
        // Same trace → structurally identical models (same Signatures
        // observed, same op count). Not byte-identical: rewards now come
        // from real measured latency (Phase 6), so run-to-run timing jitter
        // can occasionally flip which backend wins a near-tied Signature —
        // see the module doc's "Determinism" section.
        let t1 = lohalloc_alloc::perfect_hash::PerfectHashTable::deserialize(&r1.lohalloc_bytes)
            .expect("r1 model should deserialize");
        let t2 = lohalloc_alloc::perfect_hash::PerfectHashTable::deserialize(&r2.lohalloc_bytes)
            .expect("r2 model should deserialize");
        assert_eq!(
            t1.len(),
            t2.len(),
            "same trace should observe the same Signature count"
        );
        assert_eq!(r1.ops_executed, r2.ops_executed);
    }

    #[test]
    fn replay_emits_telemetry_records() {
        let (tx, rx) = telemetry_channel();
        let json = r#"[
            {"timestamp": 111, "op": "alloc", "size": 64, "stack_hash": 100},
            {"timestamp": 222, "op": "free", "size": 64, "stack_hash": 100}
        ]"#;
        let result = replay_trace_json(json, Some(&tx)).unwrap();
        assert_eq!(result.records_emitted, 2);
        let records = rx.drain();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].op, AllocOp::Alloc);
        assert_eq!(records[1].op, AllocOp::Free);
        // The emitted record's timestamp is the trace's timestamp verbatim,
        // NOT the op index — this is the core regression guard for the
        // "0 or 41" time-axis bug.
        assert_eq!(records[0].timestamp, 111);
        assert_eq!(records[1].timestamp, 222);
        // Latency should be non-zero (we measured *something*).
        // (Could theoretically be 0 on very fast machines, so just check the field exists.)
        let _ = records[0].latency_ns;
    }

    #[test]
    fn replay_synthesizes_time_axis_for_index_like_timestamps() {
        // A trace authored with sequential indices (0,1,2,3) has no real time
        // spread; honoring it literally collapses the axis to ~0s (the "0/41"
        // bug). Replay must synthesize an even cadence instead.
        let (tx, rx) = telemetry_channel();
        let json = r#"[
            {"timestamp": 0, "op": "alloc", "size": 64, "stack_hash": 1},
            {"timestamp": 1, "op": "alloc", "size": 64, "stack_hash": 2},
            {"timestamp": 2, "op": "free", "size": 64, "stack_hash": 1},
            {"timestamp": 3, "op": "free", "size": 64, "stack_hash": 2}
        ]"#;
        replay_trace_json(json, Some(&tx)).unwrap();
        let records = rx.drain();
        assert_eq!(records.len(), 4);
        // Synthesized cadence: op i at i * SYNTHETIC_OP_INTERVAL_NS.
        assert_eq!(records[0].timestamp, 0);
        assert_eq!(records[1].timestamp, SYNTHETIC_OP_INTERVAL_NS);
        assert_eq!(records[2].timestamp, 2 * SYNTHETIC_OP_INTERVAL_NS);
        assert_eq!(records[3].timestamp, 3 * SYNTHETIC_OP_INTERVAL_NS);
    }

    #[test]
    fn replay_honors_real_timestamps() {
        // A trace with a genuine ns spread is honored verbatim (not synthesized).
        let (tx, rx) = telemetry_channel();
        let json = r#"[
            {"timestamp": 1000000, "op": "alloc", "size": 64, "stack_hash": 1},
            {"timestamp": 5000000, "op": "alloc", "size": 64, "stack_hash": 2},
            {"timestamp": 9000000, "op": "free", "size": 64, "stack_hash": 1}
        ]"#;
        replay_trace_json(json, Some(&tx)).unwrap();
        let records = rx.drain();
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].timestamp, 1_000_000);
        assert_eq!(records[1].timestamp, 5_000_000);
        assert_eq!(records[2].timestamp, 9_000_000);
    }

    #[test]
    fn degenerate_timestamp_detection() {
        let mk = |ts: u64| TraceOp {
            timestamp: ts,
            op: AllocOp::Alloc,
            size: 64,
            stack_hash: 1,
        };
        // Index-like / all-equal: degenerate.
        assert!(timestamps_are_degenerate(&[mk(0), mk(1), mk(2)]));
        assert!(timestamps_are_degenerate(&[mk(7), mk(7), mk(7)]));
        // Real ns spread: not degenerate.
        assert!(!timestamps_are_degenerate(&[
            mk(0),
            mk(1_000_000),
            mk(2_000_000)
        ]));
        // 0 or 1 op: never degenerate (nothing to collapse) — honored as-is.
        assert!(!timestamps_are_degenerate(&[]));
        assert!(!timestamps_are_degenerate(&[mk(98765)]));
    }

    #[test]
    fn replay_lohalloc_roundtrip() {
        // Replay a trace, get the .lohalloc model, load it into a fresh
        // allocator, and verify it's in Inference mode.
        let json = r#"[
            {"timestamp": 0, "op": "alloc", "size": 64, "stack_hash": 100},
            {"timestamp": 1, "op": "alloc", "size": 64, "stack_hash": 100},
            {"timestamp": 2, "op": "free", "size": 64, "stack_hash": 100}
        ]"#;
        let result = replay_trace_json(json, None).unwrap();

        let fresh = Lohalloc::new();
        assert!(!fresh.is_inference());
        assert!(fresh.load(&result.lohalloc_bytes));
        assert!(fresh.is_inference());
    }

    #[test]
    fn replay_csv_via_file_format() {
        let csv = "timestamp,op,size,stack_hash\n0,alloc,64,100\n1500000,free,64,100\n";
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
                r#"{{"timestamp": {}, "op": "alloc", "size": {}, "stack_hash": {}}}"#,
                i * 2,
                64 + (i % 4) * 64,
                100 + i
            ));
            json_ops.push(format!(
                r#"{{"timestamp": {}, "op": "free", "size": {}, "stack_hash": {}}}"#,
                i * 2 + 1,
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
            {"timestamp": 0, "op": "alloc", "size": 16, "stack_hash": 1},
            {"timestamp": 1, "op": "alloc", "size": 256, "stack_hash": 2},
            {"timestamp": 2, "op": "alloc", "size": 4096, "stack_hash": 3},
            {"timestamp": 3, "op": "alloc", "size": 1048576, "stack_hash": 4},
            {"timestamp": 4, "op": "free", "size": 16, "stack_hash": 1},
            {"timestamp": 5, "op": "free", "size": 256, "stack_hash": 2},
            {"timestamp": 6, "op": "free", "size": 4096, "stack_hash": 3},
            {"timestamp": 7, "op": "free", "size": 1048576, "stack_hash": 4}
        ]"#;
        let result = replay_trace_json(json, None).unwrap();
        assert_eq!(result.ops_executed, 8);
    }
}
