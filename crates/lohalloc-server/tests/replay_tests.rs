//! Integration tests for the replay engine (`lohalloc-server::replay`).
//!
//! These tests exercise the public replay API through the crate boundary,
//! validating:
//! - JSON and CSV trace parsing (valid + malformed)
//! - Replay determinism (same trace → identical `.lohalloc` model)
//! - `.lohalloc` roundtrip (replay → export → load into fresh allocator)
//! - Telemetry emission during replay
//! - Mixed alloc/free operations
//! - Large traces under churn
//! - Edge cases: empty traces, unknown ops, missing fields
//!
//! Trace fixtures carry a required `timestamp` (ns) field/column — the replay
//! engine emits it verbatim as `TelemetryRecord.timestamp`.

use lohalloc_alloc::Lohalloc;
use lohalloc_core::{AllocOp, TelemetryRecord, TraceOp};
use lohalloc_server::{
    parse_csv_trace, parse_json_trace, replay_trace_csv, replay_trace_json, telemetry_channel,
};

// ---------------------------------------------------------------------------
// JSON parsing
// ---------------------------------------------------------------------------

#[test]
fn json_parse_valid_array() {
    let json = r#"[
        {"timestamp": 0, "op": "alloc", "size": 64, "stack_hash": 100},
        {"timestamp": 1, "op": "free", "size": 64, "stack_hash": 100},
        {"timestamp": 2, "op": "alloc", "size": 128, "stack_hash": 200}
    ]"#;
    let ops = parse_json_trace(json).unwrap();
    assert_eq!(ops.len(), 3);
    assert_eq!(ops[0].timestamp, 0);
    assert_eq!(ops[0].op, AllocOp::Alloc);
    assert_eq!(ops[0].size, 64);
    assert_eq!(ops[0].stack_hash, 100);
    assert_eq!(ops[1].op, AllocOp::Free);
    assert_eq!(ops[2].size, 128);
}

#[test]
fn json_parse_empty_array() {
    let ops = parse_json_trace("[]").unwrap();
    assert!(ops.is_empty());
}

#[test]
fn json_parse_malformed_not_json() {
    assert!(parse_json_trace("not json at all").is_err());
}

#[test]
fn json_parse_malformed_missing_field() {
    assert!(parse_json_trace(r#"[{"timestamp":0,"op":"alloc","size":64}]"#).is_err());
}

#[test]
fn json_parse_missing_timestamp_is_rejected() {
    // `timestamp` is required — a pre-timestamp (3-field) record no longer parses.
    assert!(parse_json_trace(r#"[{"op":"alloc","size":64,"stack_hash":100}]"#).is_err());
}

#[test]
fn json_parse_malformed_bad_op() {
    assert!(
        parse_json_trace(r#"[{"timestamp":0,"op":"bogus","size":64,"stack_hash":1}]"#).is_err()
    );
}

#[test]
fn json_parse_malformed_truncated() {
    assert!(parse_json_trace(r#"[{"timestamp":0,"op":"alloc""#).is_err());
}

// ---------------------------------------------------------------------------
// CSV parsing
// ---------------------------------------------------------------------------

#[test]
fn csv_parse_with_header() {
    let csv = "timestamp,op,size,stack_hash\n0,alloc,64,100\n1500000,free,64,100\n";
    let ops = parse_csv_trace(csv).unwrap();
    assert_eq!(ops.len(), 2);
    assert_eq!(ops[0].timestamp, 0);
    assert_eq!(ops[0].op, AllocOp::Alloc);
    assert_eq!(ops[1].timestamp, 1_500_000);
    assert_eq!(ops[1].op, AllocOp::Free);
}

#[test]
fn csv_parse_without_header() {
    let csv = "10,alloc,32,50\n20,free,32,50\n";
    let ops = parse_csv_trace(csv).unwrap();
    assert_eq!(ops.len(), 2);
    assert_eq!(ops[0].timestamp, 10);
}

#[test]
fn csv_parse_empty_lines() {
    let csv = "timestamp,op,size,stack_hash\n\n0,alloc,64,100\n\n\n1,free,64,100\n\n";
    let ops = parse_csv_trace(csv).unwrap();
    assert_eq!(ops.len(), 2);
}

#[test]
fn csv_parse_empty_input() {
    let ops = parse_csv_trace("").unwrap();
    assert!(ops.is_empty());
}

#[test]
fn csv_parse_wrong_column_count() {
    assert!(parse_csv_trace("0,alloc,64").is_err()); // 3 cols
    assert!(parse_csv_trace("0,alloc,64,100,extra").is_err()); // 5 cols
}

#[test]
fn csv_parse_legacy_3col_is_rejected() {
    // A pre-timestamp trace (3 columns) is now a hard error.
    assert!(parse_csv_trace("op,size,stack_hash\nalloc,64,100\n").is_err());
}

#[test]
fn csv_parse_bad_timestamp() {
    assert!(parse_csv_trace("notanumber,alloc,64,100").is_err());
}

#[test]
fn csv_parse_bad_op() {
    assert!(parse_csv_trace("0,bogus,64,100").is_err());
}

#[test]
fn csv_parse_bad_size() {
    assert!(parse_csv_trace("0,alloc,notanumber,100").is_err());
}

#[test]
fn csv_parse_bad_hash() {
    assert!(parse_csv_trace("0,alloc,64,notanumber").is_err());
}

#[test]
fn csv_parse_whitespace_tolerant() {
    let csv = "timestamp,op,size,stack_hash\n 0 , alloc , 64 , 100 \n 1 , free , 64 , 100 \n";
    let ops = parse_csv_trace(csv).unwrap();
    assert_eq!(ops.len(), 2);
    assert_eq!(ops[0].timestamp, 0);
    assert_eq!(ops[0].size, 64);
}

// ---------------------------------------------------------------------------
// Replay: basic functionality
// ---------------------------------------------------------------------------

#[test]
fn replay_simple_alloc_free() {
    let json = r#"[
        {"timestamp": 0, "op": "alloc", "size": 64, "stack_hash": 100},
        {"timestamp": 1500000, "op": "free", "size": 64, "stack_hash": 100}
    ]"#;
    let result = replay_trace_json(json, None).unwrap();
    assert_eq!(result.ops_executed, 2);
    assert!(!result.lohalloc_bytes.is_empty());
}

#[test]
fn replay_empty_trace_produces_valid_model() {
    let result = replay_trace_json("[]", None).unwrap();
    assert_eq!(result.ops_executed, 0);
    assert!(result.lohalloc_bytes.len() >= 24);
}

#[test]
fn replay_produces_loadable_model() {
    let json = r#"[
        {"timestamp": 0, "op": "alloc", "size": 64, "stack_hash": 100},
        {"timestamp": 1, "op": "alloc", "size": 64, "stack_hash": 100},
        {"timestamp": 2, "op": "alloc", "size": 256, "stack_hash": 200},
        {"timestamp": 3, "op": "free", "size": 64, "stack_hash": 100}
    ]"#;
    let result = replay_trace_json(json, None).unwrap();

    let fresh = Lohalloc::new();
    assert!(!fresh.is_inference());
    assert!(fresh.load(&result.lohalloc_bytes));
    assert!(fresh.is_inference());
}

// ---------------------------------------------------------------------------
// Replay: determinism
// ---------------------------------------------------------------------------

#[test]
fn replay_determinism_identical_models() {
    let json = r#"[
        {"timestamp": 0, "op": "alloc", "size": 64, "stack_hash": 100},
        {"timestamp": 1, "op": "alloc", "size": 128, "stack_hash": 200},
        {"timestamp": 2, "op": "alloc", "size": 64, "stack_hash": 100},
        {"timestamp": 3, "op": "free", "size": 64, "stack_hash": 100},
        {"timestamp": 4, "op": "free", "size": 128, "stack_hash": 200},
        {"timestamp": 5, "op": "free", "size": 64, "stack_hash": 100}
    ]"#;
    let r1 = replay_trace_json(json, None).unwrap();
    let r2 = replay_trace_json(json, None).unwrap();
    assert_eq!(
        r1.lohalloc_bytes, r2.lohalloc_bytes,
        "models must be identical"
    );
    assert_eq!(r1.ops_executed, r2.ops_executed);
}

// ---------------------------------------------------------------------------
// Replay: telemetry emission
// ---------------------------------------------------------------------------

#[test]
fn replay_emits_telemetry_for_each_op() {
    let (tx, rx) = telemetry_channel();
    let json = r#"[
        {"timestamp": 100, "op": "alloc", "size": 64, "stack_hash": 100},
        {"timestamp": 200, "op": "alloc", "size": 128, "stack_hash": 200},
        {"timestamp": 300, "op": "free", "size": 64, "stack_hash": 100},
        {"timestamp": 400, "op": "free", "size": 128, "stack_hash": 200}
    ]"#;
    let result = replay_trace_json(json, Some(&tx)).unwrap();
    assert_eq!(result.records_emitted, 4);

    let records: Vec<TelemetryRecord> = rx.drain();
    assert_eq!(records.len(), 4);
    assert_eq!(records[0].op, AllocOp::Alloc);
    assert_eq!(records[0].size, 64);
    assert_eq!(records[1].op, AllocOp::Alloc);
    assert_eq!(records[2].op, AllocOp::Free);
    assert_eq!(records[3].op, AllocOp::Free);
    // Emitted timestamps come straight from the trace, not the op index.
    assert_eq!(records[0].timestamp, 100);
    assert_eq!(records[1].timestamp, 200);
    assert_eq!(records[2].timestamp, 300);
    assert_eq!(records[3].timestamp, 400);
}

#[test]
fn replay_no_telemetry_without_sender() {
    let json = r#"[{"timestamp":0,"op":"alloc","size":64,"stack_hash":100}]"#;
    let result = replay_trace_json(json, None).unwrap();
    assert_eq!(result.records_emitted, 0);
}

#[test]
fn replay_telemetry_records_have_valid_fields() {
    let (tx, rx) = telemetry_channel();
    let json = r#"[{"timestamp":98765,"op":"alloc","size":64,"stack_hash":12345}]"#;
    replay_trace_json(json, Some(&tx)).unwrap();

    let records = rx.drain();
    assert_eq!(records.len(), 1);
    let r = &records[0];
    assert_eq!(r.timestamp, 98765);
    assert_eq!(r.op, AllocOp::Alloc);
    assert_eq!(r.size, 64);
    assert_eq!(r.stack_hash, 12345);
    assert_eq!(r.thread_id, 0);
    assert!(r.result_ptr != 0, "alloc should produce non-null ptr");
    assert!(
        r.fragmentation_pct >= 0.0,
        "fragmentation should be non-negative"
    );
}

// ---------------------------------------------------------------------------
// Replay: mixed sizes and large traces
// ---------------------------------------------------------------------------

#[test]
fn replay_mixed_sizes_small_to_large() {
    let json = r#"[
        {"timestamp": 0, "op": "alloc", "size": 16, "stack_hash": 1},
        {"timestamp": 1, "op": "alloc", "size": 256, "stack_hash": 2},
        {"timestamp": 2, "op": "alloc", "size": 4096, "stack_hash": 3},
        {"timestamp": 3, "op": "alloc", "size": 65536, "stack_hash": 4},
        {"timestamp": 4, "op": "alloc", "size": 1048576, "stack_hash": 5},
        {"timestamp": 5, "op": "free", "size": 16, "stack_hash": 1},
        {"timestamp": 6, "op": "free", "size": 256, "stack_hash": 2},
        {"timestamp": 7, "op": "free", "size": 4096, "stack_hash": 3},
        {"timestamp": 8, "op": "free", "size": 65536, "stack_hash": 4},
        {"timestamp": 9, "op": "free", "size": 1048576, "stack_hash": 5}
    ]"#;
    let result = replay_trace_json(json, None).unwrap();
    assert_eq!(result.ops_executed, 10);
    assert!(!result.lohalloc_bytes.is_empty());
}

#[test]
fn replay_large_trace_1000_ops() {
    let mut ops = Vec::new();
    for i in 0..500 {
        let size = 64 + (i % 8) * 64;
        ops.push(format!(
            r#"{{"timestamp": {}, "op": "alloc", "size": {}, "stack_hash": {}}}"#,
            i * 2,
            size,
            1000 + i
        ));
        ops.push(format!(
            r#"{{"timestamp": {}, "op": "free", "size": {}, "stack_hash": {}}}"#,
            i * 2 + 1,
            size,
            1000 + i
        ));
    }
    let json = format!("[{}]", ops.join(","));
    let result = replay_trace_json(&json, None).unwrap();
    assert_eq!(result.ops_executed, 1000);
    assert!(!result.lohalloc_bytes.is_empty());
}

#[test]
fn replay_allocs_without_frees() {
    let json = r#"[
        {"timestamp": 0, "op": "alloc", "size": 64, "stack_hash": 1},
        {"timestamp": 1, "op": "alloc", "size": 64, "stack_hash": 1},
        {"timestamp": 2, "op": "alloc", "size": 64, "stack_hash": 1}
    ]"#;
    let result = replay_trace_json(json, None).unwrap();
    assert_eq!(result.ops_executed, 3);
}

#[test]
fn replay_free_without_matching_alloc_is_noop() {
    let json = r#"[
        {"timestamp": 0, "op": "free", "size": 64, "stack_hash": 100},
        {"timestamp": 1, "op": "alloc", "size": 128, "stack_hash": 200},
        {"timestamp": 2, "op": "free", "size": 64, "stack_hash": 100}
    ]"#;
    let result = replay_trace_json(json, None).unwrap();
    assert_eq!(result.ops_executed, 3);
}

// ---------------------------------------------------------------------------
// CSV replay
// ---------------------------------------------------------------------------

#[test]
fn replay_csv_trace() {
    let csv = "timestamp,op,size,stack_hash\n0,alloc,64,100\n1500000,free,64,100\n";
    let result = replay_trace_csv(csv, None).unwrap();
    assert_eq!(result.ops_executed, 2);
    assert!(!result.lohalloc_bytes.is_empty());
}

#[test]
fn replay_csv_empty_trace() {
    let csv = "timestamp,op,size,stack_hash\n";
    let result = replay_trace_csv(csv, None).unwrap();
    assert_eq!(result.ops_executed, 0);
}

// ---------------------------------------------------------------------------
// Serde roundtrips
// ---------------------------------------------------------------------------

#[test]
fn trace_op_serde_roundtrip() {
    let ops = vec![
        TraceOp {
            timestamp: 0,
            op: AllocOp::Alloc,
            size: 64,
            stack_hash: 100,
        },
        TraceOp {
            timestamp: 1500000,
            op: AllocOp::Free,
            size: 64,
            stack_hash: 100,
        },
    ];
    let json = serde_json::to_string(&ops).unwrap();
    let parsed: Vec<TraceOp> = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, ops);
}

#[test]
fn alloc_op_serialized_as_lowercase() {
    assert_eq!(
        serde_json::to_string(&AllocOp::Alloc).unwrap(),
        r#""alloc""#
    );
    assert_eq!(serde_json::to_string(&AllocOp::Free).unwrap(), r#""free""#);
}

#[test]
fn telemetry_record_serde_roundtrip() {
    let record = TelemetryRecord {
        timestamp: 12345,
        op: AllocOp::Alloc,
        size: 64,
        stack_hash: 999,
        thread_id: 7,
        result_ptr: 0xdeadbeef,
        latency_ns: 500,
        fragmentation_pct: 12.5,
        backend: None,
    };
    let json = serde_json::to_string(&record).unwrap();
    assert!(json.contains("0xdeadbeef"));
    let parsed: TelemetryRecord = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.timestamp, 12345);
    assert_eq!(parsed.op, AllocOp::Alloc);
    assert_eq!(parsed.size, 64);
    assert_eq!(parsed.result_ptr, 0xdeadbeef);
    assert_eq!(parsed.latency_ns, 500);
}
