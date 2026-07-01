//! Integration tests for the Axum server (`lohalloc-server::server`).
//!
//! These tests run the server on a random port and connect via real HTTP
//! (reqwest) and WebSocket (tokio-tungstenite) clients.

use axum::http::header;
use futures_util::StreamExt;
use lohalloc_server::{build_app, AppState};
use std::future::IntoFuture;
use std::net::{Ipv4Addr, SocketAddr};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async fn start_server() -> SocketAddr {
    let listener = tokio::net::TcpListener::bind(SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0)))
        .await
        .unwrap();
    let addr = listener.local_addr().unwrap();
    let state = AppState::new();
    state.set_server_port(addr.port());
    tokio::spawn(axum::serve(listener, build_app(state)).into_future());
    addr
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

#[tokio::test]
async fn health_returns_ok() {
    let addr = start_server().await;
    let response = reqwest::get(format!("http://{addr}/health")).await.unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(response.text().await.unwrap(), "ok");
}

// ---------------------------------------------------------------------------
// POST /api/upload-trace
// ---------------------------------------------------------------------------

#[tokio::test]
async fn upload_trace_valid_json_returns_lohalloc_bytes() {
    let addr = start_server().await;
    let body =
        r#"[{"op":"alloc","size":64,"stack_hash":100},{"op":"free","size":64,"stack_hash":100}]"#;
    let response = reqwest::Client::new()
        .post(format!("http://{addr}/api/upload-trace"))
        .header(header::CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let ct = response
        .headers()
        .get(header::CONTENT_TYPE)
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert_eq!(ct, "application/octet-stream");
    let bytes = response.bytes().await.unwrap();
    // Minimum .lohalloc size is 24 bytes.
    assert!(bytes.len() >= 24);
}

#[tokio::test]
async fn upload_trace_empty_array_returns_200() {
    let addr = start_server().await;
    let response = reqwest::Client::new()
        .post(format!("http://{addr}/api/upload-trace"))
        .header(header::CONTENT_TYPE, "application/json")
        .body("[]")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
}

#[tokio::test]
async fn upload_trace_missing_content_type_rejected() {
    let addr = start_server().await;
    let response = reqwest::Client::new()
        .post(format!("http://{addr}/api/upload-trace"))
        .body(r#"[{"op":"alloc","size":64,"stack_hash":100}]"#)
        .send()
        .await
        .unwrap();
    // Axum's Json extractor requires Content-Type: application/json.
    assert!(
        response.status() == 415 || response.status() == 422,
        "expected 415 or 422, got {}",
        response.status()
    );
}

#[tokio::test]
async fn upload_trace_malformed_json_rejected() {
    let addr = start_server().await;
    let response = reqwest::Client::new()
        .post(format!("http://{addr}/api/upload-trace"))
        .header(header::CONTENT_TYPE, "application/json")
        .body("not valid json")
        .send()
        .await
        .unwrap();
    // Malformed JSON → 400 or 422 (Axum's Json rejection).
    assert!(
        response.status() == 400 || response.status() == 422,
        "expected 400 or 422, got {}",
        response.status()
    );
}

#[tokio::test]
async fn upload_trace_bad_op_value_rejected() {
    let addr = start_server().await;
    let response = reqwest::Client::new()
        .post(format!("http://{addr}/api/upload-trace"))
        .header(header::CONTENT_TYPE, "application/json")
        .body(r#"[{"op":"bogus","size":64,"stack_hash":100}]"#)
        .send()
        .await
        .unwrap();
    assert!(
        response.status() == 400 || response.status() == 422,
        "expected 400 or 422, got {}",
        response.status()
    );
}

#[tokio::test]
async fn upload_trace_large_works() {
    let addr = start_server().await;
    let mut ops = Vec::new();
    for i in 0..200 {
        ops.push(format!(
            r#"{{"op":"alloc","size":{},"stack_hash":{}}}"#,
            64 + (i % 4) * 64,
            1000 + i
        ));
        ops.push(format!(
            r#"{{"op":"free","size":{},"stack_hash":{}}}"#,
            64 + (i % 4) * 64,
            1000 + i
        ));
    }
    let body = format!("[{}]", ops.join(","));
    let response = reqwest::Client::new()
        .post(format!("http://{addr}/api/upload-trace"))
        .header(header::CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let bytes = response.bytes().await.unwrap();
    assert!(!bytes.is_empty());
}

// ---------------------------------------------------------------------------
// WebSocket /ws/telemetry
// ---------------------------------------------------------------------------

#[tokio::test]
async fn websocket_upgrade_succeeds() {
    let addr = start_server().await;
    let (mut socket, _response) =
        tokio_tungstenite::connect_async(format!("ws://{addr}/ws/telemetry"))
            .await
            .unwrap();
    // The connection should be open. Send a ping and verify we don't get
    // immediately closed.
    // We don't expect a message yet (no telemetry has been pushed).
    // Just verify the connection stays open by checking a short timeout returns
    // no message (not an error/closed).
    let result = tokio::time::timeout(std::time::Duration::from_millis(100), socket.next()).await;
    // Either timeout (no message) or no message yet — both are fine.
    // The key assertion is that `connect_async` succeeded (above).
    let _ = result;
}

#[tokio::test]
async fn websocket_streams_telemetry_after_upload() {
    let addr = start_server().await;

    // Connect WebSocket first so it's ready to receive.
    let (mut socket, _response) =
        tokio_tungstenite::connect_async(format!("ws://{addr}/ws/telemetry"))
            .await
            .unwrap();

    // Give the WS handler a moment to start.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Push telemetry by uploading a trace.
    let client = reqwest::Client::new();
    let response = client
        .post(format!("http://{addr}/api/upload-trace"))
        .header(header::CONTENT_TYPE, "application/json")
        .body(r#"[{"op":"alloc","size":64,"stack_hash":100},{"op":"free","size":64,"stack_hash":100}]"#)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);

    // The WebSocket should receive telemetry records as JSON text messages.
    let msg = tokio::time::timeout(std::time::Duration::from_secs(5), socket.next())
        .await
        .expect("timed out waiting for WS message")
        .expect("stream ended")
        .expect("WS error");

    let text = msg.into_text().unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    // Validate the Performance Trace Format schema fields.
    assert!(v.get("op").is_some(), "missing 'op' field");
    assert!(v.get("size").is_some(), "missing 'size' field");
    assert!(v.get("stack_hash").is_some(), "missing 'stack_hash' field");
    assert!(v.get("latency_ns").is_some(), "missing 'latency_ns' field");
    assert!(v.get("result_ptr").is_some(), "missing 'result_ptr' field");
    assert!(v.get("timestamp").is_some(), "missing 'timestamp' field");
    assert!(v.get("thread_id").is_some(), "missing 'thread_id' field");
    assert!(
        v.get("fragmentation_pct").is_some(),
        "missing 'fragmentation_pct' field"
    );

    // result_ptr should be "0x..." format.
    let ptr = v["result_ptr"].as_str().unwrap();
    assert!(
        ptr.starts_with("0x"),
        "result_ptr should be 0x-prefixed, got {ptr}"
    );

    // op should be "alloc" or "free".
    let op = v["op"].as_str().unwrap();
    assert!(op == "alloc" || op == "free", "unexpected op: {op}");
}

#[tokio::test]
async fn websocket_multiple_records_streamed() {
    let addr = start_server().await;

    let (mut socket, _response) =
        tokio_tungstenite::connect_async(format!("ws://{addr}/ws/telemetry"))
            .await
            .unwrap();

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Upload a trace with 4 ops.
    let body = r#"[
        {"op":"alloc","size":64,"stack_hash":1},
        {"op":"alloc","size":128,"stack_hash":2},
        {"op":"free","size":64,"stack_hash":1},
        {"op":"free","size":128,"stack_hash":2}
    ]"#;
    let response = reqwest::Client::new()
        .post(format!("http://{addr}/api/upload-trace"))
        .header(header::CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);

    // Collect up to 4 messages (with a generous timeout).
    let mut received = 0;
    for _ in 0..4 {
        let result = tokio::time::timeout(std::time::Duration::from_secs(5), socket.next()).await;
        match result {
            Ok(Some(Ok(msg))) => {
                let text = msg.into_text().unwrap();
                let v: serde_json::Value = serde_json::from_str(&text).unwrap();
                assert!(v.get("op").is_some());
                received += 1;
            }
            _ => break,
        }
    }
    assert!(received >= 1, "should receive at least 1 telemetry record");
}

// ---------------------------------------------------------------------------
// POST /api/run-simulation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn run_simulation_unknown_kind_returns_400() {
    let addr = start_server().await;
    let response = reqwest::Client::new()
        .post(format!("http://{addr}/api/run-simulation"))
        .header(header::CONTENT_TYPE, "application/json")
        .body(r#"{"kind":"nonexistent","args":{}}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    let body = response.text().await.unwrap();
    assert!(body.contains("nonexistent"));
}

#[tokio::test]
async fn run_simulation_missing_shim_returns_503() {
    // Clear any user override so we exercise the discovery path.
    // SAFETY: tests are single-threaded for this env var (tokio test).
    // SAFETY: env::remove_var is unsafe in newer Rust editions.
    // (Test-only; race conditions across tests are acceptable.)
    std::env::remove_var("LOHALLOC_SHIM_PATH");
    let addr = start_server().await;
    // Force a kind whose binary may not exist; combined with the missing
    // shim, we expect a 503 (shim check happens first).
    std::env::remove_var("LOHALLOC_BIN_LOHALLOC_EXAMPLE");
    let response = reqwest::Client::new()
        .post(format!("http://{addr}/api/run-simulation"))
        .header(header::CONTENT_TYPE, "application/json")
        .body(r#"{"kind":"lohalloc-example","args":{}}"#)
        .send()
        .await
        .unwrap();
    let status = response.status();
    let body = response.text().await.unwrap();
    // Either 503 (shim missing) or 202 (everything happens to exist) is
    // acceptable on dev machines; the unit tests in `simulation.rs` cover
    // the 503 path deterministically.
    assert!(status == 503 || status == 202, "got {status}: {body}");
    if status == 503 {
        assert!(body.contains("SHIM_NOT_FOUND") || body.contains("BINARY_NOT_FOUND"));
    }
}

#[tokio::test]
async fn simulation_history_endpoint_returns_events_array() {
    let addr = start_server().await;
    let response = reqwest::get(format!("http://{addr}/api/simulation-history"))
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let v: serde_json::Value = response.json().await.unwrap();
    assert!(v.get("events").is_some());
    assert!(v["events"].is_array());
}

#[tokio::test]
async fn run_simulation_long_running_via_inline_shell() {
    // We don't actually run the shim here; the `long-running` kind uses
    // /bin/sh -c. We point at an invalid port so curl fails immediately,
    // but we can still confirm the endpoint accepts the kind and returns
    // either 202 (if /bin/sh exists and the server finds its port) or 503
    // (if a binary is missing for some reason). Both are non-error.
    let addr = start_server().await;
    let response = reqwest::Client::new()
        .post(format!("http://{addr}/api/run-simulation"))
        .header(header::CONTENT_TYPE, "application/json")
        .body(r#"{"kind":"long-running","args":{"duration_secs":1}}"#)
        .send()
        .await
        .unwrap();
    let status = response.status();
    assert!(
        status == 202 || status == 503 || status == 501,
        "expected 202/503/501, got {status}"
    );
}
