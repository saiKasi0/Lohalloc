//! Axum server: WebSocket telemetry stream + trace replay API.
//!
//! # Endpoints
//!
//! - `GET /ws/telemetry` — WebSocket upgrade. Streams JSON `TelemetryRecord`s
//!   from the telemetry channel to connected clients.
//! - `POST /api/upload-trace` — Accepts a JSON array of `TraceOp`s in the
//!   request body, replays them through a private `Lohalloc` instance, and
//!   returns the frozen `.lohalloc` model as `application/octet-stream`.
//!
//! # State
//!
//! `AppState` holds the telemetry channel sender. The replay engine pushes
//! records to this channel; WebSocket clients drain it.

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures_util::{sink::SinkExt, stream::StreamExt};
use lohalloc_core::TraceOp;
use std::sync::Arc;

use crate::replay::{replay_trace_json, ReplayError};
use crate::telemetry::{telemetry_channel, TelemetryReceiver, TelemetrySender};

/// Shared application state.
#[derive(Clone)]
pub struct AppState {
    pub telemetry_tx: TelemetrySender,
    // We keep the receiver alive so the channel doesn't close when no
    // WebSocket client is connected. The WebSocket handler clones the
    // `Arc<TelemetryReceiver>` to drain records.
    telemetry_rx: Arc<TelemetryReceiver>,
}

impl AppState {
    /// Create a new `AppState` with a fresh telemetry channel.
    pub fn new() -> Self {
        Self::new_with_channel(telemetry_channel())
    }

    /// Create an `AppState` from an existing channel pair (for testing).
    pub fn new_with_channel((tx, rx): (TelemetrySender, TelemetryReceiver)) -> Self {
        Self {
            telemetry_tx: tx,
            telemetry_rx: Arc::new(rx),
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

/// Build the Axum router for the Lohalloc control-plane server.
pub fn build_app(state: AppState) -> Router {
    Router::new()
        .route("/ws/telemetry", get(ws_telemetry_handler))
        .route("/api/upload-trace", post(upload_trace_handler))
        .route("/health", get(health_handler))
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// `GET /health` — simple liveness check.
async fn health_handler() -> &'static str {
    "ok"
}

/// `GET /ws/telemetry` — WebSocket upgrade. Streams JSON `TelemetryRecord`s.
async fn ws_telemetry_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> Response {
    ws.on_upgrade(move |socket| handle_telemetry_socket(socket, state))
}

/// Handle a single WebSocket connection: drain the telemetry channel and
/// send each record as a JSON text message. The connection stays open until
/// the client disconnects or the server is shut down.
async fn handle_telemetry_socket(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();
    let rx = state.telemetry_rx.clone();

    // Spawn a task that drains the telemetry channel and sends JSON records.
    // `crossbeam-channel::recv` is blocking, so we use `spawn_blocking` to
    // avoid stalling the async runtime.
    let mut send_task = tokio::spawn(async move {
        loop {
            let batch = tokio::task::spawn_blocking({
                let rx = rx.clone();
                move || rx.recv_batch()
            })
            .await;

            match batch {
                Ok(Some(records)) => {
                    for record in records {
                        let json = serde_json::to_string(&record).unwrap_or_default();
                        if sender.send(Message::text(json)).await.is_err() {
                            return;
                        }
                    }
                }
                Ok(None) => {
                    return;
                }
                Err(_) => return,
            }
        }
    });

    // Spawn a task that reads incoming messages (to detect client disconnect).
    let mut recv_task = tokio::spawn(async move {
        while receiver.next().await.is_some() {
            // Ignore incoming messages — this is a server-push stream.
        }
    });

    // If either task completes, abort the other.
    tokio::select! {
        _ = &mut send_task => recv_task.abort(),
        _ = &mut recv_task => send_task.abort(),
    }
}

/// `POST /api/upload-trace` — Accepts a JSON array of `TraceOp`s, replays
/// them through a private `Lohalloc` instance, and returns the frozen
/// `.lohalloc` model bytes as `application/octet-stream`.
///
/// Returns:
/// - `200 OK` with `.lohalloc` bytes on success.
/// - `400 Bad Request` on malformed JSON.
/// - `500 Internal Server Error` on replay failure.
async fn upload_trace_handler(
    State(state): State<AppState>,
    Json(trace): Json<Vec<TraceOp>>,
) -> Response {
    // Serialize the parsed trace back to JSON for the replay engine.
    // (The replay engine's `replay_trace_json` takes a string; we round-trip
    // through serde to validate the body was a proper `Vec<TraceOp>`.)
    let json = match serde_json::to_string(&trace) {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to serialize trace: {e}"),
            )
                .into_response()
        }
    };

    match replay_trace_json(&json, Some(&state.telemetry_tx)) {
        Ok(result) => {
            // Return .lohalloc bytes as octet-stream.
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "application/octet-stream")],
                result.lohalloc_bytes,
            )
                .into_response()
        }
        Err(ReplayError::JsonParse(msg)) => (StatusCode::BAD_REQUEST, msg).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use std::future::IntoFuture;
    use std::net::{Ipv4Addr, SocketAddr};
    use tower::ServiceExt;

    #[tokio::test]
    async fn health_endpoint_returns_ok() {
        let app = build_app(AppState::new());
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn upload_trace_returns_lohalloc_bytes() {
        let app = build_app(AppState::new());
        let body = r#"[{"op":"alloc","size":64,"stack_hash":100},{"op":"free","size":64,"stack_hash":100}]"#;
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/upload-trace")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let ct = response
            .headers()
            .get(header::CONTENT_TYPE)
            .unwrap()
            .to_str()
            .unwrap();
        assert_eq!(ct, "application/octet-stream");
    }

    #[tokio::test]
    async fn upload_trace_empty_array() {
        let app = build_app(AppState::new());
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/upload-trace")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from("[]"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn upload_trace_missing_content_type_returns_422() {
        // Axum's Json extractor requires Content-Type: application/json.
        let app = build_app(AppState::new());
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/upload-trace")
                    .body(Body::from(r#"[{"op":"alloc","size":64,"stack_hash":100}]"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        // Axum returns 415 Unsupported Media Type or 422 Unprocessable Entity
        // when the Content-Type is missing for a Json extractor.
        assert!(
            response.status() == StatusCode::UNSUPPORTED_MEDIA_TYPE
                || response.status() == StatusCode::UNPROCESSABLE_ENTITY
        );
    }

    #[tokio::test]
    async fn websocket_upgrade_succeeds() {
        let listener = tokio::net::TcpListener::bind(SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0)))
            .await
            .unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(axum::serve(listener, build_app(AppState::new())).into_future());

        // Connect to the WebSocket endpoint.
        let (mut socket, _response) =
            tokio_tungstenite::connect_async(format!("ws://{addr}/ws/telemetry"))
                .await
                .unwrap();

        // Push a telemetry record by running a replay via the upload endpoint.
        let client = reqwest::Client::new();
        let response = client
            .post(format!("http://{addr}/api/upload-trace"))
            .header(header::CONTENT_TYPE, "application/json")
            .body(r#"[{"op":"alloc","size":64,"stack_hash":100},{"op":"free","size":64,"stack_hash":100}]"#)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);

        // The WebSocket should receive at least one telemetry record as JSON.
        let msg = tokio::time::timeout(std::time::Duration::from_secs(5), socket.next())
            .await
            .expect("timed out waiting for WebSocket message")
            .expect("stream ended")
            .expect("WS error");

        let text = msg.into_text().unwrap();
        // Should be a JSON object with "op" field.
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert!(v.get("op").is_some());
        assert!(v.get("size").is_some());
        assert!(v.get("stack_hash").is_some());
        assert!(v.get("latency_ns").is_some());
        assert!(v.get("result_ptr").is_some());
        // result_ptr should be "0x..." format.
        let ptr = v["result_ptr"].as_str().unwrap();
        assert!(ptr.starts_with("0x"));
    }
}
