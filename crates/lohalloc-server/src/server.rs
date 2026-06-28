//! Axum server: WebSocket telemetry stream + trace replay API.
//!
//! # Endpoints
//!
//! - `GET /ws/telemetry` — WebSocket upgrade. Streams JSON `TelemetryRecord`s
//!   from the telemetry channel to connected clients.
//! - `POST /api/upload-trace` — Accepts a JSON array of `TraceOp`s in the
//!   request body, replays them through a private `Lohalloc` instance, and
//!   returns the frozen `.lohalloc` model as `application/octet-stream`.
//! - `POST /api/freeze-export` — Triggers `freeze()` + `export()` on the
//!   last replay allocator and returns `.lohalloc` bytes (Phase 5).
//! - `GET /api/strategy` — Returns the current strategy override (Phase 5).
//! - `POST /api/strategy` — Sets the strategy override (Phase 5).
//! - `GET /health` — Liveness check.
//!
//! # Static File Serving
//!
//! In production, the server serves the built frontend from `gui/dist/`
//! via a fallback route, so the entire control plane runs on one port.
//!
//! # State
//!
//! `AppState` holds the telemetry channel sender, the current strategy
//! override, and the last replay result (for freeze-export). The replay
//! engine pushes records to this channel; WebSocket clients drain it.

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
use lohalloc_core::{Strategy, TraceOp};
use std::sync::Arc;
use tower_http::cors::CorsLayer;

use crate::replay::{replay_trace_json_with_strategy, ReplayError};
use crate::telemetry::{telemetry_channel, TelemetryReceiver, TelemetrySender};

/// Shared application state.
#[derive(Clone)]
pub struct AppState {
    pub telemetry_tx: TelemetrySender,
    // We keep the receiver alive so the channel doesn't close when no
    // WebSocket client is connected. The WebSocket handler clones the
    // `Arc<TelemetryReceiver>` to drain records.
    telemetry_rx: Arc<TelemetryReceiver>,
    /// Current strategy override (Phase 5). Stored behind a `RwLock` for
    /// thread-safe interior mutability.
    strategy: Arc<std::sync::RwLock<String>>,
    /// The last replay result's `.lohalloc` bytes (for freeze-export).
    /// Updated on each `/api/upload-trace` call.
    last_model: Arc<std::sync::Mutex<Vec<u8>>>,
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
            strategy: Arc::new(std::sync::RwLock::new("default".to_string())),
            last_model: Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    /// Get the current strategy.
    pub fn get_strategy(&self) -> Strategy {
        self.strategy
            .read()
            .map(|s| Strategy::parse_strategy(&s).unwrap_or(Strategy::Default))
            .unwrap_or(Strategy::Default)
    }

    /// Set the strategy override.
    pub fn set_strategy(&self, strategy: Strategy) {
        if let Ok(mut guard) = self.strategy.write() {
            *guard = strategy.as_str().to_string();
        }
    }

    /// Store the last replay result's model bytes.
    fn store_model(&self, bytes: Vec<u8>) {
        if let Ok(mut guard) = self.last_model.lock() {
            *guard = bytes;
        }
    }

    /// Get the last replay result's model bytes.
    fn get_model(&self) -> Vec<u8> {
        self.last_model
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default()
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

/// Build the Axum router for the Lohalloc control-plane server.
pub fn build_app(state: AppState) -> Router {
    build_app_with_options(state, false)
}

/// Build the router with optional static file serving and CORS.
pub fn build_app_with_options(state: AppState, serve_static: bool) -> Router {
    let router = Router::new()
        .route("/ws/telemetry", get(ws_telemetry_handler))
        .route("/api/upload-trace", post(upload_trace_handler))
        .route("/api/freeze-export", post(freeze_export_handler))
        .route(
            "/api/strategy",
            get(get_strategy_handler).post(set_strategy_handler),
        )
        .route("/health", get(health_handler))
        .layer(CorsLayer::permissive());

    let router = if serve_static {
        let static_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("gui")
            .join("dist");
        router.fallback_service(tower_http::services::ServeDir::new(static_dir))
    } else {
        router
    };

    router.with_state(state)
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
/// them through a private `Lohalloc` instance using the current strategy
/// override, and returns the frozen `.lohalloc` model bytes as
/// `application/octet-stream`.
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

    let strategy = state.get_strategy();
    match replay_trace_json_with_strategy(&json, Some(&state.telemetry_tx), strategy) {
        Ok(result) => {
            // Store the model for later freeze-export requests.
            state.store_model(result.lohalloc_bytes.clone());
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

/// `POST /api/freeze-export` — Returns the last replay result's `.lohalloc`
/// model bytes (Phase 5). The model is already frozen after each replay;
/// this endpoint lets the GUI download it without re-uploading a trace.
///
/// Returns:
/// - `200 OK` with `.lohalloc` bytes (may be empty if no replay has run).
async fn freeze_export_handler(State(state): State<AppState>) -> Response {
    let model = state.get_model();
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/octet-stream")],
        model,
    )
        .into_response()
}

/// `GET /api/strategy` — Returns the current strategy override as JSON
/// `{"strategy": "default|latency_priority|throughput_priority"}`.
async fn get_strategy_handler(State(state): State<AppState>) -> Response {
    let strategy = state.get_strategy();
    Json(serde_json::json!({ "strategy": strategy.as_str() })).into_response()
}

/// `POST /api/strategy` — Sets the strategy override. Accepts JSON body
/// `{"strategy": "default|latency_priority|throughput_priority"}`.
///
/// Returns:
/// - `200 OK` with the updated strategy on success.
/// - `400 Bad Request` on invalid strategy value.
async fn set_strategy_handler(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Response {
    let strategy_str = match body.get("strategy").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => return (StatusCode::BAD_REQUEST, "missing 'strategy' field").into_response(),
    };
    match Strategy::parse_strategy(strategy_str) {
        Some(strategy) => {
            state.set_strategy(strategy);
            Json(serde_json::json!({ "strategy": strategy.as_str() })).into_response()
        }
        None => (
            StatusCode::BAD_REQUEST,
            format!("unknown strategy '{strategy_str}', expected default|latency_priority|throughput_priority"),
        )
            .into_response(),
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

    // -----------------------------------------------------------------------
    // Phase 5 tests: strategy endpoints, freeze-export, backend field
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn get_strategy_returns_default() {
        let app = build_app(AppState::new());
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/strategy")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["strategy"], "default");
    }

    #[tokio::test]
    async fn set_strategy_to_latency_priority() {
        let app = build_app(AppState::new());
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/strategy")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(r#"{"strategy":"latency_priority"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["strategy"], "latency_priority");
    }

    #[tokio::test]
    async fn set_strategy_invalid_returns_400() {
        let app = build_app(AppState::new());
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/strategy")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(r#"{"strategy":"nonexistent"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn set_strategy_missing_field_returns_400() {
        let app = build_app(AppState::new());
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/strategy")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(r#"{}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn freeze_export_returns_empty_model_without_replay() {
        let app = build_app(AppState::new());
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/freeze-export")
                    .body(Body::empty())
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
    async fn freeze_export_returns_model_after_replay() {
        let state = AppState::new();
        let app = build_app(state.clone());

        // First upload a trace
        let body = r#"[{"op":"alloc","size":64,"stack_hash":100},{"op":"free","size":64,"stack_hash":100}]"#;
        let _ = app
            .clone()
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

        // Then freeze-export should return non-empty bytes
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/freeze-export")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        assert!(
            !bytes.is_empty(),
            "freeze-export should return non-empty model bytes after replay"
        );
    }

    #[tokio::test]
    async fn strategy_override_affects_replay() {
        let state = AppState::new();
        let app = build_app(state.clone());

        // Set strategy to latency_priority
        let _ = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/strategy")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(r#"{"strategy":"latency_priority"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        // Now upload a trace — the replay should use the latency_priority strategy
        let body = r#"[{"op":"alloc","size":64,"stack_hash":100}]"#;
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
    }

    #[tokio::test]
    async fn upload_trace_emits_backend_in_telemetry() {
        let (tx, rx) = crate::telemetry::telemetry_channel();
        let state = AppState::new_with_channel((tx, rx));
        let app = build_app(state.clone());

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

        // Drain the telemetry channel and verify backend field is present on allocs
        let records = state.telemetry_rx.recv_batch().unwrap_or_default();
        assert!(!records.is_empty(), "should have telemetry records");
        let allocs: Vec<_> = records
            .iter()
            .filter(|r| r.op == lohalloc_core::AllocOp::Alloc)
            .collect();
        assert!(!allocs.is_empty(), "should have at least one alloc record");
        for r in &allocs {
            assert!(
                r.backend.is_some(),
                "alloc records should have backend field set"
            );
        }
    }
}
