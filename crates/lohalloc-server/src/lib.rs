//! Lohalloc control-plane server (Phase 4).
//!
//! The server exposes two surfaces for the GUI (Phase 5) and operator tools:
//!
//! - **WebSocket telemetry stream** (`/ws/telemetry`): Streams JSON
//!   `TelemetryRecord`s from the replay engine / allocator background thread
//!   to connected clients. Uses `crossbeam-channel` for lock-free IPC.
//! - **Trace replay API** (`POST /api/upload-trace`): Accepts a JSON array
//!   of `TraceOp`s, replays them through a private `Lohalloc` instance,
//!   freezes the trained bandit, and returns the serialized `.lohalloc`
//!   model as `application/octet-stream`.
//!
//! See `COPILOT.md` → "Phase 4 — Web UI Telemetry Server (Axum)" for the
//! full specification.

pub mod replay;
pub mod server;
pub mod telemetry;

pub use replay::{
    parse_csv_trace, parse_json_trace, replay_trace_csv, replay_trace_file, replay_trace_json,
    ReplayError, ReplayResult,
};
pub use server::{build_app, AppState};
pub use telemetry::{
    telemetry_channel, telemetry_channel_with_capacity, TelemetryReceiver, TelemetrySender,
    DEFAULT_CAPACITY,
};
