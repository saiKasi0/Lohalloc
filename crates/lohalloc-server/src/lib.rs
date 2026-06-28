//! Lohalloc control-plane server (Phase 2/3 wiring — Phase 1 stub).
//!
//! In Phase 1 this crate is a minimal placeholder so the workspace builds. The
//! Axum WebSocket telemetry stream is added in Phase 2/3 once the Observer and
//! Decision Engine emit a telemetry stream over `crossbeam-channel`.
