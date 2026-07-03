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
//! - `POST /api/freeze` — Freeze the live training allocator (TensorBoard
//!   "commit"). Collapses the live MAB into a frozen routing table and
//!   stores it for `/api/freeze-export`. Returns `{"frozen_entries": N}`.
//! - `POST /api/reset-training` — Reset the live allocator back to fresh
//!   Training mode (discards frozen routing table).
//! - `GET /api/training-status` — Live-training diagnostics
//!   (`{signatures, live_allocations, inference}`) for the GUI's
//!   convergence indicator.
//! - `GET /api/strategy` — Returns the current strategy override (Phase 5).
//! - `POST /api/strategy` — Sets the strategy override (Phase 5).
//! - `GET /api/mode` — Returns `{"mode": "training"}` or
//!   `{"mode": "inference"}` depending on Decision Engine state.
//! - `GET /api/routing-table` — Returns the current routing table as a
//!   JSON array of `{hash, backend}` objects. Inference mode returns
//!   the frozen table; Training mode returns the live MAB snapshot
//!   (TensorBoard-style, current best per signature).
//! - `POST /api/telemetry` — Accepts a single `TelemetryRecord` or an array
//!   of records (Phase 5++: live mode from LD_PRELOAD shim or external
//!   producers). Forwards each record to the existing telemetry channel so
//!   `/ws/telemetry` clients receive it in real time. Returns `202 Accepted`.
//! - `GET /api/export-trace` — Returns a JSON array snapshot of the most
//!   recent telemetry records (`TelemetryRecord[]`). Backed by an
//!   independent ring buffer (`TRACE_RING_CAPACITY` records) that mirrors
//!   what `POST /api/telemetry` / `POST /api/upload-trace` push, so it does
//!   not interfere with the live WS stream.
//! - `POST /api/run-simulation` — Spawns a real Lohalloc workload with the
//!   `liblohalloc_obs` shim preloaded. Body:
//!   `{"kind": "lohalloc-example|long-running|stress-test|...", "args": {...}}`.
//!   Streams lifecycle events over `/ws/telemetry` as
//!   `{"type":"simulation","event":{...}}` messages. Returns `202 Accepted`
//!   with the spawned `pid`. Refuses requests from non-loopback IPs unless
//!   `LOHALLOC_ALLOW_REMOTE_SPAWN=1` is set in the env.
//! - `POST /api/kill-all-simulations` — Emergency stop. Sends SIGKILL to
//!   all running simulation subprocesses. Returns `{"killed": N}`.
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
        Path, State,
    },
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures_util::{sink::SinkExt, stream::StreamExt};
use lohalloc_alloc::Lohalloc;
use lohalloc_core::{AllocOp, Strategy, TelemetryRecord, TraceOp};
use serde::Deserialize;
use std::collections::VecDeque;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tracing::Instrument;

use crate::replay::{replay_trace_json_with_strategy, ReplayError};
use crate::simulation::{self, SimulationArgs, SimulationEvent, SimulationKind};
use crate::telemetry::{telemetry_channel, RawWsMessage, TelemetryReceiver, TelemetrySender};

/// Maximum number of telemetry records retained for `GET /api/export-trace`.
/// At ~200 bytes per record this caps the ring at ~13 MB.
pub const TRACE_RING_CAPACITY: usize = 65_536;

/// Minimum spacing (ns) between consecutive live telemetry timestamps.
///
/// Live records arrive from the shim in bursts — a single `POST /api/telemetry`
/// can carry thousands of records captured within the same clock tick. Stamping
/// them all with the same instant would collapse the GUI's seconds axis back to
/// a flat line, so we enforce a strictly-increasing floor of `MIN_LIVE_TS_STEP_NS`
/// per record: a burst spreads across at least `n * 1µs` while a genuinely idle
/// gap lets the real clock jump ahead. See [`AppState::next_live_timestamp`].
pub const MIN_LIVE_TS_STEP_NS: u64 = 1_000;

/// Shared application state.
#[derive(Clone)]
pub struct AppState {
    pub telemetry_tx: TelemetrySender,
    // We keep the receiver alive so the crossbeam channel doesn't close
    // when no WebSocket client is connected. WS clients now use the
    // broadcast channel instead; this is kept for the trace ring buffer
    // and tests.
    #[allow(dead_code)]
    telemetry_rx: Arc<TelemetryReceiver>,
    /// Broadcast sender for WS fan-out. Each WS client subscribes by calling
    /// `subscribe()`, getting its own receiver so records aren't stolen by
    /// other (possibly zombie) connections.
    ws_broadcast: Arc<tokio::sync::broadcast::Sender<TelemetryRecord>>,
    /// Broadcast sender for raw WS messages (simulation lifecycle events).
    ws_raw_broadcast: Arc<tokio::sync::broadcast::Sender<RawWsMessage>>,
    /// Current strategy override (Phase 5). Stored behind a `RwLock` for
    /// thread-safe interior mutability.
    strategy: Arc<std::sync::RwLock<String>>,
    /// The last replay result's `.lohalloc` bytes (for freeze-export).
    /// Updated on each `/api/upload-trace` call.
    last_model: Arc<std::sync::Mutex<Vec<u8>>>,
    /// Whether the Decision Engine is in Inference (frozen) mode. Toggled
    /// by `/api/upload-trace` and `/api/freeze-export` (set to `true`) and
    /// by `/api/strategy` (reset to `false`, since a strategy change
    /// returns the allocator to Training mode).
    is_inference: Arc<std::sync::RwLock<bool>>,
    /// Cached routing table extracted from the last frozen model. Exposed
    /// via `GET /api/routing-table` so the GUI can render the Policy
    /// Matrix without re-parsing the binary model on each request.
    routing_table: Arc<std::sync::RwLock<Vec<(u64, String)>>>,
    /// Ring buffer of recent telemetry records for `GET /api/export-trace`.
    /// Independent of the live WS stream (which drains `telemetry_rx`); a
    /// snapshot is served on demand. Capacity: [`TRACE_RING_CAPACITY`]
    /// records.
    trace_ring: Arc<std::sync::Mutex<VecDeque<TelemetryRecord>>>,
    /// The TCP port the server is bound to. Defaults to `0` (unknown) until
    /// `set_server_port` is called by `main.rs`. Used by `/api/run-simulation`
    /// to tell spawned subprocesses which port to POST telemetry to.
    server_port: Arc<std::sync::RwLock<u16>>,
    /// Active simulation subprocess handles, keyed by pid. Cleaned up on
    /// `exited` / `failed`. Sized to bound memory; oldest are evicted if
    /// the cap is reached.
    simulations: Arc<std::sync::Mutex<std::collections::HashMap<u32, SimulationHandle>>>,
    /// History of completed simulations (capped). Used by the GUI's
    /// `SimulationPanel` to show past runs even after they've exited.
    simulation_history: Arc<std::sync::Mutex<Vec<SimulationEvent>>>,
    /// Live training allocator (TensorBoard-style). Receives every
    /// incoming telemetry record from `POST /api/telemetry` so the MAB
    /// learns in real time as the shim workload runs. When the user
    /// triggers `/api/freeze`, this is what gets frozen — no replay
    /// required.
    live_alloc: Arc<Lohalloc>,
    /// Tracks `(ptr, size)` pairs from live allocations so free
    /// operations can be routed back through `dealloc_with_hash`.
    /// Pointers are stored as `usize` so this field stays `Send` —
    /// `(*mut u8, usize)` is not. Cleared on reset to training.
    live_allocations: Arc<std::sync::Mutex<Vec<(usize, usize)>>>,
    /// Last nanosecond timestamp assigned to a live telemetry record by
    /// [`AppState::next_live_timestamp`]. Guarantees a strictly-increasing
    /// time axis for the live stream regardless of what the shim/observer
    /// (or a stale binary) actually reported in the record.
    live_ts_last: Arc<std::sync::atomic::AtomicU64>,
    /// Single server-wide monotonic epoch. Live timestamps are nanoseconds
    /// **since this instant**, not since the Unix epoch: relative ns stays
    /// small (< 2^53 for ~100 days of uptime) so it round-trips through a
    /// JavaScript `number` (an IEEE-754 double) losslessly. Absolute wall-
    /// clock ns (~1.7e18) would exceed 2^53 and snap to ~1µs granularity in
    /// the browser, collapsing the GUI/CSV time axis. `Instant` is `Copy`, so
    /// every `AppState` clone shares the value captured once in `new`.
    live_epoch: std::time::Instant,
}

/// Cap on the simulation history shown in the GUI.
pub const SIMULATION_HISTORY_CAP: usize = 32;

/// Lightweight bookkeeping for a running simulation. The actual `Child`
/// handle lives behind a `tokio::sync::Mutex` so the async cleanup task
/// can `try_wait()` on it.
pub struct SimulationHandle {
    pub kind: crate::simulation::SimulationKind,
    pub started_at: std::time::Instant,
    pub child: Arc<tokio::sync::Mutex<std::process::Child>>,
    /// Set to `true` when the operator kills the sim via `/api/stop-simulation`.
    /// The watcher checks this to avoid emitting a duplicate exit event.
    pub killed_by_operator: Arc<std::sync::atomic::AtomicBool>,
}

impl AppState {
    /// Create a new `AppState` with a fresh telemetry channel.
    pub fn new() -> Self {
        Self::new_with_channel(telemetry_channel())
    }

    /// Create an `AppState` from an existing channel pair (for testing).
    pub fn new_with_channel((mut tx, rx): (TelemetrySender, TelemetryReceiver)) -> Self {
        let (ws_tx, _) = tokio::sync::broadcast::channel(4096);
        let (ws_raw_tx, _) = tokio::sync::broadcast::channel(512);
        tx.attach_broadcast(ws_tx.clone(), ws_raw_tx.clone());
        Self {
            telemetry_tx: tx,
            telemetry_rx: Arc::new(rx),
            ws_broadcast: Arc::new(ws_tx),
            ws_raw_broadcast: Arc::new(ws_raw_tx),
            strategy: Arc::new(std::sync::RwLock::new("default".to_string())),
            last_model: Arc::new(std::sync::Mutex::new(Vec::new())),
            is_inference: Arc::new(std::sync::RwLock::new(false)),
            routing_table: Arc::new(std::sync::RwLock::new(Vec::new())),
            trace_ring: Arc::new(std::sync::Mutex::new(VecDeque::with_capacity(
                TRACE_RING_CAPACITY,
            ))),
            server_port: Arc::new(std::sync::RwLock::new(0)),
            simulations: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            simulation_history: Arc::new(std::sync::Mutex::new(Vec::new())),
            live_alloc: Arc::new(Lohalloc::new()),
            live_allocations: Arc::new(std::sync::Mutex::new(Vec::<(usize, usize)>::new())),
            live_ts_last: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            live_epoch: std::time::Instant::now(),
        }
    }

    /// Assign the next strictly-increasing nanosecond timestamp for a live
    /// telemetry record, measured against the server's monotonic epoch.
    ///
    /// The shim/observer's own `timestamp` field is ignored on the live path:
    /// it is process-relative (each spawned subprocess re-anchors its own
    /// monotonic epoch at ~0) and has been observed to degenerate to tiny,
    /// repeating values (the "0 / 41" symptom). Instead we stamp against one
    /// server-wide epoch (`live_epoch`) — a single authoritative monotonic
    /// clock — flooring each reading to at least [`MIN_LIVE_TS_STEP_NS`] past
    /// the previous one so a burst of records arriving within one clock tick
    /// still gets a monotonically increasing, non-collapsing axis. Relative ns
    /// stays under 2^53, so the value survives the browser's `number` type
    /// without the precision loss that made the exported CSV time axis idle.
    pub fn next_live_timestamp(&self) -> u64 {
        use std::sync::atomic::Ordering;
        let elapsed = self.live_epoch.elapsed().as_nanos() as u64;
        // CAS loop: candidate = max(elapsed, prev + step); publish it.
        let mut prev = self.live_ts_last.load(Ordering::Relaxed);
        loop {
            let candidate = elapsed.max(prev.saturating_add(MIN_LIVE_TS_STEP_NS));
            match self.live_ts_last.compare_exchange_weak(
                prev,
                candidate,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return candidate,
                Err(actual) => prev = actual,
            }
        }
    }

    /// Record the port the server is bound to. Called by `main.rs` after
    /// `TcpListener::bind` so `/api/run-simulation` can pass the correct
    /// `LOHALLOC_OBS_PORT` to its children.
    pub fn set_server_port(&self, port: u16) {
        if let Ok(mut g) = self.server_port.write() {
            *g = port;
        }
    }

    /// Get the bound port (or `0` if unknown).
    pub fn get_server_port(&self) -> u16 {
        self.server_port.read().map(|g| *g).unwrap_or(0)
    }

    /// Register a freshly-spawned simulation in the active set.
    pub fn register_simulation(
        &self,
        pid: u32,
        kind: crate::simulation::SimulationKind,
        child: std::process::Child,
    ) {
        let handle = SimulationHandle {
            kind,
            started_at: std::time::Instant::now(),
            child: Arc::new(tokio::sync::Mutex::new(child)),
            killed_by_operator: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        };
        if let Ok(mut g) = self.simulations.lock() {
            g.insert(pid, handle);
        }
    }

    /// Look up a registered simulation. Returns `None` if the pid is not
    /// currently tracked (either never registered, or already cleaned up).
    pub fn get_simulation(
        &self,
        pid: u32,
    ) -> Option<(
        Arc<tokio::sync::Mutex<std::process::Child>>,
        Arc<std::sync::atomic::AtomicBool>,
    )> {
        self.simulations.lock().ok().and_then(|g| {
            g.get(&pid)
                .map(|h| (h.child.clone(), h.killed_by_operator.clone()))
        })
    }

    /// Get the kind of a registered simulation.
    pub fn get_simulation_kind(&self, pid: u32) -> Option<crate::simulation::SimulationKind> {
        self.simulations
            .lock()
            .ok()
            .and_then(|g| g.get(&pid).map(|h| h.kind))
    }

    /// Remove a simulation from the active set after it has exited or failed.
    /// Returns the kind so the caller can record a final lifecycle event.
    pub fn unregister_simulation(&self, pid: u32) -> Option<crate::simulation::SimulationKind> {
        self.simulations
            .lock()
            .ok()
            .and_then(|mut g| g.remove(&pid).map(|h| h.kind))
    }

    /// Kill a running simulation by pid. Sends `SIGKILL` to the child's
    /// entire process group (so grandchildren are also killed) and returns
    /// the simulation kind if found.
    pub async fn kill_simulation(&self, pid: u32) -> Option<crate::simulation::SimulationKind> {
        let (child_arc, killed_flag) = self.get_simulation(pid)?;
        // Set the operator-kill flag so the watcher doesn't emit a duplicate.
        killed_flag.store(true, std::sync::atomic::Ordering::Relaxed);
        let mut guard = child_arc.lock().await;
        let child_pid = guard.id();
        tracing::info!(pid, "kill_simulation");

        // Kill the entire process group so grandchildren are cleaned up too.
        #[cfg(unix)]
        {
            let pgid = child_pid as i32;
            let _ = unsafe { libc::kill(-pgid, libc::SIGKILL) };
        }
        let _ = guard.kill();
        let kind = self.get_simulation_kind(pid);
        self.unregister_simulation(pid);
        tracing::info!(pid, ?kind, "simulation killed");
        kind
    }

    /// Kill ALL running simulations. Returns the number of simulations killed.
    /// Used by the "Kill All" emergency-stop button.
    pub async fn kill_all_simulations(&self) -> usize {
        let pids: Vec<u32> = if let Ok(g) = self.simulations.lock() {
            g.keys().copied().collect()
        } else {
            return 0;
        };
        let mut killed = 0;
        for pid in pids {
            if self.kill_simulation(pid).await.is_some() {
                killed += 1;
            }
        }
        killed
    }

    /// Best-effort **synchronous** kill of every running simulation's
    /// process group. Unlike [`kill_all_simulations`] this needs no async
    /// runtime and never `.await`s a child lock, so it is safe to call from
    /// a panic hook or a signal-handling context where the async executor
    /// may be gone or unusable.
    ///
    /// Children are spawned as process-group leaders (`process_group(0)` in
    /// `simulation.rs`), so the group id equals the child pid — which is the
    /// map key. That lets us `SIGKILL` the whole group (child + any
    /// grandchildren) directly from the key without locking the child. The
    /// registry lock is taken with `try_lock` so a poisoned/held lock during
    /// a panic can't deadlock; we just skip cleanup in that rare case.
    /// Returns the number of process groups signalled.
    pub fn kill_all_simulations_blocking(&self) -> usize {
        let mut killed = 0;
        if let Ok(g) = self.simulations.try_lock() {
            for pid in g.keys() {
                #[cfg(unix)]
                {
                    // Negative pid targets the whole process group.
                    let _ = unsafe { libc::kill(-(*pid as i32), libc::SIGKILL) };
                }
                killed += 1;
            }
        }
        killed
    }

    /// Append a finished lifecycle event to the bounded history.
    pub fn push_simulation_history(&self, event: crate::simulation::SimulationEvent) {
        if let Ok(mut g) = self.simulation_history.lock() {
            if g.len() >= SIMULATION_HISTORY_CAP {
                g.remove(0);
            }
            g.push(event);
        }
    }

    /// Snapshot the simulation history for `GET /api/simulation-history`.
    pub fn snapshot_simulation_history(&self) -> Vec<crate::simulation::SimulationEvent> {
        self.simulation_history
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default()
    }

    /// Snapshot the currently-running simulations for `GET /api/simulation-history`.
    /// Each entry is the same `SimulationEvent` shape but with `status: "running"`.
    pub fn snapshot_active_simulations(&self) -> Vec<crate::simulation::SimulationEvent> {
        let now = std::time::Instant::now();
        self.simulations
            .lock()
            .map(|g| {
                g.iter()
                    .map(|(pid, h)| crate::simulation::SimulationEvent {
                        pid: *pid,
                        kind: h.kind.as_str().to_string(),
                        status: "running".to_string(),
                        duration_ms: now.duration_since(h.started_at).as_millis() as u64,
                        exit_code: None,
                        stdout_tail: None,
                        error: None,
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Push a record into the export-trace ring buffer. When the buffer is
    /// full, the oldest record is dropped. The lock is best-effort: if it
    /// is poisoned we silently skip the push (the telemetry channel is the
    /// authoritative live stream — losing a snapshot record is non-fatal).
    pub fn push_trace_record(&self, record: TelemetryRecord) {
        if let Ok(mut ring) = self.trace_ring.lock() {
            if ring.len() >= TRACE_RING_CAPACITY {
                ring.pop_front();
            }
            ring.push_back(record);
        }
    }

    /// Snapshot the current contents of the ring buffer in insertion order.
    /// Returns an empty `Vec` if the lock is poisoned.
    pub fn snapshot_trace_ring(&self) -> Vec<TelemetryRecord> {
        self.trace_ring
            .lock()
            .map(|r| r.iter().copied().collect())
            .unwrap_or_default()
    }

    /// Get the current strategy.
    pub fn get_strategy(&self) -> Strategy {
        self.strategy
            .read()
            .map(|s| Strategy::parse_strategy(&s).unwrap_or(Strategy::Default))
            .unwrap_or(Strategy::Default)
    }

    /// Set the strategy override. Resets the allocator to Training mode
    /// since a strategy change invalidates the frozen routing table.
    pub fn set_strategy(&self, strategy: Strategy) {
        if let Ok(mut guard) = self.strategy.write() {
            *guard = strategy.as_str().to_string();
        }
        self.set_inference(false);
        if let Ok(mut guard) = self.routing_table.write() {
            guard.clear();
        }
    }

    /// Store the last replay result's model bytes and transition to
    /// Inference mode. The routing table is extracted from the frozen
    /// model's binary format and cached for `/api/routing-table`.
    fn store_model(&self, bytes: Vec<u8>) {
        let entries = decode_routing_entries(&bytes);
        if let Ok(mut guard) = self.last_model.lock() {
            *guard = bytes;
        }
        if let Ok(mut guard) = self.routing_table.write() {
            *guard = entries;
        }
        self.set_inference(true);
    }

    /// Get the last replay result's model bytes.
    fn get_model(&self) -> Vec<u8> {
        self.last_model
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default()
    }

    /// Toggle the Inference mode flag.
    fn set_inference(&self, value: bool) {
        if let Ok(mut guard) = self.is_inference.write() {
            *guard = value;
        }
    }

    /// Read the current Inference mode.
    pub fn is_inference(&self) -> bool {
        self.is_inference.read().map(|g| *g).unwrap_or(false)
    }

    /// Get a snapshot of the cached routing table.
    fn get_routing_table(&self) -> Vec<(u64, String)> {
        self.routing_table
            .read()
            .map(|g| g.clone())
            .unwrap_or_default()
    }

    // -----------------------------------------------------------------
    // Live training allocator (TensorBoard-style)
    //
    // The server holds a single `Lohalloc` instance in Training mode and
    // feeds every incoming telemetry record from `POST /api/telemetry`
    // through its `alloc_with_hash`/`dealloc_with_hash` API. This means
    // the MAB learns from the SAME workload the shim is running, in
    // real time — no replay required. When the user triggers a freeze
    // (via `/api/freeze`), the live allocator collapses its bandit
    // weights into a frozen routing table and exports a `.lohalloc`.
    // -----------------------------------------------------------------

    /// Feed one telemetry record into the live training allocator.
    /// Allocations are actually performed; frees find the matching
    /// pointer in `live_allocations` and pass it to
    /// `dealloc_with_hash`. Pointer-tracking is necessary because the
    /// shim only sends `(op, size, stack_hash)` — no pointer — and
    /// `GlobalAlloc::dealloc` requires a pointer.
    fn feed_live_record(&self, record: &TelemetryRecord) {
        let layout = match std::alloc::Layout::from_size_align(record.size.max(1), 16) {
            Ok(l) => l,
            Err(_) => return, // Invalid layout — silently skip.
        };
        match record.op {
            AllocOp::Alloc => {
                let ptr = unsafe { self.live_alloc.alloc_with_hash(layout, record.stack_hash) };
                if !ptr.is_null() {
                    if let Ok(mut live) = self.live_allocations.lock() {
                        live.push((ptr as usize, record.size));
                    }
                }
            }
            AllocOp::Free => {
                let pair = if let Ok(mut live) = self.live_allocations.lock() {
                    live.iter()
                        .rposition(|(_, s)| *s == record.size)
                        .map(|idx| live.swap_remove(idx))
                } else {
                    None
                };
                if let Some((ptr_usize, _size)) = pair {
                    let ptr = ptr_usize as *mut u8;
                    unsafe { self.live_alloc.dealloc_with_hash(ptr, layout) };
                }
            }
        }
    }

    /// Number of live allocations the live allocator is currently
    /// tracking. Useful for diagnostics / convergence display.
    pub fn live_allocation_count(&self) -> usize {
        self.live_allocations
            .lock()
            .map(|g| g.len())
            .unwrap_or_default()
    }

    /// Number of distinct Signatures the live allocator's MAB has
    /// observed so far. 0 if in Inference mode.
    pub fn live_signature_count(&self) -> usize {
        self.live_alloc.signature_count()
    }

    /// Snapshot the live MAB's current best-backend-per-signature.
    /// Returns an empty Vec if the allocator is in Inference mode.
    pub fn live_routing_snapshot(&self) -> Vec<(u64, String)> {
        self.live_alloc
            .routing_snapshot()
            .into_iter()
            .map(|(h, b)| (h, backend_to_string(b).to_string()))
            .collect()
    }

    /// Freeze the live training allocator and store the resulting
    /// `.lohalloc` model bytes for `/api/freeze-export` download.
    /// Returns the routing table extracted from the frozen model.
    pub fn freeze_live(&self) -> Vec<(u64, String)> {
        self.live_alloc.freeze();
        let bytes = self.live_alloc.export().unwrap_or_default();
        let entries = decode_routing_entries(&bytes);
        if let Ok(mut guard) = self.last_model.lock() {
            *guard = bytes;
        }
        if let Ok(mut guard) = self.routing_table.write() {
            *guard = entries.clone();
        }
        self.set_inference(true);
        entries
    }

    /// Reset the live allocator back to fresh Training mode, discarding
    /// the frozen routing table and any live pointers. Used by the GUI's
    /// "back to training" button.
    pub fn reset_live_to_training(&self) {
        self.live_alloc.reset_to_training();
        if let Ok(mut live) = self.live_allocations.lock() {
            live.clear();
        }
        if let Ok(mut guard) = self.last_model.lock() {
            guard.clear();
        }
        if let Ok(mut guard) = self.routing_table.write() {
            guard.clear();
        }
        self.set_inference(false);
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for AppState {
    fn drop(&mut self) {
        // Only kill simulation processes when the *last* AppState clone
        // is being dropped (i.e. server shutdown).  Axum clones AppState
        // for every request via `State<AppState>`; those clones share
        // the same Arcs, so we must not kill children when a per-request
        // clone drops — that would kill every simulation immediately
        // after the handler returns.
        if Arc::strong_count(&self.simulations) > 1 {
            return;
        }
        if let Ok(mut g) = self.simulations.lock() {
            for (pid, handle) in g.iter() {
                if let Ok(mut child) = handle.child.try_lock() {
                    let child_pid = child.id();
                    tracing::info!(pid, "AppState drop: killing simulation");
                    #[cfg(unix)]
                    {
                        let _ = unsafe { libc::kill(-(child_pid as i32), libc::SIGKILL) };
                    }
                    let _ = child.kill();
                    let _ = child.wait();
                }
            }
            g.clear();
        }
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
        .route("/api/freeze", post(freeze_live_handler))
        .route("/api/reset-training", post(reset_training_handler))
        .route("/api/training-status", get(training_status_handler))
        .route(
            "/api/strategy",
            get(get_strategy_handler).post(set_strategy_handler),
        )
        .route("/api/mode", get(get_mode_handler))
        .route("/api/routing-table", get(get_routing_table_handler))
        .route("/api/telemetry", post(post_telemetry_handler))
        .route("/api/run-simulation", post(run_simulation_handler))
        .route("/api/stop-simulation/{pid}", post(stop_simulation_handler))
        .route(
            "/api/kill-all-simulations",
            post(kill_all_simulations_handler),
        )
        .route(
            "/api/simulation-history",
            get(get_simulation_history_handler),
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
    let (sender, mut receiver) = socket.split();

    // Each WS client gets its own broadcast receiver so records aren't
    // stolen by other (possibly zombie) connections. This is the fix for
    // the "records not reaching browser" bug where StrictMode remount
    // created zombie send tasks that consumed records from a shared
    // crossbeam Receiver.
    let mut ws_rx = state.ws_broadcast.subscribe();
    let mut raw_rx = state.ws_raw_broadcast.subscribe();

    // Wrap the sink so the send futures can share it.
    let sender = Arc::new(tokio::sync::Mutex::new(sender));
    let sender2 = sender.clone();

    // Future: receive telemetry records from the broadcast channel and
    // forward as JSON WS text frames. Cancelled by `select!` when the
    // client disconnects (recv_fut completes).
    let send_fut = async move {
        loop {
            match ws_rx.recv().await {
                Ok(record) => {
                    let json = serde_json::to_string(&record).unwrap_or_default();
                    let mut guard = sender.lock().await;
                    if guard.send(Message::text(json)).await.is_err() {
                        return;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                    // Client is slow — skip missed records and continue.
                    continue;
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                    return;
                }
            }
        }
    };

    // Future: receive raw (simulation event) messages from the broadcast
    // channel and forward verbatim as WS text frames.
    let raw_send_fut = async move {
        loop {
            match raw_rx.recv().await {
                Ok(msg) => {
                    let mut guard = sender2.lock().await;
                    if guard.send(Message::text(msg.0)).await.is_err() {
                        return;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                    continue;
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                    return;
                }
            }
        }
    };

    // Future: read incoming messages to detect client disconnect.
    let recv_fut = async move {
        while receiver.next().await.is_some() {
            // Ignore incoming messages — this is a server-push stream.
        }
    };

    // Select on inline futures. When any future completes, the others are
    // dropped, cancelling them. Since these are inline futures (not spawned
    // tasks), dropping them properly cancels the broadcast recv.
    tokio::select! {
        _ = send_fut => {}
        _ = raw_send_fut => {}
        _ = recv_fut => {}
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

/// `POST /api/freeze` — Freeze the live training allocator.
///
/// Collapses the live MAB's bandit weights into a frozen `PerfectHashTable`
/// and stores the resulting `.lohalloc` bytes for download via
/// `/api/freeze-export`. This is the TensorBoard-style "commit" action
/// that switches the system from live-training to inference.
///
/// Returns:
/// - `200 OK` with `{"frozen_entries": <usize>, "signatures": <usize>}`
///   on success.
async fn freeze_live_handler(State(state): State<AppState>) -> Response {
    if state.is_inference() {
        // Idempotent — already frozen. Return current state instead of
        // panicking.
        let entries = state.get_routing_table();
        return Json(serde_json::json!({
            "frozen_entries": entries.len(),
            "signatures": entries.len(),
            "already_frozen": true,
        }))
        .into_response();
    }
    let entries = state.freeze_live();
    Json(serde_json::json!({
        "frozen_entries": entries.len(),
        "signatures": entries.len(),
        "already_frozen": false,
    }))
    .into_response()
}

/// `POST /api/reset-training` — Reset the live allocator back to fresh
/// Training mode, discarding any frozen routing table and learned
/// weights. Used by the GUI's "back to training" button.
async fn reset_training_handler(State(state): State<AppState>) -> Response {
    state.reset_live_to_training();
    Json(serde_json::json!({ "mode": "training" })).into_response()
}

/// `GET /api/training-status` — Returns live-training diagnostics for the
/// GUI's convergence indicator:
///
/// - `signatures` — distinct topology hashes the MAB has observed.
/// - `live_allocations` — currently-tracked allocations awaiting free.
/// - `inference` — whether the system is in frozen mode.
async fn training_status_handler(State(state): State<AppState>) -> Response {
    Json(serde_json::json!({
        "signatures": state.live_signature_count(),
        "live_allocations": state.live_allocation_count(),
        "inference": state.is_inference(),
    }))
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

/// `GET /api/mode` — Returns the current Decision Engine mode as JSON
/// `{"mode": "training"}` or `{"mode": "inference"}`.
async fn get_mode_handler(State(state): State<AppState>) -> Response {
    let mode = if state.is_inference() {
        "inference"
    } else {
        "training"
    };
    Json(serde_json::json!({ "mode": mode })).into_response()
}

/// `GET /api/routing-table` — Returns the current routing table as a JSON
/// array of `{"hash": <u64>, "backend": <string>}` objects.
///
/// - **Inference mode**: returns the cached frozen routing table.
/// - **Training mode**: returns the live MAB snapshot (current
///   best-backend-per-signature), so the GUI can render the table as
///   it converges in real time (TensorBoard-style).
async fn get_routing_table_handler(State(state): State<AppState>) -> Response {
    let entries = if state.is_inference() {
        state.get_routing_table()
    } else {
        state.live_routing_snapshot()
    };
    let payload: Vec<serde_json::Value> = entries
        .into_iter()
        .map(|(hash, backend)| serde_json::json!({ "hash": hash, "backend": backend }))
        .collect();
    Json(payload).into_response()
}

/// `POST /api/telemetry` — Live-mode ingest endpoint (Phase 5++).
///
/// Accepts either:
/// - A single JSON `TelemetryRecord`: `{ "timestamp": ..., "op": "alloc|free", ... }`
/// - A JSON array of records: `[ {...}, {...} ]`
///
/// Each record is forwarded to the shared telemetry channel (`telemetry_tx`),
/// which the existing `/ws/telemetry` WebSocket handler drains. This means
/// the GUI's `useTelemetry()` hook, `FloatingWeb`, `TelemetrySidebar`,
/// `PolicyMatrix`, and `PerfTraceView` all receive these live records
/// without any change to their consumption logic.
///
/// Records that don't fit the bounded buffer (capacity = `DEFAULT_CAPACITY`)
/// are silently dropped, matching the replay engine's behavior — we never
/// block the producer.
///
/// Returns:
/// - `202 Accepted` with `{ "accepted": <count> }` on success.
/// - `400 Bad Request` if the body is not valid JSON or doesn't match the
///   `TelemetryRecord` schema.
/// - `415 Unsupported Media Type` if the `Content-Type` header is not
///   `application/json` (Axum's `Json` extractor enforces this).
#[tracing::instrument(level = "debug", skip_all)]
async fn post_telemetry_handler(
    State(state): State<AppState>,
    body: axum::body::Bytes,
) -> Response {
    // Peek at the first non-whitespace byte to decide single vs array.
    // This avoids the cost of deserializing to a `serde_json::Value`
    // first and re-parsing.
    let first = body.iter().find(|b| !b.is_ascii_whitespace()).copied();
    let records: Vec<lohalloc_core::TelemetryRecord> = match first {
        Some(b'[') => match serde_json::from_slice::<Vec<lohalloc_core::TelemetryRecord>>(&body) {
            Ok(rs) => rs,
            Err(e) => {
                return (StatusCode::BAD_REQUEST, format!("invalid array: {e}")).into_response();
            }
        },
        Some(_) => match serde_json::from_slice::<lohalloc_core::TelemetryRecord>(&body) {
            Ok(r) => vec![r],
            Err(e) => {
                return (StatusCode::BAD_REQUEST, format!("invalid record: {e}")).into_response();
            }
        },
        None => {
            return (StatusCode::BAD_REQUEST, "empty body").into_response();
        }
    };

    let count = records.len();
    for mut record in records {
        // Stamp a real, server-side wall-clock timestamp (ns). The shim's
        // own `timestamp` is process-relative and has been observed to
        // collapse to tiny repeating values ("0 / 41"), so we override it
        // here with a single authoritative, strictly-increasing clock —
        // this is what gives the GUI a real seconds axis for live streams.
        record.timestamp = state.next_live_timestamp();
        // TensorBoard-style: feed each record into the live training
        // allocator in real time. The MAB learns from this stream, and
        // `/api/freeze` collapses it directly without a replay.
        state.feed_live_record(&record);
        state.telemetry_tx.send(record);
    }
    tracing::debug!(count, "ingested telemetry records");
    Json(serde_json::json!({ "accepted": count })).into_response()
}

/// Request body for `POST /api/run-simulation`.
#[derive(Debug, Deserialize)]
struct RunSimulationRequest {
    /// One of `SimulationKind::ALL` (see `simulation.rs`), e.g.
    /// `"lohalloc-example"`, `"long-running"`.
    kind: String,
    /// Optional per-kind args (duration_secs for timed kinds).
    #[serde(default)]
    args: Option<SimulationArgs>,
}

/// `POST /api/run-simulation` — Spawn a real Lohalloc workload with the
/// `liblohalloc_obs` shim preloaded.
///
/// Body:
/// ```json
/// { "kind": "lohalloc-example", "args": {} }
/// ```
///
/// Returns:
/// - `202 Accepted` with `{"pid": <u32>}` on success.
/// - `400 Bad Request` on malformed JSON or unknown kind.
/// - `501 Not Implemented` on Windows (the shim is POSIX-only).
/// - `503 Service Unavailable` if the binary or shim is missing, with a
///   helpful message and build command.
///
/// Lifecycle events are streamed over `/ws/telemetry` as
/// `{"type":"simulation","event":{...}}` messages.
#[tracing::instrument(skip(state), fields(kind = %body.kind))]
async fn run_simulation_handler(
    State(state): State<AppState>,
    Json(body): Json<RunSimulationRequest>,
) -> Response {
    if !(cfg!(target_os = "macos") || cfg!(target_os = "linux")) {
        return (
            StatusCode::NOT_IMPLEMENTED,
            "simulation spawn is only supported on macOS and Linux",
        )
            .into_response();
    }

    let kind = match SimulationKind::parse(&body.kind) {
        Some(k) => k,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                format!(
                    "unknown kind '{}', expected {}",
                    body.kind,
                    SimulationKind::accepted_names()
                ),
            )
                .into_response();
        }
    };

    let shim = match simulation::find_shim_path() {
        Some(p) => p,
        None => {
            let err = simulation::missing_shim_error();
            return (StatusCode::SERVICE_UNAVAILABLE, Json(err)).into_response();
        }
    };

    if simulation::find_simulation_binary(kind).is_none() {
        let err = simulation::missing_binary_error(kind);
        return (StatusCode::SERVICE_UNAVAILABLE, Json(err)).into_response();
    }

    let args = body.args.unwrap_or_default();
    let server_port = state.get_server_port();
    if server_port == 0 {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "server port not yet known; cannot spawn simulations",
        )
            .into_response();
    }

    let child = match simulation::spawn_simulation(kind, &shim, server_port, &args) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(error = %e.message, "simulation spawn error");
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(e)).into_response();
        }
    };

    let pid = child.id();
    tracing::info!(pid, "simulation registered");
    state.register_simulation(pid, kind, child);

    // Emit the "started" event. WS clients receive it immediately.
    let started = SimulationEvent {
        pid,
        kind: kind.as_str().to_string(),
        status: "started".to_string(),
        duration_ms: 0,
        exit_code: None,
        stdout_tail: None,
        error: None,
    };
    let envelope = started.clone().into_ws_message();
    if let Ok(s) = serde_json::to_string(&envelope) {
        state.telemetry_tx.send_raw(s);
    }
    state.push_simulation_history(started);

    // Spawn a background task that polls the child and emits "exited"/"failed"
    // events when the process terminates.
    spawn_simulation_watcher(state.clone(), pid, kind, args.duration_secs);

    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({ "pid": pid, "kind": kind.as_str() })),
    )
        .into_response()
}

/// Hard ceiling for any simulation, regardless of kind. Prevents zombie
/// processes from hanging indefinitely (e.g. a `lohalloc-example` variant
/// stuck in an infinite loop).
const MAX_SIM_DURATION_SECS: u64 = 300;

/// Background task: poll a spawned simulation's exit status and emit
/// lifecycle events. Runs on the tokio runtime; uses `try_wait` so it does
/// not block other tasks. Auto-kills the process after the effective
/// deadline (the minimum of `max_duration_secs` and `MAX_SIM_DURATION_SECS`).
fn spawn_simulation_watcher(
    state: AppState,
    pid: u32,
    kind: SimulationKind,
    max_duration_secs: Option<u64>,
) {
    let span = tracing::info_span!("sim_watcher", pid, kind = kind.as_str());
    tokio::spawn(
        async move {
        // Tiny initial delay so the "started" event arrives first.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let started_at = std::time::Instant::now();

        // Effective deadline: the requested duration (if any) capped by the
        // hard ceiling. Always applies, even for kinds without a built-in
        // duration limit.
        let effective_deadline_secs = max_duration_secs
            .unwrap_or(MAX_SIM_DURATION_SECS)
            .min(MAX_SIM_DURATION_SECS);
        let effective_deadline = std::time::Duration::from_secs(effective_deadline_secs);

        loop {
            let Some((child_arc, killed_flag)) = state.get_simulation(pid) else {
                // Already cleaned up.
                return;
            };

            // Check hard timeout ceiling (applies to ALL sim kinds).
            if started_at.elapsed() >= effective_deadline {
                tracing::warn!(elapsed = ?started_at.elapsed(), "simulation duration limit reached");
                // Set the operator-kill flag so we don't emit a duplicate.
                killed_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                let mut guard = child_arc.lock().await;
                let child_pid = guard.id();
                #[cfg(unix)]
                {
                    let _ = unsafe { libc::kill(-(child_pid as i32), libc::SIGKILL) };
                }
                let _ = guard.kill();
                drop(guard);

                let duration_ms = started_at.elapsed().as_millis() as u64;
                let ev = SimulationEvent {
                    pid,
                    kind: kind.as_str().to_string(),
                    status: "exited".to_string(),
                    duration_ms,
                    exit_code: Some(0),
                    stdout_tail: None,
                    error: Some("duration limit reached".to_string()),
                };
                let envelope = ev.clone().into_ws_message();
                if let Ok(s) = serde_json::to_string(&envelope) {
                    state.telemetry_tx.send_raw(s);
                }
                state.push_simulation_history(ev);
                state.unregister_simulation(pid);
                return;
            }

            // Poll once.
            let exit_status = {
                let mut guard = child_arc.lock().await;
                guard.try_wait().ok().flatten()
            };

            if let Some(status) = exit_status {
                tracing::info!(success = status.success(), "simulation process exited");
                // If the operator already killed this sim (via
                // `/api/stop-simulation`), the kill handler already emitted
                // a "failed" event. Don't emit a duplicate.
                if killed_flag.load(std::sync::atomic::Ordering::Relaxed) {
                    tracing::debug!("killed by operator, skipping duplicate event");
                    return;
                }

                let duration_ms = started_at.elapsed().as_millis() as u64;
                let exit_code = status.code();
                let stdout_tail = read_child_stdout_tail(&child_arc).await;
                let status_str = if status.success() { "exited" } else { "failed" };
                let ev = SimulationEvent {
                    pid,
                    kind: kind.as_str().to_string(),
                    status: status_str.to_string(),
                    duration_ms,
                    exit_code,
                    stdout_tail: stdout_tail.clone(),
                    error: if status.success() {
                        None
                    } else {
                        Some(format!(
                            "process exited with code {}",
                            exit_code.unwrap_or(-1)
                        ))
                    },
                };
                let envelope = ev.clone().into_ws_message();
                if let Ok(s) = serde_json::to_string(&envelope) {
                    state.telemetry_tx.send_raw(s);
                }
                state.push_simulation_history(ev);
                state.unregister_simulation(pid);
                return;
            }

            // Emit a "running" heartbeat every 500ms so the GUI can update
            // elapsed time without polling the WS.
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            if state.get_simulation(pid).is_none() {
                return;
            }
            let duration_ms = started_at.elapsed().as_millis() as u64;
            let ev = SimulationEvent {
                pid,
                kind: kind.as_str().to_string(),
                status: "running".to_string(),
                duration_ms,
                exit_code: None,
                stdout_tail: None,
                error: None,
            };
            let envelope = ev.into_ws_message();
            if let Ok(s) = serde_json::to_string(&envelope) {
                state.telemetry_tx.send_raw(s);
            }
        }
        }
        .instrument(span),
    );
}

/// Read up to 4 KiB of the child's stdout tail. Best-effort: returns
/// `None` on any error or if the child has no captured stdout. Uses
/// blocking I/O because `std::process::ChildStdout` is not async-aware.
async fn read_child_stdout_tail(
    child_arc: &Arc<tokio::sync::Mutex<std::process::Child>>,
) -> Option<String> {
    let mut guard = child_arc.lock().await;
    let stdout = guard.stdout.take()?;
    // Move the blocking read off the async runtime.
    tokio::task::spawn_blocking(move || {
        use std::io::Read;
        let mut stdout = stdout;
        let mut buf = vec![0u8; 4096];
        let n = match stdout.read(&mut buf) {
            Ok(n) => n,
            Err(_) => return None,
        };
        if n == 0 {
            return None;
        }
        buf.truncate(n);
        Some(String::from_utf8_lossy(&buf).into_owned())
    })
    .await
    .ok()
    .flatten()
}

/// `GET /api/simulation-history` — Returns the last `SIMULATION_HISTORY_CAP`
/// simulation lifecycle events plus the currently-active set. Used by the
/// GUI's `SimulationPanel` to populate on initial page load.
async fn get_simulation_history_handler(State(state): State<AppState>) -> Response {
    let mut events = state.snapshot_active_simulations();
    let mut history = state.snapshot_simulation_history();
    events.append(&mut history);
    Json(serde_json::json!({ "events": events })).into_response()
}

/// `POST /api/stop-simulation/:pid` — Kill a running simulation by pid.
/// Sends SIGKILL to the child process and emits a "failed" lifecycle event.
async fn stop_simulation_handler(State(state): State<AppState>, Path(pid): Path<u32>) -> Response {
    let kind = state.kill_simulation(pid).await;
    match kind {
        Some(k) => {
            let ev = SimulationEvent {
                pid,
                kind: k.as_str().to_string(),
                status: "failed".to_string(),
                duration_ms: 0,
                exit_code: Some(-1),
                stdout_tail: None,
                error: Some("killed by operator".to_string()),
            };
            let envelope = ev.clone().into_ws_message();
            if let Ok(s) = serde_json::to_string(&envelope) {
                state.telemetry_tx.send_raw(s);
            }
            state.push_simulation_history(ev);
            Json(serde_json::json!({ "pid": pid, "killed": true })).into_response()
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "simulation not found", "pid": pid })),
        )
            .into_response(),
    }
}

/// `POST /api/kill-all-simulations` — Kill ALL running simulations.
/// Emergency-stop button. Sends SIGKILL to every active child process.
/// Returns `{ "killed": N }` where N is the number of simulations killed.
async fn kill_all_simulations_handler(State(state): State<AppState>) -> Response {
    let killed = state.kill_all_simulations().await;

    // Emit a "failed" lifecycle event for each killed sim.
    // Since kill_all_simulations already unregistered them, we can't
    // retrieve their kinds — but the individual kill_simulation calls
    // would have emitted events if we tracked them. Instead, we emit
    // a single summary event with pid=0.
    if killed > 0 {
        let ev = SimulationEvent {
            pid: 0,
            kind: "all".to_string(),
            status: "failed".to_string(),
            duration_ms: 0,
            exit_code: Some(-1),
            stdout_tail: None,
            error: Some(format!("kill-all: terminated {} simulation(s)", killed)),
        };
        let envelope = ev.clone().into_ws_message();
        if let Ok(s) = serde_json::to_string(&envelope) {
            state.telemetry_tx.send_raw(s);
        }
        state.push_simulation_history(ev);
    }

    Json(serde_json::json!({ "killed": killed })).into_response()
}

// ---------------------------------------------------------------------------
// Routing-table extraction
// ---------------------------------------------------------------------------

/// Decode `(hash, backend)` entries from a frozen `.lohalloc` model's binary
/// representation.
///
/// The format is owned by `lohalloc_alloc::perfect_hash::PerfectHashTable`:
///
/// ```text
/// [8]    magic    "loha11oc" (LE u64)
/// [4]    version  (LE u32)
/// [4]    count    (LE u32)
/// [N×12] entries: (hash: u64 LE, backend: u8, _pad: [u8; 3])
/// [8]    checksum (XOR of all hashes, LE u64)
/// ```
///
/// Returns an empty `Vec` if `bytes` is too short, has bad magic/version,
/// or fails checksum validation. This is best-effort decoding — the
/// authoritative parser lives in `lohalloc-alloc`; this duplicate exists
/// so the server crate doesn't need to keep a private `Lohalloc` instance
/// alive purely to enumerate entries.
fn decode_routing_entries(bytes: &[u8]) -> Vec<(u64, String)> {
    // magic(8) + version(4) + count(4) + checksum(8) = 24
    if bytes.len() < 24 {
        return Vec::new();
    }
    let mut pos = 0;

    let magic = read_u64_le(bytes, &mut pos);
    // Magic: "COLLAHOL" = 0x434f4c4c41484f4c (matches `lohalloc_alloc::perfect_hash::MAGIC`).
    if magic != 0x434f_4c4c_4148_4f4c {
        return Vec::new();
    }

    let version = read_u32_le(bytes, &mut pos);
    if version != 1 {
        return Vec::new();
    }

    let count = read_u32_le(bytes, &mut pos) as usize;
    let expected_len = 16 + count * 12 + 8;
    if bytes.len() < expected_len {
        return Vec::new();
    }

    let mut entries = Vec::with_capacity(count);
    let mut checksum: u64 = 0;
    for _ in 0..count {
        let hash = read_u64_le(bytes, &mut pos);
        let backend_byte = *bytes.get(pos).unwrap_or(&0);
        pos += 4; // backend(1) + padding(3)
        let backend = match backend_byte {
            0 => "slab",
            1 => "buddy",
            2 => "system",
            3 => "arena",
            _ => return Vec::new(),
        };
        entries.push((hash, backend.to_string()));
        checksum ^= hash;
    }

    let stored_checksum = read_u64_le(bytes, &mut pos);
    if stored_checksum != checksum {
        return Vec::new();
    }

    entries
}

fn read_u64_le(bytes: &[u8], pos: &mut usize) -> u64 {
    let end = *pos + 8;
    let slice = bytes.get(*pos..end).unwrap_or(&[0u8; 8]);
    *pos = end;
    let mut buf = [0u8; 8];
    buf.copy_from_slice(slice);
    u64::from_le_bytes(buf)
}

fn read_u32_le(bytes: &[u8], pos: &mut usize) -> u32 {
    let end = *pos + 4;
    let slice = bytes.get(*pos..end).unwrap_or(&[0u8; 4]);
    *pos = end;
    let mut buf = [0u8; 4];
    buf.copy_from_slice(slice);
    u32::from_le_bytes(buf)
}

/// Convert a `lohalloc_core::Backend` enum to its lowercase wire name
/// (matches the `rename_all = "snake_case"` setting used by the shim and
/// GUI). Keeping this in one place avoids drift between server and
/// client string formats.
fn backend_to_string(backend: lohalloc_core::Backend) -> &'static str {
    use lohalloc_core::Backend;
    match backend {
        Backend::Slab => "slab",
        Backend::Buddy => "buddy",
        Backend::System => "system",
        Backend::Arena => "arena",
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
        let body = r#"[{"timestamp":0,"op":"alloc","size":64,"stack_hash":100},{"timestamp":1500000,"op":"free","size":64,"stack_hash":100}]"#;
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
                    .body(Body::from(r#"[{"timestamp":0,"op":"alloc","size":64,"stack_hash":100}]"#))
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
            .body(r#"[{"timestamp":0,"op":"alloc","size":64,"stack_hash":100},{"timestamp":1500000,"op":"free","size":64,"stack_hash":100}]"#)
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
        let body = r#"[{"timestamp":0,"op":"alloc","size":64,"stack_hash":100},{"timestamp":1500000,"op":"free","size":64,"stack_hash":100}]"#;
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
        let body = r#"[{"timestamp":0,"op":"alloc","size":64,"stack_hash":100}]"#;
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

        let body = r#"[{"timestamp":0,"op":"alloc","size":64,"stack_hash":100},{"timestamp":1500000,"op":"free","size":64,"stack_hash":100}]"#;
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

    // -----------------------------------------------------------------------
    // Phase 5: mode + routing-table endpoints
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_mode_endpoint() {
        let app = build_app(AppState::new());
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/mode")
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
        assert_eq!(v["mode"], "training");
    }

    #[tokio::test]
    async fn test_routing_table_empty_initially() {
        let app = build_app(AppState::new());
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/routing-table")
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
        assert!(v.is_array(), "routing-table should be a JSON array");
        assert_eq!(v.as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_strategy_change_resets_mode() {
        let state = AppState::new();
        let app = build_app(state.clone());

        // Force the allocator into Inference mode (as if freeze-export had run).
        state.set_inference(true);

        // Sanity check: mode endpoint now reports "inference".
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/mode")
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
        assert_eq!(v["mode"], "inference");

        // POST /api/strategy — should revert the mode to training.
        let response = app
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
        assert_eq!(response.status(), StatusCode::OK);

        // Mode must now be back to "training".
        assert!(!state.is_inference(), "mode should revert to training");
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/mode")
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
        assert_eq!(v["mode"], "training");
    }

    // -----------------------------------------------------------------------
    // Phase 5++: POST /api/telemetry (live mode ingest)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn post_telemetry_single_record_returns_202() {
        let app = build_app(AppState::new());
        // Backend serializes as snake_case ("slab", "buddy", etc.).
        // stack_hash / result_ptr / timestamp are u64 — must be JSON numbers.
        let body = r#"{
            "timestamp": 12345,
            "op": "alloc",
            "size": 64,
            "stack_hash": 9876543210,
            "thread_id": 0,
            "result_ptr": "0x1000",
            "latency_ns": 100,
            "fragmentation_pct": 0.0,
            "backend": "slab"
        }"#;
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/telemetry")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let resp_body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&resp_body).unwrap();
        assert_eq!(v["accepted"], 1);
    }

    #[tokio::test]
    async fn post_telemetry_batch_returns_202() {
        let app = build_app(AppState::new());
        let body = r#"[
            {"timestamp":1,"op":"alloc","size":64,"stack_hash":100,"thread_id":0,"result_ptr":"0x1000","latency_ns":50,"fragmentation_pct":0.0,"backend":"slab"},
            {"timestamp":2,"op":"alloc","size":128,"stack_hash":101,"thread_id":0,"result_ptr":"0x2000","latency_ns":75,"fragmentation_pct":0.0,"backend":"buddy"},
            {"timestamp":3,"op":"free","size":64,"stack_hash":100,"thread_id":0,"result_ptr":"0x1000","latency_ns":20,"fragmentation_pct":0.0}
        ]"#;
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/telemetry")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let resp_body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&resp_body).unwrap();
        assert_eq!(v["accepted"], 3);
    }

    #[test]
    fn kill_all_simulations_blocking_is_safe_when_empty() {
        // The panic-hook / signal path must never panic or block on an empty
        // registry — it just reports zero groups signalled.
        let state = AppState::new();
        assert_eq!(state.kill_all_simulations_blocking(), 0);
    }

    #[test]
    fn next_live_timestamp_is_strictly_increasing_and_precision_safe() {
        // A burst of assignments (same clock tick) must be strictly
        // increasing so the GUI axis never collapses, and the values must
        // stay under 2^53 so they round-trip through a JS `number` losslessly
        // (the fix for the idling CSV time axis). Epoch-relative ns keeps them
        // small; 2^53 ns is ~104 days of uptime.
        let state = AppState::new();
        let mut prev = 0u64;
        for i in 0..10_000 {
            let ts = state.next_live_timestamp();
            assert!(
                ts > prev,
                "timestamp #{i} not strictly increasing: {ts} <= {prev}"
            );
            assert!(
                ts < (1u64 << 53),
                "timestamp #{i} exceeds JS-safe integer range: {ts}"
            );
            prev = ts;
        }
    }

    #[tokio::test]
    async fn post_telemetry_overrides_degenerate_timestamp_on_ws() {
        // End-to-end: a record POSTed with a degenerate "0 / 41"-style
        // timestamp must arrive on the WS with a real, rebased ns value.
        let state = AppState::new();
        let app = build_app(state.clone());

        let listener = tokio::net::TcpListener::bind(SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0)))
            .await
            .unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(axum::serve(listener, app).into_future());

        let (mut socket, _response) =
            tokio_tungstenite::connect_async(format!("ws://{addr}/ws/telemetry"))
                .await
                .unwrap();

        let client = reqwest::Client::new();
        // Both records carry timestamp 41 — exactly the degenerate,
        // non-advancing value the shim reported. The server must replace them
        // with its own strictly-increasing, JS-precision-safe clock, so the
        // two WS records must NOT both be 41 and must advance.
        for _ in 0..2 {
            let body = r#"{"timestamp":41,"op":"alloc","size":64,"stack_hash":1,"thread_id":0,"result_ptr":"0x10","latency_ns":0,"fragmentation_pct":0.0,"backend":"slab"}"#;
            let response = client
                .post(format!("http://{addr}/api/telemetry"))
                .header(header::CONTENT_TYPE, "application/json")
                .body(body)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), 200);
        }

        let mut timestamps = Vec::new();
        while timestamps.len() < 2 {
            let msg = tokio::time::timeout(std::time::Duration::from_secs(5), socket.next())
                .await
                .expect("timed out waiting for WS message")
                .expect("stream ended")
                .expect("WS error");
            let v: serde_json::Value = serde_json::from_str(&msg.into_text().unwrap()).unwrap();
            timestamps.push(v["timestamp"].as_u64().expect("timestamp is u64"));
        }
        let (first, second) = (timestamps[0], timestamps[1]);

        assert_ne!(first, 41, "server echoed the degenerate timestamp");
        assert!(
            second > first,
            "timestamps not strictly increasing: {second} <= {first}"
        );
        // Precision-safe: must round-trip through a JS `number` losslessly.
        assert!(second < (1u64 << 53), "timestamp exceeds JS-safe range: {second}");
    }

    #[tokio::test]
    async fn post_telemetry_malformed_returns_400() {
        let app = build_app(AppState::new());
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/telemetry")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(r#"{"not_a_record": true}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn post_telemetry_empty_returns_400() {
        let app = build_app(AppState::new());
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/telemetry")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(""))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn post_telemetry_arrives_on_ws_stream() {
        // End-to-end: POST a record → verify it shows up on /ws/telemetry.
        let state = AppState::new();
        let app = build_app(state.clone());

        let listener = tokio::net::TcpListener::bind(SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0)))
            .await
            .unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(axum::serve(listener, app).into_future());

        // Connect to the WebSocket.
        let (mut socket, _response) =
            tokio_tungstenite::connect_async(format!("ws://{addr}/ws/telemetry"))
                .await
                .unwrap();

        // POST a single record.
        let client = reqwest::Client::new();
        let body = r#"{
            "timestamp": 99999,
            "op": "alloc",
            "size": 256,
            "stack_hash": 3203385166,
            "thread_id": 7,
            "result_ptr": "0xDEADBEEF",
            "latency_ns": 333,
            "fragmentation_pct": 0.5,
            "backend": "slab"
        }"#;
        let response = client
            .post(format!("http://{addr}/api/telemetry"))
            .header(header::CONTENT_TYPE, "application/json")
            .body(body)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);

        // Verify it appears on the WebSocket.
        let msg = tokio::time::timeout(std::time::Duration::from_secs(5), socket.next())
            .await
            .expect("timed out waiting for WS message")
            .expect("stream ended")
            .expect("WS error");
        let text = msg.into_text().unwrap();
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["size"], 256);
        // result_ptr is serialized as "0x..." hex string by the
        // `serialize_ptr` adapter on TelemetryRecord.
        assert_eq!(v["result_ptr"], "0xdeadbeef");
        assert_eq!(v["op"], "alloc");
        assert_eq!(v["backend"], "slab");
    }

    #[tokio::test]
    async fn post_telemetry_batch_arrives_on_ws_stream() {
        let state = AppState::new();
        let app = build_app(state.clone());

        let listener = tokio::net::TcpListener::bind(SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0)))
            .await
            .unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(axum::serve(listener, app).into_future());

        let (mut socket, _response) =
            tokio_tungstenite::connect_async(format!("ws://{addr}/ws/telemetry"))
                .await
                .unwrap();

        let client = reqwest::Client::new();
        let body = r#"[
            {"timestamp":1,"op":"alloc","size":16,"stack_hash":1,"thread_id":0,"result_ptr":"0x10","latency_ns":10,"fragmentation_pct":0.0,"backend":"slab"},
            {"timestamp":2,"op":"alloc","size":32,"stack_hash":2,"thread_id":0,"result_ptr":"0x20","latency_ns":20,"fragmentation_pct":0.0,"backend":"slab"}
        ]"#;
        let response = client
            .post(format!("http://{addr}/api/telemetry"))
            .header(header::CONTENT_TYPE, "application/json")
            .body(body)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);

        // Drain two messages from the WS.
        let mut received = 0;
        while received < 2 {
            let msg = tokio::time::timeout(std::time::Duration::from_secs(5), socket.next())
                .await
                .expect("timeout")
                .expect("stream ended")
                .expect("WS error");
            let text = msg.into_text().unwrap();
            let v: serde_json::Value = serde_json::from_str(&text).unwrap();
            assert!(v.get("op").is_some());
            received += 1;
        }
        assert_eq!(received, 2);
    }

    // -----------------------------------------------------------------
    // Live training allocator tests (TensorBoard-style)
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn training_status_starts_in_training_with_zero_sigs() {
        let app = build_app(AppState::new());
        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/training-status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["signatures"], 0);
        assert_eq!(v["live_allocations"], 0);
        assert_eq!(v["inference"], false);
    }

    #[tokio::test]
    async fn post_telemetry_updates_live_signature_count() {
        let app = build_app(AppState::new());
        // Send two allocations with different stack_hashes.
        let body = r#"[
            {"timestamp":1,"op":"alloc","size":64,"stack_hash":100,"thread_id":0,"result_ptr":"0x1000","latency_ns":1,"fragmentation_pct":0.0,"backend":"slab"},
            {"timestamp":2,"op":"alloc","size":128,"stack_hash":200,"thread_id":0,"result_ptr":"0x2000","latency_ns":1,"fragmentation_pct":0.0,"backend":"slab"}
        ]"#;
        let _ = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/telemetry")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        // Status should reflect 2 distinct signatures.
        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/training-status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["signatures"], 2);
        assert_eq!(v["live_allocations"], 2);
        assert_eq!(v["inference"], false);
    }

    #[tokio::test]
    async fn routing_table_returns_live_snapshot_in_training_mode() {
        let app = build_app(AppState::new());
        let body = r#"[
            {"timestamp":1,"op":"alloc","size":64,"stack_hash":100,"thread_id":0,"result_ptr":"0x1000","latency_ns":1,"fragmentation_pct":0.0,"backend":"slab"},
            {"timestamp":2,"op":"alloc","size":128,"stack_hash":200,"thread_id":0,"result_ptr":"0x2000","latency_ns":1,"fragmentation_pct":0.0,"backend":"slab"}
        ]"#;
        let _ = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/telemetry")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        // /api/routing-table in training mode returns the live snapshot
        // (not the empty Vec that the inference path returns).
        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/routing-table")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        let mut hashes: Vec<u64> = arr.iter().map(|e| e["hash"].as_u64().unwrap()).collect();
        hashes.sort();
        assert_eq!(hashes, vec![100, 200]);
        // Backend should be a valid string.
        for entry in arr {
            assert!(entry["backend"].is_string());
        }
    }

    #[tokio::test]
    async fn freeze_live_switches_to_inference_and_persists_model() {
        let app = build_app(AppState::new());
        let body = r#"[
            {"timestamp":1,"op":"alloc","size":64,"stack_hash":100,"thread_id":0,"result_ptr":"0x1000","latency_ns":1,"fragmentation_pct":0.0,"backend":"slab"},
            {"timestamp":2,"op":"alloc","size":128,"stack_hash":200,"thread_id":0,"result_ptr":"0x2000","latency_ns":1,"fragmentation_pct":0.0,"backend":"slab"}
        ]"#;
        let _ = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/telemetry")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        // Freeze the live allocator.
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/freeze")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["already_frozen"], false);
        assert!(v["frozen_entries"].as_u64().unwrap() >= 2);

        // Mode should now be inference.
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/mode")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["mode"], "inference");

        // /api/freeze-export should now return non-empty .lohalloc bytes.
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
            "model bytes should be present after freeze"
        );
    }

    #[tokio::test]
    async fn freeze_live_is_idempotent_when_already_frozen() {
        let app = build_app(AppState::new());
        let body = r#"[{"timestamp":1,"op":"alloc","size":64,"stack_hash":100,"thread_id":0,"result_ptr":"0x1000","latency_ns":1,"fragmentation_pct":0.0,"backend":"slab"}]"#;
        let _ = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/telemetry")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        // Freeze twice — second call should NOT panic.
        let _ = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/freeze")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/freeze")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["already_frozen"], true);
    }

    #[tokio::test]
    async fn reset_training_returns_to_training_mode_and_clears_model() {
        let app = build_app(AppState::new());
        let body = r#"[{"timestamp":1,"op":"alloc","size":64,"stack_hash":100,"thread_id":0,"result_ptr":"0x1000","latency_ns":1,"fragmentation_pct":0.0,"backend":"slab"}]"#;
        let _ = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/telemetry")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        // Freeze first.
        let _ = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/freeze")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        // Reset.
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/reset-training")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        // Mode should now be training again.
        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/mode")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["mode"], "training");
    }

    #[tokio::test]
    async fn kill_all_simulations_returns_zero_when_empty() {
        let state = AppState::new();
        let killed = state.kill_all_simulations().await;
        assert_eq!(killed, 0);
    }

    #[tokio::test]
    async fn kill_all_simulations_endpoint_returns_killed_count() {
        let app = build_app(AppState::new());
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/kill-all-simulations")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["killed"], 0);
    }
}
