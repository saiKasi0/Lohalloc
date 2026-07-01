//! Server-side subprocess runner for live Lohalloc simulations.
//!
//! Spawns real Lohalloc workloads (the `lohalloc-example` smoke binary, an
//! additional `lohalloc-server` instance, or a long-running curl loop) with
//! the `liblohalloc_obs` shim preloaded so their `malloc`/`free` traffic is
//! streamed back to this server via `POST /api/telemetry`.
//!
//! # Binary discovery
//!
//! [`find_simulation_binary`] checks in order:
//! 1. `LOHALLOC_BIN_<UPPER_NAME>` environment variable
//! 2. `target/debug/<name>` and `target/release/<name>` relative to CWD
//! 3. `../target/*/<name>` (workspace-relative)
//!
//! [`find_shim_path`] checks similarly:
//! 1. `LOHALLOC_SHIM_PATH` environment variable
//! 2. `shim/build/liblohalloc_obs.{so,dylib}` relative to CWD
//! 3. `../shim/build/liblohalloc_obs.{so,dylib}`
//!
//! # Security
//!
//! The HTTP handler rejects requests from non-loopback IPs unless
//! `LOHALLOC_ALLOW_REMOTE_SPAWN=1` is set in the environment. The server
//! itself binds to `127.0.0.1` by default.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

/// The three built-in simulation kinds the GUI exposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SimulationKind {
    /// The `lohalloc-example` smoke binary (`cargo run -p lohalloc-example`).
    LohallocExample,
    /// A second `lohalloc-server` instance under the shim, demonstrating
    /// an HTTP workload.
    HttpServer,
    /// A long-running curl loop that hits `/api/mode` repeatedly to keep
    /// the allocator hot.
    LongRunning,
}

impl SimulationKind {
    /// Parse a string into a [`SimulationKind`].
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "lohalloc-example" => Some(Self::LohallocExample),
            "http-server" => Some(Self::HttpServer),
            "long-running" => Some(Self::LongRunning),
            _ => None,
        }
    }

    /// Human-friendly name used in `SimulationEvent`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LohallocExample => "lohalloc-example",
            Self::HttpServer => "http-server",
            Self::LongRunning => "long-running",
        }
    }
}

/// Optional per-kind arguments.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SimulationArgs {
    /// Port for the `http-server` kind (defaults to 4000). Ignored by other
    /// kinds.
    #[serde(default)]
    pub port: Option<u16>,
    /// Duration in seconds. Used by `long-running` (defaults to 60) and
    /// `lohalloc-example` (loops the workload until duration elapses).
    /// Ignored by `http-server`.
    #[serde(default)]
    pub duration_secs: Option<u64>,
}

/// Lifecycle event published over `/ws/telemetry` to the GUI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationEvent {
    /// Process ID, or `0` if the spawn failed before fork.
    pub pid: u32,
    /// The kind of simulation.
    pub kind: String,
    /// Lifecycle phase: `started`, `running`, `exited`, `failed`.
    pub status: String,
    /// Wall-clock duration so far (ms).
    pub duration_ms: u64,
    /// Exit code, if the process has exited.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    /// Last 4 KiB of stdout, for `failed` / `exited` events.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stdout_tail: Option<String>,
    /// Error message, for `failed` events.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl SimulationEvent {
    /// Wrap this event in the WS message envelope used by `handle_telemetry_socket`.
    pub fn into_ws_message(self) -> serde_json::Value {
        serde_json::json!({ "type": "simulation", "event": self })
    }
}

/// Errors returned by [`spawn_simulation`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationError {
    pub code: &'static str,
    pub message: String,
}

impl SimulationError {
    fn missing_binary(kind: SimulationKind) -> Self {
        let name = kind.binary_name();
        let env_key = kind.env_override_key();
        Self {
            code: "BINARY_NOT_FOUND",
            message: format!(
                "could not find `{}`. Searched: ${} env var, target/debug/{name}, \
                 target/release/{name}, ../target/*/{{debug,release}}/{name}, \
                 ../../target/*/{{debug,release}}/{name}. \
                 Build it with: cargo build --release -p {}",
                name,
                env_key,
                kind.crate_name()
            ),
        }
    }

    fn missing_shim() -> Self {
        Self {
            code: "SHIM_NOT_FOUND",
            message: "could not find `liblohalloc_obs`. Build it with: cd shim && make".to_string(),
        }
    }
}

impl std::fmt::Display for SimulationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl SimulationKind {
    fn crate_name(self) -> &'static str {
        match self {
            Self::LohallocExample => "lohalloc-example",
            Self::HttpServer => "lohalloc-server",
            Self::LongRunning => "lohalloc-example", // reuses the example binary
        }
    }

    fn binary_name(self) -> &'static str {
        match self {
            Self::LohallocExample => "lohalloc-example",
            Self::HttpServer => "lohalloc-server",
            Self::LongRunning => "lohalloc-example", // reuses the example binary
        }
    }

    fn env_override_key(self) -> String {
        // e.g. LOHALLOC_BIN_LOHALLOC_EXAMPLE
        format!(
            "LOHALLOC_BIN_{}",
            self.binary_name().to_uppercase().replace('-', "_")
        )
    }
}

/// Locate a simulation binary on disk. Returns `None` if not found.
pub fn find_simulation_binary(kind: SimulationKind) -> Option<PathBuf> {
    let name = kind.binary_name();

    // 1. Explicit env override.
    if let Ok(p) = std::env::var(kind.env_override_key()) {
        let pb = PathBuf::from(p);
        if pb.is_file() {
            return Some(pb);
        }
    }

    // 2. target/{debug,release}/<name> relative to CWD.
    for profile in &["debug", "release"] {
        let pb = PathBuf::from("target").join(profile).join(name);
        if pb.is_file() {
            return Some(pb);
        }
    }

    // 3. ../target/*/<name> (workspace-relative).
    if let Ok(cwd) = std::env::current_dir() {
        let parent = cwd.parent();
        if let Some(parent) = parent {
            for profile in &["debug", "release"] {
                let pb = parent.join("target").join(profile).join(name);
                if pb.is_file() {
                    return Some(pb);
                }
            }
            // Also check sibling workspace target dirs (e.g. server running
            // from crates/lohalloc-server -> ../../target).
            if let Some(grandparent) = parent.parent() {
                for profile in &["debug", "release"] {
                    let pb = grandparent.join("target").join(profile).join(name);
                    if pb.is_file() {
                        return Some(pb);
                    }
                }
            }
        }
    }

    None
}

/// Locate the compiled `liblohalloc_obs` shim. Returns `None` if not built.
pub fn find_shim_path() -> Option<PathBuf> {
    let candidates: &[&str] = if cfg!(target_os = "macos") {
        &["shim/build/liblohalloc_obs.dylib"]
    } else {
        &["shim/build/liblohalloc_obs.so"]
    };

    // 1. Explicit env override.
    if let Ok(p) = std::env::var("LOHALLOC_SHIM_PATH") {
        let pb = PathBuf::from(p);
        if pb.is_file() {
            return Some(pb);
        }
    }

    // 2. Relative to CWD.
    for c in candidates {
        let pb = PathBuf::from(c);
        if pb.is_file() {
            return Some(pb);
        }
    }

    // 3. ../shim/build/... (workspace-relative).
    if let Ok(cwd) = std::env::current_dir() {
        if let Some(parent) = cwd.parent() {
            for c in candidates {
                let pb = parent.join(c);
                if pb.is_file() {
                    return Some(pb);
                }
            }
            if let Some(grandparent) = parent.parent() {
                for c in candidates {
                    let pb = grandparent.join(c);
                    if pb.is_file() {
                        return Some(pb);
                    }
                }
            }
        }
    }

    None
}

/// Monotonic counter used to generate unique `SimulationEvent::pid` values
/// even when the real OS pid has not yet been allocated (spawn failures).
static NEXT_FAKE_PID: AtomicU64 = AtomicU64::new(1);

/// Build a [`Command`] for the given kind, with the shim preloaded and
/// `LOHALLOC_OBS_PORT` pointing at `server_port`. Does not spawn.
pub fn build_command(
    kind: SimulationKind,
    shim: &std::path::Path,
    server_port: u16,
    args: &SimulationArgs,
) -> Result<Command, SimulationError> {
    let binary =
        find_simulation_binary(kind).ok_or_else(|| SimulationError::missing_binary(kind))?;

    let mut cmd = match kind {
        SimulationKind::LohallocExample => {
            let mut c = Command::new(&binary);
            c.arg("--diverse");
            if let Some(d) = args.duration_secs {
                c.arg("--duration-secs").arg(d.to_string());
            }
            c
        }
        SimulationKind::HttpServer => {
            let mut c = Command::new(&binary);
            let port = args.port.unwrap_or(4000);
            // lohalloc-server uses LOHALLOC_ADDR env var, not CLI args.
            c.env("LOHALLOC_ADDR", format!("127.0.0.1:{}", port));
            c
        }
        SimulationKind::LongRunning => {
            // Run lohalloc-example with --diverse and --duration-secs to
            // generate real allocation traffic under the shim for the
            // requested duration.
            let mut c = Command::new(&binary);
            c.arg("--diverse");
            let duration = args.duration_secs.unwrap_or(60);
            c.arg("--duration-secs").arg(duration.to_string());
            c
        }
    };

    // Inject the shim into the subprocess env.
    inject_shim(&mut cmd, shim);

    // Point the shim at the parent server so its telemetry POSTs land here.
    cmd.env("LOHALLOC_OBS_PORT", server_port.to_string());

    // Pipe stdout/stderr so we can tail them if the process fails.
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    Ok(cmd)
}

/// Set `LD_PRELOAD` (Linux) or `DYLD_INSERT_LIBRARIES` (macOS) on `cmd`.
fn inject_shim(cmd: &mut Command, shim: &std::path::Path) {
    let shim_str = shim.to_string_lossy().into_owned();
    if cfg!(target_os = "macos") {
        cmd.env("DYLD_INSERT_LIBRARIES", &shim_str);
        // macOS SIP disables DYLD_INSERT_LIBRARIES for protected binaries;
        // most workloads work but some may not. We do not error out here
        // — the OS will report it on `spawn()`.
    } else if cfg!(target_os = "linux") {
        cmd.env("LD_PRELOAD", &shim_str);
    } else {
        // Windows: not supported. The handler returns 501 before reaching here.
    }
}

/// Spawn the configured simulation. Returns a `Child` handle on success.
pub fn spawn_simulation(
    kind: SimulationKind,
    shim: &std::path::Path,
    server_port: u16,
    args: &SimulationArgs,
) -> Result<Child, SimulationError> {
    let mut cmd = build_command(kind, shim, server_port, args)?;
    let start = std::time::Instant::now();
    match cmd.spawn() {
        Ok(child) => Ok(child),
        Err(e) => {
            // Synthesize a fake pid so the failure event is still unique.
            let _fake = NEXT_FAKE_PID.fetch_add(1, Ordering::Relaxed);
            let _ = start; // duration captured by the caller
            Err(SimulationError {
                code: "SPAWN_FAILED",
                message: format!("failed to spawn `{}`: {}", kind.binary_name(), e),
            })
        }
    }
}

/// Return a friendly "missing binary" error for the given kind.
pub fn missing_binary_error(kind: SimulationKind) -> SimulationError {
    SimulationError::missing_binary(kind)
}

/// Return a friendly "missing shim" error.
pub fn missing_shim_error() -> SimulationError {
    SimulationError::missing_shim()
}

/// True if the server should accept spawn requests from non-loopback IPs.
/// Controlled by the `LOHALLOC_ALLOW_REMOTE_SPAWN` env var.
pub fn allow_remote_spawn() -> bool {
    std::env::var("LOHALLOC_ALLOW_REMOTE_SPAWN")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Check whether a `SocketAddr` is loopback (127.0.0.1 / ::1).
pub fn is_loopback(addr: &std::net::SocketAddr) -> bool {
    addr.ip().is_loopback()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_parse_roundtrip() {
        for k in [
            SimulationKind::LohallocExample,
            SimulationKind::HttpServer,
            SimulationKind::LongRunning,
        ] {
            assert_eq!(SimulationKind::parse(k.as_str()), Some(k));
        }
        assert_eq!(SimulationKind::parse("nope"), None);
    }

    #[test]
    fn allow_remote_spawn_default_false() {
        // We don't actually clear the env (other tests may set it), but the
        // default must be `false` if unset. The function reads via env var;
        // if the test runner has it set this assertion is a no-op.
        if std::env::var("LOHALLOC_ALLOW_REMOTE_SPAWN").is_err() {
            assert!(!allow_remote_spawn());
        }
    }

    #[test]
    fn is_loopback_for_v4_and_v6() {
        assert!(is_loopback(&"127.0.0.1:3000".parse().unwrap()));
        assert!(is_loopback(&"[::1]:3000".parse().unwrap()));
        assert!(!is_loopback(&"10.0.0.1:3000".parse().unwrap()));
    }

    #[test]
    fn missing_binary_error_mentions_kind() {
        let e = missing_binary_error(SimulationKind::LohallocExample);
        assert_eq!(e.code, "BINARY_NOT_FOUND");
        assert!(e.message.contains("lohalloc-example"));
        assert!(e.message.contains("cargo build"));
        assert!(e.message.contains("target/debug"));
        assert!(e.message.contains("target/release"));
        assert!(e.message.contains("LOHALLOC_BIN_"));
    }

    #[test]
    fn missing_shim_error_mentions_make() {
        let e = missing_shim_error();
        assert_eq!(e.code, "SHIM_NOT_FOUND");
        assert!(e.message.contains("make"));
    }

    #[test]
    fn event_ws_envelope_has_type() {
        let ev = SimulationEvent {
            pid: 1234,
            kind: "lohalloc-example".into(),
            status: "started".into(),
            duration_ms: 0,
            exit_code: None,
            stdout_tail: None,
            error: None,
        };
        let env = ev.into_ws_message();
        assert_eq!(env["type"], "simulation");
        assert_eq!(env["event"]["pid"], 1234);
        assert_eq!(env["event"]["kind"], "lohalloc-example");
    }

    #[test]
    fn args_default_is_empty() {
        let a = SimulationArgs::default();
        assert!(a.port.is_none());
        assert!(a.duration_secs.is_none());
    }
}
