//! Lohalloc control-plane server binary.
//!
//! Run with:
//! ```not_rust
//! cargo run -p lohalloc-server
//! ```
//!
//! Listens on `127.0.0.1:3000` by default. Set `LOHALLOC_ADDR` to override.
//! Serves the built frontend from `gui/dist/` if it exists.
//!
//! Logging is via `tracing`, controlled by `RUST_LOG` (defaults to `info`
//! if unset). Set `RUST_LOG=lohalloc_server=debug` to see per-record
//! telemetry ingest and simulation-watcher heartbeat spans — see
//! `gui/DEBUGGING.md` for the full recipe.

use lohalloc_server::{build_app_with_options, AppState};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let addr: std::net::SocketAddr = std::env::var("LOHALLOC_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:3000".to_string())
        .parse()
        .expect("invalid LOHALLOC_ADDR");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    tracing::info!(%addr, "Lohalloc server listening");

    let state = AppState::new();
    state.set_server_port(addr.port());

    // Trap #1 (panic): spawned simulations are detached process-group
    // leaders, so a server panic/abort would orphan them. Kill their groups
    // synchronously from the panic hook before the default hook aborts.
    install_panic_trap(state.clone());

    // Serve static frontend files from gui/dist/ in production.
    // Trap #2 (signals): SIGINT (Ctrl-C) / SIGTERM stop the accept loop via
    // graceful shutdown, and reap every child before we exit.
    let result = axum::serve(listener, build_app_with_options(state.clone(), true))
        .with_graceful_shutdown(shutdown_signal(state.clone()))
        .await;
    if let Err(e) = result {
        tracing::error!(error = %e, "server error");
    }

    // Trap #3 (normal return): belt-and-suspenders reap on any clean exit
    // path (e.g. the accept loop ending) so no simulation is left running.
    let reaped = state.kill_all_simulations().await;
    tracing::info!(reaped, "server shut down; simulations reaped");
}

/// Install a panic hook that force-kills every running simulation's process
/// group before delegating to the previous hook. Holds an `AppState` clone
/// for the process lifetime so the registry is reachable from the hook.
fn install_panic_trap(state: AppState) {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let killed = state.kill_all_simulations_blocking();
        if killed > 0 {
            tracing::error!(killed, "panic: killed running simulations before abort");
        }
        previous(info);
    }));
}

/// Resolve once a shutdown signal arrives, then kill all simulations. Used as
/// the `with_graceful_shutdown` future: returning from it tells Axum to stop
/// accepting new connections. A long-lived WebSocket could otherwise keep the
/// graceful drain open indefinitely, so we also arm a hard-exit fallback.
async fn shutdown_signal(state: AppState) {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(e) => {
                tracing::warn!(error = %e, "failed to install SIGTERM handler");
                std::future::pending::<()>().await;
            }
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => tracing::warn!("received SIGINT — shutting down"),
        _ = terminate => tracing::warn!("received SIGTERM — shutting down"),
    }

    let killed = state.kill_all_simulations().await;
    tracing::info!(killed, "simulations terminated on shutdown");

    // If an in-flight WebSocket keeps the graceful drain from completing,
    // don't hang forever (and don't leave the operator unable to exit).
    tokio::spawn(async {
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
        tracing::warn!("graceful drain timed out — forcing exit");
        std::process::exit(0);
    });
}
