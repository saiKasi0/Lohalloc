//! Lohalloc control-plane server binary.
//!
//! Run with:
//! ```not_rust
//! cargo run -p lohalloc-server
//! ```
//!
//! Listens on `127.0.0.1:3000` by default. Set `LOHALLOC_ADDR` to override.

use lohalloc_server::{build_app, AppState};

#[tokio::main]
async fn main() {
    let addr: std::net::SocketAddr = std::env::var("LOHALLOC_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:3000".to_string())
        .parse()
        .expect("invalid LOHALLOC_ADDR");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    eprintln!("Lohalloc server listening on {addr}");

    axum::serve(listener, build_app(AppState::new()))
        .await
        .unwrap();
}
