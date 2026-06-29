//! Lohalloc live-training demo binary.
//!
//! Installs `lohalloc_alloc::Lohalloc` as the global allocator. When built
//! with `--features install-shim-sink`, dlsym's the
//! `lohalloc_telemetry_emit` symbol (exported by the C shim at
//! `shim/build/liblohalloc_obs.{so,dylib}` when preloaded via
//! `LD_PRELOAD` / `DYLD_INSERT_LIBRARIES`) and registers it as the
//! allocator's observer sink. Every alloc/free then produces a telemetry
//! record that flows through the shim → `POST /api/telemetry` → Axum WS →
//! GUI in real time.
//!
//! Run:
//!   make -C shim
//!   DYLD_INSERT_LIBRARIES=$PWD/shim/build/liblohalloc_obs.dylib \
//!     cargo run -p lohalloc-demo --features install-shim-sink --release
//!
//! Without the shim preloaded, the binary runs fine but no telemetry is
//! captured (the sink install is skipped via the feature flag).

#[cfg(feature = "install-shim-sink")]
use lohalloc_alloc::observer;
use lohalloc_alloc::Lohalloc;
use std::time::{Duration, Instant};

#[global_allocator]
static ALLOC: Lohalloc = Lohalloc::new();

/// Try to dlsym the shim's sink symbol and install it. Logs status to
/// stderr. Returns true if the sink was installed, false otherwise.
#[cfg(feature = "install-shim-sink")]
fn try_install_shim_sink() -> bool {
    use std::ffi::CString;
    let sym_name = match CString::new("lohalloc_telemetry_emit") {
        Ok(s) => s,
        Err(_) => return false,
    };

    // SAFETY: dlopen with a null pathname returns a handle for the main
    // program; combined with RTLD_GLOBAL this finds symbols exported by
    // any preloaded shared library.
    let handle = unsafe { libc::dlopen(std::ptr::null(), libc::RTLD_NOW | libc::RTLD_GLOBAL) };
    if handle.is_null() {
        eprintln!("[demo] dlopen failed: shim not loaded — telemetry disabled");
        return false;
    }

    // SAFETY: sym_name is a valid C string, handle is valid.
    let sym = unsafe { libc::dlsym(handle, sym_name.as_ptr()) };
    if sym.is_null() {
        eprintln!("[demo] dlsym('lohalloc_telemetry_emit') returned NULL — is the shim preloaded?");
        return false;
    }

    // SAFETY: the shim exports a function with the C-ABI sink signature,
    // matching `TelemetrySink`. The dlsym pointer is to a function, not data.
    let sink: observer::TelemetrySink = unsafe { std::mem::transmute(sym) };
    observer::install_sink(Some(sink));
    eprintln!("[demo] installed shim telemetry sink @ {:?}", sym);
    true
}

#[cfg(not(feature = "install-shim-sink"))]
fn try_install_shim_sink() -> bool {
    eprintln!("[demo] install-shim-sink feature disabled — telemetry sink not installed");
    false
}

fn run_churn_workload() {
    eprintln!("[demo] running churn workload...");
    let t0 = Instant::now();

    // Phase 1: dense Vec growth (small reallocs through MAB).
    let mut v: Vec<u64> = Vec::new();
    for i in 0..500_000u64 {
        v.push(i);
    }
    assert_eq!(v.len(), 500_000);

    // Phase 2: many small Boxes (slab-heavy).
    let boxes: Vec<Box<[u8; 64]>> = (0..5_000).map(|_| Box::new([0u8; 64])).collect();
    drop(boxes);

    // Phase 3: large buffer (system fallback).
    let big: Vec<u8> = vec![0xAB; 4 * 1024 * 1024];
    assert_eq!(big.len(), 4 * 1024 * 1024);
    drop(big);

    // Phase 4: HashMap churn.
    let mut m: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
    for i in 0..50_000 {
        m.insert(i, i.wrapping_mul(2));
    }
    assert_eq!(m.get(&25_000), Some(&50_000));

    // Phase 5: alloc/free bursts (pattern the GUI's floating web will light up).
    let mut arena: Vec<Vec<u8>> = Vec::new();
    for burst in 0..10 {
        let mut local: Vec<Vec<u8>> = Vec::new();
        for i in 0..100 {
            local.push(vec![burst as u8; 64 + i * 8]);
        }
        arena.extend(local);
    }
    drop(arena);

    eprintln!("[demo] workload complete in {:?}", t0.elapsed());
}

fn main() {
    eprintln!("Lohalloc live-training demo");
    eprintln!(
        "  host: {} {}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );

    let sink_installed = try_install_shim_sink();
    if sink_installed {
        // Give the consumer thread in the shim a moment to start.
        std::thread::sleep(Duration::from_millis(100));
    }

    run_churn_workload();

    // Give the shim time to flush remaining buffered records.
    if sink_installed {
        std::thread::sleep(Duration::from_millis(500));
        #[cfg(feature = "install-shim-sink")]
        observer::clear_sink();
    }

    eprintln!("[demo] DONE");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The binary's layout types must match the C shim's. If they ever
    /// drift, the C shim reads garbage — this test catches drift early.
    #[cfg(feature = "install-shim-sink")]
    #[test]
    fn shim_record_size_matches() {
        assert_eq!(
            std::mem::size_of::<observer::TelemetryCRecord>(),
            72,
            "TelemetryCRecord must be 72 bytes to match the C shim's wire format"
        );
    }

    /// Without the shim feature, the binary still runs the workload — just
    /// without telemetry.
    #[test]
    fn workload_runs_without_sink() {
        run_churn_workload();
    }
}
