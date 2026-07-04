//! Dump a `.lohalloc` model's routing table for diagnosis.
//!
//! ```sh
//! cargo run -p lohalloc-bench --bin model_dump --release -- <path-to.lohalloc>
//! ```
//!
//! Prints one line per routing entry: `hash size_class backend`, plus a
//! per-backend summary count. `size_class` follows `state::size_class_for`:
//! 0-11 are the Slab classes, 12/13 are Buddy (16KiB-64KiB, 64KiB-1MiB), 14
//! is System (>1MiB). Built to diagnose a frozen model that routes a
//! Signature to a backend its size class shouldn't reach (e.g. a Buddy-range
//! class frozen to System) — see the Step 0 investigation in
//! `crates/lohalloc-alloc/src/state.rs`'s `clamp_backend_for_size_class`.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::process::ExitCode;

use lohalloc_alloc::state::AllocatorState;
use lohalloc_core::Backend;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: model_dump <path-to.lohalloc>");
        return ExitCode::FAILURE;
    };

    let bytes = match fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("model_dump: could not read {path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let Some(state) = AllocatorState::load(&bytes) else {
        eprintln!("model_dump: {path} is malformed (bad magic/checksum)");
        return ExitCode::FAILURE;
    };

    let Some(table) = state.routing_table() else {
        eprintln!("model_dump: loaded state is not in Inference mode (unreachable via load())");
        return ExitCode::FAILURE;
    };

    let entries = table.entries();
    println!("# {} entries", entries.len());
    println!("# hash size_class backend");
    let mut by_backend: BTreeMap<&'static str, usize> = BTreeMap::new();
    for (hash, size_class, backend) in &entries {
        println!("{hash:#018x} {size_class} {}", backend.as_str());
        *by_backend.entry(backend.as_str()).or_insert(0) += 1;
    }

    println!("# summary:");
    for backend in Backend::ALL {
        let name = backend.as_str();
        println!("#   {name}: {}", by_backend.get(name).copied().unwrap_or(0));
    }

    // Sanity flag matching `clamp_backend_for_size_class`'s invariant: no
    // entry with size_class <= 13 (fits Slab/Buddy) should ever be System
    // after the Step 0 fix — flag it here too so a stale/pre-fix model is
    // caught immediately by inspection.
    let suspicious: Vec<_> = entries
        .iter()
        .filter(|(_, size_class, backend)| *backend == Backend::System && *size_class <= 13)
        .collect();
    if !suspicious.is_empty() {
        println!(
            "# WARNING: {} entries route a Slab/Buddy-range size_class to System",
            suspicious.len()
        );
    }

    ExitCode::SUCCESS
}
