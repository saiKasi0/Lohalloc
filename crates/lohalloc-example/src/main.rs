//! Lohalloc global-allocator smoke binary.
//!
//! Installs [`lohalloc_alloc::Lohalloc`] as the `#[global_allocator]` and
//! runs an allocation-heavy workload (`Vec` growth, many small `Box`es, large
//! buffers). Asserts the program completes without corruption. Run under both
//! debug and release:
//!
//! ```sh
//! cargo run -p lohalloc-example
//! cargo run -p lohalloc-example --release
//! ```
//!
//! # Telemetry (live mode)
//!
//! When built with `--features install-shim-sink`, the binary dlsym's the
//! `lohalloc_telemetry_emit` symbol exported by the C shim at
//! `shim/build/liblohalloc_obs.{so,dylib}` and installs it as the allocator's
//! observer sink. Preload the shim before launching:
//!
//! ```sh
//! make -C shim
//! cargo build -p lohalloc-example --release \
//!   --features telemetry-observer,install-shim-sink
//! DYLD_INSERT_LIBRARIES=$PWD/shim/build/liblohalloc_obs.dylib \
//!   target/release/lohalloc-example --duration-secs 60
//! ```
//!
//! With the shim loaded, every `alloc`/`free` produces a telemetry record
//! that flows: shim ring → POST /api/telemetry → server WS broadcast → GUI.

#![allow(unused)]

#[cfg(feature = "install-shim-sink")]
use lohalloc_alloc::observer;
use lohalloc_alloc::Lohalloc;
use std::time::{Duration, Instant};

#[global_allocator]
static ALLOC: Lohalloc = Lohalloc::new();

/// Try to dlsym the shim's sink symbol and install it as the observer sink.
/// Returns true if a sink was installed. Safe no-op when the
/// `install-shim-sink` feature is off (the shim is optional in this binary —
/// unlike `lohalloc-demo` which is built specifically for live-mode demos).
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
        eprintln!("[example] dlopen failed: shim not loaded — telemetry disabled");
        return false;
    }

    // SAFETY: sym_name is a valid C string, handle is valid.
    let sym = unsafe { libc::dlsym(handle, sym_name.as_ptr()) };
    if sym.is_null() {
        eprintln!(
            "[example] dlsym('lohalloc_telemetry_emit') returned NULL — is the shim preloaded?"
        );
        return false;
    }

    // SAFETY: the shim exports a function with the C-ABI sink signature,
    // matching `TelemetrySink`. The dlsym pointer is to a function, not data.
    let sink: observer::TelemetrySink = unsafe { std::mem::transmute(sym) };
    observer::install_sink(Some(sink));
    eprintln!("[example] installed shim telemetry sink @ {:?}", sym);
    true
}

#[cfg(not(feature = "install-shim-sink"))]
fn try_install_shim_sink() -> bool {
    false
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let duration_secs = parse_duration_secs(&args);
    let diverse = parse_diverse_flag(&args);

    println!(
        "Lohalloc smoke test — host: {} {}{}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        if diverse { " [diverse]" } else { "" }
    );

    // Install the shim sink before running any allocations so the very
    // first `Box`/`Vec` produces a record.
    let sink_installed = try_install_shim_sink();
    if sink_installed {
        // Give the consumer thread in the shim a moment to start so the
        // first record isn't dropped while the cond var is still unset.
        std::thread::sleep(Duration::from_millis(50));
    }

    if let Some(secs) = duration_secs {
        println!("Running for {}s...", secs);
        let deadline = Instant::now() + Duration::from_secs(secs);
        let mut iter = 0;
        if diverse {
            while Instant::now() < deadline {
                run_diverse_workload(iter);
                iter += 1;
                println!("Diverse iteration {} complete", iter);
            }
        } else {
            while Instant::now() < deadline {
                run_workload();
                iter += 1;
                println!("Iteration {} complete", iter);
            }
        }
        println!("Completed {} iterations in {}s", iter, secs);
    } else {
        if diverse {
            run_diverse_workload(0);
            println!("Lohalloc diverse smoke test PASSED");
        } else {
            run_workload();
            println!("Lohalloc smoke test PASSED");
        }
    }

    // Give the shim time to flush remaining buffered records before exit.
    if sink_installed {
        std::thread::sleep(Duration::from_millis(200));
        #[cfg(feature = "install-shim-sink")]
        observer::clear_sink();
    }
}

fn parse_diverse_flag(args: &[String]) -> bool {
    args.iter().any(|a| a == "--diverse")
}

fn parse_duration_secs(args: &[String]) -> Option<u64> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == "--duration-secs" {
            if let Some(val) = iter.next() {
                if let Ok(n) = val.parse::<u64>() {
                    return Some(n);
                }
            }
        }
    }
    None
}

fn run_workload() {
    // 1. Vec growth (many reallocs through the global allocator).
    let mut v: Vec<u64> = Vec::new();
    for i in 0..1_000_000u64 {
        v.push(i);
    }
    let sum: u64 = v.iter().sum();
    assert_eq!(sum, 1_000_000 * 999_999 / 2, "vec sum mismatch");

    // 2. Many small Boxes.
    let boxes: Vec<Box<[u8; 64]>> = (0..10_000).map(|_| Box::new([0u8; 64])).collect();
    for b in &boxes {
        assert!(b.iter().all(|&x| x == 0));
    }
    drop(boxes);

    // 3. Large buffers via the System Fallback.
    let big: Vec<u8> = vec![0xAB; 4 * 1024 * 1024]; // 4 MiB > BUDDY_MAX
    assert_eq!(big.len(), 4 * 1024 * 1024);
    assert!(big.iter().all(|&x| x == 0xAB));

    // 4. String operations (variable-size, alignment-sensitive).
    let s = "Lohalloc".repeat(1000);
    assert_eq!(s.len(), "Lohalloc".len() * 1000);

    // 5. HashMap (hashing + allocation).
    let mut m: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
    for i in 0..100_000 {
        m.insert(i, i * 2);
    }
    assert_eq!(m.get(&50_000u64), Some(&100_000));

    println!(
        "OK: completed {} vec entries, large buffer, 100k hashmap entries",
        v.len()
    );
}

// ---------------------------------------------------------------------------
// Diverse workloads — each is a separate function so the topology engine
// sees a distinct call-stack hash for every pattern.
// ---------------------------------------------------------------------------

/// Rotate through all diverse workloads. `idx` selects which workload runs
/// so successive iterations exercise different allocation patterns.
fn run_diverse_workload(idx: usize) {
    match idx % 10 {
        0 => workload_vec_small(),
        1 => workload_vec_medium(),
        2 => workload_boxes_32(),
        3 => workload_boxes_128(),
        4 => workload_string_build(),
        5 => workload_hashmap_small(),
        6 => workload_hashmap_large(),
        7 => workload_nested_structs(),
        8 => workload_buffer_1mib(),
        _ => workload_buffer_4mib(),
    }
}

fn workload_vec_small() {
    let mut v: Vec<u64> = Vec::new();
    for i in 0..1_000u64 {
        v.push(i);
    }
    let sum: u64 = v.iter().sum();
    assert_eq!(sum, 1_000 * 999 / 2);
}

fn workload_vec_medium() {
    let mut v: Vec<u64> = Vec::new();
    for i in 0..100_000u64 {
        v.push(i);
    }
    let sum: u64 = v.iter().sum();
    assert_eq!(sum, 100_000 * 99_999 / 2);
}

fn workload_boxes_32() {
    let boxes: Vec<Box<[u8; 32]>> = (0..5_000).map(|_| Box::new([0u8; 32])).collect();
    for b in &boxes {
        assert!(b.iter().all(|&x| x == 0));
    }
}

fn workload_boxes_128() {
    let boxes: Vec<Box<[u8; 128]>> = (0..2_000).map(|_| Box::new([0u8; 128])).collect();
    for b in &boxes {
        assert!(b.iter().all(|&x| x == 0));
    }
}

fn workload_string_build() {
    let s = "Lohalloc".repeat(500);
    assert_eq!(s.len(), "Lohalloc".len() * 500);
    let t = s.clone() + &s;
    assert_eq!(t.len(), s.len() * 2);
    let parts: Vec<&str> = t.split('o').collect();
    assert!(!parts.is_empty());
}

fn workload_hashmap_small() {
    let mut m: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
    for i in 0..1_000u64 {
        m.insert(i, i * 2);
    }
    assert_eq!(m.get(&500u64), Some(&1000));
}

fn workload_hashmap_large() {
    let mut m: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
    for i in 0..50_000u64 {
        m.insert(i, i * 2);
    }
    assert_eq!(m.get(&25_000u64), Some(&50_000));
}

struct Inner {
    data: [u8; 64],
}

struct Outer {
    inner: Box<Inner>,
    tag: u64,
}

fn workload_nested_structs() {
    let items: Vec<Outer> = (0..3_000)
        .map(|i| Outer {
            inner: Box::new(Inner {
                data: [i as u8; 64],
            }),
            tag: i,
        })
        .collect();
    assert_eq!(items.len(), 3_000);
    assert_eq!(items[100].tag, 100);
}

fn workload_buffer_1mib() {
    let buf: Vec<u8> = vec![0xCD; 1024 * 1024];
    assert_eq!(buf.len(), 1024 * 1024);
    assert!(buf.iter().all(|&x| x == 0xCD));
}

fn workload_buffer_4mib() {
    let buf: Vec<u8> = vec![0xAB; 4 * 1024 * 1024];
    assert_eq!(buf.len(), 4 * 1024 * 1024);
    assert!(buf.iter().all(|&x| x == 0xAB));
}
