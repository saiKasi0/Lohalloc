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
    let churn = has_flag(&args, "--churn");
    let checkerboard = has_flag(&args, "--checkerboard");
    let mixed_frag = has_flag(&args, "--mixed-fragmentation");
    let stress = has_flag(&args, "--stress");

    // Determine which workload mode to run.
    let mode = if churn {
        "churn"
    } else if checkerboard {
        "checkerboard"
    } else if mixed_frag {
        "mixed-fragmentation"
    } else if stress {
        "stress"
    } else if diverse {
        "diverse"
    } else {
        "default"
    };

    println!(
        "Lohalloc smoke test — host: {} {} [{}]",
        std::env::consts::OS,
        std::env::consts::ARCH,
        mode
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
        while Instant::now() < deadline {
            match mode {
                "churn" => run_churn_workload(iter),
                "checkerboard" => run_checkerboard_workload(iter),
                "mixed-fragmentation" => run_mixed_fragmentation_workload(iter),
                "stress" => run_diverse_workload(iter),
                "diverse" => run_diverse_workload(iter),
                _ => run_workload(),
            }
            iter += 1;
            println!("[{}] iteration {} complete", mode, iter);
        }
        println!("Completed {} iterations in {}s", iter, secs);
    } else {
        match mode {
            "churn" => {
                // Without duration, run 50 iterations so there's enough data.
                for i in 0..50 {
                    run_churn_workload(i);
                    println!("[churn] iteration {} complete", i + 1);
                }
            }
            "checkerboard" => {
                for i in 0..20 {
                    run_checkerboard_workload(i);
                    println!("[checkerboard] iteration {} complete", i + 1);
                }
            }
            "mixed-fragmentation" => {
                for i in 0..20 {
                    run_mixed_fragmentation_workload(i);
                    println!("[mixed-fragmentation] iteration {} complete", i + 1);
                }
            }
            "stress" | "diverse" => {
                run_diverse_workload(0);
                println!("Lohalloc {} smoke test PASSED", mode);
            }
            _ => {
                run_workload();
                println!("Lohalloc smoke test PASSED");
            }
        }
    }

    // Give the shim time to flush remaining buffered records before exit.
    if sink_installed {
        std::thread::sleep(Duration::from_millis(200));
        #[cfg(feature = "install-shim-sink")]
        observer::clear_sink();
    }
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
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

// ---------------------------------------------------------------------------
// High-Frequency Churn workload — rapid alloc/dealloc across all size classes.
// ---------------------------------------------------------------------------

/// Entry point for high-frequency churn. Rotates through churn patterns.
fn run_churn_workload(iter: usize) {
    match iter % 6 {
        0 => churn_tight_loop_small(),
        1 => churn_size_class_sweep(),
        2 => churn_ninety_percent_free(),
        3 => churn_interleaved_alloc_free(),
        4 => churn_box_churn(),
        _ => churn_vec_capacity_churn(),
    }
}

/// Tight loop of small alloc/free cycles — 8B to 64B.
fn churn_tight_loop_small() {
    let sizes: [usize; 4] = [8, 16, 32, 64];
    for _ in 0..500 {
        let mut ptrs: Vec<(std::ptr::NonNull<u8>, std::alloc::Layout)> = Vec::new();
        for &size in &sizes {
            let layout = std::alloc::Layout::from_size_align(size, 16).unwrap();
            for _ in 0..50 {
                unsafe {
                    let ptr = std::alloc::alloc(layout);
                    if !ptr.is_null() {
                        ptrs.push((std::ptr::NonNull::new_unchecked(ptr), layout));
                    }
                }
            }
        }
        let keep = ptrs.len() / 10;
        let to_free = ptrs.split_off(keep);
        for (ptr, layout) in to_free {
            unsafe { std::alloc::dealloc(ptr.as_ptr(), layout) };
        }
        for (ptr, layout) in ptrs {
            unsafe { std::alloc::dealloc(ptr.as_ptr(), layout) };
        }
    }
}

/// Sweep through all size classes from 8B to 1MiB, allocating and freeing.
fn churn_size_class_sweep() {
    let sizes: [usize; 13] = [
        8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536, 262144, 1048576,
    ];
    let mut live: Vec<Vec<u8>> = Vec::new();
    for &size in &sizes {
        for _ in 0..30 {
            live.push(vec![0xAB; size]);
        }
        let half = live.len() / 2;
        live.truncate(half);
    }
}

/// Allocate many items, free 90% immediately, keep 10%.
fn churn_ninety_percent_free() {
    let mut live: Vec<Box<[u8]>> = Vec::new();
    for i in 0..2000 {
        let size = 8 << (i % 14);
        live.push(vec![0xCD; size].into_boxed_slice());
        if i % 10 != 0 {
            live.pop();
        }
    }
}

/// Interleaved allocate-N-free-N pattern across mixed sizes.
fn churn_interleaved_alloc_free() {
    for _ in 0..100 {
        let mut items: Vec<Vec<u8>> = Vec::with_capacity(100);
        for i in 0..100 {
            items.push(vec![i as u8; 32 * (1 + i % 32)]);
        }
        let mut i = 0;
        while i < items.len() {
            items.remove(i);
            i += 1;
        }
    }
}

/// Box churn — allocate and free many Box<[u8]> of varying sizes.
fn churn_box_churn() {
    let mut boxes: Vec<Box<[u8]>> = Vec::new();
    for i in 0..5000 {
        let size = 16 << (i % 10);
        boxes.push(vec![i as u8; size].into_boxed_slice());
        if i % 10 < 9 {
            boxes.pop();
        }
    }
}

/// Vec capacity churn — repeated grow/shrink cycles.
fn churn_vec_capacity_churn() {
    for _ in 0..200 {
        let mut v: Vec<u64> = Vec::new();
        for i in 0..10_000 {
            v.push(i);
        }
        v.truncate(v.len() / 2);
        for i in 0..5_000 {
            v.push(i);
        }
        v.clear();
    }
}

// ---------------------------------------------------------------------------
// Checkerboard Fragmentation workload — alternating alloc/free for max
// external fragmentation.
// ---------------------------------------------------------------------------

/// Entry point for checkerboard fragmentation. Rotates through patterns.
fn run_checkerboard_workload(iter: usize) {
    match iter % 4 {
        0 => checkerboard_uniform_256(),
        1 => checkerboard_uniform_64(),
        2 => checkerboard_varied_sizes(),
        _ => checkerboard_dynamic_fill(),
    }
}

/// Allocate 10,000 items of 256 bytes, free every alternating one.
fn checkerboard_uniform_256() {
    let mut items: Vec<Vec<u8>> = (0..10_000).map(|_| vec![0xABu8; 256]).collect();
    for i in (0..items.len()).step_by(2) {
        items[i] = Vec::new();
    }
    for i in (0..items.len()).step_by(2) {
        let size = 64 + (i % 4) * 64;
        items[i] = vec![0xCDu8; size];
    }
}

/// Allocate 10,000 items of 64 bytes, free every alternating one.
fn checkerboard_uniform_64() {
    let mut items: Vec<Box<[u8]>> = (0..10_000)
        .map(|_| vec![0u8; 64].into_boxed_slice())
        .collect();
    for i in (0..items.len()).step_by(2) {
        items[i] = Vec::new().into_boxed_slice();
    }
    for i in (0..items.len()).step_by(2) {
        let size = 16 << (i % 6);
        items[i] = vec![0xEFu8; size].into_boxed_slice();
    }
}

/// Checkerboard with varied allocation sizes.
fn checkerboard_varied_sizes() {
    let mut items: Vec<Vec<u8>> = Vec::new();
    for i in 0..10_000 {
        let size = 32 << (i % 8);
        items.push(vec![i as u8; size]);
    }
    for i in (0..items.len()).step_by(2) {
        items[i] = Vec::new();
    }
    for i in (0..items.len()).step_by(2) {
        let size = 16 << (i % 10);
        items[i] = vec![0x99u8; size];
    }
}

/// Dynamic checkerboard — repeated alloc/free to create evolving fragmentation.
fn checkerboard_dynamic_fill() {
    for round in 0..5 {
        let mut items: Vec<Vec<u8>> = (0..5_000).map(|i| vec![(round + i) as u8; 128]).collect();
        for i in (0..items.len()).step_by(3) {
            items[i] = Vec::new();
        }
        for i in (0..items.len()).step_by(3) {
            let size = 64 << (i % 5);
            items[i] = vec![0x55u8; size];
        }
        for i in (1..items.len()).step_by(2) {
            items[i] = Vec::new();
        }
    }
}

// ---------------------------------------------------------------------------
// Mixed Fragmentation workload — interleaved large blocks with tiny allocs.
// ---------------------------------------------------------------------------

/// Entry point for mixed fragmentation. Rotates through patterns.
fn run_mixed_fragmentation_workload(iter: usize) {
    match iter % 5 {
        0 => mixed_large_then_tiny(),
        1 => mixed_interleaved_blocks(),
        2 => mixed_tiny_then_large(),
        3 => mixed_buddy_slab_stress(),
        _ => mixed_compaction_pattern(),
    }
}

/// Allocate large contiguous blocks, then flood with tiny allocations.
fn mixed_large_then_tiny() {
    let mut large: Vec<Vec<u8>> = Vec::new();
    large.push(vec![0xABu8; 4 * 1024 * 1024]);
    large.push(vec![0xCDu8; 1024 * 1024]);
    large.push(vec![0xEFu8; 1024 * 1024]);

    let mut tiny: Vec<Box<[u8]>> = Vec::new();
    for i in 0..10_000 {
        let size = 16 + (i % 3) * 16;
        tiny.push(vec![i as u8; size].into_boxed_slice());
    }

    let half = tiny.len() / 2;
    let _freed = tiny.split_off(half);

    large.remove(1);

    for i in 0..5_000 {
        let size = 16 + (i % 3) * 16;
        tiny.push(vec![i as u8; size].into_boxed_slice());
    }
}

/// Interleave large block allocations with tiny allocations.
fn mixed_interleaved_blocks() {
    for _ in 0..10 {
        let _big: Vec<u8> = vec![0xABu8; 1024 * 1024];

        let mut tiny: Vec<Box<[u8]>> = Vec::new();
        for i in 0..2_000 {
            let size = 16 << (i % 4);
            tiny.push(vec![i as u8; size].into_boxed_slice());
        }

        let half = tiny.len() / 2;
        tiny.truncate(half);

        let _big2: Vec<u8> = vec![0xCDu8; 4 * 1024 * 1024];

        for i in 0..2_000 {
            let size = 32 << (i % 3);
            tiny.push(vec![i as u8; size].into_boxed_slice());
        }
    }
}

/// Thousands of tiny allocations first, then large blocks on top.
fn mixed_tiny_then_large() {
    let mut tiny: Vec<Box<[u8]>> = Vec::new();
    for i in 0..20_000 {
        let size = 16 + (i % 4) * 16;
        tiny.push(vec![i as u8; size].into_boxed_slice());
    }

    let mut i = 0;
    while i < tiny.len() {
        tiny.remove(i);
        i += 1;
    }

    let _big1: Vec<u8> = vec![0xABu8; 1024 * 1024];
    let _big2: Vec<u8> = vec![0xCDu8; 4 * 1024 * 1024];
}

/// Stress the buddy allocator with power-of-two sizes mixed with slab sizes.
fn mixed_buddy_slab_stress() {
    let mut buddy_blocks: Vec<Vec<u8>> = Vec::new();
    for i in 0..20 {
        let size = 4096 << (i % 9);
        buddy_blocks.push(vec![i as u8; size]);
    }

    let mut slab_blocks: Vec<Box<[u8]>> = Vec::new();
    for i in 0..15_000 {
        let size = 8 << (i % 12);
        slab_blocks.push(vec![i as u8; size].into_boxed_slice());
    }

    let mut i = 0;
    while i < buddy_blocks.len() {
        buddy_blocks.remove(i);
        i += 2;
    }

    let mut i = 0;
    while i < slab_blocks.len() {
        slab_blocks.remove(i);
        i += 1;
    }

    for i in 0..10 {
        let size = 4096 << (i % 9);
        buddy_blocks.push(vec![i as u8; size]);
    }
    for i in 0..5_000 {
        let size = 8 << (i % 12);
        slab_blocks.push(vec![i as u8; size].into_boxed_slice());
    }
}

/// Pattern that creates compaction-like behavior: allocate, free holes,
/// then allocate to fill holes.
fn mixed_compaction_pattern() {
    for round in 0..5 {
        let mut blocks: Vec<Vec<u8>> = Vec::new();
        for i in 0..10 {
            let size = if i % 3 == 0 {
                4 * 1024 * 1024
            } else {
                1024 * 1024
            };
            blocks.push(vec![(round + i) as u8; size]);
        }

        let mut tiny: Vec<Box<[u8]>> = Vec::new();
        for i in 0..8_000 {
            let size = 16 + (i % 4) * 16;
            tiny.push(vec![i as u8; size].into_boxed_slice());
        }

        blocks.remove(2);
        blocks.remove(5);

        let half = tiny.len() / 2;
        tiny.truncate(half);

        blocks.push(vec![0xFFu8; 1024 * 1024]);
        for i in 0..4_000 {
            let size = 16 + (i % 4) * 16;
            tiny.push(vec![i as u8; size].into_boxed_slice());
        }
    }
}
