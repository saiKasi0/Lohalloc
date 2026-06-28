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

#![allow(unused)]

use lohalloc_alloc::Lohalloc;

#[global_allocator]
static ALLOC: Lohalloc = Lohalloc::new();

fn main() {
    println!("Lohalloc smoke test — host: {} {}", std::env::consts::OS, std::env::consts::ARCH);

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

    println!("OK: completed {} vec entries, large buffer, 100k hashmap entries", v.len());
    println!("Lohalloc smoke test PASSED");
}
