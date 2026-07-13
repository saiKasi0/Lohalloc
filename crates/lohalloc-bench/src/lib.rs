//! Phase 6 hypothesis-validation benchmarking support crate.
//!
//! Lohalloc claims to learn a workload's topology and route each allocation
//! to the backend best suited to its size/lifetime. This crate provides:
//!
//! - [`workloads`]: backend-pure and adversarial workload generators shared by
//!   routing-validation tests, criterion benches, and the latency profiler.
//! - [`forced`]: hand-built `.lohalloc` models that force specific call-site
//!   hashes to specific backends (Layer 1 — validates the execution plane
//!   independent of the bandit's learning).
//! - [`hypotheses`]: the routing-distribution assertions shared by
//!   `tests/routing_validation.rs` and (once trained routing lands)
//!   `tests/decision_plane.rs`.

pub mod clockinfo;
pub mod forced;
pub mod global_alloc;
pub mod hypotheses;
pub mod workloads;

/// Peak RSS of this process in bytes (`getrusage(RUSAGE_SELF).ru_maxrss` —
/// high-water resident set over the run). Cross-platform: `ru_maxrss` is KiB on
/// Linux, bytes on macOS. `0` if the syscall fails. Shared by the RSS pass in
/// `native_workload` and the `latency_profile` reporter — the memory-footprint
/// axis of the real-workload benchmarks.
pub fn peak_rss_bytes() -> u64 {
    // SAFETY: getrusage writes into the zeroed struct we hand it; no pointers
    // retained.
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut ru) != 0 {
            return 0;
        }
        let raw = ru.ru_maxrss.max(0) as u64;
        if cfg!(target_os = "macos") {
            raw
        } else {
            raw * 1024
        }
    }
}
