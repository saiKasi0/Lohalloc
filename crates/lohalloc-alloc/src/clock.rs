//! Process-wide monotonic clock.
//!
//! Unlike `observer` (feature-gated behind `telemetry-observer`), this module
//! is always compiled in: Layer 2 latency-based MAB rewards (`lib.rs::route_alloc`,
//! `dealloc`) need timing on the Training path regardless of whether
//! telemetry is enabled. Only the Training path pays for it — the frozen
//! Inference hot path never calls `now_ns()`.

use std::sync::OnceLock;
use std::time::Instant;

/// Process-wide monotonic epoch, lazily anchored on first use. Every
/// `now_ns()` call measures elapsed time against this single fixed instant
/// — never against a fresh `Instant::now()` (which would just measure the
/// gap between two back-to-back calls, i.e. ~0).
static EPOCH: OnceLock<Instant> = OnceLock::new();

/// Monotonic nanosecond timestamp relative to the shared `EPOCH`, so
/// readings are meaningfully comparable across calls (and across threads).
/// Cheap (vDSO on Linux, `mach_absolute_time` on macOS), zero allocations.
#[inline]
pub(crate) fn now_ns() -> u64 {
    let epoch = EPOCH.get_or_init(Instant::now);
    epoch.elapsed().as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn now_ns_is_monotonic() {
        use std::thread;
        use std::time::Duration;
        let t1 = now_ns();
        thread::sleep(Duration::from_millis(5));
        let t2 = now_ns();
        assert!(t2 > t1, "now_ns() must be strictly increasing");
        let delta = t2 - t1;
        assert!(
            delta >= 4_000_000,
            "delta should be at least 5ms (5_000_000ns), got {}ns",
            delta
        );
    }
}
