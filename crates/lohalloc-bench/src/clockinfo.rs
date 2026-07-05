//! Timer-resolution probe: measures the effective tick floor of
//! `std::time::Instant` on this machine.
//!
//! Why this exists: on Apple Silicon, `Instant` is backed by the 24 MHz ARM
//! generic timer, giving an effective resolution of ~41.7ns — any single
//! measured interval quantizes to a multiple of that tick. Per-op latency
//! percentiles for operations *faster than the tick* (slab/arena hot paths,
//! ~20-40ns) then read as 0/41/42/83ns buckets rather than a continuous
//! distribution, which silently invalidates p50/p99 comparisons between
//! configs (discovered during the Ladder 3 tune sweep; see COPILOT.md).
//! x86 (TSC-backed `clock_gettime`) resolves ~1ns and is unaffected.
//!
//! The probe is cheap (~a few ms) and run once per process by consumers
//! that report per-op latencies (`latency_profile`), which stamp the result
//! into their JSON so the aggregator can annotate quantized percentiles
//! instead of presenting them as real measurements.

use std::time::Instant;

/// Measured effective tick of `std::time::Instant`, in nanoseconds.
///
/// Method: hammer `Instant::now()` back-to-back and take the smallest
/// *positive* observed delta. Back-to-back calls are faster than one tick
/// on a tick-floored clock (most deltas read 0), so the smallest nonzero
/// delta is the tick itself; on a fine-grained clock it's the true call
/// overhead (~10-30ns on x86), which is the honest "smallest interval this
/// clock can distinguish" either way.
///
/// Returns at least 1 (a clock that never advanced across the whole probe
/// would be broken; 1ns is the "no measurable floor" answer).
pub fn instant_tick_ns() -> u64 {
    let mut min_positive = u64::MAX;
    let mut last = Instant::now();
    // 100k samples keeps the probe well under 10ms while making it
    // overwhelmingly likely we straddle many tick boundaries.
    for _ in 0..100_000 {
        let now = Instant::now();
        let delta = now.duration_since(last).as_nanos() as u64;
        if delta > 0 && delta < min_positive {
            min_positive = delta;
        }
        last = now;
    }
    if min_positive == u64::MAX {
        1
    } else {
        min_positive.max(1)
    }
}

/// Is a reported per-op percentile too close to the tick floor to be a
/// trustworthy measurement? Below ~3 ticks a value is dominated by
/// quantization (it can only ever have been 0, 1, or 2 ticks), so treat it
/// as a bucket label, not a latency.
pub fn is_quantized(value_ns: u64, tick_ns: u64) -> bool {
    tick_ns > 1 && value_ns < tick_ns.saturating_mul(3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_returns_sane_positive_tick() {
        let tick = instant_tick_ns();
        // Any real clock lands between "sub-ns is impossible to observe"
        // and "worse than 1us would break every timer-based test we have".
        assert!(tick >= 1, "tick must be positive");
        assert!(tick < 1_000, "tick {tick}ns implausibly coarse");
    }

    #[test]
    fn quantization_check_flags_only_near_tick_values() {
        // 42ns tick (Apple Silicon): sub-3-tick values are bucket labels.
        assert!(is_quantized(0, 42));
        assert!(is_quantized(42, 42));
        assert!(is_quantized(84, 42));
        assert!(!is_quantized(126, 42));
        assert!(!is_quantized(4_000, 42));
        // 1ns tick (x86 TSC): nothing is flagged.
        assert!(!is_quantized(0, 1));
        assert!(!is_quantized(42, 1));
    }
}
