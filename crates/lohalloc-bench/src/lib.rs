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
