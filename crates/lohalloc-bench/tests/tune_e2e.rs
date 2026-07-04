//! End-to-end tests for the Step-8 tune config across real process
//! boundaries — the piece the `tune.rs` unit tests can't cover: that a
//! `LOHALLOC_TUNE` file / `LOHALLOC_<KEY>` env var set on a *spawned child*
//! actually lands in that child's resolved config and (for a
//! global-allocator build) its freeze behavior.
//!
//! All spawn-based (the config is a process-wide `OnceLock`, so distinct
//! configs cannot coexist in one process — the same reason `tune_sweep`
//! shells out per grid point) and fully deterministic: nothing here asserts
//! on timing, only on resolved-config output and freeze-state debug lines.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// `latency_profile` built by this same `cargo test` invocation.
fn latency_profile_bin() -> &'static str {
    env!("CARGO_BIN_EXE_latency_profile")
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("lohalloc-tune-e2e");
    fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

/// Run `latency_profile --dump-config` with the given env, return stdout.
fn dump_config(envs: &[(&str, &str)]) -> String {
    let mut cmd = Command::new(latency_profile_bin());
    cmd.arg("--dump-config");
    // Isolate from any LOHALLOC_* vars in the ambient test environment.
    for (key, _) in std::env::vars() {
        if key.starts_with("LOHALLOC_") {
            cmd.env_remove(&key);
        }
    }
    for (k, v) in envs {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("failed to spawn latency_profile");
    assert!(
        out.status.success(),
        "--dump-config must exit 0 (stderr: {})",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

fn line_value<'a>(dump: &'a str, key: &str) -> &'a str {
    dump.lines()
        .find_map(|l| l.strip_prefix(&format!("{key}=")))
        .unwrap_or_else(|| panic!("no `{key}=` line in dump:\n{dump}"))
}

#[test]
fn precedence_defaults_then_file_then_focus_then_env_key() {
    // File sets focus=throughput (a preset expanding to t_ref_ns=200,
    // frag_weight=0.05) plus one explicit key; the env overrides one of
    // the focus-derived values. Full documented chain, one child process.
    let tune = scratch("precedence.tune");
    fs::write(&tune, "focus=throughput\nucb_c=1.25\n").unwrap();

    let dump = dump_config(&[
        ("LOHALLOC_TUNE", tune.to_str().unwrap()),
        ("LOHALLOC_T_REF_NS", "123"),
    ]);
    // Explicit file key applied.
    assert_eq!(line_value(&dump, "ucb_c"), "1.25");
    // Focus preset expanded (frag_weight comes only from the preset).
    assert_eq!(line_value(&dump, "frag_weight"), "0.05");
    // Env key beats the file's focus-derived t_ref_ns=200.
    assert_eq!(line_value(&dump, "t_ref_ns"), "123");
    // Untouched knobs stay at defaults.
    assert_eq!(line_value(&dump, "hysteresis"), "0.15");
}

#[test]
fn no_config_dumps_pure_defaults() {
    let dump = dump_config(&[]);
    assert_eq!(line_value(&dump, "ucb_c"), "2");
    assert_eq!(line_value(&dump, "t_ref_ns"), "50");
    assert_eq!(line_value(&dump, "frag_weight"), "0");
    assert_eq!(line_value(&dump, "freeze_mode"), "ops");
}

#[test]
fn bad_keys_and_values_warn_but_never_break_the_child() {
    // Robustness across the process boundary: a corrupt tune file must
    // leave the child running on defaults, not crash it — a production
    // process under LD_PRELOAD would otherwise die at startup.
    let tune = scratch("corrupt.tune");
    fs::write(
        &tune,
        "definitely_not_a_key=7\nucb_c=not-a-number\nt_ref_ns=75\n",
    )
    .unwrap();

    let mut cmd = Command::new(latency_profile_bin());
    cmd.arg("--dump-config")
        .env("LOHALLOC_TUNE", tune.to_str().unwrap());
    let out = cmd.output().expect("failed to spawn latency_profile");
    assert!(out.status.success());
    let dump = String::from_utf8(out.stdout).unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    // Bad entries ignored with a warning; the good key still applies.
    assert_eq!(line_value(&dump, "ucb_c"), "2", "bad value -> default kept");
    assert_eq!(
        line_value(&dump, "t_ref_ns"),
        "75",
        "good key still applies"
    );
    assert!(
        stderr.contains("unknown key") && stderr.contains("bad value"),
        "both problems must be warned about (stderr: {stderr})"
    );
}

/// The global-allocator behavior leg: a tune file's freeze knobs must reach
/// `native_workload`'s freeze driver (via `load_config_uncached` — the
/// `OnceLock` is useless there because the Rust runtime allocates before
/// `main`). Uses the prebuilt `native_workload_lohalloc` from
/// `make bench-rust-bins` (feature-gated bins can't be expressed via
/// `CARGO_BIN_EXE_*`, which builds with the test's own feature set) and
/// skips with a notice when it isn't built — CI's bench path always builds
/// it.
#[test]
fn tune_file_freeze_after_reaches_global_alloc_build() {
    let bin = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/native/build/native_workload_lohalloc");
    if !bin.exists() {
        eprintln!(
            "skipping: {} not built (run `make bench-rust-bins`)",
            bin.display()
        );
        return;
    }

    let froze = |tune: Option<&Path>| -> bool {
        let mut cmd = Command::new(&bin);
        cmd.args(["slab", "5000"]).env("LOHALLOC_DEBUG", "1");
        cmd.env_remove("LOHALLOC_TUNE")
            .env_remove("LOHALLOC_FREEZE_AFTER")
            .env_remove("LOHALLOC_MODEL")
            .env_remove("LOHALLOC_EXPORT_MODEL");
        if let Some(t) = tune {
            cmd.env("LOHALLOC_TUNE", t);
        }
        let out = cmd.output().expect("failed to spawn native_workload");
        assert!(out.status.success());
        let stderr = String::from_utf8_lossy(&out.stderr);
        let line = stderr
            .lines()
            .find(|l| l.contains("froze_during_run="))
            .unwrap_or_else(|| panic!("no froze_during_run debug line in: {stderr}"));
        line.contains("froze_during_run=true")
    };

    // Without a tune file the run never freezes...
    assert!(!froze(None), "untuned run must stay in training");
    // ...and with `freeze_after=100` from a tune FILE (no
    // LOHALLOC_FREEZE_AFTER env var involved) it must freeze mid-run.
    let tune = scratch("freeze.tune");
    fs::write(&tune, "freeze_after=100\n").unwrap();
    assert!(
        froze(Some(&tune)),
        "tune-file freeze_after must reach the global-allocator freeze driver"
    );
}
