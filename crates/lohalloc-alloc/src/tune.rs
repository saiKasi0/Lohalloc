//! Tunable training configuration (Step 8).
//!
//! Every training-path hyperparameter that used to be a hard-coded constant
//! (`EXPLORATION_C`, `HYSTERESIS_PENALTY`, `BASELINE_REWARDS`, `T_REF_NS`)
//! lives in [`TrainingConfig`], alongside the Step 8 additions: a
//! fragmentation weight for reward shaping, a `focus` preset
//! (latency / throughput / balanced) that sets the reward-shape knobs in
//! one key, and the convergence-based auto-freeze mode. **Defaults
//! reproduce the pre-Step-8 behavior exactly** (`frag_weight = 0`, same
//! constants, op-count freezing).
//!
//! # Config sources and precedence
//!
//! defaults → `LOHALLOC_TUNE=<path>` file (flat `key=value` lines, `#`
//! comments) → per-key env overrides (`LOHALLOC_<KEY>`, e.g.
//! `LOHALLOC_UCB_C=1.4`). Within each layer a `focus` preset is applied
//! *before* individual keys, so explicit keys always win over the preset
//! regardless of their order in the file. Deliberately **not JSON**: no
//! serde in a crate that runs under `LD_PRELOAD` — the parser below is a
//! few dozen lines of `str::split`. JSON lives at the harness layer
//! (`tune_sweep` takes a JSON grid and emits these `key=value` files).
//!
//! # Re-entrancy contract (load-bearing)
//!
//! [`load_from_env`] reads env vars and a file — both allocate internally,
//! and `std::env::var` is guarded by std's own non-reentrant machinery.
//! Under an interposed allocator this is exactly the documented
//! `ensure_model_loaded` deadlock class, so **`load_from_env` must only be
//! called from a bootstrap-guarded context** (`Lohalloc::with_bootstrap_guard`
//! in `lohalloc-cabi`, or plain `main()` before any training traffic in a
//! harness binary). [`config`] itself never does I/O: if `load_from_env`
//! was never called it returns the defaults — a process that doesn't opt
//! in pays nothing and behaves exactly as before.
//!
//! # Keys
//!
//! | key                 | default | meaning |
//! |---------------------|---------|---------|
//! | `focus`             | `latency` | preset: sets `(t_ref_ns, frag_weight)` — `latency` (50, 0), `throughput` (200, 0.05), `balanced` (100, 0.02) |
//! | `ucb_c`             | 2.0     | UCB1 exploration constant |
//! | `hysteresis`        | 0.15    | anti-jitter penalty on switching arms |
//! | `t_ref_ns`          | 50.0    | reward curve knee: small = tail-punishing (latency focus), large = flatter mean-cost curve (throughput focus) |
//! | `frag_weight`       | 0.0     | reward penalty per 100% internal fragmentation; 0 disables the frag computation entirely |
//! | `baseline_slab`     | 1.0     | cold-start prior reward per backend |
//! | `baseline_buddy`    | 0.8     | |
//! | `baseline_system`   | 0.3     | |
//! | `baseline_arena`    | 0.9     | |
//! | `freeze_mode`       | `ops`   | `ops` = freeze at a fixed malloc count; `converged` = freeze when the bandit stabilizes (see `BanditPolicy::is_converged`) |
//! | `converge_stable_n` | 64      | consecutive same-arm selections per Signature required by `converged` |
//! | `freeze_after`      | (none)  | op-count threshold; the env var `LOHALLOC_FREEZE_AFTER` (read by `lohalloc-cabi`) takes precedence over this key |
//! | `reward_batch`      | 16      | latency samples averaged per bandit reward update; de-quantizes the ARM ~42ns `Instant` tick floor. `1` = pre-Ladder-4 per-op behavior |
//! | `demote_fraction`   | 0.10    | freeze-time Arena-demotion volume threshold (J5-A bisect knob): `0.0` = demote every Arena verdict (J4-C blanket), `>1.0` = never demote |

/// How the training→inference transition is triggered (consumed by
/// `lohalloc-cabi`'s auto-freeze and `native_workload`'s driver — the
/// allocator itself never freezes spontaneously).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FreezeMode {
    /// Freeze after a fixed number of successful top-level allocations
    /// (`freeze_after` / `LOHALLOC_FREEZE_AFTER`). Pre-Step-8 behavior.
    Ops,
    /// Freeze once every trained Signature's arm choice has been stable
    /// for `converge_stable_n` consecutive selections and its mean reward
    /// separates from the runner-up's UCB interval. An op-count threshold,
    /// if also set, acts as a hard cap.
    Converged,
}

/// All training-path knobs. See the module doc's key table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainingConfig {
    pub ucb_c: f64,
    pub hysteresis: f64,
    /// Cold-start priors, indexed `[Slab, Buddy, System, Arena]` (the
    /// `Backend as usize` order).
    pub baseline_rewards: [f64; 4],
    pub t_ref_ns: f64,
    pub frag_weight: f64,
    pub freeze_mode: FreezeMode,
    pub converge_stable_n: u32,
    pub freeze_after: Option<u64>,
    /// Number of latency samples averaged per bandit reward update (the
    /// clock-quantization fix — see `state::PendingRewards`). Default 16
    /// averages out the ~42ns `Instant` tick floor on ARM; `1` reproduces
    /// pre-Ladder-4 per-op reward behavior bit-for-bit. Clamped to ≥1 by
    /// the parser (0 would never flush a reward).
    pub reward_batch: u32,
    /// Freeze-time Arena-demotion volume threshold (J5-A, bisect knob): a
    /// signature's Arena verdict is demoted to the size-based default when
    /// its training volume `pulls/total >= demote_fraction` (and the arena
    /// never reset, see `Lohalloc::freeze`). `0.0` demotes EVERY Arena
    /// verdict (the J4-C blanket behavior); `0.10` is the certified J5
    /// volume-selective default; `> 1.0` never demotes.
    pub demote_fraction: f64,
}

impl TrainingConfig {
    /// Compile-time defaults — the single source of truth (`Default`
    /// delegates here). `const` so the `DEFAULTS` static [`config`] falls
    /// back to before [`load_from_env`] runs needs no runtime init.
    pub const fn defaults_const() -> Self {
        Self {
            ucb_c: 2.0,
            hysteresis: 0.15,
            baseline_rewards: [1.0, 0.8, 0.3, 0.9],
            t_ref_ns: 50.0,
            frag_weight: 0.0,
            freeze_mode: FreezeMode::Ops,
            converge_stable_n: 64,
            freeze_after: None,
            reward_batch: 16,
            demote_fraction: crate::state::ARENA_DEMOTE_VOLUME_FRACTION,
        }
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self::defaults_const()
    }
}

/// The known keys, used both for file parsing and for deriving the
/// `LOHALLOC_<KEY>` env-override names.
const KEYS: [&str; 14] = [
    "focus",
    "ucb_c",
    "hysteresis",
    "t_ref_ns",
    "frag_weight",
    "baseline_slab",
    "baseline_buddy",
    "baseline_system",
    "baseline_arena",
    "freeze_mode",
    "converge_stable_n",
    "freeze_after",
    "reward_batch",
    "demote_fraction",
];

/// Apply a `focus` preset. Only sets the reward-shape pair — everything
/// else keeps its current value, and explicit `t_ref_ns`/`frag_weight`
/// keys applied *after* the preset override it (see `apply_layer`).
///
/// `latency` is pinned to the historical defaults and must stay
/// behavior-identical. These `(t_ref_ns, frag_weight)` pairs were the
/// original Step 8 design (not swept — `tune_sweep`'s grid varies
/// `ucb_c`/`hysteresis` per focus instead, see `bench/lohalloc-tune.*`
/// for the 2026-07-04 measured results checked into those files).
fn apply_focus(cfg: &mut TrainingConfig, value: &str) {
    match value {
        "latency" => {
            cfg.t_ref_ns = 50.0;
            cfg.frag_weight = 0.0;
        }
        "throughput" => {
            cfg.t_ref_ns = 200.0;
            cfg.frag_weight = 0.05;
        }
        "balanced" => {
            cfg.t_ref_ns = 100.0;
            cfg.frag_weight = 0.02;
        }
        other => eprintln!("lohalloc tune: unknown focus '{other}' ignored"),
    }
}

/// Apply one `key = value` pair. Unknown keys and unparseable values warn
/// and keep the current value — a typo in a tune file must never abort or
/// silently zero a knob.
fn apply_key(cfg: &mut TrainingConfig, key: &str, value: &str) {
    fn parse_f64(cfg_field: &mut f64, key: &str, value: &str) {
        match value.parse::<f64>() {
            Ok(v) if v.is_finite() => *cfg_field = v,
            _ => eprintln!("lohalloc tune: bad value '{value}' for {key} ignored"),
        }
    }
    match key {
        "focus" => apply_focus(cfg, value),
        "ucb_c" => parse_f64(&mut cfg.ucb_c, key, value),
        "hysteresis" => parse_f64(&mut cfg.hysteresis, key, value),
        "t_ref_ns" => parse_f64(&mut cfg.t_ref_ns, key, value),
        "frag_weight" => parse_f64(&mut cfg.frag_weight, key, value),
        "baseline_slab" => parse_f64(&mut cfg.baseline_rewards[0], key, value),
        "baseline_buddy" => parse_f64(&mut cfg.baseline_rewards[1], key, value),
        "baseline_system" => parse_f64(&mut cfg.baseline_rewards[2], key, value),
        "baseline_arena" => parse_f64(&mut cfg.baseline_rewards[3], key, value),
        "freeze_mode" => match value {
            "ops" => cfg.freeze_mode = FreezeMode::Ops,
            "converged" => cfg.freeze_mode = FreezeMode::Converged,
            other => eprintln!("lohalloc tune: bad freeze_mode '{other}' ignored"),
        },
        "converge_stable_n" => match value.parse::<u32>() {
            Ok(v) if v > 0 => cfg.converge_stable_n = v,
            _ => eprintln!("lohalloc tune: bad value '{value}' for converge_stable_n ignored"),
        },
        "freeze_after" => match value.parse::<u64>() {
            Ok(v) if v > 0 => cfg.freeze_after = Some(v),
            _ => eprintln!("lohalloc tune: bad value '{value}' for freeze_after ignored"),
        },
        "reward_batch" => match value.parse::<u32>() {
            Ok(v) if v > 0 => cfg.reward_batch = v,
            _ => eprintln!(
                "lohalloc tune: bad value '{value}' for reward_batch (must be >= 1) ignored"
            ),
        },
        "demote_fraction" => match value.parse::<f64>() {
            Ok(v) if v.is_finite() && v >= 0.0 => cfg.demote_fraction = v,
            _ => eprintln!(
                "lohalloc tune: bad value '{value}' for demote_fraction (must be >= 0) ignored"
            ),
        },
        other => eprintln!("lohalloc tune: unknown key '{other}' ignored"),
    }
}

/// Parse a flat `key=value` file body into pairs. Blank lines and `#`
/// comments are skipped; whitespace around key and value is trimmed.
fn parse_pairs(text: &str) -> Vec<(String, String)> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .filter_map(|l| {
            let (k, v) = l.split_once('=')?;
            Some((k.trim().to_string(), v.trim().to_string()))
        })
        .collect()
}

/// Apply one precedence layer: `focus` (if present) first, then every
/// other pair in order — so an explicit key in the same layer always beats
/// the preset, wherever it appears in the file.
fn apply_layer(cfg: &mut TrainingConfig, pairs: &[(String, String)]) {
    if let Some((_, v)) = pairs.iter().find(|(k, _)| k == "focus") {
        apply_focus(cfg, v);
    }
    for (k, v) in pairs.iter().filter(|(k, _)| k != "focus") {
        apply_key(cfg, k, v);
    }
}

/// Build a config from an optional file body plus explicit env-style
/// overrides — the pure core of [`load_from_env`], testable without
/// touching process env or the global `OnceLock`.
fn build_config(file_text: Option<&str>, env_pairs: &[(String, String)]) -> TrainingConfig {
    let mut cfg = TrainingConfig::default();
    if let Some(text) = file_text {
        apply_layer(&mut cfg, &parse_pairs(text));
    }
    apply_layer(&mut cfg, env_pairs);
    cfg
}

/// The installed config (T1). Null until [`load_from_env`] runs; readers
/// fall back to `DEFAULTS`. An `AtomicPtr` *upgrade* rather than a
/// `OnceLock` — same publish pattern as `Lohalloc::frozen_table` — because
/// a `OnceLock` here had a real startup-ordering bug: in a
/// `#[global_allocator]` binary the language runtime allocates before
/// `main`, and the bandit's per-allocation [`config`] read locked the
/// `OnceLock` to defaults forever, silently ignoring `LOHALLOC_TUNE` /
/// `LOHALLOC_<KEY>` for the whole process. With the pointer upgrade,
/// pre-`main` allocations run on defaults (harmless — a handful of
/// runtime-init call sites) and `main`'s [`load_from_env`] genuinely takes
/// effect for all workload traffic.
static CONFIG_PTR: core::sync::atomic::AtomicPtr<TrainingConfig> =
    core::sync::atomic::AtomicPtr::new(core::ptr::null_mut());

/// Fallback returned by [`config`] before [`load_from_env`] installs one.
static DEFAULTS: TrainingConfig = TrainingConfig::defaults_const();

/// The process-wide training config. Never does I/O: returns what
/// [`load_from_env`] installed, or the defaults if it was never called
/// (yet) — one `Acquire` load either way.
#[inline]
pub fn config() -> &'static TrainingConfig {
    let ptr = CONFIG_PTR.load(std::sync::atomic::Ordering::Acquire);
    if ptr.is_null() {
        &DEFAULTS
    } else {
        // SAFETY: only ever set by `load_from_env` to a `Box::leak`ed
        // TrainingConfig — 'static and immutable once published.
        unsafe { &*ptr }
    }
}

/// Read `LOHALLOC_TUNE` + `LOHALLOC_<KEY>` into a fresh `TrainingConfig`
/// **without** touching the process-wide `OnceLock`. For harness code that
/// needs the intended config regardless of whether the global allocator
/// already locked [`config`] to defaults during startup (see [`config`]'s
/// caveat). Does I/O — same re-entrancy contract as [`load_from_env`].
pub fn load_config_uncached() -> TrainingConfig {
    let file_text =
        std::env::var("LOHALLOC_TUNE")
            .ok()
            .and_then(|path| match std::fs::read_to_string(&path) {
                Ok(text) => Some(text),
                Err(e) => {
                    eprintln!("lohalloc tune: cannot read {path}: {e} — using defaults");
                    None
                }
            });
    let env_pairs: Vec<(String, String)> = KEYS
        .iter()
        .filter_map(|key| {
            let var = format!("LOHALLOC_{}", key.to_uppercase());
            std::env::var(var).ok().map(|v| (key.to_string(), v))
        })
        .collect();
    build_config(file_text.as_deref(), &env_pairs)
}

/// Load the config from `LOHALLOC_TUNE` (optional file) + `LOHALLOC_<KEY>`
/// env overrides and install it as the process-wide config — an *upgrade*:
/// unlike the old `OnceLock` version, this works even after pre-`main`
/// allocations already read [`config`]'s defaults (the
/// `#[global_allocator]` poisoning case — see `CONFIG_PTR`'s doc). Later
/// calls win over earlier ones; each installed config is `Box::leak`ed
/// (bounded by the number of calls, which is once per process in every
/// real caller — the same accepted-leak pattern as `reset_to_training`'s
/// frozen table).
///
/// **Must run inside a bootstrap-guarded context under an interposed
/// allocator** (see the module doc's re-entrancy contract), or from plain
/// `main` in a global-allocator binary.
pub fn load_from_env() -> &'static TrainingConfig {
    let cfg: &'static TrainingConfig = Box::leak(Box::new(load_config_uncached()));
    CONFIG_PTR.store(
        cfg as *const TrainingConfig as *mut TrainingConfig,
        std::sync::atomic::Ordering::Release,
    );
    cfg
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(list: &[(&str, &str)]) -> Vec<(String, String)> {
        list.iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn load_from_env_upgrades_after_config_was_read() {
        // The global-allocator poisoning scenario (T1): config() is read
        // first (as pre-main runtime allocations do), THEN load_from_env
        // runs — the upgrade must take effect. The old OnceLock stayed
        // locked to defaults forever in this exact sequence.
        let before = *config();
        std::env::set_var("LOHALLOC_UCB_C", "1.25");
        let loaded = load_from_env();
        std::env::remove_var("LOHALLOC_UCB_C");
        assert_eq!(loaded.ucb_c, 1.25);
        assert_eq!(
            config().ucb_c,
            1.25,
            "config() must reflect a post-first-read load_from_env (was {before:?})"
        );
        // Restore process-global state for concurrently-running tests
        // (env already cleared, so this reinstalls the defaults).
        let restored = load_from_env();
        assert_eq!(restored.ucb_c, TrainingConfig::default().ucb_c);
    }

    #[test]
    fn defaults_match_pre_step8_constants() {
        let cfg = TrainingConfig::default();
        assert_eq!(cfg.ucb_c, 2.0);
        assert_eq!(cfg.hysteresis, 0.15);
        assert_eq!(cfg.baseline_rewards, [1.0, 0.8, 0.3, 0.9]);
        assert_eq!(cfg.t_ref_ns, 50.0);
        assert_eq!(cfg.frag_weight, 0.0);
        assert_eq!(cfg.freeze_mode, FreezeMode::Ops);
        assert_eq!(cfg.freeze_after, None);
        // The bisect knob's default must stay the certified J5 constant.
        assert_eq!(
            cfg.demote_fraction,
            crate::state::ARENA_DEMOTE_VOLUME_FRACTION
        );
        assert_eq!(cfg.demote_fraction, 0.10);
    }

    #[test]
    fn demote_fraction_parses_and_rejects() {
        let mut cfg = TrainingConfig::default();
        apply_key(&mut cfg, "demote_fraction", "0.0");
        assert_eq!(cfg.demote_fraction, 0.0);
        apply_key(&mut cfg, "demote_fraction", "2.0");
        assert_eq!(cfg.demote_fraction, 2.0);
        // Rejected values keep the current setting.
        apply_key(&mut cfg, "demote_fraction", "-1");
        assert_eq!(cfg.demote_fraction, 2.0);
        apply_key(&mut cfg, "demote_fraction", "nan");
        assert_eq!(cfg.demote_fraction, 2.0);
        apply_key(&mut cfg, "demote_fraction", "junk");
        assert_eq!(cfg.demote_fraction, 2.0);
    }

    #[test]
    fn missing_file_yields_defaults() {
        assert_eq!(build_config(None, &[]), TrainingConfig::default());
    }

    #[test]
    fn file_keys_apply_and_comments_are_skipped() {
        let text = "# a comment\n\nucb_c = 1.4\nhysteresis=0.2\nfreeze_mode=converged\nconverge_stable_n=32\nfreeze_after=5000\n";
        let cfg = build_config(Some(text), &[]);
        assert_eq!(cfg.ucb_c, 1.4);
        assert_eq!(cfg.hysteresis, 0.2);
        assert_eq!(cfg.freeze_mode, FreezeMode::Converged);
        assert_eq!(cfg.converge_stable_n, 32);
        assert_eq!(cfg.freeze_after, Some(5000));
    }

    #[test]
    fn bad_key_and_bad_value_keep_defaults() {
        let text = "no_such_key=7\nucb_c=not_a_number\nfreeze_mode=maybe\nfrag_weight=inf\n";
        let cfg = build_config(Some(text), &[]);
        assert_eq!(cfg, TrainingConfig::default());
    }

    #[test]
    fn focus_presets_expand() {
        let lat = build_config(Some("focus=latency\n"), &[]);
        assert_eq!((lat.t_ref_ns, lat.frag_weight), (50.0, 0.0));
        assert_eq!(
            lat,
            TrainingConfig::default(),
            "latency focus must be behavior-identical to the defaults"
        );

        let thr = build_config(Some("focus=throughput\n"), &[]);
        assert_eq!((thr.t_ref_ns, thr.frag_weight), (200.0, 0.05));

        let bal = build_config(Some("focus=balanced\n"), &[]);
        assert_eq!((bal.t_ref_ns, bal.frag_weight), (100.0, 0.02));
    }

    #[test]
    fn explicit_key_beats_focus_regardless_of_order() {
        // Key BEFORE the preset line in the file must still win.
        let text = "frag_weight=0.5\nfocus=throughput\n";
        let cfg = build_config(Some(text), &[]);
        assert_eq!(cfg.frag_weight, 0.5, "explicit key must override preset");
        assert_eq!(cfg.t_ref_ns, 200.0, "preset still sets untouched knobs");
    }

    #[test]
    fn env_layer_beats_file_layer() {
        let cfg = build_config(
            Some("ucb_c=1.0\nfocus=throughput\n"),
            &pairs(&[("ucb_c", "3.0"), ("focus", "balanced")]),
        );
        assert_eq!(cfg.ucb_c, 3.0, "env key beats file key");
        assert_eq!(cfg.t_ref_ns, 100.0, "env focus beats file focus");
    }

    #[test]
    fn env_key_beats_env_focus() {
        let cfg = build_config(
            None,
            &pairs(&[("focus", "throughput"), ("frag_weight", "0.9")]),
        );
        assert_eq!(cfg.frag_weight, 0.9);
        assert_eq!(cfg.t_ref_ns, 200.0);
    }

    #[test]
    fn global_config_defaults_without_load() {
        // `config()` must never panic or do I/O; in the test process the
        // OnceLock may or may not have been initialized by another test,
        // but it is always *some* valid config.
        let cfg = config();
        assert!(cfg.ucb_c > 0.0);
    }
}
