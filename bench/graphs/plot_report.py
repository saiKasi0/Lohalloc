#!/usr/bin/env python3
"""Render a Phase 6 bench-report.json into PNG graphs.

Usage:
    plot_report.py <bench-report.json> <out_dir>

Reads the aggregator's report (the `rows` array) and emits grouped-bar charts:

  native-timing-<lang>.png      mean wall-clock ns/invocation, workload x allocator
  native-throughput-<lang>.png  Mops/s (ops / mean time), workload x allocator
  cache-d1-<lang>.png           cachegrind D1 (L1-data) miss rate, workload x allocator
  cache-ll-<lang>.png           cachegrind LL (last-level) miss rate, workload x allocator
  cache-d1-perop-<lang>.png     D1 misses PER OP, workload x allocator — the
                                denominator-immune view (a mode that does less
                                per-op bookkeeping shows a higher *rate* on the
                                same misses; see aggregate.rs's module doc)
  rust-latency-p99.png          lohalloc per-op alloc p99, workload x mode
  rust-latency-p50.png          lohalloc per-op alloc p50, workload x mode
  rust-latency-mean.png         lohalloc per-op alloc mean, workload x mode

Only charts with data are written. matplotlib-only (no pandas); dark palette
mirrors the project's "hardware terminal" aesthetic so the images sit well
next to the rest of the tooling.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — never touch a display
import matplotlib.pyplot as plt  # noqa: E402

# Project palette (see CLAUDE.md "GUI design system").
CANVAS = "#0A0A0A"
INK = "#E5E0D8"
INK_MUTED = "#8A857C"
INK_FAINT = "#3A3833"
# A small categorical cycle; Heat red is reserved for the lohalloc series so it
# reads as "the thing under test" against the muted baselines.
SERIES_COLORS = {
    "system": "#8A857C",
    "jemalloc": "#6E8BA3",
    "mimalloc": "#B3925A",
    "lohalloc-training": "#C7452E",
    "lohalloc-inference": "#FF2E2E",
}
FALLBACK_CYCLE = ["#7FA35A", "#9A6EA3", "#A3906E", "#5AA394", "#A35A7F"]


def _style_axes(ax, title, ylabel):
    ax.set_facecolor(CANVAS)
    ax.set_title(title, color=INK, fontsize=13, family="monospace", pad=14)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=10, family="monospace")
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(INK_FAINT)
    ax.grid(axis="y", color=INK_FAINT, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_family("monospace")


def _color_for(series, idx):
    if series in SERIES_COLORS:
        return SERIES_COLORS[series]
    return FALLBACK_CYCLE[idx % len(FALLBACK_CYCLE)]


def grouped_bar(
    out_path, title, ylabel, categories, series_values, series_errors=None, footnote=None
):
    """categories: ordered x labels. series_values: {series_name: {cat: value}}.
    series_errors (optional): same shape — per-bar stddev, rendered as an
    error bar so run-to-run noise is visible next to the mean.
    footnote (optional): small muted caption under the chart (e.g. the
    clock-tick quantization warning on per-op latency charts)."""
    if not categories or not series_values:
        return False
    series_names = sorted(series_values.keys())
    n_series = len(series_names)
    n_cat = len(categories)
    group_width = 0.8
    bar_width = group_width / max(n_series, 1)

    fig, ax = plt.subplots(figsize=(max(7, n_cat * 1.4), 4.5))
    fig.patch.set_facecolor(CANVAS)

    x_base = list(range(n_cat))
    for si, name in enumerate(series_names):
        offsets = [
            x + (si - (n_series - 1) / 2) * bar_width for x in x_base
        ]
        heights = [series_values[name].get(cat, 0.0) for cat in categories]
        yerr = None
        if series_errors and name in series_errors:
            errs = [series_errors[name].get(cat, 0.0) for cat in categories]
            if any(e > 0 for e in errs):
                yerr = errs
        ax.bar(
            offsets,
            heights,
            bar_width * 0.92,
            label=name,
            color=_color_for(name, si),
            edgecolor=CANVAS,
            linewidth=0.5,
            yerr=yerr,
            error_kw={"ecolor": INK_MUTED, "elinewidth": 0.8, "capsize": 2},
        )

    ax.set_xticks(x_base)
    ax.set_xticklabels(categories, rotation=0)
    _style_axes(ax, title, ylabel)
    legend = ax.legend(
        frameon=False, fontsize=8, labelcolor=INK, prop={"family": "monospace"}
    )
    if legend:
        legend.get_title().set_color(INK)
    fig.tight_layout()
    if footnote:
        fig.subplots_adjust(bottom=0.22)
        fig.text(
            0.01,
            0.02,
            footnote,
            color=INK_MUTED,
            fontsize=7,
            family="monospace",
        )
    fig.savefig(out_path, dpi=130, facecolor=CANVAS)
    plt.close(fig)
    print(f"  wrote {out_path}")
    return True


def series_label(allocator, mode):
    """system/jemalloc/mimalloc collapse to their name; lohalloc keeps its mode
    (training vs inference are genuinely different bars)."""
    if allocator == "lohalloc":
        return f"lohalloc-{mode}"
    return allocator


def collect(rows, source, value_key):
    """-> {lang: (ordered_categories, {series: {cat: value}})} for a given source."""
    by_lang = defaultdict(lambda: (list(), defaultdict(dict)))
    seen_cat = defaultdict(set)
    for r in rows:
        if r.get("source") != source:
            continue
        val = r.get(value_key)
        if val is None:
            continue
        lang = r.get("lang", "?")
        workload = r.get("workload", "?")
        series = series_label(r.get("allocator", "?"), r.get("mode", "?"))
        cats, sv = by_lang[lang]
        if workload not in seen_cat[lang]:
            seen_cat[lang].add(workload)
            cats.append(workload)
        sv[series][workload] = val
    # stabilize category order (entry = (categories, series_values))
    for entry in by_lang.values():
        entry[0].sort()
    return by_lang


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    report_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    report = json.loads(report_path.read_text())
    rows = report.get("rows", [])
    if not rows:
        print("no rows in report — nothing to plot", file=sys.stderr)
        return 0

    wrote_any = False

    # 1. Native timing, per language — with hyperfine's run-to-run stddev as
    # error bars, so "inference vs training" deltas smaller than the noise
    # floor are visibly not signal.
    stddev_by_lang = collect(rows, "native-timing", "stddev_ns")
    for lang, (cats, sv) in collect(rows, "native-timing", "mean_ns").items():
        errors = stddev_by_lang.get(lang, (None, None))[1]
        wrote_any |= grouped_bar(
            out_dir / f"native-timing-{lang}.png",
            f"Native mean latency — {lang}",
            "ns / invocation",
            cats,
            sv,
            series_errors=errors,
        )

    # 1b. Native throughput (same timing data, ops/sec view).
    for lang, (cats, sv) in collect(rows, "native-timing", "throughput_mops").items():
        wrote_any |= grouped_bar(
            out_dir / f"native-throughput-{lang}.png",
            f"Native throughput — {lang}",
            "Mops / s",
            cats,
            sv,
        )

    # 2. Cache miss rates, per language.
    for lang, (cats, sv) in collect(rows, "cachegrind", "d1_miss_rate").items():
        wrote_any |= grouped_bar(
            out_dir / f"cache-d1-{lang}.png",
            f"L1-data (D1) miss rate — {lang}",
            "D1 miss rate",
            cats,
            sv,
        )
    for lang, (cats, sv) in collect(rows, "cachegrind", "ll_miss_rate").items():
        wrote_any |= grouped_bar(
            out_dir / f"cache-ll-{lang}.png",
            f"Last-level (LL) miss rate — {lang}",
            "LL miss rate",
            cats,
            sv,
        )

    # 2b. Absolute D1 misses per workload op — the denominator-immune view
    # (see aggregate.rs's module doc for why the rate alone misleads on
    # training-vs-inference comparisons). Prefer the startup-subtracted
    # `_net` values (ops=1 calibration) when the run captured them; older
    # runs fall back to gross per-op.
    perop_net = collect(rows, "cachegrind", "d1_misses_per_op_net")
    perop_key, perop_suffix = (
        ("d1_misses_per_op_net", " (net of startup)") if perop_net else ("d1_misses_per_op", "")
    )
    for lang, (cats, sv) in collect(rows, "cachegrind", perop_key).items():
        wrote_any |= grouped_bar(
            out_dir / f"cache-d1-perop-{lang}.png",
            f"D1 misses per op{perop_suffix} — {lang}",
            "D1 misses / op",
            cats,
            sv,
        )

    # 3. Rust per-op alloc latency, workload x mode (single language: rust).
    # If any row was measured on a tick-floored clock (Apple Silicon: ~42ns
    # — see aggregate.rs / clockinfo.rs), caption the per-op charts: bars
    # at/near the tick are quantization buckets, not latencies.
    tick_footnote = None
    for r in rows:
        if r.get("source") == "rust-latency" and r.get("quantized"):
            tick = r.get("clock_tick_ns") or 0
            tick_footnote = (
                f"clock tick floor {tick}ns on this machine: bars at/below "
                f"~{3 * tick}ns are quantization buckets, not latencies"
            )
            break
    for lang, (cats, sv) in collect(rows, "rust-latency", "alloc_p99_ns").items():
        wrote_any |= grouped_bar(
            out_dir / "rust-latency-p99.png",
            "Rust per-op alloc p99 (hdrhistogram)",
            "p99 ns",
            cats,
            sv,
            footnote=tick_footnote,
        )
    for lang, (cats, sv) in collect(rows, "rust-latency", "alloc_p50_ns").items():
        wrote_any |= grouped_bar(
            out_dir / "rust-latency-p50.png",
            "Rust per-op alloc p50 (hdrhistogram)",
            "p50 ns",
            cats,
            sv,
            footnote=tick_footnote,
        )
    for lang, (cats, sv) in collect(rows, "rust-latency", "alloc_mean_ns").items():
        wrote_any |= grouped_bar(
            out_dir / "rust-latency-mean.png",
            "Rust per-op alloc mean (hdrhistogram)",
            "mean ns",
            cats,
            sv,
            footnote=tick_footnote,
        )

    if not wrote_any:
        print("no plottable rows found (all values null?)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
