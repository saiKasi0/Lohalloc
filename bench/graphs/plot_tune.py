#!/usr/bin/env python3
"""Render a tune_sweep tune-report.json into PNG graphs.

Usage:
    plot_tune.py <tune-report.json> <out_dir>

Input is tune_sweep's structured report:
    {"inprocess": [{"label", "focus", "workload",
                    "training":  {"p50","p99","mean","mops","rss_bytes"} | null,
                    "inference": {...} | null}, ...],
     "native":    [{"label", "focus", "workload",
                    "training"/"inference": {"mean_ns","stddev_ns"} | null}, ...]}
(a bare in-process list — the pre-native shape — is also accepted).

Per workload, horizontal bar charts over config points (inference run):

  tune-p99-<workload>.png    alloc p99 ns   (lower = better)
  tune-thpt-<workload>.png   measured Mops/s (higher = better)
  tune-rss-<workload>.png    peak RSS MiB   (lower = better; judges frag_weight)
  tune-native-<workload>.png LD_PRELOAD triple inference wall time, ms
                             (lower = better; only when swept with --native)

The always-present `defaults` baseline bar is drawn in Heat red so "did
tuning beat not tuning?" is answerable at a glance. Same palette/venv as
plot_report.py (invoked through generate.sh's third argument).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — never touch a display
import matplotlib.pyplot as plt  # noqa: E402

CANVAS = "#0A0A0A"
INK = "#E5E0D8"
INK_MUTED = "#8A857C"
INK_FAINT = "#3A3833"
HEAT = "#FF2E2E"  # the defaults baseline — "the thing everything is judged against"
BAR = "#6E8BA3"


def _style_axes(ax, title, xlabel):
    ax.set_facecolor(CANVAS)
    ax.set_title(title, color=INK, fontsize=12, family="monospace", pad=12)
    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=9, family="monospace")
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(INK_FAINT)
    ax.grid(axis="x", color=INK_FAINT, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_family("monospace")


def hbar_chart(out_path, title, xlabel, rows):
    """rows: ordered [(label, value, is_defaults)] — horizontal bars, config
    labels on the y axis (they are long: 'focus=x ucb_c=y ...')."""
    rows = [(lbl, v, d) for (lbl, v, d) in rows if v is not None]
    if not rows:
        return False
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = [HEAT if r[2] else BAR for r in rows]

    fig, ax = plt.subplots(figsize=(9, max(2.5, 0.45 * len(rows) + 1.2)))
    fig.patch.set_facecolor(CANVAS)
    y = list(range(len(rows)))
    ax.barh(y, values, 0.7, color=colors, edgecolor=CANVAS, linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # first (best-ranked) row on top
    _style_axes(ax, title, xlabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor=CANVAS)
    plt.close(fig)
    print(f"  wrote {out_path}")
    return True


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    report_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    report = json.loads(report_path.read_text())
    if isinstance(report, list):  # pre-native shape: bare in-process list
        results, native = report, []
    else:
        results = report.get("inprocess", [])
        native = report.get("native", [])
    if not results and not native:
        print("no results in report — nothing to plot", file=sys.stderr)
        return 0

    workloads = []
    for r in results + native:
        if r["workload"] not in workloads:
            workloads.append(r["workload"])

    wrote_any = False
    for workload in workloads:
        group = [r for r in results if r["workload"] == workload]

        def rows(metric, scale=1.0):
            out = []
            for r in group:
                inf = r.get("inference")
                val = inf.get(metric) if inf else None
                out.append(
                    (r["label"], val * scale if val is not None else None,
                     r["label"] == "defaults")
                )
            return out

        wrote_any |= hbar_chart(
            out_dir / f"tune-p99-{workload}.png",
            f"inference alloc p99 — {workload} (lower is better)",
            "p99 ns",
            sorted(rows("p99"), key=lambda r: (r[1] is None, r[1])),
        )
        wrote_any |= hbar_chart(
            out_dir / f"tune-thpt-{workload}.png",
            f"inference measured throughput — {workload} (higher is better)",
            "Mops / s",
            sorted(rows("mops"), key=lambda r: (r[1] is None, -(r[1] or 0))),
        )
        wrote_any |= hbar_chart(
            out_dir / f"tune-rss-{workload}.png",
            f"peak RSS — {workload} (lower is better)",
            "MiB",
            sorted(
                rows("rss_bytes", scale=1.0 / (1024 * 1024)),
                key=lambda r: (r[1] is None, r[1]),
            ),
        )

        native_group = [r for r in native if r["workload"] == workload]
        if native_group:
            native_rows = []
            for r in native_group:
                inf = r.get("inference")
                val = inf["mean_ns"] / 1e6 if inf else None
                native_rows.append((r["label"], val, r["label"] == "defaults"))
            wrote_any |= hbar_chart(
                out_dir / f"tune-native-{workload}.png",
                f"native LD_PRELOAD inference wall time — {workload} (lower is better)",
                "ms / invocation",
                sorted(native_rows, key=lambda r: (r[1] is None, r[1])),
            )

    if not wrote_any:
        print("no plottable rows found (all inference runs failed?)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
