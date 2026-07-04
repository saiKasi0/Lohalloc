#!/usr/bin/env bash
# Render a Phase 6 bench-report.json into PNG graphs.
#
#   bench/graphs/generate.sh <report.json> <out_dir> [plotter.py]
#
# The optional third arg picks the plotting script (default plot_report.py;
# tune_sweep passes plot_tune.py for its own report shape).
#
# Manages a self-contained Python venv at bench/graphs/.venv (matplotlib only,
# no pandas/seaborn) so the aggregator can shell out to it without touching the
# system Python. Invoked automatically by `aggregate` after it writes the
# report; safe to run by hand too. Exits non-zero on failure — the aggregator
# treats that as a non-fatal warning (the textual report is already complete).
set -euo pipefail

REPORT_JSON="${1:?usage: generate.sh <report.json> <out_dir>}"
OUT_DIR="${2:?usage: generate.sh <report.json> <out_dir>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "python3 not found — cannot render graphs (report is still complete)." >&2
    exit 1
fi

if [ ! -x "$VENV/bin/python" ]; then
    echo "Creating graph venv at $VENV ..." >&2
    "$PYTHON" -m venv "$VENV"
    "$VENV/bin/pip" install --quiet --upgrade pip
    "$VENV/bin/pip" install --quiet -r "$SCRIPT_DIR/requirements.txt"
fi

PLOTTER="${3:-plot_report.py}"
mkdir -p "$OUT_DIR"
exec "$VENV/bin/python" "$SCRIPT_DIR/$PLOTTER" "$REPORT_JSON" "$OUT_DIR"
