#!/usr/bin/env bash
# Decoupled cloud flow, step 2 of 2: fetch results + tear down the run started
# by cloud_provision.sh. Re-runnable and BOUNDED (default ~18 min poll, safely
# under any background-task lifetime limit) — if it's killed, just run it again;
# the terraform self-terminate net guarantees the box never leaks meanwhile.
#
#   infra/cloud_collect.sh            # poll up to POLL_MINUTES, then pull+destroy if done
#   infra/cloud_collect.sh --once     # single probe (no wait); exit 0 done / 10 still-running
#   infra/cloud_collect.sh --destroy  # abandon the run: destroy now, no pull
#
# Exit codes: 0 = done (pulled + destroyed) or destroyed; 10 = still running
# (re-run collect); 1 = error / no run in flight.
set -euo pipefail
# shellcheck source=infra/cloud_lib.sh
. "$(dirname "$0")/cloud_lib.sh"

MODE="wait"
case "${1:-}" in
--once) MODE="once" ;;
--destroy) MODE="destroy" ;;
"") MODE="wait" ;;
*)
    echo "usage: cloud_collect.sh [--once|--destroy]" >&2
    exit 1
    ;;
esac

if [ ! -f "$STATE_FILE" ]; then
    echo "No run in flight ($STATE_FILE missing). Nothing to collect." >&2
    exit 1
fi
# shellcheck disable=SC1090
. "./$STATE_FILE"
IP="$CLOUD_IP"
INSTANCE_TYPE="$CLOUD_INSTANCE_TYPE"
DEST="$CLOUD_DEST"

finish_destroy() {
    echo "== Destroying infra =="
    destroy_infra "$INSTANCE_TYPE" || echo "WARNING: destroy failed — check AWS console"
    rm -f "$STATE_FILE"
}

if [ "$MODE" = destroy ]; then
    echo "== Abandoning run (no pull) =="
    finish_destroy
    exit 0
fi

POLL_MINUTES="${POLL_MINUTES:-18}"
# ~20s cadence → 3 iterations/min. (String compares can't live in `$(( ))`.)
if [ "$MODE" = once ]; then iters=1; else iters=$((POLL_MINUTES * 3)); fi
probe='if [ -f ~/bench.rc ]; then echo "DONE $(cat ~/bench.rc)"; else echo "RUN $(tail -n 1 ~/bench.log 2>/dev/null)"; fi'
last_line=""
gone_streak=0
for ((p = 0; p < iters; p++)); do
    out="$(SSH_TRIES=6 ssh_try "$IP" "$probe" || echo UNREACHABLE)"
    case "$out" in
    "DONE "*)
        rc="${out#DONE }"
        echo "== Remote suite finished, rc=$rc — pulling into $DEST =="
        pull_results "$IP" "$DEST" || echo "  (no remote results directory found)"
        finish_destroy
        echo "== Done. Results in $DEST (remote rc=$rc) =="
        exit 0
        ;;
    UNREACHABLE)
        gone_streak=$((gone_streak + 1))
        echo "  [$(date +%H:%M:%S)] box unreachable (streak $gone_streak)"
        # A sustained outage most likely means the self-terminate net (or a
        # crash) already took the box: reconcile by destroying the leftover SG
        # and clearing state so a fresh provision can start. 6 straight misses
        # (~1 min each with ssh_try's retries) ≈ a genuinely gone instance.
        if [ "$gone_streak" -ge 6 ]; then
            echo "== Box gone for good (self-terminate net?) — results lost, reconciling =="
            finish_destroy
            exit 1
        fi
        ;;
    "RUN "*)
        gone_streak=0
        line="${out#RUN }"
        if [ -n "$line" ] && [ "$line" != "$last_line" ]; then
            echo "  [$(date +%H:%M:%S)] $line"
            last_line="$line"
        fi
        ;;
    esac
    [ "$MODE" = once ] && break
    sleep 20
done

echo "== Still running after this poll window — re-run 'infra/cloud_collect.sh' to keep waiting. =="
echo "   (The self-terminate net protects against any leak in the meantime.)"
exit 10
