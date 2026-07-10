#!/usr/bin/env bash
# probe_perf.sh — one-shot PMU capability probe.
#
# Answers the ONE question that decides whether `bench-perf` can run here or
# needs a .metal instance: does THIS host's PMU actually expose the cache /
# coherence events bench-perf wants, and does the current paranoid level let
# an unprivileged user read them?
#
# Safe to run anywhere Linux + perf exists. Never provisions anything. Run it
# first on the (cheap, non-metal) box before committing to a costly metal run.
#
#   bash bench/probe_perf.sh
#
# Exit 0 = enough events available; 2 = perf missing; 3 = insufficient PMU.
set -uo pipefail

echo "== perf PMU capability probe =="
uname -mrs
para="$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo '?')"
echo "perf_event_paranoid=$para  (need <=1 for kernel/LLC events; 2 blocks many)"

if ! command -v perf >/dev/null 2>&1; then
    echo "VERDICT: perf NOT INSTALLED — 'sudo apt-get install -y linux-tools-\$(uname -r) linux-tools-common'"
    exit 2
fi

# The events bench-perf attributes MT regressions to. Generic (arch-portable)
# names so this reads the same on Graviton and x86; raw uncore/coherence
# events (Graviton-specific) are a follow-up once these are confirmed present.
EVENTS=(cache-references cache-misses
        L1-dcache-loads L1-dcache-load-misses
        LLC-loads LLC-load-misses)
joined="$(IFS=,; echo "${EVENTS[*]}")"

# A workload with genuine memory traffic so a *supported* counter reads
# non-zero (an unsupported one still reports "<not supported>" regardless).
WORK='awk "BEGIN{s=0;for(i=0;i<30000000;i++)s+=(i*7)%13;print s}" >/dev/null'

echo "== perf stat -e $joined =="
out="$(perf stat -x, -e "$joined" -- bash -c "$WORK" 2>&1)"
echo "$out"
echo "-----"

avail=0; total=${#EVENTS[@]}
for ev in "${EVENTS[@]}"; do
    line="$(echo "$out" | grep -w -- "$ev" | head -1)"
    if   echo "$line" | grep -qi 'not supported'; then echo "  $ev: NOT SUPPORTED (PMU does not expose it)"
    elif echo "$line" | grep -qi 'not counted';   then echo "  $ev: NOT COUNTED (permissions / paranoid=$para)"
    elif [ -n "$line" ];                          then echo "  $ev: OK  [$line]"; avail=$((avail+1))
    else                                               echo "  $ev: (no output line — check perf version)"
    fi
done

echo "== VERDICT: $avail/$total target events available =="
if [ "$avail" -ge 4 ]; then
    echo "  -> This host is ENOUGH for bench-perf MT diagnostics. Run: make bench-perf"
    exit 0
else
    echo "  -> Insufficient PMU exposure on this host."
    echo "     If events are 'NOT COUNTED': 'sudo sysctl kernel.perf_event_paranoid=1' and re-probe."
    echo "     If 'NOT SUPPORTED': the (virtualized) PMU lacks them — use a .metal instance"
    echo "     (e.g. c8g.metal-24xl / c9g.metal if available in-region) for real LLC/coherence."
    exit 3
fi
