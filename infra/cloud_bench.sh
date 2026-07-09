#!/usr/bin/env bash
# One-shot ARM-only cloud benchmark run, driven from a dev machine (the CI
# equivalent is .github/workflows/bench.yml, which runs BOTH architectures).
#
#   infra/cloud_bench.sh [instance-type]     # default: c8g.4xlarge
#
# Provisions a single ARM64 EC2 instance (enable_x86=false → 2 billable
# resources: SG + instance), rsyncs the CURRENT WORKING TREE (uncommitted
# changes included — same as CI), runs infra/remote_bench.sh (full `make
# bench` + native + cachegrind suites natively), and pulls the finished run
# back into:
#
#   ./results/<datetime>_<instance-type>/
#
# The instance is ALWAYS destroyed, even on failure/interrupt (EXIT trap) —
# never leave billable EC2 running. Requires: terraform, aws credentials in
# the environment, ~/.ssh/id_ed25519(.pub).
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

INSTANCE_TYPE="${1:-c8g.4xlarge}"
STAMP="$(date +%Y%m%dT%H%M%S)"
DEST="results/${STAMP}_${INSTANCE_TYPE}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
# ServerAlive* keeps the (short) poll connections from wedging on a stalled
# box; ConnectTimeout bounds each attempt so a briefly-down host fails fast
# instead of blocking the poll loop.
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -i "$SSH_KEY")
TF=(terraform -chdir=infra)
TF_VARS=(-var "ssh_public_key=$(cat "$SSH_KEY.pub")"
         -var "arm_instance_type=$INSTANCE_TYPE"
         -var "enable_x86=false")

# Set once SSH is reachable so the cleanup trap can rescue partial results
# from a run that failed midway BEFORE tearing the (billable) instance down.
IP=""

# Best-effort SSH that tolerates a briefly-unreachable box (a mid-run reboot
# or transient network blip — exactly what cost the first two runs). Retries
# SSH_TRIES times before giving up; stdout of the remote command passes
# through. Returns the last attempt's exit status.
ssh_try() {
    local tries="${SSH_TRIES:-1}" i
    for ((i = 1; i <= tries; i++)); do
        if ssh "${SSH_OPTS[@]}" -o ConnectTimeout=10 "ubuntu@$IP" "$@"; then
            return 0
        fi
        [ "$i" -lt "$tries" ] && sleep 10
    done
    return 1
}

# Pull the newest remote run dir (+ the full log) into $DEST. Idempotent:
# safe to call from both the happy path and the rescue trap.
pull_results() {
    [ -n "$IP" ] || return 1
    local remote_run
    remote_run="$(SSH_TRIES=6 ssh_try 'ls -1d ~/lohalloc/results/*/ 2>/dev/null | sort | tail -1' || true)"
    [ -n "$remote_run" ] || return 1
    mkdir -p "$DEST"
    rsync -az -e "ssh ${SSH_OPTS[*]}" "ubuntu@$IP:${remote_run}" "$DEST/" \
        || { echo "WARNING: results rsync failed"; return 1; }
    # The detached run's console log — invaluable when a step misbehaved.
    rsync -az -e "ssh ${SSH_OPTS[*]}" "ubuntu@$IP:~/bench.log" "$DEST/bench.log" 2>/dev/null || true
    return 0
}

rescue_and_destroy() {
    # Only pull here if the happy path didn't already (a run that died before
    # writing its completion sentinel). Never let a failed pull block destroy.
    if [ -n "$IP" ] && [ ! -d "$DEST" ]; then
        echo "== Rescuing any remote results into $DEST (pre-destroy) =="
        pull_results || echo "  (no remote results directory found)"
    fi
    echo "== Destroying infra (always) =="
    "${TF[@]}" destroy -auto-approve "${TF_VARS[@]}" || \
        echo "WARNING: destroy failed — check the AWS console for stragglers!"
}
trap rescue_and_destroy EXIT

echo "== Provisioning $INSTANCE_TYPE (ARM64, single instance) =="
"${TF[@]}" init -input=false >/dev/null
"${TF[@]}" apply -auto-approve "${TF_VARS[@]}"

IP="$("${TF[@]}" output -raw arm64_public_ip)"
echo "== Instance up at $IP =="

echo "== Waiting for SSH =="
for i in $(seq 1 60); do
    if ssh "${SSH_OPTS[@]}" -o ConnectTimeout=5 "ubuntu@$IP" true 2>/dev/null; then
        break
    fi
    [ "$i" = 60 ] && { echo "SSH never came up"; exit 1; }
    sleep 5
done

echo "== Rsyncing working tree =="
ssh "${SSH_OPTS[@]}" "ubuntu@$IP" 'mkdir -p ~/lohalloc'
# Exclude every locally-built artifact dir: shipping host-arch (macOS
# Mach-O) binaries under bench/native/build or shim/build makes the
# timestamp-driven `make` treat them as up-to-date and skip the Linux
# rebuild — hyperfine then execs a Mach-O binary on Linux and dies with
# exit 2. `target`/`.git`/`results`/`node_modules` excluded for size.
rsync -az -e "ssh ${SSH_OPTS[*]}" \
    --exclude target --exclude node_modules --exclude .git --exclude results \
    --exclude 'bench/native/build' --exclude 'shim/build' --exclude 'gui/dist' \
    --exclude '*.o' \
    ./ "ubuntu@$IP:~/lohalloc/"

# Ubuntu's unattended-upgrades fired a kernel upgrade + reboot mid-run once and
# took the box (and its results) down with it. Silence the apt auto timers up
# front so nothing reboots under the long suite. Best-effort, before any apt
# work in remote_bench.sh so there's no dpkg-lock contention.
echo "== Quieting apt auto-upgrade timers (prevent mid-run reboot) =="
SSH_TRIES=3 ssh_try 'sudo systemctl disable --now \
    unattended-upgrades apt-daily.timer apt-daily-upgrade.timer \
    apt-daily.service apt-daily-upgrade.service 2>/dev/null; true' || true

# Launch the suite DETACHED (setsid + full fd redirect) so it survives SSH
# drops and runs to completion on its own. `~/bench.rc` appears only when it
# finishes and holds the suite's exit code; `~/bench.log` streams progress.
# This is the fix for both prior data losses: execution is decoupled from this
# SSH session, and completion is a file on disk, not a live channel.
echo "== Launching detached bench suite =="
ssh "${SSH_OPTS[@]}" "ubuntu@$IP" \
    'cd ~/lohalloc && chmod +x infra/remote_bench.sh && rm -f ~/bench.rc ~/bench.log && \
     setsid bash -lc "./infra/remote_bench.sh > ~/bench.log 2>&1; echo \$? > ~/bench.rc" \
        </dev/null >/dev/null 2>&1 & \
     sleep 1; echo "detached launch ok"'

# Poll for the completion sentinel, tolerant of a box that briefly vanishes
# (reboot / blip). Up to ~4h (720 * 20s) — the cachegrind pass alone is slow
# on ARM. One probe per iteration that ALWAYS exits 0 when the box is
# reachable (so ssh_try's retries fire only on a genuine outage, never on the
# normal "not done yet" state): a reachable box prints "DONE <rc>" once the
# sentinel exists, else "RUN <last log line>"; an unreachable box yields
# "UNREACHABLE" only after ssh_try exhausts its reboot-tolerant retries.
echo "== Polling for completion (new tail lines shown as they appear) =="
REMOTE_RC=""
last_line=""
probe='if [ -f ~/bench.rc ]; then echo "DONE $(cat ~/bench.rc)"; else echo "RUN $(tail -n 1 ~/bench.log 2>/dev/null)"; fi'
for ((p = 0; p < 720; p++)); do
    out="$(SSH_TRIES=6 ssh_try "$probe" || echo UNREACHABLE)"
    case "$out" in
    "DONE "*)
        REMOTE_RC="${out#DONE }"
        echo "== Remote suite finished, rc=$REMOTE_RC =="
        break
        ;;
    UNREACHABLE)
        echo "  [$(date +%H:%M:%S)] (box unreachable — retrying)"
        ;;
    "RUN "*)
        line="${out#RUN }"
        if [ -n "$line" ] && [ "$line" != "$last_line" ]; then
            echo "  [$(date +%H:%M:%S)] $line"
            last_line="$line"
        fi
        ;;
    esac
    sleep 20
done

if [ -z "$REMOTE_RC" ]; then
    echo "== Poll window elapsed with no completion sentinel — pulling whatever exists =="
fi

echo "== Pulling results into $DEST =="
pull_results || echo "  (no remote results directory found)"

# EXIT trap destroys the instance ($DEST now exists, so it won't re-pull).
exit "${REMOTE_RC:-1}"
