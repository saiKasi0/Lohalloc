#!/usr/bin/env bash
# Shared config + helpers for the DECOUPLED cloud benchmark flow
# (cloud_provision.sh + cloud_collect.sh). Sourced, never executed directly.
#
# Why decoupled: cloud_bench.sh is monolithic — it provisions, then POLLS for
# ~30-90 min in one process, then pulls + destroys. When that single long-lived
# local process is killed mid-poll (background-task lifetime limits, a dropped
# session), the detached remote suite keeps running but nothing tears the
# (billable) instance down. The split here makes provisioning one short step
# and collection a re-runnable short/medium step, and the terraform-level
# self-terminate net (var.self_terminate_minutes) guarantees the box dies on
# its own even if every local step is abandoned. No single killable task can
# strand a run.
#
# State handoff between the two scripts is `.cloud_run.env` at the repo root
# (gitignored): the provisioned IP, instance type, dest dir, and stamp.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." # repo root

# --- credentials + region (identical to cloud_bench.sh) ----------------------
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi
export TF_VAR_aws_region="${TF_VAR_aws_region:-${AWS_DEFAULT_REGION:-us-east-1}}"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
    -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -i "$SSH_KEY")
TF=(terraform -chdir=infra)

STATE_FILE=".cloud_run.env"

# Populate the global TFVARS array for a given instance type (ARM-only, single
# instance). A plain global-array setter (not a printer) so this stays portable
# to macOS's bash 3.2 — no `mapfile`. The self-terminate net uses the variable
# default (180 min) unless overridden via TF_VAR_self_terminate_minutes.
TFVARS=()
set_tf_vars() {
    TFVARS=(-var "ssh_public_key=$(cat "$SSH_KEY.pub")"
        -var "arm_instance_type=$1"
        -var "enable_x86=false")
}

# Best-effort SSH tolerant of a briefly-unreachable box (mid-run blip). Retries
# SSH_TRIES times. Returns the last attempt's status; remote stdout passes
# through.
ssh_try() {
    local ip="$1"
    shift
    local tries="${SSH_TRIES:-1}" i
    for ((i = 1; i <= tries; i++)); do
        if ssh "${SSH_OPTS[@]}" -o ConnectTimeout=10 "ubuntu@$ip" "$@"; then
            return 0
        fi
        [ "$i" -lt "$tries" ] && sleep 10
    done
    return 1
}

# Pull the newest remote run dir (+ console log) into $1 (dest). Idempotent.
pull_results() {
    local ip="$1" dest="$2" remote_run
    remote_run="$(SSH_TRIES=6 ssh_try "$ip" 'ls -1d ~/lohalloc/results/*/ 2>/dev/null | sort | tail -1' || true)"
    [ -n "$remote_run" ] || return 1
    mkdir -p "$dest"
    rsync -az -e "ssh ${SSH_OPTS[*]}" "ubuntu@$ip:${remote_run}" "$dest/" \
        || { echo "WARNING: results rsync failed"; return 1; }
    rsync -az -e "ssh ${SSH_OPTS[*]}" "ubuntu@$ip:~/bench.log" "$dest/bench.log" 2>/dev/null || true
    return 0
}

# Destroy the infra for a given instance type. Safe to call repeatedly (a
# no-op when nothing is provisioned, and reconciles a self-terminated instance
# by removing the leftover security group + clearing stale state).
destroy_infra() {
    set_tf_vars "$1"
    "${TF[@]}" destroy -auto-approve "${TFVARS[@]}"
}
