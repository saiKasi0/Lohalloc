#!/usr/bin/env bash
# Decoupled cloud flow, step 1 of 2: provision + launch the detached suite,
# then EXIT (no poll, no destroy). Pair with cloud_collect.sh to fetch results
# and tear down. Short-lived by design (~5 min) so it can't be killed mid-run.
#
#   REMOTE_SCRIPT=infra/remote_clamp_ablation.sh infra/cloud_provision.sh c9g.4xlarge
#
# The instance carries a self-terminate net (var.self_terminate_minutes, 180 by
# default) so even if you never run collect, it deprovisions on its own — no
# unbounded leak. Writes .cloud_run.env for cloud_collect.sh.
set -euo pipefail
# shellcheck source=infra/cloud_lib.sh
. "$(dirname "$0")/cloud_lib.sh"

INSTANCE_TYPE="${1:-c9g.4xlarge}"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-infra/remote_bench.sh}"
STAMP="$(date +%Y%m%dT%H%M%S)"
DEST="results/${STAMP}_${INSTANCE_TYPE}"

if [ -f "$STATE_FILE" ]; then
    echo "ERROR: $STATE_FILE exists — a run is already in flight. Run" >&2
    echo "  infra/cloud_collect.sh            # fetch/tear-down the existing run" >&2
    echo "  infra/cloud_collect.sh --destroy  # abandon it" >&2
    exit 1
fi

set_tf_vars "$INSTANCE_TYPE"

echo "== Provisioning $INSTANCE_TYPE (ARM64, self-terminate net armed) =="
"${TF[@]}" init -input=false >/dev/null
"${TF[@]}" apply -auto-approve "${TFVARS[@]}"
IP="$("${TF[@]}" output -raw arm64_public_ip)"
echo "== Instance up at $IP =="

# On ANY failure from here on, tear the box down immediately (it isn't running
# the suite yet, so nothing to salvage) rather than leaving it for the net.
fail_destroy() {
    echo "== Provision failed before launch — destroying =="
    destroy_infra "$INSTANCE_TYPE" || echo "WARNING: destroy failed — check AWS console"
    rm -f "$STATE_FILE"
}
trap fail_destroy ERR

echo "== Waiting for SSH =="
for i in $(seq 1 60); do
    if ssh "${SSH_OPTS[@]}" -o ConnectTimeout=5 "ubuntu@$IP" true 2>/dev/null; then break; fi
    [ "$i" = 60 ] && {
        echo "SSH never came up"
        false
    }
    sleep 5
done

echo "== Rsyncing working tree =="
ssh "${SSH_OPTS[@]}" "ubuntu@$IP" 'mkdir -p ~/lohalloc'
rsync -az -e "ssh ${SSH_OPTS[*]}" \
    --exclude target --exclude node_modules --exclude .git --exclude results \
    --exclude 'bench/native/build' --exclude 'shim/build' --exclude 'gui/dist' \
    --exclude '.env' --exclude '.env.*' --exclude '*.o' \
    ./ "ubuntu@$IP:~/lohalloc/"

echo "== Launching detached bench suite ($REMOTE_SCRIPT) =="
ssh "${SSH_OPTS[@]}" "ubuntu@$IP" \
    "cd ~/lohalloc && chmod +x '$REMOTE_SCRIPT' && rm -f ~/bench.rc ~/bench.log && \
     setsid bash -lc \"./'$REMOTE_SCRIPT' > ~/bench.log 2>&1; echo \\\$? > ~/bench.rc\" \
        </dev/null >/dev/null 2>&1 & \
     sleep 1; echo 'detached launch ok'"

trap - ERR
# Persist the handoff state for cloud_collect.sh.
cat >"$STATE_FILE" <<EOF
CLOUD_IP=$IP
CLOUD_INSTANCE_TYPE=$INSTANCE_TYPE
CLOUD_DEST=$DEST
CLOUD_STAMP=$STAMP
CLOUD_REMOTE_SCRIPT=$REMOTE_SCRIPT
EOF

echo
echo "== Provisioned + launched. IP=$IP =="
echo "   Suite runs detached; self-terminate net will kill the box on its own"
echo "   if abandoned. To fetch results + tear down (re-runnable, short):"
echo "     infra/cloud_collect.sh          # poll up to ~18 min, pull+destroy when done"
echo "     infra/cloud_collect.sh --once   # single probe, no wait"
echo "     infra/cloud_collect.sh --destroy # abandon: destroy now, no pull"
