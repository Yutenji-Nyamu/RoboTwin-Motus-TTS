#!/usr/bin/env bash
set -euo pipefail

# =========================
# CONFIG
# =========================
GPU_ID="$1"
THRESHOLD_MB=5
STABLE_SECONDS=10
CHECK_INTERVAL=1

shift
RUN_CMD="$*"

echo "[queue] GPU_ID=${GPU_ID}"
echo "[queue] THRESHOLD_MB=${THRESHOLD_MB}"
echo "[queue] STABLE_SECONDS=${STABLE_SECONDS}"
echo "[queue] RUN_CMD=${RUN_CMD}"
echo "[queue] start waiting at $(date)"

stable_count=0

while true; do
    used_mb=$(nvidia-smi --id="${GPU_ID}" --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -dc '0-9')

    if [ -z "$used_mb" ]; then
        used_mb=999999
    fi

    echo "[queue] $(date '+%F %T') gpu=${GPU_ID} used_mb=${used_mb} stable_count=${stable_count}/${STABLE_SECONDS}"

    if [ "$used_mb" -lt "$THRESHOLD_MB" ]; then
        stable_count=$((stable_count + CHECK_INTERVAL))
    else
        stable_count=0
    fi

    if [ "$stable_count" -ge "$STABLE_SECONDS" ]; then
        echo "[queue] GPU ${GPU_ID} has been idle for ${STABLE_SECONDS}s. Launching at $(date)"
        bash -lc "$RUN_CMD"
        echo "[queue] submitted command at $(date)"
        exit 0
    fi

    sleep "$CHECK_INTERVAL"
done
