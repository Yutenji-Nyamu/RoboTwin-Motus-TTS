#!/usr/bin/env bash

GPU_ID="${1:-0}"
INTERVAL="${2:-1}"
OUT_DIR="${3:-logs}"

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/gpu${GPU_ID}_mem_$(date +%Y%m%d_%H%M%S).csv"

echo "timestamp,gpu_index,memory_used_mib,gpu_util_percent,peak_memory_used_mib" > "$LOG"

peak=0

echo "Monitoring GPU $GPU_ID every ${INTERVAL}s"
echo "Log: $LOG"

while true; do
    line=$(nvidia-smi \
        --id="$GPU_ID" \
        --query-gpu=timestamp,index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits)

    mem=$(echo "$line" | awk -F', ' '{print $3}')

    if [ "$mem" -gt "$peak" ]; then
        peak="$mem"
    fi

    echo "$line,$peak" >> "$LOG"
    sleep "$INTERVAL"
done
