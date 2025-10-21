#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

mkdir -p results logs
CSV="results/raw_runs.csv"
: > "$CSV"
echo "workload,run_idx,bytes_in,bytes_out,ops,total_time_sec" >> "$CSV"

WORKLOADS=(addressbook mapreduce telemetry social trading sensor)
RUNS=${RUNS:-10}

for w in "${WORKLOADS[@]}"; do
  for i in $(seq 1 "$RUNS"); do
    echo ">> [$w] run $i/$RUNS"
    LOG="logs/${w}_${i}.log"
    set +e
    python3 bench_workload.py --input "workloads/${w}.npy" > "$LOG" 2>&1
    rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
      echo "warn: bench failed: $w run $i (rc=$rc)"
    fi
    BI=$(grep -oP 'BYTES_IN=\K[0-9]+' "$LOG" || echo 0)
    BO=$(grep -oP 'BYTES_OUT=\K[0-9]+' "$LOG" || echo 0)
    OP=$(grep -oP 'OPS=\K[0-9]+' "$LOG" || echo 0)
    TM=$(grep -oP 'WALL=\K[0-9. ]+' "$LOG" | tail -1 | tr -d ' ' || echo 0)
    echo "$w,$i,$BI,$BO,$OP,$TM" >> "$CSV"
  done
done

echo "Wrote $CSV"
