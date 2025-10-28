#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.."; pwd)"
OUT="$ROOT/results/raw_runs.csv"
WDIR="$ROOT/workloads"
ELF="$ROOT/bin/bench_roofline.riscv"

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required" >&2
  exit 1
fi

if ! command -v spike >/dev/null 2>&1; then
  echo "spike is required" >&2
  exit 1
fi

if [[ ! -x "$ELF" ]]; then
  echo "Missing benchmark ELF at $ELF" >&2
  exit 1
fi

echo "workload,run_idx,bytes_in,bytes_out,ops,total_time_sec" > "$OUT"
RUNS="${RUNS:-5}"

for name in addressbook mapreduce telemetry social trading sensor; do
  N=$(jq -r '.N' "$WDIR/$name.json")
  K=$(jq -r '.K' "$WDIR/$name.json")
  for i in $(seq 1 "$RUNS"); do
    LOG=$(mktemp)
    spike pk "$ELF" "$N" "$K" | tee "$LOG" >/dev/null
    BI=$(grep -m1 '^BYTES_IN='  "$LOG" | cut -d= -f2)
    BO=$(grep -m1 '^BYTES_OUT=' "$LOG" | cut -d= -f2)
    OPS=$(grep -m1 '^OPS='      "$LOG" | cut -d= -f2)
    WAL=$(grep -m1 '^WALL='     "$LOG" | cut -d= -f2)
    rm -f "$LOG"
    echo "$name,$i,$BI,$BO,$OPS,$WAL" >> "$OUT"
  done
done

echo "[OK] raw_runs.csv written"
