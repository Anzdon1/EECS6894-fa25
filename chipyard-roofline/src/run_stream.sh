#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
if [[ ! -f stream.c ]]; then
  wget -q https://www.cs.virginia.edu/stream/FTP/Code/stream.c -O stream.c
fi
gcc -O3 -fopenmp -DSTREAM_ARRAY_SIZE=120000000 -DNTIMES=20 stream.c -o stream
export OMP_NUM_THREADS="$(nproc)"
numactl --interleave=all ./stream | tee stream.out

python3 - <<'PY'
import json
import pathlib
import re

out = pathlib.Path("stream.out").read_text()
match = re.search(r"Triad:\s+([\d\.]+)\s+(\wB/s)", out)
if not match:
    match = re.search(r"Copy:\s+([\d\.]+)\s+(\wB/s)", out)
if not match:
    raise SystemExit("STREAM output not parsed; Triad/Copy not found.")
value, unit = match.group(1), match.group(2)
bandwidth = float(value)
if unit.upper().startswith("MB/S"):
    bandwidth /= 1000.0
ceilings = {"memory_bandwidth_gbps": round(bandwidth, 2)}
path = pathlib.Path("..").joinpath("results", "ceilings.json")
path.write_text(json.dumps(ceilings, indent=2))
print(f"[OK] ceilings.json with {bandwidth:.2f} GB/s")
PY
