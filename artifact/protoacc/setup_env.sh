#!/usr/bin/env bash
set -euo pipefail

# Python deps
python3 - <<'PY'
import sys, subprocess
def pip(p): subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
for p in ["numpy","pandas","matplotlib"]:
    pip(p)
print("python deps ok")
PY

# Optional: measure memory bandwidth ceiling (simple numpy memcpy loop)
python3 - <<'PY'
import numpy as np, time, json, os
n = 200_000_000  # ~0.8GB (int32)
try:
    a = np.random.randint(-2**31, 2**31-1, size=n, dtype=np.int32)
    b = np.empty_like(a)
    t0 = time.perf_counter()
    b[:] = a
    t1 = time.perf_counter()
    gbps = (a.nbytes*8/1e9)/(t1-t0)
    print(f"[CEILING] memory_bandwidth_gbps={gbps:.3f}")
    os.makedirs("results", exist_ok=True)
    with open("results/ceilings.json","w") as f:
        json.dump({"memory_bandwidth_gbps":gbps}, f)
except Exception as e:
    print(f"[CEILING] skipped: {e}")
PY

echo "Env OK"
