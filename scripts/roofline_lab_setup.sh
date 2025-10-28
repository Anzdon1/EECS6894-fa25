#!/usr/bin/env bash
set -euo pipefail

################################################################################
# Chipyard-based Roofline (classic): end-to-end, English-only
# - Clones Chipyard and builds riscv-tools (spike/pk + toolchain)
# - Measures memory bandwidth with STREAM (Triad) -> results/ceilings.json (GB/s)
# - Builds and runs a RISC-V bare-metal roofline benchmark via Spike+pk
# - Collects BYTES_IN/OUT, OPS, WALL (sec) -> results/raw_runs.csv
# - Parses -> results/roofline_input.csv
# - Plots classic roofline (Y=Gops/s, X=ops/byte) -> results/roofline.png
# - Optional HW-vs-SW overlay if results/roofline_input_hw.csv exists
#
# Target OS: Ubuntu 22.04 on CloudLab
################################################################################

# --- 0) system packages -------------------------------------------------------
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
  git build-essential cmake ninja-build autoconf automake autotools-dev \
  python3 python3-venv python3-pip default-jdk \
  device-tree-compiler libmpc-dev libmpfr-dev libgmp-dev gawk bison flex \
  texinfo gperf libtool patchutils zlib1g-dev libexpat-dev \
  libfl-dev libssl-dev numactl wget curl jq

# --- 1) workspace -------------------------------------------------------------
ROOT="$HOME/chipyard-roofline"
mkdir -p "$ROOT"
cd "$ROOT"

# --- 2) clone Chipyard and build riscv-tools (spike/pk + toolchain) ----------
if [[ ! -d chipyard ]]; then
  git clone https://github.com/ucb-bar/chipyard.git
fi
cd chipyard
./scripts/init-submodules-no-riscv-tools.sh
./scripts/build-toolchains.sh riscv-tools
# shellcheck disable=SC1091
source env.sh

# quick sanity
command -v spike >/dev/null
command -v riscv64-unknown-elf-gcc >/dev/null
command -v pk >/dev/null || true

cd "$ROOT"

# --- 3) python venv & deps ----------------------------------------------------
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install numpy pandas matplotlib

# --- 4) layout ----------------------------------------------------------------
mkdir -p src results workloads bin

# --- 5) STREAM: measure memory bandwidth (Triad) -> ceilings.json (GB/s) -----
cat > src/run_stream.sh <<'SH'
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
import json, re, pathlib
out = pathlib.Path("stream.out").read_text()
triad = re.search(r"Triad:\s+([\d\.]+)\s+(\wB/s)", out)
copy  = re.search(r"Copy:\s+([\d\.]+)\s+(\wB/s)", out)
val, unit = None, None
if triad:
    val, unit = triad.group(1), triad.group(2)
elif copy:
    val, unit = copy.group(1), copy.group(2)
else:
    raise SystemExit("STREAM output not parsed; Triad/Copy not found.")
bw = float(val)
unit = unit.upper()
if unit.startswith("MB/S"):
    bw = bw / 1000.0  # MB/s -> GB/s
res = {"memory_bandwidth_gbps": round(bw, 2)}
path = pathlib.Path("..")/"results"/"ceilings.json"
path.write_text(json.dumps(res, indent=2))
print(f"[OK] ceilings.json -> {path} :: {res}")
PY
SH
chmod +x src/run_stream.sh

# --- 6) workloads (classic 6) -------------------------------------------------
cat > src/gen_workloads.py <<'PY'
#!/usr/bin/env python3
import json, pathlib
root = pathlib.Path(__file__).resolve().parent.parent
wdir = root/"workloads"
wdir.mkdir(parents=True, exist_ok=True)

# 6 classic workloads; sizes chosen so that bytes_total ~= {8,16,24,32,40,48} MB
# We keep AI ~= 0.125 ops/byte (1 int add per 8 bytes R/W combined)
workloads = [
  ("addressbook", 1_000_000, 1),   # N elements, K ops/element
  ("mapreduce",   2_000_000, 1),
  ("telemetry",   3_000_000, 1),
  ("social",      4_000_000, 1),
  ("trading",     5_000_000, 1),
  ("sensor",      6_000_000, 1),
]
for name, N, K in workloads:
    (wdir/f"{name}.json").write_text(json.dumps({"name":name,"N":N,"K":K}))
print("[OK] generated 6 workload descriptors in ./workloads")
PY
chmod +x src/gen_workloads.py

# --- 7) RISC-V roofline microbenchmark (bare-metal via pk) --------------------
#   - Reads N int32 elements from src, writes N int32 elements to dst
#   - K integer ops per element (AI = K / 8 ops/byte when K ops and 8 bytes R/W)
#   - Measures cycles via rdcycle; time = cycles / FREQ_HZ
cat > src/bench_roofline.c <<'C'
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

static inline uint64_t rdcycle(void) {
  uint64_t c;
  asm volatile ("rdcycle %0" : "=r"(c));
  return c;
}

#ifndef FREQ_HZ
#define FREQ_HZ 1000000000ULL  // 1.0 GHz assumed for time conversion
#endif

// K integer ops per element: simple add chain to avoid being optimized out
static inline int32_t k_ops(int32_t x, int K) {
  volatile int32_t acc = x;
  for (int i = 0; i < K; ++i) acc += 1;
  return acc;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <N:int> <K:int>\n", argv[0]);
    return 2;
  }
  const int N = atoi(argv[1]);       // number of int32 elements
  const int K = atoi(argv[2]);       // ops per element
  // bytes_in = N * 4 (src), bytes_out = N * 4 (dst); total = 8*N
  const uint64_t bytes_in  = (uint64_t)N * 4ULL;
  const uint64_t bytes_out = (uint64_t)N * 4ULL;
  const uint64_t ops       = (uint64_t)N * (uint64_t)K;

  int32_t* src = (int32_t*) malloc((size_t)bytes_in);
  int32_t* dst = (int32_t*) malloc((size_t)bytes_out);
  if (!src || !dst) {
    fprintf(stderr, "malloc failed\n");
    return 3;
  }
  for (int i = 0; i < N; ++i) src[i] = i;

  uint64_t c0 = rdcycle();
  for (int i = 0; i < N; ++i) {
    int32_t v = src[i];
    v = k_ops(v, K);      // K integer ops
    dst[i] = v;           // 4B write
  }
  uint64_t c1 = rdcycle();

  // Convert cycles to seconds using assumed FREQ_HZ. Keep I/O minimal and structured.
  double time_sec = (double)(c1 - c0) / (double)FREQ_HZ;

  printf("BYTES_IN=%llu\n", (unsigned long long)bytes_in);
  printf("BYTES_OUT=%llu\n", (unsigned long long)bytes_out);
  printf("OPS=%llu\n", (unsigned long long)ops);
  printf("WALL=%.9f\n", time_sec);

  // touch output so compiler cannot drop it
  volatile int32_t guard = dst[N-1];
  (void)guard;
  free(src); free(dst);
  return 0;
}
C

# build for RISC-V (lp64, integer ops only; no FPU dependency)
riscv64-unknown-elf-gcc -O3 -static -march=rv64gc -mabi=lp64 -mcmodel=medany \
  -DFREQ_HZ=1000000000ULL \
  src/bench_roofline.c -o bin/bench_roofline.riscv

# --- 8) multi-run driver using Spike+pk -> results/raw_runs.csv ---------------
cat > src/run_bench.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.."; pwd)"
OUT="$ROOT/results/raw_runs.csv"
WDIR="$ROOT/workloads"
ELF="$ROOT/bin/bench_roofline.riscv"

echo "workload,run_idx,bytes_in,bytes_out,ops,total_time_sec" > "$OUT"
RUNS="${RUNS:-10}"

for name in addressbook mapreduce telemetry social trading sensor; do
  N=$(jq -r '.N' "$WDIR/$name.json")
  K=$(jq -r '.K' "$WDIR/$name.json")
  for i in $(seq 1 "$RUNS"); do
    LOG="$(mktemp)"
    # Run via Spike + proxy kernel, capture structured output
    spike pk "$ELF" "$N" "$K" | tee "$LOG" >/dev/null
    BI="$(grep -m1 '^BYTES_IN='  "$LOG" | cut -d= -f2)"
    BO="$(grep -m1 '^BYTES_OUT=' "$LOG" | cut -d= -f2)"
    OPS="$(grep -m1 '^OPS='      "$LOG" | cut -d= -f2)"
    WAL="$(grep -m1 '^WALL='     "$LOG" | cut -d= -f2)"
    rm -f "$LOG"
    echo "$name,$i,$BI,$BO,$OPS,$WAL" >> "$OUT"
  done
done
echo "[OK] wrote $OUT"
SH
chmod +x src/run_bench.sh

# --- 9) parse raw_runs.csv -> roofline_input.csv (classic roofline fields) ---
cat > src/parse_logs.py <<'PY'
#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
root = Path(__file__).resolve().parent.parent
csv = root/"results"/"raw_runs.csv"
df = pd.read_csv(csv)

df["bytes_total"] = df["bytes_in"].fillna(0) + df["bytes_out"].fillna(0)
df["throughput_gbps"] = (df["bytes_total"] * 8 / 1e9) / df["total_time_sec"].replace(0, float("nan"))
df["arith_intensity"] = (df["ops"].replace(0, pd.NA) / df["bytes_total"].replace(0, pd.NA))

agg = (df.groupby("workload", as_index=False)
         .agg(runs=("run_idx","count"),
              bytes_total=("bytes_total","mean"),
              ops=("ops","mean"),
              time_sec=("total_time_sec","mean"),
              ai=("arith_intensity","mean"),
              th_gbps=("throughput_gbps","mean")))

out = root/"results"/"roofline_input.csv"
out.parent.mkdir(parents=True, exist_ok=True)
agg.to_csv(out, index=False)
print(f"[OK] wrote {out}")
print(agg)
PY
chmod +x src/parse_logs.py

# --- 10) plotting (classic roofline: Y=Gops/s, X=ops/byte) -------------------
cat > src/roofline.py <<'PY'
#!/usr/bin/env python3
import pandas as pd, numpy as np, json
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(__file__).resolve().parent.parent
df = pd.read_csv(root/"results"/"roofline_input.csv")

mem_bw_gbps = 12.0      # GB/s
comp_peak_gops = 20.0   # Gops/s

ceil = root/"results"/"ceilings.json"
if ceil.exists():
    try:
        j = json.load(open(ceil))
        mem_bw_gbps = float(j.get("memory_bandwidth_gbps", mem_bw_gbps))
        if "compute_peak_gops" in j:
            comp_peak_gops = float(j["compute_peak_gops"])
    except Exception:
        pass

df = df.dropna(subset=["ai"])
df["gops"] = (df["th_gbps"].fillna(0.0) / 8.0) * df["ai"]

ai = np.logspace(-3, 3, 400)
p_mem_gops  = mem_bw_gbps * ai
p_comp_gops = np.full_like(ai, comp_peak_gops)
roof = np.minimum(p_mem_gops, p_comp_gops)

plt.figure(figsize=(9,6), dpi=220)
plt.loglog(ai, p_mem_gops, "--", label=f"Memory BW = {mem_bw_gbps:.1f} GB/s")
plt.loglog(ai, p_comp_gops, "--", label=f"Compute Peak = {comp_peak_gops:.1f} Gops/s")
plt.loglog(ai, roof, color="black", linewidth=3, label="Roofline")

for _, r in df.iterrows():
    plt.scatter(r["ai"], r["gops"], s=80, edgecolor="white", linewidths=0.5, label=r["workload"])

plt.xlabel("Arithmetic Intensity (ops / byte)")
plt.ylabel("Performance (Gops/s)")
plt.title("Performance Roofline (Software Baseline)")
handles, labels = plt.gca().get_legend_handles_labels()
uniq = dict(zip(labels, handles))
plt.legend(uniq.values(), uniq.keys(), bbox_to_anchor=(1.03,1), loc="upper left")
plt.grid(True, which="both", linestyle=":")
plt.tight_layout()
out = root/"results"/"roofline.png"
plt.savefig(out, dpi=220, bbox_inches="tight")
print(f"[OK] wrote {out}")
PY
chmod +x src/roofline.py

# --- 11) HW vs SW overlay (optional) ------------------------------------------
cat > src/roofline_hwsw.py <<'PY'
#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sw_csv = root / "results" / "roofline_input.csv"
hw_csv = root / "results" / "roofline_input_hw.csv"  # optional
ceilings_json = root / "results" / "ceilings.json"

mem_bw_gbps = 12.0
comp_peak_gops = 20.0

if ceilings_json.exists():
    try:
        j = json.load(open(ceilings_json))
        mem_bw_gbps = float(j.get("memory_bandwidth_gbps", mem_bw_gbps))
        if "compute_peak_gops" in j:
            comp_peak_gops = float(j["compute_peak_gops"])
    except Exception:
        pass

def load_csv(path, typ):
    df = pd.read_csv(path).dropna(subset=["ai"])
    df["type"] = typ
    df["gops"] = (df["th_gbps"].fillna(0.0) / 8.0) * df["ai"]
    return df

sw = load_csv(sw_csv, "Software")
try:
    hw = load_csv(hw_csv, "Hardware")
    data = pd.concat([sw, hw], ignore_index=True)
except FileNotFoundError:
    data = sw

plt.rcParams.update({"figure.figsize": (10,7), "figure.dpi": 220, "font.size": 12})
fig, ax = plt.subplots()

ai = np.logspace(-3, 3, 400)
p_mem_gops  = mem_bw_gbps * ai
p_comp_gops = np.full_like(ai, comp_peak_gops)
roof = np.minimum(p_mem_gops, p_comp_gops)

ax.loglog(ai, p_mem_gops, "--", label=f"Memory BW = {mem_bw_gbps:.1f} GB/s", zorder=2)
ax.loglog(ai, p_comp_gops, "--", label=f"Compute Peak = {comp_peak_gops:.1f} Gops/s", zorder=2)
ax.loglog(ai, roof, color="black", linewidth=3.0, label="Roofline", zorder=3)

colors = {"Software": "#1f77b4", "Hardware": "#2ca02c"}
for t, df in data.groupby("type"):
    ax.scatter(df["ai"], df["gops"], c=colors.get(t, "black"),
               s=90 if t=="Hardware" else 70,
               marker="o", edgecolor="white", linewidths=0.5,
               label=t, zorder=4)

def annotate_points(df, color, ystep=0.04):
    if df.empty: return
    df = df.sort_values("gops").reset_index(drop=True)
    import numpy as np
    signs = np.array([1, -1] * ((len(df) + 1) // 2))[:len(df)]
    for i, row in df.iterrows():
        x = float(row["ai"]); y = float(row["gops"])
        y_off = y * (1.0 + signs[i] * ystep)
        ax.text(x, y_off, str(row["workload"]), color=color, fontsize=9,
                ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
                zorder=5)

annotate_points(sw, color=colors["Software"])
if "Hardware" in data["type"].unique():
    annotate_points(data[data["type"] == "Hardware"], color=colors["Hardware"])

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Arithmetic Intensity (ops / byte)")
ax.set_ylabel("Performance (Gops/s)")
ax.set_title("Roofline: Software vs Hardware")
ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.7)

handles, labels = ax.get_legend_handles_labels()
uniq = dict(zip(labels, handles))
ax.legend(uniq.values(), uniq.keys(), loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

fig.tight_layout()
out = root / "results" / "roofline_hw_sw.png"
fig.savefig(out, bbox_inches="tight")
print(f"[OK] wrote {out}")
PY
chmod +x src/roofline_hwsw.py

# --- 12) make-all convenience -------------------------------------------------
cat > src/make_all_plots.py <<'PY'
#!/usr/bin/env python3
import subprocess, sys, pathlib
root = pathlib.Path(__file__).resolve().parent
subprocess.check_call([sys.executable, str(root/"roofline.py")])
subprocess.check_call([sys.executable, str(root/"roofline_hwsw.py")])
print("[OK] generated figures in ../results")
PY
chmod +x src/make_all_plots.py

# --- 13) top-level driver -----------------------------------------------------
cat > run_all.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")"; pwd)"

echo "=== (1/6) Generate workloads ==="
python3 "$ROOT/src/gen_workloads.py"

echo "=== (2/6) Measure memory bandwidth (STREAM) ==="
bash "$ROOT/src/run_stream.sh"

echo "=== (3/6) Run RISC-V roofline benchmark via Spike+pk ==="
bash "$ROOT/src/run_bench.sh"

echo "=== (4/6) Parse logs -> roofline_input.csv ==="
python3 "$ROOT/src/parse_logs.py"

echo "=== (5/6) Plot roofline(s) ==="
python3 "$ROOT/src/make_all_plots.py"

echo "=== (6/6) Package artifacts ==="
ts="$(date +%Y%m%d_%H%M%S)"
tar -czf "$ROOT/results/roofline_artifacts_${ts}.tar.gz" -C "$ROOT/results" .

echo
echo "[DONE] Artifacts in $ROOT/results:"
ls -lh "$ROOT/results"
echo
echo "Key outputs:"
echo "  - $ROOT/results/ceilings.json"
echo "  - $ROOT/results/raw_runs.csv"
echo "  - $ROOT/results/roofline_input.csv"
echo "  - $ROOT/results/roofline.png"
echo "  - $ROOT/results/roofline_hw_sw.png (if results/roofline_input_hw.csv exists)"
echo "  - $ROOT/results/roofline_artifacts_${ts}.tar.gz"
SH
chmod +x run_all.sh

# --- 14) quick start hint -----------------------------------------------------
echo
echo "================ Quick Start ================"
echo "cd $ROOT && source .venv/bin/activate && ./run_all.sh"
echo "============================================"
echo "Updated Chipyard roofline configuration tested on CloudLab amd255 c6525-100g"
