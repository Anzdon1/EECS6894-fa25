#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")"; pwd)"

# === Step 0: System deps ======================================================
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
  git build-essential cmake ninja-build autoconf automake autotools-dev \
  python3 python3-venv python3-pip default-jdk \
  device-tree-compiler libmpc-dev libmpfr-dev libgmp-dev gawk bison flex \
  texinfo gperf libtool patchutils zlib1g-dev libexpat-dev \
  libfl-dev libssl-dev numactl wget curl jq

# === Step 1: Python env =======================================================
python3 -m venv "$ROOT/.venv"
source "$ROOT/.venv/bin/activate"
python -m pip install --upgrade pip wheel
python -m pip install numpy pandas matplotlib

# === Step 2: Chipyard + riscv-tools ==========================================
if [[ ! -d "$ROOT/chipyard" ]]; then
  git clone https://github.com/ucb-bar/chipyard.git "$ROOT/chipyard"
fi
cd "$ROOT/chipyard"
./scripts/init-submodules-no-riscv-tools.sh
./scripts/build-toolchains.sh riscv-tools
source env.sh
cd "$ROOT"

# === Step 3: STREAM bandwidth =================================================
mkdir -p "$ROOT/src" "$ROOT/results" "$ROOT/workloads" "$ROOT/bin"
bash "$ROOT/src/run_stream.sh"

# === Step 4: Workloads + Build benchmark =====================================
python3 "$ROOT/src/gen_workloads.py"
riscv64-unknown-elf-gcc -O3 -static -march=rv64gc -mabi=lp64 -mcmodel=medany \
  -DFREQ_HZ=1000000000ULL \
  "$ROOT/src/bench_roofline.c" -o "$ROOT/bin/bench_roofline.riscv"

# === Step 5: Run & Parse ======================================================
bash "$ROOT/src/run_bench.sh"
python3 "$ROOT/src/parse_logs.py"

# === Step 6: Plot & Package ===================================================
python3 "$ROOT/src/roofline.py"
python3 "$ROOT/src/roofline_hwsw.py"
ts="$(date +%Y%m%d_%H%M%S)"
tar -czf "$ROOT/results/roofline_artifacts_${ts}.tar.gz" -C "$ROOT/results" .
echo "[DONE] Results in $ROOT/results/"
ls -lh "$ROOT/results"
