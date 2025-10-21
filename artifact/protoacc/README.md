# ProtoAcc Artifact (No-Chipyard Baseline)

This artifact builds a reproducible **software-only** roofline for the ProtoAcc
accelerator using six hyperprotobench-like workloads (synthetic int32 streams).
It measures:

- Bytes processed (input+output)
- Number of encode operations (OPS = number of int32 elements)
- Elapsed wall time (seconds)

Then it generates a **Performance Roofline**:
- X-axis: Arithmetic Intensity (ops/byte)
- Y-axis: Performance (Gb/s)
- Two ceilings: measured memory bandwidth (Gb/s) and measured compute peak (Gb/s-equivalent)

Later, replace the software bench (`bench_workload.py`) with your real
accelerator runs (Chipyard/FPGA). The rest stays the same.

## Quickstart

```bash
cd artifact/protoacc
bash setup_env.sh
python3 gen_workloads.py                  # create six .npy workloads
bash run_bench.sh                         # run N trials per workload -> results/raw_runs.csv
python3 parse_logs.py                     # aggregate to results/roofline_input.csv
python3 roofline.py                       # write results/roofline.png
```

## Replace with hardware (when ready)

Edit `run_bench.sh` and replace:
```
python3 bench_workload.py --input workloads/<name>.npy
```
with your hardware command, but ensure it prints lines:
```
BYTES_IN=<int>
BYTES_OUT=<int>
OPS=<int>
WALL=<float seconds>
```
This keeps the rest of the pipeline unchanged.

## Files

- `setup_env.sh`     : install python deps; measure memory-bandwidth ceiling
- `gen_workloads.py` : generate six int32 workloads (.npy)
- `bench_workload.py`: vectorized zigzag encode; prints metrics
- `run_bench.sh`     : run all 6 workloads N times and log CSV
- `parse_logs.py`    : aggregate CSV to per-workload means
- `roofline.py`      : plot the performance roofline
- `results/`         : outputs (CSV/PNG)
- `workloads/`       : generated .npy files
