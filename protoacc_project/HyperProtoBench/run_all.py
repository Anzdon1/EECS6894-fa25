#!/usr/bin/env python3
import os, subprocess, csv, re, json, time, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import re

# ============ 参数 ============

DRAW_ONLY = ("--draw-only" in sys.argv)
RUN_TIME = 10

ts = time.strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"run_all_{ts}.log"

def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

log(f"===== START RUN at {ts} =====")
log(f"DRAW_ONLY = {DRAW_ONLY}")

# ============ 命令执行，屏蔽所有 likwid ============

def run_cmd(cmd, cwd=None, can_log=True):
    if can_log:
        log(f"[CMD] {cmd} (cwd={cwd})")

    out = subprocess.run(
        cmd, cwd=cwd, shell=True,
        capture_output=True, text=True
    )

    # stdout
    if out.stdout:
        for line in out.stdout.splitlines():
            if can_log:
                log("  [stdout] " + line)

    # stderr 始终打印
    if out.stderr:
        for line in out.stderr.splitlines():
            if can_log:
                log("  [stderr] " + line)

    if out.returncode != 0:
        if can_log:
            log(f"  [ERROR] return code {out.returncode}")

    return out.stdout


# ============ 提取数值 ============

def parse_value(pattern, text):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


# ============ Roofline Limits 读取或测量 ============

LIMIT_FILE = "roofline_limits.json"

if DRAW_ONLY and os.path.exists(LIMIT_FILE):
# if os.path.exists(LIMIT_FILE):
    log("Loading cached roofline_limits.json ...")
    with open(LIMIT_FILE) as f:
        data = json.load(f)
        pi = data["pi"]
        beta = data["beta"]
        ridge = data["ridge"]
else:

    log("Measuring platform with likwid-bench ...")

    pi_output = run_cmd("sudo likwid-bench -t peakflops -w S0:1GB:1", can_log=False)
    beta_output = run_cmd("sudo likwid-bench -t stream -w S0:1GB:1", can_log=False)

    pi_val = parse_value(r"MFlops/s:\s*([0-9\.]+)", pi_output)
    beta_val = parse_value(r"MByte/s:\s*([0-9\.]+)", beta_output)

    if pi_val is None or beta_val is None:
        log("ERROR: Could not parse peakflops or stream output")
        exit(1)

    pi = pi_val * 1e6
    beta = beta_val * 1e6
    ridge = pi / beta

    with open(LIMIT_FILE, "w") as f:
        json.dump({"pi": pi, "beta": beta, "ridge": ridge}, f, indent=4)

log(f"pi={pi}, beta={beta}, ridge={ridge}")

# ============ bench 列表 ============

bench_dirs = [f"bench{i}" for i in range(6) if os.path.isdir(f"bench{i}")]
results = []


# ============ DRAW_ONLY 模式：只画图 ============

CSV_FILE = "roofline_combined.csv"

if DRAW_ONLY:
    if not os.path.exists(CSV_FILE):
        log("ERROR: roofline_combined.csv missing in --draw-only mode.")
        exit(1)

    log("Loading cached roofline_combined.csv ...")
    with open(CSV_FILE) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            bench, t, W, Q, T, I, P = row
            results.append([bench, t, float(W), float(Q), float(T),
                            float(I), float(P)])
else:
    # ============ 正常执行：测量所有 bench ============
    for bench in bench_dirs:
        log(f"===== PROCESS {bench} =====")

        bp = os.path.abspath(bench)

        # ===== normal =====
        log(f"[{bench}] make normal")
        run_cmd("make", cwd=bp)

        runs = []
        for i in range(RUN_TIME):
            log(f"--- Run normal {i} ---")

            perf_f = run_cmd("sudo likwid-perfctr -C 0 -g FLOPS_DP ./benchmark_mlir_exec", cwd=bp, can_log=False)
            perf_m = run_cmd("sudo likwid-perfctr -C 0 -g MEM ./benchmark_mlir_exec", cwd=bp, can_log=False)

            # T = parse_value(r"Runtime \(RDTSC\) \[s\]\s*\|\s*([0-9\.]+)", perf_f)
            F = parse_value(r"DP \[MFLOP/s\]\s*\|\s*([0-9\.]+)", perf_f)
            B = parse_value(r"Memory bandwidth \[MBytes/s\]\s*\|\s*([0-9\.]+)", perf_m)

            output = run_cmd("./benchmark_mlir_exec", cwd=bp, can_log=False)
            search = re.search(r"(\d+)\s+ns per iter", output)
            T = int(search.group(1))*1e-9


            if T and F and B:
                P = F * 1e6
                Q = B * 1e6 * T
                I = (P * T) / Q
                W = P * T
                runs.append((W, Q, T, I, P))

        if runs:
            W, Q, T, I, P = np.mean(np.array(runs), axis=0)
            results.append([bench, "normal", W, Q, T, I, P])

        # ===== protoacc =====
        log(f"[{bench}] make protoacc")
        run_cmd("make protoacc", cwd=bp)

        runs = []
        for i in range(RUN_TIME):
            log(f"--- Run protoacc {i} ---")

            perf_f = run_cmd("sudo likwid-perfctr -C 0 -g FLOPS_DP ./benchmark_mlir_exec_protoacc", cwd=bp, can_log=False)
            perf_m = run_cmd("sudo likwid-perfctr -C 0 -g MEM ./benchmark_mlir_exec_protoacc", cwd=bp, can_log=False)

            # T = parse_value(r"Runtime \(RDTSC\) \[s\]\s*\|\s*([0-9\.]+)", perf_f)
            F = parse_value(r"DP \[MFLOP/s\]\s*\|\s*([0-9\.]+)", perf_f)
            B = parse_value(r"Memory bandwidth \[MBytes/s\]\s*\|\s*([0-9\.]+)", perf_m)

            output = run_cmd("./benchmark_mlir_exec_protoacc", cwd=bp, can_log=False)
            search = re.search(r"(\d+)\s+ns per iter", output)
            T = int(search.group(1))*1e-9

            if T and F and B:
                P = F * 1e6
                Q = B * 1e6 * T
                I = (P * T) / Q
                W = P * T
                runs.append((W, Q, T, I, P))

        if runs:
            W, Q, T, I, P = np.mean(np.array(runs), axis=0)
            results.append([bench, "protoacc", W, Q, T, I, P])


    # ============ 写 CSV ============

    with open(CSV_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bench", "type", "W", "Q", "T", "I", "P"])
        w.writerows(results)

    log("Results written to roofline_combined.csv")


# ============ 绘图 ============

Ispace = np.logspace(-7, 1, 300)
Proot = np.minimum(pi, Ispace * beta)

plt.figure(figsize=(9, 7))
plt.loglog(Ispace, Proot, 'k--', label="Roofline")

def darken(color, factor=0.6):
    r, g, b = to_rgb(color)
    return (r*factor, g*factor, b*factor)

cmap = plt.get_cmap("tab10")
bench_colors = {}

for idx, bench in enumerate(bench_dirs):
    bench_colors[bench] = cmap(idx % 10)

    # normal
    r = [x for x in results if x[0] == bench and x[1] == "normal"]
    if r:
        I, P = r[0][5], r[0][6]
        plt.loglog(I, P, 'o', color=bench_colors[bench])

    # protoacc
    r = [x for x in results if x[0] == bench and x[1] == "protoacc"]
    if r:
        I, P = r[0][5], r[0][6]
        plt.loglog(I, P, '^', color=bench_colors[bench])

# legend
legend_handles = []
legend_labels = []
for bench in bench_dirs:
    legend_handles.append(plt.Line2D([0], [0], marker='o', linestyle='', color=bench_colors[bench]))
    legend_labels.append(f"{bench} normal")
    legend_handles.append(plt.Line2D([0], [0], marker='^', linestyle='', color=darken(bench_colors[bench])))
    legend_labels.append(f"{bench} protoacc")

plt.legend(legend_handles, legend_labels, fontsize=8)
plt.xlabel("Flops/Byte")
plt.ylabel("Flops/s")
plt.grid(True, which='both', linestyle=':')
plt.tight_layout()
plt.savefig("roofline_compare.png", dpi=300)

log("roofline_compare.png generated.")
log("===== END =====")
