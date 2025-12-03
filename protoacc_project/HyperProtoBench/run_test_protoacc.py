import os
import subprocess
import csv
from datetime import datetime
import re

# --- Configurations ---
BENCH_DIRS = [f"bench{i}" for i in range(6)]
RESULT_CSV = "results.csv"
RUNS = 20  

# Log file with run_test_protoacc_yymmdd_hhmmss.log
log_name = "run_test_protoacc_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"

log = open(log_name, "w")
rows = [("bench", "mode", "avg_ser_us", "avg_des_us")]


def log_write(msg: str) -> None:
    print(msg)
    log.write(msg + "\n")


for bench in BENCH_DIRS:
    log_write(f"===== Processing {bench} =====")

    if not os.path.isdir(bench):
        log_write(f"[Warning] Bench dir {bench} does not exist, skip.")
        rows.append((bench, "normal", -1, -1))
        rows.append((bench, "protoacc", -1, -1))
        continue

    os.chdir(bench)

    # --- Run make and make protoacc ---
    log_write("-- Running make --")
    p_make = subprocess.run(["make"], text=True, capture_output=True)
    log_write(p_make.stdout)
    log_write(p_make.stderr)

    if p_make.returncode != 0:
        log_write(f"[Error] make failed in {bench}, skip both modes.")
        rows.append((bench, "normal", -1, -1))
        rows.append((bench, "protoacc", -1, -1))
        os.chdir("..")
        continue

    log_write("-- Running make protoacc --")
    p_proto = subprocess.run(["make", "protoacc"], text=True, capture_output=True)
    log_write(p_proto.stdout)
    log_write(p_proto.stderr)

    protoacc_ok = (p_proto.returncode == 0)

    # --- Run benchmarks ---
    for exe, mode in [("./benchmark_mlir_exec", "normal"),
                      ("./benchmark_mlir_exec_protoacc", "protoacc")]:

        if mode == "protoacc" and not protoacc_ok:
            log_write(f"[Warning] make protoacc failed in {bench}, skip {mode}.")
            rows.append((bench, mode, -1, -1))
            continue

        if not os.path.exists(exe):
            log_write(f"[Warning] {exe} not found in {bench}, skip {mode}.")
            rows.append((bench, mode, -1, -1))
            continue

        ser_times = []
        deser_times = []

        log_write(f"-- Running {exe} ({mode}) {RUNS} times --")

        for i in range(RUNS):
            p_run = subprocess.run([exe], text=True, capture_output=True)
            out = p_run.stdout

            ser = re.search(r"(\d+)\s+us\s+ser", out)
            des = re.search(r"(\d+)\s+us\s+des", out)

            if ser and des:
                ser_times.append(int(ser.group(1)))
                deser_times.append(int(des.group(1)))
            else:
                log_write(f"[Warning] Missing ser/des info in run {i} of {bench}-{mode}")

        if ser_times and deser_times:
            avg_ser = sum(ser_times) / len(ser_times)
            avg_des = sum(deser_times) / len(deser_times)
        else:
            avg_ser = -1
            avg_des = -1

        rows.append((bench, mode, avg_ser, avg_des))
        log_write(f"Averages for {bench}-{mode}: ser={avg_ser}, des={avg_des}")

    os.chdir("..")

# --- Write CSV ---
with open(RESULT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

log_write(f"===== Completed. Results saved to {RESULT_CSV} =====")
log.close()
print(f"Done. Log written to {log_name}")
