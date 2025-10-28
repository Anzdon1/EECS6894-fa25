#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "roofline_input.csv"

if not CSV.exists():
    raise SystemExit(f"Missing roofline input CSV: {CSV}")

df = pd.read_csv(CSV).dropna(subset=["ai"])

mem_bw_gbps = 12.0
comp_peak_gops = 20.0
ceilings = ROOT / "results" / "ceilings.json"
if ceilings.exists():
    try:
        data = json.loads(ceilings.read_text())
        mem_bw_gbps = float(data.get("memory_bandwidth_gbps", mem_bw_gbps))
        if "compute_peak_gops" in data:
            comp_peak_gops = float(data["compute_peak_gops"])
    except Exception:
        pass

df["gops"] = (df["th_gbps"].fillna(0.0) / 8.0) * df["ai"]

ai = np.logspace(-3, 3, 400)
mem_line = mem_bw_gbps * ai
comp_line = np.full_like(ai, comp_peak_gops)
roof = np.minimum(mem_line, comp_line)

plt.figure(figsize=(8, 6), dpi=220)
plt.loglog(ai, mem_line, "--", label=f"Memory BW = {mem_bw_gbps:.1f} GB/s")
plt.loglog(ai, comp_line, "--", label=f"Compute Peak = {comp_peak_gops:.1f} Gops/s")
plt.loglog(ai, roof, color="black", linewidth=3, label="Roofline")

for _, row in df.iterrows():
    plt.scatter(row["ai"], row["gops"], s=80, edgecolor="white", linewidths=0.5, label=row["workload"])

plt.xlabel("Arithmetic Intensity (ops / byte)")
plt.ylabel("Performance (Gops/s)")
plt.title("Roofline (Software Baseline)")
handles, labels = plt.gca().get_legend_handles_labels()
unique = dict(zip(labels, handles))
plt.legend(unique.values(), unique.keys(), bbox_to_anchor=(1.03, 1), loc="upper left")
plt.grid(True, which="both", linestyle=":")
plt.tight_layout()
output = ROOT / "results" / "roofline.png"
plt.savefig(output, dpi=220, bbox_inches="tight")
print(f"[OK] roofline.png written -> {output}")
