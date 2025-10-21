#!/usr/bin/env python3
import pandas as pd, numpy as np, json
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(__file__).resolve().parent
df = pd.read_csv(root/"results"/"roofline_input.csv")

mem_bw = 12.0  # fallback (GB/s)
ceil = root/"results"/"ceilings.json"
if ceil.exists():
    try:
        mem_bw = float(json.load(open(ceil))["memory_bandwidth_gbps"])
    except Exception:
        pass

comp_peak = 20.0

ai = np.logspace(-3, 3, 200)
perf_bw = mem_bw * ai
perf_cp = np.ones_like(ai) * comp_peak
roof = np.minimum(perf_bw, perf_cp)

plt.figure(figsize=(8,6))
plt.loglog(ai, roof, label="Roofline", linewidth=2)
plt.loglog(ai, perf_bw, "--", label=f"Memory BW ({mem_bw:.1f} GB/s)")
plt.loglog(ai, perf_cp, "--", label=f"Compute Peak ({comp_peak:.1f} GB/s)")

for _, r in df.dropna(subset=["ai","th_gbps"]).iterrows():
    plt.scatter(r["ai"], r["th_gbps"], s=80, label=r["workload"])

plt.xlabel("Arithmetic Intensity (ops / byte)")
plt.ylabel("Performance (GB/s)")
plt.title("ProtoAcc Performance Roofline (Software Baseline)")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.grid(True, which="both", linestyle=":")
plt.tight_layout()
out = root/"results"/"roofline.png"
plt.savefig(out, dpi=200)
print(f"Wrote {out}")
