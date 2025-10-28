#!/usr/bin/env python3
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

root = Path(__file__).resolve().parent
sw_csv = root / "results" / "roofline_input.csv"
hw_csv = root / "results" / "roofline_input_hw.csv"
ceil   = root / "results" / "ceilings.json"

mem_bw_gbps = 12.0
comp_peak_gops = 20.0
if ceil.exists():
    try:
        j = json.load(open(ceil))
        mem_bw_gbps   = float(j.get("memory_bandwidth_gbps", mem_bw_gbps))
        comp_peak_gops = float(j.get("compute_peak_gops", comp_peak_gops))
    except Exception:
        pass

def load(path, typ):
    df = pd.read_csv(path).dropna(subset=["ai"])
    df["type"] = typ
    df["gops"] = (df["th_gbps"].fillna(0.0) / 8.0) * df["ai"]
    return df

sw = load(sw_csv, "Software")
try:
    hw = load(hw_csv, "Hardware")
    data = pd.concat([sw, hw], ignore_index=True)
except FileNotFoundError:
    data = sw

plt.rcParams.update({"figure.figsize": (10,7), "figure.dpi": 220, "font.size": 12})
fig, ax = plt.subplots()

ai = np.logspace(-3, 3, 400)
p_mem  = mem_bw_gbps * ai
p_comp = np.full_like(ai, comp_peak_gops)
roof   = np.minimum(p_mem, p_comp)

ax.loglog(ai, p_mem,  "--", label=f"Memory BW = {mem_bw_gbps:.1f} GB/s", zorder=2)
ax.loglog(ai, p_comp, "--", label=f"Compute Peak = {comp_peak_gops:.1f} Gops/s", zorder=2)
ax.loglog(ai, roof, "k", lw=3, label="Roofline", zorder=3)

colors = {"Software":"#1f77b4","Hardware":"#2ca02c"}
for t, df in data.groupby("type"):
    ax.scatter(df["ai"], df["gops"], c=colors.get(t,"black"),
               s=90 if t=="Hardware" else 70,
               marker="o", edgecolor="white", linewidths=0.5, label=t, zorder=4)

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Arithmetic Intensity (ops / byte)")
ax.set_ylabel("Performance (Gops/s)")
ax.set_title("Roofline: Software vs Hardware (Classic)")
ax.grid(True, which="both", ls=":", lw=0.6, alpha=0.7)
h,l = ax.get_legend_handles_labels()
u = dict(zip(l,h))
ax.legend(u.values(), u.keys(), loc="upper left", bbox_to_anchor=(1.02,1.0), borderaxespad=0.0)

fig.tight_layout()
out = root / "results" / "roofline_hw_sw.png"
fig.savefig(out, bbox_inches="tight")
print(f"Wrote {out}")
