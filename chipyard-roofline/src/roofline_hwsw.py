#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SW_CSV = ROOT / "results" / "roofline_input.csv"
HW_CSV = ROOT / "results" / "roofline_input_hw.csv"

if not SW_CSV.exists():
  raise SystemExit(f"Missing software CSV: {SW_CSV}")

def load(kind: str, path: Path) -> pd.DataFrame:
  frame = pd.read_csv(path).dropna(subset=["ai"])
  frame["type"] = kind
  frame["gops"] = (frame["th_gbps"].fillna(0.0) / 8.0) * frame["ai"]
  return frame

sw = load("Software", SW_CSV)
if HW_CSV.exists():
  hw = load("Hardware", HW_CSV)
  combined = pd.concat([sw, hw], ignore_index=True)
else:
  combined = sw

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

plt.rcParams.update({"figure.figsize": (10, 7), "figure.dpi": 220, "font.size": 12})
fig, ax = plt.subplots()

ai = np.logspace(-3, 3, 400)
mem_line = mem_bw_gbps * ai
comp_line = np.full_like(ai, comp_peak_gops)
roof = np.minimum(mem_line, comp_line)

ax.loglog(ai, mem_line, "--", label=f"Memory BW = {mem_bw_gbps:.1f} GB/s", zorder=2)
ax.loglog(ai, comp_line, "--", label=f"Compute Peak = {comp_peak_gops:.1f} Gops/s", zorder=2)
ax.loglog(ai, roof, color="black", linewidth=3.0, label="Roofline", zorder=3)

COLORS = {"Software": "#1f77b4", "Hardware": "#2ca02c"}
for label, frame in combined.groupby("type"):
  ax.scatter(
    frame["ai"],
    frame["gops"],
    c=COLORS.get(label, "black"),
    s=90 if label == "Hardware" else 70,
    marker="o",
    edgecolor="white",
    linewidths=0.5,
    label=label,
    zorder=4,
  )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Arithmetic Intensity (ops / byte)")
ax.set_ylabel("Performance (Gops/s)")
ax.set_title("Roofline: Software vs Hardware")
ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.7)

handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys(), loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

fig.tight_layout()
output = ROOT / "results" / "roofline_hw_sw.png"
fig.savefig(output, bbox_inches="tight")
print(f"[OK] roofline_hw_sw.png written -> {output}")
