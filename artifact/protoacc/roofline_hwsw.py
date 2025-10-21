import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------
# Inputs and defaults
# ----------------------------
root = Path(__file__).resolve().parent
sw_csv = root / "results" / "roofline_input.csv"         # software baseline
hw_csv = root / "results" / "roofline_input_hw.csv"      # hardware results (optional)
ceilings_json = root / "results" / "ceilings.json"       # optional measured memory BW

# Throughput units on Y axis are GB/s (not FLOPs/s).
# Arithmetic Intensity (X) is ops/byte.
MEM_BW_GBPS = 12.0
COMP_PEAK_GBPS = 20.0

# Override memory BW from measured ceiling if available
if ceilings_json.exists():
    try:
        MEM_BW_GBPS = float(json.load(open(ceilings_json))["memory_bandwidth_gbps"])
    except Exception:
        pass

# ----------------------------
# Load data
# ----------------------------
sw = pd.read_csv(sw_csv)
sw["type"] = "Software"
try:
    hw = pd.read_csv(hw_csv)
    hw["type"] = "Hardware"
    data = pd.concat([sw, hw], ignore_index=True)
except FileNotFoundError:
    data = sw

# ----------------------------
# Figure cosmetics
# ----------------------------
plt.rcParams.update({
    "figure.figsize": (10, 7),
    "figure.dpi": 200,
    "font.size": 13,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 12,
    "lines.linewidth": 2.0,
})

fig, ax = plt.subplots()

# ----------------------------
# Roofline envelope and ceilings
# ----------------------------
ai = np.logspace(-3, 3, 400)  # ops/byte
mem_line = MEM_BW_GBPS * ai                # GB/s
comp_line = np.full_like(ai, COMP_PEAK_GBPS)

# Draw ceilings as dashed guides
ax.loglog(ai, mem_line, linestyle="--", color="#FF8C00",
          label=f"Memory BW = {MEM_BW_GBPS:.1f} GB/s", zorder=2)
ax.loglog(ai, comp_line, linestyle="--", color="#808080",
          label=f"Compute Peak = {COMP_PEAK_GBPS:.1f} GB/s", zorder=2)

# Draw the roofline envelope on top
roof = np.minimum(mem_line, comp_line)
ax.loglog(ai, roof, color="black", linewidth=3.0, label="Roofline", zorder=3)

# ----------------------------
# Scatter points
# ----------------------------
colors = {"Software": "#1f77b4", "Hardware": "#2ca02c"}  # blue, green
markers = {"Software": "o", "Hardware": "o"}
sizes = {"Software": 70, "Hardware": 90}

for t, df in data.groupby("type"):
    ax.scatter(df["ai"], df["th_gbps"],
               c=colors.get(t, "black"), s=sizes.get(t, 70),
               marker=markers.get(t, "o"), edgecolor="white", linewidths=0.5,
               label=t, zorder=4)

# ----------------------------
# Optional point labels (de-overlapped)
# ----------------------------
def annotate_points(df, color, ystep=0.03):
    # Sort by throughput and alternate small y-offsets to reduce overlap
    if df.empty:
        return
    df = df.sort_values("th_gbps").reset_index(drop=True)
    signs = np.array([1, -1] * ((len(df) + 1) // 2))[:len(df)]
    for i, row in df.iterrows():
        x = float(row["ai"])
        y = float(row["th_gbps"])
        # Offset in log space: convert a small additive offset on linear scale
        # into a multiplicative factor on log scale
        y_off = y * (1.0 + signs[i] * ystep)
        ax.text(x, y_off, str(row["workload"]),
                color=color, fontsize=9, ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
                zorder=5)

# Toggle this to False if you prefer no inline labels
LABEL_POINTS = True
if LABEL_POINTS:
    annotate_points(sw, color=colors["Software"])
    if "Hardware" in data["type"].unique():
        annotate_points(data[data["type"] == "Hardware"], color=colors["Hardware"])

# ----------------------------
# Axes, grid, legend
# ----------------------------
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Arithmetic Intensity (ops / byte)")
ax.set_ylabel("Performance (GB/s)")
ax.set_title("ProtoAcc Roofline: Software vs Hardware")

ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.7)
# Place legend outside for clarity
leg = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

fig.tight_layout()
out = root / "results" / "roofline_hw_sw.png"
fig.savefig(out, bbox_inches="tight")
print(f"Saved {out}")
