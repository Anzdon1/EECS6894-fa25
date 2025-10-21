import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
RES.mkdir(parents=True, exist_ok=True)

SW_CSV = RES / "roofline_input.csv"
HW_CSV = RES / "roofline_input_hw.csv"   # optional
CEIL_JSON = RES / "ceilings.json"

# Defaults (can be overridden by ceilings.json for memory BW)
MEM_BW_GBPS = 12.0
COMP_PEAK_GBPS = 20.0

if CEIL_JSON.exists():
    try:
        MEM_BW_GBPS = float(json.load(open(CEIL_JSON))["memory_bandwidth_gbps"])
    except Exception:
        pass

def load_data():
    sw = pd.read_csv(SW_CSV)
    sw["type"] = "Software"
    try:
        hw = pd.read_csv(HW_CSV)
        hw["type"] = "Hardware"
        data = pd.concat([sw, hw], ignore_index=True)
    except FileNotFoundError:
        data = sw
    return sw, data

def plot_roofline_hwsw():
    sw, data = load_data()

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

    ai = np.logspace(-3, 3, 400)
    mem_line = MEM_BW_GBPS * ai
    comp_line = np.full_like(ai, COMP_PEAK_GBPS)
    roof = np.minimum(mem_line, comp_line)

    ax.loglog(ai, mem_line, "--", color="#FF8C00",
              label=f"Memory BW = {MEM_BW_GBPS:.1f} GB/s", zorder=2)
    ax.loglog(ai, comp_line, "--", color="#808080",
              label=f"Compute Peak = {COMP_PEAK_GBPS:.1f} GB/s", zorder=2)
    ax.loglog(ai, roof, color="black", linewidth=3.0, label="Roofline", zorder=3)

    colors = {"Software": "#1f77b4", "Hardware": "#2ca02c"}
    for t, df in data.groupby("type"):
        ax.scatter(df["ai"], df["th_gbps"],
                   c=colors.get(t, "black"), s=90 if t=="Hardware" else 70,
                   marker="o", edgecolor="white", linewidths=0.5,
                   label=t, zorder=4)

    def annotate_points(df, color, ystep=0.04):
        if df.empty: return
        df = df.sort_values("th_gbps").reset_index(drop=True)
        signs = np.array([1, -1] * ((len(df) + 1) // 2))[:len(df)]
        for i, row in df.iterrows():
            x = float(row["ai"])
            y = float(row["th_gbps"])
            y_off = y * (1.0 + signs[i] * ystep)
            ax.text(x, y_off, str(row["workload"]),
                    color=color, fontsize=9, ha="left", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
                    zorder=5)

    annotate_points(sw, color=colors["Software"])
    if "Hardware" in data["type"].unique():
        annotate_points(data[data["type"] == "Hardware"], color=colors["Hardware"])

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (ops / byte)")
    ax.set_ylabel("Performance (GB/s)")
    ax.set_title("ProtoAcc Roofline: Software vs Hardware")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    out = RES / "roofline_hw_sw.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")

def plot_workload_throughput():
    df = pd.read_csv(SW_CSV)
    plt.figure(figsize=(9,5), dpi=200)
    plt.bar(df["workload"], df["th_gbps"], color="#1f77b4")
    plt.ylabel("Throughput (GB/s)")
    plt.title("Workload Throughput (Software Baseline)")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", ls="--", lw=0.5, alpha=0.7)
    plt.tight_layout()
    out = RES / "workload_throughput.png"
    plt.savefig(out, dpi=200)
    print(f"Saved {out}")

def plot_ai_vs_throughput():
    df = pd.read_csv(SW_CSV)
    plt.figure(figsize=(7,5), dpi=200)
    plt.scatter(df["ai"], df["th_gbps"], s=90, color="#1f77b4", edgecolor="white", linewidths=0.5)
    for _, r in df.iterrows():
        plt.text(r["ai"]*1.02, r["th_gbps"]*1.02, r["workload"], fontsize=9)
    plt.xlabel("Arithmetic Intensity (ops / byte)")
    plt.ylabel("Throughput (GB/s)")
    plt.title("Workload Distribution (Software Baseline)")
    plt.grid(True, ls="--", lw=0.5, alpha=0.7)
    plt.tight_layout()
    out = RES / "ai_vs_throughput.png"
    plt.savefig(out, dpi=200)
    print(f"Saved {out}")

if __name__ == "__main__":
    plot_roofline_hwsw()
    plot_workload_throughput()
    plot_ai_vs_throughput()
