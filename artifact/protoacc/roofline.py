#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HyperProtoBench Roofline (Throughput vs AI)
  • Y: Performance (GB/s)  – log scale
  • X: Arithmetic Intensity (ops/byte) – log scale
  • 自动保证 Memory Roof 可见，纵轴拉伸，更清晰
"""

import argparse, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

# bench 名字映射
NAME_MAP = {
    "addressbook": "addressbook (bench0)",
    "mapreduce":   "mapreduce (bench1)",
    "telemetry":   "telemetry (bench2)",
    "social":      "social (bench3)",
    "trading":     "trading (bench4)",
    "sensor":      "sensor (bench5)",
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="artifact/protoacc/results/roofline_input.csv")
    p.add_argument("--ceilings", default="artifact/protoacc/results/ceilings.json")
    p.add_argument("--outfile", default="artifact/protoacc/results/roofline.png")
    p.add_argument("--ylim", nargs=2, type=float, default=None)
    p.add_argument("--no-jitter", action="store_true")
    return p.parse_args()

def load(csv_path, ceilings_path):
    df = pd.read_csv(csv_path)
    if "th_gbps" not in df.columns and "throughput_gbps" in df.columns:
        df = df.rename(columns={"throughput_gbps": "th_gbps"})
    ceilings = {}
    if Path(ceilings_path).exists():
        ceilings = json.loads(Path(ceilings_path).read_text())
    return df, ceilings

def tight_ylim(df, bw, user=None):
    if user: 
        lo, hi = user
        return max(min(lo, hi), 1e-3), max(lo, hi)
    y = np.asarray(df["th_gbps"], float)
    y = y[y > 0]
    if y.size == 0:
        return (1e-2, max(10.0, bw or 10.0))
    ylo = np.percentile(y, 5) / 2.0
    yhi = np.percentile(y, 95) * 3.0
    if bw and bw > 0:
        yhi = max(yhi, bw * 1.25)
        ylo = min(ylo, bw / 8.0)
    if yhi <= ylo:
        yhi = ylo * 10
    return max(ylo, 1e-3), yhi

def plot(df, ceilings, outfile, ylim=None, jitter=True):
    plt.figure(figsize=(8, 6), dpi=150)
    x = np.asarray(df["ai"], float)
    y = np.asarray(df["th_gbps"], float)
    names = [NAME_MAP.get(n, n) for n in df["workload"]]
    bw = float(ceilings.get("memory_bandwidth_gbps", 0.0))

    plt.xscale("log"); plt.yscale("log")

    # Memory roof
    if bw > 0:
        plt.hlines(bw, xmin=max(min(x)*0.5, 1e-3), xmax=max(x)*10,
                   colors="k", linewidth=3, label=f"Memory Roof = {bw:.1f} GB/s")

    # 斜率引导线
    if x.size and y.size:
        medx, medy = np.median(x), np.median(y)
        c = medy / medx
        xs = np.logspace(np.log10(max(min(x)*0.5,1e-3)), np.log10(max(x)*10), 100)
        plt.plot(xs, c*xs, "--", color="#f5a142", linewidth=2, alpha=0.9, label="Slope guide")

    # jitter
    xp = x.copy()
    if jitter:
        rng = np.random.default_rng(42)
        xp = x * (1.0 + rng.uniform(-0.05, 0.05, size=x.shape))

    # scatter
    plt.scatter(xp, y, s=45, color="#2f7ed8", zorder=3)
    for i, (xx, yy, name) in enumerate(zip(xp, y, names)):
        plt.text(xx*1.05, yy*1.1, name, fontsize=9, color="#1f4e79")

    ylo, yhi = tight_ylim(df, bw, user=tuple(ylim) if ylim else None)
    # 确保 Roof 可见
    if bw > 0 and yhi < bw * 1.05:
        yhi = bw * 1.2
    plt.ylim(ylo, yhi)
    plt.xlim(max(min(x)/2, 1e-3), max(x)*4)

    plt.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)
    plt.xlabel("Arithmetic Intensity (ops/byte)")
    plt.ylabel("Performance (GB/s)")
    plt.title(f"HyperProtoBench: Software Roofline (bench0–bench5)\nCeiling (memory) = {bw:.1f} GB/s")
    plt.legend(loc="lower right", fontsize=9, frameon=False)
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"Saved {outfile}")

def main():
    args = parse_args()
    df, ceilings = load(args.csv, args.ceilings)
    plot(df, ceilings, args.outfile, ylim=args.ylim, jitter=(not args.no_jitter))

if __name__ == "__main__":
    main()
