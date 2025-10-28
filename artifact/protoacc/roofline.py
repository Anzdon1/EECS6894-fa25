#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Roofline (Throughput vs Arithmetic Intensity) for HyperProtoBench
- Y: Performance (GB/s), log scale
- X: Arithmetic Intensity (ops/byte), log scale
Enhancements:
  * Auto-computed tight y-limits to spread points
  * Optional x-jitter so same-AI workloads don't overlap
  * Reads ceilings.json for memory bandwidth ceiling
Usage:
  python3 artifact/protoacc/roofline.py [--ylim 3 15] [--no-jitter] [--outfile results/roofline.png]
"""
import argparse, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="artifact/protoacc/results/roofline_input.csv")
    p.add_argument("--ceilings", default="artifact/protoacc/results/ceilings.json")
    p.add_argument("--outfile", default="artifact/protoacc/results/roofline.png")
    p.add_argument("--ylim", nargs=2, type=float, default=None, help="Y-axis limits in GB/s")
    p.add_argument("--no-jitter", action="store_true", help="Disable x-axis jitter")
    return p.parse_args()

def load(csv_path, ceilings_path):
    df = pd.read_csv(csv_path)
    if "th_gbps" not in df.columns and "throughput_gbps" in df.columns:
        df = df.rename(columns={"throughput_gbps":"th_gbps"})
    assert {"workload","ai","th_gbps"}.issubset(df.columns), df.columns
    ceilings = {}
    p = Path(ceilings_path)
    if p.exists():
        ceilings = json.loads(p.read_text())
    return df, ceilings

def tight_ylim(df, bw, user=None):
    if user: 
        lo, hi = user
        return max(min(lo, hi), 1e-3), max(lo, hi)
    y = np.asarray(df["th_gbps"], float)
    y = y[y>0]
    if y.size==0:
        return (1e-2, max(10.0, (bw or 10.0)))
    ylo = np.percentile(y, 5)/1.6
    yhi = np.percentile(y, 95)*1.8
    if bw and bw>0:
        yhi = max(yhi, bw*1.25)
        ylo = min(ylo, bw/6.0)
    if yhi<=ylo: yhi=ylo*10
    return max(ylo,1e-3), yhi

def plot(df, ceilings, outfile, ylim=None, jitter=True):
    plt.figure(figsize=(8,6), dpi=140)
    x = np.asarray(df["ai"], float)
    y = np.asarray(df["th_gbps"], float)
    names = df["workload"].astype(str).tolist()
    bw = float(ceilings.get("memory_bandwidth_gbps", 0.0))

    plt.xscale("log"); plt.yscale("log")

    # memory roof
    if bw>0:
        plt.hlines(bw, xmin=max(min(x)*0.5,1e-3), xmax=max(x)*10, colors="k", linewidth=3, label=f"Memory Roof = {bw:.1f} GB/s")

    # visual slope guide through med point
    if x.size and y.size and np.all(x>0) and np.all(y>0):
        medx, medy = float(np.median(x)), float(np.median(y))
        if medx>0 and medy>0:
            c = medy/medx
            xs = np.logspace(np.log10(max(min(x)*0.5,1e-3)), np.log10(max(x)*10), 100)
            plt.plot(xs, c*xs, "--", color="#f5a142", linewidth=2, alpha=0.9, label="Slope guide")

    # jitter to avoid overlap on same AI
    xp = x.copy()
    if jitter:
        rng = np.random.default_rng(42)
        xp = x*(1.0 + rng.uniform(-0.04,0.04,size=x.shape))

    plt.scatter(xp, y, s=38, color="#2f7ed8", zorder=3)
    for i,(xx,yy,name) in enumerate(zip(xp,y,names)):
        ha = "left" if (i%2==0) else "right"
        va = "bottom" if (i%3!=0) else "top"
        plt.text(xx*(1.06 if ha=="left" else 0.94), yy*(1.06 if va=="bottom" else 0.94),
                 name, fontsize=9, color="#1f4e79", ha=ha, va=va)

    ylo,yhi = tight_ylim(df, bw, user=tuple(ylim) if ylim else None)
    plt.ylim(ylo, yhi)
    if x.size:
        plt.xlim(max(min(x)/2.0,1e-3), max(x)*4.0)

    plt.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)
    plt.xlabel("Arithmetic Intensity (ops/byte)")
    plt.ylabel("Performance (GB/s)")
    title = "HyperProtoBench: Software Roofline (Throughput vs AI)"
    if bw>0: title += f"\nCeiling (memory) = {bw:.1f} GB/s"
    plt.title(title); plt.legend(loc="lower right", fontsize=9, frameon=False)
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(outfile, dpi=140, bbox_inches="tight")
    print(f"Saved {outfile}")

def main():
    args = parse_args()
    df, ceilings = load(args.csv, args.ceilings)
    plot(df, ceilings, args.outfile, ylim=args.ylim, jitter=(not args.no_jitter))
if __name__ == "__main__":
    main()
