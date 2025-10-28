#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "raw_runs.csv"

if not CSV.exists():
    raise SystemExit(f"Missing input CSV: {CSV}")

df = pd.read_csv(CSV)
df["bytes_total"] = df["bytes_in"].fillna(0) + df["bytes_out"].fillna(0)
df["throughput_gbps"] = (df["bytes_total"] * 8 / 1e9) / df["total_time_sec"].replace(0, float("nan"))
df["arith_intensity"] = df["ops"].replace(0, pd.NA) / df["bytes_total"].replace(0, pd.NA)

agg = (
    df.groupby("workload", as_index=False)
      .agg(
          runs=("run_idx", "count"),
          bytes_total=("bytes_total", "mean"),
          ops=("ops", "mean"),
          time_sec=("total_time_sec", "mean"),
          ai=("arith_intensity", "mean"),
          th_gbps=("throughput_gbps", "mean"),
      )
)

out = ROOT / "results" / "roofline_input.csv"
out.parent.mkdir(parents=True, exist_ok=True)
agg.to_csv(out, index=False)

print(f"[OK] wrote {out}")
print(agg)
