import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sw_csv = "results/roofline_input.csv"
hw_csv = "results/roofline_input_hw.csv"

MEM_BW = 17.2
COMP_PEAK = 220

sw = pd.read_csv(sw_csv)
sw["type"] = "Software"
try:
    hw = pd.read_csv(hw_csv)
    hw["type"] = "Hardware"
    data = pd.concat([sw, hw])
except FileNotFoundError:
    print("Warning: no hardware data file found, plotting software only.")
    data = sw

plt.figure(figsize=(8, 6))
ai = np.logspace(-3, 3, 100)
bw_line = MEM_BW * ai
plt.loglog(ai, bw_line, '--', label=f"Memory BW = {MEM_BW:.1f} GB/s", color="red")
plt.hlines(COMP_PEAK, ai[0], ai[-1], linestyles='--', colors='gray',
           label=f"Compute Peak = {COMP_PEAK:.0f} GFLOPs/s")

colors = {"Software": "blue", "Hardware": "green"}
for t, df in data.groupby("type"):
    plt.scatter(df["ai"], df["th_gbps"], label=t, c=colors.get(t, "black"), s=80)

for _, r in sw.iterrows():
    plt.text(r["ai"] * 1.05, r["th_gbps"], r["workload"], fontsize=8, color="blue")

if "Hardware" in data["type"].unique():
    for _, r in data[data["type"] == "Hardware"].iterrows():
        plt.text(r["ai"] * 1.05, r["th_gbps"], r["workload"], fontsize=8, color="green")

plt.xlabel("Arithmetic Intensity (Ops/Byte)")
plt.ylabel("Throughput (GB/s)")
plt.title("ProtoAcc Roofline: Software vs Hardware")
plt.legend()
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.savefig("results/roofline_hw_sw.png", dpi=200)
print("Saved results/roofline_hw_sw.png")
