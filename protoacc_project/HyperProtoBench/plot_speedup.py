import csv
import matplotlib.pyplot as plt
import numpy as np

# 读取 results.csv
with open("results.csv") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

benches = []
speedup_des = []

# 计算每个 bench 的 decode 加速比
for b in sorted(set(r["bench"] for r in rows)):
    n = next(r for r in rows if r["bench"] == b and r["mode"] == "normal")
    p = next(r for r in rows if r["bench"] == b and r["mode"] == "protoacc")
    nd = float(n["avg_des_us"])
    pd = float(p["avg_des_us"])
    benches.append(b)
    speedup_des.append(nd / pd)

x = np.arange(len(benches))

plt.figure(figsize=(7, 4))
bars = plt.bar(x, speedup_des, width=0.6)

plt.xticks(x, benches)
plt.yscale("log")  # bench2/bench5 太大，用 log 轴比较好看
plt.ylabel("Decode speedup (normal_des / proto_des)")
plt.title("Decode speedup per bench (log scale)")

# 在柱子上标注数值（例如 3.0x, 40.6x）
for bar, val in zip(bars, speedup_des):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height * 1.05,
        f"{val:.1f}x",
        ha="center",
        va="bottom",
        fontsize=8,
    )

plt.tight_layout()
plt.savefig("decode_speedup.png", dpi=200)
plt.show()
