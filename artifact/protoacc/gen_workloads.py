#!/usr/bin/env python3
import numpy as np, os, json

os.makedirs("workloads", exist_ok=True)
meta = []
specs = [
    ("addressbook",  1_000_000),
    ("mapreduce",    2_000_000),
    ("telemetry",    3_000_000),
    ("social",       4_000_000),
    ("trading",      5_000_000),
    ("sensor",       6_000_000),
]
for name, n in specs:
    rng = np.random.default_rng(seed=hash(name) & 0xffffffff)
    x = rng.integers(low=-2**15, high=2**15-1, size=n, dtype=np.int32)
    path = f"workloads/{name}.npy"
    np.save(path, x)
    meta.append({"name":name,"elements":int(n),"bytes":int(x.nbytes), "path":path})

with open("workloads/meta.json","w") as f:
    json.dump(meta, f, indent=2)
print("Generated workloads -> workloads/*.npy")
