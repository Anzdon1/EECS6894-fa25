#!/usr/bin/env python3
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKLOAD_DIR = ROOT / "workloads"
WORKLOAD_DIR.mkdir(parents=True, exist_ok=True)

SPEC = [
    ("addressbook", 1_000_000, 1),
    ("mapreduce", 2_000_000, 1),
    ("telemetry", 3_000_000, 1),
    ("social", 4_000_000, 1),
    ("trading", 5_000_000, 1),
    ("sensor", 6_000_000, 1),
]

for name, size, ops_per_elem in SPEC:
    payload = {"name": name, "N": size, "K": ops_per_elem}
    (WORKLOAD_DIR / f"{name}.json").write_text(json.dumps(payload))

print("[OK] workloads ready")
