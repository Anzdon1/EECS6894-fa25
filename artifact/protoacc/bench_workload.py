#!/usr/bin/env python3
import argparse, time, numpy as np

def zigzag_encode_int32(x: np.ndarray) -> np.ndarray:
    # ZigZag: (x<<1) ^ (x>>31), in uint32 then cast back (payload size preserved)
    ux = x.astype(np.int32)
    enc = ((ux.astype(np.int64) << 1) ^ (ux.astype(np.int64) >> 31)).astype(np.uint32)
    return enc

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to .npy int32 array")
    args = ap.parse_args()

    arr = np.load(args.input, mmap_mode="r")
    n = arr.size
    bytes_in = arr.nbytes
    t0 = time.perf_counter()
    out = zigzag_encode_int32(np.asarray(arr))
    t1 = time.perf_counter()
    bytes_out = out.nbytes
    ops = int(n)

    print(f"BYTES_IN={bytes_in}")
    print(f"BYTES_OUT={bytes_out}")
    print(f"OPS={ops}")
    print(f"WALL={t1-t0:.6f}")
