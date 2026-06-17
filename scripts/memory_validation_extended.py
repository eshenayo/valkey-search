#!/usr/bin/env python3
"""Extended memory investigation: tracks RSS and counters through each phase.

Expects valkey-server already running on PORT with the search module loaded.
"""

import os
import re
import subprocess
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import redis

PORT = 6399
DIM = 128


def get_pid():
    result = subprocess.run(
        ["pgrep", "-f", f"valkey-server.*:{PORT}"],
        capture_output=True, text=True
    )
    pids = result.stdout.strip().split("\n")
    return int(pids[0]) if pids and pids[0] else None


def get_vmrss_bytes(pid):
    with open(f"/proc/{pid}/status", "r") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


def get_search_mem(r):
    info = r.info("modules")
    return {
        "used": int(info.get("search_used_memory_bytes", 0)),
        "svs_rt": int(info.get("search_svs_runtime_memory_bytes", 0)),
    }


def mib(b):
    return b / (1024 * 1024)


def snapshot(pid, r, label):
    vmrss = get_vmrss_bytes(pid)
    sm = get_search_mem(r)
    return {"label": label, "vmrss": vmrss, "used": sm["used"], "svs_rt": sm["svs_rt"]}


def print_snap(s, prev=None):
    print(f"  [{s['label']}]")
    print(f"    VmRSS:        {mib(s['vmrss']):>8.1f} MiB", end="")
    if prev:
        print(f"  (Δ {mib(s['vmrss'] - prev['vmrss']):+.1f})")
    else:
        print()
    print(f"    used_memory:  {mib(s['used']):>8.1f} MiB", end="")
    if prev:
        print(f"  (Δ {mib(s['used'] - prev['used']):+.1f})")
    else:
        print()
    print(f"    svs_runtime:  {mib(s['svs_rt']):>8.1f} MiB", end="")
    if prev:
        print(f"  (Δ {mib(s['svs_rt'] - prev['svs_rt']):+.1f})")
    else:
        print()
    if prev and (s['vmrss'] - prev['vmrss']) > 0:
        ratio = ((s['used'] - prev['used']) * 100) / (s['vmrss'] - prev['vmrss'])
        print(f"    ratio:        {ratio:>8.0f}%")


def load_vectors(r, start, count, dim):
    """Load `count` vectors starting from doc:start."""
    pipe = r.pipeline(transaction=False)
    for i in range(start, start + count):
        vec = np.random.rand(dim).astype(np.float32).tobytes()
        pipe.hset(f"doc:{i}", mapping={"vec": vec})
        if (i - start + 1) % 1000 == 0:
            pipe.execute()
            pipe = r.pipeline(transaction=False)
    pipe.execute()


def wait_indexed(r, idx_name, target):
    """Wait until num_docs >= target."""
    for _ in range(600):
        info_raw = r.execute_command("FT.INFO", idx_name)
        for i in range(0, len(info_raw), 2):
            k = info_raw[i].decode() if isinstance(info_raw[i], bytes) else str(info_raw[i])
            if k == "num_docs":
                nd = int(info_raw[i + 1])
                if nd >= target:
                    return nd
                break
        time.sleep(1)
    return -1


def main():
    pid = get_pid()
    if not pid:
        print("ERROR: valkey-server not running on port", PORT)
        sys.exit(1)

    r = redis.Redis(host="localhost", port=PORT, decode_responses=False)
    r.execute_command("FLUSHALL")
    time.sleep(1)

    np.random.seed(42)
    snapshots = []

    print("=" * 70)
    print("EXTENDED MEMORY INVESTIGATION")
    print(f"DIM={DIM}, testing incremental behavior")
    print("=" * 70)

    # Stage 0: Baseline
    s = snapshot(pid, r, "empty")
    snapshots.append(s)
    print_snap(s)

    # Stage 1: Load 10K vectors (no index)
    print(f"\n--- Loading 10K vectors (no index) ---")
    load_vectors(r, 0, 10000, DIM)
    time.sleep(1)
    s = snapshot(pid, r, "10K hashes loaded")
    print_snap(s, snapshots[-1])
    snapshots.append(s)

    # Stage 2: Create SVS index (triggers build + flush of 10K)
    print(f"\n--- Creating SVS index (build + flush 10K) ---")
    r.execute_command(
        "FT.CREATE", "idx", "ON", "HASH",
        "PREFIX", "1", "doc:",
        "SCHEMA", "vec", "VECTOR", "SVS", "10",
        "TYPE", "FLOAT32", "DIM", str(DIM),
        "DISTANCE_METRIC", "L2",
        "GRAPH_MAX_DEGREE", "32",
        "CONSTRUCTION_WINDOW_SIZE", "128",
    )
    nd = wait_indexed(r, "idx", 10000)
    time.sleep(2)
    s = snapshot(pid, r, f"index ready (10K, nd={nd})")
    print_snap(s, snapshots[-1])
    snapshots.append(s)

    # Stage 3: Add 10K more vectors (incremental, will trigger FlushBuffer)
    print(f"\n--- Adding 10K more vectors (incremental flush) ---")
    load_vectors(r, 10000, 10000, DIM)
    nd = wait_indexed(r, "idx", 20000)
    time.sleep(2)
    s = snapshot(pid, r, f"after +10K (nd={nd})")
    print_snap(s, snapshots[-1])
    snapshots.append(s)

    # Stage 4: Add another 10K
    print(f"\n--- Adding another 10K (incremental flush) ---")
    load_vectors(r, 20000, 10000, DIM)
    nd = wait_indexed(r, "idx", 30000)
    time.sleep(2)
    s = snapshot(pid, r, f"after +10K (nd={nd})")
    print_snap(s, snapshots[-1])
    snapshots.append(s)

    # Stage 5: Add another 10K
    print(f"\n--- Adding another 10K (incremental flush) ---")
    load_vectors(r, 30000, 10000, DIM)
    nd = wait_indexed(r, "idx", 40000)
    time.sleep(2)
    s = snapshot(pid, r, f"after +10K (nd={nd})")
    print_snap(s, snapshots[-1])
    snapshots.append(s)

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY: CUMULATIVE FROM BASELINE (post-hash-load)")
    print("=" * 70)
    baseline = snapshots[1]  # After hashes loaded, before index
    for s in snapshots[2:]:
        d_rss = s["vmrss"] - baseline["vmrss"]
        d_used = s["used"] - baseline["used"]
        d_svs = s["svs_rt"] - baseline["svs_rt"]
        ratio = (d_used * 100 / d_rss) if d_rss > 0 else 0
        print(f"  {s['label']:<35} ΔRSS={mib(d_rss):>7.1f}  Δused={mib(d_used):>7.1f}  Δsvs_rt={mib(d_svs):>7.1f}  ratio={ratio:.0f}%")

    # Stage 6: Drop index — check memory is properly freed
    print(f"\n--- Dropping index ---")
    pre_drop = snapshot(pid, r, "pre-drop")
    r.execute_command("FT.DROPINDEX", "idx")
    time.sleep(3)
    post_drop = snapshot(pid, r, "post-drop")
    print_snap(post_drop, pre_drop)

    print(f"\n  Reported freed (used):    {mib(pre_drop['used'] - post_drop['used']):.1f} MiB")
    print(f"  Reported freed (svs_rt):  {mib(pre_drop['svs_rt'] - post_drop['svs_rt']):.1f} MiB")
    print(f"  Actual RSS freed:         {mib(pre_drop['vmrss'] - post_drop['vmrss']):.1f} MiB")

    # Cleanup
    r.execute_command("FLUSHALL")
    print("\nDone.")


if __name__ == "__main__":
    main()
