#!/usr/bin/env python3
"""Empirical validation of memory accounting on svs-memory-accounting branch.

Expects valkey-server already running on PORT with the search module loaded.
Start with:
  LD_LIBRARY_PATH=.build-release/_deps/svs-src/lib \
    valkey-server --port 6399 --loadmodule .build-release/libsearch.so \
    --save "" --appendonly no --daemonize yes
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
N_VECTORS = 10000
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


def capture_metrics(pid, r, stage_name):
    vmrss = get_vmrss_bytes(pid)
    server_mem = r.info("memory")
    search_mem = r.info("modules")
    return {
        "stage": stage_name,
        "vmrss_bytes": vmrss,
        "server_used_memory": server_mem.get("used_memory", 0),
        "search_used_memory_bytes": int(search_mem.get("search_used_memory_bytes", 0)),
        "search_svs_runtime_memory_bytes": int(search_mem.get("search_svs_runtime_memory_bytes", 0)),
    }


def print_metrics(m):
    print(f"  VmRSS:                    {m['vmrss_bytes'] / (1024*1024):.1f} MiB")
    print(f"  Server used_memory:       {m['server_used_memory'] / (1024*1024):.1f} MiB")
    print(f"  Search used_memory_bytes: {m['search_used_memory_bytes'] / (1024*1024):.1f} MiB")
    print(f"  Search svs_runtime_bytes: {m['search_svs_runtime_memory_bytes'] / (1024*1024):.1f} MiB")


def parse_smaps(pid):
    buckets = {}
    try:
        with open(f"/proc/{pid}/smaps", "r") as f:
            lines = f.readlines()
    except PermissionError:
        return None

    i = 0
    while i < len(lines):
        m = re.match(
            r"^([0-9a-f]+)-([0-9a-f]+)\s+(\S+)\s+\S+\s+\S+\s+\S+\s*(.*)",
            lines[i],
        )
        if m:
            path = m.group(4).strip() or "(anon)"
            if "[heap]" in path:
                category = "[heap]"
            elif "libsvs_runtime" in path:
                category = "libsvs_runtime.so"
            elif "libsearch" in path:
                category = "libsearch.so"
            elif "[stack" in path:
                category = "[stack]"
            elif path == "(anon)":
                category = "(anon mmap)"
            elif ".so" in path:
                category = "other .so"
            else:
                category = "other"

            for j in range(i + 1, min(i + 25, len(lines))):
                if lines[j].startswith("Rss:"):
                    rss_kb = int(lines[j].split()[1])
                    buckets.setdefault(category, 0)
                    buckets[category] += rss_kb * 1024
                    break
        i += 1
    return buckets


def main():
    pid = get_pid()
    if not pid:
        print("ERROR: valkey-server not running on port", PORT)
        sys.exit(1)

    print(f"Connected to valkey-server PID: {pid}")
    r = redis.Redis(host="localhost", port=PORT, decode_responses=False)
    r.execute_command("FLUSHALL")
    time.sleep(1)

    # ===== STAGE 0: EMPTY SERVER =====
    print("\n" + "=" * 60)
    print("STAGE 0: EMPTY SERVER BASELINE")
    print("=" * 60)
    stage0 = capture_metrics(pid, r, "stage0_empty")
    print_metrics(stage0)

    # ===== STAGE 1: LOAD VECTORS =====
    print(f"\n{'=' * 60}")
    print(f"STAGE 1: LOADING {N_VECTORS} VECTORS (DIM={DIM}, NO INDEX)")
    print("=" * 60)

    np.random.seed(42)
    pipe = r.pipeline(transaction=False)
    for i in range(N_VECTORS):
        vec = np.random.rand(DIM).astype(np.float32).tobytes()
        pipe.hset(f"doc:{i}", mapping={"vec": vec})
        if (i + 1) % 1000 == 0:
            pipe.execute()
            pipe = r.pipeline(transaction=False)
            if (i + 1) % 5000 == 0:
                print(f"  Loaded {i+1}/{N_VECTORS}...")
    pipe.execute()
    time.sleep(2)

    stage1 = capture_metrics(pid, r, "stage1_hashes")
    print_metrics(stage1)
    d_vmrss = stage1["vmrss_bytes"] - stage0["vmrss_bytes"]
    d_search = stage1["search_used_memory_bytes"] - stage0["search_used_memory_bytes"]
    print(f"\n  Δ VmRSS:            +{d_vmrss / (1024*1024):.1f} MiB")
    print(f"  Δ search_used_mem:  +{d_search / (1024*1024):.1f} MiB")

    # ===== STAGE 2: CREATE SVS INDEX =====
    print(f"\n{'=' * 60}")
    print("STAGE 2: CREATE SVS INDEX + BACKFILL")
    print("=" * 60)

    r.execute_command(
        "FT.CREATE", "idx",
        "ON", "HASH",
        "PREFIX", "1", "doc:",
        "SCHEMA",
        "vec", "VECTOR", "SVS", "10",
        "TYPE", "FLOAT32",
        "DIM", str(DIM),
        "DISTANCE_METRIC", "L2",
        "GRAPH_MAX_DEGREE", "32",
        "CONSTRUCTION_WINDOW_SIZE", "128",
    )
    print("  FT.CREATE issued, waiting for backfill...")

    for attempt in range(300):
        info_raw = r.execute_command("FT.INFO", "idx")
        info_dict = {}
        for i in range(0, len(info_raw), 2):
            k = info_raw[i].decode() if isinstance(info_raw[i], bytes) else str(info_raw[i])
            info_dict[k] = info_raw[i + 1]

        num_docs = int(info_dict.get("num_docs", 0))
        if num_docs >= N_VECTORS:
            print(f"  Backfill complete! num_docs={num_docs}")
            break
        if (attempt + 1) % 15 == 0:
            print(f"  Indexing... num_docs={num_docs}/{N_VECTORS}")
        time.sleep(1)
    else:
        print(f"  WARNING: timed out, num_docs={num_docs}")

    time.sleep(3)
    stage2 = capture_metrics(pid, r, "stage2_indexed")
    print_metrics(stage2)

    # ===== ANALYSIS =====
    print(f"\n{'=' * 60}")
    print("ANALYSIS: DOUBLE-COUNTING CHECK")
    print("=" * 60)

    d_vmrss_2 = stage2["vmrss_bytes"] - stage1["vmrss_bytes"]
    d_search_2 = stage2["search_used_memory_bytes"] - stage1["search_used_memory_bytes"]
    svs_rt = stage2["search_svs_runtime_memory_bytes"]

    print(f"\n  Index-phase deltas (Stage 1 → 2):")
    print(f"    ΔVmRSS:                +{d_vmrss_2 / (1024*1024):.1f} MiB")
    print(f"    Δsearch_used_memory:   +{d_search_2 / (1024*1024):.1f} MiB")
    print(f"    svs_runtime_memory:     {svs_rt / (1024*1024):.1f} MiB")

    if d_vmrss_2 > 0:
        ratio = (d_search_2 * 100) / d_vmrss_2
        print(f"\n  *** DIAGNOSTIC RATIO: {ratio:.0f}% ***")
        print(f"      (Δsearch_used_memory / ΔVmRSS)")
        if ratio > 150:
            print("\n  RESULT: DOUBLE-COUNTING LIKELY (ratio > 150%)")
        elif ratio > 110:
            print("\n  RESULT: Slightly elevated — minor noise (110-150%)")
        elif ratio >= 50:
            print("\n  RESULT: OK — no double-counting (50-110%)")
        else:
            print("\n  RESULT: LOW — some SVS memory untracked (<50%)")
    else:
        print("\n  WARNING: VmRSS did not increase")

    # ===== SMAPS =====
    print(f"\n{'=' * 60}")
    print("SMAPS DECOMPOSITION")
    print("=" * 60)
    smaps = parse_smaps(pid)
    if smaps:
        print(f"\n  {'Region':<25} {'RSS MiB':>10}")
        print(f"  {'-'*36}")
        for cat in sorted(smaps.keys(), key=lambda k: -smaps[k]):
            mib = smaps[cat] / (1024 * 1024)
            if mib >= 0.1:
                print(f"  {cat:<25} {mib:>10.1f}")
    else:
        print("  Could not read smaps")

    # ===== MODULE VS SERVER =====
    print(f"\n{'=' * 60}")
    print("MODULE VS SERVER BREAKDOWN")
    print("=" * 60)
    server_total = stage2["server_used_memory"]
    module_total = stage2["search_used_memory_bytes"]
    module_heap = module_total - svs_rt

    print(f"\n  Server INFO used_memory:  {server_total / (1024*1024):.1f} MiB")
    print(f"  Search module total:      {module_total / (1024*1024):.1f} MiB")
    print(f"    module heap:            {module_heap / (1024*1024):.1f} MiB")
    print(f"    svs_runtime:            {svs_rt / (1024*1024):.1f} MiB")
    print(f"  Process VmRSS:            {stage2['vmrss_bytes'] / (1024*1024):.1f} MiB")
    print(f"\n  Estimated server-only:    {max(0, server_total - module_heap) / (1024*1024):.1f} MiB")

    # Cleanup
    print(f"\n{'=' * 60}")
    print("CLEANUP")
    print("=" * 60)
    r.execute_command("FT.DROPINDEX", "idx")
    r.execute_command("FLUSHALL")
    print("  Done (server still running).")


if __name__ == "__main__":
    main()
