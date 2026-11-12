#!/usr/bin/env python3
"""
SVS_VAMANA churn benchmark for Valkey Search.

Measures one module build under a delete-and-insert workload at a constant
live count, so two builds (for example before and after a change) can be
compared on the same server binary and hardware. Run it once per module and
compare the JSON outputs.

Phases:
  1. Ingest: pipelined HSETs, then the time until the index has caught up.
  2. Search at rest: QPS, latency and recall@k against exact neighbours.
  3. Churn: each round deletes a fraction of the live keys, inserts as many
     new ones, then updates as many existing keys with new vectors, one
     command at a time, while readers keep searching. Reports write
     throughput and latency, search QPS and latency during the churn,
     memory_bytes and deleted_slots per round, and recall@k after.
  4. Persistence: BGSAVE (duration and copy-on-write bytes), SAVE duration,
     and DEBUG RELOAD (save plus load) duration.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
from typing import Dict, List, Optional

import numpy as np
import valkey

from ft_info_parser import FTInfoParser

INDEX = "svs_bench"
PREFIX = "d:"


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def start_server(server: str, module: str, port: int, work_dir: str,
                 log: str = ""):
    conf = os.path.join(work_dir, "valkey.conf")
    with open(conf, "w") as f:
        f.write(f"port {port}\n")
        f.write('save ""\n')
        f.write(f"dir {work_dir}\n")
        f.write("enable-debug-command yes\n")
        if log:
            f.write(f"logfile {os.path.abspath(log)}\n")
            f.write("loglevel verbose\n")
        module_args = " --log-level verbose" if log else ""
        f.write(f"loadmodule {os.path.abspath(module)}{module_args}\n")
    proc = subprocess.Popen([os.path.abspath(server), conf],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            preexec_fn=os.setsid)
    deadline = time.time() + 15
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("valkey-server exited: " +
                               proc.stderr.read().decode(errors="replace"))
        try:
            if valkey.Valkey(port=port, socket_timeout=1).ping():
                return proc
        except valkey.exceptions.ConnectionError:
            time.sleep(0.1)
    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    raise TimeoutError("valkey-server did not start")


def stop_server(proc):
    if proc is None or proc.poll() is not None:
        return
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def index_info(client) -> FTInfoParser:
    return FTInfoParser(client.execute_command("FT.INFO", INDEX))


def algorithm(client) -> dict:
    return index_info(client).attributes[0]["index"]["algorithm"]


def wait_settled(client, size: int, timeout: float) -> float:
    """Seconds until the index holds `size` vectors with nothing queued."""
    start = time.perf_counter()
    while True:
        info = index_info(client)
        if (info.mutation_queue_size == 0 and
                info.attributes[0]["index"]["size"] == size):
            return time.perf_counter() - start
        if time.perf_counter() - start > timeout:
            raise TimeoutError(f"index did not settle at {size} vectors")
        time.sleep(0.05)


def latency_summary(latencies_ms: List[float]) -> Dict[str, float]:
    if not latencies_ms:
        return {}
    a = np.array(latencies_ms)
    return {"avg": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def knn(client, query: bytes, k: int, ef: Optional[int]) -> List[int]:
    q = f"*=>[KNN {k} @v $q" + (f" EF_RUNTIME {ef}]" if ef else "]")
    reply = client.execute_command(
        "FT.SEARCH", INDEX, q, "NOCONTENT", "LIMIT", "0", str(k),
        "PARAMS", "2", "q", query, "DIALECT", "2")
    return [int(key[len(PREFIX):]) for key in reply[1:]]


class Dataset:
    """Live vectors by key number, and exact neighbours over them."""

    def __init__(self, dim: int, metric: str, seed: int):
        self.dim = dim
        self.metric = metric
        self.rng = np.random.default_rng(seed)
        self.live: Dict[int, np.ndarray] = {}
        self.next_id = 0

    def new_vectors(self, n: int) -> List[int]:
        ids = list(range(self.next_id, self.next_id + n))
        vectors = self.rng.random((n, self.dim), dtype=np.float32)
        for i, v in zip(ids, vectors):
            self.live[i] = v
        self.next_id += n
        return ids

    def sample_live(self, n: int) -> List[int]:
        keys = np.fromiter(self.live.keys(), dtype=np.int64)
        return self.rng.choice(keys, size=n, replace=False).tolist()

    def queries(self, n: int) -> np.ndarray:
        return self.rng.random((n, self.dim), dtype=np.float32)

    def exact_top_k(self, queries: np.ndarray, k: int) -> List[set]:
        ids = np.fromiter(self.live.keys(), dtype=np.int64)
        data = np.stack([self.live[i] for i in ids])
        if self.metric == "L2":
            # -||q - d||^2 without materializing queries x data x dim.
            scores = 2 * queries @ data.T - (data * data).sum(1)[None, :]
        else:
            if self.metric == "COSINE":
                data = data / np.linalg.norm(data, axis=1, keepdims=True)
                queries = queries / np.linalg.norm(queries, axis=1,
                                                   keepdims=True)
            scores = queries @ data.T
        top = np.argpartition(-scores, k, axis=1)[:, :k]
        return [set(ids[row].tolist()) for row in top]


def recall(client, data: Dataset, n_queries: int, k: int, ef) -> float:
    queries = data.queries(n_queries)
    expected = data.exact_top_k(queries, k)
    hits = sum(len(set(knn(client, q.tobytes(), k, ef)) & e)
               for q, e in zip(queries, expected))
    return hits / (k * n_queries)


def search_load(pool, queries: List[bytes], k: int, ef, stop: threading.Event,
                worker: int):
    client = valkey.Valkey(connection_pool=pool)
    latencies, i = [], worker
    while not stop.is_set():
        t0 = time.perf_counter()
        knn(client, queries[i % len(queries)], k, ef)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        i += 1
    return latencies


def run_searches(pool, args, queries, duration=None, stop=None):
    """Readers until `stop` is set (or for `duration` seconds)."""
    stop = stop or threading.Event()
    with concurrent.futures.ThreadPoolExecutor(args.readers) as ex:
        start = time.perf_counter()
        futures = [ex.submit(search_load, pool, queries, args.k,
                             args.ef_runtime, stop, w)
                   for w in range(args.readers)]
        if duration is not None:
            time.sleep(duration)
            stop.set()
        latencies = [lat for f in futures for lat in f.result()]
        elapsed = time.perf_counter() - start
    return {"qps": len(latencies) / elapsed,
            "latency_ms": latency_summary(latencies)}


def write_ops(pool, ops):
    """Runs (command, key, vector) one at a time; returns latencies in ms."""
    client = valkey.Valkey(connection_pool=pool)
    latencies = []
    for command, key, vector in ops:
        t0 = time.perf_counter()
        if command == "DEL":
            client.delete(key)
        else:
            client.hset(key, mapping={"v": vector})
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return latencies


def run_writes(pool, writers: int, ops) -> Dict:
    shards = [ops[w::writers] for w in range(writers)]
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(writers) as ex:
        latencies = [lat for f in [ex.submit(write_ops, pool, s)
                                   for s in shards] for lat in f.result()]
    elapsed = time.perf_counter() - start
    return {"ops": len(ops), "elapsed_sec": elapsed,
            "ops_per_sec": len(ops) / elapsed,
            "latency_ms": latency_summary(latencies)}


def timed(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def bgsave(client) -> Dict:
    start = time.perf_counter()
    client.execute_command("BGSAVE")
    while client.info("persistence")["rdb_bgsave_in_progress"]:
        time.sleep(0.02)
    elapsed = time.perf_counter() - start
    persistence = client.info("persistence")
    if persistence["rdb_last_bgsave_status"] != "ok":
        raise RuntimeError("BGSAVE failed")
    return {"elapsed_sec": elapsed,
            "cow_bytes": persistence.get("rdb_last_cow_size")}


def run_benchmark(args) -> Dict:
    port = args.port or find_free_port()
    work_dir = tempfile.mkdtemp(prefix="svs_churn_bench_")
    proc = pool = None
    try:
        proc = start_server(args.server, args.module, port, work_dir,
                            args.server_log)
        pool = valkey.ConnectionPool(port=port, max_connections=256)
        client = valkey.Valkey(connection_pool=pool)
        params = ["TYPE", "FLOAT32", "DIM", str(args.dim),
                  "DISTANCE_METRIC", args.metric]
        if args.compression != "NONE":
            params += ["COMPRESSION", args.compression]
        client.execute_command(
            "FT.CREATE", INDEX, "PREFIX", "1", PREFIX, "SCHEMA", "v",
            "VECTOR", "SVS_VAMANA", str(len(params)), *params)
        data = Dataset(args.dim, args.metric, args.seed)
        results = {"config": {**vars(args),
                              "module_sha256": sha256_of(args.module)}}

        # 1. Ingest.
        ids = data.new_vectors(args.num_vectors)
        ops = [("HSET", f"{PREFIX}{i}", data.live[i].tobytes()) for i in ids]
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(args.writers) as ex:
            for f in [ex.submit(_pipelined_hset, pool, ops[w::args.writers],
                                args.batch_size)
                      for w in range(args.writers)]:
                f.result()
        sent = time.perf_counter() - start
        caught_up = wait_settled(client, args.num_vectors, args.timeout)
        results["ingest"] = {
            "vectors": args.num_vectors, "send_sec": sent,
            "indexed_sec": sent + caught_up,
            "vectors_per_sec": args.num_vectors / (sent + caught_up),
            "memory_bytes": algorithm(client)["memory_bytes"]}
        print(f"ingest: {results['ingest']}")

        queries = [q.tobytes() for q in data.queries(1000)]

        # 2. Search at rest.
        results["search_at_rest"] = run_searches(
            pool, args, queries, duration=args.search_duration)
        results["search_at_rest"]["recall"] = recall(
            client, data, args.recall_queries, args.k, args.ef_runtime)
        print(f"search at rest: {results['search_at_rest']}")

        # 3. Churn while readers search.
        per_round = int(args.num_vectors * args.churn_fraction)
        rounds, writes = [], []
        stop = threading.Event()
        with concurrent.futures.ThreadPoolExecutor(1) as ex:
            readers = ex.submit(run_searches, pool, args, queries, None, stop)
            churn_start = time.perf_counter()
            for r in range(args.rounds):
                victims = data.sample_live(per_round)
                deletes = run_writes(pool, args.writers,
                                     [("DEL", f"{PREFIX}{i}", None)
                                      for i in victims])
                for i in victims:
                    del data.live[i]
                wait_settled(client, len(data.live), args.timeout)
                after_delete = algorithm(client)
                new_ids = data.new_vectors(per_round)
                inserts = run_writes(pool, args.writers,
                                     [("HSET", f"{PREFIX}{i}",
                                       data.live[i].tobytes())
                                      for i in new_ids])
                wait_settled(client, len(data.live), args.timeout)
                after_insert = algorithm(client)
                updated = data.sample_live(per_round)
                for i, v in zip(updated, data.queries(per_round)):
                    data.live[i] = v
                updates = run_writes(pool, args.writers,
                                     [("HSET", f"{PREFIX}{i}",
                                       data.live[i].tobytes())
                                      for i in updated])
                wait_settled(client, len(data.live), args.timeout)
                after_update = algorithm(client)
                rounds.append({
                    "round": r + 1,
                    "delete": deletes, "insert": inserts, "update": updates,
                    "deleted_slots_after_delete":
                        after_delete.get("deleted_slots"),
                    "deleted_slots_after_insert":
                        after_insert.get("deleted_slots"),
                    "deleted_slots_after_update":
                        after_update.get("deleted_slots"),
                    "memory_bytes": after_update["memory_bytes"]})
                writes.append((deletes, inserts, updates))
                print(f"churn round {r + 1}: memory_bytes="
                      f"{after_update['memory_bytes']} deleted_slots="
                      f"{after_update.get('deleted_slots')} p99 ms: insert="
                      f"{inserts['latency_ms'].get('p99', 0):.1f} update="
                      f"{updates['latency_ms'].get('p99', 0):.1f}")
            churn_elapsed = time.perf_counter() - churn_start
            stop.set()
            search_during = readers.result()
        total_ops = sum(op["ops"] for ops in writes for op in ops)

        def p99_max(kind):
            return max(op["latency_ms"]["p99"] for op in
                       (w[kind] for w in writes))

        def ops_per_sec(kind):
            return (sum(w[kind]["ops"] for w in writes) /
                    sum(w[kind]["elapsed_sec"] for w in writes))

        results["churn"] = {
            "rounds": rounds,
            "write_ops_per_sec": total_ops / churn_elapsed,
            "delete_ops_per_sec": ops_per_sec(0),
            "insert_ops_per_sec": ops_per_sec(1),
            "update_ops_per_sec": ops_per_sec(2),
            "delete_latency_ms_p99_max": p99_max(0),
            "insert_latency_ms_p99_max": p99_max(1),
            "update_latency_ms_p99_max": p99_max(2),
            "memory_bytes_growth":
                rounds[-1]["memory_bytes"] - results["ingest"]["memory_bytes"],
            "search_during_churn": search_during,
            "recall_after": recall(client, data, args.recall_queries, args.k,
                                   args.ef_runtime)}
        print(f"churn: write ops/s={results['churn']['write_ops_per_sec']:.0f}"
              f" memory growth={results['churn']['memory_bytes_growth']}"
              f" recall={results['churn']['recall_after']:.4f}")

        # 4. Persistence.
        before = algorithm(client)
        persistence = {"deleted_slots_before": before.get("deleted_slots"),
                       "bgsave": bgsave(client)}
        persistence["save_sec"] = timed(lambda: client.execute_command("SAVE"))
        persistence["debug_reload_sec"] = timed(
            lambda: client.execute_command("DEBUG", "RELOAD"))
        persistence["settled_after_reload_sec"] = wait_settled(
            client, len(data.live), args.timeout)
        persistence["memory_bytes_after_reload"] = \
            algorithm(client)["memory_bytes"]
        results["persistence"] = persistence
        print(f"persistence: {persistence}")

        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(results, f, indent=2)
            print(f"results written to {args.output_json}")
        return results
    finally:
        if pool is not None:
            pool.disconnect()
        stop_server(proc)
        shutil.rmtree(work_dir, ignore_errors=True)


def _pipelined_hset(pool, ops, batch_size: int):
    pipe = valkey.Valkey(connection_pool=pool).pipeline(transaction=False)
    for n, (_, key, vector) in enumerate(ops, 1):
        pipe.hset(key, mapping={"v": vector})
        if n % batch_size == 0:
            pipe.execute()
    pipe.execute()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--server", required=True, help="valkey-server binary")
    p.add_argument("--module", required=True, help="libsearch.so to measure")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--num_vectors", type=int, default=100000)
    p.add_argument("--metric", default="L2", choices=["L2", "IP", "COSINE"])
    p.add_argument("--compression", default="NONE",
                   choices=["NONE", "FP16", "SQ8"])
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--churn_fraction", type=float, default=0.1,
                   help="share of the live keys deleted and re-inserted "
                        "each round")
    p.add_argument("--writers", type=int, default=8)
    p.add_argument("--readers", type=int, default=8)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--ef_runtime", type=int, default=None)
    p.add_argument("--search_duration", type=float, default=20)
    p.add_argument("--recall_queries", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=500)
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_json", default="")
    p.add_argument("--server_log", default="",
                   help="keep the valkey-server log at this path")
    run_benchmark(p.parse_args())


if __name__ == "__main__":
    main()
