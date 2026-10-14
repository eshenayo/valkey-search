"""Issue #8 integration checks: SVS_VAMANA creates no threads (the installed
pool has size() == 1) and search latency under load stays in HNSW's regime."""

import struct
import threading
import time

from valkey.client import Valkey

from ft_info_parser import FTInfoParser
from valkey_search_test_case import ValkeySearchTestCaseBase
from valkeytestframework.conftest import resource_port_tracker


DIM = 4
NUM_DOCS = 200
NUM_CLIENTS = 8
LOAD_SECONDS = 2.0
KNN_K = 10
LATENCY_BOUND = 10.0


def _fp32(values):
    return struct.pack(f"<{len(values)}f", *values)


def _doc_vector(i):
    return [float(i)] + [0.0] * (DIM - 1)


def _threads(pid: int) -> int:
    """Thread count of the server process, from /proc/<pid>/status."""
    with open(f"/proc/{pid}/status") as status:
        for line in status:
            if line.startswith("Threads:"):
                return int(line.split()[1])
    raise AssertionError(f"no Threads: line in /proc/{pid}/status")


def _wait_indexed(client: Valkey, index: str, n: int, timeout: float = 60.0):
    """Block until FT.INFO shows n docs, an empty mutation queue and no
    backfill in progress."""
    deadline = time.monotonic() + timeout
    while True:
        info = FTInfoParser(client.execute_command("FT.INFO", index))
        if (info.num_docs >= n and info.mutation_queue_size == 0
                and not info.backfill_in_progress):
            return
        if time.monotonic() > deadline:
            raise AssertionError(
                f"{index}: {info.num_docs}/{n} docs indexed after {timeout}s "
                f"(mutation_queue_size={info.mutation_queue_size})")
        time.sleep(0.05)


def _knn(client: Valkey, index: str, query, k: int = KNN_K):
    return client.execute_command(
        "FT.SEARCH", index, f"*=>[KNN {k} @v $q]",
        "PARAMS", "2", "q", _fp32(query), "DIALECT", "2", "LIMIT", "0", str(k))


def _search_load(clients, index: str, seconds: float, sampler=None):
    """Run one FT.SEARCH loop per client for `seconds`. Returns the list of
    per-call latencies (seconds) across all clients. `sampler`, if given, is
    called every 100 ms on the main thread while the load runs."""
    stop = threading.Event()
    latencies = [[] for _ in clients]
    errors = []

    def worker(i, client):
        query = [NUM_DOCS / 2 + 0.5] + [0.0] * (DIM - 1)
        try:
            while not stop.is_set():
                t0 = time.perf_counter()
                reply = _knn(client, index, query)
                latencies[i].append(time.perf_counter() - t0)
                assert reply[0] == KNN_K, reply[0]
        except Exception as exc:  # surfaced on the main thread below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i, c))
               for i, c in enumerate(clients)]
    for t in threads:
        t.start()
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        time.sleep(0.1)
        if sampler is not None:
            sampler()
    stop.set()
    for t in threads:
        t.join()
    assert not errors, errors[0]
    return [lat for per_client in latencies for lat in per_client]


def _p99(samples):
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))]


class TestSVSThreadpool(ValkeySearchTestCaseBase):

    def _create(self, client: Valkey, index: str, algorithm: str, prefix: str):
        client.execute_command(
            "FT.CREATE", index, "PREFIX", "1", prefix,
            "SCHEMA", "v", "VECTOR", algorithm,
            "6", "TYPE", "FLOAT32", "DIM", str(DIM), "DISTANCE_METRIC", "L2")

    def _populate(self, client: Valkey, prefix: str):
        for i in range(NUM_DOCS):
            client.hset(f"{prefix}{i}", mapping={"v": _fp32(_doc_vector(i))})

    def _clients(self):
        return [self.server.get_new_client() for _ in range(NUM_CLIENTS)]

    def test_thread_count_unchanged_by_svs_build_ingest_and_search(self):
        """No thread is created at FT.CREATE, at bootstrap, while adding
        points, or under 8 concurrent searches."""
        client: Valkey = self.server.get_new_client()
        pid = self.server.pid()

        baseline = _threads(pid)
        self._create(client, "thr", "SVS_VAMANA", "thr:")
        after_create = _threads(pid)
        self._populate(client, "thr:")
        _wait_indexed(client, "thr", NUM_DOCS)
        after_ingest = _threads(pid)

        during_search = []
        _search_load(self._clients(), "thr", LOAD_SECONDS,
                     sampler=lambda: during_search.append(_threads(pid)))
        assert during_search, "no samples taken during the search load"

        observed = {
            "baseline": baseline,
            "after_create": after_create,
            "after_ingest": after_ingest,
            "during_search_min": min(during_search),
            "during_search_max": max(during_search),
        }
        print(f"threads: {observed}")
        assert all(v == baseline for v in observed.values()), observed

    def test_p99_search_latency_under_8_clients_within_10x_of_hnsw(self):
        """One search occupies one reader thread, like HNSW; same 200 docs in
        both indexes."""
        client: Valkey = self.server.get_new_client()
        self._create(client, "lat_svs", "SVS_VAMANA", "lat:")
        self._create(client, "lat_hnsw", "HNSW", "lat:")
        self._populate(client, "lat:")
        _wait_indexed(client, "lat_svs", NUM_DOCS)
        _wait_indexed(client, "lat_hnsw", NUM_DOCS)

        query = [NUM_DOCS / 2 + 0.5] + [0.0] * (DIM - 1)
        for index in ("lat_svs", "lat_hnsw"):
            for _ in range(50):  # warm up
                _knn(client, index, query)

        clients = self._clients()
        p99 = {}
        samples = {}
        for index in ("lat_hnsw", "lat_svs"):
            lat = _search_load(clients, index, LOAD_SECONDS)
            assert len(lat) >= 100, f"{index}: only {len(lat)} searches"
            p99[index] = _p99(lat)
            samples[index] = len(lat)

        detail = (f"p99 svs={p99['lat_svs'] * 1e3:.3f} ms "
                  f"(n={samples['lat_svs']}) vs "
                  f"hnsw={p99['lat_hnsw'] * 1e3:.3f} ms "
                  f"(n={samples['lat_hnsw']}), bound {LATENCY_BOUND}x")
        print(detail)
        assert p99["lat_svs"] <= LATENCY_BOUND * p99["lat_hnsw"], detail
