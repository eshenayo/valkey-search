"""LVQ and LeanVec compression on SVS_VAMANA, end to end.

Every kind fills uncompressed, converts once at TRAINING_THRESHOLD and then
answers queries. Lossy storage reorders close neighbors, so recall is judged by
how much of FLAT's top 10 it returns rather than by score error. A converted
index survives SAVE + DEBUG RELOAD with the same keys and scores.

Needs the prebuilt SVS C API (build.sh --svs-prebuilt) on an Intel CPU; skips
elsewhere.
"""

import numpy as np
import pytest
from valkey.client import Valkey
from valkeytestframework.util import waiters

from ft_info_parser import FTInfoParser
from test_svs_smoke import leanvec_min_training_threshold
from indexes import ClusterTestUtils
from valkey_search_test_case import (
    ValkeySearchClusterTestCaseDebugMode,
    ValkeySearchTestCaseBase,
    ValkeySearchTestCaseDebugMode,
)
from valkeytestframework.conftest import resource_port_tracker


KINDS = ["LVQ4", "LVQ8", "LVQ4X4", "LVQ4X8",
         "LEANVEC4X4", "LEANVEC4X8", "LEANVEC8X8"]
DIM = 32
THRESHOLD = 256
NUM_DOCS = 384
DELETED = 16
NUM_QUERIES = 20
K = 10
# LeanVec searches on the reduced vectors and reranks the search window at full
# dimension, so a window of K leaves recall to the reduced vectors alone. The
# default window for LeanVec is not tuned yet.
LEANVEC_SEARCH_WINDOW = 4 * K
MIN_RECALL = 0.8


def _vectors(n, seed):
    """Seeded vectors near an 8-dimensional subspace, like real embeddings,
    so LeanVec's reduction to DIM / 4 keeps most of the structure."""
    rng = np.random.default_rng(seed)
    basis = np.random.default_rng(0).standard_normal((8, DIM))
    latent = rng.standard_normal((n, 8))
    noise = 0.05 * rng.standard_normal((n, DIM))
    return (latent @ basis + noise).astype(np.float32)


def _v_index(client: Valkey, index: str) -> dict:
    return FTInfoParser(
        client.execute_command("FT.INFO", index)
    ).attributes[0]["index"]


def _counter(client: Valkey, name: str) -> int:
    return int(client.info("SEARCH").get(f"search_{name}", 0))


def _knn(client: Valkey, index: str, query) -> dict:
    """{key: score} for the top K, in rank order."""
    reply = client.execute_command(
        "FT.SEARCH", index, f"*=>[KNN {K} @v $q]",
        "PARAMS", "2", "q", query.tobytes(), "LIMIT", "0", str(K),
        "DIALECT", "2",
    )
    out = {}
    for i in range(1, len(reply), 2):
        payload = dict(zip(reply[i + 1][::2], reply[i + 1][1::2]))
        out[reply[i]] = float(payload[b"__v_score"])
    return out


class TestSVSLvqLeanVec(ValkeySearchTestCaseDebugMode):

    # FLOAT16 input fills an FP16 index, so its conversion starts from a
    # different source than FLOAT32's.
    @pytest.mark.parametrize("vtype", ["FLOAT32", "FLOAT16"])
    @pytest.mark.parametrize("metric", ["L2", "COSINE"])
    @pytest.mark.parametrize("compression", KINDS)
    def test_converts_at_threshold_and_survives_reload(self, compression,
                                                       metric, vtype):
        client: Valkey = self.server.get_new_client()
        if not int(client.info("SEARCH").get(
                "search_svs_lvq_leanvec_supported", 0)):
            pytest.skip("needs the prebuilt SVS C API on an Intel CPU")

        svs = ["TYPE", vtype, "DIM", str(DIM), "DISTANCE_METRIC", metric,
               "COMPRESSION", compression,
               "TRAINING_THRESHOLD", str(THRESHOLD)]
        if compression.startswith("LEANVEC"):
            svs += ["REDUCE", str(DIM // 4),
                    "SEARCH_WINDOW_SIZE", str(LEANVEC_SEARCH_WINDOW)]
        flat = ["TYPE", vtype, "DIM", str(DIM), "DISTANCE_METRIC", metric]
        for index, algorithm, params in (("svs", "SVS_VAMANA", svs),
                                         ("flat", "FLAT", flat)):
            client.execute_command(
                "FT.CREATE", index, "PREFIX", "1", "d:",
                "SCHEMA", "v", "VECTOR", algorithm, str(len(params)), *params,
            )

        dtype = np.float16 if vtype == "FLOAT16" else np.float32
        docs = _vectors(NUM_DOCS, seed=1).astype(dtype)
        conversions = _counter(client, "svs_conversions_count")
        failures = _counter(client, "svs_conversion_failures_count")

        def wait_for_size(size):
            for index in ("svs", "flat"):
                waiters.wait_for_equal(
                    lambda: _v_index(client, index)["size"], size, timeout=30)

        def add(start, stop):
            for i in range(start, stop):
                client.hset(f"d:{i}", mapping={"v": docs[i].tobytes()})
            wait_for_size(stop)

        add(0, THRESHOLD - 1)
        assert _v_index(client, "svs")["algorithm"]["compression_pending"] \
            == 1
        assert _counter(client, "svs_conversions_count") == conversions

        # FT.INFO counts a key before its add, and the conversion it
        # triggers, finish; wait on the conversion itself.
        add(THRESHOLD - 1, THRESHOLD)
        waiters.wait_for_equal(
            lambda: _v_index(client, "svs")["algorithm"]["compression_pending"],
            0, timeout=30)
        assert _counter(client, "svs_conversions_count") == conversions + 1

        # Writes, modifies and deletes keep landing after the conversion,
        # without converting again. The deletes make SAVE consolidate and
        # compact the converted index.
        add(THRESHOLD, NUM_DOCS)
        for i in range(DELETED):
            client.delete(f"d:{i}")
        client.hset(f"d:{DELETED}", mapping={"v": docs[0].tobytes()})
        wait_for_size(NUM_DOCS - DELETED)
        assert _counter(client, "svs_conversions_count") == conversions + 1
        assert _counter(client, "svs_conversion_failures_count") == failures

        queries = _vectors(NUM_QUERIES, seed=2).astype(dtype)
        before = [_knn(client, "svs", q) for q in queries]
        found = sum(len(result.keys() & _knn(client, "flat", q).keys())
                    for q, result in zip(queries, before))
        recall = found / (K * NUM_QUERIES)
        assert recall >= MIN_RECALL, f"recall@{K} vs FLAT {recall:.2f}"
        deleted = {f"d:{i}".encode() for i in range(DELETED)}
        assert not any(deleted & result.keys() for result in before)
        assert next(iter(_knn(client, "svs", docs[0]))) == \
            f"d:{DELETED}".encode()

        # SAVE consolidates and compacts the deleted slots first, which can
        # reorder close neighbors; the reload must then change nothing.
        client.execute_command("SAVE")
        saved = [_knn(client, "svs", q) for q in queries]
        client.execute_command("DEBUG", "RELOAD")
        info = _v_index(client, "svs")
        assert info["size"] == NUM_DOCS - DELETED
        # Loaded from the saved SVS index, not rebuilt and converted again.
        assert _counter(client, "svs_conversions_count") == conversions + 1
        assert info["algorithm"]["compression_pending"] == 0
        assert info["algorithm"]["compression"] == \
            f"SVS_COMPRESSION_{compression}"
        assert info["algorithm"].get("leanvec_dims") == \
            (DIM // 4 if compression.startswith("LEANVEC") else None)
        for q, (query, result) in enumerate(zip(queries, saved)):
            after = _knn(client, "svs", query)
            assert list(after) == list(result), f"query {q} keys"
            assert list(after.values()) == pytest.approx(
                list(result.values()), abs=1e-6), f"query {q} scores"

    @pytest.mark.parametrize("metric", ["L2", "COSINE"])
    def test_leanvec_at_training_floor_matches_full_training(self, metric):
        """LeanVec trained at its TRAINING_THRESHOLD floor is within 0.05
        recall@10 of one trained on all 10k vectors (10k x 128-d, variance
        spread over every dimension). Window 100, so the comparison measures
        the learned projection rather than the search window."""
        client: Valkey = self.server.get_new_client()
        if not int(client.info("SEARCH").get(
                "search_svs_lvq_leanvec_supported", 0)):
            pytest.skip("needs the prebuilt SVS C API on an Intel CPU")
        n, dim, window = 10_000, 128, 100
        floor = leanvec_min_training_threshold(dim)
        rot, _ = np.linalg.qr(
            np.random.default_rng(0).standard_normal((dim, dim)))
        scale = np.arange(1, dim + 1) ** -0.5

        def vectors(count, seed):
            rng = np.random.default_rng(seed)
            return ((rng.standard_normal((count, dim)) * scale) @ rot).astype(
                np.float32)

        indexes = [(kind, t) for kind in KINDS if kind.startswith("LEANVEC")
                   for t in (floor, n)]
        for kind, t in indexes:
            params = ["TYPE", "FLOAT32", "DIM", str(dim),
                      "DISTANCE_METRIC", metric, "COMPRESSION", kind,
                      "TRAINING_THRESHOLD", str(t),
                      "SEARCH_WINDOW_SIZE", str(window)]
            client.execute_command(
                "FT.CREATE", f"{kind.lower()}_{t}", "PREFIX", "1", "f:",
                "SCHEMA", "v", "VECTOR", "SVS_VAMANA", str(len(params)),
                *params)
        flat = ["TYPE", "FLOAT32", "DIM", str(dim), "DISTANCE_METRIC", metric]
        client.execute_command("FT.CREATE", "flat", "PREFIX", "1", "f:",
                               "SCHEMA", "v", "VECTOR", "FLAT",
                               str(len(flat)), *flat)
        docs = vectors(n, 1)

        def insert(start, stop):
            pipe = client.pipeline(transaction=False)
            for i in range(start, stop):
                pipe.hset(f"f:{i}", mapping={"v": docs[i].tobytes()})
            pipe.execute()

        # An index converts with every vector it holds by then, and writers
        # keep adding while a conversion waits for its lock. Insert exactly
        # the floor first, so the floor indexes train on that many.
        insert(0, floor)
        for kind, t in indexes:
            waiters.wait_for_equal(
                lambda: (_v_index(client, f"{kind.lower()}_{t}")["size"],
                         _v_index(client, f"{kind.lower()}_{t}")["algorithm"]
                         ["compression_pending"]),
                (floor, 0 if t == floor else 1), timeout=120)
        insert(floor, n)
        for kind, t in indexes:
            waiters.wait_for_equal(
                lambda: (_v_index(client, f"{kind.lower()}_{t}")["size"],
                         _v_index(client, f"{kind.lower()}_{t}")["algorithm"]
                         ["compression_pending"]),
                (n, 0), timeout=120)
        waiters.wait_for_equal(lambda: _v_index(client, "flat")["size"], n,
                               timeout=120)

        queries = vectors(200, 2)
        truth = [_knn(client, "flat", q).keys() for q in queries]

        def recall(index):
            return np.mean([len(_knn(client, index, q).keys() & e) / K
                            for q, e in zip(queries, truth)])

        for kind in sorted({kind for kind, _ in indexes}):
            at_floor = recall(f"{kind.lower()}_{floor}")
            full = recall(f"{kind.lower()}_{n}")
            assert at_floor >= full - 0.05, (
                f"{kind} {metric}: recall@{K} {at_floor:.3f} at the floor "
                f"({floor}) vs {full:.3f} trained on {n}")

    @pytest.mark.parametrize("metric", ["L2", "COSINE"])
    @pytest.mark.parametrize("compression", KINDS)
    def test_loads_without_storage_where_unsupported(self, compression,
                                                     metric):
        """An RDB with an LVQ/LeanVec index loads on a build or CPU that
        can't create it: the rest of the keyspace loads, the log, FT.INFO
        and INFO name what is missing, searches that need the index fail
        with that reason, exact pre-filtered searches still run, and writes
        stay tracked. Back on a build that supports it, the index is rebuilt
        from the keyspace, including those writes. The debug-only switch
        stands in for the build or CPU."""
        client: Valkey = self.server.get_new_client()
        if not int(client.info("SEARCH").get(
                "search_svs_lvq_leanvec_supported", 0)):
            pytest.skip("needs the prebuilt SVS C API on an Intel CPU")
        dim, threshold, n = 16, 64, 80
        params = ["TYPE", "FLOAT32", "DIM", str(dim), "DISTANCE_METRIC",
                  metric, "COMPRESSION", compression,
                  "TRAINING_THRESHOLD", str(threshold)]
        client.execute_command(
            "FT.CREATE", "svs", "PREFIX", "1", "d:", "SCHEMA", "v", "VECTOR",
            "SVS_VAMANA", str(len(params)), *params)
        client.execute_command(
            "FT.CREATE", "hnsw", "PREFIX", "1", "h:", "SCHEMA", "v", "VECTOR",
            "HNSW", "6", "TYPE", "FLOAT32", "DIM", str(dim),
            "DISTANCE_METRIC", metric)
        rng = np.random.default_rng(3)
        docs = rng.standard_normal((n + 1, dim)).astype(np.float32)
        for i in range(n):
            client.hset(f"d:{i}", mapping={"v": docs[i].tobytes()})
            client.hset(f"h:{i}", mapping={"v": docs[i].tobytes()})
        client.set("plain", "value")
        waiters.wait_for_equal(
            lambda: (_v_index(client, "svs")["size"],
                     _v_index(client, "svs")["algorithm"]
                     ["compression_pending"]), (n, 0), timeout=30)
        keys = client.dbsize()

        def knn(index, query, inkeys=None):
            filt = ["INKEYS", str(len(inkeys)), *inkeys] if inkeys else []
            return client.execute_command(
                "FT.SEARCH", index, "*=>[KNN 3 @v $q]", *filt, "PARAMS", "2",
                "q", query.tobytes(), "NOCONTENT", "DIALECT", "2")

        flag = "search.svs-lvq-leanvec-force-unsupported"
        client.execute_command("CONFIG", "SET", flag, "yes")
        try:
            client.execute_command("DEBUG", "RELOAD")
            assert client.dbsize() == keys
            assert client.get("plain") == b"value"
            info = client.info("SEARCH")
            assert int(info["search_svs_unavailable_indexes"]) == 1
            assert int(info["search_svs_lvq_leanvec_supported"]) == 0
            missing = "requires an SVS build with LVQ/LeanVec support"
            algorithm = _v_index(client, "svs")["algorithm"]
            assert missing in algorithm["unavailable"]
            # Nothing converts without storage; the settings are kept.
            assert algorithm["compression_pending"] == 0
            assert algorithm["training_threshold"] == threshold
            assert algorithm.get("leanvec_dims") == \
                (dim // 2 if compression.startswith("LEANVEC") else None)
            assert self.server.verify_string_in_logfile(missing)
            with pytest.raises(Exception) as exc_info:
                knn("svs", docs[0])
            assert missing in str(exc_info.value)
            assert knn("hnsw", docs[0])[0] == 3
            # Exact pre-filtered search runs from the tracked vectors.
            near = knn("svs", docs[0], inkeys=["d:0", "d:1", "d:2"])
            assert near[0] == 3 and near[1] == b"d:0"
            client.hset(f"d:{n}", mapping={"v": docs[n].tobytes()})
            waiters.wait_for_equal(
                lambda: _v_index(client, "svs")["size"], n + 1, timeout=30)
        finally:
            client.execute_command("CONFIG", "SET", flag, "no")

        client.execute_command("DEBUG", "RELOAD")
        assert int(client.info("SEARCH")["search_svs_unavailable_indexes"]) \
            == 0
        algorithm = _v_index(client, "svs")["algorithm"]
        assert "unavailable" not in algorithm
        waiters.wait_for_equal(
            lambda: (_v_index(client, "svs")["size"],
                     _v_index(client, "svs")["algorithm"]
                     ["compression_pending"]), (n + 1, 0), timeout=30)
        assert knn("svs", docs[n])[1] == f"d:{n}".encode()


class TestSVSLvqLeanVecUnavailableCluster(ValkeySearchClusterTestCaseDebugMode,
                                          ClusterTestUtils):

    @pytest.mark.parametrize("metric", ["L2", "COSINE"])
    @pytest.mark.parametrize("compression", KINDS)
    def test_reason_survives_consistent_fanout(self, compression, metric):
        """Across shards that can't create an LVQ/LeanVec index, the reason
        reaches the client under CONSISTENT fan-out (FT.SEARCH) and consistent
        results (FT.HYBRID), not a consistency-failure message."""
        if not int(self.client_for_primary(0).info("SEARCH").get(
                "search_svs_lvq_leanvec_supported", 0)):
            pytest.skip("needs the prebuilt SVS C API on an Intel CPU")
        self.execute_primaries(["flushall sync"])
        cluster = self.new_cluster_client()
        dim, n = 8, 30
        cluster.execute_command(
            "FT.CREATE", "svs", "ON", "HASH", "PREFIX", "1", "d:", "SCHEMA",
            "title", "TEXT", "v", "VECTOR", "SVS_VAMANA", "8", "TYPE",
            "FLOAT32", "DIM", str(dim), "DISTANCE_METRIC", metric,
            "COMPRESSION", compression)
        rng = np.random.default_rng(5)
        docs = rng.standard_normal((n, dim)).astype(np.float32)
        for i in range(n):
            cluster.hset(f"d:{i}", mapping={"title": "hello",
                                            "v": docs[i].tobytes()})
        waiters.wait_for_equal(
            lambda: sum(FTInfoParser(self.client_for_primary(i).execute_command(
                "FT.INFO", "svs")).num_docs
                for i in range(self.CLUSTER_SIZE)), n, timeout=30)

        flag = "search.svs-lvq-leanvec-force-unsupported"
        missing = "requires an SVS build with LVQ/LeanVec support"
        self.config_set(flag, "yes")
        try:
            self.execute_primaries(["DEBUG", "RELOAD"])
            # A CONSISTENT search aborts a node whose cached cluster map doesn't
            # yet hold its own shard (CHECK in PerformSlotConsistencyCheck, not
            # SVS); let the map settle first.
            self.wait_for_cluster_topology_to_settle()
            node = self.client_for_primary(0)
            with pytest.raises(Exception) as exc_info:
                node.execute_command(
                    "FT.SEARCH", "svs", "*=>[KNN 3 @v $q]", "PARAMS", "2",
                    "q", docs[0].tobytes(), "ALLSHARDS", "CONSISTENT",
                    "DIALECT", "2")
            assert missing in str(exc_info.value)
            self.config_set("search.enable-consistent-results", "yes")
            try:
                with pytest.raises(Exception) as exc_info:
                    node.execute_command(
                        "FT.HYBRID", "svs", "SEARCH", "@title:hello",
                        "VSIM", "@v", "$q", "KNN", "2", "K", "3",
                        "PARAMS", "2", "q", docs[0].tobytes())
                assert missing in str(exc_info.value)
            finally:
                self.config_set("search.enable-consistent-results", "no")
        finally:
            self.config_set(flag, "no")

    @pytest.mark.parametrize("metric", ["L2", "COSINE"])
    @pytest.mark.parametrize("compression", KINDS)
    def test_definition_from_metadata_on_unsupported_shards(self, compression,
                                                            metric):
        """An index created on a shard that supports LVQ/LeanVec reaches,
        through cluster metadata, shards that don't: they load it without
        storage instead of rejecting it, and a CONSISTENT search reports the
        reason."""
        if not int(self.client_for_primary(0).info("SEARCH").get(
                "search_svs_lvq_leanvec_supported", 0)):
            pytest.skip("needs the prebuilt SVS C API on an Intel CPU")
        self.execute_primaries(["flushall sync"])
        flag = "search.svs-lvq-leanvec-force-unsupported"
        missing = "requires an SVS build with LVQ/LeanVec support"
        unsupported = range(1, self.CLUSTER_SIZE)
        dim = 8
        for i in unsupported:
            self.client_for_primary(i).execute_command(
                "CONFIG", "SET", flag, "yes")
        try:
            node = self.client_for_primary(0)
            node.execute_command(
                "FT.CREATE", "svs", "ON", "HASH", "PREFIX", "1", "d:",
                "SCHEMA", "v", "VECTOR", "SVS_VAMANA", "8", "TYPE", "FLOAT32",
                "DIM", str(dim), "DISTANCE_METRIC", metric,
                "COMPRESSION", compression)
            for i in unsupported:
                waiters.wait_for_true(
                    lambda: b"svs" in self.client_for_primary(i)
                    .execute_command("FT._LIST"), timeout=30)
                assert missing in _v_index(
                    self.client_for_primary(i), "svs")["algorithm"][
                        "unavailable"]
            assert "unavailable" not in _v_index(node, "svs")["algorithm"]
            cluster = self.new_cluster_client()
            rng = np.random.default_rng(7)
            docs = rng.standard_normal((30, dim)).astype(np.float32)
            for j, v in enumerate(docs):
                cluster.hset(f"d:{j}", mapping={"v": v.tobytes()})
            # A CONSISTENT search aborts a node whose cached cluster map doesn't
            # yet hold its own shard (CHECK in PerformSlotConsistencyCheck, not
            # SVS); let the map settle first.
            self.wait_for_cluster_topology_to_settle()
            with pytest.raises(Exception) as exc_info:
                node.execute_command(
                    "FT.SEARCH", "svs", "*=>[KNN 3 @v $q]", "PARAMS", "2",
                    "q", docs[0].tobytes(), "ALLSHARDS", "CONSISTENT",
                    "DIALECT", "2")
            assert missing in str(exc_info.value)
        finally:
            for i in unsupported:
                self.client_for_primary(i).execute_command(
                    "CONFIG", "SET", flag, "no")


class TestSVSLvqLeanVecSwitchNeedsDebugMode(ValkeySearchTestCaseBase):
    """The switch that makes LVQ/LeanVec unsupported is for tests: set from a
    config file on a server without debug mode, it is ignored."""

    def get_config_file_lines(self, testdir, port):
        return super().get_config_file_lines(testdir, port) + [
            "search.svs-lvq-leanvec-force-unsupported yes"]

    def test_switch_ignored_without_debug_mode(self):
        client: Valkey = self.server.get_new_client()
        info = client.info("SEARCH")
        if not int(info.get("search_svs_c_api_prebuilt", 0)):
            pytest.skip("needs the prebuilt SVS C API")
        assert int(info["search_svs_lvq_leanvec_supported"]) == 1
        client.execute_command(
            "FT.CREATE", "lvq", "SCHEMA", "v", "VECTOR", "SVS_VAMANA", "8",
            "TYPE", "FLOAT32", "DIM", "8", "DISTANCE_METRIC", "L2",
            "COMPRESSION", "LVQ8")
        assert "unavailable" not in _v_index(client, "lvq")["algorithm"]
