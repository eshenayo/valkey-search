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
from valkey_search_test_case import ValkeySearchTestCaseDebugMode
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
