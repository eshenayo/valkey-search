"""SVS_VAMANA under churn: inserts after deletes write into the deleted slots,
so memory stays flat at a constant live count, searches never return a
deleted key and results keep matching FLAT. Also covers a replica full sync
taken in the middle of churn."""

import random
import struct

import pytest
from valkey import Valkey
from valkeytestframework.conftest import resource_port_tracker
from valkeytestframework.util import waiters

from ft_info_parser import FTInfoParser
from valkey_search_test_case import ValkeySearchTestCaseBase

DIM = 8


def _fp32(values):
    return struct.pack(f"<{len(values)}f", *values)


def _info(client: Valkey, index: str) -> FTInfoParser:
    return FTInfoParser(client.execute_command("FT.INFO", index))


def _algorithm(client: Valkey, index: str) -> dict:
    return _info(client, index).attributes[0]["index"]["algorithm"]


def _create(client: Valkey, index: str, algorithm: str, params=()):
    params = ["TYPE", "FLOAT32", "DIM", str(DIM), "DISTANCE_METRIC", "L2",
              *params]
    client.execute_command(
        "FT.CREATE", index, "PREFIX", "1", "d:",
        "SCHEMA", "v", "VECTOR", algorithm, str(len(params)), *params,
    )


def _wait_settled(client: Valkey, index: str, size: int):
    def settled():
        info = _info(client, index)
        return (info.mutation_queue_size == 0 and
                info.attributes[0]["index"]["size"] == size)

    waiters.wait_for_equal(settled, True, timeout=30)


def _knn(client: Valkey, index: str, query, k: int):
    reply = client.execute_command(
        "FT.SEARCH", index, f"*=>[KNN {k} @v $q]", "NOCONTENT",
        "PARAMS", "2", "q", _fp32(query), "DIALECT", "2",
    )
    return [int(key[2:]) for key in reply[1:]]


class _Churn:
    """Keys d:<n> with random vectors; each round deletes `churn` live keys,
    waits until `index` has processed the deletes, then inserts as many new
    keys."""

    def __init__(self, client: Valkey, index: str, seed: int):
        self.client = client
        self.index = index
        self.rng = random.Random(seed)
        self.live = {}
        self.deleted = set()
        self.next_id = 0

    def insert(self, n: int):
        for _ in range(n):
            vector = [self.rng.uniform(-1, 1) for _ in range(DIM)]
            self.client.hset(f"d:{self.next_id}", mapping={"v": _fp32(vector)})
            self.live[self.next_id] = vector
            self.next_id += 1

    def round(self, churn: int):
        victims = self.rng.sample(sorted(self.live), churn)
        self.client.delete(*[f"d:{i}" for i in victims])
        for i in victims:
            del self.live[i]
        self.deleted.update(victims)
        live = len(self.live)
        _wait_settled(self.client, self.index, live)
        assert _algorithm(self.client, self.index)["deleted_slots"] == churn
        self.insert(churn)
        _wait_settled(self.client, self.index, live + churn)

    def exact_top_k(self, query, k: int):
        dist = sorted((sum((a - b) ** 2 for a, b in zip(v, query)), i)
                      for i, v in self.live.items())
        return [i for _, i in dist[:k]]


class TestSVSChurn(ValkeySearchTestCaseBase):

    NUM_DOCS = 500
    ROUNDS = 20
    CHURN = 100  # 2,000 inserts in all
    K = 10

    def _churn(self, client: Valkey):
        """At a constant live count, each round's inserts take every deleted
        slot, so deleted_slots returns to 0."""
        churn = _Churn(client, "svs", seed=3)
        churn.insert(self.NUM_DOCS)
        _wait_settled(client, "svs", self.NUM_DOCS)
        memory_after_fill = _algorithm(client, "svs")["memory_bytes"]
        for _ in range(self.ROUNDS):
            churn.round(self.CHURN)
            assert _algorithm(client, "svs")["deleted_slots"] == 0
        return churn, memory_after_fill

    @pytest.mark.xfail(
        strict=True,
        reason="SVS's concurrent index keeps stale reverse-edge entries "
               "until consolidation, so its graph grows under churn though "
               "no slot is added (intel/ScalableVectorSearch#469)")
    def test_memory_stays_flat(self):
        """Reusing slots adds none, so memory_bytes stays flat (2,000 new
        slots would add hundreds of KiB)."""
        client: Valkey = self.server.get_new_client()
        # The smallest SVS allocation blocks, so growth would show.
        _create(client, "svs", "SVS_VAMANA", ["INITIAL_CAP", "16"])
        _, memory_after_fill = self._churn(client)
        growth = _algorithm(client, "svs")["memory_bytes"] - memory_after_fill
        assert growth < 64 * 1024, f"SVS grew {growth} bytes under churn"

    def test_results_match_flat(self):
        """Under churn SVS keeps returning FLAT's neighbours and never a
        deleted key."""
        client: Valkey = self.server.get_new_client()
        _create(client, "svs", "SVS_VAMANA")
        _create(client, "flat", "FLAT")
        churn, _ = self._churn(client)
        _wait_settled(client, "flat", self.NUM_DOCS)

        rng = random.Random(4)
        matched = 0
        queries = [[rng.uniform(-1, 1) for _ in range(DIM)] for _ in range(20)]
        for query in queries:
            svs = _knn(client, "svs", query, self.K)
            assert not set(svs) & churn.deleted
            matched += len(set(svs) & set(_knn(client, "flat", query, self.K)))
        assert matched / (self.K * len(queries)) >= 0.9


class TestSVSChurnReplicaFullSync(ValkeySearchTestCaseBase):

    NUM_DOCS = 200
    CHURN = 50
    K = 5

    @pytest.mark.parametrize("setup_test", [{"replica_count": 1}],
                             indirect=True)
    def test_replica_full_sync_mid_churn(self):
        """A replica that full-syncs while the primary holds deleted slots
        loads none of them, and keeps answering exactly while the churn
        continues on the primary."""
        primary = self.get_primary_connection()
        replica = self.get_replica_connection(0)
        _create(primary, "svs", "SVS_VAMANA")
        churn = _Churn(primary, "svs", seed=5)
        churn.insert(self.NUM_DOCS)
        _wait_settled(primary, "svs", self.NUM_DOCS)
        churn.round(self.CHURN)
        # Leave deleted slots on the primary for the sync's RDB.
        victims = sorted(churn.live)[:self.CHURN]
        primary.delete(*[f"d:{i}" for i in victims])
        for i in victims:
            del churn.live[i]
        churn.deleted.update(victims)
        _wait_settled(primary, "svs", len(churn.live))
        assert _algorithm(primary, "svs")["deleted_slots"] == self.CHURN

        sync_full_before = primary.info("stats")["sync_full"]
        replica.execute_command("REPLICAOF", "NO", "ONE")
        self.rg.setup_replications_cmd()
        assert primary.info("stats")["sync_full"] == sync_full_before + 1
        waiters.wait_for_true(
            lambda: replica.info("replication")["master_link_status"] == "up",
            timeout=30,
        )
        _wait_settled(replica, "svs", len(churn.live))
        assert _algorithm(replica, "svs")["deleted_slots"] == 0
        # The sync's save ran in a forked child; the primary keeps its slots.
        assert _algorithm(primary, "svs")["deleted_slots"] == self.CHURN

        def assert_exact(client):
            for query in list(churn.live.values())[:5] + [[0.0] * DIM]:
                got = _knn(client, "svs", query, self.K)
                assert got == churn.exact_top_k(query, self.K)
                assert not set(got) & churn.deleted

        assert_exact(primary)
        assert_exact(replica)

        churn.insert(self.CHURN)  # takes the primary's deleted slots
        _wait_settled(primary, "svs", self.NUM_DOCS)
        assert _algorithm(primary, "svs")["deleted_slots"] == 0
        waiters.wait_for_true(
            lambda: replica.info("replication")["master_link_status"] == "up",
            timeout=30,
        )
        primary.execute_command("WAIT", "1", "5000")
        _wait_settled(replica, "svs", self.NUM_DOCS)
        assert_exact(primary)
        assert_exact(replica)
