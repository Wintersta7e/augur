"""Tests for PersistenceManager correlation graph save/load/list.

The correlator flushes its in-memory NetworkX DiGraph to Redis at
session end, where reflection engine and MCP tools can inspect it.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock


from tabula.persistence import PersistenceManager

GRAPH_KEY_PREFIX = "augur:nexus:graph:"
GRAPH_INDEX_KEY = "augur:nexus:graph:_index"

SAMPLE_GRAPH_DATA = {
    "directed": True,
    "multigraph": False,
    "graph": {},
    "nodes": [
        {
            "id": "chess:white:2026-03-17T14:30:00+00:00",
            "domain": "chess",
            "entity": "white",
            "severity": "low",
            "timestamp": "2026-03-17T14:30:00+00:00",
        },
        {
            "id": "typing:user:2026-03-17T14:29:48+00:00",
            "domain": "typing",
            "entity": "user",
            "severity": "low",
            "timestamp": "2026-03-17T14:29:48+00:00",
        },
    ],
    "links": [
        {
            "source": "chess:white:2026-03-17T14:30:00+00:00",
            "target": "typing:user:2026-03-17T14:29:48+00:00",
            "temporal_lag": 12.0,
            "escalation_rule": "LOW+LOW\u2192MEDIUM",
            "combined_severity": "MEDIUM",
            "domains": ["chess", "typing"],
        }
    ],
}


class TestSaveCorrelationGraph:
    def test_save_writes_json_at_session_specific_key(self) -> None:
        mock_redis = MagicMock()
        pm = PersistenceManager(mock_redis)

        pm.save_correlation_graph("sess-abc", SAMPLE_GRAPH_DATA)

        # set call: key and JSON value
        mock_redis.set.assert_called_once()
        called_key, called_value = mock_redis.set.call_args[0]
        assert called_key == f"{GRAPH_KEY_PREFIX}sess-abc"
        assert json.loads(called_value) == SAMPLE_GRAPH_DATA

    def test_save_also_pushes_session_id_onto_index(self) -> None:
        mock_redis = MagicMock()
        pm = PersistenceManager(mock_redis)

        pm.save_correlation_graph("sess-abc", SAMPLE_GRAPH_DATA)

        mock_redis.lpush.assert_called_once_with(GRAPH_INDEX_KEY, "sess-abc")

    def test_save_trims_index_to_1000_entries(self) -> None:
        mock_redis = MagicMock()
        pm = PersistenceManager(mock_redis)

        pm.save_correlation_graph("sess-xyz", SAMPLE_GRAPH_DATA)

        mock_redis.ltrim.assert_called_once_with(GRAPH_INDEX_KEY, 0, 999)

    def test_save_handles_empty_graph(self) -> None:
        mock_redis = MagicMock()
        pm = PersistenceManager(mock_redis)

        empty_graph = {
            "directed": True,
            "multigraph": False,
            "graph": {},
            "nodes": [],
            "links": [],
        }
        pm.save_correlation_graph("sess-empty", empty_graph)

        mock_redis.set.assert_called_once()
        _, called_value = mock_redis.set.call_args[0]
        assert json.loads(called_value) == empty_graph


class TestLoadCorrelationGraph:
    def test_load_returns_none_when_key_missing(self) -> None:
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        pm = PersistenceManager(mock_redis)

        result = pm.load_correlation_graph("sess-missing")

        assert result is None
        mock_redis.get.assert_called_once_with(f"{GRAPH_KEY_PREFIX}sess-missing")

    def test_load_returns_parsed_dict_when_present(self) -> None:
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps(SAMPLE_GRAPH_DATA).encode()
        pm = PersistenceManager(mock_redis)

        result = pm.load_correlation_graph("sess-abc")

        assert result == SAMPLE_GRAPH_DATA

    def test_load_handles_string_return_from_decoded_responses_client(self) -> None:
        # redis[hiredis] with decode_responses=True returns str
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps(SAMPLE_GRAPH_DATA)
        pm = PersistenceManager(mock_redis)

        result = pm.load_correlation_graph("sess-abc")

        assert result == SAMPLE_GRAPH_DATA


class TestListCorrelationGraphs:
    def test_list_empty_index_returns_empty_list(self) -> None:
        mock_redis = MagicMock()
        mock_redis.lrange.return_value = []
        pm = PersistenceManager(mock_redis)

        result = pm.list_correlation_graphs()

        assert result == []
        mock_redis.lrange.assert_called_once_with(GRAPH_INDEX_KEY, 0, 49)

    def test_list_returns_session_ids_in_index_order(self) -> None:
        mock_redis = MagicMock()
        mock_redis.lrange.return_value = [b"sess-3", b"sess-2", b"sess-1"]
        pm = PersistenceManager(mock_redis)

        result = pm.list_correlation_graphs()

        assert result == ["sess-3", "sess-2", "sess-1"]

    def test_list_handles_string_returns_from_decoded_client(self) -> None:
        mock_redis = MagicMock()
        mock_redis.lrange.return_value = ["sess-a", "sess-b"]
        pm = PersistenceManager(mock_redis)

        result = pm.list_correlation_graphs()

        assert result == ["sess-a", "sess-b"]

    def test_list_respects_custom_limit(self) -> None:
        mock_redis = MagicMock()
        mock_redis.lrange.return_value = []
        pm = PersistenceManager(mock_redis)

        pm.list_correlation_graphs(limit=10)

        mock_redis.lrange.assert_called_once_with(GRAPH_INDEX_KEY, 0, 9)


class TestRoundTrip:
    def test_save_then_load_via_shared_mock(self) -> None:
        store: dict[str, bytes] = {}
        mock_redis = MagicMock()
        # save_correlation_graph now passes ex=SESSION_KEY_TTL_S (LEAK-12 fix),
        # so the lambda must accept keyword arguments.
        mock_redis.set.side_effect = lambda k, v, **kw: store.__setitem__(k, v)
        mock_redis.get.side_effect = lambda k: store.get(k)

        pm = PersistenceManager(mock_redis)
        pm.save_correlation_graph("sess-round", SAMPLE_GRAPH_DATA)

        assert pm.load_correlation_graph("sess-round") == SAMPLE_GRAPH_DATA
