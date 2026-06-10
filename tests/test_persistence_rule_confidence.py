"""Tests for PersistenceManager rule confidence save/load.

The reflection engine's matrix tuning analysis writes per-rule
EWMA confidence + restore_target snapshots to Redis. Schema:
    {rule_key: {"confidence": float, "restore_target": str | None}}
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock


from tabula.persistence import PersistenceManager

KEY = "augur:config:escalation_confidence"

SAMPLE_STATE = {
    "LOW+LOW": {"confidence": 0.78, "restore_target": "MEDIUM"},
    "MEDIUM+MEDIUM": {"confidence": 0.91, "restore_target": "HIGH"},
}


class TestSaveRuleConfidence:
    def test_save_writes_json_at_known_key(self) -> None:
        mock_redis = MagicMock()
        pm = PersistenceManager(mock_redis)

        pm.save_rule_confidence(SAMPLE_STATE)

        mock_redis.set.assert_called_once()
        called_key, called_value = mock_redis.set.call_args[0]
        assert called_key == KEY
        assert json.loads(called_value) == SAMPLE_STATE

    def test_save_handles_empty_state(self) -> None:
        mock_redis = MagicMock()
        pm = PersistenceManager(mock_redis)

        pm.save_rule_confidence({})

        mock_redis.set.assert_called_once()
        _, called_value = mock_redis.set.call_args[0]
        assert json.loads(called_value) == {}


class TestLoadRuleConfidence:
    def test_load_returns_none_when_key_missing(self) -> None:
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        pm = PersistenceManager(mock_redis)

        assert pm.load_rule_confidence() is None
        mock_redis.get.assert_called_once_with(KEY)

    def test_load_returns_parsed_dict_when_present(self) -> None:
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps(SAMPLE_STATE).encode()
        pm = PersistenceManager(mock_redis)

        result = pm.load_rule_confidence()

        assert result == SAMPLE_STATE

    def test_load_handles_string_return_from_decoded_client(self) -> None:
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps(SAMPLE_STATE)
        pm = PersistenceManager(mock_redis)

        result = pm.load_rule_confidence()

        assert result == SAMPLE_STATE


class TestRoundTrip:
    def test_save_then_load_via_shared_mock(self) -> None:
        store: dict[str, bytes] = {}
        mock_redis = MagicMock()
        mock_redis.set.side_effect = lambda k, v: store.__setitem__(k, v)
        mock_redis.get.side_effect = lambda k: store.get(k)

        pm = PersistenceManager(mock_redis)
        pm.save_rule_confidence(SAMPLE_STATE)

        assert pm.load_rule_confidence() == SAMPLE_STATE
