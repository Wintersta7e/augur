"""Tests for PersistenceManager escalation matrix save/load.

The correlator loads this matrix on every anomaly event to support
runtime tuning by the reflection engine (not cached).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock


from tabula.persistence import PersistenceManager

MATRIX_KEY = "augur:config:escalation_matrix"

SAMPLE_MATRIX = {
    "version": "1.0",
    "rules": {
        "LOW+LOW": "MEDIUM",
        "LOW+MEDIUM": "MEDIUM",
        "LOW+HIGH": "HIGH",
        "MEDIUM+MEDIUM": "HIGH",
        "MEDIUM+HIGH": "HIGH",
        "HIGH+HIGH": "HIGH",
    },
}


class TestSaveEscalationMatrix:
    def test_save_writes_full_dict_to_known_key(self) -> None:
        mock_redis = MagicMock()
        pm = PersistenceManager(mock_redis)

        pm.save_escalation_matrix(SAMPLE_MATRIX)

        mock_redis.set.assert_called_once()
        called_key, called_value = mock_redis.set.call_args[0]
        assert called_key == MATRIX_KEY
        assert json.loads(called_value) == SAMPLE_MATRIX

    def test_save_preserves_version_field(self) -> None:
        mock_redis = MagicMock()
        pm = PersistenceManager(mock_redis)

        pm.save_escalation_matrix({"version": "2.0", "rules": {}})

        _, called_value = mock_redis.set.call_args[0]
        assert json.loads(called_value)["version"] == "2.0"


class TestLoadEscalationMatrix:
    def test_load_returns_none_when_key_missing(self) -> None:
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        pm = PersistenceManager(mock_redis)

        assert pm.load_escalation_matrix() is None

    def test_load_returns_parsed_dict_when_present(self) -> None:
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps(SAMPLE_MATRIX).encode()
        pm = PersistenceManager(mock_redis)

        result = pm.load_escalation_matrix()

        assert result == SAMPLE_MATRIX
        mock_redis.get.assert_called_once_with(MATRIX_KEY)

    def test_load_handles_string_return_from_redis(self) -> None:
        # Some redis clients decode to str when decode_responses=True
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps(SAMPLE_MATRIX)
        pm = PersistenceManager(mock_redis)

        result = pm.load_escalation_matrix()

        assert result == SAMPLE_MATRIX


class TestRoundTrip:
    def test_save_then_load_via_shared_mock(self) -> None:
        # Simulate Redis with a dict-backed stand-in
        store: dict[str, bytes] = {}
        mock_redis = MagicMock()
        mock_redis.set.side_effect = lambda k, v: store.__setitem__(k, v)
        mock_redis.get.side_effect = lambda k: store.get(k)

        pm = PersistenceManager(mock_redis)
        pm.save_escalation_matrix(SAMPLE_MATRIX)

        assert pm.load_escalation_matrix() == SAMPLE_MATRIX
