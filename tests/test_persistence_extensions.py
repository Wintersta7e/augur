"""Tests for PersistenceManager extensions added during the deep review fix loop.

Covers:
- save/load_reflection (ARCH-03 fix)
- save/load_last_anomaly, save/load_last_advice (ARCH-07 consolidation)
- mark_tuning_applied / is_tuning_applied (ARCH-02 fix)
- 30-day TTL on per-session keys (LEAK-12 fix)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock


from blackboard.persistence import (
    SESSION_KEY_TTL_S,
    TUNING_APPLIED_TTL_S,
    PersistenceManager,
)


class TestSaveReflection:
    def test_save_writes_key_with_ttl(self) -> None:
        mock_redis = MagicMock()
        pm = PersistenceManager(mock_redis)
        report = {"session_id": "sess-1", "foo": "bar"}

        pm.save_reflection("sess-1", report)

        mock_redis.set.assert_called_once()
        args, kwargs = mock_redis.set.call_args
        assert args[0] == "augur:reflect:sess-1"
        assert json.loads(args[1]) == report
        assert kwargs.get("ex") == SESSION_KEY_TTL_S

    def test_load_returns_none_when_absent(self) -> None:
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        pm = PersistenceManager(mock_redis)
        assert pm.load_reflection("sess-missing") is None

    def test_load_returns_parsed_dict(self) -> None:
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps({"a": 1}).encode()
        pm = PersistenceManager(mock_redis)
        assert pm.load_reflection("sess-1") == {"a": 1}


class TestSaveLastAnomaly:
    def test_save_writes_live_state_key_without_ttl(self) -> None:
        mock_redis = MagicMock()
        pm = PersistenceManager(mock_redis)
        anomaly = {"domain": "chess", "severity": "high"}

        pm.save_last_anomaly(anomaly)

        mock_redis.set.assert_called_once()
        args, kwargs = mock_redis.set.call_args
        assert args[0] == "augur:detection:last_anomaly"
        assert json.loads(args[1]) == anomaly
        # Live state — no TTL
        assert "ex" not in kwargs

    def test_load_returns_none_when_absent(self) -> None:
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        pm = PersistenceManager(mock_redis)
        assert pm.load_last_anomaly() is None

    def test_load_returns_parsed_dict(self) -> None:
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps({"s": "high"}).encode()
        pm = PersistenceManager(mock_redis)
        assert pm.load_last_anomaly() == {"s": "high"}


class TestSaveLastAdvice:
    def test_save_writes_live_state_key_without_ttl(self) -> None:
        mock_redis = MagicMock()
        pm = PersistenceManager(mock_redis)
        advice = {"domain": "chess", "advice": "think more"}

        pm.save_last_advice(advice)

        mock_redis.set.assert_called_once()
        args, kwargs = mock_redis.set.call_args
        assert args[0] == "augur:reasoning:last_advice"
        assert json.loads(args[1]) == advice
        assert "ex" not in kwargs

    def test_load_returns_parsed_dict(self) -> None:
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps({"advice": "x"})
        pm = PersistenceManager(mock_redis)
        assert pm.load_last_advice() == {"advice": "x"}


class TestTuningAppliedMarker:
    def test_mark_sets_key_with_tuning_ttl(self) -> None:
        mock_redis = MagicMock()
        pm = PersistenceManager(mock_redis)

        pm.mark_tuning_applied("sess-abc")

        mock_redis.set.assert_called_once()
        args, kwargs = mock_redis.set.call_args
        assert args[0] == "augur:correlation:tuning_applied:sess-abc"
        assert args[1] == "1"
        assert kwargs.get("ex") == TUNING_APPLIED_TTL_S

    def test_is_applied_returns_false_when_key_missing(self) -> None:
        mock_redis = MagicMock()
        mock_redis.exists.return_value = 0
        pm = PersistenceManager(mock_redis)
        assert pm.is_tuning_applied("sess-missing") is False

    def test_is_applied_returns_true_when_key_present(self) -> None:
        mock_redis = MagicMock()
        mock_redis.exists.return_value = 1
        pm = PersistenceManager(mock_redis)
        assert pm.is_tuning_applied("sess-abc") is True


class TestSessionKeyTTLs:
    """LEAK-12: per-session keys must have a TTL to prevent unbounded growth."""

    def test_save_feedback_sets_ttl(self) -> None:
        mock_redis = MagicMock()
        pm = PersistenceManager(mock_redis)
        pm.save_feedback("sess-1", {"advice_events": []})
        # First call is r.set(key, value, ex=TTL)
        args, kwargs = mock_redis.set.call_args
        assert kwargs.get("ex") == SESSION_KEY_TTL_S

    def test_save_correlation_graph_sets_ttl(self) -> None:
        mock_redis = MagicMock()
        pm = PersistenceManager(mock_redis)
        pm.save_correlation_graph("sess-1", {"nodes": [], "edges": []})
        args, kwargs = mock_redis.set.call_args
        assert kwargs.get("ex") == SESSION_KEY_TTL_S

    def test_save_reflection_sets_ttl(self) -> None:
        mock_redis = MagicMock()
        pm = PersistenceManager(mock_redis)
        pm.save_reflection("sess-1", {"session_id": "sess-1"})
        args, kwargs = mock_redis.set.call_args
        assert kwargs.get("ex") == SESSION_KEY_TTL_S
