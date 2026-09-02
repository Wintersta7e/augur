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

import fakeredis
import pytest

from tabula.persistence import (
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
        assert args[0] == "augur:disciplina:sess-1"
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
        assert args[0] == "augur:vigil:last_anomaly"
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
        assert args[0] == "augur:consilium:last_advice"
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

        pm.mark_tuning_applied("sess-abc", pass_name="correlation")

        mock_redis.set.assert_called_once()
        args, kwargs = mock_redis.set.call_args
        assert args[0] == "augur:tuning_applied:correlation:sess-abc"
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


class TestJsonHelpers:
    """The private ``_set_json`` / ``_get_json`` boilerplate-collapsing helpers.

    These back every uniform ``set(key, json.dumps(x))`` /
    ``json.loads(get(key)) or default`` method, so their contract is load-
    bearing: the underlying ``redis.set`` call must be byte-for-byte identical
    to the old inline form (no spurious ``ex`` kwarg), the absent-key path must
    return the caller's default, and a present-but-corrupt value must still
    raise (never be silently swallowed — the corrupt-tolerant loaders that DO
    swallow keep their own bespoke bodies and never route through here).
    """

    # -- _set_json ----------------------------------------------------------

    def test_set_json_without_ttl_omits_ex_kwarg(self) -> None:
        mock_redis = MagicMock()
        pm = PersistenceManager(mock_redis)

        pm._set_json("augur:test:k", {"a": 1})

        mock_redis.set.assert_called_once()
        args, kwargs = mock_redis.set.call_args
        assert args[0] == "augur:test:k"
        assert json.loads(args[1]) == {"a": 1}
        # Exact-shape contract: no ``ex`` keyword for a no-TTL write, matching
        # the prior inline ``self._r.set(key, json.dumps(x))``.
        assert "ex" not in kwargs

    def test_set_json_with_ttl_passes_ex_kwarg(self) -> None:
        mock_redis = MagicMock()
        pm = PersistenceManager(mock_redis)

        pm._set_json("augur:test:k", {"a": 1}, ex=SESSION_KEY_TTL_S)

        mock_redis.set.assert_called_once()
        args, kwargs = mock_redis.set.call_args
        assert args[0] == "augur:test:k"
        assert json.loads(args[1]) == {"a": 1}
        assert kwargs.get("ex") == SESSION_KEY_TTL_S

    def test_set_json_serializes_via_json_dumps(self) -> None:
        # Round-trip through fakeredis (bytes mode, like the live client).
        pm = PersistenceManager(fakeredis.FakeStrictRedis())
        pm._set_json("augur:test:k", {"nested": [1, 2, {"x": True}]})
        assert pm._get_json("augur:test:k") == {"nested": [1, 2, {"x": True}]}

    # -- _get_json ----------------------------------------------------------

    def test_get_json_absent_returns_none_default(self) -> None:
        pm = PersistenceManager(fakeredis.FakeStrictRedis())
        assert pm._get_json("augur:test:missing") is None

    def test_get_json_absent_returns_custom_default(self) -> None:
        pm = PersistenceManager(fakeredis.FakeStrictRedis())
        sentinel: dict = {}
        assert pm._get_json("augur:test:missing", sentinel) is sentinel
        assert pm._get_json("augur:test:missing", []) == []

    def test_get_json_present_decodes_bytes_mode(self) -> None:
        # Live connect_redis client has NO decode_responses -> bytes come back.
        r = fakeredis.FakeStrictRedis()
        r.set("augur:test:k", json.dumps({"a": 1}))
        pm = PersistenceManager(r)
        assert pm._get_json("augur:test:k") == {"a": 1}

    def test_get_json_present_decodes_decode_mode(self) -> None:
        r = fakeredis.FakeStrictRedis(decode_responses=True)
        r.set("augur:test:k", json.dumps({"a": 1}))
        pm = PersistenceManager(r)
        assert pm._get_json("augur:test:k") == {"a": 1}

    def test_get_json_present_default_is_ignored(self) -> None:
        r = fakeredis.FakeStrictRedis()
        r.set("augur:test:k", json.dumps({"a": 1}))
        pm = PersistenceManager(r)
        # A present value wins over the default.
        assert pm._get_json("augur:test:k", {"fallback": True}) == {"a": 1}

    def test_get_json_corrupt_value_raises(self) -> None:
        # Preserves prior behavior: methods using this pattern raised on a
        # present-but-malformed value (the corrupt-tolerant loaders that catch
        # JSONDecodeError keep their own bespoke bodies, not this helper).
        r = fakeredis.FakeStrictRedis()
        r.set("augur:test:k", b"{not valid json")
        pm = PersistenceManager(r)
        with pytest.raises(json.JSONDecodeError):
            pm._get_json("augur:test:k")

    # -- public methods still satisfy their pre-refactor contract -----------

    def test_refactored_loader_returns_none_when_absent(self) -> None:
        pm = PersistenceManager(fakeredis.FakeStrictRedis())
        assert pm.load_baseline("chess", "move", "e2e4") is None
        assert pm.load_escalation_matrix() is None

    def test_refactored_save_load_round_trip(self) -> None:
        pm = PersistenceManager(fakeredis.FakeStrictRedis())
        pm.save_baseline("chess", "move", "e2e4", {"mean": 1.0, "observation_count": 5})
        assert pm.load_baseline("chess", "move", "e2e4") == {
            "mean": 1.0,
            "observation_count": 5,
        }
