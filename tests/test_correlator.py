"""Unit tests for reasoning/correlator.py.

Pure-logic tests only — no NATS or live Redis. Redis is mocked.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from reasoning.correlator import (
    CORRELATION_WINDOW_S,
    DEFAULT_ESCALATION_MATRIX,
    PRUNE_WINDOW_S,
    SEVERITY_ORDER,
    add_to_window,
    lookup_escalation,
    normalize_rule_key,
    parse_timestamp,
    query_window,
)


def _make_anomaly(
    domain: str,
    entity: str,
    severity: str,
    ts_iso: str,
    value: float = 1.0,
) -> dict:
    return {
        "domain": domain,
        "stream_id": f"{domain}_stream",
        "entity": entity,
        "event_type": "test",
        "value": value,
        "unit": "s",
        "context": {},
        "session_id": "test-session",
        "baseline_mean": 0.0,
        "baseline_std": 1.0,
        "deviation_score": 2.5,
        "anomaly_score": 0.5,
        "severity": severity,
        "timestamp": ts_iso,
    }


class TestNormalizeRuleKey:
    def test_rank_order_not_alphabetical(self) -> None:
        # Alphabetical would give 'HIGH+LOW' — wrong
        assert normalize_rule_key("HIGH", "LOW") == "LOW+HIGH"

    def test_low_medium_rank_ordered(self) -> None:
        assert normalize_rule_key("MEDIUM", "LOW") == "LOW+MEDIUM"

    def test_medium_high_rank_ordered(self) -> None:
        assert normalize_rule_key("HIGH", "MEDIUM") == "MEDIUM+HIGH"

    def test_same_severity_pair(self) -> None:
        assert normalize_rule_key("LOW", "LOW") == "LOW+LOW"
        assert normalize_rule_key("MEDIUM", "MEDIUM") == "MEDIUM+MEDIUM"
        assert normalize_rule_key("HIGH", "HIGH") == "HIGH+HIGH"

    def test_lowercase_inputs_are_uppercased(self) -> None:
        # Detector emits lowercase — correlator is the boundary
        assert normalize_rule_key("low", "high") == "LOW+HIGH"
        assert normalize_rule_key("low", "low") == "LOW+LOW"

    def test_mixed_case_inputs(self) -> None:
        assert normalize_rule_key("Low", "Medium") == "LOW+MEDIUM"
        assert normalize_rule_key("HIGH", "low") == "LOW+HIGH"

    def test_unknown_severity_returns_none(self) -> None:
        assert normalize_rule_key("CRITICAL", "LOW") is None
        assert normalize_rule_key("LOW", "UNKNOWN") is None
        assert normalize_rule_key("", "LOW") is None


class TestConstants:
    def test_correlation_window_is_30_seconds(self) -> None:
        assert CORRELATION_WINDOW_S == 30

    def test_prune_window_is_twice_correlation_window(self) -> None:
        # Prune boundary must always be derived from query window
        assert PRUNE_WINDOW_S == 2 * CORRELATION_WINDOW_S

    def test_severity_order_ranks_low_medium_high(self) -> None:
        assert SEVERITY_ORDER["LOW"] < SEVERITY_ORDER["MEDIUM"]
        assert SEVERITY_ORDER["MEDIUM"] < SEVERITY_ORDER["HIGH"]


class TestLookupEscalation:
    """lookup_escalation(sev1, sev2, matrix) -> (combined, rule_label)."""

    def test_low_low_escalates_to_medium(self) -> None:
        combined, rule = lookup_escalation("low", "low", DEFAULT_ESCALATION_MATRIX)
        assert combined == "MEDIUM"
        assert rule == "LOW+LOW→MEDIUM"

    def test_low_high_escalates_to_high(self) -> None:
        combined, rule = lookup_escalation("low", "high", DEFAULT_ESCALATION_MATRIX)
        assert combined == "HIGH"
        assert rule == "LOW+HIGH→HIGH"

    def test_medium_medium_escalates_to_high(self) -> None:
        combined, rule = lookup_escalation(
            "medium", "medium", DEFAULT_ESCALATION_MATRIX
        )
        assert combined == "HIGH"
        assert rule == "MEDIUM+MEDIUM→HIGH"

    def test_high_high_stays_high(self) -> None:
        combined, rule = lookup_escalation("high", "high", DEFAULT_ESCALATION_MATRIX)
        assert combined == "HIGH"
        assert rule == "HIGH+HIGH→HIGH"

    def test_all_six_defined_pairs_present(self) -> None:
        # Guard against a future edit removing matrix entries
        pairs = [
            ("low", "low"),
            ("low", "medium"),
            ("low", "high"),
            ("medium", "medium"),
            ("medium", "high"),
            ("high", "high"),
        ]
        for s1, s2 in pairs:
            combined, _ = lookup_escalation(s1, s2, DEFAULT_ESCALATION_MATRIX)
            assert combined in {"MEDIUM", "HIGH"}

    def test_unknown_rule_falls_back_to_higher_severity(self) -> None:
        # Matrix missing the requested entry
        sparse_matrix = {"version": "1.0", "rules": {"LOW+LOW": "MEDIUM"}}
        combined, rule = lookup_escalation("low", "high", sparse_matrix)
        assert combined == "HIGH"
        assert rule is None  # fallback path signals no matrix hit

    def test_unknown_severity_falls_back_to_higher(self) -> None:
        combined, rule = lookup_escalation("weird", "high", DEFAULT_ESCALATION_MATRIX)
        assert combined == "HIGH"
        assert rule is None

    def test_unknown_severity_both_sides_uppercases_and_picks_first(self) -> None:
        # Pathological case: both unknown — return the first uppercased, no rule
        combined, rule = lookup_escalation("weird", "other", DEFAULT_ESCALATION_MATRIX)
        assert combined == "WEIRD"  # caller already dropped in real flow
        assert rule is None


class TestParseTimestamp:
    def test_iso_utc_z_suffix(self) -> None:
        ts = parse_timestamp("2026-03-17T14:30:00+00:00")
        assert ts > 0

    def test_two_timestamps_produce_expected_delta(self) -> None:
        t1 = parse_timestamp("2026-03-17T14:30:00+00:00")
        t2 = parse_timestamp("2026-03-17T14:30:30+00:00")
        assert abs((t2 - t1) - 30.0) < 0.001


class TestAddToWindow:
    def test_zadd_called_with_json_member_and_timestamp_score(self) -> None:
        mock_redis = MagicMock()
        anomaly = _make_anomaly("chess", "white", "low", "2026-03-17T14:30:00+00:00")

        add_to_window(mock_redis, anomaly)

        mock_redis.zadd.assert_called_once()
        args, _ = mock_redis.zadd.call_args
        key, mapping = args
        assert key == "augur:correlation:window"
        # mapping is {json_str: score}
        assert len(mapping) == 1
        member, score = next(iter(mapping.items()))
        assert json.loads(member)["domain"] == "chess"
        expected = parse_timestamp("2026-03-17T14:30:00+00:00")
        assert abs(score - expected) < 0.001

    def test_prune_call_uses_prune_window_boundary(self) -> None:
        mock_redis = MagicMock()
        anomaly = _make_anomaly("chess", "white", "low", "2026-03-17T14:30:00+00:00")

        add_to_window(mock_redis, anomaly)

        # ZREMRANGEBYSCORE is the pruning primitive
        mock_redis.zremrangebyscore.assert_called_once()
        args, _ = mock_redis.zremrangebyscore.call_args
        key, min_score, max_score = args
        assert key == "augur:correlation:window"
        assert min_score == "-inf"
        now = parse_timestamp("2026-03-17T14:30:00+00:00")
        # Prune everything older than now - 60s
        assert abs(max_score - (now - PRUNE_WINDOW_S)) < 0.001


class TestQueryWindow:
    def test_returns_events_from_other_domains_only(self) -> None:
        primary_ts = "2026-03-17T14:30:00+00:00"
        primary = _make_anomaly("chess", "white", "low", primary_ts)

        other_domain = _make_anomaly(
            "typing", "user", "low", "2026-03-17T14:29:50+00:00"
        )
        same_domain = _make_anomaly(
            "chess", "black", "low", "2026-03-17T14:29:55+00:00"
        )

        mock_redis = MagicMock()
        # ZRANGEBYSCORE returns members (bytes) for the 30s window
        mock_redis.zrangebyscore.return_value = [
            json.dumps(other_domain).encode(),
            json.dumps(same_domain).encode(),
            json.dumps(primary).encode(),
        ]

        results = query_window(mock_redis, primary)

        assert len(results) == 1
        assert results[0]["domain"] == "typing"

    def test_zrangebyscore_called_with_inclusive_30s_window(self) -> None:
        primary_ts = "2026-03-17T14:30:00+00:00"
        primary = _make_anomaly("chess", "white", "low", primary_ts)

        mock_redis = MagicMock()
        mock_redis.zrangebyscore.return_value = []

        query_window(mock_redis, primary)

        args, _ = mock_redis.zrangebyscore.call_args
        key, min_score, max_score = args
        assert key == "augur:correlation:window"
        now = parse_timestamp(primary_ts)
        assert abs(max_score - now) < 0.001
        assert abs(min_score - (now - CORRELATION_WINDOW_S)) < 0.001

    def test_excludes_primary_itself_when_already_in_set(self) -> None:
        # Primary shares domain with itself, so domain filter drops it
        primary = _make_anomaly("chess", "white", "low", "2026-03-17T14:30:00+00:00")
        mock_redis = MagicMock()
        mock_redis.zrangebyscore.return_value = [json.dumps(primary).encode()]

        assert query_window(mock_redis, primary) == []

    def test_handles_string_members_from_decoded_redis(self) -> None:
        primary = _make_anomaly("chess", "white", "low", "2026-03-17T14:30:00+00:00")
        other = _make_anomaly("typing", "user", "low", "2026-03-17T14:29:55+00:00")

        mock_redis = MagicMock()
        mock_redis.zrangebyscore.return_value = [json.dumps(other)]  # str, not bytes

        results = query_window(mock_redis, primary)
        assert len(results) == 1
        assert results[0]["domain"] == "typing"
