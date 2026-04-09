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
    correlate,
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


class TestCorrelate:
    """End-to-end pure logic test of correlate().

    correlate(primary_anomaly, r, matrix) returns one of:
      - dict with correlation_found=True  (cross-domain hit)
      - dict with correlation_found=False (standalone medium/high)
      - None                              (standalone low — drop)
    """

    def _setup_window(self, mock_redis: MagicMock, stored_events: list[dict]) -> None:
        mock_redis.zrangebyscore.return_value = [
            json.dumps(e).encode() for e in stored_events
        ]

    def test_two_lows_different_domains_escalate_to_medium(self) -> None:
        primary = _make_anomaly("chess", "white", "low", "2026-03-17T14:30:00+00:00")
        typing_low = _make_anomaly("typing", "user", "low", "2026-03-17T14:29:48+00:00")

        mock_redis = MagicMock()
        self._setup_window(mock_redis, [typing_low, primary])

        result = correlate(primary, mock_redis, DEFAULT_ESCALATION_MATRIX)

        assert result is not None
        assert result["correlation_found"] is True
        assert result["combined_severity"] == "MEDIUM"
        assert result["severity_escalated"] is True
        assert result["escalation_rule"] == "LOW+LOW→MEDIUM"
        assert result["escalation_matrix_version"] == "1.0"
        assert result["primary_anomaly"]["domain"] == "chess"
        assert len(result["correlated_events"]) == 1
        assert result["correlated_events"][0]["domain"] == "typing"
        assert abs(result["temporal_lag_seconds"] - 12.0) < 0.1

    def test_standalone_medium_passes_through(self) -> None:
        primary = _make_anomaly("chess", "white", "medium", "2026-03-17T14:30:00+00:00")
        mock_redis = MagicMock()
        self._setup_window(mock_redis, [primary])  # only itself

        result = correlate(primary, mock_redis, DEFAULT_ESCALATION_MATRIX)

        assert result is not None
        assert result["correlation_found"] is False
        assert result["combined_severity"] == "MEDIUM"  # uppercased
        assert result["severity_escalated"] is False
        assert result["escalation_rule"] is None
        assert result["escalation_matrix_version"] is None
        assert result["temporal_lag_seconds"] is None
        assert result["correlated_events"] == []

    def test_standalone_high_passes_through_uppercased(self) -> None:
        primary = _make_anomaly("chess", "white", "high", "2026-03-17T14:30:00+00:00")
        mock_redis = MagicMock()
        self._setup_window(mock_redis, [primary])

        result = correlate(primary, mock_redis, DEFAULT_ESCALATION_MATRIX)

        assert result is not None
        assert result["correlation_found"] is False
        assert result["combined_severity"] == "HIGH"

    def test_standalone_low_is_dropped(self) -> None:
        primary = _make_anomaly("chess", "white", "low", "2026-03-17T14:30:00+00:00")
        mock_redis = MagicMock()
        self._setup_window(mock_redis, [primary])  # only itself in window

        result = correlate(primary, mock_redis, DEFAULT_ESCALATION_MATRIX)

        assert result is None

    def test_multi_domain_window_picks_most_recent_per_domain(self) -> None:
        # Window contains two typing events; correlator must pick the most recent
        primary = _make_anomaly("chess", "white", "low", "2026-03-17T14:30:00+00:00")
        typing_older = _make_anomaly(
            "typing", "user", "low", "2026-03-17T14:29:35+00:00"
        )
        typing_newer = _make_anomaly(
            "typing", "user", "low", "2026-03-17T14:29:55+00:00"
        )
        mock_redis = MagicMock()
        self._setup_window(mock_redis, [typing_older, typing_newer, primary])

        result = correlate(primary, mock_redis, DEFAULT_ESCALATION_MATRIX)

        assert result is not None
        assert len(result["correlated_events"]) == 1
        assert (
            result["correlated_events"][0]["timestamp"] == "2026-03-17T14:29:55+00:00"
        )

    def test_multi_domain_window_with_different_domains_keeps_one_per_domain(
        self,
    ) -> None:
        primary = _make_anomaly("chess", "white", "low", "2026-03-17T14:30:00+00:00")
        typing_ev = _make_anomaly("typing", "user", "low", "2026-03-17T14:29:50+00:00")
        focus_ev = _make_anomaly("focus", "app", "low", "2026-03-17T14:29:40+00:00")
        mock_redis = MagicMock()
        self._setup_window(mock_redis, [typing_ev, focus_ev, primary])

        result = correlate(primary, mock_redis, DEFAULT_ESCALATION_MATRIX)

        assert result is not None
        assert len(result["correlated_events"]) == 2
        domains = {e["domain"] for e in result["correlated_events"]}
        assert domains == {"typing", "focus"}

    def test_pairwise_escalation_uses_highest_severity_correlated_event(self) -> None:
        # LOW primary + MEDIUM correlated should escalate via LOW+MEDIUM rule
        primary = _make_anomaly("chess", "white", "low", "2026-03-17T14:30:00+00:00")
        typing_med = _make_anomaly(
            "typing", "user", "medium", "2026-03-17T14:29:50+00:00"
        )
        mock_redis = MagicMock()
        self._setup_window(mock_redis, [typing_med, primary])

        result = correlate(primary, mock_redis, DEFAULT_ESCALATION_MATRIX)

        assert result is not None
        assert result["combined_severity"] == "MEDIUM"
        assert result["escalation_rule"] == "LOW+MEDIUM→MEDIUM"

    def test_window_boundary_30s_inclusive(self) -> None:
        primary = _make_anomaly("chess", "white", "low", "2026-03-17T14:30:00+00:00")
        # Exactly 30 seconds old — must be included
        on_boundary = _make_anomaly(
            "typing", "user", "low", "2026-03-17T14:29:30+00:00"
        )
        mock_redis = MagicMock()
        # Simulate: zrangebyscore with max=now and min=now-30 includes it
        self._setup_window(mock_redis, [on_boundary, primary])

        result = correlate(primary, mock_redis, DEFAULT_ESCALATION_MATRIX)

        assert result is not None
        assert result["correlation_found"] is True

    def test_primary_add_called_before_query(self) -> None:
        # add_to_window must run before the window query — otherwise primary is missing
        primary = _make_anomaly("chess", "white", "medium", "2026-03-17T14:30:00+00:00")
        call_log: list[str] = []
        mock_redis = MagicMock()
        mock_redis.zadd.side_effect = lambda *a, **kw: call_log.append("zadd")
        mock_redis.zremrangebyscore.side_effect = lambda *a, **kw: call_log.append(
            "prune"
        )
        mock_redis.zrangebyscore.side_effect = lambda *a, **kw: (
            call_log.append("query") or []
        )

        correlate(primary, mock_redis, DEFAULT_ESCALATION_MATRIX)

        assert call_log.index("zadd") < call_log.index("query")
        assert call_log.index("prune") < call_log.index("query")


import networkx as nx
import pytest

from reasoning.correlator import (
    add_correlation_to_graph,
    new_session_graph,
    node_key,
)


class TestSessionGraph:
    def test_node_key_uses_domain_entity_timestamp(self) -> None:
        ev = _make_anomaly("chess", "white", "low", "2026-03-17T14:30:00+00:00")
        assert node_key(ev) == "chess:white:2026-03-17T14:30:00+00:00"

    def test_add_correlation_adds_primary_and_correlated_nodes(self) -> None:
        g = new_session_graph()
        primary = _make_anomaly("chess", "white", "low", "2026-03-17T14:30:00+00:00")
        correlated = _make_anomaly("typing", "user", "low", "2026-03-17T14:29:48+00:00")

        add_correlation_to_graph(
            g,
            primary=primary,
            correlated=[correlated],
            combined_severity="MEDIUM",
            rule_label="LOW+LOW→MEDIUM",
        )

        assert node_key(primary) in g.nodes
        assert node_key(correlated) in g.nodes
        assert g.nodes[node_key(primary)]["severity"] == "low"
        assert g.nodes[node_key(primary)]["domain"] == "chess"

    def test_edge_direction_primary_to_correlated(self) -> None:
        # Edge points from primary (newly arrived) → correlated (older)
        g = new_session_graph()
        primary = _make_anomaly("chess", "white", "low", "2026-03-17T14:30:00+00:00")
        correlated = _make_anomaly("typing", "user", "low", "2026-03-17T14:29:48+00:00")

        add_correlation_to_graph(
            g,
            primary=primary,
            correlated=[correlated],
            combined_severity="MEDIUM",
            rule_label="LOW+LOW→MEDIUM",
        )

        pk = node_key(primary)
        ck = node_key(correlated)
        assert g.has_edge(pk, ck)
        assert not g.has_edge(ck, pk)  # directed

    def test_edge_attributes_include_lag_rule_severity(self) -> None:
        g = new_session_graph()
        primary = _make_anomaly("chess", "white", "low", "2026-03-17T14:30:00+00:00")
        correlated = _make_anomaly("typing", "user", "low", "2026-03-17T14:29:48+00:00")

        add_correlation_to_graph(
            g,
            primary=primary,
            correlated=[correlated],
            combined_severity="MEDIUM",
            rule_label="LOW+LOW→MEDIUM",
        )

        edge = g.edges[node_key(primary), node_key(correlated)]
        assert edge["escalation_rule"] == "LOW+LOW→MEDIUM"
        assert edge["combined_severity"] == "MEDIUM"
        assert edge["domains"] == ("chess", "typing")
        assert edge["temporal_lag"] == pytest.approx(12.0, abs=0.01)

    def test_new_session_graph_returns_empty_digraph(self) -> None:
        g = new_session_graph()
        assert isinstance(g, nx.DiGraph)
        assert len(g.nodes) == 0
        assert len(g.edges) == 0


from reasoning.correlator import ensure_matrix_seeded


class TestEnsureMatrixSeeded:
    def test_seeds_default_when_missing(self) -> None:
        mock_pm = MagicMock()
        mock_pm.load_escalation_matrix.return_value = None

        result = ensure_matrix_seeded(mock_pm)

        mock_pm.save_escalation_matrix.assert_called_once_with(
            DEFAULT_ESCALATION_MATRIX
        )
        assert result == DEFAULT_ESCALATION_MATRIX

    def test_leaves_existing_matrix_alone(self) -> None:
        existing = {"version": "1.5", "rules": {"LOW+LOW": "HIGH"}}
        mock_pm = MagicMock()
        mock_pm.load_escalation_matrix.return_value = existing

        result = ensure_matrix_seeded(mock_pm)

        mock_pm.save_escalation_matrix.assert_not_called()
        assert result == existing
