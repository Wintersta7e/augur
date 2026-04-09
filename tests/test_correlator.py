"""Unit tests for reasoning/correlator.py.

Pure-logic tests only — no NATS or live Redis. Redis is mocked.
"""

from __future__ import annotations



from reasoning.correlator import (
    CORRELATION_WINDOW_S,
    PRUNE_WINDOW_S,
    SEVERITY_ORDER,
    normalize_rule_key,
)


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
