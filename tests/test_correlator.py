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


from reasoning.correlator import (  # noqa: E402
    DEFAULT_ESCALATION_MATRIX,
    lookup_escalation,
)


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
