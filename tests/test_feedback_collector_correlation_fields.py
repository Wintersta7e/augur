"""Tests for the PendingAdvice correlation metadata extension.

Phase 3B added correlation_found and correlated_domains to the advice
payload, but the feedback collector never captured them. This extends
PendingAdvice with four correlation fields (correlation_found,
correlated_domains, rule_key, escalation_rule) so the reflection
engine can tune the escalation matrix based on per-rule feedback.
"""

from __future__ import annotations


from responsum.feedback_collector import PendingAdvice


def _base_kwargs() -> dict:
    return {
        "advice_id": "adv-0001",
        "domain": "chess",
        "entity": "white",
        "severity": "low",
        "baseline_mean": 8.2,
        "timestamp": "2026-04-09T12:00:00+00:00",
    }


class TestDefaultCorrelationFields:
    def test_default_correlation_found_is_false(self) -> None:
        p = PendingAdvice(**_base_kwargs())
        assert p.correlation_found is False

    def test_default_correlated_domains_is_empty_list(self) -> None:
        p = PendingAdvice(**_base_kwargs())
        assert p.correlated_domains == []

    def test_default_rule_key_is_none(self) -> None:
        p = PendingAdvice(**_base_kwargs())
        assert p.rule_key is None

    def test_default_escalation_rule_is_none(self) -> None:
        p = PendingAdvice(**_base_kwargs())
        assert p.escalation_rule is None


class TestExplicitCorrelationFields:
    def test_all_fields_set_explicitly(self) -> None:
        p = PendingAdvice(
            **_base_kwargs(),
            correlation_found=True,
            correlated_domains=["typing", "focus"],
            rule_key="LOW+LOW",
            escalation_rule="LOW+LOW\u2192MEDIUM",
        )
        assert p.correlation_found is True
        assert p.correlated_domains == ["typing", "focus"]
        assert p.rule_key == "LOW+LOW"
        assert p.escalation_rule == "LOW+LOW\u2192MEDIUM"

    def test_correlated_domains_none_becomes_empty_list(self) -> None:
        p = PendingAdvice(**_base_kwargs(), correlated_domains=None)
        assert p.correlated_domains == []


class TestToRecordIncludesCorrelationFields:
    def test_record_has_all_four_fields(self) -> None:
        p = PendingAdvice(
            **_base_kwargs(),
            correlation_found=True,
            correlated_domains=["typing"],
            rule_key="LOW+LOW",
            escalation_rule="LOW+LOW\u2192MEDIUM",
        )
        record = p.to_record()
        assert record["correlation_found"] is True
        assert record["correlated_domains"] == ["typing"]
        assert record["rule_key"] == "LOW+LOW"
        assert record["escalation_rule"] == "LOW+LOW\u2192MEDIUM"

    def test_record_defaults_present_for_standalone_advice(self) -> None:
        p = PendingAdvice(**_base_kwargs())
        record = p.to_record()
        assert record["correlation_found"] is False
        assert record["correlated_domains"] == []
        assert record["rule_key"] is None
        assert record["escalation_rule"] is None

    def test_record_preserves_malformed_escalation_rule(self) -> None:
        # The feedback path does not parse escalation_rule — it stores
        # whatever the advisor sent. Verify garbage is round-tripped.
        p = PendingAdvice(
            **_base_kwargs(),
            escalation_rule="garbage_string",
        )
        record = p.to_record()
        assert record["escalation_rule"] == "garbage_string"
