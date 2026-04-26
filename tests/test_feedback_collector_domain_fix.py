"""Tests for the primary_domain fix and new metadata capture in PendingAdvice."""

from perception.feedback_collector import PendingAdvice, _resolve_primary_domain


def test_pending_advice_records_correct_domain():
    p = PendingAdvice(
        advice_id="abc",
        domain="typing",
        entity="kbd",
        severity="medium",
        baseline_mean=0.4,
        timestamp="2026-04-25T12:00:00+00:00",
    )
    record = p.to_record()
    assert record["domain"] == "typing"


def test_pending_advice_carries_new_fields():
    p = PendingAdvice(
        advice_id="abc",
        domain="chess",
        entity="white",
        severity="low",
        baseline_mean=12.5,
        timestamp="2026-04-25T12:00:00+00:00",
        correlation_found=True,
        correlated_domains=["typing"],
        rule_key="LOW+LOW",
        escalation_rule="LOW+LOW→MEDIUM",
        involved_domains=["chess", "typing"],
        temporal_lag_seconds=5.0,
        correlation_span_s=5.0,
        rule_window_s=30.0,
    )
    record = p.to_record()
    assert record["involved_domains"] == ["chess", "typing"]
    assert record["temporal_lag_seconds"] == 5.0
    assert record["correlation_span_s"] == 5.0
    assert record["rule_window_s"] == 30.0


def test_pending_advice_new_fields_default_none():
    p = PendingAdvice(
        advice_id="abc",
        domain="chess",
        entity="white",
        severity="low",
        baseline_mean=12.5,
        timestamp="2026-04-25T12:00:00+00:00",
    )
    record = p.to_record()
    assert record["involved_domains"] == []
    assert record["temporal_lag_seconds"] is None
    assert record["correlation_span_s"] is None
    assert record["rule_window_s"] is None




def test_resolve_primary_domain_from_top_level():
    assert _resolve_primary_domain({"domain": "typing"}) == "typing"


def test_resolve_primary_domain_from_primary_anomaly():
    assert (
        _resolve_primary_domain({"primary_anomaly": {"domain": "typing"}}) == "typing"
    )


def test_resolve_primary_domain_top_level_wins_over_primary_anomaly():
    assert (
        _resolve_primary_domain(
            {
                "domain": "typing",
                "primary_anomaly": {"domain": "chess"},
            }
        )
        == "typing"
    )


def test_resolve_primary_domain_falls_back_to_chess():
    assert _resolve_primary_domain({}) == "chess"
