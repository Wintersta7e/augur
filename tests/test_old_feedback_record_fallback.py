"""Tests for graceful handling of old feedback records lacking new fields."""

from blackboard.config import AugurConfig
from reasoning.reflection_engine import (
    _attribution_weights,
    analyze_correlation_window_tuning,
    analyze_precision,
)


def test_old_record_attribution_falls_back_to_primary():
    old_record = {"correlation_found": True, "domain": "chess"}  # no involved_domains
    assert _attribution_weights(old_record) == {"chess": 1.0}


def test_old_record_precision_treated_as_standalone():
    feedback = {
        "advice_events": [
            {
                "domain": "chess",
                "correlation_found": True,
                "explicit_rating": "y",
                "behavioral_score": 0.8,
            }
            for _ in range(3)
        ],
        "session_summary": {"total_advice": 3},
    }
    result = analyze_precision(
        feedback,
        {"chess": {"sigma_threshold": 2.0}},
        AugurConfig.from_env(),
    )
    # Falls back to {chess: 1.0} for each event
    assert result["per_domain"]["chess"]["total_anomalies"] == 3.0


def test_old_record_window_tuning_skipped_for_missing_span():
    feedback = {
        "advice_events": [
            {"correlation_found": True, "rule_key": "LOW+LOW"},  # no correlation_span_s
        ],
    }
    result = analyze_correlation_window_tuning(feedback, {}, {}, AugurConfig.from_env())
    assert result["rules_evaluated"] == 0
