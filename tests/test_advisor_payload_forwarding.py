"""Test that the advisor forwards new correlation payload fields into advice."""

from reasoning.augur_advisor import _build_advice_event


def _make_correlation_payload() -> dict:
    return {
        "primary_anomaly": {
            "domain": "chess",
            "entity": "white",
            "severity": "low",
            "value": 12.5,
            "timestamp": "2026-04-25T12:00:00+00:00",
        },
        "correlated_events": [
            {
                "domain": "typing",
                "entity": "kbd",
                "severity": "low",
                "value": 0.4,
                "timestamp": "2026-04-25T11:59:55+00:00",
            },
        ],
        "correlation_found": True,
        "rule_key": "LOW+LOW",
        "escalation_rule": "LOW+LOW→MEDIUM",
        "combined_severity": "MEDIUM",
        "involved_domains": ["chess", "typing"],
        "temporal_lag_seconds": 5.0,
        "correlation_span_s": 5.0,
        "rule_window_s": 30.0,
    }


def test_advice_event_forwards_involved_domains():
    payload = _make_correlation_payload()
    advice = _build_advice_event(payload, advice_text="...", model_used="qwen2.5:32b")
    assert advice["involved_domains"] == ["chess", "typing"]


def test_advice_event_forwards_correlation_span_s():
    payload = _make_correlation_payload()
    advice = _build_advice_event(payload, advice_text="...", model_used="qwen2.5:32b")
    assert advice["correlation_span_s"] == 5.0


def test_advice_event_forwards_rule_window_s():
    payload = _make_correlation_payload()
    advice = _build_advice_event(payload, advice_text="...", model_used="qwen2.5:32b")
    assert advice["rule_window_s"] == 30.0


def test_advice_event_handles_missing_new_fields_gracefully():
    payload = {
        "primary_anomaly": {
            "domain": "chess",
            "entity": "w",
            "severity": "medium",
            "value": 1.0,
            "timestamp": "2026-04-25T12:00:00+00:00",
        },
        "correlated_events": [],
        "correlation_found": False,
    }
    advice = _build_advice_event(payload, advice_text="...", model_used="qwen2.5:32b")
    # missing new fields → empty list / None defaults
    assert advice["involved_domains"] == ["chess"]
    assert advice["correlation_span_s"] is None
    assert advice["rule_window_s"] is None
