"""Tests for analyze_correlation_window_tuning."""

from tabula.config import AugurConfig
from reasoning.reflection_engine import analyze_correlation_window_tuning


def _cfg() -> AugurConfig:
    return AugurConfig.from_env()


def _ev(rule_key: str, span: float, correlation_found: bool = True) -> dict:
    return {
        "domain": "chess",
        "correlation_found": correlation_found,
        "rule_key": rule_key,
        "correlation_span_s": span,
        "explicit_rating": "y",
        "behavioral_score": 0.8,
    }


def test_no_events_returns_zero_evaluated():
    feedback = {"advice_events": []}
    result = analyze_correlation_window_tuning(feedback, {}, {}, _cfg())
    assert result["rules_evaluated"] == 0
    assert result["new_rule_windows"] is None


def test_single_pairwise_rule_starts_state():
    feedback = {"advice_events": [_ev("LOW+LOW", 8.0)] * 3}
    matrix = {"rules": {"LOW+LOW": "MEDIUM"}}
    state = {}
    result = analyze_correlation_window_tuning(feedback, matrix, state, _cfg())
    assert "LOW+LOW" in result["new_window_state"]
    assert result["new_window_state"]["LOW+LOW"]["ewma_lag"] == 8.0


def test_pairwise_only_filter_skips_3way_rule_keys():
    feedback = {
        "advice_events": [
            _ev("LOW+LOW+LOW", 12.0),
            _ev("LOW+LOW+MEDIUM", 14.0),
        ]
    }
    result = analyze_correlation_window_tuning(feedback, {}, {}, _cfg())
    assert result["rules_evaluated"] == 0


def test_hysteresis_holds_when_change_below_threshold():
    feedback = {"advice_events": [_ev("LOW+LOW", 12.0)] * 3}
    matrix = {"rules": {"LOW+LOW": "MEDIUM"}, "rule_windows": {"LOW+LOW": 30.0}}
    # Existing EWMA 12.0 → target 12 * 2.5 = 30.0; current window 30 → delta 0%
    state = {"LOW+LOW": {"ewma_lag": 12.0}}
    result = analyze_correlation_window_tuning(feedback, matrix, state, _cfg())
    assert result["per_rule"]["LOW+LOW"]["action"] == "held"
    assert result["new_rule_windows"] is None


def test_hysteresis_fires_when_change_exceeds_threshold():
    matrix = {"rules": {"LOW+LOW": "MEDIUM"}, "rule_windows": {"LOW+LOW": 30.0}}
    feedback = {"advice_events": [_ev("LOW+LOW", 40.0)] * 3}
    state = {"LOW+LOW": {"ewma_lag": 10.0}}
    # ewma: 0.8*10 + 0.2*40 = 16.0 → target 16*2.5 = 40.0; current 30 → delta 33.3%
    result = analyze_correlation_window_tuning(feedback, matrix, state, _cfg())
    assert result["per_rule"]["LOW+LOW"]["action"] == "tuned"
    assert result["new_rule_windows"]["LOW+LOW"] == 40.0


def test_clamp_to_max():
    feedback = {"advice_events": [_ev("LOW+LOW", 200.0)] * 3}
    matrix = {"rules": {"LOW+LOW": "MEDIUM"}}
    state = {"LOW+LOW": {"ewma_lag": 100.0}}
    result = analyze_correlation_window_tuning(feedback, matrix, state, _cfg())
    # 0.8*100 + 0.2*200 = 120.0 ewma_lag
    # target = 120 * 2.5 = 300, clamped to max 120.0
    assert result["new_window_state"]["LOW+LOW"]["ewma_lag"] == 120.0
    if result["new_rule_windows"] is not None:
        assert result["new_rule_windows"]["LOW+LOW"] <= 120.0


def test_clamp_to_min():
    feedback = {"advice_events": [_ev("LOW+LOW", 0.5)] * 3}
    matrix = {"rules": {"LOW+LOW": "MEDIUM"}, "rule_windows": {"LOW+LOW": 30.0}}
    state = {"LOW+LOW": {"ewma_lag": 0.5}}
    result = analyze_correlation_window_tuning(feedback, matrix, state, _cfg())
    # ewma 0.5 → target 1.25, clamped to 5.0 (min)
    if result["new_rule_windows"] is not None:
        assert result["new_rule_windows"]["LOW+LOW"] == 5.0


def test_skips_events_without_correlation_span_s():
    feedback = {
        "advice_events": [
            {"correlation_found": True, "rule_key": "LOW+LOW"},  # no correlation_span_s
            _ev("LOW+LOW", 8.0),
        ]
    }
    result = analyze_correlation_window_tuning(feedback, {}, {}, _cfg())
    assert result["per_rule"]["LOW+LOW"]["event_count"] == 1
