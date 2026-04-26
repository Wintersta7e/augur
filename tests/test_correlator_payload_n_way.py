"""Tests for _build_correlation_payload with N-way + new payload fields."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

from blackboard.config import AugurConfig
from reasoning.correlator import (
    DEFAULT_ESCALATION_MATRIX,
    _build_correlation_payload,
    correlate,
)


def _ts(seconds_ago: float) -> str:
    return datetime.fromtimestamp(
        datetime.now(timezone.utc).timestamp() - seconds_ago, timezone.utc
    ).isoformat()


def _ev(domain: str, severity: str, lag_s: float, entity: str = "e1") -> dict:
    return {
        "domain": domain,
        "severity": severity,
        "entity": entity,
        "timestamp": _ts(lag_s),
        "value": 1.0,
    }


def _config() -> AugurConfig:
    return AugurConfig.from_env()


# Pairwise payload structure (regression check) ------------------------------


def test_pairwise_payload_carries_new_fields():
    primary = _ev("chess", "low", 0)
    correlated = [_ev("typing", "low", 5)]
    payload = _build_correlation_payload(
        primary, correlated, DEFAULT_ESCALATION_MATRIX, _config()
    )
    assert payload["rule_key"] == "LOW+LOW"
    assert payload["combined_severity"] == "MEDIUM"
    assert payload["temporal_lag_seconds"] == 5.0  # closest event
    assert payload["correlation_span_s"] == 5.0  # max lag
    assert payload["involved_domains"] == ["chess", "typing"]
    assert payload["rule_window_s"] == 30.0  # default


# 3-way payload structure ----------------------------------------------------


def test_3way_payload_uses_3way_rule():
    primary = _ev("chess", "low", 0)
    correlated = [_ev("typing", "low", 5), _ev("focus", "low", 10)]
    payload = _build_correlation_payload(
        primary, correlated, DEFAULT_ESCALATION_MATRIX, _config()
    )
    assert payload["rule_key"] == "LOW+LOW+LOW"
    assert payload["combined_severity"] == "MEDIUM"  # per default 3-way rule
    assert payload["escalation_rule"] == "LOW+LOW+LOW→MEDIUM"
    assert payload["correlation_span_s"] == 10.0  # furthest of the two
    assert payload["temporal_lag_seconds"] == 5.0  # closest of the two
    assert payload["involved_domains"] == ["chess", "focus", "typing"]


def test_3way_with_mixed_severities():
    primary = _ev("chess", "low", 0)
    correlated = [_ev("typing", "medium", 5), _ev("focus", "high", 8)]
    payload = _build_correlation_payload(
        primary, correlated, DEFAULT_ESCALATION_MATRIX, _config()
    )
    assert payload["rule_key"] == "LOW+MEDIUM+HIGH"
    assert payload["combined_severity"] == "HIGH"


# Per-rule window threading --------------------------------------------------


def test_payload_uses_per_rule_window_when_present():
    matrix = dict(DEFAULT_ESCALATION_MATRIX)
    matrix["rule_windows"] = {"LOW+LOW": 25.0}
    primary = _ev("chess", "low", 0)
    correlated = [_ev("typing", "low", 5)]
    payload = _build_correlation_payload(primary, correlated, matrix, _config())
    assert payload["rule_window_s"] == 25.0


# Pass-through payload (no correlation) --------------------------------------


def test_passthrough_carries_new_fields_as_null_or_singleton():
    primary = _ev("chess", "medium", 0)
    redis = MagicMock()
    # No prior events in window → no correlation found, pass-through path
    redis.zrangebyscore.return_value = []
    payload = correlate(primary, redis, DEFAULT_ESCALATION_MATRIX, _config())
    assert payload is not None
    assert payload["correlation_found"] is False
    assert payload["correlation_span_s"] is None
    assert payload["rule_window_s"] is None
    assert payload["involved_domains"] == ["chess"]


# Filter end-to-end ---------------------------------------------------------


def test_correlate_filters_candidate_beyond_pairwise_window():
    primary = _ev("chess", "low", 0)
    far_cand = _ev("typing", "low", 50)  # 50s lag — beyond 30s default
    redis = MagicMock()
    import json

    redis.zrangebyscore.return_value = [json.dumps(far_cand).encode()]
    payload = correlate(primary, redis, DEFAULT_ESCALATION_MATRIX, _config())
    # candidate filtered out → standalone low → drop (returns None)
    assert payload is None
