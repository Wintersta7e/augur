"""Tests for _attribution_weights — multi-domain feedback attribution helper."""

from reasoning.reflection_engine import _attribution_weights


def test_standalone_advice_full_weight_to_primary():
    event = {"correlation_found": False, "domain": "chess"}
    assert _attribution_weights(event) == {"chess": 1.0}


def test_correlated_advice_equal_weighted_across_domains():
    event = {
        "correlation_found": True,
        "domain": "chess",
        "involved_domains": ["chess", "typing"],
    }
    weights = _attribution_weights(event)
    assert weights == {"chess": 0.5, "typing": 0.5}


def test_correlated_3way_one_third_each():
    event = {
        "correlation_found": True,
        "domain": "chess",
        "involved_domains": ["chess", "typing", "focus"],
    }
    weights = _attribution_weights(event)
    assert weights["chess"] == weights["typing"] == weights["focus"]
    assert sum(weights.values()) == 1.0


def test_old_record_without_involved_domains_falls_back_to_primary():
    event = {"correlation_found": True, "domain": "chess"}  # no involved_domains
    assert _attribution_weights(event) == {"chess": 1.0}


def test_correlation_found_with_empty_involved_domains():
    event = {
        "correlation_found": True,
        "domain": "chess",
        "involved_domains": [],
    }
    # Empty list → fall back to primary
    assert _attribution_weights(event) == {"chess": 1.0}


def test_missing_domain_falls_back_to_unknown():
    event = {"correlation_found": False}
    assert _attribution_weights(event) == {"unknown": 1.0}
