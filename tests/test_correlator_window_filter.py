"""Unit tests for window resolution and pairwise filtering helpers."""

from datetime import datetime, timezone

from nexus.correlator import (
    compute_prune_window,
    compute_query_window,
    filter_by_pairwise_window,
    get_rule_window,
)


def _ts(seconds_ago: float) -> str:
    """Return an ISO timestamp `seconds_ago` seconds before now."""
    epoch = datetime.now(timezone.utc).timestamp() - seconds_ago
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


# get_rule_window -------------------------------------------------------------


def test_get_rule_window_falls_back_to_default_when_rule_key_none():
    assert get_rule_window(None, {}, default_s=30.0) == 30.0


def test_get_rule_window_falls_back_to_default_when_not_in_map():
    matrix = {"rule_windows": {"LOW+LOW": 25.0}}
    assert get_rule_window("MEDIUM+HIGH", matrix, default_s=30.0) == 30.0


def test_get_rule_window_returns_override():
    matrix = {"rule_windows": {"LOW+LOW": 25.0}}
    assert get_rule_window("LOW+LOW", matrix, default_s=30.0) == 25.0


# compute_query_window --------------------------------------------------------


def test_compute_query_window_no_overrides():
    assert compute_query_window({}, default_s=30.0) == 30.0


def test_compute_query_window_takes_max_of_default_and_overrides():
    matrix = {"rule_windows": {"LOW+LOW": 45.0, "MEDIUM+HIGH": 20.0}}
    assert compute_query_window(matrix, default_s=30.0) == 45.0


def test_compute_query_window_default_dominates_when_overrides_smaller():
    matrix = {"rule_windows": {"LOW+LOW": 10.0}}
    assert compute_query_window(matrix, default_s=30.0) == 30.0


# compute_prune_window --------------------------------------------------------


def test_compute_prune_window_doubles_query():
    assert compute_prune_window(30.0) == 60.0
    assert compute_prune_window(45.0) == 90.0


# filter_by_pairwise_window --------------------------------------------------


def test_filter_keeps_candidate_within_pairwise_default():
    primary = {"severity": "LOW", "domain": "chess", "timestamp": _ts(0)}
    cand = {"severity": "LOW", "domain": "typing", "timestamp": _ts(20)}
    matrix = {"rule_windows": {}}  # no overrides — default applies
    result = filter_by_pairwise_window(primary, [cand], matrix, default_window_s=30.0)
    assert result == [cand]


def test_filter_drops_candidate_beyond_pairwise_default():
    primary = {"severity": "LOW", "domain": "chess", "timestamp": _ts(0)}
    cand = {"severity": "LOW", "domain": "typing", "timestamp": _ts(40)}
    matrix = {"rule_windows": {}}
    result = filter_by_pairwise_window(primary, [cand], matrix, default_window_s=30.0)
    assert result == []


def test_filter_uses_per_rule_override():
    primary = {"severity": "LOW", "domain": "chess", "timestamp": _ts(0)}
    cand = {"severity": "LOW", "domain": "typing", "timestamp": _ts(20)}
    matrix = {"rule_windows": {"LOW+LOW": 15.0}}  # narrower than default
    result = filter_by_pairwise_window(primary, [cand], matrix, default_window_s=30.0)
    assert result == []  # 20s lag > 15s rule window


def test_filter_handles_unknown_severity_with_default_window():
    primary = {"severity": "FAKE", "domain": "chess", "timestamp": _ts(0)}
    cand = {"severity": "LOW", "domain": "typing", "timestamp": _ts(20)}
    matrix = {"rule_windows": {}}
    result = filter_by_pairwise_window(primary, [cand], matrix, default_window_s=30.0)
    assert result == [cand]  # falls back to default 30s
