"""Unit tests for N-way rule_key normalization and escalation lookup."""

from nexus.correlator import (
    DEFAULT_ESCALATION_MATRIX,
    lookup_escalation_n_way,
    normalize_rule_key_n_way,
)


# normalize_rule_key_n_way ---------------------------------------------------


def test_normalize_pairwise():
    assert normalize_rule_key_n_way(["LOW", "MEDIUM"]) == "LOW+MEDIUM"


def test_normalize_pairwise_lowercase_inputs():
    assert normalize_rule_key_n_way(["medium", "low"]) == "LOW+MEDIUM"


def test_normalize_three_way():
    assert normalize_rule_key_n_way(["HIGH", "LOW", "MEDIUM"]) == "LOW+MEDIUM+HIGH"


def test_normalize_three_way_all_low():
    assert normalize_rule_key_n_way(["LOW", "LOW", "LOW"]) == "LOW+LOW+LOW"


def test_normalize_unknown_severity_returns_none():
    assert normalize_rule_key_n_way(["LOW", "FAKE"]) is None


def test_normalize_empty_returns_none():
    assert normalize_rule_key_n_way([]) is None


# lookup_escalation_n_way ---------------------------------------------------


def test_lookup_pairwise_hit():
    sev, rule = lookup_escalation_n_way(["LOW", "LOW"], DEFAULT_ESCALATION_MATRIX)
    assert sev == "MEDIUM"
    assert rule == "LOW+LOW→MEDIUM"


def test_lookup_3way_hit_with_local_matrix():
    """Use a local matrix — DEFAULT_ESCALATION_MATRIX doesn't yet ship 3-way
    rules at this point in the build sequence (Task 4 adds them). The test
    proves the lookup logic; the default-matrix coverage is in Task 4."""
    matrix = {"version": "1.0", "rules": {"LOW+LOW+LOW": "MEDIUM"}}
    sev, rule = lookup_escalation_n_way(["LOW", "LOW", "LOW"], matrix)
    assert sev == "MEDIUM"
    assert rule == "LOW+LOW+LOW→MEDIUM"


def test_lookup_4way_falls_back_to_max():
    """No 4-way rule anywhere → falls back to max severity."""
    matrix = {"version": "1.0", "rules": {}}  # explicitly empty
    sev, rule = lookup_escalation_n_way(["LOW", "LOW", "LOW", "MEDIUM"], matrix)
    assert sev == "MEDIUM"
    assert rule is None  # no 4-way default rule


def test_lookup_single_severity_returns_unchanged():
    sev, rule = lookup_escalation_n_way(["HIGH"], DEFAULT_ESCALATION_MATRIX)
    assert sev == "HIGH"
    assert rule is None


def test_lookup_empty_returns_low_no_rule():
    sev, rule = lookup_escalation_n_way([], DEFAULT_ESCALATION_MATRIX)
    assert sev == "LOW"
    assert rule is None
