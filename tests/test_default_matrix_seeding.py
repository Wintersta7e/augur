"""Unit tests for additive matrix seeding (routed through the shared CAS writer)."""

from unittest.mock import MagicMock

import fakeredis

from nexus import matrix_ops
from nexus.correlator import DEFAULT_ESCALATION_MATRIX, ensure_matrix_seeded
from tabula.persistence import PersistenceManager


def _pm():
    return PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))


def test_default_matrix_contains_pairwise_rules():
    rules = DEFAULT_ESCALATION_MATRIX["rules"]
    assert rules["LOW+LOW"] == "MEDIUM"
    assert rules["LOW+HIGH"] == "HIGH"
    assert rules["HIGH+HIGH"] == "HIGH"


def test_default_matrix_contains_3way_rules():
    rules = DEFAULT_ESCALATION_MATRIX["rules"]
    assert rules["LOW+LOW+LOW"] == "MEDIUM"
    assert rules["LOW+LOW+MEDIUM"] == "MEDIUM"
    assert rules["LOW+LOW+HIGH"] == "HIGH"
    assert rules["LOW+MEDIUM+MEDIUM"] == "HIGH"
    assert rules["LOW+MEDIUM+HIGH"] == "HIGH"
    assert rules["LOW+HIGH+HIGH"] == "HIGH"
    assert rules["MEDIUM+MEDIUM+MEDIUM"] == "HIGH"
    assert rules["MEDIUM+MEDIUM+HIGH"] == "HIGH"
    assert rules["MEDIUM+HIGH+HIGH"] == "HIGH"
    assert rules["HIGH+HIGH+HIGH"] == "HIGH"


def test_seeding_writes_default_when_empty():
    pm = _pm()
    result = ensure_matrix_seeded(pm)
    assert result == DEFAULT_ESCALATION_MATRIX
    assert pm.load_escalation_matrix() == DEFAULT_ESCALATION_MATRIX


def test_seeding_preserves_operator_changes_and_version():
    pm = _pm()
    pm.save_escalation_matrix({"version": "1.5", "rules": {"LOW+LOW": "HIGH"}})
    result = ensure_matrix_seeded(pm)
    assert result["rules"]["LOW+LOW"] == "HIGH"  # operator override preserved
    assert result["rules"]["LOW+LOW+LOW"] == "MEDIUM"  # missing defaults added
    assert result["rules"]["HIGH+HIGH+HIGH"] == "HIGH"
    assert result["version"] == "1.5"  # existing version preserved
    assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "HIGH"


def test_seeding_preserves_rule_windows():
    pm = _pm()
    pm.save_escalation_matrix(
        {
            "version": "1.0",
            "rules": {"LOW+LOW": "MEDIUM"},
            "rule_windows": {"LOW+LOW": 25.0},
        }
    )
    result = ensure_matrix_seeded(pm)
    assert result["rule_windows"] == {"LOW+LOW": 25.0}


def test_seeding_no_write_when_already_complete(monkeypatch):
    pm = _pm()
    pm.save_escalation_matrix(DEFAULT_ESCALATION_MATRIX.copy())
    # No defaults missing -> the CAS writer must not be invoked at all.
    spy = MagicMock()
    monkeypatch.setattr(matrix_ops, "apply_matrix_update", spy)
    ensure_matrix_seeded(pm)
    spy.assert_not_called()
