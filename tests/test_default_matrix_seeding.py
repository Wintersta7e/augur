"""Unit tests for additive matrix seeding."""

from unittest.mock import MagicMock

from nexus.correlator import DEFAULT_ESCALATION_MATRIX, ensure_matrix_seeded


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
    pm = MagicMock()
    pm.load_escalation_matrix.return_value = None
    result = ensure_matrix_seeded(pm)
    pm.save_escalation_matrix.assert_called_once_with(DEFAULT_ESCALATION_MATRIX)
    assert result == DEFAULT_ESCALATION_MATRIX


def test_seeding_preserves_operator_changes_to_existing_rules():
    pm = MagicMock()
    existing = {
        "version": "1.0",
        "rules": {"LOW+LOW": "HIGH"},  # operator override
    }
    pm.load_escalation_matrix.return_value = existing
    result = ensure_matrix_seeded(pm)
    # Operator's HIGH override is preserved
    assert result["rules"]["LOW+LOW"] == "HIGH"
    # Missing defaults get added
    assert result["rules"]["LOW+LOW+LOW"] == "MEDIUM"
    assert result["rules"]["HIGH+HIGH+HIGH"] == "HIGH"
    # save called with the merged matrix
    pm.save_escalation_matrix.assert_called_once()


def test_seeding_preserves_rule_windows():
    pm = MagicMock()
    existing = {
        "version": "1.0",
        "rules": {"LOW+LOW": "MEDIUM"},
        "rule_windows": {"LOW+LOW": 25.0},
    }
    pm.load_escalation_matrix.return_value = existing
    result = ensure_matrix_seeded(pm)
    assert result["rule_windows"] == {"LOW+LOW": 25.0}


def test_seeding_no_save_when_already_complete():
    pm = MagicMock()
    pm.load_escalation_matrix.return_value = DEFAULT_ESCALATION_MATRIX.copy()
    ensure_matrix_seeded(pm)
    # No mutation needed → no save
    pm.save_escalation_matrix.assert_not_called()
