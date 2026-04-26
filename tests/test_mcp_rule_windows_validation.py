"""Tests for set_escalation_matrix validation of rule_windows."""

from unittest.mock import patch


def _call_validate(rules, rule_windows=None, version="1.0"):
    """Helper that calls the MCP validation directly via the module-level helpers."""
    from augur_mcp.augur_server import (
        _validate_escalation_rules,
        _validate_escalation_matrix_rule_windows,
    )

    err = _validate_escalation_rules(rules)
    if err:
        return err
    if rule_windows is not None:
        from blackboard.config import AugurConfig

        return _validate_escalation_matrix_rule_windows(
            rule_windows, AugurConfig.from_env()
        )
    return None


def test_max_escalation_rules_bumped_to_40():
    from augur_mcp.augur_server import MAX_ESCALATION_RULES

    assert MAX_ESCALATION_RULES == 40


def test_rules_count_under_new_cap_accepted():
    from reasoning.correlator import DEFAULT_ESCALATION_MATRIX

    err = _call_validate(DEFAULT_ESCALATION_MATRIX["rules"])
    assert err is None  # 16 rules <= 40 cap


def test_rule_windows_pairwise_only_accepted():
    err = _call_validate(
        {"LOW+LOW": "MEDIUM"},
        rule_windows={"LOW+LOW": 25.0},
    )
    assert err is None


def test_rule_windows_3way_key_rejected():
    err = _call_validate(
        {"LOW+LOW": "MEDIUM"},
        rule_windows={"LOW+LOW+LOW": 45.0},
    )
    assert err is not None
    assert "pairwise" in err.lower()


def test_rule_windows_value_below_min_rejected():
    err = _call_validate(
        {"LOW+LOW": "MEDIUM"},
        rule_windows={"LOW+LOW": 1.0},  # below min 5.0
    )
    assert err is not None


def test_rule_windows_value_above_max_rejected():
    err = _call_validate(
        {"LOW+LOW": "MEDIUM"},
        rule_windows={"LOW+LOW": 200.0},  # above max 120.0
    )
    assert err is not None


def test_rule_windows_non_numeric_rejected():
    err = _call_validate(
        {"LOW+LOW": "MEDIUM"},
        rule_windows={"LOW+LOW": "thirty"},
    )
    assert err is not None


def test_rule_windows_too_many_rejected():
    err = _call_validate(
        {"LOW+LOW": "MEDIUM"},
        rule_windows={f"K{i}+K{i}": 25.0 for i in range(50)},
    )
    assert err is not None


def test_rule_windows_none_accepted():
    err = _call_validate({"LOW+LOW": "MEDIUM"}, rule_windows=None)
    assert err is None


def test_rule_windows_empty_dict_accepted():
    err = _call_validate({"LOW+LOW": "MEDIUM"}, rule_windows={})
    assert err is None


# Preservation: omitting rule_windows must NOT erase existing ones --------


def test_set_escalation_matrix_preserves_existing_rule_windows():
    """Caller omitting rule_windows in set_escalation_matrix must not erase
    a previously-tuned rule_windows entry on the live matrix."""
    from unittest.mock import MagicMock

    from augur_mcp.augur_server import set_escalation_matrix

    fake_pm = MagicMock()
    fake_pm.load_escalation_matrix.return_value = {
        "version": "1.0",
        "rules": {"LOW+LOW": "MEDIUM"},
        "rule_windows": {"LOW+LOW": 25.0},
    }

    with patch("augur_mcp.augur_server._persistence_ctx") as ctx:
        ctx.return_value.__enter__.return_value = fake_pm
        ctx.return_value.__exit__.return_value = None
        result = set_escalation_matrix(
            rules={"LOW+LOW": "HIGH"},  # rule changed
            version="1.0",
            # rule_windows omitted intentionally
        )

    assert result.get("status") == "saved"
    saved = fake_pm.save_escalation_matrix.call_args.args[0]
    assert saved["rules"] == {"LOW+LOW": "HIGH"}
    # rule_windows preserved from existing matrix
    assert saved["rule_windows"] == {"LOW+LOW": 25.0}


def test_set_escalation_matrix_explicit_empty_rule_windows_clears():
    """Caller explicitly passing rule_windows={} should clear them."""
    from unittest.mock import MagicMock

    from augur_mcp.augur_server import set_escalation_matrix

    fake_pm = MagicMock()
    fake_pm.load_escalation_matrix.return_value = {
        "version": "1.0",
        "rules": {"LOW+LOW": "MEDIUM"},
        "rule_windows": {"LOW+LOW": 25.0},
    }

    with patch("augur_mcp.augur_server._persistence_ctx") as ctx:
        ctx.return_value.__enter__.return_value = fake_pm
        ctx.return_value.__exit__.return_value = None
        result = set_escalation_matrix(
            rules={"LOW+LOW": "MEDIUM"},
            version="1.0",
            rule_windows={},  # explicit empty
        )

    assert result.get("status") == "saved"
    saved = fake_pm.save_escalation_matrix.call_args.args[0]
    assert saved["rule_windows"] == {}
