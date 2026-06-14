"""Tests for nexus.matrix_ops — apply_matrix_update patch/replace + CAS."""

import fakeredis
from tabula.persistence import PersistenceManager
from nexus import matrix_ops


def _pm():
    pm = PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))
    pm.save_escalation_matrix(
        {
            "version": "v7",
            "rules": {"LOW+LOW": "MEDIUM", "MEDIUM+MEDIUM": "HIGH"},
            "rule_windows": {"LOW+LOW": 30.0},
        }
    )
    return pm


def test_patch_merges_without_dropping_others_and_keeps_version():
    pm = _pm()
    out = matrix_ops.apply_matrix_update(
        pm, rules={"MEDIUM+MEDIUM": "LOW"}, mode="patch"
    )
    assert out["status"] == "saved"
    m = pm.load_escalation_matrix()
    assert m["rules"] == {"LOW+LOW": "MEDIUM", "MEDIUM+MEDIUM": "LOW"}
    assert m["rule_windows"] == {"LOW+LOW": 30.0}
    assert m["version"] == "v7"


def test_replace_overwrites_rules():
    pm = _pm()
    matrix_ops.apply_matrix_update(
        pm, rules={"HIGH+HIGH": "HIGH"}, version="v8", mode="replace"
    )
    assert pm.load_escalation_matrix()["rules"] == {"HIGH+HIGH": "HIGH"}


def test_replace_empty_windows_clears():
    pm = _pm()
    matrix_ops.apply_matrix_update(
        pm, rules={"LOW+LOW": "MEDIUM"}, rule_windows={}, version="v8", mode="replace"
    )
    m = pm.load_escalation_matrix()
    assert m.get("rule_windows", {}) == {}


def test_patch_rejects_invalid_target():
    pm = _pm()
    out = matrix_ops.apply_matrix_update(pm, rules={"LOW+LOW": "BOGUS"}, mode="patch")
    assert (
        "error" in out and pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "MEDIUM"
    )


def test_patch_validates_merged_cap(monkeypatch):
    pm = _pm()
    monkeypatch.setattr(matrix_ops, "MAX_ESCALATION_RULES", 2)
    out = matrix_ops.apply_matrix_update(pm, rules={"HIGH+HIGH": "HIGH"}, mode="patch")
    assert "error" in out and "HIGH+HIGH" not in pm.load_escalation_matrix()["rules"]
