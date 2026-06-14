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


def test_patch_returns_prior_rules_from_committed_snapshot():
    pm = _pm()
    out = matrix_ops.apply_matrix_update(
        pm, rules={"MEDIUM+MEDIUM": "LOW"}, mode="patch"
    )
    # prior comes from the watched snapshot actually used for the committed write,
    # so a reversible caller can anchor a rollback to the real pre-write value.
    assert out["prior_rules"] == {"MEDIUM+MEDIUM": "HIGH"}


def test_patch_returns_prior_rule_windows_from_snapshot():
    pm = _pm()
    out = matrix_ops.apply_matrix_update(
        pm, rule_windows={"LOW+LOW": 45.0}, mode="patch"
    )
    assert out["prior_rule_windows"] == {"LOW+LOW": 30.0}


def test_prior_is_none_for_new_rule_key():
    pm = _pm()
    out = matrix_ops.apply_matrix_update(pm, rules={"HIGH+HIGH": "HIGH"}, mode="patch")
    assert out["prior_rules"] == {"HIGH+HIGH": None}


def test_patch_rejects_oversized_preserved_version():
    pm = _pm()
    # An existing over-cap version (save_escalation_matrix does not validate) must
    # not be silently re-committed by a patch that preserves it.
    pm.save_escalation_matrix(
        {
            "version": "x" * (matrix_ops.MAX_ESCALATION_VERSION_LEN + 1),
            "rules": {"LOW+LOW": "MEDIUM"},
        }
    )
    out = matrix_ops.apply_matrix_update(pm, rules={"LOW+LOW": "HIGH"}, mode="patch")
    assert "error" in out and "version" in out["error"]
    assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "MEDIUM"  # no clobber


def test_watcherror_retries_then_succeeds():
    import redis as _redis

    pm = _pm()

    class _FlakyPipe:
        def __init__(self, real, fail):
            self._real, self._fail = real, fail

        def __enter__(self):
            self._real.__enter__()
            return self

        def __exit__(self, *a):
            return self._real.__exit__(*a)

        def execute(self):
            if self._fail[0] > 0:
                self._fail[0] -= 1
                raise _redis.WatchError("simulated contention")
            return self._real.execute()

        def __getattr__(self, name):
            return getattr(self._real, name)

    class _FlakyClient:
        def __init__(self, real, fail):
            self._real, self._fail = real, fail

        def pipeline(self):
            return _FlakyPipe(self._real.pipeline(), self._fail)

        def __getattr__(self, name):
            return getattr(self._real, name)

    pm._r = _FlakyClient(pm._r, [1])  # fail the first execute, then succeed on retry
    out = matrix_ops.apply_matrix_update(pm, rules={"LOW+LOW": "HIGH"}, mode="patch")
    assert out["status"] == "saved"
    assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "HIGH"
