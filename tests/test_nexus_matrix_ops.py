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


def test_validate_patch_accepts_valid_isolated_patches():
    from tabula.config import AugurConfig

    assert matrix_ops.validate_patch(rules={"LOW+HIGH": "HIGH"}) is None
    assert (
        matrix_ops.validate_patch(rule_windows={"LOW+LOW": 60.0}, config=AugurConfig())
        is None
    )


def test_validate_patch_rejects_without_touching_redis():
    # The pre-arm validator mirrors apply_matrix_update's checks and never
    # needs a client — malformed patches are refused before any side effect.
    from tabula.config import AugurConfig

    assert matrix_ops.validate_patch(rules={"LOW+LOW": "BANANA"}) is not None
    assert matrix_ops.validate_patch(rules={"typing+chess": "HIGH"}) is not None
    cfg = AugurConfig()
    assert (
        matrix_ops.validate_patch(rule_windows={"LOW+LOW": 9999.0}, config=cfg)
        is not None
    )


def test_rule_windows_keys_must_be_severity_pairs():
    # Symmetry with rules keys: window keys whose parts are not severities
    # are rejected (windows for keys that can never be rules made no sense).
    from tabula.config import AugurConfig

    cfg = AugurConfig()
    err = matrix_ops._validate_escalation_matrix_rule_windows(
        {"typing+activity": 30.0}, cfg
    )
    assert err is not None and "severity" in err
    assert (
        matrix_ops._validate_escalation_matrix_rule_windows({"LOW+HIGH": 30.0}, cfg)
        is None
    )


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


class _FlakyPipe:
    """Wraps a real pipeline, raising WatchError on the first `fail[0]` executes."""

    def __init__(self, real, fail):
        self._real, self._fail = real, fail

    def __enter__(self):
        self._real.__enter__()
        return self

    def __exit__(self, *a):
        return self._real.__exit__(*a)

    def execute(self):
        import redis as _redis

        if self._fail[0] > 0:
            self._fail[0] -= 1
            raise _redis.WatchError("simulated contention")
        return self._real.execute()

    def __getattr__(self, name):
        return getattr(self._real, name)


class _FlakyClient:
    """Hands out _FlakyPipe instances sharing a mutable fail counter."""

    def __init__(self, real, fail):
        self._real, self._fail = real, fail

    def pipeline(self):
        return _FlakyPipe(self._real.pipeline(), self._fail)

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_watcherror_retries_then_succeeds():
    pm = _pm()
    pm._r = _FlakyClient(pm._r, [1])  # fail the first execute, then succeed on retry
    out = matrix_ops.apply_matrix_update(pm, rules={"LOW+LOW": "HIGH"}, mode="patch")
    assert out["status"] == "saved"
    assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "HIGH"


def test_retries_zero_still_attempts_one_write():
    # `_retries` bounds RE-tries, not the initial attempt: with _retries=0 the
    # writer must still make exactly one (un-retried) write rather than reporting
    # a false 'contention' error without ever touching Redis.
    pm = _pm()
    out = matrix_ops.apply_matrix_update(
        pm, rules={"LOW+LOW": "HIGH"}, mode="patch", _retries=0
    )
    assert out["status"] == "saved"
    assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "HIGH"


def test_retries_zero_does_not_retry_on_contention():
    # _retries=0 means the single attempt is NOT retried: one WatchError is fatal
    # and surfaces as the contention error (fail-open guarantee is the caller's).
    pm = _pm()
    pm._r = _FlakyClient(pm._r, [1])  # fail the only execute
    out = matrix_ops.apply_matrix_update(
        pm, rules={"LOW+LOW": "HIGH"}, mode="patch", _retries=0
    )
    assert "error" in out and "contention" in out["error"]
    assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "MEDIUM"  # unchanged


def test_retry_exhaustion_returns_contention_error():
    # Persistent WATCH/CAS contention that outlasts every retry must surface the
    # contention error and leave the matrix untouched. With the default of 5
    # retries there are 6 total attempts; fail more than that to exhaust them.
    pm = _pm()
    pm._r = _FlakyClient(pm._r, [99])  # always raise WatchError
    out = matrix_ops.apply_matrix_update(pm, rules={"LOW+LOW": "HIGH"}, mode="patch")
    assert "error" in out and "contention" in out["error"]
    assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "MEDIUM"  # unchanged


def test_corrupt_matrix_in_redis_fails_safe():
    # A corrupt (non-JSON) matrix blob must not crash the writer: the json.loads
    # failure is caught by the broad fail-safe handler and returned as an error,
    # leaving the corrupt value in place rather than committing on top of garbage.
    pm = _pm()
    pm._r.set(matrix_ops._MATRIX_KEY, b"{not valid json")
    out = matrix_ops.apply_matrix_update(pm, rules={"LOW+LOW": "HIGH"}, mode="patch")
    assert "error" in out
    assert pm._r.get(matrix_ops._MATRIX_KEY) == b"{not valid json"  # not clobbered


def test_missing_matrix_replace_creates_from_empty():
    # No matrix key at all (fresh Redis): a replace must succeed by treating the
    # absent value as an empty matrix rather than failing or erroring.
    pm = PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))
    assert pm._r.get(matrix_ops._MATRIX_KEY) is None
    out = matrix_ops.apply_matrix_update(
        pm, rules={"LOW+LOW": "MEDIUM"}, version="v1", mode="replace"
    )
    assert out["status"] == "saved"
    assert pm.load_escalation_matrix()["rules"] == {"LOW+LOW": "MEDIUM"}
