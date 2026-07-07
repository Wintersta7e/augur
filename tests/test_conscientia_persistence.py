"""Conscientia persistence — capped JSON lists via PersistenceManager."""

import fakeredis
import pytest

from tabula.persistence import PersistenceManager


def _pm():
    return PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))


def test_verdict_roundtrip_newest_first():
    pm = _pm()
    pm.save_conscientia_verdict({"proposal_id": "a", "recommendation": "needs_human"})
    pm.save_conscientia_verdict({"proposal_id": "b", "recommendation": "reject"})
    got = pm.load_conscientia_verdicts(limit=10)
    assert [v["proposal_id"] for v in got] == ["b", "a"]


def test_verdicts_capped_at_200():
    pm = _pm()
    for i in range(210):
        pm.save_conscientia_verdict({"proposal_id": f"p{i}"})
    assert len(pm.load_conscientia_verdicts(limit=500)) == 200


def test_violation_roundtrip_and_cap():
    pm = _pm()
    for i in range(510):
        pm.save_conscientia_violation({"surface": "advice", "code": f"c{i}"})
    got = pm.load_conscientia_violations(limit=1000)
    assert len(got) == 500 and got[0]["code"] == "c509"


def test_limits_are_keyword_only():
    pm = _pm()
    with pytest.raises(TypeError):
        pm.load_conscientia_verdicts(5)  # type: ignore[misc]
