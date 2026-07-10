"""Vox renders augur.conscientia.violation (charter blocks, Task 10's record shape)."""

from vox import console_display as V


def _violation_payload() -> dict:
    """A representative augur.conscientia.violation payload (make_violation shape)."""
    return {
        "surface": "advice",
        "code": "forbidden_valence",
        "detail": "matched 'take a break'",
        "principle": "restraint",
        "decision_id": "d1",
        "state_key": "single:typing:user",
        "domain": "typing",
        "entity": "user",
        "session_id": "sess1",
        "regenerated": False,
        "ts": 1234.5,
        "charter_version": 1,
    }


def test_subject_constant():
    assert V.SUBJECT_CONSCIENTIA_VIOLATION == "augur.conscientia.violation"


def test_render_conscientia_violation_shows_principle_and_detail():
    out = V.render_conscientia_violation(_violation_payload())
    assert "CONSCIENTIA" in out
    assert "restraint" in out
    assert "matched 'take a break'" in out
    assert "advice" in out


def test_render_conscientia_violation_handles_missing_fields():
    # Defensive: a malformed/partial payload must not crash the display.
    out = V.render_conscientia_violation({})
    assert "CONSCIENTIA" in out


def test_violation_dedup_suppresses_identical_repeat():
    last_violations: dict = {}
    payload = _violation_payload()

    # Before rendering, an identical violation is not yet deduped.
    assert V.dedup_should_suppress_violation(last_violations, payload) is False

    # After rendering, update_last_violations records (surface, domain) -> detail
    # so an identical repeat is suppressed.
    V.update_last_violations(last_violations, payload)
    assert V.dedup_should_suppress_violation(last_violations, payload) is True


def test_violation_dedup_does_not_suppress_different_detail():
    last_violations: dict = {}
    payload = _violation_payload()
    V.update_last_violations(last_violations, payload)

    other = dict(payload, detail="matched 'you seem distracted'")
    assert V.dedup_should_suppress_violation(last_violations, other) is False


def test_violation_dedup_does_not_suppress_different_entity():
    # Two entities tripping the same charter pattern are distinct alerts —
    # entity is part of the dedup key, not collapsed into it.
    last_violations: dict = {}
    payload = _violation_payload()
    V.update_last_violations(last_violations, payload)

    other = dict(payload, entity="user2")
    assert V.dedup_should_suppress_violation(last_violations, other) is False
