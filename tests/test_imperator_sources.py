import fakeredis
from tabula.persistence import PersistenceManager
from imperator import sources


def _pm():
    return PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))


def test_current_sid_extracts_from_dict(monkeypatch):
    pm = _pm()
    monkeypatch.setattr(
        pm, "load_current_session", lambda: {"session_id": "s7", "x": 1}
    )
    assert sources._current_sid(pm) == "s7"
    monkeypatch.setattr(pm, "load_current_session", lambda: None)
    assert sources._current_sid(pm) is None


def test_resolve_latest_reflection_iterates_to_newest_existing(monkeypatch):
    pm = _pm()
    monkeypatch.setattr(pm, "load_current_session", lambda: {"session_id": "s3"})
    monkeypatch.setattr(
        pm,
        "get_all_feedback",
        lambda limit=20: [
            {"session_id": "s3"},
            {"session_id": "s2"},
            {"session_id": "s1"},
        ],
    )
    reports = {
        "s2": {"timestamp": "2026-06-14T00:00:02+00:00", "analyses": {"precision": {}}},
        "s1": {"timestamp": "2026-06-14T00:00:01+00:00", "analyses": {}},
    }
    monkeypatch.setattr(pm, "load_reflection", lambda sid: reports.get(sid))
    out = sources.resolve_latest_reflection(pm)
    assert out["timestamp"] == "2026-06-14T00:00:02+00:00"


def test_resolve_latest_reflection_none_when_no_reports(monkeypatch):
    pm = _pm()
    monkeypatch.setattr(pm, "load_current_session", lambda: None)
    monkeypatch.setattr(pm, "get_all_feedback", lambda limit=20: [{"session_id": "sX"}])
    monkeypatch.setattr(pm, "load_reflection", lambda sid: None)
    assert sources.resolve_latest_reflection(pm) is None


def test_resolve_latest_decision_schema_mismatch(monkeypatch):
    pm = _pm()
    monkeypatch.setattr(
        pm,
        "load_last_advice",
        lambda: {"timestamp": "2026-06-14T00:00:00+00:00", "decision_id": "d1"},
    )
    monkeypatch.setattr(
        pm,
        "load_silence_records",
        lambda limit=1: [
            {"ts": 9_999_999_999.0, "arm": "habituation", "reason": "muted"}
        ],
    )
    out = sources.resolve_latest_decision(pm)
    assert out["decision"] == "suppressed"
    assert out["arm"] == "habituation"


def test_resolve_reception_matches_decision_id(monkeypatch):
    pm = _pm()
    monkeypatch.setattr(pm, "load_current_session", lambda: {"session_id": "s1"})
    monkeypatch.setattr(
        pm,
        "get_feedback",
        lambda sid: {
            "advice_events": [
                {"decision_id": "old", "explicit_rating": "y"},
                {"decision_id": "d9", "explicit_rating": "n", "behavioral_score": 0.2},
            ]
        },
    )
    out = sources.resolve_reception(pm, {"decision_id": "d9"})
    assert out["explicit_rating"] == "n"
    assert out["behavioral_score"] == 0.2


def test_resolve_reception_none_without_match(monkeypatch):
    pm = _pm()
    monkeypatch.setattr(pm, "load_current_session", lambda: {"session_id": "s1"})
    monkeypatch.setattr(
        pm, "get_feedback", lambda sid: {"advice_events": [{"decision_id": "x"}]}
    )
    assert sources.resolve_reception(pm, {"decision_id": "d9"}) is None
