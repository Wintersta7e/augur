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


class _Cfg:
    imperator_rate_window_s = 900.0
    imperator_baseline_trained_obs = 15
    gate_max_consecutive_suppressions = 8
    correlation_window_lag_multiplier = 3.0
    correlation_window_min_s = 5.0
    correlation_window_max_s = 120.0
    correlation_window_tuning_hysteresis_pct = 0.2
    correlation_window_s = 30.0


def test_windowed_rates_excludes_probe(monkeypatch):
    pm = _pm()
    now = 1000.0
    monkeypatch.setattr(
        pm,
        "load_silence_records",
        lambda limit: [{"ts": now - 10}, {"ts": now - 5}, {"ts": now - 5000}],
    )
    monkeypatch.setattr(
        pm,
        "load_emissions",
        lambda limit: [
            {"ts": now - 8, "probe": False},
            {"ts": now - 7, "probe": True},
            {"ts": now - 6, "audit_only": True},
            {"ts": now - 9000, "probe": False},
        ],
    )
    out = sources.windowed_rates(pm, now=now, window_s=300.0)
    assert out["advice_volume"]["delivered"] == 1
    assert out["advice_volume"]["suppressed"] == 2
    assert round(out["suppression_rate"], 3) == round(2 / 3, 3)


def test_build_blind_spots_all_five_kinds(monkeypatch):
    pm = _pm()
    monkeypatch.setattr(
        pm, "load_rule_confidence", lambda: {"R_LOW": {"confidence": 0.3}}
    )
    monkeypatch.setattr(
        pm,
        "load_escalation_matrix",
        lambda: {"rules": {"R_LOW": {}, "R_NEW": {}}, "rule_windows": {"R_W": 30.0}},
    )
    monkeypatch.setattr(
        pm, "load_rule_window_state", lambda: {"R_W": {"ewma_lag": 40.0}}
    )
    monkeypatch.setattr(pm, "load_self_tolerance", lambda: ["typing:alice"])
    monkeypatch.setattr(
        pm,
        "load_all_channel_stats",
        lambda: {"activity:ide": {"consecutive_suppressions": 7}},
    )
    spots = sources._build_blind_spots(pm, {"untrained": 2, "by_domain": {}}, _Cfg())
    kinds = {s["kind"] for s in spots}
    assert kinds == {
        "low_confidence_rule",
        "never_evaluated_rule",
        "mis_sized_window",
        "muted_channel",
        "starving_channel",
        "undertrained_baselines",
    }


def test_gather_uses_history_and_no_data_utility(monkeypatch):
    pm = _pm()
    monkeypatch.setattr(pm, "load_current_session", lambda: {"session_id": "s1"})
    monkeypatch.setattr(pm, "load_health_snapshot", lambda: {"faculties": {}})
    monkeypatch.setattr(pm, "load_last_advice", lambda: None)
    monkeypatch.setattr(
        sources,
        "resolve_latest_reflection",
        lambda _pm: {
            "analyses": {"utility": {"reason": "No advice events to evaluate"}},
            "adjustments": {"sigma_adjusted": True, "matrix_mutated": False},
        },
    )
    monkeypatch.setattr(sources, "resolve_reception", lambda _pm, a: None)
    monkeypatch.setattr(
        sources,
        "windowed_rates",
        lambda _pm, now, window_s: {
            "suppression_rate": 0.0,
            "advice_volume": {"delivered": 0, "suppressed": 0},
        },
    )
    monkeypatch.setattr(
        pm,
        "scan_baseline_maturity",
        lambda trained_obs: {"total": 0, "trained": 0, "untrained": 0, "by_domain": {}},
    )
    monkeypatch.setattr(sources, "_build_blind_spots", lambda _pm, b, cfg: [])
    monkeypatch.setattr(pm, "load_advice_rate", lambda: {"rate_ewma": 0.4})
    monkeypatch.setattr(
        pm,
        "get_history",
        lambda domain, limit=1: [
            {
                "value": 88.0,
                "entity": "ide",
                "context": {"focused_app": "ide", "new_app": "ide"},
                "timestamp": "2026-06-14T00:00:00+00:00",
            }
        ],
    )
    out = sources.gather(pm, {"escalation_tier": "MEDIUM"}, now=10.0, cfg=_Cfg())
    assert out["session_id"] == "s1"
    assert out["activity"] == "ide"
    assert out["intensity_ewma"] == 88.0
    assert out["utility_no_data"] is True
    assert out["dismissal_rate"] == 0.4
    assert out["recent_self_tuning"] == {
        "sigma_adjusted": True,
        "matrix_mutated": False,
    }
