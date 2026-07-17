"""Praesagium matcher (spec 2026-07-09 §5, §5.1, §5.2, §5.3, §6.1, §6.2).

Task 6 covers the prediction lifecycle: arming (match_patterns), resolution
(resolve_open_predictions), the deterministic forewarning template
(render_forewarning/_humanize), and the foreseen payload
(build_foreseen_payload). Async paths run through asyncio.run with a
fakeredis-backed PersistenceManager and a recording (raise-capable) publish
stub.

Task 7 (below) covers the runner surface built on top of that core:
make_on_anomaly (the augur.vigil.anomaly callback), on_session_end (the
augur.session.end bookkeeping stub), warm_cooldowns (restart warm-start), and
the PR2 structural pins (no task fan-out, no httpx/ollama import). run()/main()
are exercised only indirectly here (they need real Redis/NATS); the callback
factories they wire together are fully testable without NATS.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from datetime import datetime, timezone

import fakeredis
import pytest

import praesagium.matcher as matcher_module
from conscientia.screens import screen_advice_text
from limen.gate import build_signature
from praesagium.matcher import (
    SUBJECT_FORESEEN,
    SUBJECT_RESOLVED,
    _humanize,
    build_foreseen_payload,
    make_on_anomaly,
    match_patterns,
    on_session_end,
    render_forewarning,
    resolve_open_predictions,
    warm_cooldowns,
)
from tabula.config import AugurConfig
from tabula.persistence import PersistenceManager


# -- fixtures / helpers -------------------------------------------------------


def _pm() -> PersistenceManager:
    return PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))


def _cfg(**overrides) -> AugurConfig:
    return AugurConfig(**overrides)


def _pattern(
    pid: str = "abc123def456",
    antecedent: str = "typing:user",
    consequent: str = "activity:editor",
    *,
    status: str = "active",
    window_s: float = 120.0,
    support_sessions: int = 4,
    conf_lower: float = 0.62,
    lift: float = 2.13,
) -> dict:
    return {
        "pattern_id": pid,
        "antecedent": antecedent,
        "consequent": consequent,
        "window_s": window_s,
        "support_sessions": support_sessions,
        "n": 5,
        "k": 4,
        "conf": 0.8,
        "conf_lower": conf_lower,
        "lift": lift,
        "lag_median_s": 40.0,
        "lag_p90_s": 90.0,
        "status": status,
        "hit_rate": None,
        "resolutions": 0,
        "created_at": 1.0,
        "mined_at": 2.0,
        "retired_at": None,
        "retired_reason": None,
        "repass_streak": 1,
    }


def _seed_patterns(pm: PersistenceManager, *patterns: dict) -> None:
    blob = {
        "version": 1,
        "mined_at": 100.0,
        "hit_rate_watermark": 0.0,
        "patterns": {p["pattern_id"]: p for p in patterns},
    }
    pm.save_praesagium_patterns(blob)


class RecordingPublish:
    """Async publish stub: records (subject, data) and can raise per-subject."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[tuple[str, bytes]] = []
        self.fail_on = fail_on

    async def __call__(self, subject: str, data: bytes) -> None:
        self.calls.append((subject, data))
        if self.fail_on in (subject, "all"):
            raise RuntimeError("nats down")

    def subjects(self) -> list[str]:
        return [s for s, _ in self.calls]

    def payloads(self, subject: str) -> list[dict]:
        return [json.loads(d) for s, d in self.calls if s == subject]


# -- _humanize ----------------------------------------------------------------


def test_humanize_domain_entity():
    assert _humanize("typing:user") == "typing (user)"


def test_humanize_no_colon_passthrough():
    assert _humanize("standalone") == "standalone"


def test_humanize_splits_on_first_colon_only():
    assert _humanize("activity:app:vscode") == "activity (app:vscode)"


# -- render_forewarning -------------------------------------------------------


def test_render_forewarning_exact_string():
    p = _pattern(
        antecedent="typing:user",
        consequent="activity:editor",
        window_s=120.0,
        support_sessions=4,
        conf_lower=0.62,
    )
    assert render_forewarning(p) == (
        "Forewarning: in 4 recent sessions, typing (user) was followed by "
        "activity (editor) within ~120s (confidence ≥ 62%). "
        "typing (user) was just observed."
    )


def test_render_forewarning_int_casts_truncate():
    p = _pattern(window_s=90.7, conf_lower=0.449)
    text = render_forewarning(p)
    assert "~90s" in text  # int() truncates 90.7
    assert "≥ 44%" in text  # int(0.449 * 100) == 44


@pytest.mark.parametrize(
    "antecedent,consequent",
    [
        ("typing:user", "activity:editor"),
        ("activity:browser", "typing:user"),
        ("chess:white", "chess:black"),
        ("activity:terminal", "activity:meeting"),
        ("typing:host", "activity:idle"),
    ],
)
def test_template_never_trips_charter(antecedent, consequent):
    # Template-vs-charter pin: the deterministic forewarning must survive
    # Conscientia's default output screen for a matrix of realistic patterns.
    p = _pattern(antecedent=antecedent, consequent=consequent)
    verdict = screen_advice_text(render_forewarning(p), AugurConfig())
    assert verdict.ok is True


# -- build_foreseen_payload + PR1a signature ---------------------------------


def test_build_foreseen_payload_all_fields():
    pat = _pattern(pid="p1", conf_lower=0.6234, lift=2.137, support_sessions=4)
    pred = {"prediction_id": "pred-1", "forewarning_text": "FW TEXT"}
    payload = build_foreseen_payload(pat, pred, "sess-9")

    prim = payload["primary_anomaly"]
    assert prim["domain"] == "praesagium"
    assert prim["entity"] == "p1"
    assert prim["severity"] == "medium"
    assert prim["event_type"] == "forewarning"
    assert prim["value"] == round(0.6234, 3)
    assert prim["unit"] == "confidence"
    assert prim["session_id"] == "sess-9"
    assert isinstance(prim["timestamp"], str)
    assert prim["baseline_mean"] is None
    assert prim["baseline_std"] is None
    assert prim["deviation_score"] is None
    assert prim["baseline_observation_count"] is None

    ctx = prim["context"]
    assert ctx["antecedent"] == "typing:user"
    assert ctx["consequent"] == "activity:editor"
    assert ctx["window_s"] == 120.0
    assert ctx["support_sessions"] == 4
    assert ctx["lift"] == round(2.137, 2)
    assert ctx["label"] == "typing:user → activity:editor"

    assert payload["correlated_events"] == []
    assert payload["correlation_found"] is False
    assert payload["combined_severity"] == "MEDIUM"
    assert payload["temporal_lag_seconds"] is None
    assert payload["correlation_span_s"] is None
    assert payload["severity_escalated"] is False
    assert payload["escalation_rule"] is None
    assert payload["escalation_matrix_version"] is None
    assert payload["rule_key"] is None
    assert payload["rule_window_s"] is None
    assert payload["involved_domains"] == ["praesagium"]
    assert isinstance(payload["timestamp"], str)
    assert payload["session_id"] == "sess-9"  # top-level, not only nested
    assert payload["source"] == "anticipatory"

    ant = payload["anticipatory"]
    assert ant["pattern_id"] == "p1"
    assert ant["prediction_id"] == "pred-1"
    assert ant["antecedent"] == "typing:user"
    assert ant["consequent"] == "activity:editor"
    assert ant["window_s"] == 120.0
    assert ant["conf_lower"] == 0.6234
    assert ant["support_sessions"] == 4
    assert ant["forewarning_text"] == "FW TEXT"


def test_foreseen_payload_signature_pr1a():
    pat = _pattern(pid="pat-xyz")
    pred = {"prediction_id": "pred-1", "forewarning_text": "t"}
    payload = build_foreseen_payload(pat, pred, "s")
    sig = build_signature(payload)
    assert sig.exempt is False
    assert sig.path == "single"
    assert sig.state_key == "single:praesagium:pat-xyz"
    assert sig.ungateable is False


def test_foreseen_payload_json_serializable():
    payload = build_foreseen_payload(
        _pattern(), {"prediction_id": "x", "forewarning_text": "t"}, "s"
    )
    # Must round-trip cleanly (it is published as json.dumps(...).encode()).
    assert json.loads(json.dumps(payload))["source"] == "anticipatory"


# -- match_patterns: arming ---------------------------------------------------


def _open_by_pattern(pm: PersistenceManager) -> dict[str, dict]:
    return {r["pattern_id"]: r for r in pm.load_praesagium_open_predictions()}


def test_match_arms_active_pattern_emit_off():
    pm = _pm()
    _seed_patterns(pm, _pattern(pid="p1", antecedent="typing:user"))
    pub = RecordingPublish()
    cooldowns: dict[str, float] = {}
    cfg = _cfg(praesagium_emit_enabled=False)

    asyncio.run(
        match_patterns(
            pm,
            pub,
            "typing:user",
            ts=1000.0,
            payload={"session_id": "s1"},
            cfg=cfg,
            cooldowns=cooldowns,
        )
    )

    opens = _open_by_pattern(pm)
    assert "p1" in opens
    rec = opens["p1"]
    assert rec["emit_attempted"] is False
    assert rec["emitted_at"] is None
    assert rec["antecedent"] == "typing:user"
    assert rec["consequent"] == "activity:editor"
    assert rec["window_s"] == 120.0
    assert rec["created_ts"] == 1000.0
    assert rec["deadline_ts"] == 1000.0 + 120.0
    assert rec["session_id"] == "s1"
    assert rec["conf_lower"] == 0.62
    assert isinstance(rec["prediction_id"], str) and rec["prediction_id"]
    assert rec["forewarning_text"] == render_forewarning(_pattern(pid="p1"))
    assert cooldowns["p1"] == 1000.0
    assert pub.calls == []  # emit off -> no publish


def test_match_skips_non_active_and_wrong_antecedent():
    pm = _pm()
    _seed_patterns(
        pm,
        _pattern(pid="prov", antecedent="typing:user", status="provisional"),
        _pattern(pid="ret", antecedent="typing:user", status="retired"),
        _pattern(pid="other", antecedent="activity:browser", status="active"),
        _pattern(pid="hit", antecedent="typing:user", status="active"),
    )
    pub = RecordingPublish()
    cooldowns: dict[str, float] = {}
    asyncio.run(
        match_patterns(
            pm,
            pub,
            "typing:user",
            ts=10.0,
            payload={"session_id": "s"},
            cfg=_cfg(),
            cooldowns=cooldowns,
        )
    )
    opens = _open_by_pattern(pm)
    assert set(opens) == {"hit"}  # only the active + antecedent match


def test_match_emit_on_publishes_and_stamps_emitted_at():
    pm = _pm()
    _seed_patterns(pm, _pattern(pid="p1", antecedent="typing:user"))
    pub = RecordingPublish()
    cfg = _cfg(praesagium_emit_enabled=True)
    asyncio.run(
        match_patterns(
            pm,
            pub,
            "typing:user",
            ts=2000.0,
            payload={"session_id": "s1"},
            cfg=cfg,
            cooldowns={},
        )
    )
    assert pub.subjects() == [SUBJECT_FORESEEN]
    foreseen = pub.payloads(SUBJECT_FORESEEN)[0]
    assert foreseen["source"] == "anticipatory"
    assert foreseen["primary_anomaly"]["entity"] == "p1"

    rec = _open_by_pattern(pm)["p1"]
    assert rec["emit_attempted"] is True
    assert rec["emitted_at"] == 2000.0  # stamped on publish success


def test_match_emit_on_publish_failure_keeps_record_emitted_at_none():
    pm = _pm()
    _seed_patterns(pm, _pattern(pid="p1", antecedent="typing:user"))
    pub = RecordingPublish(fail_on=SUBJECT_FORESEEN)
    cfg = _cfg(praesagium_emit_enabled=True)
    # Publish raising must be swallowed (no exception escapes match_patterns).
    asyncio.run(
        match_patterns(
            pm,
            pub,
            "typing:user",
            ts=3000.0,
            payload={"session_id": "s1"},
            cfg=cfg,
            cooldowns={},
        )
    )
    rec = _open_by_pattern(pm)["p1"]
    assert rec["emit_attempted"] is True
    assert rec["emitted_at"] is None  # never stamped when publish raised
    assert pub.subjects() == [SUBJECT_FORESEEN]  # attempt was made


def test_match_cooldown_skips_recent_arm():
    pm = _pm()
    _seed_patterns(pm, _pattern(pid="p1", antecedent="typing:user"))
    cfg = _cfg()  # cooldown default 600s
    cooldowns = {"p1": 1000.0}
    pub = RecordingPublish()
    asyncio.run(
        match_patterns(
            pm,
            pub,
            "typing:user",
            ts=1000.0 + 100.0,  # 100s < 600s cooldown
            payload={"session_id": "s"},
            cfg=cfg,
            cooldowns=cooldowns,
        )
    )
    assert pm.load_praesagium_open_predictions() == []
    assert cooldowns["p1"] == 1000.0  # untouched


def test_match_cooldown_expired_arms_again():
    pm = _pm()
    _seed_patterns(pm, _pattern(pid="p1", antecedent="typing:user"))
    cfg = _cfg()
    cooldowns = {"p1": 1000.0}
    asyncio.run(
        match_patterns(
            pm,
            RecordingPublish(),
            "typing:user",
            ts=1000.0 + 700.0,  # 700s > 600s cooldown
            payload={"session_id": "s"},
            cfg=cfg,
            cooldowns=cooldowns,
        )
    )
    assert len(pm.load_praesagium_open_predictions()) == 1
    assert cooldowns["p1"] == 1700.0


def test_match_no_open_duplicate_skips():
    pm = _pm()
    _seed_patterns(pm, _pattern(pid="p1", antecedent="typing:user"))
    # Pre-existing open prediction for p1 -> second arm must be skipped.
    pm.save_praesagium_open_prediction(
        {"prediction_id": "existing", "pattern_id": "p1", "deadline_ts": 9e9}
    )
    asyncio.run(
        match_patterns(
            pm,
            RecordingPublish(),
            "typing:user",
            ts=5000.0,
            payload={"session_id": "s"},
            cfg=_cfg(),
            cooldowns={},
        )
    )
    opens = pm.load_praesagium_open_predictions()
    assert len(opens) == 1
    assert opens[0]["prediction_id"] == "existing"  # no new arm


def test_match_cap_refusal_skips_with_no_emit():
    pm = _pm()
    _seed_patterns(pm, _pattern(pid="p1", antecedent="typing:user"))
    # Fill the open hash to cap=1 with a DIFFERENT pattern so no-open-duplicate
    # does not fire; the new arm must be cap-refused.
    pm.save_praesagium_open_prediction(
        {"prediction_id": "other-pred", "pattern_id": "other"}, cap=1
    )
    pub = RecordingPublish()
    cooldowns: dict[str, float] = {}
    asyncio.run(
        match_patterns(
            pm,
            pub,
            "typing:user",
            ts=6000.0,
            payload={"session_id": "s"},
            cfg=_cfg(praesagium_open_predictions_cap=1, praesagium_emit_enabled=True),
            cooldowns=cooldowns,
        )
    )
    opens = _open_by_pattern(pm)
    assert "p1" not in opens  # refused
    assert pub.calls == []  # no publish when arm refused
    assert "p1" not in cooldowns  # cooldown only updates on successful arm


def test_match_no_patterns_blob_is_noop():
    pm = _pm()  # nothing seeded
    asyncio.run(
        match_patterns(
            pm,
            RecordingPublish(),
            "typing:user",
            ts=1.0,
            payload={"session_id": "s"},
            cfg=_cfg(),
            cooldowns={},
        )
    )
    assert pm.load_praesagium_open_predictions() == []


# -- stale cooldown pruning on blob reload -----------------------------------


def test_match_prunes_stale_cooldowns_on_blob_reload():
    # A pid retired/evicted from the patterns blob must be dropped from the
    # in-memory cooldowns dict on every reload -- nothing else ever prunes it,
    # so it would otherwise accumulate forever.
    pm = _pm()
    _seed_patterns(pm, _pattern(pid="live", antecedent="typing:user"))
    cooldowns = {"stale-gone": 500.0, "live": 500.0}
    asyncio.run(
        match_patterns(
            pm,
            RecordingPublish(),
            "typing:user",
            ts=1000.0 + 700.0,  # past live's cooldown -> re-arms and refreshes it
            payload={"session_id": "s"},
            cfg=_cfg(),
            cooldowns=cooldowns,
        )
    )
    assert "stale-gone" not in cooldowns
    assert "live" in cooldowns


def test_match_prunes_stale_cooldowns_even_when_no_pattern_matches_key():
    # The prune runs unconditionally on a valid blob, independent of whether
    # any pattern's antecedent matches this call's key.
    pm = _pm()
    _seed_patterns(pm, _pattern(pid="live", antecedent="typing:user"))
    cooldowns = {"stale-gone": 1.0}
    asyncio.run(
        match_patterns(
            pm,
            RecordingPublish(),
            "activity:browser",  # matches no pattern's antecedent
            ts=1000.0,
            payload={"session_id": "s"},
            cfg=_cfg(),
            cooldowns=cooldowns,
        )
    )
    assert cooldowns == {}


# -- resolve_open_predictions -------------------------------------------------


def _arm(pm, pid, antecedent, consequent, *, created_ts, window_s=120.0, sid="s"):
    rec = {
        "prediction_id": f"pred-{pid}",
        "pattern_id": pid,
        "antecedent": antecedent,
        "consequent": consequent,
        "window_s": window_s,
        "created_ts": created_ts,
        "deadline_ts": created_ts + window_s,
        "session_id": sid,
        "conf_lower": 0.6,
        "emit_attempted": False,
        "emitted_at": None,
        "forewarning_text": "t",
    }
    pm.save_praesagium_open_prediction(rec)
    return rec


def test_resolve_fulfilled_within_deadline():
    pm = _pm()
    _arm(pm, "p1", "typing:user", "activity:editor", created_ts=1000.0, window_s=120.0)
    pub = RecordingPublish()
    n = asyncio.run(
        resolve_open_predictions(
            pm, pub, "activity:editor", "medium", 1050.0, cfg=_cfg()
        )
    )
    assert n == 1
    assert pm.load_praesagium_open_predictions() == []  # removed
    resolved = pm.load_praesagium_resolved()
    assert len(resolved) == 1
    r = resolved[0]
    assert r["outcome"] == "fulfilled"
    assert r["lag_s"] == pytest.approx(50.0)
    assert r["pattern_id"] == "p1"
    assert r["resolved_ts"] == 1050.0
    assert pub.subjects() == [SUBJECT_RESOLVED]


def test_resolve_high_severity_also_fulfils():
    pm = _pm()
    _arm(pm, "p1", "typing:user", "activity:editor", created_ts=0.0)
    n = asyncio.run(
        resolve_open_predictions(
            pm, RecordingPublish(), "activity:editor", "high", 30.0, cfg=_cfg()
        )
    )
    assert n == 1
    assert pm.load_praesagium_resolved()[0]["outcome"] == "fulfilled"


def test_resolve_low_severity_does_not_fulfil():
    pm = _pm()
    _arm(pm, "p1", "typing:user", "activity:editor", created_ts=0.0)
    n = asyncio.run(
        resolve_open_predictions(
            pm, RecordingPublish(), "activity:editor", "low", 30.0, cfg=_cfg()
        )
    )
    assert n == 0
    assert len(pm.load_praesagium_open_predictions()) == 1  # still open
    assert pm.load_praesagium_resolved() == []


def test_resolve_consequent_mismatch_no_fulfil():
    pm = _pm()
    _arm(pm, "p1", "typing:user", "activity:editor", created_ts=0.0)
    n = asyncio.run(
        resolve_open_predictions(
            pm, RecordingPublish(), "something:else", "high", 30.0, cfg=_cfg()
        )
    )
    assert n == 0
    assert len(pm.load_praesagium_open_predictions()) == 1


def test_resolve_after_deadline_within_grace_stays_open():
    pm = _pm()
    _arm(pm, "p1", "typing:user", "activity:editor", created_ts=0.0, window_s=100.0)
    # deadline 100; event at 103 with grace 5 -> not fulfilled (past deadline),
    # not expired (103 < 100+5). Stays open.
    n = asyncio.run(
        resolve_open_predictions(
            pm,
            RecordingPublish(),
            "activity:editor",
            "medium",
            103.0,
            cfg=_cfg(praesagium_expiry_grace_s=5.0),
        )
    )
    assert n == 0
    assert len(pm.load_praesagium_open_predictions()) == 1


def test_resolve_expiry_past_deadline_plus_grace():
    pm = _pm()
    _arm(pm, "p1", "typing:user", "activity:editor", created_ts=0.0, window_s=100.0)
    pub = RecordingPublish()
    # deadline 100; event at 106 > 100 + 5 grace -> expired.
    n = asyncio.run(
        resolve_open_predictions(
            pm,
            pub,
            "unrelated:key",
            "medium",
            106.0,
            cfg=_cfg(praesagium_expiry_grace_s=5.0),
        )
    )
    assert n == 1
    r = pm.load_praesagium_resolved()[0]
    assert r["outcome"] == "expired"
    assert r["lag_s"] is None
    assert pub.subjects() == [SUBJECT_RESOLVED]


def test_resolve_replay_same_event_is_noop():
    pm = _pm()
    _arm(pm, "p1", "typing:user", "activity:editor", created_ts=1000.0)
    args = (pm, RecordingPublish(), "activity:editor", "medium", 1050.0)
    first = asyncio.run(resolve_open_predictions(*args, cfg=_cfg()))
    second = asyncio.run(resolve_open_predictions(*args, cfg=_cfg()))
    assert first == 1
    assert second == 0  # atomic op -> replay resolves nothing
    assert len(pm.load_praesagium_resolved()) == 1  # exactly once


def test_resolve_publish_failure_swallowed_but_counts():
    pm = _pm()
    _arm(pm, "p1", "typing:user", "activity:editor", created_ts=1000.0)
    pub = RecordingPublish(fail_on=SUBJECT_RESOLVED)
    n = asyncio.run(
        resolve_open_predictions(
            pm, pub, "activity:editor", "medium", 1050.0, cfg=_cfg()
        )
    )
    assert n == 1  # resolution counted despite publish failure
    assert len(pm.load_praesagium_resolved()) == 1
    assert pub.subjects() == [SUBJECT_RESOLVED]  # attempt was made


# -- corrupt open-prediction records are skipped, not fatal ------------------


def test_resolve_corrupt_open_records_skipped_valid_still_resolves():
    # Three corrupt shapes coexisting with one valid open record: a non-dict
    # JSON value, a dict missing "prediction_id", and a dict with a
    # non-numeric "deadline_ts". None of these may raise or be counted; the
    # valid record must resolve normally and the corrupt hash entries must be
    # left untouched (resolve_open_predictions never writes to them).
    pm = _pm()
    key = "augur:praesagium:predictions:open"
    pm._r.hset(key, "bad-nondict", json.dumps("not-a-dict"))
    pm._r.hset(
        key,
        "bad-missing-pid",
        json.dumps({"deadline_ts": 9999.0, "consequent": "activity:editor"}),
    )
    pm._r.hset(
        key,
        "bad-deadline",
        json.dumps(
            {
                "prediction_id": "bad-deadline",
                "deadline_ts": "soon",
                "consequent": "activity:editor",
            }
        ),
    )
    _arm(pm, "p1", "typing:user", "activity:editor", created_ts=1000.0)

    pub = RecordingPublish()
    n = asyncio.run(
        resolve_open_predictions(
            pm, pub, "activity:editor", "medium", 1050.0, cfg=_cfg()
        )
    )

    assert n == 1  # only the valid record resolves; corrupt ones uncounted
    resolved = pm.load_praesagium_resolved()
    assert len(resolved) == 1
    assert resolved[0]["prediction_id"] == "pred-p1"
    # The 3 corrupt entries remain in the hash exactly as seeded (untouched);
    # the valid one was removed on resolve.
    assert pm._r.hlen(key) == 3
    remaining_ids = set(pm._r.hkeys(key))
    assert remaining_ids == {b"bad-nondict", b"bad-missing-pid", b"bad-deadline"}


# -- self-review: resolve-then-match ordering ---------------------------------


def test_b_event_fulfils_one_and_arms_another_when_ordered():
    # A->B (p1) and B->C (p2), both active. A already fired -> p1 open.
    # A B event must, at the call site (resolve THEN match), fulfil p1 and arm
    # p2 -- and must NOT fulfil the prediction it just armed.
    pm = _pm()
    _seed_patterns(
        pm,
        _pattern(pid="p1", antecedent="a:x", consequent="b:y"),
        _pattern(pid="p2", antecedent="b:y", consequent="c:z"),
    )
    _arm(pm, "p1", "a:x", "b:y", created_ts=1000.0)
    cfg = _cfg()
    pub = RecordingPublish()

    async def _drive() -> int:
        n = await resolve_open_predictions(pm, pub, "b:y", "medium", 1050.0, cfg=cfg)
        await match_patterns(
            pm, pub, "b:y", 1050.0, {"session_id": "s"}, cfg=cfg, cooldowns={}
        )
        return n

    n = asyncio.run(_drive())
    assert n == 1  # p1 fulfilled

    resolved = pm.load_praesagium_resolved()
    assert [r["pattern_id"] for r in resolved] == ["p1"]
    assert resolved[0]["outcome"] == "fulfilled"

    opens = _open_by_pattern(pm)
    assert set(opens) == {"p2"}  # p1 removed, p2 newly armed
    assert opens["p2"]["consequent"] == "c:z"


# -- runner helpers (Task 7) ---------------------------------------------------


class _Msg:
    """Minimal NATS-message stand-in: only ``.data`` (bytes) is read."""

    def __init__(self, data: bytes) -> None:
        self.data = data


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _anomaly(
    *,
    domain: str = "typing",
    entity: str = "user",
    severity: str = "medium",
    ts: float = 1000.0,
    session_id: str | None = "s1",
) -> dict:
    payload: dict = {
        "domain": domain,
        "entity": entity,
        "severity": severity,
        "timestamp": _iso(ts),
    }
    if session_id is not None:
        payload["session_id"] = session_id
    return payload


def _keyspace(pm: PersistenceManager) -> set:
    return set(pm._r.keys())


# -- make_on_anomaly: decode / kill-switch / unkeyable -------------------------


def test_on_anomaly_malformed_json_no_raise():
    pm = _pm()
    pub = RecordingPublish()
    cb = make_on_anomaly(pm, _cfg(), pub, {})
    asyncio.run(cb(_Msg(b"not json")))  # must not raise
    assert _keyspace(pm) == set()
    assert pub.calls == []


def test_on_anomaly_disabled_zero_writes_zero_publishes():
    # PR3 parity: praesagium_enabled=False -> zero pm writes, zero publishes,
    # checked even with an armable pattern already seeded.
    pm = _pm()
    _seed_patterns(pm, _pattern(pid="p1", antecedent="typing:user"))
    before = _keyspace(pm)
    pub = RecordingPublish()
    cb = make_on_anomaly(pm, _cfg(praesagium_enabled=False), pub, {})
    asyncio.run(cb(_Msg(json.dumps(_anomaly()).encode())))
    assert _keyspace(pm) == before
    assert pub.calls == []


def test_on_anomaly_unkeyable_no_writes():
    pm = _pm()
    pub = RecordingPublish()
    cb = make_on_anomaly(pm, _cfg(), pub, {})
    payload = _anomaly(domain="", entity="user")  # canonical_key -> None
    asyncio.run(cb(_Msg(json.dumps(payload).encode())))
    assert _keyspace(pm) == set()
    assert pub.calls == []


def test_on_anomaly_unparseable_timestamp_no_writes():
    pm = _pm()
    pub = RecordingPublish()
    cb = make_on_anomaly(pm, _cfg(), pub, {})
    payload = _anomaly()
    payload["timestamp"] = "not-a-timestamp"
    asyncio.run(cb(_Msg(json.dumps(payload).encode())))
    assert _keyspace(pm) == set()
    assert pub.calls == []


# -- make_on_anomaly: episode + resolve/match ordering -------------------------


def test_on_anomaly_missing_session_id_skips_episode_but_still_arms():
    pm = _pm()
    _seed_patterns(pm, _pattern(pid="p1", antecedent="typing:user"))
    pub = RecordingPublish()
    cooldowns: dict[str, float] = {}
    cb = make_on_anomaly(pm, _cfg(), pub, cooldowns)
    payload = _anomaly(session_id=None)
    asyncio.run(cb(_Msg(json.dumps(payload).encode())))

    assert pm.list_praesagium_episode_sessions() == []  # no episode recorded
    opens = _open_by_pattern(pm)
    assert "p1" in opens  # resolve+match still ran: the pattern armed


def test_on_anomaly_happy_path_appends_episode_with_configured_cap():
    pm = _pm()
    pub = RecordingPublish()
    cfg = _cfg(praesagium_episode_cap_per_session=137)
    seen_caps: list[int] = []
    orig_append = pm.append_praesagium_episode

    def _spy(session_id, entry, *, cap=None):
        seen_caps.append(cap)
        return orig_append(session_id, entry, cap=cap)

    pm.append_praesagium_episode = _spy
    cb = make_on_anomaly(pm, cfg, pub, {})
    payload = _anomaly(session_id="sess-x")
    asyncio.run(cb(_Msg(json.dumps(payload).encode())))

    assert seen_caps == [137]
    episodes = pm.load_praesagium_episodes("sess-x")
    assert len(episodes) == 1
    assert episodes[0]["k"] == "typing:user"
    assert episodes[0]["s"] == "medium"


def test_on_anomaly_episode_append_explodes_swallowed_resolve_match_still_run():
    # An internal pm explosion inside the episode-write step must not stop
    # resolve+match from running (fail silent-and-logged, spec sec 11).
    pm = _pm()
    _seed_patterns(pm, _pattern(pid="p1", antecedent="typing:user"))

    def _boom(*_args, **_kwargs):
        raise RuntimeError("redis exploded")

    pm.append_praesagium_episode = _boom
    pub = RecordingPublish()
    cooldowns: dict[str, float] = {}
    cb = make_on_anomaly(pm, _cfg(), pub, cooldowns)
    payload = _anomaly(session_id="s1")
    asyncio.run(cb(_Msg(json.dumps(payload).encode())))  # must not raise

    opens = _open_by_pattern(pm)
    assert "p1" in opens  # resolve+match still ran despite the append blowing up


def test_on_anomaly_resolve_then_match_ordering_via_callback():
    # End-to-end through the callback: a B event both fulfils an open p1
    # prediction and arms p2 (b:y -> c:z), without fulfilling the one it just
    # armed (mirrors test_b_event_fulfils_one_and_arms_another_when_ordered).
    pm = _pm()
    _seed_patterns(
        pm,
        _pattern(pid="p1", antecedent="a:x", consequent="b:y"),
        _pattern(pid="p2", antecedent="b:y", consequent="c:z"),
    )
    _arm(pm, "p1", "a:x", "b:y", created_ts=1000.0)
    pub = RecordingPublish()
    cb = make_on_anomaly(pm, _cfg(), pub, {})
    payload = _anomaly(
        domain="b", entity="y", severity="medium", ts=1050.0, session_id="s"
    )
    asyncio.run(cb(_Msg(json.dumps(payload).encode())))

    resolved = pm.load_praesagium_resolved()
    assert [r["pattern_id"] for r in resolved] == ["p1"]
    assert resolved[0]["outcome"] == "fulfilled"
    opens = _open_by_pattern(pm)
    assert set(opens) == {"p2"}


# -- on_session_end -------------------------------------------------------


def test_on_session_end_malformed_payload_no_raise():
    asyncio.run(on_session_end(_Msg(b"not json")))  # must not raise


def test_on_session_end_well_formed_no_raise():
    asyncio.run(on_session_end(_Msg(json.dumps({"session_id": "s1"}).encode())))


# -- warm_cooldowns -------------------------------------------------------


def test_warm_cooldowns_max_of_open_and_newest_resolved_created_ts():
    pm = _pm()
    _arm(pm, "p1", "typing:user", "activity:editor", created_ts=1000.0)  # stays open
    _arm(pm, "p2", "typing:user", "activity:editor", created_ts=500.0)
    asyncio.run(
        resolve_open_predictions(
            pm, RecordingPublish(), "activity:editor", "medium", 550.0, cfg=_cfg()
        )
    )
    _arm(pm, "p2", "typing:user", "activity:editor", created_ts=900.0)
    asyncio.run(
        resolve_open_predictions(
            pm, RecordingPublish(), "activity:editor", "medium", 950.0, cfg=_cfg()
        )
    )

    cooldowns = warm_cooldowns(pm)
    assert cooldowns["p1"] == 1000.0
    assert cooldowns["p2"] == 900.0  # the NEWEST resolved record's created_ts


def test_warm_cooldowns_open_beats_older_resolved_same_pattern():
    pm = _pm()
    _arm(pm, "p1", "typing:user", "activity:editor", created_ts=100.0)
    asyncio.run(
        resolve_open_predictions(
            pm, RecordingPublish(), "activity:editor", "medium", 150.0, cfg=_cfg()
        )
    )
    _arm(pm, "p1", "typing:user", "activity:editor", created_ts=800.0)  # re-armed, open

    cooldowns = warm_cooldowns(pm)
    assert cooldowns["p1"] == 800.0  # max(open=800, resolved=100)


def test_warm_cooldowns_corrupt_entries_skipped():
    pm = _pm()
    pm.save_praesagium_open_prediction({"prediction_id": "x", "pattern_id": "p1"})
    pm.save_praesagium_open_prediction(
        {"prediction_id": "y", "pattern_id": "", "created_ts": 5.0}
    )
    pm.save_praesagium_open_prediction(
        {"prediction_id": "z", "pattern_id": "p2", "created_ts": 42.0}
    )

    cooldowns = warm_cooldowns(pm)
    assert cooldowns == {"p2": 42.0}


def test_warm_cooldowns_empty_state_is_empty_dict():
    pm = _pm()
    assert warm_cooldowns(pm) == {}


def test_warm_cooldowns_finds_resolution_deeper_than_default_load_limit():
    # load_praesagium_resolved's own default (limit=50) would miss this
    # record entirely; warm_cooldowns must read past it (default
    # resolved_limit=MAX_PRAESAGIUM_RESOLVED) so a warm start doesn't lose
    # a pattern's cooldown just because 50+ other resolutions logged since.
    pm = _pm()
    log_key = "augur:praesagium:predictions:log"
    pm._r.lpush(
        log_key,
        json.dumps(
            {
                "prediction_id": "pred-deep",
                "pattern_id": "deep-pattern",
                "outcome": "fulfilled",
                "resolved_ts": 1000.0,
                "created_ts": 777.0,
            }
        ),
    )
    # 54 newer resolutions for other patterns push "deep-pattern" to index 54
    # (the 55th entry -- past the default load_praesagium_resolved limit=50).
    for i in range(54):
        pm._r.lpush(
            log_key,
            json.dumps(
                {
                    "prediction_id": f"pred-other-{i}",
                    "pattern_id": "other-pattern",
                    "outcome": "fulfilled",
                    "resolved_ts": 1001.0 + i,
                    "created_ts": 500.0 + i,
                }
            ),
        )

    # Sanity: the default shallow read no longer sees the deep record.
    assert all(r["pattern_id"] != "deep-pattern" for r in pm.load_praesagium_resolved())

    cooldowns = warm_cooldowns(pm)
    assert cooldowns["deep-pattern"] == 777.0


# -- structural pins (PR2) -----------------------------------------------------


def test_no_task_fan_out_in_matcher_source():
    src = inspect.getsource(matcher_module)
    assert "create_task(" not in src


def test_no_llm_imports_in_matcher_source():
    # Import-statement pin (not a bare substring check -- the module's own
    # docstrings legitimately mention "httpx"/"ollama" when describing the
    # PR2 purity invariant they satisfy).
    import_lines = [
        line.strip()
        for line in inspect.getsource(matcher_module).splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    assert not any("httpx" in line for line in import_lines)
    assert not any("ollama" in line.lower() for line in import_lines)


# PR3 heartbeat carve-out (runner-level) ---------------------------------------
#
# Spec sec 11 PR3: praesagium_enabled=False => zero praesagium publishes/
# writes, BUT the process heartbeat on augur.system.heartbeat continues
# (liveness != activity; a disabled-but-required faculty must not read dead).
# run() implements this by starting the heartbeat unconditionally on
# praefectus_enabled and only gating the augur.vigil.anomaly /
# augur.session.end subscriptions on praesagium_enabled. These three tests
# pin that at the run()-process level (make_on_anomaly's own kill-switch
# check is already covered by test_on_anomaly_disabled_zero_writes_zero_publishes
# above).


class _FakeSub:
    """Minimal NATS-subscription stand-in: only ``.unsubscribe()`` is read."""

    def __init__(self) -> None:
        self.unsubscribed = False

    async def unsubscribe(self) -> None:
        self.unsubscribed = True


class _FakeNC:
    """Minimal NATS-client stand-in for run(): records subscribe/publish/close."""

    def __init__(self) -> None:
        self.subscribed: list[str] = []
        self.published: list[tuple[str, bytes]] = []
        self.closed = False

    async def subscribe(self, subject, cb=None):
        self.subscribed.append(subject)
        return _FakeSub()

    async def publish(self, subject, data) -> None:
        self.published.append((subject, data))

    async def close(self) -> None:
        self.closed = True


def _patch_run(
    monkeypatch, cfg: AugurConfig
) -> tuple[_FakeNC, list[tuple[str, float]]]:
    """Patch run()'s connect/config/heartbeat seams; return (fake_nc, hb_calls).

    connect_redis -> a fresh fakeredis client (PersistenceManager itself is
    real -- it only stores whatever client it's given, so there's no need to
    mock it separately). nats.connect -> a _FakeNC. AugurConfig.from_env ->
    cfg. start_heartbeat -> a spy that records (faculty, interval_s) and
    delegates to the real implementation, so the heartbeat carve-out is
    exercised end-to-end (a real asyncio.Task publishing through the fake
    NATS client) rather than merely pinning the call arguments.
    """
    import nats

    monkeypatch.setattr(
        matcher_module.AugurConfig, "from_env", classmethod(lambda cls: cfg)
    )
    monkeypatch.setattr(
        matcher_module,
        "connect_redis",
        lambda config: fakeredis.FakeStrictRedis(decode_responses=False),
    )

    fake_nc = _FakeNC()

    async def _fake_connect(url, connect_timeout=None):
        return fake_nc

    monkeypatch.setattr(nats, "connect", _fake_connect)

    hb_calls: list[tuple[str, float]] = []
    orig_start_heartbeat = matcher_module.start_heartbeat

    def _spy_start_heartbeat(nc, faculty, interval_s):
        hb_calls.append((faculty, interval_s))
        return orig_start_heartbeat(nc, faculty, interval_s)

    monkeypatch.setattr(matcher_module, "start_heartbeat", _spy_start_heartbeat)

    return fake_nc, hb_calls


async def _run_to_steady_state_then_cancel() -> None:
    """Start matcher.run() as a task, let it reach the sleep-forever loop,
    cancel it, and assert teardown propagates ONLY CancelledError (no other
    exception escapes the finally block's cleanup)."""
    task = asyncio.create_task(matcher_module.run())
    for _ in range(50):
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_run_disabled_heartbeats_but_never_subscribes(monkeypatch):
    # (a) praesagium_enabled=False + praefectus_enabled=True -> start_heartbeat
    # called with faculty "praesagium" AND nc.subscribe never called.
    cfg = AugurConfig(praesagium_enabled=False, praefectus_enabled=True)
    fake_nc, hb_calls = _patch_run(monkeypatch, cfg)

    asyncio.run(_run_to_steady_state_then_cancel())

    assert hb_calls == [("praesagium", cfg.praefectus_heartbeat_interval_s)]
    assert fake_nc.subscribed == []


def test_run_enabled_subscribes_both_subjects(monkeypatch):
    # (b) praesagium_enabled=True -> both augur.vigil.anomaly and
    # augur.session.end subscribed (heartbeat still runs -- praefectus_enabled
    # defaults True).
    cfg = AugurConfig(praesagium_enabled=True, praefectus_enabled=True)
    fake_nc, hb_calls = _patch_run(monkeypatch, cfg)

    asyncio.run(_run_to_steady_state_then_cancel())

    assert fake_nc.subscribed == [
        matcher_module.SUBSCRIBE_ANOMALY,
        matcher_module.SUBSCRIBE_SESSION_END,
    ]
    assert hb_calls == [("praesagium", cfg.praefectus_heartbeat_interval_s)]


def test_run_praefectus_disabled_no_heartbeat(monkeypatch):
    # (c) praefectus_enabled=False -> start_heartbeat not called, regardless
    # of praesagium_enabled (subscriptions still happen independently).
    cfg = AugurConfig(praesagium_enabled=True, praefectus_enabled=False)
    fake_nc, hb_calls = _patch_run(monkeypatch, cfg)

    asyncio.run(_run_to_steady_state_then_cancel())

    assert hb_calls == []
    assert fake_nc.subscribed == [
        matcher_module.SUBSCRIBE_ANOMALY,
        matcher_module.SUBSCRIBE_SESSION_END,
    ]
