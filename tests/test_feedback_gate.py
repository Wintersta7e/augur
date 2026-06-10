"""Gate MRT data-path tests for the feedback collector (spec §9).

Task 9.1: PendingAdvice gains decision_id/probe/mrt_eligible/p_fire/
behavioral_finalized (+ to_record); on_advice reads decision_id (plus
mrt_eligible/p_fire/probe) from the advice payload so the fired arm can be
joined to its feedback by exact key and inverse-probability-weighted.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import responsum.feedback_collector as fc
from responsum.feedback_collector import PendingAdvice


def _base_kwargs() -> dict[str, Any]:
    return {
        "advice_id": "adv-0001",
        "domain": "chess",
        "entity": "white",
        "severity": "medium",
        "baseline_mean": 8.2,
        "timestamp": "2026-06-06T12:00:00+00:00",
    }


# ── PendingAdvice gate fields ────────────────────────────────────────────────


class TestDefaultGateFields:
    def test_default_decision_id_is_none(self) -> None:
        assert PendingAdvice(**_base_kwargs()).decision_id is None

    def test_default_probe_is_false(self) -> None:
        assert PendingAdvice(**_base_kwargs()).probe is False

    def test_default_mrt_eligible_is_false(self) -> None:
        assert PendingAdvice(**_base_kwargs()).mrt_eligible is False

    def test_default_p_fire_is_none(self) -> None:
        assert PendingAdvice(**_base_kwargs()).p_fire is None

    def test_default_behavioral_finalized_tracks_finalized(self) -> None:
        # behavioral_finalized in to_record mirrors the runtime finalized flag.
        p = PendingAdvice(**_base_kwargs())
        assert p.to_record()["behavioral_finalized"] is False


class TestExplicitGateFields:
    def test_all_gate_fields_set_explicitly(self) -> None:
        p = PendingAdvice(
            **_base_kwargs(),
            decision_id="dec-abc123",
            probe=True,
            mrt_eligible=True,
            p_fire=0.1,
        )
        assert p.decision_id == "dec-abc123"
        assert p.probe is True
        assert p.mrt_eligible is True
        assert p.p_fire == 0.1


class TestToRecordIncludesGateFields:
    def test_record_has_all_gate_fields(self) -> None:
        p = PendingAdvice(
            **_base_kwargs(),
            decision_id="dec-abc123",
            probe=True,
            mrt_eligible=True,
            p_fire=0.1,
        )
        rec = p.to_record()
        assert rec["decision_id"] == "dec-abc123"
        assert rec["probe"] is True
        assert rec["mrt_eligible"] is True
        assert rec["p_fire"] == 0.1
        assert "behavioral_finalized" in rec

    def test_record_defaults_present_for_legacy_advice(self) -> None:
        rec = PendingAdvice(**_base_kwargs()).to_record()
        assert rec["decision_id"] is None
        assert rec["probe"] is False
        assert rec["mrt_eligible"] is False
        assert rec["p_fire"] is None
        assert rec["behavioral_finalized"] is False

    def test_behavioral_finalized_true_after_scoring(self) -> None:
        p = PendingAdvice(**_base_kwargs())
        for _ in range(fc.POST_ADVICE_TRACK_MOVES):
            p.add_post_move(8.0)
        assert p.finalized is True
        assert p.to_record()["behavioral_finalized"] is True

    def test_record_preserves_existing_correlation_fields(self) -> None:
        # The new gate fields must not displace the existing record schema.
        p = PendingAdvice(**_base_kwargs(), correlation_found=True)
        rec = p.to_record()
        assert rec["correlation_found"] is True
        assert rec["advice_id"] == "adv-0001"
        assert rec["domain"] == "chess"


# ── on_advice reads the gate fields from the advice payload ──────────────────


async def _drive_on_advice(monkeypatch: pytest.MonkeyPatch, payload: dict) -> list:
    """Start run(), capture the on_advice closure, feed it one advice msg.

    Mocks NATS + Redis so only the on_advice extraction path executes; returns
    the list of PendingAdvice instances created during the call.
    """
    captured: dict[str, Any] = {}

    async def fake_subscribe(subject: str, cb: Any) -> MagicMock:
        if subject == fc.SUBJECT_ADVICE:
            captured["on_advice"] = cb
        sub = MagicMock()
        sub.unsubscribe = AsyncMock()
        return sub

    fake_nc = MagicMock()
    fake_nc.subscribe = fake_subscribe
    fake_nc.close = AsyncMock()
    fake_nc.publish = AsyncMock()

    async def fake_connect(*args: Any, **kwargs: Any) -> MagicMock:
        return fake_nc

    monkeypatch.setattr(fc.nats, "connect", fake_connect)

    fake_pm = MagicMock()
    fake_pm.load_baseline.return_value = {"ewma_mean": 7.5}
    fake_pm.save_feedback.return_value = None
    monkeypatch.setattr(fc, "PersistenceManager", lambda _r: fake_pm)

    fake_redis = MagicMock()
    fake_redis.get.return_value = json.dumps({"session_id": "sess-1"})
    monkeypatch.setattr(fc, "connect_redis", lambda _c: fake_redis)

    # No real stdin prompt — return immediately ("skip").
    async def fake_read(_timeout: float) -> str:
        return "s"

    monkeypatch.setattr(fc, "read_stdin_with_timeout", fake_read)

    # Capture every PendingAdvice constructed during on_advice.
    created: list[PendingAdvice] = []
    real_pending = fc.PendingAdvice

    def spy_pending(*args: Any, **kwargs: Any) -> PendingAdvice:
        inst = real_pending(*args, **kwargs)
        created.append(inst)
        return inst

    monkeypatch.setattr(fc, "PendingAdvice", spy_pending)

    task = asyncio.create_task(fc.run())
    # Let run() register subscriptions.
    for _ in range(50):
        await asyncio.sleep(0)
        if "on_advice" in captured:
            break

    msg = MagicMock()
    msg.data = json.dumps(payload).encode()
    await captured["on_advice"](msg)

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    return created


@pytest.mark.asyncio
async def test_on_advice_reads_gate_fields_from_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "domain": "chess",
        "player": "white",
        "severity": "medium",
        "think_time": 9.0,
        "decision_id": "dec-xyz",
        "mrt_eligible": True,
        "p_fire": 0.1,
        "probe": True,
    }
    created = await _drive_on_advice(monkeypatch, payload)
    assert len(created) == 1
    p = created[0]
    assert p.decision_id == "dec-xyz"
    assert p.mrt_eligible is True
    assert p.p_fire == 0.1
    assert p.probe is True


@pytest.mark.asyncio
async def test_on_advice_gate_fields_default_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "domain": "chess",
        "player": "white",
        "severity": "medium",
        "think_time": 9.0,
    }
    created = await _drive_on_advice(monkeypatch, payload)
    assert len(created) == 1
    p = created[0]
    assert p.decision_id is None
    assert p.mrt_eligible is False
    assert p.p_fire is None
    assert p.probe is False


# ── PendingGateDecision (the MRT withheld/control arm) ───────────────────────


def _suppressed_payload(**overrides: Any) -> dict[str, Any]:
    """A full §8 augur.limen.suppressed payload (primary domain+entity)."""
    payload = {
        "decision_id": "dec-w001",
        "state_key": "single:chess:white:medium",
        "domain": "chess",
        "entity": "white",
        "value": 14.0,
        "baseline_mean": 7.5,
        "severity": "medium",
        "session_id": "sess-1",
        "arm": "habituation",
        "reason": "habituated channel",
        "mrt_eligible": True,
        "p_withhold": 0.9,
        "timestamp": "2026-06-06T12:00:00+00:00",
    }
    payload.update(overrides)
    return payload


class TestPendingGateDecision:
    def test_carries_core_fields(self) -> None:
        from responsum.feedback_collector import PendingGateDecision

        d = PendingGateDecision(
            decision_id="dec-w001",
            state_key="single:chess:white:medium",
            domain="chess",
            entity="white",
            severity="medium",
            baseline_mean=7.5,
            timestamp="2026-06-06T12:00:00+00:00",
            mrt_eligible=True,
            p_withhold=0.9,
            reason="habituated channel",
        )
        assert d.decision_id == "dec-w001"
        assert d.state_key == "single:chess:white:medium"
        assert d.domain == "chess"
        assert d.entity == "white"
        assert d.severity == "medium"
        assert d.baseline_mean == 7.5
        assert d.mrt_eligible is True
        assert d.p_withhold == 0.9
        assert d.reason == "habituated channel"

    def test_to_record_has_spec_fields(self) -> None:
        from responsum.feedback_collector import PendingGateDecision

        d = PendingGateDecision(
            decision_id="dec-w001",
            state_key="single:chess:white:medium",
            domain="chess",
            entity="white",
            severity="medium",
            baseline_mean=7.5,
            timestamp="2026-06-06T12:00:00+00:00",
            mrt_eligible=True,
            p_withhold=0.9,
            reason="habituated channel",
        )
        rec = d.to_record()
        for key in (
            "decision_id",
            "state_key",
            "domain",
            "entity",
            "severity",
            "mrt_eligible",
            "p_withhold",
            "baseline_mean",
            "behavioral_score",
            "behavioral_finalized",
            "explicit_rating",
            "reason",
            "timestamp",
        ):
            assert key in rec, f"missing {key}"
        assert rec["behavioral_finalized"] is False
        assert rec["explicit_rating"] == "no_response"

    def test_behavioral_score_matches_pending_advice(self) -> None:
        # Same post-decision behavioral-score computation as PendingAdvice.
        from responsum.feedback_collector import PendingGateDecision

        kwargs = dict(
            domain="chess",
            entity="white",
            severity="medium",
            baseline_mean=8.2,
            timestamp="2026-06-06T12:00:00+00:00",
        )
        adv = PendingAdvice(advice_id="adv-1", **kwargs)
        dec = PendingGateDecision(
            decision_id="dec-1",
            state_key="single:chess:white:medium",
            mrt_eligible=False,
            p_withhold=None,
            reason="r",
            **kwargs,
        )
        for v in (9.0, 8.0, 7.0):
            adv.add_post_move(v)
            dec.add_post_move(v)
        assert dec.finalized is True
        assert dec.behavioral_score == adv.behavioral_score
        assert dec.to_record()["behavioral_finalized"] is True


# ── on_suppressed subscriber + gate_decision_events persistence ──────────────


async def _drive_subscribers(
    monkeypatch: pytest.MonkeyPatch,
    feed: Any,
) -> dict[str, Any]:
    """Start run(), capture all subscriber closures, run a feed coroutine.

    ``feed(captured, ctx)`` is awaited with the captured callbacks and a ctx
    dict carrying the fake pm so the test can inspect persisted records.
    Returns the ctx dict.
    """
    captured: dict[str, Any] = {}
    subjects = {
        fc.SUBJECT_ADVICE: "on_advice",
        fc.SUBJECT_PERCEPTION: "on_perception",
        fc.SUBJECT_SESSION_END: "on_session_end",
        fc.SUBJECT_SUPPRESSED: "on_suppressed",
    }

    async def fake_subscribe(subject: str, cb: Any) -> MagicMock:
        if subject in subjects:
            captured[subjects[subject]] = cb
        sub = MagicMock()
        sub.unsubscribe = AsyncMock()
        return sub

    fake_nc = MagicMock()
    fake_nc.subscribe = fake_subscribe
    fake_nc.close = AsyncMock()
    fake_nc.publish = AsyncMock()

    async def fake_connect(*args: Any, **kwargs: Any) -> MagicMock:
        return fake_nc

    monkeypatch.setattr(fc.nats, "connect", fake_connect)

    fake_pm = MagicMock()
    fake_pm.load_baseline.return_value = {"ewma_mean": 7.5}
    saved: list[dict] = []

    def _save_feedback(_sid: str, record: dict) -> None:
        saved.append(record)

    fake_pm.save_feedback.side_effect = _save_feedback
    monkeypatch.setattr(fc, "PersistenceManager", lambda _r: fake_pm)

    fake_redis = MagicMock()
    fake_redis.get.return_value = json.dumps({"session_id": "sess-1"})
    monkeypatch.setattr(fc, "connect_redis", lambda _c: fake_redis)

    async def fake_read(_timeout: float) -> str:
        return "s"

    monkeypatch.setattr(fc, "read_stdin_with_timeout", fake_read)

    task = asyncio.create_task(fc.run())
    for _ in range(50):
        await asyncio.sleep(0)
        if len(captured) == len(subjects):
            break

    ctx = {"pm": fake_pm, "nc": fake_nc, "saved": saved}
    await feed(captured, ctx)

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    return ctx


def _suppressed_msg(**overrides: Any) -> MagicMock:
    msg = MagicMock()
    msg.data = json.dumps(_suppressed_payload(**overrides)).encode()
    return msg


def _perception_msg(domain: str, entity: str, value: float) -> MagicMock:
    msg = MagicMock()
    msg.data = json.dumps(
        {
            "domain": domain,
            "stream_id": domain,
            "entity": entity,
            "event_type": "move",
            "value": value,
            "unit": "s",
            "context": {},
            "timestamp": "2026-06-06T12:00:01+00:00",
            "session_id": "sess-1",
        }
    ).encode()
    return msg


@pytest.mark.asyncio
async def test_on_suppressed_creates_gate_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def feed(cap: dict, ctx: dict) -> None:
        await cap["on_suppressed"](_suppressed_msg())
        await cap["on_session_end"](MagicMock())

    ctx = await _drive_subscribers(monkeypatch, feed)
    record = ctx["saved"][-1]
    events = record["gate_decision_events"]
    assert len(events) == 1
    ev = events[0]
    assert ev["decision_id"] == "dec-w001"
    assert ev["state_key"] == "single:chess:white:medium"
    assert ev["domain"] == "chess"
    assert ev["entity"] == "white"
    assert ev["severity"] == "medium"
    assert ev["mrt_eligible"] is True
    assert ev["p_withhold"] == 0.9
    assert ev["baseline_mean"] == 7.5
    assert ev["reason"] == "habituated channel"


@pytest.mark.asyncio
async def test_on_suppressed_tracks_post_decision_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def feed(cap: dict, ctx: dict) -> None:
        await cap["on_suppressed"](_suppressed_msg())
        for v in (7.0, 6.8, 6.5):
            await cap["on_perception"](_perception_msg("chess", "white", v))
        await cap["on_session_end"](MagicMock())

    ctx = await _drive_subscribers(monkeypatch, feed)
    record = ctx["saved"][-1]
    ev = record["gate_decision_events"][0]
    assert ev["behavioral_finalized"] is True
    assert ev["behavioral_score"] > 0.0


@pytest.mark.asyncio
async def test_withheld_only_session_persisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A session with NO advice but withheld decisions must NOT be dropped.
    async def feed(cap: dict, ctx: dict) -> None:
        await cap["on_suppressed"](_suppressed_msg())
        await cap["on_session_end"](MagicMock())

    ctx = await _drive_subscribers(monkeypatch, feed)
    # save_feedback must have been called (early-return removed).
    assert ctx["pm"].save_feedback.called
    record = ctx["saved"][-1]
    assert record["advice_events"] == []
    assert len(record["gate_decision_events"]) == 1
    assert record["session_summary"]["total_gate_decisions"] == 1


@pytest.mark.asyncio
async def test_session_summary_counts_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    advice_payload = {
        "domain": "chess",
        "player": "black",
        "severity": "high",
        "think_time": 9.0,
        "decision_id": "dec-fire",
    }

    async def feed(cap: dict, ctx: dict) -> None:
        msg = MagicMock()
        msg.data = json.dumps(advice_payload).encode()
        await cap["on_advice"](msg)
        await cap["on_suppressed"](_suppressed_msg())
        await cap["on_session_end"](MagicMock())

    ctx = await _drive_subscribers(monkeypatch, feed)
    summary = ctx["saved"][-1]["session_summary"]
    assert summary["total_advice"] == 1
    assert summary["total_gate_decisions"] == 1


@pytest.mark.asyncio
async def test_single_tracker_per_decision_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two augur.limen.suppressed messages for the same decision_id must
    # create exactly ONE PendingGateDecision tracker.
    async def feed(cap: dict, ctx: dict) -> None:
        await cap["on_suppressed"](_suppressed_msg(decision_id="dup-1"))
        await cap["on_suppressed"](_suppressed_msg(decision_id="dup-1"))
        await cap["on_session_end"](MagicMock())

    ctx = await _drive_subscribers(monkeypatch, feed)
    events = ctx["saved"][-1]["gate_decision_events"]
    assert len(events) == 1
    assert events[0]["decision_id"] == "dup-1"


@pytest.mark.asyncio
async def test_probe_creates_no_gate_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A probe fires real advice -> on_advice makes a PendingAdvice(probe=True);
    # on_suppressed is NOT called for a probe, so no PendingGateDecision exists.
    probe_payload = {
        "domain": "chess",
        "player": "white",
        "severity": "medium",
        "think_time": 9.0,
        "decision_id": "dec-probe",
        "probe": True,
        "mrt_eligible": True,
        "p_fire": 0.1,
    }

    async def feed(cap: dict, ctx: dict) -> None:
        msg = MagicMock()
        msg.data = json.dumps(probe_payload).encode()
        await cap["on_advice"](msg)
        await cap["on_session_end"](MagicMock())

    ctx = await _drive_subscribers(monkeypatch, feed)
    record = ctx["saved"][-1]
    assert record["gate_decision_events"] == []
    assert len(record["advice_events"]) == 1
    assert record["advice_events"][0]["probe"] is True
