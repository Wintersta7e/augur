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

import perception.feedback_collector as fc
from perception.feedback_collector import PendingAdvice


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
