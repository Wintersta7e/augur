"""Conscientia review hook wired into Disciplina's reflection cycle (Task 11).

Mirrors the run_reflection wiring harness used for the Gate/Memory passes in
tests/test_reflection_gate.py's TestRunReflectionWiring: a real
PersistenceManager over fakeredis, AsyncMock() for http_client/nc, and a
plain AugurConfig() for config.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import fakeredis
import pytest

from tabula.config import AugurConfig
from tabula.persistence import PersistenceManager
from disciplina.reflection_engine import run_reflection

CONFIG = AugurConfig()


def _pm() -> PersistenceManager:
    return PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=True))


def _feedback(session_id: str) -> dict:
    return {"session_id": session_id, "advice_events": []}


def _seed_gated_proposal(pm: PersistenceManager, pid: str = "g1") -> None:
    pm.save_proposal(
        {
            "proposal_id": pid,
            "dedupe_key": f"dk-{pid}",
            "kind": "code",
            "target": "vigil/x.py",
            "klass": "gated",
            "ts": 1.0,
            "action": {"patch": "x"},
            "status": "logged",
        }
    )


@pytest.mark.asyncio
async def test_conscientia_review_rides_the_reflection_cycle() -> None:
    pm = _pm()
    _seed_gated_proposal(pm)
    feedback = _feedback("sess-conscientia")
    pm.save_feedback("sess-conscientia", feedback)

    report = await run_reflection(
        "sess-conscientia",
        feedback,
        pm,
        fakeredis.FakeStrictRedis(decode_responses=True),
        AsyncMock(),
        AsyncMock(),
        CONFIG,
    )

    assert report["conscientia"]["reviewed"] == 1
    assert len(pm.load_conscientia_verdicts(limit=5)) == 1


@pytest.mark.asyncio
async def test_conscientia_disabled_is_zero_shape() -> None:
    pm = _pm()
    _seed_gated_proposal(pm)
    feedback = _feedback("sess-conscientia-off")
    pm.save_feedback("sess-conscientia-off", feedback)
    cfg = AugurConfig(conscientia_enabled=False)

    report = await run_reflection(
        "sess-conscientia-off",
        feedback,
        pm,
        fakeredis.FakeStrictRedis(decode_responses=True),
        AsyncMock(),
        AsyncMock(),
        cfg,
    )

    assert report["conscientia"]["reviewed"] == 0
    assert pm.load_conscientia_verdicts(limit=5) == []


@pytest.mark.asyncio
async def test_conscientia_review_failure_is_non_fatal(monkeypatch) -> None:
    pm = _pm()
    feedback = _feedback("sess-conscientia-fail")
    pm.save_feedback("sess-conscientia-fail", feedback)

    async def _boom(pm, nc, cfg):
        raise RuntimeError("conscientia exploded")

    monkeypatch.setattr("conscientia.auditor.run_conscientia_review", _boom)

    report = await run_reflection(
        "sess-conscientia-fail",
        feedback,
        pm,
        fakeredis.FakeStrictRedis(decode_responses=True),
        AsyncMock(),
        AsyncMock(),
        CONFIG,
    )

    assert "conscientia exploded" in report["conscientia"]["error"]
