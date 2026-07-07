"""Auditor — idempotent review sweep over the gated log."""

import json
from unittest.mock import AsyncMock, MagicMock

import fakeredis
import pytest

from conscientia.auditor import run_conscientia_review
from tabula.config import AugurConfig
from tabula.persistence import (
    MAX_CONSCIENTIA_VERDICTS,
    MAX_IMPERATOR_PROPOSALS,
    PersistenceManager,
)

CFG = AugurConfig()


def _pm():
    return PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))


def _seed(pm, pid, klass="gated", kind="code", target="vigil/x.py"):
    pm.save_proposal(
        {
            "proposal_id": pid,
            "dedupe_key": f"dk-{pid}",
            "kind": kind,
            "target": target,
            "klass": klass,
            "ts": 1.0,
            "action": {"patch": "x"},
            "status": "logged",
        }
    )


@pytest.mark.asyncio
async def test_reviews_only_gated_and_publishes():
    pm = _pm()
    _seed(pm, "g1")
    _seed(pm, "g2", target="limen/gate.py")
    _seed(pm, "s1", klass="safe", kind="sigma")
    nc = MagicMock()
    nc.publish = AsyncMock()
    out = await run_conscientia_review(pm, nc, CFG)
    assert out == {"reviewed": 2, "recommendations": {"reject": 1, "needs_human": 1}}
    assert nc.publish.await_count == 2
    subj, payload = nc.publish.await_args.args
    assert subj == "augur.conscientia.verdict"
    assert json.loads(payload.decode())["charter_version"]


@pytest.mark.asyncio
async def test_idempotent_across_sweeps():
    pm = _pm()
    _seed(pm, "g1")
    await run_conscientia_review(pm, None, CFG)
    out2 = await run_conscientia_review(pm, None, CFG)
    assert out2["reviewed"] == 0
    assert len(pm.load_conscientia_verdicts(limit=50)) == 1


@pytest.mark.asyncio
async def test_idempotent_at_full_proposal_cap():
    """Regression for the coupled-caps invariant (MAX_CONSCIENTIA_VERDICTS
    >= MAX_IMPERATOR_PROPOSALS). Seed more gated proposals than the
    proposal store can hold; one sweep must fully review whatever survives
    the trim, and a second sweep must find nothing left to re-review."""
    pm = _pm()
    for i in range(MAX_IMPERATOR_PROPOSALS + 10):
        _seed(pm, f"p{i}")
    out1 = await run_conscientia_review(pm, None, CFG)
    # The proposal store trims to MAX_IMPERATOR_PROPOSALS entries, so that's
    # exactly how many gated proposals are visible to (and reviewed by) the
    # first sweep.
    assert out1["reviewed"] == MAX_IMPERATOR_PROPOSALS
    # Every one of those got a verdict persisted -- only guaranteed because
    # MAX_CONSCIENTIA_VERDICTS >= MAX_IMPERATOR_PROPOSALS.
    visible_verdicts = pm.load_conscientia_verdicts(
        limit=max(MAX_CONSCIENTIA_VERDICTS, MAX_IMPERATOR_PROPOSALS)
    )
    assert len(visible_verdicts) == min(
        MAX_IMPERATOR_PROPOSALS, MAX_CONSCIENTIA_VERDICTS
    )
    out2 = await run_conscientia_review(pm, None, CFG)
    assert out2["reviewed"] == 0


@pytest.mark.asyncio
async def test_disabled_noops():
    pm = _pm()
    _seed(pm, "g1")
    out = await run_conscientia_review(pm, None, AugurConfig(conscientia_enabled=False))
    assert out == {"reviewed": 0, "recommendations": {"reject": 0, "needs_human": 0}}
    assert pm.load_conscientia_verdicts(limit=50) == []


@pytest.mark.asyncio
async def test_publish_failure_is_non_fatal():
    pm = _pm()
    _seed(pm, "g1")
    nc = MagicMock()
    nc.publish = AsyncMock(side_effect=RuntimeError("nats down"))
    out = await run_conscientia_review(pm, nc, CFG)
    assert out["reviewed"] == 1
    assert len(pm.load_conscientia_verdicts(limit=50)) == 1
