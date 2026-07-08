"""Integration test: Conscientia's gated-review sweep rides Disciplina's live
reflection cycle (fast tier — real Redis + NATS, no Ollama).

The live counterpart to ``tests/test_conscientia_reflection_hook.py``: a real
``disciplina`` subprocess started via the ``pipeline`` fixture, a seeded gated
proposal plus matching feedback, an ``augur.disciplina.trigger`` publish, and
assertions on BOTH the persisted verdict (``load_conscientia_verdicts``) and the
``augur.conscientia.verdict`` event.

Empty ``advice_events`` keep the utility pass off the Ollama prompt-mutation
path, so the whole reflection runs without an LLM — this is a fast test.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tabula.persistence import PersistenceManager

# The gated proposal's target sits outside charter.PROTECTED_SURFACES, so the
# rubric's recommendation is "needs_human" (not "reject").
_PROPOSAL_ID = "conscientia-live-g1"
_SESSION_ID = "conscientia-live"


@pytest.mark.parametrize("pipeline", [["disciplina"]], indirect=True)
@pytest.mark.asyncio
async def test_conscientia_verdict_from_live_reflection(
    pipeline, nats_conn, redis_client
) -> None:
    """A gated proposal swept during the live reflection cycle yields a
    persisted verdict AND an ``augur.conscientia.verdict`` event."""
    pm = PersistenceManager(redis_client)

    # Subscribe FIRST — before the proposal even exists. NATS core has no
    # persistence, and a rebuilt containerized disciplina also listens on
    # augur.disciplina.trigger, so subscribing ahead of the seed guarantees we
    # catch every verdict event produced for our proposal.
    received: list[dict] = []

    async def _capture(msg) -> None:  # type: ignore[no-untyped-def]
        received.append(json.loads(msg.data.decode()))

    sub = await nats_conn.subscribe("augur.conscientia.verdict", cb=_capture)

    # Seed one gated proposal and matching feedback. The feedback is required:
    # on_trigger bails before run_reflection if get_feedback(session_id) is None,
    # so without it the conscientia pass never runs.
    pm.save_proposal(
        {
            "proposal_id": _PROPOSAL_ID,
            "dedupe_key": f"dk-{_PROPOSAL_ID}",
            "kind": "code",
            "target": "vigil/x.py",
            "klass": "gated",
            "ts": 1.0,
            "action": {"patch": "x"},
            "status": "logged",
        }
    )
    pm.save_feedback(_SESSION_ID, {"session_id": _SESSION_ID, "advice_events": []})

    await nats_conn.publish(
        "augur.disciplina.trigger",
        json.dumps({"session_id": _SESSION_ID}).encode(),
    )

    # Poll the durable verdict store. The reflection runs several passes before
    # the conscientia sweep; keep the timeout generous for the WSL box.
    verdict: dict | None = None
    for _ in range(300):  # up to ~60s at 0.2s
        matches = [
            v
            for v in pm.load_conscientia_verdicts(limit=50)
            if v.get("proposal_id") == _PROPOSAL_ID
        ]
        if matches:
            verdict = matches[0]
            break
        await asyncio.sleep(0.2)

    # Let the verdict event land (the auditor persists then publishes, so the
    # event trails the durable write by a beat).
    for _ in range(25):  # up to ~5s
        if any(m.get("proposal_id") == _PROPOSAL_ID for m in received):
            break
        await asyncio.sleep(0.2)

    await sub.unsubscribe()

    assert verdict is not None, "No conscientia verdict persisted within timeout"
    assert verdict["recommendation"] == "needs_human"
    assert verdict["charter_version"]

    event_matches = [m for m in received if m.get("proposal_id") == _PROPOSAL_ID]
    assert event_matches, "No augur.conscientia.verdict event received"
    assert event_matches[0]["recommendation"] == "needs_human"
