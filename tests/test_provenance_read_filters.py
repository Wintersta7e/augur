"""CL11 — non-learnable sessions are excluded from learning pools at READ time.

The "filter-at-read" group (feedback, gate logs, reflection reports, read-models)
is always written (``@non_learning_write``); enforcement happens where those
records are READ back into a learning decision. This pins the feedback pool
filter (spec §4.3d): under ENFORCE, ``get_all_feedback`` drops a non-learnable
session's feedback; under OFF/REPORT the pool is untouched (pre-enforcement
behaviour is unchanged).
"""

from __future__ import annotations

import json

import fakeredis
import pytest

from tabula.persistence import PersistenceManager
from tabula.provenance import (
    ProvenanceMode,
    get_provenance_mode,
    set_provenance_mode,
)
from tabula.session import REDIS_KEY_META, build_session_meta


@pytest.fixture(autouse=True)
def _restore_mode():
    prev = get_provenance_mode()
    yield
    set_provenance_mode(prev)


def _pm() -> PersistenceManager:
    return PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=True))


def _seed(pm: PersistenceManager, sid: str, origin: str) -> None:
    pm._r.set(
        REDIS_KEY_META.format(sid=sid),
        json.dumps(
            build_session_meta(sid, origin=origin, created_by="x", started_at="t")
        ),
    )
    pm.save_feedback(sid, {"score": 1})
    pm._r.lpush("augur:responsum:_index", sid)


def test_off_returns_all_feedback() -> None:
    set_provenance_mode(ProvenanceMode.OFF)
    pm = _pm()
    _seed(pm, "real-1", "real")
    _seed(pm, "synth-1", "synthetic")
    sids = {fb["session_id"] for fb in pm.get_all_feedback()}
    assert sids == {"real-1", "synth-1"}  # OFF: unchanged, both present


def test_report_returns_all_feedback() -> None:
    set_provenance_mode(ProvenanceMode.REPORT)
    pm = _pm()
    _seed(pm, "real-1", "real")
    _seed(pm, "synth-1", "synthetic")
    sids = {fb["session_id"] for fb in pm.get_all_feedback()}
    assert sids == {"real-1", "synth-1"}  # REPORT measures, does not filter


def test_enforce_excludes_non_learnable_feedback() -> None:
    set_provenance_mode(ProvenanceMode.ENFORCE)
    pm = _pm()
    _seed(pm, "real-1", "real")
    _seed(pm, "synth-1", "synthetic")
    _seed(pm, "unattr-1", "unattributed")
    sids = {fb["session_id"] for fb in pm.get_all_feedback()}
    assert sids == {"real-1"}  # only the learnable session survives the pool


def _seed_reflection(pm: PersistenceManager, sid: str, origin: str, ts: str) -> None:
    pm._r.set(
        REDIS_KEY_META.format(sid=sid),
        json.dumps(
            build_session_meta(sid, origin=origin, created_by="x", started_at="t")
        ),
    )
    pm.save_reflection(sid, {"session_id": sid, "timestamp": ts})


def test_enforce_excludes_non_learnable_reflection_from_imperator() -> None:
    # §4.3e: the Imperator read-model excludes a non-learnable session's reflection.
    from imperator.sources import resolve_latest_reflection

    pm = _pm()
    # The synthetic session is the CURRENT one and has the NEWER reflection, so
    # without the filter it would win; the real session's older one must be chosen.
    pm._r.set("augur:session:current", json.dumps({"session_id": "synth-1"}))
    _seed_reflection(pm, "synth-1", "synthetic", "2026-07-17T13:00:00+00:00")
    _seed_reflection(pm, "real-1", "real", "2026-07-17T12:00:00+00:00")
    _seed(pm, "real-1", "real")  # puts real-1 in the recent-session pool

    set_provenance_mode(ProvenanceMode.OFF)
    assert resolve_latest_reflection(pm)["session_id"] == "synth-1"  # newer wins

    set_provenance_mode(ProvenanceMode.ENFORCE)
    assert resolve_latest_reflection(pm)["session_id"] == "real-1"  # synthetic excluded


def _spawned_for(pm: PersistenceManager, subject: str, payload: dict) -> int:
    """Run the improver's dispatch callback once; return how many cycles it spawned."""
    import asyncio

    from imperator import improver
    from tests.test_imperator_improver import _Cfg, _Msg

    cfg = _Cfg()
    cfg.imperator_ii_min_interval_s = 0.0
    spawned: list = []

    async def scenario() -> None:
        on_msg = improver.make_on_msg(
            pm,
            cfg,
            None,
            lock=asyncio.Lock(),
            last_run=[0.0],
            spawn=lambda coro: (spawned.append(coro), coro.close()),
            publish=lambda s, d: None,
        )
        await on_msg(_Msg(subject, payload))

    asyncio.run(scenario())
    return len(spawned)


def test_enforce_skips_a_reflection_trigger_from_a_non_learnable_session() -> None:
    # §4.3e second layer: filtering the read-model keeps a synthetic reflection
    # out of the self-model, but the trigger itself would still spend an LLM
    # cycle reasoning about it. Drop it at the dispatch path too.
    pm = _pm()
    _seed_reflection(pm, "synth-1", "synthetic", "2026-07-17T13:00:00+00:00")
    payload = {"session_id": "synth-1", "timestamp": "2030-01-01T00:00:00+00:00"}

    set_provenance_mode(ProvenanceMode.OFF)
    assert _spawned_for(pm, "augur.disciplina.complete", payload) == 1

    set_provenance_mode(ProvenanceMode.ENFORCE)
    assert _spawned_for(pm, "augur.disciplina.complete", payload) == 0


def test_enforce_still_runs_a_user_driven_dialogue_trigger() -> None:
    # The dialogue trigger is a direct user action carrying no session; it is not
    # perception learning and must never be gated on a session's provenance.
    pm = _pm()
    set_provenance_mode(ProvenanceMode.ENFORCE)
    assert (
        _spawned_for(
            pm, "augur.imperator.ii.trigger", {"reason": "dialogue", "ts": 1.0}
        )
        == 1
    )
