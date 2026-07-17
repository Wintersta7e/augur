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
