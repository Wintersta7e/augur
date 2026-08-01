"""CL11 — gate logs carry their session, and only the LEARNING reads filter them.

The emission/silence logs are written unconditionally (``@non_learning_write``):
the online arms (refractory / global pressure / duplicate) must see every event
that really happened — including a synthetic driver's — or the gate would spam a
live console during a shakeout.  Enforcement therefore belongs at the *learning*
reads: ``analyze_gate``'s offline MRT/IPW readout and the Imperator self-model's
windowed rates, which opt in with ``learnable_only=True``.

That split is only possible because each record now carries the ``session_id``
of the ``LearnContext`` threaded into ``gate.record_*`` (spec §4.3c).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import fakeredis
import pytest

from disciplina.reflection_engine import analyze_gate
from limen.gate import Gate, GateDecision, build_signature
from tabula.config import AugurConfig
from tabula.persistence import PersistenceManager
from tabula.provenance import (
    LearnContext,
    ProvenanceMode,
    get_provenance_mode,
    set_provenance_mode,
)
from tabula.session import REDIS_KEY_META, build_session_meta
from tests.conftest import SINGLE_MEDIUM
from tests.test_advisor_gate_flow import NOW, _run, _scheduler
from tests.test_reflection_gate import _feedback, _gate_decision

CONFIG = AugurConfig()

REAL_SID = "real-1"
SYNTH_SID = "synth-1"


@pytest.fixture(autouse=True)
def _restore_mode():
    prev = get_provenance_mode()
    yield
    set_provenance_mode(prev)


def _pm() -> PersistenceManager:
    return PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=True))


def _mint(pm: PersistenceManager, sid: str, origin: str) -> None:
    pm._r.set(
        REDIS_KEY_META.format(sid=sid),
        json.dumps(
            build_session_meta(sid, origin=origin, created_by="x", started_at="t")
        ),
    )


def _emission(sid: str | None, decision_id: str = "e1") -> dict:
    return {
        "ts": 1.0,
        "decision_id": decision_id,
        "session_id": sid,
        "state_key": "single:chess:board",
        "severity": "medium",
        "tier": 2,
        "probe": False,
        "audit_only": False,
        "withheld_reason": None,
        "mrt_eligible": False,
        "p_fire": None,
    }


def _silence(sid: str | None, decision_id: str = "s1") -> dict:
    return {
        "ts": 1.0,
        "decision_id": decision_id,
        "session_id": sid,
        "state_key": "single:chess:board",
        "domain": "chess",
        "entity": "board",
        "severity": "medium",
        "arm": "bet_hedge",
        "reason": "low_credibility_class",
        "metrics": {},
        "mrt_eligible": True,
        "p_withhold": 0.9,
    }


# ── write side: the record carries the event's session ───────────────────────


def test_emission_record_carries_the_event_session(fake_pm, cfg) -> None:
    gate = Gate(config=cfg)
    sig = build_signature(SINGLE_MEDIUM)
    gate.record_delivery_success(
        sig,
        fake_pm,
        NOW,
        decision=GateDecision.fire("passed_all_arms"),
        tier=2,
        ctx=LearnContext(REAL_SID, True, "real"),
    )
    assert fake_pm.load_emissions(limit=10)[0]["session_id"] == REAL_SID


def test_silence_record_carries_the_event_session(fake_pm, cfg) -> None:
    gate = Gate(config=cfg)
    sig = build_signature(SINGLE_MEDIUM)
    assert gate.record_suppression(
        GateDecision.suppress("habituated", deciding_arm="habituation"),
        sig,
        fake_pm,
        NOW,
        ctx=LearnContext(SYNTH_SID, False, "synthetic"),
    )
    assert fake_pm.load_silence_records(limit=10)[0]["session_id"] == SYNTH_SID


def test_gate_log_session_is_none_without_a_context(fake_pm, cfg) -> None:
    # OFF/REPORT tolerate a context-less call; the record must still be
    # well-formed, and a null session reads as non-learnable (fail-closed).
    gate = Gate(config=cfg)
    sig = build_signature(SINGLE_MEDIUM)
    gate.record_delivery_success(
        sig, fake_pm, NOW, decision=GateDecision.fire("passed_all_arms"), tier=2
    )
    assert fake_pm.load_emissions(limit=10)[0]["session_id"] is None


async def test_tier1_note_threads_the_learn_context(
    fake_pm, cfg, nc, http_client, lane
) -> None:
    """A Tier-1 downgrade must carry provenance like every other delivery.

    Without it, ENFORCE makes each downstream ``@learned_write`` raise inside the
    advisor's ``_safe`` wrapper: the note still delivers, but its channel
    silently stops adapting (no habituation, no advice-rate, no channel stats).
    """
    _mint(fake_pm, REAL_SID, "real")
    set_provenance_mode(ProvenanceMode.ENFORCE)
    gate = Gate(config=cfg)
    gate.evaluate = lambda *a, **k: GateDecision.downgrade(  # type: ignore[assignment]
        "cost_tier_downgrade", deciding_arm="cost_tier_router", tier=1
    )

    await _run(
        payload={**SINGLE_MEDIUM, "session_id": REAL_SID},
        gate=gate,
        scheduler=_scheduler(),
        pm=fake_pm,
        nc=nc,
        http_client=http_client,
        config=cfg,
        lane=lane,
    )

    sig = build_signature(SINGLE_MEDIUM)
    assert fake_pm.load_habituation(sig.state_key)  # the channel still adapts
    assert fake_pm.load_emissions(limit=10)[0]["session_id"] == REAL_SID


# ── read side: runtime unfiltered, learning reads opt in ─────────────────────


def test_runtime_reads_stay_unfiltered_under_enforce() -> None:
    # The online arms must still see a synthetic burst: those advices really
    # were delivered, so they must still refract and count toward pressure.
    pm = _pm()
    _mint(pm, SYNTH_SID, "synthetic")
    pm.save_emission(_emission(SYNTH_SID))
    pm.save_silence_record(_silence(SYNTH_SID))
    set_provenance_mode(ProvenanceMode.ENFORCE)
    assert len(pm.load_emissions(limit=10)) == 1
    assert len(pm.load_silence_records(limit=10)) == 1


def test_learnable_only_excludes_non_learnable_sessions_under_enforce() -> None:
    pm = _pm()
    _mint(pm, REAL_SID, "real")
    _mint(pm, SYNTH_SID, "synthetic")
    for sid in (REAL_SID, SYNTH_SID):
        pm.save_emission(_emission(sid, decision_id=f"e-{sid}"))
        pm.save_silence_record(_silence(sid, decision_id=f"s-{sid}"))
    set_provenance_mode(ProvenanceMode.ENFORCE)

    emissions = pm.load_emissions(limit=10, learnable_only=True)
    silences = pm.load_silence_records(limit=10, learnable_only=True)
    assert [e["session_id"] for e in emissions] == [REAL_SID]
    assert [s["session_id"] for s in silences] == [REAL_SID]


@pytest.mark.parametrize("mode", [ProvenanceMode.OFF, ProvenanceMode.REPORT])
def test_learnable_only_is_inert_before_enforce(mode: ProvenanceMode) -> None:
    pm = _pm()
    _mint(pm, SYNTH_SID, "synthetic")
    pm.save_emission(_emission(SYNTH_SID))
    pm.save_silence_record(_silence(SYNTH_SID))
    set_provenance_mode(mode)
    assert len(pm.load_emissions(limit=10, learnable_only=True)) == 1
    assert len(pm.load_silence_records(limit=10, learnable_only=True)) == 1


def test_learnable_only_drops_unprovenanced_records_under_enforce() -> None:
    # Fail-closed: a record written before this stamp existed (or by a path with
    # no context in hand) has no evidence it may train, so it is excluded.
    pm = _pm()
    pm.save_emission(_emission(None))
    pm.save_silence_record(_silence(None))
    set_provenance_mode(ProvenanceMode.ENFORCE)
    assert pm.load_emissions(limit=10, learnable_only=True) == []
    assert pm.load_silence_records(limit=10, learnable_only=True) == []


def test_analyze_gate_mrt_readout_ignores_synthetic_silences() -> None:
    """A synthetic silence must not count as MRT-unobservable in the readout.

    ``get_all_feedback`` already drops the synthetic session's feedback under
    ENFORCE (CL11), so without the log filter the orphaned silence looks like a
    withheld decision whose outcome was never observed — inflating the
    unobservable rate the gate tunes against.
    """
    pm = _pm()
    _mint(pm, REAL_SID, "real")
    _mint(pm, SYNTH_SID, "synthetic")
    pm.save_silence_record(_silence(REAL_SID, decision_id="wh-real"))
    pm.save_silence_record(_silence(SYNTH_SID, decision_id="wh-synth"))
    for sid in (REAL_SID, SYNTH_SID):
        pm.save_feedback(
            sid,
            _feedback(
                sid,
                gate_decision_events=[
                    _gate_decision(decision_id=f"wh-{sid.split('-')[0]}")
                ],
            ),
        )
        pm._r.lpush("augur:responsum:_index", sid)

    set_provenance_mode(ProvenanceMode.ENFORCE)
    mrt = analyze_gate(REAL_SID, pm, CONFIG)["mrt"]
    assert mrt["withheld_n"] == 1  # only the real withheld decision
    assert mrt["unobservable_rate"] == 0.0  # the synthetic silence is not "missing"


def test_imperator_windowed_rates_ignore_synthetic_gate_activity() -> None:
    from imperator.sources import windowed_rates

    pm = _pm()
    _mint(pm, REAL_SID, "real")
    _mint(pm, SYNTH_SID, "synthetic")
    pm.save_emission(_emission(REAL_SID, decision_id="e-real"))
    for i in range(3):
        pm.save_silence_record(_silence(SYNTH_SID, decision_id=f"s-{i}"))

    set_provenance_mode(ProvenanceMode.OFF)
    assert windowed_rates(pm, 2.0, 3600.0)["suppression_rate"] == pytest.approx(0.75)

    set_provenance_mode(ProvenanceMode.ENFORCE)
    rates = windowed_rates(pm, 2.0, 3600.0)
    assert rates["suppression_rate"] == 0.0  # the synthetic burst is not a blind spot
    assert rates["advice_volume"] == {
        "delivered": 1,
        "suppressed": 0,
        "total_decisions": 1,
    }


# ── the MCP/dialogue readouts stay honest (not learning reads) ───────────────


def test_operator_readout_still_shows_every_silence_under_enforce() -> None:
    # An introspection view that hid real events would make the console lie;
    # only the learning reads opt into filtering.
    pm = _pm()
    _mint(pm, SYNTH_SID, "synthetic")
    pm.save_silence_record(_silence(SYNTH_SID))
    set_provenance_mode(ProvenanceMode.ENFORCE)

    from imperator.dialogue.context import assemble

    ctx: Any = assemble(pm, NOW, CONFIG)
    assert len(ctx.recent_suppressions) == 1


# `nc`, `http_client`, `lane`, `fake_pm`, `cfg` come from tests/conftest.py; the
# async advisor flow needs AsyncMock publishes, mirroring test_advisor_gate_flow.
@pytest.fixture
def nc() -> MagicMock:
    n = MagicMock()
    n.publish = AsyncMock()
    return n


@pytest.fixture
def http_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def lane() -> MagicMock:
    return MagicMock()
