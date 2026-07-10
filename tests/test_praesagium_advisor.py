"""Consilium's anticipatory delivery lane for Praesagium forewarnings.

Task 8 of the Praesagium build (spec ``2026-07-09-praesagium-design.md`` §6.3,
invariant PR1b). These drive the real ``process_message`` / ``_clamp_foreseen``
/ ``conscientia_finalize_text`` entry points with the LLM (``query_ollama``) and
NATS (``nc.publish``) mocked — mirroring the harness idioms of
``tests/test_conscientia_output_screen.py``.

The load-bearing guarantees pinned here:

- a well-formed foreseen payload is delivered from a **deterministic template**
  (no Ollama call, ``model == "anticipatory-template"``, ``latency_ms == 0.0``);
- PR1b: ``_clamp_foreseen`` force-clamps every exemption-granting field before
  ``build_signature`` ever sees it, and DROPS malformed/spoofed envelopes so
  they never reach the gate;
- the anticipatory S2 screen runs with regeneration DISABLED — a violating
  template blocks straight away (no LLM), the ``allow_regeneration`` default
  stays ``True`` for every legacy caller;
- tier-1 downgrade wording states the prediction ("Forewarning withheld"), not
  an observation, and suppression tolerates the None baseline fields.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from limen.gate import Gate, GateDecision, build_signature
from consilium import advisor
from praesagium.matcher import build_foreseen_payload, render_forewarning
from tests.conftest import SINGLE_MEDIUM_TYPING
from tests.test_advisor_gate_flow import _published_on, _run, _scheduler

VIOLATION_SUBJECT = "augur.conscientia.violation"


def _nc() -> MagicMock:
    n = MagicMock()
    n.publish = AsyncMock()
    return n


def _fire_gate(cfg):
    """A gate that ordinary-fires at tier 2 (cost-tier downgrade disabled)."""
    cfg2 = replace(cfg, gate_cost_tier_enabled=False)
    return Gate(arms=[], config=cfg2), cfg2


def _raising_ollama() -> AsyncMock:
    """A query_ollama stub that RAISES if awaited — the anticipatory lane and
    the disabled-regeneration S2 block must never touch the LLM."""
    return AsyncMock(side_effect=AssertionError("Ollama must not be called"))


def _pattern(**over: Any) -> dict:
    p = {
        "pattern_id": "abc123def456",
        "antecedent": "typing:user",
        "consequent": "activity_intensity:app",
        "window_s": 120.0,
        "support_sessions": 3,
        "conf_lower": 0.44,
        "lift": 1.8,
    }
    p.update(over)
    return p


def _foreseen(pattern: dict | None = None, session_id: str | None = "sess-1") -> dict:
    """A real, correctly-shaped foreseen envelope via the matcher's builder."""
    pattern = pattern or _pattern()
    prediction = {
        "prediction_id": "pred-1",
        "forewarning_text": render_forewarning(pattern),
    }
    return build_foreseen_payload(pattern, prediction, session_id)


async def _drive_foreseen(
    payload: dict,
    *,
    gate: Gate,
    pm: Any,
    nc: MagicMock,
    config: Any,
    query_ollama: Any,
) -> None:
    """Mirror ``advisor.run()``'s ``on_foreseen``: clamp → drop | process_message.

    Kept in lockstep with the real closure (decode is elided — payload is
    already a dict): the clamp is the security boundary, so a dropped envelope
    returns BEFORE ``process_message`` and the gate is never consulted.
    """
    clamped = advisor._clamp_foreseen(payload)
    if clamped is None:
        return
    await advisor.process_message(
        payload=clamped,
        gate=gate,
        scheduler=_scheduler(),
        pm=pm,
        nc=nc,
        http_client=MagicMock(),
        redis_client=None,
        classifier_lane=MagicMock(),
        config=config,
        now=1000.0,
        query_ollama=query_ollama,
    )


# ── Happy path: LLM-free template delivery ────────────────────────────────────


async def test_anticipatory_template_delivers(fake_pm, cfg) -> None:
    gate, cfg2 = _fire_gate(cfg)
    nc = _nc()
    pattern = _pattern()
    payload = _foreseen(pattern)
    expected = render_forewarning(pattern)
    query_ollama = _raising_ollama()
    await _run(
        payload=payload,
        gate=gate,
        scheduler=_scheduler(),
        pm=fake_pm,
        nc=nc,
        http_client=MagicMock(),
        config=cfg2,
        query_ollama=query_ollama,
    )

    advice = _published_on(nc, advisor.PUBLISH_SUBJECT)
    assert len(advice) == 1
    ev = advice[0]
    # Template text verbatim — no LLM rephrasing.
    assert ev["advice"] == expected
    assert ev["model"] == "anticipatory-template"
    assert ev["latency_ms"] == 0.0
    # Provenance: domain is the durable join key; source/anticipatory are extras.
    assert ev["domain"] == "praesagium"
    assert ev["entity"] == pattern["pattern_id"]
    assert ev["source"] == "anticipatory"
    assert ev["anticipatory"]["pattern_id"] == pattern["pattern_id"]
    assert ev["anticipatory"]["forewarning_text"] == expected
    # Ollama never touched; a clean screen never regenerates.
    assert query_ollama.await_count == 0
    assert "conscientia_regenerated" not in ev
    # Delivered → gate emission recorded.
    assert len(fake_pm.load_emissions(limit=10)) == 1
    # No violation surface.
    assert fake_pm.load_conscientia_violations(limit=10) == []


# ── PR1b: validate-and-clamp at the boundary ─────────────────────────────────


def test_clamp_forces_never_exempt() -> None:
    payload = _foreseen()
    # Forge every exemption-granting field.
    payload["correlation_found"] = True
    payload["combined_severity"] = "HIGH"
    payload["correlated_events"] = [{"domain": "typing"}]
    payload["involved_domains"] = ["typing", "chess"]
    payload["primary_anomaly"]["severity"] = "high"

    clamped = advisor._clamp_foreseen(payload)
    assert clamped is not None
    # The clamp neutralized the forgery in the returned payload.
    assert clamped["correlation_found"] is False
    assert clamped["correlated_events"] == []
    assert clamped["combined_severity"] == "MEDIUM"
    assert clamped["involved_domains"] == ["praesagium"]
    assert clamped["primary_anomaly"]["severity"] == "medium"
    # build_signature over the clamped payload can never be exempt/high.
    sig = build_signature(clamped)
    assert sig.exempt is False
    assert sig.severity == "medium"
    assert sig.path == "single"
    assert sig.state_key == "single:praesagium:abc123def456"
    assert sig.ungateable is False


def _drop_missing_forewarning(p: dict) -> None:
    p["anticipatory"].pop("forewarning_text")


def _drop_empty_forewarning(p: dict) -> None:
    p["anticipatory"]["forewarning_text"] = ""


def _drop_nonstr_forewarning(p: dict) -> None:
    p["anticipatory"]["forewarning_text"] = 123


def _drop_missing_pattern_id(p: dict) -> None:
    p["anticipatory"].pop("pattern_id")


def _drop_empty_pattern_id(p: dict) -> None:
    p["anticipatory"]["pattern_id"] = ""


def _drop_wrong_domain(p: dict) -> None:
    p["primary_anomaly"]["domain"] = "typing"


def _drop_nondict_anticipatory(p: dict) -> None:
    p["anticipatory"] = "not-a-dict"


def _drop_missing_anticipatory(p: dict) -> None:
    p.pop("anticipatory")


def _drop_empty_entity(p: dict) -> None:
    p["primary_anomaly"]["entity"] = ""


def _drop_nondict_primary(p: dict) -> None:
    p["primary_anomaly"] = None


def _drop_wrong_source(p: dict) -> None:
    p["source"] = "detected"


@pytest.mark.parametrize(
    "mutate",
    [
        _drop_missing_forewarning,
        _drop_empty_forewarning,
        _drop_nonstr_forewarning,
        _drop_missing_pattern_id,
        _drop_empty_pattern_id,
        _drop_wrong_domain,
        _drop_nondict_anticipatory,
        _drop_missing_anticipatory,
        _drop_empty_entity,
        _drop_nondict_primary,
        _drop_wrong_source,
    ],
)
def test_clamp_drops_malformed(mutate) -> None:
    payload = _foreseen()
    mutate(payload)
    assert advisor._clamp_foreseen(payload) is None


def test_clamp_drops_non_dict_payload() -> None:
    assert advisor._clamp_foreseen("nope") is None  # type: ignore[arg-type]
    assert advisor._clamp_foreseen(None) is None  # type: ignore[arg-type]
    assert advisor._clamp_foreseen([]) is None  # type: ignore[arg-type]


async def test_dropped_envelope_never_reaches_gate(fake_pm, cfg) -> None:
    gate, cfg2 = _fire_gate(cfg)
    spy = MagicMock(side_effect=AssertionError("gate.evaluate must not run"))
    gate.evaluate = spy  # type: ignore[assignment]
    nc = _nc()
    bad = _foreseen()
    _drop_missing_forewarning(bad)
    await _drive_foreseen(
        bad, gate=gate, pm=fake_pm, nc=nc, config=cfg2, query_ollama=_raising_ollama()
    )
    # Dropped before the gate: no evaluate, no publish of any kind.
    assert spy.call_count == 0
    assert nc.publish.await_count == 0


async def test_valid_envelope_reaches_gate_and_delivers(fake_pm, cfg) -> None:
    gate, cfg2 = _fire_gate(cfg)
    orig = gate.evaluate
    spy = MagicMock(side_effect=orig)
    gate.evaluate = spy  # type: ignore[assignment]
    nc = _nc()
    await _drive_foreseen(
        _foreseen(),
        gate=gate,
        pm=fake_pm,
        nc=nc,
        config=cfg2,
        query_ollama=_raising_ollama(),
    )
    # Positive control: the clamp passed a valid envelope through to the gate.
    assert spy.call_count == 1
    assert len(_published_on(nc, advisor.PUBLISH_SUBJECT)) == 1


# ── S2 block: regeneration disabled in the anticipatory lane ──────────────────


async def test_s2_block_no_regeneration(fake_pm, cfg) -> None:
    # An extra output pattern matching the template forces a violation; the
    # anticipatory lane passes allow_regeneration=False, so it blocks with no
    # LLM call (D10 semantics: violation recorded + published, no delivery).
    cfg2 = replace(
        cfg,
        gate_cost_tier_enabled=False,
        conscientia_output_extra_patterns=("forewarning",),
    )
    gate = Gate(arms=[], config=cfg2)
    nc = _nc()
    query_ollama = _raising_ollama()
    await _run(
        payload=_foreseen(),
        gate=gate,
        scheduler=_scheduler(),
        pm=fake_pm,
        nc=nc,
        http_client=MagicMock(),
        config=cfg2,
        query_ollama=query_ollama,
    )

    # No advice delivered.
    assert _published_on(nc, advisor.PUBLISH_SUBJECT) == []
    # One "advice"-surface violation, NOT regenerated, carrying praesagium domain.
    violations = fake_pm.load_conscientia_violations(limit=10)
    assert len(violations) == 1
    assert violations[0]["surface"] == "advice"
    assert violations[0]["regenerated"] is False
    assert violations[0]["domain"] == "praesagium"
    assert violations[0]["entity"] == "abc123def456"
    assert violations[0]["state_key"] == "single:praesagium:abc123def456"
    assert len(_published_on(nc, VIOLATION_SUBJECT)) == 1
    # Regeneration disabled → the LLM was never called.
    assert query_ollama.await_count == 0
    # Blocked (D10) → no gate emission.
    assert fake_pm.load_emissions(limit=10) == []


# ── allow_regeneration keyword: default unchanged, False forces retries=0 ──────


async def test_finalize_default_allows_regeneration(fake_pm, cfg) -> None:
    # Default (allow_regeneration=True) behaves EXACTLY as before: a forbidden
    # draft triggers one corrective regeneration (regenerate_max=1), which the
    # single query_ollama call here supplies as clean recovered text.
    nc = _nc()
    query_ollama = AsyncMock(return_value=("Rate is elevated.", 2.0))
    text, regenerated = await advisor.conscientia_finalize_text(
        "please take a break",
        "PROMPT",
        query_ollama=query_ollama,
        http_client=MagicMock(),
        config=cfg,
        pm=fake_pm,
        nc=nc,
        decision=GateDecision.fire("x"),
        domain="typing",
        entity="user",
        session_id=None,
        state_key="single:typing:user",
    )
    assert text == "Rate is elevated."
    assert regenerated is True
    assert query_ollama.await_count == 1
    assert fake_pm.load_conscientia_violations(limit=10) == []


async def test_finalize_no_regeneration_blocks_immediately(fake_pm, cfg) -> None:
    # allow_regeneration=False forces retries=0: a forbidden draft blocks with
    # no LLM call.
    nc = _nc()
    query_ollama = _raising_ollama()
    text, regenerated = await advisor.conscientia_finalize_text(
        "please take a break",
        "PROMPT",
        query_ollama=query_ollama,
        http_client=MagicMock(),
        config=cfg,
        pm=fake_pm,
        nc=nc,
        decision=GateDecision.fire("x"),
        domain="praesagium",
        entity="abc123def456",
        session_id=None,
        state_key="single:praesagium:abc123def456",
        allow_regeneration=False,
    )
    assert text is None  # blocked
    assert regenerated is False
    assert query_ollama.await_count == 0
    violations = fake_pm.load_conscientia_violations(limit=10)
    assert len(violations) == 1
    assert violations[0]["surface"] == "advice"
    assert violations[0]["regenerated"] is False


# ── Tier-1 downgrade wording ──────────────────────────────────────────────────


def _downgrade_gate(cfg):
    gate = Gate(config=cfg)
    d = GateDecision.downgrade(
        "cost_tier_downgrade", deciding_arm="cost_tier_router", tier=1
    )
    gate.evaluate = lambda *a, **k: d  # type: ignore[assignment]
    return gate


async def test_tier1_anticipatory_wording(fake_pm, cfg) -> None:
    gate = _downgrade_gate(cfg)
    nc = _nc()
    pattern = _pattern()
    query_ollama = _raising_ollama()
    await _run(
        payload=_foreseen(pattern),
        gate=gate,
        scheduler=_scheduler(),
        pm=fake_pm,
        nc=nc,
        http_client=MagicMock(),
        config=cfg,
        query_ollama=query_ollama,
    )

    notes = _published_on(nc, advisor.PUBLISH_SUBJECT)
    assert len(notes) == 1
    note = notes[0]
    assert note["tier"] == 1
    # Prediction wording, not observation.
    assert "Forewarning withheld" in note["advice"]
    assert "was observed" not in note["advice"]
    # Humanized antecedent/consequent + pattern id.
    assert "typing (user)" in note["advice"]
    assert "activity_intensity (app)" in note["advice"]
    assert pattern["pattern_id"] in note["advice"]
    # Clean note → no LLM, delivered → gate emission recorded.
    assert query_ollama.await_count == 0
    assert len(fake_pm.load_emissions(limit=10)) == 1


async def test_tier1_non_anticipatory_wording_unchanged(fake_pm, cfg) -> None:
    gate = _downgrade_gate(cfg)
    nc = _nc()
    await _run(
        payload=SINGLE_MEDIUM_TYPING,
        gate=gate,
        scheduler=_scheduler(),
        pm=fake_pm,
        nc=nc,
        http_client=MagicMock(),
        config=cfg,
        query_ollama=AsyncMock(return_value=("clean", 1.0)),
    )
    notes = _published_on(nc, advisor.PUBLISH_SUBJECT)
    assert len(notes) == 1
    assert "was observed" in notes[0]["advice"]
    assert "Forewarning withheld" not in notes[0]["advice"]


# ── Suppression: None baselines tolerated ─────────────────────────────────────


async def test_suppression_none_baselines(fake_pm, cfg) -> None:
    gate = Gate(config=cfg)
    d = GateDecision.suppress(
        "habituation", deciding_arm="habituation", mrt_eligible=True, p_withhold=0.9
    )
    gate.evaluate = lambda *a, **k: d  # type: ignore[assignment]
    nc = _nc()
    query_ollama = _raising_ollama()
    await _run(
        payload=_foreseen(),
        gate=gate,
        scheduler=_scheduler(),
        pm=fake_pm,
        nc=nc,
        http_client=MagicMock(),
        config=cfg,
        query_ollama=query_ollama,
    )

    # No advice; exactly one suppressed event.
    assert _published_on(nc, advisor.PUBLISH_SUBJECT) == []
    suppressed = _published_on(nc, advisor.SUBJECT_SUPPRESSED)
    assert len(suppressed) == 1
    s = suppressed[0]
    assert s["domain"] == "praesagium"
    assert s["state_key"] == "single:praesagium:abc123def456"
    # The None baseline fields pass through untouched (no arithmetic).
    assert s["baseline_mean"] is None
    assert s["baseline_std"] is None
    assert s["deviation_score"] is None
    assert s["baseline_observation_count"] is None
    assert query_ollama.await_count == 0
