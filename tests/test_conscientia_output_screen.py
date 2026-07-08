"""Conscientia S2 — advice-output valence screen with corrective regeneration.

The recorded headline finding (R1): live advice text could carry forbidden
valence (e.g. "take a short break") because the prompt-safety guard only ever
screened prompt MUTATIONS, never the LLM's *output*. These tests drive the real
``process_message`` entry point (via the gate-flow harness) and pin the block /
retry / fail-open behavior of ``conscientia_finalize_text`` at BOTH finalize
points — the tier-2 advice path and the tier-1 note path.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

from limen.gate import Gate, GateDecision
from consilium import advisor
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


# ── Tier-2 advice path ───────────────────────────────────────────────────────


async def test_clean_advice_delivers_untouched(fake_pm, cfg) -> None:
    gate, cfg2 = _fire_gate(cfg)
    nc = _nc()
    query_ollama = AsyncMock(
        return_value=("Your typing rate rose above baseline.", 12.3)
    )
    await _run(
        payload=SINGLE_MEDIUM_TYPING,
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
    assert advice[0]["advice"] == "Your typing rate rose above baseline."
    assert "conscientia_regenerated" not in advice[0]
    # A clean screen never regenerates: exactly one LLM call.
    assert query_ollama.await_count == 1
    # No violation surface at all.
    assert fake_pm.load_conscientia_violations(limit=10) == []
    assert _published_on(nc, VIOLATION_SUBJECT) == []
    # Delivered → gate emission recorded.
    assert len(fake_pm.load_emissions(limit=10)) == 1


async def test_forbidden_advice_blocked_after_failed_retry(fake_pm, cfg) -> None:
    gate, cfg2 = _fire_gate(cfg)
    nc = _nc()
    # Forbidden both times → the single retry cannot recover → block.
    query_ollama = AsyncMock(
        side_effect=[("please take a break now", 1.0), ("still, take a break", 2.0)]
    )
    await _run(
        payload=SINGLE_MEDIUM_TYPING,
        gate=gate,
        scheduler=_scheduler(),
        pm=fake_pm,
        nc=nc,
        http_client=MagicMock(),
        config=cfg2,
        query_ollama=query_ollama,
    )

    # No advice published.
    assert _published_on(nc, advisor.PUBLISH_SUBJECT) == []
    # Exactly one violation record, surface "advice", regenerated True.
    violations = fake_pm.load_conscientia_violations(limit=10)
    assert len(violations) == 1
    assert violations[0]["surface"] == "advice"
    assert violations[0]["regenerated"] is True
    # Delivery context threaded onto the record.
    assert violations[0]["domain"] == "typing"
    assert violations[0]["entity"] == "user"
    assert violations[0]["decision_id"]  # non-empty decision linkage
    # Join axis: state_key comes from the gate Signature (single:{domain}:
    # {entity}), NOT GateDecision (which has no such field).
    assert violations[0]["state_key"] == "single:typing:user"
    # Exactly one violation event published.
    assert len(_published_on(nc, VIOLATION_SUBJECT)) == 1
    # Original + one regeneration attempt.
    assert query_ollama.await_count == 2
    # Gate delivery-success NOT recorded (spec D10) → no emission.
    assert fake_pm.load_emissions(limit=10) == []


async def test_retry_llm_failure_blocks(fake_pm, cfg) -> None:
    gate, cfg2 = _fire_gate(cfg)
    nc = _nc()
    # Forbidden first reply; the regeneration LLM call itself raises. Spec
    # failure-mode table: "Regeneration LLM call fails/times out -> degrade ->
    # skip retry, proceed to block" — it must NOT fail open to the ORIGINAL
    # forbidden text via the helper's outer except.
    query_ollama = AsyncMock(
        side_effect=[("please take a break now", 1.0), RuntimeError("ollama down")]
    )
    await _run(
        payload=SINGLE_MEDIUM_TYPING,
        gate=gate,
        scheduler=_scheduler(),
        pm=fake_pm,
        nc=nc,
        http_client=MagicMock(),
        config=cfg2,
        query_ollama=query_ollama,
    )

    # No advice published — the forbidden original must never leak out.
    assert _published_on(nc, advisor.PUBLISH_SUBJECT) == []
    # Exactly one violation record; a retry was attempted (it just failed).
    violations = fake_pm.load_conscientia_violations(limit=10)
    assert len(violations) == 1
    assert violations[0]["surface"] == "advice"
    assert violations[0]["regenerated"] is True
    # Violation publish was attempted.
    assert len(_published_on(nc, VIOLATION_SUBJECT)) == 1
    # Original call + the failed regeneration attempt — no further retries.
    assert query_ollama.await_count == 2
    # Gate delivery-success NOT recorded (spec D10) → no emission.
    assert fake_pm.load_emissions(limit=10) == []


async def test_retry_recovers_and_marks_payload(fake_pm, cfg) -> None:
    gate, cfg2 = _fire_gate(cfg)
    nc = _nc()
    # First forbidden, second clean → recovered on the single retry.
    query_ollama = AsyncMock(
        side_effect=[
            ("you should take a break", 1.0),
            ("Typing rate is elevated.", 2.0),
        ]
    )
    await _run(
        payload=SINGLE_MEDIUM_TYPING,
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
    assert advice[0]["advice"] == "Typing rate is elevated."
    assert advice[0]["conscientia_regenerated"] is True
    # No violation was recorded/published — the retry recovered.
    assert fake_pm.load_conscientia_violations(limit=10) == []
    assert _published_on(nc, VIOLATION_SUBJECT) == []
    # The retry prompt carried the corrective instruction (CORRECTIVE_SUFFIX).
    assert query_ollama.await_count == 2
    retry_prompt = query_ollama.await_args_list[1].args[0]
    assert "previous draft was refused" in retry_prompt
    # Delivered → emission recorded.
    assert len(fake_pm.load_emissions(limit=10)) == 1


async def test_screen_exception_fails_open(fake_pm, cfg, monkeypatch) -> None:
    gate, cfg2 = _fire_gate(cfg)
    nc = _nc()

    def _boom(*a, **k):
        raise RuntimeError("screen bug")

    # Invariant C3: a Conscientia bug must not silence the pipeline.
    monkeypatch.setattr(advisor.conscientia_screens, "screen_advice_text", _boom)
    # Forbidden text — but the screen itself blows up, so it must deliver anyway.
    query_ollama = AsyncMock(return_value=("please take a break", 1.0))
    await _run(
        payload=SINGLE_MEDIUM_TYPING,
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
    assert advice[0]["advice"] == "please take a break"  # unscreened, fail-open
    assert "conscientia_regenerated" not in advice[0]
    assert fake_pm.load_conscientia_violations(limit=10) == []
    assert _published_on(nc, VIOLATION_SUBJECT) == []


# ── Tier-1 note path ─────────────────────────────────────────────────────────


async def test_note_path_also_screened(fake_pm, cfg) -> None:
    # A DOWNGRADE decision routes to the templated Tier-1 note. Forbid a phrase
    # the template always emits ("no full analysis") and disable regeneration so
    # the note blocks immediately and deterministically (no LLM on this path).
    cfg2 = replace(
        cfg,
        conscientia_output_extra_patterns=("no full analysis",),
        conscientia_regenerate_on_violation=False,
    )
    gate = Gate(config=cfg2)
    d = GateDecision.downgrade(
        "cost_tier_downgrade", deciding_arm="cost_tier_router", tier=1
    )
    gate.evaluate = lambda *a, **k: d  # type: ignore[assignment]
    nc = _nc()
    query_ollama = AsyncMock(return_value=("clean", 1.0))
    await _run(
        payload=SINGLE_MEDIUM_TYPING,
        gate=gate,
        scheduler=_scheduler(),
        pm=fake_pm,
        nc=nc,
        http_client=MagicMock(),
        config=cfg2,
        query_ollama=query_ollama,
    )

    # The note is NOT published.
    assert _published_on(nc, advisor.PUBLISH_SUBJECT) == []
    # It is screened by the same shared surface → one "advice" violation record.
    violations = fake_pm.load_conscientia_violations(limit=10)
    assert len(violations) == 1
    assert violations[0]["surface"] == "advice"
    assert violations[0]["regenerated"] is False  # regeneration disabled
    assert len(_published_on(nc, VIOLATION_SUBJECT)) == 1
    # Regeneration off → the note path never touched the LLM.
    assert query_ollama.await_count == 0
    # Gate delivery-success NOT recorded (spec D10) → no tier-1 emission.
    assert fake_pm.load_emissions(limit=10) == []


async def test_note_path_violation_carries_state_key(fake_pm, cfg) -> None:
    # Finding-2 coverage for the SECOND conscientia_finalize_text call site
    # (the tier-1 note path, ``_publish_tier1_note``) — state_key must come
    # from the same gate Signature there too, not just on the tier-2 path.
    cfg2 = replace(
        cfg,
        conscientia_output_extra_patterns=("no full analysis",),
        conscientia_regenerate_on_violation=False,
    )
    gate = Gate(config=cfg2)
    d = GateDecision.downgrade(
        "cost_tier_downgrade", deciding_arm="cost_tier_router", tier=1
    )
    gate.evaluate = lambda *a, **k: d  # type: ignore[assignment]
    nc = _nc()
    query_ollama = AsyncMock(return_value=("clean", 1.0))
    await _run(
        payload=SINGLE_MEDIUM_TYPING,
        gate=gate,
        scheduler=_scheduler(),
        pm=fake_pm,
        nc=nc,
        http_client=MagicMock(),
        config=cfg2,
        query_ollama=query_ollama,
    )

    violations = fake_pm.load_conscientia_violations(limit=10)
    assert len(violations) == 1
    assert violations[0]["state_key"] == "single:typing:user"
