"""Praesagium mining wired into Disciplina's reflection cycle (Task 9).

Mirrors the run_reflection wiring harness used for the Conscientia pass in
tests/test_conscientia_reflection_hook.py: a real PersistenceManager over
fakeredis, AsyncMock() for http_client/nc, and a plain AugurConfig() for
config.

Also pins the detector-loop containment (spec 2026-07-09 §4.7): praesagium
advice events must never reach analyze_precision's or analyze_utility's
detector-tuning writes (Vigil thresholds / Ollama prompt mutation), while
leaving other domains' math byte-identical.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import fakeredis
import pytest

import disciplina.reflection_engine
from tabula.config import AugurConfig
from tabula.persistence import PersistenceManager
from disciplina.reflection_engine import run_reflection

CONFIG = AugurConfig()


def _pm() -> PersistenceManager:
    return PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=True))


def _advice(
    *,
    domain: str = "typing",
    entity: str = "user",
    severity: str = "medium",
    explicit: str = "no_response",
    behavioral: float = 0.5,
    behavioral_finalized: bool = True,
    unmeasurable: bool = False,
    correlation_found: bool = False,
    involved_domains: list[str] | None = None,
) -> dict:
    return {
        "advice_id": "adv",
        "domain": domain,
        "entity": entity,
        "severity": severity,
        "explicit_rating": explicit,
        "behavioral_score": behavioral,
        "behavioral_finalized": behavioral_finalized,
        "unmeasurable": unmeasurable,
        "outcome_metric_version": 2,
        "correlation_found": correlation_found,
        "involved_domains": involved_domains or [],
    }


def _feedback(session_id: str, *, advice_events: list[dict] | None = None) -> dict:
    return {
        "session_id": session_id,
        "advice_events": advice_events or [],
        "session_summary": {"total_advice": len(advice_events or [])},
    }


async def _run(session_id: str, feedback: dict, pm: PersistenceManager, config=CONFIG):
    return await run_reflection(
        session_id,
        feedback,
        pm,
        fakeredis.FakeStrictRedis(decode_responses=True),
        AsyncMock(),
        AsyncMock(),
        config,
    )


# ── pass 9 wiring ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_praesagium_mining_rides_the_reflection_cycle(monkeypatch) -> None:
    pm = _pm()
    feedback = _feedback("sess-praesagium")
    pm.save_feedback("sess-praesagium", feedback)

    sentinel = {
        "active": 3,
        "provisional": 1,
        "retired": 0,
        "promoted": 1,
        "reactivated": 0,
        "corpus_sessions": 5,
        "resolutions_folded": 2,
        "expired_open": 0,
    }

    def _stub(session_id, pm, config):
        return sentinel

    monkeypatch.setattr("praesagium.miner.run_praesagium_mining", _stub)

    report = await _run("sess-praesagium", feedback, pm)

    assert report["praesagium"] == sentinel


@pytest.mark.asyncio
async def test_praesagium_mining_failure_is_non_fatal(monkeypatch) -> None:
    pm = _pm()
    feedback = _feedback("sess-praesagium-fail")
    pm.save_feedback("sess-praesagium-fail", feedback)

    def _boom(session_id, pm, config):
        raise RuntimeError("mining exploded")

    monkeypatch.setattr("praesagium.miner.run_praesagium_mining", _boom)

    report = await _run("sess-praesagium-fail", feedback, pm)

    assert "mining exploded" in report["praesagium"]["error"]
    # Reflection still completed — other passes ran.
    assert "gate" in report["analyses"]
    assert "conscientia" in report
    assert report["session_id"] == "sess-praesagium-fail"


@pytest.mark.asyncio
async def test_praesagium_mining_skip_shape_passes_through(monkeypatch) -> None:
    pm = _pm()
    feedback = _feedback("sess-praesagium-skip")
    pm.save_feedback("sess-praesagium-skip", feedback)

    def _stub(session_id, pm, config):
        return {"skipped": True, "reason": "disabled"}

    monkeypatch.setattr("praesagium.miner.run_praesagium_mining", _stub)

    report = await _run("sess-praesagium-skip", feedback, pm)

    assert report["praesagium"] == {"skipped": True, "reason": "disabled"}


# ── detector-loop containment ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_praesagium_only_feedback_writes_no_thresholds(monkeypatch) -> None:
    """Praesagium-only feedback must not cause analyze_precision to persist
    Vigil thresholds for a domain with no sensor."""
    pm = _pm()
    events = [
        _advice(domain="praesagium", explicit="n", behavioral=0.1),
        _advice(domain="praesagium", explicit="n", behavioral=0.2),
        _advice(domain="praesagium", explicit="y", behavioral=0.9),
    ]
    feedback = _feedback("sess-praesagium-precision", advice_events=events)
    pm.save_feedback("sess-praesagium-precision", feedback)

    save_calls: list[tuple] = []
    orig_save = pm.save_thresholds

    def _tracking_save(domain, thresholds_dict, *, ctx=None):
        save_calls.append((domain, thresholds_dict))
        return orig_save(domain, thresholds_dict, ctx=ctx)

    monkeypatch.setattr(pm, "save_thresholds", _tracking_save)
    monkeypatch.setattr(
        "praesagium.miner.run_praesagium_mining",
        lambda session_id, pm, config: {"skipped": True, "reason": "disabled"},
    )

    report = await _run("sess-praesagium-precision", feedback, pm)

    assert "praesagium" not in report["analyses"]["precision"]["per_domain"]
    assert not any(domain == "praesagium" for domain, _ in save_calls)
    assert pm.load_thresholds("praesagium") is None


@pytest.mark.asyncio
async def test_praesagium_only_feedback_never_mutates_prompt(monkeypatch) -> None:
    """Praesagium-only feedback must not trigger prompt mutation via Ollama
    for a prompt nothing loads."""
    pm = _pm()
    # Low explicit/behavioral scores would normally trip needs_prompt_mutation.
    events = [
        _advice(domain="praesagium", explicit="n", behavioral=0.0),
        _advice(domain="praesagium", explicit="n", behavioral=0.0),
        _advice(domain="praesagium", explicit="n", behavioral=0.0),
    ]
    feedback = _feedback("sess-praesagium-utility", advice_events=events)
    pm.save_feedback("sess-praesagium-utility", feedback)

    mutate_calls: list[tuple] = []

    async def _tracking_mutate(
        pm, domain, utility_result, http_client, config, *, ctx=None
    ):
        mutate_calls.append((pm, domain, utility_result, http_client, config))
        return None

    monkeypatch.setattr("disciplina.reflection_engine.mutate_prompt", _tracking_mutate)
    monkeypatch.setattr(
        "praesagium.miner.run_praesagium_mining",
        lambda session_id, pm, config: {"skipped": True, "reason": "disabled"},
    )

    report = await _run("sess-praesagium-utility", feedback, pm)

    assert report["analyses"]["utility"]["needs_prompt_mutation"] is False
    assert not any(domain == "praesagium" for _, domain, *_ in mutate_calls)


@pytest.mark.asyncio
async def test_mixed_domain_feedback_typing_unaffected_by_filter(
    monkeypatch,
) -> None:
    """A mixed praesagium + typing feedback set must produce IDENTICAL
    typing-domain precision/utility results to a typing-only run — the
    praesagium filter must not over-filter other domains."""
    pm_typing_only = _pm()
    pm_mixed = _pm()

    typing_events = [
        _advice(domain="typing", explicit="y", behavioral=0.9),
        _advice(domain="typing", explicit="y", behavioral=0.8),
        _advice(domain="typing", explicit="n", behavioral=0.1),
    ]
    praesagium_events = [
        _advice(domain="praesagium", explicit="n", behavioral=0.0),
        _advice(domain="praesagium", explicit="n", behavioral=0.0),
    ]

    feedback_typing_only = _feedback(
        "sess-typing-only", advice_events=list(typing_events)
    )
    feedback_mixed = _feedback(
        "sess-mixed", advice_events=list(typing_events) + list(praesagium_events)
    )
    pm_typing_only.save_feedback("sess-typing-only", feedback_typing_only)
    pm_mixed.save_feedback("sess-mixed", feedback_mixed)

    def _stub(session_id, pm, config):
        return {"skipped": True, "reason": "disabled"}

    monkeypatch.setattr("praesagium.miner.run_praesagium_mining", _stub)

    report_typing_only = await _run(
        "sess-typing-only", feedback_typing_only, pm_typing_only
    )
    report_mixed = await _run("sess-mixed", feedback_mixed, pm_mixed)

    precision_typing_only = report_typing_only["analyses"]["precision"]["per_domain"][
        "typing"
    ]
    precision_mixed = report_mixed["analyses"]["precision"]["per_domain"]["typing"]
    assert precision_mixed == precision_typing_only

    utility_typing_only = report_typing_only["analyses"]["utility"]
    utility_mixed = report_mixed["analyses"]["utility"]
    assert utility_mixed == utility_typing_only

    # And praesagium never shows up in precision's per-domain results.
    assert "praesagium" not in report_mixed["analyses"]["precision"]["per_domain"]


# ── _derive_domain exclusion (majority-praesagium session) ─────────────────


@pytest.mark.asyncio
async def test_majority_praesagium_session_mutates_real_domain_not_praesagium(
    monkeypatch,
) -> None:
    """Reviewer's repro composition: praesagium events are the numeric
    majority (3) but a minority of low-scoring typing events (2) earns the
    prompt mutation. _derive_domain must exclude praesagium from candidate
    counting so the derived domain (and therefore mutate_prompt's target) is
    "typing", not "praesagium" -- otherwise a live Ollama call gets burned on
    a dead prompt key nothing ever loads while typing gets nothing."""
    from disciplina.reflection_engine import _derive_domain

    pm = _pm()
    events = [
        _advice(domain="praesagium", explicit="no_response", behavioral=0.5),
        _advice(domain="praesagium", explicit="no_response", behavioral=0.5),
        _advice(domain="praesagium", explicit="no_response", behavioral=0.5),
        _advice(domain="typing", explicit="n", behavioral=0.0),
        _advice(domain="typing", explicit="n", behavioral=0.0),
    ]
    feedback = _feedback("sess-mixed-majority-praesagium", advice_events=events)
    pm.save_feedback("sess-mixed-majority-praesagium", feedback)

    assert _derive_domain(feedback) == "typing"

    mutate_calls: list[str] = []
    orig_mutate = disciplina.reflection_engine.mutate_prompt

    async def _spy(pm_, domain, utility_result, http_client, config_, *, ctx=None):
        mutate_calls.append(domain)
        return await orig_mutate(
            pm_, domain, utility_result, http_client, config_, ctx=ctx
        )

    monkeypatch.setattr("disciplina.reflection_engine.mutate_prompt", _spy)
    monkeypatch.setattr(
        "praesagium.miner.run_praesagium_mining",
        lambda session_id, pm, config: {"skipped": True, "reason": "disabled"},
    )

    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock(return_value=None)
    fake_resp.json = MagicMock(
        return_value={"response": "A brand new mutated prompt, long enough text."}
    )
    http_client = AsyncMock()
    http_client.post = AsyncMock(return_value=fake_resp)

    report = await run_reflection(
        "sess-mixed-majority-praesagium",
        feedback,
        pm,
        fakeredis.FakeStrictRedis(decode_responses=True),
        http_client,
        AsyncMock(),
        CONFIG,
    )

    assert mutate_calls == ["typing"]
    assert "praesagium" not in mutate_calls
    assert pm.load_prompt("praesagium") is None
    assert report["session_id"] == "sess-mixed-majority-praesagium"


@pytest.mark.asyncio
async def test_all_praesagium_session_derives_default_domain() -> None:
    """An all-praesagium session has no real-domain candidates left after the
    exclusion, so _derive_domain must fall through to DEFAULT_DOMAIN rather
    than returning "praesagium"."""
    from disciplina.reflection_engine import DEFAULT_DOMAIN, _derive_domain

    feedback = _feedback(
        "sess-all-praesagium",
        advice_events=[
            _advice(domain="praesagium", explicit="n", behavioral=0.0),
            _advice(domain="praesagium", explicit="n", behavioral=0.0),
            _advice(domain="praesagium", explicit="n", behavioral=0.0),
        ],
    )

    derived = _derive_domain(feedback)

    assert derived == DEFAULT_DOMAIN
    assert derived != "praesagium"
