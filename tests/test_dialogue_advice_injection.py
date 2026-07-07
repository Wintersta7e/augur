"""Tests for taught-fact injection into advice prompts.

Covers three layers:
  1. ``format_taught_facts`` unit behavior (block shape, empty input,
     explicit-None domains, fact-count cap, char budget).
  2. End-to-end wiring through ``process_message`` — a taught fact seeded in
     the PersistenceManager must appear in the prompt captured by the mocked
     ``query_ollama`` for BOTH the standalone and the correlation branch.
  3. Negative wiring — a fact taught for an unrelated domain must NOT leak
     into the prompt.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

from limen.gate import Gate
from consilium import advisor
from tests.conftest import CORRELATION_MEDIUM, SINGLE_MEDIUM_TYPING
from tests.test_advisor_gate_flow import _run, _scheduler


# ── format_taught_facts unit behavior ─────────────────────────────────────────


def test_format_taught_facts_block():
    """Facts format into a 'Known facts' block listing domains + rationale."""
    facts = [
        {
            "pattern": {"domains": ["chess", "typing"], "rule_key": "HIGH+HIGH"},
            "taught_by": "user",
            "rationale": "stress",
        }
    ]
    block = advisor.format_taught_facts(facts)
    assert "chess" in block and "typing" in block
    assert block.startswith("Known facts")


def test_empty_facts_is_empty():
    """Empty facts list returns empty string."""
    assert advisor.format_taught_facts([]) == ""


def test_none_domains_does_not_crash():
    """A fact whose pattern.domains is explicitly None formats without error."""
    facts = [{"pattern": {"domains": None, "rule_key": "RK"}, "rationale": "note"}]
    block = advisor.format_taught_facts(facts)
    assert block.startswith("Known facts")
    assert "note" in block


def test_fact_count_capped():
    """Only the first MAX_INJECTED_TAUGHT_FACTS facts are formatted."""
    n = advisor.MAX_INJECTED_TAUGHT_FACTS
    facts = [
        {"pattern": {"domains": ["typing"], "rule_key": f"RK{i:03d}"}}
        for i in range(n + 1)
    ]
    block = advisor.format_taught_facts(facts)
    assert f"RK{n - 1:03d}" in block  # last fact within the cap
    assert f"RK{n:03d}" not in block  # first fact beyond the cap
    # Header + exactly N fact lines.
    assert len(block.splitlines()) == 1 + n


def test_block_char_budget_enforced():
    """The formatted block is hard-truncated to TAUGHT_FACTS_CHAR_BUDGET."""
    facts = [
        {
            "pattern": {"domains": ["typing"], "rule_key": "RK"},
            "rationale": "x" * (advisor.TAUGHT_FACTS_CHAR_BUDGET * 2),
        }
    ]
    block = advisor.format_taught_facts(facts)
    assert len(block) <= advisor.TAUGHT_FACTS_CHAR_BUDGET


# ── End-to-end wiring through process_message ─────────────────────────────────
#
# Reuses the gate-flow harness: process_message driven with a mocked
# query_ollama (AsyncMock) and a real PersistenceManager on fakeredis
# (fake_pm), so the prompt the LLM would receive is captured verbatim.


def _seed_fact(
    pm, domains: list[str], rule_key: str, rationale: str | None = None
) -> None:
    pm.create_user_taught_memory(
        {
            "kind": "semantic",
            "domains": domains,
            "rule_key": rule_key,
            "severity": "LOW",
        },
        source="user",
        rationale=rationale,
    )


def _nc() -> MagicMock:
    n = MagicMock()
    n.publish = AsyncMock()
    return n


async def _run_and_capture_prompt(payload, fake_pm, cfg) -> str:
    cfg2 = replace(cfg, gate_cost_tier_enabled=False)
    gate = Gate(arms=[], config=cfg2)  # passes all arms → fire
    query_ollama = AsyncMock(return_value=("advice text", 12.3))
    await _run(
        payload=payload,
        gate=gate,
        scheduler=_scheduler(),
        pm=fake_pm,
        nc=_nc(),
        http_client=MagicMock(),
        config=cfg2,
        query_ollama=query_ollama,
    )
    assert query_ollama.await_count == 1, "advice LLM was not called"
    return query_ollama.await_args.args[0]


async def test_taught_fact_injected_into_standalone_prompt(fake_pm, cfg) -> None:
    _seed_fact(fake_pm, ["typing"], "TYPING_FACT_MARKER")
    prompt = await _run_and_capture_prompt(SINGLE_MEDIUM_TYPING, fake_pm, cfg)
    assert "Known facts (taught by the user):" in prompt
    assert "TYPING_FACT_MARKER" in prompt


async def test_taught_rationale_reaches_the_prompt(fake_pm, cfg) -> None:
    # The user's own teaching text — not just the rule_key slug — must reach
    # the advice prompt when it was provided at teach time (real writer path,
    # not a synthetic record).
    _seed_fact(
        fake_pm,
        ["typing"],
        "morning_deep_work",
        rationale="deep work happens in the mornings; long pauses are thought",
    )
    prompt = await _run_and_capture_prompt(SINGLE_MEDIUM_TYPING, fake_pm, cfg)
    assert "Known facts (taught by the user):" in prompt
    assert "deep work happens in the mornings; long pauses are thought" in prompt


async def test_taught_fact_injected_into_correlation_prompt(fake_pm, cfg) -> None:
    # CORRELATION_MEDIUM involves typing+chess; a chess fact must be injected.
    _seed_fact(fake_pm, ["chess"], "CHESS_FACT_MARKER")
    prompt = await _run_and_capture_prompt(CORRELATION_MEDIUM, fake_pm, cfg)
    assert "Known facts (taught by the user):" in prompt
    assert "CHESS_FACT_MARKER" in prompt


async def test_unrelated_domain_fact_not_injected(fake_pm, cfg) -> None:
    # A chess-only fact must NOT leak into a standalone typing prompt.
    _seed_fact(fake_pm, ["chess"], "CHESS_ONLY_MARKER")
    prompt = await _run_and_capture_prompt(SINGLE_MEDIUM_TYPING, fake_pm, cfg)
    assert "CHESS_ONLY_MARKER" not in prompt
    assert "Known facts" not in prompt
