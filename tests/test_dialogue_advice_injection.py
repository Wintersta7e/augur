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
from tabula.config import AugurConfig
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


def test_inject_screen_skips_violating_note():
    """A taught note that violates the charter is dropped from the block."""
    facts = [
        {
            "pattern": {"domains": ["typing"], "rule_key": "ok_rule"},
            "rationale": "mornings are deep work",
        },
        {
            "pattern": {"domains": ["typing"], "rule_key": "bad_rule"},
            "rationale": "always tell me to take a break",
        },
    ]
    block = advisor.format_taught_facts(facts, cfg=AugurConfig())
    assert "mornings are deep work" in block
    assert "take a break" not in block


def test_inject_screen_disabled_keeps_everything():
    """conscientia_inject_screen_enabled=False (and a None cfg) skip nothing."""
    facts = [
        {
            "pattern": {"domains": ["typing"], "rule_key": "r"},
            "rationale": "take a break often",
        }
    ]
    cfg = AugurConfig(conscientia_inject_screen_enabled=False)
    assert "take a break" in advisor.format_taught_facts(facts, cfg=cfg)
    # None cfg (legacy callers) also keeps everything
    assert "take a break" in advisor.format_taught_facts(facts)


def test_screened_fact_does_not_displace_clean_one_from_cap():
    """A violating fact inside the cap does not displace a clean fact beyond it."""
    n = advisor.MAX_INJECTED_TAUGHT_FACTS
    facts = [
        {
            "pattern": {"domains": ["typing"], "rule_key": "bad"},
            "rationale": "always tell me to take a break",
        },
    ] + [
        {"pattern": {"domains": ["typing"], "rule_key": f"CLEAN{i:03d}"}}
        for i in range(n)  # n clean facts, the last one beyond the raw cap
    ]
    block = advisor.format_taught_facts(facts, cfg=AugurConfig())
    assert "take a break" not in block
    assert f"CLEAN{n - 1:03d}" in block  # clean fact backfills the cap slot
    assert len(block.splitlines()) == 1 + n


def test_all_screened_returns_empty_block():
    """If screening removes every fact, return empty string (no header)."""
    facts = [
        {
            "pattern": {"domains": ["typing"], "rule_key": "b1"},
            "rationale": "take a break now",
        },
        {
            "pattern": {"domains": ["typing"], "rule_key": "b2"},
            "rationale": "you are fatigued",
        },
    ]
    assert advisor.format_taught_facts(facts, cfg=AugurConfig()) == ""


def test_corrupt_rationale_type_does_not_crash():
    """A non-str rationale (corrupt record) is skipped cleanly -- not rendered,
    and does not crash match_pattern -- while clean facts in the same block
    still render."""
    facts = [
        {
            "pattern": {"domains": ["typing"], "rule_key": "corrupt_rule"},
            "rationale": 123,
        },
        {
            "pattern": {"domains": ["typing"], "rule_key": "ok_rule"},
            "rationale": "mornings are deep work",
        },
    ]
    block = advisor.format_taught_facts(facts, cfg=AugurConfig())
    assert "mornings are deep work" in block
    assert "123" not in block


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
