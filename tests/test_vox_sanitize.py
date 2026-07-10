"""Vox render backstop: externally-sourced strings interpolated by the
telemetry/advice renderers are stripped of C0/C1/DEL control bytes and Unicode
bidirectional controls before reaching the terminal, ALWAYS-ON (independent of
conscientia_enabled -- this is render mechanism, not policy). The sanitizer
shares its character class with the Conscientia output screen via the single
public ``CONTROL_CHARS_RE`` so the two can never drift.

Two guarantees per renderer:
- a payload carrying an ANSI clear-screen escape (\\x1b[2J) and an RLO override
  (U+202E) renders output containing NEITHER sequence;
- an ordinary (control-free) payload renders byte-identically to the
  pre-backstop renderer (frozen oracle reconstructed from vox's own constants).
"""

from __future__ import annotations

import textwrap

from conscientia.screens import CONTROL_CHARS_RE
from vox import console_display as V

# The two malicious sequences seeded into every payload below.
ESC_CLEAR = "\x1b[2J"  # ANSI erase-display
RLO = "‮"  # right-to-left override (BiDi)


def _assert_clean(out: str) -> None:
    assert ESC_CLEAR not in out
    assert RLO not in out


# ── sanitize_external primitive ──────────────────────────────────────────────


def test_sanitize_external_shares_class_with_screen():
    # Exactly one definition of the character class.
    assert V.CONTROL_CHARS_RE is CONTROL_CHARS_RE


def test_sanitize_external_strips_control_and_bidi_keeps_tab_newline():
    assert V.sanitize_external("a\x1b[2Jb‮c") == "a[2Jbc"
    # \t and \n are legitimate and preserved (same class as the S2 screen).
    assert V.sanitize_external("line\tone\ntwo") == "line\tone\ntwo"


# ── render_advice ────────────────────────────────────────────────────────────


def _advice_ordinary() -> dict:
    return {
        "domain": "typing",
        "entity": "user",
        "advice": "Your typing rhythm has been unusually fast for the last few minutes.",
        "value": 3.2,
        "severity": "medium",
        "model": "qwen2.5:32b",
        "timestamp": "2026-07-09T12:00:00+00:00",
        "latency_ms": 812.4,
    }


def test_render_advice_strips_control_and_bidi():
    payload = _advice_ordinary()
    payload["advice"] = f"Rate elevated.{ESC_CLEAR}{RLO}evil"
    payload["entity"] = f"user{ESC_CLEAR}{RLO}"
    _assert_clean(V.render_advice(payload))


def test_render_advice_ordinary_byte_identical():
    payload = _advice_ordinary()
    wrapped = "\n".join(
        f"  {line}"
        for line in textwrap.fill(
            payload["advice"], width=V.WRAP_WIDTH - 4
        ).splitlines()
    )
    expected = "\n".join(
        [
            "",
            V.THICK_SEPARATOR,
            f"  {V.BOLD}AUGUR ADVISOR{V.RESET}  {V.severity_badge('medium')}  "
            f"{V.FG_CYAN}TYPING{V.RESET}/user",
            V.SEPARATOR,
            f"{V.FG_WHITE}{wrapped}{V.RESET}",
            V.SEPARATOR,
            f"  {V.FG_GRAY}Model: qwen2.5:32b  |  LLM latency: 812ms{V.RESET}",
            V.THICK_SEPARATOR,
            "",
        ]
    )
    assert V.render_advice(payload) == expected


# ── render_anomaly_line ──────────────────────────────────────────────────────


def _anomaly_ordinary() -> dict:
    return {
        "domain": "typing",
        "entity": "user",
        "severity": "low",
        "value": 18.0,
        "baseline_mean": 3.5,
        "deviation_score": 4.1,
        "context": {"avg_wpm": 52},
        "timestamp": "2026-03-17T14:29:48+00:00",
    }


def test_render_anomaly_line_strips_control_and_bidi():
    # Generic-domain fallback exercises domain + entity + value + baseline.
    payload = {
        "domain": f"weird{ESC_CLEAR}",
        "entity": f"host{RLO}",
        "severity": "low",
        "value": f"7{ESC_CLEAR}",
        "baseline_mean": f"3{RLO}",
        "deviation_score": 2.0,
        "timestamp": "2026-03-17T14:29:48+00:00",
    }
    _assert_clean(V.render_anomaly_line(payload))
    # And the typing branch (entity + wpm).
    typ = _anomaly_ordinary()
    typ["entity"] = f"user{ESC_CLEAR}{RLO}"
    typ["context"] = {"avg_wpm": f"52{ESC_CLEAR}"}
    _assert_clean(V.render_anomaly_line(typ))


def test_render_anomaly_line_ordinary_byte_identical():
    expected = (
        f"{V.DIM}14:29:48{V.RESET}  {V.severity_badge('low')}  "
        f"{V.FG_GRAY}user  wpm=52  dev=4.1σ{V.RESET}"
    )
    assert V.render_anomaly_line(_anomaly_ordinary()) == expected


# ── render_correlation ───────────────────────────────────────────────────────


def _correlation_ordinary() -> dict:
    return {
        "primary_anomaly": {
            "domain": "chess",
            "entity": "white",
            "value": 47.2,
            "baseline_mean": 8.2,
            "deviation_score": 5.7,
        },
        "correlated_events": [
            {
                "domain": "typing",
                "entity": "user",
                "value": 18.0,
                "baseline_mean": 3.5,
                "deviation_score": 4.1,
            }
        ],
        "correlation_found": True,
        "temporal_lag_seconds": 12.0,
        "combined_severity": "MEDIUM",
        "escalation_rule": "LOW+LOW→MEDIUM",
        "timestamp": "2026-03-17T14:30:00+00:00",
    }


def test_render_correlation_strips_control_and_bidi():
    payload = _correlation_ordinary()
    payload["primary_anomaly"]["domain"] = f"chess{ESC_CLEAR}"
    payload["primary_anomaly"]["entity"] = f"white{RLO}"
    payload["primary_anomaly"]["value"] = f"47{ESC_CLEAR}"
    payload["correlated_events"][0]["entity"] = f"user{RLO}"
    payload["escalation_rule"] = f"RULE{ESC_CLEAR}"
    payload["combined_severity"] = f"MEDIUM{RLO}"
    _assert_clean(V.render_correlation(payload))
    # Pass-through (standalone) branch too.
    standalone = {
        "primary_anomaly": {
            "domain": f"typing{ESC_CLEAR}",
            "entity": f"user{RLO}",
            "value": f"9{ESC_CLEAR}",
            "baseline_mean": f"1{RLO}",
            "deviation_score": "2",
        },
        "correlated_events": [],
        "correlation_found": False,
        "combined_severity": "HIGH",
    }
    _assert_clean(V.render_correlation(standalone))


def test_render_correlation_ordinary_byte_identical():
    color = V.FG_YELLOW + V.BOLD
    badge = f"{color}[MEDIUM]{V.RESET}"
    expected = "\n".join(
        [
            "",
            V.THICK_SEPARATOR,
            f"{V.BOLD}⚠ CORRELATION{V.RESET} {badge}  {V.FG_CYAN}chess + typing{V.RESET}",
            V.SEPARATOR,
            f"  {V.FG_GRAY}Temporal lag: 12.0s   rule: LOW+LOW→MEDIUM{V.RESET}",
            f"  {V.FG_WHITE}chess{V.RESET}/white: value=47.2 baseline=8.2 dev=5.7σ",
            f"  {V.FG_WHITE}typing{V.RESET}/user: value=18.0 baseline=3.5 dev=4.1σ",
            V.THICK_SEPARATOR,
            "",
        ]
    )
    assert V.render_correlation(_correlation_ordinary()) == expected


# ── render_suppression ───────────────────────────────────────────────────────


def _suppression_ordinary() -> dict:
    return {
        "domain": "typing",
        "entity": "user",
        "severity": "medium",
        "reason": "habituated",
        "timestamp": "2026-06-07T10:00:00",
    }


def test_render_suppression_strips_control_and_bidi():
    payload = _suppression_ordinary()
    payload["domain"] = f"typing{ESC_CLEAR}"
    payload["entity"] = f"user{RLO}"
    payload["reason"] = f"habituated{ESC_CLEAR}{RLO}"
    _assert_clean(V.render_suppression(payload))


def test_render_suppression_ordinary_byte_identical():
    expected = (
        f"{V.DIM}10:00:00{V.RESET}  "
        f"{V.FG_GRAY}typing/user medium  (silent: habituated){V.RESET}"
    )
    assert V.render_suppression(_suppression_ordinary()) == expected
