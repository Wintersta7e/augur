"""Decode-boundary sanitization.

Vox runs EVERY decoded NATS payload through ``sanitize_payload`` exactly ONCE,
at the ``on_*`` callback boundary (``_make_callbacks``), immediately after
``json.loads`` and before any dedup or render call. So no renderer -- current
or future -- can leak C0/C1/DEL control bytes or Unicode BiDi controls to the
terminal, even for fields a renderer interpolates raw (e.g. render_advice's
``severity``/``domain``/``model`` and the chess ``player``/``move``, or
render_proposal's LLM-generated ``rationale``).

These tests drive the REAL callbacks built by ``_make_callbacks`` -- not the
renderers directly -- so a renderer added later and wired into ``run()`` without
its own guard is still covered. The ``*_covered`` guard forces any new callback
into the parametrized set.

Note on the residue: ``CONTROL_CHARS_RE`` strips the ESC *control byte* but not
the printable ``[2J`` that follows it, so a seeded ``"\\x1b[2J"`` leaves the
harmless text ``"[2J"`` -- the dangerous ANSI *escape sequence* is broken. The
security assertion (``_assert_clean``) therefore checks the full sequence is
absent, exactly like the existing ``test_vox_sanitize`` oracle.
"""

from __future__ import annotations

import asyncio
import copy
import json

import pytest

from conscientia.screens import CONTROL_CHARS_RE
from vox import console_display as V

ESC_CLEAR = "\x1b[2J"  # ANSI erase-display
RLO = "‮"  # right-to-left override (BiDi / Trojan-Source)
E = ESC_CLEAR + RLO  # both malicious sequences, seeded into every string leaf


# ── plumbing ─────────────────────────────────────────────────────────────────


class _Msg:
    """Minimal stand-in for nats.aio.client.Msg (only ``.data`` is read)."""

    def __init__(self, payload: object) -> None:
        self.data = json.dumps(payload).encode()


def _drive(subject: str, payload: object, capsys) -> str:
    """Deliver ``payload`` to the real callback wired to ``subject``; return stdout."""
    callbacks = V._make_callbacks({}, {})
    asyncio.run(callbacks[subject](_Msg(payload)))
    return capsys.readouterr().out


def _assert_clean(out: str) -> None:
    assert ESC_CLEAR not in out
    assert RLO not in out


def _strip(s: str) -> str:
    """The exact leaf contract sanitize_payload must honour on strings."""
    return CONTROL_CHARS_RE.sub("", s)


def _seed(obj: object) -> object:
    """Append the two malicious sequences to EVERY string leaf (values only)."""
    if isinstance(obj, str):
        return obj + E
    if isinstance(obj, dict):
        return {k: _seed(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_seed(v) for v in obj]
    return obj


# ── one control-free payload per subject (byte-parity oracle inputs) ─────────

ORDINARY = {
    V.SUBJECT_ANOMALY: {
        "domain": "typing",
        "entity": "user",
        "severity": "low",
        "value": 18.0,
        "baseline_mean": 3.5,
        "deviation_score": 4.1,
        "context": {"avg_wpm": 52},
        "timestamp": "2026-03-17T14:29:48+00:00",
    },
    V.SUBJECT_ADVICE: {
        "domain": "typing",
        "entity": "user",
        "advice": "Your typing rhythm has been unusually fast.",
        "severity": "medium",
        "model": "qwen2.5:32b",
        "timestamp": "2026-07-09T12:00:00+00:00",
        "latency_ms": 812.4,
    },
    V.SUBJECT_CORRELATION: {
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
        "escalation_rule": "LOW+LOW->MEDIUM",
        "timestamp": "2026-03-17T14:30:00+00:00",
    },
    V.SUBJECT_SUPPRESSED: {
        "domain": "typing",
        "entity": "user",
        "severity": "medium",
        "reason": "habituated",
        "timestamp": "2026-06-07T10:00:00",
    },
    V.SUBJECT_CONSCIENTIA_VIOLATION: {
        "surface": "advice",
        "code": "forbidden_valence",
        "detail": "matched 'take a break'",
        "principle": "restraint",
        "domain": "typing",
        "entity": "user",
    },
    V.SUBJECT_HEALTH: {
        "faculty": "consilium",
        "reason": "consilium_stall",
        "transition": "degraded",
        "overall": "degraded",
        "ts": 1.0,
    },
    V.SUBJECT_REFLECT: {
        "session_id": "abcdef1234",
        "analyses": {},
        "adjustments": {},
    },
    V.SUBJECT_AUSPICES: {
        "salience": {"value": 0.42, "fresh": True},
        "activity": {"value": "ide", "fresh": True},
    },
    V.SUBJECT_SELF_MODEL: {
        "competence": {"value": 0.6, "fresh": True},
        "blind_spots": {
            "value": [{"kind": "low_confidence_rule", "detail": "r"}],
            "fresh": True,
        },
    },
    V.SUBJECT_PROPOSAL: {
        "klass": "safe",
        "kind": "escalation_rule",
        "target": "LOW+LOW",
        "status": "logged",
        "rationale": "improves precision",
    },
}

# The renderer each subject feeds, for the byte-parity oracle.
_RENDER = {
    V.SUBJECT_ANOMALY: V.render_anomaly_line,
    V.SUBJECT_ADVICE: V.render_advice,
    V.SUBJECT_CORRELATION: V.render_correlation,
    V.SUBJECT_SUPPRESSED: V.render_suppression,
    V.SUBJECT_CONSCIENTIA_VIOLATION: V.render_conscientia_violation,
    V.SUBJECT_HEALTH: V.render_health,
    V.SUBJECT_REFLECT: V.render_reflection,
    V.SUBJECT_AUSPICES: V.render_auspices,
    V.SUBJECT_SELF_MODEL: V.render_self_model,
    V.SUBJECT_PROPOSAL: V.render_proposal,
}

_SUBJECTS = sorted(ORDINARY)


def _malicious(subject: str) -> object:
    """``ORDINARY[subject]`` with every string leaf carrying the malicious
    sequences -- except routing enums whose value selects the renderer under
    test (an attacker controls the data fields, not which renderer fires)."""
    payload = _seed(ORDINARY[subject])
    if subject == V.SUBJECT_ANOMALY:
        # severity routes to the low-severity one-liner; keep it "low" so the
        # callback actually renders (attacker still controls entity/value/etc.).
        payload["severity"] = "low"
    return payload


# ── the structural pin: every wired callback sanitizes ───────────────────────


def test_all_callbacks_covered():
    """Every callback wired into run() must appear in these fixtures, so a new
    renderer cannot silently escape the parametrized coverage below."""
    wired = set(V._make_callbacks({}, {}))
    assert set(ORDINARY) == wired
    assert set(_RENDER) == wired


@pytest.mark.parametrize("subject", _SUBJECTS)
def test_callback_strips_control_and_bidi(subject, capsys):
    """Driving each real callback with a payload whose every string leaf carries
    an ANSI clear-screen escape and a BiDi override yields output with NEITHER."""
    out = _drive(subject, _malicious(subject), capsys)
    _assert_clean(out)
    assert out.strip(), "callback produced no output (vacuous pass guard)"


@pytest.mark.parametrize("subject", _SUBJECTS)
def test_callback_ordinary_byte_identical(subject, capsys):
    """For a control-free payload the boundary is identity: callback stdout is
    byte-for-byte the renderer's output (+ the print newline), so no ordinary
    render changed."""
    payload = ORDINARY[subject]
    got = _drive(subject, payload, capsys)
    expected = _RENDER[subject](json.loads(json.dumps(payload))) + "\n"
    assert got == expected


# ── render_advice fields that were interpolated raw ─────────────────────────


def test_render_advice_severity_model_chess_fields_sanitized(capsys):
    """severity (severity_badge .upper() fallback), model, and the chess
    player/move all reached the terminal raw before the boundary."""
    out = _drive(
        V.SUBJECT_ADVICE,
        {
            "domain": "chess",  # routing value selects the chess branch
            "severity": "sev" + E,  # unknown -> severity_badge falls back to .upper()
            "advice": "Consider developing a piece." + E,
            "model": "qwen2.5:32b" + E,
            "player": "white" + E,
            "move": "e4" + E,
            "think_time": 3.0,
            "latency_ms": 500.0,
        },
        capsys,
    )
    _assert_clean(out)
    assert out.strip()


def test_render_advice_domain_upper_sanitized(capsys):
    """The non-chess branch interpolates ``{domain.upper()}`` -- a spoofed domain
    with escape bytes must not survive there."""
    payload = _seed(ORDINARY[V.SUBJECT_ADVICE])  # domain "typing" + E -> else branch
    out = _drive(V.SUBJECT_ADVICE, payload, capsys)
    _assert_clean(out)
    assert "TYPING" in out  # domain.upper() rendered, control bytes gone


# ── render_proposal LLM-generated free text ─────────────────────────────────


def test_render_proposal_llm_rationale_sanitized(capsys):
    """rationale is set from parsed model JSON (imperator/reasoner.py) -- the LLM
    itself can emit escape bytes, no spoofing needed. target/kind likewise."""
    out = _drive(
        V.SUBJECT_PROPOSAL,
        {
            "klass": "gated" + E,
            "kind": "code_change" + E,
            "target": "consilium/advisor.py" + E,
            "status": "logged" + E,
            "rationale": "Tighten the gate;" + E + "then wipe the screen.",
        },
        capsys,
    )
    _assert_clean(out)
    assert "Tighten the gate;" in out


# ── sanitize_payload primitive ───────────────────────────────────────────────


def test_sanitize_payload_shares_regex_with_screen():
    # Exactly one definition of the character class, shared with the screen.
    assert V.CONTROL_CHARS_RE is CONTROL_CHARS_RE


def test_sanitize_payload_strips_control_keeps_printable_and_ws():
    # Only the ESC control byte + BiDi override are removed; the printable "[2J"
    # residue and \t/\n survive -- identical contract to sanitize_external.
    assert V.sanitize_payload("a\x1b[2Jb‮c") == "a[2Jbc"
    assert V.sanitize_payload("line\tone\ntwo") == "line\tone\ntwo"


def test_sanitize_payload_walks_nested_dict_and_list():
    src = {"a": "x" + E, "b": {"c": "y" + E, "d": ["z" + E, {"e": "w" + E}]}}
    assert V.sanitize_payload(src) == {
        "a": _strip("x" + E),
        "b": {"c": _strip("y" + E), "d": [_strip("z" + E), {"e": _strip("w" + E)}]},
    }


def test_sanitize_payload_sanitizes_string_keys():
    assert V.sanitize_payload({"k" + E: "v" + E}) == {_strip("k" + E): _strip("v" + E)}


def test_sanitize_payload_non_str_leaves_untouched():
    src = {"i": 0, "f": 0.0, "b": False, "t": True, "n": None, "big": 12345}
    out = V.sanitize_payload(src)
    assert out == src
    # Not coerced to "0"/"0.0"/"False"/"None": types and identities preserved.
    assert isinstance(out["i"], int) and out["i"] == 0
    assert isinstance(out["f"], float) and out["f"] == 0.0
    assert out["b"] is False
    assert out["t"] is True
    assert out["n"] is None


def test_sanitize_payload_scalar_passthrough():
    assert V.sanitize_payload(0) == 0
    assert V.sanitize_payload(False) is False
    assert V.sanitize_payload(None) is None
    assert V.sanitize_payload("a" + E) == _strip("a" + E)


def test_sanitize_payload_idempotent():
    src = {"a": "x" + E, "l": ["y" + E], "d": {"k" + E: "z" + E}}
    once = V.sanitize_payload(src)
    assert V.sanitize_payload(once) == once


def test_sanitize_payload_does_not_mutate_input():
    src = {"a": "x" + E, "l": ["y" + E], "d": {"k": "z" + E}}
    snapshot = copy.deepcopy(src)
    result = V.sanitize_payload(src)
    assert src == snapshot  # input structure unchanged
    assert result is not src  # a NEW structure is returned
    assert result["l"] is not src["l"]
    assert result["d"] is not src["d"]
