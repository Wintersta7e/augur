"""session_id rides at the TOP LEVEL of detected + foreseen payloads (spec 5.2).

Provenance enforcement needs the EVENT's session id, not "whatever session is
current". Today it survives only nested in ``primary_anomaly``, which is exactly
why the advisor falls back to reading ``augur:session:current`` (answering "what
is current now?", not "which session produced this event"). This pins the
*additive* top-level ``session_id`` on every payload builder so a downstream
consumer can read event-level provenance directly, without digging into a nested
dict whose absence is a trap for the next reader.

Purely additive — no behaviour change: an absent session id surfaces as ``None``
at the top level, and every other field is untouched. The migration that makes a
consumer actually *read* this field (and gate learning on it) is a later step.
"""

from __future__ import annotations

from consilium.advisor import _clamp_foreseen
from nexus.correlator import (
    DEFAULT_ESCALATION_MATRIX,
    _build_correlation_payload,
    _build_passthrough_payload,
)
from praesagium.matcher import build_foreseen_payload
from tabula.config import AugurConfig


def _primary(session_id: str | None = "sess-7", severity: str = "low") -> dict:
    return {
        "domain": "typing",
        "severity": severity,
        "entity": "e1",
        "timestamp": "2026-07-17T12:00:00+00:00",
        "value": 1.0,
        "session_id": session_id,
    }


def _pattern() -> dict:
    return {
        "pattern_id": "p1",
        "antecedent": "typing:user",
        "consequent": "activity:editor",
        "window_s": 120.0,
        "support_sessions": 4,
        "conf_lower": 0.62,
        "lift": 2.1,
    }


def _prediction() -> dict:
    return {"prediction_id": "pred-1", "forewarning_text": "FW"}


# -- Nexus detected payloads -------------------------------------------------


def test_correlation_payload_carries_session_id_top_level() -> None:
    payload = _build_correlation_payload(
        _primary("sess-42"),
        [_primary("sess-42")],
        DEFAULT_ESCALATION_MATRIX,
        AugurConfig.from_env(),
    )
    assert payload["session_id"] == "sess-42"
    # unchanged: it still rides nested too (no consumer moved yet)
    assert payload["primary_anomaly"]["session_id"] == "sess-42"


def test_passthrough_payload_carries_session_id_top_level() -> None:
    payload = _build_passthrough_payload(_primary("sess-9", severity="medium"))
    assert payload["session_id"] == "sess-9"
    assert payload["primary_anomaly"]["session_id"] == "sess-9"


def test_detected_payload_session_id_is_none_when_absent() -> None:
    # Additive + graceful: a primary with no session id yields a top-level None,
    # never a KeyError — the propagation must not assume the field is present.
    corr_primary = _primary()
    del corr_primary["session_id"]
    corr = _build_correlation_payload(
        corr_primary, [_primary("x")], DEFAULT_ESCALATION_MATRIX, AugurConfig.from_env()
    )

    pass_primary = _primary(severity="high")
    del pass_primary["session_id"]
    passthrough = _build_passthrough_payload(pass_primary)

    assert corr["session_id"] is None
    assert passthrough["session_id"] is None


# -- Praesagium foreseen payload ---------------------------------------------


def test_foreseen_payload_carries_session_id_top_level() -> None:
    payload = build_foreseen_payload(_pattern(), _prediction(), "sess-9")
    assert payload["session_id"] == "sess-9"
    assert payload["primary_anomaly"]["session_id"] == "sess-9"


def test_foreseen_payload_session_id_none_passes_through() -> None:
    payload = build_foreseen_payload(_pattern(), _prediction(), None)
    assert payload["session_id"] is None


def test_clamp_preserves_top_level_session_id() -> None:
    payload = build_foreseen_payload(_pattern(), _prediction(), "sess-9")
    clamped = _clamp_foreseen(payload)
    assert clamped is not None
    assert clamped["session_id"] == "sess-9"


def test_clamp_backfills_session_id_from_primary_when_top_level_missing() -> None:
    # A publisher (or spoofer) that omits the top-level field but carries it
    # nested must still leave the clamp boundary with event-level provenance.
    payload = build_foreseen_payload(_pattern(), _prediction(), "sess-3")
    del payload["session_id"]
    clamped = _clamp_foreseen(payload)
    assert clamped is not None
    assert clamped["session_id"] == "sess-3"
