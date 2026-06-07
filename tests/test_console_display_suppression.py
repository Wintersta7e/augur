"""Unit tests for console_display gate-suppression rendering (spec §8)."""

from __future__ import annotations

from output.console_display import (
    dedup_should_suppress,
    render_suppression,
    update_last_rendered,
)


def _suppressed_payload() -> dict:
    """A representative augur.advisor.suppressed payload (spec §8)."""
    return {
        "decision_id": "d1",
        "state_key": "single:typing:user",
        "domain": "typing",
        "entity": "user",
        "value": 2.0,
        "baseline_mean": 1.0,
        "severity": "medium",
        "session_id": "sess1",
        "arm": "habituation",
        "reason": "habituated",
        "mrt_eligible": False,
        "p_withhold": None,
        "timestamp": "2026-06-07T10:00:00",
    }


def test_render_suppression_shows_silent_reason():
    out = render_suppression(_suppressed_payload())
    assert "(silent: habituated)" in out
    assert "typing" in out.lower()
    assert "user" in out


def test_render_suppression_handles_missing_fields():
    # Defensive: a malformed payload must not crash the display.
    out = render_suppression({})
    assert "silent" in out.lower()


def test_suppression_dedups_via_update_last_rendered():
    """render path + update_last_rendered makes dedup fire on the primary."""
    last_rendered: dict[str, tuple[str, str]] = {}
    payload = _suppressed_payload()

    # Before rendering the suppression, the originating anomaly is not deduped.
    assert dedup_should_suppress(last_rendered, payload) is False

    # After rendering, update_last_rendered records (domain, entity, timestamp)
    # keyed on the PRIMARY anomaly so a later identical anomaly is suppressed.
    update_last_rendered(last_rendered, payload)
    assert dedup_should_suppress(last_rendered, payload) is True
