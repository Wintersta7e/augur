"""Vox renders anticipatory advice (source == "anticipatory") with a FORESEEN
badge; ordinary advice is unchanged (spec 2026-07-09 Praesagium §10)."""

from vox import console_display as V


def _ordinary_advice_payload() -> dict:
    """A representative augur.consilium.advice payload (non-anticipatory)."""
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


def _anticipatory_advice_payload() -> dict:
    """A representative anticipatory advice payload (matcher template lane,
    spec §6.1/§6.3): domain/entity come from the pattern's primary_anomaly,
    model is the fixed "anticipatory-template" sentinel, source discriminates
    the branch."""
    return {
        "domain": "praesagium",
        "entity": "a1b2c3d4e5f6",
        "advice": (
            "Forewarning: in 3 recent sessions, typing (user) was followed by "
            "activity_focus (user) within ~600s (confidence >= 42%). "
            "typing (user) was just observed."
        ),
        "value": 0.42,
        "severity": "medium",
        "model": "anticipatory-template",
        "timestamp": "2026-07-09T12:00:00+00:00",
        "latency_ms": 0.0,
        "source": "anticipatory",
        "anticipatory": {
            "pattern_id": "a1b2c3d4e5f6",
            "prediction_id": "pred1",
            "antecedent": "typing:user",
            "consequent": "activity_focus:user",
            "window_s": 600,
            "conf_lower": 0.42,
            "support_sessions": 3,
            "forewarning_text": (
                "Forewarning: in 3 recent sessions, typing (user) was followed by "
                "activity_focus (user) within ~600s (confidence >= 42%). "
                "typing (user) was just observed."
            ),
        },
    }


def test_ordinary_advice_has_no_foreseen_badge():
    out = V.render_advice(_ordinary_advice_payload())
    assert "FORESEEN" not in out
    assert "AUGUR ADVISOR" in out


def test_anticipatory_advice_has_foreseen_badge():
    out = V.render_advice(_anticipatory_advice_payload())
    assert "FORESEEN" in out
    assert "AUGUR ADVISOR" in out


def test_foreseen_badge_precedes_advisor_label():
    out = V.render_advice(_anticipatory_advice_payload())
    foreseen_idx = out.index("FORESEEN")
    advisor_idx = out.index("AUGUR ADVISOR")
    assert foreseen_idx < advisor_idx


def test_ordinary_advice_rendering_unchanged_byte_for_byte():
    """Regression pin: the non-anticipatory rendering path must be byte-identical
    to the pre-Praesagium renderer. Captured from the renderer prior to the
    FORESEEN badge change (git show HEAD:vox/console_display.py::render_advice
    on the parent commit, activity/typing branch)."""
    payload = _ordinary_advice_payload()
    out = V.render_advice(payload)

    expected_lines = [
        "",
        V.THICK_SEPARATOR,
        f"  {V.BOLD}AUGUR ADVISOR{V.RESET}  {V.severity_badge('medium')}  "
        f"{V.FG_CYAN}TYPING{V.RESET}/user",
        V.SEPARATOR,
        f"{V.FG_WHITE}"
        + "\n".join(
            f"  {line}"
            for line in __import__("textwrap")
            .fill(payload["advice"], width=V.WRAP_WIDTH - 4)
            .splitlines()
        )
        + f"{V.RESET}",
        V.SEPARATOR,
        f"  {V.FG_GRAY}Model: qwen2.5:32b  |  LLM latency: 812ms{V.RESET}",
        V.THICK_SEPARATOR,
        "",
    ]
    assert out == "\n".join(expected_lines)
