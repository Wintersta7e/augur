"""Anti-valence debias of the typing + generic prompts (spec 1D)."""

from consilium.advisor import build_generic_prompt, build_typing_prompt

_ANOM = {
    "domain": "typing",
    "entity": "keyboard",
    "value": 1.8,
    "unit": "seconds",
    "event_type": "pause",
    "baseline_mean": 0.9,
    "deviation_score": 3.0,
    "severity": "medium",
    "context": {"avg_wpm": 60, "keypress_count": 200, "pause_position": 15},
}


def test_typing_prompt_drops_prescriptive_valence():
    p = build_typing_prompt(_ANOM, None, "SYS").lower()
    assert "take a break" not in p
    assert "stuck on a problem, distracted, or fatigued" not in p
    # anti-valence structure present
    assert "default" in p and "normal" in p
    assert "over-report" in p or "few observations" in p


def test_generic_prompt_drops_prescriptive_valence():
    p = build_generic_prompt({**_ANOM, "domain": "futuredomain"}, None, "SYS").lower()
    assert "take a break" not in p
    assert "only if" in p  # intervene-only-if-warranted framing
