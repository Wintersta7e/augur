"""Typed teaching-intent taxonomy + confirmation matchers."""

from __future__ import annotations

import math

INTENT_KINDS = frozenset(
    {
        "query",
        "teach_context_directive",
        "teach_semantic_fact",
        "correct_silence",
        "correct_noise",
        "correct_advice_quality",
        "tune_rule",
        "undo",
    }
)
_REQUIRE_TARGET = INTENT_KINDS - {"query", "undo"}
_AFFIRMATIVES = {"yes", "y", "ok", "okay", "do it", "confirm", "apply", "yep", "sure"}

# Numeric bounds mirror Disciplina's own tuning writer — the sole autonomous
# tuner of these values. Sigma: clamped into [config.sigma_min, config.sigma_max]
# (disciplina/reflection_engine.py raise/lower clamps; defaults 1.5 / 5.0 in
# tabula/config.py). Habituation floor: swept into [0.0, GATE_FLOOR_MAX=0.6]
# (disciplina/reflection_engine.py). Non-finite values are rejected outright:
# a NaN sigma makes vigil's `deviation >= sigma_threshold` comparison silently
# always-False (disabling detection for a domain), and a floor > 1.0 produces
# a negative habituation cap downstream.
#
# Layer divergence: validate_intent has no cfg parameter (brief-fixed
# signature), so these are the compiled defaults, while apply.py reads the
# env-overridable cfg.sigma_min/sigma_max. If AUGUR_SIGMA_MIN/MAX diverge from
# the defaults, this layer may accept intents the apply layer then rejects —
# fails closed, never unsafe.
_SIGMA_MIN, _SIGMA_MAX = 1.5, 5.0
_FLOOR_MIN, _FLOOR_MAX = 0.0, 0.6


def _validated_number(value: object, name: str, lo: float, hi: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"intent action {name} must be a number")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"intent action {name} must be finite")
    if not (lo <= v <= hi):
        raise ValueError(f"intent action {name} must be within [{lo}, {hi}], got {v}")
    return v


def validate_intent(intent: dict) -> dict:
    if not isinstance(intent, dict):
        raise ValueError("intent must be an object")
    kind = intent.get("kind")
    if kind not in INTENT_KINDS:
        raise ValueError(f"unknown intent kind: {kind!r}")
    if kind in _REQUIRE_TARGET and not intent.get("target"):
        raise ValueError(f"intent {kind} requires a target")
    action = intent.get("action") or {}
    if not isinstance(action, dict):
        raise ValueError("intent.action must be an object")
    action = dict(action)  # normalized copy; never mutate the caller's dict
    if "sigma" in action:
        action["sigma"] = _validated_number(
            action["sigma"], "sigma", _SIGMA_MIN, _SIGMA_MAX
        )
    if action.get("op") == "floor_set":
        action["value"] = _validated_number(
            action.get("value"), "floor value", _FLOOR_MIN, _FLOOR_MAX
        )
    return {
        "kind": kind,
        "target": intent.get("target"),
        "action": action,
        "rationale": str(intent.get("rationale", ""))[:280],
    }


def is_affirmative(text: str) -> bool:
    return text.strip().lower().rstrip(".!") in _AFFIRMATIVES


def matches_heavy_phrase(text: str, phrase: str) -> bool:
    return phrase.lower() in text.lower()
