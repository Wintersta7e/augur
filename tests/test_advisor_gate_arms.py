"""Unit tests for the gate suppressor arms (spec §5).

Each arm is a pure private method ``_arm_<name>(self, sig, state, config, now,
rng) -> GateDecision | None``.  These tests drive arms through the real
``Gate.evaluate`` pipeline so arm ordering + the non-exempt-HIGH bypass are
exercised together (spec §5).
"""

from __future__ import annotations

from reasoning.advisor_gate import Gate, build_signature
from tests.conftest import SINGLE_HIGH_TYPING, SINGLE_MEDIUM_TYPING


def test_central_tolerance_suppresses_medium_not_high(fake_pm, cfg):
    """Arm 1: a learned-self channel suppresses a medium; a HIGH bypasses it."""
    fake_pm.add_self_tolerance("single:typing:user")
    g = Gate()

    medium = g.evaluate(build_signature(SINGLE_MEDIUM_TYPING), fake_pm, cfg, now=1.0)
    assert medium.action == "suppress"
    assert medium.reason == "central_tolerance_learned_self"
    assert medium.deciding_arm == "central_tolerance"

    # HIGH bypass: a standalone high skips the learned/recurrence suppressors.
    high = g.evaluate(build_signature(SINGLE_HIGH_TYPING), fake_pm, cfg, now=1.0)
    assert high.action == "fire"
