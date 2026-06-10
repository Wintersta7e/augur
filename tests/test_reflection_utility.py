"""Tests for analyze_utility — prompt mutation decision.

This function computes the weighted utility score (60% explicit, 40% behavioral)
that determines whether Augur should rewrite its own LLM prompt. A wrong score
means either unnecessary prompt mutations (destroying a good prompt) or failure
to mutate a bad one.
"""

from __future__ import annotations

import pytest

from tabula.config import AugurConfig
from disciplina.reflection_engine import analyze_utility

_CFG = AugurConfig()
UTILITY_MUTATION_THRESHOLD = _CFG.utility_mutation_threshold


def _make_feedback(
    ratings: list[str],
    behavioral_scores: list[float] | None = None,
) -> dict:
    events = []
    for i, rating in enumerate(ratings):
        # COV-06: explicitly set correlation_found=False on every event.
        # The analyze_utility filter treats missing-key as falsy, but
        # being explicit protects this fixture against a future tightening
        # of the filter (e.g., to `e.get("correlation_found") is False`,
        # which would classify missing-key events as "unknown" and possibly
        # change their contribution).
        ev = {"explicit_rating": rating, "correlation_found": False}
        if behavioral_scores and i < len(behavioral_scores):
            ev["behavioral_score"] = behavioral_scores[i]
            # A provided behavioral score is a finalized, measurable outcome
            # (real records carry these flags; the new filter keys on them).
            ev["behavioral_finalized"] = True
            ev["unmeasurable"] = False
        events.append(ev)
    return {
        "advice_events": events,
        "session_summary": {"total_advice": len(events)},
    }


class TestUtilityEmptySession:
    def test_no_events_returns_perfect_score(self) -> None:
        result = analyze_utility(
            {"advice_events": [], "session_summary": {"total_advice": 0}},
            _CFG,
        )
        assert result["utility_score"] == 1.0
        assert result["needs_prompt_mutation"] is False


class TestExplicitScoring:
    """Verify the explicit component: y=1.0, n=0.0, no_response=0.5."""

    def test_all_positive(self) -> None:
        feedback = _make_feedback(["y", "y"])
        result = analyze_utility(feedback, _CFG)
        assert result["explicit_component"] == pytest.approx(1.0)

    def test_all_negative(self) -> None:
        feedback = _make_feedback(["n", "n"])
        result = analyze_utility(feedback, _CFG)
        assert result["explicit_component"] == pytest.approx(0.0)

    def test_all_no_response(self) -> None:
        feedback = _make_feedback(["no_response", "no_response"])
        result = analyze_utility(feedback, _CFG)
        assert result["explicit_component"] == pytest.approx(0.5)

    def test_mixed(self) -> None:
        feedback = _make_feedback(["y", "n"])
        result = analyze_utility(feedback, _CFG)
        assert result["explicit_component"] == pytest.approx(0.5)


class TestWeightedCombination:
    """Verify the 60/40 weighting between explicit and behavioral."""

    def test_all_positive_all_high_behavioral(self) -> None:
        feedback = _make_feedback(["y", "y"], [0.9, 0.9])
        result = analyze_utility(feedback, _CFG)
        # explicit=1.0, behavioral=0.9 -> 0.6*1.0 + 0.4*0.9 = 0.96
        assert result["utility_score"] == pytest.approx(0.96, abs=0.01)

    def test_all_negative_zero_behavioral(self) -> None:
        feedback = _make_feedback(["n", "n"], [0.1, 0.1])
        result = analyze_utility(feedback, _CFG)
        # explicit=0.0, behavioral=0.1 -> 0.6*0.0 + 0.4*0.1 = 0.04
        assert result["utility_score"] == pytest.approx(0.04, abs=0.01)

    def test_no_behavioral_defaults_to_half(self) -> None:
        # Events with no behavioral_score (or 0) get filtered out,
        # behavioral_avg defaults to 0.5
        feedback = _make_feedback(["y", "y"])
        result = analyze_utility(feedback, _CFG)
        # explicit=1.0, behavioral=0.5 -> 0.6*1.0 + 0.4*0.5 = 0.8
        assert result["utility_score"] == pytest.approx(0.8, abs=0.01)


class TestMutationDecision:
    """Verify prompt mutation is triggered at the right threshold."""

    def test_low_utility_triggers_mutation(self) -> None:
        feedback = _make_feedback(["n", "n"], [0.1, 0.1])
        result = analyze_utility(feedback, _CFG)
        assert result["utility_score"] < UTILITY_MUTATION_THRESHOLD
        assert result["needs_prompt_mutation"] is True

    def test_high_utility_no_mutation(self) -> None:
        feedback = _make_feedback(["y", "y"], [0.9, 0.9])
        result = analyze_utility(feedback, _CFG)
        assert result["needs_prompt_mutation"] is False

    def test_single_event_never_triggers_mutation(self) -> None:
        """Even with low utility, a single event is not enough to mutate."""
        feedback = _make_feedback(["n"], [0.0])
        result = analyze_utility(feedback, _CFG)
        assert result["needs_prompt_mutation"] is False
