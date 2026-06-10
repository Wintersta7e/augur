"""Tests for the analyze_utility correlated-event filter.

Prompt mutation mutates the chess domain's stored system prompt. The
advisor's correlation path does NOT use stored prompts (it uses
build_correlation_prompt, which is self-contained). So correlated
advice with bad feedback should NOT drive prompt mutation — that
would be fixing the wrong knob.

analyze_utility now filters out events with correlation_found=True
before computing its score.
"""

from __future__ import annotations

from tabula.config import AugurConfig
from reasoning.reflection_engine import analyze_utility

CONFIG = AugurConfig()


def _advice(explicit: str, behavioral: float, correlation_found: bool) -> dict:
    return {
        "advice_id": "adv",
        "domain": "chess",
        "entity": "white",
        "severity": "medium",
        "explicit_rating": explicit,
        "behavioral_score": behavioral,
        "behavioral_finalized": True,
        "unmeasurable": False,
        "think_times_after": [],
        "baseline_mean_at_time": 5.0,
        "timestamp": "2026-04-09T12:00:00+00:00",
        "correlation_found": correlation_found,
    }


class TestAnalyzeUtilityCorrelationFilter:
    def test_excludes_correlation_found_true_events(self) -> None:
        # 2 correlated bad events + 1 standalone good event
        # Without the filter: 3 events, 1 positive → poor utility → mutation flagged
        # With the filter:    1 event,  1 positive → great utility → no mutation
        feedback = {
            "advice_events": [
                _advice("n", 0.0, correlation_found=True),
                _advice("n", 0.0, correlation_found=True),
                _advice("y", 1.0, correlation_found=False),
            ],
            "session_summary": {"total_advice": 3},
        }
        result = analyze_utility(feedback, CONFIG)
        assert result["needs_prompt_mutation"] is False
        # Computed from only the standalone event (y, 1.0) → explicit=1.0, behavioral=1.0
        assert result["utility_score"] == 1.0

    def test_unchanged_for_all_standalone_feedback(self) -> None:
        # Pure standalone feedback should produce the same result as before the fix
        feedback = {
            "advice_events": [
                _advice("y", 0.9, correlation_found=False),
                _advice("y", 0.8, correlation_found=False),
            ],
            "session_summary": {"total_advice": 2},
        }
        result = analyze_utility(feedback, CONFIG)
        # explicit_avg = 1.0, behavioral_avg = 0.85, utility = 0.6*1.0 + 0.4*0.85 = 0.94
        assert result["utility_score"] == 0.94
        assert result["needs_prompt_mutation"] is False

    def test_empty_when_all_correlated(self) -> None:
        # All events filtered out → treated as "no advice to evaluate" baseline
        feedback = {
            "advice_events": [
                _advice("n", 0.0, correlation_found=True),
                _advice("n", 0.0, correlation_found=True),
            ],
            "session_summary": {"total_advice": 2},
        }
        result = analyze_utility(feedback, CONFIG)
        # Same as empty feedback: utility 1.0, no mutation, reason "No advice events to evaluate"
        assert result["utility_score"] == 1.0
        assert result["needs_prompt_mutation"] is False
        assert "No advice events to evaluate" in result["reason"]
