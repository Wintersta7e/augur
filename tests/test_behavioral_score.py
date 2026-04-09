"""Tests for PendingAdvice behavioral scoring.

The behavioral score feeds directly into analyze_utility, which decides
prompt mutation. If behavioral scoring is wrong, the utility signal is
corrupted and Augur mutates (or preserves) prompts based on bad data.
"""

from __future__ import annotations

import pytest

from perception.feedback_collector import PendingAdvice, POST_ADVICE_TRACK_MOVES


def _make_pending(baseline_mean: float = 5.0) -> PendingAdvice:
    return PendingAdvice(
        advice_id="test-001",
        domain="chess",
        entity="white",
        severity="medium",
        baseline_mean=baseline_mean,
        timestamp="2025-01-01T00:00:00Z",
    )


class TestBehavioralScoreComputation:
    """Verify _compute_behavioral_score under controlled conditions."""

    def test_faster_than_baseline_scores_high(self) -> None:
        p = _make_pending(baseline_mean=10.0)
        # All moves faster than baseline
        for _ in range(POST_ADVICE_TRACK_MOVES):
            p.add_post_move(5.0)  # ratio = 0.5 -> fast
        assert p.finalized
        assert p.behavioral_score > 0.7

    def test_much_slower_than_baseline_scores_low(self) -> None:
        p = _make_pending(baseline_mean=5.0)
        for _ in range(POST_ADVICE_TRACK_MOVES):
            p.add_post_move(20.0)  # ratio = 4.0 -> very slow
        assert p.finalized
        assert p.behavioral_score < 0.3

    def test_at_baseline_scores_near_half(self) -> None:
        p = _make_pending(baseline_mean=5.0)
        for _ in range(POST_ADVICE_TRACK_MOVES):
            p.add_post_move(5.0)  # ratio = 1.0 -> at baseline
        assert p.finalized
        assert 0.4 <= p.behavioral_score <= 0.85

    def test_normalizing_trend_gets_bonus(self) -> None:
        """Times trending toward baseline should score higher than stable slow."""
        p_normalizing = _make_pending(baseline_mean=5.0)
        # Trending from slow to baseline
        p_normalizing.add_post_move(10.0)
        p_normalizing.add_post_move(7.0)
        p_normalizing.add_post_move(5.5)

        p_stable_slow = _make_pending(baseline_mean=5.0)
        # Consistently slightly slow
        p_stable_slow.add_post_move(7.5)
        p_stable_slow.add_post_move(7.5)
        p_stable_slow.add_post_move(7.5)

        assert p_normalizing.behavioral_score > p_stable_slow.behavioral_score

    def test_zero_baseline_defaults_to_half(self) -> None:
        p = _make_pending(baseline_mean=0.0)
        for _ in range(POST_ADVICE_TRACK_MOVES):
            p.add_post_move(5.0)
        assert p.behavioral_score == pytest.approx(0.5)


class TestPostMoveTracking:
    """Verify add_post_move behavior and auto-finalization."""

    def test_tracks_correct_number_of_moves(self) -> None:
        p = _make_pending()
        for i in range(POST_ADVICE_TRACK_MOVES + 5):
            p.add_post_move(5.0)
        assert len(p.think_times_after) == POST_ADVICE_TRACK_MOVES

    def test_not_finalized_before_enough_moves(self) -> None:
        p = _make_pending()
        p.add_post_move(5.0)
        assert not p.finalized

    def test_finalized_after_enough_moves(self) -> None:
        p = _make_pending()
        for _ in range(POST_ADVICE_TRACK_MOVES):
            p.add_post_move(5.0)
        assert p.finalized


class TestToRecord:
    """Verify serialization for persistence."""

    def test_record_contains_all_fields(self) -> None:
        p = _make_pending()
        for _ in range(POST_ADVICE_TRACK_MOVES):
            p.add_post_move(5.0)
        record = p.to_record()

        assert record["advice_id"] == "test-001"
        assert record["domain"] == "chess"
        assert record["entity"] == "white"
        assert record["explicit_rating"] == "no_response"
        assert isinstance(record["behavioral_score"], float)
        assert len(record["think_times_after"]) == POST_ADVICE_TRACK_MOVES
