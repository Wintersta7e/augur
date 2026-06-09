"""Tests for PendingAdvice behavioral scoring.

The behavioral score feeds directly into analyze_utility, which decides
prompt mutation. If behavioral scoring is wrong, the utility signal is
corrupted and Augur mutates (or preserves) prompts based on bad data.
"""

from __future__ import annotations

import pytest

from perception.feedback_collector import PendingAdvice, POST_ADVICE_TRACK_MOVES


def _make_pending(
    baseline_mean: float = 10.0,
    baseline_std: float = 2.0,
    deviation_at_decision: float = 3.0,
    baseline_observation_count: int = 50,
) -> PendingAdvice:
    p = PendingAdvice(
        advice_id="test-001",
        domain="chess",
        entity="white",
        severity="medium",
        baseline_mean=baseline_mean,
        timestamp="2025-01-01T00:00:00Z",
    )
    # Decision-time-frozen snapshot the σ-space metric scores against. Set on the
    # instance (PendingAdvice wires these as constructor kwargs in a later task;
    # instance assignment works either way).
    p.baseline_std = baseline_std
    p.deviation_at_decision = deviation_at_decision
    p.baseline_observation_count = baseline_observation_count
    return p


class TestBehavioralScoreComputation:
    """Verify the domain-agnostic surprise-reduction score (spec §1A) via the
    PendingAdvice subclass. The scoring math itself is covered exhaustively in
    test_feedback_outcome_metric.py; these confirm the subclass wires into it."""

    def test_return_to_baseline_scores_high(self) -> None:
        # dev0 = 3σ; post-decision values sit on the mean (0σ) → surprise removed.
        p = _make_pending()
        for _ in range(POST_ADVICE_TRACK_MOVES):
            p.add_post_move(10.0)
        assert p.finalized and not p.unmeasurable
        assert p.behavioral_score > 0.7

    def test_stays_anomalous_scores_low(self) -> None:
        # post-decision stays 3σ off → surprise unchanged.
        p = _make_pending()
        for _ in range(POST_ADVICE_TRACK_MOVES):
            p.add_post_move(16.0)  # |16-10|/2 = 3σ
        assert p.finalized
        assert p.behavioral_score < 0.3

    def test_partial_return_scores_mid(self) -> None:
        p = _make_pending()
        for _ in range(POST_ADVICE_TRACK_MOVES):
            p.add_post_move(14.0)  # 2σ → surprise 4 of 9 → ~0.56
        assert p.finalized
        assert 0.4 <= p.behavioral_score <= 0.7

    def test_improving_trend_beats_worsening(self) -> None:
        """Same mean surprise, but a shrinking-deviation window gets the bonus."""
        improving = _make_pending()
        for v in (16.0, 13.0, 10.0):  # 3σ→1.5σ→0σ
            improving.add_post_move(v)
        worsening = _make_pending()
        for v in (10.0, 13.0, 16.0):  # 0σ→1.5σ→3σ (same mean surprise)
            worsening.add_post_move(v)
        assert improving.behavioral_score > worsening.behavioral_score

    def test_degenerate_std_is_unmeasurable_half(self) -> None:
        p = _make_pending(baseline_std=0.0)  # σ below floor
        for _ in range(POST_ADVICE_TRACK_MOVES):
            p.add_post_move(10.0)
        assert p.finalized and p.unmeasurable
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
