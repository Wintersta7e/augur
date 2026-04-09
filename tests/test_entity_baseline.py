"""Tests for EntityBaseline — EWMA math, scoring, and serialization.

These guard the statistical foundation that every anomaly detection decision
depends on. If the EWMA update or scoring is wrong, Augur's baselines drift
and the entire detection layer produces garbage.
"""

from __future__ import annotations

import math

import pytest

from detection.anomaly_detector import EntityBaseline


class TestEWMAUpdate:
    """Verify EWMA mean and variance tracking."""

    def test_first_observation_sets_mean_exactly(self) -> None:
        bl = EntityBaseline()
        bl.update(10.0, alpha=0.3)
        assert bl.ewma_mean == 10.0
        assert bl.ewma_var == 0.0
        assert bl.observation_count == 1

    def test_second_observation_applies_alpha(self) -> None:
        bl = EntityBaseline()
        bl.update(10.0, alpha=0.3)
        bl.update(20.0, alpha=0.3)
        # mean = 10 + 0.3 * (20 - 10) = 13.0
        assert bl.ewma_mean == pytest.approx(13.0)
        assert bl.observation_count == 2

    def test_variance_grows_with_spread(self) -> None:
        bl = EntityBaseline()
        bl.update(10.0, alpha=0.3)
        bl.update(20.0, alpha=0.3)
        # var = (1 - 0.3) * (0.0 + 0.3 * 10^2) = 0.7 * 30 = 21.0
        assert bl.ewma_var == pytest.approx(21.0)

    def test_stable_values_converge_to_low_variance(self) -> None:
        bl = EntityBaseline()
        for _ in range(50):
            bl.update(5.0, alpha=0.3)
        assert bl.ewma_mean == pytest.approx(5.0, abs=0.01)
        assert bl.ewma_var < 0.01

    def test_alpha_one_tracks_last_value(self) -> None:
        bl = EntityBaseline()
        bl.update(10.0, alpha=1.0)
        bl.update(99.0, alpha=1.0)
        assert bl.ewma_mean == pytest.approx(99.0)

    def test_alpha_zero_never_moves_after_second(self) -> None:
        bl = EntityBaseline()
        bl.update(10.0, alpha=0.0)
        bl.update(99.0, alpha=0.0)
        # mean stays at 10 after second observation: 10 + 0*(99-10) = 10
        assert bl.ewma_mean == pytest.approx(10.0)


class TestEWMAStd:
    """Verify standard deviation derivation."""

    def test_zero_variance_gives_zero_std(self) -> None:
        bl = EntityBaseline()
        bl.update(5.0, alpha=0.3)
        assert bl.ewma_std == 0.0

    def test_negative_variance_clamped_to_zero(self) -> None:
        bl = EntityBaseline()
        bl.ewma_var = -0.001  # should not happen, but guard against it
        assert bl.ewma_std == 0.0

    def test_positive_variance_gives_sqrt(self) -> None:
        bl = EntityBaseline()
        bl.ewma_var = 9.0
        assert bl.ewma_std == pytest.approx(3.0)


class TestScoring:
    """Verify deviation and HST scoring."""

    def test_deviation_zero_when_std_near_zero(self) -> None:
        bl = EntityBaseline()
        bl.update(5.0, alpha=0.3)  # single obs, var=0, std<0.01
        deviation, _ = bl.score(100.0)
        assert deviation == 0.0  # can't compute sigma with no spread

    def test_deviation_scales_with_distance(self) -> None:
        bl = EntityBaseline()
        bl.ewma_mean = 10.0
        bl.ewma_var = 4.0  # std = 2.0
        deviation, _ = bl.score(16.0)
        # |16 - 10| / 2 = 3.0 sigma
        assert deviation == pytest.approx(3.0)

    def test_deviation_symmetric(self) -> None:
        bl = EntityBaseline()
        bl.ewma_mean = 10.0
        bl.ewma_var = 4.0
        dev_high, _ = bl.score(16.0)
        dev_low, _ = bl.score(4.0)
        assert dev_high == pytest.approx(dev_low)

    def test_hst_score_returns_numeric(self) -> None:
        bl = EntityBaseline()
        # Train HST with some data so it has something to score against
        for v in [5.0, 5.1, 4.9, 5.2, 4.8]:
            bl.update(v, alpha=0.3)
        _, hst_score = bl.score(5.0)
        assert isinstance(hst_score, (int, float))  # River may return int 0
        assert 0.0 <= hst_score <= 1.0


class TestSerialization:
    """Verify state dict round-trip preserves baseline state."""

    def test_round_trip_preserves_state(self) -> None:
        bl = EntityBaseline()
        for v in [3.0, 7.0, 5.0, 12.0]:
            bl.update(v, alpha=0.3)

        state = bl.to_state_dict()
        restored = EntityBaseline.from_state_dict(state)

        assert restored.ewma_mean == pytest.approx(bl.ewma_mean)
        assert restored.ewma_var == pytest.approx(bl.ewma_var)
        assert restored.observation_count == bl.observation_count

    def test_from_empty_dict_gives_defaults(self) -> None:
        restored = EntityBaseline.from_state_dict({})
        assert restored.ewma_mean == 0.0
        assert restored.ewma_var == 0.0
        assert restored.observation_count == 0

    def test_hst_not_serialized(self) -> None:
        """HST model is not persisted — only EWMA state is.
        A restored baseline starts with a fresh HST. This is by design:
        HST rebuilds quickly from incoming data."""
        bl = EntityBaseline()
        for v in [1.0, 2.0, 3.0]:
            bl.update(v, alpha=0.3)
        state = bl.to_state_dict()
        assert "hst" not in state
