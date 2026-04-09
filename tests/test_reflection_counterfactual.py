"""Tests for analyze_counterfactual — threshold variant replay.

This function replays recent history against +-10% sigma variants to
recommend threshold changes. If the EWMA replay or flag counting is wrong,
Augur makes threshold recommendations based on phantom data.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from reasoning.reflection_engine import analyze_counterfactual


def _make_pm_with_history(events: list[dict]) -> MagicMock:
    """Create a PersistenceManager mock that returns the given history."""
    pm = MagicMock()
    # get_history returns newest-first (Redis LRANGE order)
    pm.get_history.return_value = list(reversed(events))
    return pm


class TestCounterfactualNoHistory:
    def test_empty_history_returns_no_recommendation(self) -> None:
        pm = MagicMock()
        pm.get_history.return_value = []
        result = analyze_counterfactual(pm, "chess", {"sigma_threshold": 2.0})
        assert result["events_replayed"] == 0
        assert "No history" in result["recommendation"]


class TestCounterfactualVariants:
    """Verify that +-10% variants are computed correctly."""

    def test_variant_thresholds(self) -> None:
        events = [{"entity": "white", "value": float(i)} for i in range(10)]
        pm = _make_pm_with_history(events)
        result = analyze_counterfactual(pm, "chess", {"sigma_threshold": 2.0})

        assert result["variants"]["current"]["sigma"] == 2.0
        assert result["variants"]["minus_10pct"]["sigma"] == pytest.approx(1.8)
        assert result["variants"]["plus_10pct"]["sigma"] == pytest.approx(2.2)


class TestCounterfactualFlagCounting:
    """Verify that lower threshold catches more and higher catches fewer."""

    def test_lower_threshold_flags_more_or_equal(self) -> None:
        # Create events with one clear outlier
        events = [{"entity": "white", "value": 5.0}] * 20
        events.append({"entity": "white", "value": 50.0})  # extreme outlier
        pm = _make_pm_with_history(events)

        result = analyze_counterfactual(
            pm,
            "chess",
            {"sigma_threshold": 2.0, "ewma_alpha": 0.3},
        )
        lower = result["variants"]["minus_10pct"]["would_flag"]
        current = result["variants"]["current"]["would_flag"]
        higher = result["variants"]["plus_10pct"]["would_flag"]

        assert lower >= current
        assert current >= higher


class TestCounterfactualEWMAReplay:
    """Verify the EWMA replay matches EntityBaseline math."""

    def test_first_event_always_zero_deviation(self) -> None:
        """First event for any entity has no baseline to deviate from."""
        events = [{"entity": "white", "value": 100.0}]
        pm = _make_pm_with_history(events)
        result = analyze_counterfactual(
            pm,
            "chess",
            {"sigma_threshold": 0.01, "ewma_alpha": 0.3},
        )
        # Even with a near-zero threshold, the first event should have 0 deviation
        assert result["variants"]["current"]["would_flag"] == 0

    def test_multi_entity_tracked_separately(self) -> None:
        """Each entity should have its own baseline in the replay."""
        events = [
            {"entity": "white", "value": 5.0},
            {"entity": "black", "value": 5.0},
            {"entity": "white", "value": 5.0},
            {"entity": "black", "value": 5.0},
        ]
        pm = _make_pm_with_history(events)
        result = analyze_counterfactual(
            pm,
            "chess",
            {"sigma_threshold": 2.0, "ewma_alpha": 0.3},
        )
        # Stable values — nothing should flag
        assert result["variants"]["current"]["would_flag"] == 0
