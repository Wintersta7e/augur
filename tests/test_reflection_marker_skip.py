"""Tests that the unified marker correctly skips both step 4 and step 5."""

from unittest.mock import AsyncMock, MagicMock
import pytest

from tabula.config import AugurConfig
from disciplina.reflection_engine import run_reflection


@pytest.mark.asyncio
async def test_marker_set_skips_both_step_4_and_5():
    feedback = {
        "advice_events": [
            {
                "domain": "chess",
                "correlation_found": True,
                "rule_key": "LOW+LOW",
                "explicit_rating": "y",
                "behavioral_score": 0.9,
                "correlation_span_s": 8.0,
                "involved_domains": ["chess", "typing"],
            }
        ],
        "session_summary": {"total_advice": 1},
    }
    pm = MagicMock()
    pm.is_tuning_applied.return_value = True  # already applied
    pm.load_thresholds.return_value = None
    pm.get_history.return_value = []

    nc = AsyncMock()

    report = await run_reflection(
        "session-DUP",
        feedback,
        pm,
        MagicMock(),
        AsyncMock(),
        nc,
        AugurConfig.from_env(),
    )

    # Neither matrix nor state saves should occur
    pm.save_escalation_matrix.assert_not_called()
    pm.save_tuning_state.assert_not_called()
    pm.mark_tuning_applied.assert_not_called()
    # Both analyses report skipped
    assert report["analyses"]["correlation_tuning"].get("skipped") is True
    assert report["analyses"]["correlation_window_tuning"].get("skipped") is True
