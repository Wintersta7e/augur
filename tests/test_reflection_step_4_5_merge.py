"""Tests for run_reflection: step 4 + step 5 merge into single matrix save."""

from unittest.mock import AsyncMock, MagicMock
import pytest

from blackboard.config import AugurConfig
from reasoning.reflection_engine import run_reflection


@pytest.mark.asyncio
async def test_step_4_and_5_save_single_merged_matrix():
    """Both step 4 (rule-confidence) and step 5 (window) should merge
    their changes into one matrix save call, not two."""
    feedback = {
        "advice_events": [
            {
                "domain": "chess",
                "correlation_found": True,
                "rule_key": "LOW+LOW",
                "explicit_rating": "n",
                "behavioral_score": 0.1,
                "correlation_span_s": 50.0,
                "involved_domains": ["chess", "typing"],
            }
            for _ in range(5)
        ],
        "session_summary": {"total_advice": 5},
    }

    pm = MagicMock()
    pm.is_tuning_applied.return_value = False
    pm.load_escalation_matrix.return_value = {
        "version": "1.0",
        "rules": {"LOW+LOW": "MEDIUM"},
        "rule_windows": {},
    }
    pm.load_rule_confidence.return_value = {}
    pm.load_rule_window_state.return_value = {}
    pm.load_thresholds.return_value = None
    pm.get_history.return_value = []
    pm.get_feedback.return_value = feedback

    nc = AsyncMock()
    redis_client = MagicMock()
    http_client = AsyncMock()

    report = await run_reflection(
        "session-123",
        feedback,
        pm,
        redis_client,
        http_client,
        nc,
        AugurConfig.from_env(),
    )

    # Single matrix save with BOTH new rules and new rule_windows
    save_calls = pm.save_escalation_matrix.call_args_list
    assert len(save_calls) == 1, f"Expected 1 matrix save, got {len(save_calls)}"
    saved = save_calls[0].args[0]
    assert "rules" in saved
    assert "rule_windows" in saved


@pytest.mark.asyncio
async def test_window_state_persisted_independently_from_matrix():
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
            for _ in range(3)
        ],
        "session_summary": {"total_advice": 3},
    }
    pm = MagicMock()
    pm.is_tuning_applied.return_value = False
    pm.load_escalation_matrix.return_value = {
        "version": "1.0",
        "rules": {"LOW+LOW": "MEDIUM"},
    }
    pm.load_rule_confidence.return_value = {}
    pm.load_rule_window_state.return_value = {}
    pm.load_thresholds.return_value = None
    pm.get_history.return_value = []

    nc = AsyncMock()

    await run_reflection(
        "session-X", feedback, pm, MagicMock(), AsyncMock(), nc, AugurConfig.from_env()
    )

    # Round-3: state saves go through the atomic pm.save_tuning_state(...) call
    # (a single MULTI/EXEC pipeline writing both confidence + window_state).
    pm.save_tuning_state.assert_called_once()
    kwargs = pm.save_tuning_state.call_args.kwargs
    # window_state should contain the new EWMA for LOW+LOW
    assert kwargs.get("window_state") is not None
    assert "LOW+LOW" in kwargs["window_state"]


@pytest.mark.asyncio
async def test_marker_set_after_both_writes():
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
            for _ in range(3)
        ],
        "session_summary": {"total_advice": 3},
    }
    pm = MagicMock()
    pm.is_tuning_applied.return_value = False
    pm.load_escalation_matrix.return_value = {
        "version": "1.0",
        "rules": {"LOW+LOW": "MEDIUM"},
    }
    pm.load_rule_confidence.return_value = {}
    pm.load_rule_window_state.return_value = {}
    pm.load_thresholds.return_value = None
    pm.get_history.return_value = []

    nc = AsyncMock()

    await run_reflection(
        "session-Y", feedback, pm, MagicMock(), AsyncMock(), nc, AugurConfig.from_env()
    )

    pm.mark_tuning_applied.assert_called_once_with("session-Y")


@pytest.mark.asyncio
async def test_marker_NOT_set_when_matrix_save_fails():
    """Closes the race Codex flagged: a failed matrix save must NOT mark
    the session applied. Otherwise next-session retry can't recover."""
    import redis as redis_lib

    feedback = {
        "advice_events": [
            {
                "domain": "chess",
                "correlation_found": True,
                "rule_key": "LOW+LOW",
                "explicit_rating": "n",
                "behavioral_score": 0.1,
                "correlation_span_s": 50.0,
                "involved_domains": ["chess", "typing"],
            }
            for _ in range(5)
        ],
        "session_summary": {"total_advice": 5},
    }
    pm = MagicMock()
    pm.is_tuning_applied.return_value = False
    pm.load_escalation_matrix.return_value = {
        "version": "1.0",
        "rules": {"LOW+LOW": "MEDIUM"},
        "rule_windows": {},
    }
    pm.load_rule_confidence.return_value = {}
    pm.load_rule_window_state.return_value = {}
    pm.load_thresholds.return_value = None
    pm.get_history.return_value = []
    # Matrix save fails — marker MUST NOT be set
    pm.save_escalation_matrix.side_effect = redis_lib.RedisError("simulated")

    nc = AsyncMock()

    await run_reflection(
        "session-FAIL",
        feedback,
        pm,
        MagicMock(),
        AsyncMock(),
        nc,
        AugurConfig.from_env(),
    )

    pm.mark_tuning_applied.assert_not_called()


@pytest.mark.asyncio
async def test_marker_NOT_set_when_state_save_fails():
    """rule_confidence or rule_window_state save failure also blocks the marker."""
    import redis as redis_lib

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
            for _ in range(3)
        ],
        "session_summary": {"total_advice": 3},
    }
    pm = MagicMock()
    pm.is_tuning_applied.return_value = False
    pm.load_escalation_matrix.return_value = {
        "version": "1.0",
        "rules": {"LOW+LOW": "MEDIUM"},
    }
    pm.load_rule_confidence.return_value = {}
    pm.load_rule_window_state.return_value = {}
    pm.load_thresholds.return_value = None
    pm.get_history.return_value = []
    pm.save_tuning_state.side_effect = redis_lib.RedisError("simulated")

    nc = AsyncMock()

    await run_reflection(
        "session-FAIL2",
        feedback,
        pm,
        MagicMock(),
        AsyncMock(),
        nc,
        AugurConfig.from_env(),
    )

    pm.mark_tuning_applied.assert_not_called()
