"""Tests for run_reflection: step 4 + step 5 merge into a single matrix CAS patch."""

from unittest.mock import ANY, AsyncMock, MagicMock, call
import pytest

from tabula.config import AugurConfig
from disciplina.reflection_engine import run_reflection
from nexus import matrix_ops


def _ok_update(*args, **kwargs):
    """Success stand-in for matrix_ops.apply_matrix_update (real CAS needs Redis)."""
    return {
        "status": "saved",
        "matrix": {"version": "1.0", "rules": {}, "rule_windows": {}},
        "prior_rules": {},
        "prior_rule_windows": {},
    }


@pytest.mark.asyncio
async def test_step_4_and_5_save_single_merged_matrix(monkeypatch):
    """Step 4 (rule-confidence) and step 5 (window) merge into ONE CAS patch
    call, not two separate writes."""
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
    pm.get_prompt_score_pair.return_value = (None, None)  # 1E prompt pass no-op
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

    update_mock = MagicMock(side_effect=_ok_update)
    monkeypatch.setattr(matrix_ops, "apply_matrix_update", update_mock)

    nc = AsyncMock()
    await run_reflection(
        "session-123",
        feedback,
        pm,
        MagicMock(),
        AsyncMock(),
        nc,
        AugurConfig.from_env(),
    )

    # Single merged matrix write, routed through the shared CAS helper in patch mode.
    assert update_mock.call_count == 1
    assert update_mock.call_args.kwargs.get("mode") == "patch"


@pytest.mark.asyncio
async def test_window_state_persisted_independently_from_matrix(monkeypatch):
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
    pm.get_prompt_score_pair.return_value = (None, None)  # 1E prompt pass no-op
    pm.load_escalation_matrix.return_value = {
        "version": "1.0",
        "rules": {"LOW+LOW": "MEDIUM"},
    }
    pm.load_rule_confidence.return_value = {}
    pm.load_rule_window_state.return_value = {}
    pm.load_thresholds.return_value = None
    pm.get_history.return_value = []

    monkeypatch.setattr(
        matrix_ops, "apply_matrix_update", MagicMock(side_effect=_ok_update)
    )
    nc = AsyncMock()

    await run_reflection(
        "session-X", feedback, pm, MagicMock(), AsyncMock(), nc, AugurConfig.from_env()
    )

    # Round-3: state saves go through the atomic pm.save_tuning_state(...) call
    # (a single MULTI/EXEC pipeline writing both confidence + window_state).
    pm.save_tuning_state.assert_called_once()
    kwargs = pm.save_tuning_state.call_args.kwargs
    assert kwargs.get("window_state") is not None
    assert "LOW+LOW" in kwargs["window_state"]


@pytest.mark.asyncio
async def test_marker_set_after_both_writes(monkeypatch):
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
    pm.get_prompt_score_pair.return_value = (None, None)  # 1E prompt pass no-op
    pm.load_escalation_matrix.return_value = {
        "version": "1.0",
        "rules": {"LOW+LOW": "MEDIUM"},
    }
    pm.load_rule_confidence.return_value = {}
    pm.load_rule_window_state.return_value = {}
    pm.load_thresholds.return_value = None
    pm.get_history.return_value = []

    monkeypatch.setattr(
        matrix_ops, "apply_matrix_update", MagicMock(side_effect=_ok_update)
    )
    nc = AsyncMock()

    await run_reflection(
        "session-Y", feedback, pm, MagicMock(), AsyncMock(), nc, AugurConfig.from_env()
    )

    # The gate pass (pass_name="gate") also marks the session; assert the
    # correlation marker specifically was set.
    pm.mark_tuning_applied.assert_any_call(
        "session-Y", pass_name="correlation", ctx=ANY
    )


@pytest.mark.asyncio
async def test_marker_NOT_set_when_matrix_save_fails(monkeypatch):
    """A failed matrix CAS must NOT mark the session applied, so the next
    session's reflection can retry the tuning."""
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
    pm.get_prompt_score_pair.return_value = (None, None)  # 1E prompt pass no-op
    pm.load_escalation_matrix.return_value = {
        "version": "1.0",
        "rules": {"LOW+LOW": "MEDIUM"},
        "rule_windows": {},
    }
    pm.load_rule_confidence.return_value = {}
    pm.load_rule_window_state.return_value = {}
    pm.load_thresholds.return_value = None
    pm.get_history.return_value = []
    # Matrix CAS fails — marker MUST NOT be set.
    monkeypatch.setattr(
        matrix_ops,
        "apply_matrix_update",
        MagicMock(return_value={"error": "simulated contention"}),
    )

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

    # The independent gate pass may still mark pass_name="gate"; the correlation
    # marker specifically must NOT be set when the matrix save fails.
    assert (
        call("session-FAIL", pass_name="correlation", ctx=ANY)
        not in pm.mark_tuning_applied.call_args_list
    )


@pytest.mark.asyncio
async def test_marker_NOT_set_when_state_save_fails(monkeypatch):
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
    pm.get_prompt_score_pair.return_value = (None, None)  # 1E prompt pass no-op
    pm.load_escalation_matrix.return_value = {
        "version": "1.0",
        "rules": {"LOW+LOW": "MEDIUM"},
    }
    pm.load_rule_confidence.return_value = {}
    pm.load_rule_window_state.return_value = {}
    pm.load_thresholds.return_value = None
    pm.get_history.return_value = []
    monkeypatch.setattr(
        matrix_ops, "apply_matrix_update", MagicMock(side_effect=_ok_update)
    )
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

    assert (
        call("session-FAIL2", pass_name="correlation", ctx=ANY)
        not in pm.mark_tuning_applied.call_args_list
    )
