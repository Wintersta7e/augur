"""Integration: window adaptation persists across reflection runs.

Session 1: emit pairwise correlations with mean lag ~50s.
Reflection runs, EWMA → window tunes upward.
Session 2: correlator picks up tuned window via fresh matrix load.

Requires Redis + NATS running.
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from blackboard.config import AugurConfig
from blackboard.persistence import PersistenceManager
from reasoning.correlator import (
    add_to_window,
    correlate,
    ensure_matrix_seeded,
)
from reasoning.reflection_engine import run_reflection

pytestmark = pytest.mark.asyncio


async def test_window_tunes_after_session_with_long_lag(
    redis_client, nats_conn
) -> None:
    """Reflection on a session with 50s correlation_span_s should tune the
    LOW+LOW window upward; session 2 picks up the tuned window."""
    pm = PersistenceManager(redis_client)
    ensure_matrix_seeded(pm)

    feedback = {
        "session_id": "session-1-window",
        "advice_events": [
            {
                "advice_id": f"adv-{i}",
                "domain": "chess",
                "entity": "white",
                "severity": "low",
                "explicit_rating": "y",
                "behavioral_score": 0.85,
                "correlation_found": True,
                "rule_key": "LOW+LOW",
                "involved_domains": ["chess", "typing"],
                "correlation_span_s": 50.0,
                "temporal_lag_seconds": 50.0,
                "rule_window_s": 30.0,
            }
            for i in range(5)
        ],
        "session_summary": {"total_advice": 5},
    }
    pm.save_feedback("session-1-window", feedback)

    http_client = httpx.AsyncClient()
    try:
        await run_reflection(
            "session-1-window",
            feedback,
            pm,
            redis_client,
            http_client,
            nats_conn,
            AugurConfig.from_env(),
        )
    finally:
        await http_client.aclose()

    # Verify window state was persisted
    state = pm.load_rule_window_state()
    assert "LOW+LOW" in state
    assert state["LOW+LOW"]["ewma_lag"] > 30.0  # absorbed the 50s lag

    # Verify rule_windows in matrix was updated
    # EWMA: new_lag = 50.0 (no prior), target_window = min(50*2.5, 120) = 120.0
    # delta_pct = |120-30|/30 = 3.0 >= 0.20 hysteresis → tuned to 120.0
    matrix = pm.load_escalation_matrix()
    assert matrix is not None
    assert "rule_windows" in matrix
    assert "LOW+LOW" in matrix["rule_windows"]
    assert matrix["rule_windows"]["LOW+LOW"] > 30.0

    # ─── Session 2: tuned window allows pair beyond default 30s to correlate ───
    s2_config = AugurConfig.from_env()
    now2 = datetime.now(timezone.utc).timestamp()

    def _ts2(seconds_ago: float) -> str:
        return datetime.fromtimestamp(now2 - seconds_ago, timezone.utc).isoformat()

    typing_event = {
        "domain": "typing",
        "entity": "kbd",
        "severity": "low",
        "value": 0.5,
        "timestamp": _ts2(33.0),
    }
    chess_event = {
        "domain": "chess",
        "entity": "white",
        "severity": "low",
        "value": 12.0,
        "timestamp": _ts2(0.0),
    }

    tuned_matrix = pm.load_escalation_matrix()
    add_to_window(redis_client, typing_event, prune_window_s=240.0)

    payload = correlate(chess_event, redis_client, tuned_matrix, s2_config)

    assert payload is not None, "Tuned window should allow 33s pair to correlate"
    assert payload["correlation_found"] is True
    assert payload["rule_key"] == "LOW+LOW"
