"""Integration: 3 candidates with one beyond pairwise window → 2-way result.

Verifies that pairwise filtering correctly drops the third candidate
and the resulting correlation falls back to a 2-way rule_key.

Requires Redis running.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tabula.config import AugurConfig
from tabula.persistence import PersistenceManager
from nexus.correlator import (
    correlate,
    ensure_matrix_seeded,
)

pytestmark = pytest.mark.asyncio
_CONFIG = AugurConfig.from_env()


def _anomaly(domain: str, severity: str, ts: datetime) -> dict:
    return {
        "domain": domain,
        "stream_id": f"{domain}_stream",
        "entity": f"{domain}-e1",
        "event_type": "test_event",
        "value": 1.0,
        "unit": "s",
        "context": {"label": "integration"},
        "session_id": "integration-test-3way-filter",
        "baseline_mean": 1.0,
        "baseline_std": 0.5,
        "deviation_score": 3.0,
        "severity": severity,
        "timestamp": ts.isoformat(),
    }


async def test_3way_candidate_falls_back_to_2way_when_third_beyond_window(
    redis_client, nats_conn
) -> None:
    """3 candidates: focus 50s old (beyond 30s window), typing 10s, chess primary.
    Pairwise filter drops focus → 2-way LOW+LOW remains."""
    pm = PersistenceManager(redis_client)
    matrix = ensure_matrix_seeded(pm)

    now = datetime.now(timezone.utc)
    focus_event = _anomaly("focus", "low", now - timedelta(seconds=50))
    typing_event = _anomaly("typing", "low", now - timedelta(seconds=10))
    chess_event = _anomaly("chess", "low", now)

    # Build the window in chronological order
    correlate(focus_event, redis_client, matrix, _CONFIG)  # standalone low → None
    correlate(
        typing_event, redis_client, matrix, _CONFIG
    )  # may correlate; we don't care here

    # Chess arrives — 50s focus is beyond window, 10s typing is within → 2-way
    p3 = correlate(chess_event, redis_client, matrix, _CONFIG)
    assert p3 is not None
    assert p3["correlation_found"] is True
    assert p3["rule_key"] == "LOW+LOW", f"Expected pairwise; got {p3['rule_key']}"
    assert set(p3["involved_domains"]) == {"chess", "typing"}
    # focus must NOT be in correlated_events
    correlated_domains = {e["domain"] for e in p3["correlated_events"]}
    assert correlated_domains == {"typing"}, f"focus leaked: {correlated_domains}"
