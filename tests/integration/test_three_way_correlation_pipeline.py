"""Integration: 3 anomalies from 3 domains within window → 3-way correlation event.

Requires Redis running (docker compose up -d).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tabula.config import AugurConfig
from tabula.persistence import PersistenceManager
from reasoning.correlator import (
    correlate,
    ensure_matrix_seeded,
)

pytestmark = pytest.mark.asyncio
_CONFIG = AugurConfig.from_env()


def _anomaly(domain: str, severity: str, ts: datetime, value: float = 1.0) -> dict:
    return {
        "domain": domain,
        "stream_id": f"{domain}_stream",
        "entity": f"{domain}-e1",
        "event_type": "test_event",
        "value": value,
        "unit": "s",
        "context": {"label": "integration"},
        "session_id": "integration-test-3way",
        "baseline_mean": 1.0,
        "baseline_std": 0.5,
        "deviation_score": 3.0,
        "severity": severity,
        "timestamp": ts.isoformat(),
    }


async def test_three_way_correlation_emits_3way_rule_key(
    redis_client, nats_conn
) -> None:
    """3 LOW anomalies from 3 domains within window → rule_key=LOW+LOW+LOW."""
    pm = PersistenceManager(redis_client)
    matrix = ensure_matrix_seeded(pm)

    now = datetime.now(timezone.utc)
    typing_event = _anomaly("typing", "low", now - timedelta(seconds=10))
    focus_event = _anomaly("focus", "low", now - timedelta(seconds=5))
    chess_event = _anomaly("chess", "low", now)

    # First anomaly arrives — standalone low → drop (None)
    p1 = correlate(typing_event, redis_client, matrix, _CONFIG)
    assert p1 is None

    # Second arrives — pairwise typing+focus → MEDIUM correlation
    p2 = correlate(focus_event, redis_client, matrix, _CONFIG)
    assert p2 is not None
    # The 2-way correlation may have rule_key=LOW+LOW or be a passthrough,
    # depending on severity gate. Don't assert on this intermediate.

    # Third arrives — chess primary, typing+focus correlated → 3-way LOW+LOW+LOW
    p3 = correlate(chess_event, redis_client, matrix, _CONFIG)
    assert p3 is not None, "Expected 3-way correlation"
    assert p3["correlation_found"] is True
    assert p3["rule_key"] == "LOW+LOW+LOW"
    assert p3["combined_severity"] == "MEDIUM"
    assert set(p3["involved_domains"]) == {"chess", "typing", "focus"}
    assert p3["correlation_span_s"] is not None
    assert p3["rule_window_s"] is not None
    assert p3["escalation_rule"] == "LOW+LOW+LOW→MEDIUM"
