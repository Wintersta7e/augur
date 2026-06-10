"""Integration test: cross-domain correlator pipeline.

Runs the correlator's core logic against real Redis + NATS (no Ollama):
- Publish two LOW anomalies (chess + typing) within the 30s window
- Assert that a MEDIUM escalated correlation event is produced
- Assert that the payload contains both original events in full

Does NOT require Ollama — the advisor is not in this test. We're testing
the detector→correlator→NATS path only.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tabula.config import AugurConfig
from tabula.persistence import PersistenceManager
from reasoning.correlator import (
    DEFAULT_ESCALATION_MATRIX,
    correlate,
    ensure_matrix_seeded,
)

pytestmark = pytest.mark.asyncio
_CONFIG = AugurConfig.from_env()


def _anomaly(
    domain: str,
    entity: str,
    severity: str,
    ts: datetime,
    value: float = 10.0,
) -> dict:
    return {
        "domain": domain,
        "stream_id": f"{domain}_stream",
        "entity": entity,
        "event_type": "test_event",
        "value": value,
        "unit": "s",
        "context": {"label": "integration"},
        "session_id": "integration-test",
        "baseline_mean": 1.0,
        "baseline_std": 0.5,
        "deviation_score": 3.0,
        "anomaly_score": 0.4,
        "severity": severity,
        "timestamp": ts.isoformat(),
    }


async def test_two_lows_different_domains_produce_medium_correlation(
    redis_client,
    nats_conn,
) -> None:
    """Both events flow through correlate() against real Redis.

    Note: the redis_client fixture already clears augur:* keys on entry.
    """
    pm = PersistenceManager(redis_client)
    matrix = ensure_matrix_seeded(pm)
    assert matrix == DEFAULT_ESCALATION_MATRIX

    now = datetime.now(timezone.utc)
    typing_event = _anomaly("typing", "user", "low", now - timedelta(seconds=12))
    chess_event = _anomaly("chess", "white", "low", now)

    # Publish events to the window in chronological order
    first = correlate(typing_event, redis_client, matrix, _CONFIG)
    assert first is None, "standalone low typing event should be dropped"

    second = correlate(chess_event, redis_client, matrix, _CONFIG)
    assert second is not None
    assert second["correlation_found"] is True
    assert second["combined_severity"] == "MEDIUM"
    assert second["escalation_rule"] == "LOW+LOW\u2192MEDIUM"
    assert second["escalation_matrix_version"] == "1.0"

    # Primary is chess (just arrived); correlated is typing (older)
    assert second["primary_anomaly"]["domain"] == "chess"
    assert len(second["correlated_events"]) == 1
    assert second["correlated_events"][0]["domain"] == "typing"

    # Temporal lag is ~12 seconds
    assert 11.0 < second["temporal_lag_seconds"] < 13.0


async def test_standalone_high_passes_through_redis(
    redis_client,
    nats_conn,
) -> None:
    pm = PersistenceManager(redis_client)
    matrix = ensure_matrix_seeded(pm)

    now = datetime.now(timezone.utc)
    high = _anomaly("chess", "white", "high", now, value=99.9)

    result = correlate(high, redis_client, matrix, _CONFIG)

    assert result is not None
    assert result["correlation_found"] is False
    assert result["combined_severity"] == "HIGH"
    assert result["severity_escalated"] is False
    assert result["correlated_events"] == []


async def test_same_domain_lows_do_not_correlate(
    redis_client,
    nats_conn,
) -> None:
    pm = PersistenceManager(redis_client)
    matrix = ensure_matrix_seeded(pm)

    now = datetime.now(timezone.utc)
    chess1 = _anomaly("chess", "white", "low", now - timedelta(seconds=10))
    chess2 = _anomaly("chess", "black", "low", now)

    first = correlate(chess1, redis_client, matrix, _CONFIG)
    second = correlate(chess2, redis_client, matrix, _CONFIG)

    assert first is None
    assert second is None  # same-domain low + low → both dropped
