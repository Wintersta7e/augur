"""Integration: cross-domain feedback drives sigma adjustments in TWO domains.

Verifies the breaking change to analyze_precision: a high-precision
correlated session should lower sigma for both involved domains.

Requires Redis + NATS running.
"""

from __future__ import annotations

import httpx
import pytest

from tabula.config import AugurConfig
from tabula.persistence import PersistenceManager
from disciplina.reflection_engine import run_reflection

pytestmark = pytest.mark.asyncio


async def test_correlated_feedback_lowers_sigma_in_both_domains(
    redis_client,
    nats_conn,
) -> None:
    pm = PersistenceManager(redis_client)
    pm.save_thresholds(
        "chess",
        {"sigma_threshold": 2.0, "ewma_alpha": 0.3, "hst_threshold": 0.7},
    )
    pm.save_thresholds(
        "typing",
        {"sigma_threshold": 2.0, "ewma_alpha": 0.3, "hst_threshold": 0.7},
    )

    feedback = {
        "session_id": "session-multi",
        "advice_events": [
            {
                "advice_id": f"adv-{i}",
                "domain": "chess",
                "entity": "white",
                "severity": "low",
                "explicit_rating": "y",
                "behavioral_score": 0.9,
                "correlation_found": True,
                "rule_key": "LOW+LOW",
                "involved_domains": ["chess", "typing"],
                "correlation_span_s": 5.0,
                "rule_window_s": 30.0,
            }
            for i in range(5)
        ],
        "session_summary": {"total_advice": 5},
    }
    pm.save_feedback("session-multi", feedback)

    http_client = httpx.AsyncClient()
    try:
        report = await run_reflection(
            "session-multi",
            feedback,
            pm,
            redis_client,
            http_client,
            nats_conn,
            AugurConfig.from_env(),
        )
    finally:
        await http_client.aclose()

    # Each event contributes 0.5 to chess and 0.5 to typing → 2.5 weighted total per domain
    # All useful (rating=y) → precision 1.0 → high precision → lower sigma
    per_domain = report["analyses"]["precision"]["per_domain"]
    assert "chess" in per_domain
    assert "typing" in per_domain
    assert per_domain["chess"]["action"] == "lower_sigma"
    assert per_domain["typing"]["action"] == "lower_sigma"

    chess_thresholds = pm.load_thresholds("chess")
    typing_thresholds = pm.load_thresholds("typing")
    assert chess_thresholds["sigma_threshold"] < 2.0
    assert typing_thresholds["sigma_threshold"] < 2.0
