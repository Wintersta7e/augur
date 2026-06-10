"""Withheld-arm decision-time snapshot wiring (spec §1A): publish_suppressed_event
threads the detector snapshot into the suppressed payload, and PendingGateDecision
freezes + serializes it. Also pins the BLOCKER-1 signature ordering (defaulted
args after the required mrt_eligible/p_withhold/reason)."""

import json
from unittest.mock import AsyncMock

import pytest

from responsum.feedback_collector import PendingGateDecision
from limen.gate import GateDecision, build_signature
from consilium.advisor import publish_suppressed_event


@pytest.mark.asyncio
async def test_suppressed_payload_carries_snapshot():
    payload = {
        "primary_anomaly": {
            "domain": "activity_intensity",
            "entity": "someapp.exe",
            "value": 120.0,
            "baseline_mean": 60.0,
            "baseline_std": 15.0,
            "deviation_score": 4.0,
            "baseline_observation_count": 50,
            "timestamp": "2026-06-09T00:00:00+00:00",
        },
        "combined_severity": "medium",
    }
    sig = build_signature(payload)
    decision = GateDecision.suppress("habituation")
    nc = AsyncMock()
    await publish_suppressed_event(nc, sig, decision, payload, redis_client=None)
    sent = json.loads(nc.publish.call_args[0][1].decode())
    assert sent["baseline_std"] == 15.0
    assert sent["deviation_score"] == 4.0
    assert sent["baseline_observation_count"] == 50


def test_pending_gate_decision_freezes_snapshot():
    p = PendingGateDecision(
        decision_id="d1",
        state_key="single:typing:keyboard",
        domain="typing",
        entity="keyboard",
        severity="medium",
        baseline_mean=0.9,
        timestamp="t",
        mrt_eligible=True,
        p_withhold=0.9,
        reason="habituation",
        baseline_std=0.3,
        deviation_at_decision=3.0,
        baseline_observation_count=40,
        session_id="s1",
    )
    assert p.baseline_std == 0.3
    assert p.session_id == "s1"
    assert p.selected_for_rating is False
    assert p.withheld_rating_p is None
    rec = p.to_record()
    assert rec["baseline_std_at_time"] == 0.3
    assert rec["deviation_at_decision"] == 3.0
    assert rec["baseline_observation_count"] == 40
    assert rec["outcome_metric_version"] == 2
    assert rec["unmeasurable"] is False
    assert rec["withheld_rating_p"] is None
