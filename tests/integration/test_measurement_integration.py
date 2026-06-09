"""Integration: the detector emits a consistent decision-time (pre-update)
baseline snapshot on the anomaly payload (spec 1A/§4.3 — the CRITICAL fix).

Robust against the subprocess-subscribe race: a warmup event is injected first
and its baseline key awaited before the rest, so the detector is provably
subscribed before the events that matter. Run on a slow box with
AUGUR_TEST_STARTUP_WAIT_S bumped (e.g. 12).

The fired/withheld σ-score path is covered exhaustively by the unit suite
(test_fired_arm_snapshot, test_withheld_arm_snapshot, test_feedback_outcome_metric)
and exercised end-to-end by the rest of the integration suite.
"""

import asyncio
import json

import pytest

from .conftest import inject_perception_event, wait_for_redis_key

pytestmark = pytest.mark.asyncio

_DOMAIN = "inttest_meas"
_ENTITY = "probe1"


async def _inject(nc, value: float) -> None:
    await inject_perception_event(
        nc,
        domain=_DOMAIN,
        entity=_ENTITY,
        event_type="sample",
        value=value,
        unit="units",
        context={},
        session_id="meas-int-1",
    )


@pytest.mark.parametrize("pipeline", [["detector"]], indirect=True)
async def test_detector_emits_decision_time_snapshot(pipeline, redis_client, nats_conn):
    baseline_key = f"augur:profile:{_DOMAIN}:{_ENTITY}"

    # 1) Warmup event + wait for its baseline → detector is provably subscribed
    #    (defeats the subscribe-vs-publish race deterministically).
    await _inject(nats_conn, 10.0)
    assert await wait_for_redis_key(redis_client, baseline_key, timeout=15.0), (
        "detector never created the baseline — it likely had not subscribed; "
        "bump AUGUR_TEST_STARTUP_WAIT_S"
    )

    # 2) Train past min_observations (15) with VARIED values so ewma_std > 0.01
    #    (the zero-variance gotcha would otherwise make deviation 0).
    for i in range(20):
        await _inject(nats_conn, 10.0 + (2.0 if i % 2 else -2.0))
        await asyncio.sleep(0.05)

    # 3) A large deviation-driven anomaly on this NON-chess entity.
    await _inject(nats_conn, 40.0)
    assert await wait_for_redis_key(
        redis_client, "augur:detection:last_anomaly", timeout=15.0
    )
    await asyncio.sleep(0.3)  # let the final write settle

    raw = redis_client.get("augur:detection:last_anomaly")
    payload = json.loads(raw.decode() if isinstance(raw, bytes) else raw)

    # The decision-time snapshot must be present and self-consistent.
    assert payload["domain"] == _DOMAIN and payload["entity"] == _ENTITY
    assert "baseline_std" in payload and payload["baseline_std"] > 0.0
    assert "baseline_observation_count" in payload
    assert payload["baseline_observation_count"] >= 15  # trained
    assert "drift_reset" in payload and isinstance(payload["drift_reset"], bool)

    # deviation_score is the PRE-update z-score; it must be consistent with the
    # emitted (pre-update) baseline_mean/std — not a post-update blend.
    expected_dev = abs(40.0 - payload["baseline_mean"]) / payload["baseline_std"]
    assert payload["deviation_score"] == pytest.approx(expected_dev, abs=0.05)
    assert payload["deviation_score"] > 4.0  # genuinely anomalous
    assert payload["severity"] in ("medium", "high")
