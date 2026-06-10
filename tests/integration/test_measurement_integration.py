"""Integration: the detector emits a consistent decision-time (pre-update)
baseline snapshot on the anomaly payload (spec 1A/§4.3 — the CRITICAL fix).

Robust against the subprocess-subscribe race: warmup events are RESENT in a
loop until the baseline key appears (NATS core has no persistence, so a warmup
published before the detector subscribed is dropped — resending retries it),
so the detector is provably subscribed before the events that matter. On a slow
box, bumping AUGUR_TEST_STARTUP_WAIT_S just reduces the number of retries.

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


@pytest.mark.parametrize("pipeline", [["vigil"]], indirect=True)
async def test_detector_emits_decision_time_snapshot(pipeline, redis_client, nats_conn):
    baseline_key = f"augur:vigil:profile:{_DOMAIN}:{_ENTITY}"

    # 1) Warmup until the baseline appears — RESEND in a loop so a warmup dropped
    #    before the detector subscribed (NATS core, no persistence) is retried.
    #    This actually defeats the race rather than merely detecting it.
    subscribed = False
    for _ in range(30):  # ~15s worst case
        await _inject(nats_conn, 10.0)
        if await wait_for_redis_key(redis_client, baseline_key, timeout=0.5):
            subscribed = True
            break
    assert subscribed, (
        "detector never created the baseline after repeated warmups — it likely "
        "never subscribed; check the detector subprocess / bump AUGUR_TEST_STARTUP_WAIT_S"
    )

    # 2) Train past min_observations (15) with VARIED values so ewma_std > 0.01
    #    (the zero-variance gotcha would otherwise make deviation 0).
    for i in range(20):
        await _inject(nats_conn, 10.0 + (2.0 if i % 2 else -2.0))
        await asyncio.sleep(0.05)

    # 3) A large deviation-driven anomaly on this NON-chess entity.
    await _inject(nats_conn, 40.0)
    assert await wait_for_redis_key(
        redis_client, "augur:vigil:last_anomaly", timeout=15.0
    )
    await asyncio.sleep(0.3)  # let the final write settle

    raw = redis_client.get("augur:vigil:last_anomaly")
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
