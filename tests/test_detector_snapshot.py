"""Decision-time (pre-update) baseline snapshot invariant (spec §4.3).

The detector emits ``deviation_score`` from a pre-update score and must emit
``baseline_mean``/``baseline_std`` from the SAME pre-update state, so the
downstream outcome metric freezes a consistent decision-time baseline. This
pins the property the on_event fix relies on; the emitted-payload assertion is
in tests/integration/test_measurement_integration.py.
"""

from blackboard.contracts import PerceptionEvent
from detection.anomaly_detector import EntityBaseline, build_anomaly_payload


def test_payload_emits_pre_update_baseline_not_post_update():
    """Guards the spec §4.3 CRITICAL fix at the UNIT level: the emitted payload
    must carry the PRE-update baseline, not the post-update mean. The pure
    builder makes the post-update regression impossible (it only sees the
    snapshot), and this pins it."""
    bl = EntityBaseline()
    for v in (10.0, 8.0, 12.0, 9.0, 11.0) * 4:  # train, varied → std > 0.01
        bl.update(v, 0.3)
    mean_before, std_before, obs_before = (
        bl.ewma_mean,
        bl.ewma_std,
        bl.observation_count,
    )
    value = 40.0
    deviation, hst = bl.score(value)  # pre-update z-score
    bl.update(value, 0.3)  # post-update mean now differs
    assert bl.ewma_mean != mean_before  # sanity: the update moved the mean

    event = PerceptionEvent(
        domain="d",
        stream_id="d_stream",
        entity="e",
        event_type="sample",
        value=value,
        unit="u",
        context={},
        timestamp="2026-06-09T00:00:00+00:00",
        session_id="s",
    )
    payload = build_anomaly_payload(
        event,
        deviation=deviation,
        hst_score=hst,
        severity="high",
        mean_before=mean_before,
        std_before=std_before,
        obs_before=obs_before,
        drift_reset=False,
        timestamp="2026-06-09T00:00:00+00:00",
    )
    # Emitted baseline is the PRE-update snapshot, NOT the post-update mean.
    assert payload["baseline_mean"] == round(mean_before, 3)
    assert payload["baseline_mean"] != round(bl.ewma_mean, 3)
    assert payload["baseline_observation_count"] == obs_before
    # deviation_score is consistent with the emitted (pre-update) baseline.
    expected = abs(value - payload["baseline_mean"]) / payload["baseline_std"]
    assert abs(payload["deviation_score"] - round(expected, 3)) < 0.05


def test_pre_update_snapshot_is_consistent_with_deviation():
    bl = EntityBaseline()
    for _ in range(30):
        bl.update(10.0, 0.3)
    for v in (8.0, 12.0, 9.0, 11.0):  # inject spread so std > 0.01
        bl.update(v, 0.3)

    mean_before, std_before = bl.ewma_mean, bl.ewma_std
    value = 30.0
    deviation, _ = bl.score(value)  # pre-update z-score the payload emits

    # baseline_mean/std emitted alongside deviation_score MUST be these
    # pre-update values, not the post-update ones.
    assert abs(deviation - abs(value - mean_before) / std_before) < 1e-6

    bl.update(value, 0.3)  # what on_event does AFTER snapshotting
    assert bl.ewma_mean != mean_before  # post-update differs — must not be emitted
