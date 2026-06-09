"""Decision-time (pre-update) baseline snapshot invariant (spec §4.3).

The detector emits ``deviation_score`` from a pre-update score and must emit
``baseline_mean``/``baseline_std`` from the SAME pre-update state, so the
downstream outcome metric freezes a consistent decision-time baseline. This
pins the property the on_event fix relies on; the emitted-payload assertion is
in tests/integration/test_measurement_integration.py.
"""

from detection.anomaly_detector import EntityBaseline


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
