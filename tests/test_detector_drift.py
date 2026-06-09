"""River drift detector + deliberate baseline reset (spec 1C)."""

from detection.anomaly_detector import EntityBaseline


def _warm(bl, center, n, spread=2.0, alpha=0.3):
    # Varied values so ewma_std stays > 0.01 (the detector skips a degenerate-σ
    # feed — the zero-variance gotcha). A constant stream would starve the
    # drift detector of any scale-free signal.
    for i in range(n):
        bl.update(center + (spread if i % 2 else -spread), alpha)


def test_mean_shift_triggers_reset():
    bl = EntityBaseline()
    bl.enable_drift(
        "adwin", min_observations=15, cooldown_obs=30, restart_std_factor=1.0
    )
    _warm(bl, 10.0, 40)  # stable around 10
    for _ in range(60):  # abrupt shift to 30
        bl.update(30.0, 0.3)
    assert bl.drift_resets >= 1
    # After reset the mean tracks the new regime, not a runaway blend.
    assert abs(bl.ewma_mean - 30.0) < 5.0


def test_no_reset_while_warming():
    bl = EntityBaseline()
    bl.enable_drift(
        "adwin", min_observations=15, cooldown_obs=30, restart_std_factor=1.0
    )
    for i in range(10):  # below min_observations, wild early values
        bl.update(float(i) * 5, 0.3)
    assert bl.drift_resets == 0


def test_cooldown_blocks_rereset():
    bl = EntityBaseline()
    bl.enable_drift(
        "adwin", min_observations=5, cooldown_obs=1000, restart_std_factor=1.0
    )
    _warm(bl, 10.0, 20)
    for _ in range(40):
        bl.update(40.0, 0.3)
    first = bl.drift_resets
    for _ in range(5):  # within cooldown
        bl.update(80.0, 0.3)
    assert bl.drift_resets == first


def test_disabled_never_resets():
    bl = EntityBaseline()  # drift not enabled
    _warm(bl, 10.0, 20)
    for _ in range(40):
        bl.update(40.0, 0.3)
    assert getattr(bl, "drift_resets", 0) == 0


def test_reset_variance_positive():
    bl = EntityBaseline()
    bl.enable_drift("adwin", min_observations=5, cooldown_obs=0, restart_std_factor=1.0)
    _warm(bl, 10.0, 20)
    for _ in range(40):
        bl.update(40.0, 0.3)
    assert bl.ewma_var > 0.0  # restart variance, never zeroed
