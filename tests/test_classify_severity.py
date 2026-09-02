"""Tests for classify_severity — the gate that controls LLM activation.

If this function misclassifies severity, either the LLM fires on noise
(wasting Ollama calls) or misses genuine anomalies (defeating the purpose).
"""

from __future__ import annotations


import random

from vigil.anomaly_detector import (
    DEFAULT_THRESHOLDS,
    EntityBaseline,
    classify_severity,
)

# Use the actual project defaults for boundary tests
MEDIUM = DEFAULT_THRESHOLDS["severity_medium_sigma"]  # 2.5
HIGH = DEFAULT_THRESHOLDS["severity_high_sigma"]  # 4.0


class TestSeverityBoundaries:
    """Verify exact boundary behavior with default thresholds."""

    def test_below_medium_is_low(self) -> None:
        assert classify_severity(2.0, MEDIUM, HIGH) == "low"

    def test_at_medium_sigma_is_medium(self) -> None:
        assert classify_severity(MEDIUM, MEDIUM, HIGH) == "medium"

    def test_between_medium_and_high_is_medium(self) -> None:
        assert classify_severity(3.0, MEDIUM, HIGH) == "medium"

    def test_at_high_sigma_is_high(self) -> None:
        assert classify_severity(HIGH, MEDIUM, HIGH) == "high"

    def test_above_high_sigma_is_high(self) -> None:
        assert classify_severity(10.0, MEDIUM, HIGH) == "high"


class TestSigmaIsTheOnlyInput:
    """Severity is a pure function of deviation.

    A HalfSpaceTrees score used to be able to force high/medium independently
    of sigma. It was removed: fed a single scalar it returned a constant —
    exactly 0.0 for any value >= 1 under River's default [0, 1] feature range,
    and ~0.95 for every input once the feature was rescaled to fit — so the
    override fired either never or always, depending only on the unit the
    sensor happened to publish in.
    """

    def test_a_sub_medium_deviation_cannot_be_escalated(self) -> None:
        assert classify_severity(1.0, MEDIUM, HIGH) == "low"

    def test_just_below_medium_is_low(self) -> None:
        assert classify_severity(MEDIUM - 0.01, MEDIUM, HIGH) == "low"

    def test_just_below_high_is_medium(self) -> None:
        assert classify_severity(HIGH - 0.01, MEDIUM, HIGH) == "medium"


class TestEdgeCases:
    """Guard against degenerate inputs."""

    def test_zero_deviation_is_low(self) -> None:
        assert classify_severity(0.0, MEDIUM, HIGH) == "low"

    def test_negative_deviation_treated_as_low(self) -> None:
        # Shouldn't happen (abs value), but classify_severity should not crash
        assert classify_severity(-5.0, MEDIUM, HIGH) == "low"


class TestThresholdsMatchTheMeasuredNull:
    """The sigma thresholds are calibrated against the ESTIMATOR, not a z-table.

    `ewma_var` at `alpha` has effective sample size (2-alpha)/alpha, so
    `|value - mean| / sigma` is t-like and the Gaussian tail probabilities do
    not apply. At the previous alpha=0.3 (effective n=5.7) the nominal "2.0
    sigma" threshold fired on 14% of stationary normal input and "4.0 sigma" on
    1.1% — 190x its nominal rate, on the severity that bypasses correlation
    entirely and is exempt at the gate.

    This is the guard against changing `ewma_alpha` without re-deriving the
    thresholds: it fails if the realized tail rates drift from the design
    intent (~5% fire, ~1% medium, ~0.1% high).
    """

    @staticmethod
    def _null_rates(alpha: float, trials: int = 700, n: int = 120, burn: int = 40):
        rng = random.Random(23)
        devs = []
        for _ in range(trials):
            bl = EntityBaseline()
            for i in range(n):
                v = rng.gauss(100.0, 10.0)
                if i >= burn:
                    devs.append(bl.score(v))
                bl.update(v, alpha)
        total = len(devs)
        return {
            "fire": sum(d >= DEFAULT_THRESHOLDS["sigma_threshold"] for d in devs)
            / total,
            "medium": sum(
                d >= DEFAULT_THRESHOLDS["severity_medium_sigma"] for d in devs
            )
            / total,
            "high": sum(d >= DEFAULT_THRESHOLDS["severity_high_sigma"] for d in devs)
            / total,
        }

    def test_realized_tail_rates_match_the_design_intent(self) -> None:
        r = self._null_rates(DEFAULT_THRESHOLDS["ewma_alpha"])
        assert 0.035 <= r["fire"] <= 0.075, f"fire rate {r['fire']:.4f} off target ~5%"
        assert 0.005 <= r["medium"] <= 0.020, (
            f"medium rate {r['medium']:.4f} off target ~1%"
        )
        assert r["high"] <= 0.004, f"high rate {r['high']:.4f} off target ~0.1%"

    def test_the_old_alpha_would_now_fail_this_gate(self) -> None:
        """Pins that the check has teeth: alpha=0.3 blows every band."""
        r = self._null_rates(0.3)
        assert r["fire"] > 0.075
        assert r["high"] > 0.004
