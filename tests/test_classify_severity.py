"""Tests for classify_severity — the gate that controls LLM activation.

If this function misclassifies severity, either the LLM fires on noise
(wasting Ollama calls) or misses genuine anomalies (defeating the purpose).
"""

from __future__ import annotations


from vigil.anomaly_detector import classify_severity, DEFAULT_THRESHOLDS

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
