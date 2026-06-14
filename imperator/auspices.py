"""Pure compute for the auspices read-model (the user's current situation).

No I/O, no Redis, no asyncio. `now` is injected so tests are deterministic.
"""

from __future__ import annotations

SCHEMA_VERSION = 1
_W_ANOMALY, _W_ESCALATION, _W_CORRELATION, _W_INTENSITY = 0.35, 0.30, 0.20, 0.15
# Spec §5.5: tier_int = {quiescent:0, MEDIUM:2, HIGH:3}. Nexus only ever
# publishes MEDIUM/HIGH as combined_severity (standalone LOW anomalies are
# dropped, never escalated), so LOW/unknown fall through the .get default to 0.
_TIER_INT = {"quiescent": 0, "MEDIUM": 2, "HIGH": 3}
_INTENSITY_NORM_CAP = 300.0


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def field(value, fresh: bool, now: float) -> dict:
    return {"value": value, "fresh": fresh, "as_of": now}


def salience(
    anomaly_load: float,
    escalation_tier: str,
    has_active_correlation: bool,
    intensity_ewma: float,
) -> float:
    """Bounded weighted fusion -> how much the user's state commands attention."""
    return _clamp(
        _W_ANOMALY * _clamp((anomaly_load or 0.0) / 3.0)
        + _W_ESCALATION * (_TIER_INT.get(escalation_tier, 0) / 3.0)
        + _W_CORRELATION * (1.0 if has_active_correlation else 0.0)
        + _W_INTENSITY * _clamp((intensity_ewma or 0.0) / _INTENSITY_NORM_CAP)
    )


def compute_auspices(inputs: dict, now: float, prev: dict, cfg) -> dict:
    """Fold gathered inputs into the auspices snapshot. Pure + deterministic."""

    def present(key):
        return key in inputs and inputs[key] is not None

    # The salience headline is only a current reading when at least one of the
    # attention signals it fuses is actually present; in warmup it collapses to
    # the quiescent floor (0.0) and must not be advertised fresh, or the reasoner
    # would trust a not-yet-gathered "all calm" as a real reading.
    salience_fresh = (
        present("anomaly_load")
        or present("escalation_tier")
        or present("intensity_ewma")
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now,
        "session_id": inputs.get("session_id"),
        "activity": field(inputs.get("activity"), present("activity"), now),
        "intensity": field(
            inputs.get("intensity_ewma"), present("intensity_ewma"), now
        ),
        "anomaly_load": field(inputs.get("anomaly_load"), present("anomaly_load"), now),
        "escalation_tier": field(
            inputs.get("escalation_tier"), present("escalation_tier"), now
        ),
        "active_correlations": field(
            inputs.get("active_correlations"), present("active_correlations"), now
        ),
        "last_advice_and_reception": field(
            {
                "advice": inputs.get("last_advice"),
                "reception": inputs.get("reception"),
                "latest_decision": inputs.get("latest_decision"),
            },
            present("last_advice") or present("latest_decision"),
            now,
        ),
        "pipeline_health": field(
            inputs.get("pipeline_health_rollup"), present("pipeline_health_rollup"), now
        ),
        "salience": field(
            salience(
                inputs.get("anomaly_load") or 0.0,
                inputs.get("escalation_tier") or "quiescent",
                bool(inputs.get("has_active_correlation")),
                inputs.get("intensity_ewma") or 0.0,
            ),
            salience_fresh,
            now,
        ),
    }
