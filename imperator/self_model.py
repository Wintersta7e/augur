"""Pure compute for the self-model read-model (Augur's own state). No I/O."""

from __future__ import annotations

SCHEMA_VERSION = 1
_W_PRECISION, _W_UTILITY, _W_DISMISS, _W_COVERAGE, _W_HEALTH = (
    0.30,
    0.25,
    0.20,
    0.15,
    0.10,
)
_BLIND_PENALTY = 0.05


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def field(value, fresh: bool, now: float) -> dict:
    return {"value": value, "fresh": fresh, "as_of": now}


def competence(
    precision: float,
    utility: float,
    utility_no_data: bool,
    dismissal_rate: float,
    coverage_depth: float,
    health_score: float,
    n_blind_spots: int,
    coverage_no_data: bool = False,
) -> float:
    """Inward headline. Higher = better; the inverse = room to grow."""
    utility_adj = 0.5 if utility_no_data else (utility or 0.0)
    coverage_adj = 0.5 if coverage_no_data else (coverage_depth or 0.0)
    return _clamp(
        _W_PRECISION * (precision or 0.0)
        + _W_UTILITY * utility_adj
        + _W_DISMISS * (1.0 - (dismissal_rate or 0.0))
        + _W_COVERAGE * coverage_adj
        + _W_HEALTH * (health_score or 0.0)
        - _BLIND_PENALTY * min(n_blind_spots, 5) / 5.0
    )


def compute_self_model(inputs: dict, now: float, prev: dict, cfg) -> dict:
    def present(key):
        return key in inputs and inputs[key] is not None

    report_fresh = present("precision")
    coverage = inputs.get("coverage") or {}
    coverage_no_data = bool(inputs.get("coverage_no_data"))
    blind = inputs.get("blind_spots") or []

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now,
        "session_id": inputs.get("session_id"),
        "precision": field(inputs.get("precision"), report_fresh, now),
        "utility": field(inputs.get("utility"), present("utility"), now),
        "mrt": field(inputs.get("mrt"), present("mrt"), now),
        "suppression_rate": field(
            inputs.get("suppression_rate"), present("suppression_rate"), now
        ),
        "dismissal_rate": field(
            inputs.get("dismissal_rate"), present("dismissal_rate"), now
        ),
        "advice_volume": field(
            inputs.get("advice_volume"), present("advice_volume"), now
        ),
        "pipeline_health": field(
            inputs.get("pipeline_health_full"), present("pipeline_health_full"), now
        ),
        "coverage": field(coverage, present("coverage") and not coverage_no_data, now),
        "blind_spots": field(blind, True, now),
        "recent_self_tuning": field(
            inputs.get("recent_self_tuning"), present("recent_self_tuning"), now
        ),
        "competence": field(
            competence(
                inputs.get("precision") or 0.0,
                inputs.get("utility") or 0.0,
                bool(inputs.get("utility_no_data")),
                inputs.get("dismissal_rate") or 0.0,
                coverage.get("coverage_depth", 0.0),
                inputs.get("health_score") or 0.0,
                len(blind),
                coverage_no_data,
            ),
            True,
            now,
        ),
    }
