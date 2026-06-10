"""Per-arm + per-domain reliability audit (spec §7)."""

from tabula.config import AugurConfig
from reasoning.reflection_engine import _behavioral_audit_per_arm


def _row(
    rating,
    score,
    domain="chess",
    finalized=True,
    unmeasurable=False,
    arm="fired",
    version=2,
):
    return {
        "explicit_rating": rating,
        "behavioral_score": score,
        "behavioral_finalized": finalized,
        "unmeasurable": unmeasurable,
        "outcome_metric_version": version,
        "domain": domain,
        "_arm": arm,
    }


def test_excludes_unmeasurable_and_no_response():
    cfg = AugurConfig(gate_behavioral_min_samples=2)
    rows = [
        _row("y", 0.9),
        _row("n", 0.1),
        _row("y", 0.5, unmeasurable=True),
        _row("no_response", 0.8),
    ]
    assert _behavioral_audit_per_arm(rows, cfg)["overall"]["genuine_samples"] == 2


def test_excludes_old_metric_version():
    cfg = AugurConfig(gate_behavioral_min_samples=2)
    rows = [
        _row("y", 0.9),
        _row("n", 0.1),
        _row("y", 0.8, version=1),
        _row("n", 0.2, version=None),
    ]
    out = _behavioral_audit_per_arm(rows, cfg)["overall"]
    assert out["genuine_samples"] == 2  # only the v2 rows
    assert out["excluded_old_version"] == 2


def test_per_domain_and_per_arm_breakdown():
    cfg = AugurConfig(gate_behavioral_min_samples=2)
    rows = [
        _row("y", 0.9, domain="chess", arm="fired"),
        _row("n", 0.1, domain="chess", arm="fired"),
        _row("y", 0.8, domain="typing", arm="withheld"),
        _row("n", 0.2, domain="typing", arm="withheld"),
    ]
    out = _behavioral_audit_per_arm(rows, cfg)
    assert "chess" in out["per_domain"] and "typing" in out["per_domain"]
    assert "fired" in out["per_arm"] and "withheld" in out["per_arm"]
    # perfectly-correlated chess slice → correlation 1.0
    assert out["per_domain"]["chess"]["correlation"] == 1.0
