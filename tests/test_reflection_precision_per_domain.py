"""Tests for per-domain analyze_precision refactor."""

from tabula.config import AugurConfig
from disciplina.reflection_engine import analyze_precision


def _cfg() -> AugurConfig:
    return AugurConfig.from_env()


def _thresholds(domain: str, sigma: float = 2.0) -> dict:
    return {"sigma_threshold": sigma, "ewma_alpha": 0.3, "hst_threshold": 0.7}


def test_precision_empty_feedback():
    feedback = {"advice_events": [], "session_summary": {"total_advice": 0}}
    result = analyze_precision(feedback, {}, _cfg())
    assert result["per_domain"] == {}
    assert result["domains_evaluated"] == []


def test_precision_standalone_chess_only():
    feedback = {
        "advice_events": [
            {
                "domain": "chess",
                "explicit_rating": "y",
                "behavioral_score": 0.8,
                "correlation_found": False,
            },
            {
                "domain": "chess",
                "explicit_rating": "y",
                "behavioral_score": 0.8,
                "correlation_found": False,
            },
        ],
        "session_summary": {"total_advice": 2},
    }
    result = analyze_precision(feedback, {"chess": _thresholds("chess")}, _cfg())
    assert "chess" in result["per_domain"]
    assert result["per_domain"]["chess"]["total_anomalies"] == 2.0
    assert result["per_domain"]["chess"]["useful"] == 2.0
    assert result["per_domain"]["chess"]["precision_ratio"] == 1.0
    assert result["per_domain"]["chess"]["action"] == "lower_sigma"  # high precision


def test_precision_correlated_distributes_across_domains():
    feedback = {
        "advice_events": [
            {
                "domain": "chess",
                "explicit_rating": "y",
                "behavioral_score": 0.8,
                "correlation_found": True,
                "involved_domains": ["chess", "typing"],
            },
            {
                "domain": "chess",
                "explicit_rating": "y",
                "behavioral_score": 0.8,
                "correlation_found": True,
                "involved_domains": ["chess", "typing"],
            },
            {
                "domain": "chess",
                "explicit_rating": "y",
                "behavioral_score": 0.8,
                "correlation_found": True,
                "involved_domains": ["chess", "typing"],
            },
            {
                "domain": "chess",
                "explicit_rating": "y",
                "behavioral_score": 0.8,
                "correlation_found": True,
                "involved_domains": ["chess", "typing"],
            },
        ],
        "session_summary": {"total_advice": 4},
    }
    thresholds = {
        "chess": _thresholds("chess"),
        "typing": _thresholds("typing"),
    }
    result = analyze_precision(feedback, thresholds, _cfg())
    # 4 events, 0.5 weight each → 2.0 weighted total per domain
    assert result["per_domain"]["chess"]["total_anomalies"] == 2.0
    assert result["per_domain"]["typing"]["total_anomalies"] == 2.0


def test_precision_below_threshold_signal_no_action():
    """Domain with weighted_total < 2.0 has no action."""
    feedback = {
        "advice_events": [
            {
                "domain": "chess",
                "explicit_rating": "y",
                "behavioral_score": 0.8,
                "correlation_found": True,
                "involved_domains": ["chess", "typing"],
            },
        ],
        "session_summary": {"total_advice": 1},
    }
    thresholds = {"chess": _thresholds("chess"), "typing": _thresholds("typing")}
    result = analyze_precision(feedback, thresholds, _cfg())
    # 0.5 weight per domain → below 2.0 threshold
    assert result["per_domain"]["chess"]["action"] == "none"


def test_precision_low_per_domain_raises_sigma():
    feedback = {
        "advice_events": [
            {
                "domain": "chess",
                "explicit_rating": "n",
                "behavioral_score": 0.2,
                "correlation_found": False,
            },
            {
                "domain": "chess",
                "explicit_rating": "n",
                "behavioral_score": 0.2,
                "correlation_found": False,
            },
            {
                "domain": "chess",
                "explicit_rating": "n",
                "behavioral_score": 0.2,
                "correlation_found": False,
            },
        ],
        "session_summary": {"total_advice": 3},
    }
    result = analyze_precision(feedback, {"chess": _thresholds("chess")}, _cfg())
    assert result["per_domain"]["chess"]["action"] == "raise_sigma"
    assert result["per_domain"]["chess"]["sigma_after"] > 2.0
