"""Metric-compat fixes (spec §7): IPW excludes unmeasurable/old-version/rated
rows; utility + correlation treat a finalized 0.0 as a valid negative outcome."""

from blackboard.config import AugurConfig
from reasoning.reflection_engine import _mrt_ipw_readout, analyze_utility


def test_ipw_excludes_unmeasurable_old_version_and_rated():
    emissions = [{"mrt_eligible": True, "decision_id": "f1", "p_fire": 0.5}]
    fired = [
        {
            "decision_id": "f1",
            "behavioral_finalized": True,
            "behavioral_score": 1.0,
            "p_fire": 0.5,
            "outcome_metric_version": 2,
            "unmeasurable": False,
        }
    ]
    silences = [
        {"mrt_eligible": True, "decision_id": "w1", "p_withhold": 0.5},
        {"mrt_eligible": True, "decision_id": "w2", "p_withhold": 0.5},  # unmeasurable
        {"mrt_eligible": True, "decision_id": "w3", "p_withhold": 0.5},  # old version
        {"mrt_eligible": True, "decision_id": "w4", "p_withhold": 0.5},  # rated
    ]
    withheld = [
        {
            "decision_id": "w1",
            "behavioral_finalized": True,
            "behavioral_score": 0.0,
            "p_withhold": 0.5,
            "outcome_metric_version": 2,
            "unmeasurable": False,
            "withheld_rating_p": None,
        },
        {
            "decision_id": "w2",
            "behavioral_finalized": True,
            "behavioral_score": 0.5,
            "p_withhold": 0.5,
            "outcome_metric_version": 2,
            "unmeasurable": True,
            "withheld_rating_p": None,
        },
        {
            "decision_id": "w3",
            "behavioral_finalized": True,
            "behavioral_score": 0.9,
            "p_withhold": 0.5,
            "unmeasurable": False,
            "withheld_rating_p": None,
        },  # no outcome_metric_version → treated as old (v1)
        {
            "decision_id": "w4",
            "behavioral_finalized": True,
            "behavioral_score": 0.2,
            "p_withhold": 0.5,
            "outcome_metric_version": 2,
            "unmeasurable": False,
            "withheld_rating_p": 0.5,
        },  # rated → stratified out
    ]
    out = _mrt_ipw_readout(emissions, silences, fired, withheld)
    assert out["withheld_n"] == 1  # only w1 survives
    assert out["fired_n"] == 1


def test_utility_treats_finalized_zero_as_valid():
    feedback = {
        "advice_events": [
            {
                "explicit_rating": "no_response",
                "behavioral_score": 0.0,
                "behavioral_finalized": True,
                "unmeasurable": False,
                "domain": "typing",
            }
        ]
    }
    res = analyze_utility(feedback, AugurConfig())
    # behavioral component must reflect the 0.0 (not be ignored as "missing")
    assert res["behavioral_component"] < 0.5


def test_utility_ignores_unmeasurable_and_unfinalized():
    feedback = {
        "advice_events": [
            {
                "explicit_rating": "no_response",
                "behavioral_score": 0.0,
                "behavioral_finalized": True,
                "unmeasurable": True,  # excluded
                "domain": "typing",
            },
            {
                "explicit_rating": "no_response",
                "behavioral_score": 0.0,
                "behavioral_finalized": False,  # excluded
                "unmeasurable": False,
                "domain": "typing",
            },
        ]
    }
    res = analyze_utility(feedback, AugurConfig())
    # No finalized+measurable behavioral rows → fall back to 0.5
    assert res["behavioral_component"] == 0.5
