"""Domain-agnostic surprise-reduction outcome metric (spec 2026-06-09 §1A)."""

from responsum.feedback_collector import _BehavioralTracker


def _tracker(mean, std, dev0, obs0=50, window=3, min_post_obs=2, min_obs=15):
    t = _BehavioralTracker(
        baseline_std=std,
        deviation_at_decision=dev0,
        baseline_observation_count=obs0,
        window=window,
        min_baseline_std=0.01,
        trend_bonus=0.1,
        min_post_obs=min_post_obs,
        min_observations=min_obs,
    )
    t.baseline_mean = mean
    return t


def test_full_return_to_baseline_scores_high():
    t = _tracker(10.0, 2.0, 3.0)
    for v in (10.0, 10.0, 10.0):
        t.add_post_move(v)
    assert t.finalized and not t.unmeasurable
    assert t.behavioral_score >= 0.9


def test_no_return_scores_low():
    t = _tracker(10.0, 2.0, 3.0)
    for v in (16.0, 16.0, 16.0):  # stays 3σ off
        t.add_post_move(v)
    assert t.finalized and t.behavioral_score <= 0.1


def test_half_surprise_removed_scores_mid():
    t = _tracker(0.0, 1.0, 4.0)
    for v in (2.83, 2.83, 2.83):  # ~8/16 surprise
        t.add_post_move(v)
    assert 0.4 <= t.behavioral_score <= 0.6


def test_direction_symmetry():
    a = _tracker(10.0, 2.0, 3.0)
    b = _tracker(10.0, 2.0, 3.0)
    for v in (11.0, 11.0, 11.0):
        a.add_post_move(v)
    for v in (9.0, 9.0, 9.0):
        b.add_post_move(v)
    assert a.behavioral_score == b.behavioral_score


def test_trend_bonus_only_when_net_positive_and_shrinking():
    shrink = _tracker(10.0, 2.0, 3.0)
    for v in (12.0, 11.0, 10.0):
        shrink.add_post_move(v)
    flat = _tracker(10.0, 2.0, 3.0)
    for v in (11.0, 11.0, 11.0):
        flat.add_post_move(v)
    assert shrink.behavioral_score >= flat.behavioral_score


def test_trend_bonus_does_not_rescue_net_negative():
    # surprise grows overall (well past dev0=3) but last < first → still ~0.
    t = _tracker(10.0, 2.0, 3.0)
    for v in (24.0, 22.0, 20.0):  # 7σ→6σ→5σ, all >> dev0
        t.add_post_move(v)
    assert t.behavioral_score == 0.0


def test_degenerate_std_is_unmeasurable():
    t = _tracker(10.0, 0.0, 3.0)  # σ below floor
    for v in (10.0, 10.0, 10.0):
        t.add_post_move(v)
    assert t.finalized and t.unmeasurable and t.behavioral_score == 0.5


def test_zero_decision_deviation_is_unmeasurable():
    t = _tracker(10.0, 2.0, 0.0)  # dev0 below MIN_DECISION_DEVIATION
    for v in (10.0, 10.0, 10.0):
        t.add_post_move(v)
    assert t.finalized and t.unmeasurable and t.behavioral_score == 0.5


def test_untrained_baseline_is_unmeasurable():
    t = _tracker(10.0, 2.0, 3.0, obs0=5, min_obs=15)
    for v in (10.0, 10.0, 10.0):
        t.add_post_move(v)
    assert t.finalized and t.unmeasurable and t.behavioral_score == 0.5


def test_below_min_post_obs_not_finalized_not_unmeasurable():
    t = _tracker(10.0, 2.0, 3.0, min_post_obs=2)
    t.add_post_move(10.0)
    t._compute_behavioral_score()
    assert not t.finalized and not t.unmeasurable


def test_metric_version_is_2():
    t = _tracker(10.0, 2.0, 3.0)
    assert t.outcome_metric_version == 2


def test_window_respected():
    t = _BehavioralTracker(
        baseline_std=2.0,
        deviation_at_decision=3.0,
        baseline_observation_count=50,
        window=2,
        min_baseline_std=0.01,
        trend_bonus=0.1,
        min_post_obs=2,
        min_observations=15,
    )
    t.baseline_mean = 10.0
    t.add_post_move(10.0)
    assert not t.finalized
    t.add_post_move(10.0)
    assert t.finalized
