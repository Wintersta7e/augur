"""Praesagium pattern-mining math: Wilson bound, trial collapse, one-B-per-trial
match counting, two-pass window recount, session-conditional lift, lag stability,
and the full ``mine_corpus`` promotion gate.

Spec: docs/superpowers/specs/2026-07-09-praesagium-design.md Sec 4.2-4.5 + Sec 11 PR5.
Pure functions only -- no Redis/NATS fixtures.

Every hand-computed expectation (Wilson bounds, pattern ids, P90, lift) is
independently precomputed and hardcoded below -- the tests do NOT mirror the
implementation's own call, so a regression in the formula is caught rather than
masked.

Conventions pinned here (must match patterns.py docstrings):
  * P90(v) = sorted(v)[min(len-1, ceil(0.9*len)-1)]  (nearest-rank, rounded up).
  * Trial collapse reference = the *trial start* (NOT the previous occurrence):
    an A within lag_min of the trial's START folds in; a burst wider than lag_min
    re-arms (see test_collapse_trial_start_reference). Regression bursts are
    therefore constructed to span <= lag_min so they fold to ONE trial (PR5c).
  * Stability IQR = statistics.quantiles(lags, n=4)[2] - [0] (exclusive method).
"""

import math

import pytest

from praesagium.patterns import (
    _FINITE_LIFT_CAP,
    collapse_trials,
    count_matches,
    merge_blob,
    mine_corpus,
    pattern_id,
    wilson_lower,
)
from tabula.config import AugurConfig

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _ep(key: str, sev: str, t: float) -> dict:
    return {"k": key, "s": sev, "t": float(t)}


def _cfg(**over) -> AugurConfig:
    return AugurConfig(**over)


# ---------------------------------------------------------------------------
# wilson_lower -- hand-computed pins (independent of implementation)
# ---------------------------------------------------------------------------


def test_wilson_3_of_3_spec_pin():
    # The spec-pinned borderline: a 3-of-3 streak yields ~0.4385, passing the
    # default conf floor 0.4 barely (deliberate -- then sits in probation).
    assert wilson_lower(3, 3) == pytest.approx(0.438494, abs=1e-3)


def test_wilson_30_of_35():
    assert wilson_lower(30, 35) == pytest.approx(0.706241, abs=1e-3)


def test_wilson_3_of_8_pseudo_replication_pin():
    assert wilson_lower(3, 8) == pytest.approx(0.136842, abs=1e-3)


def test_wilson_9_of_10():
    assert wilson_lower(9, 10) == pytest.approx(0.595844, abs=1e-3)


def test_wilson_100_of_100():
    assert wilson_lower(100, 100) == pytest.approx(0.963005, abs=1e-3)


def test_wilson_monotone_below_phat():
    # Wilson lower bound is always strictly below the point estimate for k<n.
    assert wilson_lower(3, 5) < 3 / 5


def test_wilson_le_phat_always():
    for k, n in [(1, 2), (3, 4), (5, 10), (7, 7), (0, 3), (50, 60)]:
        assert wilson_lower(k, n) <= k / n + 1e-12


def test_wilson_zero_successes_is_zero():
    assert wilson_lower(0, 5) == pytest.approx(0.0, abs=1e-9)


def test_wilson_n_zero_guarded():
    # n==0 must not divide-by-zero; returns 0.0 (no evidence).
    assert wilson_lower(0, 0) == 0.0


def test_wilson_z_param_widens_interval():
    # A larger z (99% vs 95%) gives a lower bound.
    assert wilson_lower(3, 5, z=2.576) < wilson_lower(3, 5, z=1.96)


# ---------------------------------------------------------------------------
# pattern_id -- stable sha256(f"{A}→{B}")[:12]
# ---------------------------------------------------------------------------


def test_pattern_id_hardcoded():
    # sha256("typing:user→activity:app")[:12], computed independently.
    assert pattern_id("typing:user", "activity:app") == "5655efe4e3f1"


def test_pattern_id_hardcoded_short_keys():
    assert pattern_id("A", "B") == "ce55591040ed"


def test_pattern_id_is_12_hex():
    pid = pattern_id("typing:user", "activity:app")
    assert len(pid) == 12
    int(pid, 16)  # valid hex, raises otherwise


def test_pattern_id_stable_across_calls():
    assert pattern_id("x:1", "y:2") == pattern_id("x:1", "y:2")


def test_pattern_id_order_sensitive():
    assert pattern_id("A", "B") != pattern_id("B", "A")


# ---------------------------------------------------------------------------
# collapse_trials -- burst folding (trial-start reference, inclusive boundary)
# ---------------------------------------------------------------------------


def test_collapse_burst_is_one_trial():
    assert collapse_trials([0, 2, 4, 100], 10) == [0, 100]


def test_collapse_trial_start_reference():
    # Reference is the TRIAL START, not the previous occurrence: a slow drift
    # spaced <= lag_min re-arms once it leaves the start's lag_min radius.
    # Chaining (prev-occurrence) would collapse all four to [0]; trial-start
    # yields [0, 16] (16 is >10 from start 0 -> new trial; 24 folds into 16).
    assert collapse_trials([0, 8, 16, 24], 10) == [0, 16]


def test_collapse_boundary_inclusive():
    # "within lag_min" is inclusive: exactly lag_min after the start folds in.
    assert collapse_trials([0, 10, 21], 10) == [0, 21]


def test_collapse_empty():
    assert collapse_trials([], 10) == []


def test_collapse_single():
    assert collapse_trials([5], 10) == [5]


def test_collapse_all_spread():
    assert collapse_trials([0, 100, 200], 10) == [0, 100, 200]


def test_collapse_sorts_unsorted_input():
    assert collapse_trials([100, 0, 4, 2], 10) == [0, 100]


# ---------------------------------------------------------------------------
# count_matches -- one B satisfies at most one trial, smallest qualifying lag
# ---------------------------------------------------------------------------


def test_count_one_b_one_trial():
    assert count_matches([0, 100], [30], 10, 900) == (1, [30])


def test_count_b_consumed_by_at_most_one_trial():
    # Two trials, a single B -> exactly one success (oldest trial consumes it).
    assert count_matches([0, 20], [25], 10, 900) == (1, [25])


def test_count_smallest_qualifying_lag():
    # Among qualifying B's the earliest (smallest lag) is taken.
    assert count_matches([0], [15, 50], 10, 900) == (1, [15])


def test_count_too_early_no_match():
    # lag == lo is NOT in the open-lower window (lag_min, hi].
    assert count_matches([0], [10], 10, 900) == (0, [])


def test_count_lower_bound_exclusive():
    assert count_matches([0], [11], 10, 900) == (1, [11])


def test_count_upper_bound_inclusive():
    assert count_matches([0], [900], 10, 900) == (1, [900])
    assert count_matches([0], [901], 10, 900) == (0, [])


def test_count_too_far_deferred_to_later_trial():
    # 950 is out of window for the trial at 0 (lag 950>900) but in window for
    # the trial at 500 (lag 450) -- the pointer must not discard it early.
    assert count_matches([0, 500], [950], 10, 900) == (1, [450])


def test_count_multiple_trials_each_one_b():
    assert count_matches([0, 300, 600], [100, 400, 700], 10, 900) == (
        3,
        [100, 100, 100],
    )


def test_count_sorts_unsorted_b_times():
    assert count_matches([0], [50, 15], 10, 900) == (1, [15])


def test_count_no_b_no_success():
    assert count_matches([0, 100, 200], [], 10, 900) == (0, [])


# ---------------------------------------------------------------------------
# mine_corpus -- clean promotion + full artifact shape
# ---------------------------------------------------------------------------

# 5 sessions: A(low)@0 -> B(medium)@50, filler C(low)@2000 to enlarge D_s so the
# session-conditional lift is comfortably above 1.5. n=k=5, W=62.5.
_A = "typing:user"
_B = "activity:app"
_C = "gamma:filler"


def _basic_corpus() -> dict:
    corpus = {}
    for i in range(5):
        sid = f"s{i}"
        corpus[sid] = [
            _ep(_A, "low", 0),
            _ep(_B, "medium", 50),
            _ep(_C, "low", 2000),
        ]
    return corpus


def test_basic_promotion_artifact_fields():
    result = mine_corpus(_basic_corpus(), _cfg())
    pid = pattern_id(_A, _B)  # "5655efe4e3f1"
    assert pid == "5655efe4e3f1"
    assert set(result.keys()) == {pid}  # only (A,B) promotes; fillers inert
    art = result[pid]
    # every spec Sec 4.5 field present
    assert set(art.keys()) == {
        "pattern_id",
        "antecedent",
        "consequent",
        "window_s",
        "support_sessions",
        "n",
        "k",
        "conf",
        "conf_lower",
        "lift",
        "lag_median_s",
        "lag_p90_s",
        "status",
        "hit_rate",
        "resolutions",
        "created_at",
        "mined_at",
        "retired_at",
        "retired_reason",
        "repass_streak",
    }
    assert art["pattern_id"] == pid
    assert art["antecedent"] == _A
    assert art["consequent"] == _B
    assert art["window_s"] == pytest.approx(62.5)  # P90([50]*5)=50 -> 50*1.25
    assert art["support_sessions"] == 5
    assert art["n"] == 5
    assert art["k"] == 5
    assert art["conf"] == pytest.approx(1.0)
    assert art["conf_lower"] == pytest.approx(0.565506, abs=1e-3)  # wilson(5,5)
    assert art["lift"] == pytest.approx(32.5, rel=0.02)
    assert art["lag_median_s"] == pytest.approx(50.0)
    assert art["lag_p90_s"] == pytest.approx(50.0)
    # lifecycle defaults (Task 5's miner stamps created_at/mined_at + merges)
    assert art["status"] == "provisional"
    assert art["hit_rate"] is None
    assert art["resolutions"] == 0
    assert art["created_at"] is None
    assert art["mined_at"] is None
    assert art["retired_at"] is None
    assert art["retired_reason"] is None
    assert art["repass_streak"] == 0


def test_status_always_provisional():
    for art in mine_corpus(_basic_corpus(), _cfg()).values():
        assert art["status"] == "provisional"


def test_empty_corpus_yields_nothing():
    assert mine_corpus({}, _cfg()) == {}


# ---------------------------------------------------------------------------
# REGRESSION 1 -- activity confound (session-conditional lift rejects it)
# ---------------------------------------------------------------------------


def _activity_confound_corpus() -> dict:
    """10 'active' sessions: A(low) every 300s (0..2700), B(medium) every 100s
    (100..3000, 30/session). Each A deterministically catches a B at lag 100 ->
    p_hat=1.0 with the near-certain B rate, so the honest lift collapses to ~1.40.
    Plus 20 empty sessions (skipped: <2 episodes) that a *pooled* null would have
    used to deflate lambda -- the session-conditional lambda ignores them.

    lambda_B = 300 / (10 * 3000) = 0.01 ; W = P90([100]*100)*1.25 = 125 ;
    P0 = 1 - exp(-1.25) = 0.7135 ; lift = 1.0/0.7135 = 1.4016 < 1.5 -> REJECT.
    """
    corpus = {}
    for s in range(10):
        eps = [_ep("desk:A", "low", 300 * i) for i in range(10)]  # 0..2700
        eps += [_ep("desk:B", "medium", 100 * j) for j in range(1, 31)]  # 100..3000
        corpus[f"active{s}"] = eps
    for s in range(20):
        corpus[f"empty{s}"] = [_ep("noise:x", "low", 0)]  # 1 episode -> skipped
    return corpus


def test_regression_activity_confound_promotes_nothing():
    # Passes support (10), confidence (p_hat=1), and stability (IQR 0) -- ONLY the
    # session-conditional lift (1.40 < 1.5) blocks it. Deleting the lift clause
    # would promote (desk:A, desk:B); this asserts it does not.
    result = mine_corpus(_activity_confound_corpus(), _cfg())
    assert pattern_id("desk:A", "desk:B") not in result
    assert result == {}


# ---------------------------------------------------------------------------
# REGRESSION 2 -- pseudo-replication (trial collapse + padding lowers Wilson)
# ---------------------------------------------------------------------------


def _pseudo_replication_corpus() -> dict:
    """3 burst sessions: 10 A(low) tightly within lag_min (0..9, span 9 <= 10 ->
    ONE trial), then one B(medium)@30. Plus 5 sessions with a single lone A and no
    B (a low filler at 100 clears the <2-episode skip floor).

    Trial collapse -> n = 3 + 5 = 8, k = 3. wilson(3,8) = 0.137 < 0.4 -> REJECT.
    """
    corpus = {}
    for s in range(3):
        eps = [_ep("burst:A", "low", float(i)) for i in range(10)]  # 0..9
        eps.append(_ep("burst:B", "medium", 30))
        corpus[f"burst{s}"] = eps
    for s in range(5):
        corpus[f"lone{s}"] = [
            _ep("burst:A", "low", 0),
            _ep("noise:c", "low", 100),  # unrelated, clears <2-episode floor
        ]
    return corpus


def test_regression_pseudo_replication_not_promoted():
    # Confidence gate: wilson(3,8)=0.137 < 0.4. (Support=3 and lift both PASS, so
    # deleting the Wilson clause would promote -- this pins that it does not.)
    result = mine_corpus(_pseudo_replication_corpus(), _cfg())
    assert pattern_id("burst:A", "burst:B") not in result
    assert result == {}


# ---------------------------------------------------------------------------
# collapse deflation at the corpus level (uniquely exercises trial collapse)
# ---------------------------------------------------------------------------


def _burst_with_many_b_corpus() -> dict:
    """3 sessions, each a tight burst of 10 A(low)@0..9 (ONE trial) plus 10
    B(medium)@[30,40,...,120] all inside the window. WITH collapse: n=3, k=3,
    conf_lower ~= wilson(3,3)=0.4385. WITHOUT collapse each burst would be 10
    trials each catching a distinct B -> conf_lower ~= wilson(30,30)=0.88.
    """
    corpus = {}
    for s in range(3):
        eps = [_ep("bx:A", "low", float(i)) for i in range(10)]  # 0..9 -> 1 trial
        eps += [_ep("bx:B", "medium", 30 + 10 * j) for j in range(10)]  # 30..120
        corpus[f"b{s}"] = eps
    return corpus


def test_collapse_deflates_confidence_to_honest_value():
    # cfg loosened only on lift so collapse/confidence are what we observe.
    result = mine_corpus(_burst_with_many_b_corpus(), _cfg(praesagium_lift_min=1.0))
    pid = pattern_id("bx:A", "bx:B")
    assert pid in result
    art = result[pid]
    assert art["n"] == 3  # one collapsed trial per session
    assert art["k"] == 3
    # honest (collapsed) confidence, NOT the inflated ~0.88 of 30 fake trials.
    assert art["conf_lower"] == pytest.approx(0.438494, abs=1e-3)


# ---------------------------------------------------------------------------
# TWO-PASS recount -- all promotion stats use the (lag_min, W] numbers
# ---------------------------------------------------------------------------


def _two_pass_corpus() -> dict:
    """10 sessions, each A(low)@0 -> B(medium)@lag. Nine lags are 20..100; the
    tenth is 850. Pass-1 window (10,900] admits all ten (k1=10). P90 of the ten
    lags = 100 -> W = 125. Pass-2 window (10,125] drops the 850 success (k2=9).
    """
    lags = [20, 30, 40, 50, 60, 70, 80, 90, 100, 850]
    corpus = {}
    for i, lag in enumerate(lags):
        corpus[f"tp{i}"] = [_ep("tp:A", "low", 0), _ep("tp:B", "medium", float(lag))]
    return corpus


def test_two_pass_recount_uses_pass2_numbers():
    # lift lowered so the recount is isolated; lag-850 success must be dropped.
    result = mine_corpus(_two_pass_corpus(), _cfg(praesagium_lift_min=1.2))
    pid = pattern_id("tp:A", "tp:B")
    assert pid in result
    art = result[pid]
    assert art["window_s"] == pytest.approx(125.0)  # P90=100 * 1.25
    assert art["n"] == 10
    assert art["k"] == 9  # NOT 10 -- the lag-850 trial fails pass 2
    assert art["conf"] == pytest.approx(0.9)  # 9/10, not 10/10
    assert art["conf_lower"] == pytest.approx(0.595844, abs=1e-3)  # wilson(9,10)
    # lag stats are computed over pass-2 lags only (850 excluded)
    assert art["lag_median_s"] == pytest.approx(60.0)
    assert art["lag_p90_s"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# STABILITY -- IQR rule (k>=4) and range proxy (k<4)
# ---------------------------------------------------------------------------


def _scattered_corpus(lags: list[float], key_a: str, key_b: str) -> dict:
    """One session per lag: A(low)@0 -> B(medium)@lag, plus a far filler(low)@5000
    so D_s is large and lift never becomes the rejecting gate (isolates stability).
    """
    corpus = {}
    for i, lag in enumerate(lags):
        corpus[f"{key_a}{i}"] = [
            _ep(key_a, "low", 0),
            _ep(key_b, "medium", float(lag)),
            _ep("far:z", "low", 5000),
        ]
    return corpus


def test_stability_iqr_rejects_scattered_lags():
    # k=8, lags [10..15, 800, 850]: IQR=592.5 > median(13.5)*2=27 -> REJECT.
    # Support(8), confidence(1.0), lift(~6) all PASS; only stability blocks it.
    result = mine_corpus(
        _scattered_corpus([10, 11, 12, 13, 14, 15, 800, 850], "iqr:A", "iqr:B"),
        _cfg(),
    )
    assert pattern_id("iqr:A", "iqr:B") not in result


def test_stability_iqr_control_promotes():
    # Same shape, tight lags [10..17]: IQR=4.5 <= 27 -> stability PASSES.
    result = mine_corpus(
        _scattered_corpus([10, 11, 12, 13, 14, 15, 16, 17], "iqr:A", "iqr:B"),
        _cfg(),
    )
    assert pattern_id("iqr:A", "iqr:B") in result


def test_stability_range_proxy_rejects_at_low_k():
    # k=3 (<4): range proxy. lags [15,20,600]: 585 > median(20)*2=40 -> REJECT.
    result = mine_corpus(
        _scattered_corpus([15, 20, 600], "rng:A", "rng:B"),
        _cfg(),
    )
    assert pattern_id("rng:A", "rng:B") not in result


def test_stability_range_proxy_control_promotes():
    # k=3, tight lags [15,20,25]: range 10 <= 40 -> PASS.
    result = mine_corpus(
        _scattered_corpus([15, 20, 25], "rng:A", "rng:B"),
        _cfg(),
    )
    assert pattern_id("rng:A", "rng:B") in result


# ---------------------------------------------------------------------------
# SUPPORT gate -- distinct sessions, isolated from the other gates
# ---------------------------------------------------------------------------


def _two_session_strong_corpus(n_sessions: int) -> dict:
    """n_sessions, each with 4 A-trials (A@0,300,600,900) each catching a B at
    lag 100 (B medium@100,400,700,1000). Per session n=k=4; total n=k=4*sessions
    so confidence is strong regardless of session count -- only support varies.
    """
    corpus = {}
    for s in range(n_sessions):
        eps = [_ep("sup:A", "low", 300 * i) for i in range(4)]
        eps += [_ep("sup:B", "medium", 100 + 300 * i) for i in range(4)]
        corpus[f"sup{s}"] = eps
    return corpus


def test_support_gate_rejects_two_sessions():
    # 2 distinct sessions < support_min 3. Confidence (wilson(8,8)=0.68), lift
    # (~2.5) and stability all PASS -- only support blocks it.
    result = mine_corpus(_two_session_strong_corpus(2), _cfg())
    assert pattern_id("sup:A", "sup:B") not in result


def test_support_gate_admits_three_sessions():
    result = mine_corpus(_two_session_strong_corpus(3), _cfg())
    assert pattern_id("sup:A", "sup:B") in result


def test_support_counts_distinct_sessions_not_trials():
    # Duplicating one session's episodes must NOT raise support (still 1 session).
    dup = {
        "one": [_ep("ds:A", "low", 300 * i) for i in range(4)]
        + [_ep("ds:B", "medium", 100 + 300 * i) for i in range(4)],
    }
    result = mine_corpus(dup, _cfg())
    # single session -> support 1 < 3 -> not promoted, however strong the trials.
    assert pattern_id("ds:A", "ds:B") not in result


# ---------------------------------------------------------------------------
# severity + pair rules
# ---------------------------------------------------------------------------


def test_consequent_low_severity_never_matched():
    # B occurs only at low severity -> never a consequent -> no pattern.
    corpus = {
        f"s{i}": [_ep("cs:A", "low", 0), _ep("cs:B", "low", 50), _ep(_C, "low", 2000)]
        for i in range(5)
    }
    assert mine_corpus(corpus, _cfg()) == {}


def test_antecedent_counts_at_all_severities():
    # A occurs only at LOW severity yet still drives a promotable pattern.
    result = mine_corpus(_basic_corpus(), _cfg())
    art = result[pattern_id(_A, _B)]
    assert art["antecedent"] == _A  # _A episodes are all "low"


def test_high_severity_consequent_matched():
    corpus = {
        f"s{i}": [_ep("hs:A", "low", 0), _ep("hs:B", "high", 50), _ep(_C, "low", 2000)]
        for i in range(5)
    }
    result = mine_corpus(corpus, _cfg())
    assert pattern_id("hs:A", "hs:B") in result


def test_no_self_pairs_mined():
    # A key that both precedes and follows itself must never yield an (X,X) pair.
    corpus = {
        f"s{i}": [_ep("solo:x", "medium", 0), _ep("solo:x", "medium", 50)]
        for i in range(5)
    }
    result = mine_corpus(corpus, _cfg())
    assert pattern_id("solo:x", "solo:x") not in result
    assert result == {}


def test_short_sessions_skipped():
    # Sessions with <2 episodes contribute no trials.
    corpus = {f"s{i}": [_ep("sh:A", "low", 0)] for i in range(10)}
    assert mine_corpus(corpus, _cfg()) == {}


def test_lift_rejects_ultra_frequent_consequent():
    # A consequent so frequent that catching one in-window is near-certain under
    # the null is rejected by design (forewarning the near-inevitable has no
    # value). B every 20s drives lambda high -> lift ~ 1.4 < 1.5 -> REJECT.
    corpus = {}
    for i in range(5):
        eps = [_ep("uf:A", "low", 0)]
        eps += [_ep("uf:B", "medium", float(t)) for t in range(20, 3000, 20)]
        corpus[f"uf{i}"] = eps
    result = mine_corpus(corpus, _cfg())
    assert pattern_id("uf:A", "uf:B") not in result


# ===========================================================================
# merge_blob -- lifecycle: probation promotion, watermark hit-rate fold,
# retire, probation-reactivate, bound. Pure; inputs built by hand (Sec 4.6).
# ===========================================================================

_NOW = 1000.0


def _art(a: str = "c:A", b: str = "c:B", **over) -> dict:
    """A fresh candidate as mine_corpus emits it (created_at/mined_at = None)."""
    d = {
        "pattern_id": pattern_id(a, b),
        "antecedent": a,
        "consequent": b,
        "window_s": 125.0,
        "support_sessions": 3,
        "n": 3,
        "k": 3,
        "conf": 1.0,
        "conf_lower": 0.44,
        "lift": 2.0,
        "lag_median_s": 60.0,
        "lag_p90_s": 100.0,
        "status": "provisional",
        "hit_rate": None,
        "resolutions": 0,
        "created_at": None,
        "mined_at": None,
        "retired_at": None,
        "retired_reason": None,
        "repass_streak": 0,
    }
    d.update(over)
    return d


def _pat(a: str = "c:A", b: str = "c:B", **over) -> dict:
    """A prior-blob pattern entry (numeric created_at/mined_at)."""
    d = _art(a, b)
    d.update({"created_at": 100.0, "mined_at": 100.0})
    d.update(over)
    return d


def _cands(*arts: dict) -> dict:
    return {x["pattern_id"]: x for x in arts}


def _prev(*pats: dict, watermark: float = 0.0, mined_at: float = 100.0) -> dict:
    return {
        "version": 1,
        "mined_at": mined_at,
        "hit_rate_watermark": watermark,
        "patterns": {p["pattern_id"]: p for p in pats},
    }


def _res(pid: str, outcome: str, ts: float) -> dict:
    return {"pattern_id": pid, "outcome": outcome, "resolved_ts": float(ts)}


def _assert_finite_stamps(blob: dict) -> None:
    """No pattern in a saved blob may carry inf/nan/None where a number is due."""
    assert math.isfinite(blob["mined_at"])
    assert math.isfinite(blob["hit_rate_watermark"])
    for p in blob["patterns"].values():
        assert isinstance(p["created_at"], (int, float)) and math.isfinite(
            p["created_at"]
        )
        assert isinstance(p["mined_at"], (int, float)) and math.isfinite(p["mined_at"])
        assert math.isfinite(p["lift"])
        assert math.isfinite(p["conf_lower"])
        if p["hit_rate"] is not None:
            assert math.isfinite(p["hit_rate"])


# -- shape + first-run (prev=None) ------------------------------------------


def test_merge_blob_shape():
    blob = merge_blob(None, _cands(_art()), [], _NOW, _cfg(), corpus_newest_ts=500.0)
    assert blob["version"] == 1
    assert blob["mined_at"] == _NOW
    assert set(blob.keys()) == {"version", "mined_at", "hit_rate_watermark", "patterns"}


def test_first_run_all_provisional_and_stamped():
    blob = merge_blob(
        None,
        _cands(_art("x:A", "x:B"), _art("y:A", "y:B")),
        [],
        _NOW,
        _cfg(),
        corpus_newest_ts=999.0,
    )
    assert len(blob["patterns"]) == 2
    for p in blob["patterns"].values():
        assert p["status"] == "provisional"  # first pass -> probation, never active
        assert p["created_at"] == _NOW  # first-promotion stamp
        assert p["mined_at"] == _NOW  # every-save stamp
        assert p["repass_streak"] == 1
        assert p["hit_rate"] is None
        assert p["resolutions"] == 0
    _assert_finite_stamps(blob)


def test_first_run_stamps_over_candidate_none_defaults():
    # mine_corpus emits created_at/mined_at = None; merge MUST overwrite them.
    cand = _art(created_at=None, mined_at=None)
    blob = merge_blob(None, _cands(cand), [], _NOW, _cfg(), corpus_newest_ts=1.0)
    p = next(iter(blob["patterns"].values()))
    assert p["created_at"] == _NOW
    assert p["mined_at"] == _NOW
    _assert_finite_stamps(blob)


def test_first_run_watermark_from_data():
    blob = merge_blob(
        None,
        _cands(_art()),
        [
            _res(pattern_id("c:A", "c:B"), "fulfilled", 100.0),
            _res(pattern_id("c:A", "c:B"), "expired", 200.0),
        ],
        _NOW,
        _cfg(),
        corpus_newest_ts=1.0,
    )
    # watermark comes from resolution data (max folded ts), NOT from mined_at.
    assert blob["hit_rate_watermark"] == 200.0
    assert blob["mined_at"] == _NOW


def test_first_run_no_resolutions_watermark_zero():
    blob = merge_blob(None, _cands(_art()), [], _NOW, _cfg(), corpus_newest_ts=1.0)
    assert blob["hit_rate_watermark"] == 0.0


# -- provisional -> active (probation, Sec 4.4-5) ---------------------------


def test_provisional_promotes_with_new_data():
    prev = _prev(_pat(status="provisional", created_at=100.0, repass_streak=1))
    blob = merge_blob(prev, _cands(_art()), [], _NOW, _cfg(), corpus_newest_ts=200.0)
    p = blob["patterns"][pattern_id("c:A", "c:B")]
    assert p["status"] == "active"  # re-passed AND corpus postdates created_at
    assert p["created_at"] == 100.0  # original created_at preserved
    assert p["repass_streak"] == 2
    assert p["mined_at"] == _NOW


def test_provisional_stays_without_new_data():
    prev = _prev(_pat(status="provisional", created_at=100.0, repass_streak=1))
    blob = merge_blob(prev, _cands(_art()), [], _NOW, _cfg(), corpus_newest_ts=50.0)
    p = blob["patterns"][pattern_id("c:A", "c:B")]
    assert p["status"] == "provisional"  # no session postdating created_at


def test_provisional_stays_when_corpus_ts_equals_created_at():
    prev = _prev(_pat(status="provisional", created_at=100.0))
    blob = merge_blob(prev, _cands(_art()), [], _NOW, _cfg(), corpus_newest_ts=100.0)
    p = blob["patterns"][pattern_id("c:A", "c:B")]
    assert p["status"] == "provisional"  # strict > required, equality does not arm


# -- hit-rate fold (Sec 4.6-3) ----------------------------------------------


def test_fold_first_initializes_then_ewma():
    prev = _prev(_pat(status="active", hit_rate=None, resolutions=0), watermark=0.0)
    pid = pattern_id("c:A", "c:B")
    blob = merge_blob(
        prev,
        _cands(_art()),
        [_res(pid, "fulfilled", 10.0), _res(pid, "expired", 20.0)],
        _NOW,
        _cfg(),
        corpus_newest_ts=1.0,
    )
    p = blob["patterns"][pid]
    # init to 1.0 (fulfilled), then 0.8*1.0 + 0.2*0.0 = 0.8 (alpha=0.2).
    assert p["hit_rate"] == pytest.approx(0.8)
    assert p["resolutions"] == 2


def test_fold_ewma_from_existing():
    prev = _prev(_pat(status="active", hit_rate=0.5, resolutions=3), watermark=0.0)
    pid = pattern_id("c:A", "c:B")
    blob = merge_blob(
        prev,
        _cands(_art()),
        [_res(pid, "fulfilled", 10.0)],
        _NOW,
        _cfg(),
        corpus_newest_ts=1.0,
    )
    p = blob["patterns"][pid]
    assert p["hit_rate"] == pytest.approx(0.6)  # 0.8*0.5 + 0.2*1.0
    assert p["resolutions"] == 4


def test_fold_skips_at_or_below_watermark():
    prev = _prev(_pat(status="active", hit_rate=None, resolutions=0), watermark=50.0)
    pid = pattern_id("c:A", "c:B")
    blob = merge_blob(
        prev,
        _cands(_art()),
        [
            _res(pid, "expired", 40.0),
            _res(pid, "expired", 50.0),
            _res(pid, "fulfilled", 60.0),
        ],
        _NOW,
        _cfg(),
        corpus_newest_ts=1.0,
    )
    p = blob["patterns"][pid]
    # only ts=60 (>50) folds -> init to 1.0; ts=40 and ts=50 ignored.
    assert p["hit_rate"] == pytest.approx(1.0)
    assert p["resolutions"] == 1
    assert blob["hit_rate_watermark"] == 60.0


def test_fold_watermark_stays_when_nothing_new():
    prev = _prev(_pat(status="active"), watermark=500.0)
    pid = pattern_id("c:A", "c:B")
    blob = merge_blob(
        prev,
        _cands(_art()),
        [_res(pid, "expired", 400.0), _res(pid, "fulfilled", 450.0)],
        _NOW,
        _cfg(),
        corpus_newest_ts=1.0,
    )
    assert blob["hit_rate_watermark"] == 500.0  # nothing above watermark folded


def test_fold_is_time_ordered_not_list_ordered():
    # EWMA is order-dependent: merge must fold by resolved_ts, not list order.
    prev = _prev(_pat(status="active", hit_rate=None, resolutions=0), watermark=0.0)
    pid = pattern_id("c:A", "c:B")
    blob = merge_blob(
        prev,
        _cands(_art()),
        [_res(pid, "expired", 20.0), _res(pid, "fulfilled", 10.0)],  # out of order
        _NOW,
        _cfg(),
        corpus_newest_ts=1.0,
    )
    p = blob["patterns"][pid]
    # time order is fulfilled@10 then expired@20 -> init 1.0 then 0.8, NOT 0.2.
    assert p["hit_rate"] == pytest.approx(0.8)


# -- retirement (Sec 4.6-4) --------------------------------------------------


def test_retire_on_low_hit_rate():
    prev = _prev(_pat(status="active", hit_rate=0.2, resolutions=5), watermark=0.0)
    pid = pattern_id("c:A", "c:B")
    blob = merge_blob(prev, _cands(_art()), [], _NOW, _cfg(), corpus_newest_ts=1.0)
    p = blob["patterns"][pid]
    assert p["status"] == "retired"
    assert p["retired_reason"] == "hit_rate"
    assert p["retired_at"] == _NOW


def test_no_retire_below_min_resolutions():
    # hit_rate is bad but resolutions (4) < retire_min (5) -> stays active.
    prev = _prev(_pat(status="active", hit_rate=0.2, resolutions=4), watermark=0.0)
    pid = pattern_id("c:A", "c:B")
    blob = merge_blob(prev, _cands(_art()), [], _NOW, _cfg(), corpus_newest_ts=1.0)
    assert blob["patterns"][pid]["status"] == "active"


def test_revalidation_retire_when_absent_from_candidates():
    prev = _prev(_pat(status="active", hit_rate=0.9, resolutions=8, repass_streak=4))
    pid = pattern_id("c:A", "c:B")
    blob = merge_blob(prev, {}, [], _NOW, _cfg(), corpus_newest_ts=1.0)
    p = blob["patterns"][pid]
    assert p["status"] == "retired"
    assert p["retired_reason"] == "revalidation"
    assert p["retired_at"] == _NOW
    assert p["hit_rate"] == 0.9  # belief preserved for audit
    assert p["resolutions"] == 8
    assert p["repass_streak"] == 0  # reset -- it did not re-pass


def test_repass_streak_increments_on_repass():
    prev = _prev(_pat(status="active", repass_streak=2, created_at=100.0))
    blob = merge_blob(prev, _cands(_art()), [], _NOW, _cfg(), corpus_newest_ts=1.0)
    assert blob["patterns"][pattern_id("c:A", "c:B")]["repass_streak"] == 3


# -- probation reactivation (Sec 4.6-5) --------------------------------------


def test_hit_rate_retired_reactivates_on_second_repass():
    prev = _prev(
        _pat(
            status="retired",
            retired_reason="hit_rate",
            retired_at=90.0,
            hit_rate=0.1,
            resolutions=7,
            repass_streak=1,
        )
    )
    pid = pattern_id("c:A", "c:B")
    blob = merge_blob(prev, _cands(_art()), [], _NOW, _cfg(), corpus_newest_ts=1.0)
    p = blob["patterns"][pid]
    assert p["status"] == "provisional"  # probation, not straight to active
    assert p["hit_rate"] is None  # belief reset
    assert p["resolutions"] == 0
    assert p["retired_at"] is None
    assert p["retired_reason"] is None
    assert p["repass_streak"] == 2
    assert p["created_at"] == _NOW  # fresh probation window


def test_hit_rate_retired_stays_on_first_repass():
    prev = _prev(
        _pat(
            status="retired",
            retired_reason="hit_rate",
            retired_at=90.0,
            hit_rate=0.1,
            resolutions=7,
            repass_streak=0,
        )
    )
    pid = pattern_id("c:A", "c:B")
    blob = merge_blob(prev, _cands(_art()), [], _NOW, _cfg(), corpus_newest_ts=1.0)
    p = blob["patterns"][pid]
    assert p["status"] == "retired"  # needs 2 consecutive re-passes
    assert p["repass_streak"] == 1
    assert p["hit_rate"] == 0.1  # belief still frozen while retired


def test_revalidation_retired_reactivates_immediately():
    prev = _prev(
        _pat(
            status="retired",
            retired_reason="revalidation",
            retired_at=90.0,
            hit_rate=0.9,
            resolutions=8,
            repass_streak=0,
        )
    )
    pid = pattern_id("c:A", "c:B")
    blob = merge_blob(prev, _cands(_art()), [], _NOW, _cfg(), corpus_newest_ts=1.0)
    p = blob["patterns"][pid]
    assert p["status"] == "provisional"  # back via the normal probation path
    assert p["hit_rate"] is None
    assert p["resolutions"] == 0
    assert p["retired_reason"] is None
    assert p["created_at"] == _NOW


def test_retired_keeps_folding_stragglers():
    prev = _prev(
        _pat(
            status="retired",
            retired_reason="hit_rate",
            retired_at=90.0,
            hit_rate=0.1,
            resolutions=7,
            repass_streak=0,
        ),
        watermark=50.0,
    )
    pid = pattern_id("c:A", "c:B")
    # absent from candidates (stays retired) but a straggler resolves.
    blob = merge_blob(
        prev, {}, [_res(pid, "fulfilled", 60.0)], _NOW, _cfg(), corpus_newest_ts=1.0
    )
    p = blob["patterns"][pid]
    assert p["status"] == "retired"  # still retired
    assert p["resolutions"] == 8  # straggler folded
    assert p["hit_rate"] == pytest.approx(0.28)  # 0.8*0.1 + 0.2*1.0
    assert blob["hit_rate_watermark"] == 60.0


# -- bound (Sec 4.6-6) -------------------------------------------------------


def test_bound_drops_provisionals_before_actives():
    prev = _prev(
        _pat("a:A", "a:B", status="active", conf_lower=0.5),
        _pat("b:A", "b:B", status="active", conf_lower=0.5),
    )
    cands = _cands(
        _art("a:A", "a:B"),
        _art("b:A", "b:B"),
        _art("n:A", "n:B", conf_lower=0.9),  # new provisional, high conf
    )
    blob = merge_blob(
        prev, cands, [], _NOW, _cfg(praesagium_max_patterns=2), corpus_newest_ts=1.0
    )
    pids = set(blob["patterns"])
    assert pattern_id("n:A", "n:B") not in pids  # provisional dropped first
    assert pattern_id("a:A", "a:B") in pids
    assert pattern_id("b:A", "b:B") in pids


def test_bound_drops_lowest_conf_lower_within_status():
    cands = _cands(
        _art("a:A", "a:B", conf_lower=0.5),
        _art("b:A", "b:B", conf_lower=0.6),
        _art("c:A", "c:B", conf_lower=0.7),
    )
    prev = _prev(
        _pat("a:A", "a:B", status="active", conf_lower=0.5),
        _pat("b:A", "b:B", status="active", conf_lower=0.6),
        _pat("c:A", "c:B", status="active", conf_lower=0.7),
    )
    blob = merge_blob(
        prev, cands, [], _NOW, _cfg(praesagium_max_patterns=2), corpus_newest_ts=1.0
    )
    pids = set(blob["patterns"])
    assert pattern_id("a:A", "a:B") not in pids  # lowest conf_lower dropped
    assert pattern_id("b:A", "b:B") in pids
    assert pattern_id("c:A", "c:B") in pids


def test_bound_prunes_retired_oldest_first():
    prev = _prev(
        _pat(
            "a:A", "a:B", status="retired", retired_reason="hit_rate", retired_at=100.0
        ),
        _pat(
            "b:A", "b:B", status="retired", retired_reason="hit_rate", retired_at=200.0
        ),
    )
    blob = merge_blob(
        prev, {}, [], _NOW, _cfg(praesagium_max_patterns=1), corpus_newest_ts=1.0
    )
    pids = set(blob["patterns"])
    assert pattern_id("a:A", "a:B") not in pids  # oldest retired_at pruned
    assert pattern_id("b:A", "b:B") in pids


# -- finite-lift invariant (task-5 binding) ---------------------------------


def test_infinite_lift_clamped_to_finite():
    cand = _art(lift=math.inf)
    blob = merge_blob(None, _cands(cand), [], _NOW, _cfg(), corpus_newest_ts=1.0)
    p = next(iter(blob["patterns"].values()))
    assert math.isfinite(p["lift"])
    assert p["lift"] == _FINITE_LIFT_CAP
    _assert_finite_stamps(blob)


def test_nan_stats_sanitized():
    cand = _art(conf=math.nan, lag_median_s=math.nan)
    blob = merge_blob(None, _cands(cand), [], _NOW, _cfg(), corpus_newest_ts=1.0)
    p = next(iter(blob["patterns"].values()))
    assert math.isfinite(p["conf"])
    assert math.isfinite(p["lag_median_s"])
    _assert_finite_stamps(blob)
