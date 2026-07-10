"""Praesagium's pattern-mining math: the honest statistical core that turns a
cross-session corpus of episode streams into promotable "A precedes B within
window W" candidates.

Pure functions only -- NO I/O (no Redis, no NATS, no clock-authoritative
timestamps). ``mine_corpus`` returns candidate artifacts that pass EVERY
promotion test of spec Sec 4.4; the lifecycle merge (created_at/mined_at
stamping, probation, hit-rate fold, retirement) is Task 5's miner and is
deliberately absent here -- lifecycle fields are emitted at their pre-merge
defaults (status="provisional", created_at=None, ...).

Spec: docs/superpowers/specs/2026-07-09-praesagium-design.md Sec 4.1-4.5 + PR5.

Pinned conventions
------------------
* **Wilson lower bound** (Sec 4.4-2): the exact score-interval formula with
  z=1.96; ``wilson_lower(3, 3) ~= 0.4385``.
* **P90** (window discovery + lag_p90 stat): nearest-rank rounded up,
  ``sorted(v)[min(len-1, ceil(0.9*len)-1)]`` (e.g. P90 of three equal lags is
  the value itself; P90 over 100 values is index 89).
* **Trial collapse** (Sec 4.2, anti-pseudo-replication): the reference point is
  the *trial start*. An A occurrence within ``lag_min`` (inclusive) of the
  current trial's START folds into it; the first A beyond that radius re-arms a
  new trial. A burst wider than ``lag_min`` therefore re-arms rather than
  collapsing into a single endless trial -- matching PR5c's "N occurrences
  within lag_min of a trial start count as ONE trial".
* **Match window** ``(lag_min, hi]``: open below (a B exactly at ``lag_min`` does
  NOT match), closed above. Each B occurrence satisfies at most one trial,
  consumed by the oldest qualifying trial; a trial takes its smallest qualifying
  lag.
* **Stability IQR** (Sec 4.4-4): ``statistics.quantiles(lags, n=4)`` (exclusive
  method), ``IQR = q3 - q1``.
"""

from __future__ import annotations

import hashlib
import math
import statistics
from typing import Any

_CONSEQUENT_SEVERITIES = ("medium", "high")


def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    """Wilson score-interval lower bound for a binomial proportion (Sec 4.4-2).

    ``p_lower = (p + z^2/2n - z*sqrt(p(1-p)/n + z^2/4n^2)) / (1 + z^2/n)`` with
    ``p = k/n``. Returns 0.0 when ``n <= 0`` (no evidence). The result is clamped
    to be non-negative (guards floating-point undershoot at k=0). Small-sample
    honest: ``wilson_lower(3, 3) ~= 0.4385``.
    """
    if n <= 0:
        return 0.0
    p = k / n
    z2 = z * z
    center = p + z2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    lower = (center - margin) / (1 + z2 / n)
    return max(0.0, lower)


def pattern_id(antecedent: str, consequent: str) -> str:
    """Stable 12-hex identity of an ordered (A, B) pair (Sec 4.5).

    ``sha256(f"{A}→{B}").hexdigest()[:12]`` -- order-sensitive and stable
    across re-mines.
    """
    return hashlib.sha256(f"{antecedent}→{consequent}".encode()).hexdigest()[:12]


def collapse_trials(a_times: list[float], lag_min: float) -> list[float]:
    """Fold a burst of antecedent occurrences into independent trials (Sec 4.2).

    Reference = the trial START. Walking occurrences in time order, an A within
    ``lag_min`` (inclusive) of the current trial's start folds into that trial;
    the first A beyond ``lag_min`` from the start opens a new trial. Returns the
    ascending list of trial-start times. See module docstring for why the
    reference is the start, not the previous occurrence.
    """
    if not a_times:
        return []
    ordered = sorted(a_times)
    trials = [ordered[0]]
    start = ordered[0]
    for t in ordered[1:]:
        if t - start <= lag_min:
            continue  # folds into the current trial
        trials.append(t)
        start = t
    return trials


def count_matches(
    trials: list[float], b_times: list[float], lo: float, hi: float
) -> tuple[int, list[float]]:
    """Count successful trials and their qualifying lags over ``(lo, hi]``.

    A trial at start ``s`` succeeds on the earliest unconsumed B with
    ``s + lo < b <= s + hi`` (smallest qualifying lag). Each B satisfies at most
    one trial and trials are served oldest-first. Returns ``(k, lags)`` where
    ``k == len(lags)`` and each lag is ``b - s``.

    Both inputs are sorted defensively. A single forward pointer suffices: a B
    too early for a trial (``b <= s + lo``) is even earlier for every later
    (larger-start) trial, so it is skipped permanently; a B too far
    (``b > s + hi``) is left for later trials whose window reaches it.
    """
    starts = sorted(trials)
    bs = sorted(b_times)
    lags: list[float] = []
    ptr = 0
    m = len(bs)
    for start in starts:
        floor = start + lo
        ceil = start + hi
        while ptr < m and bs[ptr] <= floor:
            ptr += 1
        if ptr < m and bs[ptr] <= ceil:
            lags.append(bs[ptr] - start)
            ptr += 1
    return len(lags), lags


def _p90(values: list[float]) -> float:
    """Nearest-rank 90th percentile, rounded up (see module docstring)."""
    ordered = sorted(values)
    idx = min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)
    return ordered[idx]


def _stable(lags: list[float], k: int, median_lag: float, ratio: float) -> bool:
    """Lag-stability test (Sec 4.4-4): IQR rule for k>=4, range proxy for k<4."""
    threshold = median_lag * ratio
    if k >= 4:
        q1, _q2, q3 = statistics.quantiles(lags, n=4)  # exclusive method
        spread = q3 - q1
    else:
        spread = max(lags) - min(lags)
    return spread <= threshold


def _valid_episode(entry: Any) -> bool:
    return (
        isinstance(entry, dict)
        and isinstance(entry.get("k"), str)
        and isinstance(entry.get("s"), str)
        and isinstance(entry.get("t"), (int, float))
        and not isinstance(entry.get("t"), bool)
    )


class _Session:
    """Per-session mining view: antecedent times (all severities) and consequent
    times (medium/high only) per canonical key, plus the session duration D_s."""

    __slots__ = ("ante", "cons", "d_s")

    def __init__(
        self, ante: dict[str, list[float]], cons: dict[str, list[float]], d_s: float
    ):
        self.ante = ante
        self.cons = cons
        self.d_s = d_s


def _build_sessions(corpus: dict[str, list[dict]]) -> dict[str, _Session]:
    sessions: dict[str, _Session] = {}
    for sid, episodes in corpus.items():
        eps = sorted((e for e in episodes if _valid_episode(e)), key=lambda e: e["t"])
        if len(eps) < 2:
            continue  # Sec 4.1: skip sessions with < 2 episodes
        d_s = max(eps[-1]["t"] - eps[0]["t"], 60.0)  # 60s floor
        ante: dict[str, list[float]] = {}
        cons: dict[str, list[float]] = {}
        for e in eps:
            ante.setdefault(e["k"], []).append(e["t"])
            if e["s"] in _CONSEQUENT_SEVERITIES:
                cons.setdefault(e["k"], []).append(e["t"])
        sessions[sid] = _Session(ante, cons, d_s)
    return sessions


def _mine_pair(
    a: str,
    b: str,
    sessions: dict[str, _Session],
    cfg: Any,
) -> dict[str, Any] | None:
    """Run the two-pass discovery + Sec 4.4 promotion tests for one (A, B) pair.

    Returns a provisional artifact dict if every test passes, else None.
    """
    lag_min = cfg.praesagium_lag_min_s
    lag_max = cfg.praesagium_lag_max_s

    # A-sessions: sessions containing >=1 antecedent occurrence of A. Trials are
    # collapsed once (collapse depends only on lag_min -> identical across passes).
    a_sessions: list[tuple[str, list[float], list[float], float]] = []
    for sid, sess in sessions.items():
        a_times = sess.ante.get(a)
        if not a_times:
            continue
        trials = collapse_trials(a_times, lag_min)
        b_times = sess.cons.get(b, [])
        a_sessions.append((sid, trials, b_times, sess.d_s))
    if not a_sessions:
        return None

    # PASS 1 -- window discovery over (lag_min, lag_max].
    lags1: list[float] = []
    for _sid, trials, b_times, _d_s in a_sessions:
        _k, session_lags = count_matches(trials, b_times, lag_min, lag_max)
        lags1.extend(session_lags)
    if not lags1:
        return None  # no successful trial -> no window to fix, nothing to mine

    window = min(max(_p90(lags1) * cfg.praesagium_window_margin, lag_min), lag_max)

    # PASS 2 -- honest recount over (lag_min, W]; ALL promotion stats use these.
    n = 0
    k = 0
    lags2: list[float] = []
    support_sids: set[str] = set()
    b_occurrences = 0  # medium/high B occurrences across A-sessions
    sum_d_s = 0.0
    for sid, trials, b_times, d_s in a_sessions:
        k_s, session_lags = count_matches(trials, b_times, lag_min, window)
        n += len(trials)
        k += k_s
        lags2.extend(session_lags)
        if k_s > 0:
            support_sids.add(sid)
        b_occurrences += len(b_times)
        sum_d_s += d_s

    if n == 0 or k == 0:
        return None

    # Sec 4.4-1 -- support (distinct sessions with >=1 pass-2 success).
    support = len(support_sids)
    if support < cfg.praesagium_support_min_sessions:
        return None

    # Sec 4.4-2 -- Wilson-bounded confidence over the runtime window.
    p_hat = k / n
    p_lower = wilson_lower(k, n)
    if p_lower < cfg.praesagium_conf_lower_min:
        return None

    # Sec 4.4-3 -- lift over a session-conditional Poisson null (A-sessions only).
    lam_b = b_occurrences / sum_d_s if sum_d_s > 0 else 0.0
    p0 = 1.0 - math.exp(-lam_b * window)
    lift = p_hat / p0 if p0 > 0 else math.inf
    if lift < cfg.praesagium_lift_min:
        return None

    # Sec 4.4-4 -- lag stability.
    median_lag = statistics.median(lags2)
    if not _stable(lags2, k, median_lag, cfg.praesagium_lag_stability_ratio):
        return None

    return {
        "pattern_id": pattern_id(a, b),
        "antecedent": a,
        "consequent": b,
        "window_s": window,
        "support_sessions": support,
        "n": n,
        "k": k,
        "conf": p_hat,
        "conf_lower": p_lower,
        "lift": lift,
        "lag_median_s": median_lag,
        "lag_p90_s": float(_p90(lags2)),
        # Sec 4.4-5 -- first-pass passers enter probation. Lifecycle fields below
        # are pre-merge defaults; Task 5's miner stamps created_at/mined_at and
        # merges hit_rate/resolutions/repass_streak.
        "status": "provisional",
        "hit_rate": None,
        "resolutions": 0,
        "created_at": None,
        "mined_at": None,
        "retired_at": None,
        "retired_reason": None,
        "repass_streak": 0,
    }


def mine_corpus(corpus: dict[str, list[dict]], cfg: Any) -> dict[str, dict]:
    """Mine cross-session A->B temporal patterns (Sec 4.1-4.5).

    ``corpus`` maps session_id -> episode list (entries ``{"k","s","t"}`` as built
    by ``praesagium.episodes``). Returns ``{pattern_id: artifact}`` for candidates
    passing EVERY promotion test; each artifact has ``status="provisional"`` and
    the full Sec 4.5 field set. Full recompute, idempotent by construction; no I/O.
    """
    sessions = _build_sessions(corpus)

    # Candidate (A, B) pairs: A any-severity antecedent key, B a medium/high
    # consequent key, A != B, co-present in >= 1 session.
    candidates: set[tuple[str, str]] = set()
    for sess in sessions.values():
        for a in sess.ante:
            for b in sess.cons:
                if a != b:
                    candidates.add((a, b))

    results: dict[str, dict] = {}
    for a, b in candidates:
        artifact = _mine_pair(a, b, sessions, cfg)
        if artifact is not None:
            results[artifact["pattern_id"]] = artifact
    return results


# ===========================================================================
# Lifecycle merge (Sec 4.6) -- the miner's pure core: fold a fresh candidate
# set + resolved-outcome history into the previous blob, stamping created_at
# (first promotion) and mined_at (every save), moving patterns through
# probation / hit-rate retirement / probation-reactivation, and bounding the
# blob. NO I/O and NO clock read here either -- ``now`` is passed in.
# ===========================================================================

# inf lift (a consequent absent from its session-conditional null, P0 -> 0)
# promotes by design; the blob is JSON-serialised, so it is clamped to a large
# finite sentinel. No inf/nan may ever reach save_praesagium_patterns.
_FINITE_LIFT_CAP = 1e9

_RECOGNIZED_OUTCOMES = ("fulfilled", "expired")


def _finite(value: Any, *, cap: float, default: float) -> float:
    """Coerce *value* to a finite float; nan/garbage -> *default*, +-inf -> +-cap.

    The single guard behind the finite-lift / no-nan blob invariant.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(f):
        return default
    if math.isinf(f):
        return cap if f > 0 else -cap
    return f


def _sanitize_stats(src: dict) -> dict:
    """Copy the Sec 4.5 statistic fields from *src*, forcing every float finite."""
    return {
        "window_s": _finite(src.get("window_s"), cap=_FINITE_LIFT_CAP, default=0.0),
        "support_sessions": int(src.get("support_sessions") or 0),
        "n": int(src.get("n") or 0),
        "k": int(src.get("k") or 0),
        "conf": _finite(src.get("conf"), cap=1.0, default=0.0),
        "conf_lower": _finite(src.get("conf_lower"), cap=1.0, default=0.0),
        "lift": _finite(src.get("lift"), cap=_FINITE_LIFT_CAP, default=0.0),
        "lag_median_s": _finite(
            src.get("lag_median_s"), cap=_FINITE_LIFT_CAP, default=0.0
        ),
        "lag_p90_s": _finite(src.get("lag_p90_s"), cap=_FINITE_LIFT_CAP, default=0.0),
    }


def _streak(p: dict) -> int:
    return int(p.get("repass_streak") or 0)


def _kept_created_at(prev_p: dict, now: float) -> float:
    """Preserve a prior created_at (first-promotion stamp); repair a bad one."""
    ca = _finite_or_none(prev_p.get("created_at"))
    return ca if ca is not None else float(now)


def _clamp_hit_rate(hr: Any) -> float | None:
    f = _finite_or_none(hr)
    if f is None:
        return None
    return min(1.0, max(0.0, f))


def _finite_or_none(v: Any) -> float | None:
    """A finite float or None: bool, non-numeric, and inf/nan all map to None."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _fresh_probation(
    pid: str, a: Any, b: Any, stats: dict, now: float, *, repass_streak: int
) -> dict:
    """A pattern entering (or re-entering) probation: belief slate wiped, a new
    created_at window so provisional->active demands data newer than *now*."""
    return {
        "pattern_id": pid,
        "antecedent": a,
        "consequent": b,
        **stats,
        "status": "provisional",
        "hit_rate": None,
        "resolutions": 0,
        "created_at": float(now),
        "mined_at": float(now),
        "retired_at": None,
        "retired_reason": None,
        "repass_streak": repass_streak,
    }


def _merge_one(
    pid: str,
    prev_p: dict | None,
    cand: dict | None,
    now: float,
    corpus_newest_ts: float,
) -> dict:
    """Reconcile one pid's prior state with this mine's candidate (Sec 4.6-2/4/5).

    Hit-rate retirement (Sec 4.6-4, post-fold) and bounding (Sec 4.6-6) are
    applied by ``merge_blob`` after every pid is reconciled here.
    """
    if cand is None:
        # Absent from this mine's candidates -> did not re-pass Sec 4.4.
        assert prev_p is not None  # a pid in neither set cannot reach here
        p = dict(prev_p)
        p.update(_sanitize_stats(prev_p))
        p["pattern_id"] = pid
        p["antecedent"] = prev_p.get("antecedent")
        p["consequent"] = prev_p.get("consequent")
        p["created_at"] = _kept_created_at(prev_p, now)
        p["mined_at"] = float(now)
        p["hit_rate"] = _clamp_hit_rate(prev_p.get("hit_rate"))
        p["resolutions"] = int(prev_p.get("resolutions") or 0)
        p["repass_streak"] = 0  # streak resets when a mine does not re-pass
        if prev_p.get("status") == "retired":
            p["status"] = "retired"
            p["retired_at"] = _finite_or_none(prev_p.get("retired_at"))
            p["retired_reason"] = prev_p.get("retired_reason")
        else:
            p["status"] = "retired"  # revalidation retirement
            p["retired_at"] = float(now)
            p["retired_reason"] = "revalidation"
        return p

    # Re-passing this mine.
    stats = _sanitize_stats(cand)
    a = cand.get("antecedent")
    b = cand.get("consequent")

    if prev_p is None:
        return _fresh_probation(pid, a, b, stats, now, repass_streak=1)

    prev_status = prev_p.get("status")
    if prev_status == "retired":
        streak = _streak(prev_p) + 1
        if prev_p.get("retired_reason") == "revalidation":
            # Belief was never impugned; return via the normal probation path.
            return _fresh_probation(pid, a, b, stats, now, repass_streak=streak)
        # hit-rate retired: reactivate only after 2 consecutive re-passes.
        if streak >= 2:
            return _fresh_probation(pid, a, b, stats, now, repass_streak=streak)
        p = dict(prev_p)
        p.update(stats)
        p["pattern_id"] = pid
        p["antecedent"] = a
        p["consequent"] = b
        p["status"] = "retired"
        p["retired_at"] = _finite_or_none(prev_p.get("retired_at"))
        p["retired_reason"] = "hit_rate"
        p["created_at"] = _kept_created_at(prev_p, now)
        p["mined_at"] = float(now)
        p["hit_rate"] = _clamp_hit_rate(prev_p.get("hit_rate"))
        p["resolutions"] = int(prev_p.get("resolutions") or 0)
        p["repass_streak"] = streak
        return p

    # prev provisional or active -> keep belief + created_at, refresh stats.
    created_at = _kept_created_at(prev_p, now)
    if prev_status == "active":
        status = "active"
    else:  # provisional -> active only with a session postdating created_at
        status = "active" if corpus_newest_ts > created_at else "provisional"
    return {
        "pattern_id": pid,
        "antecedent": a,
        "consequent": b,
        **stats,
        "status": status,
        "hit_rate": _clamp_hit_rate(prev_p.get("hit_rate")),
        "resolutions": int(prev_p.get("resolutions") or 0),
        "created_at": created_at,
        "mined_at": float(now),
        "retired_at": None,
        "retired_reason": None,
        "repass_streak": _streak(prev_p) + 1,
    }


def _fold_resolutions(
    merged: dict[str, dict],
    resolutions: list[dict],
    prev_watermark: float,
    prev_watermark_ids: set[str],
    watermark_ids_known: bool,
    cfg: Any,
) -> tuple[float, list[str]]:
    """Fold resolved outcomes into per-pattern EWMA hit rates (Sec 4.6-3).

    A resolution folds iff ``resolved_ts > prev_watermark``, OR it lands EXACTLY
    at the watermark (``resolved_ts == prev_watermark``) with a prediction_id not
    already folded there (*prev_watermark_ids*). This exact fold catches same-ts
    resolutions LPUSHed after a mid-batch miner snapshot, which a strict ``>``
    comparison silently drops forever. A same-ts resolution lacking a
    string prediction_id is skipped (unidentifiable -> can't dedup -> would
    double-fold). Folds run in strict time order (EWMA is order-dependent);
    retired patterns keep folding stragglers. Returns
    ``(watermark, watermark_ids)``: the watermark from DATA (max considered ts,
    never regressing below *prev_watermark*), and the prediction_ids folded AT
    the new watermark ts -- accumulated with *prev_watermark_ids* when the
    watermark does not advance (bounded by resolutions-per-anomaly at a single
    ts, a handful).

    *watermark_ids_known* is False iff the prior blob predates the exact-fold
    rule and never carried ``hit_rate_watermark_ids`` at all -- that is NOT the
    same as a present-but-empty set. A missing field means the old strict-`>`
    code already folded whatever resolution(s) established *prev_watermark* on
    a prior mine; without this flag, an empty *prev_watermark_ids* looks
    identical to "nothing folded there yet" and the watermark-establishing
    resolution re-folds once on the first post-upgrade mine (extra
    ``resolutions`` increment + extra EWMA step, i.e. a migration double-fold).
    When NOT known, every ``ts == prev_watermark`` resolution is treated as
    already-folded (old strict-`>` semantics, for this one migration mine
    only).
    """
    alpha = float(cfg.praesagium_hit_rate_alpha)
    considered: list[tuple[float, str | None, str | None, float | None]] = []
    for r in resolutions:
        if not isinstance(r, dict):
            continue
        ts = _finite_or_none(r.get("resolved_ts"))
        if ts is None or ts < prev_watermark:
            continue
        pred_raw = r.get("prediction_id")
        pred = pred_raw if isinstance(pred_raw, str) else None
        if ts == prev_watermark and (
            not watermark_ids_known or pred is None or pred in prev_watermark_ids
        ):
            # Already folded at the watermark: known-migration (ids missing),
            # unidentifiable, or a known duplicate. Skip.
            continue
        outcome = r.get("outcome")
        if outcome in _RECOGNIZED_OUTCOMES:
            value = 1.0 if outcome == "fulfilled" else 0.0
            considered.append((ts, r.get("pattern_id"), pred, value))
        else:
            # Unrecognised outcome: advances the watermark (seen) but folds
            # into no belief -- prevents a garbage record pinning the watermark.
            considered.append((ts, None, pred, None))

    if not considered:
        return prev_watermark, sorted(prev_watermark_ids)

    considered.sort(key=lambda x: x[0])
    watermark = prev_watermark
    for fold_ts, fold_pid, _pred, fold_value in considered:
        if fold_ts > watermark:
            watermark = fold_ts
        if fold_pid is None or fold_value is None:
            continue
        p = merged.get(fold_pid)
        if p is None:
            continue
        current = p.get("hit_rate")
        if current is None:
            p["hit_rate"] = fold_value
        else:
            p["hit_rate"] = (1.0 - alpha) * float(current) + alpha * fold_value
        p["resolutions"] = int(p.get("resolutions") or 0) + 1

    new_ids: set[str] = (
        set(prev_watermark_ids) if watermark == prev_watermark else set()
    )
    for fold_ts, _fold_pid, pred, _fold_value in considered:
        if fold_ts == watermark and pred is not None:
            new_ids.add(pred)
    return watermark, sorted(new_ids)


def _apply_hit_rate_retire(merged: dict[str, dict], now: float, cfg: Any) -> None:
    """Retire a live pattern whose measured hit rate has failed (Sec 4.6-4)."""
    retire_min = int(cfg.praesagium_retire_min_resolutions)
    retire_below = float(cfg.praesagium_hit_rate_retire_below)
    for p in merged.values():
        if p.get("status") == "retired":
            continue
        hr = p.get("hit_rate")
        if hr is None:
            continue
        if int(p.get("resolutions") or 0) >= retire_min and float(hr) < retire_below:
            p["status"] = "retired"
            p["retired_at"] = float(now)
            p["retired_reason"] = "hit_rate"
            p["repass_streak"] = 0  # reactivation counts fresh post-retire re-passes


def _bound(merged: dict[str, dict], cfg: Any) -> None:
    """Cap non-retired and retired patterns to max_patterns each (Sec 4.6-6).

    Non-retired: keep the best -- actives before provisionals, higher
    ``conf_lower`` first (so drops take provisionals before actives, lowest
    ``conf_lower`` first within a status). Retired: keep the newest
    ``retired_at`` (drop oldest first).
    """
    max_patterns = int(cfg.praesagium_max_patterns)
    non_retired = [p for p in merged.values() if p.get("status") != "retired"]
    retired = [p for p in merged.values() if p.get("status") == "retired"]

    keep: set[str] = set()
    non_retired.sort(
        key=lambda p: (
            0 if p.get("status") == "active" else 1,
            -_finite(p.get("conf_lower"), cap=1.0, default=0.0),
        )
    )
    for p in non_retired[:max_patterns]:
        keep.add(p["pattern_id"])
    retired.sort(key=lambda p: -(_finite_or_none(p.get("retired_at")) or 0.0))
    for p in retired[:max_patterns]:
        keep.add(p["pattern_id"])

    for pid in list(merged):
        if pid not in keep:
            del merged[pid]


def merge_blob(
    prev: dict | None,
    candidates: dict[str, dict],
    resolutions: list[dict],
    now: float,
    cfg: Any,
    *,
    corpus_newest_ts: float,
) -> dict:
    """Merge a fresh candidate set + resolved history into the patterns blob.

    Pure (Sec 4.6): ``now`` and ``corpus_newest_ts`` are passed in, never read.
    *resolutions* should already be time-ordered by the caller (the miner
    reverses the newest-first log); this function additionally sorts defensively
    before the order-dependent EWMA fold. Returns the new blob
    ``{"version", "mined_at", "hit_rate_watermark", "hit_rate_watermark_ids",
    "patterns"}`` -- every pattern carries numeric ``created_at``/``mined_at``
    and only finite floats. ``hit_rate_watermark_ids`` are the prediction_ids
    folded AT the watermark ts (the exact-fold dedup set).
    """
    prev_patterns: dict[str, dict] = {}
    prev_watermark = 0.0
    prev_watermark_ids: set[str] = set()
    watermark_ids_known = True  # no prior blob at all -> nothing to migrate
    if isinstance(prev, dict):
        pp = prev.get("patterns")
        if isinstance(pp, dict):
            prev_patterns = pp
        wm = _finite_or_none(prev.get("hit_rate_watermark"))
        if wm is not None:
            prev_watermark = wm
        wm_ids = prev.get("hit_rate_watermark_ids")
        # A blob written before the exact-fold rule existed never carried this
        # field at all -- distinct from a present-but-empty list. See
        # _fold_resolutions docstring.
        watermark_ids_known = wm_ids is not None
        if isinstance(wm_ids, list):
            prev_watermark_ids = {x for x in wm_ids if isinstance(x, str)}

    merged: dict[str, dict] = {}
    for pid in set(prev_patterns) | set(candidates):
        merged[pid] = _merge_one(
            pid, prev_patterns.get(pid), candidates.get(pid), now, corpus_newest_ts
        )

    new_watermark, new_watermark_ids = _fold_resolutions(
        merged,
        resolutions,
        prev_watermark,
        prev_watermark_ids,
        watermark_ids_known,
        cfg,
    )
    _apply_hit_rate_retire(merged, now, cfg)
    _bound(merged, cfg)

    return {
        "version": 1,
        "mined_at": float(now),
        "hit_rate_watermark": float(new_watermark),
        "hit_rate_watermark_ids": new_watermark_ids,
        "patterns": merged,
    }
