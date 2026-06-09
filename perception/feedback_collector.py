"""Feedback collector — gathers explicit and behavioral signals after advice.

Runs alongside the chess board as a separate process. Subscribes to advice
events, prompts for user feedback, tracks post-advice move patterns, and
persists aggregated feedback records per session.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, NamedTuple

import nats
import redis

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blackboard.config import AugurConfig
from blackboard.connections import connect_redis
from blackboard.contracts import PerceptionEvent
from blackboard.persistence import PersistenceManager


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class TrackingKey(NamedTuple):
    domain: str
    entity: str


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("feedback_collector")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUBJECT_ADVICE = "augur.reasoning.advice"
SUBJECT_PERCEPTION = "augur.perception.>"
SUBJECT_SESSION_END = "augur.session.end"
SUBJECT_FEEDBACK_COMPLETE = "augur.feedback.complete"
# Gate suppression (the MRT withheld/control arm). Only gate-decision
# suppressions are published here; infra non-deliveries use a separate subject
# so PendingGateDecision never tracks an infra drop (spec §8).
SUBJECT_SUPPRESSED = "augur.advisor.suppressed"

EXPLICIT_TIMEOUT_S = 10
POST_ADVICE_TRACK_MOVES = 3

# Outcome-metric constants (spec 2026-06-09 §1A/§4)
_EPS = 1e-9
MIN_DECISION_DEVIATION = 1.0  # below this, an HST-only fire — sigma-metric unmeasurable
MIN_POST_OBS = 2  # minimum post-decision observations to score
OUTCOME_METRIC_VERSION = 2  # v1 = legacy chess think-time formula

# ---------------------------------------------------------------------------
# ANSI helpers (for the inline prompt)
# ---------------------------------------------------------------------------
CYAN = "\033[96m"
YELLOW = "\033[93m"
GRAY = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_primary_domain(advice_data: dict) -> str | None:
    """Return the actual primary domain for an advice payload, or None if missing.

    Used by on_advice. Returns None for malformed payloads so the caller can
    log+skip rather than silently mis-attributing to a default domain.
    """
    return advice_data.get("domain") or advice_data.get("primary_anomaly", {}).get(
        "domain"
    )


# ---------------------------------------------------------------------------
# Pending advice tracker
# ---------------------------------------------------------------------------


class _BehavioralTracker:
    """Shared post-decision outcome scoring (spec 2026-06-09 §1A).

    Both the fired arm (``PendingAdvice``) and the withheld/control arm
    (``PendingGateDecision``) compute the same DOMAIN-AGNOSTIC surprise-reduction
    score: the fraction of decision-time surprise removed in the post-decision
    window. Under a Gaussian baseline, surprise ∝ deviation²; the score is
    direction-agnostic and measured against the baseline frozen at decision time
    (baseline_mean, baseline_std, deviation_at_decision, baseline_observation_count),
    so the MRT compares the same outcome across arms. ``think_times_after`` keeps
    its legacy name but now holds generic post-decision values for any domain.
    Unmeasurable fires (degenerate σ, HST-only dev₀, or untrained baseline) score
    a neutral 0.5 with ``unmeasurable=True``.
    """

    baseline_mean: float

    def __init__(
        self,
        *,
        baseline_std: float = 0.0,
        deviation_at_decision: float = 0.0,
        baseline_observation_count: int = 0,
        window: int = POST_ADVICE_TRACK_MOVES,
        min_baseline_std: float = 0.01,
        trend_bonus: float = 0.1,
        min_post_obs: int = MIN_POST_OBS,
        min_observations: int = 15,
    ) -> None:
        self.explicit_rating: Literal["y", "n", "no_response"] = "no_response"
        self.think_times_after: list[float] = []
        self.behavioral_score: float = 0.0
        self.finalized = False
        self.unmeasurable = False
        self.outcome_metric_version = OUTCOME_METRIC_VERSION
        # Decision-time-frozen baseline
        self.baseline_std = baseline_std
        self.deviation_at_decision = deviation_at_decision
        self.baseline_observation_count = baseline_observation_count
        # Scoring config (threaded from AugurConfig at construction)
        self._window = window
        self._min_baseline_std = min_baseline_std
        self._trend_bonus = trend_bonus
        self._min_post_obs = min_post_obs
        self._min_observations = min_observations

    def add_post_move(self, value: float) -> None:
        if len(self.think_times_after) < self._window:
            self.think_times_after.append(round(value, 3))
        if len(self.think_times_after) >= self._window:
            self._compute_behavioral_score()

    def _compute_behavioral_score(self) -> None:
        n = len(self.think_times_after)
        if n < self._min_post_obs:
            # Incomplete window (session-end path) — not scorable yet.
            self.finalized = False
            return
        self.finalized = True
        # Measurability gate (spec §4.1/§4.4): the sigma-surprise metric only
        # applies to a trained, deviation-driven fire against a non-degenerate σ.
        if (
            self.baseline_std < self._min_baseline_std
            or self.deviation_at_decision < MIN_DECISION_DEVIATION
            or self.baseline_observation_count < self._min_observations
        ):
            self.unmeasurable = True
            self.behavioral_score = 0.5
            return

        devs = [
            abs(v - self.baseline_mean) / self.baseline_std
            for v in self.think_times_after
        ]
        surprise_after = sum(d * d for d in devs) / len(devs)
        surprise_before = self.deviation_at_decision**2  # ≥ 1.0 by the gate
        score = 1.0 - surprise_after / max(surprise_before, _EPS)
        # σ-space trajectory bonus: ONLY when net-positive AND shrinking, so it
        # can't rescue a net-worsening window into a positive score.
        if surprise_after < surprise_before and len(devs) >= 2 and devs[-1] < devs[0]:
            score += self._trend_bonus
        self.behavioral_score = round(min(1.0, max(0.0, score)), 3)


class PendingAdvice(_BehavioralTracker):
    """Tracks one piece of advice waiting for feedback signals."""

    def __init__(
        self,
        advice_id: str,
        domain: str,
        entity: str,
        severity: str,
        baseline_mean: float,
        timestamp: str,
        baseline_std: float = 0.0,
        deviation_at_decision: float = 0.0,
        baseline_observation_count: int = 0,
        correlation_found: bool = False,
        correlated_domains: list[str] | None = None,
        rule_key: str | None = None,
        escalation_rule: str | None = None,
        # NEW Phase 3 polish fields:
        involved_domains: list[str] | None = None,
        temporal_lag_seconds: float | None = None,
        correlation_span_s: float | None = None,
        rule_window_s: float | None = None,
        # Advisor-gate MRT fields (spec §9): decision_id joins this fired
        # decision to its emission/silence/feedback; probe marks a bet-hedge
        # probe-fire; mrt_eligible/p_fire make the fired arm IPW-weightable.
        decision_id: str | None = None,
        probe: bool = False,
        mrt_eligible: bool = False,
        p_fire: float | None = None,
        # Scoring config (threaded from AugurConfig at construction)
        window: int = POST_ADVICE_TRACK_MOVES,
        min_baseline_std: float = 0.01,
        trend_bonus: float = 0.1,
        min_observations: int = 15,
    ) -> None:
        super().__init__(
            baseline_std=baseline_std,
            deviation_at_decision=deviation_at_decision,
            baseline_observation_count=baseline_observation_count,
            window=window,
            min_baseline_std=min_baseline_std,
            trend_bonus=trend_bonus,
            min_observations=min_observations,
        )
        self.advice_id = advice_id
        self.domain = domain
        self.entity = entity
        self.severity = severity
        self.baseline_mean = baseline_mean
        self.timestamp = timestamp
        # Correlation metadata (added for matrix tuning)
        self.correlation_found = correlation_found
        self.correlated_domains = correlated_domains or []
        self.rule_key = rule_key
        self.escalation_rule = escalation_rule
        # NEW Phase 3 polish fields
        self.involved_domains = involved_domains or []
        self.temporal_lag_seconds = temporal_lag_seconds
        self.correlation_span_s = correlation_span_s
        self.rule_window_s = rule_window_s
        # Advisor-gate MRT fields (spec §9)
        self.decision_id = decision_id
        self.probe = probe
        self.mrt_eligible = mrt_eligible
        self.p_fire = p_fire

    def to_record(self) -> dict:
        return {
            "advice_id": self.advice_id,
            "domain": self.domain,
            "entity": self.entity,
            "severity": self.severity,
            "explicit_rating": self.explicit_rating,
            "behavioral_score": self.behavioral_score,
            "think_times_after": self.think_times_after,
            "baseline_mean_at_time": self.baseline_mean,
            "baseline_std_at_time": self.baseline_std,
            "deviation_at_decision": self.deviation_at_decision,
            "baseline_observation_count": self.baseline_observation_count,
            "unmeasurable": self.unmeasurable,
            "outcome_metric_version": self.outcome_metric_version,
            "timestamp": self.timestamp,
            # Correlation metadata (added for matrix tuning)
            "correlation_found": self.correlation_found,
            "correlated_domains": self.correlated_domains,
            "rule_key": self.rule_key,
            "escalation_rule": self.escalation_rule,
            # NEW Phase 3 polish fields
            "involved_domains": self.involved_domains,
            "temporal_lag_seconds": self.temporal_lag_seconds,
            "correlation_span_s": self.correlation_span_s,
            "rule_window_s": self.rule_window_s,
            # Advisor-gate MRT fields (spec §9). behavioral_finalized lets the
            # offline audit tell an unfinalized 0.0 from a genuine low score.
            "decision_id": self.decision_id,
            "probe": self.probe,
            "mrt_eligible": self.mrt_eligible,
            "p_fire": self.p_fire,
            "behavioral_finalized": self.finalized,
        }


# ---------------------------------------------------------------------------
# Pending gate-decision tracker (MRT withheld/control arm, spec §9)
# ---------------------------------------------------------------------------


class PendingGateDecision(_BehavioralTracker):
    """Tracks one gate-withheld decision for the MRT control arm.

    Created only from ``augur.advisor.suppressed`` (gate suppressions). Carries
    the same post-decision behavioral tracking as ``PendingAdvice`` so the MRT
    compares the same outcome across the fired and withheld arms, joined by
    ``decision_id``. Probe-fired decisions get NO ``PendingGateDecision`` (a
    probe fires real advice → ``on_advice`` already makes a ``PendingAdvice``).
    """

    def __init__(
        self,
        decision_id: str,
        state_key: str,
        domain: str,
        entity: str,
        severity: str,
        baseline_mean: float,
        timestamp: str,
        mrt_eligible: bool,
        p_withhold: float | None,
        reason: str,
        # Defaulted args MUST follow the required ones above (no default-before-
        # required — that is a SyntaxError). Decision-time snapshot + 1B fields.
        baseline_std: float = 0.0,
        deviation_at_decision: float = 0.0,
        baseline_observation_count: int = 0,
        session_id: str | None = None,
        # Scoring config (threaded from AugurConfig at construction)
        window: int = POST_ADVICE_TRACK_MOVES,
        min_baseline_std: float = 0.01,
        trend_bonus: float = 0.1,
        min_observations: int = 15,
    ) -> None:
        super().__init__(
            baseline_std=baseline_std,
            deviation_at_decision=deviation_at_decision,
            baseline_observation_count=baseline_observation_count,
            window=window,
            min_baseline_std=min_baseline_std,
            trend_bonus=trend_bonus,
            min_observations=min_observations,
        )
        self.decision_id = decision_id
        self.state_key = state_key
        self.domain = domain
        self.entity = entity
        self.severity = severity
        self.baseline_mean = baseline_mean
        self.timestamp = timestamp
        self.mrt_eligible = mrt_eligible
        self.p_withhold = p_withhold
        self.reason = reason
        self.session_id = session_id
        # 1B withheld-rating state: selection is set at suppression; the rating
        # probability is set only when a prompt is actually issued (the IPW
        # exclusion key), so a selected-but-never-prompted row stays in the estimand.
        self.selected_for_rating = False
        self.withheld_rating_p: float | None = None

    def to_record(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "state_key": self.state_key,
            "domain": self.domain,
            "entity": self.entity,
            "severity": self.severity,
            "mrt_eligible": self.mrt_eligible,
            "p_withhold": self.p_withhold,
            "baseline_mean": self.baseline_mean,
            "baseline_std_at_time": self.baseline_std,
            "deviation_at_decision": self.deviation_at_decision,
            "baseline_observation_count": self.baseline_observation_count,
            "unmeasurable": self.unmeasurable,
            "outcome_metric_version": self.outcome_metric_version,
            "withheld_rating_p": self.withheld_rating_p,
            "behavioral_score": self.behavioral_score,
            "behavioral_finalized": self.finalized,
            "explicit_rating": self.explicit_rating,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Async stdin reader
# ---------------------------------------------------------------------------


# LEAK-08: use a dedicated single-worker executor for stdin reads so a
# timed-out read that leaves a thread blocked on sys.stdin.readline does
# not consume a slot in the default asyncio executor pool (which is shared
# with every other loop.run_in_executor call). The dedicated executor
# caps the collateral damage to one leaked thread per pending read.
# The executor is intentionally module-global so it is reused across
# multiple read_stdin_with_timeout calls and cleaned up at process exit.
_stdin_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="augur-stdin"
)


async def read_stdin_with_timeout(timeout: float) -> str | None:
    """Non-blocking stdin read with timeout. Returns None on timeout.

    NOTE: On Linux there is no portable way to cancel a blocking
    ``sys.stdin.readline`` call once it has started. If this function
    times out, the underlying thread remains blocked on readline until
    the user eventually types something. We contain the damage by using
    a dedicated single-worker executor (LEAK-08) so a stalled read does
    not consume slots from the shared default pool; the next call will
    queue behind the blocked thread if one is already outstanding, but
    in practice users respond to at most one prompt at a time so queuing
    is never observed during normal operation.
    """
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(_stdin_executor, sys.stdin.readline),
            timeout=timeout,
        )
        return result.strip().lower() if result else None
    except asyncio.TimeoutError:
        return None


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------


async def run() -> None:
    config = AugurConfig.from_env()

    # ARCH-11: shared connect_redis helper (previously inlined here).
    redis_client = connect_redis(config)
    pm = PersistenceManager(redis_client)

    nc = await nats.connect(
        config.nats_url, connect_timeout=config.nats_connect_timeout
    )
    log.info("NATS connected (%s)", config.nats_url)

    # State
    current_session_id: str | None = None
    advice_events: list[PendingAdvice] = []
    gate_decision_events: list[PendingGateDecision] = []  # MRT withheld arm
    # (domain, entity) -> active tracker. Both arms key on the primary
    # (domain, entity); one tracker per decision_id (spec §9: never two).
    active_tracking: dict[TrackingKey, PendingAdvice | PendingGateDecision] = {}
    # decision_ids already tracked (either arm) — dedup so a re-published
    # augur.advisor.suppressed never creates a second PendingGateDecision.
    tracked_decision_ids: set[str] = set()

    def get_session_id() -> str:
        # Intentionally laxer than blackboard.session.get_active_session:
        # accepts ended sessions (so late-arriving feedback for a just-ended
        # session is still attributed correctly) and falls back to a fresh
        # uuid rather than dropping feedback on a session-record race.
        nonlocal current_session_id
        if current_session_id is None:
            raw = redis_client.get("augur:session:current")
            if raw:
                data = json.loads(raw)
                current_session_id = data.get("session_id", str(uuid.uuid4()))
            else:
                current_session_id = str(uuid.uuid4())
        return current_session_id

    def build_session_summary() -> dict:
        total = len(advice_events)
        explicit_pos = sum(1 for a in advice_events if a.explicit_rating == "y")
        explicit_neg = sum(1 for a in advice_events if a.explicit_rating == "n")
        behavioral_scores = [a.behavioral_score for a in advice_events if a.finalized]
        avg_behavioral = (
            round(sum(behavioral_scores) / len(behavioral_scores), 3)
            if behavioral_scores
            else 0.0
        )
        return {
            "total_advice": total,
            "explicit_positive": explicit_pos,
            "explicit_negative": explicit_neg,
            "avg_behavioral_score": avg_behavioral,
            # MRT withheld/control arm (spec §9): the summary counts both arms.
            "total_gate_decisions": len(gate_decision_events),
        }

    def save_current_feedback() -> None:
        sid = get_session_id()
        record = {
            "session_id": sid,
            "advice_events": [a.to_record() for a in advice_events],
            # Parallel withheld-arm list so a withheld-only session is not lost.
            "gate_decision_events": [d.to_record() for d in gate_decision_events],
            "session_summary": build_session_summary(),
        }
        try:
            pm.save_feedback(sid, record)
            log.info("Saved feedback for session %s", sid)
        except redis.RedisError as exc:
            log.error("Failed to save feedback: %s", exc)

    # -- Advice handler ------------------------------------------------------
    async def on_advice(msg: nats.aio.client.Msg) -> None:
        try:
            data = json.loads(msg.data.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        # Pull primary domain from advice payload (was hardcoded "chess")
        primary_domain = _resolve_primary_domain(data)
        if primary_domain is None:
            log.error(
                "on_advice: missing domain in advice payload; skipping: %s",
                data.get("advice_id", "?"),
            )
            return

        entity = data.get("player", "?")
        severity = data.get("severity", "?")
        move = data.get("move", "?")
        think_time = data.get("think_time", 0)

        # Correlation metadata from the advisor (Phase 3B + matrix-tuning follow-up)
        correlation_found = bool(data.get("correlation_found", False))
        correlated_domains = data.get("correlated_domains") or []
        rule_key = data.get("rule_key")
        escalation_rule = data.get("escalation_rule")
        # NEW Phase 3 polish fields
        involved_domains = data.get("involved_domains") or []
        temporal_lag_seconds = data.get("temporal_lag_seconds")
        correlation_span_s = data.get("correlation_span_s")
        rule_window_s = data.get("rule_window_s")
        # Advisor-gate MRT linkage (spec §9): decision_id joins this fired
        # decision to its emission/silence/feedback by exact key; probe marks a
        # bet-hedge probe-fire; mrt_eligible/p_fire make it IPW-weightable.
        decision_id = data.get("decision_id")
        probe = bool(data.get("probe", False))
        mrt_eligible = bool(data.get("mrt_eligible", False))
        p_fire = data.get("p_fire")

        # Freeze the decision-time baseline (μ₀, σ₀, dev₀, obs₀) for the
        # domain-agnostic outcome metric (spec 1A). Prefer the advice payload's
        # values (carried pre-update from the triggering anomaly); fall back to
        # Redis for baseline_mean/std.
        baseline_raw = pm.load_baseline(primary_domain, entity)
        baseline_mean = data.get("baseline_mean")
        if baseline_mean is None:
            baseline_mean = (
                baseline_raw.get("ewma_mean", think_time)
                if baseline_raw
                else think_time
            )
        baseline_std = data.get("baseline_std")
        if baseline_std is None and baseline_raw:
            baseline_std = baseline_raw.get("ewma_var", 0.0) ** 0.5
        baseline_std = float(baseline_std or 0.0)
        deviation_at_decision = float(data.get("deviation_score") or 0.0)
        baseline_obs = int(data.get("baseline_observation_count") or 0)

        advice_id = str(uuid.uuid4())[:8]
        pending = PendingAdvice(
            advice_id=advice_id,
            domain=primary_domain,  # was "chess"
            entity=entity,
            severity=severity,
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            deviation_at_decision=deviation_at_decision,
            baseline_observation_count=baseline_obs,
            timestamp=datetime.now(timezone.utc).isoformat(),
            correlation_found=correlation_found,
            correlated_domains=correlated_domains,
            rule_key=rule_key,
            escalation_rule=escalation_rule,
            involved_domains=involved_domains,
            temporal_lag_seconds=temporal_lag_seconds,
            correlation_span_s=correlation_span_s,
            rule_window_s=rule_window_s,
            decision_id=decision_id,
            probe=probe,
            mrt_eligible=mrt_eligible,
            p_fire=p_fire,
            window=config.post_decision_window,
            min_baseline_std=config.min_baseline_std,
            trend_bonus=config.outcome_trend_bonus,
            min_observations=config.min_observations,
        )
        advice_events.append(pending)
        # Mark the decision_id tracked so a re-published augur.advisor.suppressed
        # for the same decision never also creates a PendingGateDecision (spec
        # §9: one tracker per decision_id — a probe-fire is tracked here only).
        if decision_id:
            tracked_decision_ids.add(decision_id)

        # BUG-04: if there is already a pending advice tracked for this
        # entity, finalize its behavioural score before replacing it.
        # Without this, the displaced advice would stay at behavioural=0.0
        # and finalized=False for the lifetime of the session, silently
        # corrupting the feedback record used by the reflection engine.
        tracking_key = TrackingKey(primary_domain, entity)
        displaced = active_tracking.get(tracking_key)
        if displaced is not None and not displaced.finalized:
            displaced._compute_behavioral_score()
            log.debug(
                "Finalized displaced tracker before overwriting %s",
                tracking_key,
            )
        active_tracking[tracking_key] = pending

        log.info(
            "Advice received for %s (%s, %s) — awaiting feedback",
            entity,
            move,
            severity,
        )

        # Prompt for explicit feedback (non-blocking with timeout)
        print(
            f"\n{CYAN}[AUGUR]{RESET} Was this advice useful? "
            f"{BOLD}[y/n/s]{RESET} "
            f"{GRAY}({EXPLICIT_TIMEOUT_S}s to respond, s=skip){RESET} ",
            end="",
            flush=True,
        )

        response = await read_stdin_with_timeout(EXPLICIT_TIMEOUT_S)

        if response in ("y", "yes"):
            pending.explicit_rating = "y"
            print(f"{CYAN}Recorded: positive{RESET}", flush=True)
        elif response in ("n", "no"):
            pending.explicit_rating = "n"
            print(f"{YELLOW}Recorded: negative{RESET}", flush=True)
        elif response in ("s", "skip"):
            pending.explicit_rating = "no_response"
            print(f"{GRAY}Skipped{RESET}", flush=True)
        else:
            pending.explicit_rating = "no_response"
            print(f"\n{GRAY}No response — logged as neutral{RESET}", flush=True)

        # Save intermediate feedback
        save_current_feedback()

    # -- Suppressed handler (MRT withheld/control arm, spec §9) --------------
    async def on_suppressed(msg: nats.aio.client.Msg) -> None:
        try:
            data = json.loads(msg.data.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        decision_id = data.get("decision_id")
        if not decision_id:
            log.error("on_suppressed: missing decision_id; skipping")
            return
        # One tracker per decision_id (spec §9): a re-published suppressed event
        # (or a probe already tracked by on_advice) must not create a second.
        if decision_id in tracked_decision_ids:
            log.debug(
                "on_suppressed: decision %s already tracked; skipping", decision_id
            )
            return

        domain = data.get("domain")
        entity = data.get("entity")
        if domain is None or entity is None:
            log.error("on_suppressed: missing primary domain/entity; skipping")
            return

        pending = PendingGateDecision(
            decision_id=decision_id,
            state_key=data.get("state_key", ""),
            domain=domain,
            entity=entity,
            severity=data.get("severity", "?"),
            baseline_mean=data.get("baseline_mean", 0.0),
            timestamp=data.get("timestamp", ""),
            mrt_eligible=bool(data.get("mrt_eligible", False)),
            p_withhold=data.get("p_withhold"),
            reason=data.get("reason", ""),
            baseline_std=float(data.get("baseline_std") or 0.0),
            deviation_at_decision=float(data.get("deviation_score") or 0.0),
            baseline_observation_count=int(data.get("baseline_observation_count") or 0),
            session_id=data.get("session_id"),
            window=config.post_decision_window,
            min_baseline_std=config.min_baseline_std,
            trend_bonus=config.outcome_trend_bonus,
            min_observations=config.min_observations,
        )
        gate_decision_events.append(pending)
        tracked_decision_ids.add(decision_id)

        # Same post-decision tracking as the fired arm: register on the primary
        # (domain, entity), finalizing any displaced tracker first (BUG-04).
        tracking_key = TrackingKey(domain, entity)
        displaced = active_tracking.get(tracking_key)
        if displaced is not None and not displaced.finalized:
            displaced._compute_behavioral_score()
        active_tracking[tracking_key] = pending

        log.info(
            "Gate suppressed %s (%s/%s) — tracking withheld arm",
            decision_id,
            domain,
            entity,
        )
        save_current_feedback()

    # -- Perception handler (post-advice move tracking) ----------------------
    async def on_perception(msg: nats.aio.client.Msg) -> None:
        try:
            event = PerceptionEvent.from_json(msg.data)
        except (ValueError, TypeError, UnicodeDecodeError) as exc:
            log.debug("Skipping bad perception event in feedback tracker: %s", exc)
            return

        entity = event.entity
        tracking_key = TrackingKey(event.domain, entity)
        if tracking_key not in active_tracking:
            return

        pending = active_tracking[tracking_key]
        pending.add_post_move(event.value)

        log.info(
            "Post-advice move %d/%d for %s: %.2f%s (baseline=%.2f)",
            len(pending.think_times_after),
            POST_ADVICE_TRACK_MOVES,
            entity,
            event.value,
            event.unit,
            pending.baseline_mean,
        )

        if pending.finalized:
            log.info(
                "Behavioral score for %s: %.3f",
                entity,
                pending.behavioral_score,
            )
            del active_tracking[tracking_key]
            save_current_feedback()

    # -- Session end handler -------------------------------------------------
    async def on_session_end(msg: nats.aio.client.Msg) -> None:
        # A withheld-only session (gate decisions but no advice) must NOT be
        # dropped — its gate_decision_events carry the MRT control arm (spec §9).
        if not advice_events and not gate_decision_events:
            log.info(
                "Session ended with no advice or gate decisions — nothing to finalize"
            )
            return

        # Force-finalize any pending tracking
        for pending in active_tracking.values():
            if not pending.finalized:
                pending._compute_behavioral_score()
        active_tracking.clear()

        save_current_feedback()
        summary = build_session_summary()

        log.info(
            "Session finalized: %d advice, %d withheld, %d positive, %d negative, "
            "avg behavioral=%.3f",
            summary["total_advice"],
            summary["total_gate_decisions"],
            summary["explicit_positive"],
            summary["explicit_negative"],
            summary["avg_behavioral_score"],
        )

        # Publish completion event
        try:
            await nc.publish(
                SUBJECT_FEEDBACK_COMPLETE,
                json.dumps(
                    {
                        "session_id": get_session_id(),
                        "summary": summary,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                ).encode(),
            )
            log.info("Published feedback complete to %s", SUBJECT_FEEDBACK_COMPLETE)
        except Exception as exc:
            log.error("Failed to publish feedback complete: %s", exc)

    # -- Subscribe -----------------------------------------------------------
    # R2-LEAK-01: save subscription handles so unsubscribe() is called on
    # shutdown rather than relying on nc.close() to tear them down abruptly.
    # This matches the pattern in correlator.py, reflection_engine.py,
    # console_display.py, and anomaly_detector.py — feedback_collector was
    # missed in Round 1's LEAK-05/06/07 batch.
    sub_advice = await nc.subscribe(SUBJECT_ADVICE, cb=on_advice)
    sub_perception = await nc.subscribe(SUBJECT_PERCEPTION, cb=on_perception)
    sub_session_end = await nc.subscribe(SUBJECT_SESSION_END, cb=on_session_end)
    sub_suppressed = await nc.subscribe(SUBJECT_SUPPRESSED, cb=on_suppressed)

    log.info(
        "Subscribed to: %s, %s, %s, %s",
        SUBJECT_ADVICE,
        SUBJECT_PERCEPTION,
        SUBJECT_SESSION_END,
        SUBJECT_SUPPRESSED,
    )
    log.info("Waiting for advice events...")

    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        try:
            await sub_advice.unsubscribe()
            await sub_perception.unsubscribe()
            await sub_session_end.unsubscribe()
            await sub_suppressed.unsubscribe()
        except Exception as exc:
            log.debug("Unsubscribe failed during shutdown: %s", exc)
        await nc.close()
        log.info("Shut down cleanly")


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log.info("Interrupted")


if __name__ == "__main__":
    main()
