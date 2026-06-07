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

EXPLICIT_TIMEOUT_S = 10
POST_ADVICE_TRACK_MOVES = 3

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


class PendingAdvice:
    """Tracks one piece of advice waiting for feedback signals."""

    def __init__(
        self,
        advice_id: str,
        domain: str,
        entity: str,
        severity: str,
        baseline_mean: float,
        timestamp: str,
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
    ) -> None:
        self.advice_id = advice_id
        self.domain = domain
        self.entity = entity
        self.severity = severity
        self.baseline_mean = baseline_mean
        self.timestamp = timestamp
        self.explicit_rating: Literal["y", "n", "no_response"] = "no_response"
        self.think_times_after: list[float] = []
        self.behavioral_score: float = 0.0
        self.finalized = False
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

    def add_post_move(self, value: float) -> None:
        if len(self.think_times_after) < POST_ADVICE_TRACK_MOVES:
            self.think_times_after.append(round(value, 3))
        if len(self.think_times_after) >= POST_ADVICE_TRACK_MOVES:
            self._compute_behavioral_score()

    def _compute_behavioral_score(self) -> None:
        if not self.think_times_after or self.baseline_mean <= 0:
            self.behavioral_score = 0.5
            return

        scores: list[float] = []
        for t in self.think_times_after:
            ratio = t / self.baseline_mean
            if ratio <= 1.0:
                # At or faster than baseline — positive signal
                scores.append(min(1.0, 1.0 - (ratio - 0.5) * 0.5))
            elif ratio <= 1.5:
                # Slightly slower — neutral
                scores.append(0.5)
            else:
                # Much slower — negative signal
                scores.append(max(0.0, 1.0 - (ratio - 1.0) * 0.5))

        # Check if times are normalizing (trending toward baseline)
        if len(self.think_times_after) >= 2:
            diffs = [
                abs(self.think_times_after[i] - self.baseline_mean)
                for i in range(len(self.think_times_after))
            ]
            if diffs[-1] < diffs[0]:
                # Normalizing — bonus
                scores.append(0.8)

        self.behavioral_score = round(sum(scores) / len(scores), 3) if scores else 0.5
        self.finalized = True

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
    active_tracking: dict[
        TrackingKey, PendingAdvice
    ] = {}  # (domain, entity) -> pending advice

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
        }

    def save_current_feedback() -> None:
        sid = get_session_id()
        record = {
            "session_id": sid,
            "advice_events": [a.to_record() for a in advice_events],
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

        # Read baseline mean for the ACTUAL primary domain (was hardcoded "chess")
        baseline_raw = pm.load_baseline(primary_domain, entity)
        baseline_mean = (
            baseline_raw.get("ewma_mean", think_time) if baseline_raw else think_time
        )

        advice_id = str(uuid.uuid4())[:8]
        pending = PendingAdvice(
            advice_id=advice_id,
            domain=primary_domain,  # was "chess"
            entity=entity,
            severity=severity,
            baseline_mean=baseline_mean,
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
        )
        advice_events.append(pending)

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
                "Finalized displaced pending advice %s before overwriting %s",
                displaced.advice_id,
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
        if not advice_events:
            log.info("Session ended with no advice events — nothing to finalize")
            return

        # Force-finalize any pending tracking
        for pending in active_tracking.values():
            if not pending.finalized:
                pending._compute_behavioral_score()
        active_tracking.clear()

        save_current_feedback()
        summary = build_session_summary()

        log.info(
            "Session finalized: %d advice, %d positive, %d negative, "
            "avg behavioral=%.3f",
            summary["total_advice"],
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

    log.info(
        "Subscribed to: %s, %s, %s",
        SUBJECT_ADVICE,
        SUBJECT_PERCEPTION,
        SUBJECT_SESSION_END,
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
