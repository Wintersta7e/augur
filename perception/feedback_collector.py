"""Feedback collector — gathers explicit and behavioral signals after advice.

Runs alongside the chess board as a separate process. Subscribes to advice
events, prompts for user feedback, tracks post-advice move patterns, and
persists aggregated feedback records per session.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import nats
import redis

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blackboard.contracts import PerceptionEvent
from blackboard.persistence import PersistenceManager

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
    ) -> None:
        self.advice_id = advice_id
        self.domain = domain
        self.entity = entity
        self.severity = severity
        self.baseline_mean = baseline_mean
        self.timestamp = timestamp
        self.explicit_rating: str = "no_response"
        self.think_times_after: list[float] = []
        self.behavioral_score: float = 0.0
        self.finalized = False

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
        }


# ---------------------------------------------------------------------------
# Async stdin reader
# ---------------------------------------------------------------------------


async def read_stdin_with_timeout(timeout: float) -> str | None:
    """Non-blocking stdin read with timeout. Returns None on timeout."""
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, sys.stdin.readline),
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

    redis_client = redis.Redis(
        host=config.redis_host,
        port=config.redis_port,
        socket_connect_timeout=config.redis_connect_timeout,
    )
    redis_client.ping()
    log.info("Redis connected (%s)", config.redis_url)

    pm = PersistenceManager(redis_client)

    nc = await nats.connect(
        config.nats_url, connect_timeout=config.nats_connect_timeout
    )
    log.info("NATS connected (%s)", config.nats_url)

    # State
    current_session_id: str | None = None
    advice_events: list[PendingAdvice] = []
    active_tracking: dict[str, PendingAdvice] = {}  # entity -> pending advice

    def get_session_id() -> str:
        nonlocal current_session_id
        if current_session_id is None:
            # Try to read from Redis
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

        entity = data.get("player", "?")
        severity = data.get("severity", "?")
        move = data.get("move", "?")
        think_time = data.get("think_time", 0)

        # Read baseline mean from Redis
        baseline_raw = pm.load_baseline(
            "chess",
            entity,
        )
        baseline_mean = (
            baseline_raw.get("ewma_mean", think_time) if baseline_raw else think_time
        )

        advice_id = str(uuid.uuid4())[:8]
        pending = PendingAdvice(
            advice_id=advice_id,
            domain="chess",
            entity=entity,
            severity=severity,
            baseline_mean=baseline_mean,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        advice_events.append(pending)
        active_tracking[entity] = pending

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
        except (json.JSONDecodeError, TypeError, KeyError):
            return

        entity = event.entity
        if entity not in active_tracking:
            return

        pending = active_tracking[entity]
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
            del active_tracking[entity]
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
    await nc.subscribe(SUBJECT_ADVICE, cb=on_advice)
    await nc.subscribe(SUBJECT_PERCEPTION, cb=on_perception)
    await nc.subscribe(SUBJECT_SESSION_END, cb=on_session_end)

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
        await nc.close()
        log.info("Shut down cleanly")


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log.info("Interrupted")


if __name__ == "__main__":
    main()
