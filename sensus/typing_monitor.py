"""Typing rhythm monitor — second perception domain proving generic architecture.

Captures system-wide keypresses via the 'keyboard' library, tracks
inter-keypress intervals, detects pauses (> 3s gaps), and publishes
PerceptionEvents to NATS on 'augur.sensus.typing'.

Requires root on Linux (keyboard library needs /dev/input access).
Run with: sudo .venv/bin/python sensus/typing_monitor.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import keyboard
import nats
import redis

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tabula.config import AugurConfig
from tabula.connections import connect_redis
from tabula.contracts import PerceptionEvent
from tabula.heartbeat import start_heartbeat
from tabula.persistence import PersistenceManager
from tabula.session import SessionManager

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("typing_monitor")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NATS_SUBJECT = "augur.sensus.typing"

DOMAIN = "typing"
STREAM_ID = "typing_rhythm"
ENTITY = "user"

PAUSE_THRESHOLD_S = 3.0  # gap > 3s = pause event
ROLLING_WINDOW_S = 5.0  # window for rolling speed
MIN_KEYPRESSES_BASELINE = 20  # minimum before scoring begins
SAMPLE_INTERVAL = 30  # publish a rhythm sample every N keypresses

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
GRAY = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"

# ---------------------------------------------------------------------------
# Typing tracker
# ---------------------------------------------------------------------------


class TypingTracker:
    """Tracks typing rhythm, detects pauses, computes rolling stats."""

    def __init__(self) -> None:
        self.keypress_count: int = 0
        self.last_keypress_time: float = 0.0
        self.keypresses_since_last_pause: int = 0
        self.session_start: float = time.monotonic()

        # Rolling window of recent keypress timestamps
        self._recent_times: deque[float] = deque()
        # Inter-keypress intervals for baseline
        self._intervals: deque[float] = deque(maxlen=200)

    def on_keypress(self) -> dict | None:
        """Process a keypress. Returns event info dict if something should be published."""
        now = time.monotonic()
        self.keypress_count += 1
        self.keypresses_since_last_pause += 1

        # Maintain rolling window
        self._recent_times.append(now)
        cutoff = now - ROLLING_WINDOW_S
        while self._recent_times and self._recent_times[0] < cutoff:
            self._recent_times.popleft()

        result = None

        if self.last_keypress_time > 0:
            gap = now - self.last_keypress_time
            self._intervals.append(gap)

            # Pause detection
            if gap >= PAUSE_THRESHOLD_S:
                result = {
                    "type": "pause",
                    "value": round(gap, 3),
                    "keypresses_since_last_pause": self.keypresses_since_last_pause - 1,
                }
                self.keypresses_since_last_pause = 1

        # Periodic rhythm sample
        if self.keypress_count > 0 and self.keypress_count % SAMPLE_INTERVAL == 0:
            avg_interval = self.avg_interval_ms()
            if avg_interval > 0:
                result = {
                    "type": "sample",
                    "value": round(avg_interval, 3),
                }

        self.last_keypress_time = now
        return result

    def rolling_kps(self) -> float:
        """Keypresses per second over the rolling window."""
        if len(self._recent_times) < 2:
            return 0.0
        span = self._recent_times[-1] - self._recent_times[0]
        if span < 0.01:
            return 0.0
        return (len(self._recent_times) - 1) / span

    def rolling_wpm(self) -> float:
        """Approximate words per minute (assuming 5 chars per word)."""
        return self.rolling_kps() * 60.0 / 5.0

    def avg_interval_ms(self) -> float:
        """Average inter-keypress interval in milliseconds."""
        if not self._intervals:
            return 0.0
        # Exclude pauses from the average
        normal = [i for i in self._intervals if i < PAUSE_THRESHOLD_S]
        if not normal:
            return 0.0
        return sum(normal) / len(normal) * 1000.0


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------


async def run() -> None:
    # ARCH-04 / ARCH-11: route Redis + NATS through AugurConfig and the
    # shared connect_redis helper so Docker deploy mode (and any AUGUR_*
    # env override) applies uniformly to this component.
    config = AugurConfig.from_env()

    redis_client = connect_redis(config)
    pm = PersistenceManager(redis_client)

    # Load existing baseline
    existing = pm.load_baseline(DOMAIN, ENTITY)
    if existing:
        log.info(
            "Restored baseline: %d observations, mean=%.2fms",
            existing.get("observation_count", 0),
            existing.get("ewma_mean", 0),
        )

    # Session
    session_mgr = SessionManager(redis_client)
    session_id = session_mgr.start()
    log.info("Session started: %s", session_id)

    # NATS
    nc = await nats.connect(
        config.nats_url, connect_timeout=config.nats_connect_timeout
    )
    hb_task = (
        start_heartbeat(nc, "sensus.typing", config.praefectus_heartbeat_interval_s)
        if config.praefectus_enabled
        else None
    )
    log.info("NATS connected (%s)", config.nats_url)

    # Publish session start
    try:
        await nc.publish(
            "augur.session.start",
            json.dumps(
                {
                    "session_id": session_id,
                    "domain": DOMAIN,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ).encode(),
        )
    except Exception as exc:
        log.error("Failed to publish session start: %s", exc)

    tracker = TypingTracker()

    # Queue for passing keypress events from the callback thread to asyncio
    event_queue: asyncio.Queue[dict] = asyncio.Queue()

    def on_key_event(e: keyboard.KeyboardEvent) -> None:
        """Callback from keyboard library (runs in a separate thread)."""
        if e.event_type != keyboard.KEY_DOWN:
            return
        result = tracker.on_keypress()
        if result is not None:
            # Thread-safe put into asyncio queue
            try:
                event_queue.put_nowait(result)
            except asyncio.QueueFull:
                # Backpressure: consumer is slow, drop this keypress rather than
                # block the keyboard hook thread (would freeze the UI).
                pass

    # Register global hook
    keyboard.hook(on_key_event)
    log.info("Keyboard hook registered (listening for keypresses)")

    print(f"\n{CYAN}{BOLD}  Augur Typing Monitor{RESET}", flush=True)
    print(f"{GRAY}  Listening for keypresses...{RESET}", flush=True)
    print(
        f"{GRAY}  Pause threshold: {PAUSE_THRESHOLD_S}s | Sample every {SAMPLE_INTERVAL} keys{RESET}",
        flush=True,
    )
    print(f"{GRAY}  Press Ctrl+C to stop.{RESET}\n", flush=True)

    try:
        while True:
            try:
                result = await asyncio.wait_for(event_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # Print periodic status
                if tracker.keypress_count > 0 and tracker.keypress_count % 50 == 0:
                    wpm = tracker.rolling_wpm()
                    log.info(
                        "Status: %d keypresses, %.1f WPM",
                        tracker.keypress_count,
                        wpm,
                    )
                continue

            ts = datetime.now(timezone.utc).isoformat()
            wpm = tracker.rolling_wpm()

            if result["type"] == "pause":
                event = PerceptionEvent(
                    domain=DOMAIN,
                    stream_id=STREAM_ID,
                    entity=ENTITY,
                    event_type="pause",
                    value=result["value"],
                    unit="seconds",
                    context={
                        "avg_wpm": round(wpm, 1),
                        "keypress_count": tracker.keypress_count,
                        "pause_position": result["keypresses_since_last_pause"],
                    },
                    timestamp=ts,
                    session_id=session_id,
                )
                log.info(
                    "%sPAUSE%s  %.1fs gap  (after %d keys, %.0f WPM)",
                    YELLOW,
                    RESET,
                    result["value"],
                    result["keypresses_since_last_pause"],
                    wpm,
                )

            elif result["type"] == "sample":
                event = PerceptionEvent(
                    domain=DOMAIN,
                    stream_id=STREAM_ID,
                    entity=ENTITY,
                    event_type="sample",
                    value=result["value"],
                    unit="ms",
                    context={
                        "avg_wpm": round(wpm, 1),
                        "keypress_count": tracker.keypress_count,
                        "label": f"{result['value']:.0f}ms avg interval",
                    },
                    timestamp=ts,
                    session_id=session_id,
                )
                log.info(
                    "%sSAMPLE%s  avg interval=%.0fms  (%d keys, %.0f WPM)",
                    GREEN,
                    RESET,
                    result["value"],
                    tracker.keypress_count,
                    wpm,
                )
            else:
                continue

            # Publish to NATS
            try:
                await nc.publish(NATS_SUBJECT, event.to_bytes())
                log.debug("Published to %s", NATS_SUBJECT)
            except Exception as exc:
                log.error("NATS publish failed: %s", exc)

            # Persist baseline update (use avg interval as the tracked value
            # for samples, and pause duration for pauses)
            try:
                # Save a lightweight baseline state
                baseline_state = {
                    "ewma_mean": tracker.avg_interval_ms(),
                    "observation_count": tracker.keypress_count,
                }
                pm.save_baseline(
                    DOMAIN,
                    ENTITY,
                    baseline_state,
                    ctx=pm.resolve_learn_context(event.session_id),
                )
            except redis.RedisError as exc:
                log.error("Failed to persist baseline: %s", exc)

    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        if hb_task is not None:
            hb_task.cancel()
        keyboard.unhook_all()
        log.info("Keyboard hooks removed")

        # Session end
        session_mgr.end()
        try:
            await nc.publish(
                "augur.session.end",
                json.dumps(
                    {
                        "session_id": session_id,
                        "domain": DOMAIN,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                ).encode(),
            )
            await nc.flush()
        except Exception as exc:
            log.error("Failed to publish session end: %s", exc)

        await nc.close()
        log.info("Shut down cleanly")


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log.info("Interrupted")


if __name__ == "__main__":
    main()
