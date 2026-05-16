"""Activity perception daemon — Windows-side, publishes to WSL NATS.

Emits two PerceptionEvent streams:
  * augur.perception.activity_focus     — on every active-window change
  * augur.perception.activity_intensity — every activity_sampling_s seconds

Windows-only at runtime (Win32 hooks + global input listeners), but the
module is intentionally importable on Linux CI so unit tests can patch
the OS deps via sys.modules injection.

Run on the Windows host:
    pip install -r requirements-windows.txt
    python -m perception.activity_monitor
"""

from __future__ import annotations

import asyncio
import logging
import math
import socket
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from blackboard.contracts import PerceptionEvent
from blackboard.session import get_active_session

log = logging.getLogger("activity_monitor")

# Lazy import: keep the module importable on Linux CI. Tests inject fakes
# via sys.modules BEFORE importing this file; the real CLI entrypoint
# checks _WIN32_AVAILABLE and exits cleanly if the deps are missing.
try:
    import win32gui  # type: ignore[import-not-found]
    import win32process  # type: ignore[import-not-found]
    import psutil  # type: ignore[import-not-found]
    from pynput import keyboard as _kb  # type: ignore[import-not-found]
    from pynput import mouse as _mouse  # type: ignore[import-not-found]

    _WIN32_AVAILABLE = True
except ImportError:  # pragma: no cover - Linux CI path
    win32gui = None  # type: ignore[assignment]
    win32process = None  # type: ignore[assignment]
    psutil = None  # type: ignore[assignment]
    _kb = None  # type: ignore[assignment]
    _mouse = None  # type: ignore[assignment]
    _WIN32_AVAILABLE = False


# ---------------------------------------------------------------------------
# Pure helpers (no OS calls)
# ---------------------------------------------------------------------------

_UNKNOWN_APP = "<unknown>"
_NO_FOREGROUND_APP = "<no_foreground>"
_DENIED_APP = "<denied>"
_GONE_APP = "<gone>"


def _normalize_app_name(exe_path: str | None) -> str:
    """Strip path, drop `.exe`, lowercase. Returns "<unknown>" for empty input."""
    if not exe_path:
        return _UNKNOWN_APP
    # Handle both / and \ path separators.
    name = exe_path.replace("\\", "/").rsplit("/", 1)[-1]
    if name.lower().endswith(".exe"):
        name = name[:-4]
    return name.lower() or _UNKNOWN_APP


def _resolve_title(
    app: str, raw_title: str | None, allowlist: tuple[str, ...]
) -> str | None:
    """Return raw_title if app is in allowlist; else None (privacy default)."""
    if not raw_title:
        return None
    if app in allowlist:
        return raw_title
    return None


def _clamp_idle_seconds(last_input_time: float, span_start: float, now: float) -> float:
    """Compute the idle portion of a [span_start, now] window.

    If last_input_time predates span_start, the whole span is idle.
    Otherwise idle = max(0, now - last_input_time), clamped to span duration.
    """
    span_duration = max(0.0, now - span_start)
    if last_input_time < span_start:
        return span_duration
    idle = max(0.0, now - last_input_time)
    return min(idle, span_duration)


@dataclass
class _CounterState:
    """Mutable, thread-safe input counters for an intensity window.

    drain() atomically reads keystrokes + mouse_events and zeroes them.
    last_input_time is the monotonic-clock timestamp of the most recent input; it
    is NOT reset by drain because it's used to compute idle_seconds in
    the NEXT sample as well.
    """

    keystrokes: int = 0
    mouse_events: int = 0
    last_input_time: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_keystroke(self) -> None:
        with self._lock:
            self.keystrokes += 1

    def record_mouse_event(self) -> None:
        with self._lock:
            self.mouse_events += 1

    def touch(self, now: float) -> None:
        with self._lock:
            self.last_input_time = now

    def drain(self) -> tuple[int, int, float]:
        with self._lock:
            k = self.keystrokes
            m = self.mouse_events
            t = self.last_input_time
            self.keystrokes = 0
            self.mouse_events = 0
            return k, m, t


@dataclass
class _FocusState:
    """Tracks the current focused app + the dwell window for the previous app.

    on_focus_change emits a focus_change event for the PREVIOUS span on
    every transition AFTER the first. The first transition primes state
    only (no zero-dwell sample).
    """

    sampling_s: float
    title_allowlist: tuple[str, ...]
    source_id: str
    session_id: str

    current_app: str | None = None
    current_title: str | None = None
    current_focus_started_at: float | None = None
    current_span_id: str | None = None
    last_input_time: float = 0.0

    def on_focus_change(
        self,
        new_app: str,
        new_title: str | None,
        now: float,
    ) -> PerceptionEvent | None:
        """Return a focus_change PerceptionEvent, or None on the first call."""
        prev_app = self.current_app
        prev_title = self.current_title
        prev_started = self.current_focus_started_at
        prev_span = self.current_span_id

        # Advance state to the NEW focus regardless of whether we emit.
        self.current_app = new_app
        self.current_title = new_title
        self.current_focus_started_at = now
        self.current_span_id = str(uuid.uuid4())

        if prev_app is None or prev_started is None:
            return None  # first-event skip

        total_dwell = max(0.0, now - prev_started)
        # idle = time the user stopped giving input before the focus changed.
        # If last_input_time predates the focus start (stale), treat the whole
        # span as idle.
        idle_dwell = _clamp_idle_seconds(self.last_input_time, prev_started, now)
        active_dwell = total_dwell - idle_dwell

        ctx = {
            "prev_app": prev_app,
            "new_app": new_app,
            "prev_title": _resolve_title(prev_app, prev_title, self.title_allowlist),
            "new_title": _resolve_title(new_app, new_title, self.title_allowlist),
            "active_dwell_s": active_dwell,
            "idle_dwell_s": idle_dwell,
            "total_dwell_s": total_dwell,
            "source_id": self.source_id,
            "span_id": prev_span,
        }

        return PerceptionEvent(
            domain="activity_focus",
            stream_id="activity_focus",
            entity=prev_app,
            event_type="focus_change",
            value=math.log1p(active_dwell),
            unit="log1p_seconds",
            context=ctx,
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=self.session_id,
        )


@dataclass
class _IntensityWindow:
    """Bounded-by-focus-span input-intensity sampler."""

    sampling_s: float
    min_events: int
    min_window_s: float
    title_allowlist: tuple[str, ...]
    source_id: str
    session_id: str

    window_started_at: float = 0.0
    span_id: str | None = None
    keystrokes: int = 0
    mouse_events: int = 0
    last_input_time: float = 0.0

    def reset(self, new_started_at: float, new_span_id: str | None) -> None:
        self.window_started_at = new_started_at
        self.span_id = new_span_id
        self.keystrokes = 0
        self.mouse_events = 0
        # last_input_time is intentionally NOT reset.

    def build(
        self,
        focused_app: str,
        focused_title: str | None,
        now: float,
    ) -> PerceptionEvent | None:
        """Return an intensity_sample PerceptionEvent, or None if dropped."""
        window_duration = max(0.0, now - self.window_started_at)
        total_events = self.keystrokes + self.mouse_events
        if window_duration < self.min_window_s:
            return None
        if total_events < self.min_events:
            return None

        ipm = (60.0 * total_events / window_duration) if window_duration > 0 else 0.0

        idle_seconds = _clamp_idle_seconds(
            self.last_input_time, self.window_started_at, now
        )

        ctx = {
            "focused_app": focused_app,
            "title": _resolve_title(focused_app, focused_title, self.title_allowlist),
            "keystroke_count": self.keystrokes,
            "mouse_event_count": self.mouse_events,
            "idle_seconds": idle_seconds,
            "window_duration_s": window_duration,
            "source_id": self.source_id,
            "span_id": self.span_id,
        }

        return PerceptionEvent(
            domain="activity_intensity",
            stream_id="activity_intensity",
            entity=focused_app,
            event_type="intensity_sample",
            value=ipm,
            unit="ipm",
            context=ctx,
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=self.session_id,
        )


@dataclass
class _SessionReader:
    """Wraps get_active_session with last_seen tracking + change detection.

    On every read, updates `last_seen` and sets `changed_since_last`
    when the session_id changes between reads. Callers use this signal
    to flush the drop log and reset state machines.
    """

    redis_client: Any  # redis.Redis at runtime
    max_age_h: float

    last_seen: str | None = None
    changed_since_last: bool = False

    def read_current(self) -> str | None:
        session_id = get_active_session(self.redis_client, max_age_h=self.max_age_h)
        self._update_seen(session_id)
        return session_id

    def _update_seen(self, session_id: str | None) -> None:
        self.changed_since_last = (session_id is not None) and (
            session_id != self.last_seen
        )
        self.last_seen = session_id


class _DroppedEventLog:
    """Best-effort drop log for NATS publish failures. Does NOT replay.

    Holds the last N (capacity) payloads that couldn't reach NATS so the
    operator can see what was lost. `dropped_total` increments on every
    overflow (oldest pushed out) AND on every flush (events discarded on
    session change because replaying into a new session would contaminate
    detection — see spec §8). Surfaced via log warnings; not a buffer to
    drain back into NATS.
    """

    def __init__(self, capacity: int = 200) -> None:
        self._capacity = capacity
        self._dq: deque[dict[str, Any]] = deque(maxlen=capacity)
        self.dropped_total: int = 0

    def enqueue(self, payload: dict[str, Any]) -> None:
        if len(self._dq) == self._capacity:
            self.dropped_total += 1
            log.warning(
                "activity_monitor: drop-log full (capacity=%d, total dropped=%d), "
                "oldest event evicted",
                self._capacity,
                self.dropped_total,
            )
        self._dq.append(payload)

    def drain(self) -> list[dict[str, Any]]:
        """For tests/inspection only. Production code does not call this."""
        out = list(self._dq)
        self._dq.clear()
        return out

    def flush(self) -> None:
        n = len(self._dq)
        if n:
            self.dropped_total += n
            log.info(
                "activity_monitor: drop-log flushed on session change "
                "(%d events discarded, total dropped=%d)",
                n,
                self.dropped_total,
            )
        self._dq.clear()


def _probe_nats_reachable(nats_url: str, timeout_s: float = 5.0) -> bool:
    """TCP-connect probe. Real protocol handshake happens later via nats-py."""
    try:
        from urllib.parse import urlparse

        parsed = urlparse(nats_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 4222
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except (OSError, ValueError):
        return False


def _probe_redis_reachable(redis_url: str, timeout_s: float = 5.0) -> bool:
    """TCP-connect probe for Redis (default port 6379)."""
    try:
        from urllib.parse import urlparse

        parsed = urlparse(redis_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 6379
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except (OSError, ValueError):
        return False


class ActivityMonitor:
    """Windows-host daemon. Composes the helpers above + NATS/Redis I/O."""

    SUBJECT_FOCUS = "augur.perception.activity_focus"
    SUBJECT_INTENSITY = "augur.perception.activity_intensity"

    def __init__(
        self,
        config: Any,  # AugurConfig at runtime
        redis_client: object,
        nats_client: Any,
    ) -> None:
        self.config = config
        self.redis_client = redis_client
        self.nats = nats_client
        self.counter = _CounterState()
        self.drops = _DroppedEventLog(capacity=200)
        self.session_reader = _SessionReader(
            redis_client, max_age_h=config.session_max_age_h
        )
        self.focus: _FocusState | None = None
        self.intensity: _IntensityWindow | None = None
        self.allowlist = tuple(
            s.strip() for s in config.activity_title_allowlist.split(",") if s.strip()
        )

    def _build_focus_state(self, session_id: str) -> _FocusState:
        return _FocusState(
            sampling_s=self.config.activity_sampling_s,
            title_allowlist=self.allowlist,
            source_id=self.config.activity_source_id,
            session_id=session_id,
        )

    def _build_intensity_window(self, session_id: str) -> _IntensityWindow:
        return _IntensityWindow(
            sampling_s=self.config.activity_sampling_s,
            min_events=self.config.activity_intensity_min_events,
            min_window_s=self.config.activity_intensity_min_window_s,
            title_allowlist=self.allowlist,
            source_id=self.config.activity_source_id,
            session_id=session_id,
        )

    def _resolve_session(self) -> str | None:
        sid = self.session_reader.read_current()
        # On session change: reset the state machines entirely so the next
        # iteration rebuilds fresh _FocusState / _IntensityWindow under the new
        # session_id. Keeping their carry-over state would emit a span
        # belonging to the old session tagged with the new session_id.
        if self.session_reader.changed_since_last:
            self.drops.flush()
            self.focus = None
            self.intensity = None
            # Counter holds last_input_time; safe to keep (input that arrived
            # in old session won't be attributed to new since both classes will
            # be rebuilt from scratch).
        return sid

    def _get_foreground(self) -> tuple[str, str | None]:
        """Return (app_name, window_title). Distinguishes failure modes."""
        if not _WIN32_AVAILABLE:
            return (_UNKNOWN_APP, None)
        try:
            hwnd = win32gui.GetForegroundWindow()
            if not hwnd:
                return (_NO_FOREGROUND_APP, None)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "activity_monitor: GetForegroundWindow failed (%s): %s",
                type(exc).__name__,
                exc,
            )
            return (_UNKNOWN_APP, None)
        try:
            title = win32gui.GetWindowText(hwnd) or None
        except Exception:  # noqa: BLE001
            title = None
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            if not pid:
                return (_NO_FOREGROUND_APP, title)
            exe = psutil.Process(pid).name()
        except psutil.AccessDenied:
            return (_DENIED_APP, title)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            return (_GONE_APP, title)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "activity_monitor: process lookup failed (%s): %s",
                type(exc).__name__,
                exc,
            )
            return (_UNKNOWN_APP, title)
        return (_normalize_app_name(exe), title)

    async def _publish(self, subject: str, payload: PerceptionEvent) -> None:
        try:
            await self.nats.publish(subject, payload.to_bytes())
        except Exception as exc:  # noqa: BLE001
            log.warning("activity_monitor: publish failed, buffering: %s", exc)
            # Drop log expects dicts (legacy shape); convert before storing.
            self.drops.enqueue(
                {
                    "subject": subject,
                    "payload": payload.to_json()
                    if hasattr(payload, "to_json")
                    else str(payload),
                }
            )

    async def run(self) -> None:  # pragma: no cover - exercised manually
        """Main loop. Real Win32 hook wiring happens here.

        On every poll tick we:
          1. resolve the current session (returns None if absent/stale/ended)
          2. if no session, sleep + retry
          3. poll active window; on change, emit focus_change + reset intensity
          4. if sampling_s elapsed since last intensity sample, emit one
        """
        sampling = self.config.activity_sampling_s
        last_sampled: float | None = None
        last_app: str | None = None
        last_title: str | None = None

        # Install global input listeners (best-effort).
        listeners: list[Any] = []
        last_listener_check = time.monotonic()
        if _WIN32_AVAILABLE:

            def _on_key(_ev: Any) -> None:
                try:
                    now = time.monotonic()
                    self.counter.record_keystroke()
                    self.counter.touch(now)
                except Exception as exc:  # noqa: BLE001
                    log.warning("activity_monitor: _on_key callback failed: %s", exc)

            def _on_mouse(*_args: Any) -> None:
                try:
                    now = time.monotonic()
                    self.counter.record_mouse_event()
                    self.counter.touch(now)
                except Exception as exc:  # noqa: BLE001
                    log.warning("activity_monitor: _on_mouse callback failed: %s", exc)

            try:
                kbd = _kb.Listener(on_press=_on_key)  # type: ignore[union-attr]
                kbd.start()
                listeners.append(kbd)
                mse = _mouse.Listener(on_click=_on_mouse, on_scroll=_on_mouse)  # type: ignore[union-attr]
                mse.start()
                listeners.append(mse)
            except Exception as exc:  # noqa: BLE001
                log.warning("activity_monitor: input listeners failed: %s", exc)

        try:
            while True:
                sid = self._resolve_session()
                if sid is None:
                    log.info("activity_monitor: waiting for active session...")
                    await asyncio.sleep(5.0)
                    continue

                if self.focus is None:
                    self.focus = self._build_focus_state(sid)
                    last_sampled = (
                        time.monotonic()
                    )  # start sampling timer from focus init
                    last_app = None
                    last_title = None
                if self.intensity is None:
                    self.intensity = self._build_intensity_window(sid)
                    self.intensity.reset(
                        new_started_at=time.monotonic(),
                        new_span_id=self.focus.current_span_id,
                    )

                now_mono = time.monotonic()

                # Periodic listener health check (every ~10s).
                if now_mono - last_listener_check >= 10.0:
                    for listener in listeners:
                        if hasattr(listener, "is_alive") and not listener.is_alive():
                            log.error(
                                "activity_monitor: input listener died (counter drain "
                                "will return zeros). Daemon will continue but input "
                                "intensity readings are unreliable."
                            )
                    last_listener_check = now_mono

                app, title = self._get_foreground()

                # Pull counters and update intensity state.
                k, m, last_input = self.counter.drain()
                self.intensity.keystrokes += k
                self.intensity.mouse_events += m
                if last_input:
                    self.intensity.last_input_time = last_input
                    self.focus.last_input_time = last_input

                # Focus change?
                if app != last_app:
                    # Emit any in-flight intensity sample BEFORE the focus change,
                    # truncated to the current focus span.
                    if last_app is not None and self.intensity.window_started_at > 0:
                        ev = self.intensity.build(
                            focused_app=last_app, focused_title=last_title, now=now_mono
                        )
                        if ev is not None:
                            await self._publish(self.SUBJECT_INTENSITY, ev)

                    ev_focus = self.focus.on_focus_change(
                        new_app=app, new_title=title, now=now_mono
                    )
                    if ev_focus is not None:
                        await self._publish(self.SUBJECT_FOCUS, ev_focus)

                    self.intensity.reset(
                        new_started_at=now_mono, new_span_id=self.focus.current_span_id
                    )
                    last_sampled = now_mono
                    last_app, last_title = app, title

                # Periodic intensity sample?
                elif last_sampled is not None and now_mono - last_sampled >= sampling:
                    ev = self.intensity.build(
                        focused_app=app, focused_title=title, now=now_mono
                    )
                    if ev is not None:
                        await self._publish(self.SUBJECT_INTENSITY, ev)
                    self.intensity.reset(
                        new_started_at=now_mono, new_span_id=self.focus.current_span_id
                    )
                    last_sampled = now_mono

                await asyncio.sleep(0.25)
        finally:
            for listener in listeners:
                try:
                    listener.stop()
                except Exception as exc:  # noqa: BLE001
                    log.debug("activity_monitor: listener.stop() failed: %s", exc)


def main() -> None:  # pragma: no cover - CLI entrypoint
    """CLI entrypoint. Refuses to run if Win32 deps are missing."""
    if not _WIN32_AVAILABLE:
        sys.stderr.write(
            "activity_monitor: Windows dependencies missing.\n"
            "  Install: pip install -r requirements-windows.txt\n"
        )
        sys.exit(2)

    # Lazy imports for runtime-only paths (keeps tests importable).
    from blackboard.config import AugurConfig
    from blackboard.connections import connect_redis
    import nats

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = AugurConfig.from_env()
    if not _probe_nats_reachable(config.nats_url):
        sys.stderr.write(
            f"activity_monitor: NATS unreachable at {config.nats_url}.\n"
            "  1. Verify WSL2 distro is running:   wsl -l -v\n"
            "  2. Verify docker-compose is up:     docker compose ps\n"
            "  3. Or set explicit IP:              "
            "export AUGUR_NATS_URL=nats://$(wsl hostname -I | awk '{print $1}'):4222\n"
        )
        sys.exit(3)

    if not _probe_redis_reachable(config.redis_url):
        sys.stderr.write(
            f"activity_monitor: Redis unreachable at {config.redis_url}.\n"
            "  1. Verify WSL2 distro is running:   wsl -l -v\n"
            "  2. Verify docker-compose is up:     docker compose ps\n"
            "  3. Or set explicit IP:              "
            "export AUGUR_REDIS_URL=redis://$(wsl hostname -I | awk '{print $1}'):6379\n"
        )
        sys.exit(4)

    redis_client = connect_redis(config)

    async def _run() -> None:
        nc = await nats.connect(
            config.nats_url, connect_timeout=config.nats_connect_timeout
        )
        mon = ActivityMonitor(config, redis_client, nc)
        try:
            await mon.run()
        finally:
            await nc.drain()

    asyncio.run(_run())


if __name__ == "__main__":  # pragma: no cover
    main()
