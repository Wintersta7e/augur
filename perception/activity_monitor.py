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

import logging
import math  # noqa: F401
import threading
import time  # noqa: F401
import uuid  # noqa: F401
from dataclasses import dataclass, field
from datetime import datetime, timezone  # noqa: F401

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


@dataclass
class _CounterState:
    """Mutable, thread-safe input counters for an intensity window.

    drain() atomically reads keystrokes + mouse_events and zeroes them.
    last_input_time is the unix timestamp of the most recent input; it
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
    idle_threshold_s: float
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
    ) -> dict | None:
        """Return a focus_change PerceptionEvent dict, or None on the first call."""
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
        if self.last_input_time < prev_started:
            idle_dwell = total_dwell
        else:
            idle_dwell = max(0.0, now - self.last_input_time)
            if idle_dwell > total_dwell:
                idle_dwell = total_dwell
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

        return {
            "domain": "activity_focus",
            "stream_id": "activity_focus",
            "entity": prev_app,
            "event_type": "focus_change",
            "value": math.log1p(active_dwell),
            "unit": "log1p_seconds",
            "context": ctx,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
        }


@dataclass
class _IntensityWindow:
    """Bounded-by-focus-span input-intensity sampler.

    Counts keystrokes + mouse events between resets. build() emits a
    sample if the window is long enough AND has enough events; otherwise
    returns None (caller resets either way).
    """

    sampling_s: float
    min_events: int
    min_window_s: float
    idle_threshold_s: float
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
    ) -> dict | None:
        """Return an intensity_sample PerceptionEvent dict, or None if dropped."""
        window_duration = max(0.0, now - self.window_started_at)
        total_events = self.keystrokes + self.mouse_events
        if window_duration < self.min_window_s:
            return None
        if total_events < self.min_events:
            return None

        ipm = (60.0 * total_events / window_duration) if window_duration > 0 else 0.0

        if self.last_input_time < self.window_started_at:
            idle_seconds = window_duration
        else:
            idle_seconds = max(0.0, now - self.last_input_time)
            if idle_seconds > window_duration:
                idle_seconds = window_duration

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

        return {
            "domain": "activity_intensity",
            "stream_id": "activity_intensity",
            "entity": focused_app,
            "event_type": "intensity_sample",
            "value": ipm,
            "unit": "ipm",
            "context": ctx,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
        }


class ActivityMonitor:
    """Windows-host daemon. See module docstring."""

    def __init__(self) -> None:
        # Concrete construction lands in Tasks 5–8.
        raise NotImplementedError("ActivityMonitor wiring lands in later tasks")
