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
import threading
from dataclasses import dataclass, field

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


class ActivityMonitor:
    """Windows-host daemon. See module docstring."""

    def __init__(self) -> None:
        # Concrete construction lands in Tasks 5–8.
        raise NotImplementedError("ActivityMonitor wiring lands in later tasks")
