"""Unit tests for perception/activity_monitor.py — helpers + event builders.

OS modules (pywin32/pynput/psutil) are injected as MagicMock into
sys.modules BEFORE activity_monitor is imported, so the module is
importable on Linux CI even though its real deps are Windows-only.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def fake_win_modules(monkeypatch):
    """Inject fakes for Windows-only deps so the module imports cleanly."""
    fakes = {
        "win32gui": MagicMock(),
        "win32process": MagicMock(),
        "psutil": MagicMock(),
        "pynput": MagicMock(),
        "pynput.keyboard": MagicMock(),
        "pynput.mouse": MagicMock(),
    }
    for name, mod in fakes.items():
        monkeypatch.setitem(sys.modules, name, mod)
    # Force re-import in case a prior test loaded it.
    sys.modules.pop("perception.activity_monitor", None)
    yield fakes


def _import_module():
    import perception.activity_monitor as mod  # noqa: PLC0415

    return mod


def test_module_importable_on_linux_ci():
    mod = _import_module()
    assert hasattr(mod, "ActivityMonitor")
    # Fakes are injected so the win-available flag is True.
    assert mod._WIN32_AVAILABLE is True


def test_normalize_app_name_strips_path_and_lowercases():
    mod = _import_module()
    assert mod._normalize_app_name(r"C:\Program Files\Code\Code.exe") == "code"
    assert mod._normalize_app_name("Chrome.exe") == "chrome"
    assert mod._normalize_app_name("/usr/bin/Firefox") == "firefox"
    assert mod._normalize_app_name("") == "<unknown>"
    assert mod._normalize_app_name(None) == "<unknown>"


def test_normalize_app_name_handles_no_extension():
    mod = _import_module()
    assert mod._normalize_app_name("bash") == "bash"


def test_resolve_title_returns_none_when_app_not_in_allowlist():
    mod = _import_module()
    assert mod._resolve_title("chrome", "Personal email - inbox", ("code",)) is None


def test_resolve_title_returns_title_when_app_in_allowlist():
    mod = _import_module()
    assert (
        mod._resolve_title("code", "main.py - augur - VS Code", ("code", "terminal"))
        == "main.py - augur - VS Code"
    )


def test_resolve_title_handles_empty_allowlist():
    mod = _import_module()
    assert mod._resolve_title("anything", "anything", ()) is None


def test_resolve_title_handles_none_title():
    mod = _import_module()
    assert mod._resolve_title("code", None, ("code",)) is None


def test_drain_counters_returns_and_resets():
    mod = _import_module()
    state = mod._CounterState()
    state.keystrokes = 17
    state.mouse_events = 4
    state.last_input_time = 1234.5
    k, m, last = state.drain()
    assert (k, m, last) == (17, 4, 1234.5)
    assert state.keystrokes == 0
    assert state.mouse_events == 0
    # last_input_time is intentionally NOT reset (still used to detect idle).
    assert state.last_input_time == 1234.5


def test_drain_counters_thread_safe_under_concurrent_increment(monkeypatch):
    """drain() takes the lock; concurrent increments should not lose counts."""
    import threading

    mod = _import_module()
    state = mod._CounterState()

    stop = threading.Event()

    def hammer():
        while not stop.is_set():
            state.record_keystroke()

    t = threading.Thread(target=hammer, daemon=True)
    t.start()
    try:
        # Drain a few times; nothing should crash.
        for _ in range(50):
            state.drain()
    finally:
        stop.set()
        t.join(timeout=1.0)
