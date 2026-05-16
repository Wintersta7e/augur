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


# ---------------------------------------------------------------------------
# Focus event builder
# ---------------------------------------------------------------------------


def _make_focus_state(mod, sampling_s=10.0, allowlist=()):
    return mod._FocusState(
        sampling_s=sampling_s,
        idle_threshold_s=60.0,
        title_allowlist=allowlist,
        source_id="test-host",
        session_id="sess-1",
    )


def test_focus_event_skipped_on_first_change(monkeypatch):
    mod = _import_module()
    state = _make_focus_state(mod)
    monkeypatch.setattr(mod.time, "monotonic", lambda: 100.0)
    # No prior focus: first change should return None.
    ev = state.on_focus_change(
        new_app="code",
        new_title="main.py - VS Code",
        now=100.0,
    )
    assert ev is None
    # State now remembers code as the active focus.
    assert state.current_app == "code"
    assert state.current_title == "main.py - VS Code"
    assert state.current_focus_started_at == 100.0


def test_focus_event_built_on_second_change():
    import math

    mod = _import_module()
    state = _make_focus_state(mod, allowlist=("code", "chrome"))
    # Prime with a first focus at t=100s.
    state.on_focus_change(new_app="code", new_title="main.py", now=100.0)
    # 30s later (with 5s idle inside the span), switch to chrome.
    state.last_input_time = 125.0  # last input was 5s before the switch
    ev = state.on_focus_change(new_app="chrome", new_title="news.com", now=130.0)
    assert ev is not None
    assert ev["domain"] == "activity_focus"
    assert ev["stream_id"] == "activity_focus"
    assert ev["entity"] == "code"  # entity = PREVIOUS app (whose dwell this measures)
    assert ev["event_type"] == "focus_change"
    assert ev["unit"] == "log1p_seconds"
    # value = log1p(active_dwell). active = 30s total - 5s idle = 25s.
    assert ev["value"] == pytest.approx(math.log1p(25.0), rel=1e-6)
    ctx = ev["context"]
    assert ctx["prev_app"] == "code"
    assert ctx["new_app"] == "chrome"
    assert ctx["prev_title"] == "main.py"
    assert ctx["new_title"] == "news.com"
    assert ctx["active_dwell_s"] == pytest.approx(25.0, rel=1e-6)
    assert ctx["idle_dwell_s"] == pytest.approx(5.0, rel=1e-6)
    assert ctx["total_dwell_s"] == pytest.approx(30.0, rel=1e-6)
    assert ctx["source_id"] == "test-host"
    assert ctx["span_id"]  # uuid for the PREVIOUS span
    assert ev["session_id"] == "sess-1"
    assert ev["timestamp"]  # ISO format from datetime.now(timezone.utc)


def test_focus_event_titles_filtered_when_app_not_in_allowlist():
    mod = _import_module()
    state = _make_focus_state(mod, allowlist=("code",))
    state.on_focus_change(new_app="code", new_title="main.py", now=100.0)
    state.last_input_time = 130.0
    ev = state.on_focus_change(new_app="chrome", new_title="secret-doc", now=130.0)
    ctx = ev["context"]
    assert ctx["prev_title"] == "main.py"  # allowed
    assert ctx["new_title"] is None  # filtered


def test_focus_event_clamps_idle_to_total_dwell():
    """Defensive: if last_input_time precedes focus start, idle == total."""
    mod = _import_module()
    state = _make_focus_state(mod)
    state.on_focus_change(new_app="code", new_title="x", now=100.0)
    # last_input_time is BEFORE current focus started (stale)
    state.last_input_time = 50.0
    ev = state.on_focus_change(new_app="chrome", new_title="y", now=110.0)
    ctx = ev["context"]
    assert ctx["total_dwell_s"] == pytest.approx(10.0, rel=1e-6)
    assert ctx["idle_dwell_s"] == pytest.approx(10.0, rel=1e-6)
    assert ctx["active_dwell_s"] == 0.0


def test_focus_event_assigns_new_span_id_after_change():
    mod = _import_module()
    state = _make_focus_state(mod)
    state.on_focus_change(new_app="code", new_title="x", now=100.0)
    first_span = state.current_span_id
    state.last_input_time = 110.0
    state.on_focus_change(new_app="chrome", new_title="y", now=110.0)
    assert state.current_span_id != first_span
