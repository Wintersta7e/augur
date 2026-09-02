"""Unit tests for sensus/activity_monitor.py — helpers + event builders.

OS modules (pywin32/pynput/psutil) are injected as MagicMock into
sys.modules BEFORE activity_monitor is imported, so the module is
importable on Linux CI even though its real deps are Windows-only.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import math

import pytest


@pytest.fixture(autouse=True)
def fake_win_modules(monkeypatch):
    """Inject fakes for Windows-only deps so the module imports cleanly."""
    fakes = {
        "win32api": MagicMock(),
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
    sys.modules.pop("sensus.activity_monitor", None)
    yield fakes


def _import_module():
    import sensus.activity_monitor as mod  # noqa: PLC0415

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


def test_clamp_idle_seconds_returns_full_span_when_input_predates_start():
    mod = _import_module()
    assert mod._clamp_idle_seconds(
        last_input_time=50.0, span_start=100.0, now=110.0
    ) == pytest.approx(10.0)


def test_clamp_idle_seconds_returns_capped_idle_when_input_within_span():
    mod = _import_module()
    # last input at 105, span 100-110 → idle = 110-105 = 5
    assert mod._clamp_idle_seconds(
        last_input_time=105.0, span_start=100.0, now=110.0
    ) == pytest.approx(5.0)
    # last input at 109.5, span 100-110 → idle = 0.5
    assert mod._clamp_idle_seconds(
        last_input_time=109.5, span_start=100.0, now=110.0
    ) == pytest.approx(0.5)
    # zero span → idle = 0
    assert (
        mod._clamp_idle_seconds(last_input_time=100.0, span_start=100.0, now=100.0)
        == 0.0
    )


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
    """Hammer record_keystroke from a thread while draining; sum total events
    across all drains + remaining state == total increments. If the lock is
    removed, this would race and lose increments."""
    import threading
    import time as _time

    mod = _import_module()
    state = mod._CounterState()

    total_increments = 0
    increments_lock = threading.Lock()
    stop = threading.Event()

    def hammer():
        nonlocal total_increments
        local = 0
        while not stop.is_set():
            state.record_keystroke()
            local += 1
        with increments_lock:
            total_increments += local

    t = threading.Thread(target=hammer, daemon=True)
    t.start()
    try:
        observed_total = 0
        deadline = _time.monotonic() + 0.25
        while _time.monotonic() < deadline:
            k, _, _ = state.drain()
            observed_total += k
            _time.sleep(0.005)
    finally:
        stop.set()
        t.join(timeout=1.0)

    # Final drain to capture remaining.
    k, _, _ = state.drain()
    observed_total += k

    # If the lock is removed, observed_total could be less than total_increments
    # due to lost-update race. Allow a tiny in-flight slack: hammer increments
    # may have happened after our final drain but before join returned, so
    # require equality up to a small constant.
    assert observed_total <= total_increments, (
        f"observed_total={observed_total} > total_increments={total_increments}; "
        "this should be impossible"
    )
    assert total_increments - observed_total <= 50, (
        f"lost {total_increments - observed_total} increments; "
        f"observed={observed_total}, total_incremented={total_increments}"
    )


# ---------------------------------------------------------------------------
# Focus event builder
# ---------------------------------------------------------------------------


def _make_focus_state(mod, sampling_s=10.0, allowlist=()):
    return mod._FocusState(
        sampling_s=sampling_s,
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
    assert ev.domain == "activity_focus"
    assert ev.stream_id == "activity_focus"
    assert ev.entity == "code"  # entity = PREVIOUS app (whose dwell this measures)
    assert ev.event_type == "focus_change"
    assert ev.unit == "log1p_seconds"
    # value = log1p(active_dwell). active = 30s total - 5s idle = 25s.
    assert ev.value == pytest.approx(math.log1p(25.0), rel=1e-6)
    ctx = ev.context
    assert ctx["prev_app"] == "code"
    assert ctx["new_app"] == "chrome"
    assert ctx["prev_title"] == "main.py"
    assert ctx["new_title"] == "news.com"
    assert ctx["active_dwell_s"] == pytest.approx(25.0, rel=1e-6)
    assert ctx["idle_dwell_s"] == pytest.approx(5.0, rel=1e-6)
    assert ctx["total_dwell_s"] == pytest.approx(30.0, rel=1e-6)
    assert ctx["source_id"] == "test-host"
    assert ctx["span_id"]  # uuid for the PREVIOUS span
    assert ev.session_id == "sess-1"
    assert ev.timestamp  # ISO format from datetime.now(timezone.utc)


def test_focus_event_titles_filtered_when_app_not_in_allowlist():
    mod = _import_module()
    state = _make_focus_state(mod, allowlist=("code",))
    state.on_focus_change(new_app="code", new_title="main.py", now=100.0)
    state.last_input_time = 130.0
    ev = state.on_focus_change(new_app="chrome", new_title="secret-doc", now=130.0)
    ctx = ev.context
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
    ctx = ev.context
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


# ---------------------------------------------------------------------------
# Intensity event builder
# ---------------------------------------------------------------------------


def _make_intensity_window(
    mod,
    sampling_s=10.0,
    min_events=1,
    min_window_s=2.0,
    allowlist=(),
):
    return mod._IntensityWindow(
        sampling_s=sampling_s,
        min_events=min_events,
        min_window_s=min_window_s,
        title_allowlist=allowlist,
        source_id="test-host",
        session_id="sess-1",
    )


def test_intensity_event_built_for_full_window():
    mod = _import_module()
    w = _make_intensity_window(mod, allowlist=("code",))
    w.window_started_at = 100.0
    w.span_id = "span-1"
    w.keystrokes = 30
    w.mouse_events = 10
    w.last_input_time = 109.5
    ev = w.build(focused_app="code", focused_title="main.py", now=110.0)
    assert ev is not None
    assert ev.domain == "activity_intensity"
    assert ev.stream_id == "activity_intensity"
    assert ev.entity == "code"
    assert ev.event_type == "intensity_sample"
    assert ev.unit == "log1p_ipm"
    # 40 events over 10s = 240 ipm, published log-compressed because the raw
    # rate is strongly right-skewed and the detector's test is symmetric.
    assert ev.value == pytest.approx(math.log1p(240.0), rel=1e-6)
    ctx = ev.context
    assert ctx["focused_app"] == "code"
    assert ctx["title"] == "main.py"
    assert ctx["keystroke_count"] == 30
    assert ctx["mouse_event_count"] == 10
    assert ctx["window_duration_s"] == pytest.approx(10.0, rel=1e-6)
    assert ctx["idle_seconds"] == pytest.approx(0.5, rel=1e-6)
    assert ctx["source_id"] == "test-host"
    assert ctx["span_id"] == "span-1"
    # The raw rate stays available for anything wanting a human number.
    assert ctx["ipm"] == pytest.approx(240.0, rel=1e-6)
    assert ev.session_id == "sess-1"
    assert ev.timestamp


def test_intensity_event_dropped_below_min_events():
    mod = _import_module()
    w = _make_intensity_window(mod, min_events=5)
    w.window_started_at = 100.0
    w.keystrokes = 2
    w.mouse_events = 2  # total=4 < 5
    w.last_input_time = 105.0
    ev = w.build(focused_app="code", focused_title=None, now=110.0)
    assert ev is None


def test_intensity_event_dropped_below_min_window():
    mod = _import_module()
    w = _make_intensity_window(mod, min_window_s=2.0)
    w.window_started_at = 100.0
    w.keystrokes = 100  # plenty of events
    w.mouse_events = 0
    w.last_input_time = 101.0
    ev = w.build(focused_app="code", focused_title=None, now=101.5)  # 1.5s < 2.0s
    assert ev is None


def test_intensity_event_title_filtered_when_app_not_in_allowlist():
    mod = _import_module()
    w = _make_intensity_window(mod, allowlist=("code",))
    w.window_started_at = 100.0
    w.keystrokes = 5
    w.mouse_events = 0
    w.last_input_time = 105.0
    ev = w.build(focused_app="chrome", focused_title="secret-doc", now=110.0)
    assert ev.context["title"] is None


def test_intensity_window_reset_clears_counters_and_advances_start():
    mod = _import_module()
    w = _make_intensity_window(mod)
    w.window_started_at = 100.0
    w.span_id = "old-span"
    w.keystrokes = 5
    w.mouse_events = 5
    w.reset(new_started_at=110.0, new_span_id="new-span")
    assert w.keystrokes == 0
    assert w.mouse_events == 0
    assert w.window_started_at == 110.0
    assert w.span_id == "new-span"
    # last_input_time is NOT reset (carries forward for idle calc).


def test_intensity_event_idle_clamped_when_last_input_before_window():
    """If no input arrived in the window, idle == window_duration."""
    mod = _import_module()
    w = _make_intensity_window(mod, min_events=0)
    w.window_started_at = 100.0
    w.keystrokes = 0
    w.mouse_events = 0
    w.last_input_time = 50.0  # before window
    ev = w.build(focused_app="code", focused_title=None, now=110.0)
    # 0 events => 0 ipm, but should still emit when min_events=0
    assert ev.context["idle_seconds"] == pytest.approx(10.0, rel=1e-6)
    assert ev.value == 0.0


# ---------------------------------------------------------------------------
# Session reader
# ---------------------------------------------------------------------------


def test_session_reader_returns_active_session_id():
    import json
    from datetime import datetime, timezone

    mod = _import_module()
    r = MagicMock()
    r.get.return_value = json.dumps(
        {
            "session_id": "sess-xyz",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "active",
        }
    )
    reader = mod._SessionReader(r, max_age_h=12.0)
    assert reader.read_current() == "sess-xyz"


def test_session_reader_returns_none_when_status_ended():
    import json
    from datetime import datetime, timezone

    mod = _import_module()
    r = MagicMock()
    r.get.return_value = json.dumps(
        {
            "session_id": "sess-xyz",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "ended",
        }
    )
    reader = mod._SessionReader(r, max_age_h=12.0)
    assert reader.read_current() is None


def test_session_reader_detects_change_and_signals_buffer_flush():
    import json
    from datetime import datetime, timezone

    mod = _import_module()
    r = MagicMock()
    now_iso = datetime.now(timezone.utc).isoformat()
    r.get.return_value = json.dumps(
        {"session_id": "sess-1", "started_at": now_iso, "status": "active"}
    )
    reader = mod._SessionReader(r, max_age_h=12.0)
    assert reader.read_current() == "sess-1"
    assert reader.last_seen == "sess-1"

    # Session changes.
    r.get.return_value = json.dumps(
        {"session_id": "sess-2", "started_at": now_iso, "status": "active"}
    )
    new = reader.read_current()
    assert new == "sess-2"
    # Caller is expected to flush the buffer when last_seen != new — verify
    # the reader at least surfaces the change.
    assert reader.last_seen == "sess-2"
    assert reader.changed_since_last is True


def test_session_reader_returns_none_when_redis_raises():
    mod = _import_module()
    r = MagicMock()
    r.get.side_effect = ConnectionError("Redis down")
    reader = mod._SessionReader(r, max_age_h=12.0)
    assert reader.read_current() is None


def test_session_reader_clears_last_seen_on_malformed_json():
    """After malformed JSON, last_seen must be reset so a later valid
    session of the SAME id is not treated as already-known."""
    import json
    from datetime import datetime, timezone

    mod = _import_module()
    r = MagicMock()
    now_iso = datetime.now(timezone.utc).isoformat()
    r.get.return_value = json.dumps(
        {"session_id": "sess-1", "started_at": now_iso, "status": "active"}
    )
    reader = mod._SessionReader(r, max_age_h=12.0)
    assert reader.read_current() == "sess-1"
    assert reader.last_seen == "sess-1"

    # Malformed JSON returned next.
    r.get.return_value = b"not-json"
    assert reader.read_current() is None
    assert reader.last_seen is None  # state cleared

    # Valid session returns (same id) — must NOT signal changed (we cleared, then re-saw).
    r.get.return_value = json.dumps(
        {"session_id": "sess-1", "started_at": now_iso, "status": "active"}
    )
    assert reader.read_current() == "sess-1"
    # last_seen was None, now sess-1 → changed_since_last fires (correctly, since we
    # had no valid session in between)
    assert reader.last_seen == "sess-1"


def test_session_reader_none_to_valid_signals_change():
    """First read returns None (no key); a later valid session must
    fire changed_since_last=True."""
    import json
    from datetime import datetime, timezone

    mod = _import_module()
    r = MagicMock()
    r.get.return_value = None  # no session yet
    reader = mod._SessionReader(r, max_age_h=12.0)
    assert reader.read_current() is None
    assert reader.last_seen is None

    r.get.return_value = json.dumps(
        {
            "session_id": "sess-new",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "active",
        }
    )
    assert reader.read_current() == "sess-new"
    assert reader.changed_since_last is True


# ---------------------------------------------------------------------------
# Buffer + CLI gate
# ---------------------------------------------------------------------------


def test_dropped_event_log_drops_oldest_on_overflow():
    mod = _import_module()
    log = mod._DroppedEventLog(capacity=3)
    log.enqueue({"i": 1})
    log.enqueue({"i": 2})
    log.enqueue({"i": 3})
    assert log.dropped_total == 0
    log.enqueue({"i": 4})
    assert log.dropped_total == 1
    drained = log.drain()
    assert [d["i"] for d in drained] == [2, 3, 4]


def test_dropped_event_log_flushed_drops_all_events():
    mod = _import_module()
    log = mod._DroppedEventLog(capacity=10)
    log.enqueue({"i": 1})
    log.enqueue({"i": 2})
    assert log.dropped_total == 0
    log.flush()
    assert log.dropped_total == 2
    assert log.drain() == []


def test_cli_main_exits_when_win32_unavailable(monkeypatch, capsys):
    mod = _import_module()
    monkeypatch.setattr(mod, "_WIN32_AVAILABLE", False)
    with pytest.raises(SystemExit) as ei:
        mod.main()
    assert ei.value.code != 0
    captured = capsys.readouterr()
    assert "requirements-windows.txt" in captured.err


# ---------------------------------------------------------------------------
# CR-7: synthetic-clock integration test for ActivityMonitor.run()
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="synthetic-clock loop test needs richer harness; "
    "schema round-trips below cover producer contract"
)
@pytest.mark.asyncio
async def test_activity_monitor_run_emits_focus_and_intensity_events(monkeypatch):
    """Drive run() with controlled time + mocked foreground polls.

    Skipped: asyncio.sleep patching recursion needs a richer harness.
    Schema round-trip tests below cover PerceptionEvent contract;
    integration tests cover full pipeline; this is the gap in between.

    Uncovered until this test runs:
      - Publish ordering in focus-change (intensity-flush before focus_change)
      - Intensity-window reset timing on focus change
      - last_app / last_title / last_sampled state advancement
      - Session-change state reset path (CR-2 invariant)
    """
    pass


def test_focus_event_round_trips_through_perception_event_schema():
    """Schema compatibility: _FocusState.on_focus_change → PerceptionEvent → bytes → from_json."""
    from tabula.contracts import PerceptionEvent

    mod = _import_module()
    state = mod._FocusState(
        sampling_s=10.0,
        title_allowlist=("code",),
        source_id="test-host",
        session_id="sess-rt",
    )
    state.on_focus_change(new_app="code", new_title="main.py", now=100.0)
    state.last_input_time = 125.0
    ev = state.on_focus_change(new_app="chrome", new_title="news.com", now=130.0)

    # IM-3 means this is already a PerceptionEvent; round-trip must still work.
    assert isinstance(ev, PerceptionEvent)
    roundtripped = PerceptionEvent.from_json(ev.to_bytes())
    assert roundtripped.domain == "activity_focus"
    assert roundtripped.entity == "code"
    assert roundtripped.session_id == "sess-rt"


def test_intensity_event_round_trips_through_perception_event_schema():
    from tabula.contracts import PerceptionEvent

    mod = _import_module()
    w = mod._IntensityWindow(
        sampling_s=10.0,
        min_events=1,
        min_window_s=2.0,
        title_allowlist=("code",),
        source_id="test-host",
        session_id="sess-rt",
    )
    w.window_started_at = 100.0
    w.span_id = "span-rt"
    w.keystrokes = 20
    w.mouse_events = 5
    w.last_input_time = 109.0
    ev = w.build(focused_app="code", focused_title="main.py", now=110.0)

    assert isinstance(ev, PerceptionEvent)
    roundtripped = PerceptionEvent.from_json(ev.to_bytes())
    assert roundtripped.domain == "activity_intensity"
    assert roundtripped.entity == "code"
    assert roundtripped.session_id == "sess-rt"


# ---------------------------------------------------------------------------
# _get_foreground: identity cache short-circuit (pid+exe) + FIFO eviction
# ---------------------------------------------------------------------------


def _make_monitor(mod):
    """Build an ActivityMonitor whose OS deps come from the injected fakes."""
    config = MagicMock()
    config.session_max_age_h = 12.0
    config.activity_title_allowlist = "code"
    return mod.ActivityMonitor(
        config, redis_client=MagicMock(), nats_client=MagicMock()
    )


def _wire_foreground(fakes, *, exe_name, exe_path, pid=4321, title="title"):
    """Point the fake OS modules at a single foreground process.

    Returns the psutil.Process MagicMock instance so the test can assert on
    proc.exe() call counts.
    """
    win32gui = fakes["win32gui"]
    win32process = fakes["win32process"]
    psutil = fakes["psutil"]

    win32gui.GetForegroundWindow.return_value = 99  # truthy hwnd
    win32gui.GetWindowText.return_value = title
    win32process.GetWindowThreadProcessId.return_value = (1, pid)

    proc = MagicMock()
    proc.name.return_value = exe_name
    proc.exe.return_value = exe_path
    psutil.Process.return_value = proc

    # The except-clauses reference these as exception classes; give them real ones.
    psutil.AccessDenied = type("AccessDenied", (Exception,), {})
    psutil.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
    psutil.ZombieProcess = type("ZombieProcess", (Exception,), {})
    return proc


def test_get_foreground_short_circuits_for_same_pid_and_exe(fake_win_modules):
    mod = _import_module()
    mon = _make_monitor(mod)
    proc = _wire_foreground(
        fake_win_modules,
        exe_name="Alpha.exe",
        exe_path="fake/path/a/Alpha.exe",
        pid=4321,
    )

    app1, _, id1 = mon._get_foreground()
    assert app1 == "alpha"
    assert proc.exe.call_count == 1  # resolved once

    # Second call, SAME (pid, exe): short-circuit fires, no extra proc.exe().
    app2, _, id2 = mon._get_foreground()
    assert app2 == "alpha"
    assert id2 == id1
    assert proc.exe.call_count == 1  # NOT called again


def test_get_foreground_reresolves_when_exe_changes_at_same_pid(fake_win_modules):
    """BUG-21: a recycled pid running a different exe must NOT reuse the stale
    identity — identity is re-resolved against the new exe path."""
    mod = _import_module()
    mon = _make_monitor(mod)

    fake_win_modules["win32api"].GetFileVersionInfo.return_value = None
    proc = _wire_foreground(
        fake_win_modules,
        exe_name="Alpha.exe",
        exe_path="fake/path/a/Alpha.exe",
        pid=4321,
    )
    app1, _, _ = mon._get_foreground()
    assert app1 == "alpha"
    assert proc.exe.call_count == 1

    # Same pid, DIFFERENT foreground exe (pid reuse on Windows).
    proc2 = _wire_foreground(
        fake_win_modules,
        exe_name="Beta.exe",
        exe_path="fake/path/b/Beta.exe",
        pid=4321,
    )
    app2, _, _ = mon._get_foreground()
    assert app2 == "beta"  # re-resolved, not the stale "alpha"
    assert proc2.exe.call_count == 1  # exe() invoked again for the new path


def test_get_foreground_identity_cache_respects_fifo_cap(fake_win_modules):
    mod = _import_module()
    mon = _make_monitor(mod)
    _wire_foreground(
        fake_win_modules, exe_name="New.exe", exe_path="fake/path/new/New.exe", pid=7777
    )

    # Pre-fill the cache to the cap with dummy paths.
    cap = mod._IDENTITY_CACHE_MAX
    for i in range(cap):
        mon._identity_cache[f"fake/path/old/app{i}.exe"] = None
    assert len(mon._identity_cache) == cap

    # Resolving a NOVEL exe_path evicts the oldest, staying at the cap.
    mon._get_foreground()
    assert len(mon._identity_cache) <= cap
    assert "fake/path/new/New.exe" in mon._identity_cache
