"""OS identity attribution: focus uses previous app, intensity uses focused app."""

from sensus.activity_monitor import (
    _FocusState,
    _IntensityWindow,
    _resolve_file_description,
)


def _focus_state():
    return _FocusState(
        sampling_s=10.0, title_allowlist=(), source_id="test-host", session_id="sess-1"
    )


def _intensity_window():
    return _IntensityWindow(
        sampling_s=10.0,
        min_events=1,
        min_window_s=2.0,
        title_allowlist=(),
        source_id="test-host",
        session_id="sess-1",
    )


def test_resolve_file_description_none_without_win32():
    # On non-Windows CI, _WIN32_AVAILABLE is False -> always None.
    assert _resolve_file_description(None) is None
    assert _resolve_file_description("C:/Apps/alpha.exe") is None


def test_focus_event_carries_previous_app_identity():
    fs = _focus_state()
    assert (
        fs.on_focus_change("alpha_app", "t1", now=100.0, new_identity="Alpha Browser")
        is None
    )
    ev = fs.on_focus_change("beta_app", "t2", now=160.0, new_identity="Beta Editor")
    assert ev is not None
    assert ev.entity == "alpha_app"
    assert ev.context["app_identity"] == "Alpha Browser"


def test_focus_event_omits_identity_when_none():
    fs = _focus_state()
    fs.on_focus_change("alpha_app", "t1", now=100.0, new_identity=None)
    ev = fs.on_focus_change("beta_app", "t2", now=160.0, new_identity="Beta Editor")
    assert "app_identity" not in ev.context


def test_intensity_event_carries_focused_identity():
    iw = _intensity_window()
    iw.reset(new_started_at=0.0, new_span_id="span-1")
    iw.keystrokes = 10
    ev = iw.build(
        focused_app="alpha_app",
        focused_title="t",
        now=10.0,
        focused_identity="Alpha Browser",
    )
    assert ev is not None
    assert ev.entity == "alpha_app"
    assert ev.context["app_identity"] == "Alpha Browser"


def test_intensity_event_omits_identity_when_none():
    iw = _intensity_window()
    iw.reset(new_started_at=0.0, new_span_id="span-1")
    iw.keystrokes = 10
    ev = iw.build(
        focused_app="alpha_app", focused_title="t", now=10.0, focused_identity=None
    )
    assert "app_identity" not in ev.context
