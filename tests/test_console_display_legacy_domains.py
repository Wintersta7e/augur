"""Regression tests: render_anomaly_line + render_advice for chess/typing.

T10 of activity-perception introduced a domain-dispatch in
output/console_display.py but accidentally keyed it on file names
(chess_board, typing_monitor) instead of the actual domain values
(chess, typing). These tests lock the correct keys in place.
"""

from __future__ import annotations

from output.console_display import render_advice, render_anomaly_line


def test_render_anomaly_line_chess_uses_player_move_format():
    data = {
        "domain": "chess",
        "player": "white",
        "move": "Nf3",
        "think_time": 12.4,
        "deviation_score": 2.1,
        "severity": "MEDIUM",
        "timestamp": "2026-05-16T12:00:00+00:00",
    }
    line = render_anomaly_line(data)
    # The chess branch (not the generic fallback) must render the move
    assert "white" in line
    assert "Nf3" in line
    assert "think=" in line  # chess-specific field


def test_render_anomaly_line_typing_uses_entity_wpm_format():
    data = {
        "domain": "typing",
        "entity": "user",
        "wpm": 42,
        "deviation_score": 1.3,
        "severity": "LOW",
        "timestamp": "2026-05-16T12:00:00+00:00",
    }
    line = render_anomaly_line(data)
    assert "user" in line
    assert "wpm=" in line  # typing-specific field
    assert "42" in line


def test_render_advice_chess_uses_player_move_format():
    data = {
        "domain": "chess",
        "player": "black",
        "move": "e5",
        "severity": "HIGH",
        "think_time": 38.0,
        "advice": "Consider taking time here.",
        "model": "qwen2.5:32b",
        "latency_ms": 1234,
    }
    out = render_advice(data)
    # Chess advice branch uses player + move + think time
    assert "BLACK" in out or "black" in out.lower()
    assert "e5" in out
    assert "Think time" in out or "think_time" in out
    assert "Consider taking" in out
