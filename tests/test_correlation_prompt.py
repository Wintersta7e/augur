"""Tests for advisor correlation-path functions.

describe_signal, build_correlation_prompt, and the routing change on
payload.correlation_found.
"""

from __future__ import annotations



from reasoning.augur_advisor import describe_signal


def _chess_event() -> dict:
    return {
        "domain": "chess",
        "entity": "white",
        "event_type": "move",
        "value": 47.2,
        "unit": "seconds",
        "context": {"move_san": "Nf3", "move_number": 12},
        "baseline_mean": 8.2,
        "baseline_std": 2.1,
        "deviation_score": 5.7,
        "severity": "low",
    }


def _typing_event() -> dict:
    return {
        "domain": "typing",
        "entity": "user",
        "event_type": "pause",
        "value": 18.0,
        "unit": "seconds",
        "context": {"wpm_before": 65, "wpm_after": 39, "avg_wpm": 52},
        "baseline_mean": 3.5,
        "baseline_std": 1.2,
        "deviation_score": 4.1,
        "severity": "low",
    }


class TestDescribeSignal:
    def test_chess_mentions_move_thinktime_baseline_deviation(self) -> None:
        line = describe_signal("chess", _chess_event())
        assert "Nf3" in line
        assert "47" in line  # think time
        assert "8.2" in line  # baseline
        assert "5.7" in line  # deviation

    def test_chess_one_line(self) -> None:
        line = describe_signal("chess", _chess_event())
        assert "\n" not in line

    def test_typing_mentions_pause_and_wpm(self) -> None:
        line = describe_signal("typing", _typing_event())
        assert "18" in line  # pause seconds
        assert "52" in line or "wpm" in line.lower()

    def test_typing_one_line(self) -> None:
        line = describe_signal("typing", _typing_event())
        assert "\n" not in line

    def test_unknown_domain_fallback(self) -> None:
        event = {
            "domain": "focus",
            "entity": "app",
            "event_type": "switch",
            "value": 5,
            "unit": "count",
            "context": {},
            "baseline_mean": 1,
            "deviation_score": 3.0,
            "severity": "low",
        }
        line = describe_signal("focus", event)
        assert "focus" in line.lower()
        assert "\n" not in line
