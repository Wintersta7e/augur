"""Vox renders augur.praefectus.health transitions."""

from vox import console_display as V


def test_subject_constant():
    assert V.SUBJECT_HEALTH == "augur.praefectus.health"


def test_render_health_degraded():
    line = V.render_health(
        {
            "faculty": "consilium",
            "reason": "consilium_stall",
            "transition": "degraded",
            "overall": "degraded",
            "ts": 1.0,
        }
    )
    assert "consilium" in line and "consilium_stall" in line


def test_render_health_dead():
    line = V.render_health(
        {
            "faculty": "vox",
            "reason": "never_started",
            "transition": "dead",
            "overall": "dead",
            "ts": 1.0,
        }
    )
    assert "vox" in line and "never_started" in line


def test_render_health_recovered():
    line = V.render_health(
        {
            "faculty": "vox",
            "reason": "lost",
            "transition": "recovered",
            "overall": "alive",
            "ts": 1.0,
        }
    )
    assert "vox" in line and "recover" in line.lower()
