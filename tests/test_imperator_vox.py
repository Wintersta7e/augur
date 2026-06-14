from vox.console_display import render_auspices, render_self_model


def test_render_auspices_warming_up():
    assert "warming up" in render_auspices({"status": "warming_up"}).lower()


def test_render_auspices_shows_salience():
    out = render_auspices(
        {
            "schema_version": 1,
            "salience": {"value": 0.42, "fresh": True},
            "activity": {"value": "ide", "fresh": True},
        }
    )
    assert "0.42" in out and "ide" in out


def test_render_self_model_lists_blind_spots():
    out = render_self_model(
        {
            "competence": {"value": 0.6, "fresh": True},
            "blind_spots": {
                "value": [{"kind": "low_confidence_rule", "detail": "r"}],
                "fresh": True,
            },
        }
    )
    assert "low_confidence_rule" in out and "0.6" in out
