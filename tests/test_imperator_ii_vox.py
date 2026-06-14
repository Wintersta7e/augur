from vox.console_display import render_proposal


def test_render_proposal():
    out = render_proposal(
        {
            "klass": "safe",
            "kind": "escalation_rule",
            "target": "LOW+LOW",
            "status": "logged",
            "rationale": "improves precision",
        }
    )
    assert "escalation_rule" in out and "LOW+LOW" in out and "improves precision" in out
