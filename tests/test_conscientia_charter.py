"""Charter — principles, protected surfaces, pattern assembly."""

from dataclasses import FrozenInstanceError

import pytest

from conscientia import charter
from tabula.config import AugurConfig


def test_principles_shape_and_order():
    pids = [p.pid for p in charter.PRINCIPLES]
    assert pids == [
        "pietas",
        "restraint",
        "reversibility",
        "transparency",
        "containment",
    ]
    assert all(p.title and p.text for p in charter.PRINCIPLES)


def test_principles_immutable():
    with pytest.raises(FrozenInstanceError):
        charter.PRINCIPLES[0].text = "changed"  # type: ignore[misc]


def test_protected_surfaces_cover_the_guardian_itself():
    for needle in (
        "conscientia/",
        "limen/",
        "imperator/apply.py",
        "imperator/proposals.py",
        "nexus/matrix_ops.py",
        "augur:conscientia:",
    ):
        assert needle in charter.PROTECTED_SURFACES


def test_pattern_assembly_extends_config():
    cfg = AugurConfig(
        conscientia_output_extra_patterns=("extra one",),
        conscientia_teach_extra_patterns=("extra two",),
    )
    out = charter.output_patterns(cfg)
    teach = charter.teach_patterns(cfg)
    assert set(cfg.prompt_forbidden_patterns) <= set(out)
    assert "extra one" in out and "extra one" not in teach
    assert "extra two" in teach and "extra two" not in out


def test_render_charter_is_pure_data():
    doc = charter.render_charter()
    assert doc["version"] == charter.CHARTER_VERSION
    assert [p["pid"] for p in doc["principles"]] == [p.pid for p in charter.PRINCIPLES]
    assert doc["protected_surfaces"] == list(charter.PROTECTED_SURFACES)


def test_pattern_assembly_tolerates_partial_cfg():
    class _Partial:
        prompt_forbidden_patterns = ("take a break",)

    assert charter.output_patterns(_Partial()) == ("take a break",)
    assert charter.teach_patterns(object()) == ()
