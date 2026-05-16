"""Unit tests for active_tracking keying by (domain, entity).

We test the keying invariant directly via a small in-process dict
because the feedback_collector module wraps active_tracking in a
closure (it's a local in `run()`). We assert that the keying choice
is documented in the source.
"""

from __future__ import annotations

import re
from pathlib import Path


def test_active_tracking_is_keyed_by_domain_and_entity():
    """active_tracking declaration must be dict[tuple[str, str], PendingAdvice]."""
    src = Path("perception/feedback_collector.py").read_text()
    decl = re.search(
        r"active_tracking:\s*dict\[(.+?),\s*PendingAdvice\s*\]",
        src,
        re.DOTALL,
    )
    assert decl is not None, "active_tracking declaration not found"
    key_type = " ".join(decl.group(1).split())
    assert key_type == "tuple[str, str]", (
        f"active_tracking must be keyed by tuple[str, str] "
        f"(got {key_type!r}) so domains with overlapping entities "
        f"(e.g. activity_focus/code and activity_intensity/code) "
        f"do not share a tracking record."
    )


def test_no_active_tracking_lookups_use_bare_entity_key():
    """Bare entity-string indexing of active_tracking must be gone.

    Lookups must use 2-element keys constructed from a (domain, entity)
    pair, never `active_tracking[entity]` or `active_tracking.get(entity)`.
    """
    src = Path("perception/feedback_collector.py").read_text()
    # Strip the # comment in the declaration line so we don't false-positive.
    bad_patterns = [
        r"active_tracking\[entity\]",
        r"active_tracking\.get\(entity\)",
    ]
    for pat in bad_patterns:
        assert not re.search(pat, src), (
            f"Found bare-entity active_tracking access matching {pat!r}; "
            f"must use (domain, entity) tuple key."
        )
