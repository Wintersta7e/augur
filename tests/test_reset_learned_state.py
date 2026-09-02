"""Classifier tests for scripts/reset_learned_state.py.

The two branches that must never be wrong — a mistake destroys real data:
  * a user-taught (semantic) memory must be KEPT;
  * a real perception-baseline entity must be KEPT.
Everything contaminated is deleted; anything unrecognized is kept and flagged.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.reset_learned_state import classify_key  # noqa: E402


def _no_json(_key: str) -> object:
    """get_json stub for keys whose content is not consulted."""
    return None


class TestVigilProfiles:
    def test_real_app_baseline_is_kept(self) -> None:
        d = classify_key(
            "augur:vigil:profile:activity_focus:focus_change:some_app", _no_json
        )
        assert d.action == "keep"

    def test_real_typing_baseline_is_kept(self) -> None:
        assert (
            classify_key("augur:vigil:profile:typing:sample:user", _no_json).action
            == "keep"
        )

    def test_entity_containing_a_colon_is_kept(self) -> None:
        """The entity is the key's tail, so a ':' in an app name survives."""
        assert (
            classify_key(
                "augur:vigil:profile:activity_focus:focus_change:host:app", _no_json
            ).action
            == "keep"
        )

    def test_legacy_pre_series_baseline_is_deleted(self) -> None:
        """One EWMA over every event_type of an entity may straddle units."""
        d = classify_key("augur:vigil:profile:typing:user", _no_json)
        assert d.action == "delete"
        assert "legacy" in d.reason

    def test_synthetic_suffix_entity_is_deleted(self) -> None:
        assert (
            classify_key(
                "augur:vigil:profile:typing:pause:user_a2b60936", _no_json
            ).action
            == "delete"
        )
        assert (
            classify_key(
                "augur:vigil:profile:activity_focus:focus_change:code_a2b60936",
                _no_json,
            ).action
            == "delete"
        )

    def test_chess_domain_entity_is_deleted(self) -> None:
        assert (
            classify_key("augur:vigil:profile:chess:move:white", _no_json).action
            == "delete"
        )


class TestMemoria:
    def test_user_taught_semantic_memory_is_protected(self) -> None:
        d = classify_key(
            "augur:memoria:dsr:abc", lambda _k: {"memory_kind": "semantic"}
        )
        assert d.action == "keep"
        assert "PROTECTED" in d.reason

    def test_episodic_memory_is_deleted(self) -> None:
        d = classify_key(
            "augur:memoria:dsr:abc",
            lambda _k: {
                "memory_kind": "episodic",
                "source_sessions": ["session-1-window"],
            },
        )
        assert d.action == "delete"

    def test_unreadable_memory_is_deleted_not_kept(self) -> None:
        # A corrupt/missing memoria record is not a semantic memory, so it is
        # not protected — it goes. (Fail toward cleanup, not toward keeping junk.)
        assert classify_key("augur:memoria:dsr:abc", _no_json).action == "delete"


class TestLearnedPolicyAndSessions:
    def test_tuned_threshold_is_deleted(self) -> None:
        assert (
            classify_key("augur:vigil:thresholds:typing", _no_json).action == "delete"
        )

    def test_nexus_matrix_is_deleted(self) -> None:
        assert classify_key("augur:nexus:matrix", _no_json).action == "delete"

    def test_limen_adaptive_is_deleted(self) -> None:
        assert classify_key("augur:limen:credibility", _no_json).action == "delete"

    def test_session_scoped_records_are_deleted(self) -> None:
        for k in (
            "augur:disciplina:some-session",
            "augur:responsum:some-session",
            "augur:tuning_applied:gate:some-session",
            "augur:praesagium:episodes:some-session",
        ):
            assert classify_key(k, _no_json).action == "delete", k


class TestKeepAndUnknown:
    def test_session_count_is_kept(self) -> None:
        assert classify_key("augur:session:count", _no_json).action == "keep"

    def test_health_is_kept(self) -> None:
        assert classify_key("augur:praefectus:health", _no_json).action == "keep"

    def test_unrecognized_key_is_kept_and_flagged(self) -> None:
        d = classify_key("augur:some_new_faculty:state", _no_json)
        assert d.action == "keep"
        assert d.reason.startswith("UNRECOGNIZED")
