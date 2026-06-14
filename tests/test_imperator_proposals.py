import pytest

from imperator import proposals as P


class _Cfg:
    min_prompt_len = 20
    prompt_forbidden_patterns = ("as an ai", "take a break")


def test_dedupe_key_stable_across_action_changes():
    a = P.make_proposal(
        kind="escalation_rule",
        target="LOW+LOW",
        action={"target": "MEDIUM"},
        rationale="r",
    )
    b = P.make_proposal(
        kind="escalation_rule",
        target="LOW+LOW",
        action={"target": "LOW"},
        rationale="r",
    )
    assert a["dedupe_key"] == b["dedupe_key"] and a["proposal_id"] != b["proposal_id"]


def test_normalize_klass_overrides_llm_claim():
    p = P.make_proposal(
        kind="code", target="x.py", action={}, rationale="r", klass="safe"
    )
    P.normalize_klass(p)
    assert p["klass"] == "gated"
    p2 = P.make_proposal(
        kind="sigma", target="typing", action={}, rationale="r", klass="safe"
    )
    P.normalize_klass(p2)
    assert p2["klass"] == "safe"


def test_gate_forces_gated_to_logged():
    p = P.normalize_klass(
        P.make_proposal(kind="code", target="x.py", action={}, rationale="r")
    )
    out = P.gate(p, cfg=_Cfg(), recent_self_tuning={}, applied_keys=set())
    assert out["klass"] == "gated" and out["status"] == "logged"


def test_gate_skips_already_applied_only():
    p = P.normalize_klass(
        P.make_proposal(
            kind="escalation_rule",
            target="LOW+LOW",
            action={"target": "MEDIUM"},
            rationale="r",
        )
    )
    assert (
        P.gate(
            dict(p), cfg=_Cfg(), recent_self_tuning={}, applied_keys={p["dedupe_key"]}
        )["status"]
        == "skipped"
    )
    assert (
        P.gate(dict(p), cfg=_Cfg(), recent_self_tuning={}, applied_keys=set())["status"]
        != "skipped"
    )


def test_gate_rejects_unsafe_prompt():
    p = P.normalize_klass(
        P.make_proposal(
            kind="prompt_strategy",
            target="typing",
            action={"text": "short"},
            rationale="r",
        )
    )
    assert (
        P.gate(p, cfg=_Cfg(), recent_self_tuning={}, applied_keys=set())["status"]
        == "logged"
    )


def test_matches_recent_defers_rule_when_matrix_mutated():
    p = P.normalize_klass(
        P.make_proposal(
            kind="escalation_rule",
            target="LOW+LOW",
            action={"target": "MEDIUM"},
            rationale="r",
        )
    )
    moved = {"matrix_mutated": True, "windows_tuned": False, "sigma_values": {}}
    assert (
        P.gate(dict(p), cfg=_Cfg(), recent_self_tuning=moved, applied_keys=set())[
            "status"
        ]
        == "skipped"
    )
    still = {"matrix_mutated": False, "windows_tuned": False, "sigma_values": {}}
    assert (
        P.gate(dict(p), cfg=_Cfg(), recent_self_tuning=still, applied_keys=set())[
            "status"
        ]
        != "skipped"
    )


def test_matches_recent_no_substring_collision_on_domain():
    # The old substring scan skipped domain "type" merely because "type" is a
    # substring of "typing" in the recent blob; the structured check keys off
    # the exact domain, so "type" is not deferred.
    p = P.normalize_klass(
        P.make_proposal(
            kind="prompt_strategy",
            target="type",
            action={"domain": "type", "text": "x" * 40},
            rationale="r",
        )
    )
    recent = {"prompt_mutated": False, "sigma_values": {"typing": 3.0}}
    assert (
        P.gate(dict(p), cfg=_Cfg(), recent_self_tuning=recent, applied_keys=set())[
            "status"
        ]
        != "skipped"
    )


def test_matches_recent_defers_prompt_when_domain_sigma_moved():
    p = P.normalize_klass(
        P.make_proposal(
            kind="prompt_strategy",
            target="typing",
            action={"domain": "typing", "text": "x" * 40},
            rationale="r",
        )
    )
    recent = {"prompt_mutated": False, "sigma_values": {"typing": 3.0}}
    assert (
        P.gate(dict(p), cfg=_Cfg(), recent_self_tuning=recent, applied_keys=set())[
            "status"
        ]
        == "skipped"
    )


@pytest.mark.parametrize(
    "kind", sorted(P._KIND_KLASS) + ["totally_unknown", "", "ESCALATION_RULE"]
)
def test_invariant_only_safe_kinds_are_auto_applicable(kind):
    # Privilege cannot be self-escalated: after the deterministic normalize_klass,
    # the ONLY auto-applicable kinds are the safe-auto set, and code/structural or
    # any unknown kind is always gated (never auto-applied) — for every kind.
    p = P.normalize_klass(
        P.make_proposal(
            kind=kind, target="LOW+LOW", action={"target": "MEDIUM"}, rationale="r"
        )
    )
    p["status"] = "logged"
    if P.is_auto_applicable(p):
        assert kind in P._AUTO_APPLY_KINDS
    if kind in ("code", "structural") or kind not in P._KIND_KLASS:
        assert P.is_auto_applicable(p) is False
