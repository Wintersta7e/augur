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


def test_matches_recent_prompt_mutation_does_not_defer_unrelated_domain():
    # The over-defer bug: a prompt was mutated this session (for the reflection's
    # own domain — always present in sigma_values), but a prompt_strategy proposal
    # targeting an UNRELATED domain the reflection never touched must NOT defer.
    # The mutated domain is the one that moved; an off-target domain has no fresh
    # change to thrash, so deferring it just costs a wasted cycle.
    p = P.normalize_klass(
        P.make_proposal(
            kind="prompt_strategy",
            target="chess",
            action={"domain": "chess", "text": "x" * 40},
            rationale="r",
        )
    )
    recent = {"prompt_mutated": True, "sigma_values": {"typing": 3.0}}
    assert (
        P.gate(dict(p), cfg=_Cfg(), recent_self_tuning=recent, applied_keys=set())[
            "status"
        ]
        != "skipped"
    )
    # The session-wide prompt_mutated flag must not change an unrelated domain's
    # fate: a `chess` proposal sees the same (non-skipped) outcome flag-on or
    # flag-off, because the fix scopes the defer to the proposal's own domain.
    flag_off = {"prompt_mutated": False, "sigma_values": {"typing": 3.0}}
    assert (
        P.gate(dict(p), cfg=_Cfg(), recent_self_tuning=flag_off, applied_keys=set())[
            "status"
        ]
        != "skipped"
    )


def test_matches_recent_prompt_mutation_defers_its_own_domain():
    # The flip side: when a prompt was mutated, the proposal that targets the
    # mutated domain (which is the reflection's domain, so its sigma after-state
    # is in sigma_values) DOES defer a cycle to avoid thrashing the fresh prompt.
    p = P.normalize_klass(
        P.make_proposal(
            kind="prompt_strategy",
            target="typing",
            action={"domain": "typing", "text": "x" * 40},
            rationale="r",
        )
    )
    recent = {"prompt_mutated": True, "sigma_values": {"typing": 3.0}}
    assert (
        P.gate(dict(p), cfg=_Cfg(), recent_self_tuning=recent, applied_keys=set())[
            "status"
        ]
        == "skipped"
    )


def test_matches_recent_defers_window_rule_when_windows_tuned():
    # An escalation_rule proposal carrying a `window` action targets the window
    # surface: it defers on `windows_tuned`, and is unaffected by `matrix_mutated`.
    p = P.normalize_klass(
        P.make_proposal(
            kind="escalation_rule",
            target="LOW+LOW",
            action={"target": "MEDIUM", "window": 45},
            rationale="r",
        )
    )
    tuned = {"matrix_mutated": False, "windows_tuned": True, "sigma_values": {}}
    assert (
        P.gate(dict(p), cfg=_Cfg(), recent_self_tuning=tuned, applied_keys=set())[
            "status"
        ]
        == "skipped"
    )
    # Matrix moved but windows did not — a window-action rule does NOT defer.
    matrix_only = {"matrix_mutated": True, "windows_tuned": False, "sigma_values": {}}
    assert (
        P.gate(dict(p), cfg=_Cfg(), recent_self_tuning=matrix_only, applied_keys=set())[
            "status"
        ]
        != "skipped"
    )


def test_matches_recent_window_rule_not_deferred_when_windows_untuned():
    # Windows untuned this session → a window-action escalation_rule is free to land.
    p = P.normalize_klass(
        P.make_proposal(
            kind="escalation_rule",
            target="LOW+LOW",
            action={"target": "MEDIUM", "window": 45},
            rationale="r",
        )
    )
    still = {"matrix_mutated": True, "windows_tuned": False, "sigma_values": {}}
    assert (
        P.gate(dict(p), cfg=_Cfg(), recent_self_tuning=still, applied_keys=set())[
            "status"
        ]
        != "skipped"
    )


@pytest.mark.parametrize("kind", ["sigma", "gate_calibration", "observe_more"])
def test_matches_recent_fallthrough_safe_non_auto_kinds(kind):
    # The safe-but-non-auto kinds are never auto-applied, so _matches_recent has no
    # anti-thrash branch for them: it falls through to False regardless of how much
    # Disciplina tuned this session. (gate() then leaves them `logged`, not skipped.)
    p = P.normalize_klass(
        P.make_proposal(
            kind=kind, target="typing", action={"domain": "typing"}, rationale="r"
        )
    )
    everything_moved = {
        "matrix_mutated": True,
        "windows_tuned": True,
        "prompt_mutated": True,
        "sigma_values": {"typing": 3.0},
    }
    assert P._matches_recent(p, everything_moved) is False
    # gate() routes a safe-non-auto kind to `logged` (not `skipped`).
    assert (
        P.gate(
            dict(p),
            cfg=_Cfg(),
            recent_self_tuning=everything_moved,
            applied_keys=set(),
        )["status"]
        == "logged"
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
