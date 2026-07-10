"""Gated-proposal rubric — recommendation matrix."""

from conscientia.review import review_gated
from tabula.config import AugurConfig

CFG = AugurConfig()


def _gated(kind="code", target="vigil/anomaly_detector.py", action=None, klass="gated"):
    return {
        "proposal_id": "p1",
        "dedupe_key": "dk1",
        "kind": kind,
        "target": target,
        "klass": klass,
        "ts": 1.0,
        "action": action if action is not None else {"patch": "small"},
    }


def test_normal_code_proposal_needs_human():
    rec = review_gated(_gated(), CFG)
    assert rec["recommendation"] == "needs_human"
    by = {c["check"]: c["ok"] for c in rec["checks"]}
    assert by["klass_gated"] and not by["reversible"] and not by["tests_verifiable"]
    assert by["protected_surface"] and by["bounded"]
    assert rec["proposal_id"] == "p1" and rec["reviewed_at"] > 0


def test_protected_surface_rejected():
    rec = review_gated(_gated(target="limen/gate.py"), CFG)
    assert rec["recommendation"] == "reject"


def test_non_gated_record_rejected():
    rec = review_gated(_gated(klass="safe"), CFG)
    assert rec["recommendation"] == "reject"
    assert not {c["check"]: c["ok"] for c in rec["checks"]}["klass_gated"]


def test_unbounded_action_fails_bounded_but_still_needs_human():
    rec = review_gated(_gated(action={"patch": "x" * 5000}), CFG)
    by = {c["check"]: c["ok"] for c in rec["checks"]}
    assert not by["bounded"]
    assert rec["recommendation"] == "needs_human"  # bounded is advisory, not a rejector


def test_nested_payload_over_2kib_fails_bounded():
    # Spec R4's second clause: total ≤ 4 KiB but a single embedded code blob
    # > 2 KiB is exactly the smuggled-big-patch shape the clause exists for.
    action = {"files": [{"path": "a.py", "patch": "x" * 3000}], "note": "small"}
    rec = review_gated(_gated(action=action), CFG)
    by = {c["check"]: c["ok"] for c in rec["checks"]}
    assert not by["bounded"]
    assert rec["recommendation"] == "needs_human"
    note = next(c["note"] for c in rec["checks"] if c["check"] == "bounded")
    assert "2" in note and "KiB" in note


def test_nested_payloads_under_2kib_pass_bounded():
    action = {"files": [{"path": "a.py", "patch": "x" * 1000}], "note": "y" * 1000}
    rec = review_gated(_gated(action=action), CFG)
    assert {c["check"]: c["ok"] for c in rec["checks"]}["bounded"]


def test_oversized_nested_dict_key_fails_bounded():
    # Keys are payload capacity too — a blob can hide there.
    rec = review_gated(_gated(action={"z" * 3000: "v"}), CFG)
    assert not {c["check"]: c["ok"] for c in rec["checks"]}["bounded"]


def test_never_approve():
    # exhaustively: no input yields anything but reject/needs_human
    for kind in ("code", "structural"):
        for target in ("vigil/x.py", "conscientia/charter.py"):
            rec = review_gated(_gated(kind=kind, target=target), CFG)
            assert rec["recommendation"] in ("reject", "needs_human")
