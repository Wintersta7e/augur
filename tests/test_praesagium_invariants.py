"""Praesagium invariants PR1-PR8 (spec 2026-07-09 sec 11).

This file is the single cross-cutting audit surface for the anticipation
faculty: each PR section pins one spec-sec-11 invariant with its verbatim
quote in the section docstring, over parametrized matrices and hand-rolled
generators (the Conscientia precedent: tests/test_conscientia_invariants.py --
no hypothesis). Per-task test files cover behaviors; THIS file pins the
properties that must hold no matter which task last touched the code.

Sources exercised (all shipped, Tasks 1-13):
  * praesagium/matcher.py  -- build_foreseen_payload, make_on_anomaly,
    match_patterns, resolve_open_predictions, render_forewarning.
  * praesagium/miner.py    -- run_praesagium_mining (expiry sweep).
  * praesagium/patterns.py -- mine_corpus, merge_blob, count_matches.
  * consilium/advisor.py   -- _clamp_foreseen (the PR1b boundary clamp).
  * limen/gate.py          -- build_signature (the exempt/path/state_key truth).
  * tabula/persistence.py  -- the bounded PM stores + atomic resolve op.
"""

from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from pathlib import Path

import fakeredis
import pytest

from consilium.advisor import _clamp_foreseen
from limen.gate import build_signature
from praesagium.matcher import (
    SUBJECT_FORESEEN,
    build_foreseen_payload,
    make_on_anomaly,
    render_forewarning,
)
from praesagium.miner import run_praesagium_mining
from praesagium.patterns import count_matches, merge_blob, mine_corpus, pattern_id
from tabula.config import AugurConfig
from tabula.persistence import PersistenceManager

REPO_ROOT = Path(__file__).resolve().parent.parent
MATCHER_SRC = (REPO_ROOT / "praesagium" / "matcher.py").read_text(encoding="utf-8")


# ===========================================================================
# Shared helpers (self-contained, mirroring tests/test_praesagium_matcher.py)
# ===========================================================================


def _pm() -> PersistenceManager:
    return PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))


def _cfg(**over) -> AugurConfig:
    return AugurConfig(**over)


def _ep(k: str, s: str, t: float) -> dict:
    return {"k": k, "s": s, "t": float(t)}


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


class _Msg:
    """Minimal NATS-message stand-in: only ``.data`` (bytes) is read."""

    def __init__(self, data: bytes) -> None:
        self.data = data


def _anomaly(
    *,
    domain: str = "typing",
    entity: str = "user",
    severity: str = "medium",
    ts: float = 1000.0,
    session_id: str | None = "s1",
) -> dict:
    payload: dict = {
        "domain": domain,
        "entity": entity,
        "severity": severity,
        "timestamp": _iso(ts),
    }
    if session_id is not None:
        payload["session_id"] = session_id
    return payload


class RecordingPublish:
    """Async publish stub: records every (subject, data); can raise per-subject."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[tuple[str, bytes]] = []
        self.fail_on = fail_on

    async def __call__(self, subject: str, data: bytes) -> None:
        self.calls.append((subject, data))
        if self.fail_on in (subject, "all"):
            raise RuntimeError("nats down")

    def subjects(self) -> list[str]:
        return [s for s, _ in self.calls]


def _pattern(
    *,
    pid: str = "abc123def456",
    antecedent: str = "typing:user",
    consequent: str = "activity:editor",
    status: str = "active",
    window_s: float = 120.0,
    support_sessions: int = 4,
    conf_lower: float = 0.62,
    lift: float = 2.13,
) -> dict:
    return {
        "pattern_id": pid,
        "antecedent": antecedent,
        "consequent": consequent,
        "window_s": window_s,
        "support_sessions": support_sessions,
        "n": 5,
        "k": 4,
        "conf": 0.8,
        "conf_lower": conf_lower,
        "lift": lift,
        "lag_median_s": 40.0,
        "lag_p90_s": 90.0,
        "status": status,
        "hit_rate": None,
        "resolutions": 0,
        "created_at": 1.0,
        "mined_at": 2.0,
        "retired_at": None,
        "retired_reason": None,
        "repass_streak": 1,
    }


def _seed_patterns(pm: PersistenceManager, *patterns: dict) -> None:
    pm.save_praesagium_patterns(
        {
            "version": 1,
            "mined_at": 100.0,
            "hit_rate_watermark": 0.0,
            "patterns": {p["pattern_id"]: p for p in patterns},
        }
    )


def _prae_keys(pm: PersistenceManager) -> list:
    return pm._r.keys("augur:praesagium:*")


# ===========================================================================
# PR1 -- never exempt, enforced at the boundary
# ===========================================================================
#
# Spec sec 11, PR1: "never exempt, enforced at the boundary. (a) By
# construction: every matcher-built foreseen payload has
# correlation_found=False, severity medium => build_signature(payload).exempt
# is False, path == "single". (b) By clamping: for ARBITRARY payloads arriving
# on augur.praesagium.foreseen (including forged correlation_found=True +
# "HIGH"), Consilium's on_foreseen validate-and-clamp guarantees the payload
# that reaches process_message cannot be exempt or high. Both pinned over
# generated matrices. The single most important constraint in the faculty."


# -- PR1a: by construction ----------------------------------------------------

_PR1A_PATTERNS = [
    {
        "pattern_id": pid,
        "antecedent": a,
        "consequent": b,
        "window_s": w,
        "support_sessions": s,
        "conf_lower": cl,
        "lift": lift,
    }
    for pid, a, b in (
        ("abc123def456", "typing:user", "activity:editor"),
        ("deadbeef0001", "activity:browser", "typing:user"),
        ("0f0f0f0f0f0f", "chess:white", "chess:black"),
        ("ffffffffffff", "activity:app:vscode", "activity:meeting"),
    )
    for w in (30.0, 125.0, 900.0)
    for s, cl, lift in ((3, 0.40, 1.5), (10, 0.87, 12.3), (4, 0.62, 2.13))
]


@pytest.mark.parametrize("session_id", ["sess-1", None])
@pytest.mark.parametrize("pattern", _PR1A_PATTERNS)
def test_pr1a_generated_foreseen_never_exempt(pattern, session_id):
    prediction = {
        "prediction_id": "pred-" + pattern["pattern_id"],
        "forewarning_text": render_forewarning(pattern),
    }
    payload = build_foreseen_payload(pattern, prediction, session_id)

    # The payload envelope itself is never-exempt-shaped.
    assert payload["correlation_found"] is False
    assert payload["correlated_events"] == []
    assert payload["combined_severity"] == "MEDIUM"
    assert payload["primary_anomaly"]["severity"] == "medium"

    sig = build_signature(payload)
    assert sig.exempt is False
    assert sig.correlation_found is False
    assert sig.severity == "medium"
    assert sig.path == "single"
    assert sig.state_key == f"single:praesagium:{pattern['pattern_id']}"
    assert sig.ungateable is False  # pid is always non-empty


# -- PR1b: by clamping (arbitrary/forged payloads) ---------------------------


def _forged_valid(**over) -> dict:
    """A WELL-FORMED anticipatory envelope carrying maximally hostile danger
    fields (correlation_found=True, combined_severity HIGH, escalation rule,
    junk) -- everything _clamp_foreseen must neutralize rather than drop."""
    payload = {
        "source": "anticipatory",
        "anticipatory": {
            "pattern_id": "patDEAD",
            "forewarning_text": "Forewarning: typing (user) precedes activity (app).",
            "antecedent": "typing:user",
            "consequent": "activity:app",
        },
        "primary_anomaly": {
            "domain": "praesagium",
            "entity": "patDEAD",
            "severity": "high",
            "value": 0.9,
        },
        "correlation_found": True,
        "correlated_events": [{"forged": 1}],
        "combined_severity": "HIGH",
        "involved_domains": ["typing", "activity", "praesagium"],
        "escalation_rule": "FORGED_RULE",
        "severity_escalated": True,
        "malicious_junk": {"__proto__": "x", "nested": [1, 2, 3]},
        "temporal_lag_seconds": 5,
    }
    payload.update(over)
    return payload


# Forged-but-valid matrix: every combination MUST clamp (never drop), because
# the anticipatory block + praesagium domain + non-empty entity are intact.
_PR1B_FORGED_VALID = [
    _forged_valid(
        correlation_found=cf,
        combined_severity=cs,
        correlated_events=ce,
        involved_domains=idoms,
        primary_anomaly={
            "domain": "praesagium",
            "entity": "patDEAD",
            "severity": psev,
            "value": val,
        },
    )
    for cf in (True, False)
    for cs in ("HIGH", "MEDIUM", "CRITICAL", "low", "high")
    for ce in ([], [{"e": 1}])
    for idoms in (["a", "b"], ["praesagium"])
    for psev, val in (("high", 0.9), ("medium", 0.4))
]

# Malformed matrix: every entry MUST drop (return None).
_PR1B_MALFORMED = [
    "not-a-dict",
    123,
    None,
    _forged_valid(source="detection"),
    _forged_valid(source=None),
    _forged_valid(anticipatory=None),
    _forged_valid(anticipatory={}),
    _forged_valid(anticipatory={"pattern_id": "", "forewarning_text": "t"}),
    _forged_valid(anticipatory={"pattern_id": "p", "forewarning_text": ""}),
    _forged_valid(anticipatory={"pattern_id": "p"}),
    _forged_valid(anticipatory={"forewarning_text": "t"}),
    _forged_valid(anticipatory="not-a-dict"),
    _forged_valid(primary_anomaly={"domain": "typing", "entity": "x", "value": 1}),
    _forged_valid(primary_anomaly={"domain": "praesagium", "entity": ""}),
    _forged_valid(primary_anomaly={"domain": "praesagium"}),
    _forged_valid(primary_anomaly={"domain": "praesagium", "entity": 42}),
    _forged_valid(primary_anomaly="not-a-dict"),
    _forged_valid(primary_anomaly=None),
]


@pytest.mark.parametrize("payload", _PR1B_FORGED_VALID)
def test_pr1b_forged_valid_always_clamps_to_non_exempt(payload):
    clamped = _clamp_foreseen(payload)
    # A well-formed anticipatory envelope is NEVER dropped -- it is clamped.
    assert clamped is not None
    assert clamped["correlation_found"] is False
    assert clamped["correlated_events"] == []
    assert clamped["combined_severity"] == "MEDIUM"
    assert clamped["primary_anomaly"]["severity"] == "medium"
    assert clamped["involved_domains"] == ["praesagium"]

    sig = build_signature(clamped)
    assert sig.exempt is False
    assert sig.severity == "medium"
    assert sig.path == "single"


@pytest.mark.parametrize("payload", _PR1B_MALFORMED)
def test_pr1b_malformed_is_dropped(payload):
    assert _clamp_foreseen(payload) is None


@pytest.mark.parametrize("payload", _PR1B_FORGED_VALID + _PR1B_MALFORMED)
def test_pr1b_no_third_outcome(payload):
    """The exhaustive PR1b property: for ANY payload, _clamp_foreseen either
    DROPS it (None) or returns a payload that build_signature can never call
    exempt or high -- there is no third outcome."""
    result = _clamp_foreseen(payload)
    if result is None:
        return  # dropped -- acceptable
    sig = build_signature(result)
    assert sig.exempt is False
    assert sig.severity == "medium"
    assert sig.path == "single"
    assert result["combined_severity"] == "MEDIUM"
    assert result["correlation_found"] is False


# ===========================================================================
# PR2 -- hot-path purity
# ===========================================================================
#
# Spec sec 11, PR2: "hot-path purity. on_anomaly never mines, never calls an
# LLM, never writes the patterns blob. Structural pin: praesagium/matcher.py
# imports no httpx/ollama; behavioral pin: a matcher event never mutates
# augur:praesagium:patterns. Handler is strictly sequential (no task fan-out)
# -- structural pin."


def test_pr2_matcher_imports_no_llm_modules():
    # AST-level (not substring: the docstring mentions "httpx/ollama"): no
    # import statement in matcher.py may pull in httpx or any ollama package.
    forbidden = {"httpx", "ollama"}
    tree = ast.parse(MATCHER_SRC)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert forbidden.isdisjoint(imported_roots), imported_roots & forbidden


def test_pr2_matcher_no_task_fan_out():
    # Strictly sequential: no create_task anywhere in the anomaly path (the
    # emitted_at safety proof depends on the handler running to completion).
    assert "create_task" not in MATCHER_SRC


def test_pr2_on_anomaly_never_writes_patterns_blob():
    # A FULL arm + resolve cycle must leave augur:praesagium:patterns byte-
    # identical -- the matcher only READS the blob (the miner is the writer).
    pm = _pm()
    _seed_patterns(
        pm,
        _pattern(
            pid="p1",
            antecedent="typing:user",
            consequent="activity:editor",
            window_s=120.0,
        ),
    )
    blob_before = pm._r.get("augur:praesagium:patterns")

    pub = RecordingPublish()
    cb = make_on_anomaly(pm, _cfg(), pub, {})
    import asyncio

    # Event 1: the antecedent arms a prediction.
    asyncio.run(cb(_Msg(json.dumps(_anomaly(entity="user", ts=1000.0)).encode())))
    # Event 2: the consequent resolves it (fulfilled, within window).
    asyncio.run(
        cb(
            _Msg(
                json.dumps(
                    _anomaly(
                        domain="activity", entity="editor", severity="medium", ts=1010.0
                    )
                ).encode()
            )
        )
    )

    # The cycle actually ran (an open was armed, then resolved into the log).
    assert pm.load_praesagium_resolved(), "resolve did not run -- test is vacuous"
    # ...yet the patterns blob is untouched.
    assert pm._r.get("augur:praesagium:patterns") == blob_before


# ===========================================================================
# PR3 -- kill-switch parity (scoped)
# ===========================================================================
#
# Spec sec 11, PR3: "kill-switch parity (scoped). praesagium_enabled=False =>
# zero publishes on augur.praesagium.* subjects and zero writes to any
# augur:praesagium:* key, from every surface (matcher, miner; the consilium
# branch is unreachable -- no foreseen events exist). Carve-out: the process
# heartbeat on augur.system.heartbeat continues ... praesagium_emit_enabled=
# False => zero augur.praesagium.foreseen publishes ever, while episodes/
# predictions/resolutions continue (watch-first). Byte-parity matrix like
# Conscientia's C5."


def test_pr3_matcher_callback_disabled_zero_writes_zero_publishes():
    import asyncio

    # (i) Nothing seeded: a disabled handler creates NO augur:praesagium:* keys.
    pm = _pm()
    pub = RecordingPublish()
    cb = make_on_anomaly(pm, _cfg(praesagium_enabled=False), pub, {})
    asyncio.run(cb(_Msg(json.dumps(_anomaly()).encode())))
    assert _prae_keys(pm) == []
    assert pub.calls == []

    # (ii) Even with an armable pattern seeded, the disabled handler mutates
    # nothing and publishes nothing (byte-parity with pre-Praesagium).
    pm2 = _pm()
    _seed_patterns(pm2, _pattern(pid="p1", antecedent="typing:user"))
    before = set(pm2._r.keys())
    pub2 = RecordingPublish()
    cb2 = make_on_anomaly(pm2, _cfg(praesagium_enabled=False), pub2, {})
    asyncio.run(cb2(_Msg(json.dumps(_anomaly()).encode())))
    assert set(pm2._r.keys()) == before
    assert pub2.calls == []


def test_pr3_miner_disabled_zero_writes():
    pm = _pm()
    # Even with a corpus present, the disabled gate returns before any read.
    pm.append_praesagium_episode("s0", _ep("typing:user", "low", 0.0))
    pm.append_praesagium_episode("s0", _ep("activity:app", "medium", 50.0))
    keys_before = set(pm._r.keys())

    result = run_praesagium_mining("s0", pm, _cfg(praesagium_enabled=False))

    assert result == {"skipped": True, "reason": "disabled"}
    # No NEW key created; specifically no patterns blob and no tuning marker.
    assert set(pm._r.keys()) == keys_before
    assert pm.load_praesagium_patterns() is None
    assert not pm.is_tuning_applied("s0", pass_name="praesagium")


def test_pr3_emit_disabled_no_foreseen_while_predictions_accumulate():
    import asyncio

    # Watch-first: enabled but emit OFF. Three armable patterns on distinct
    # antecedents; each anomaly arms a prediction but NONE publishes foreseen.
    pm = _pm()
    _seed_patterns(
        pm,
        _pattern(pid="pa", antecedent="typing:user", consequent="activity:editor"),
        _pattern(pid="pb", antecedent="activity:browser", consequent="typing:user"),
        _pattern(pid="pc", antecedent="chess:white", consequent="chess:black"),
    )
    pub = RecordingPublish()
    cfg = _cfg(praesagium_emit_enabled=False)  # enabled True (default), emit OFF
    cooldowns: dict[str, float] = {}
    cb = make_on_anomaly(pm, cfg, pub, cooldowns)

    for domain, entity in (
        ("typing", "user"),
        ("activity", "browser"),
        ("chess", "white"),
    ):
        asyncio.run(
            cb(_Msg(json.dumps(_anomaly(domain=domain, entity=entity)).encode()))
        )

    # Predictions accumulated (watch-first learning continues) ...
    opens = pm.load_praesagium_open_predictions()
    assert {r["pattern_id"] for r in opens} == {"pa", "pb", "pc"}
    # ... but not a single foreseen event went out.
    assert SUBJECT_FORESEEN not in pub.subjects()


# ===========================================================================
# PR4 -- bounded state
# ===========================================================================
#
# Spec sec 11, PR4: "bounded state. Episodes <= cap per session (LTRIM);
# episode index <= 1000; non-retired patterns <= praesagium_max_patterns and
# retired pruned to the same bound; open predictions <= cap (save refuses; cap
# is unreachable by invariant ...); resolved log <= cap (LTRIM). No Praesagium
# key grows without bound."


def test_pr4_episodes_capped_per_session():
    pm = _pm()
    for i in range(60):
        pm.append_praesagium_episode("one", _ep("a:b", "low", float(i)), cap=25)
    assert pm._r.llen("augur:praesagium:episodes:one") == 25
    # LTRIM keeps the NEWEST (RPUSH tail): last 25 timestamps 35..59.
    eps = pm.load_praesagium_episodes("one")
    assert [e["t"] for e in eps] == [float(i) for i in range(35, 60)]


def test_pr4_episode_index_capped_at_1000():
    pm = _pm()
    for i in range(1005):
        pm.append_praesagium_episode(f"sess{i}", _ep("a:b", "low", 1.0))
    assert pm._r.llen("augur:praesagium:episodes:_index") == 1000


def test_pr4_open_predictions_refused_at_cap():
    pm = _pm()
    saved = [
        pm.save_praesagium_open_prediction({"prediction_id": f"o{i}"}, cap=5)
        for i in range(12)
    ]
    assert sum(saved) == 5  # only the first 5 NEW ids stored
    assert saved[5:] == [False] * 7  # every subsequent NEW id refused
    assert pm._r.hlen("augur:praesagium:predictions:open") == 5


def test_pr4_resolved_log_ltrimmed_to_cap():
    pm = _pm()
    for i in range(20):
        pm.save_praesagium_open_prediction({"prediction_id": f"r{i}"}, cap=1000)
    for i in range(20):
        pm.resolve_praesagium_prediction(
            f"r{i}",
            {"prediction_id": f"r{i}", "outcome": "expired", "resolved_ts": float(i)},
            cap=7,
        )
    assert pm._r.llen("augur:praesagium:predictions:log") == 7


def _bare_candidate(pid: str, conf_lower: float) -> dict:
    return {
        "pattern_id": pid,
        "antecedent": f"a:{pid}",
        "consequent": f"b:{pid}",
        "window_s": 100.0,
        "support_sessions": 3,
        "n": 4,
        "k": 3,
        "conf": 0.75,
        "conf_lower": conf_lower,
        "lift": 2.0,
        "lag_median_s": 40.0,
        "lag_p90_s": 90.0,
    }


def test_pr4_patterns_blob_non_retired_bounded():
    cfg = _cfg(praesagium_max_patterns=3)
    candidates = {
        f"c{i}": _bare_candidate(f"c{i}", conf_lower=0.4 + i * 0.01) for i in range(10)
    }
    blob = merge_blob(None, candidates, [], now=1000.0, cfg=cfg, corpus_newest_ts=0.0)
    patterns = blob["patterns"]
    non_retired = [p for p in patterns.values() if p.get("status") != "retired"]
    assert len(non_retired) <= 3


def test_pr4_patterns_blob_retired_bounded():
    cfg = _cfg(praesagium_max_patterns=2)
    # Previous blob carrying 6 retired patterns; no candidates re-pass them.
    prev = {
        "version": 1,
        "mined_at": 0.0,
        "hit_rate_watermark": 0.0,
        "patterns": {
            f"old{i}": {
                **_bare_candidate(f"old{i}", conf_lower=0.5),
                "status": "retired",
                "retired_at": float(i),
                "retired_reason": "hit_rate",
                "hit_rate": 0.1,
                "resolutions": 9,
                "created_at": 0.0,
                "mined_at": 0.0,
                "repass_streak": 0,
            }
            for i in range(6)
        },
    }
    blob = merge_blob(prev, {}, [], now=1000.0, cfg=cfg, corpus_newest_ts=0.0)
    retired = [p for p in blob["patterns"].values() if p.get("status") == "retired"]
    assert len(retired) <= 2
    # The newest retired_at survive (drop oldest first).
    kept = {p["pattern_id"] for p in retired}
    assert kept == {"old4", "old5"}


# ===========================================================================
# PR5 -- honest math (end-to-end THROUGH mine_corpus)
# ===========================================================================
#
# Spec sec 11, PR5: "honest math. (a) adding A-trials with no B strictly lowers
# p_hat and p_lower; (b) duplicating one session's episodes Nx leaves support
# unchanged; (c) burst-collapse: N A-occurrences within lag_min of a trial
# start count as ONE trial, and one B occurrence satisfies at most one trial;
# (d) a consequent B so common in A-sessions that P0 >= p_hat/lift_min is never
# promoted; (e) k <= n, p_lower <= p_hat always; (f) pass-2 recount: every
# promoted pattern's conf is computed over exactly (lag_min, W]."

_A, _B, _C = "typing:user", "activity:app", "gamma:filler"
# Permissive but bound-legal cfg: lets small honest corpora promote so the
# corpus-level consequences are observable (retire_below < conf_lower_min).
_PERMISSIVE = dict(
    praesagium_conf_lower_min=0.05,
    praesagium_lift_min=1.0,
    praesagium_support_min_sessions=3,
    praesagium_hit_rate_retire_below=0.02,
)


def _ab_corpus(nsess: int = 5) -> dict:
    return {
        f"s{i}": [_ep(_A, "low", 0), _ep(_B, "medium", 50), _ep(_C, "low", 2000)]
        for i in range(nsess)
    }


def test_pr5a_adding_failing_a_trials_lowers_confidence():
    cfg = _cfg(**_PERMISSIVE)
    pid = pattern_id(_A, _B)

    base = _ab_corpus(5)
    with_fail = dict(base)
    for j in range(3):  # A occurs, no B -> failing trials (n up, k flat)
        with_fail[f"fail{j}"] = [_ep(_A, "low", 0), _ep(_C, "low", 2000)]

    art0 = mine_corpus(base, cfg)[pid]
    art1 = mine_corpus(with_fail, cfg)[pid]

    assert art1["n"] > art0["n"]
    assert art1["k"] == art0["k"]
    assert art1["conf"] < art0["conf"]
    assert art1["conf_lower"] < art0["conf_lower"]


def test_pr5b_duplicating_session_leaves_support_unchanged():
    cfg = _cfg(**_PERMISSIVE)
    pid = pattern_id(_A, _B)

    base = _ab_corpus(3)
    chatty = {k: list(v) for k, v in base.items()}
    chatty["s0"] = chatty["s0"] * 4  # one chatty session, 4x its episodes

    base_support = mine_corpus(base, cfg)[pid]["support_sessions"]
    chatty_support = mine_corpus(chatty, cfg)[pid]["support_sessions"]
    assert base_support == chatty_support == 3


def test_pr5c_burst_collapses_to_one_trial():
    cfg = _cfg(**_PERMISSIVE)
    pid = pattern_id(_A, _B)
    # 4 A-occurrences within lag_min (default 10) of the trial start -> ONE
    # trial per session; without collapse n would be 12 (3 sessions x 4).
    corpus = {
        f"b{i}": [
            _ep(_A, "low", 0),
            _ep(_A, "low", 3),
            _ep(_A, "low", 6),
            _ep(_A, "low", 9),
            _ep(_B, "medium", 50),
            _ep(_C, "low", 2000),
        ]
        for i in range(3)
    }
    art = mine_corpus(corpus, cfg)[pid]
    assert art["n"] == 3  # one collapsed trial per session
    assert art["k"] == 3


def test_pr5c_one_b_satisfies_at_most_one_trial():
    cfg = _cfg(**_PERMISSIVE)
    pid = pattern_id(_A, _B)
    # Two independent trials per session (A@0 and A@200, gap > lag_min) but a
    # SINGLE B@50 -- only trial-1 may consume it. So per session n=2, k=1.
    corpus = {
        f"tt{i}": [_ep(_A, "low", 0), _ep(_B, "medium", 50), _ep(_A, "low", 200)]
        for i in range(3)
    }
    art = mine_corpus(corpus, cfg)[pid]
    assert art["n"] == 6  # 2 trials x 3 sessions
    assert art["k"] == 3  # the single B per session satisfies exactly one trial


def _common_b_corpus(bn: int) -> dict:
    """10 A-sessions: A(low) every 300s (10 per session); B(medium) every 100s
    for the first ``bn`` slots. Large ``bn`` => B near-certain in-window => the
    session-conditional P0 rises until lift drops below lift_min."""
    corpus = {}
    for s in range(10):
        eps = [_ep("desk:A", "low", 300 * i) for i in range(10)]
        eps += [_ep("desk:B", "medium", 100 * j) for j in range(1, bn + 1)]
        corpus[f"a{s}"] = eps
    return corpus


def _rare_b_corpus() -> dict:
    """Same antecedent schedule as ``_common_b_corpus`` but B fires just once
    after each A (10/session) with a far filler inflating D_s -- only the lift
    discriminant differs from the over-common case, so this one PROMOTES."""
    corpus = {}
    for s in range(10):
        eps = [_ep("desk:A", "low", 300 * i) for i in range(10)]
        eps += [_ep("desk:B", "medium", 300 * i + 100) for i in range(10)]
        eps += [_ep("far:z", "low", 100_000)]
        corpus[f"a{s}"] = eps
    return corpus


def test_pr5d_over_common_consequent_rejected_but_rare_promoted():
    dpid = pattern_id("desk:A", "desk:B")
    # Over-common B (30/session): P0 high -> lift < 1.5 -> REJECTED.
    assert dpid not in mine_corpus(_common_b_corpus(30), _cfg())
    # Rare B (10/session), everything else equal -> PROMOTED (lift is the gate).
    result = mine_corpus(_rare_b_corpus(), _cfg())
    assert dpid in result
    assert result[dpid]["lift"] >= _cfg().praesagium_lift_min


def _two_pass_corpus() -> dict:
    lags = [20, 30, 40, 50, 60, 70, 80, 90, 100, 850]
    return {
        f"tp{i}": [_ep("tp:A", "low", 0), _ep("tp:B", "medium", float(lag))]
        for i, lag in enumerate(lags)
    }


@pytest.mark.parametrize(
    "corpus_fn,cfg_over",
    [
        (lambda: _ab_corpus(5), _PERMISSIVE),
        (_two_pass_corpus, dict(praesagium_lift_min=1.2)),
        (_rare_b_corpus, {}),
    ],
)
def test_pr5e_k_le_n_and_conf_lower_le_conf(corpus_fn, cfg_over):
    result = mine_corpus(corpus_fn(), _cfg(**cfg_over))
    assert result, "corpus promoted nothing -- test would be vacuous"
    for art in result.values():
        assert 0 <= art["k"] <= art["n"]
        assert art["conf"] == pytest.approx(art["k"] / art["n"])
        assert art["conf_lower"] <= art["conf"] + 1e-12
        assert art["conf_lower"] >= 0.0


@pytest.mark.parametrize(
    "corpus_fn,cfg_over,key_a,key_b",
    [
        (lambda: _ab_corpus(5), _PERMISSIVE, _A, _B),
        (_two_pass_corpus, dict(praesagium_lift_min=1.2), "tp:A", "tp:B"),
    ],
)
def test_pr5f_conf_computed_over_exactly_lag_min_to_window(
    corpus_fn, cfg_over, key_a, key_b
):
    """Every promoted pattern's conf is k/n recounted over (lag_min, W]. Both
    corpora place exactly one A@0 per session, so trials == [0.0] per session
    and an independent count_matches recount reproduces n, k over the artifact's
    own window -- verifying the mined conf names exactly the runtime window."""
    cfg = _cfg(**cfg_over)
    corpus = corpus_fn()
    art = mine_corpus(corpus, cfg)[pattern_id(key_a, key_b)]
    lag_min = cfg.praesagium_lag_min_s
    window = art["window_s"]

    n_rec = k_rec = 0
    for episodes in corpus.values():
        b_times = [
            e["t"] for e in episodes if e["k"] == key_b and e["s"] in ("medium", "high")
        ]
        k_s, _ = count_matches([0.0], b_times, lag_min, window)
        n_rec += 1
        k_rec += k_s

    assert (n_rec, k_rec) == (art["n"], art["k"])
    assert art["conf"] == pytest.approx(k_rec / n_rec)


# ===========================================================================
# PR6 -- exactly-once resolution
# ===========================================================================
#
# Spec sec 11, PR6: "exactly-once resolution. Every prediction resolves to
# exactly one of fulfilled/expired via the atomic append-iff-removed op;
# duplicate delivery, crash-between-ops, and replay produce no second
# resolution and no lost outcome; after resolution + the miner's expiry sweep,
# no open prediction with a past deadline survives."


def test_pr6_duplicate_resolve_is_a_no_op():
    pm = _pm()
    pm.save_praesagium_open_prediction(
        {
            "prediction_id": "x1",
            "pattern_id": "p",
            "created_ts": 1.0,
            "deadline_ts": 9.0,
        }
    )
    rec = {"prediction_id": "x1", "outcome": "fulfilled", "resolved_ts": 5.0}
    assert pm.resolve_praesagium_prediction("x1", rec) is True
    # Replay / duplicate delivery: no second resolution, no double log entry.
    assert pm.resolve_praesagium_prediction("x1", rec) is False
    assert pm.load_praesagium_open_predictions() == []
    assert len(pm.load_praesagium_resolved()) == 1


class _CrashOnceLpushPipeline:
    """Wraps a real fakeredis pipeline; its FIRST ``lpush`` raises to simulate a
    crash mid-CAS (after HDEL is buffered inside MULTI, before EXEC commits)."""

    def __init__(self, real) -> None:
        self._real = real
        self._armed = True

    def __enter__(self):
        self._real.__enter__()
        return self

    def __exit__(self, *exc):
        return self._real.__exit__(*exc)

    def lpush(self, *a, **k):
        if self._armed:
            self._armed = False
            raise RuntimeError("simulated crash mid-CAS")
        return self._real.lpush(*a, **k)

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_pr6_crash_mid_cas_open_survives_and_retry_resolves_once():
    pm = _pm()
    open_rec = {
        "prediction_id": "x1",
        "pattern_id": "p",
        "created_ts": 1.0,
        "deadline_ts": 9.0,
    }
    pm.save_praesagium_open_prediction(open_rec)
    resolved = {"prediction_id": "x1", "outcome": "fulfilled", "resolved_ts": 5.0}

    real_pipeline = pm._r.pipeline
    pm._r.pipeline = lambda *a, **k: _CrashOnceLpushPipeline(real_pipeline(*a, **k))
    with pytest.raises(RuntimeError, match="crash mid-CAS"):
        pm.resolve_praesagium_prediction("x1", resolved)
    pm._r.pipeline = real_pipeline  # un-crash for the retry

    # The MULTI never committed: the open record survives UNTOUCHED, log empty.
    survivors = pm.load_praesagium_open_predictions()
    assert len(survivors) == 1
    assert survivors[0]["prediction_id"] == "x1"
    assert pm.load_praesagium_resolved() == []

    # Retry resolves exactly once; a further replay is a no-op.
    assert pm.resolve_praesagium_prediction("x1", resolved) is True
    assert pm.resolve_praesagium_prediction("x1", resolved) is False
    assert pm.load_praesagium_open_predictions() == []
    assert len(pm.load_praesagium_resolved()) == 1


def test_pr6_miner_expiry_sweep_leaves_no_past_deadline_open():
    pm = _pm()
    # An open prediction whose deadline is long past (session ended, no further
    # matcher events) must be expired by the miner's sweep.
    pm.save_praesagium_open_prediction(
        {
            "prediction_id": "stale1",
            "pattern_id": "p",
            "created_ts": 1.0,
            "deadline_ts": 2.0,  # far in the past relative to time.time()
            "session_id": "s0",
        }
    )
    run_praesagium_mining("mine-sess", pm, _cfg())

    import time

    now = time.time()
    survivors = pm.load_praesagium_open_predictions()
    assert all(r["deadline_ts"] >= now for r in survivors)
    assert survivors == []  # the only open was stale
    resolved = pm.load_praesagium_resolved()
    assert any(
        r["prediction_id"] == "stale1" and r["outcome"] == "expired" for r in resolved
    )


# ===========================================================================
# PR7 -- no delivery bypass
# ===========================================================================
#
# Spec sec 11, PR7: "no delivery bypass. The matcher never publishes on
# augur.consilium.advice; forewarnings reach the user only through Consilium's
# gated process_message path (structural pin on matcher source + behavioral
# test)."


def test_pr7_matcher_source_never_names_advice_subject():
    assert "augur.consilium.advice" not in MATCHER_SRC


def test_pr7_foreseen_publish_only_to_praesagium_subject():
    import asyncio

    pm = _pm()
    _seed_patterns(
        pm, _pattern(pid="p1", antecedent="typing:user", consequent="activity:editor")
    )
    pub = RecordingPublish()
    cfg = _cfg(praesagium_emit_enabled=True)  # emit ARMED
    cb = make_on_anomaly(pm, cfg, pub, {})
    asyncio.run(cb(_Msg(json.dumps(_anomaly(entity="user")).encode())))

    subjects = pub.subjects()
    # A foreseen event WAS published (the arm+emit path ran) ...
    assert SUBJECT_FORESEEN in subjects
    # ... and EVERY publish stayed within the faculty's own namespace -- the
    # matcher never bypasses Consilium onto the advice subject.
    assert all(s.startswith("augur.praesagium.") for s in subjects)
    assert "augur.consilium.advice" not in subjects


# ===========================================================================
# PR8 -- single-writer patterns
# ===========================================================================
#
# Spec sec 11, PR8: "single-writer patterns. Only run_praesagium_mining calls
# save_praesagium_patterns (structural pin across the repo source, a la the
# matrix-write-path guard)."


def test_pr8_save_patterns_call_sites_bounded():
    needle = "save_praesagium_patterns("
    offenders = []
    for path in REPO_ROOT.rglob("*.py"):
        parts = set(path.parts)
        if parts & {".venv", "__pycache__", "site-packages", "build", ".git"}:
            continue
        if needle not in path.read_text(encoding="utf-8"):
            continue
        rel = path.relative_to(REPO_ROOT)
        # Allowed: the miner (the single writer), the PM (the definition), tests.
        if rel.parts[0] == "tests":
            continue
        if rel.as_posix() in {"praesagium/miner.py", "tabula/persistence.py"}:
            continue
        offenders.append(rel.as_posix())
    assert offenders == [], (
        f"unexpected save_praesagium_patterns call sites: {offenders}"
    )
