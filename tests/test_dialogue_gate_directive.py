"""Unit tests for the Imperator III taught-directive pre-check (spec §7.2).

Two concerns:

* ``PersistenceManager.load_focused_app`` — the single source of truth for
  the currently-focused app, reading the ``augur:vigil:history:{activity_focus,
  activity_intensity}`` keys (also consumed by ``imperator/sources.py::gather``
  for the ``activity`` read-model field).
* ``Gate.evaluate``'s taught-directive pre-check itself, driven through a
  minimal ``_PM`` stub (not the full ``fake_pm`` fixture) — the pre-check
  returns before ``Gate._load_state`` on the non-exempt path, so the stub
  needs only ``load_dialogue_directives`` + ``load_focused_app`` +
  ``can_track_gate_state`` (unused here, but kept for parity with a real PM's
  surface / documented in the task report).
"""

from __future__ import annotations

from tabula.config import AugurConfig
from tabula.contracts import PerceptionEvent

from limen import gate as G
from tests.conftest import EXEMPT_PAYLOAD, SINGLE_MEDIUM_TYPING

# ── Step 1: the focused-app reader ──────────────────────────────────────────


def test_load_focused_app_prefers_newer_intensity_stream(fake_pm):
    fake_pm.append_event(
        PerceptionEvent(
            domain="activity_focus",
            stream_id="activity_focus",
            entity="editor",
            event_type="focus_change",
            value=0.0,
            unit="none",
            context={"new_app": "editor"},
            timestamp="2026-06-14T00:00:00+00:00",
            session_id="s1",
        )
    )
    fake_pm.append_event(
        PerceptionEvent(
            domain="activity_intensity",
            stream_id="activity_intensity",
            entity="ide",
            event_type="intensity_tick",
            value=88.0,
            unit="pct",
            context={"focused_app": "ide"},
            timestamp="2026-06-14T00:00:10+00:00",
            session_id="s1",
        )
    )
    assert fake_pm.load_focused_app() == "ide"


def test_load_focused_app_falls_back_to_focus_stream(fake_pm):
    fake_pm.append_event(
        PerceptionEvent(
            domain="activity_focus",
            stream_id="activity_focus",
            entity="editor",
            event_type="focus_change",
            value=0.0,
            unit="none",
            context={"new_app": "editor"},
            timestamp="2026-06-14T00:00:00+00:00",
            session_id="s1",
        )
    )
    assert fake_pm.load_focused_app() == "editor"


def test_load_focused_app_none_when_no_activity_history(fake_pm):
    assert fake_pm.load_focused_app() is None


# ── Step 2: the gate pre-check ──────────────────────────────────────────────


class _PM:
    """Minimal PM stub — the pre-check returns before ``_load_state`` runs,
    so no other read/write method is ever reached on the non-exempt path."""

    def __init__(self, app, directives):
        self._app, self._d = app, directives

    def load_focused_app(self, **_k):
        return self._app

    def load_dialogue_directives(self):
        return self._d

    def can_track_gate_state(self, *a, **k):
        return True


def _sig(exempt: bool = False) -> G.Signature:
    """A representative gate signature; ``exempt`` selects the high+correlated
    payload (invariant B) vs. an ordinary single/medium payload."""
    payload = EXEMPT_PAYLOAD if exempt else SINGLE_MEDIUM_TYPING
    return G.build_signature(payload)


def _gate_cfg() -> AugurConfig:
    return AugurConfig()


def test_directive_suppresses_when_app_matches():
    pm = _PM(
        "appX",
        [
            {
                "directive_id": "d1",
                "predicate": {"context": "focused_app", "match": "appX"},
                "action": "suppress",
                "scope": "all",
            }
        ],
    )
    g = G.Gate()
    dec = g.evaluate(_sig(exempt=False), pm, _gate_cfg(), now=100.0)
    assert dec.action == "suppress"
    assert dec.reason == "taught_directive:d1"
    assert dec.deciding_arm == "taught_directive"


def test_directive_never_suppresses_exempt_high_correlated():
    """Invariant B wins: an exempt (high+correlated) signature fires before the
    directive pre-check is even reached — the taught directive never fires."""
    pm = _PM(
        "appX",
        [
            {
                "directive_id": "d1",
                "predicate": {"context": "focused_app", "match": "appX"},
                "action": "suppress",
                "scope": "all",
            }
        ],
    )
    g = G.Gate()
    dec = g.evaluate(_sig(exempt=True), pm, _gate_cfg(), now=100.0)
    assert dec.action == "fire"
    assert dec.reason == "exempt_high_correlated"


def test_no_directives_is_a_noop(fake_pm, cfg):
    """No directives ⇒ the pre-check falls through to the normal pipeline
    (driven through the real fake_pm so the rest of evaluate() actually runs)."""
    g = G.Gate()
    dec = g.evaluate(_sig(exempt=False), fake_pm, cfg, now=1.0)
    assert dec.deciding_arm != "taught_directive"


def test_directive_predicate_mismatch_does_not_suppress(fake_pm, cfg):
    """A directive for a different app than the one currently focused must not
    fire — proves the pre-check is not a blanket suppress-if-any-directive.

    Uses the real fake_pm (not the minimal _PM stub): a non-matching directive
    does NOT short-circuit, so evaluate() falls through into the full
    biological pipeline, which needs the full PersistenceManager surface.
    """
    fake_pm.load_focused_app = lambda **_k: "appY"
    fake_pm.add_dialogue_directive(
        {
            "directive_id": "d1",
            "predicate": {"context": "focused_app", "match": "appX"},
            "action": "suppress",
            "scope": "all",
        }
    )
    g = G.Gate()
    dec = g.evaluate(_sig(exempt=False), fake_pm, cfg, now=100.0)
    assert dec.deciding_arm != "taught_directive"


def test_directive_unknown_action_does_not_suppress(fake_pm, cfg):
    """Only action in {"suppress", "downgrade"} is consulted by this
    pre-check; any other value (a future/unrecognized action) is a no-op."""
    fake_pm.load_focused_app = lambda **_k: "appX"
    fake_pm.add_dialogue_directive(
        {
            "directive_id": "d1",
            "predicate": {"context": "focused_app", "match": "appX"},
            "action": "annotate",
            "scope": "all",
        }
    )
    g = G.Gate()
    dec = g.evaluate(_sig(exempt=False), fake_pm, cfg, now=100.0)
    assert dec.deciding_arm != "taught_directive"


# ── Task-13-follow-up: "downgrade" directive consult ────────────────────────


def test_directive_downgrade_action_produces_downgrade_decision():
    """A taught "downgrade" directive matching the focused app returns a
    DOWNGRADE decision (not suppress, not fire) — the confirmed teach must
    actually change gate behavior, not just be echoed back."""
    pm = _PM(
        "appX",
        [
            {
                "directive_id": "d1",
                "predicate": {"context": "focused_app", "match": "appX"},
                "action": "downgrade",
                "scope": "all",
            }
        ],
    )
    g = G.Gate()
    dec = g.evaluate(_sig(exempt=False), pm, _gate_cfg(), now=100.0)
    assert dec.action == "downgrade"
    assert dec.reason == "taught_directive:d1"
    assert dec.deciding_arm == "taught_directive"


def test_directive_downgrade_never_overrides_exempt_high_correlated():
    """Invariant B wins over a taught downgrade too: an exempt (high+
    correlated) signature fires before the directive pre-check is reached."""
    pm = _PM(
        "appX",
        [
            {
                "directive_id": "d1",
                "predicate": {"context": "focused_app", "match": "appX"},
                "action": "downgrade",
                "scope": "all",
            }
        ],
    )
    g = G.Gate()
    dec = g.evaluate(_sig(exempt=True), pm, _gate_cfg(), now=100.0)
    assert dec.action == "fire"
    assert dec.reason == "exempt_high_correlated"


def test_directive_downgrade_scoped_to_other_domain_does_not_downgrade(fake_pm, cfg):
    """A downgrade directive scoped to ["chess"] must not touch a typing
    anomaly even when the focused-app predicate matches."""
    fake_pm.load_focused_app = lambda **_k: "appX"
    fake_pm.add_dialogue_directive(
        {
            "directive_id": "d1",
            "predicate": {"context": "focused_app", "match": "appX"},
            "action": "downgrade",
            "scope": ["chess"],
        }
    )
    g = G.Gate()
    dec = g.evaluate(_sig(exempt=False), fake_pm, cfg, now=100.0)
    assert dec.deciding_arm != "taught_directive"


def test_directive_scoped_to_matching_domain_suppresses():
    """A directive scoped to the event's domain applies (design spec §7.2:
    scope = "all" | ["<domain>", ...])."""
    pm = _PM(
        "appX",
        [
            {
                "directive_id": "d1",
                "predicate": {"context": "focused_app", "match": "appX"},
                "action": "suppress",
                "scope": ["typing"],
            }
        ],
    )
    g = G.Gate()
    # SINGLE_MEDIUM_TYPING → signature.domain == "typing", in scope.
    dec = g.evaluate(_sig(exempt=False), pm, _gate_cfg(), now=100.0)
    assert dec.action == "suppress"
    assert dec.deciding_arm == "taught_directive"


def test_directive_scoped_to_other_domain_does_not_suppress(fake_pm, cfg):
    """A directive scoped to ["chess"] must NOT silence a typing anomaly even
    when the focused-app predicate matches — scope restricts the domains the
    taught silence covers. Falls through to the real pipeline ⇒ real fake_pm."""
    fake_pm.load_focused_app = lambda **_k: "appX"
    fake_pm.add_dialogue_directive(
        {
            "directive_id": "d1",
            "predicate": {"context": "focused_app", "match": "appX"},
            "action": "suppress",
            "scope": ["chess"],
        }
    )
    g = G.Gate()
    dec = g.evaluate(_sig(exempt=False), fake_pm, cfg, now=100.0)
    assert dec.deciding_arm != "taught_directive"


def test_directive_absent_scope_applies_to_all_domains():
    """A directive with no scope key at all behaves like scope="all"."""
    pm = _PM(
        "appX",
        [
            {
                "directive_id": "d1",
                "predicate": {"context": "focused_app", "match": "appX"},
                "action": "suppress",
            }
        ],
    )
    g = G.Gate()
    dec = g.evaluate(_sig(exempt=False), pm, _gate_cfg(), now=100.0)
    assert dec.action == "suppress"
    assert dec.deciding_arm == "taught_directive"


def test_directive_missing_focused_app_does_not_suppress(fake_pm, cfg):
    """No focused-app signal available (None, the natural default when no
    activity history exists yet) ⇒ a directive can never match."""
    fake_pm.add_dialogue_directive(
        {
            "directive_id": "d1",
            "predicate": {"context": "focused_app", "match": "appX"},
            "action": "suppress",
            "scope": "all",
        }
    )
    assert fake_pm.load_focused_app() is None
    g = G.Gate()
    dec = g.evaluate(_sig(exempt=False), fake_pm, cfg, now=100.0)
    assert dec.deciding_arm != "taught_directive"


def test_teach_to_gate_end_to_end_directive_matches(fake_pm, cfg):
    """Flagship F14 regression: a taught context directive routed with the LIVE
    focused app, confirmed-applied, produces a stored directive the gate
    actually matches -- no false "applied" for an inert directive (invariant 7).

    Drives the real seam: assemble (fills ctx.focused_app from load_focused_app)
    -> route (fills predicate.match from ctx.focused_app) -> apply_confirmed
    (stores it) -> Gate.evaluate (matches on the SAME load_focused_app value).
    """
    from imperator.dialogue import context as C, router as R

    fake_pm.load_focused_app = lambda **_k: "deepwork.exe"
    ctx = C.assemble(fake_pm, now=100.0, cfg=cfg)
    assert ctx.focused_app == "deepwork.exe"
    pending = R.route(
        {
            "kind": "teach_context_directive",
            "target": "deep work",
            "action": {"action": "suppress", "scope": "all"},
            "rationale": "stay silent in deep work",
        },
        ctx,
        pm=fake_pm,
        cfg=cfg,
    )
    applied = R.apply_confirmed(pending, pm=fake_pm, cfg=cfg, session_id="s1")
    assert applied["status"] == "applied"
    stored = fake_pm.load_dialogue_directives()
    assert len(stored) == 1
    assert stored[0]["predicate"] == {
        "context": "focused_app",
        "match": "deepwork.exe",
    }
    dec = G.Gate().evaluate(_sig(exempt=False), fake_pm, cfg, now=200.0)
    assert dec.action == "suppress"
    assert dec.deciding_arm == "taught_directive"


def test_directive_bare_string_scope_fails_closed(fake_pm, cfg):
    """F2 regression: a stored directive whose scope is a bare non-"all" string
    (a malformed/legacy record) must NOT widen suppression to a domain it does
    not name. _directive_scope_allows fails closed."""
    fake_pm.load_focused_app = lambda **_k: "appX"
    fake_pm.add_dialogue_directive(
        {
            "directive_id": "d1",
            "predicate": {"context": "focused_app", "match": "appX"},
            "action": "suppress",
            "scope": "chess",  # bare string, NOT a list -> must not cover typing
        }
    )
    dec = G.Gate().evaluate(_sig(exempt=False), fake_pm, cfg, now=100.0)
    assert dec.deciding_arm != "taught_directive"


def test_load_focused_app_stale_read_returns_none(fake_pm):
    """F3 residual: a focus/intensity event older than max_age_s is treated as
    absent, so a stalled activity Sensor can't keep reporting an app the user
    left. Omitting the bounds preserves the raw newest-entry read."""
    from tabula.persistence import _to_epoch

    ts = "2026-06-14T00:00:00+00:00"
    fake_pm.append_event(
        PerceptionEvent(
            domain="activity_intensity",
            stream_id="activity_intensity",
            entity="ide",
            event_type="intensity_tick",
            value=88.0,
            unit="pct",
            context={"focused_app": "ide"},
            timestamp=ts,
            session_id="s1",
        )
    )
    epoch = _to_epoch(ts)
    assert fake_pm.load_focused_app(now=epoch + 100.0, max_age_s=300.0) == "ide"
    assert fake_pm.load_focused_app(now=epoch + 400.0, max_age_s=300.0) is None
    assert fake_pm.load_focused_app() == "ide"  # no bound -> unfiltered


def test_directive_stale_focused_app_does_not_suppress(fake_pm, cfg):
    """F3 residual at the gate: when the last focus event is stale (older than
    cfg.focused_app_max_age_s) load_focused_app returns None, the taught
    directive stops matching, and the gate falls through (fails toward FIRE)."""
    from tabula.persistence import _to_epoch

    ts = "2026-06-14T00:00:00+00:00"
    fake_pm.append_event(
        PerceptionEvent(
            domain="activity_intensity",
            stream_id="activity_intensity",
            entity="appX",
            event_type="intensity_tick",
            value=50.0,
            unit="pct",
            context={"focused_app": "appX"},
            timestamp=ts,
            session_id="s1",
        )
    )
    fake_pm.add_dialogue_directive(
        {
            "directive_id": "d1",
            "predicate": {"context": "focused_app", "match": "appX"},
            "action": "suppress",
            "scope": "all",
        }
    )
    stale_now = _to_epoch(ts) + cfg.focused_app_max_age_s + 100.0
    dec = G.Gate().evaluate(_sig(exempt=False), fake_pm, cfg, now=stale_now)
    assert dec.deciding_arm != "taught_directive"
