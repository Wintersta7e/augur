"""Pure health/supervision engine for Praefectus. Redis-free, no AugurConfig
mutation: plain dicts/dataclasses in, new state out. The monitor (praefectus/
monitor.py) owns all I/O. See
docs/superpowers/specs/2026-06-10-praefectus-supervision-health-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

HEARTBEAT_SUBJECT = "augur.system.heartbeat"
HEALTH_SUBJECT = "augur.praefectus.health"

# Structural (not user-tunable): who must be alive vs who is optional.
REQUIRED_FACULTIES: tuple[str, ...] = (
    "vigil",
    "nexus",
    "consilium",
    "responsum",
    "disciplina",
    "vox",
    "praefectus",
    "imperator",
    "imperator_ii",
    "praesagium",
)
OPTIONAL_COMPONENTS: tuple[str, ...] = (
    "sensus.chess",
    "sensus.typing",
    "sensus.activity",
)

# overall_state severity order (higher = worse).
_SEVERITY = {
    "dead": 4,
    "stale": 3,
    "degraded": 2,
    "warming_up": 1,
    "unknown": 1,
    "alive": 0,
    "absent": 0,
    "ok": 0,
}


@dataclass
class FacultyHealth:
    faculty: str
    required: bool
    seen: bool = False
    last_heartbeat: float | None = None
    last_event_ts: float | None = None
    liveness_state: str = "unknown"
    activity_state: str = "ok"
    overall_state: str = "unknown"
    reasons: list[str] = field(default_factory=list)  # alerting (dead/degraded)
    flags: list[str] = field(
        default_factory=list
    )  # observability-only (e.g. reflection_lag)


@dataclass
class ActivityWindow:
    detected_mh: list[float] = field(
        default_factory=list
    )  # MEDIUM/HIGH nexus.detected ts
    advice: list[float] = field(default_factory=list)
    suppressed: list[float] = field(default_factory=list)
    delivery_failure: list[float] = field(default_factory=list)
    conscientia_block: list[float] = field(
        default_factory=list
    )  # advice-surface augur.conscientia.violation (spec D10 block; a terminal)


@dataclass
class StallVerdict:
    degraded: bool
    reasons: list[str]


@dataclass
class HealthReport:
    started_at: float
    now: float
    faculties: dict[str, FacultyHealth]
    entered: list[tuple[str, str]]  # (faculty, reason) newly degraded/dead
    cleared: list[tuple[str, str]]  # (faculty, reason) recovered


def classify_event(subject: str) -> tuple[str, str | None]:
    """Route a bus message BEFORE faculty mapping.

    ('heartbeat', None)        for augur.system.heartbeat
    ('activity', '<faculty>')  for a faculty work subject (augur.<faculty>.<...>)
    ('ignore', None)           for augur.praefectus.* (own output) / augur.session.*
                               (cross-cutting) / other augur.system.* / non-augur.
    """
    if subject == HEARTBEAT_SUBJECT:
        return ("heartbeat", None)
    if (
        subject.startswith("augur.praefectus.")
        or subject.startswith("augur.session.")
        or subject.startswith("augur.system.")
    ):
        return ("ignore", None)
    parts = subject.split(".")
    if len(parts) >= 2 and parts[0] == "augur":
        return ("activity", parts[1])
    return ("ignore", None)


def initial_states(started_at: float) -> dict[str, FacultyHealth]:
    """Registry seeded at monitor startup: a record for every required faculty so a
    faculty that never starts is tracked from t0. Optional components appear on first
    heartbeat. (started_at is accepted for symmetry/future use.)"""
    return {f: FacultyHealth(faculty=f, required=True) for f in REQUIRED_FACULTIES}


def record_heartbeat(
    states: dict[str, FacultyHealth], faculty: str | None, ts: float
) -> None:
    """Stamp last_heartbeat/seen for a known faculty. Required faculties are pre-seeded;
    optional components register on first beat; an unrecognized id is ignored (caller
    may debug-log) to prevent unbounded registry growth."""
    if not faculty:
        return
    st = states.get(faculty)
    if st is None:
        if faculty in OPTIONAL_COMPONENTS:
            st = FacultyHealth(faculty=faculty, required=False)
            states[faculty] = st
        else:
            return
    st.seen = True
    st.last_heartbeat = ts


def _is_praesagium_state_key(payload: dict) -> bool:
    """True when a suppressed/delivery_failure event's Signature.state_key
    (advisor.py publish_suppressed_event/publish_delivery_failure_event: both
    carry ``"state_key": signature.state_key``) is the single-path praesagium
    channel (limen/gate.py build_signature: ``f"single:{domain}:{entity}"``)."""
    return str(payload.get("state_key", "")).startswith("single:praesagium:")


def record_activity(
    states: dict[str, FacultyHealth],
    window: ActivityWindow,
    subject: str,
    payload: dict,
    now: float,
    cfg,
) -> None:
    """Push the subject's signal class into the window (for the stall math) and stamp
    last_event_ts on the owning faculty if it is in the registry. Then prune the window
    to the effective stall horizon so memory stays bounded.

    Praesagium-attributable terminals (spec 2026-07-09 §7.2) are excluded from
    the consilium stall accounting: an anticipatory advice/suppressed/
    delivery-failure/advice-surface-violation is a terminal of PRAESAGIUM work
    (a foreseen forewarning), not of a nexus detection, so counting it toward
    consilium's terminals could mask a genuinely stalled real detection in the
    same window. The carrier field is pinned per publisher's real payload:
    ``domain == "praesagium"`` for advice/violation events (advisor.py
    _build_advice_event / conscientia/screens.py make_violation both carry
    ``"domain"``); ``state_key`` prefix ``"single:praesagium:"`` for
    suppressed/delivery-failure events (advisor.py publish_suppressed_event /
    publish_delivery_failure_event both carry ``"state_key"``).
    """
    if subject == "augur.nexus.detected":
        if str(payload.get("combined_severity", "")).upper() in ("MEDIUM", "HIGH"):
            window.detected_mh.append(now)
    elif subject == "augur.consilium.advice":
        if str(payload.get("domain", "")) != "praesagium":
            window.advice.append(now)
    elif subject == "augur.limen.suppressed":
        if not _is_praesagium_state_key(payload):
            window.suppressed.append(now)
    elif subject == "augur.limen.delivery_failure":
        if not _is_praesagium_state_key(payload):
            window.delivery_failure.append(now)
    elif subject == "augur.conscientia.violation":
        # A Conscientia block on the advice surface (spec D10) is deliberate and
        # terminal for that detection: consilium serviced it, Conscientia refused
        # delivery. Count it so it does not read as a stall. Other surfaces (e.g.
        # "teach") are not consilium-advice terminals and must not count here.
        # A praesagium-domain advice-surface block is a terminal of anticipatory
        # work, not of a nexus detection -- excluded for the same reason as above.
        if (
            str(payload.get("surface", "")) == "advice"
            and str(payload.get("domain", "")) != "praesagium"
        ):
            window.conscientia_block.append(now)

    _, faculty = classify_event(subject)
    if faculty and faculty in states:
        states[faculty].last_event_ts = now

    cutoff = now - cfg.effective_stall_window_s
    window.detected_mh = [t for t in window.detected_mh if t >= cutoff]
    window.advice = [t for t in window.advice if t >= cutoff]
    window.suppressed = [t for t in window.suppressed if t >= cutoff]
    window.delivery_failure = [t for t in window.delivery_failure if t >= cutoff]
    window.conscientia_block = [t for t in window.conscientia_block if t >= cutoff]


def liveness(state: FacultyHealth, now: float, started_at: float, cfg) -> str:
    """'alive'|'stale'|'dead'|'absent'|'warming_up'|'unknown' from the heartbeat clock.
    A never-seen required faculty is warming_up during the warmup grace, then unknown
    until the horizon (started_at + warmup + dead_after), then dead (never_started).
    Optional never-seen is absent (not an alert). Seen-then-late goes alive→stale→dead
    (lost)."""
    hb = state.last_heartbeat
    if hb is None:
        if not state.required:
            return "absent"
        if now <= started_at + cfg.praefectus_warmup_s:
            return "warming_up"  # cold-start grace
        if now <= started_at + cfg.praefectus_warmup_s + cfg.praefectus_dead_after_s:
            return "unknown"  # past grace, not yet confirmed dead
        return "dead"  # never_started
    age = now - hb
    if age <= cfg.praefectus_stale_after_s:
        return "alive"
    if age <= cfg.praefectus_dead_after_s:
        return "stale"
    return "dead"


def stall_signal(window: ActivityWindow, now: float, cfg) -> StallVerdict:
    """Windowed MEDIUM/HIGH nexus.detected vs {advice|suppressed|delivery_failure|
    conscientia_block} deficit + a delivery_failure spike → degraded, with reasons.
    Rate-based (not per-event) so anti-starvation coalescing within tolerance does
    not false-trigger.

    consilium_stall counts only detections aged past the servicing grace (one
    ollama_timeout — the worst-case time to turn a detection into a terminal via a
    cold-start LLM call) as the deficit numerator: a detection still within the grace
    is in-flight work consilium has not yet had time to service (a long LLM call, or a
    fresh restart), NOT a stall. An idle consilium with no inbound detections never
    trips this — there is no pending work to fail. Terminals still count over the whole
    window, so a late terminal clears earlier pending work. An advice-surface
    Conscientia block (spec D10) is also a terminal: consilium serviced the
    detection, Conscientia refused the output — that is not consilium degradation.
    """
    cutoff = now - cfg.effective_stall_window_s
    # servicing grace = one ollama_timeout (cold-start call is the worst case), always
    # < effective_stall_window_s (>=300 >= 2*timeout > timeout) so the band is non-empty.
    grace_cutoff = now - float(cfg.ollama_timeout)
    # deficit numerator = detections old enough to have produced a terminal by now;
    # fresh in-flight detections are excluded so BUSY/restart does not false-trigger.
    detected = [t for t in window.detected_mh if cutoff <= t <= grace_cutoff]
    terminals = (
        [t for t in window.advice if t >= cutoff]
        + [t for t in window.suppressed if t >= cutoff]
        + [t for t in window.delivery_failure if t >= cutoff]
        + [t for t in window.conscientia_block if t >= cutoff]
    )
    dfs = [t for t in window.delivery_failure if t >= cutoff]
    reasons: list[str] = []
    if (
        len(detected) >= cfg.praefectus_stall_min_events
        and len(terminals) < len(detected) - cfg.praefectus_stall_tolerance
    ):
        reasons.append("consilium_stall")
    if len(dfs) >= cfg.praefectus_delivery_failure_spike:
        reasons.append("delivery_failures")
    return StallVerdict(degraded=bool(reasons), reasons=reasons)


def _worse(a: str, b: str) -> str:
    return a if _SEVERITY.get(a, 0) >= _SEVERITY.get(b, 0) else b


def evaluate(
    states: dict[str, FacultyHealth],
    window: ActivityWindow,
    now: float,
    started_at: float,
    cfg,
) -> HealthReport:
    """Recompute liveness_state/activity_state/overall_state per faculty and the
    transition delta vs the previous tick (reasons stored on each FacultyHealth).
    Alert reasons are dead (never_started/lost) + degraded; stale is reported in the
    state but is intentionally NOT an alert reason (avoids heartbeat-flap noise)."""
    verdict = stall_signal(window, now, cfg)
    entered: list[tuple[str, str]] = []
    cleared: list[tuple[str, str]] = []
    for fac, st in states.items():
        st.liveness_state = liveness(st, now, started_at, cfg)
        if fac == "consilium":
            st.activity_state = "degraded" if verdict.degraded else "ok"
            activity_reasons = list(verdict.reasons) if verdict.degraded else []
        else:
            st.activity_state = "ok"
            activity_reasons = []
        st.overall_state = _worse(st.liveness_state, st.activity_state)

        new_reasons: list[str] = []
        if st.liveness_state == "dead":
            new_reasons.append("never_started" if not st.seen else "lost")
        new_reasons.extend(activity_reasons)

        prev, cur = set(st.reasons), set(new_reasons)
        entered.extend((fac, r) for r in (cur - prev))
        cleared.extend((fac, r) for r in (prev - cur))
        st.reasons = new_reasons

    # Observability-only (never alerts): reflection lag on Disciplina — a
    # responsum.complete with no disciplina.complete following within the window.
    dst = states.get("disciplina")
    rst = states.get("responsum")
    if dst is not None and rst is not None:
        r_ts = rst.last_event_ts
        d_ts = dst.last_event_ts
        lag = (
            r_ts is not None
            and (d_ts is None or d_ts < r_ts)
            and (now - r_ts) > cfg.effective_reflection_window_s
        )
        dst.flags = ["reflection_lag"] if lag else []

    return HealthReport(started_at, now, states, entered, cleared)


def summarize(report: HealthReport) -> dict:
    """The JSON the MCP tool + augur.praefectus.health payload return."""
    return {
        "started_at": report.started_at,
        "ts": report.now,
        "uptime_s": report.now - report.started_at,
        "faculties": {
            fac: {
                "liveness": st.liveness_state,
                "activity": st.activity_state,
                "overall": st.overall_state,
                "reasons": list(st.reasons),
                "flags": list(st.flags),
                "required": st.required,
                "last_heartbeat": st.last_heartbeat,
                "last_event_ts": st.last_event_ts,
            }
            for fac, st in report.faculties.items()
        },
    }
