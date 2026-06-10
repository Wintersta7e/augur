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
    to the effective stall horizon so memory stays bounded."""
    if subject == "augur.nexus.detected":
        if str(payload.get("combined_severity", "")).upper() in ("MEDIUM", "HIGH"):
            window.detected_mh.append(now)
    elif subject == "augur.consilium.advice":
        window.advice.append(now)
    elif subject == "augur.limen.suppressed":
        window.suppressed.append(now)
    elif subject == "augur.limen.delivery_failure":
        window.delivery_failure.append(now)

    _, faculty = classify_event(subject)
    if faculty and faculty in states:
        states[faculty].last_event_ts = now

    cutoff = now - cfg.effective_stall_window_s
    window.detected_mh = [t for t in window.detected_mh if t >= cutoff]
    window.advice = [t for t in window.advice if t >= cutoff]
    window.suppressed = [t for t in window.suppressed if t >= cutoff]
    window.delivery_failure = [t for t in window.delivery_failure if t >= cutoff]
