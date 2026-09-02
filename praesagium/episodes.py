"""Praesagium's pure episode substrate: canonical keys and compact episode
entries built from vigil anomaly payloads.

No Redis/NATS imports here -- the matcher (praesagium/matcher.py, built in a
later task) owns persistence via PersistenceManager. See
docs/superpowers/specs/2026-07-09-praesagium-design.md Sec 3.1-3.2.
"""

from __future__ import annotations

from datetime import datetime

from tabula.contracts import is_sentinel_entity


def canonical_key(anomaly: dict) -> str | None:
    """Session-invariant identity of an anomaly stream: '{domain}:{entity}'.

    Returns None (record nothing) when domain is missing/falsy or entity is
    missing/'?'/'' -- mirrors Limen's ungateable rule (gate.py:253) -- or when
    the entity is a daemon sentinel.

    The sentinel case is not cosmetic. `<no_foreground>` precedes nearly every
    app switch, so as an antecedent it has high support and genuine lift for
    "some app gains focus". It would clear the Wilson lower bound and the
    session-conditional null and be promoted as a real pattern, because the
    promotion math is built to reject coincidence, not to reject a placeholder
    that is definitionally present. Cheaper to exclude at the source.
    """
    domain = anomaly.get("domain")
    entity = anomaly.get("entity")
    if not domain or entity in (None, "?", "") or is_sentinel_entity(entity):
        return None
    return f"{domain}:{entity}"


def parse_epoch(iso: str | None) -> float | None:
    """ISO-8601 string (as emitted by vigil) -> epoch seconds.

    Aware-only contract: vigil always emits timezone-aware UTC timestamps;
    timezone-naive strings are treated as unparseable rather than silently
    interpreted in the host's local timezone. None on anything unparseable
    -- missing, garbage, naive, or not a string at all. Never raises.
    """
    if not isinstance(iso, str):
        return None
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return None
    return dt.timestamp()


def build_episode(anomaly: dict) -> dict | None:
    """Compact episode entry ({'k', 's', 't'}) from a vigil anomaly payload.

    None when the anomaly is unkeyable (see canonical_key) or its timestamp
    doesn't parse (see parse_epoch) -- spec Sec 3.2.
    """
    key = canonical_key(anomaly)
    if key is None:
        return None
    epoch = parse_epoch(anomaly.get("timestamp"))
    if epoch is None:
        return None
    severity = anomaly.get("severity") or "low"
    return {"k": key, "s": str(severity).lower(), "t": epoch}
