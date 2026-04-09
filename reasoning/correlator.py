"""Cross-domain correlator.

Subscribes to NATS 'augur.detection.anomaly', maintains a Redis-backed
sliding window + in-memory NetworkX DiGraph of correlations, and emits
enriched events on 'augur.correlation.detected' to the advisor.

Two LOW-severity anomalies from different domains within 30 seconds
escalate to MEDIUM via a Redis-stored matrix, producing reasoning that
neither signal would trigger individually.
"""

from __future__ import annotations

import asyncio  # noqa: F401 — used in run loop (Task 8)
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import nats  # noqa: F401 — used in NATS subscriber (Task 7)
import networkx as nx
import redis

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blackboard.config import AugurConfig  # noqa: F401 — used in run loop (Task 8)
from blackboard.persistence import PersistenceManager  # noqa: F401 — used in matrix load (Task 5)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("correlator")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUBSCRIBE_ANOMALY = "augur.detection.anomaly"
SUBSCRIBE_DEBUG_DUMP = "augur.debug.graph_dump"
PUBLISH_CORRELATION = "augur.correlation.detected"

REDIS_KEY_WINDOW = "augur:correlation:window"
REDIS_KEY_MATRIX = "augur:config:escalation_matrix"

CORRELATION_WINDOW_S = 30  # query window (seconds back from primary)
PRUNE_WINDOW_S = 2 * CORRELATION_WINDOW_S  # derived: 60s buffer for clock skew

SEVERITY_ORDER: dict[str, int] = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

DEFAULT_ESCALATION_MATRIX: dict = {
    "version": "1.0",
    "rules": {
        "LOW+LOW": "MEDIUM",
        "LOW+MEDIUM": "MEDIUM",
        "LOW+HIGH": "HIGH",
        "MEDIUM+MEDIUM": "HIGH",
        "MEDIUM+HIGH": "HIGH",
        "HIGH+HIGH": "HIGH",
    },
}

SEVERITY_GATE_PASSTHROUGH = {"MEDIUM", "HIGH"}  # forwarded even with no correlation


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def normalize_rule_key(sev1: str, sev2: str) -> str | None:
    """Sort a severity pair by rank (LOW<MEDIUM<HIGH) and return 'A+B'.

    Uppercases inputs first because the detector emits lowercase severity.
    Returns None for unknown severity levels — caller must fall back to
    the higher of the two original severities.
    """
    s1, s2 = sev1.upper(), sev2.upper()
    if s1 not in SEVERITY_ORDER or s2 not in SEVERITY_ORDER:
        return None
    pair = sorted([s1, s2], key=lambda s: SEVERITY_ORDER[s])
    return f"{pair[0]}+{pair[1]}"


def lookup_escalation(
    sev1: str,
    sev2: str,
    matrix: dict,
) -> tuple[str, str | None]:
    """Look up a severity pair in the escalation matrix.

    Returns ``(combined_severity, rule_label)`` where:
    - ``combined_severity`` is the escalated severity (uppercase).
    - ``rule_label`` is ``"LOW+LOW→MEDIUM"`` on matrix hit, ``None`` on miss.

    On matrix miss, falls back to ``max(sev1, sev2)`` by rank order. If
    neither severity is recognized at all, returns the first input
    uppercased with ``rule_label=None`` — the caller should have dropped
    this event before reaching lookup, so this path is a safety net only.
    """
    key = normalize_rule_key(sev1, sev2)
    rules = matrix.get("rules", {})

    if key is not None and key in rules:
        combined = rules[key]
        return combined, f"{key}→{combined}"

    # Fallback: take the higher of the two by rank
    s1, s2 = sev1.upper(), sev2.upper()
    r1 = SEVERITY_ORDER.get(s1, -1)
    r2 = SEVERITY_ORDER.get(s2, -1)
    if r1 < 0 and r2 < 0:
        return s1, None
    combined = s1 if r1 >= r2 else s2
    return combined, None


# ---------------------------------------------------------------------------
# Redis window helpers
# ---------------------------------------------------------------------------


def parse_timestamp(iso_ts: str) -> float:
    """Parse ISO-8601 timestamp to unix epoch seconds (float)."""
    return datetime.fromisoformat(iso_ts).timestamp()


def _decode_member(member: bytes | str) -> dict:
    """Decode a Redis sorted-set member (bytes or str) to a dict."""
    if isinstance(member, bytes):
        member = member.decode()
    return json.loads(member)


def add_to_window(r: redis.Redis, anomaly: dict) -> None:
    """Add an anomaly to the correlation window and prune old entries.

    Member: JSON-serialized anomaly dict.
    Score: unix timestamp parsed from ``anomaly['timestamp']``.
    Prune: removes entries with score <= ``now - PRUNE_WINDOW_S``.
    """
    score = parse_timestamp(anomaly["timestamp"])
    member = json.dumps(anomaly)
    r.zadd(REDIS_KEY_WINDOW, {member: score})
    # Score-based prune: remove everything older than the prune boundary.
    r.zremrangebyscore(REDIS_KEY_WINDOW, "-inf", score - PRUNE_WINDOW_S)


def query_window(r: redis.Redis, primary: dict) -> list[dict]:
    """Return anomalies from other domains within the last CORRELATION_WINDOW_S.

    Filters out same-domain events (correlation is cross-domain by definition).
    """
    now = parse_timestamp(primary["timestamp"])
    start = now - CORRELATION_WINDOW_S
    primary_domain = primary["domain"]

    raw_members = r.zrangebyscore(REDIS_KEY_WINDOW, start, now)
    results: list[dict] = []
    for m in raw_members:
        try:
            event = _decode_member(m)
        except (json.JSONDecodeError, UnicodeDecodeError):
            log.warning("Skipping unparseable window member")
            continue
        if event.get("domain") != primary_domain:
            results.append(event)
    return results


def _pick_most_recent_per_domain(events: list[dict]) -> list[dict]:
    """Group by domain, keep only the most recent event per domain."""
    by_domain: dict[str, dict] = {}
    for ev in events:
        d = ev["domain"]
        if d not in by_domain or parse_timestamp(ev["timestamp"]) > parse_timestamp(
            by_domain[d]["timestamp"]
        ):
            by_domain[d] = ev
    return list(by_domain.values())


def _highest_severity(events: list[dict]) -> dict:
    """Return the event with the highest severity by rank."""
    return max(
        events,
        key=lambda e: SEVERITY_ORDER.get(e["severity"].upper(), -1),
    )


def _build_correlation_payload(
    primary: dict,
    correlated: list[dict],
    matrix: dict,
) -> dict:
    """Assemble the correlated-event payload published on augur.correlation.detected."""
    # Pairwise escalation uses the HIGHEST-severity correlated event
    driver = _highest_severity(correlated)
    combined_severity, rule_label = lookup_escalation(
        primary["severity"], driver["severity"], matrix
    )

    # Temporal lag = primary - closest correlated event
    primary_ts = parse_timestamp(primary["timestamp"])
    closest = min(
        correlated,
        key=lambda e: abs(primary_ts - parse_timestamp(e["timestamp"])),
    )
    lag = primary_ts - parse_timestamp(closest["timestamp"])

    return {
        "primary_anomaly": primary,
        "correlated_events": correlated,
        "correlation_found": True,
        "temporal_lag_seconds": round(lag, 3),
        "combined_severity": combined_severity,
        "severity_escalated": combined_severity != primary["severity"].upper(),
        "escalation_rule": rule_label,
        "escalation_matrix_version": matrix.get("version") if rule_label else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _build_passthrough_payload(primary: dict) -> dict:
    """Assemble the pass-through payload for standalone medium/high events."""
    return {
        "primary_anomaly": primary,
        "correlated_events": [],
        "correlation_found": False,
        "temporal_lag_seconds": None,
        "combined_severity": primary["severity"].upper(),
        "severity_escalated": False,
        "escalation_rule": None,
        "escalation_matrix_version": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def correlate(
    primary: dict,
    r: redis.Redis,
    matrix: dict,
) -> dict | None:
    """Run the full correlation pipeline for one anomaly.

    Returns:
      - correlated payload dict  (correlation found across domains)
      - passthrough payload dict (standalone medium/high — forward as-is)
      - None                     (standalone low — drop, advisor ignores)
    """
    # 1. Add primary to window, prune old entries
    add_to_window(r, primary)

    # 2. Query for other-domain events in the last 30s
    other_domain_events = query_window(r, primary)

    # 3. Collapse to one event per domain (most recent)
    grouped = _pick_most_recent_per_domain(other_domain_events)

    if grouped:
        return _build_correlation_payload(primary, grouped, matrix)

    # No correlation — gate by severity
    severity = primary.get("severity", "low").upper()
    if severity in SEVERITY_GATE_PASSTHROUGH:
        return _build_passthrough_payload(primary)

    return None  # drop low


# ---------------------------------------------------------------------------
# Session graph (in-memory NetworkX DiGraph — not persisted in this phase)
# ---------------------------------------------------------------------------


def new_session_graph() -> nx.DiGraph:
    """Return a fresh empty DiGraph for a new session."""
    return nx.DiGraph()


def node_key(anomaly: dict) -> str:
    """Unique node key: ``{domain}:{entity}:{timestamp}``."""
    return f"{anomaly['domain']}:{anomaly['entity']}:{anomaly['timestamp']}"


def _add_anomaly_node(graph: nx.DiGraph, anomaly: dict) -> str:
    """Add an anomaly as a node; return the node key."""
    key = node_key(anomaly)
    graph.add_node(
        key,
        domain=anomaly["domain"],
        entity=anomaly["entity"],
        severity=anomaly["severity"],
        timestamp=anomaly["timestamp"],
    )
    return key


def add_correlation_to_graph(
    graph: nx.DiGraph,
    primary: dict,
    correlated: list[dict],
    combined_severity: str,
    rule_label: str | None,
) -> None:
    """Add nodes for primary + correlated events and directed edges primary→correlated.

    Each edge carries ``temporal_lag``, ``escalation_rule``, ``combined_severity``,
    and a ``domains`` tuple describing the cross-domain pair. ``temporal_lag`` is
    always positive (primary timestamp minus correlated timestamp).
    """
    pk = _add_anomaly_node(graph, primary)
    primary_ts = parse_timestamp(primary["timestamp"])

    for ev in correlated:
        ck = _add_anomaly_node(graph, ev)
        lag = primary_ts - parse_timestamp(ev["timestamp"])
        graph.add_edge(
            pk,
            ck,
            temporal_lag=round(lag, 3),
            escalation_rule=rule_label,
            combined_severity=combined_severity,
            domains=(primary["domain"], ev["domain"]),
        )


def dump_graph(graph: nx.DiGraph) -> None:
    """Log all nodes and edges of the session graph at INFO level."""
    log.info("=== Session graph dump ===")
    log.info("Nodes (%d):", len(graph.nodes))
    for k, data in graph.nodes(data=True):
        log.info(
            "  %s  severity=%s  domain=%s",
            k,
            data.get("severity"),
            data.get("domain"),
        )
    log.info("Edges (%d):", len(graph.edges))
    for u, v, data in graph.edges(data=True):
        log.info(
            "  %s → %s  lag=%.1fs  rule=%s  severity=%s",
            u,
            v,
            data.get("temporal_lag", 0),
            data.get("escalation_rule"),
            data.get("combined_severity"),
        )
    log.info("=== End graph dump ===")
