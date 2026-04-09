"""Cross-domain correlator.

Subscribes to NATS 'augur.detection.anomaly', maintains a Redis-backed
sliding window + in-memory NetworkX DiGraph of correlations, and emits
enriched events on 'augur.correlation.detected' to the advisor.

Two LOW-severity anomalies from different domains within 30 seconds
escalate to MEDIUM via a Redis-stored matrix, producing reasoning that
neither signal would trigger individually.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import nats
import nats.aio.client
import networkx as nx
from networkx.readwrite import json_graph
import redis

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blackboard.config import AugurConfig
from blackboard.persistence import PersistenceManager

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
SUBSCRIBE_SESSION_END = "augur.session.end"
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


def flush_graph_to_redis(
    graph: nx.DiGraph,
    pm: PersistenceManager,
    session_id: str,
) -> None:
    """Serialize the session DiGraph via node_link_data and persist it.

    Saves an empty-graph placeholder for sessions with no correlations so
    consumers can distinguish "session had no correlations" from "session
    has never existed". The tuple-valued ``domains`` edge attribute is
    converted to a list by the JSON round-trip inside PersistenceManager.
    """
    graph_data = json_graph.node_link_data(graph)
    pm.save_correlation_graph(session_id, graph_data)
    log.info(
        "Flushed correlation graph for session %s (%d nodes, %d edges)",
        session_id,
        len(graph.nodes),
        len(graph.edges),
    )


# ---------------------------------------------------------------------------
# Redis helper
# ---------------------------------------------------------------------------


def connect_redis(config: AugurConfig) -> redis.Redis:
    client = redis.Redis(
        host=config.redis_host,
        port=config.redis_port,
        socket_connect_timeout=config.redis_connect_timeout,
    )
    client.ping()
    log.info("Redis connected")
    return client


def ensure_matrix_seeded(pm: PersistenceManager) -> dict:
    """Write the default escalation matrix if absent; return the active matrix."""
    existing = pm.load_escalation_matrix()
    if existing is None:
        pm.save_escalation_matrix(DEFAULT_ESCALATION_MATRIX)
        log.info("Seeded default escalation matrix (version=1.0)")
        return DEFAULT_ESCALATION_MATRIX
    log.info(
        "Loaded existing escalation matrix (version=%s)",
        existing.get("version", "unknown"),
    )
    return existing


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------


async def run() -> None:
    config = AugurConfig.from_env()

    redis_client = connect_redis(config)
    pm = PersistenceManager(redis_client)

    ensure_matrix_seeded(pm)  # write default if absent

    nc = await nats.connect(
        config.nats_url, connect_timeout=config.nats_connect_timeout
    )
    log.info("NATS connected (%s)", config.nats_url)

    session_graph = new_session_graph()

    async def on_anomaly(msg: nats.aio.client.Msg) -> None:
        try:
            anomaly = json.loads(msg.data.decode())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            log.warning("Bad anomaly payload: %s", exc)
            return

        # Load matrix fresh on every event so reflection-engine tuning takes
        # effect without restarting the correlator.
        matrix = pm.load_escalation_matrix() or DEFAULT_ESCALATION_MATRIX

        try:
            payload = correlate(anomaly, redis_client, matrix)
        except Exception as exc:
            log.error("correlate() failed: %s", exc, exc_info=True)
            return

        if payload is None:
            log.debug(
                "Dropped standalone low %s/%s",
                anomaly.get("domain"),
                anomaly.get("entity"),
            )
            return

        # Accumulate into session graph (only when correlation actually found)
        if payload["correlation_found"]:
            add_correlation_to_graph(
                session_graph,
                primary=payload["primary_anomaly"],
                correlated=payload["correlated_events"],
                combined_severity=payload["combined_severity"],
                rule_label=payload["escalation_rule"],
            )
            log.warning(
                "  \u2605 CORRELATION [%s] %s + %s  (rule=%s, lag=%.1fs)",
                payload["combined_severity"],
                payload["primary_anomaly"]["domain"],
                ", ".join(e["domain"] for e in payload["correlated_events"]),
                payload["escalation_rule"],
                payload["temporal_lag_seconds"],
            )
        else:
            log.info(
                "  Pass-through [%s] %s/%s",
                payload["combined_severity"],
                payload["primary_anomaly"]["domain"],
                payload["primary_anomaly"]["entity"],
            )

        try:
            await nc.publish(PUBLISH_CORRELATION, json.dumps(payload).encode())
            log.info("Published correlation to %s", PUBLISH_CORRELATION)
        except Exception as exc:
            log.error("NATS publish failed: %s", exc)

    async def on_debug_dump(_msg: nats.aio.client.Msg) -> None:
        dump_graph(session_graph)

    async def on_session_end(msg: nats.aio.client.Msg) -> None:
        nonlocal session_graph
        try:
            payload = json.loads(msg.data.decode())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            log.warning("Bad session.end payload: %s", exc)
            return
        session_id = payload.get("session_id")
        if not session_id:
            log.warning("session.end payload missing session_id: %s", payload)
            return
        try:
            flush_graph_to_redis(session_graph, pm, session_id)
        except Exception as exc:
            log.error("Failed to flush correlation graph: %s", exc, exc_info=True)
            return
        # Reset to a fresh empty DiGraph for the next session
        session_graph = new_session_graph()
        log.info("Session graph reset after flush (session_id=%s)", session_id)

    await nc.subscribe(SUBSCRIBE_ANOMALY, cb=on_anomaly)
    await nc.subscribe(SUBSCRIBE_DEBUG_DUMP, cb=on_debug_dump)
    await nc.subscribe(SUBSCRIBE_SESSION_END, cb=on_session_end)

    log.info("Subscribed to %s", SUBSCRIBE_ANOMALY)
    log.info("Subscribed to %s (debug graph dump)", SUBSCRIBE_DEBUG_DUMP)
    log.info("Subscribed to %s (session end graph flush)", SUBSCRIBE_SESSION_END)
    log.info(
        "Window: %ds query / %ds prune buffer",
        CORRELATION_WINDOW_S,
        PRUNE_WINDOW_S,
    )
    log.info("Waiting for anomalies...")

    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await nc.close()
        log.info("Shut down cleanly")


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log.info("Interrupted")


if __name__ == "__main__":
    main()
