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
from blackboard.connections import connect_redis
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

# Safety valve for an in-memory session graph that never receives a
# session.end message (e.g., publisher crash, network partition). Once a
# session accumulates this many nodes, the correlator flushes the graph
# under a synthetic "orphaned-<unix_ts>" session id and resets in-memory
# state. Prevents the DiGraph from growing unbounded (LEAK-09).
MAX_SESSION_GRAPH_NODES = 10_000

SEVERITY_ORDER: dict[str, int] = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

DEFAULT_ESCALATION_MATRIX: dict = {
    "version": "1.0",
    "rules": {
        # Pairwise (existing)
        "LOW+LOW": "MEDIUM",
        "LOW+MEDIUM": "MEDIUM",
        "LOW+HIGH": "HIGH",
        "MEDIUM+MEDIUM": "HIGH",
        "MEDIUM+HIGH": "HIGH",
        "HIGH+HIGH": "HIGH",
        # 3-way (NEW)
        "LOW+LOW+LOW": "MEDIUM",
        "LOW+LOW+MEDIUM": "MEDIUM",
        "LOW+LOW+HIGH": "HIGH",
        "LOW+MEDIUM+MEDIUM": "HIGH",
        "LOW+MEDIUM+HIGH": "HIGH",
        "LOW+HIGH+HIGH": "HIGH",
        "MEDIUM+MEDIUM+MEDIUM": "HIGH",
        "MEDIUM+MEDIUM+HIGH": "HIGH",
        "MEDIUM+HIGH+HIGH": "HIGH",
        "HIGH+HIGH+HIGH": "HIGH",
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


def normalize_rule_key_n_way(severities: list[str]) -> str | None:
    """Sort N severities by rank (LOW<MEDIUM<HIGH) and join with '+'.

    Uppercases inputs first because the detector emits lowercase.
    Returns None for empty input or any unknown severity — caller must
    fall back to max-severity behaviour.
    """
    if not severities:
        return None
    upper = [s.upper() for s in severities]
    if any(s not in SEVERITY_ORDER for s in upper):
        return None
    sorted_severities = sorted(upper, key=lambda s: SEVERITY_ORDER[s])
    return "+".join(sorted_severities)


def lookup_escalation_n_way(
    severities: list[str],
    matrix: dict,
) -> tuple[str, str | None]:
    """Look up an N-severity tuple in the escalation matrix.

    Returns ``(combined_severity, rule_label)`` where:
    - ``combined_severity`` is the escalated severity (uppercase).
    - ``rule_label`` is e.g. ``"LOW+LOW+LOW→MEDIUM"`` on hit, ``None`` on miss.

    Defensive: callers normally pass len >= 2 (correlation found path).
    For len == 1 returns ``(severities[0].upper(), None)`` without matrix lookup.
    For empty list returns ``("LOW", None)``.

    On miss, falls back to ``max(severities)`` by rank order.
    """
    if not severities:
        return "LOW", None
    upper = [s.upper() for s in severities]
    if len(upper) == 1:
        return upper[0], None

    key = normalize_rule_key_n_way(upper)
    rules = matrix.get("rules", {})
    if key is not None and key in rules:
        combined = rules[key]
        return combined, f"{key}→{combined}"

    # Fallback: max severity by rank
    valid = [s for s in upper if s in SEVERITY_ORDER]
    if not valid:
        return upper[0], None
    return max(valid, key=lambda s: SEVERITY_ORDER[s]), None


def get_rule_window(
    rule_key: str | None,
    matrix: dict,
    default_s: float,
) -> float:
    """Resolve per-rule window from matrix.rule_windows, fall back to default."""
    if rule_key is None:
        return default_s
    rule_windows = matrix.get("rule_windows", {})
    return rule_windows.get(rule_key, default_s)


def compute_query_window(matrix: dict, default_s: float) -> float:
    """Return max(default, max(rule_windows.values())). Used to size the candidate pool."""
    rule_windows = matrix.get("rule_windows", {})
    if not rule_windows:
        return default_s
    return max(default_s, max(rule_windows.values()))


def compute_prune_window(query_window_s: float) -> float:
    """Return 2 * query_window for clock-skew buffer."""
    return 2.0 * query_window_s


def filter_by_pairwise_window(
    primary: dict,
    candidates: list[dict],
    matrix: dict,
    default_window_s: float,
) -> list[dict]:
    """Drop candidates whose lag exceeds their pairwise rule's window.

    For each candidate, compute the pairwise rule_key (primary↔candidate),
    look up its window, and keep the candidate only if
    (primary_ts - candidate_ts) <= rule_window. Unknown severities fall
    back to default_window_s.
    """
    primary_ts = parse_timestamp(primary["timestamp"])
    primary_sev = primary.get("severity", "")
    survivors: list[dict] = []
    for cand in candidates:
        cand_ts = parse_timestamp(cand["timestamp"])
        lag = primary_ts - cand_ts
        rule_key = normalize_rule_key_n_way([primary_sev, cand.get("severity", "")])
        window = get_rule_window(rule_key, matrix, default_window_s)
        if lag <= window:
            survivors.append(cand)
    return survivors


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

    # Structural attribution: rule_key is derived from severities directly so
    # the reflection engine can tune regardless of matrix-miss.
    rule_key = normalize_rule_key(primary["severity"], driver["severity"])

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
        "rule_key": rule_key,
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
        "rule_key": None,
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


# Redis connection comes from blackboard.connections.connect_redis
# (imported above) — previously duplicated inline here.


def ensure_matrix_seeded(pm: PersistenceManager) -> dict:
    """Write the default matrix if absent; additively merge missing default
    rules into an existing matrix without overwriting operator changes.

    Operator deletion of a default rule is non-durable: it will be re-seeded
    on next startup. The intended operator model is to set the rule's target
    to LOW (which short-circuits escalation) for durable disable.

    rule_windows in an existing matrix are preserved untouched.
    """
    existing = pm.load_escalation_matrix()
    if existing is None:
        pm.save_escalation_matrix(DEFAULT_ESCALATION_MATRIX)
        log.info("Seeded default escalation matrix (version=1.0)")
        return DEFAULT_ESCALATION_MATRIX

    existing_rules = existing.get("rules", {})
    default_rules = DEFAULT_ESCALATION_MATRIX["rules"]

    merged_rules = dict(existing_rules)
    added: list[str] = []
    for k, v in default_rules.items():
        if k not in merged_rules:
            merged_rules[k] = v
            added.append(k)

    if not added:
        log.info(
            "Loaded existing escalation matrix (version=%s, no defaults missing)",
            existing.get("version", "unknown"),
        )
        return existing

    merged = {
        "version": existing.get("version", "1.0"),
        "rules": merged_rules,
    }
    if "rule_windows" in existing:
        merged["rule_windows"] = existing["rule_windows"]

    pm.save_escalation_matrix(merged)
    log.info(
        "Seeded %d missing default rules into existing matrix: %s",
        len(added),
        ", ".join(added),
    )
    return merged


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
    # BUG-05: dedup marker so a duplicate session.end for the same id
    # cannot flush a freshly-emptied graph over the real persisted one.
    last_flushed_session_id: str | None = None

    async def on_anomaly(msg: nats.aio.client.Msg) -> None:
        nonlocal session_graph
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
            # ARCH-12: correlation events are normal operation, not warnings.
            # Use INFO with a ★ prefix for visibility without conflating
            # the log level with genuine warnings/errors.
            log.info(
                "  \u2605 CORRELATION [%s] %s + %s  (rule=%s, lag=%.1fs)",
                payload["combined_severity"],
                payload["primary_anomaly"]["domain"],
                ", ".join(e["domain"] for e in payload["correlated_events"]),
                payload["escalation_rule"],
                payload["temporal_lag_seconds"],
            )

            # LEAK-09: safety valve. If session.end never arrives, the
            # in-memory graph grows unbounded. Once nodes exceed the cap,
            # flush under a synthetic session id and reset.
            if len(session_graph.nodes) >= MAX_SESSION_GRAPH_NODES:
                synthetic_id = f"orphaned-{int(datetime.now(timezone.utc).timestamp())}"
                log.warning(
                    "Session graph exceeded %d nodes without session.end; "
                    "auto-flushing under %s to prevent unbounded growth",
                    MAX_SESSION_GRAPH_NODES,
                    synthetic_id,
                )
                try:
                    flush_graph_to_redis(session_graph, pm, synthetic_id)
                except Exception as exc:
                    log.error("Safety-valve flush failed: %s", exc, exc_info=True)
                session_graph = new_session_graph()
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
        nonlocal session_graph, last_flushed_session_id
        try:
            payload = json.loads(msg.data.decode())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            log.warning("Bad session.end payload: %s", exc)
            # BUG-02: reset the graph even on parse failure so a malformed
            # session.end cannot contaminate future sessions with carried-
            # over correlations from the current session.
            session_graph = new_session_graph()
            return

        session_id = payload.get("session_id")
        if not session_id:
            log.warning("session.end payload missing session_id: %s", payload)
            # BUG-02: same reset for missing session_id.
            session_graph = new_session_graph()
            return

        # BUG-05: skip re-flush when the same session.end arrives twice
        # (duplicate publish, retry, replay). Without this guard, the second
        # flush would overwrite the real persisted graph with an empty one.
        if session_id == last_flushed_session_id:
            log.warning(
                "Duplicate session.end for %s — skipping re-flush",
                session_id,
            )
            return

        try:
            flush_graph_to_redis(session_graph, pm, session_id)
            last_flushed_session_id = session_id
            log.info("Session graph flushed (session_id=%s)", session_id)
        except Exception as exc:
            log.error(
                "Failed to flush correlation graph for %s: %s",
                session_id,
                exc,
                exc_info=True,
            )
            # last_flushed_session_id is intentionally NOT set on failure.
            # R2-BUG-03: a retry of the same session.end will hit the
            # finally-reset below with an empty graph, so the correlation
            # data from this session is not recoverable from in-memory
            # state. A retry will flush an empty graph and overwrite any
            # partial data that made it to Redis during the failed first
            # attempt. This is an acceptable trade-off: leaving the full
            # graph in memory (the alternative) would contaminate all
            # subsequent sessions with stale correlations — BUG-01. Losing
            # one session's correlations is better than cross-session
            # state corruption.
        finally:
            # BUG-01: always reset the graph, even on flush failure. Leaving
            # the failed session's state in memory would contaminate every
            # subsequent session with stale correlations until process restart.
            session_graph = new_session_graph()

    # LEAK-05: save subscription handles so unsubscribe() is called on
    # shutdown rather than relying on nc.close() to tear them down abruptly.
    sub_anomaly = await nc.subscribe(SUBSCRIBE_ANOMALY, cb=on_anomaly)
    sub_debug = await nc.subscribe(SUBSCRIBE_DEBUG_DUMP, cb=on_debug_dump)
    sub_session = await nc.subscribe(SUBSCRIBE_SESSION_END, cb=on_session_end)

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
        try:
            await sub_anomaly.unsubscribe()
            await sub_debug.unsubscribe()
            await sub_session.unsubscribe()
        except Exception as exc:
            log.debug("Unsubscribe failed during shutdown: %s", exc)
        await nc.close()
        log.info("Shut down cleanly")


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log.info("Interrupted")


if __name__ == "__main__":
    main()
