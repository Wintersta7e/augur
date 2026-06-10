"""Augur MCP server — tools for pipeline lifecycle, event injection,
state inspection, and control. All Redis I/O is routed through
PersistenceManager via context managers that close sockets on exit.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import signal
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import nats as nats_client
import redis
from fastmcp import FastMCP

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tabula.config import AugurConfig  # noqa: E402
from tabula.contracts import PerceptionEvent  # noqa: E402
from tabula.persistence import PersistenceManager  # noqa: E402

log = logging.getLogger("augur.mcp")

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_config = AugurConfig.from_env()

_processes: dict[str, dict[str, Any]] = {}

COMPONENT_COMMANDS: dict[str, list[str]] = {
    "detector": [sys.executable, "-m", "detection.anomaly_detector"],
    "correlator": [sys.executable, "-m", "reasoning.correlator"],
    "advisor": [sys.executable, "-m", "reasoning.augur_advisor"],
    "feedback": [sys.executable, "-m", "perception.feedback_collector"],
    "reflection": [sys.executable, "-m", "reasoning.reflection_engine"],
    "vox": [sys.executable, "-m", "vox.console_display"],
}

# SEC-02: allowlist for domain / entity / stream_id values received through
# MCP tool arguments. Unsanitized values would land in NATS subjects
# (f"augur.perception.{domain}") and Redis keys (f"augur:profile:{domain}:{entity}"),
# where a ":" or "." or wildcard character can break downstream parsing or
# reach unintended keyspace. Keep it narrow.
_SAFE_LABEL_RE = re.compile(r"^[a-z0-9_]{1,64}$")


def _validate_label(value: str, field: str) -> str | None:
    """Return an error message if value fails the safe-label check, else None."""
    if not isinstance(value, str):
        return f"{field} must be a string, got {type(value).__name__}"
    if not _SAFE_LABEL_RE.match(value):
        return (
            f"{field} must match {_SAFE_LABEL_RE.pattern} "
            f"(lowercase letters, digits, underscore; 1-64 chars)"
        )
    return None


# ---------------------------------------------------------------------------
# Redis + PersistenceManager context managers
# ---------------------------------------------------------------------------
#
# LEAK-01 / LEAK-02: every MCP tool that needs a Redis client must close
# it on exit, otherwise each call orphans a socket on the underlying
# connection pool. The context managers below make the correct pattern
# concise enough that every tool can opt into it with a single `with`.
#
# ARCH-07: `decode_responses=True` was previously set on these clients
# while PersistenceManager uses bytes internally. Mismatching decode
# modes made it impossible to share a client between MCP tools and PM
# without surprises, so the decode flag is now off and json.loads
# handles both str and bytes uniformly.


def _new_redis() -> redis.Redis:
    return redis.Redis(
        host=_config.redis_host,
        port=_config.redis_port,
        socket_connect_timeout=_config.redis_connect_timeout,
    )


@contextmanager
def _redis_ctx() -> Iterator[redis.Redis]:
    """Context-managed Redis client that always closes on exit."""
    client = _new_redis()
    try:
        yield client
    finally:
        try:
            client.close()
        except Exception as exc:
            log.debug("Redis client close failed: %s", exc)


@contextmanager
def _persistence_ctx() -> Iterator[PersistenceManager]:
    """Context-managed PersistenceManager that closes the Redis client."""
    with _redis_ctx() as client:
        yield PersistenceManager(client)


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP("augur", instructions="Augur pipeline control.")

# ===========================================================================
# Lifecycle tools
# ===========================================================================


@mcp.tool()
async def start_pipeline(components: list[str] | None = None) -> dict[str, Any]:
    """Launch Augur components as subprocesses.

    Args:
        components: Names to start. Defaults to all five components
            (detector, advisor, feedback, reflection, display).

    Returns:
        Per-component status dict with 'status', 'pid', and 'error' keys.
    """
    targets = components if components is not None else list(COMPONENT_COMMANDS.keys())
    results: dict[str, Any] = {}

    for name in targets:
        if name not in COMPONENT_COMMANDS:
            results[name] = {"status": "error", "error": f"Unknown component: {name}"}
            continue

        # Already running?
        existing = _processes.get(name)
        if existing is not None:
            proc: asyncio.subprocess.Process = existing["proc"]
            if proc.returncode is None:
                results[name] = {
                    "status": "already_running",
                    "pid": proc.pid,
                }
                continue
            # LEAK-11: dead entry — pop it so a stale Process object does
            # not linger in the dict after we replace it below.
            _processes.pop(name, None)

        cmd = COMPONENT_COMMANDS[name]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(PROJECT_ROOT),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            _processes[name] = {
                "proc": proc,
                "started_at": time.time(),
                "cmd": cmd,
            }
            results[name] = {"status": "started", "pid": proc.pid}
        except Exception as exc:
            results[name] = {"status": "error", "error": str(exc)}

    return results


@mcp.tool()
async def stop_pipeline(components: list[str] | None = None) -> dict[str, Any]:
    """Stop running Augur components.

    Args:
        components: Names to stop. Defaults to all currently tracked processes.

    Returns:
        Per-component status dict.
    """
    targets = components if components is not None else list(_processes.keys())
    results: dict[str, Any] = {}

    for name in targets:
        entry = _processes.get(name)
        if entry is None:
            results[name] = {"status": "not_tracked"}
            continue

        proc: asyncio.subprocess.Process = entry["proc"]
        if proc.returncode is not None:
            results[name] = {"status": "already_exited", "returncode": proc.returncode}
            _processes.pop(name, None)
            continue

        try:
            proc.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
                results[name] = {"status": "stopped", "returncode": proc.returncode}
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                results[name] = {"status": "killed", "returncode": proc.returncode}
        except Exception as exc:
            results[name] = {"status": "error", "error": str(exc)}
        finally:
            _processes.pop(name, None)

    return results


@mcp.tool()
async def pipeline_status() -> dict[str, Any]:
    """Return running/exited status, PIDs, and uptime for all tracked processes.

    Returns:
        Dict mapping component name to status info.
    """
    now = time.time()
    result: dict[str, Any] = {}

    for name, entry in _processes.items():
        proc: asyncio.subprocess.Process = entry["proc"]
        uptime = now - entry["started_at"]
        if proc.returncode is None:
            result[name] = {
                "status": "running",
                "pid": proc.pid,
                "uptime_seconds": round(uptime, 1),
            }
        else:
            result[name] = {
                "status": "exited",
                "pid": proc.pid,
                "returncode": proc.returncode,
                "uptime_seconds": round(uptime, 1),
            }

    # Components not in _processes are simply not started
    for name in COMPONENT_COMMANDS:
        if name not in result:
            result[name] = {"status": "not_started"}

    return result


@mcp.tool()
async def check_infrastructure() -> dict[str, Any]:
    """Ping Redis, NATS, and Ollama. Return OK/FAIL for each service.

    Returns:
        Dict with 'redis', 'nats', and 'ollama' keys each containing
        'status' (ok/fail) and optional 'error' or 'detail'.
    """
    import httpx

    result: dict[str, Any] = {}

    # Redis ping — LEAK-01: context manager guarantees close on exit
    try:
        with _redis_ctx() as r:
            r.ping()
        result["redis"] = {"status": "ok", "url": _config.redis_url}
    except Exception as exc:
        result["redis"] = {"status": "fail", "error": str(exc)}

    # NATS ping
    try:
        nc = await asyncio.wait_for(
            nats_client.connect(
                _config.nats_url,
                connect_timeout=_config.nats_connect_timeout,
            ),
            timeout=_config.nats_connect_timeout + 1,
        )
        await nc.close()
        result["nats"] = {"status": "ok", "url": _config.nats_url}
    except Exception as exc:
        result["nats"] = {"status": "fail", "error": str(exc)}

    # Ollama probe
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{_config.ollama_url}/api/tags")
            if resp.status_code == 200:
                result["ollama"] = {
                    "status": "ok",
                    "url": _config.ollama_url,
                    "detail": resp.json(),
                }
            else:
                result["ollama"] = {
                    "status": "fail",
                    "url": _config.ollama_url,
                    "error": f"HTTP {resp.status_code}",
                }
    except Exception as exc:
        result["ollama"] = {"status": "fail", "error": str(exc)}

    return result


# ===========================================================================
# Injection tools
# ===========================================================================


@mcp.tool()
async def inject_event(
    domain: str,
    entity: str,
    event_type: str,
    value: float,
    unit: str,
    context: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Create a PerceptionEvent and publish it to NATS augur.perception.{domain}.

    Args:
        domain: Perception domain, e.g. "chess" or "typing".
        entity: Named entity within the domain, e.g. "white" or "user".
        event_type: Type label, e.g. "move" or "keypress".
        value: Primary numeric signal value.
        unit: Unit string, e.g. "seconds" or "wpm".
        context: Optional domain-specific extras dict.
        session_id: Session identifier. Auto-generated if not provided.

    Returns:
        Dict with 'status', 'subject', 'session_id', and 'event' keys.
    """
    # SEC-02: validate caller-supplied labels before they reach NATS
    # subjects or Redis keys.
    for field, value_ in (
        ("domain", domain),
        ("entity", entity),
        ("event_type", event_type),
    ):
        err = _validate_label(value_, field)
        if err is not None:
            return {"status": "error", "error": err}

    sid = session_id or str(uuid.uuid4())
    event = PerceptionEvent(
        domain=domain,
        stream_id=f"{domain}_injected",
        entity=entity,
        event_type=event_type,
        value=value,
        unit=unit,
        context=context or {},
        timestamp=datetime.now(timezone.utc).isoformat(),
        session_id=sid,
    )
    subject = f"augur.perception.{domain}"

    # LEAK-03: try/finally guarantees the NATS connection is closed even
    # if nc.publish raises.
    try:
        nc = await asyncio.wait_for(
            nats_client.connect(
                _config.nats_url,
                connect_timeout=_config.nats_connect_timeout,
            ),
            timeout=_config.nats_connect_timeout + 1,
        )
    except Exception as exc:
        return {"status": "error", "error": f"NATS connect failed: {exc}"}

    try:
        await nc.publish(subject, event.to_bytes())
        return {
            "status": "published",
            "subject": subject,
            "session_id": sid,
            "event": json.loads(event.to_json()),
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
    finally:
        try:
            await nc.close()
        except Exception as exc:
            log.debug("NATS close failed after inject_event: %s", exc)


@mcp.tool()
async def inject_sequence(
    events: list[dict[str, Any]],
    delay_ms: int = 100,
) -> dict[str, Any]:
    """Publish multiple PerceptionEvents with a delay between each.

    Each event dict requires: domain, entity, value.
    Optional per-event: event_type, unit, context, session_id.

    A single session_id is shared across the sequence unless overridden
    per-event.

    Args:
        events: List of event dicts.
        delay_ms: Milliseconds to wait between events.

    Returns:
        Dict with 'published', 'errors', and 'session_id' keys.
    """
    shared_sid = str(uuid.uuid4())
    delay_s = delay_ms / 1000.0
    published: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    try:
        nc = await asyncio.wait_for(
            nats_client.connect(
                _config.nats_url,
                connect_timeout=_config.nats_connect_timeout,
            ),
            timeout=_config.nats_connect_timeout + 1,
        )
    except Exception as exc:
        return {"status": "error", "error": f"NATS connect failed: {exc}"}

    try:
        for i, ev in enumerate(events):
            try:
                domain = ev["domain"]
                entity = ev["entity"]
                value = float(ev["value"])
                event_type = ev.get("event_type", "event")
                unit = ev.get("unit", "")
                context = ev.get("context") or {}
                sid = ev.get("session_id") or shared_sid

                # SEC-02: validate caller-supplied labels per event
                for field, value_ in (
                    ("domain", domain),
                    ("entity", entity),
                    ("event_type", event_type),
                ):
                    err = _validate_label(value_, field)
                    if err is not None:
                        raise ValueError(err)

                event = PerceptionEvent(
                    domain=domain,
                    stream_id=f"{domain}_injected",
                    entity=entity,
                    event_type=event_type,
                    value=value,
                    unit=unit,
                    context=context,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    session_id=sid,
                )
                subject = f"augur.perception.{domain}"
                await nc.publish(subject, event.to_bytes())
                published.append({"index": i, "subject": subject, "entity": entity})
            except Exception as exc:
                errors.append({"index": i, "error": str(exc)})

            if i < len(events) - 1:
                await asyncio.sleep(delay_s)
    finally:
        try:
            await nc.close()
        except Exception as exc:
            log.debug("NATS close failed after inject_sequence: %s", exc)

    return {
        "session_id": shared_sid,
        "published": published,
        "errors": errors,
        "total": len(events),
    }


# ===========================================================================
# Inspection tools
# ===========================================================================


@mcp.tool()
def get_baseline(domain: str, entity: str) -> dict[str, Any]:
    """Read the persisted EWMA baseline for a domain/entity pair."""
    try:
        with _persistence_ctx() as pm:
            baseline = pm.load_baseline(domain, entity)
        if baseline is None:
            return {"error": "not found", "domain": domain, "entity": entity}
        return {"domain": domain, "entity": entity, "baseline": baseline}
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def get_last_anomaly(domain: str | None = None) -> dict[str, Any]:
    """Read the last anomaly event from Redis (ARCH-07: via PersistenceManager)."""
    try:
        with _persistence_ctx() as pm:
            data = pm.load_last_anomaly()
        if data is None:
            return {"error": "not found"}
        if domain is not None and data.get("domain") != domain:
            return {"error": "not found", "requested_domain": domain}
        return data
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def get_last_advice(domain: str | None = None) -> dict[str, Any]:
    """Read the last LLM advice from Redis (ARCH-07: via PersistenceManager)."""
    try:
        with _persistence_ctx() as pm:
            data = pm.load_last_advice()
        if data is None:
            return {"error": "not found"}
        if domain is not None and data.get("domain") != domain:
            return {"error": "not found", "requested_domain": domain}
        return data
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def get_session(session_id: str | None = None) -> dict[str, Any]:
    """Read session info from Redis.

    If ``session_id`` is None, returns the current session metadata.
    Otherwise returns the reflection report for that session.
    """
    try:
        with _persistence_ctx() as pm:
            if session_id is None:
                current = pm.load_current_session()
                if current is None:
                    return {"error": "no current session"}
                return current
            report = pm.load_reflection(session_id)
        if report is None:
            return {"error": "not found", "session_id": session_id}
        return report
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def get_reflection(session_id: str | None = None) -> dict[str, Any]:
    """Read reflection report (ARCH-07: via PersistenceManager).

    If session_id is None, falls back to the current session from Redis.
    """
    try:
        with _persistence_ctx() as pm:
            sid = session_id
            if sid is None:
                current = pm.load_current_session()
                if current is not None:
                    sid = current.get("session_id")
            if sid is None:
                return {"error": "no session_id available"}
            report = pm.load_reflection(sid)
        if report is None:
            return {"error": "not found", "session_id": sid}
        return report
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def list_sessions(limit: int = 10) -> dict[str, Any]:
    """List recent sessions using feedback records."""
    try:
        with _persistence_ctx() as pm:
            sessions = pm.get_all_feedback(limit=limit)
        return {"sessions": sessions, "count": len(sessions)}
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def get_thresholds(domain: str | None = None) -> dict[str, Any]:
    """Read detection thresholds from persistence or return config defaults.

    Args:
        domain: Domain to look up. If None, returns config defaults.

    Returns:
        Threshold dict.
    """
    try:
        if domain is None:
            return {
                "source": "config_defaults",
                "default_sigma_threshold": _config.default_sigma_threshold,
                "hst_threshold": _config.hst_threshold,
                "severity_medium_sigma": _config.severity_medium_sigma,
                "severity_high_sigma": _config.severity_high_sigma,
                "min_observations": _config.min_observations,
            }
        with _persistence_ctx() as pm:
            thresholds = pm.load_thresholds(domain)
        if thresholds is None:
            return {
                "source": "config_defaults",
                "domain": domain,
                "default_sigma_threshold": _config.default_sigma_threshold,
                "hst_threshold": _config.hst_threshold,
                "severity_medium_sigma": _config.severity_medium_sigma,
                "severity_high_sigma": _config.severity_high_sigma,
                "min_observations": _config.min_observations,
            }
        return {"source": "persistence", "domain": domain, "thresholds": thresholds}
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def get_config() -> dict[str, Any]:
    """Return the current AugurConfig as a JSON-serializable dict.

    Returns:
        All configuration fields and their current values.
    """
    return _config.as_dict()


@mcp.tool()
def get_correlation_graph(session_id: str) -> dict[str, Any]:
    """Read a persisted cross-domain correlation graph from Redis.

    Args:
        session_id: Session whose graph was flushed on augur.session.end.

    Returns:
        Dict in networkx node_link_data format with 'directed', 'nodes',
        'edges' keys (NetworkX 3.4+ uses 'edges'; 3.3 and older used
        'links'), or {'error': 'not found'} if the session has no
        persisted graph.
    """
    try:
        with _persistence_ctx() as pm:
            graph = pm.load_correlation_graph(session_id)
        if graph is None:
            return {"error": "not found", "session_id": session_id}
        return {"session_id": session_id, "graph": graph}
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def list_correlation_graphs(limit: int = 50) -> dict[str, Any]:
    """List recent session ids that have persisted correlation graphs."""
    try:
        with _persistence_ctx() as pm:
            ids = pm.list_correlation_graphs(limit=limit)
        return {"session_ids": ids, "count": len(ids)}
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def dump_correlation_window() -> dict[str, Any]:
    """Return the current contents of the correlator's sliding window.

    Reads the augur:correlation:window sorted set directly and returns
    one entry per member: {anomaly: dict, score: unix_timestamp}.
    Useful for verifying what the correlator currently sees as "recent"
    when debugging correlation misses.
    """
    try:
        with _redis_ctx() as r:
            raw_members = r.zrevrangebyscore(
                "augur:correlation:window", "+inf", "-inf", withscores=True
            )
        window: list[dict[str, Any]] = []
        for member, score in raw_members:
            member_str = member.decode() if isinstance(member, bytes) else member
            try:
                anomaly = json.loads(member_str)
            except json.JSONDecodeError:
                continue
            window.append({"anomaly": anomaly, "score": float(score)})
        return {"window": window, "count": len(window)}
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def get_escalation_matrix() -> dict[str, Any]:
    """Read the current cross-domain escalation matrix from Redis."""
    try:
        with _persistence_ctx() as pm:
            matrix = pm.load_escalation_matrix()
        if matrix is None:
            return {"error": "not set"}
        return {"matrix": matrix}
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def get_app_descriptors() -> dict[str, Any]:
    """Read the autonomously-learned app->descriptor map from Redis."""
    try:
        with _persistence_ctx() as pm:
            descriptors = pm.load_app_descriptors()
        return {"descriptors": descriptors}
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def get_gate_silences(limit: int = 100) -> dict[str, Any]:
    """Return recent gate suppression records and per-arm counts.

    Read-only.  Reads up to *limit* silence records from
    ``augur:gate:silences`` (newest first) and derives a per-arm frequency
    tally from the returned records.

    Args:
        limit: Maximum number of records to return (default 100).

    Returns:
        Dict with 'silences' (list of records), 'arm_counts' (dict mapping
        arm name to suppression count), and 'total' (int).
    """
    try:
        with _persistence_ctx() as pm:
            silences = pm.load_silence_records(limit=limit)
        arm_counts: dict[str, int] = {}
        for rec in silences:
            arm = rec.get("arm")
            if arm:
                arm_counts[arm] = arm_counts.get(arm, 0) + 1
        return {
            "silences": silences,
            "arm_counts": arm_counts,
            "total": len(silences),
        }
    except Exception as exc:
        return {"error": str(exc)}


_VALID_SEVERITIES = {"LOW", "MEDIUM", "HIGH"}

# SEC-04: caps on matrix size. Without these, a caller could pass
# thousands of rules or very long keys, amplifying memory and Redis-read
# latency (the correlator re-reads the matrix on every anomaly event).
# Bumped from 20 → 40 to accommodate 6 pairwise + 10 3-way defaults
# plus room for future expansion.
MAX_ESCALATION_RULES = 40
MAX_ESCALATION_RULE_KEY_LEN = 32
MAX_ESCALATION_VERSION_LEN = 32
MAX_RULE_WINDOWS = 40


def _validate_escalation_rules(rules: dict[str, str]) -> str | None:
    """Return an error message if rules fail shape validation, else None."""
    if not isinstance(rules, dict):
        return "rules must be a dict"
    if len(rules) > MAX_ESCALATION_RULES:
        return f"too many rules: {len(rules)} (max {MAX_ESCALATION_RULES})"
    for key, value in rules.items():
        if not isinstance(key, str) or "+" not in key:
            return f"invalid rule key (expected 'A+B'): {key!r}"
        if len(key) > MAX_ESCALATION_RULE_KEY_LEN:
            return (
                f"rule key too long: {len(key)} chars "
                f"(max {MAX_ESCALATION_RULE_KEY_LEN})"
            )
        parts = key.split("+")
        if not all(p in _VALID_SEVERITIES for p in parts):
            return (
                f"invalid severity in rule key {key!r}: "
                f"each part must be one of {sorted(_VALID_SEVERITIES)}"
            )
        if not isinstance(value, str) or value not in _VALID_SEVERITIES:
            return (
                f"invalid rule value for {key!r}: "
                f"must be one of {sorted(_VALID_SEVERITIES)}, got {value!r}"
            )
    return None


def _validate_escalation_matrix_rule_windows(
    rule_windows: dict | None,
    config: AugurConfig,
) -> str | None:
    """Validate optional rule_windows dict in the matrix.

    Pairwise-only this phase (one '+'). Values must be numeric within
    [correlation_window_min_s, correlation_window_max_s]. Returns None
    if valid; an error string otherwise.
    """
    if rule_windows is None:
        return None
    if not isinstance(rule_windows, dict):
        return "rule_windows must be a dict"
    if len(rule_windows) > MAX_RULE_WINDOWS:
        return f"too many rule_windows: {len(rule_windows)} (max {MAX_RULE_WINDOWS})"
    for key, value in rule_windows.items():
        if not isinstance(key, str):
            return f"invalid rule_windows key type: {type(key).__name__}"
        if len(key) > MAX_ESCALATION_RULE_KEY_LEN:
            return (
                f"rule_windows key '{key}' exceeds {MAX_ESCALATION_RULE_KEY_LEN} chars"
            )
        if key.count("+") != 1:
            return (
                f"rule_windows key '{key}' must be pairwise (one '+'); "
                f"N-way windows are not yet supported."
            )
        if not isinstance(value, (int, float)):
            return f"rule_windows value for '{key}' must be numeric"
        if not (
            config.correlation_window_min_s
            <= float(value)
            <= config.correlation_window_max_s
        ):
            return (
                f"rule_windows[{key}]={value} outside "
                f"[{config.correlation_window_min_s}, {config.correlation_window_max_s}]"
            )
    return None


@mcp.tool()
def set_escalation_matrix(
    rules: dict[str, str],
    version: str = "1.0",
    rule_windows: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Write a new escalation matrix to Redis for runtime tuning.

    If rule_windows is omitted, any existing rule_windows in the current
    matrix are preserved (so callers updating only ``rules`` don't erase
    tuned windows).
    """
    # R2-SEC-01: the version string is written to Redis as part of the
    # matrix JSON and the correlator re-reads the matrix on every anomaly
    # event. An unbounded version string would amplify the per-event
    # Redis read+deserialize cost. Cap it.
    if not isinstance(version, str):
        return {"error": f"version must be a string, got {type(version).__name__}"}
    if len(version) > MAX_ESCALATION_VERSION_LEN:
        return {
            "error": (
                f"version string too long: {len(version)} chars "
                f"(max {MAX_ESCALATION_VERSION_LEN})"
            )
        }

    err = _validate_escalation_rules(rules)
    if err is not None:
        return {"error": err}

    config = AugurConfig.from_env()
    err = _validate_escalation_matrix_rule_windows(rule_windows, config)
    if err is not None:
        return {"error": err}

    matrix: dict = {"version": version, "rules": rules}
    try:
        with _persistence_ctx() as pm:
            if rule_windows is None:
                # Preserve existing rule_windows; don't erase tuned windows
                # when the caller only wants to update rules.
                existing = pm.load_escalation_matrix() or {}
                if "rule_windows" in existing:
                    matrix["rule_windows"] = existing["rule_windows"]
            else:
                matrix["rule_windows"] = rule_windows
            pm.save_escalation_matrix(matrix)
        return {"status": "saved", "matrix": matrix}
    except Exception as exc:
        return {"error": str(exc)}


# ===========================================================================
# Control tools
# ===========================================================================


@mcp.tool()
async def trigger_reflection(session_id: str | None = None) -> dict[str, Any]:
    """Trigger the reflection engine by publishing to NATS augur.reflect.trigger.

    Args:
        session_id: Optional session to reflect on. Uses current session if None.

    Returns:
        Dict with 'status' and 'session_id'.
    """
    # Resolve session id first (may need a Redis read).
    sid = session_id
    if sid is None:
        try:
            with _persistence_ctx() as pm:
                current = pm.load_current_session()
            if current is not None:
                sid = current.get("session_id")
        except Exception as exc:
            return {"status": "error", "error": f"Redis read failed: {exc}"}

    payload = json.dumps({"session_id": sid}).encode()

    # LEAK-04: guarantee nc.close() even if publish raises.
    try:
        nc = await asyncio.wait_for(
            nats_client.connect(
                _config.nats_url,
                connect_timeout=_config.nats_connect_timeout,
            ),
            timeout=_config.nats_connect_timeout + 1,
        )
    except Exception as exc:
        return {"status": "error", "error": f"NATS connect failed: {exc}"}

    try:
        await nc.publish("augur.reflect.trigger", payload)
        return {"status": "triggered", "session_id": sid}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
    finally:
        try:
            await nc.close()
        except Exception as exc:
            log.debug("NATS close failed after trigger_reflection: %s", exc)


@mcp.tool()
def flush_state(confirm: bool = False) -> dict[str, Any]:
    """Delete all augur:* keys from Redis.

    Args:
        confirm: Must be True to proceed. Acts as a safety guard.

    Returns:
        Dict with 'status' and 'deleted_count'.
    """
    if not confirm:
        return {
            "status": "aborted",
            "reason": "Pass confirm=True to actually flush Redis state.",
        }
    try:
        with _redis_ctx() as r:
            keys = r.keys("augur:*")
            if keys:
                deleted = r.delete(*keys)
            else:
                deleted = 0
        return {"status": "flushed", "deleted_count": deleted}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


# ===========================================================================
# Entry point
# ===========================================================================


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
