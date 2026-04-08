"""Augur MCP server — 16 tools for pipeline lifecycle, event injection,
state inspection, and control.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nats as nats_client
import redis
from fastmcp import FastMCP

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackboard.config import AugurConfig  # noqa: E402
from blackboard.contracts import PerceptionEvent  # noqa: E402
from blackboard.persistence import PersistenceManager  # noqa: E402

log = logging.getLogger("augur.mcp")

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_config = AugurConfig.from_env()

_processes: dict[str, dict[str, Any]] = {}

COMPONENT_COMMANDS: dict[str, list[str]] = {
    "detector": [sys.executable, "-m", "detection.anomaly_detector"],
    "advisor": [sys.executable, "-m", "reasoning.augur_advisor"],
    "feedback": [sys.executable, "-m", "perception.feedback_collector"],
    "reflection": [sys.executable, "-m", "reasoning.reflection_engine"],
    "display": [sys.executable, "-m", "output.console_display"],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_redis() -> redis.Redis:
    return redis.Redis(
        host=_config.redis_host,
        port=_config.redis_port,
        socket_connect_timeout=_config.redis_connect_timeout,
        decode_responses=True,
    )


def _get_persistence() -> PersistenceManager:
    return PersistenceManager(_get_redis())


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

    # Redis ping
    try:
        r = _get_redis()
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

    try:
        nc = await asyncio.wait_for(
            nats_client.connect(
                _config.nats_url,
                connect_timeout=_config.nats_connect_timeout,
            ),
            timeout=_config.nats_connect_timeout + 1,
        )
        await nc.publish(subject, event.to_bytes())
        await nc.close()
        return {
            "status": "published",
            "subject": subject,
            "session_id": sid,
            "event": json.loads(event.to_json()),
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


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
        await nc.close()

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
    """Read the persisted EWMA baseline for a domain/entity pair.

    Args:
        domain: Perception domain, e.g. "chess".
        entity: Named entity, e.g. "white".

    Returns:
        Baseline dict or {'error': 'not found'}.
    """
    try:
        pm = _get_persistence()
        baseline = pm.load_baseline(domain, entity)
        if baseline is None:
            return {"error": "not found", "domain": domain, "entity": entity}
        return {"domain": domain, "entity": entity, "baseline": baseline}
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def get_last_anomaly(domain: str | None = None) -> dict[str, Any]:
    """Read the last anomaly event from Redis.

    Args:
        domain: Optional domain filter. If provided, only return if matching.

    Returns:
        Last anomaly dict or {'error': 'not found'}.
    """
    try:
        r = _get_redis()
        raw = r.get("augur:detection:last_anomaly")
        if raw is None:
            return {"error": "not found"}
        data = json.loads(raw)
        if domain is not None and data.get("domain") != domain:
            return {"error": "not found", "requested_domain": domain}
        return data
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def get_last_advice(domain: str | None = None) -> dict[str, Any]:
    """Read the last LLM advice from Redis.

    Args:
        domain: Optional domain filter. If provided, only return if matching.

    Returns:
        Last advice dict or {'error': 'not found'}.
    """
    try:
        r = _get_redis()
        raw = r.get("augur:reasoning:last_advice")
        if raw is None:
            return {"error": "not found"}
        data = json.loads(raw)
        if domain is not None and data.get("domain") != domain:
            return {"error": "not found", "requested_domain": domain}
        return data
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def get_session(session_id: str | None = None) -> dict[str, Any]:
    """Read session info from Redis.

    Args:
        session_id: Specific session to look up. Reads augur:reflect:{session_id}.
            If None, reads augur:session:current.

    Returns:
        Session dict or {'error': 'not found'}.
    """
    try:
        r = _get_redis()
        if session_id is None:
            raw = r.get("augur:session:current")
            if raw is None:
                return {"error": "no current session"}
            return json.loads(raw)
        raw = r.get(f"augur:reflect:{session_id}")
        if raw is None:
            return {"error": "not found", "session_id": session_id}
        return json.loads(raw)
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def get_reflection(session_id: str | None = None) -> dict[str, Any]:
    """Read reflection report from Redis.

    Args:
        session_id: Session to read. If None, reads current session first,
            then looks up its reflection.

    Returns:
        Reflection dict or {'error': 'not found'}.
    """
    try:
        r = _get_redis()
        sid = session_id
        if sid is None:
            raw_session = r.get("augur:session:current")
            if raw_session is not None:
                session_data = json.loads(raw_session)
                sid = session_data.get("session_id")
        if sid is None:
            return {"error": "no session_id available"}
        raw = r.get(f"augur:reflect:{sid}")
        if raw is None:
            return {"error": "not found", "session_id": sid}
        return json.loads(raw)
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def list_sessions(limit: int = 10) -> dict[str, Any]:
    """List recent sessions using feedback records.

    Args:
        limit: Maximum number of sessions to return.

    Returns:
        Dict with 'sessions' list and 'count'.
    """
    try:
        pm = _get_persistence()
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
        pm = _get_persistence()
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
    try:
        r = _get_redis()
        sid = session_id
        if sid is None:
            raw_session = r.get("augur:session:current")
            if raw_session is not None:
                session_data = json.loads(raw_session)
                sid = session_data.get("session_id")

        payload = json.dumps({"session_id": sid}).encode()

        nc = await asyncio.wait_for(
            nats_client.connect(
                _config.nats_url,
                connect_timeout=_config.nats_connect_timeout,
            ),
            timeout=_config.nats_connect_timeout + 1,
        )
        await nc.publish("augur.reflect.trigger", payload)
        await nc.close()

        return {"status": "triggered", "session_id": sid}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


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
        r = _get_redis()
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
