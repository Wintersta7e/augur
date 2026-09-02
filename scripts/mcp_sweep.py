#!/usr/bin/env python3
"""Live MCP surface sweep: drive every augur_mcp tool over real stdio.

Spawns the FastMCP server as a subprocess (the same wire surface a real MCP
client sees), lists its tools, then calls all of them against the running
deploy stack: happy path plus validation edges (bad labels, malformed
sequence entries, invalid ratings, oversized matrices, limit clamps).

Read-only tools are scored on their documented response shapes against
whatever live pipeline state exists; write tools use run-unique
entities/sessions so they never collide with other test traffic. The
escalation matrix is restored to its prior value after the roundtrip check.
flush_state is exercised only through its confirm=False refusal unless
--flush is passed (then it really wipes augur:* as the final step).

dialogue_turn needs Ollama; from WSL export AUGUR_OLLAMA_URL to the Windows
host gateway (auto-detected from `ip route` when unset). --skip-dialogue
skips the three dialogue tools for infra-only runs.

Usage: .venv/bin/python scripts/mcp_sweep.py [--flush] [--skip-dialogue]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import uuid
from datetime import timedelta
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_TOOLS = {
    "start_pipeline",
    "stop_pipeline",
    "pipeline_status",
    "check_infrastructure",
    "inject_event",
    "inject_sequence",
    "get_baseline",
    "get_last_anomaly",
    "get_last_advice",
    "get_pipeline_health",
    "get_auspices",
    "get_self_model",
    "get_session",
    "get_reflection",
    "list_sessions",
    "get_thresholds",
    "get_config",
    "get_correlation_graph",
    "list_correlation_graphs",
    "dump_correlation_window",
    "get_escalation_matrix",
    "get_app_descriptors",
    "get_gate_silences",
    "get_proposals",
    "get_conscientia_charter",
    "get_conscientia_verdicts",
    "get_conscientia_violations",
    "get_praesagium_patterns",
    "get_praesagium_predictions",
    "dialogue_turn",
    "dialogue_history",
    "dialogue_pending",
    "set_escalation_matrix",
    "trigger_reflection",
    "submit_feedback",
    "flush_state",
}

_POOL = [3.2, 3.8, 3.5, 4.0, 3.1, 3.9, 3.3, 3.7, 3.4, 3.6]


def detect_ollama_url() -> str:
    if os.environ.get("AUGUR_OLLAMA_URL"):
        return os.environ["AUGUR_OLLAMA_URL"]
    try:
        out = subprocess.run(
            ["ip", "route"], capture_output=True, text=True, timeout=5
        ).stdout
        for line in out.splitlines():
            if line.startswith("default via "):
                return f"http://{line.split()[2]}:11434"
    except Exception:
        pass
    return "http://127.0.0.1:11434"


def parse(result) -> dict:
    """Normalize a CallToolResult into the tool's dict payload."""
    sc = getattr(result, "structuredContent", None)
    if isinstance(sc, dict):
        return sc.get("result", sc)
    for block in result.content or []:
        text = getattr(block, "text", None)
        if text:
            try:
                return json.loads(text)
            except (json.JSONDecodeError, TypeError):
                return {"_raw": text}
    return {}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--flush",
        action="store_true",
        help="really flush augur:* state as the final step",
    )
    ap.add_argument(
        "--skip-dialogue",
        action="store_true",
        help="skip the three dialogue tools (no Ollama needed)",
    )
    args = ap.parse_args()

    rid = uuid.uuid4().hex[:8]
    entity = f"sweep_{rid}"
    dlg_session = f"mcp-sweep-{rid}"
    rows: list[tuple[str, bool, str]] = []

    def row(name: str, ok: bool, detail: str) -> None:
        rows.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:34s} {detail}", flush=True)

    params = StdioServerParameters(
        command=str(_ROOT / ".venv" / "bin" / "python"),
        args=["-m", "augur_mcp.augur_server"],
        env={**os.environ, "AUGUR_OLLAMA_URL": detect_ollama_url()},
        cwd=str(_ROOT),
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            async def call(
                tool: str, arguments: dict | None = None, timeout_s: float = 30.0
            ) -> dict:
                res = await session.call_tool(
                    tool,
                    arguments or {},
                    read_timeout_seconds=timedelta(seconds=timeout_s),
                )
                return parse(res)

            # ---- surface: the tool list itself
            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            row(
                "list_tools (36 tools)",
                names == EXPECTED_TOOLS,
                f"{len(names)} tools, missing={sorted(EXPECTED_TOOLS - names)}, "
                f"extra={sorted(names - EXPECTED_TOOLS)}",
            )

            # ---- seed: run-unique baseline so entity-scoped reads have data
            seed = [
                {
                    "domain": "typing",
                    "entity": entity,
                    "event_type": "pause",
                    "value": _POOL[i % 10],
                    "unit": "seconds",
                }
                for i in range(14)
            ]
            r = await call(
                "inject_sequence", {"events": seed, "delay_ms": 30}, timeout_s=60
            )
            sid = r.get("session_id", "")
            published = r.get("published") or []
            row(
                "inject_sequence (seed)",
                len(published) == 14 and not r.get("errors"),
                f"published={len(published)} errors={len(r.get('errors', []))}",
            )
            await asyncio.sleep(2.0)  # let vigil ingest before baseline read

            # ---- infrastructure / status reads
            r = await call("check_infrastructure")
            row(
                "check_infrastructure",
                r.get("redis", {}).get("status") == "ok"
                and r.get("nats", {}).get("status") == "ok",
                f"redis={r.get('redis', {}).get('status')} "
                f"nats={r.get('nats', {}).get('status')} "
                f"ollama={r.get('ollama', {}).get('status')}",
            )

            r = await call("pipeline_status")
            n_running = sum(
                1
                for v in r.values()
                if isinstance(v, dict) and v.get("status") == "running"
            )
            row(
                "pipeline_status",
                "vigil" in r and "vox" in r,
                f"components={len(r)} native_running={n_running}",
            )

            r = await call("get_config")
            row(
                "get_config",
                "ollama_model" in r,
                f"model={r.get('ollama_model')} keys={len(r)}",
            )

            r = await call("get_pipeline_health")
            fac = r.get("faculties", {})
            row(
                "get_pipeline_health",
                len(fac) >= 8,
                f"{len(fac)} faculties, "
                f"alive={sum(1 for v in fac.values() if v.get('liveness') == 'alive')}",
            )

            # ---- imperator read-models
            r = await call("get_auspices")
            row("get_auspices", "error" not in r and bool(r), f"keys={sorted(r)[:6]}")
            r = await call("get_self_model")
            row("get_self_model", "error" not in r and bool(r), f"keys={sorted(r)[:6]}")

            # ---- state reads over live pipeline data
            # event_type is part of the baseline's identity: one entity can
            # emit several streams on different scales, so a baseline is scoped
            # to (domain, event_type, entity). The seed above publishes "pause".
            r = await call(
                "get_baseline",
                {"domain": "typing", "event_type": "pause", "entity": entity},
            )
            row(
                "get_baseline (seeded entity)",
                "baseline" in r,
                f"resp_keys={sorted(r)[:6]}",
            )

            r = await call("get_last_anomaly", {"domain": "typing"})
            row(
                "get_last_anomaly",
                "error" not in r and bool(r),
                f"keys={sorted(r)[:5]}",
            )

            # unfiltered: correlated advice carries domain="multi", so a
            # domain filter can legitimately miss the newest record
            adv = await call("get_last_advice")
            adv_rec = adv.get("advice") if isinstance(adv.get("advice"), dict) else adv
            decision_id = (adv_rec or {}).get("decision_id")
            row(
                "get_last_advice",
                "error" not in adv and bool(adv),
                f"decision_id={'present' if decision_id else 'absent'} "
                f"domain={(adv_rec or {}).get('domain')}",
            )

            r = await call("get_thresholds", {"domain": "typing"})
            row("get_thresholds", "error" not in r, f"keys={sorted(r)[:5]}")

            # response nests the record under "matrix"
            r = await call("get_escalation_matrix")
            mrec = r.get("matrix") or {}
            matrix_rules = mrec.get("rules")
            matrix_version = mrec.get("version", "1.0")
            matrix_windows = mrec.get("rule_windows")
            row(
                "get_escalation_matrix",
                "error" not in r and isinstance(matrix_rules, dict),
                f"{len(matrix_rules or {})} rules version={matrix_version} "
                f"windows={matrix_windows}",
            )

            r = await call("get_app_descriptors")
            row("get_app_descriptors", "error" not in r, f"keys={sorted(r)[:5]}")

            r = await call("get_gate_silences", {"limit": 5})
            row(
                "get_gate_silences",
                "error" not in r,
                f"count={len(r.get('silences', r.get('records', [])))}",
            )

            r = await call("get_proposals", {"limit": 5})
            row(
                "get_proposals",
                "error" not in r,
                f"count={len(r.get('proposals', []))}",
            )

            # ---- conscientia surface (charter is pure code/data; verdicts and
            # violations are legitimately empty right after a flush)
            r = await call("get_conscientia_charter")
            row(
                "get_conscientia_charter",
                "error" not in r and bool(r.get("principles")),
                f"version={r.get('version')} principles={len(r.get('principles', []))}",
            )

            r = await call("get_conscientia_verdicts", {"limit": 5})
            row(
                "get_conscientia_verdicts",
                "error" not in r and "verdicts" in r,
                f"count={r.get('count')}",
            )

            r = await call("get_conscientia_violations", {"limit": 5})
            row(
                "get_conscientia_violations",
                "error" not in r and "violations" in r,
                f"count={r.get('count')}",
            )

            # ---- praesagium surface (patterns/predictions may legitimately
            # be empty on a fresh stack -- score presence of the documented
            # response shape, not non-emptiness)
            r = await call("get_praesagium_patterns", {"limit": 5})
            row(
                "get_praesagium_patterns",
                "error" not in r and "patterns" in r and "count" in r,
                f"count={r.get('count')} mined_at={r.get('mined_at')}",
            )

            r = await call("get_praesagium_predictions", {"limit": 5})
            row(
                "get_praesagium_predictions",
                "error" not in r and "open" in r and "resolved" in r and "counts" in r,
                f"counts={r.get('counts')}",
            )

            r = await call("list_sessions", {"limit": 5})
            sessions = r.get("sessions", [])
            row(
                "list_sessions",
                isinstance(sessions, list) and sessions,
                f"count={len(sessions)}",
            )

            # inject sessions are not registered sessions: not-found is the
            # documented contract, a crash/absent-echo is not
            r = await call("get_session", {"session_id": sid})
            row(
                "get_session (unregistered -> not-found contract)",
                r.get("session_id") == sid and ("error" in r or "session" in r),
                f"keys={sorted(r)[:5]}",
            )

            r = await call("get_reflection")
            row(
                "get_reflection (responds)",
                isinstance(r, dict) and bool(r),
                f"keys={sorted(r)[:5]}",
            )

            r = await call("list_correlation_graphs", {"limit": 5})
            graph_ids = r.get("session_ids", [])
            row(
                "list_correlation_graphs",
                "count" in r,
                f"count={r.get('count')} ids={graph_ids[:2]}",
            )

            if graph_ids:
                r = await call("get_correlation_graph", {"session_id": graph_ids[0]})
                row(
                    "get_correlation_graph",
                    "error" not in r,
                    f"session={graph_ids[0][:20]} keys={sorted(r)[:5]}",
                )
            else:
                r = await call("get_correlation_graph", {"session_id": sid})
                row(
                    "get_correlation_graph (absent -> error contract)",
                    "error" in r,
                    f"keys={sorted(r)[:5]}",
                )

            r = await call("dump_correlation_window")
            row("dump_correlation_window", "error" not in r, f"keys={sorted(r)[:5]}")

            # ---- write tools: happy + validation edges
            r = await call(
                "inject_event",
                {
                    "domain": "typing",
                    "entity": entity,
                    "event_type": "pause",
                    "value": 4.1,
                    "unit": "seconds",
                },
            )
            row(
                "inject_event (valid)",
                r.get("status") in ("published", "ok"),
                f"status={r.get('status')} subject={r.get('subject')}",
            )

            r = await call(
                "inject_event",
                {
                    "domain": "Bad Domain!",
                    "entity": entity,
                    "event_type": "x",
                    "value": 1.0,
                    "unit": "u",
                },
            )
            row(
                "inject_event (bad domain rejected)",
                "error" in r or r.get("status") == "error",
                f"resp={str(r)[:70]}",
            )

            r = await call(
                "inject_sequence",
                {
                    "events": [
                        {"domain": "typing", "entity": entity, "value": 3.5},
                        {"domain": "typing", "entity": entity},  # missing value
                    ],
                    "delay_ms": 10,
                },
                timeout_s=30,
            )
            row(
                "inject_sequence (partial errors)",
                len(r.get("published") or []) == 1 and len(r.get("errors") or []) == 1,
                f"published={len(r.get('published') or [])} "
                f"errors={len(r.get('errors') or [])}",
            )

            fb_id = decision_id or f"sweep-fake-{rid}"
            r = await call("submit_feedback", {"decision_id": fb_id, "rating": "y"})
            row(
                "submit_feedback (publishes)",
                r.get("status") == "submitted",
                f"status={r.get('status')} "
                f"decision_id={'real' if decision_id else 'synthetic'}",
            )
            r = await call("submit_feedback", {"decision_id": "x", "rating": "amazing"})
            row(
                "submit_feedback (bad rating rejected)",
                "error" in r,
                f"resp={str(r)[:60]}",
            )

            r = await call("trigger_reflection", {"session_id": sid})
            row(
                "trigger_reflection",
                r.get("status") in ("published", "triggered"),
                f"status={r.get('status')}",
            )

            # matrix roundtrip: valid severity-pair rules, sweep version;
            # rule_windows omitted must be preserved; then restore
            test_rules = dict(matrix_rules) if matrix_rules else {"LOW+HIGH": "HIGH"}
            r = await call(
                "set_escalation_matrix",
                {"rules": test_rules, "version": f"sweep-{rid}"},
            )
            ok_write = "error" not in r
            r2 = await call("get_escalation_matrix")
            m2 = r2.get("matrix") or {}
            roundtrip = (
                m2.get("rules") == test_rules
                and m2.get("version") == f"sweep-{rid}"
                and m2.get("rule_windows") == matrix_windows
            )
            restore = await call(
                "set_escalation_matrix",
                {
                    "rules": matrix_rules or {},
                    "version": matrix_version,
                    **({"rule_windows": matrix_windows} if matrix_windows else {}),
                },
            )
            row(
                "set_escalation_matrix (roundtrip+restore)",
                ok_write and roundtrip and "error" not in restore,
                f"write_ok={ok_write} roundtrip={roundtrip} "
                f"restored={'error' not in restore}",
            )

            # note: the 40-rule count cap is unreachable through valid input
            # (only 9 severity-pair keys exist) — this exercises key
            # validation on a bulk write instead
            too_many = {f"a{i}+b{i}": "HIGH" for i in range(25)}
            r = await call("set_escalation_matrix", {"rules": too_many})
            row(
                "set_escalation_matrix (bulk invalid keys rejected)",
                "error" in r,
                f"resp={str(r)[:60]}",
            )

            # ---- dialogue tools (real LLM)
            if args.skip_dialogue:
                print("  [skip] dialogue_turn/history/pending (--skip-dialogue)")
            else:
                r = await call(
                    "dialogue_turn",
                    {
                        "session_id": dlg_session,
                        "message": "In one sentence, what have you noticed about "
                        "my typing today?",
                    },
                    timeout_s=240,
                )
                row(
                    "dialogue_turn",
                    bool(r.get("reply")) and not r.get("error"),
                    f"reply={str(r.get('reply'))[:50]!r} err={r.get('error')}",
                )

                r = await call(
                    "dialogue_history", {"session_id": dlg_session, "limit": 5}
                )
                turns = r.get("turns", [])
                row("dialogue_history", len(turns) >= 1, f"turns={len(turns)}")

                r = await call(
                    "dialogue_history", {"session_id": dlg_session, "limit": 0}
                )
                row(
                    "dialogue_history (limit=0 clamped)",
                    "error" not in r,
                    f"turns={len(r.get('turns', []))}",
                )

                r = await call("dialogue_pending", {"session_id": dlg_session})
                row("dialogue_pending", "error" not in r, f"pending={r.get('pending')}")

            # ---- native component lifecycle (vox is display-only, safe)
            r = await call("start_pipeline", {"components": ["vox"]})
            started = "vox" in str(r.get("started", r))
            await asyncio.sleep(1.5)
            st = await call("pipeline_status")
            vox_status = (st.get("vox") or {}).get("status")
            r2 = await call("stop_pipeline", {"components": ["vox"]})
            stopped = "vox" in str(r2.get("stopped", r2))
            row(
                "start/stop_pipeline (vox)",
                started and vox_status == "running" and stopped,
                f"started={started} status={vox_status} stopped={stopped}",
            )

            # ---- flush_state: refusal without confirm, real flush on --flush
            r = await call("flush_state", {})
            refused = r.get("status") == "aborted"
            probe = await call("get_thresholds")
            row(
                "flush_state (refuses without confirm)",
                refused and "error" not in probe,
                f"status={r.get('status')} state_intact={'error' not in probe}",
            )

            if args.flush:
                r = await call("flush_state", {"confirm": True})
                row(
                    "flush_state (confirm=True)",
                    r.get("status") == "flushed",
                    f"deleted={r.get('deleted_count')}",
                )

    print("\n" + "=" * 72)
    print("MCP SWEEP REPORT")
    print("=" * 72)
    passed = sum(1 for _, ok, _ in rows if ok)
    for name, ok, detail in rows:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:34s} {detail}")
    print("=" * 72)
    print(f"OVERALL: {passed}/{len(rows)} PASS")
    return 0 if passed == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
