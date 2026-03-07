"""Console display for Augur chess advisor output.

Subscribes to NATS for anomaly detections and LLM advice,
renders them as color-coded console output.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import textwrap
from datetime import datetime, timezone

import nats

# ---------------------------------------------------------------------------
# Logging (minimal — this module IS the display)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("console_display")

# ---------------------------------------------------------------------------
# ANSI escape codes
# ---------------------------------------------------------------------------
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

FG_RED = "\033[91m"
FG_YELLOW = "\033[93m"
FG_GREEN = "\033[92m"
FG_CYAN = "\033[96m"
FG_WHITE = "\033[97m"
FG_GRAY = "\033[90m"

BG_RED = "\033[41m"
BG_YELLOW = "\033[43m"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NATS_URL = "nats://localhost:4222"
SUBJECT_ANOMALY = "augur.detection.anomaly"
SUBJECT_ADVICE = "augur.reasoning.advice"
SUBJECT_REFLECT = "augur.reflect.complete"
WRAP_WIDTH = 80

# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

SEVERITY_STYLE = {
    "low": (FG_GREEN, "LOW"),
    "medium": (FG_YELLOW + BOLD, "MEDIUM"),
    "high": (FG_RED + BOLD, "HIGH"),
}

SEPARATOR = f"{FG_GRAY}{'─' * WRAP_WIDTH}{RESET}"
THICK_SEPARATOR = f"{FG_CYAN}{'━' * WRAP_WIDTH}{RESET}"


def severity_badge(severity: str) -> str:
    color, label = SEVERITY_STYLE.get(severity, (FG_WHITE, severity.upper()))
    return f"{color}[{label}]{RESET}"


def render_anomaly_line(data: dict) -> str:
    """One-line notification for low-severity anomalies."""
    player = data.get("player", "?")
    move = data.get("move", "?")
    think = data.get("think_time", 0)
    dev = data.get("deviation_score", 0)
    badge = severity_badge(data.get("severity", "low"))
    ts = _short_ts(data.get("timestamp", ""))
    return (
        f"{DIM}{ts}{RESET}  "
        f"{badge}  "
        f"{FG_GRAY}{player} played {move}  "
        f"think={think:.1f}s  dev={dev:.1f}σ{RESET}"
    )


def render_advice(data: dict) -> str:
    """Multi-line formatted block for LLM advice."""
    player = data.get("player", "?")
    move = data.get("move", "?")
    severity = data.get("severity", "?")
    think_time = data.get("think_time", 0)
    advice_text = data.get("advice", "(no advice)")
    model = data.get("model", "?")
    latency_ms = data.get("latency_ms", 0)

    badge = severity_badge(severity)
    player_color = FG_WHITE + BOLD if player == "white" else FG_CYAN + BOLD

    # Word-wrap the advice text
    wrapped = textwrap.fill(advice_text, width=WRAP_WIDTH - 4)
    indented = "\n".join(f"  {line}" for line in wrapped.splitlines())

    lines = [
        "",
        THICK_SEPARATOR,
        f"  {BOLD}AUGUR ADVISOR{RESET}  {badge}  "
        f"{player_color}{player.upper()}{RESET} played {BOLD}{move}{RESET}",
        SEPARATOR,
        f"  {FG_GRAY}Think time:{RESET}  {BOLD}{think_time:.1f}s{RESET}",
        SEPARATOR,
        f"{FG_WHITE}{indented}{RESET}",
        SEPARATOR,
        f"  {FG_GRAY}Model: {model}  |  LLM latency: {latency_ms:.0f}ms{RESET}",
        THICK_SEPARATOR,
        "",
    ]
    return "\n".join(lines)


def render_reflection(data: dict) -> str:
    """End-of-session reflection summary block."""
    session_id = data.get("session_id", "?")
    analyses = data.get("analyses", {})
    adjustments = data.get("adjustments", {})

    precision = analyses.get("precision", {})
    utility = analyses.get("utility", {})
    counterfactual = analyses.get("counterfactual", {})

    prec_ratio = precision.get("precision_ratio", 0)
    prec_color = FG_GREEN if prec_ratio >= 0.7 else FG_YELLOW if prec_ratio >= 0.4 else FG_RED
    util_score = utility.get("utility_score", 0)
    util_color = FG_GREEN if util_score >= 0.7 else FG_YELLOW if util_score >= 0.4 else FG_RED

    sigma_line = ""
    if adjustments.get("sigma_adjusted"):
        sigma_line = (
            f"\n  {FG_YELLOW}Sigma threshold adjusted to "
            f"{adjustments.get('sigma_value', '?')}{RESET}"
        )

    prompt_line = ""
    if adjustments.get("prompt_mutated"):
        prompt_line = f"\n  {FG_YELLOW}LLM prompt mutated for next session{RESET}"

    cf_rec = counterfactual.get("recommendation", "")

    lines = [
        "",
        THICK_SEPARATOR,
        f"  {BOLD}AUGUR REFLECTION{RESET}  {FG_GRAY}session {session_id[:8]}...{RESET}",
        SEPARATOR,
        f"  {FG_GRAY}Precision:{RESET}   {prec_color}{BOLD}{prec_ratio:.0%}{RESET}"
        f"  {FG_GRAY}({precision.get('escalated', 0)}/{precision.get('total_anomalies', 0)} useful){RESET}",
        f"  {FG_GRAY}Utility:{RESET}     {util_color}{BOLD}{util_score:.2f}{RESET}"
        f"  {FG_GRAY}(explicit={utility.get('explicit_component', 0):.2f},"
        f" behavioral={utility.get('behavioral_component', 0):.2f}){RESET}",
        f"  {FG_GRAY}Counterfactual:{RESET}  {FG_WHITE}{cf_rec}{RESET}",
    ]

    if sigma_line:
        lines.append(sigma_line)
    if prompt_line:
        lines.append(prompt_line)

    lines.extend([
        SEPARATOR,
        f"  {FG_GRAY}{precision.get('reason', '')}{RESET}",
        THICK_SEPARATOR,
        "",
    ])
    return "\n".join(lines)


def _short_ts(iso_ts: str) -> str:
    """Extract HH:MM:SS from an ISO timestamp, or return as-is."""
    try:
        dt = datetime.fromisoformat(iso_ts)
        return dt.strftime("%H:%M:%S")
    except (ValueError, TypeError):
        return iso_ts[:8] if iso_ts else "??:??:??"

# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------

BANNER = f"""{FG_CYAN}{BOLD}
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║       █████╗ ██╗   ██╗ ██████╗ ██╗   ██╗██████╗           ║
    ║      ██╔══██╗██║   ██║██╔════╝ ██║   ██║██╔══██╗          ║
    ║      ███████║██║   ██║██║  ███╗██║   ██║██████╔╝          ║
    ║      ██╔══██║██║   ██║██║   ██║██║   ██║██╔══██╗          ║
    ║      ██║  ██║╚██████╔╝╚██████╔╝╚██████╔╝██║  ██║          ║
    ║      ╚═╝  ╚═╝ ╚═════╝  ╚═════╝  ╚═════╝ ╚═╝  ╚═╝          ║
    ║                                                           ║
    ║          Neurosymbolic Chess Timing Analyzer              ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
{RESET}
{FG_GRAY}  Listening for events...
  ├─ {FG_GREEN}anomalies{FG_GRAY}    →  {SUBJECT_ANOMALY}
  ├─ {FG_CYAN}advice{FG_GRAY}       →  {SUBJECT_ADVICE}
  └─ {FG_YELLOW}reflection{FG_GRAY}   →  {SUBJECT_REFLECT}
{RESET}"""

# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

async def run() -> None:
    nc = await nats.connect(NATS_URL, connect_timeout=5)

    print(BANNER, flush=True)

    async def on_anomaly(msg: nats.aio.client.Msg) -> None:
        try:
            data = json.loads(msg.data.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        severity = data.get("severity", "low")
        if severity == "low":
            print(render_anomaly_line(data), flush=True)

    async def on_advice(msg: nats.aio.client.Msg) -> None:
        try:
            data = json.loads(msg.data.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        print(render_advice(data), flush=True)

    async def on_reflection(msg: nats.aio.client.Msg) -> None:
        try:
            data = json.loads(msg.data.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        print(render_reflection(data), flush=True)

    await nc.subscribe(SUBJECT_ANOMALY, cb=on_anomaly)
    await nc.subscribe(SUBJECT_ADVICE, cb=on_advice)
    await nc.subscribe(SUBJECT_REFLECT, cb=on_reflection)

    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await nc.close()


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print(f"\n{FG_GRAY}Augur display stopped.{RESET}")


if __name__ == "__main__":
    main()
