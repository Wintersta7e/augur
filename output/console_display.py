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
from datetime import datetime
from pathlib import Path

import nats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blackboard.config import AugurConfig  # noqa: E402

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
SUBJECT_ANOMALY = "augur.detection.anomaly"
SUBJECT_ADVICE = "augur.reasoning.advice"
SUBJECT_CORRELATION = "augur.correlation.detected"
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
    domain = data.get("domain", "?")
    entity = data.get("entity", "?")
    severity = data.get("severity", "low")
    badge = severity_badge(severity)
    ts = _short_ts(data.get("timestamp", ""))
    dev = data.get("deviation_score", 0)

    if domain == "chess":
        player = data.get("player", "?")
        move = data.get("move", "?")
        think = data.get("think_time", 0)
        return (
            f"{DIM}{ts}{RESET}  "
            f"{badge}  "
            f"{FG_GRAY}{player} played {move}  "
            f"think={think:.1f}s  dev={dev:.1f}σ{RESET}"
        )

    if domain == "typing":
        wpm = (data.get("context") or {}).get("avg_wpm", "?")
        return (
            f"{DIM}{ts}{RESET}  "
            f"{badge}  "
            f"{FG_GRAY}{entity}  "
            f"wpm={wpm}  dev={dev:.1f}σ{RESET}"
        )

    if domain == "activity_focus":
        ctx = data.get("context", {}) or {}
        active = ctx.get("active_dwell_s", "?")
        new_app = ctx.get("new_app", "?")
        baseline = data.get("baseline_mean", "?")
        return (
            f"{DIM}{ts}{RESET}  "
            f"{badge}  "
            f"{FG_GRAY}ACTIVITY_FOCUS/{entity}: "
            f"dwell {active}s (then {new_app}). "
            f"baseline {baseline} log1p_s. dev {dev:.1f}σ{RESET}"
        )

    if domain == "activity_intensity":
        ctx = data.get("context", {}) or {}
        value = data.get("value", "?")
        keystrokes = ctx.get("keystroke_count", "?")
        baseline = data.get("baseline_mean", "?")
        return (
            f"{DIM}{ts}{RESET}  "
            f"{badge}  "
            f"{FG_GRAY}ACTIVITY_INTENSITY/{entity}: "
            f"{value} ipm (keys={keystrokes}). baseline {baseline}. dev {dev:.1f}σ{RESET}"
        )

    # Generic fallback for unknown domains
    value = data.get("value", "?")
    baseline = data.get("baseline_mean", "?")
    return (
        f"{DIM}{ts}{RESET}  "
        f"{badge}  "
        f"{FG_GRAY}{domain}/{entity}: "
        f"value={value} baseline={baseline} dev={dev:.1f}σ{RESET}"
    )


def render_correlation(data: dict) -> str:
    """Block render for a correlation event from augur.correlation.detected.

    Correlated event: multi-domain MEDIUM/HIGH block with contributing signals.
    Pass-through event: single-domain block identifying it as standalone.
    """
    primary = data.get("primary_anomaly", {})
    correlated = data.get("correlated_events", [])
    combined = str(data.get("combined_severity", "?")).upper()
    rule = data.get("escalation_rule")

    color_map = {
        "LOW": FG_GREEN,
        "MEDIUM": FG_YELLOW + BOLD,
        "HIGH": FG_RED + BOLD,
    }
    color = color_map.get(combined, FG_WHITE)
    badge = f"{color}[{combined}]{RESET}"

    if data.get("correlation_found"):
        all_domains = [primary.get("domain", "?")] + [
            e.get("domain", "?") for e in correlated
        ]
        header = (
            f"{BOLD}\u26a0 CORRELATION{RESET} {badge}  "
            f"{FG_CYAN}{' + '.join(all_domains)}{RESET}"
        )
        lag = data.get("temporal_lag_seconds")
        lag_line = (
            f"  {FG_GRAY}Temporal lag: {lag}s" if lag is not None else f"  {FG_GRAY}"
        )
        if rule:
            lag_line += f"   rule: {rule}{RESET}"
        else:
            lag_line += f"{RESET}"

        sig_lines: list[str] = []
        for ev in [primary, *correlated]:
            d = ev.get("domain", "?")
            e = ev.get("entity", "?")
            v = ev.get("value", "?")
            b = ev.get("baseline_mean", "?")
            dev = ev.get("deviation_score", "?")
            sig_lines.append(
                f"  {FG_WHITE}{d}{RESET}/{e}: value={v} baseline={b} dev={dev}\u03c3"
            )

        return "\n".join(
            [
                "",
                THICK_SEPARATOR,
                header,
                SEPARATOR,
                lag_line,
                *sig_lines,
                THICK_SEPARATOR,
                "",
            ]
        )

    # Pass-through (standalone medium/high — no correlation)
    domain = primary.get("domain", "?")
    entity = primary.get("entity", "?")
    value = primary.get("value", "?")
    baseline = primary.get("baseline_mean", "?")
    dev = primary.get("deviation_score", "?")
    return "\n".join(
        [
            "",
            THICK_SEPARATOR,
            f"  {BOLD}STANDALONE{RESET} {badge}  {FG_CYAN}{domain}{RESET}/{entity}",
            SEPARATOR,
            f"  value={value}  baseline={baseline}  dev={dev}\u03c3",
            THICK_SEPARATOR,
            "",
        ]
    )


def dedup_should_suppress(last_rendered: dict, anomaly: dict) -> bool:
    """Return True if this anomaly was already rendered as a low-severity one-liner.

    Matches on (domain, entity, timestamp) — all three must agree to suppress.
    """
    domain = anomaly.get("domain")
    if domain is None:
        return False
    prev = last_rendered.get(domain)
    if prev is None:
        return False
    prev_entity, prev_ts = prev
    return prev_entity == anomaly.get("entity") and prev_ts == anomaly.get("timestamp")


def update_last_rendered(last_rendered: dict, anomaly: dict) -> None:
    """Record this anomaly as the last rendered one-liner for its domain."""
    domain = anomaly.get("domain")
    if domain is None:
        return
    last_rendered[domain] = (anomaly.get("entity"), anomaly.get("timestamp"))


def render_advice(data: dict) -> str:
    """Multi-line formatted block for LLM advice."""
    domain = data.get("domain", "?")
    severity = data.get("severity", "?")
    advice_text = data.get("advice", "(no advice)")
    model = data.get("model", "?")
    latency_ms = data.get("latency_ms", 0)

    badge = severity_badge(severity)

    # Word-wrap the advice text
    wrapped = textwrap.fill(advice_text, width=WRAP_WIDTH - 4)
    indented = "\n".join(f"  {line}" for line in wrapped.splitlines())

    # Chess domain (legacy)
    if domain == "chess":
        player = data.get("player", "?")
        move = data.get("move", "?")
        think_time = data.get("think_time", 0)
        player_color = FG_WHITE + BOLD if player == "white" else FG_CYAN + BOLD

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

    # Activity or typing domains
    entity = data.get("entity", "?")
    lines = [
        "",
        THICK_SEPARATOR,
        f"  {BOLD}AUGUR ADVISOR{RESET}  {badge}  "
        f"{FG_CYAN}{domain.upper()}{RESET}/{entity}",
        SEPARATOR,
        f"{FG_WHITE}{indented}{RESET}",
        SEPARATOR,
        f"  {FG_GRAY}Model: {model}  |  LLM latency: {latency_ms:.0f}ms{RESET}",
        THICK_SEPARATOR,
        "",
    ]
    return "\n".join(lines)


def render_reflection(data: dict) -> str:
    """End-of-session reflection summary block.

    Handles both the new per-domain shape (Phase 3+) and the legacy
    single-domain shape (pre-Phase-3) for backward compatibility with
    historical reflection records.
    """
    session_id = data.get("session_id", "?")
    analyses = data.get("analyses", {})
    adjustments = data.get("adjustments", {})

    precision = analyses.get("precision", {})
    utility = analyses.get("utility", {})
    counterfactual = analyses.get("counterfactual", {})

    util_score = utility.get("utility_score", 0)
    util_color = (
        FG_GREEN if util_score >= 0.7 else FG_YELLOW if util_score >= 0.4 else FG_RED
    )

    cf_rec = counterfactual.get("recommendation", "")

    lines: list[str] = [
        "",
        THICK_SEPARATOR,
        f"  {BOLD}AUGUR REFLECTION{RESET}  {FG_GRAY}session {session_id[:8]}...{RESET}",
        SEPARATOR,
    ]

    # Precision — per-domain (new) or top-level scalar (legacy)
    per_domain = precision.get("per_domain")
    if per_domain:
        lines.append(f"  {FG_GRAY}Precision (per-domain):{RESET}")
        for domain, result in per_domain.items():
            ratio = result.get("precision_ratio", 0)
            action = result.get("action", "none")
            prec_color = (
                FG_GREEN if ratio >= 0.7 else FG_YELLOW if ratio >= 0.4 else FG_RED
            )
            lines.append(
                f"    {FG_WHITE}{domain}{RESET}: "
                f"{prec_color}{BOLD}{ratio:.0%}{RESET}  [{action}]"
            )
    elif "precision_ratio" in precision:
        # Legacy single-domain shape
        prec_ratio = precision.get("precision_ratio", 0)
        prec_color = (
            FG_GREEN
            if prec_ratio >= 0.7
            else FG_YELLOW
            if prec_ratio >= 0.4
            else FG_RED
        )
        lines.append(
            f"  {FG_GRAY}Precision:{RESET}   {prec_color}{BOLD}{prec_ratio:.0%}{RESET}"
            f"  {FG_GRAY}({precision.get('escalated', 0)}/{precision.get('total_anomalies', 0)} useful){RESET}"
        )

    lines.append(
        f"  {FG_GRAY}Utility:{RESET}     {util_color}{BOLD}{util_score:.2f}{RESET}"
        f"  {FG_GRAY}(explicit={utility.get('explicit_component', 0):.2f},"
        f" behavioral={utility.get('behavioral_component', 0):.2f}){RESET}"
    )
    lines.append(f"  {FG_GRAY}Counterfactual:{RESET}  {FG_WHITE}{cf_rec}{RESET}")

    # Sigma adjustments — per-domain map (new) or scalar (legacy)
    sigma_values = adjustments.get("sigma_values")
    if sigma_values:
        lines.append(f"  {FG_GRAY}Sigma after:{RESET}")
        for domain, value in sigma_values.items():
            lines.append(
                f"    {FG_WHITE}{domain}{RESET}: {FG_YELLOW}{value:.2f}{RESET}"
            )
    elif adjustments.get("sigma_adjusted"):
        # Legacy single value
        lines.append(
            f"  {FG_YELLOW}Sigma threshold adjusted to "
            f"{adjustments.get('sigma_value', '?')}{RESET}"
        )

    if adjustments.get("prompt_mutated"):
        lines.append(f"  {FG_YELLOW}LLM prompt mutated for next session{RESET}")

    if adjustments.get("matrix_mutated"):
        lines.append(f"  {FG_GRAY}Matrix:{RESET} updated")

    # Window tuning (new in Phase 3)
    window_tuning = analyses.get("correlation_window_tuning", {})
    if adjustments.get("windows_tuned") and window_tuning.get("per_rule"):
        lines.append(f"  {FG_GRAY}Windows tuned:{RESET}")
        for rule_key, rec in window_tuning["per_rule"].items():
            if rec.get("action") == "tuned":
                lines.append(
                    f"    {FG_WHITE}{rule_key}{RESET}: "
                    f"{rec.get('window_before')}s {FG_GRAY}→{RESET} "
                    f"{FG_CYAN}{rec.get('window_after')}s{RESET}"
                )

    lines.extend(
        [
            SEPARATOR,
            f"  {FG_GRAY}{precision.get('reason', '')}{RESET}",
            THICK_SEPARATOR,
            "",
        ]
    )
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
    ║          Neurosymbolic Cognitive Assistant                ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
{RESET}
{FG_GRAY}  Listening for events...
  ├─ {FG_GREEN}anomalies{FG_GRAY}    →  {SUBJECT_ANOMALY}
  ├─ {FG_YELLOW}correlations{FG_GRAY} →  {SUBJECT_CORRELATION}
  ├─ {FG_CYAN}advice{FG_GRAY}       →  {SUBJECT_ADVICE}
  └─ {FG_YELLOW}reflection{FG_GRAY}   →  {SUBJECT_REFLECT}
{RESET}"""

# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------


async def run() -> None:
    config = AugurConfig.from_env()
    nc = await nats.connect(
        config.nats_url, connect_timeout=config.nats_connect_timeout
    )

    last_rendered: dict[str, tuple[str, str]] = {}

    print(BANNER, flush=True)

    async def on_anomaly(msg: nats.aio.client.Msg) -> None:
        try:
            data = json.loads(msg.data.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        severity = data.get("severity", "low")
        if severity == "low":
            print(render_anomaly_line(data), flush=True)
            update_last_rendered(last_rendered, data)

    async def on_correlation(msg: nats.aio.client.Msg) -> None:
        try:
            data = json.loads(msg.data.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        primary = data.get("primary_anomaly", {})
        if dedup_should_suppress(last_rendered, primary):
            # The primary already showed as a low-severity one-liner
            # when it first arrived on augur.detection.anomaly — do not
            # render it again as part of the correlation block.
            # The correlation block still prints (this flag only
            # suppresses the earlier one-liner's retention).
            pass
        print(render_correlation(data), flush=True)

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

    # LEAK-07: save subscription handles so unsubscribe() is called on
    # shutdown rather than relying on nc.close() to tear them down abruptly
    # mid-render.
    sub_anomaly = await nc.subscribe(SUBJECT_ANOMALY, cb=on_anomaly)
    sub_correlation = await nc.subscribe(SUBJECT_CORRELATION, cb=on_correlation)
    sub_advice = await nc.subscribe(SUBJECT_ADVICE, cb=on_advice)
    sub_reflect = await nc.subscribe(SUBJECT_REFLECT, cb=on_reflection)

    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        try:
            await sub_anomaly.unsubscribe()
            await sub_correlation.unsubscribe()
            await sub_advice.unsubscribe()
            await sub_reflect.unsubscribe()
        except Exception as exc:
            log.debug("Unsubscribe failed during shutdown: %s", exc)
        await nc.close()


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print(f"\n{FG_GRAY}Augur display stopped.{RESET}")


if __name__ == "__main__":
    main()
