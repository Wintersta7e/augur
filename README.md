# Augur

[![CI](https://github.com/Wintersta7e/augur/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Wintersta7e/augur/actions/workflows/ci.yml)
[![CodeQL](https://github.com/Wintersta7e/augur/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/Wintersta7e/augur/actions/workflows/codeql.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A hybrid neurosymbolic AI system that combines neural perception with symbolic reasoning to detect, interpret, and respond to complex patterns in streaming behavioral data. Augur detects anomalies across independent perception domains, correlates signals that fire together, decides whether the moment is worth speaking to, asks a local LLM for advice, collects feedback, and tunes its own parameters after every session.

> **Status:** active personal research project. Currently validated with four independent perception domains (chess move timing, system-wide typing rhythm, per-app focus dwell, per-app interaction intensity) sharing the same detection, correlation, and reasoning pipeline. Cross-domain reasoning verified end-to-end against a local Ollama `qwen2.5:32b` model.

## The pantheon

Augur is organized as a **pantheon of named faculties** — Latin-named subsystems layered over a shared blackboard. The faculties are decentralized: they coordinate only through durable Redis state and NATS events, never direct calls. The naming is an identity/charter layer; the runtime stays a blackboard.

| Faculty | Latin sense | Role |
|---|---|---|
| **Tabula** | the slate | Shared base — config, the `PerceptionEvent` contract, sessions, and all Redis I/O |
| **Sensus** | the senses | Perception sensors (chess timing, typing rhythm, app focus/intensity) |
| **Vigil** | the watch | Domain-agnostic anomaly detection |
| **Nexus** | the binding | Cross-domain correlation + escalation matrix |
| **Consilium** | the counsel | The local-LLM advisor |
| **Limen** | the threshold | The biological "stay-silent" gate that decides whether to speak |
| **Responsum** | the reply | Feedback collection (explicit + behavioral) |
| **Disciplina** | the training | Post-session reflection and self-tuning |
| **Vox** | the voice | Terminal rendering of what the system surfaces |

## What makes it different

- **Domain-agnostic detection** — adding a new perception source requires zero changes to Vigil (the detector). Any publisher that emits a `PerceptionEvent` to an `augur.sensus.<domain>` NATS subject is picked up automatically.
- **N-way cross-domain correlation** — Nexus combines signals from any number of domains inside an adaptive correlation window using a rule-based escalation matrix. Default rules ship for both pairwise (e.g. `LOW+LOW→MEDIUM`) and 3-way (`LOW+LOW+LOW→MEDIUM`) escalation. Two or three low-severity signals firing together escalate to higher severity and trigger LLM advice that references the *combination* rather than any one signal alone.
- **Biological stay-silent gate** — before any LLM call, Limen decides suppress / fire / downgrade. A multi-arm gate (habituation, refractory burden, credibility, reservoir/rate limiting, and more) keeps Augur quiet unless the moment is genuinely worth speaking to, with hard invariants (never silence a high-severity correlated event; fail open to firing on any error; never silence a trackable channel forever) and an offline self-audit that tunes the gate from outcomes.
- **Adaptive correlation windows** — Disciplina (reflection) EWMA-updates an observed-lag estimate per pairwise rule and tunes that rule's window upward or downward (with hysteresis to prevent flapping). Rules that fire mostly within 8s get tighter windows; rules with consistently long lag get more headroom. All bounded to `[5s, 120s]` by default.
- **Self-tuning escalation rules** — per-rule EWMA confidence with hysteresis. Rules that consistently produce useful advice stay; rules that repeatedly miss have their confidence decayed toward their pre-mutation state. Disciplina updates the escalation matrix in Redis; Nexus reloads it on every event without restart.
- **Per-domain feedback attribution** — sessions that produced cross-domain advice distribute their feedback signal across all involved domains (1/N weighting). Sigma thresholds tune independently per domain; a chess+typing correlation that gets thumbs-up lowers chess *and* typing detection thresholds in lockstep.
- **Self-improvement loop** — after each session, Disciplina runs six analysis passes (per-domain precision, utility, counterfactual, correlation-rule tuning, correlation-window tuning, and a gate audit) and adjusts sigma thresholds, mutates LLM prompts via Ollama, updates escalation confidence, tunes per-rule windows, and self-tunes the gate.
- **Blackboard architecture** — Redis holds durable state, NATS carries events between faculties. Each faculty is testable in isolation; new domains plug in without touching detection or reasoning.

## Architecture

```
  sensus/* (chess, typing, activity)
         │
         │ NATS: augur.sensus.<domain>  (wildcard)
         ▼
  vigil/anomaly_detector.py              EWMA + River HalfSpaceTrees
         │
         │ NATS: augur.vigil.anomaly
         ▼
  nexus/correlator.py                    Redis sorted-set window + NetworkX DiGraph
         │                                escalation matrix lookup
         │ NATS: augur.nexus.detected
         ▼
  consilium/advisor.py                   Limen stay-silent gate → Ollama qwen2.5:32b
         │                                (limen/ decides suppress / fire / downgrade
         │                                 before the LLM call; suppressions are logged)
         │ NATS: augur.consilium.advice   (+ augur.limen.suppressed on a suppression)
         ▼
  responsum/feedback_collector.py        explicit + behavioral scoring
         │
         │ NATS: augur.responsum.complete
         ▼
  disciplina/reflection_engine.py        six-pass analysis → parameter tuning
         │
         │ NATS: augur.disciplina.complete
         ▼
  vox/console_display.py                 ANSI renderer with dedup
```

`tabula/` is the shared base every faculty imports (config, the `PerceptionEvent` contract, sessions, and `PersistenceManager` — all Redis I/O). `limen/` runs in-process inside Consilium, gating the LLM call. At session end, Nexus flushes the in-memory NetworkX DiGraph to Redis for later cross-session analysis. All live state is queryable and mutable via a 23-tool FastMCP server.

## Faculties

| Package | Purpose |
|---|---|
| `tabula/` | Shared base: `AugurConfig` (env-var config), `PerceptionEvent` contract, `SessionManager`, `PersistenceManager` (all Redis I/O), shared connections |
| `sensus/` | Perception sensors publishing to `augur.sensus.<domain>`: `chess_board.py`, `typing_monitor.py`, `activity_monitor.py` (Windows-host) |
| `vigil/` | Domain-agnostic anomaly detector (EWMA + River HalfSpaceTrees, wildcard NATS subscription) |
| `nexus/` | Cross-domain correlator — Redis sorted-set window + NetworkX session DiGraph + runtime-loadable escalation matrix |
| `consilium/` | Multi-domain LLM advisor (Ollama) + app-descriptor classifier |
| `limen/` | Biological stay-silent gate (`gate.py` + `scheduler.py`) — suppress/fire/downgrade before the LLM, runs in-process within Consilium |
| `responsum/` | Feedback collector — explicit + behavioral + correlation-aware scoring |
| `disciplina/` | Reflection engine — six-pass post-session self-improvement |
| `vox/` | ANSI terminal display with domain-scoped dedup and correlation rendering |
| `augur_mcp/` | FastMCP server with 23 tools (lifecycle, injection, inspection, control) |
| `infrastructure/` | Launcher script (6-slot pipeline), connectivity and persistence smoke tests |
| `tests/` | 888 unit (mocked) + 41 fast integration (real Redis/NATS) + 4 slow (real Ollama) |

## Prerequisites

- Python 3.12+
- Docker (for Redis 7 and NATS 2 + JetStream)
- [Ollama](https://ollama.com) with `qwen2.5:32b` (or any model — configurable) pulled
- Linux: the typing monitor uses the `keyboard` library which needs root for system-wide keypress capture. The chess board does not need root.

## Quick start

```bash
# 1. Start Redis + NATS
docker compose up -d

# 2. Set up Python environment
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 3. Verify connectivity
.venv/bin/python infrastructure/test_connections.py

# 4. Pull the LLM model
ollama pull qwen2.5:32b

# 5. Run the tests (unit only, no Ollama needed)
.venv/bin/pytest tests/ --ignore=tests/integration

# 6. Run the fast integration tests (needs Redis + NATS)
.venv/bin/pytest tests/integration/ -m "not slow"

# 7. Start the full pipeline (dev mode)
bash infrastructure/run_augur.sh

# 8. In another terminal, start a perception source
.venv/bin/python sensus/chess_board.py
# or, for system-wide typing (Linux: requires sudo):
sudo .venv/bin/python sensus/typing_monitor.py
```

### Fully containerized mode

```bash
docker compose -f docker-compose.yml -f docker-compose.deploy.yml up
```

All six pipeline faculties (Vigil, Nexus, Consilium, Responsum, Disciplina, Vox) run as containers; Ollama stays on the host (for GPU access). Consilium has a healthcheck-gated `depends_on` from Nexus.

## Configuration

All faculties read from `AugurConfig` (`tabula/config.py`), a frozen dataclass with ~30 fields. Any field can be overridden via environment variables using the `AUGUR_` prefix:

```bash
export AUGUR_NATS_URL=nats://remotehost:4222
export AUGUR_REDIS_HOST=redis.internal
export AUGUR_OLLAMA_URL=http://host.docker.internal:11434
export AUGUR_OLLAMA_MODEL=llama3.2:3b
export AUGUR_DEFAULT_SIGMA_THRESHOLD=2.5
```

There are no hardcoded connection strings anywhere in the codebase.

## Adding a new perception domain

The design rule is: **a new perception source must require zero changes to Vigil or Nexus**. In practice:

1. Create `sensus/<your_source>.py` that publishes `PerceptionEvent`s to `augur.sensus.<your_domain>`.
2. In `consilium/advisor.py`, register a prompt builder in `DOMAIN_HANDLERS` and a one-line summary function in `DOMAIN_DESCRIBERS`. A test (`tests/test_advisor_activity.py::test_domain_handlers_and_describers_keys_match`) pins the two registries to identical keys so future drift is caught.
3. Optionally extend `vox/console_display.py` with a domain-aware branch in `render_anomaly_line` / `render_advice` so anomalies surface human-meaningful values instead of the generic fallback.

Vigil picks up the new domain automatically (wildcard NATS subscription). Nexus will start finding cross-domain patterns as soon as two domains emit anomalies inside the same correlation window (default 30s, adaptive per pairwise rule via Disciplina).

## Optional Windows companion: activity perception

`sensus/activity_monitor.py` is a Windows-host daemon that observes the foreground app, its dwell time, and the interaction intensity within it (keystrokes + mouse activity per 10s window). It publishes two streams — `activity_focus` (per-app dwell) and `activity_intensity` (per-app interaction rate) — to the same NATS instance the rest of the pipeline uses, so cross-domain correlations against chess or typing come for free.

The daemon runs separately from `infrastructure/run_augur.sh` (which is WSL/Linux-side only). It refuses to start without its Win32 deps or without reaching NATS + Redis, and prints explicit remediation hints. It does NOT create sessions — it reads `augur:session:current` from Redis and waits until another perception source (chess_board or typing_monitor) has populated it.

```bash
# On the Windows host (NOT in WSL):
pip install -r requirements-windows.txt
python -m sensus.activity_monitor
```

Window titles are NEVER captured by default. The `AUGUR_ACTIVITY_TITLE_ALLOWLIST` environment variable opts in per app (e.g., `AUGUR_ACTIVITY_TITLE_ALLOWLIST="code,terminal"`); apps not on the allowlist contribute only their executable name to events.

The daemon's NATS publish layer is a best-effort drop log (capacity-bounded `_DroppedEventLog`), not a replay buffer. Events lost during a NATS disconnect are surfaced in logs (`dropped_total`) and discarded — replaying them across session boundaries would contaminate Vigil's receive-time anomaly stamps.

## MCP server

The `augur_mcp` package exposes a FastMCP server with 23 tools covering pipeline lifecycle, event injection, state inspection, escalation-matrix mutation, and gate-silence inspection. Useful for programmatic testing, automated smoke checks, and future autonomous operation without a human in the loop.

## Dependencies

Runtime (`requirements.txt`):

```
python-chess    # Chess rules and move validation (GPL-3.0 — see License note below)
pygame          # Board GUI (LGPL-2.1)
river           # Online machine learning - HalfSpaceTrees (BSD-3)
redis[hiredis]  # Blackboard shared state (MIT)
nats-py         # Message bus client (Apache-2.0)
httpx           # Async HTTP client for Ollama (BSD-3)
keyboard        # System-wide keypress capture for typing_monitor (MIT)
fastmcp         # MCP server framework (Apache-2.0)
networkx        # Session correlation DiGraph (BSD-3)
```

Optional Windows-host companion (`requirements-windows.txt` — install only on the machine running `sensus/activity_monitor.py`):

```
pywin32         # Win32 API access for active-window detection (PSF/BSD-style)
psutil          # Process introspection (BSD-3)
pynput          # Global keyboard/mouse listeners (LGPL-3.0)
```

Dev:

```
pytest, pytest-asyncio
```

## License

The Augur codebase is licensed under **MIT** — see [`LICENSE`](LICENSE).

**Important note on the chess perception module and `python-chess`:**

- `sensus/chess_board.py` imports `python-chess`, which is licensed under **GPL-3.0**.
- The rest of the Augur codebase (Vigil, Nexus, Consilium, Limen, Disciplina, the MCP server, persistence, Tabula, typing monitor, Vox, tests, infrastructure) does **not** depend on `python-chess` and is cleanly MIT.
- If you redistribute a combined work that includes `chess_board.py` together with `python-chess`, the GPL-3.0 terms of `python-chess` may apply to that combined work under a conservative reading of the GPL. This is a longstanding grey area for Python's dynamic imports and has not historically been enforced against hobby projects, but it is worth knowing.
- **If you need a strictly MIT codebase**, simply exclude `sensus/chess_board.py` from your build — the typing monitor and your own perception sources are unaffected and the rest of the system works without it.
- For personal use, research, and non-redistributed deployments, none of this is a practical concern.

## Acknowledgments

Augur was built as a personal research project with substantial AI-assisted development using [Claude Code](https://claude.com/claude-code) (Anthropic). Architecture decisions, design direction, code review, and refactoring were driven by the author; implementation was iteratively produced and reviewed through extended Claude Code sessions. The MIT license reflects permissions granted by the author over their directed contributions.

## Security notes

- Redis and NATS ports (`6379`, `4222`, `8222`) are bound to `127.0.0.1` in `docker-compose.yml`. **Do not rebind them to `0.0.0.0`** without adding authentication — the NATS monitoring port discloses the full subscription topology.
- The `keyboard` library captures all system-wide keypresses when the typing monitor is running. This is a personal-use pattern; do not run it on a shared or public machine.
- All MCP tool inputs are validated against a strict allowlist (`^[a-z0-9_]{1,64}$` for labels) and bounded length caps for escalation matrix keys.
- Session-scoped Redis keys (feedback, correlation graphs, reflection reports) have a 30-day TTL. Long-lived keys are only those that represent persistent state (baselines, prompts, thresholds).

## Status and roadmap

**Shipped:**
- Phases 1–2 foundation (chess perception, anomaly detection, Ollama advisor, console display)
- Phase 2 generic architecture (PerceptionEvent contract, PersistenceManager, typing monitor as second domain)
- Phase 2 self-improvement (feedback collection, reflection with precision/utility/counterfactual analysis)
- Phase 2.5 infrastructure (AugurConfig, MCP server, Docker dual-mode, integration test framework)
- **Phase 3 symbolic reasoning** — NetworkX knowledge graph, escalation matrix symbolic rules, cross-domain correlation, self-tuning via EWMA confidence. Live Ollama verification confirmed cross-domain reasoning produces qualitatively richer advice than per-signal alone.
- **Phase 3 polish** — N-way correlation with default 3-way rules, adaptive per-rule correlation windows tuned by EWMA over observed lag, per-domain feedback attribution with 1/N weighting, atomic state persistence (single MULTI/EXEC pipeline for matrix + state), unified tuning marker that only commits when all writes succeed.
- **Workstation activity perception** — two new perception domains (`activity_focus`, `activity_intensity`) emitted by an optional Windows-host daemon, validating that the domain-agnostic plumbing generalizes beyond timing-of-an-event-stream domains.
- **The stay-silent gate (Limen)** — a multi-arm biological gate that decides suppress/fire/downgrade before each LLM call, with hard safety invariants and an offline self-audit that tunes it from outcomes.
- **Causal-measurement outcome metric** — a domain-agnostic surprise-reduction proximal-outcome metric (σ-space return-to-baseline) with a calibration-era control arm, replacing the chess-only behavioral score.
- **The Great Renaming** — the codebase is organized as the named-faculty pantheon described above (a behavior-preserving refactor; the decentralized blackboard is unchanged).

**In progress / open:**
- Cross-session pattern mining — per-session graphs are persisted but not yet queried across sessions.
- 4-way and higher correlation rules — matrix supports it structurally; defaults ship up to 3-way; adaptive windows are pairwise-only.
- Gate and adaptive-window defaults need calibration against ~10 real sessions before tuning further.

**Future faculties:**
- **Memoria** — a hot/warm/cold knowledge store with long-term cross-session memory.
- OS-level signals as a separate `system` domain (CPU/memory pressure, network bursts, screen-lock).
- Biometric signals as a separate `biometric` domain (webcam attention, mic ambient, wearables).
- Self-modification beyond parameters and prompts (symbolic rule mutation with rollback).
