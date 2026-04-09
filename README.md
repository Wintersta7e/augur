# Augur

A hybrid neurosymbolic AI system that combines neural perception with symbolic reasoning to detect, interpret, and respond to complex patterns in streaming behavioral data. Augur detects anomalies across independent perception domains, correlates signals that fire together, asks a local LLM for advice, collects feedback, and tunes its own parameters after every session.

> **Status:** active personal research project. Currently validated with two independent perception domains (chess move timing and system-wide typing rhythm) sharing the same detection, correlation, and reasoning pipeline. Cross-domain reasoning verified end-to-end against a local Ollama `qwen2.5:32b` model.

## What makes it different

- **Domain-agnostic detection** — adding a new perception source requires zero changes to the detector. Any publisher that emits `PerceptionEvent` to a `augur.perception.<domain>` NATS subject is picked up automatically.
- **Cross-domain correlation** — a rule-based escalation matrix combines signals from different domains inside a rolling 30-second window. Two low-severity signals firing together can escalate to medium or high severity and trigger LLM advice that references the *combination* rather than either signal alone.
- **Self-tuning escalation rules** — per-rule EWMA confidence with hysteresis. Rules that consistently produce useful advice stay; rules that repeatedly miss have their confidence decayed toward their pre-mutation state. The reflection engine updates the escalation matrix in Redis; the correlator reloads it on every event without restart.
- **Self-improvement loop** — after each session, the reflection engine runs four analysis passes (precision, utility, counterfactual, correlation tuning) and adjusts sigma thresholds, mutates LLM prompts via Ollama, and updates escalation confidence.
- **Blackboard architecture** — Redis holds durable state, NATS carries events between components. Six independent subsystems, each testable in isolation.

## Architecture

```
  perception/* (chess, typing, ...)
         │
         │ NATS: augur.perception.<domain>  (wildcard)
         ▼
  detection/anomaly_detector.py          EWMA + River HalfSpaceTrees
         │
         │ NATS: augur.detection.anomaly
         ▼
  reasoning/correlator.py                Redis sorted-set window + NetworkX DiGraph
         │                                escalation matrix lookup
         │ NATS: augur.correlation.detected
         ▼
  reasoning/augur_advisor.py             Ollama qwen2.5:32b
         │                                cross-domain prompt construction
         │ NATS: augur.reasoning.advice
         ▼
  perception/feedback_collector.py       explicit + behavioral scoring
         │
         │ NATS: augur.feedback.complete
         ▼
  reasoning/reflection_engine.py         4-pass analysis → parameter tuning
         │
         │ NATS: augur.reflect.complete
         ▼
  output/console_display.py              ANSI renderer with dedup
```

At session end, the correlator flushes the in-memory NetworkX DiGraph to Redis for later cross-session analysis. All live state is queryable and mutable via a 21-tool FastMCP server.

## Components

| Directory | Purpose |
|---|---|
| `blackboard/` | Shared state layer: `AugurConfig` (env-var config), `PerceptionEvent` contract, `SessionManager`, `PersistenceManager` (all Redis I/O) |
| `perception/` | Input sources publishing to `augur.perception.<domain>`. Includes `chess_board.py`, `typing_monitor.py`, `feedback_collector.py` |
| `detection/` | Domain-agnostic anomaly detector (EWMA + River HalfSpaceTrees, wildcard NATS subscription) |
| `reasoning/` | `correlator.py` (cross-domain), `augur_advisor.py` (Ollama LLM), `reflection_engine.py` (self-improvement) |
| `output/` | ANSI terminal display with domain-scoped dedup and correlation rendering |
| `augur_mcp/` | FastMCP server with 21 tools (lifecycle, injection, inspection, control) |
| `infrastructure/` | Launcher script (6-slot pipeline), connectivity and persistence smoke tests |
| `tests/` | 302 tests total — 278 unit (mocked) + 21 fast integration (real Redis/NATS) + 3 slow (real Ollama) |

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
.venv/bin/python perception/chess_board.py
# or, for system-wide typing (Linux: requires sudo):
sudo .venv/bin/python perception/typing_monitor.py
```

### Fully containerized mode

```bash
docker compose -f docker-compose.yml -f docker-compose.deploy.yml up
```

All six pipeline components run as containers; Ollama stays on the host (for GPU access). The correlator service has a healthcheck-gated `depends_on` from the advisor.

## Configuration

All components read from `AugurConfig` (`blackboard/config.py`), a frozen dataclass with ~30 fields. Any field can be overridden via environment variables using the `AUGUR_` prefix:

```bash
export AUGUR_NATS_URL=nats://remotehost:4222
export AUGUR_REDIS_HOST=redis.internal
export AUGUR_OLLAMA_URL=http://host.docker.internal:11434
export AUGUR_OLLAMA_MODEL=llama3.2:3b
export AUGUR_DEFAULT_SIGMA_THRESHOLD=2.5
```

There are no hardcoded connection strings anywhere in the codebase.

## Adding a new perception domain

The design rule is: **a new perception source must require zero changes to detection or reasoning**. In practice:

1. Create `perception/<your_source>.py` that publishes `PerceptionEvent`s to `augur.perception.<your_domain>`
2. Add a `describe_signal` case in `reasoning/correlator.py` (one-liner formatter)
3. Add a `DOMAIN_HANDLERS` entry in `reasoning/augur_advisor.py` (domain-specific prompt prefix)

The detector picks up the new domain automatically (wildcard NATS subscription). The correlator will start finding cross-domain patterns as soon as two domains emit anomalies inside the same 30s window.

## MCP server

The `augur_mcp` package exposes a FastMCP server with 21 tools covering pipeline lifecycle, event injection, state inspection, and escalation-matrix mutation. Useful for programmatic testing, automated smoke checks, and future autonomous operation without a human in the loop.

## Dependencies

Runtime:

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

Dev:

```
pytest, pytest-asyncio
```

## License

The Augur codebase is licensed under **MIT** — see [`LICENSE`](LICENSE).

**Important note on the chess perception module and `python-chess`:**

- `perception/chess_board.py` imports `python-chess`, which is licensed under **GPL-3.0**.
- The rest of the Augur codebase (detection, correlator, advisor, reflection engine, MCP server, persistence, blackboard, typing monitor, console display, tests, infrastructure) does **not** depend on `python-chess` and is cleanly MIT.
- If you redistribute a combined work that includes `chess_board.py` together with `python-chess`, the GPL-3.0 terms of `python-chess` may apply to that combined work under a conservative reading of the GPL. This is a longstanding grey area for Python's dynamic imports and has not historically been enforced against hobby projects, but it is worth knowing.
- **If you need a strictly MIT codebase**, simply exclude `perception/chess_board.py` from your build — the typing monitor and your own perception sources are unaffected and the rest of the system works without it.
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
- Phase 2 self-improvement (feedback collector, reflection engine with precision/utility/counterfactual analysis)
- Phase 2.5 infrastructure (AugurConfig, MCP server, Docker dual-mode, integration test framework)
- **Phase 3 symbolic reasoning** — NetworkX knowledge graph, escalation matrix symbolic rules, cross-domain correlation, self-tuning via EWMA confidence. Live Ollama verification confirmed cross-domain reasoning produces qualitatively richer advice than per-signal alone.

**In progress / open:**
- Three-domain (and higher) correlation — matrix supports it structurally, `correlate()` is pairwise-only
- Cross-session pattern mining — per-session graphs are persisted but not yet queried across sessions (may be subsumed by Phase 6)
- Adaptive correlation window
- Multi-domain feedback attribution

**Future phases:**
- Phase 4 — richer behavioral inference (session fingerprinting, longitudinal modeling)
- Phase 5 — additional perception domains (code editing, application focus)
- Phase 6 — Hot/Warm/Cold knowledge store with long-term cross-session memory
- Phase 7 — self-modification beyond parameters and prompts (symbolic rule mutation with rollback)
