# Augur

**A local-first neurosymbolic companion that watches your behavioral signals and speaks only when it matters.**

[![CI](https://github.com/Wintersta7e/augur/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Wintersta7e/augur/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Wintersta7e/augur/graph/badge.svg)](https://codecov.io/gh/Wintersta7e/augur)
[![CodeQL](https://github.com/Wintersta7e/augur/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/Wintersta7e/augur/actions/workflows/codeql.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3120/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Status](https://img.shields.io/badge/status-personal%20%C2%B7%20actively%20developed-brightgreen)](#status)

## Why

A single-user ambient companion that learns behavioral rhythms (chess move
timing, typing cadence, which app is focused and how hard it is being worked),
notices when something is off, correlates signals that fire together, and offers
a just-in-time nudge from a local LLM — only when the moment earns the
interruption. Local-first, runs entirely against a local Ollama model, sends no
telemetry.

A personal research project, not a product. MIT-licensed and clonable, with no
adoption goal and no support guarantees; issues and PRs are read.

The system is a **blackboard**: faculties coordinate only through durable Redis
state and a NATS event bus, never direct calls, so a new sense or reasoning step
plugs in without touching the rest. Names are an identity layer over that
blackboard, not central orchestration — **Tabula** (shared slate), **Sensus**
(senses), **Vigil** (the watch), **Nexus** (binding), **Consilium** (counsel),
**Limen** (the threshold), **Responsum** (feedback), **Disciplina** (training),
**Vox** (voice), **Praefectus** (marshal), **Memoria** (memory), **Imperator**
(self-improvement), **Conscientia** (conscience), **Praesagium** (foresight).

## Status

Active personal research project. The full pipeline — perception → detection →
correlation → gated LLM advice → feedback → self-tuning — is implemented and
covered by **2431 unit + 57 fast integration + 5 slow** tests under strict ruff
lints and green CI, verified end-to-end across four perception domains against a
local Ollama `qwen2.5:32b`. Redis key and NATS subject contracts can still shift
between commits: treat it as a working pipeline you can build and run, not a
packaged app.

**Not built:** a vision sense (**Visus**), Praefectus *output arbitration* +
lifecycle (its supervision/health tier is built), rule induction over mined
patterns, packaged installers.

**Explicitly declined:** multi-user/team mode, cloud sync, telemetry, hosted
surface.

## What's implemented

### Perception & detection
- **Sensus** — chess move timing, system-wide typing rhythm, and an optional
  Windows daemon for per-app focus dwell + interaction intensity.
- **Vigil** — domain-agnostic EWMA detection on a wildcard NATS subscription; a
  new sense needs zero detector changes. A baseline is scoped to one measurement
  **series** — `(domain, event_type, entity)` — and records the unit it was
  trained in, refusing a mismatch: one entity may publish several streams on
  different scales. Sigma thresholds are calibrated against the estimator's
  measured null, not a z-table, because the EWMA variance is t-like at the
  configured alpha. River supplies the ADWIN/Page-Hinkley drift detector that
  triggers deliberate baseline resets.

### Correlation & escalation
- **Nexus** — Redis sorted-set window + NetworkX session graph + a
  runtime-loadable escalation matrix, with pairwise and 3-way rules. Per-rule
  EWMA confidence with hysteresis and adaptive per-rule windows learned from
  observed lag. Two anomalies sharing a sensor's `span_id` do not correlate:
  one sensor tick emitting twice is not two detectors agreeing.

### The stay-silent gate (Limen)
- A multi-arm biological gate decides **suppress / fire / downgrade** before any
  LLM call — habituation, refractory burden, credibility, reservoir/rate-limit,
  and more.
- Hard invariants: never silence a high-severity correlated event; fail **open**
  to firing on any error; never silence a trackable channel forever.

### Memory (Memoria)
- Hot/Warm/Cold tiering with an FSRS forgetting curve on an **active-session
  clock** — a week away erases nothing; only sessions count.
- Recurring patterns consolidate toward Cold; one-offs fade and are **archived,
  never hard-deleted**, by a session-end sweep.

### Anticipation (Praesagium)
- Records a compact episode stream per session, then mines ordered "A precedes B
  within W" pairs offline during reflection.
- **Honest promotion** — cross-session support, a Wilson lower bound on
  confidence, lift over a *session-conditional* null (so "both happen when you're
  at the desk" is rejected), lag stability, and a probation mine against fresh
  data.
- **Self-verifying** — every armed prediction resolves exactly once (fulfilled or
  expired), so each pattern carries a measured hit rate and retires itself when
  behavior drifts. **Speaking is off by default**; a forewarning is a
  deterministic template that traverses the same gate as any advice and can never
  claim the gate's danger exemption.

### Reasoning, feedback & self-improvement
- **Consilium** — local-LLM advice over Ollama, with cross-domain prompts that
  reason about the *combination* of signals rather than any one alone.
- **Responsum** — behavioral scoring with per-domain 1/N attribution, plus
  explicit ratings. The interactive prompt runs only with a TTY; headless, ratings
  arrive on `augur.responsum.feedback` or via the MCP `submit_feedback` tool,
  which defaults to the most recent advice.
- **Disciplina** — seven analysis passes (precision, utility, counterfactual,
  correlation, window, gate, memory) plus the Conscientia review and Praesagium
  mining sweeps, tuning thresholds, prompts, the escalation matrix and the gate
  itself. Runs on a cadence during a session as well as at session end, so a
  session that is killed rather than closed still learns.
- **Praefectus** — heartbeat liveness for every faculty, a conservative
  pipeline-stall signal, an MCP health tool, and degradation alerts through Vox.
- **Imperator** — deterministic self-model and auspices read-models, an LLM
  reasoner over its own measured blind spots emitting ranked proposals, and a
  conversation faculty (ask what it saw and why; teach it corrections). Only a
  safe, reversible class can auto-apply, behind a flag that ships **off**.
- **Conscientia** — five screens that can refuse: advice output, teaching, prompt
  injection, pre-apply, and offline review of anything the self-improvement engine
  wants a human for. Fail directions are deliberate: the output screen fails
  *open* (never silences the pipeline), teach and apply fail *closed*. Its charter
  is code with no write path.

### Interfaces
- **Vox** — ANSI console renderer with domain-scoped dedup; every payload is
  stripped of control, escape and bidirectional characters as it is decoded.
- **`augur_mcp`** — a 36-tool FastMCP server for lifecycle, event injection, state
  inspection, runtime tuning, pipeline health, dialogue, learned patterns and
  predictions, charter/verdict inspection, and explicit advice feedback.
- **Docker dual-mode** — native dev, or fully containerized deploy across 10
  faculty components.

## Stack

- **Python 3.12**, **Redis** (durable blackboard state), **NATS + JetStream**
  (event bus)
- **Ollama** local LLM (`qwen2.5:32b` default, configurable), **River** (online
  drift detection), **NetworkX** (session correlation graphs)
- **FastMCP** (control server), **pytest + fakeredis** (tests), **ruff**
  (lint/format), **Docker** (dev + deploy)

All connection strings and tunables live in `AugurConfig` (`tabula/config.py`),
overridable via `AUGUR_*` environment variables — no hardcoded endpoints anywhere.

## Quick start

```bash
# 1. Start Redis + NATS
docker compose up -d

# 2. Python env + deps
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 3. Pull the local LLM
ollama pull qwen2.5:32b

# 4. Tests (unit needs no infra; fast integration needs Redis + NATS)
.venv/bin/pytest tests/ --ignore=tests/integration
.venv/bin/pytest tests/integration/ -m "not slow"

# 5. Run the pipeline (dev mode)
bash infrastructure/run_augur.sh

# 6. In another terminal, start a perception source
.venv/bin/python sensus/chess_board.py
sudo .venv/bin/python sensus/typing_monitor.py   # system-wide typing (Linux: root)
```

Fully containerized (the ten faculty components run as containers; Ollama stays
on the host for GPU access):

```bash
docker compose -f docker-compose.yml -f docker-compose.deploy.yml up
```

The optional Windows activity daemon (`sensus/activity_monitor.py`) runs on the
Windows host — `pip install -r requirements-windows.txt && python -m sensus.activity_monitor`.

## Layout

```
augur/
├── tabula/        # shared base: AugurConfig, PerceptionEvent contract, sessions,
│                  #   PersistenceManager (ALL Redis I/O)
├── sensus/        # perception sensors → augur.sensus.<domain>
├── vigil/         # domain-agnostic anomaly detector → augur.vigil.anomaly
├── nexus/         # cross-domain correlator → augur.nexus.detected
├── consilium/     # local-LLM advisor (+ app-descriptor classifier)
├── limen/         # the stay-silent gate (runs in-process inside Consilium)
├── responsum/     # feedback collector → augur.responsum.complete
├── disciplina/    # reflection engine → augur.disciplina.complete
├── memoria/       # pure FSRS/tier/sweep logic (no Redis) for the memory spine
├── praefectus/    # faculty supervision/health monitor → augur.praefectus.health
├── imperator/     # self-model, proposals, apply, dialogue → augur.imperator.*
├── conscientia/   # value core: charter-as-code, screens, gated review
├── praesagium/    # anticipation: episodes, pattern miner, prediction matcher
├── vox/           # ANSI console renderer
├── augur_mcp/     # FastMCP control server (36 tools)
├── infrastructure/# run_augur.sh launcher + connection/persistence smoke tests
└── tests/         # 2431 unit (mocked) + 62 integration (real Redis/NATS/Ollama)
```

**Data flow:** `sensus.* → vigil.anomaly → nexus.detected → consilium (+ limen
gate, + the Conscientia output screen) → consilium.advice → responsum.complete →
disciplina.complete → vox`. Disciplina runs its reflection passes on a cadence and
again at session end, when Nexus also flushes its correlation graph to Redis.
**Praesagium** rides the raw anomaly stream, recording episodes and resolving
predictions; when armed, a forewarning enters Consilium on its own subject and
traverses the same gate as any other advice. **Praefectus** rides the whole bus
(`augur.>`) — every faculty heartbeats on `augur.system.heartbeat` and Praefectus
publishes `augur.praefectus.health` liveness/degradation transitions.

## Design principles

1. **Local-first and private.** No telemetry, no cloud, no account.
2. **Decentralized blackboard.** State only through Redis, events only through
   NATS — never direct calls.
3. **Domain-agnostic.** A new sense is a publisher on `augur.sensus.<domain>` plus
   a prompt handler; detection and correlation are unchanged.
4. **Speak rarely, with weight.** The gate would rather stay silent, and fails
   open on error.
5. **Self-tuning, with safety floors.** Reflection adjusts thresholds, prompts,
   the matrix and the gate; hard invariants and floor-protected memories are never
   tuned away.
6. **Watch before you act.** Anything that could act on its own ships switched
   off: self-modification applies nothing until armed, and anticipation scores its
   own predictions before it may speak one.
7. **A conscience that can say no.** Screens sit in front of output, teaching and
   self-edits; the charter has no write path, so the system cannot rewrite what it
   is allowed to become.
8. **Measure, don't assume.** Detector thresholds come from the measured null of
   the estimator actually in use; every learned claim carries the evidence that
   produced it.
9. **Inspectable.** Every baseline, graph, reflection, pattern, prediction outcome
   and gate decision is in Redis and queryable through the MCP server.

## Security notes

- The typing monitor captures **all system-wide keypresses** while running
  (Linux: needs root). Don't run it on a shared or public machine.
- Redis and NATS ports (`6379`, `4222`, `8222`) bind to `127.0.0.1` in
  `docker-compose.yml`. **Don't rebind to `0.0.0.0`** without auth — the NATS
  monitoring port discloses the full subscription topology.
- All MCP tool inputs are validated against a strict allowlist
  (`^[a-z0-9_]{1,64}$`) with bounded length caps.
- NATS subjects are unauthenticated on the loopback bus, so any local process can
  publish one. Payloads are stripped of control, escape and bidirectional
  characters as Vox decodes them, and the anticipation lane — which reaches the
  console without passing through an LLM — rejects them at its entry gate.
- Session-scoped Redis keys (feedback, correlation graphs, reflections) carry a
  30-day TTL; only learned state (baselines, prompts, matrix, memories) is durable.

## License

[MIT](LICENSE) — with one caveat: `sensus/chess_board.py` imports `chess`
(**GPL-3.0**). The rest of the codebase does not depend on it and is cleanly MIT;
exclude that one file for a strictly-MIT build (the typing monitor and your own
perception sources are unaffected). For personal, non-redistributed use this is
not a practical concern.

## Acknowledgments

Augur is a personal research project built with substantial AI-assisted
development via [Claude Code](https://claude.com/claude-code) (Anthropic).
Architecture, design direction, and review are the author's; implementation was
produced and reviewed across extended sessions.
