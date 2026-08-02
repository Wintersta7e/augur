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

I built Augur for myself — a single-user ambient companion that learns my
behavioral rhythms (chess move timing, typing cadence, which app I'm in and how
hard I'm working it), notices when something is off, correlates signals that
fire together, and offers a just-in-time nudge from a local LLM — but only when
the moment is genuinely worth interrupting. It's **local-first**, runs entirely
on my machine against a local Ollama model, and sends **no telemetry**.

It's a personal research project, not a product — but it's open source under
MIT, and if the architecture is useful to you, you're welcome to clone it.
There's no adoption goal and no support guarantees, but issues and PRs are read.

The whole system is a **blackboard**: independent faculties coordinate only
through durable Redis state and a NATS event bus — never direct calls — so a new
sense or a new reasoning step plugs in without touching the rest. The faculties
carry Latin names (an identity/charter layer over the blackboard, not central
orchestration): **Tabula** (the shared slate), **Sensus** (senses), **Vigil**
(the watch), **Nexus** (binding/correlation), **Consilium** (counsel/LLM),
**Limen** (the threshold — the stay-silent gate), **Responsum** (feedback),
**Disciplina** (training/reflection), **Vox** (the voice), **Praefectus** (the
marshal — supervision/health), **Memoria** (memory), **Imperator** (the seat of
self-improvement), **Conscientia** (the conscience — a value core that can refuse),
and **Praesagium** (foresight — anticipation).

## Status

Active personal research project. The full pipeline — perception → detection →
correlation → gated LLM advice → feedback → self-tuning — is implemented and
covered by a large suite (**2245 unit + 53 fast integration + 5 slow**) under
strict ruff lints and green CI. Verified end-to-end with four perception domains
against a local Ollama `qwen2.5:32b`. It's still rough in places and the Redis
key / NATS subject contracts can shift between commits — treat it as a working
pipeline you can build and run, not a finished, packaged app.

**Implemented:**
- The perception→advice pipeline across the faculty pantheon (Sensus → Vigil →
  Nexus → Consilium → Responsum → Disciplina → Vox)
- The biological stay-silent gate (Limen) — suppress / fire / downgrade before
  every LLM call, with hard safety invariants
- Memoria — a Hot/Warm/Cold memory spine with FSRS decay on an active-session
  clock, consolidated by a session-end sweep
- N-way cross-domain correlation with a self-tuning escalation matrix + adaptive
  per-rule windows
- Session-end reflection: seven analysis passes (precision / utility /
  counterfactual / correlation / window / gate / memory) plus the Conscientia
  review and Praesagium mining sweeps
- **Praefectus** (supervision/health) — heartbeat liveness for every faculty, a
  conservative pipeline-stall signal, an MCP health tool, and degradation alerts
  surfaced through Vox
- **Imperator** (self-improvement) — deterministic self-model + auspices
  read-models, an LLM reasoner over its own blind spots that emits ranked
  proposals, and a conversation faculty (ask it what it saw and why; teach it
  corrections). Only a *safe*, reversible class of proposal can auto-apply, and
  applying is **off by default**
- **Conscientia** (the value core) — five screens that can refuse: advice output,
  teaching, prompt injection, pre-apply, and offline review of anything the
  self-improvement engine wants a human for. Its charter is code, not data, with
  no write path
- **Praesagium** (anticipation) — learns cross-session "A precedes B within W"
  patterns, arms predictions, and verifies every one of them against what
  actually happened. Forewarnings are deterministic templates that enter the same
  gated path as any advice and can never claim the gate's danger exemption.
  **Speaking is off by default**: it learns and measures its own hit rate first
- A 36-tool FastMCP control server; Docker dual-mode (native dev or fully
  containerized deploy — 10 faculty components)

**Not done yet:** a vision sense (**Visus**), Praefectus *output arbitration* +
lifecycle (the supervision/health increment is built; arbitration and restart are
deferred), and packaged installers.

## Features

### Perception & detection
- **Domain-agnostic anomaly detection (Vigil)** — EWMA + River HalfSpaceTrees on
  a wildcard NATS subscription; a new sense requires zero detector changes
- **Sensors (Sensus)** — chess move timing, system-wide typing rhythm, and (an
  optional Windows daemon) per-app focus dwell + interaction intensity

### Correlation & escalation
- **N-way correlation (Nexus)** — Redis sorted-set window + a NetworkX session
  graph + a runtime-loadable escalation matrix; pairwise and 3-way default rules
- **Self-tuning** — per-rule EWMA confidence with hysteresis, plus adaptive
  per-rule correlation windows learned from observed lag

### The stay-silent gate (Limen)
- A multi-arm biological gate decides **suppress / fire / downgrade** before any
  LLM call (habituation, refractory burden, credibility, reservoir/rate-limit, …)
- Hard invariants: never silence a high-severity correlated event; fail **open**
  to firing on any error; never silence a trackable channel forever

### Memory (Memoria)
- **Hot/Warm/Cold tiering** with an FSRS forgetting curve on an **active-session
  clock** — a week away doesn't erase memory; only sessions count
- Recurring patterns consolidate toward Cold; one-offs fade and are **archived,
  never hard-deleted**, by a session-end "sleep" sweep

### Anticipation (Praesagium)
- **Cross-session pattern mining** — records a compact episode stream per session,
  then mines ordered "A precedes B within W" pairs offline during reflection
- **Honest promotion** — a pattern is believed only past cross-session support, a
  Wilson lower bound on confidence, lift over a *session-conditional* null (so
  "both happen when you're at the desk" is rejected), lag stability, and a
  probation mine against fresh data
- **Self-verifying** — every armed prediction resolves exactly once (fulfilled or
  expired), so each pattern carries a measured hit rate and retires itself when
  your behavior drifts

### Reasoning, feedback & self-improvement
- **Local-LLM advice (Consilium)** over Ollama, with cross-domain prompts that
  reason about the *combination* of signals, not any one alone
- **Feedback (Responsum)** — explicit (interactive, or headless via the
  `augur.responsum.feedback` subject / MCP `submit_feedback`) + behavioral scoring
  with per-domain 1/N attribution; **reflection (Disciplina)** runs its analysis
  passes per session and tunes thresholds, prompts, the escalation matrix, and the
  gate itself
- **Self-improvement (Imperator)** — reasons over its own measured blind spots and
  proposes changes; only reversible, sanctioned edits can auto-apply, gated behind
  a flag that ships **off**. Everything structural is logged for review
- **A conscience (Conscientia)** — screens that refuse rather than comply: it can
  block its own advice, refuse what you try to teach it, and decline to apply a
  self-modification. Fail directions are deliberate — the output screen fails
  *open* (never silences the pipeline), the teach and apply screens fail *closed*

### Interfaces
- **Vox** — an ANSI console renderer with domain-scoped dedup; every payload is
  stripped of control, escape, and bidirectional characters as it is decoded
- **`augur_mcp`** — a 36-tool FastMCP server for lifecycle, event injection,
  state inspection, runtime tuning, pipeline-health, dialogue, learned patterns
  and predictions, charter/verdict inspection, and explicit advice feedback

### Planned / architected (not built)
- A vision sense (**Visus**), Praefectus *output arbitration* + lifecycle (its
  supervision/health tier is built), and rule induction over the mined patterns

### Explicitly declined (not on the roadmap)
- Multi-user / team mode • Cloud sync • Telemetry • Hosted/SaaS surface • Marketing

## Stack

- **Python 3.12**, **Redis** (durable blackboard state), **NATS + JetStream**
  (event bus)
- **Ollama** local LLM (`qwen2.5:32b` default, configurable), **River** (online
  ML — HalfSpaceTrees), **NetworkX** (session correlation graphs)
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
└── tests/         # 2245 unit (mocked) + 58 integration (real Redis/NATS/Ollama)
```

**Data flow:** `sensus.* → vigil.anomaly → nexus.detected → consilium (+ limen
gate, + the Conscientia output screen) → consilium.advice → responsum.complete →
disciplina.complete → vox`. At session end, Nexus flushes its correlation graph to
Redis and Disciplina runs the reflection passes (including the Memoria
consolidation sweep, the Conscientia review of anything gated, and the Praesagium
pattern mine). **Praesagium** also rides the raw anomaly stream, recording episodes
and resolving predictions; when armed, a forewarning enters Consilium on its own
subject and traverses the same gate as any other advice. **Praefectus** rides the
whole bus (`augur.>`) — every faculty heartbeats on `augur.system.heartbeat` and
Praefectus publishes `augur.praefectus.health` liveness/degradation transitions.

## Design principles

1. **Local-first and private.** Everything runs on your machine against a local
   LLM. No telemetry, no cloud, no account.
2. **Decentralized blackboard.** Faculties share state only through Redis and
   events only through NATS — never direct calls. New faculties plug in cleanly.
3. **Domain-agnostic.** A new perception source is a publisher on
   `augur.sensus.<domain>` plus a prompt handler — zero changes to detection or
   correlation.
4. **Speak rarely, with weight.** The Limen gate would rather stay silent than
   interrupt; it fires only when a moment earns it (and fails open on error).
5. **Self-tuning, with safety floors.** Reflection adjusts thresholds, prompts,
   the escalation matrix, and the gate — but hard invariants and floor-protected
   memories are never tuned away.
6. **Watch before you act.** Every capability that could act on its own arrives
   switched off: self-modification applies nothing until armed, and anticipation
   learns and scores its own predictions before it is allowed to speak one.
7. **A conscience that can say no.** Screens sit in front of the system's output,
   its teaching, and its own self-edits. Its charter lives in code with no write
   path — the system cannot rewrite what it is allowed to become.
8. **Inspectable.** Every baseline, graph, reflection, learned pattern, prediction
   outcome, and gate decision is in Redis and queryable through the MCP server.
   Persistence is centralized.
9. **A named pantheon.** The faculties are an identity/charter layer over the
   blackboard — the architecture stays decentralized; the names give it a spine.

## Security notes

- The typing monitor captures **all system-wide keypresses** while running
  (Linux: needs root). This is a personal-use pattern; don't run it on a shared
  or public machine.
- Redis and NATS ports (`6379`, `4222`, `8222`) bind to `127.0.0.1` in
  `docker-compose.yml`. **Don't rebind to `0.0.0.0`** without auth — the NATS
  monitoring port discloses the full subscription topology.
- All MCP tool inputs are validated against a strict allowlist
  (`^[a-z0-9_]{1,64}$`) with bounded length caps.
- NATS subjects are unauthenticated on the loopback bus, so any local process can
  publish one. Payloads are stripped of control, escape, and bidirectional
  characters as Vox decodes them — a renderer never prints attacker-chosen bytes
  to your terminal — and the anticipation lane, which reaches the console without
  passing through an LLM, additionally rejects them at its entry gate.
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
