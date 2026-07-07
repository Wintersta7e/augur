# scripts

Helper scripts for Augur development & live testing.

| Script | What it does | How to run |
|---|---|---|
| `inject_and_observe.py` | Inject synthetic typing/activity perception events into a running pipeline and observe the **forward chain** (anomaly → correlate → advice). Forward-chain shakeout. | `.venv/bin/python scripts/inject_and_observe.py [--llm-wait 150]` |
| `complete_loop_test.py` | Drive the **complete all-faculties loop** (forward chain + explicit feedback + session.end + reflection + Memoria sweep + Imperator II proposal) and score every faculty PASS/FAIL, incl. a Praefectus health readout and a best-effort Limen SUPPRESS attempt. | `.venv/bin/python scripts/complete_loop_test.py` |
| `stress_soak.py` | **Stress / scale / soak** driver: a high-rate event burst across many entities (throughput + baseline-model scale), then a sustained soak over a hot pool. Pair with `docker stats` sampling for leak detection. | `.venv/bin/python scripts/stress_soak.py [--soak-s 180]` |
| `teaching_session.py` | **Dialogue teaching-session driver**: exercise the Imperator III dialogue engine through scripted teaching arcs (fact→taught, directive→dropped→fire→undo), scoring persisted state + NATS dialogue events. `--stub-llm` injects a canned-JSON query_fn (full arc flow on live Redis+NATS, no Ollama needed); default is real Ollama, where intent classification can vary run-to-run. | `.venv/bin/python scripts/teaching_session.py [--run-id tag] [--stub-llm] [--dry-run]` |
| `taught_e2e_test.py` | **Taught-knowledge downstream effects**: fact→Consilium injection block, server-authoritative directive predicates, directive suppress/undo/scope/downgrade against the live gate, staleness refusal. Stubbed intents, real faculties. | `.venv/bin/python scripts/taught_e2e_test.py [--llm-wait 150]` |
| `gate_probe_test.py` | **Limen gate-arm probes**: reservoir Schmitt, absolute refractory, habituation under repetition + anti-starvation release (invariant D), high+correlated exemption (invariant B), silence-record persistence + MRT fields (invariant A). Best on a freshly flushed stack. | `.venv/bin/python scripts/gate_probe_test.py [--llm-wait 150]` |
| `mcp_sweep.py` | **MCP surface sweep**: drives all 31 augur_mcp tools over real stdio — happy paths against live pipeline state + validation edges; escalation matrix roundtrip is self-restoring. `--flush` really wipes augur:* as the final step; `--skip-dialogue` avoids Ollama. | `.venv/bin/python scripts/mcp_sweep.py [--flush] [--skip-dialogue]` |
| `armed_apply_test.py` | **Imperator II armed-apply rehearsal**: scratch-target apply machinery under `imperator_ii_apply_enabled` (anchors, dedupe, prompt-safety, klass gating) + an optional containerized armed cycle via `docker-compose.arm.yml`, fully restored + disarmed afterwards. | `.venv/bin/python scripts/armed_apply_test.py [--skip-container]` |

All four connect to the **deploy stack** (or dev pipeline): NATS on `127.0.0.1:4222`, Redis on
`127.0.0.1:6379`, with Ollama (`qwen2.5:32b`) reachable by the faculty containers via
`host.docker.internal` (from WSL, export `AUGUR_OLLAMA_URL=http://<default-gateway-ip>:11434`).
Use run-unique entity/session IDs so each run gets a fresh baseline or dialogue context.
For a clean, fully-attributable run, flush Redis and restart the faculty containers first.
