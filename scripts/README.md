# scripts

Helper scripts for Augur development & live testing.

| Script | What it does | How to run |
|---|---|---|
| `inject_and_observe.py` | Inject synthetic typing/activity perception events into a running pipeline and observe the **forward chain** (anomaly → correlate → advice). Forward-chain shakeout. | `.venv/bin/python scripts/inject_and_observe.py [--llm-wait 150]` |
| `complete_loop_test.py` | Drive the **complete all-faculties loop** (forward chain + explicit feedback + session.end + reflection + Memoria sweep + Imperator II proposal) and score every faculty PASS/FAIL, incl. a Praefectus health readout and a best-effort Limen SUPPRESS attempt. | `.venv/bin/python scripts/complete_loop_test.py` |
| `stress_soak.py` | **Stress / scale / soak** driver: a high-rate event burst across many entities (throughput + baseline-model scale), then a sustained soak over a hot pool. Pair with `docker stats` sampling for leak detection. | `.venv/bin/python scripts/stress_soak.py [--soak-s 180]` |

All three connect to the **deploy stack** (or dev pipeline): NATS on `127.0.0.1:4222`, Redis on
`127.0.0.1:6379`, with Ollama (`qwen2.5:32b`) reachable by the faculty containers via
`host.docker.internal`. Use run-unique entity names so each run gets a fresh baseline.
For a clean, fully-attributable run, flush Redis and restart the faculty containers first.
