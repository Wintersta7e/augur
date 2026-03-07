# Augur

A hybrid neurosymbolic AI system that combines neural perception with symbolic reasoning to detect, interpret, and respond to complex patterns in streaming data.

Currently implemented as a **chess timing analyzer** -- a playable two-player chess board that captures move timing data, detects anomalous think times using online machine learning, and triggers LLM-powered strategic advice when players struggle.

## Architecture

Augur uses a **blackboard architecture** where specialized subsystems communicate through shared state (Redis) and message passing (NATS).

```
  Chess Board (pygame)
        |
        | NATS: augur.perception.chess
        v
  Anomaly Detector (River HalfSpaceTrees + EWMA)
        |
        | NATS: augur.detection.anomaly
        +----------------------------+
        v                            v
  Chess Advisor             Console Display
  (Ollama / qwen2.5:32b)   (low severity alerts)
        |
        | NATS: augur.reasoning.advice
        v
  Console Display (full advice blocks)
```

### Subsystems

| Directory        | File                  | Purpose                                                      |
|------------------|-----------------------|--------------------------------------------------------------|
| `perception/`    | `chess_board.py`      | Playable two-player chess board, captures move timing data    |
| `detection/`     | `anomaly_detector.py` | Online anomaly detection using EWMA baselines + HalfSpaceTrees |
| `reasoning/`     | `chess_advisor.py`    | LLM-powered chess analysis triggered by anomalies            |
| `output/`        | `console_display.py`  | Color-coded terminal display of detections and advice         |
| `blackboard/`    | --                    | Shared state (Redis) and event coordination (NATS)           |
| `infrastructure/`| `run_augur.sh`, `test_connections.py` | Launcher script, connection tests          |

### Data Flow

| NATS Subject              | Payload                                                    | Producer   | Consumer(s)         |
|---------------------------|------------------------------------------------------------|------------|---------------------|
| `augur.perception.chess`  | player, move_uci, move_san, think_time_seconds, move_number | chess_board | anomaly_detector   |
| `augur.detection.anomaly` | player, move, think_time, deviation_score, severity        | anomaly_detector | chess_advisor, console_display |
| `augur.reasoning.advice`  | player, move, advice, severity, model, latency_ms          | chess_advisor | console_display   |

### Redis Keys

| Key                          | Type   | Purpose                              |
|------------------------------|--------|--------------------------------------|
| `augur:chess:last_move`      | String | Last move JSON                       |
| `augur:chess:move_history`   | List   | Rolling last 20 moves                |
| `augur:detection:last_anomaly` | String | Most recent anomaly event          |
| `augur:reasoning:last_advice`  | String | Most recent LLM advice             |

## Prerequisites

- Python 3.12+
- Docker (for Redis and NATS)
- [Ollama](https://ollama.com) with `qwen2.5:32b` pulled
- Ollama must be running on Windows (not inside WSL). The advisor connects via `http://host.docker.internal:11434` by default. Override with `OLLAMA_URL` env var if needed.

## Getting Started

```bash
# 1. Start infrastructure services
docker compose up -d

# 2. Verify Redis and NATS are reachable
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python infrastructure/test_connections.py

# 3. Pull the LLM model (if not already available)
ollama pull qwen2.5:32b

# 4. Start the Augur pipeline (detector + advisor + display)
infrastructure/run_augur.sh

# 5. In a second terminal, start the chess board
.venv/bin/python perception/chess_board.py
```

## Usage

### Chess Board Controls

| Input  | Action            |
|--------|-------------------|
| Click  | Select / move piece |
| `R`    | Reset board       |
| `U`    | Undo last move    |

### How Detection Works

- Each player's think times are tracked independently using an EWMA (exponentially weighted moving average) for mean and variance
- After 3 moves per player, new moves are scored for anomaly
- Two detection methods run in parallel:
  - **Statistical**: flags moves deviating >= 2 standard deviations from the player's baseline
  - **ML**: River's HalfSpaceTrees online anomaly detector (scores 0-1)
- Severity levels: **low** (mild), **medium** (significant), **high** (extreme)

### How Advice Works

- Only **medium** and **high** severity anomalies trigger an LLM query (conserving resources)
- The advisor enriches the anomaly with board context from Redis (move history, current position)
- Ollama generates detailed strategic analysis of the position and situation
- A concurrency lock prevents piling up requests during slow LLM responses

## Dependencies

```
python-chess    # Chess logic and move validation
pygame          # Board GUI
river           # Online machine learning (HalfSpaceTrees)
redis           # Blackboard shared state
nats-py         # Message bus client
httpx           # Async HTTP client for Ollama
```

## License

MIT
