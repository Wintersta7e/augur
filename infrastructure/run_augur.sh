#!/usr/bin/env bash
# Start all Augur backend components in the correct order.
# Chess board (sensus/chess_board.py) is started separately by the user.
#
# Optional Windows companion (not started by this script):
#   On the Windows host (NOT in WSL), install requirements-windows.txt
#   and run:
#     python -m sensus.activity_monitor
#   It will publish to WSL NATS via localhost:4222 and read the current
#   session from Redis. Make sure a perception source that calls
#   session_mgr.start() (chess_board or typing_monitor) has already
#   populated augur:session:current — the activity monitor reads, does
#   not create, the session.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV="$PROJECT_DIR/.venv"
PYTHON="$VENV/bin/python"
LOG_DIR="$PROJECT_DIR/logs"

# Colors
CYAN='\033[96m'
GREEN='\033[92m'
YELLOW='\033[93m'
GRAY='\033[90m'
BOLD='\033[1m'
RESET='\033[0m'

# ---------------------------------------------------------------------------

if [ ! -f "$PYTHON" ]; then
	echo -e "${YELLOW}Virtual environment not found at $VENV${RESET}"
	echo "Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
	exit 1
fi

mkdir -p "$LOG_DIR"

# Clean up children on exit
PIDS=()
cleanup() {
	echo ""
	echo -e "${GRAY}Stopping Augur components...${RESET}"
	for pid in "${PIDS[@]}"; do
		kill "$pid" 2>/dev/null || true
	done
	wait 2>/dev/null
	echo -e "${GRAY}All components stopped.${RESET}"
}
trap cleanup EXIT INT TERM

echo -e "${CYAN}${BOLD}"
echo "  Starting Augur pipeline..."
echo -e "${RESET}"

# 1. Anomaly detector (must be first — listens for chess moves)
echo -ne "  [1/6] Anomaly detector ...  "
$PYTHON "$PROJECT_DIR/vigil/anomaly_detector.py" \
	>"$LOG_DIR/anomaly_detector.log" 2>&1 &
PIDS+=($!)
sleep 1
if kill -0 "${PIDS[-1]}" 2>/dev/null; then
	echo -e "${GREEN}started${RESET}  (PID ${PIDS[-1]})"
else
	echo -e "\033[91mFAILED${RESET}  — check $LOG_DIR/anomaly_detector.log"
	exit 1
fi

# 2. Correlator (cross-domain correlation — must be between detector and advisor)
echo -ne "  [2/6] Correlator        ...  "
$PYTHON "$PROJECT_DIR/reasoning/correlator.py" \
	>"$LOG_DIR/correlator.log" 2>&1 &
PIDS+=($!)
sleep 1
if kill -0 "${PIDS[-1]}" 2>/dev/null; then
	echo -e "${GREEN}started${RESET}  (PID ${PIDS[-1]})"
else
	echo -e "\033[91mFAILED${RESET}  — check $LOG_DIR/correlator.log"
	exit 1
fi

# 3. Augur advisor (multi-domain LLM advisor — listens for correlation events)
echo -ne "  [3/6] Augur advisor     ...  "
$PYTHON "$PROJECT_DIR/reasoning/augur_advisor.py" \
	>"$LOG_DIR/augur_advisor.log" 2>&1 &
PIDS+=($!)
sleep 1
if kill -0 "${PIDS[-1]}" 2>/dev/null; then
	echo -e "${GREEN}started${RESET}  (PID ${PIDS[-1]})"
else
	echo -e "\033[91mFAILED${RESET}  — check $LOG_DIR/augur_advisor.log"
	exit 1
fi

# 4. Feedback collector (listens for advice + perception events)
echo -ne "  [4/6] Feedback collector...  "
$PYTHON "$PROJECT_DIR/perception/feedback_collector.py" \
	>"$LOG_DIR/feedback_collector.log" 2>&1 &
PIDS+=($!)
sleep 1
if kill -0 "${PIDS[-1]}" 2>/dev/null; then
	echo -e "${GREEN}started${RESET}  (PID ${PIDS[-1]})"
else
	echo -e "\033[91mFAILED${RESET}  — check $LOG_DIR/feedback_collector.log"
	exit 1
fi

# 5. Reflection engine (triggers at end of session)
echo -ne "  [5/6] Reflection engine ...  "
$PYTHON "$PROJECT_DIR/reasoning/reflection_engine.py" \
	>"$LOG_DIR/reflection_engine.log" 2>&1 &
PIDS+=($!)
sleep 1
if kill -0 "${PIDS[-1]}" 2>/dev/null; then
	echo -e "${GREEN}started${RESET}  (PID ${PIDS[-1]})"
else
	echo -e "\033[91mFAILED${RESET}  — check $LOG_DIR/reflection_engine.log"
	exit 1
fi

# 6. Console display (runs in foreground — output goes to terminal)
echo -e "  [6/6] Console display   ...  ${GREEN}starting (foreground)${RESET}"
echo ""
echo -e "${GREEN}${BOLD}  All components running.${RESET}"
echo ""
echo -e "${GRAY}  Logs: $LOG_DIR/${RESET}"
echo -e "${GRAY}  PIDs: ${PIDS[*]}${RESET}"
echo ""
echo -e "  Start perception sources in other terminals:"
echo -e "  ${CYAN}$PYTHON $PROJECT_DIR/sensus/chess_board.py${RESET}"
echo -e "  ${CYAN}sudo $PYTHON $PROJECT_DIR/sensus/typing_monitor.py${RESET}"
echo ""
echo -e "${GRAY}  Press Ctrl+C to stop all components.${RESET}"
echo ""

# Run console display in foreground (blocks until Ctrl+C)
$PYTHON "$PROJECT_DIR/vox/console_display.py"
