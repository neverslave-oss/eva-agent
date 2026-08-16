#!/usr/bin/env bash
cd "$(dirname "$0")"
LOG="sim6_output_$(date +%Y%m%d_%H%M%S).log"
PIDFILE="sim6.pid"
TIMEOUT=900  # 15 minutes

echo "Starting Sim6 simulation (timeout ${TIMEOUT}s)..." > "$LOG"
echo "Timestamp: $(date)" >> "$LOG"

timeout $TIMEOUT python3 sim6/run_sim6.py >> "$LOG" 2>&1 &
PID=$!
echo $PID > "$PIDFILE"

echo "Sim6 started with PID $PID, logging to $LOG"
echo "Follow with: tail -f $LOG"
echo "PID saved to $PIDFILE"

# Wait a moment and show initial output
sleep 3
tail -20 "$LOG"