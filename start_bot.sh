#!/usr/bin/env bash
# kernel-evolving Telegram bot startup
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOT_LOG="/tmp/kernel_evolving_bot.log"

echo "[kernel-evolving-bot] Stopping previous bot..."
pkill -f "python3 telegram_bot.py" 2>/dev/null || true
sleep 1

# Source .env (same as API)
if [[ -f "$REPO/.env" ]]; then
    set -a; source "$REPO/.env"; set +a
    echo "[kernel-evolving-bot] env loaded from .env"
else
    echo "[kernel-evolving-bot] ERROR: no .env found"
    exit 1
fi

cd "$REPO/src"
nohup python3 telegram_bot.py >> "$BOT_LOG" 2>&1 &
BOT_PID=$!
echo "[kernel-evolving-bot] Bot started (PID $BOT_PID) — log: $BOT_LOG"

sleep 2
if tail -5 "$BOT_LOG" 2>/dev/null | grep -q "Polling"; then
    echo "[kernel-evolving-bot] ✅ Bot polling"
else
    echo "[kernel-evolving-bot] ❌ Bot may have failed — check $BOT_LOG"
    exit 1
fi