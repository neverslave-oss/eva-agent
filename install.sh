#!/usr/bin/env bash
# kernel-evolving — Self-evolving sandbox companion
# Standalone self-evolving agent. Works independently or alongside kernel base.
set -euo pipefail

INSTALL_DIR="${PWD}"
echo ""
echo "  🧬 Kernel-evolving — Self-Evolving Agent"
echo "  v1.8.3-evolving"
echo ""
echo "  Prerequisites:"
echo "    ✅ Python 3.11+ (required)"
echo "    ✅ Python 3.11+"
echo "    ✅ CUDA GPU 8GB+ VRAM (shared model server)"
echo "    ✅ OpenAI API key (for Tier 2 skill synthesis)"
echo ""

command -v python3 >/dev/null 2>&1 || { echo "❌ Python 3.11+ required"; exit 1; }

# Check kernel base is running
if curl -sf http://localhost:8769/health > /dev/null 2>if ! curl -sf http://localhost:8769/health > /dev/null 2>&1; then1; then
  echo "  ℹ️  kernel base detected on :8769 — peer delegation features will be available"
  echo "    Continuing anyway (kernel-evolving can run standalone)..."
fi

# Python deps
echo "📦 Installing Python dependencies..."
python3 -m venv .venv 2>/dev/null || true
source .venv/bin/activate 2>/dev/null || true
pip install -q -r requirements.txt

# .env — symlink to kernel base if present, otherwise create
if [ ! -f ".env" ]; then
  KERNEL_ENV="${KERNEL_DIR:-$HOME/.kernel-agent}/.env"
  if [ -f "$KERNEL_ENV" ]; then
    ln -sf "$KERNEL_ENV" .env
    echo "  ✅ .env linked from kernel base"
  else
    cat > .env << 'EOF'
KERNEL_EVO_TELEGRAM_BOT_TOKEN=your_bot_token_here
KERNEL_EVO_TELEGRAM_CHAT_ID=your_telegram_chat_id_here
OPENAI_API_KEY=your_openai_key_here
EVOLUTION_ENABLED=true
EOF
    echo "  ⚙️  Edit .env — add Telegram token + OpenAI API key"
  fi
fi

echo ""
echo "  ✅ kernel-evolving ready"
echo ""
echo "  Start the sandbox:"
echo "    bash start.sh"
echo ""
echo "  Then from kernel base Telegram bot:"
echo "    /evolve          → evolution menu"
echo "    /evolve status   → current state"
echo "    /evolve task <t> → trigger one evolution cycle"
echo ""
