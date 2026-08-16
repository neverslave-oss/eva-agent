#!/usr/bin/env bash
# ============================================================================
# portability_scan.sh
# Recursively scan kernel-evolving codebase for hardcoded paths & references
# that would affect portability to another user/machine.
#
# Output: workspace/<timestamp>-portability-report.md
# ============================================================================
set -o pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%dT%H%M%S)"
REPORT_DIR="${REPO_DIR}/tmp"
mkdir -p "${REPORT_DIR}"
REPORT="${REPORT_DIR}/${TIMESTAMP}-portability-report.md"

# ── helpers ──────────────────────────────────────────────────────────────────
count() { wc -l < "$1" | tr -d ' '; }
heading() { printf "\n## %s\n\n" "$1" >> "$2"; }
subheading() { printf "### %s\n\n" "$1" >> "$2"; }
bullet() { printf -- "- %s\n" "$1" >> "$2"; }
code() { printf '`%s`\n' "$1" >> "$2"; }
sep() { printf "\n---\n" >> "$1"; }
hits_file() { mktemp --tmpdir portability_hits_XXXXXX; }

# ── init report ──────────────────────────────────────────────────────────────
{
  echo "# Portability Scan — ${TIMESTAMP}"
  echo "**Repo:** \`${REPO_DIR}\`"
  echo "**Scanned:** $(date -R)"
  echo ""
  echo "Scans for:"
  echo "- Absolute filesystem paths (user/machine-specific)"
  echo "- User-specific home references (~/, \\\$HOME)"
  echo "- Hardcoded usernames (pacificDev, fabio, etc.)"
  echo "- Hardcoded mount points (/mnt/)"
  echo "- Hardcoded model paths / snapshot hashes"
  echo "- Hardcoded IP addresses (not localhost/127.0.0.1/::1)"
  echo "- Hardcoded port numbers in source (not tests/docs)"
  echo "- Environment variable names that are deployment-specific"
  echo ""
} > "$REPORT"

TOTAL_HITS=0

# ── 1. Absolute paths ────────────────────────────────────────────────────────
heading "Absolute Filesystem Paths" "$REPORT"
subheading "/home/ — user home directories" "$REPORT"
HITS=$(hits_file)
# Exclude common false positives: container paths, pip, npm, venv, etc.
grep -rnI --include='*.py' --include='*.sh' --include='*.yaml' --include='*.yml' \
     --include='*.json' --include='*.toml' --include='*.cfg' --include='*.ini' \
     --include='*.md' --include='Dockerfile' --include='*.txt' \
     -oE '/home/[a-zA-Z0-9._-]+/[^"'"'"'\s:;)\]]+' "${REPO_DIR}/src" \
     2>/dev/null | grep -vE '(\.venv|__pycache__|node_modules|miniconda|\.cache)' \
     | sort -u >> "$HITS"
if [ -s "$HITS" ]; then
  while IFS= read -r line; do
    bullet "${line}" "$REPORT"
    ((TOTAL_HITS++))
  done < "$HITS"
else
  bullet "None found" "$REPORT"
fi
rm -f "$HITS"

subheading "/mnt/ — Windows mount points (WSL specific)" "$REPORT"
HITS=$(hits_file)
grep -rnI --include='*.py' --include='*.sh' --include='*.yaml' --include='*.yml' \
     --include='*.json' --include='*.toml' --include='*.cfg' --include='*.ini' \
     --include='*.md' \
     -oE '/mnt/[a-zA-Z]/[^"'"'"'\s:;)\]]+' "${REPO_DIR}/src" 2>/dev/null \
     | sort -u >> "$HITS"
if [ -s "$HITS" ]; then
  while IFS= read -r line; do
    bullet "${line}" "$REPORT"
    ((TOTAL_HITS++))
  done < "$HITS"
else
  bullet "None found" "$REPORT"
fi
rm -f "$HITS"

# ── 2. Hardcoded usernames ───────────────────────────────────────────────────
heading "Hardcoded Usernames / Identifiers" "$REPORT"
subheading "References to 'pacificDev'" "$REPORT"
HITS=$(hits_file)
grep -rnI --include='*.py' --include='*.sh' --include='*.yaml' --include='*.yml' \
     --include='*.json' --include='*.cfg' --include='*.toml' --include='*.ini' \
     --include='*.md' \
     -i 'pacificDev' "${REPO_DIR}/src" 2>/dev/null | sort -u >> "$HITS"
if [ -s "$HITS" ]; then
  while IFS= read -r line; do bullet "${line}" "$REPORT"; ((TOTAL_HITS++)); done < "$HITS"
else
  bullet "None found" "$REPORT"
fi
rm -f "$HITS"

subheading "References to 'fabio'" "$REPORT"
HITS=$(hits_file)
grep -rnI --include='*.py' --include='*.sh' --include='*.yaml' --include='*.yml' \
     --include='*.json' --include='*.cfg' --include='*.toml' \
     -i -E '(fabio|pacifici)' "${REPO_DIR}/src" 2>/dev/null | grep -vE '(fabiopacifici|github\.com|__pycache__)' | sort -u >> "$HITS"
if [ -s "$HITS" ]; then
  while IFS= read -r line; do bullet "${line}" "$REPORT"; ((TOTAL_HITS++)); done < "$HITS"
else
  bullet "None found" "$REPORT"
fi
rm -f "$HITS"

# ── 3. Hardcoded home dirs in config ─────────────────────────────────────────
heading "Hardcoded Home / Config Directories" "$REPORT"
SUBDIRS=("~/.openclaw" "~/.kernel" "~/.kernel-evolving" "~/.config" "\$HOME")

for dir in "${SUBDIRS[@]}"; do
  subheading "References to \`${dir}\`" "$REPORT"
  HITS=$(hits_file)
  esc_dir="$(echo "${dir}" | sed 's/\./\\./g; s/\$/\\$/g')"
  grep -rnI --include='*.py' --include='*.sh' --include='*.yaml' --include='*.yml' \
       --include='*.json' --include='*.cfg' --include='*.toml' --include='*.ini' \
       -E "${esc_dir}" "${REPO_DIR}/src" 2>/dev/null \
       | grep -vE '(node_modules|__pycache__)' | sort -u >> "$HITS"
  if [ -s "$HITS" ]; then
    while IFS= read -r line; do bullet "${line}" "$REPORT"; ((TOTAL_HITS++)); done < "$HITS"
  else
    bullet "None found" "$REPORT"
  fi
  rm -f "$HITS"
done

# ── 4. Model paths / snapshots ──────────────────────────────────────────────
heading "Hardcoded Model Paths / Snapshot Hashes" "$REPORT"
subheading "HuggingFace snapshot hashes (40+ hex chars)" "$REPORT"
HITS=$(hits_file)
grep -rnI --include='*.py' --include='*.sh' --include='*.yaml' --include='*.yml' \
     --include='*.json' --include='*.toml' --include='*.cfg' --include='*.ini' \
     -oE '[a-f0-9]{40,}' "${REPO_DIR}/src" 2>/dev/null \
     | grep -viE '(commit|sha|hash|git)' | sort -u >> "$HITS"
if [ -s "$HITS" ]; then
  while IFS= read -r line; do bullet "${line}" "$REPORT"; ((TOTAL_HITS++)); done < "$HITS"
else
  bullet "None found" "$REPORT"
fi
rm -f "$HITS"

subheading "Hardcoded model paths" "$REPORT"
HITS=$(hits_file)
grep -rnI --include='*.py' --include='*.sh' \
     -E '(model_path|model_dir|checkpoint|weights_path|snapshot|model_name_or_path)\s*[=:]' "${REPO_DIR}/src" 2>/dev/null \
     | grep -vE '(\.env|os\.environ|config|__pycache__)' | sort -u >> "$HITS"
if [ -s "$HITS" ]; then
  while IFS= read -r line; do bullet "${line}" "$REPORT"; ((TOTAL_HITS++)); done < "$HITS"
else
  bullet "None found" "$REPORT"
fi
rm -f "$HITS"

# ── 5. Hardcoded IP addresses ───────────────────────────────────────────────
heading "Hardcoded IP Addresses" "$REPORT"
HITS=$(hits_file)
grep -rnI --include='*.py' --include='*.sh' --include='*.yaml' --include='*.yml' \
     --include='*.json' --include='*.toml' --include='*.cfg' --include='*.ini' \
     -oE '\b([0-9]{1,3}\.){3}[0-9]{1,3}\b' "${REPO_DIR}/src" 2>/dev/null \
     | grep -vE '(0\.0\.0\.0|127\.0\.0\.1|255\.255\.255\.255)' | sort -u >> "$HITS"
if [ -s "$HITS" ]; then
  while IFS= read -r line; do bullet "${line}" "$REPORT"; ((TOTAL_HITS++)); done < "$HITS"
else
  bullet "None found" "$REPORT"
fi
rm -f "$HITS"

# ── 6. Hardcoded port numbers in source code ────────────────────────────────
heading "Hardcoded Port Numbers in Source" "$REPORT"
subheading "Common service ports in .py and .sh files" "$REPORT"
HITS=$(hits_file)
# Look for numbers 1024-9999 near common port assignment patterns
grep -rnI --include='*.py' --include='*.sh' --include='*.yaml' --include='*.yml' \
     --include='*.toml' --include='*.cfg' --include='*.ini' \
     -E '(port\s*[=:]\s*[0-9]{2,5}|localhost:[0-9]{2,5}|:[0-9]{2,5}[^0-9])' "${REPO_DIR}/src" 2>/dev/null \
     | grep -vE '(\.pyc|__pycache__|test_|docs/)' | sort -u >> "$HITS"
if [ -s "$HITS" ]; then
  while IFS= read -r line; do bullet "${line}" "$REPORT"; ((TOTAL_HITS++)); done < "$HITS"
else
  bullet "None found" "$REPORT"
fi
rm -f "$HITS"

# ── 7. Environment variables that look deployment-specific ──────────────────
heading "Deployment-Specific Environment Variables" "$REPORT"
subheading "Env vars sourced in scripts/launchers" "$REPORT"
HITS=$(hits_file)
grep -rnI --include='*.sh' --include='*.py' --include='*.yaml' --include='*.yml' \
     -E '(os\.environ|environ\.get|\$\{[A-Z_]+\}|\$[A-Z][A-Z_]+)' "${REPO_DIR}/src" 2>/dev/null \
     | grep -viE '(PATH|HOME|USER|SHELL|TERM|LANG|LC_|PWD|HOSTNAME|TZ|EDITOR|PYTHON|CUDA_|TRANSFORMERS_|HF_|http_proxy|https_proxy|no_proxy)' \
     | sort -u >> "$HITS"
if [ -s "$HITS" ]; then
  while IFS= read -r line; do bullet "${line}" "$REPORT"; ((TOTAL_HITS++)); done < "$HITS"
else
  bullet "None found" "$REPORT"
fi
rm -f "$HITS"

# ── 8. Hardcoded shell commands with absolute paths ──────────────────────────
heading "Shell Commands with Hardcoded Absolute Paths" "$REPORT"
HITS=$(hits_file)
grep -rnI --include='*.py' \
     -E "(subprocess\.(run|call|check_call|check_output|Popen)|os\.system|shutil\.which)" \
     "${REPO_DIR}/src" 2>/dev/null | sort -u >> "$HITS"
if [ -s "$HITS" ]; then
  subheading "subprocess / os.system / which calls (check for hardcoded commands)" "$REPORT"
  while IFS= read -r line; do bullet "${line}" "$REPORT"; ((TOTAL_HITS++)); done < "$HITS"
else
  bullet "None found" "$REPORT"
fi
rm -f "$HITS"

# ── 9. Config files with deployment-specific values ─────────────────────────
heading "Config Files with Potential Deployment-Specific Values" "$REPORT"
HITS=$(hits_file)
find "${REPO_DIR}/src" \( -name 'config.yaml' -o -name 'config.yml' -o -name 'config.json' -o -name 'config.toml' -o -name '*.cfg' -o -name '*.ini' \) \
     -not -path '*/node_modules/*' -not -path '*/__pycache__/*' 2>/dev/null \
     | sort >> "$HITS"
if [ -s "$HITS" ]; then
  while IFS= read -r f; do
    rel="${f#$REPO_DIR/}"
    bullet "${rel}" "$REPORT"
    ((TOTAL_HITS++))
  done < "$HITS"
else
  bullet "None found" "$REPORT"
fi
rm -f "$HITS"

# ── 10. env files / .env.example ────────────────────────────────────────────
heading "Environment / Dotenv Files" "$REPORT"
HITS=$(hits_file)
find "${REPO_DIR}" -maxdepth 2 \( -name '.env*' -o -name '*.env' -o -name 'env.sh' \) \
     -not -path '*/node_modules/*' -not -path '*/__pycache__/*' 2>/dev/null \
     | sort >> "$HITS"
if [ -s "$HITS" ]; then
  subheading "Found dotenv files — check for hardcoded values" "$REPORT"
  while IFS= read -r f; do
    rel="${f#$REPO_DIR/}"
    bullet "${rel}" "$REPORT"
  done < "$HITS"
else
  bullet "None found" "$REPORT"
fi
rm -f "$HITS"

# ── Summary ──────────────────────────────────────────────────────────────────
sep "$REPORT"
{
  echo ""
  echo "## Summary"
  echo ""
  echo "**Total hits found:** ${TOTAL_HITS}"
  echo ""
  echo "### Priority items to fix for portability:"
  echo ""
  echo "1. **Absolute /home/ paths** — must be replaced with \`os.path.expanduser('~')\` or env vars"
  echo "2. **/mnt/ paths** — Windows/WSL specific; needs config abstraction"
  echo "3. **Hardcoded usernames** — should be dynamic or in config only"
  echo "4. **Model snapshot hashes** — should be config-driven, not hardcoded"
  echo "5. **Hardcoded ports** — should be env-configurable with defaults"
  echo "6. **Deployment-specific env vars** — should be documented in .env.example"
  echo ""
} >> "$REPORT"

echo "✅ Portability scan complete — ${TOTAL_HITS} hits"
echo "📄 Report: ${REPORT}"

echo "done:${TIMESTAMP}:${TOTAL_HITS}" > "${REPORT_DIR}/.portability_scan_done"
