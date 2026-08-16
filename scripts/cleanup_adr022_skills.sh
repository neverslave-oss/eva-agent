#!/usr/bin/env bash
# ADR-022 cleanup: remove auto-evolved skills that cause false semantic matches
# Run this on the target host after deploying the ADR-022 branch.

set -euo pipefail

SKILLS_DIRS=(
    "${HOME}/.kernel-evolving/ecosystem/private/skills"
    "${SKILLS_DIR:-}"
)

SKILLS_TO_REMOVE=(
    "send-workspace-user-md"
)

for dir in "${SKILLS_DIRS[@]}"; do
    [ -z "$dir" ] && continue
    for skill in "${SKILLS_TO_REMOVE[@]}"; do
        target="${dir}/${skill}"
        if [ -d "$target" ]; then
            echo "[cleanup] Removing auto-evolved skill: $target"
            rm -rf "$target"
        else
            echo "[cleanup] Not found (OK): $target"
        fi
    done
done

echo "[cleanup] Done."
