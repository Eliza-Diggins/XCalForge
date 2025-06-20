#!/bin/bash
# =============================================================================
# setup_env.sh - Dispatch to the appropriate system-specific environment script
# =============================================================================
# Usage:
#   source bootstrap/setup_env.sh <system>
# Example:
#   source bootstrap/setup_env.sh chpc
#
# This will source `bootstrap/setup_env_chpc.sh`.
# The script must be sourced to correctly activate environments.
# =============================================================================

set -e

SYSTEM="$1"

if [[ -z "$SYSTEM" ]]; then
  echo "[XCalForge]: ERROR - No system specified."
  echo "Usage: source bootstrap/setup_env.sh <system>"
  exit 1
fi

SCRIPT="bootstrap/setup_env_${SYSTEM}.sh"

if [[ -f "$SCRIPT" ]]; then
  echo "[XCalForge]: Setting up environment for '$SYSTEM'..."
  source "$SCRIPT"
else
  echo "[XCalForge]: ERROR - No setup script found for system '$SYSTEM'"
  echo "Expected: $SCRIPT"
  exit 1
fi
