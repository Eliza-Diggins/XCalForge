#!/bin/bash
# ============================================================================
# setup_env_chpc.sh - Set up environment for SpecForge on CHPC systems
# ============================================================================
# Usage:
#   source bootstrap/setup_env_chpc.sh
# This script should be sourced to configure the shell session.

set -e

# ----------------------------
# Configuration
# ----------------------------
PYTHON="python"
MPI_MODULE="openmpi/4.1.6"
HEASOFT_MODULE="heasoft/6.31.1"

echo "[SpecForge]: Loading CHPC modules..."
module purge
module load "$MPI_MODULE"
module load "$HEASOFT_MODULE"
echo "[SpecForge]: Modules loaded."

echo "[SpecForge]: Loading virtual environment..."
source .venv/bin/activate

echo "[SpecForge]: Environment setup complete. Ready for use."
