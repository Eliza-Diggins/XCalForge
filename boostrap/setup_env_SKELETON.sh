#!/bin/bash
# =============================================================================
# setup_env_SKELETON.sh - Template script to initialize XCalForge environment
# =============================================================================
# Usage:
#   source bootstrap/setup_env_<your_cluster>.sh
#
# This script should be sourced (not executed) to configure your shell
# session with the necessary modules and virtual environment for XCalForge.
#
# INSTRUCTIONS:
#   1. Copy this file to setup_env_<your_cluster>.sh
#   2. Replace the module versions and paths below as needed
#   3. Add an entry in the Makefile or README for user convenience
# =============================================================================

set -e  # Exit immediately on error

# ----------------------------
# CONFIGURATION
# ----------------------------
PYTHON="python"                       # Adjust if a specific Python module is required
MPI_MODULE="openmpi/<version>"        # e.g., openmpi/4.1.3
HEASOFT_MODULE="heasoft/<version>"    # e.g., heasoft/6.31.1
VENV_PATH=".venv"                     # Relative or absolute path to your virtual environment

# ----------------------------
# MODULE LOADING
# ----------------------------
echo "[XCalForge]: Loading system modules..."
module purge
module load "$MPI_MODULE"
module load "$HEASOFT_MODULE"
echo "[XCalForge]: Modules loaded."

# ----------------------------
# ENVIRONMENT ACTIVATION
# ----------------------------
echo "[XCalForge]: Activating virtual environment at $VENV_PATH..."
source "$VENV_PATH/bin/activate"
echo "[XCalForge]: Environment activated."

# ----------------------------
# SUCCESS
# ----------------------------
echo "[XCalForge]: Setup complete. You may now use XCalForge CLI or Python modules."
