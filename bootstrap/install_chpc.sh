#!/bin/bash
# ============================================================================
# install_chpc.sh - Installation script for SpecForge on CHPC systems
# ============================================================================
# This script sets up the SpecForge environment on CHPC by:
# - Loading the appropriate modules (HEASoft and OpenMPI)
# - Initializing HEASoft
# - Creating a Python virtual environment
# - Installing MPI bindings
# - Verifying XSPEC and MPI
# - Installing the SpecForge Python package
#
# Usage:
#   bash bootstrap/install_chpc.sh
# This script is intended to be run from the Makefile via:
#   make install-chpc
#
# IMPORTANT:
# - Ensure that HEASoft and XSPEC are properly installed and configured in your user environment.
# - You may need to adjust versions based on system availability.
# ============================================================================
set -e # ENSURE HARD FAILURES (any error kills install.)

# ---------------------------- #
# Configuration Settings.      #
# ---------------------------- #
# IMPORTANT: Please ensure that these are correct for your
# use case and environement configuration.
PYTHON="python"
MPI_MODULE="openmpi/4.1.3"
HEASOFT_MODULE="heasoft/6.31.1"


# --- Load the Modules --- #
echo "[Specforge]: Loading CHPC modules..."
module purge
module load "$MPI_MODULE"
module load "$HEASOFT_MODULE"
echo "[Specforge]: Loading CHPC modules... [DONE]"

# --- Configure Python --- #
# For CHPC, the python installation is venved
# off of the heasoft installation, so we don't need
# to do this explicitly.
echo "[Specforge]: Loading base python..."
module purge
module load "$MPI_MODULE"
module load "$HEASOFT_MODULE"
echo "[Specforge]: Loading base python... [DONE]"

# --- Create the VENV -- #
# This will create the venv using the specified
# python module. By default, this is created in ./.venv
echo "[Specforge]: Building venv..."
$PYTHON -m venv .venv
source .venv/bin/activate
echo "[Specforge]: Building venv... [DONE]"

# --- Install MPI4PY in the env --- #
# This can be modified in special cases.
echo "[Specforge]: Installing MPI4PY..."
pip install mpi4py
echo "[Specforge]: Installing MPI4PY... [DONE]"

# --- VERIFY --- #
echo "[Specforge]: Verifying xspec and mpi4py availability... "
python -c 'import xspec; import mpi4py.MPI; print("MPI and XSPEC successfully loaded")'

echo "[Specforge]: Installing specforge..."
pip install . --no-build-isolation
echo "[Specforge]: Installed specforge."
