#!/bin/bash
# ============================================================================
# install_SKELETON.sh - Template installation script for XCalForge on HPC systems
# ============================================================================
# This script is a reference for setting up the XCalForge environment
# on HPC systems. It assumes use of modules for environment configuration,
# and a venv-based Python install with mpi4py and PyXspec support.
#
# USAGE:
#   cp bootstrap/install_SKELETON.sh bootstrap/install_<your_cluster>.sh
#   Modify paths, modules, and Python setup as needed.
#
# Then run:
#   make install-<your_cluster>
#
# ============================================================================
set -e  # Exit on any error

# -------------------------------
# CONFIGURATION
# -------------------------------
PYTHON="python"                    # Command to invoke Python
MPI_MODULE="openmpi/<version>"     # Replace <version> with correct one
HEASOFT_MODULE="heasoft/<version>" # Replace <version> with correct one

# -------------------------------
# LOAD MODULES
# -------------------------------
echo "[XCalForge]: Loading modules..."
module purge
module load "$MPI_MODULE"
module load "$HEASOFT_MODULE"
echo "[XCalForge]: Modules loaded."

# -------------------------------
# PYTHON ENVIRONMENT
# -------------------------------
echo "[XCalForge]: Setting up virtual environment..."
$PYTHON -m venv .venv
source .venv/bin/activate
echo "[XCalForge]: Virtual environment created at .venv"

# -------------------------------
# INSTALL DEPENDENCIES
# -------------------------------
echo "[XCalForge]: Installing mpi4py..."
pip install mpi4py

echo "[XCalForge]: Verifying XSPEC + MPI availability..."
python -c "import xspec; import mpi4py.MPI; print('XSPEC and MPI are available.')"

echo "[XCalForge]: Installing XCalForge package..."
pip install . --no-build-isolation
echo "[XCalForge]: Installation complete."

# -------------------------------
# FINAL MESSAGE
# -------------------------------
echo "[XCalForge]: Done! Use 'source bootstrap/setup_env_<your_cluster>.sh' to initialize this environment in future sessions."
