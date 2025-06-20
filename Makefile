# ================================================
# Makefile - SpecForge HPC Setup
# ================================================
# This Makefile simplifies installation and cleanup for SpecForge.
# It is designed to support HPC systems where software modules
# (e.g., HEASOFT, MPI) must be explicitly loaded prior to installation.
#
# ----------------------------
# How to Use
# ----------------------------
#
# 1. For CHPC users:
#    Run:
#        make install-chpc
#
#    This will:
#      - Load HEASOFT and OpenMPI modules
#      - Initialize HEASOFT environment
#      - Create a Python virtual environment
#      - Install mpi4py with mpicc
#      - Verify xspec and MPI bindings
#      - Install SpecForge in the environment
#
# 2. For users on other systems:
#    Create your own installation script based on `bootstrap/install_chpc.sh`.
#    Save it as `install_<system>.sh` in the `bootstrap/` folder.
#
#    Then add a new target like:
#        install-<system>:
#            bash bootstrap/install_<system>.sh
#
#    Example for a system named "astrofarm":
#        install-astrofarm:
#            bash bootstrap/install_astrofarm.sh
#
#    Then run:
#        make install-astrofarm
#
# 3. Cleanup:
#    Run:
#        make clean
#    This removes all build artifacts and caches.

# ----------------------------
# INSTALL: CHPC setup
# ----------------------------
.PHONY: install-chpc clean
.DEFAULT_GOAL := help

install-chpc:
	@echo "Running CHPC install script..."
	@chmod +x bootstrap/*.sh
	@bash bootstrap/install_chpc.sh

# ----------------------------
# CLEAN: remove build artifacts
# ----------------------------
clean:
	find . -name "*.so" -delete
	find . -name "__pycache__" -type d -exec rm -r {} +
	rm -rf build dist *.egg-info
	@echo "Cleaned up build files."

help:
	@echo "Usage:"
	@echo "  make install-chpc   # Install SpecForge on CHPC"
	@echo "  make clean          # Clean build artifacts"
