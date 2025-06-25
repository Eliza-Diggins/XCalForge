#!/usr/bin/env python
"""
example_build_arf_library.py

This script demonstrates how to create and generate a Gaussian ARF modification
library using the SpecForge infrastructure.

We assume that the base ARF and RMF files are present at the indicated paths.
"""
import numpy as np

from xforge.library import GaussianARFLibrary
from xforge.utilities.env import configure_xspec, get_mpi, get_xspec

# Configure XSPEC settings and obtain the MPI information
# for the runtime process.
xspec = get_xspec()
configure_xspec()

comm = get_mpi()
rank = comm.Get_rank()

# -------------------------------
# Step 1: Define library location
# -------------------------------
library_dir = "/scratch/general/vast/u1281896/mod_arf_lib_test"  # Output directory to store the library
base_arf = "base.arf"  # Path to base ARF file
base_rmf = "base.rmf"  # Path to base RMF file

# -------------------------------
# Step 2: Define parameter grid
# -------------------------------
parameters = {
    "mu": np.linspace(0.1, 12.0, 15),  # keV center of the Gaussian distortion
    "sigma": np.linspace(0.1, 5, 2),  # keV width of the distortion
    "A": np.linspace(-0.1, 0.1, 2),  # amplitude of the distortion
}

# -------------------------------
# Step 3: Create the library
# -------------------------------
if rank == 0:
    _ = GaussianARFLibrary.create_library(
        directory=library_dir,
        parameters=parameters,
        base_arf="/uufs/astro.utah.edu/common/home/u1281896/emma_project/unmodified_arf.arf",
        base_rmf="/uufs/astro.utah.edu/common/home/u1281896/emma_project/center_6102.rmf",
        overwrite=True,
    )

# Ensure rank 0 creates the library before we load it.
comm.Barrier()
lib = GaussianARFLibrary(library_dir)
lib.set_logging_level("DEBUG")

# -------------------------------
# Step 4: Generate library data
# -------------------------------
temperatures = np.linspace(0.5, 15, 3)  # keV, simulated plasma temperatures

lib.generate_library(temperatures)
