#!/usr/bin/env python
"""
example_build_arf_library.py

This script demonstrates how to create and generate a Gaussian ARF modification
library using the SpecForge infrastructure.

We assume that the base ARF and RMF files are present at the indicated paths.
"""
import numpy as np

from xforge.library import GaussianARFLibrary

# -------------------------------
# Step 1: Define library location
# -------------------------------
library_dir = "my_arf_library"  # Output directory to store the library
base_arf = "base.arf"  # Path to base ARF file
base_rmf = "base.rmf"  # Path to base RMF file

# -------------------------------
# Step 2: Define parameter grid
# -------------------------------
parameters = {
    "mu": np.linspace(1.0, 3.0, 5),  # keV center of the Gaussian distortion
    "sigma": np.linspace(0.1, 0.3, 3),  # keV width of the distortion
    "A": np.linspace(-0.2, 0.2, 5),  # amplitude of the distortion
}

# -------------------------------
# Step 3: Create the library
# -------------------------------
print("[Example] Creating library...")
lib = GaussianARFLibrary.create_library(
    directory=library_dir,
    parameters=parameters,
    base_arf=base_arf,
    base_rmf=base_rmf,
    overwrite=True,
)

# -------------------------------
# Step 4: Generate library data
# -------------------------------
temperatures = np.linspace(0.5, 5.0, 10)  # keV, simulated plasma temperatures

print("[Example] Generating library...")
lib.generate_library(temperatures)
print("[Example] Library generation complete!")
