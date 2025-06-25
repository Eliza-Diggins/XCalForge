"""
===============================================
ARF Modification Library Generation (Gaussian)
===============================================

This example demonstrates how to generate a calibration modification library using
**XCalForge's** ARF modification framework with Gaussian perturbations.
"""
# %%
# The ARF (Auxiliary Response File) characterizes an X-ray telescope's effective area curve,
# which determines how many photons are detected at each energy. Systematic uncertainties
# in the ARF directly affect measured spectra and inferred astrophysical parameters.
#
# **XCalForge** supports injecting controlled distortions into ARFs to quantify the
# impact of such uncertainties on temperature recovery.
#
# Gaussian Modification Scheme
# ----------------------------
#
# The modified ARF is defined as:
#
# .. math::
#
#     A'(E) = A(E) \times \left( 1 + A_0 \, e^{- \left( \frac{E - \mu}{\sigma} \right)^2} \right)
#
# Where:
#
# - :math:`A(E)` is the baseline effective area at energy :math:`E`,
# - :math:`A_0` is the amplitude of the modification (unitless, fractional change),
# - :math:`\mu` is the center energy of the Gaussian (keV),
# - :math:`\sigma` is the width (keV) controlling the spread of the modification.
#
# This scheme simulates local, energy-dependent distortions—representing, for example,
# uncertainties from coating thickness, contamination layers, or detector edges.
#
# The resulting library provides a grid of synthetic spectra exploring how these calibration perturbations
# propagate into biased temperature measurements.
#
# .. important::
#
#   The first half of this example, the section covering generating the library
#   is presented in a somewhat schematic manner as the actual code necessary will
#   depend somewhat on specific paths and behaviors on your system.
#
# Setup
# -----
#
# The first step in the setup for this workflow is to ensure that any necessary environmental
# elements are configured. By default, XCalForge requires XSPEC (via PyXSPEC) and also
# requires MPI for parallelization. Details on how to configure these for different
# systems can be found at :ref:`install`. Before running the script, it
# is then necessary to have self-consistently loaded the environment.
#
# On a system like the University of Utah's CHPC, this might look like
#
# .. code-block:: bash
#
#   $ ml heasoft/6.31.1 openmpi/4.1.6
#   $ source /path/to/venv/bin/activate
#
# One the environment is configured, we're ready to begin the actual code.
#
# Library Setup
# '''''''''''''
#
# To begin, we'll need the relevant modification library class (:class:`~library.base.ModificationLibrary`)
# for the particular perturbations we wish to consider. In most cases, these need to be
# written by the user for their particular use case; however, gaussian arf modification is
# built in via the :class:`~library.arf_mod.GaussianARFLibrary`.
#
# We'll begin by importing this class and performing some configuration for XSPEC. This includes
# setting the test statistic, the chatter level, and some other things via :func:`utilities.env.configure_xspec`.
#
# .. code-block:: python
#
#    # Import necessary modules.
#    import numpy as np
#    from xforge.library import GaussianARFLibrary
#    from xforge.utilities import configure_xspec, get_config, get_mpi, get_xspec, clear_xspec
#    from pathlib import Path
#    import os
#
#    # Fetch XSPEC and configure it.
#    xspec = get_xspec()
#    configure_xspec()
#
# Creating the Library
# --------------------
#
# To create the library, we'll need to know the desired path of the library
# and the path to the default ARF and RMF files being used in the simulation.
#
# .. code-block:: python
#
#    library_dir = "path/to/library"
#    base_arf,base_rmf = "path/to/arf", "path/to/rmf"
#
# In addition to the relevant directories, we also need to specify the
# correct set of parameters from which to create the discretized parameter space.
# For this example, we'll consider a very minimal library:
#
# .. code-block:: python
#
#    parameters = {
#                  'mu': np.linspace(1,10,15),
#                  'sigma': [1],
#                  'A': [0.08, -0.08]
#                  }
#
# Now that the setup has been performed, we can build the library:
#
# .. code-block:: python
#
#     comm = get_mpi()
#     rank = comm.Get_rank()
#     if rank == 0:
#         _ = GaussianARFLibrary.create_library(
#             directory=library_dir,
#             parameters=parameters,
#             base_arf=base_arf,
#             base_rmf=base_rmf,
#             overwrite=True,
#         )
#
#     # Ensure RANK 0 completes before continuing.
#     comm.Barrier()
#
# Generating the Library
# ----------------------
#
# Now for the big part: generating the library data. This is done by passing
# a list of temperatures to :meth:`~library.base.ModificationLibrary.generate_library`.
#
# .. code-block:: python
#
#     temperatures = np.linspace(0.5,15,8)
#     lib.generate_library()
#
# An instance of the library generated from this proceedure is stored in
# the ``/docs/galleries/examples/bin/gauss_arf_lib`` directory.
#
# %%
# Inspecting the Data
# -------------------
#
# Once the library has been generated, we can begin inspecting the data!

from pathlib import Path

import h5py
import numpy as np

# Get the path to the data relative to this file.
data_path = Path("../bin/gauss_arf_lib/library.h5")
print(data_path)

with h5py.File(data_path, "r") as fio:
    # See what data is available
    print(fio.keys())

    # The `LIBRARY` contains the data.
    # The `PARAMS` contains the parameters.

    library, temperatures = np.asarray(fio["LIBRARY/results"]), np.asarray(
        fio["LIBRARY/temps"]
    )
    mus = np.asarray(fio["PARAMS/mu"])

# %%
# the ``library`` will be a ``(15,1,2,8,3)`` array of data for
# us to inspect. The final axis contains the lower, mean, and upper bounds on
# the parameter estimate.
#
# We can use this to make a plot of the data.
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
cmap = plt.cm.viridis

# Extract the temperatures that we are going
# to plot.
T_unmod = temperatures
T_mod = library[0, :, 0, :, :]

vmin, vmax = np.amin(T_unmod), np.amax(T_unmod)
norm = plt.Normalize(vmin=vmin, vmax=vmax)


# Plot each mu as a scatter point set, color-coded by mu
for mui, mu in enumerate(mus):
    # Compute the color.
    color = cmap(norm(mu))

    yerr = (T_mod[mui, :, 2] - T_mod[mui, :, 1], T_mod[mui, :, 1] - T_mod[mui, :, 0])

    ax.errorbar(
        T_unmod,
        T_mod[mui, :, 1],
        yerr=yerr,
        mec=color,
        mfc="w",
        ls="-",
        color=color,
        capsize=2,
    )

# plot the 1-1 line.
ax.plot([vmin, vmax], [vmin, vmax], color="k")


# Labels and colorbar
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
cbar.set_label("Modification Center Energy (keV) [μ]")

ax.set_xlabel("Unmodified Temperature (keV)")
ax.set_ylabel("Recovered Modified Temperature (keV)")
ax.set_title("Temperature Bias from ARF Gaussian Modifications")
ax.grid(True)

plt.show()

# %%
# Residuals Plot
# --------------
# In addition to the raw recovered temperatures, it's often useful to visualize
# the residuals between the unmodified and modified results:
#
# .. math::
#
#     \Delta T = T_\mathrm{mod} - T_\mathrm{unmod}
#
# This highlights both systematic offsets and parameter-dependent biases.
#

fig, ax = plt.subplots(figsize=(8, 6))
cmap = plt.cm.viridis

# Compute residuals
residuals = T_mod[:, :, 1] - T_unmod[None, :]

# Setup color normalization
vmin, vmax = np.amin(mus), np.amax(mus)
norm = plt.Normalize(vmin=vmin, vmax=vmax)

# Plot each μ as a scatter set with error bars
for mui, mu in enumerate(mus):
    color = cmap(norm(mu))

    # Symmetric error propagated from T_mod uncertainties
    yerr = 0.5 * (T_mod[mui, :, 2] - T_mod[mui, :, 0])

    ax.errorbar(
        T_unmod,
        residuals[mui],
        yerr=yerr,
        ls="-",
        color=color,
        capsize=2,
        marker="o",
        mfc="w",
        mec=color,
    )

# Horizontal zero reference line
ax.axhline(0, color="k", linestyle="--", linewidth=1)

# Labels and colorbar
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
cbar.set_label("Modification Center Energy (keV) [μ]")

ax.set_xlabel("Unmodified Temperature (keV)")
ax.set_ylabel("Temperature Bias ΔT (keV)")
ax.set_title("Residual Temperature Bias from ARF Gaussian Modifications")
ax.grid(True)

plt.show()

# %%
