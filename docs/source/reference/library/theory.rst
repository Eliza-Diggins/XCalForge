.. _library_gen_theory:
Library Generation
==================

The library generation process is a critical component of the `xrmodifier` pipeline.
It produces a structured, synthetic dataset that maps calibration modifications to observable temperature discrepancies
using forward simulations. This dataset serves as training data for machine learning emulators, enabling fast,
differentiable inference of calibration corrections across X-ray instruments.

The library generation process is contained within the :mod:`library` module. For details on the
implementation and usage of libraries, see :ref:`library_gen_user`.

Overview
--------

Library generation constructs a high-dimensional parameter grid representing calibration modifiers and true physical temperatures.
For each grid point, the process simulates X-ray spectra with a modified Ancillary Response File (ARF), performs spectral fitting,
and records the temperature recovery performance. The results are stored in an HDF5-backed library file suitable for downstream emulation and inference.

Motivation
----------

Direct, on-the-fly simulation and fitting during inference is computationally intractable due to:

- The expense of XSPEC-based spectral simulations.
- The need to explore a continuous, high-dimensional calibration space.
- The requirement for differentiable, fast evaluation of model outputs.

Thus, `xrmodifier` uses a two-phase approach:

1. **Offline Library Generation** — Exhaustively simulate and record the relationship between calibration modifiers and temperature recovery.
2. **Emulator Training** — Train neural networks to interpolate and approximate these relationships.

This approach provides a general way to test a number of possible calibration modifications and determine the
best fitting instance for a particular observed discrepancy.

Library Construction
---------------------

The library generation process proceeds through a structured series of steps to produce a synthetic dataset suitable for emulator training and calibration inference.

The core components of this process are:

1. **Define Modification Parameterization**
2. **Generate Simulation Configurations**
3. **Run Forward Simulations**
4. **Record Results to Disk**

Each step is designed to produce an exhaustive, reproducible set of simulations covering the relevant calibration space.

Define Modification Parameterization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first step is to select a parametric family of calibration modifications :math:`\mathcal{F}`. The most common example is the Gaussian ARF modifier:

.. math::

    \phi_\theta(\Psi_1)(E) = \Psi_1(E) \cdot \left[ 1 + A \cdot \exp\left( -\frac{(E - \mu)^2}{2\sigma^2} \right) \right]

where:

- :math:`\Psi_1(E)` is the baseline telescope ARF as a function of energy.
- :math:`\theta = (\mu, \sigma, A)` are the learned parameters representing center, width, and amplitude of the distortion.

Each parameter :math:`\alpha_i` in :math:`\theta` is discretized into a finite set of values, constructing a parameter grid:

.. math::

    \mathcal{G} = \mathcal{A}_1 \times \mathcal{A}_2 \times \ldots \times \mathcal{A}_d

where :math:`d` is the number of modifier parameters.

In addition, a discrete set of true source temperatures :math:`\mathcal{T}` is defined:

.. math::

    \mathcal{T} = \{ T_1, T_2, \ldots, T_{N_T} \}

The full simulation space is:

.. math::

    \mathcal{G} \times \mathcal{T} = \{ (\theta, T_{\text{true}}) \}

This defines the total number of simulations required:

.. math::

    |\mathcal{G} \times \mathcal{T}| = \left( \prod_{i=1}^d N_i \right) \times N_T

Generate Simulation Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each grid point :math:`(\theta, T_{\text{true}})`, the process builds the necessary simulation configuration:

1. Apply the calibration modifier :math:`\phi_\theta` to the baseline ARF, producing a modified ARF file.
2. Prepare XSPEC input files, including:

   - Modified ARF
   - Response Matrix File (RMF)
   - Exposure time, background models, etc.

The configuration fully specifies how to simulate and analyze that grid point.

Forward Simulation and Fitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given each configuration:

1. **Simulate Synthetic Spectrum**

   Generate synthetic photon data using XSPEC's `fakeit` functionality:

   .. math::

       D_\theta \sim \mathcal{S}(T_{\text{true}}, \Psi_1')

2. **Fit with Original Calibration**

   Fit the synthetic data using the unmodified ARF:

   .. math::

       T_1 = \arg\max_T \mathcal{L}(D_\theta \mid T, \Psi_1)

3. **Fit with Modified Calibration**

   Refit using the modified ARF:

   .. math::

       T_1' = \arg\max_T \mathcal{L}(D_\theta \mid T, \Psi_1')

Record Results
~~~~~~~~~~~~~~

The output of each simulation includes the resulting modified temperature
obtained from fitting with the unmodified configuration along with relevant
measures of the uncertainty.

These results are stored in an HDF5 library file with structure:

.. code-block:: text

    library.h5
    ├── PARAMS/           # Parameter grids for each modifier dimension
    ├── LIBRARY/
        ├── results       # Array: (N1, N2, ..., N_d, N_T, 3)
        ├── temps         # Temperature grid

The results array records:


Parallelization and Efficiency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The library generation process is parallelized using MPI:

- The parameter grid is divided across processes.
- Each rank simulates and fits a unique subset of the grid.
- Intermediate results are saved to temporary, rank-specific HDF5 files.
- Rank 0 merges these outputs using efficient, Dask-based HDF5 finalization.

This enables scalable, efficient exploration of complex modifier spaces on HPC systems.
