Library Generation
==================

This section demonstrates how to create, configure, and populate calibration modification
libraries with **XCalForge**.

Modification libraries serve as structured, disk-backed datasets capturing
how instrumental calibration uncertainties propagate into astrophysical parameter
recovery—typically focused on temperature estimates from X-ray spectra.

The examples below walk through:

- Initializing new libraries with defined parameter grids,
- Applying controlled calibration modifications (e.g., ARF perturbations),
- Running synthetic data generation using MPI-parallel workflows,
- Finalizing and inspecting library outputs for downstream analysis,

Whether you're validating a new calibration model or building training datasets
for machine learning inference, these examples provide a hands-on introduction
to the library generation process in XCalForge.

.. contents::
   :local:
   :depth: 1
