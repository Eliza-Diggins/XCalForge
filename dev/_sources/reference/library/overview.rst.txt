.. _library_gen_user:

===============================
Library Generation User Guide
===============================

This guide explains how to build, run, and manage synthetic calibration modification libraries using the `xrmodifier` framework.

For theoretical background on the principles and motivations of library generation, see :ref:`library_gen_theory`.

Overview
--------

The primary goal of library generation is to produce structured, reproducible datasets that map configuration modifications
to observable temperature recovery performance. These datasets serve as the training foundation for fast, differentiable
emulators used in downstream calibration inference.

The library system supports fully generic modifications to the telescope configuration and hyperparameters, including:

- Changes to spectral models (e.g., different astrophysical models in XSPEC)
- Perturbations to instrumental response files (e.g., ARFs, RMFs)
- Variations in exposure settings or background modeling
- Arbitrary extensions defined by the user

The class :class:`xforge.library.base.ModificationLibrary` provides the base interface and tooling
to implement such modification libraries.

Library Structure
-----------------

A generated library has a well-defined, self-contained directory structure:

.. code-block:: text

    library_dir/
    ├── library.h5     # HDF5 data file (parameter grid, results, temperatures)
    ├── config.yaml    # Persistent configuration file
    ├── logs/          # Per-rank log files
    ├── cache/         # Temporary working directory (cleared by default after generation)
    ├── bin/           # Optional directory for persistent outputs (e.g., modified ARFs)

Details:

- **`library.h5`** contains:

  - `PARAMS` group: Parameter grids for modification hyperparameters.
  - `LIBRARY` group (after generation): Simulation temperatures and results arrays.

- **`config.yaml`** uses :class:`utils.ConfigManager` for structured access and storage of run settings.
- **`logs/`** provides separate log files for each MPI rank.
- **`cache/`** holds transient data such as synthetic spectra or working files during generation.
- **`bin/`** is optional for outputs intended to persist beyond the generation process.

Re-loading a library from disk is done via:

.. code-block:: python

    from xforge.library import GaussianARFLibrary

    lib = GaussianARFLibrary("/path/to/library_dir")

The library validates structure and loads its configuration and parameter space automatically.

The Simulation Process
----------------------

Before diving into user modifications, it is important to understand the simulation and fitting workflow that underpins
the library generation process:

**Simulate with Modified Configuration:**

- Synthetic photon spectra are generated using the *modified* telescope configuration, which reflects the true but unknown
  calibration state of the instrument.
- This incorporates distortions or biases introduced by calibration errors (e.g., ARF distortions).
- In practice, this step uses XSPEC's ``fakeit`` functionality with modified inputs to produce simulated observational data.

**Fit with Modified Configuration:**

- The same modified configuration can be used to fit the simulated data.
- This serves as a sanity check: fitting with the correct calibration should accurately recover the true physical parameters
  (e.g., plasma temperature).
- Some workflows omit this step, focusing only on recovery errors under incorrect calibration assumptions.

**Fit with Unmodified Configuration:**

- The simulated data (generated with the modified, true configuration) is analyzed as if the telescope calibration were correct,
  using the *unmodified* configuration currently in operational use.
- This mimics real-world scenarios where outdated or imperfect calibration files produce incorrect parameter estimates.

**Record Results:**

- The primary quantity stored in the library is the discrepancy between known truth and recovered parameters when using
  the unmodified configuration.
- This enables building a mapping from calibration modifications to temperature recovery errors.

**Temperature Sweep**

- The above steps are repeated across a user-defined grid of plasma temperatures.
- This ensures the library captures calibration effects over the full relevant physical parameter space.

This process is repeated across the entire parameter grid of calibration modifications and temperatures in a distributed,
MPI-parallelized workflow. The result is a dense, HDF5-backed library capturing how calibration errors distort temperature
inference across a wide range of conditions.

Building Simulation Pipelines
''''''''''''''''''''''''''''''

The abstract class :class:`~library.base.ModificationLibrary` defines the general structure and sequence of operations
required for a calibration modification library. To create a custom library, users
subclass :class:`~library.base.ModificationLibrary` and implement abstract classes to configure the behavior. There are
3 parts of the simulation which can be configured:

1. **Configuration**

   The first area for customization is the **configuration**, which defines all parameters
   necessary to simulate the telescope response. This includes, but is not limited to:

   - Ancillary Response Files (ARFs)
   - Redistribution Matrix Files (RMFs)
   - Exposure settings
   - Background files
   - Instrument-specific corrections

   These configurations control how XSPEC generates synthetic spectra via the `fakeit` interface.
   In `xrmodifier`, the user provides both a baseline (trusted) configuration and a modified (perturbed) configuration to test calibration hypotheses.

   The two abstract methods responsible for defining configurations are:

   .. code-block:: python

       def generate_unmodified_configuration(self, id: int, **parameters) -> dict:
           ...

       def generate_modified_configuration(self, id: int, **parameters) -> dict:
           ...

   Both methods must return a dictionary containing the required file paths and settings for XSPEC, typically structured as:

   .. code-block:: python

       {
           "response": "/path/to/rmf.rmf",
           "arf": "/path/to/arf.arf",
           "exposure": 50000,  # Optional, defaults to 50 ks
           "background": "/path/to/bkg.pha",  # Optional
           "correction": "/path/to/corr.arf",  # Optional
           "backExposure": 50000,  # Optional
       }

    **Modified Configuration**

    - Represents the *assumed true* configuration of the telescope, including a calibration modification (e.g., distorted ARF).
    - This is used to generate the synthetic, simulated data, which reflects the "ground truth" we are attempting to recover.
    - In practice, this produces synthetic spectra using the modified calibration, which mimics real-world observations where the instrument contains unknown biases.

    **Unmodified Configuration**

    - Represents the *assumed incorrect* configuration currently in use by the telescope (e.g., pre-launch calibration files or outdated response matrices).
    - This configuration is applied during the fitting stage, where the synthetic data (generated with the modified, biased configuration) is analyzed as if the calibration were correct.
    - The discrepancy between the fitted parameters (e.g., recovered temperature) and the known simulation input reflects the calibration-induced bias.

2. **Model Definitions**

   The second area for customization is the **astrophysical model**, which defines the physical source being simulated and fit.
   These models are constructed using XSPEC and typically include components such as thermal plasma models, absorption models, or more complex source descriptions.

   The user defines two key methods for model construction:

   .. code-block:: python

       def generate_model_unmodified(self, T, **parameters):
           ...

       def generate_model_modified(self, T, **parameters):
           ...

   Both methods build and return an XSPEC model instance configured for the specified temperature ``T`` and parameter grid point ``**parameters``.

   **Modified Model**

   - Represents the *physical truth* model used in the simulation, incorporating the effects of calibration distortions via the modified configuration.
   - This model is used alongside the modified configuration to generate the synthetic spectrum with XSPEC's ``fakeit``.
   - For example, the modified model might use a distorted ARF or updated effective area curve, reflecting the true instrument state.

   **Unmodified Model**

   - Represents the model used during spectral fitting under the assumption that the calibration is correct (i.e., using the unmodified configuration).
   - Fitting the synthetic data with this model reveals how calibration errors propagate into parameter estimation biases (e.g., incorrect temperature recovery).

   In many cases, the astrophysical model structure is identical between the modified and unmodified cases—only the calibration files differ. However, advanced workflows can modify the source model itself if desired.

3. **Fitting Procedures**

   The final component under user control is the **fitting logic**, where the synthetic spectra are analyzed to extract physical parameters.

   The user provides two methods responsible for performing XSPEC fits and extracting results:

   .. code-block:: python

       def fit_unmodified(self, config: dict, **parameters) -> tuple:
           ...

       def fit_modified(self, config: dict, **parameters) -> tuple:
           ...

   Both methods:

   - Receive a configuration dictionary specifying response files, ARFs, exposure, etc.
   - Fit the corresponding spectrum loaded into XSPEC.
   - Return relevant fit statistics, typically including:

     - The best-fit temperature
     - Fit quality metrics (e.g., C-statistic, goodness-of-fit)
     - Any additional diagnostics relevant to the modification being tested

   **Standard Return Convention**

   While users can modify outputs as needed, the typical return structure is:

   .. code-block:: python

       (best_fit_temperature, stat_value, additional_metric)

   These results are stored in the HDF5 library file for downstream use.
