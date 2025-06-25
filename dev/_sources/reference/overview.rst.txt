.. _overview:

=================================
XCalForge | System Overview
=================================

**XCalForge** is an integrated framework for instrument cross-calibration,
designed to quantify and model systematic discrepancies in high-energy astrophysical observations.
The library is designed to answer the following generic question:

    Given two telescopes (Telescope A and Telescope B), and assuming that Telescope A represents
    the "ground truth," what modifications to the calibration, modeling, or fitting could correct
    the discrepancies between the findings of the two telescopes.

Impetus
=======

Persistent discrepancies in X-ray temperature measurements across observatories such as
**Chandra**, **XMM-Newton**, **NuSTAR**, and now **XRISM**, have motivated extensive
cross-calibration efforts within the astrophysical community. Numerous studies have]
demonstrated that fitted plasma temperatures, particularly for hot galaxy clusters, vary systematically between instruments.
For example, broad analyses by the IACHEC consortium and others have shown that **XMM-Newton's** EPIC detectors
often yield temperatures up to 20–30% lower than those derived from **Chandra's** ACIS at high temperatures :cite:p:`Schellenberger2015`.

Similar temperature discrepancies, on the order of 10–15%, have been observed between **Chandra**
and **NuSTAR** in cluster observations :cite:p:`2022MNRAS.517.5594W,2025AAS...24541206P,2023ApJ...958..112P`.
These differences propagate into critical quantities such as cluster mass estimates, introducing significant systematic
uncertainty into cosmological constraints derived from X-ray data.

The root causes of these discrepancies are linked to uncertainties in instrument
effective area calibrations, especially at low energies (<2 keV), as well as
differences in detector response and potential modeling systematics
:cite:p:`2022MNRAS.517.5594W`. Recent studies
using **XRISM's** Resolve and Xtend instruments show promising internal consistency,
but preliminary results suggest that soft-band calibration challenges
persist. As highlighted by IACHEC, correcting these inter-mission
biases is essential to ensure reliable, reproducible measurements of key astrophysical
parameters across observatories :cite:p:`2025arXiv250117199G`. **XCalForge** provides a unified framework
to simulate, emulate, and quantify these systematic effects, enabling the community to better
understand, model, and mitigate cross-calibration temperature offsets.




📊 Methodology
==================

XCalForge is a modular framework designed to quantify and mitigate calibration-induced
systematics in X-ray data. It combines synthetic simulations, machine learning emulators,
and statistical inference to efficiently explore complex calibration parameter spaces.
The workflow consists of three integrated stages:

1. **Library Generation**: Systematic, high-performance generation of synthetic calibration libraries across configurable parameter grids.
2. **Emulation**: Development of machine learning models that emulate calibration effects and enable rapid predictions across parameter space.
3. **Inference & Bias Quantification**: Application of statistical tools to infer optimal
   calibration adjustments and quantify systematic biases in observations.

Together, these steps provide a scalable, data-driven pathway to understand and control
calibration uncertainties.

Library Generation
--------------------

The first stage of the **XCalForge** workflow is the generation of synthetic calibration libraries.
This process builds structured datasets that map controlled, parameterized calibration modifications
to their measurable effects on temperature recovery in X-ray observations. These libraries form the
foundation for machine learning emulator training and downstream statistical inference.

**Formalism**

For a given telescope calibration state :math:`\Theta` and an observed dataset :math:`\mathcal{D}`,
spectral fitting recovers physical parameters (e.g., plasma temperature) by maximizing the likelihood:

.. math::

    \mathcal{L}(\mathcal{D} \mid T, \Theta)

If the calibration :math:`\Theta` is imperfect—due to uncertainties in effective area,
response matrices, etc., the recovered parameter :math:`T_{\mathrm{fit}}` may differ
systematically from the true astrophysical value. The central goal of library generation
is to quantify how *modifications* to the assumed calibration affect this relationship.
A parametric family of calibration modifications :math:`\phi_{\theta}` is introduced, where:

- :math:`\theta` are user-defined parameters describing the modification (e.g., Gaussian ARF distortions).
- Applying :math:`\phi_{\theta}` produces a modified calibration state:

.. math::

    \Theta' = \phi_{\theta}(\Theta)

Synthetic spectra are generated using the modified calibration :math:`\Theta'`, simulating observations
that reflect potential calibration biases. The simulated data :math:`\mathcal{D}_{\theta}` is analyzed
using both:

- The *unmodified* calibration :math:`\Theta`, mimicking real-world analyses with imperfect knowledge.
- The *modified* calibration :math:`\Theta'`, serving as a ground-truth consistency check.

This enables measurement of the discrepancy:

.. math::

    \Delta T(\theta, T_{\mathrm{true}}) = T_{\mathrm{fit}}(\Theta) - T_{\mathrm{true}}

capturing how the assumed calibration leads to parameter biases as a function of
modification parameters :math:`\theta` and true temperatures :math:`T_{\mathrm{true}}`.

**Parameter Space and Mapping**:

To explore these effects, the modification parameters are discretized into a grid:

.. math::

    \mathcal{G} = \mathcal{A}_1 \times \mathcal{A}_2 \times \ldots \times \mathcal{A}_d

and a set of true source temperatures :math:`\mathcal{T} = \{ T_1, T_2, \ldots, T_{N_T} \}` is defined.

The library thus establishes a dense, precomputed mapping:

.. math::

    (\theta, T_{\mathrm{true}}) \longrightarrow \Delta T(\theta, T_{\mathrm{true}})

This dataset enables training fast, differentiable emulators to interpolate this mapping across parameter space,
avoiding costly on-the-fly simulations during inference.

.. note::

    The success of this approach depends critically on selecting a meaningful, physically motivated parameterization :math:`\theta`
    of calibration modifications. Typical examples include:

    - Gaussian distortions to the ARF effective area
    - Energy-dependent scaling of response matrices
    - User-defined perturbations based on known instrument systematics

**XCalForge** provides flexible tools to define and test such modification families within the library generation framework.

For further theoretical background, see :ref:`library_gen_theory`.
For detailed implementation guidance, refer to :ref:`library_gen_user`.


Emulation
------------

Leverage machine learning to build surrogate models that approximate calibration effects:

- Train neural emulators using generated spectral libraries
- Enable rapid interpolation within parameter spaces
- Reduce computational cost for inference and bias estimation

These emulators accelerate large-scale statistical studies and observational analyses.


Inference & Bias Quantification
---------------------------------

Apply statistical inference pipelines to model and quantify calibration-induced systematics:

- Perform temperature bias studies across instruments
- Fit observational datasets using emulator-enhanced models
- Quantify systematic offsets with robust uncertainty estimates

---

.. bibliography::
