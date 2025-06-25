.. _reference:

=============================
XCalForge | Reference Manual
=============================

**XCalForge** is a comprehensive toolkit for studying cross calibration issues in
X-ray astronomy.

This reference provides everything you need to install, operate,
and understand XCalForge, from technical setup to theoretical background.

.. contents::
   :local:
   :depth: 2
   :backlinks: none


🚀 Getting Started
===================

Everything you need to set up and deploy **XCalForge** for your research environment.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: 🔧 Installation Guide
      :link: install.html
      :text-align: center

      Step-by-step instructions for environment setup, dependencies, and HPC deployment with MPI.


📚 Core Workflows
===================

Master the core functionality of **XCalForge**, from generating synthetic spectral
libraries to applying statistical inference on observational datasets. At its core, **XCalForge** enables a
complete calibration workflow for X-ray astronomy, combining simulation,
machine learning, and statistical modeling:

Overview
--------

The core workflow in XCalForge involves simulating observations using different
telescope calibrations to construct a library of temperature discrepancies. These are
then used in an emulator to identify possible configuration changes to re-callibrate
telescopes.

.. toctree::
   :maxdepth: 1

   overview.rst


Library Generation
-------------------

Build high-fidelity synthetic libraries with support for distributed computing and large parameter spaces.

.. toctree::
   :maxdepth: 1
   :caption: Library Generation

   library/overview.rst


Inference & Machine Learning
-----------------------------

Leverage neural emulators and statistical methods to analyze simulation outputs or observational datasets.



📖 Background & Theory
=======================

Understand the scientific principles and statistical models behind **XCalForge**.

.. toctree::
   :maxdepth: 1
   :caption: Theoretical Foundations

   theory/libgen.rst


💻 Technical Notes
===================



🎯 Need Help?
===================

For troubleshooting, feature requests, or contributing:

- Visit the `GitHub Repository <https://github.com/eliza-diggins/XCalForge>`_
- Open an issue or discussion thread
