.. _reference:

=====================
XCalForge Reference
=====================

.. contents::
   :local:
   :depth: 2

Getting Started
---------------

Installation
~~~~~~~~~~~~

Step-by-step instructions for setting up the XCalForge environment, managing dependencies, and preparing your system. Includes guidance for HPC deployments, XSPEC configuration, and MPI support.

.. toctree::
   :titlesonly:
   :caption: Install
   :maxdepth: 1

   install.rst

Library Generation
~~~~~~~~~~~~~~~~~~

Comprehensive guide to building synthetic data libraries using parameter grids. Covers grid construction, distributed simulation with MPI, and efficient finalization with HDF5 outputs.

.. toctree::
   :titlesonly:
   :caption: Library Generation
   :maxdepth: 1

   library/overview.rst

Inference and Machine Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Overview of performing scientific inference with XCalForge. Includes workflows for training neural emulators, modeling temperature distortions, and applying parameter inference to observational data.

Background
----------

This section covers various aspects of the underlying theory and statistical approach of the XCalForge approach:

.. toctree::
   :titlesonly:
   :caption: Background
   :maxdepth: 1

   theory/libgen.rst
