XCalForge
=============================

|isort| |black| |Pre-Commit| |docformatter| |NUMPSTYLE| |Commits| |Contributors| |docs|

.. raw:: html

   <hr style="height:2px;background-color:black">

**XCalForge** is a research-oriented framework for analyzing and correcting calibration discrepancies
between X-ray telescopes using simulation-driven modeling and machine learning.

XCalForge enables users to:

- Apply parametric modifications to instrument response files (e.g., ARFs)
- Generate synthetic spectra via XSPEC with modified configurations
- Fit models to simulated data to quantify calibration impact
- Train neural emulators to predict the effect of modifications
- Infer optimal global calibration adjustments from observational discrepancies

The system is optimized for HPC environments and supports:

- MPI-parallel synthetic library generation
- HDF5 output for scalable data storage
- Seamless PyXspec integration for XSPEC simulation control

Whether correcting ARF biases or validating temperature consistency across missions,
**XCalForge** provides a principled and extensible toolkit for addressing
instrument cross-calibration challenges in X-ray astronomy.

Getting Started
----------------

To get started with XCalForge, install it by following the instructions on the :ref:`install` page. Once
the code is installed, check out the :ref:`reference` guide for detailed explanation about how to use the code.

Contents
------------------------------


.. toctree::
   :maxdepth: 1

   api
   reference/index

Indices and Tables
------------------------------

* :ref:`genindex` – General index of all documented terms
* :ref:`modindex` – Python module index
* :ref:`search` – Search the documentation

.. |docs| image:: https://img.shields.io/badge/docs-latest-brightgreen
   :target: https://eliza-diggins.github.io/XCalForge/build/html/index.html

.. |Pre-Commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://pre-commit.com/

.. |Contributors| image:: https://img.shields.io/github/contributors/Eliza-Diggins/XCalForge
   :target: https://github.com/Eliza-Diggins/XCalForge/graphs/contributors

.. |Commits| image:: https://img.shields.io/github/last-commit/Eliza-Diggins/XCalForge

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000
   :target: https://github.com/psf/black

.. |isort| image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336
   :target: https://pycqa.github.io/isort/

.. |NUMPSTYLE| image:: https://img.shields.io/badge/%20style-numpy-459db9
   :target: https://numpydoc.readthedocs.io/en/latest/format.html

.. |docformatter| image:: https://img.shields.io/badge/%20formatter-docformatter-fedcba
   :target: https://github.com/PyCQA/docformatter
