XCalForge
===============

|isort| |black| |Pre-Commit| |docformatter| |NUMPSTYLE| |Commits| |CONTRIBUTORS| |docs|

.. raw:: html

   <hr style="height:2px;background-color:black">


**XCalForge** is a research-oriented framework for analyzing and correcting calibration discrepancies
 between X-ray telescopes through simulation-driven modeling and machine learning.

It allows users to:

- Apply parametric modifications to instrument response files (e.g., ARFs)
- Generate synthetic spectra via XSPEC with modified configurations
- Fit models to simulated data to evaluate calibration impact
- Train neural emulators to learn the effect of modifications
- Infer optimal global calibration adjustments from observed discrepancies

The system is optimized for use on HPC environments and supports MPI-parallel library generation, HDF5 output, and integration with PyXspec for seamless XSPEC simulation control.

Whether you're correcting for ARF biases or validating temperature consistency across missions, **XCalForge** provides a principled and
extensible toolkit for tackling instrument cross-calibration challenges in X-ray astronomy.


Contents
--------

.. raw:: html

   <hr style="height:10px;background-color:black">

.. toctree::
   :maxdepth: 1

   api
   reference/index

Indices and Tables
------------------

.. raw:: html

   <hr style="height:10px;background-color:black">

* :ref:`genindex` – General index of all documented terms
* :ref:`modindex` – Python module index
* :ref:`search` – Search the documentation


.. |docs| image:: https://img.shields.io/badge/docs-latest-brightgreen
   :target: https://eliza-diggins.github.io/pisces/build/html/index.html

.. |Pre-Commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://pre-commit.com/

.. |Xenon| image:: https://img.shields.io/badge/Xenon-enabled-red
   :target: https://xenon.readthedocs.io/en/latest/

.. |Tests| image:: https://github.com/Pisces-Project/PyMetric/actions/workflows/run_tests.yml/badge.svg

.. |Contributors| image:: https://img.shields.io/github/contributors/Pisces-Project/PyMetric
   :target: https://github.com/Eliza-Diggins/pisces/graphs/contributors

.. |Commits| image:: https://img.shields.io/github/last-commit/Pisces-Project/PyMetric

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000
   :target: https://github.com/psf/black

.. |isort| image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336
   :target: https://pycqa.github.io/isort/

.. |NUMPSTYLE| image:: https://img.shields.io/badge/%20style-numpy-459db9
    :target: https://numpydoc.readthedocs.io/en/latest/format.html

.. |docformatter| image:: https://img.shields.io/badge/%20formatter-docformatter-fedcba
    :target: https://github.com/PyCQA/docformatter

.. |License| image:: https://img.shields.io/pypi/l/pymetric-lib
.. |Wheel| image:: https://img.shields.io/pypi/wheel/pymetric-lib
.. |PyVersion| image:: https://img.shields.io/pypi/pyversions/pymetric-lib
.. |PyPi| image:: https://img.shields.io/pypi/v/pymetric-lib
