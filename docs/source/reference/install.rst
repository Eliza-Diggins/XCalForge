.. _install:
Installation
============

**XCalForge** is designed for high-performance computing (HPC) environments and depends on system-level software
including **MPI**, **HEASOFT**, and **PyXspec**. To streamline installation, it uses a
modular ``Makefile`` and per-system bootstrap scripts. If you encounter issues with the bootstrapping process, reach out
to the maintainer or consult your system administrator.

Dependencies
------------

The core runtime and simulation framework depends on the following:

- `numpy <http://www.numpy.org>`__ — Numerical operations
- `h5py <http://www.h5py.org>`__ — HDF5 file interaction
- `astropy <https://www.astropy.org/>`__ — Table and FITS file handling
- `mpi4py <https://mpi4py.readthedocs.io/>`__ — Python bindings for MPI
- `PyXSPEC <https://heasarc.gsfc.nasa.gov/xanadu/xspec/python/html/index.html>`__ — XSPEC bindings via HEASOFT

System-level requirements (pre-installed on most HPCs):

- A valid MPI implementation (e.g., `OpenMPI`, `IntelMPI`)
- A working installation of `HEASoft` (v6.30+ recommended) with PyXspec enabled

Cloning the Repository
----------------------

The first step in installing XCalForge is to clone this github repository into
your preferred location:

.. code-block:: bash

    git clone https://github.com/eliza-diggins/XCalForge.git
    cd XCalForge

This will be the location of the installation, so ensure you have things where you want them. Because of
its reliance on system level software, ``XCalForge`` cannot be automatically installed by systems like ``pip`` and
requires some additional effort to set up on a system-by-system level.

Bootstrapping
-------------

Once the repository has been cloned, a **bootstrapping** script is required to fully install
the software. The bootstrapping script effectively ensures that the dependencies are available, the relevant
HPC modules are available, and then creates a python virtual environment and installs the package.

In some systems, bootstrapping scripts are already available. If your system does not already
have a bootstrap available, then you will need to write it. The ``/bootstrap/install_SKELETON.sh`` file
contains a model for the steps to help make things easy.

**Available bootstraps:**

- ``install_chpc.sh`` — for CHPC (University of Utah)

Adding New bootstraps
*********************

To add support for a new system:

1. Copy the template and name it appropriately:

   .. code-block:: bash

       cp bootstrap/install_SKELETON.sh bootstrap/install_<system>.sh

2. Modify module names, paths, or environment setup for your cluster.
3. Add a corresponding `Makefile` target:

   .. code-block:: make

       install-<system>:
           bash bootstrap/install_<system>.sh

Installing From Bootstrap
-------------------------

Once a bootstrap script exists, you can install the entire module using

.. code-block:: bash

   make install-<system>

This will automatically perform all of the relevant bootstrapping steps.


Environment Initialization
--------------------------

Before running any XCalForge tools, you must load your environment if you have not already done so.
This is done via a **setup script**, which ensures the proper modules and virtual environment are activated.

Run:

.. code-block:: bash

    source setup_env.sh <system>

Where `chpc` matches the system you're on (must correspond to a `setup_env_<system>.sh` script).

This loads the HEASoft and MPI modules and activates the Python virtual environment created during install.

Advanced Installation
---------------------

To install optional dependencies or modify pip installation options, first set up the environment
and then install via pip.

.. code-block:: bash

    pip install . --no-build-isolation

To install additional optional dependency groups (e.g., for documentation, testing, etc.):

.. code-block:: bash

    pip install .[docs]
    pip install .[dev]

Or install all extras at once:

.. code-block:: bash

    pip install ".[all]"

Verifying Installation
----------------------

You can verify MPI and XSPEC bindings with:

.. code-block:: bash

    python -c "import mpi4py.MPI; import xspec; print('MPI and XSPEC successfully loaded')"

You’re now ready to start using **XCalForge** to simulate instrument modifications and perform inference.
