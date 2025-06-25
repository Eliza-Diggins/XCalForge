""" Environment management utilities for
XCalForge.
"""
import importlib
import logging
import sys
from typing import TYPE_CHECKING, Any, Union

from .config import xcfconfig

if TYPE_CHECKING:
    # special imports for type hint resolution
    # during static type checking.
    from mpi4py.MPI import Comm


# --------------------------------- #
# Environment Configuration Funcs.  #
# --------------------------------- #
# Functions for configuring the environment at the MPI
# and XSPEC levels.
def get_xspec():
    """
    Safely load the XSPEC PyXspec module, avoiding re-imports that can cause CLI-level deadlocks.

    Returns
    -------
    module
        The imported `xspec` module.

    Raises
    ------
    ImportError
        If the `xspec` module is not available or cannot be loaded.

    Notes
    -----
    - PyXspec can enter an unstable state if `xspec` is imported more than once across
      subprocesses or reloaded dynamically (e.g., during hot-reload or interactive development).
    - This function ensures that the module is loaded only once and reuses it afterward.
    - It is *strongly recommended* that all modules use `get_xspec()` rather than
      importing `xspec` directly.

    Example
    -------
    >>> xspec = get_xspec()
    >>> xspec.AllData.clear()
    """
    if "xspec" in sys.modules:
        return sys.modules["xspec"]

    try:
        xspec = importlib.import_module("xspec")
        return xspec
    except ImportError as e:
        logging.error(
            "Failed to import xspec module. Ensure PyXspec is installed and configured."
        )
        raise e


def configure_xspec():
    """
    Configure the XSPEC environment using the XCalForge configuration
    standards. This should be run before any standard operations.

    Notes
    -----
    - Should be called once at the start of XSPEC usage within XCalForge.
    - Safe to call multiple times with different settings if necessary.
    - Uses `get_xspec()` to avoid unsafe re-imports across processes.

    Example
    -------
    >>> configure_xspec(fit_statistic='cstat', chatter=5)
    """
    xspec = get_xspec()

    # Set global XSPEC verbosity
    xspec.Xset.chatter = xcfconfig["xspec.chatter"]

    # Set fitting configuration
    xspec.Fit.statMethod = xcfconfig["xspec.fit_statistic"]
    xspec.Fit.statTest = xcfconfig["xspec.test_statistic"]
    xspec.Fit.query = "yes"
    xspec.Fit.criticalDelta = float(xcfconfig["xspec.fit_convergence_tol"])
    xspec.Fit.nIterations = int(xcfconfig["xspec.max_fit_iterations"])


def clear_xspec():
    """Erase the current XSPEC model and data references."""
    xspec = get_xspec()
    xspec.AllData.clear()
    xspec.AllModels.clear()


def get_mpi(comm_world: bool = True) -> Union[Any, "Comm"]:
    """
    Safely load the MPI communicator from mpi4py, with configurable return behavior.

    Parameters
    ----------
    comm_world : bool, optional
        If True (default), returns the global communicator `MPI.COMM_WORLD`.
        If False, returns the `MPI` module itself for advanced operations.

    Returns
    -------
    mpi4py.MPI.Comm or mpi4py.MPI
        - `MPI.COMM_WORLD` communicator if `comm_world` is True.
        - Full `MPI` module if `comm_world` is False.

    Raises
    ------
    ImportError
        If `mpi4py` is not available or cannot be loaded.

    Notes
    -----
    - Ensures `mpi4py` is imported only once, avoiding redundant imports.
    - Supports environments where MPI may not be available by default (e.g., serial fallbacks).
    - Logs errors clearly if `mpi4py` cannot be imported.
    - Recommended to use this accessor rather than importing `mpi4py.MPI` directly to avoid issues in parallel or hybrid environments.

    Examples
    --------
    >>> comm = get_mpi()
    >>> rank = comm.Get_rank()
    >>> size = comm.Get_size()

    For direct access to the MPI module:

    >>> MPI = get_mpi(comm_world=False)
    >>> custom_comm = MPI.COMM_SELF
    """
    if "mpi4py.MPI" in sys.modules:
        if comm_world:
            return sys.modules["mpi4py.MPI"].COMM_WORLD
        else:
            return sys.modules["mpi4py.MPI"]

    try:
        from mpi4py import MPI

        if comm_world:
            return MPI.COMM_WORLD
        else:
            return MPI
    except ImportError as e:
        logging.error(
            "Failed to import mpi4py. Ensure MPI and mpi4py are correctly installed and configured."
        )
        raise e
