"""
Library generation base classes.

These are the base classes for generating the modification libraries
which are then used to train the interpolators.
"""
import logging
import os
import tempfile
import time
import traceback
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

from xforge.spectra import generate_synthetic_spectrum, group_min_counts
from xforge.utilities import (
    ConfigManager,
    RankFormatter,
    clear_xspec,
    get_mpi,
    get_xspec,
    spec_logger,
    xcfconfig,
)

from .utils import ProgressMonitor, sync_progress

# ---------------------------------- #
# Type Checking                      #
# ---------------------------------- #
if TYPE_CHECKING:
    # import special types and type hints so that they
    # are readable to static type checkers.
    from mpi4py.MPI import Comm

# Ensure access to XSPEC via PyXSPEC. This needs
# to be done carefully because there can only be one CLI active
# per process.
xspec = get_xspec()
MPI: Any = get_mpi(comm_world=False)

# -------------------------------------- #
# Modification Library Base Class        #
# -------------------------------------- #


class ModificationLibrary(ABC):
    """
    Abstract base class for calibration modification libraries.

    Modification libraries are structured, disk-backed datasets that map
    telescope configuration modifications to observable temperature recovery
    performance through synthetic X-ray simulations.

    This class provides core logic for:

    - Managing a reproducible library directory structure
    - Loading and validating parameter grids from HDF5
    - Tracking configuration and metadata via YAML
    - Parallelized, MPI-based synthetic data generation with XSPEC
    - Storing results for downstream machine learning or inference tasks

    Subclasses must implement concrete behavior for:

    1. Generating *unmodified* configurations:
       Defines the baseline, nominal telescope setup used during fitting,
       representing an outdated, incorrect, or biased calibration state.

    2. Generating *modified* configurations:
       Defines perturbed telescope setups incorporating proposed calibration
       modifications. Used to generate synthetic "true" observations.

    3. Constructing XSPEC models for both configurations at a given physical parameter
       (typically temperature).

    4. Fitting routines for extracting temperature estimates from synthetic spectra
       using both configurations.

    Library Generation Workflow:

    - A multi-dimensional grid of calibration parameters is specified.
    - For each grid point:
        * Synthetic spectra are generated with the modified configuration (true calibration).
        * Spectra are analyzed with both modified and unmodified configurations.
        * Discrepancies in recovered temperatures are recorded.
    - This process is distributed across MPI ranks for efficiency.
    - Final results are consolidated into a HDF5-backed library.

    The resulting library provides a structured dataset to:

    - Train machine learning emulators for calibration inference.
    - Quantify the impact of calibration uncertainties on temperature recovery.
    - Systematically explore complex calibration modifications.

    See Also:
    ---------
    :meth:`create_library` : Initialize new libraries with specified parameter grids.
    :meth:`generate_library` : Run synthetic data generation in parallel.
    """

    # ================================ #
    # CLASS FLAGS                      #
    # ================================ #
    # The class flags are simple markers that are used
    # for various behaviors in the code below. These may need to
    # be modified in subclasses. Please read the relevant documentation
    # in order to determine if a particular flag needs to be modified.
    __PARAMETERS__: List[str] = []
    """
    List of allowed parameter names for this modification library.

    - These parameters define the axes of the parameter grid for the library.
    - Each element should be a string corresponding to a valid hyperparameter (e.g., "mu", "sigma", "amplitude").
    - Parameter arrays for these keys are stored in the "PARAMS" group of the HDF5 file.
    - If any parameter in the input grid does not appear in this list, an error is raised during initialization.

    Subclasses MUST override this to explicitly declare the valid parameters for the modification model.
    """

    __CONFIG__: Dict[str, Any] = {
        "logging.file_level": xcfconfig["logging.library.file_level"],
        "logging.term_level": xcfconfig["logging.library.terminal_level"],
        "logging.fmt": xcfconfig["logging.library.format"],
        "prog.nsync": xcfconfig["libgen.nsync"],
    }
    """
    Default configuration dictionary for the library.

    - Populates the persistent `config.yaml` during library creation.
    - Supports hierarchical, nested configuration via dotted keys.
    - Can be overridden at library creation time using the `config` argument.
    - Common options include:

        * `"logging.level"`: Default logging verbosity (e.g., "INFO", "DEBUG").
        * `"logging.fmt"`: Optional custom logging format string.
        * Additional subclass-specific settings (e.g., file paths, hyperparameters).

    Subclasses can extend this to provide additional default options relevant to the specific modification model.
    """

    __OUTPUT_SHAPE__: Tuple[int, ...] = (3,)
    """
    Shape of the stored output for each parameter-temperature grid point.

    Defaults to (3,), corresponding to (recovered temperature, fit statistic, uncertainty).
    Subclasses may override to extend this, e.g., (5,) for additional diagnostics.
    """

    # ========================== #
    # Initialization             #
    # ========================== #
    # These methods are all concerned with loading libraries from
    # disk. They can be modified in subclasses, but should be kept
    # relatively consistent.
    def __init__(self, directory: Union[str, Path]):
        """
        Load an existing modification library from disk.

        This initializer loads the library's configuration, parameter grid, and output paths,
        preparing the object for read access, logging, and potential downstream simulation or analysis.

        Parameters
        ----------
        directory : str or Path
            Path to the root directory of the library. This directory must already contain
            a valid library structure, including:

            - `library.h5` : HDF5 file with parameter grids and (optionally) simulation results.
            - `config.yaml` : Persistent configuration file defining logging levels, runtime options, etc.
            - `logs/` : Directory for per-rank log outputs (will be created if missing).
            - `cache/` : Directory for temporary working files (will be created if missing).
            - `bin/` : Optional directory for persistent generated files (e.g., ARFs).

        Raises
        ------
        ValueError
            If the specified directory does not exist.
        FileNotFoundError
            If critical components of the library structure are missing, such as `config.yaml` or `library.h5`.

        Notes
        -----
        - This method does NOT generate a new library; it is strictly for loading and interacting with
        pre-existing libraries generated via `create_library`.
        - The configuration is loaded using :class:`~xforge.utils.ConfigManager` and
        made accessible via the `.config` property.
        - Logger setup is performed automatically, including rank-aware log file creation in distributed environments.

        See Also
        --------
        :meth:`create_library` : For creating new libraries with validated structure and default configuration.
        """
        # Configure the libraries directory and ensure that
        # the path exists before proceeding.
        self.__directory__ = Path(directory).expanduser().resolve()
        if not self.__directory__.exists():
            raise ValueError(f"The directory `{self.__directory__}` does not exist.")

        # Construct the ``__assets__`` dictionary to act as a container
        # around various different file assests for quick access and
        # iteration control.
        self.__assets__: Dict[str, Any] = {}

        # Validate that the file structure is sufficient.
        self.__validate_structures__()

        # Load the configuration file.
        self.__init_config__()

        # Setup the logger.
        self.__init_logger__()

        # Initialize the parameters.
        self.__init_parameters__()

        self.__post_init__()

        self.logger.info("Loaded Library @ %s.", self.__directory__)

    def __validate_structures__(self):
        """
        Validates the required file and directory structure for the modification library.

        Ensures that critical components are present in the library directory, including:
        - Cache directory for temporary generation data.
        - Bin directory for persistent generated files (e.g., ARFs).
        - Combined output HDF5 file (must exist).
        - Configuration YAML file (must exist).

        These paths are stored in `self.__assets__` for downstream use.

        Raises
        ------
        FileNotFoundError
            If any required file or directory is missing (aside from cache/bin which will be created).
        """
        self.__cachedir__ = self.__directory__ / "cache"
        self.__tempdir__ = None
        self.__bindir__ = self.__directory__ / "bin"
        self.__logdir__ = self.__directory__ / "logs"
        self.__configpath__ = self.__directory__ / "config.yaml"
        self.__datapath__ = self.__directory__ / "library.h5"

        if not self.__configpath__.exists():
            raise FileNotFoundError(
                f"Missing required config file: {self.__configpath__}"
            )
        if not self.__datapath__.exists():
            raise FileNotFoundError(f"Missing required data file: {self.__datapath__}")

        self.__cachedir__.mkdir(parents=True, exist_ok=True)
        self.__bindir__.mkdir(parents=True, exist_ok=True)
        self.__logdir__.mkdir(parents=True, exist_ok=True)

        self.__assets__.update(
            {
                "cache": self.__cachedir__,
                "bin": self.__bindir__,
                "log": self.__logdir__,
                "config": self.__configpath__,
                "data": self.__datapath__,
            }
        )

    def __init_config__(self):
        """
        Loads the configuration file using the `ConfigManager` class.
        """
        self.__config__ = ConfigManager(self.__configpath__)

    def __init_logger__(self):
        """
        Sets up:

        - self.logger: Logs to file with full formatting
        - self.console_logger: Plain console output, used explicitly by user code

        Logging behavior is explicit. No hidden rank-based console suppression.
        """
        from xforge.utilities.env import get_mpi

        # Resolve necessary logger information before proceeding with
        # the setup procedures.
        comm: "Comm" = get_mpi()
        rank: int = comm.Get_rank()
        logger_name = f"Library.{self.__directory__.name}.Rank{rank}"

        # Configure the rank specific logger for this MPI rank.
        # This is the logger that should be used for most purposes.
        # By default, only very high level warnings and errors on this
        # logger will go to stdout. Everything else is dumped to files in
        # the logs.
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(self.config.get("logging.file_level", logging.DEBUG))
        self.logger.propagate = False  # Avoid duplicating to root logger

        if not self.logger.hasHandlers():
            fmt = self.config.get(
                "logging.fmt",
                "%(asctime)s [%(levelname)s] [RANK=%(rank)s]: %(message)s",
            )
            formatter = RankFormatter(fmt, rank=rank)

            logfile = self.__directory__ / f"logs/rank_{rank}.log"
            file_handler = logging.FileHandler(logfile, mode="w")
            file_handler.setLevel(self.config.get("logging.level", logging.DEBUG))
            file_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)

        # Set the XSPEC logger so that we don't get XSPEC output
        # to the stdout.
        xspec.Xset.openLog(str(self.__directory__ / f"logs/xspec_rank_{rank}.log"))

    def __init_parameters__(self):
        """
        Loads the parameter arrays from the 'PARAMS' group in the HDF5 data file.

        Raises
        ------
        KeyError
            If the 'PARAMS' group is not found.
        """
        # Load the HDF5 file with the data. This is where we store the
        # parameter arrays as well.

        with h5py.File(self.__datapath__, "r") as f:
            # Check that we have a PARAMS group in the
            # HDF5.
            if "PARAMS" not in f:
                raise KeyError(f"'PARAMS' group not found in {self.__datapath__}")

            param_group: h5py.Group = f["PARAMS"]  # type: ignore[index]

            # Iterate through the keys and select only the
            # permissible parameters. Then ensure that all parameters
            # are present.
            self.__parameters__ = {}
            for key in param_group.keys():
                # If the key is unknown, we raise a warning
                # and then skip along.
                if key not in self.__class__.__PARAMETERS__:
                    self.logger.warning(
                        "Found key %s in `PARAMS`, which is not in __PARAMETERS__.", key
                    )
                    continue

                # If they key is known, they we extract the
                # data and set it.
                self.__parameters__[key] = param_group[key][...]  # type: ignore[index]

        # Finally, ensure that all of the necessary parameters
        # are specified. If they are not, we raise an error.
        if any(i not in self.__parameters__ for i in self.__PARAMETERS__):
            raise ValueError("Missing parameters!")

    @abstractmethod
    def __post_init__(self):
        """
        Perform subclass-specific initialization logic after the base library is loaded.

        This method is called automatically at the end of `__init__`, after the following steps:

        - The library directory structure has been validated.
        - The configuration file (`config.yaml`) has been loaded.
        - The logger has been initialized with rank-aware output.
        - The parameter grid has been loaded from the HDF5 data file.

        Subclasses can override this method to:

        - Perform additional sanity checks specific to the modification model.
        - Load supplementary resources (e.g., precomputed files, model templates).
        - Configure internal attributes required for the generation process.
        - Adjust default configuration entries before simulation begins.

        Notes
        -----
        - This method should avoid heavy operations like file I/O unless strictly necessary.
        - It should NOT initiate simulations, fitting, or XSPEC operations.
        - Always call `super().__post_init__()` in subclass overrides to preserve base functionality if applicable.

        Example
        -------
        A subclass might implement:

        .. code-block:: python

            def __post_init__(self):
                super().__post_init__()
                self.my_model_path = self.__bindir__ / "baseline_model.xcm"
                if not self.my_model_path.exists():
                    raise FileNotFoundError("Baseline model not found: {self.my_model_path}")

        """
        ...

    # ============================ #
    # Building Libraries.          #
    # ============================ #
    # These methods focus on ensuring that new libraries
    # can be created smoothly. These do not actually run the
    # library generation process.
    @classmethod
    def create_library(
        cls,
        directory: Union[str, Path],
        parameters: Dict[str, Sequence],
        *args,
        overwrite: bool = False,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize a new modification library directory on disk with the required structure,
        parameter grid, and optional configuration file.

        This method prepares the library directory but does not run the generation process.
        After creation, the library instance can be used to perform simulations and populate results.

        Parameters
        ----------
        directory : str or Path
            Path to the root of the new library. This will contain the generated HDF5 data,
            configuration files, cache, and outputs.
        parameters : dict[str, Sequence]
            Parameter grid specifying each axis of the modification space. Keys must match the
            class's allowed `__PARAMETERS__`. Each value should be a 1D array-like sequence.
        overwrite : bool, default = False
            Whether to overwrite an existing directory. If False and the target directory exists,
            a `ValueError` is raised. If True, the directory is removed and recreated.
        config : dict, optional
            Optional dictionary of configuration values to write to `config.yaml`.
            These override defaults defined in `__CONFIG__`.
        args, kwargs:
            Additional arguments and kwargs for subclass implementations.


        Returns
        -------
        ModificationLibrary
            An instance of the initialized library ready for generation or inspection.

        Raises
        ------
        ValueError
            If the directory exists and `overwrite` is False.
        """
        import gc
        import shutil

        # Resolve the specified directory in
        # which to create the new library.
        directory = Path(directory).expanduser().resolve()

        # Check for overwrite. The try, except structure here is
        # present to catch issues when file systems leave hanging files
        # that catch shutil.
        if directory.exists():
            if not overwrite:
                raise ValueError(
                    f"Directory `{directory}` already exists. "
                    "Use `overwrite=True` to recreate it."
                )

            # Attempt to remove the existing
            # directory but accept failure if we are not
            # able to successfully do so.
            try:
                gc.collect()
                shutil.rmtree(directory)
            except Exception as e:
                raise OSError(
                    f"Failed to delete directory {directory}.\n"
                    "This is likely the result of hanging references in the directory"
                    "due to file system behavior, not XCalForge. Try deleting manually.\n"
                    f"Error: {e}"
                )

        # Create required subdirectories and files
        cls.__create_structures__(directory)
        cls.__write_datafile__(directory, parameters)
        cls.__write_config__(directory, config=config)

        # Instantiate and return the initialized library
        spec_logger.info("Created library @ %s. (Class=%s)", directory, cls.__name__)
        return cls(directory)

    @classmethod
    def __create_structures__(cls, directory: Path):
        """
        Create the required subdirectories for the new library.

        Parameters
        ----------
        directory : str or Path
            The base library directory.
        parameters : dict
            Parameter dictionary used for validation (optional).
        """
        (directory / "cache").mkdir(parents=True, exist_ok=True)
        (directory / "bin").mkdir(parents=True, exist_ok=True)

    @classmethod
    def __write_datafile__(cls, directory: Path, parameters: Dict[str, Sequence]):
        """
        Write the HDF5 library file with the parameter grid.

        Parameters
        ----------
        directory : str or Path
            Path to the library directory.
        parameters : dict[str, Sequence]
            Dictionary of parameter arrays.
        """
        datafile_path = directory / "library.h5"

        with h5py.File(datafile_path, "w") as f:
            param_group = f.create_group("PARAMS")

            for key, values in parameters.items():
                if key not in cls.__PARAMETERS__:
                    raise ValueError(
                        f"Invalid parameter key `{key}`. Allowed: {cls.__PARAMETERS__}"
                    )
                param_group.create_dataset(key, data=np.array(values, dtype="f8"))

    @classmethod
    def __write_config__(
        cls, directory: Union[str, Path], config: Optional[Dict[str, Any]] = None
    ):
        """
        Write the config.yaml file using a ConfigManager.

        Ensures that default fields are present, but allows user overrides and additions.

        Parameters
        ----------
        directory : str or Path
            Path to the library directory.
        config : dict, optional
            Optional dictionary of configuration values. May contain dotted or nested keys.
        """
        config_path = Path(directory) / "config.yaml"
        cfg = ConfigManager(config_path, autosave=True)

        # Merge defaults and user-specified config; user config takes priority
        merged_config = dict(cls.__CONFIG__)
        if config is not None:
            merged_config.update(config)

        for key, value in merged_config.items():
            cfg[key] = value

    # ========================== #
    # Dunder Methods             #
    # ========================== #
    def __del__(self):
        """
        Destructor to clean up temporary directories when the instance is deleted.

        This attempts to remove the per-rank temporary directory if it exists.
        Non-fatal errors during deletion are suppressed to avoid interfering with interpreter shutdown.
        """
        try:
            self.clear_tempdir()
            xspec.Xset.closeLog()
        except Exception:
            # Suppress all exceptions during interpreter shutdown
            pass

    # ========================== #
    # Properties                 #
    # ========================== #
    @property
    def config(self) -> ConfigManager:
        """
        Access the YAML-backed configuration for this library.

        Provides persistent, dictionary-like access to all stored configuration options,
        typically loaded from `config.yaml`.

        Returns
        -------
        ConfigManager
            The configuration manager for this library instance.
        """
        return self.__config__

    @property
    def parameters(self) -> Dict[str, np.ndarray]:
        """
        Access the parameter grid defining the modification space.

        The parameters are loaded from the `PARAMS` group in the HDF5 file at initialization,
        and provide the 1D arrays defining each axis of the parameter lattice.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary mapping parameter names to their 1D value arrays.
        """
        return self.__parameters__

    @property
    def size(self) -> int:
        """
        Total number of unique parameter combinations in the modification grid.

        Computed as the product of the lengths of all parameter arrays.

        Returns
        -------
        int
            Number of grid points (i.e., total simulations per temperature).
        """
        return int(np.prod([len(values) for values in self.__parameters__.values()]))

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Shape of the parameter grid across all modification axes.

        Each element in the tuple corresponds to the length of one parameter array.

        Returns
        -------
        tuple of int
            The dimensional shape of the parameter lattice.
        """
        return tuple(len(values) for values in self.__parameters__.values())

    @property
    def is_generated(self) -> bool:
        """
        Check if the synthetic library results have been generated.

        Returns True if the 'LIBRARY' group exists in the HDF5 file, indicating that
        results and temperature arrays are present.

        Returns
        -------
        bool
            True if results exist, False otherwise.
        """
        with h5py.File(self.__assets__["data"], "r") as f:
            return "LIBRARY" in f

    @property
    def temperatures(self) -> Optional[np.ndarray]:
        """
        Access the list of simulated temperatures in the library.

        Returns
        -------
        np.ndarray or None
            Array of temperature values used in the simulations, or None if results
            have not yet been generated.
        """
        if not self.is_generated:
            return None

        with h5py.File(self.__assets__["data"], "r") as f:
            return f["LIBRARY"]["temps"][...]  # type: ignore[index]

    @property
    def tempdir(self) -> Path:
        """
        Node-local, per-rank temporary working directory.

        Uses $TMPDIR if available (e.g., on HPC clusters),
        falls back to system default via tempfile.gettempdir().

        The directory is isolated by:
        - Library name
        - MPI rank

        Returns
        -------
        Path
            Path to the node-local temporary directory for this rank.
        """
        if self.__tempdir__ is None:
            tmp_root = os.environ.get("TMPDIR", tempfile.gettempdir())
            comm = get_mpi()
            rank = comm.Get_rank()

            self.__tempdir__ = Path(tmp_root) / f"{self.__directory__.name}_rank_{rank}"
            self.__tempdir__.mkdir(parents=True, exist_ok=True)
        else:
            pass

        return self.__tempdir__

    # ------------------------------------ #
    # Utility Methods                      #
    # ------------------------------------ #
    @contextmanager
    def library(self, mode: str = "r"):
        """
        Context manager for safe, lazy access to the generated library results.

        This provides access to:

        - The `results` dataset, containing parameter recovery outputs.
        - The `temps` dataset, containing the temperature grid.

        Example usage:

        .. code-block:: python

            with lib.library() as (results, temps):
                print(results.shape)
                print(temps)

        Parameters
        ----------
        mode : str, default = "r"
            File mode for opening the HDF5 file (e.g., "r", "r+", "a").

        Yields
        ------
        Tuple[h5py.Dataset, h5py.Dataset]
            Tuple of (results, temperatures) datasets from the 'LIBRARY' group.

        Raises
        ------
        RuntimeError
            If the library has not yet been generated and no 'LIBRARY' group exists.
        """
        if not self.is_generated:
            raise RuntimeError(
                "This library has not been generated. Run the generation process before accessing results."
            )

        with h5py.File(self.__assets__["data"], mode) as f:
            lib_group = f["LIBRARY"]
            yield lib_group["results"], lib_group["temps"]  # type: ignore[index]

    def set_logging_level(
        self, level: Union[int, str], handlers: Optional[List[logging.Handler]] = None
    ):
        """
        Update the logging level for the library logger and its handlers.

        Parameters
        ----------
        level : int or str
            Desired logging level (e.g., logging.DEBUG or "DEBUG").
        handlers : list[logging.Handler], optional
            List of specific handlers to update. If None, all attached handlers are updated.
        """
        import logging

        # Update the config for persistence
        level = logging.getLevelName(level) if isinstance(level, int) else level
        self.config["logging.level"] = level

        # Update logger level
        self.logger.setLevel(level)

        # Select handlers
        target_handlers = handlers if handlers is not None else self.logger.handlers

        # Update handler levels
        for h in target_handlers:
            h.setLevel(level)

        self.logger.debug("Logger level set to %s.", logging.getLevelName(level))

    def clear_tempdir(self):
        """
        Completely removes the per-rank temporary directory and resets internal state.

        The directory will be recreated automatically on next access.
        """
        import shutil

        if self.__tempdir__ is not None and self.__tempdir__.exists():
            try:
                shutil.rmtree(self.__tempdir__)
                self.logger.debug("Deleted tempdir: %s", self.__tempdir__)
            except Exception as e:
                self.logger.warning(
                    "Failed to delete tempdir %s: %s", self.__tempdir__, e
                )

        self.__tempdir__ = None

    # ================================ #
    # Library Generation - User Hooks  #
    # ================================ #
    @abstractmethod
    def generate_unmodified_configuration(self, id: int, **parameters) -> dict:
        """
        Define the unmodified (baseline) telescope configuration for a given parameter grid point.

        The "unmodified" configuration represents the current or assumed telescope setup in operational use.
        It reflects how the instrument is calibrated in practice, even if that calibration is incorrect
        or outdated. This configuration will be used during the fitting stage to mimic real-world biases
        in parameter recovery.

        Typical fields include:

        - "response": Path to RMF file (energy redistribution matrix)
        - "arf": Path to ARF file (effective area curve)
        - "exposure": Exposure time in seconds (default: 50,000)
        - Optional: "background", "correction", "backExposure"

        Users may dynamically generate or retrieve these files based on `parameters`, for example to test
        different RMF versions, apply no modification to ARFs, or implement known flawed configurations.

        Parameters
        ----------
        id : int
            Global linear index into the parameter grid for this simulation.
        **parameters : dict
            Dictionary of hyperparameter values for this grid point, as defined by the library's parameter grid.

        Returns
        -------
        dict
            Dictionary of file paths and settings required by XSPEC's `fakeit` and fitting routines.
        """
        ...

    @abstractmethod
    def generate_modified_configuration(self, id: int, **parameters) -> dict:
        """
        Define the modified telescope configuration incorporating calibration perturbations.

        The "modified" configuration represents the true, unknown state of the instrument after applying
        a simulated calibration distortion (e.g., ARF bias, instrument drift). This is used to generate
        synthetic photon spectra that reflect calibration errors.

        The modified configuration typically mirrors the unmodified version, with selective changes introduced,
        such as distorted ARFs or different instrumental responses.

        Parameters
        ----------
        id : int
            Global linear index into the parameter grid for this simulation.
        **parameters : dict
            Dictionary of hyperparameter values for this grid point, controlling the modification (e.g., Gaussian ARF parameters).

        Returns
        -------
        dict
            Dictionary of file paths and settings defining the perturbed telescope configuration for synthetic data generation.
        """
        ...

    @abstractmethod
    def generate_model_unmodified(self, T: float, **parameters):
        """
        Construct the XSPEC spectral model corresponding to the unmodified configuration.

        This defines the astrophysical emission model (e.g., thermal plasma) applied when fitting
        the synthetic spectra with the unmodified, baseline calibration.

        This should match the scientific model expected by the user but must assume the unmodified instrument setup.
        In most cases, the same astrophysical model is used in both unmodified and modified stages,
        but with different calibration files driving the response.

        Parameters
        ----------
        T : float
            Physical parameter for the simulation, typically the true plasma temperature.
        **parameters : dict
            Dictionary of hyperparameter values for this grid point.

        Returns
        -------
        xspec.Model
            The XSPEC model instance, fully constructed and ready for use with `AllData` for fitting.
        """
        ...

    @abstractmethod
    def generate_model_modified(self, T: float, **parameters):
        """
        Construct the XSPEC spectral model corresponding to the modified, perturbed configuration.

        This model should incorporate the same astrophysical description as the unmodified case
        but reflects the application of calibration errors (via modified response files or models).
        The resulting synthetic data generated with this model is considered the "ground truth"
        for the purposes of recovery testing.

        Parameters
        ----------
        T : float
            Physical parameter for the simulation, typically the true plasma temperature.
        **parameters : dict
            Dictionary of hyperparameter values for this grid point, controlling the calibration modification.

        Returns
        -------
        xspec.Model
            The XSPEC model instance, fully constructed with the modified configuration for synthetic data generation.
        """
        ...

    @abstractmethod
    def fit_unmodified(self, config: dict, **parameters) -> Tuple[float, float, float]:
        """
        Fit the synthetic dataset using the unmodified configuration.

        This step evaluates how inaccurate calibration files bias parameter recovery.
        It loads the synthetic data (generated with the modified, true configuration),
        applies the unmodified setup, and extracts relevant fitting results.

        Typical return values may include:

        - Recovered temperature (float)
        - Fit statistic (e.g., chi-squared)
        - Uncertainty estimate (optional, set to 0 or NaN if not used)

        Parameters
        ----------
        config : dict
            Configuration dictionary from `generate_unmodified_configuration`.
        **parameters : dict
            Dictionary of hyperparameter values for this grid point.

        Returns
        -------
        Tuple[float, float, float]
            Results of the fit, typically (recovered_temperature, fit_statistic, uncertainty).
        """
        ...

    @abstractmethod
    def fit_modified(self, config: dict, **parameters) -> Tuple[float, float, float]:
        """
        Optionally fit the synthetic dataset using the modified configuration.

        Fitting with the correct, perturbed calibration serves as a sanity check.
        Ideally, this process should recover the true physical parameters (e.g., input temperature)
        within statistical uncertainties.

        This step may be omitted or skipped by returning placeholder values if not required.

        Parameters
        ----------
        config : dict
            Configuration dictionary from `generate_modified_configuration`.
        **parameters : dict
            Dictionary of hyperparameter values for this grid point.

        Returns
        -------
        Tuple[float, float, float]
            Results of the fit, typically (recovered_temperature, fit_statistic, uncertainty).
        """
        ...

    # ================================ #
    # Library Generation - Tooling.    #
    # ================================ #
    # These are methods which are used as tools in the generation
    # of the library but which don't generally require modification.
    def assign_mpi_load(self, mpi_size, mpi_rank) -> Tuple[int, int]:
        """
        Given a specific number of MPI ranks (mpi_size) and a specific MPI rank id
        (mpi_rank), allocate the correct set of linearlized indices for the worker.
        Each rank receives a contiguous block of indices to process:

        - The first `r = size % mpi_size` ranks receive `q + 1` items.
        - The remaining `mpi_size - r` ranks receive `q` items.

        Returns
        -------
        start : int
            Inclusive start index in the global flattened parameter grid for this rank.
        stop : int
            Exclusive stop index in the global flattened parameter grid for this rank.
        """
        # Compute base chunk size and remainder. Use the
        # computed value to get the starts and stops.
        q, r = divmod(self.size, mpi_size)
        chunk_sizes = [q + 1 if i < r else q for i in range(mpi_size)]

        starts = np.cumsum([0] + chunk_sizes[:-1])
        stops = np.cumsum(chunk_sizes)

        # Log the assignment and then return the
        # starting and stopping indices.
        self.logger.debug(
            "[LOAD ASSIGNMENT] [%s,%s] N=%s.",
            starts[mpi_rank],
            stops[mpi_rank],
            stops[mpi_rank] - starts[mpi_rank],
        )

        return int(starts[mpi_rank]), int(stops[mpi_rank])

    def _create_rank_output_file(
        self, start: int, stop: int, temperatures: Sequence[float], mpi_rank
    ) -> Path:
        """
        Create an HDF5 output file for this MPI rank to store fitting results.

        The file is saved in the output directory under the name:
        `libgen_rank_<rank>.h5`, and contains a dataset named 'results' with
        shape `(stop - start, len(temperatures), 3)`.

        Parameters
        ----------
        start : int
            Start index (inclusive) of the parameter grid assigned to this rank.
        stop : int
            Stop index (exclusive) of the parameter grid assigned to this rank.
        temperatures : Sequence[float]
            List of temperature values used in the simulation.
        mpi_rank: int
            The MPI rank for this process.

        Returns
        -------
        Path
            The path to the output file that was created.
        """
        # Determine the correct path to the temporary rank-specific
        # data. This SHOULD always be in the ``temp`` directory as defined
        # above. Typically in a subdirectory (outputs).
        rank_output_directory = self.__cachedir__ / "outputs"
        rank_output_directory.mkdir(exist_ok=True, parents=True)

        # Create the rank output file given the MPI rank.
        filename = f"libgen_rank_{mpi_rank}.h5"
        filepath = rank_output_directory / filename

        # Now build the dataset. To do so, we open the
        # HDF5 file and insert a dataset in the main directory
        # containing the data.
        with h5py.File(filepath, "w") as fio:
            # Determine how many temperature samples
            # we're taking in order to build the array.
            Ntemp = len(temperatures)
            shape = (stop - start, Ntemp, *self.__class__.__OUTPUT_SHAPE__)

            # Now create the dataset.
            fio.create_dataset("results", shape=shape, dtype="f8", compression="gzip")

            # Add some additional metadata flags.
            fio.attrs["rank"] = mpi_rank
            fio.attrs["start_index"] = start
            fio.attrs["stop_index"] = stop
            fio.attrs["temperature_count"] = Ntemp
            fio.attrs["parameter_count"] = stop - start

        self.logger.debug("[RANK OUTPUT] Built output file: %s.", filepath)
        return filepath

    def _build_synthetic_data(self, idx: int, tidx: int, T, config, **parameters):
        """
        Generate synthetic photon data for a given parameter combination and temperature.

        Uses the modified configuration and model to simulate a synthetic observation via XSPEC's
        `fakeit` functionality. Saves the generated spectrum to a `.pha` file, applies grouping,
        and returns the path to the rebinned spectrum.

        Parameters
        ----------
        idx : int
            Global index of the parameter grid point.
        tidx : int
            Index of the temperature within the current temperature list.
        T : float
            Temperature used to construct the synthetic model.
        config : dict
            Simulation configuration (e.g., response, arf, exposure).
        **parameters : dict
            Parameter values at this grid point.

        Returns
        -------
        Path
            Path to the generated, grouped synthetic `.pha` file.
        """
        # Prepare output path
        synth_dir = self.tempdir
        synth_dir.mkdir(parents=True, exist_ok=True)
        synth_path = synth_dir / f"synth.{idx}.{tidx}.pha"

        # Generate synthetic spectrum
        generate_synthetic_spectrum(
            synth_path,
            model_generator=lambda: self.generate_model_modified(T, **parameters),
            response=config["response"],
            arf=config["arf"],
            exposure=config.get("exposure", 50_000),
            background=config.get("background", ""),
            correction=config.get("correction", ""),
            backExposure=config.get("backExposure", ""),
            overwrite=True,
        )

        # Apply grouping to mimic grppha behavior
        group_min_counts(synth_path, pha_path_out=None, min_counts=3, overwrite=True)

        return synth_path

    def _write_output(
        self,
        result_mod: Union[Sequence[float], np.ndarray],
        result_unmod: Union[Sequence[float], np.ndarray],
        true_temperature: float,
        **parameters,
    ) -> np.ndarray:
        """
        Construct the result array to store for a grid point.

        This provides the default logic for saving recovery results:
        - Recovered temperature from unmodified configuration
        - Fit statistic
        - Uncertainty estimate

        Subclasses may override to include additional outputs (e.g., diagnostics, multiple fit results).

        Parameters
        ----------
        result_mod : Sequence[float] or np.ndarray
            Fit results using the modified configuration (sanity check).
        result_unmod : Sequence[float] or np.ndarray
            Fit results using the unmodified configuration (primary recovery).
        true_temperature : float
            True input temperature used to generate the synthetic spectrum.
        **parameters : dict
            Parameter values for this grid point (provided for optional subclass use).

        Returns
        -------
        np.ndarray
            Array of shape `__OUTPUT_SHAPE__` containing results to store.

        Notes
        -----
        - The returned array must exactly match `__OUTPUT_SHAPE__`.
        - Default implementation assumes shape (3,), storing:
            [recovered temperature, fit statistic, uncertainty]
        - Subclasses should explicitly document and control their storage layout if overriding.
        """
        output = np.full(self.__OUTPUT_SHAPE__, np.nan, dtype=float)

        if self.__OUTPUT_SHAPE__ == (3,):
            try:
                output[0] = float(result_unmod[0])  # Recovered temperature
                output[1] = float(result_unmod[1])  # Fit statistic
                output[2] = float(result_unmod[2])  # Uncertainty
            except (IndexError, TypeError) as e:
                raise ValueError(f"Invalid result_unmod format: {result_unmod}") from e
        else:
            raise NotImplementedError(
                f"Subclasses with custom __OUTPUT_SHAPE__={self.__OUTPUT_SHAPE__} "
                "must override `write_output` to define storage behavior."
            )

        return output

    def _finalize_library(self, temperatures: Sequence[float], mpisize: int):
        """
        Merge per-rank results into final library and clean up temporary files.

        Parameters
        ----------
        temperatures : Sequence[float]
            Array of simulated temperatures.
        mpisize : int
            Total number of MPI processes.
        """
        import dask.array as da
        from dask.array.core import slices_from_chunks
        from dask.base import compute

        # Perform some basic setup operations. Log that this
        # is occuring and get the list of dask arrays and
        # datasets ready.
        self.logger.info("Finalizing library using Dask...")
        Ntemp = len(temperatures)
        output_dir = self.__cachedir__ / "outputs"

        datasets = []
        dask_arrays = []

        # Begin the merger process. We seek each rank of
        # the operation and its datafile. For each, we merge
        # the resulting dask array and then drop the data
        # into the full output.
        try:
            # Load rank-specific outputs as Dask arrays
            for rank in range(mpisize):
                path = output_dir / f"libgen_rank_{rank}.h5"
                f = h5py.File(path, "r")
                datasets.append(f)
                dask_arrays.append(da.from_array(f["results"], chunks="auto"))

            # Combine and reshape
            combined = da.concatenate(dask_arrays, axis=0)
            expected_shape = (*self.shape, Ntemp, *self.__class__.__OUTPUT_SHAPE__)
            reshaped = combined.reshape(expected_shape)

            # Write to final library file
            with h5py.File(self.__datapath__, "a") as f:
                if "LIBRARY" in f:
                    del f["LIBRARY"]
                lib_group = f.create_group("LIBRARY")
                lib_group.create_dataset(
                    "temps", data=np.array(temperatures), dtype="f8"
                )

                results = lib_group.create_dataset(
                    "results", shape=expected_shape, dtype="f8", compression="gzip"
                )

                # Efficient block-wise write to limit memory spikes
                for block, s in zip(
                    reshaped.to_delayed().flatten(), slices_from_chunks(reshaped.chunks)
                ):
                    results[s] = compute(block)[0]

            self.logger.info("Library finalized at %s", self.__datapath__)

        finally:
            # Close HDF5 handles
            for f in datasets:
                f.close()

            # Remove temporary rank files
            for rank in range(mpisize):
                path = output_dir / f"libgen_rank_{rank}.h5"
                if path.exists():
                    path.unlink()
                    self.logger.debug("Deleted %s", path)

        self.logger.info("Library finalization complete.")

    def cleanup_temperature_iteration(
        self, global_parameter_index: int, temperature_index: int
    ):
        """
        Optional hook to clean up after each temperature iteration.

        Default behavior: removes synthetic PHA files to limit disk usage.

        Subclasses may extend this for additional cleanup tasks.

        Parameters
        ----------
        global_parameter_index : int
            Flattened index into the parameter grid.
        temperature_index : int
            Index of the current temperature in the temperatures array.
        """
        synth_path = (
            self.tempdir / f"synth.{global_parameter_index}.{temperature_index}.pha"
        )
        if synth_path.exists():
            synth_path.unlink()
            self.logger.debug("Deleted synthetic spectrum: %s", synth_path)

    def cleanup_parameter_iteration(self, global_parameter_index: int):
        """
        Optional hook to clean up after completing all temperatures for a parameter point.

        Default: no-op.

        Subclasses may override to delete temporary files, cached models, etc.

        Parameters
        ----------
        global_parameter_index : int
            Flattened index into the parameter grid.
        """
        pass  # Subclasses can override if needed

    # ------------------------------- #
    # Simulation                      #
    # ------------------------------- #
    def generate_library(self, temperatures, clear_cache: bool = True):
        """
        Run the distributed, MPI-parallelized generation and fitting process for this library.

        Each MPI rank processes a subset of the parameter grid, performing:
        - Synthetic data generation with the modified configuration.
        - Optional fitting with the modified configuration (sanity check).
        - Fitting with the unmodified configuration (bias quantification).
        - Storing results in rank-specific HDF5 files.

        Rank 0 consolidates outputs into the final library and optionally clears the working cache.

        Parameters
        ----------
        temperatures : Sequence[float]
            Array of true temperatures to simulate at each grid point.
        clear_cache : bool, default = True
            Whether to delete temporary working files (e.g., synthetic spectra, per-rank results) after library finalization.

        Notes
        -----
        - The process assumes a valid parameter grid and configuration system are defined.
        - Results include recovery errors based on the unmodified instrument configuration, mimicking real-world inference biases.
        - The library must be finalized on rank 0 after all ranks complete.
        """
        # @@ Configure the run @@ #
        # On this processor, we're going to configure the run by
        # determining the relevant start and stop indices and
        # creating the data file. This requires realizing the MPI configuration.
        _mpicomm = get_mpi()
        _mpirank = _mpicomm.Get_rank()
        _mpisize = _mpicomm.Get_size()

        _prog_monitor: ProgressMonitor = None  # type: ignore
        if _mpirank == 0:
            # Provide some information to the log about
            # the overall run.
            print("\n\n====================================================")
            print("Running `generate_library` on %s." % self.__directory__.name)
            print("")
            print("RUN PARAMETERS:")
            print("---------------")
            print("  MPI PROCS = %s" % _mpisize)
            print("  TEMPS = %s" % len(temperatures))
            print("  NSIMS = %s" % self.size)
            print("====================================================")

            # Setup the monitor.
            _prog_monitor = ProgressMonitor(
                _mpisize,
                library_name=self.__directory__.name,
            )

        # Configure the start and stop point for
        # this process and construct the datafile into
        # which this process stores its data.
        pstart_idx, pstop_idx = self.assign_mpi_load(_mpisize, _mpirank)
        data_file = self._create_rank_output_file(
            pstart_idx, pstop_idx, temperatures, _mpirank
        )
        self.clear_tempdir()

        # Compute some private variables for use in
        # tracking, logging, etc. going forward.
        _num_simulations_rank = (pstop_idx - pstart_idx) * len(temperatures)
        _num_params = pstop_idx - pstart_idx
        _rank_info = (pstart_idx, pstop_idx)

        # Gather from all MPI ranks and print a summary to
        # the stdout.
        _all_rank_info = _mpicomm.gather(_rank_info, root=0)
        if _mpirank == 0:
            print("\nParameter Space Assignment Summary:")
            print("===============================================")
            print(f"{'Rank':>4} | {'Param IDs':<19} | {'Simulations':>10}")
            print("-----------------------------------------------")
            for r, (start, stop) in enumerate(_all_rank_info):  # type: ignore
                num = (stop - start) * len(temperatures)
                print(f"{r:>4} | [{start:>7} , {stop:<7}) | {num:>10}")
            print("===============================================\n")

        # --------------------------- #
        # Begin Simulation Run.       #
        # --------------------------- #
        # This section of the code performs the central iterations
        # through the parameter space and over the relevant temperatures
        # to fill in the data.
        try:
            with h5py.File(data_file, "a") as df:
                # Iterate through each of the local and global
                # parameter indices so that the simlation may be
                # performed at each iteration.
                _simulations_completed = 0
                _progress = 0
                _is_complete = False
                for lidx, gidx in enumerate(range(pstart_idx, pstop_idx)):
                    # log the loop start information.
                    _parameter_start_time = time.perf_counter()
                    self.logger.info(
                        "[Param %5d] (%4d / %4d) | Progress: %6.2f%%",
                        gidx,
                        lidx + 1,
                        _num_params,
                        _progress,
                    )

                    # Construct the N-dim index from the gidx so that we can access
                    # the parameter dictionary given our position in the workload.
                    midx = np.unravel_index(gidx, self.shape)
                    iteration_parameters = {
                        paramkey: paramvalue[midx[param_idx]]
                        for param_idx, (paramkey, paramvalue) in enumerate(
                            self.__parameters__.items()
                        )
                    }

                    # Generate the configurations for this parameter set. These
                    # are going to be used during the fitting procedure.
                    mod_config = self.generate_modified_configuration(
                        gidx, **iteration_parameters
                    )
                    unmod_config = self.generate_unmodified_configuration(
                        gidx, **iteration_parameters
                    )
                    self.logger.debug(
                        "[Param %5d] (%4d / %4d) |   Generated mod and unmod configs.",
                        gidx,
                        lidx + 1,
                        _num_params,
                    )

                    # @@ Enter Temperature Loop @@ #
                    # Now for each of the library parameter values, we iterate over all
                    # of the temperatures in order to sample the parameter space.
                    for tid, T in enumerate(temperatures):
                        # Log some loop start information.
                        _temperature_start_time = time.perf_counter()
                        self.logger.debug(
                            "[Param %5d: Temp %5d] (%4d / %4d) | Progress: %6.2f%%",
                            gidx,
                            tid,
                            _simulations_completed,
                            _num_simulations_rank,
                            _progress,
                        )

                        # Build the synthetic data for this
                        # run by generating the modified model and then
                        # generating synthetic data.
                        _ = self.generate_model_modified(T, **iteration_parameters)
                        synth_path = self._build_synthetic_data(
                            gidx, tid, T, mod_config, **iteration_parameters
                        )
                        xspec.AllData.clear()
                        xspec.AllModels.clear()

                        self.logger.debug(
                            "[Param %5d: Temp %5d] (%4d / %4d) |   Built synthetic data.",
                            gidx,
                            tid,
                            _simulations_completed,
                            _num_simulations_rank,
                        )

                        # Perform the fit to the modified case and then
                        # to the unmodified case.
                        _ = self.generate_model_unmodified(T, **iteration_parameters)
                        _ = xspec.Spectrum(str(synth_path))
                        result_mod = self.fit_modified(mod_config)
                        self.logger.debug(
                            "[Param %5d: Temp %5d] (%4d / %4d) |   Fit to modified config.",
                            gidx,
                            tid,
                            _simulations_completed,
                            _num_simulations_rank,
                        )

                        xspec.AllData.clear()
                        xspec.AllModels.clear()
                        _ = self.generate_model_unmodified(T, **iteration_parameters)
                        _ = xspec.Spectrum(str(synth_path))
                        result_unmod = self.fit_unmodified(unmod_config)
                        self.logger.debug(
                            "[Param %5d: Temp %5d] (%4d / %4d) |   Fit to unmodified config.",
                            gidx,
                            tid,
                            _simulations_completed,
                            _num_simulations_rank,
                        )

                        # Construct output and write it to the rank
                        # level output.
                        output = self._write_output(
                            result_mod=result_mod,
                            result_unmod=result_unmod,
                            true_temperature=T,
                            **iteration_parameters,
                        )
                        df["results"][lidx, tid, :] = output  # type: ignore[index]
                        self.logger.debug(
                            "[Param %5d: Temp %5d] (%4d / %4d) |   Wrote output.",
                            gidx,
                            tid,
                            _simulations_completed,
                            _num_simulations_rank,
                        )

                        # Clean up the teperature iteration. This requires
                        # that we compute a few things before moving on to the
                        # next one.
                        self.cleanup_temperature_iteration(gidx, tid)
                        self.logger.debug(
                            "[Param %5d: Temp %5d] (%4d / %4d) |   Cleaned up temperature iteration.",
                            gidx,
                            tid,
                            _simulations_completed,
                            _num_simulations_rank,
                        )

                        _temperature_elapsed_time = (
                            time.perf_counter() - _temperature_start_time
                        )
                        _simulations_completed += 1
                        _progress = 100 * (
                            _simulations_completed / _num_simulations_rank
                        )
                        self.logger.debug(
                            "[Param %5d: Temp %5d] (%4d / %4d) | [DONE]: %6.2fs",
                            gidx,
                            tid,
                            _simulations_completed,
                            _num_simulations_rank,
                            _temperature_elapsed_time,
                        )

                        clear_xspec()

                        # ----------- END OF TEMP ITERATION ---------------- #

                    # Clean up the parameter level loop before proceeding.
                    self.cleanup_parameter_iteration(gidx)
                    self.logger.debug(
                        "[Param %5d] (%4d / %4d) |   Cleaned up parameter iteration.",
                        gidx,
                        lidx + 1,
                        _num_params,
                    )

                    _parameter_elapsed_time = (
                        time.perf_counter() - _parameter_start_time
                    )
                    self.logger.info(
                        "[Param %5d] (%4d / %4d) | [DONE]: %6.2fs",
                        gidx,
                        lidx + 1,
                        _num_params,
                        _parameter_elapsed_time,
                    )

                    _is_complete = (lidx + 1) == _num_params
                    status_local = (lidx + 1, _num_params, _is_complete)

                    # Sync every NSYNC or at completion
                    if ((lidx + 1) % self.config["prog.nsync"] == 0) or _is_complete:
                        _ = sync_progress(
                            _mpicomm, _mpirank, _prog_monitor, *status_local
                        )

            # We now exit except for the RANK-0 process, which will complete by
            # combining the relevant data files and cleaning up the environment.
            _mpicomm.Barrier()
            if _mpirank == 0:
                try:
                    self._finalize_library(temperatures, _mpisize)
                finally:
                    if clear_cache:
                        import shutil

                        shutil.rmtree(self.__assets__["cache"])
                        self.__assets__["cache"].mkdir(parents=True, exist_ok=True)

        except Exception:
            # SOMETHING FAILED. We proceed by logging to the core logger, the file logger,
            # and printing the error message before killing everthing.
            self.logger.critical(
                "Fatal error on rank %d:\n%s", _mpirank, traceback.format_exc()
            )
            spec_logger.critical(
                "Fatal error on rank %d:\n%s", _mpirank, traceback.format_exc()
            )

            # Abort all ranks immediately
            _mpicomm.Abort(1)
