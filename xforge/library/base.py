"""
Library generation base classes.

These are the base classes for generating the modification libraries
which are then used to train the interpolators.
"""
import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

from xforge.utils import ConfigManager, get_mpi, get_xspec

# Ensure access to XSPEC via PyXSPEC. This needs
# to be done carefully because there can only be one CLI active
# per process.
xspec = get_xspec()


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
        "logging.file_level": "INFO",
        "logging.term_level": "INFO",
        "logging.fmt": "%(asctime)s - %(levelname)s - %(message)s",
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

        self.logger.info("Loading Library @ %s.", self.__directory__)

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
        from xforge.utils import (  # Local import to avoid cyclic dependency
            ConfigManager,
        )

        self.__config__ = ConfigManager(self.__configpath__)

    def __init_logger__(self):
        """
        Initializes the internal logger for the library.
        Sets up both per-rank log files and (optionally) console output for rank 0.
        """
        # Obtain the name of the logger and load in
        # MPI to check what configuration system we're going
        # to be using.
        comm = get_mpi()
        rank = comm.Get_rank()

        logger_name = f"Library.{self.__directory__.name}"

        # Configure for parallelism
        self.logger = logging.getLogger(logger_name + f".{rank}")
        self.logger.setLevel(self.config.get("logging.file_level", logging.DEBUG))

        # Avoid adding multiple handlers if already initialized
        if self.logger.hasHandlers():
            return

        # construct the formatter.
        fmt = self.config.get(
            "logging.fmt", "%(asctime)s - %(levelname)s - %(message)s"
        )
        formatter = logging.Formatter(fmt)

        # Create the output log file.
        logfile = self.__directory__ / f"logs/rank_{rank}.log"
        filehandler = logging.FileHandler(logfile, mode="w")
        filehandler.setLevel(self.config.get("logging.level", logging.DEBUG))
        filehandler.setFormatter(formatter)
        self.logger.addHandler(filehandler)

        if rank == 0:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(
                self.config.get("logging.term_level", logging.DEBUG)
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

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

            param_group: h5py.Group = f["PARAMS"]

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
                self.__parameters__[key] = param_group[key][...]

        # Finally, ensure that all of the necessary parameters
        # are specified. If they are not, we raise an error.
        if any(i not in self.__parameters__ for i in self.__PARAMETERS__):
            raise ValueError("Missing parameters!")

    @abstractmethod
    def __post_init__(self):
        pass

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

        directory = Path(directory).expanduser().resolve()

        # Handle overwrite logic
        if directory.exists():
            if not overwrite:
                raise ValueError(
                    f"Directory `{directory}` already exists. Use `overwrite=True` to recreate it."
                )
            gc.collect()
            shutil.rmtree(directory)

        # Create required subdirectories and files
        cls.__create_structures__(directory)
        cls.__write_datafile__(directory, parameters)
        cls.__write_config__(directory, config=config)

        # Instantiate and return the initialized library
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
        from xforge.utils import ConfigManager

        config_path = Path(directory) / "config.yaml"
        cfg = ConfigManager(config_path, autosave=True)

        # Merge defaults and user-specified config; user config takes priority
        merged_config = dict(cls.__CONFIG__)
        if config is not None:
            merged_config.update(config)

        for key, value in merged_config.items():
            cfg[key] = value

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
            return f["LIBRARY"]["temps"][...]

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
            yield lib_group["results"], lib_group["temps"]

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

    # -- Tooling -- #
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
        # Compute base chunk size and remainder
        q, r = divmod(self.size, mpi_size)

        # Determine the number of items for each rank
        chunk_sizes = [q + 1 if i < r else q for i in range(mpi_size)]

        # Compute prefix sums to get start/stop indices
        starts = np.cumsum([0] + chunk_sizes[:-1])
        stops = np.cumsum(chunk_sizes)

        self.logger.debug(
            "Assigned process %s to %s parameters from %s to %s.",
            mpi_rank,
            stops[mpi_rank] - starts[mpi_rank],
            starts[mpi_rank],
            stops[mpi_rank],
        )

        return int(starts[mpi_rank]), int(stops[mpi_rank])

    def create_rank_output_file(
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
        self.logger.info("Building output file: %s.", filepath)
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

        return filepath

    def build_synthetic_data(self, idx: int, tidx: int, T, config, **parameters):
        """
        Generate synthetic photon data for a given parameter combination and temperature.

        This method uses the modified configuration and model to simulate a synthetic
        observation via XSPEC's `fakeit` functionality. It saves the generated spectrum
        to a `.pha` file and returns the loaded data.

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
        xspec.Spectrum
            The synthetic XSPEC spectrum loaded into memory.

        Notes
        -----
        This function clears all existing XSPEC models and data, both before
        and after generating the synthetic dataset.
        """

        # Clear all existing XSPEC states
        xspec.AllModels.clear()
        xspec.AllData.clear()

        # Construct the model and set it as the
        _ = self.generate_model_modified(T, **parameters)

        # Generate synthetic spectrum using mod_config and mod_model
        synth_dir = self.__cachedir__ / "synth"
        synth_dir.mkdir(parents=True, exist_ok=True)
        synth_path = synth_dir / f"synth_{idx}_{tidx}.pha"
        xspec.AllData.clear()
        fakeit = xspec.FakeitSettings(
            response=config["response"],
            arf=config["arf"],
            exposure=config.get("exposure", 50_000),
            background=config.get("background", ""),
            correction=config.get("correction", ""),
            backExposure=config.get("backExposure", ""),
            fileName=str(synth_path),
        )

        # Clear existing spectra to avoid collisions
        xspec.AllData.fakeit(1, [fakeit])

        logging.info(
            "Created synthetic data (%s,%s) with parameters %s.",
            idx,
            tidx,
            str(parameters),
        )

        # Return the generated synthetic datasets
        return xspec.AllData(1)

    def _finalize_library(self, temperatures: Sequence[float], mpisize: int):
        """
        Finalize the generation of the synthetic library by merging results
        from all MPI ranks using Dask and writing into the HDF5 file.

        Parameters
        ----------
        temperatures : Sequence[float]
            The array of simulated temperatures.
        mpisize : int
            The total number of MPI processes used in the generation.
        """
        import dask.array as da
        from dask import compute
        from dask.array.core import slices_from_chunks

        self.logger.info("Finalizing library with Dask...")
        Ntemp = len(temperatures)
        output_dir = self.__cachedir__ / "outputs"

        datasets = []
        darrs = []

        try:
            # Load all per-rank result datasets as Dask arrays
            for rank in range(mpisize):
                file_path = output_dir / f"libgen_rank_{rank}.h5"
                f = h5py.File(file_path, "r")
                datasets.append(f)

                dset = f["results"]
                darrs.append(da.from_array(dset, chunks="auto"))

            # Concatenate and reshape
            flat_dask_array = da.concatenate(darrs, axis=0)
            full_shape = (*self.shape, Ntemp, *self.__class__.__OUTPUT_SHAPE__)
            reshaped = flat_dask_array.reshape(full_shape)

            # Write to final HDF5 file
            with h5py.File(self.__datapath__, "a") as f:
                if "LIBRARY" in f:
                    del f["LIBRARY"]
                lib_group = f.create_group("LIBRARY")
                lib_group.create_dataset(
                    "temps", data=np.array(temperatures), dtype="f8"
                )

                result_ds = lib_group.create_dataset(
                    "results", shape=reshaped.shape, dtype="f8", compression="gzip"
                )

                # Write each chunk block-by-block to avoid memory overuse
                for block, slices in zip(
                    reshaped.to_delayed().flatten(), slices_from_chunks(reshaped.chunks)
                ):
                    result_ds[slices] = compute(block)[0]

            self.logger.info("Library written to: %s", self.__datapath__)

        finally:
            # Close all HDF5 datasets
            for f in datasets:
                f.close()

            # Clean up per-rank output files
            for rank in range(mpisize):
                file_path = output_dir / f"libgen_rank_{rank}.h5"
                if file_path.exists():
                    file_path.unlink()
                    self.logger.debug("Deleted %s", file_path)

        self.logger.info("Library finalization complete.")

    def write_output(
        self,
        result_mod: Union[Tuple, np.ndarray],
        result_unmod: Union[Tuple, np.ndarray],
        true_temperature: float,
        **parameters,
    ) -> np.ndarray:
        """
        Construct the array to store in the results dataset for this grid point.

        Parameters
        ----------
        result_mod : tuple or array-like, optional
            Fit results using the modified configuration (e.g., sanity check).
        result_unmod : tuple or array-like, optional
            Fit results using the unmodified configuration (primary target).
        true_temperature : float
            The true input temperature used in the synthetic simulation.
        **parameters : dict
            Dictionary of parameter values at this grid point.

        Returns
        -------
        np.ndarray
            Array matching `__OUTPUT_SHAPE__` to store in the results dataset.

        Notes
        -----
        The default implementation stores the recovered unmodified temperature,
        fit statistic, and uncertainty. Subclasses may override to customize outputs.
        """
        output = np.zeros(self.__OUTPUT_SHAPE__, dtype=float)
        output[0] = result_unmod[0]  # Recovered temperature
        output[1] = result_unmod[1]  # Fit statistic
        output[2] = result_unmod[2]  # Uncertainty

        return output

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

        # Determine range of parameter indices for this rank
        pstart_idx, pstop_idx = self.assign_mpi_load(_mpisize, _mpirank)

        # Open output file and get write handle
        data_file = self.create_rank_output_file(
            pstart_idx, pstop_idx, temperatures, _mpirank
        )

        with h5py.File(data_file, "a") as df:
            # @@ Iterate through run @@ #
            # We now iterate through the run of indices and
            # perform the analysis at each index.
            __NOP_TOTAL__ = (pstop_idx - pstart_idx) * len(temperatures)
            for lidx, gidx in enumerate(range(pstart_idx, pstop_idx)):
                self.logger.debug("Start iteration (%s,%s).", lidx, gidx)
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

                # @@ Enter Temperature Loop @@ #
                # Now for each of the library parameter values, we iterate over all
                # of the temperatures in order to sample the parameter space.
                for tid, T in enumerate(temperatures):
                    self.logger.debug("Start temperature iteration %s.", tid)
                    # Build the modified model so that we can use it
                    # to create the synthetic data.
                    _ = self.generate_model_modified(T, **iteration_parameters)

                    # Build the synthetic data.
                    self.build_synthetic_data(
                        gidx, tid, T, mod_config, **iteration_parameters
                    )

                    # Fit using unmodified model
                    result_mod = self.fit_modified(mod_config)

                    # Clear the model and rebuild with the
                    # unmodified
                    xspec.AllModels.clear()
                    _ = self.generate_model_unmodified(T, **iteration_parameters)

                    # Build the mod result
                    result_unmod = self.fit_unmodified(unmod_config)

                    # Construct output
                    output = self.write_output(
                        result_mod=result_mod,
                        result_unmod=result_unmod,
                        true_temperature=T,
                        **iteration_parameters,
                    )

                    # Optionally: fit using modified model (if relevant)
                    # mod_result = self.fit_modified()

                    # Store result
                    df["results"][lidx, tid, :] = output

                    # Determine % complete on this process.
                    __PERCNT_DONE__ = (
                        100 * (lidx * len(temperatures) + tid) / __NOP_TOTAL__
                    )

                    self.logger.info(
                        "[Rank %d] Iter %d/%d | ParamIdx: %d | TempIdx: %d | Progress: %.2f%%",
                        _mpirank,
                        lidx + 1,
                        pstop_idx - pstart_idx,
                        gidx,
                        tid,
                        __PERCNT_DONE__,
                    )

                    self.logger.debug("Start temperature iteration %s. [DONE]", tid)

                self.logger.debug("Start iteration (%s,%s). [DONE]", lidx, gidx)

        # We now exit except for the RANK-0 process, which will complete by
        # combining the relevant data files and cleaning up the environment.
        if _mpirank == 0:
            try:
                self._finalize_library(temperatures, _mpisize)
            finally:
                if clear_cache:
                    import shutil

                    shutil.rmtree(self.__assets__["cache"])
                    self.__assets__["cache"].mkdir(parents=True, exist_ok=True)

    # ------------------------------------ #
    # Utility Methods                      #
    # ------------------------------------ #
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

        # Convert string levels to integer values
        if isinstance(level, str):
            level_upper = level.upper()
            level = getattr(logging, level_upper, None)
            if not isinstance(level, int):
                raise ValueError(f"Invalid logging level: {level_upper}")

        # Update the config for persistence
        self.config["logging.level"] = logging.getLevelName(level)

        # Update logger level
        self.logger.setLevel(level)

        # Select handlers
        target_handlers = handlers if handlers is not None else self.logger.handlers

        # Update handler levels
        for h in target_handlers:
            h.setLevel(level)

        self.logger.debug("Logger level set to %s.", logging.getLevelName(level))
