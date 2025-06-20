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
    Base class for loading and interacting with a pre-generated synthetic fitting library.

    This class provides common logic for loading configuration, parameter space, and output
    data from a specified directory. It assumes the library has already been generated and
    contains a configuration file, parameter definitions, and combined results.
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
    The ALLOWED set of parameter names (each element) for this
    modification library class. If parameters are detected which do not
    exist in __PARAMETERS__, then an error is raised.
    """
    __CONFIG__: Dict[str, Any] = {"logging.level": "INFO"}

    # ========================== #
    # Initialization             #
    # ========================== #
    # These methods are all concerned with loading libraries from
    # disk. They can be modified in subclasses, but should be kept
    # relatively consistent.
    def __init__(self, directory: Union[str, Path]):
        """
        Initialize a `ModificationLibrary` from an existing library directory.

        Parameters
        ----------
        directory : str or Path
            Path to the root library directory containing the required structure.

        Raises
        ------
        ValueError
            If the directory does not exist.
        FileNotFoundError
            If critical components (like config.yaml or library.h5) are missing.
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
        self.logger.setLevel(self.config.get("logging.level", logging.DEBUG))

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
            console_handler.setLevel(logging.INFO)
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
        Create a new library directory with required structure and parameter file.

        Parameters
        ----------
        directory : str or Path
            Path to the root of the new library. If the directory already exists,
            the behavior is dictated by the `overwrite` parameter.
        parameters : dict[str, Sequence]
            Parameter grid as a dictionary of 1D arrays/lists for each axis. These must
            match the class's allowed parameters.
        overwrite : bool
            Whether to overwrite an existing directory. If False and directory exists, an error is raised.
        config : dict, optional
            Optional configuration dictionary to write to `config.yaml`. If None, a default is used.

        Returns
        -------
        ModificationLibrary
            An instance of the initialized library.
        """
        directory = Path(directory).expanduser().resolve()

        # Handle overwrite logic
        if directory.exists():
            if not overwrite:
                raise ValueError(
                    f"Directory `{directory}` already exists. Use `overwrite=True` to recreate."
                )
            import shutil

            shutil.rmtree(directory)

        # Create necessary folders
        cls.__create_structures__(directory)

        # Write the data file with parameter arrays
        cls.__write_datafile__(directory, parameters)

        # Write the configuration YAML
        cls.__write_config__(directory, config=config)

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
                param_group.create_dataset(key, data=np.array(values))

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
    def config(self) -> "ConfigManager":
        """
        Accessor for the YAML-backed configuration.

        Returns
        -------
        ConfigManager
            Configuration manager that supports nested dictionary-style access.
        """
        return self.__config__

    @property
    def parameters(self) -> Dict[str, np.ndarray]:
        """
        Accessor for the parameter arrays loaded from disk.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary of parameter arrays (1D), as stored in the 'PARAMS' HDF5 group.
        """
        return self.__parameters__

    @property
    def size(self) -> int:
        """
        Total number of parameter combinations in the grid.

        Returns
        -------
        int
            The product of the lengths of all parameter value lists, representing the
            total number of points in the parameter lattice to be explored.
        """
        return int(np.prod([len(pv) for pk, pv in self.__parameters__.items()]))

    @property
    def shape(self) -> tuple:
        return tuple(len(pv) for _, pv in self.__parameters__.items())

    @property
    def is_generated(self) -> bool:
        """
        Returns True if the library has been generated, i.e.,
        if the 'LIBRARY' group exists in the HDF5 file.

        Returns
        -------
        bool
            True if the synthetic library has been fully generated.
        """
        with h5py.File(self.__assets__["data"], "r") as f:
            return "LIBRARY" in f

    @property
    def temperatures(self) -> Optional[np.ndarray]:
        """
        Return the temperature grid used in the synthetic library.

        Returns
        -------
        np.ndarray or None
            Array of temperature values used in the simulation, or None if not generated.
        """
        if not self.is_generated:
            return None

        with h5py.File(self.__assets__["data"], "r") as f:
            return f["LIBRARY"]["temps"][...]

    @contextmanager
    def library(self, mode="r"):
        """
        Context manager for lazy access to the generated library results.

        Yields
        ------
        Tuple[h5py.Dataset, h5py.Dataset]
            A tuple of (results, temperatures) datasets from the 'LIBRARY' group.

        Raises
        ------
        RuntimeError
            If the library has not been generated.
        """
        if not self.is_generated:
            raise RuntimeError(
                "This library has not been generated. No results available."
            )

        with h5py.File(self.__assets__["data"], mode) as f:
            lib_group = f["LIBRARY"]
            yield lib_group["results"], lib_group["temps"]

    # ================================ #
    # Library Generation               #
    # ================================ #
    @abstractmethod
    def generate_unmodified_configuration(self, id: int, **parameters) -> dict:
        """
        Generate the configuration dictionary for the unmodified system.

        This configuration should define file paths and simulation parameters required
        for generating or loading the standard model, including ARF, RMF, exposure, etc.

        Parameters
        ----------
        id : int
            Global index in the parameter space (flattened).
        **parameters : dict
            Dictionary of parameter values at this grid point.

        Returns
        -------
        dict
            Configuration dictionary containing file references and simulation settings.
        """
        ...

    @abstractmethod
    def generate_modified_configuration(self, id: int, **parameters) -> dict:
        """
        Generate the configuration dictionary for the modified system.

        This configuration typically corresponds to a perturbed version of the unmodified
        configuration—e.g., with a distorted ARF or a changed response matrix.

        Parameters
        ----------
        id : int
            Global index in the parameter space (flattened).
        **parameters : dict
            Dictionary of parameter values at this grid point.

        Returns
        -------
        dict
            Configuration dictionary for the modified system.
        """
        ...

    @abstractmethod
    def generate_model_unmodified(self, T, **parameters):
        """
        Build and return the XSPEC model for the unmodified configuration.

        This should construct the physical model (e.g., `tbabs*apec`) corresponding to the
        unmodified simulation configuration for a given temperature.

        Parameters
        ----------
        T : float
            Temperature (or other synthetic variable) used to build the model.
        **parameters : dict
            Dictionary of parameter values at this grid point.

        Returns
        -------
        xspec.Model
            The unmodified XSPEC model instance.
        """
        ...

    @abstractmethod
    def generate_model_modified(self, T, **parameters):
        """
        Build and return the XSPEC model for the modified configuration.

        This typically reflects a perturbed version of the base model—e.g., using a
        modified response or artificial distortion in the instrument model.

        Parameters
        ----------
        T : float
            Temperature (or other synthetic variable) used to build the model.
        **parameters : dict
            Dictionary of parameter values at this grid point.

        Returns
        -------
        xspec.Model
            The modified XSPEC model instance.
        """
        ...

    @abstractmethod
    def fit_unmodified(self, config, **parameters) -> float:
        """
        Perform fitting on the synthetic data using the unmodified configuration.

        This function should load the synthetic dataset (created by the modified model),
        apply the unmodified model, and extract the best-fit parameters or relevant statistics.

        Parameters
        ----------
        config : dict
            Unmodified configuration dictionary.
        **parameters : dict
            Dictionary of parameter values at this grid point.

        Returns
        -------
        float or array-like
            Result of the fit (e.g., recovered temperature or fit statistic).
        """
        ...

    @abstractmethod
    def fit_modified(self, config, **parameters) -> float:
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
            shape = (stop - start, Ntemp, 3)

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
            full_shape = (*self.shape, Ntemp, 3)
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

    def generate_library(self, temperatures, clear_cache: bool = True):
        """
        Runs the generation and fitting loop for this MPI rank. For each assigned
        parameter set and temperature, it creates synthetic data, performs both unmodified
        and modified fits, and writes the result to a rank-specific HDF5 file.

        Parameters
        ----------
        temperatures : Sequence[float]
            List of temperature values to loop over in synthetic data generation.
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
                    mod_result = self.fit_modified(mod_config)

                    # Clear the model and rebuild with the
                    # unmodified
                    xspec.AllModels.clear()
                    _ = self.generate_model_unmodified(T, **iteration_parameters)

                    # Build the mod result
                    unmod_result = self.fit_unmodified(unmod_config)

                    # Optionally: fit using modified model (if relevant)
                    # mod_result = self.fit_modified()

                    # Store result
                    df["results"][
                        lidx, tid, 0
                    ] = unmod_result  # or mod_result, depending on target

                    self.logger.info(
                        "Iteration (%s,%s,%s) -- T_mod=%s, T_unmod=%s, T_True=%s",
                        lidx,
                        gidx,
                        tid,
                        mod_result,
                        unmod_result,
                        T,
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
