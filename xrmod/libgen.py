"""
The ``libgen.py`` module is responsible for generating the training
library of modifications. In order to do so, the LibProtocol class is used to
define and configure the behavior of the library during its generation.
"""
from typing import Union, Dict, Sequence, Tuple
from pathlib import Path
import xspec
import os
import numpy as np
from mpi4py import MPI
import h5py
from abc import ABC, abstractmethod

class LibProtocol(ABC):
    __lib_params__ = []
    """ 
    List of valid parameter names expected in the parameter grid. 

    Subclasses must override this with a list of strings corresponding to the 
    parameter names used for generating the modification space.

    Each string in the list will be used as a key in the ``parameters`` dictionary 
    during initialization. These define the dimensionality and bounds of the 
    modification lattice.

    Example:
        __lib_params__ = ["mu", "sigma", "A"]
    """

    # --------------------------------- #
    # Initialization                    #
    # --------------------------------- #
    # This part of the code should usually never change!
    def __init__(self, lib_dir: Union[str,Path], parameters: Dict[str,Sequence], **kwargs):
        """
        Initialize a LibProtocol instance for generating a synthetic fitting library.

        This sets up the working directory structure, validates the parameter grid 
        against the protocol's defined parameter list, and prepares MPI coordination.
        Subclasses may inject additional initialization behavior via `__post_init__`.

        Parameters
        ----------
        lib_dir : str or Path
            Path to the root directory where the library will be generated. This directory 
            will contain subdirectories for output, logs, synthetic data, configuration files, 
            binaries, and stable references.
        
        parameters : dict[str, Sequence]
            Dictionary defining the parameter grid. Each key must match an entry in 
            `__lib_params__`, and each value must be a sequence of discrete values to 
            span for that parameter. Together, these define the full modification lattice 
            for library generation.
        
        **kwargs : dict
            Additional keyword arguments forwarded to subclass-specific post-initialization 
            logic via `__post_init__`.

        Raises
        ------
        NotImplementedError
            If `__lib_params__` is not defined in the subclass.
        ValueError
            If an entry in the `parameters` dictionary does not match an expected key in 
            `__lib_params__`.

        Notes
        -----
        - Only rank 0 is responsible for creating the directory structure; other ranks 
          will wait on MPI synchronization.
        - Parameter values are internally coerced to lists and stored as `self.__params__`.
        - The MPI communicator is always `MPI.COMM_WORLD`.
        """
        # @@ Parse Library Directory @@ #
        # This step in the __init__ procedure ensures that the library directory
        # is found, exists / is created, and is stored internally as a path object
        # so that we can work with it later.
        self.__libdir__ = Path(lib_dir).expanduser().resolve()
        self.__libdir__.mkdir(parents=True, exist_ok=True)

        # Construct the additional directories.
        self.__outdir__ = Path(os.path.join(self.__libdir__,"out"))
        self.__logdir__ = Path(os.path.join(self.__libdir__,"log"))
        self.__bindir__ = Path(os.path.join(self.__libdir__,"bin"))
        self.__stable_dir__ = Path(os.path.join(self.__libdir__,"stable"))
        self.__synth_dir__ = Path(os.path.join(self.__libdir__,"synthetic"))
        self.__configs__ = Path(os.path.join(self.__libdir__,"configs"))

        self.__directories__ = dict(
            output=self.__outdir__,
            log=self.__logdir__,
            bin=self.__bindir__,
            stable=self.__stable_dir__,
            synth=self.__synth_dir__,
            config=self.__configs__
        )

        # Check the rank. Only the rank zero object needs
        # to actually create the directories. Everyone else just waits
        # for the start signal from the rank zero process.
        if MPI.COMM_WORLD.Get_rank() == 0:
            # This is the leader. Build all the directories.
            for _directory_path in self.__directories__.values():
                _directory_path.mkdir(parents=True,exist_ok=True)
        
        # @@ Parameter Management @@ #
        # Parse the relevant parameters and ensure that they are valid
        # parameters for this parameterization.
        self.__params__ = {}

        if not self.__lib_params__:
            raise NotImplementedError("Subclasses must define '__lib_params__' to specify required parameter names.")

        for paramkey,paramval in parameters.items():

            # Check that the parameter key is a legitimate
            # parameter for the class.
            if paramkey not in self.__lib_params__:
                raise ValueError(f"Unknown libgen parameter: `{paramkey}`")

            # Otherwise, we're going to coerce the inputs to
            # a list and then go from there.
            self.__params__[paramkey] = list(paramval)
        
        # @@ MPI CONFIGURATION @@ #
        self.__comm__ = MPI.COMM_WORLD

        # Pass off to the post initializer #
        self.__post_init__(**kwargs)

    # --------------------------------- #
    # Properties.                       #
    # --------------------------------- #
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
        return int(np.prod([len(pv) for pk, pv in self.__params__.items()]))

    @property
    def shape(self) -> tuple:
        """
        Shape of the parameter grid as a tuple of dimension sizes.

        Returns
        -------
        tuple of int
            A tuple whose length equals the number of parameters, with each entry 
            representing the number of values for that parameter. Matches the shape 
            used to index the N-dimensional parameter space.
        """
        return tuple(len(pv) for _, pv in self.__params__.items())
    
    @property
    def comm(self):
        """
        MPI communicator for the current protocol.

        Returns
        -------
        mpi4py.MPI.Comm
            The communicator used for rank and size queries, defaulting to MPI.COMM_WORLD.
        """
        return self.__comm__
    
    @property
    def mpi_rank(self) -> int:
        """
        Rank of the current MPI process.

        Returns
        -------
        int
            The rank ID (0-indexed) of the current process in the communicator.
        """
        return self.__comm__.Get_rank()

    @property
    def mpi_size(self) -> int:
        """
        Total number of MPI processes participating in the communicator.

        Returns
        -------
        int
            The total number of ranks (MPI processes) active in the communicator.
        """
        return self.__comm__.Get_size()

    # --------------------------------- #
    # Abstract Sockets                  #
    # --------------------------------- #
    # These methods are provided in 
    # subclasses defined by the user. They
    # define the customizable properties of
    # the relevant library.
    def __post_init__(self,**kwargs):
        """
        Hook method for performing additional setup in subclasses after base initialization.

        This method is called automatically at the end of the base class `__init__` method.
        Subclasses can override it to perform setup specific to the particular modification 
        protocol (e.g., setting up XSPEC models, loading additional data, etc.).

        Parameters
        ----------
        **kwargs : dict
            Optional keyword arguments passed from the main constructor, which may be used 
            for customization in derived classes.

        Notes
        -----
        The default implementation sets up default XSPEC environment options, such as:
        - Suppressing plotting windows (`Plot.device = "/null"`)
        - Setting abundance and cross-section tables
        - Choosing fitting statistic and method
        - Reducing XSPEC chatter

        These defaults can be extended or overridden in subclasses.
        """
        # Use null plotting device to suppress any GUI popups
        xspec.Plot.device = "/null"

        # Set default abundances and cross sections
        xspec.Xset.abund = "wilm"
        xspec.Xset.xsect = "vern"

        # Set XSPEC fit method and parameters
        xspec.Fit.statMethod = "cstat"
        xspec.Fit.method = "leven"

        # Suppress XSPEC chatter (set to 10 for minimal, 0 for silent)
        xspec.Xset.chatter = 10

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
    def generate_modified_configuration(self,id: int, **parameters) -> dict:
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
    def fit_unmodified(self, config, **parameters):
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
    def fit_modified(self, config, **parameters):
        """
        Perform fitting on the synthetic data using the modified configuration.

        This may be used as a diagnostic to see if the modified model can recover
        input parameters from its own simulated data.

        Parameters
        ----------
        config : dict
            Modified configuration dictionary.
        **parameters : dict
            Dictionary of parameter values at this grid point.

        Returns
        -------
        float or array-like
            Result of the fit (e.g., recovered temperature or fit statistic).
        """
        ...

    # ---------------------------------- #
    # Tools                              #
    # ---------------------------------- #
    # These are methods that are used elsewhere as tooling.
    # They generally don't need modification.
    def distribute_parameters(self) -> Tuple[int, int]:
        """
        Compute the range of flattened parameter indices assigned to this MPI rank.

        The full parameter space (a grid with shape defined by `self.shape`) is flattened
        and divided as evenly as possible across all MPI ranks. Each rank receives a
        contiguous block of indices to process:
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
        q, r = divmod(self.size, self.mpi_size)

        # Determine the number of items for each rank
        chunk_sizes = [q + 1 if i < r else q for i in range(self.mpi_size)]

        # Compute prefix sums to get start/stop indices
        starts = np.cumsum([0] + chunk_sizes[:-1])
        stops = np.cumsum(chunk_sizes)

        return starts[self.mpi_rank], stops[self.mpi_rank]
    
    def create_rank_output_file(self,start:int, stop:int, temperatures: Sequence[float]) -> h5py.File:
        """
        Create an HDF5 output file for this MPI rank to store fitting results.

        The file is saved in the output directory under the name:
        `libgen_rank_<rank>.h5`, and contains a dataset named 'results' with
        shape `(stop - start, len(temperatures))`.

        Parameters
        ----------
        start : int
            Start index (inclusive) of the parameter grid assigned to this rank.
        stop : int
            Stop index (exclusive) of the parameter grid assigned to this rank.
        temperatures : Sequence[float]
            List of temperature values used in the simulation.

        Returns
        -------
        h5py.File
            Open file handle for writing synthetic fit results.
        """
        rank = self.mpi_rank
        T_count = len(temperatures)

        # Build output filename and path
        filename = f"libgen_rank_{rank}.h5"
        filepath = self.__outdir__ / filename

        # Open HDF5 file and create dataset
        f = h5py.File(filepath, "w")
        f.create_dataset(
            "results",
            shape=(stop-start, T_count),
            dtype="f8",
            compression="gzip"
        )

        # Store metadata (optional but helpful)
        f.attrs["rank"] = rank
        f.attrs["start_index"] = start
        f.attrs["stop_index"] = stop
        f.attrs["temperature_count"] = T_count
        f.attrs["parameter_count"] = stop-start

        return f
    
    def build_synthetic_data(self,idx :int, tidx: int, T, config, **parameters):
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
        mod_model = self.generate_model_modified( T, **parameters)

        # Generate synthetic spectrum using mod_config and mod_model
        synth_path = self.__synth_dir__ / f"synth_{idx}_{tidx}.pha"
        xspec.AllData.clear()
        fakeit = xspec.FakeitSettings(response=config["response"],
                                arf=config["arf"],
                                exposure=config.get("exposure", 50_000),
                                background=config.get("background", ""),
                                correction=config.get("correction", ""),
                                backExposure=config.get("backExposure", ""),
                                fileName=str(synth_path),
                                applyStats=False)
        
        # Clear existing spectra to avoid collisions
        xspec.AllData.fakeit(1, [fakeit])

        # Return the generated synthetic datasets
        return xspec.AllData(1)
    
    def combine_output_files(self, temperature_count: int):
        """
        Combine all per-rank HDF5 output files into a single unified HDF5 file.

        This method is typically called by rank 0 after all ranks finish processing.
        It aggregates the 'results' datasets from each file into one file named
        `libgen_combined.h5` and optionally removes the individual rank-specific files.

        Parameters
        ----------
        temperature_count : int
            Number of temperature values used (i.e., number of columns in the result arrays).

        Raises
        ------
        FileNotFoundError
            If any expected rank-specific output file is missing.
        """
        combined_path = self.__outdir__ / "libgen_combined.h5"
        with h5py.File(combined_path, "w") as fout:

            # Preallocate full result array
            full_shape = (self.size, temperature_count)
            dset = fout.create_dataset(
                "results",
                shape=full_shape,
                dtype="f8",
                compression="gzip"
            )

            for rank in range(self.mpi_size):
                fname = self.__outdir__ / f"libgen_rank_{rank}.h5"
                if not fname.exists():
                    raise FileNotFoundError(f"Missing output file from rank {rank}: {fname}")

                with h5py.File(fname, "r") as fr:
                    start = fr.attrs["start_index"]
                    stop = fr.attrs["stop_index"]
                    dset[start:stop, :] = fr["results"][...]

                # Optional: remove per-rank file to clean up
                os.remove(fname)

            fout.attrs["parameters"] = str(self.__lib_params__)
            fout.attrs["parameter_shape"] = str(self.shape)
            fout.attrs["temperature_count"] = temperature_count

    # ---------------------------------- #
    # Runner                             #
    # ---------------------------------- #
    def generate_library(self, temperatures):
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
        # creating the data file.

        # Determine range of parameter indices for this rank
        pstart_idx, pstop_idx = self.distribute_parameters()

        # Open output file and get write handle
        data_file = self.create_rank_output_file(pstart_idx, pstop_idx, temperatures)
        output_dataset = data_file["results"]


        # @@ Iterate through run @@ #
        # We now iterate through the run of indices and
        # perform the analysis at each index.
        for lidx, gidx in enumerate(range(pstart_idx, pstop_idx)):

            # Construct the N-dim index from the gidx so that we can access
            # the parameter dictionary given our position in the workload.
            midx = np.unravel_index(gidx, self.shape)
            iteration_parameters = {paramkey: paramvalue[midx[param_idx]] for param_idx, (paramkey,paramvalue) in enumerate(self.__params__.items())}


            # Generate the configurations for this parameter set. These
            # are going to be used during the fitting procedure.
            mod_config = self.generate_modified_configuration(gidx, **iteration_parameters)
            unmod_config = self.generate_unmodified_configuration(gidx, **iteration_parameters)

            # @@ Enter Temperature Loop @@ #
            # Now for each of the library parameter values, we iterate over all
            # of the temperatures in order to sample the parameter space.
            for tid, T in enumerate(temperatures):

                # Build the modified model so that we can use it
                # to create the synthetic data.
                mod_model = self.generate_model_modified(T, **iteration_parameters)

                # Build the synthetic data.
                self.build_synthetic_data(gidx,tid, T, mod_config, **iteration_parameters)

                # Fit using unmodified model
                mod_result = self.fit_modified(mod_config)

                # Clear the model and rebuild with the
                # unmodified 
                xspec.AllModels.clear()

                # Build the mod result
                unmod_result = self.fit_unmodified(unmod_config)

                # Optionally: fit using modified model (if relevant)
                # mod_result = self.fit_modified()

                # Store result
                output_dataset[lidx, tid] = unmod_result  # or mod_result, depending on target

        data_file.close()


# ================================== #
# SPECIFIC LIBGEN PROTOCOLS          #
# ================================== #
class GaussianARFProtocol(LibProtocol):
    """
    Characteristic protocol to modify ARFs vis-a-vis a 
    gaussian modification to the ARF.
    """
    __lib_params__ = ["mu", "sigma", "A"]

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)

        # Load a stable ARF and energy grid template
        self.__stable_arf__ = kwargs.get("base_arf","base_arf.arf")
        self._stable_arf_path = self.__stable_dir__ / self.__stable_arf__

        if not self._stable_arf_path.exists():
            raise FileNotFoundError(f"Stable ARF file `{self.__stable_arf__}` not found in ./stable/ directory.")

        # Load ARF grid once (for efficiency)
        self.__base_arf_table__ = self._load_arf(self._stable_arf_path)


    def _load_arf(self, path):
        """
        Load an arf from a path and extract the energy bins and the
        effective areas.
        """
        from astropy.table import Table

        # load the table
        arf_table = Table.read(path)

        # The ARF MUST have ENERG_LO, ENERG_HI, and SPECRESP.
        return arf_table

    def apply_gaussian_modification(self, mu, sigma, A):
        """
        Returns a modified effective area array with Gaussian bump.
        """
        # Create the copied table
        _new_table = self.__base_arf_table__.copy()


        gaussian = lambda x: A * np.exp(-((x-mu)/sigma)**2)

        bin_centers = 0.5*(_new_table['ENERG_LO'] + _new_table['ENERG_HI'])

        _new_table['SPECRESP'] *= (1+gaussian(bin_centers))

        return _new_table

    def generate_unmodified_configuration(self, id, **parameters):
        return {
            "response": str(self.__stable_dir__ / "base.rmf"),
            "arf": str(self._stable_arf_path),
            "exposure": 50000
        }

    def generate_modified_configuration(self, id, **parameters):
        mod_arf_path = self.__bindir__ / f"mod_arf_{id}.arf"

        if not mod_arf_path.exists():
            # Compute modified effective area
            mu = parameters["mu"]
            sigma = parameters["sigma"]
            A = parameters["A"]
            new_table = self.apply_gaussian_modification(mu, sigma, A)
            new_table.write(mod_arf_path,format='fits')

        return {
            "response": str(self.__stable_dir__ / "base.rmf"),
            "arf": str(mod_arf_path),
            "exposure": 50000
        }

    def generate_model_unmodified(self, T, **parameters):
        m = xspec.Model("tbabs*apec")
        m(1).values = 0.1   # nH
        m(2).values = T     # kT
        m(3).values = 0.3   # abundance
        m(4).values = 0.05  # redshift
        m(5).values = 1.0   # norm
        return m

    def generate_model_modified(self, T, **parameters):
        return self.generate_model_unmodified(T, **parameters)

    def fit_unmodified(self, config, **parameters):
        # Update response/arf for spectrum
        xspec.AllData(1).response = config["response"]
        xspec.AllData(1).arf = config["arf"]
        xspec.AllModels.setPars({2: "1.0"})  # Reset T
        xspec.Fit.perform()
        return float(xspec.AllModels(1)(2).values[0])

    def fit_modified(self, config, **parameters):
        xspec.AllData(1).response = config["response"]
        xspec.AllData(1).arf = config["arf"]
        xspec.AllModels.setPars({2: "1.0"})
        xspec.Fit.perform()
        return float(xspec.AllModels(1)(2).values[0])


