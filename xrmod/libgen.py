"""
The ``libgen.py`` module is responsible for generating the training
library of modifications. In order to do so, the LibProtocol class is used to
define and configure the behavior of the library during its generation.
"""
from typing import Union, Dict, Sequence, Tuple
from pathlib import Path
import os
import xspec
import numpy as np
from mpi4py import MPI
import h5py
from abc import ABC, abstractmethod

class LibProtocol(ABC):
    __lib_params__ = []
    """ Valid parameters for the library protocol. These should be
    filled by the user when building the subclass for their workflow.

    ``__lib_params__`` is a list of strings each of which may be specified as
    a modification parameter in the ``__init__`` kwargs. These should be the
    parameters of the modification map.

    Example: For a gaussian shift in the ARF, you might take mu, sigma, A as
    the params.
    """
    
    def __init__(self, lib_dir: Union[str,Path], parameters: Dict[str,Sequence], **kwargs):
        """
        Construct the LibProtocol object in order to get all the settings configured for
        proceeding.

        Parameters
        ----------

        lib_dir: str
            The directory where the library is going to be generated.
        parameters: dict
            The parameter values to use for building the grid of the parameter space.
        **kwargs:
            Additional kwargs which can be used for subclassing.
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
        return int(np.prod([len(pv) for pk,pv in self.__params__.items()]))

    @property
    def shape(self) -> tuple:
        return tuple(len(pv) for _,pv in self.__params__.items())
    
    @property
    def comm(self):
        return self.__comm__
    
    @property
    def mpi_rank(self) -> int:
        return self.__comm__.Get_rank()

    @property
    def mpi_size(self) -> int:
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
        The ``__post_init__`` method may be written to add further initialization
        behavior to subclasses. By default, it does nothing.
        """
        pass

    @abstractmethod
    def generate_unmodified_configuration(self, id: int, **parameters) -> dict:
        """
        This method should be used to produce the unmodified configuration
        for the simulation.

        Returns
        -------
        dict
            The configuration dictionary. It may return any of the
            following:

            - ``response`` (required): The RMF file to use for the
              simulated data.
            - ``arf`` (required): The ARF file used to simulate the data.
            - ``background`` (optional): The background.
            - ``exposure`` (required): The exposure time.
            - ``correction`` (optional): Optional correction norm factor.
            - ``backExposure`` (optional): Optional background exposure time modifier.

              For exposure and correction, if left empty fakeit will use the values from the
              original spectrum, or 1.0 if not based on an original spectrum. Each of these may be entered as a string or float.
        
        Notes
        -----

        We HIGHLY suggest that ``response`` and ``arf`` be EITHER a fixed ``.rmf`` and ``.arf`` stored
        in the ``./stable`` directory of the library or that they be generated based on the ``parameters``, in which case
        they should be placed in the ``./bin`` and named ``unmod_arf_{id}.arf``.
        """

    @abstractmethod
    def generate_modified_configuration(self,id: int, **parameters) -> dict:
        """
        This method should be used to produce the modified configuration
        for the simulation.

        Returns
        -------
        dict
            The configuration dictionary. It may return any of the
            following:

            - ``response`` (required): The RMF file to use for the
              simulated data.
            - ``arf`` (required): The ARF file used to simulate the data.
            - ``background`` (optional): The background.
            - ``exposure`` (required): The exposure time.
            - ``correction`` (optional): Optional correction norm factor.
            - ``backExposure`` (optional): Optional background exposure time modifier.

              For exposure and correction, if left empty fakeit will use the values from the
              original spectrum, or 1.0 if not based on an original spectrum. Each of these may be entered as a string or float.
        
        Notes
        -----

        We HIGHLY suggest that ``response`` and ``arf`` be EITHER a fixed ``.rmf`` and ``.arf`` stored
        in the ``./stable`` directory of the library or that they be generated based on the ``parameters``, in which case
        they should be placed in the ``./bin`` and named ``unmod_arf_{id}.arf``.
        """

    @abstractmethod
    def generate_model_unmodified(self, T, **parameters):
        """
        This method is called to create the relevant XSPEC mode
        when fitting to the UNMODIFIED configuration with the synthetic data.

        By default, this is also used in ``generate_model_modified`` which results
        in identical models being used (with potentially different configurations);
        however, these may be modified to parameterized modifications which change the
        presumed model instead of the configuration.
        """
        pass

    @abstractmethod
    def generate_model_modified(self, T, **parameters):
        """
        This method is called to create the relevant XSPEC mode
        when fitting to the MODIFIED configuration with the synthetic data.

        It is also the model used to generate the spectrum for the synethetic data.
        """
        return self.generate_model_unmodified(T, **parameters)
    
    @abstractmethod
    def fit_unmodified(self, config, **parameters):
        pass

    @abstractmethod
    def fit_modified(self, config, **parameters):
        pass

    # ---------------------------------- #
    # Tools                              #
    # ---------------------------------- #
    # These are methods that are used elsewhere as tooling.
    # They generally don't need modification.
    def distribute_parameters(self) -> Tuple[int, int]:
        """
        Determine the contiguous linear index range assigned to this MPI rank
        for the flattened parameter grid.

        The full parameter grid (with shape [N1, N2, ..., Nk]) has total size `self.size`.
        This space is partitioned as evenly as possible across MPI processes:
        - The first `r = self.size % mpi_size` ranks get `q + 1` elements,
        - The remaining `mpi_size - r` ranks get `q` elements.

        Returns
        -------
        start : int
            Inclusive starting index in the flattened parameter space for this rank.
        stop : int
            Exclusive ending index in the flattened parameter space for this rank.
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
        Creates an HDF5 output file for this MPI rank in the output directory,
        with a dataset for storing the results of synthetic fits.

        The output file will be created as:
            <lib_dir>/out/libgen_rank_<rank>.h5

        The dataset inside will be named 'results' and shaped as:
            (N_local, len(temperatures))

        where N_local is the number of parameter grid points assigned to this rank.

        Parameters
        ----------
        temperatures : Sequence[float]
            The list of input temperatures used in the synthetic simulations.

        Returns
        -------
        h5py.File
            The open HDF5 file handle (in append mode) for writing results.
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
        Generate synthetic photon data using the modified model generator and the modified configuration.

        Parameters
        ----------
        temperature : float
            Temperature value passed to the model generator.
        
        config: dict
            The configuration settings.
        
        Returns
        -------
        datasets : list of xspec.Spectrum
            List of synthetic spectra generated from the fakeit run(s).
            
        Notes
        -----
        What this does (in order) is
        
        1. Clear any existing models and build a new one. This ensures clarity and
        may be relaxed later if we deem in interferes with efficiency.
        2. We then clear all of our data (photon lists / observations) from memory and create
        a fake data file. This is then loaded into XSPEC as AllData(1).
        3. We then return this data object.
        
        This will CLEAR ANY EXISTING XSPEC MODELS AND DATA
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
