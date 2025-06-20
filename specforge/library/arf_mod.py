"""
Library modification routines predicated on the use of 
ARF modifications.
"""
from .base import ModificationLibrary
from specforge.utils import get_xspec
from pathlib import Path
from astropy.table import Table
import numpy as np

xspec = get_xspec()


class GaussianARFLibrary(ModificationLibrary):
    # ================================ #
    # CLASS FLAGS                      #
    # ================================ #
    # The class flags are simple markers that are used
    # for various behaviors in the code below. These may need to
    # be modified in subclasses. Please read the relevant documentation
    # in order to determine if a particular flag needs to be modified.
    __PARAMETERS__ = ["mu", "sigma", "A"]
    __CONFIG__ = {
        "logging.level": "INFO",
        "input.arf": "base.arf",
        "input.rmf": "base.rmf"
    }

    # ========================== #
    # Initialization             #
    # ========================== #
    # These methods are all concerned with loading libraries from
    # disk. They can be modified in subclasses, but should be kept
    # relatively consistent.
    def __post_init__(self):
        # In the post-init for the ARF modification, we
        # require that the configuration correctly points to
        # existing arf and rmf files. If it does not, we need
        # to warn the user.
        self.__assets__['base_arf'] = Path(self.__config__['input.arf']).expanduser().resolve()
        self.__assets__['base_rmf'] = Path(self.__config__['input.rmf']).expanduser().resolve()

        if not self.__assets__['base_arf'].exists():
            self.logger.warning("The base arf at %s doesn't exist. Update the config.", self.__assets__['base_arf'])
        if not self.__assets__['base_rmf'].exists():
            self.logger.warning("The base rmf at %s doesn't exist. Update the config.", self.__assets__['base_rmf'])

        # Create the buffers for storing the loaded 
        # arf and rmf if they are asked for.
        self._loaded_base_arf = None
        self._loaded_base_rmf = None

    # ============================ #
    # Building Libraries.          #
    # ============================ #
    # These methods focus on ensuring that new libraries
    # can be created smoothly. These do not actually run the
    # library generation process.
    @classmethod
    def create_library(cls,
                       directory,
                       parameters,
                       base_arf,
                       base_rmf,
                       overwrite: bool = False,
                       config = None
                       ):
        # Ensure that the base arf and base rmf exist 
        # and are paths.
        base_arf,base_rmf = Path(base_arf).expanduser().resolve(), Path(base_rmf).expanduser().resolve()

        config = config if config is not None else {}

        config['input.arf'] = str(base_arf)
        config['input.rmf'] = str(base_rmf)

        return super().create_library(directory,parameters,overwrite=overwrite,config=config)

    # ========================== #
    # Properties                 #
    # ========================== #
    @property
    def base_arf_table(self):
        if self._loaded_base_arf is None:
            self._loaded_base_arf = Table.read(self.__assets__['base_arf'])

        return self._loaded_base_arf

    @property
    def base_rmf_table(self):
        if self._loaded_base_rmf is None:
            self._loaded_base_rmf = Table.read(self.__assets__['base_rmf'])

        return self._loaded_base_rmf

    # ================================ #
    # Library Generation               #
    # ================================ #
    def _apply_gaussian_mod(self, mu, sigma, A):
        """
        Apply a Gaussian modification to the base ARF.

        Returns
        -------
        Table
            Modified ARF table.
        """
        arf = self.base_arf_table.copy()
        bin_centers = 0.5 * (arf["ENERG_LO"] + arf["ENERG_HI"])
        profile = A * np.exp(-((bin_centers - mu) / sigma) ** 2)
        arf["SPECRESP"] *= (1.0 + profile)
        return arf

    def generate_unmodified_configuration(self, id: int, **parameters):
        """
        Return the unperturbed configuration.
        """
        return {
            "response": str(self.__assets__['base_rmf']),
            "arf": str(self.__assets__['base_arf']),
            "exposure": 50000
        }

    def generate_modified_configuration(self, id: int, **parameters):
        """
        Create modified ARF file and return the config.
        """
        mod_path = self.__bindir__ / f"mod_arf_{id}.arf"
        if not mod_path.exists():
            mu = parameters["mu"]
            sigma = parameters["sigma"]
            A = parameters["A"]
            mod_arf = self._apply_gaussian_mod(mu, sigma, A)
            mod_arf.write(mod_path, format="fits")
        return {
            "response": str(self.__assets__['base_rmf']),
            "arf": str(mod_path),
            "exposure": 50000
        }

    def generate_model_unmodified(self, T, **parameters):
        """
        Construct XSPEC model for unmodified case.
        """
        model = xspec.Model("tbabs*apec")
        model.setPars({
            1: 0.1,   # nH
            2: T,     # kT
            3: 0.3,   # abundance
            4: 0.05,  # redshift
            5: 1.0    # norm
        })
        return model

    def generate_model_modified(self, T, **parameters):
        """
        Same model used for modified and unmodified.
        """
        return self.generate_model_unmodified(T, **parameters)

    def fit_unmodified(self, config, **parameters):
        """
        Perform fit with unmodified configuration.
        """
        xspec.AllData(1).response = config["response"]
        xspec.AllData(1).response.arf = config["arf"]
        xspec.AllModels(1).setPars({2: "1.0"})
        xspec.Fit.perform()
        return float(xspec.AllModels(1)(2).values[0])

    def fit_modified(self, config, **parameters):
        """
        Perform fit with modified configuration.
        """
        xspec.AllData(1).response = config["response"]
        xspec.AllData(1).response.arf = config["arf"]
        xspec.AllModels(1).setPars({2: "1.0"})
        xspec.Fit.perform()
        return float(xspec.AllModels(1)(2).values[0])

if __name__ == 'main':
    print("Hello")