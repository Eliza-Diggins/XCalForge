"""
Example implementation of a modification library for ARF calibration distortions.
"""

from pathlib import Path
from typing import Sequence, Tuple, Union

import numpy as np

from xforge.utilities.env import get_xspec

from .base import ModificationLibrary

xspec = get_xspec()


class MyCustomLibrary(ModificationLibrary):
    """
    ModificationLibrary for simulating Gaussian perturbations to ARF files.

    Defines:
    - Allowed hyperparameters (mu: bias center, sigma: bias width)
    - Logic for generating distorted ARFs
    - XSPEC model construction
    - Fitting routines for temperature recovery analysis
    """

    # ------------------------------------ #
    # Required Class Flags                 #
    # ------------------------------------ #
    __PARAMETERS__ = ["mu", "sigma"]

    __CONFIG__ = {
        "logging.file_level": "DEBUG",
        "logging.term_level": "INFO",
        "logging.fmt": "%(asctime)s [%(levelname)s] [RANK=%(rank)s]: %(message)s",
    }

    __OUTPUT_SHAPE__ = (3,)
    """
    (recovered_temperature, fit_statistic, uncertainty)
    """

    # ------------------------------------ #
    # Optional Post-Initialization         #
    # ------------------------------------ #
    def __post_init__(self):
        """
        Ensure required response files exist and pre-cache baseline ARF path.
        """
        self.baseline_rmf = Path("responses/default.rmf")
        self.baseline_arf = Path("responses/default.arf")

        if not self.baseline_rmf.exists() or not self.baseline_arf.exists():
            raise FileNotFoundError("Baseline response files are missing.")

    # ------------------------------------ #
    # Required Configuration Generators    #
    # ------------------------------------ #
    def generate_unmodified_configuration(self, id: int, **parameters) -> dict:
        """
        Baseline configuration used during recovery fits.
        """
        ...

    def generate_modified_configuration(self, id: int, **parameters) -> dict:
        """
        Generate a Gaussian-distorted ARF for synthetic data.
        """
        ...

    # ------------------------------------ #
    # XSPEC Model Builders                 #
    # ------------------------------------ #
    def generate_model_unmodified(self, T: float, **parameters):
        """
        XSPEC model used for fitting with unmodified calibration.
        """
        return xspec.Model(f"tbabs*apec & kT={T}")

    def generate_model_modified(self, T: float, **parameters):
        """
        XSPEC model used for synthetic data generation with modified calibration.
        """
        return xspec.Model(f"tbabs*apec & kT={T}")

    # ------------------------------------ #
    # Fitting Routines                     #
    # ------------------------------------ #
    def fit_unmodified(self, config: dict, **parameters) -> Tuple[float, float, float]:
        """
        Fit spectrum with unmodified configuration and extract recovery results.
        """
        spectrum = xspec.AllData(1)
        spectrum.response = config["response"]
        spectrum.arf = config["arf"]

        xspec.Fit.perform()
        recovered_T = float(xspec.AllModels(1).apec.kT.values[0])
        fit_stat = float(xspec.Fit.statistic)
        uncertainty = float(xspec.AllModels(1).apec.kT.sigma)

        return recovered_T, fit_stat, uncertainty

    def fit_modified(self, config: dict, **parameters) -> Tuple[float, float, float]:
        """
        Optional sanity check fit with the modified (distorted) configuration.
        """
        spectrum = xspec.AllData(1)
        spectrum.response = config["response"]
        spectrum.arf = config["arf"]

        xspec.Fit.perform()
        recovered_T = float(xspec.AllModels(1).apec.kT.values[0])
        fit_stat = float(xspec.Fit.statistic)
        uncertainty = float(xspec.AllModels(1).apec.kT.sigma)

        return recovered_T, fit_stat, uncertainty

    # ------------------------------------ #
    # Optional Output Customization        #
    # ------------------------------------ #
    def _write_output(
        self,
        result_mod: Union[Sequence[float], np.ndarray],
        result_unmod: Union[Sequence[float], np.ndarray],
        true_temperature: float,
        **parameters,
    ) -> np.ndarray:
        """
        Default output: (recovered_temperature, fit_statistic, uncertainty)
        """
        output = np.full(self.__OUTPUT_SHAPE__, np.nan, dtype=float)

        try:
            output[0] = float(result_unmod[0])
            output[1] = float(result_unmod[1])
            output[2] = float(result_unmod[2])
        except (IndexError, TypeError) as e:
            raise ValueError(f"Invalid result_unmod format: {result_unmod}") from e

        return output

    # ------------------------------------ #
    # Recommended Cleanup Hooks            #
    # ------------------------------------ #
    def cleanup_temperature_iteration(
        self, global_parameter_index: int, temperature_index: int
    ):
        """
        Remove synthetic PHA files after each temperature simulation to limit disk usage.
        """
        synth_path = (
            self.tempdir / f"synth.{global_parameter_index}.{temperature_index}.pha"
        )
        if synth_path.exists():
            synth_path.unlink()
            self.logger.debug("Deleted synthetic spectrum: %s", synth_path)

    def cleanup_parameter_iteration(self, global_parameter_index: int):
        """
        Example cleanup: optionally remove temporary ARFs if generated per parameter.
        """
        mu = self.__parameters__["mu"]
        sigma = self.__parameters__["sigma"]

        # Reconstruct index for this parameter combination
        midx = np.unravel_index(global_parameter_index, self.shape)
        mu_val = mu[midx[self.__PARAMETERS__.index("mu")]]
        sigma_val = sigma[midx[self.__PARAMETERS__.index("sigma")]]

        distorted_arf = (
            self.__bindir__ / f"distorted_mu{mu_val:.3f}_sigma{sigma_val:.3f}.arf"
        )
        if distorted_arf.exists():
            distorted_arf.unlink()
            self.logger.debug("Deleted distorted ARF: %s", distorted_arf)
