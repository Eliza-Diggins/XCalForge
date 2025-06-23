"""Skeleton template for custom library generation implementations.
"""
from typing import Tuple

import numpy as np

from xforge.utils import get_xspec

from .base import ModificationLibrary

xspec = get_xspec()


class MyCustomLibrary(ModificationLibrary):
    """
    Example subclass of ModificationLibrary for a specific calibration modification study.

    This subclass defines:
    - Allowed hyperparameters
    - Default configuration options
    - All required hooks for synthetic data generation and fitting
    """

    # ------------------------------------ #
    # Required Class Flags                 #
    # ------------------------------------ #
    __PARAMETERS__ = ["mu", "sigma"]
    """
    Allowed parameter grid keys. Must match keys provided when creating the library.
    """

    __CONFIG__ = {
        "logging.file_level": "INFO",
        "logging.term_level": "INFO",
        "logging.fmt": "%(asctime)s - %(levelname)s - %(message)s",
        # Add project-specific defaults here
    }

    __OUTPUT_SHAPE__ = (3,)
    """
    Output shape for each grid point: (recovered_temp, fit_statistic, uncertainty).
    Extend if needed.
    """

    # ------------------------------------ #
    # Required Initialization Hook         #
    # ------------------------------------ #
    def __post_init__(self):
        """
        Optional post-initialization logic after base class loads parameters and config.
        """
        pass  # Optional: Load extra resources or validate setup

    # ------------------------------------ #
    # Required Generation Hooks            #
    # ------------------------------------ #
    def generate_unmodified_configuration(self, id: int, **parameters) -> dict:
        """
        Return the unmodified telescope configuration for XSPEC fitting.
        """
        return {
            "response": "responses/default.rmf",
            "arf": "responses/default.arf",
            "exposure": 50_000,
        }

    def generate_modified_configuration(self, id: int, **parameters) -> dict:
        """
        Return the modified telescope configuration for synthetic data generation.
        """
        mu = parameters["mu"]
        sigma = parameters["sigma"]

        # Example: Create distorted ARF based on parameters
        arf_path = f"bin/distorted_mu{mu}_sigma{sigma}.arf"

        return {
            "response": "responses/default.rmf",
            "arf": arf_path,
            "exposure": 50_000,
        }

    def generate_model_unmodified(self, T: float, **parameters):
        """
        Return the XSPEC model for unmodified configuration.
        """
        return xspec.Model(f"tbabs*apec & kT={T}")

    def generate_model_modified(self, T: float, **parameters):
        """
        Return the XSPEC model for modified configuration (typically same as unmodified).
        """
        return xspec.Model(f"tbabs*apec & kT={T}")

    def fit_unmodified(self, config: dict, **parameters) -> Tuple[float, float, float]:
        """
        Fit synthetic spectrum with unmodified configuration and extract results.
        """
        # Example: Fake extraction, replace with real XSPEC logic
        recovered_T = np.random.normal(loc=parameters["mu"], scale=parameters["sigma"])
        fit_stat = np.random.uniform(0, 10)
        uncertainty = np.random.uniform(0.01, 0.1)
        return recovered_T, fit_stat, uncertainty

    def fit_modified(self, config: dict, **parameters) -> Tuple[float, float, float]:
        """
        Optional: Fit with the modified configuration (sanity check).
        """
        true_T = parameters.get("true_temperature", 1.0)
        fit_stat = np.random.uniform(0, 10)
        uncertainty = np.random.uniform(0.01, 0.1)
        return true_T, fit_stat, uncertainty

    # ------------------------------------ #
    # Optional: Customize Output Format    #
    # ------------------------------------ #
    def write_output(
        self, result_mod, result_unmod, true_temperature: float, **parameters
    ) -> np.ndarray:
        """
        Customize output format stored in the HDF5 library.
        """
        output = np.zeros(self.__OUTPUT_SHAPE__, dtype=float)
        output[0] = result_unmod[0]  # Recovered temperature
        output[1] = result_unmod[1]  # Fit statistic
        output[2] = result_unmod[2]  # Uncertainty
        return output
