"""
Library for Gaussian ARF modification studies.
"""
from pathlib import Path

import numpy as np
from astropy.table import Table

from xforge.utilities.env import get_xspec

from .base import ModificationLibrary

xspec = get_xspec()


class GaussianARFLibrary(ModificationLibrary):
    """
    ModificationLibrary for simulating Gaussian perturbations to ARF effective areas.

    This library applies localized, energy-dependent distortions to the Auxiliary Response File (ARF),
    modeling realistic calibration errors that affect X-ray effective area measurements.

    The perturbation is defined as a multiplicative fractional modification applied to the baseline ARF:

    .. math::

        S'(E) = S(E) \\times \\left[ 1 + A \\exp\\left( -\\frac{(E - \\mu)^2}{\\sigma^2} \\right) \\right]

    Where:

    - :math:`S(E)` is the original effective area at energy :math:`E` from the baseline ARF.
    - :math:`S'(E)` is the modified effective area at energy :math:`E`.
    - :math:`\\mu` is the center of the Gaussian distortion in keV.
    - :math:`\\sigma` is the width (standard deviation) of the Gaussian in keV.
    - :math:`A` is the fractional amplitude of the distortion (unitless; positive or negative).

    This framework enables systematic exploration of how localized miscalibrations in effective area
    impact astrophysical parameter recovery, such as biases in X-ray temperature measurements.

    **Workflow**

    For each grid point defined by :math:`(\\mu, \\sigma, A)`:

    1. A modified ARF is generated with the Gaussian perturbation applied.
    2. Synthetic X-ray spectra are simulated using the modified ARF and baseline RMF.
    3. The synthetic spectra are analyzed with the unmodified configuration to quantify recovery bias.
    4. Optionally, the modified configuration is also used to verify correct parameter recovery.
    5. Results are stored in a structured, disk-backed HDF5 library for downstream analysis.

    **Grid Parameters**

    - ``mu`` (:math:`\\mu`) : Center of Gaussian distortion in keV.
    - ``sigma`` (:math:`\\sigma`) : Width of Gaussian distortion in keV.
    - ``A`` : Amplitude of distortion (unitless, fractional).

    **Configuration Options**

    - ``input.arf`` : Path to the baseline ARF file (FITS format, required).
    - ``input.rmf`` : Path to the baseline RMF file (FITS format, required).
    - ``error.threshold`` : Delta fit-statistic threshold for XSPEC uncertainty estimation (default = 1.0).

    **Output Format**

    The final library results are stored with shape:

    .. math::

        (N_\\mu, N_\\sigma, N_A, N_T, 3)

    Where:

    - :math:`N_\\mu, N_\\sigma, N_A` are the grid sizes for :math:`\\mu`, :math:`\\sigma`, and :math:`A`.
    - :math:`N_T` is the number of simulated temperatures.
    - The final dimension contains:

      1. Lower confidence bound on recovered temperature.
      2. Best-fit recovered temperature.
      3. Upper confidence bound on recovered temperature.

    **Notes**

    - The same astrophysical model (:code:`tbabs*apec`) is used for both modified and unmodified fits.
    - Temporary files (e.g., synthetic spectra, modified ARFs) are automatically cleaned up after use.
    - Supports MPI-parallel generation for efficient exploration of large parameter grids.
    - Designed for applications in calibration uncertainty quantification, emulator training, and bias analysis.

    **See Also**

    - :class:`ModificationLibrary` : Base class providing core library structure and simulation logic.
    """

    __PARAMETERS__ = ["mu", "sigma", "A"]
    """
    List of allowed hyperparameters for the Gaussian ARF modification grid.

    - ``mu`` (:math:`\\mu`) : Center of the Gaussian perturbation in keV.
    - ``sigma`` (:math:`\\sigma`) : Standard deviation (width) of the Gaussian perturbation in keV.
    - ``A`` : Amplitude of the perturbation, expressed as a fractional distortion (unitless).

    These parameters define the axes of the modification grid explored during library generation.
    """

    __CONFIG__ = {
        "input.arf": "base.arf",
        "input.rmf": "base.rmf",
        "error.threshold": 1.0,
        **ModificationLibrary.__CONFIG__,
    }
    """
    Default configuration options for the Gaussian ARF library.

    - ``input.arf`` : Path to the baseline ARF file (FITS format) used as the unmodified reference.
    - ``input.rmf`` : Path to the baseline RMF file (FITS format) defining energy redistribution.
    - ``error.threshold`` : Delta fit-statistic (:math:`\\Delta \\chi^2`) used for XSPEC uncertainty estimation.

    These defaults may be overridden when creating the library via ``create_library(..., config=...)``.

    Inherits additional options from :class:`ModificationLibrary`.
    """

    __OUTPUT_SHAPE__ = (3,)
    """
    Shape of the stored result array for each grid point and temperature.

    The default structure is:

    .. math::

        (T_\\mathrm{lo},\\; T_\\mathrm{fit},\\; T_\\mathrm{hi})

    Where:

    - :math:`T_\\mathrm{lo}` : Lower bound of the confidence interval on the recovered temperature.
    - :math:`T_\\mathrm{fit}` : Best-fit recovered temperature.
    - :math:`T_\\mathrm{hi}` : Upper bound of the confidence interval on the recovered temperature.

    This format matches common astrophysical uncertainty reporting.
    """

    def __post_init__(self):
        """
        Validate configuration and cache baseline response paths.
        """
        self.__assets__["base_arf"] = (
            Path(self.config["input.arf"]).expanduser().resolve()
        )
        self.__assets__["base_rmf"] = (
            Path(self.config["input.rmf"]).expanduser().resolve()
        )

        if not self.__assets__["base_arf"].exists():
            raise FileNotFoundError(
                f"Base ARF not found: {self.__assets__['base_arf']}"
            )
        if not self.__assets__["base_rmf"].exists():
            raise FileNotFoundError(
                f"Base RMF not found: {self.__assets__['base_rmf']}"
            )

        self._loaded_base_arf = None
        self._loaded_base_rmf = None

    @classmethod
    def create_library(
        cls, directory, parameters, base_arf, base_rmf, *, overwrite=False, config=None
    ):
        """
        Create a new Gaussian ARF modification library.

        Parameters
        ----------
        directory : str or Path
            Target library directory.
        parameters : dict
            Grid of hyperparameter values.
        base_arf : str or Path
            Baseline ARF file.
        base_rmf : str or Path
            Baseline RMF file.
        overwrite : bool, optional
            Allow overwriting existing library.
        config : dict, optional
            Additional configuration overrides.
        """
        base_arf = Path(base_arf).expanduser().resolve()
        base_rmf = Path(base_rmf).expanduser().resolve()

        if not base_arf.exists() or not base_rmf.exists():
            raise FileNotFoundError(
                "Baseline ARF or RMF not found for library creation."
            )

        config = dict(config) if config else {}
        config["input.arf"] = str(base_arf)
        config["input.rmf"] = str(base_rmf)

        return super().create_library(
            directory, parameters, overwrite=overwrite, config=config
        )

    @property
    def base_arf_table(self):
        if self._loaded_base_arf is None:
            self._loaded_base_arf = Table.read(self.__assets__["base_arf"])
        return self._loaded_base_arf

    @property
    def base_rmf_table(self):
        if self._loaded_base_rmf is None:
            self._loaded_base_rmf = Table.read(self.__assets__["base_rmf"])
        return self._loaded_base_rmf

    def _apply_gaussian_mod(self, mu, sigma, A):
        """
        Apply Gaussian perturbation to ARF.

        Returns
        -------
        astropy.table.Table
            Modified ARF table.
        """
        arf = self.base_arf_table.copy()
        bin_centers = 0.5 * (arf["ENERG_LO"] + arf["ENERG_HI"])
        profile = A * np.exp(-(((bin_centers - mu) / sigma) ** 2))
        arf["SPECRESP"] *= 1.0 + profile
        return arf

    def generate_unmodified_configuration(self, id: int, **parameters):
        return {
            "response": str(self.__assets__["base_rmf"]),
            "arf": str(self.__assets__["base_arf"]),
            "exposure": 50_000,
        }

    def generate_modified_configuration(self, id: int, **parameters):
        """
        Generate modified ARF file for this parameter grid point.
        """
        mod_path = self.tempdir / f"modarf.{id}.arf"
        if not mod_path.exists():
            mod_arf = self._apply_gaussian_mod(
                parameters["mu"], parameters["sigma"], parameters["A"]
            )
            mod_arf.write(mod_path, format="fits")
        return {
            "response": str(self.__assets__["base_rmf"]),
            "arf": str(mod_path),
            "exposure": 50_000,
        }

    def generate_model_unmodified(self, T, **parameters):
        """
        Construct XSPEC model for both modified and unmodified configurations.
        """
        model = xspec.Model("tbabs*apec")
        model.setPars(
            {
                1: 0.1,  # nH (fixed)
                2: T,  # kT (true temperature or initial guess)
                3: 0.3,  # abundance
                4: 0.05,  # redshift
                5: 1.0,  # normalization
            }
        )
        return model

    def generate_model_modified(self, T, **parameters):
        return self.generate_model_unmodified(T, **parameters)

    def fit_unmodified(self, config, **parameters):
        """
        Fit spectrum with unmodified configuration and compute uncertainties.

        Returns
        -------
        (lower_bound, best_fit_temperature, upper_bound)
        """
        xspec.AllData(1).response = config["response"]
        xspec.AllData(1).response.arf = config["arf"]
        xspec.AllModels(1).setPars({2: "1.0"})

        xspec.Fit.perform()

        delta_chi = self.config.get("error.threshold", 1.0)
        xspec.Fit.error(f"{delta_chi} 2")

        T_fit = float(xspec.AllModels(1)(2).values[0])
        T_lo, T_hi, status = xspec.AllModels(1)(2).error

        if status != "FFFFFFFFF":
            self.logger.warning("Non-standard XSPEC error status: %s", status)

        return (T_lo, T_fit, T_hi)

    def fit_modified(self, config, **parameters):
        """
        Optional sanity check fit with modified configuration.
        """
        return self.fit_unmodified(config, **parameters)

    def cleanup_temperature_iteration(
        self, global_parameter_index: int, temperature_index: int
    ):
        """
        Remove synthetic spectrum after each temperature simulation.
        """
        path = self.tempdir / f"synth.{global_parameter_index}.{temperature_index}.pha"
        if path.exists():
            path.unlink()
            self.logger.debug("Deleted synthetic spectrum: %s", path)

    def cleanup_parameter_iteration(self, global_parameter_index: int):
        """
        Remove modified ARF after completing all temperatures for a grid point.
        """
        path = self.tempdir / f"modarf.{global_parameter_index}.arf"
        if path.exists():
            path.unlink()
            self.logger.debug("Deleted modified ARF: %s", path)
