"""
Minor utility functions for interacting with spectra
during other workflows.
"""
from pathlib import Path

import numpy as np
from astropy.io import fits


def group_min_counts(
    pha_path,
    pha_path_out=None,
    *,
    min_counts=3,
    counts_column="COUNTS",
    grouping_column="GROUPING",
    ext=1,
    overwrite=True,
):
    """
    Reproduce `grppha group min X` behavior for XSPEC-compatible PHA files.

    Parameters
    ----------
    pha_path : str or Path
        Input PHA file.
    pha_path_out : str or Path, optional
        Output file. If None, modify in-place (requires `overwrite=True`).
    min_counts : int, default=3
        Minimum counts per group.
    counts_column : str, default="COUNTS"
        Column with counts data.
    grouping_column : str, default="GROUPING"
        Column for XSPEC grouping metadata. Created if missing.
    ext : int, default=1
        FITS extension with spectral table.
    overwrite : bool, default=True
        Allow overwriting output or input file.
    """
    pha_path = Path(pha_path).expanduser().resolve()
    inplace = pha_path_out is None

    if inplace and not overwrite:
        raise ValueError("In-place modification requires overwrite=True.")

    if not inplace:
        pha_path_out = Path(pha_path_out).expanduser().resolve()
        if pha_path_out.exists() and not overwrite:
            raise FileExistsError(f"{pha_path_out} exists and overwrite=False.")

    # Copy file if needed
    if inplace:
        hdul = fits.open(pha_path, mode="update")
    else:
        with fits.open(pha_path) as hdul_in:  # type: ignore
            hdul_in.writeto(pha_path_out, overwrite=True)
        hdul = fits.open(pha_path_out, mode="update")

    try:
        spec_hdu = hdul[ext]
        spectrum = spec_hdu.data
        counts = np.asarray(spectrum[counts_column])

        if grouping_column not in spectrum.columns.names:
            from astropy.io.fits import BinTableHDU, ColDefs, Column

            grouping = np.zeros_like(counts, dtype=np.int16)
            new_col = Column(name=grouping_column, format="I", array=grouping)
            new_cols = ColDefs(spectrum.columns) + new_col
            new_hdu = BinTableHDU.from_columns(new_cols, header=spec_hdu.header)

            # Replace HDU in file
            hdul[ext] = new_hdu
            spectrum = new_hdu.data
        else:
            spectrum[grouping_column][:] = 0

        # Apply XSPEC grouping logic
        bin_lo = 0
        while bin_lo < len(counts):
            total = 0
            bin_hi = bin_lo
            while bin_hi < len(counts) and total < min_counts:
                total += counts[bin_hi]
                bin_hi += 1

            spectrum[grouping_column][bin_lo] = 1  # Group start
            spectrum[grouping_column][bin_lo + 1 : bin_hi] = -1  # Group members
            bin_lo = bin_hi

        hdul.flush()

    finally:
        hdul.close()
