from pathlib import Path

from xforge.utilities import clear_xspec, configure_xspec, get_xspec

# Access and configure XSPEC.
xspec = get_xspec()
configure_xspec()


def generate_synthetic_spectrum(
    output_path,
    model_generator,
    response,
    arf,
    *args,
    exposure=50_000,
    background="",
    correction="",
    backExposure="",
    overwrite=True,
    **kwargs,
):
    """
    Generate a synthetic spectrum from an XSPEC model using `fakeit`.

    Parameters
    ----------
    output_path : str or Path
        Path to save the generated `.pha` spectrum.
    model_generator : callable
        Function that constructs and returns the XSPEC model. This function
        should clear existing models if needed and fully define the model state.
    response : str
        Path to RMF file (energy redistribution).
    arf : str
        Path to ARF file (effective area curve).
    exposure : float, default=50000
        Exposure time in seconds.
    background : str, optional
        Path to background file.
    correction : str, optional
        Path to correction file.
    backExposure : float, optional
        Background exposure time in seconds.
    overwrite : bool, default=True
        Whether to overwrite existing file.
    args,kwargs:
        Additional inputs to the model fixture.

    Returns
    -------
    Path
        Path to the generated `.pha` file.

    Notes
    -----
    - Clears XSPEC state before generating synthetic data.
    - Assumes `model_generator` defines the full XSPEC model state.
    """
    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists and overwrite=False.")

    # Clear existing XSPEC state
    clear_xspec()

    # Build the model
    _ = model_generator(*args, **kwargs)

    # Generate synthetic spectrum
    fakeit = xspec.FakeitSettings(
        response=response,
        arf=arf,
        exposure=exposure,
        background=background,
        correction=correction,
        backExposure=backExposure,
        fileName=str(output_path),
    )
    xspec.AllData.fakeit(1, [fakeit])

    return output_path
