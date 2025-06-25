"""
XCalForge tools for interacting with and modifying spectra.
"""
__all__ = ["group_min_counts", "generate_synthetic_spectrum"]

from .genspec import generate_synthetic_spectrum
from .specutils import group_min_counts
