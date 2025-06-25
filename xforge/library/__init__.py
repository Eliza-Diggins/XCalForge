"""
Library generation module for SpecForge.

The Library module manages the generation of synthetic modification libraries
which are then used for modeling temperature differences between telescopes.

In SpecForge, a "Library" is constructed to span some parameter space of possible modifications
to a telescope's configuration, the fitting model, etc. Each library parameterizes the set of possible
modifications into a grid of parameter values and then (for each parameter value) builds synthetic data
and fits with both the original and modified scenarios. This is done over a range of temperatures to
produce a discretized library of the effects of modifying the configuration.
"""
from .arf_mod import GaussianARFLibrary
