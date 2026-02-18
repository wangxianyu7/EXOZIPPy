"""Plotting package for EXOZIPPy traditional fitting workflows."""

from .mcmc import chain_to_arviz, plot_corner, plot_trace
from .rv import plotrv
from .sed import plotsed
from .transit import plottran

# Optional new-style aliases
plot_transit = plottran
plot_rv = plotrv
plot_sed = plotsed

__all__ = [
    "plottran",
    "plotrv",
    "plotsed",
    "plot_corner",
    "plot_trace",
    "chain_to_arviz",
    "plot_transit",
    "plot_rv",
    "plot_sed",
]
