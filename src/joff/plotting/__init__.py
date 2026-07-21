"""Plotting public API."""

from .base import BasePlotter, PlotResult
from .data import DataPlotter
from .fault_detection import FaultDetectionPlotter
from .figure import FigureFactory
from .figure import FigureSpec
from .flow import FlowPlotter
from .koopman import KoopmanPlotter
from .palettes import Palette, PaletteRegistry
from .prediction import PredictionPlotter
from .theme import PlotTheme
from .training import TrainingPlotter

__all__ = [
    "BasePlotter",
    "DataPlotter",
    "FaultDetectionPlotter",
    "FigureFactory",
    "FigureSpec",
    "FlowPlotter",
    "KoopmanPlotter",
    "Palette",
    "PaletteRegistry",
    "PredictionPlotter",
    "PlotResult",
    "PlotTheme",
    "TrainingPlotter",
]
