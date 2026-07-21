"""Data diagnostic plotters."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ._arrays import as_1d
from .base import BasePlotter
from .figure import FigureSpec
from .theme import PlotTheme


class DataPlotter(BasePlotter):
    """Plot data split and outlier diagnostics."""

    def __init__(
        self,
        theme: PlotTheme | str | None = None,
        *,
        figure: FigureSpec | None = None,
    ) -> None:
        super().__init__(theme=theme, figure=figure)

    def split_distribution(self, split_labels: Any) -> Figure:
        """Plot counts for train/eval/test split labels."""

        labels = np.asarray(split_labels).reshape(-1)
        unique, counts = np.unique(labels, return_counts=True)
        with self.theme.context(figure_size=self.figure.size):
            figure, ax = plt.subplots(figsize=self.figure.size, dpi=self.figure.dpi)
            ax.bar([str(item) for item in unique], counts)
            ax.set_xlabel("Split")
            ax.set_ylabel("Samples")
            ax.set_title("Split Distribution")
            figure.tight_layout()
        return figure

    def outlier_marks(self, values: Any, outlier_indices: Any) -> Figure:
        """Plot a series with marked outlier indices."""

        series = as_1d(values)
        indices = np.asarray(outlier_indices, dtype=int).reshape(-1)
        valid = indices[(indices >= 0) & (indices < series.shape[0])]
        with self.theme.context(figure_size=self.figure.size):
            figure, ax = plt.subplots(figsize=self.figure.size, dpi=self.figure.dpi)
            x = np.arange(series.shape[0])
            ax.plot(x, series, linewidth=1.2)
            if valid.size:
                ax.scatter(valid, series[valid], color="tab:red", s=22, zorder=3)
            ax.set_xlabel("Sample")
            ax.set_ylabel("Value")
            ax.set_title("Outlier Marks")
            figure.tight_layout()
        return figure
