"""Koopman diagnostic plotters."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ._arrays import as_1d
from .base import BasePlotter
from .figure import FigureSpec
from .theme import PlotTheme


class KoopmanPlotter(BasePlotter):
    """Plot Koopman contribution diagnostics."""

    def __init__(
        self,
        theme: PlotTheme | str | None = None,
        *,
        figure: FigureSpec | None = None,
    ) -> None:
        super().__init__(theme=theme, figure=figure)

    def contribution(self, values: Any, *, feature_names: list[str] | None = None) -> Figure:
        """Plot contribution magnitudes as a bar chart."""

        contributions = as_1d(values)
        labels = feature_names or [str(idx) for idx in range(contributions.shape[0])]
        if len(labels) != contributions.shape[0]:
            raise ValueError(
                f"feature_names length must match contribution count. "
                f"Current lengths: {len(labels)} and {contributions.shape[0]}."
            )
        order = np.argsort(-np.abs(contributions))
        with self.theme.context(figure_size=self.figure.size):
            figure, ax = plt.subplots(figsize=self.figure.size, dpi=self.figure.dpi)
            ax.bar([labels[idx] for idx in order], contributions[order])
            ax.set_xlabel("Feature")
            ax.set_ylabel("Contribution")
            ax.set_title("Koopman Contributions")
            figure.autofmt_xdate(rotation=30, ha="right")
            figure.tight_layout()
        return figure
