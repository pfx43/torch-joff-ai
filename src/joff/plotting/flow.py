"""Flow diagnostic plotters."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ._arrays import as_2d
from .base import BasePlotter
from .figure import FigureSpec
from .theme import PlotTheme


class FlowPlotter(BasePlotter):
    """Plot latent-space diagnostics for flow models."""

    def __init__(
        self,
        theme: PlotTheme | str | None = None,
        *,
        figure: FigureSpec | None = None,
    ) -> None:
        super().__init__(theme=theme, figure=figure)

    def z_distribution(self, z: Any, *, bins: int = 24) -> Figure:
        """Plot flattened latent ``z`` distribution."""

        latent = as_2d(z).reshape(-1)
        with self.theme.context(figure_size=self.figure.size):
            figure, ax = plt.subplots(figsize=self.figure.size, dpi=self.figure.dpi)
            ax.hist(latent, bins=bins, color="tab:blue", alpha=0.8)
            ax.set_xlabel("z")
            ax.set_ylabel("Count")
            ax.set_title("Latent Distribution")
            figure.tight_layout()
        return figure
