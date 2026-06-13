"""Training history plots."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .base import BasePlotter
from .figure import FigureSpec
from .theme import PlotTheme


class TrainingPlotter(BasePlotter):
    """Plot trainer history outputs."""

    def __init__(
        self,
        theme: PlotTheme | str | None = None,
        *,
        figure: FigureSpec | None = None,
    ) -> None:
        super().__init__(theme=theme, figure=figure)

    def loss_curve(self, history: list[dict[str, Any]]) -> Figure:
        """Return a figure containing the training loss curve."""

        with self.theme.context(figure_size=self.figure.size):
            figure, ax = plt.subplots(figsize=self.figure.size, dpi=self.figure.dpi)
            epochs = [row.get("epoch", idx) for idx, row in enumerate(history)]
            losses = [row["train/loss"] for row in history]
            ax.plot(epochs, losses, marker="o", linewidth=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Training Loss")
            figure.tight_layout()
        return figure
