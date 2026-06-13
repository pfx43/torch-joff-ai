"""Prediction plotters."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ._arrays import as_2d
from .base import BasePlotter
from .figure import FigureSpec
from .theme import PlotTheme


class PredictionPlotter(BasePlotter):
    """Plot predicted values against targets."""

    def __init__(
        self,
        theme: PlotTheme | str | None = None,
        *,
        figure: FigureSpec | None = None,
    ) -> None:
        super().__init__(theme=theme, figure=figure)

    def series(self, y_true: Any, y_pred: Any, *, target: int = 0) -> Figure:
        """Plot one target as true and predicted series."""

        true = as_2d(y_true)
        pred = as_2d(y_pred)
        _check_same_shape(true, pred)
        if target >= true.shape[1]:
            raise ValueError(
                f"target index must be < {true.shape[1]}. Current input: {target}."
            )
        with self.theme.context(figure_size=self.figure.size):
            figure, ax = plt.subplots(figsize=self.figure.size, dpi=self.figure.dpi)
            x = np.arange(true.shape[0])
            ax.plot(x, true[:, target], label="true", linewidth=1.5)
            ax.plot(x, pred[:, target], label="predicted", linewidth=1.5)
            ax.set_xlabel("Sample")
            ax.set_ylabel("Value")
            ax.set_title("Prediction Series")
            ax.legend()
            figure.tight_layout()
        return figure

    def scatter_true_pred(self, y_true: Any, y_pred: Any, *, target: int = 0) -> Figure:
        """Plot true-versus-predicted scatter for one target."""

        true = as_2d(y_true)
        pred = as_2d(y_pred)
        _check_same_shape(true, pred)
        if target >= true.shape[1]:
            raise ValueError(
                f"target index must be < {true.shape[1]}. Current input: {target}."
            )
        values = np.concatenate([true[:, target], pred[:, target]])
        low = float(np.min(values))
        high = float(np.max(values))
        with self.theme.context(figure_size=self.figure.size):
            figure, ax = plt.subplots(figsize=self.figure.size, dpi=self.figure.dpi)
            ax.scatter(true[:, target], pred[:, target], s=18, alpha=0.8)
            ax.plot([low, high], [low, high], color="tab:red", linewidth=1.0)
            ax.set_xlabel("True")
            ax.set_ylabel("Predicted")
            ax.set_title("True vs Predicted")
            figure.tight_layout()
        return figure


def _check_same_shape(true: np.ndarray, pred: np.ndarray) -> None:
    if true.shape != pred.shape:
        raise ValueError(f"y_true and y_pred must share shape. Current: {true.shape} and {pred.shape}.")
