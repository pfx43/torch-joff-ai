"""Fault-detection plotters."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ._arrays import as_1d
from .base import BasePlotter
from .figure import FigureSpec
from .theme import PlotTheme


class FaultDetectionPlotter(BasePlotter):
    """Plot fault-detection statistics and metrics."""

    def __init__(
        self,
        theme: PlotTheme | str | None = None,
        *,
        figure: FigureSpec | None = None,
    ) -> None:
        super().__init__(theme=theme, figure=figure)

    def stat_curve(
        self,
        scores: Any,
        *,
        threshold: float | None = None,
        labels: Any | None = None,
    ) -> Figure:
        """Plot anomaly scores with optional threshold and fault region."""

        score_array = as_1d(scores)
        label_array = None if labels is None else as_1d(labels).astype(int)
        if label_array is not None and label_array.shape[0] != score_array.shape[0]:
            raise ValueError(
                f"labels must match scores length. Current lengths: "
                f"{label_array.shape[0]} and {score_array.shape[0]}."
            )
        with self.theme.context(figure_size=self.figure.size):
            figure, ax = plt.subplots(figsize=self.figure.size, dpi=self.figure.dpi)
            indices = np.arange(score_array.shape[0])
            ax.plot(indices, score_array, linewidth=1.5)
            if threshold is not None:
                ax.axhline(float(threshold), color="tab:red", linewidth=1.2)
            if label_array is not None and np.any(label_array == 1):
                first_fault = int(np.argmax(label_array == 1))
                ax.axvspan(first_fault, score_array.shape[0] - 1, color="tab:red", alpha=0.08)
            ax.set_xlabel("Sample")
            ax.set_ylabel("Score")
            ax.set_title("Fault Detection Scores")
            figure.tight_layout()
        return figure

    def far_mdr_bar(self, metrics: dict[str, float]) -> Figure:
        """Plot FAR/MDR/FDR metric bars when available."""

        names = [name for name in ("FAR", "MDR", "FDR") if name in metrics]
        if not names:
            raise ValueError(
                f"metrics must contain at least one of FAR, MDR, FDR. Current keys: {sorted(metrics)}."
            )
        values = [float(metrics[name]) for name in names]
        with self.theme.context(figure_size=self.figure.size):
            figure, ax = plt.subplots(figsize=self.figure.size, dpi=self.figure.dpi)
            ax.bar(names, values, color=["tab:blue", "tab:orange", "tab:green"][: len(names)])
            ax.set_ylim(0.0, max(1.0, max(values)))
            ax.set_ylabel("Rate")
            ax.set_title("Fault Detection Metrics")
            figure.tight_layout()
        return figure
