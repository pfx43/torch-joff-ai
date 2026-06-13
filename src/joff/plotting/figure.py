"""Figure factory helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .theme import PlotTheme, resolve_theme


@dataclass(frozen=True)
class FigureSpec:
    """Physical figure size and DPI for reproducible saved plots."""

    width: float
    height: float
    dpi: int = 300

    @property
    def size(self) -> tuple[float, float]:
        """Return matplotlib ``figsize``."""

        return (self.width, self.height)

    @classmethod
    def paper_single(cls) -> "FigureSpec":
        """Return a single-column paper figure size."""

        return cls(3.45, 2.15)

    @classmethod
    def paper_double(cls) -> "FigureSpec":
        """Return a double-column paper figure size."""

        return cls(7.16, 2.35)

    @classmethod
    def paper_short(cls) -> "FigureSpec":
        """Return a wide, short paper figure size."""

        return cls(7.16, 1.80)

    @classmethod
    def paper_tall(cls) -> "FigureSpec":
        """Return a wide, taller paper figure size."""

        return cls(7.16, 3.20)

    @classmethod
    def paper_double_short(cls) -> "FigureSpec":
        """Return the default wide-and-narrow paper figure size."""

        return cls.paper_short()

    @classmethod
    def presentation(cls) -> "FigureSpec":
        """Return a presentation-friendly widescreen figure size."""

        return cls(10.0, 5.6)


class FigureFactory:
    """Create matplotlib figures under a temporary plot theme."""

    def __init__(
        self,
        theme: PlotTheme | str | None = None,
        *,
        spec: FigureSpec | None = None,
    ) -> None:
        self.theme = resolve_theme(theme)
        self.spec = spec or FigureSpec(
            self.theme.figure_size[0],
            self.theme.figure_size[1],
        )

    def figure(self) -> Figure:
        """Create a new themed figure."""

        with self.theme.context(figure_size=self.spec.size):
            figure = plt.figure(figsize=self.spec.size, dpi=self.spec.dpi)
        return figure

    def subplots(self, **kwargs: Any) -> tuple[Figure, Axes]:
        """Create themed matplotlib subplots."""

        with self.theme.context(figure_size=self.spec.size):
            figure, ax = plt.subplots(figsize=self.spec.size, dpi=self.spec.dpi, **kwargs)
        return figure, ax
