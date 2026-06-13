"""Plot themes that avoid import-time global matplotlib changes."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import matplotlib.pyplot as plt
from cycler import cycler

from .palettes import Palette, PaletteRegistry


class PlotTheme:
    """Temporary matplotlib rcParams theme."""

    def __init__(
        self,
        *,
        font_size: int = 11,
        figure_size: tuple[float, float] = (5.0, 3.0),
        font_family: str | None = None,
        palette: Palette | str | None = None,
        grid: bool = True,
    ) -> None:
        self.font_size = font_size
        self.figure_size = figure_size
        self.font_family = font_family
        self.palette = PaletteRegistry().get(palette) if isinstance(palette, str) else palette
        self.grid = grid

    @classmethod
    def paper(
        cls,
        *,
        font_size: int = 13,
        font_family: str | None = None,
        palette: Palette | str | None = None,
    ) -> "PlotTheme":
        """Return a large-font paper theme."""

        return cls(
            font_size=font_size,
            figure_size=(7.16, 2.35),
            font_family=font_family,
            palette=palette,
        )

    @classmethod
    def paper_compact(
        cls,
        *,
        font_family: str | None = None,
        palette: Palette | str | None = None,
    ) -> "PlotTheme":
        """Return a compact paper theme for dense figures."""

        return cls.paper(font_size=12, font_family=font_family, palette=palette)

    @classmethod
    def paper_ieee(
        cls,
        *,
        font_family: str | None = None,
        palette: Palette | str | None = None,
    ) -> "PlotTheme":
        """Return an IEEE-sized paper theme without optional dependencies."""

        return cls(font_size=10, figure_size=(3.45, 2.15), font_family=font_family, palette=palette)

    @classmethod
    def from_name(cls, name: str) -> "PlotTheme":
        """Resolve a built-in theme by name."""

        normalized = name.strip().lower()
        if normalized in {"default", "standard"}:
            return cls()
        if normalized == "paper":
            return cls.paper()
        if normalized == "paper_compact":
            return cls.paper_compact()
        if normalized == "paper_ieee":
            return cls.paper_ieee()
        raise ValueError(
            f"Unknown plot theme {name!r}. Legal options are: default, paper, "
            "paper_compact, paper_ieee."
        )

    @contextmanager
    def context(self, *, figure_size: tuple[float, float] | None = None) -> Iterator[None]:
        """Apply rcParams only inside this context."""

        rc = {
            "font.size": self.font_size,
            "axes.labelsize": self.font_size,
            "xtick.labelsize": max(self.font_size - 1, 1),
            "ytick.labelsize": max(self.font_size - 1, 1),
            "legend.fontsize": max(self.font_size - 1, 1),
            "figure.figsize": figure_size or self.figure_size,
            "axes.grid": self.grid,
            "grid.alpha": 0.25,
        }
        if self.font_family is not None:
            rc["font.family"] = self.font_family
        if self.palette is not None:
            rc["axes.prop_cycle"] = cycler(color=list(self.palette.colors))
        with plt.rc_context(rc):
            yield


def resolve_theme(theme: PlotTheme | str | None) -> PlotTheme:
    """Resolve a theme object, built-in theme name, or ``None``."""

    if theme is None:
        return PlotTheme()
    if isinstance(theme, PlotTheme):
        return theme
    return PlotTheme.from_name(theme)
