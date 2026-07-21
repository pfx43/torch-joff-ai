"""Base plotting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from matplotlib.figure import Figure

from joff.artifacts import ArtifactStore

from .figure import FigureSpec
from .theme import PlotTheme, resolve_theme


@dataclass(frozen=True)
class PlotResult:
    """Figure result returned by plotters."""

    figure: Figure


class BasePlotter:
    """Base class for plotters that share theme, figure spec, and saving."""

    def __init__(
        self,
        theme: PlotTheme | str | None = None,
        *,
        figure: FigureSpec | None = None,
    ) -> None:
        self.theme = resolve_theme(theme)
        self.figure = figure or FigureSpec(
            self.theme.figure_size[0],
            self.theme.figure_size[1],
        )

    def save(
        self,
        figure: Figure,
        store: ArtifactStore,
        name: str,
        *,
        formats: tuple[str, ...] = ("pdf", "svg", "png"),
    ) -> list[Path]:
        """Save ``figure`` through an artifact store in one or more formats."""

        paths = store.save_figure(
            f"plots/{name}",
            figure,
            dpi=self.figure.dpi,
            formats=formats,
        )
        return paths if isinstance(paths, list) else [paths]
