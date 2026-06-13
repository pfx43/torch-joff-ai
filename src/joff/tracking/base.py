"""Experiment tracking protocols."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from matplotlib.figure import Figure


@dataclass(frozen=True)
class RunInfo:
    """Metadata describing one tracked run."""

    run_id: str
    name: str | None = None
    tags: dict[str, str] | None = None


class Tracker(Protocol):
    """Protocol shared by local and optional external trackers."""

    def start_run(self, run_info: RunInfo) -> None:
        """Start a tracked run."""

    def log_config(self, config: Any) -> None:
        """Log resolved configuration."""

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Log one scalar metric."""

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log multiple scalar metrics."""

    def log_artifact(self, path: str | Path, name: str | None = None) -> None:
        """Log an artifact path."""

    def log_figure(self, figure: Figure, name: str) -> None:
        """Log a figure artifact."""

    def end_run(self, status: str = "finished") -> None:
        """End a tracked run."""


class CompositeTracker:
    """Fan out tracking calls to multiple trackers."""

    def __init__(self, trackers: list[Tracker]) -> None:
        self.trackers = trackers

    def start_run(self, run_info: RunInfo) -> None:
        """Start all trackers."""

        for tracker in self.trackers:
            tracker.start_run(run_info)

    def log_config(self, config: Any) -> None:
        """Log config to all trackers."""

        for tracker in self.trackers:
            tracker.log_config(config)

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Log one metric to all trackers."""

        for tracker in self.trackers:
            tracker.log_metric(name, value, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to all trackers."""

        for tracker in self.trackers:
            tracker.log_metrics(metrics, step=step)

    def log_artifact(self, path: str | Path, name: str | None = None) -> None:
        """Log an artifact to all trackers."""

        for tracker in self.trackers:
            tracker.log_artifact(path, name=name)

    def log_figure(self, figure: Figure, name: str) -> None:
        """Log a figure to all trackers."""

        for tracker in self.trackers:
            tracker.log_figure(figure, name)

    def end_run(self, status: str = "finished") -> None:
        """End all trackers."""

        for tracker in self.trackers:
            tracker.end_run(status=status)
