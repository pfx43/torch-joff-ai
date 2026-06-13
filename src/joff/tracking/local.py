"""Local file-backed tracker."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from matplotlib.figure import Figure

from joff.artifacts import ArtifactStore, RunLogger

from .base import RunInfo


class LocalTracker:
    """Track configs, metrics, artifacts, and figures under an ArtifactStore."""

    def __init__(self, store: ArtifactStore | str | Path) -> None:
        self.store = store if isinstance(store, ArtifactStore) else ArtifactStore(store, "tracking")
        self.logger = RunLogger(self.store.resolve("tracking/events.jsonl"))
        self.run_info: RunInfo | None = None

    def start_run(self, run_info: RunInfo) -> None:
        """Start a local tracked run."""

        self.run_info = run_info
        self.logger.log_event(
            "run_start",
            run_id=run_info.run_id,
            name=run_info.name,
            tags=run_info.tags or {},
        )

    def log_config(self, config: Any) -> None:
        """Save config and log a config event."""

        path = self.store.save_json("tracking/config.json", _json_ready(config))
        self.logger.log_event("config", path=path)

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Log one local scalar metric."""

        self.logger.log_event("metric", name=name, value=float(value), step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log multiple local scalar metrics."""

        for name, value in metrics.items():
            self.log_metric(name, value, step=step)

    def log_artifact(self, path: str | Path, name: str | None = None) -> None:
        """Log an artifact path without copying it."""

        self.logger.log_event("artifact", path=Path(path), name=name)

    def log_figure(self, figure: Figure, name: str) -> None:
        """Save and log a figure artifact."""

        path = self.store.save_figure(f"tracking/figures/{name}.png", figure)
        self.logger.log_event("figure", name=name, path=path)

    def end_run(self, status: str = "finished") -> None:
        """End a local tracked run."""

        self.logger.log_event(
            "run_end",
            run_id=None if self.run_info is None else self.run_info.run_id,
            status=status,
        )


def _json_ready(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value
