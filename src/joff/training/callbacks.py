"""Small callback protocol for trainer lifecycle hooks."""

from __future__ import annotations

from typing import Any, Protocol


class Callback(Protocol):
    """Trainer callback protocol."""

    def on_fit_start(self, context: dict[str, Any]) -> None:
        """Called before the first epoch."""

    def on_epoch_end(self, context: dict[str, Any]) -> None:
        """Called after each epoch."""

    def on_fit_end(self, context: dict[str, Any]) -> None:
        """Called after training finishes."""


class NoOpCallback:
    """Callback with empty lifecycle hooks."""

    def on_fit_start(self, context: dict[str, Any]) -> None:
        """Do nothing."""

    def on_epoch_end(self, context: dict[str, Any]) -> None:
        """Do nothing."""

    def on_fit_end(self, context: dict[str, Any]) -> None:
        """Do nothing."""


class HistoryCallback(NoOpCallback):
    """Collect epoch rows observed by the trainer."""

    def __init__(self) -> None:
        self.history: list[dict[str, float]] = []

    def on_epoch_end(self, context: dict[str, Any]) -> None:
        """Append the current epoch row if present."""

        row = context.get("epoch_metrics")
        if isinstance(row, dict):
            self.history.append(dict(row))

