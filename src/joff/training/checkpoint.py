"""Checkpoint manager that saves state_dict-based checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datetime import datetime, timezone

import random
import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class CheckpointSaveResult:
    """Paths saved or updated by a checkpoint step."""

    last_path: Path | None
    best_path: Path | None
    best_updated: bool


class CheckpointManager:
    """Save and restore model/optimizer state without serializing whole model objects."""

    def __init__(
        self,
        root: str | Path,
        *,
        monitor: str = "train/loss",
        mode: str = "min",
        save_last: bool = True,
        save_best: bool = True,
    ) -> None:
        self.root = Path(root)
        self.monitor = monitor
        self.mode = mode
        self.save_last_enabled = save_last
        self.save_best_enabled = save_best
        self.best_value: float | None = None
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def last_path(self) -> Path:
        """Path to the latest checkpoint."""

        return self.root / "last.pt"

    @property
    def best_path(self) -> Path:
        """Path to the best checkpoint."""

        return self.root / "best.pt"

    def save_epoch(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        epoch: int,
        metrics: dict[str, float],
        config: dict[str, Any] | None = None,
        resolved_config: dict[str, Any] | None = None,
        extra_state: dict[str, Any] | None = None,
    ) -> CheckpointSaveResult:
        """Save last and maybe best checkpoint for one epoch."""

        record = self._record(model, optimizer, epoch, metrics, config, resolved_config, extra_state)
        last_path = self.save(record, self.last_path) if self.save_last_enabled else None
        best_updated = False
        best_path: Path | None = None
        value = _metric_value(metrics, self.monitor)
        if self.save_best_enabled and value is not None and self._is_better(value):
            self.best_value = value
            best_path = self.save(record, self.best_path)
            best_updated = True
        return CheckpointSaveResult(last_path=last_path, best_path=best_path, best_updated=best_updated)

    def save(self, record: dict[str, Any], path: str | Path) -> Path:
        """Save a checkpoint record to ``path``."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(record, output)
        return output

    def load(
        self,
        path: str | Path,
        *,
        model: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Load a checkpoint and optionally restore model/optimizer states."""

        checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
        if model is not None:
            model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint

    def load_best(
        self,
        *,
        model: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Load ``best.pt`` and optionally restore model/optimizer states."""

        return self.load(
            self.best_path,
            model=model,
            optimizer=optimizer,
            map_location=map_location,
        )

    def load_last(
        self,
        *,
        model: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Load ``last.pt`` and optionally restore model/optimizer states."""

        return self.load(
            self.last_path,
            model=model,
            optimizer=optimizer,
            map_location=map_location,
        )

    def _record(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        epoch: int,
        metrics: dict[str, float],
        config: dict[str, Any] | None,
        resolved_config: dict[str, Any] | None,
        extra_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        model_config = config or _model_config(model)
        full_config = resolved_config or model_config
        return {
            "model_class": model.__class__.__name__,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "metrics": dict(metrics),
            "config": model_config,
            "resolved_config": full_config,
            "rng_state": _rng_state(),
            "extra_state": dict(extra_state or {}),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return value < self.best_value
        if self.mode == "max":
            return value > self.best_value
        raise ValueError(f"Unknown checkpoint mode {self.mode!r}. Legal options are: min, max.")


def _metric_value(metrics: dict[str, float], key: str) -> float | None:
    value = metrics.get(key)
    if value is None and key == "train/loss":
        value = metrics.get("loss")
    if value is None:
        return None
    return float(value)


def _model_config(model: nn.Module) -> dict[str, Any] | None:
    to_config = getattr(model, "to_config", None)
    if callable(to_config):
        return to_config()
    return None


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state
