"""Base model protocol for joff models."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from warnings import warn

import torch
from torch import nn

from joff.core.config import ModelConfig

if TYPE_CHECKING:
    from joff.training import TrainingResult


LossOutput = torch.Tensor | dict[str, torch.Tensor | dict[str, torch.Tensor]]


class BaseModel(nn.Module):
    """Base class for joff PyTorch models."""

    model_type: str = "base"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

    def to_config(self) -> dict[str, Any]:
        """Return a serializable model configuration."""

        return self.config.model_dump(mode="json")

    def fit(
        self,
        data: Any,
        *,
        max_epochs: int = 1,
        device: str | torch.device = "auto",
        optimizer: dict[str, Any] | None = None,
        seed: int | None = None,
        checkpoint_dir: str | Path | None = None,
        monitor: str | None = None,
        mode: str = "min",
    ) -> "TrainingResult":
        """Deprecated compatibility wrapper around :class:`joff.Trainer`."""

        _warn_deprecated("model.fit")
        return self._fit_with_trainer(
            data,
            max_epochs=max_epochs,
            device=device,
            optimizer=optimizer,
            seed=seed,
            checkpoint_dir=checkpoint_dir,
            monitor=monitor,
            mode=mode,
        )

    def run(
        self,
        data: Any,
        *,
        max_epochs: int = 1,
        device: str | torch.device = "auto",
        optimizer: dict[str, Any] | None = None,
        seed: int | None = None,
        checkpoint_dir: str | Path | None = None,
        monitor: str | None = None,
        mode: str = "min",
    ) -> "TrainingResult":
        """Deprecated alias for :meth:`fit` that delegates to :class:`joff.Trainer`."""

        _warn_deprecated("model.run")
        return self._fit_with_trainer(
            data,
            max_epochs=max_epochs,
            device=device,
            optimizer=optimizer,
            seed=seed,
            checkpoint_dir=checkpoint_dir,
            monitor=monitor,
            mode=mode,
        )

    def test(
        self,
        data: Any,
        *,
        device: str | torch.device = "auto",
    ) -> dict[str, float]:
        """Deprecated compatibility wrapper around :meth:`joff.Trainer.evaluate`."""

        _warn_deprecated("model.test")
        from joff.training import Trainer

        return Trainer(max_epochs=0, device=device).evaluate(self, data)

    def load(
        self,
        checkpoint_path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> "BaseModel":
        """Deprecated state-dict checkpoint loader returning ``self``."""

        _warn_deprecated("model.load")
        checkpoint = torch.load(Path(checkpoint_path), map_location=map_location, weights_only=False)
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        self.load_state_dict(state_dict)
        return self

    def compute_loss(
        self,
        batch: Any,
        output: Any,
        loss_context: dict[str, Any] | None = None,
    ) -> LossOutput:
        """Compute a scalar loss for ``batch`` and ``output``."""

        raise NotImplementedError

    def _fit_with_trainer(
        self,
        data: Any,
        *,
        max_epochs: int,
        device: str | torch.device,
        optimizer: dict[str, Any] | None,
        seed: int | None,
        checkpoint_dir: str | Path | None,
        monitor: str | None,
        mode: str,
    ) -> "TrainingResult":
        from joff.training import Trainer

        trainer = Trainer(
            max_epochs=max_epochs,
            device=device,
            optimizer=optimizer,
            seed=seed,
            checkpoint_dir=checkpoint_dir,
            monitor=monitor,
            mode=mode,
        )
        return trainer.fit(self, data)


def _warn_deprecated(name: str) -> None:
    warn(
        f"{name} is deprecated and kept only as a compatibility wrapper. "
        "Use Trainer, Experiment, or Study for new code.",
        DeprecationWarning,
        stacklevel=3,
    )


def batch_inputs(batch: Any) -> torch.Tensor:
    """Extract model inputs from a tensor, tuple/list, or mapping batch."""

    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, dict):
        for key in ("x", "input", "inputs", "features", "history", "past"):
            if key in batch:
                return batch[key]
    if isinstance(batch, (tuple, list)) and batch:
        return batch[0]
    raise TypeError(
        "Cannot extract inputs from batch. Legal batch forms are Tensor, non-empty tuple/list, "
        "or dict containing one of: x, input, inputs, features, history, past."
    )


def batch_targets(batch: Any, *, fallback: torch.Tensor | None = None) -> torch.Tensor:
    """Extract supervised targets from a batch, optionally falling back to inputs."""

    if isinstance(batch, dict):
        for key in ("y", "target", "targets", "target_future", "future", "label", "labels"):
            if key in batch:
                return batch[key]
    if isinstance(batch, (tuple, list)) and len(batch) > 1:
        return batch[1]
    if fallback is not None:
        return fallback
    return batch_inputs(batch)


def regression_loss(output: torch.Tensor, target: torch.Tensor, loss: str) -> torch.Tensor:
    """Compute a supported regression loss."""

    if loss == "mse":
        return torch.nn.functional.mse_loss(output, target)
    if loss == "mae":
        return torch.nn.functional.l1_loss(output, target)
    if loss == "smooth_l1":
        return torch.nn.functional.smooth_l1_loss(output, target)
    raise ValueError(
        f"Unknown loss {loss!r}. Legal options are: mse, mae, smooth_l1. Current input: {loss!r}."
    )
