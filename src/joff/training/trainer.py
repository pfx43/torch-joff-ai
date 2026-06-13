"""Minimal PyTorch trainer for smoke experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from joff.core.device import resolve_device
from joff.core.seed import seed_everything
from joff.evaluation import ReconstructionEvaluator, RegressionEvaluator
from joff.training.callbacks import Callback
from joff.training.checkpoint import CheckpointManager
from joff.training.optim import build_optimizer


@dataclass(frozen=True)
class TrainingResult:
    """History returned by :meth:`Trainer.fit`."""

    history: list[dict[str, float]]
    checkpoint_paths: dict[str, Path]


class Trainer:
    """Small training loop that keeps model logic separate from data and artifacts."""

    def __init__(
        self,
        max_epochs: int = 1,
        *,
        device: str | torch.device = "auto",
        optimizer: dict[str, Any] | None = None,
        seed: int | None = None,
        monitor: str | None = None,
        mode: str = "min",
        checkpoint_dir: str | Path | None = None,
        save_last: bool = True,
        save_best: bool = True,
        callbacks: list[Callback] | None = None,
        checkpoint_config: dict[str, Any] | None = None,
        resolved_config: dict[str, Any] | None = None,
        checkpoint_extra_state: dict[str, Any] | None = None,
    ) -> None:
        self.max_epochs = max_epochs
        self.device = resolve_device(device)
        self.optimizer_config = optimizer or {"type": "adam", "lr": 1e-3, "weight_decay": 0.0}
        self.seed = seed
        self.monitor = monitor
        self.mode = mode
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.save_last = save_last
        self.save_best = save_best
        self.callbacks = callbacks or []
        self.checkpoint_config = checkpoint_config
        self.resolved_config = resolved_config
        self.checkpoint_extra_state = checkpoint_extra_state

    def fit(self, model: nn.Module, data: Any) -> TrainingResult:
        """Train ``model`` on ``data`` and return epoch history."""

        if self.seed is not None:
            seed_everything(self.seed)
        model.to(self.device)
        optimizer = build_optimizer(model, self.optimizer_config)
        checkpoint_manager = self._checkpoint_manager()
        history: list[dict[str, float]] = []
        checkpoint_paths: dict[str, Path] = {}
        train_loader = data.train_dataloader()
        context: dict[str, Any] = {"model": model, "trainer": self, "optimizer": optimizer}
        for callback in self.callbacks:
            callback.on_fit_start(context)
        for epoch in range(self.max_epochs):
            model.train()
            total_loss = 0.0
            count = 0
            component_totals: dict[str, float] = {}
            for batch_idx, batch in enumerate(train_loader):
                batch = _move_to_device(batch, self.device)
                optimizer.zero_grad(set_to_none=True)
                if hasattr(model, "training_step"):
                    step = model.training_step(batch, batch_idx)
                    loss_info = _normalize_loss_output(step)
                else:
                    output = model(batch)
                    loss_info = _normalize_loss_output(_compute_loss(model, batch, output))
                loss = loss_info["loss"]
                loss.backward()
                optimizer.step()
                batch_size = _batch_size(batch)
                total_loss += float(loss.detach().cpu()) * batch_size
                for name, component in loss_info["losses"].items():
                    component_totals[name] = (
                        component_totals.get(name, 0.0) + float(component.detach().cpu()) * batch_size
                    )
                count += batch_size
            row = {"epoch": float(epoch), "train/loss": total_loss / max(count, 1)}
            for name, value in component_totals.items():
                row[f"train/{name}_loss"] = value / max(count, 1)
            test_loader_factory = getattr(data, "test_dataloader", None)
            if callable(test_loader_factory):
                test_loader = test_loader_factory()
                if test_loader is not None:
                    test_metrics = _evaluate_loader(model, test_loader, self.device)
                    row.update({f"test/{key}": value for key, value in test_metrics.items()})
            history.append(row)
            context.update({"epoch": epoch, "epoch_metrics": row})
            if checkpoint_manager is not None:
                saved = checkpoint_manager.save_epoch(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=row,
                    config=self.checkpoint_config,
                    resolved_config=self.resolved_config,
                    extra_state=self.checkpoint_extra_state,
                )
                if saved.last_path is not None:
                    checkpoint_paths["last"] = saved.last_path
                if saved.best_path is not None:
                    checkpoint_paths["best"] = saved.best_path
            for callback in self.callbacks:
                callback.on_epoch_end(context)
        context.update({"history": history, "checkpoint_paths": checkpoint_paths})
        for callback in self.callbacks:
            callback.on_fit_end(context)
        return TrainingResult(history=history, checkpoint_paths=checkpoint_paths)

    @torch.no_grad()
    def evaluate(self, model: nn.Module, data: Any) -> dict[str, float]:
        """Evaluate ``model`` on test data if present, otherwise train data."""

        model.to(self.device)
        model.eval()
        loader = data.test_dataloader() or data.train_dataloader()
        return _evaluate_loader(model, loader, self.device)

    def _checkpoint_manager(self) -> CheckpointManager | None:
        if self.checkpoint_dir is None:
            return None
        return CheckpointManager(
            self.checkpoint_dir,
            monitor=self.monitor or "train/loss",
            mode=self.mode,
            save_last=self.save_last,
            save_best=self.save_best,
        )


@torch.no_grad()
def _evaluate_loader(model: nn.Module, loader: Any, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    count = 0
    component_totals: dict[str, float] = {}
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    reconstruction_mode = False
    for batch in loader:
        batch = _move_to_device(batch, device)
        output = model(batch)
        loss_info = _normalize_loss_output(_compute_loss(model, batch, output))
        loss = loss_info["loss"]
        prediction, target, is_reconstruction = _prediction_and_target(batch, output)
        predictions.append(prediction.detach().cpu())
        targets.append(target.detach().cpu())
        reconstruction_mode = reconstruction_mode or is_reconstruction
        batch_size = _batch_size(batch)
        total_loss += float(loss.detach().cpu()) * batch_size
        for name, component in loss_info["losses"].items():
            component_totals[name] = (
                component_totals.get(name, 0.0) + float(component.detach().cpu()) * batch_size
            )
        count += batch_size
    metrics = {"loss": total_loss / max(count, 1)}
    for name, value in component_totals.items():
        metrics[f"{name}_loss"] = value / max(count, 1)
    if predictions and targets:
        y_pred = torch.cat(predictions, dim=0)
        y_true = torch.cat(targets, dim=0)
        evaluator = ReconstructionEvaluator() if reconstruction_mode else RegressionEvaluator()
        report = evaluator.evaluate(y_true, y_pred)
        metrics.update(report.to_flat_dict())
    return metrics


def _compute_loss(model: nn.Module, batch: Any, output: Any) -> torch.Tensor:
    if hasattr(model, "compute_loss"):
        return model.compute_loss(batch, output)
    target = batch[1] if isinstance(batch, (tuple, list)) and len(batch) > 1 else batch[0]
    if isinstance(output, dict) and "reconstruction" in output:
        output = output["reconstruction"]
    return F.mse_loss(output, target)


def _prediction_and_target(batch: Any, output: Any) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if isinstance(output, dict):
        if "prediction" in output:
            prediction = output["prediction"]
            target = _target_from_batch(batch, fallback=_input_from_batch(batch))
            return prediction, _align_target(target, prediction), False
        if "reconstruction" in output:
            prediction = output["reconstruction"]
            target = _input_from_batch(batch)
            return prediction, _align_target(target, prediction), True
    prediction = output
    target = _target_from_batch(batch, fallback=_input_from_batch(batch))
    return prediction, _align_target(target, prediction), False


def _input_from_batch(batch: Any) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, dict):
        for key in ("x", "input", "inputs", "features", "history", "past"):
            if key in batch:
                return batch[key]
    if isinstance(batch, (tuple, list)) and batch:
        return batch[0]
    raise TypeError("Cannot extract input tensor from batch.")


def _target_from_batch(batch: Any, *, fallback: torch.Tensor) -> torch.Tensor:
    if isinstance(batch, dict):
        for key in ("y", "target", "targets", "target_future", "future", "label", "labels"):
            if key in batch:
                return batch[key]
    if isinstance(batch, (tuple, list)) and len(batch) > 1:
        return batch[1]
    return fallback


def _align_target(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    if target.shape == prediction.shape:
        return target
    if prediction.ndim == 3 and target.ndim == 2 and target.shape[0] == prediction.shape[0]:
        return target.unsqueeze(1).expand(-1, prediction.shape[1], -1)
    return target


def _normalize_loss_output(value: Any) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        return {"loss": value, "losses": {}}
    if isinstance(value, dict):
        if "loss" not in value:
            raise ValueError("Loss dict must contain key 'loss'. Legal keys include 'loss' and 'losses'.")
        loss = value["loss"]
        if not isinstance(loss, torch.Tensor):
            raise TypeError(f"Loss value must be a torch.Tensor. Current input: {type(loss).__name__}.")
        raw_losses = value.get("losses", {})
        if raw_losses is None:
            raw_losses = {}
        if not isinstance(raw_losses, dict):
            raise TypeError(
                f"Loss components must be a mapping. Current input: {type(raw_losses).__name__}."
            )
        losses: dict[str, torch.Tensor] = {}
        for name, component in raw_losses.items():
            if isinstance(component, torch.Tensor):
                losses[str(name)] = component
        return {"loss": loss, "losses": losses}
    if hasattr(value, "loss"):
        return _normalize_loss_output({"loss": value.loss, "losses": getattr(value, "losses", {})})
    raise TypeError(
        f"Unsupported loss output {type(value).__name__}. Legal options are Tensor, dict, "
        "or object with a loss attribute."
    )


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _batch_size(batch: Any) -> int:
    if isinstance(batch, torch.Tensor):
        return int(batch.shape[0])
    if isinstance(batch, (tuple, list)) and batch:
        return _batch_size(batch[0])
    if isinstance(batch, dict):
        first = next(iter(batch.values()))
        return _batch_size(first)
    return 1
