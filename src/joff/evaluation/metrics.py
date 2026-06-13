"""Regression and reconstruction evaluators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class MetricReport:
    """Structured metric report with overall and optional per-target metrics."""

    overall: dict[str, float]
    per_target: list[dict[str, float | int]]

    def to_flat_dict(self, *, prefix: str = "") -> dict[str, float]:
        """Return overall metrics as a flat dictionary."""

        return {f"{prefix}{key.lower()}": value for key, value in self.overall.items()}

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable nested representation."""

        return {"overall": self.overall, "per_target": self.per_target}


class RegressionEvaluator:
    """Compute safe regression metrics for prediction tasks."""

    def evaluate(self, y_true: Any, y_pred: Any) -> MetricReport:
        """Evaluate predictions against targets."""

        target = _as_2d(y_true)
        prediction = _as_2d(y_pred)
        if target.shape != prediction.shape:
            raise ValueError(
                f"y_true and y_pred must have the same shape. Current shapes: "
                f"{target.shape} and {prediction.shape}."
            )
        errors = prediction - target
        overall = {
            "MSE": float(np.mean(errors**2)),
            "RMSE": float(np.sqrt(np.mean(errors**2))),
            "MAE": float(np.mean(np.abs(errors))),
            "R2": _safe_r2(target, prediction),
        }
        per_target = []
        for idx in range(target.shape[1]):
            target_i = target[:, idx]
            pred_i = prediction[:, idx]
            err_i = pred_i - target_i
            per_target.append(
                {
                    "Target": idx,
                    "MSE": float(np.mean(err_i**2)),
                    "RMSE": float(np.sqrt(np.mean(err_i**2))),
                    "MAE": float(np.mean(np.abs(err_i))),
                    "R2": _safe_r2(target_i[:, None], pred_i[:, None]),
                }
            )
        return MetricReport(overall=overall, per_target=per_target)


class ReconstructionEvaluator(RegressionEvaluator):
    """Compute reconstruction metrics for AE/VAE/NICE-like outputs."""

    def evaluate(self, y_true: Any, y_pred: Any) -> MetricReport:
        """Evaluate reconstructions against original inputs."""

        report = super().evaluate(y_true, y_pred)
        target = _as_2d(y_true)
        prediction = _as_2d(y_pred)
        max_abs = float(np.max(np.abs(prediction - target))) if target.size else float("nan")
        overall = dict(report.overall)
        overall["MaxAbs"] = max_abs
        return MetricReport(overall=overall, per_target=report.per_target)


def _safe_r2(target: np.ndarray, prediction: np.ndarray) -> float:
    ss_res = float(np.sum((target - prediction) ** 2))
    centered = target - np.mean(target, axis=0, keepdims=True)
    ss_tot = float(np.sum(centered**2))
    if ss_tot <= 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _as_2d(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim == 2:
        return array
    if array.ndim >= 3:
        return array.reshape(array.shape[0], -1)
    raise ValueError(f"Expected at least 1D metric input. Current shape: {array.shape}.")

