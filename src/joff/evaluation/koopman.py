"""Koopman contribution evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class KoopmanContributionReport:
    """Koopman contribution metrics and per-dimension contributions."""

    overall: dict[str, float]
    per_dimension: list[dict[str, float | int]]

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable report."""

        return {"overall": self.overall, "per_dimension": self.per_dimension}

    def to_flat_dict(self, *, prefix: str = "") -> dict[str, float]:
        """Return overall metrics as a flat dictionary."""

        return {f"{prefix}{key}": value for key, value in self.overall.items()}


class KoopmanContributionEvaluator:
    """Evaluate first- and second-order Koopman contribution diagnostics."""

    def __init__(self, *, sparsity_threshold: float = 1e-6, eps: float = 1e-8) -> None:
        if sparsity_threshold < 0:
            raise ValueError(
                f"sparsity_threshold must be non-negative. Current input: {sparsity_threshold}."
            )
        if eps <= 0:
            raise ValueError(f"eps must be positive. Current input: {eps}.")
        self.sparsity_threshold = sparsity_threshold
        self.eps = eps

    def evaluate(
        self,
        first_order: Any,
        second_order: Any | None = None,
        *,
        diagonal: Any | None = None,
        fm: Any | None = None,
        cross: Any | None = None,
    ) -> KoopmanContributionReport:
        """Evaluate contribution diagnostics from tensors or an output mapping."""

        if isinstance(first_order, dict):
            output = first_order
            first_order = _required(output, "first_order")
            second_order = second_order if second_order is not None else output.get("second_order")
            diagonal = diagonal if diagonal is not None else output.get("second_order_diagonal")
            fm = fm if fm is not None else output.get("second_order_fm")
            cross = cross if cross is not None else output.get("second_order_cross")
        first = _as_2d(first_order, name="first_order")
        second = np.zeros_like(first) if second_order is None else _as_2d(second_order, name="second_order")
        if first.shape != second.shape:
            raise ValueError(
                f"first_order and second_order must share shape. Current shapes: "
                f"{first.shape} and {second.shape}."
            )
        diagonal_contribution = _mean_abs_optional(diagonal)
        fm_contribution = _mean_abs_optional(fm)
        cross_contribution = _mean_abs_optional(cross)
        if fm_contribution is None and cross_contribution is None:
            fm_contribution = float(np.mean(np.abs(second)))
            cross_contribution = fm_contribution
        if diagonal_contribution is None:
            diagonal_contribution = 0.0
        if fm_contribution is None:
            fm_contribution = 0.0
        if cross_contribution is None:
            cross_contribution = 0.0
        first_abs = np.abs(first)
        second_abs = np.abs(second)
        first_total = float(np.mean(first_abs))
        second_total = float(np.mean(second_abs))
        overall = {
            "first_order_mean_abs": first_total,
            "second_order_mean_abs": second_total,
            "second_to_first_ratio": float(second_total / max(first_total, self.eps)),
            "second_order_sparsity": float(np.mean(second_abs <= self.sparsity_threshold)),
            "diagonal_mean_abs": float(diagonal_contribution),
            "fm_mean_abs": float(fm_contribution),
            "cross_mean_abs": float(cross_contribution),
        }
        per_dimension = []
        for idx in range(first.shape[1]):
            first_i = float(np.mean(first_abs[:, idx]))
            second_i = float(np.mean(second_abs[:, idx]))
            per_dimension.append(
                {
                    "Dimension": idx,
                    "first_order_mean_abs": first_i,
                    "second_order_mean_abs": second_i,
                    "second_to_first_ratio": float(second_i / max(first_i, self.eps)),
                    "second_order_sparsity": float(
                        np.mean(second_abs[:, idx] <= self.sparsity_threshold)
                    ),
                }
            )
        return KoopmanContributionReport(overall=overall, per_dimension=per_dimension)


def _required(output: dict[str, Any], key: str) -> Any:
    if key not in output:
        raise KeyError(
            f"Koopman output mapping must contain key {key!r}. "
            f"Legal keys include: first_order, second_order."
        )
    return output[key]


def _mean_abs_optional(value: Any | None) -> float | None:
    if value is None:
        return None
    array = _as_2d(value, name="contribution")
    return float(np.mean(np.abs(array)))


def _as_2d(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim >= 2:
        return array.reshape(array.shape[0], -1)
    raise ValueError(f"{name} must be at least 1D. Current shape: {array.shape}.")
