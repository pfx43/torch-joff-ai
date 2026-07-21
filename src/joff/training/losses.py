"""Reusable training loss components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch.nn import functional as F


RegressionLossName = Literal["mse", "mae", "smooth_l1"]
PredictionLossMode = Literal["all_time", "weighted_last", "last_only", "multi_lift_last"]


@dataclass(frozen=True)
class LossBundle:
    """Scalar total loss plus named scalar components."""

    loss: torch.Tensor
    losses: dict[str, torch.Tensor]

    def to_dict(self) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Return the Trainer-compatible loss dictionary."""

        return {"loss": self.loss, "losses": self.losses}


class PredictionLoss:
    """Prediction loss with NKN-compatible temporal modes."""

    def __init__(
        self,
        *,
        mode: PredictionLossMode = "all_time",
        loss: RegressionLossName = "mse",
        final_weight: float = 2.0,
    ) -> None:
        if mode not in {"all_time", "weighted_last", "last_only", "multi_lift_last"}:
            raise ValueError(
                f"Unknown prediction loss mode {mode!r}. Legal options are: "
                "all_time, weighted_last, last_only, multi_lift_last."
            )
        if final_weight <= 0:
            raise ValueError(f"final_weight must be positive. Current input: {final_weight}.")
        self.mode = mode
        self.loss = loss
        self.final_weight = final_weight

    def __call__(self, prediction: Any, target: torch.Tensor) -> torch.Tensor:
        """Compute prediction loss for a tensor or multi-lift collection."""

        if self.mode == "multi_lift_last":
            predictions = _multi_predictions(prediction)
            losses = [self._last_only_loss(item, target) for item in predictions]
            return torch.stack(losses).mean()
        if not isinstance(prediction, torch.Tensor):
            raise TypeError(
                f"PredictionLoss expects a torch.Tensor for mode {self.mode!r}. "
                f"Current input: {type(prediction).__name__}."
            )
        if self.mode == "last_only":
            return self._last_only_loss(prediction, target)
        aligned_prediction, aligned_target = _align_prediction_target(prediction, target)
        if self.mode == "all_time" or aligned_prediction.ndim < 3:
            return _reduced_loss(aligned_prediction, aligned_target, self.loss)
        return self._weighted_last_loss(aligned_prediction, aligned_target)

    def _last_only_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, tgt = _align_prediction_target(prediction, target)
        if pred.ndim >= 3:
            pred = pred[:, -1, :]
            tgt = tgt[:, -1, :]
        return _reduced_loss(pred, tgt, self.loss)

    def _weighted_last_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        elementwise = _elementwise_loss(prediction, target, self.loss)
        steps = prediction.shape[1]
        weights = torch.linspace(
            1.0,
            self.final_weight,
            steps=steps,
            device=prediction.device,
            dtype=prediction.dtype,
        )
        weights = weights / weights.mean()
        return (elementwise * weights.view(1, steps, 1)).mean()


class NICELoss:
    """Gaussian prior and log-determinant loss for NICE-style flows."""

    def __init__(
        self,
        *,
        prior_weight: float = 1.0,
        log_det_weight: float = 1.0,
        reconstruction_weight: float = 0.0,
        transform: Literal["none", "sigmoid", "softplus"] = "none",
    ) -> None:
        if transform not in {"none", "sigmoid", "softplus"}:
            raise ValueError(
                f"Unknown NICE loss transform {transform!r}. Legal options are: none, sigmoid, softplus."
            )
        self.prior_weight = prior_weight
        self.log_det_weight = log_det_weight
        self.reconstruction_weight = reconstruction_weight
        self.transform = transform

    def __call__(
        self,
        z: torch.Tensor,
        log_det: torch.Tensor,
        *,
        reconstruction: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
    ) -> LossBundle:
        """Compute NICE loss components."""

        prior = 0.5 * z.pow(2).sum(dim=-1).mean()
        log_det_loss = -log_det.mean()
        reconstruction_loss = z.new_tensor(0.0)
        if reconstruction is not None and target is not None:
            reconstruction_loss = F.mse_loss(reconstruction, target.to(reconstruction))
        total = (
            self.prior_weight * prior
            + self.log_det_weight * log_det_loss
            + self.reconstruction_weight * reconstruction_loss
        )
        return LossBundle(
            loss=_transform_loss(total, self.transform),
            losses={
                "prior": prior,
                "log_det": log_det_loss,
                "reconstruction": reconstruction_loss,
            },
        )


class LiftRegularization:
    """Regularize a lift/Koopman tensor with a configured norm and weight."""

    def __init__(
        self,
        *,
        target: str | None = None,
        norm: Literal["l1", "l2"] = "l2",
        weight: float = 1.0,
    ) -> None:
        if norm not in {"l1", "l2"}:
            raise ValueError(f"Unknown lift regularization norm {norm!r}. Legal options are: l1, l2.")
        self.target = target
        self.norm = norm
        self.weight = weight

    def __call__(self, value: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute weighted regularization."""

        tensor = self._resolve(value)
        if self.weight == 0:
            return tensor.new_tensor(0.0)
        if self.norm == "l1":
            penalty = tensor.abs().mean()
        else:
            penalty = tensor.pow(2).mean()
        return self.weight * penalty

    def _resolve(self, value: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        if self.target is None:
            raise ValueError("LiftRegularization target is required when input is a mapping.")
        try:
            return value[self.target]
        except KeyError as exc:
            raise KeyError(
                f"LiftRegularization target {self.target!r} was not found. "
                f"Legal options are: {', '.join(sorted(value))}."
            ) from exc


class SecondOrderPenalty:
    """Penalty for excessive second-order Koopman contribution."""

    def __init__(
        self,
        *,
        max_ratio: float | None = None,
        max_abs: float | None = None,
        weight: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        if max_ratio is not None and max_ratio <= 0:
            raise ValueError(f"max_ratio must be positive. Current input: {max_ratio}.")
        if max_abs is not None and max_abs <= 0:
            raise ValueError(f"max_abs must be positive. Current input: {max_abs}.")
        if eps <= 0:
            raise ValueError(f"eps must be positive. Current input: {eps}.")
        self.max_ratio = max_ratio
        self.max_abs = max_abs
        self.weight = weight
        self.eps = eps

    def __call__(self, first_order: torch.Tensor, second_order: torch.Tensor) -> torch.Tensor:
        """Compute contribution penalty."""

        if self.weight == 0 or (self.max_ratio is None and self.max_abs is None):
            return second_order.new_tensor(0.0)
        penalties: list[torch.Tensor] = []
        if self.max_abs is not None:
            penalties.append(torch.relu(second_order.abs() - self.max_abs))
        if self.max_ratio is not None:
            ratio = second_order.abs() / (first_order.abs() + self.eps)
            penalties.append(torch.relu(ratio - self.max_ratio))
        return self.weight * torch.stack([penalty.mean() for penalty in penalties]).sum()


def _multi_predictions(prediction: Any) -> list[torch.Tensor]:
    if isinstance(prediction, torch.Tensor):
        return [prediction]
    if isinstance(prediction, dict):
        for key in ("multi_lift", "predictions", "prediction"):
            if key in prediction:
                return _multi_predictions(prediction[key])
    if isinstance(prediction, (list, tuple)) and prediction:
        if all(isinstance(item, torch.Tensor) for item in prediction):
            return list(prediction)
    raise TypeError(
        "multi_lift_last expects a Tensor, a non-empty list/tuple of tensors, or a dict "
        "containing one of: multi_lift, predictions, prediction."
    )


def _align_prediction_target(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    aligned_target = target.to(device=prediction.device, dtype=prediction.dtype)
    if prediction.shape == aligned_target.shape:
        return prediction, aligned_target
    if (
        prediction.ndim == 3
        and aligned_target.ndim == 2
        and prediction.shape[0] == aligned_target.shape[0]
        and prediction.shape[-1] == aligned_target.shape[-1]
    ):
        return prediction, aligned_target.unsqueeze(1).expand(-1, prediction.shape[1], -1)
    if (
        prediction.ndim == 2
        and aligned_target.ndim == 3
        and prediction.shape[0] == aligned_target.shape[0]
        and prediction.shape[-1] == aligned_target.shape[-1]
    ):
        return prediction, aligned_target[:, -1, :]
    if (
        prediction.ndim == 3
        and aligned_target.ndim == 3
        and aligned_target.shape[1] == 1
        and prediction.shape[0] == aligned_target.shape[0]
        and prediction.shape[-1] == aligned_target.shape[-1]
    ):
        return prediction, aligned_target.expand(-1, prediction.shape[1], -1)
    raise ValueError(
        f"Prediction and target shapes are incompatible. Current shapes: "
        f"{tuple(prediction.shape)} and {tuple(target.shape)}."
    )


def _reduced_loss(prediction: torch.Tensor, target: torch.Tensor, loss: RegressionLossName) -> torch.Tensor:
    if loss == "mse":
        return F.mse_loss(prediction, target)
    if loss == "mae":
        return F.l1_loss(prediction, target)
    if loss == "smooth_l1":
        return F.smooth_l1_loss(prediction, target)
    raise ValueError(
        f"Unknown regression loss {loss!r}. Legal options are: mse, mae, smooth_l1."
    )


def _elementwise_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss: RegressionLossName,
) -> torch.Tensor:
    if loss == "mse":
        return F.mse_loss(prediction, target, reduction="none")
    if loss == "mae":
        return F.l1_loss(prediction, target, reduction="none")
    if loss == "smooth_l1":
        return F.smooth_l1_loss(prediction, target, reduction="none")
    raise ValueError(
        f"Unknown regression loss {loss!r}. Legal options are: mse, mae, smooth_l1."
    )


def _transform_loss(
    value: torch.Tensor,
    transform: Literal["none", "sigmoid", "softplus"],
) -> torch.Tensor:
    if transform == "none":
        return value
    if transform == "sigmoid":
        return torch.sigmoid(value)
    if transform == "softplus":
        return F.softplus(value)
    raise ValueError(
        f"Unknown loss transform {transform!r}. Legal options are: none, sigmoid, softplus."
    )
