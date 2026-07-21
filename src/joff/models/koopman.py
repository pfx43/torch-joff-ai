"""Neural Koopman Network built on top of NICE."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from joff.core.config import ModelConfig
from joff.core.errors import BuildError
from joff.core.factory import register_model
from joff.training.losses import LiftRegularization, NICELoss, PredictionLoss, SecondOrderPenalty

from .base import BaseModel, batch_inputs, batch_targets
from .flow import NICE


class NKN(BaseModel):
    """Minimal Neural Koopman Network using a NICE flow and Koopman latent dynamics."""

    model_type = "nkn"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        input_dim = _resolve_input_dim(config)
        output_dim = config.output_dim or input_dim
        flow_config = _flow_config(config, input_dim)
        koopman = _koopman_config(config)
        self.flow = NICE(flow_config)
        self.linear_koopman = nn.Linear(input_dim, input_dim, bias=False)
        self.second_order_enabled = bool(koopman.get("second_order", False))
        self.second_order_ratio = float(koopman.get("second_order_ratio", 1.0))
        self.regularization_weight = float(koopman.get("regularization_weight", 0.0))
        self.regularization_norm = str(koopman.get("regularization_norm", "l2")).lower()
        self.prediction_loss_weight = float(koopman.get("prediction_loss_weight", 1.0))
        self.nice_loss_weight = float(koopman.get("nice_loss_weight", 1.0))
        self.prediction_loss = PredictionLoss(
            mode=str(koopman.get("prediction_loss_mode", "all_time")),
            loss=config.loss,
            final_weight=float(koopman.get("prediction_final_weight", 2.0)),
        )
        self.flow_loss = NICELoss(
            transform=str(koopman.get("nice_loss_transform", "none")),
        )
        self.lift_regularization = LiftRegularization(
            norm=self.regularization_norm,
        )
        self.second_order_penalty = SecondOrderPenalty(
            max_ratio=koopman.get("contribution_max_ratio"),
            max_abs=koopman.get("contribution_max_abs"),
            weight=float(koopman.get("contribution_penalty_weight", 0.0)),
        )
        self.input_dim = input_dim
        self.output_dim = output_dim
        if self.second_order_enabled:
            rank = int(koopman.get("fm_rank", 4))
            if rank <= 0:
                raise BuildError(f"NKN fm_rank must be positive. Current input: {rank}.")
            self.fm_v = nn.Parameter(torch.empty(input_dim, rank))
            self.fm_out = nn.Linear(rank, input_dim, bias=False)
            nn.init.xavier_uniform_(self.fm_v)
        else:
            self.fm_v = None
            self.fm_out = None
        self.prediction_head = nn.Identity() if output_dim == input_dim else nn.Linear(input_dim, output_dim)

    def forward(self, batch: Any) -> dict[str, torch.Tensor]:
        """Predict future values from the current history batch."""

        x = batch_inputs(batch)
        z, log_det = self.flow.transform(x)
        first_order = self.linear_koopman(z)
        second_order = self._second_order(z)
        z_next = first_order + self.second_order_ratio * second_order
        decoded = self.flow.inverse(z_next)
        prediction = self.prediction_head(decoded)
        return {
            "prediction": prediction,
            "decoded": decoded,
            "latent": z,
            "z": z,
            "z_next": z_next,
            "first_order": first_order,
            "second_order": second_order,
            "log_det": log_det,
        }

    def compute_loss(
        self,
        batch: Any,
        output: dict[str, torch.Tensor],
        loss_context: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Compute prediction, flow, and optional regularization losses."""

        target = batch_targets(batch, fallback=batch_inputs(batch))
        prediction = self.prediction_loss(output["prediction"], target)
        flow = self.flow_loss(output["z"], output["log_det"]).loss
        regularization = self._regularization(output)
        contribution_penalty = self.second_order_penalty(
            output["first_order"],
            output["second_order"],
        )
        total = (
            self.prediction_loss_weight * prediction
            + self.nice_loss_weight * flow
            + self.regularization_weight * regularization
            + contribution_penalty
        )
        return {
            "loss": total,
            "losses": {
                "prediction": prediction,
                "flow": flow,
                "regularization": regularization,
                "second_order_penalty": contribution_penalty,
            },
        }

    def _second_order(self, z: torch.Tensor) -> torch.Tensor:
        if not self.second_order_enabled:
            return torch.zeros_like(z)
        assert self.fm_v is not None and self.fm_out is not None
        projected = z @ self.fm_v
        pair_features = 0.5 * (projected.pow(2) - (z.pow(2) @ self.fm_v.pow(2)))
        return self.fm_out(pair_features)

    def _regularization(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.second_order_enabled or self.fm_v is None:
            return output["prediction"].new_tensor(0.0)
        return self.lift_regularization(self.fm_v)


def _resolve_input_dim(config: ModelConfig) -> int:
    if config.input_dim is not None:
        return config.input_dim
    if config.struct and isinstance(config.struct[0], int):
        return config.struct[0]
    raise BuildError(
        "NKN requires input_dim or a struct whose first item is an integer. "
        f"Current input was: input_dim={config.input_dim!r}, struct={config.struct!r}."
    )


def _flow_config(config: ModelConfig, input_dim: int) -> ModelConfig:
    flow = dict(config.flow or {})
    flow.setdefault("type", "nice")
    flow.setdefault("input_dim", input_dim)
    flow.setdefault("hidden", config.hidden)
    flow.setdefault("act", config.act)
    flow.setdefault("coupling_layers", config.coupling_layers)
    flow.setdefault("scaling_mode", config.scaling_mode)
    flow.setdefault("odd_even_grouping", config.odd_even_grouping)
    return ModelConfig.model_validate(flow)


def _koopman_config(config: ModelConfig) -> dict[str, Any]:
    defaults = {
        "second_order": False,
        "second_order_ratio": 1.0,
        "fm_rank": 4,
        "prediction_loss_weight": 1.0,
        "prediction_loss_mode": "all_time",
        "prediction_final_weight": 2.0,
        "nice_loss_weight": 1.0,
        "nice_loss_transform": "none",
        "regularization_weight": 0.0,
        "regularization_norm": "l2",
        "contribution_max_ratio": None,
        "contribution_max_abs": None,
        "contribution_penalty_weight": 0.0,
    }
    defaults.update(config.koopman or {})
    return defaults


register_model("nkn", NKN, aliases=("neural_koopman_network", "koopman"), replace=True)
