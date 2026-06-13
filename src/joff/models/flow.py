"""NICE normalizing flow model."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from joff.core.config import ModelConfig
from joff.core.errors import BuildError
from joff.core.factory import register_model
from joff.layers.builder import build_mlp
from joff.training.losses import NICELoss

from .base import BaseModel, batch_inputs


class AdditiveCoupling(nn.Module):
    """One additive coupling layer with a fixed binary mask."""

    def __init__(
        self,
        input_dim: int,
        hidden: list[int | str] | str,
        act: list[str] | str,
        mask: torch.Tensor,
        *,
        use_scale: bool = False,
    ) -> None:
        super().__init__()
        self.register_buffer("mask", mask.to(dtype=torch.bool))
        identity_dim = int(self.mask.sum().item())
        transform_dim = input_dim - identity_dim
        if identity_dim == 0 or transform_dim == 0:
            raise BuildError(
                f"NICE coupling mask must keep and transform at least one feature. "
                f"input_dim={input_dim}, identity_dim={identity_dim}, transform_dim={transform_dim}."
            )
        self.net = build_mlp(
            input_dim=identity_dim,
            output_dim=transform_dim,
            hidden=hidden,
            act=act,
            output_act="a",
        )
        self.log_scale = nn.Parameter(torch.zeros(input_dim)) if use_scale else None

    def forward(self, x: torch.Tensor, *, reverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the coupling layer and return output plus log determinant."""

        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        base = x
        if reverse and self.log_scale is not None:
            base = x * torch.exp(-self.log_scale)
            log_det = log_det - self.log_scale.sum()
        y = base.clone()
        identity = base[:, self.mask]
        transform = base[:, ~self.mask]
        shift = self.net(identity)
        y[:, ~self.mask] = transform - shift if reverse else transform + shift
        if not reverse and self.log_scale is not None:
            y = y * torch.exp(self.log_scale)
            log_det = log_det + self.log_scale.sum()
        return y, log_det


class NICE(BaseModel):
    """Non-linear Independent Components Estimation flow with additive couplings."""

    model_type = "nice"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        input_dim = _resolve_input_dim(config)
        self.input_dim = input_dim
        self.couplings = nn.ModuleList()
        for layer_idx in range(config.coupling_layers):
            mask = _coupling_mask(
                input_dim,
                layer_idx=layer_idx,
                odd_even_grouping=config.odd_even_grouping,
            )
            self.couplings.append(
                AdditiveCoupling(
                    input_dim=input_dim,
                    hidden=config.hidden,
                    act=config.act,
                    mask=mask,
                    use_scale=config.scaling_mode == "every",
                )
            )
        self.log_scale = (
            nn.Parameter(torch.zeros(input_dim)) if config.scaling_mode == "last" else None
        )
        self.nice_loss = NICELoss(prior_weight=float(config.prior_loss_weight))

    def forward(self, batch: Any) -> dict[str, torch.Tensor]:
        """Transform inputs to latent ``z`` and expose the log determinant."""

        x = batch_inputs(batch)
        z, log_det = self.transform(x)
        reconstruction = self.inverse(z)
        return {
            "z": z,
            "latent": z,
            "log_det": log_det,
            "reconstruction": reconstruction,
        }

    def transform(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward flow from data space to latent space."""

        z = x
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for coupling in self.couplings:
            z, layer_log_det = coupling(z)
            log_det = log_det + layer_log_det
        if self.log_scale is not None:
            z = z * torch.exp(self.log_scale)
            log_det = log_det + self.log_scale.sum()
        return z, log_det

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Invert latent vectors back to data space."""

        x = z
        if self.log_scale is not None:
            x = x * torch.exp(-self.log_scale)
        for coupling in reversed(self.couplings):
            x, _ = coupling(x, reverse=True)
        return x

    def compute_loss(
        self,
        batch: Any,
        output: dict[str, torch.Tensor],
        loss_context: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Compute Gaussian prior negative log likelihood proxy."""

        return self.nice_loss(
            output["z"],
            output["log_det"],
            reconstruction=output["reconstruction"],
            target=batch_inputs(batch),
        ).to_dict()


def _resolve_input_dim(config: ModelConfig) -> int:
    if config.input_dim is not None:
        return config.input_dim
    if config.struct and isinstance(config.struct[0], int):
        return config.struct[0]
    raise BuildError(
        "NICE requires input_dim or a struct whose first item is an integer. "
        f"Current input was: input_dim={config.input_dim!r}, struct={config.struct!r}."
    )


def _coupling_mask(input_dim: int, *, layer_idx: int, odd_even_grouping: bool) -> torch.Tensor:
    indices = torch.arange(input_dim)
    if odd_even_grouping:
        return (indices % 2) == (layer_idx % 2)
    half = input_dim // 2
    mask = torch.zeros(input_dim, dtype=torch.bool)
    if layer_idx % 2 == 0:
        mask[:half] = True
    else:
        mask[half:] = True
    return mask


register_model("nice", NICE, aliases=("flow",), replace=True)
