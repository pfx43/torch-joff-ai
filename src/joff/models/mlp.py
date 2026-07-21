"""MLP model wrapper around the dense layer builder."""

from __future__ import annotations

from typing import Any

import torch

from joff.core.config import ModelConfig
from joff.core.errors import BuildError
from joff.core.factory import register_model
from joff.layers.builder import build_mlp, resolve_widths

from .base import BaseModel, batch_inputs, batch_targets, regression_loss

_ACT_TRANSLATION = str.maketrans({"，": ",", "；": ",", "、": ","})


class MLP(BaseModel):
    """Plain dense network for regression and general feature mapping."""

    model_type = "mlp"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        input_dim, hidden, output_dim, hidden_act, output_act = _resolve_mlp_config(config)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net = build_mlp(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden=hidden,
            act=hidden_act,
            output_act=output_act,
            dropout=config.dropout,
            batch_norm=config.batch_norm,
        )

    def forward(self, batch: Any) -> torch.Tensor:
        """Run a forward pass on a tensor-like batch."""

        return self.net(batch_inputs(batch))

    def compute_loss(
        self,
        batch: Any,
        output: torch.Tensor,
        loss_context: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Compute supervised regression loss against batch targets."""

        target = batch_targets(batch, fallback=batch_inputs(batch))
        loss = regression_loss(output, target, self.config.loss)
        return {"loss": loss, "losses": {self.config.loss: loss}}


def _resolve_mlp_config(config: ModelConfig) -> tuple[int, list[int], int, list[str], str]:
    if config.struct:
        dims = _resolve_full_struct(config)
        if len(dims) < 2:
            raise BuildError(
                f"MLP struct must resolve to at least input and output widths. Current input: {dims}."
            )
        hidden = dims[1:-1]
        hidden_act, output_act = _split_activations(config.act, len(dims) - 1, config.output_act)
        return dims[0], hidden, dims[-1], hidden_act, output_act
    if config.input_dim is None or config.output_dim is None:
        raise BuildError(
            "MLP requires input_dim and output_dim when struct is not provided. "
            "Legal options: provide {'input_dim': int, 'output_dim': int, 'hidden': [...]} "
            "or a full struct such as [10, '*2', '/2']."
        )
    hidden = resolve_widths(config.input_dim, config.output_dim, config.hidden)
    hidden_act, output_act = _split_activations(config.act, len(hidden) + 1, config.output_act)
    return config.input_dim, hidden, config.output_dim, hidden_act, output_act


def _resolve_full_struct(config: ModelConfig) -> list[int]:
    first = config.struct[0]
    if isinstance(first, int):
        input_dim = first
        rest = config.struct[1:]
    elif isinstance(first, str) and first.lower() in {"auto", "i", "input", "input_dim"}:
        if config.input_dim is None:
            raise BuildError(
                f"Struct starts with {first!r}, but input_dim is None. "
                "Legal options: set input_dim or start struct with an integer."
            )
        input_dim = config.input_dim
        rest = config.struct[1:] if first.lower() == "auto" else config.struct
    else:
        if config.input_dim is None:
            raise BuildError(
                "Struct first item must be an integer or input_dim must be provided. "
                f"Current input was: {config.struct!r}."
            )
        input_dim = config.input_dim
        rest = config.struct
    return [input_dim, *resolve_widths(input_dim, config.output_dim, rest)]


def _split_activations(
    act: list[str] | str,
    linear_count: int,
    default_output_act: str,
) -> tuple[list[str], str]:
    if linear_count <= 0:
        return [], default_output_act
    if isinstance(act, str):
        names = [part.strip() for part in act.translate(_ACT_TRANSLATION).split(",") if part.strip()]
    else:
        names = act or ["a"]
    if len(names) == linear_count:
        return names[:-1], names[-1]
    if len(names) == linear_count + 1:
        return names[1:-1], names[-1]
    hidden_count = max(0, linear_count - 1)
    if len(names) == 1:
        return names * hidden_count, default_output_act
    return names[:hidden_count], default_output_act


register_model("mlp", MLP, aliases=("fcnn",), replace=True)
