"""RNN/GRU/LSTM sequence regressor."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from joff.core.config import ModelConfig
from joff.core.errors import BuildError
from joff.core.factory import register_model

from .base import BaseModel, batch_inputs, batch_targets, regression_loss


class SequenceRegressor(BaseModel):
    """Sequence-to-one or sequence-to-sequence regressor backed by RNN/GRU/LSTM."""

    model_type = "sequence"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        input_dim = _resolve_input_dim(config)
        output_dim = config.output_dim
        if output_dim is None:
            raise BuildError(
                "SequenceRegressor requires output_dim. Legal options: set output_dim to the "
                "target feature count."
            )
        hidden_size = config.hidden_size or _hidden_size_from_hidden(config)
        recurrent_type = _resolve_recurrent_type(config)
        recurrent_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[recurrent_type]
        self.recurrent_type = recurrent_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.sequence_output = config.sequence_output
        self.rnn = recurrent_cls(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
        )
        direction_factor = 2 if config.bidirectional else 1
        self.head = nn.Linear(hidden_size * direction_factor, output_dim)

    def forward(self, batch: Any) -> torch.Tensor:
        """Run the recurrent encoder and prediction head."""

        x = _as_sequence(batch_inputs(batch), self.input_dim)
        sequence, _state = self.rnn(x)
        if self.sequence_output == "all":
            return self.head(sequence)
        return self.head(sequence[:, -1, :])

    def compute_loss(
        self,
        batch: Any,
        output: torch.Tensor,
        loss_context: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Compute supervised sequence regression loss."""

        target = batch_targets(batch)
        if output.ndim == 3 and target.ndim == 2:
            target = target.unsqueeze(1).expand(-1, output.shape[1], -1)
        loss = regression_loss(output, target, self.config.loss)
        return {"loss": loss, "losses": {self.config.loss: loss}}


def _resolve_input_dim(config: ModelConfig) -> int:
    if config.input_dim is not None:
        return config.input_dim
    if config.struct and isinstance(config.struct[0], int):
        return config.struct[0]
    raise BuildError(
        "SequenceRegressor requires input_dim or a struct whose first item is an integer. "
        f"Current input was: input_dim={config.input_dim!r}, struct={config.struct!r}."
    )


def _hidden_size_from_hidden(config: ModelConfig) -> int:
    hidden = config.hidden
    if isinstance(hidden, str):
        first = hidden.replace("，", ",").replace("；", ",").replace("、", ",").split(",")[0]
        return int(first.strip())
    if hidden:
        first_value = hidden[0]
        if isinstance(first_value, int):
            return first_value
    return 32


def _resolve_recurrent_type(config: ModelConfig) -> str:
    model_type = config.type.strip().lower()
    if model_type in {"rnn", "gru", "lstm"}:
        return model_type
    return config.recurrent_type


def _as_sequence(x: torch.Tensor, input_dim: int) -> torch.Tensor:
    if x.ndim == 3:
        if x.shape[-1] != input_dim:
            raise BuildError(
                f"Sequence input last dimension must match input_dim={input_dim}. "
                f"Current shape: {tuple(x.shape)}."
            )
        return x
    if x.ndim == 2:
        if x.shape[-1] != input_dim:
            raise BuildError(
                f"2D sequence input feature dimension must match input_dim={input_dim}. "
                f"Current shape: {tuple(x.shape)}."
            )
        return x.unsqueeze(1)
    raise BuildError(
        f"SequenceRegressor input must be 2D or 3D. Current shape: {tuple(x.shape)}."
    )


register_model("sequence", SequenceRegressor, aliases=("sequence_regressor",), replace=True)
register_model("rnn", SequenceRegressor, replace=True)
register_model("gru", SequenceRegressor, replace=True)
register_model("lstm", SequenceRegressor, replace=True)
