"""Masked attention models and mask factories."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from joff.core.config import ModelConfig
from joff.core.errors import BuildError
from joff.core.factory import register_model

from .base import BaseModel, batch_inputs, batch_targets, regression_loss


_MASK_TYPES = {
    "none",
    "temporal",
    "causal",
    "spatial",
    "time_lagged",
    "topological",
    "diagonal",
    "learnable",
}


class AttentionMaskFactory:
    """Create additive attention masks for sequence or token attention."""

    def __init__(
        self,
        mask_type: str = "none",
        *,
        lag: int | None = None,
        topology: list[list[float]] | torch.Tensor | None = None,
    ) -> None:
        normalized = mask_type.strip().lower()
        if normalized not in _MASK_TYPES:
            legal = ", ".join(sorted(_MASK_TYPES))
            raise BuildError(
                f"Unknown attention mask {mask_type!r}. Legal options are: {legal}. "
                f"Current input: {mask_type!r}."
            )
        self.mask_type = normalized
        self.lag = lag
        self.topology = None if topology is None else torch.as_tensor(topology)

    def build(
        self,
        sequence_length: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor | None:
        """Return an additive ``(target, source)`` attention mask or ``None``."""

        if sequence_length <= 0:
            raise BuildError(
                f"Attention sequence_length must be positive. Current input: {sequence_length}."
            )
        if self.mask_type in {"none", "spatial", "learnable"}:
            return None
        if self.mask_type in {"temporal", "causal"}:
            blocked = _future_mask(sequence_length, device=device)
        elif self.mask_type == "time_lagged":
            blocked = self._time_lagged_mask(sequence_length, device=device)
        elif self.mask_type == "diagonal":
            blocked = torch.eye(sequence_length, device=device, dtype=torch.bool)
        elif self.mask_type == "topological":
            blocked = self._topological_mask(sequence_length, device=device)
        else:
            raise BuildError(
                f"Unsupported attention mask {self.mask_type!r}. Legal options are: "
                f"{', '.join(sorted(_MASK_TYPES))}."
            )
        return _additive_mask(blocked, dtype=dtype)

    def _time_lagged_mask(
        self,
        sequence_length: int,
        *,
        device: torch.device | None,
    ) -> torch.Tensor:
        lag = sequence_length if self.lag is None else self.lag
        if lag <= 0:
            raise BuildError(f"attention_lag must be positive. Current input: {lag}.")
        query = torch.arange(sequence_length, device=device).unsqueeze(1)
        key = torch.arange(sequence_length, device=device).unsqueeze(0)
        return (key > query) | ((query - key) > lag)

    def _topological_mask(
        self,
        sequence_length: int,
        *,
        device: torch.device | None,
    ) -> torch.Tensor:
        if self.topology is None:
            raise BuildError(
                "attention_topology is required when attention_mask='topological'. "
                "Legal input is a square adjacency matrix where non-zero entries are attendable."
            )
        topology = self.topology.to(device=device, dtype=torch.float32)
        if topology.shape != (sequence_length, sequence_length):
            raise BuildError(
                f"attention_topology shape must be ({sequence_length}, {sequence_length}). "
                f"Current shape: {tuple(topology.shape)}."
            )
        return topology == 0


class MaskedMultiheadAttention(nn.Module):
    """Batch-first multihead attention with fixed or learnable additive masks."""

    def __init__(
        self,
        input_dim: int,
        *,
        embed_dim: int | None = None,
        num_heads: int = 1,
        dropout: float = 0.0,
        mask_factory: AttentionMaskFactory | None = None,
        learnable_mask: bool = False,
        max_sequence_length: int = 512,
    ) -> None:
        super().__init__()
        resolved_embed_dim = embed_dim or input_dim
        if resolved_embed_dim % num_heads != 0:
            raise BuildError(
                f"Attention embed_dim must be divisible by num_heads. "
                f"Current input: embed_dim={resolved_embed_dim}, num_heads={num_heads}."
            )
        if not 0 <= dropout < 1:
            raise BuildError(
                f"attention_dropout must be in [0, 1). Current input: {dropout}."
            )
        self.input_dim = input_dim
        self.embed_dim = resolved_embed_dim
        self.num_heads = num_heads
        self.max_sequence_length = max_sequence_length
        self.mask_factory = mask_factory or AttentionMaskFactory()
        self.input_projection = (
            nn.Identity()
            if input_dim == resolved_embed_dim
            else nn.Linear(input_dim, resolved_embed_dim)
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=resolved_embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.learnable_mask = (
            nn.Parameter(torch.zeros(max_sequence_length, max_sequence_length))
            if learnable_mask
            else None
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply masked self-attention and return sequence output plus weights."""

        sequence = _as_sequence(x, self.input_dim)
        if sequence.shape[1] > self.max_sequence_length:
            raise BuildError(
                f"Attention sequence length exceeds max_sequence_length={self.max_sequence_length}. "
                f"Current length: {sequence.shape[1]}."
            )
        projected = self.input_projection(sequence)
        attn_mask = self._attention_mask(
            projected.shape[1],
            device=projected.device,
            dtype=projected.dtype,
        )
        output, weights = self.attention(
            projected,
            projected,
            projected,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        return output, weights

    def _attention_mask(
        self,
        sequence_length: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        fixed_mask = self.mask_factory.build(sequence_length, device=device, dtype=dtype)
        if self.learnable_mask is None:
            return fixed_mask
        learned = self.learnable_mask[:sequence_length, :sequence_length].to(
            device=device,
            dtype=dtype,
        )
        if fixed_mask is None:
            return learned
        return fixed_mask + learned


class Attention(BaseModel):
    """Masked multihead attention regressor for sequence or token inputs."""

    model_type = "attention"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        input_dim = _resolve_input_dim(config)
        output_dim = config.output_dim or input_dim
        embed_dim = config.embed_dim or config.hidden_size or input_dim
        mask_factory = AttentionMaskFactory(
            config.attention_mask,
            lag=config.attention_lag,
            topology=config.attention_topology,
        )
        learnable_mask = config.attention_mask == "learnable"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.sequence_output = config.sequence_output
        self.attention = MaskedMultiheadAttention(
            input_dim,
            embed_dim=embed_dim,
            num_heads=config.num_heads,
            dropout=float(config.attention_dropout),
            mask_factory=mask_factory,
            learnable_mask=learnable_mask,
            max_sequence_length=config.max_sequence_length,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, output_dim)

    def forward(self, batch: Any) -> dict[str, torch.Tensor]:
        """Run masked self-attention and produce a regression prediction."""

        sequence, weights = self.attention(batch_inputs(batch))
        sequence = self.norm(sequence)
        if self.sequence_output == "all":
            prediction = self.head(sequence)
        else:
            prediction = self.head(sequence[:, -1, :])
        return {
            "prediction": prediction,
            "sequence": sequence,
            "attention_weights": weights,
        }

    def compute_loss(
        self,
        batch: Any,
        output: dict[str, torch.Tensor],
        loss_context: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Compute supervised regression loss for the attention prediction."""

        target = _align_target(batch_targets(batch, fallback=batch_inputs(batch)), output["prediction"])
        loss = regression_loss(output["prediction"], target, self.config.loss)
        return {"loss": loss, "losses": {self.config.loss: loss}}


def _resolve_input_dim(config: ModelConfig) -> int:
    if config.input_dim is not None:
        return config.input_dim
    if config.struct and isinstance(config.struct[0], int):
        return config.struct[0]
    raise BuildError(
        "Attention requires input_dim or a struct whose first item is an integer. "
        f"Current input was: input_dim={config.input_dim!r}, struct={config.struct!r}."
    )


def _as_sequence(x: torch.Tensor, input_dim: int) -> torch.Tensor:
    if x.ndim == 3:
        if x.shape[-1] != input_dim:
            raise BuildError(
                f"Attention input last dimension must match input_dim={input_dim}. "
                f"Current shape: {tuple(x.shape)}."
            )
        return x
    if x.ndim == 2:
        if x.shape[-1] != input_dim:
            raise BuildError(
                f"2D attention input feature dimension must match input_dim={input_dim}. "
                f"Current shape: {tuple(x.shape)}."
            )
        return x.unsqueeze(1)
    raise BuildError(f"Attention input must be 2D or 3D. Current shape: {tuple(x.shape)}.")


def _align_target(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    if target.shape == prediction.shape:
        return target
    if prediction.ndim == 2 and target.ndim == 3 and target.shape[0] == prediction.shape[0]:
        return target[:, -1, :]
    if prediction.ndim == 3 and target.ndim == 2 and target.shape[0] == prediction.shape[0]:
        return target.unsqueeze(1).expand(-1, prediction.shape[1], -1)
    return target


def _future_mask(sequence_length: int, *, device: torch.device | None) -> torch.Tensor:
    return torch.triu(
        torch.ones(sequence_length, sequence_length, device=device, dtype=torch.bool),
        diagonal=1,
    )


def _additive_mask(blocked: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.zeros(blocked.shape, device=blocked.device, dtype=dtype)
    return mask.masked_fill(blocked, float("-inf"))


register_model(
    "attention",
    Attention,
    aliases=("masked_attention", "masked_multihead_attention"),
    replace=True,
)
