"""Minimal autoencoder/DAE model for the Phase 1 skeleton."""

from __future__ import annotations

from typing import Any

import torch

from joff.core.config import ModelConfig
from joff.core.errors import BuildError
from joff.core.factory import register_model
from joff.layers.builder import build_mlp, resolve_widths

from .base import BaseModel, batch_inputs, regression_loss
from .mlp import _resolve_full_struct, _split_activations


class DAE(BaseModel):
    """Denoising-autoencoder-shaped module with reconstruction output."""

    model_type = "dae"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        input_dim, encoder_hidden, latent_dim = _resolve_dae_dims(config)
        encoder_act, latent_act = _split_activations(config.act, len(encoder_hidden) + 1, "a")
        decoder_hidden = _resolve_decoder_hidden(config, input_dim, latent_dim, encoder_hidden)
        decoder_act = list(reversed(encoder_act)) if encoder_act else ["r"]
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = build_mlp(
            input_dim=input_dim,
            output_dim=latent_dim,
            hidden=encoder_hidden,
            act=encoder_act or ["r"],
            output_act=latent_act,
            dropout=config.dropout,
            batch_norm=config.batch_norm,
        )
        self.decoder = build_mlp(
            input_dim=latent_dim,
            output_dim=input_dim,
            hidden=decoder_hidden,
            act=decoder_act,
            output_act=config.output_act,
            dropout=config.dropout,
            batch_norm=config.batch_norm,
        )

    def forward(self, batch: Any) -> dict[str, torch.Tensor]:
        """Encode and reconstruct the input batch."""

        x = batch_inputs(batch)
        encoder_input = x
        if self.training and self.config.noise_std > 0:
            encoder_input = x + torch.randn_like(x) * float(self.config.noise_std)
        z = self.encoder(encoder_input)
        reconstruction = self.decoder(z)
        return {"reconstruction": reconstruction, "latent": z}

    def compute_loss(
        self,
        batch: Any,
        output: dict[str, torch.Tensor],
        loss_context: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Compute reconstruction loss."""

        reconstruction = regression_loss(
            output["reconstruction"], batch_inputs(batch), self.config.loss
        )
        return {"loss": reconstruction, "losses": {"reconstruction": reconstruction}}


def _resolve_dae_dims(config: ModelConfig) -> tuple[int, list[int], int]:
    if config.struct:
        dims = _resolve_full_struct(config)
        if len(dims) < 2:
            raise BuildError(
                f"DAE struct must resolve to at least input and latent widths. Current input: {dims}."
            )
        return dims[0], dims[1:-1], dims[-1]
    if config.input_dim is None or config.latent_dim is None:
        raise BuildError(
            "DAE requires either a full struct such as [10, '*10', '/2'] or both "
            "input_dim and latent_dim."
        )
    hidden = config.encoder_hidden or config.hidden
    encoder_hidden = resolve_widths(config.input_dim, config.latent_dim, hidden)
    return config.input_dim, encoder_hidden, config.latent_dim


def _resolve_decoder_hidden(
    config: ModelConfig,
    input_dim: int,
    latent_dim: int,
    encoder_hidden: list[int],
) -> list[int]:
    if config.decoder_hidden == "mirror" or not config.decoder_hidden:
        return list(reversed(encoder_hidden))
    return resolve_widths(latent_dim, input_dim, config.decoder_hidden)


register_model("dae", DAE, aliases=("ae", "autoencoder"), replace=True)
