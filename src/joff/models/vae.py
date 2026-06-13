"""Variational autoencoder model."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from joff.core.config import ModelConfig
from joff.core.errors import BuildError
from joff.core.factory import register_model
from joff.layers.builder import build_mlp, resolve_widths

from .base import BaseModel, batch_inputs, regression_loss
from .autoencoder import _resolve_decoder_hidden
from .mlp import _resolve_full_struct, _split_activations


class VAE(BaseModel):
    """Variational autoencoder with reconstruction and KL loss components."""

    model_type = "vae"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        input_dim, encoder_hidden, latent_dim = _resolve_vae_dims(config)
        encoder_act, encoder_output_act = _split_activations(
            config.act, len(encoder_hidden) + 1, "a"
        )
        feature_dim = encoder_hidden[-1] if encoder_hidden else input_dim
        encoder_body_hidden = encoder_hidden[:-1] if encoder_hidden else []
        if encoder_hidden:
            self.encoder = build_mlp(
                input_dim=input_dim,
                output_dim=feature_dim,
                hidden=encoder_body_hidden,
                act=encoder_act or ["r"],
                output_act=encoder_output_act,
                dropout=config.dropout,
                batch_norm=config.batch_norm,
            )
        else:
            self.encoder = nn.Identity()
        decoder_hidden = _resolve_decoder_hidden(config, input_dim, latent_dim, encoder_hidden)
        decoder_act = list(reversed(encoder_act)) if encoder_act else ["r"]
        self.mu = nn.Linear(feature_dim, latent_dim)
        self.logvar = nn.Linear(feature_dim, latent_dim)
        self.decoder = build_mlp(
            input_dim=latent_dim,
            output_dim=input_dim,
            hidden=decoder_hidden,
            act=decoder_act,
            output_act=config.output_act,
            dropout=config.dropout,
            batch_norm=config.batch_norm,
        )
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    def forward(self, batch: Any) -> dict[str, torch.Tensor]:
        """Encode, sample, and reconstruct the input batch."""

        x = batch_inputs(batch)
        features = self.encoder(x)
        mu = self.mu(features)
        logvar = self.logvar(features)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return {"reconstruction": reconstruction, "latent": z, "mu": mu, "logvar": logvar}

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample a latent vector during training and use ``mu`` during eval."""

        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def compute_loss(
        self,
        batch: Any,
        output: dict[str, torch.Tensor],
        loss_context: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Compute reconstruction plus KL divergence loss."""

        reconstruction = regression_loss(
            output["reconstruction"], batch_inputs(batch), self.config.loss
        )
        kl = -0.5 * torch.mean(
            torch.sum(1 + output["logvar"] - output["mu"].pow(2) - output["logvar"].exp(), dim=1)
        )
        total = reconstruction + float(self.config.kl_weight) * kl
        return {"loss": total, "losses": {"reconstruction": reconstruction, "kl": kl}}


def _resolve_vae_dims(config: ModelConfig) -> tuple[int, list[int], int]:
    if config.struct:
        dims = _resolve_full_struct(config)
        if len(dims) < 2:
            raise BuildError(
                f"VAE struct must resolve to at least input and latent widths. Current input: {dims}."
            )
        return dims[0], dims[1:-1], dims[-1]
    if config.input_dim is None or config.latent_dim is None:
        raise BuildError(
            "VAE requires either a full struct such as [10, '*2', '/4'] or both "
            "input_dim and latent_dim."
        )
    hidden = config.encoder_hidden or config.hidden
    encoder_hidden = resolve_widths(config.input_dim, config.latent_dim, hidden)
    return config.input_dim, encoder_hidden, config.latent_dim


register_model("vae", VAE, aliases=("variational_autoencoder",), replace=True)
