"""GAN and WGAN model components."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from joff.core.config import ModelConfig
from joff.core.errors import BuildError
from joff.core.factory import register_model
from joff.layers.builder import build_mlp

from .base import BaseModel, batch_inputs, batch_targets


class Generator(nn.Module):
    """Dense generator that maps latent noise to data-space samples."""

    def __init__(
        self,
        noise_dim: int,
        output_dim: int,
        *,
        hidden: list[int | str] | str,
        act: list[str] | str,
        output_act: str,
    ) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.output_dim = output_dim
        self.net = build_mlp(
            input_dim=noise_dim,
            output_dim=output_dim,
            hidden=hidden,
            act=act,
            output_act=output_act,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate samples from latent noise."""

        return self.net(z)


class Discriminator(nn.Module):
    """Dense discriminator or WGAN critic for data-space samples."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden: list[int | str] | str,
        act: list[str] | str,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.net = build_mlp(
            input_dim=input_dim,
            output_dim=1,
            hidden=hidden,
            act=act,
            output_act="a",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score real or generated samples."""

        return self.net(x)


class GAN(BaseModel):
    """GAN/WGAN container exposing generator, discriminator, and adversarial losses."""

    model_type = "gan"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        data_dim = _resolve_data_dim(config)
        noise_dim = config.noise_dim or config.latent_dim
        if noise_dim is None:
            raise BuildError(
                "GAN requires noise_dim or latent_dim. Legal options: set noise_dim for "
                "latent noise width, or set latent_dim as a compatibility alias."
            )
        self.data_dim = data_dim
        self.noise_dim = noise_dim
        self.loss_type = "wgan" if config.type.strip().lower() == "wgan" else config.gan_loss
        generator_hidden = config.generator_hidden or config.hidden
        discriminator_hidden = config.discriminator_hidden or config.hidden
        self.generator = Generator(
            noise_dim,
            data_dim,
            hidden=generator_hidden,
            act=config.act,
            output_act=config.output_act,
        )
        self.discriminator = Discriminator(
            data_dim,
            hidden=discriminator_hidden,
            act=config.act,
        )

    def forward(self, batch: Any) -> dict[str, torch.Tensor]:
        """Generate samples and score them, using real samples when present."""

        reference = batch_inputs(batch)
        real = _maybe_real_batch(batch, self.data_dim)
        z = _latent_from_batch(reference, self.noise_dim)
        generated = self.generator(z)
        output = {
            "generated": generated,
            "prediction": generated,
            "z": z,
            "fake_score": self.discriminator(generated),
        }
        if real is not None:
            output["real_score"] = self.discriminator(real)
        return output

    def generate(
        self,
        batch_size_or_noise: int | torch.Tensor,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Generate samples from a batch size or explicit latent noise tensor."""

        if isinstance(batch_size_or_noise, torch.Tensor):
            z = _as_feature_matrix(batch_size_or_noise, self.noise_dim, name="noise")
        else:
            resolved_device = next(self.parameters()).device if device is None else torch.device(device)
            z = torch.randn(batch_size_or_noise, self.noise_dim, device=resolved_device, dtype=dtype)
        return self.generator(z)

    def compute_loss(
        self,
        batch: Any,
        output: dict[str, torch.Tensor],
        loss_context: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Compute adversarial generator and discriminator losses."""

        real = _real_batch(batch, self.data_dim)
        generated = output["generated"]
        if self.loss_type == "bce":
            generator_loss, discriminator_loss = self._bce_losses(real, generated)
        elif self.loss_type == "wgan":
            generator_loss, discriminator_loss = self._wgan_losses(real, generated)
        else:
            raise BuildError(
                f"Unknown gan_loss {self.loss_type!r}. Legal options are: bce, wgan. "
                f"Current input: {self.loss_type!r}."
            )
        total = generator_loss + discriminator_loss
        return {
            "loss": total,
            "losses": {
                "generator": generator_loss,
                "discriminator": discriminator_loss,
            },
        }

    def _bce_losses(self, real: torch.Tensor, generated: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        real_score = self.discriminator(real)
        fake_score_for_discriminator = self.discriminator(generated.detach())
        fake_score_for_generator = self._score_generated_with_frozen_discriminator(generated)
        real_targets = torch.ones_like(real_score)
        fake_targets = torch.zeros_like(fake_score_for_discriminator)
        discriminator_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(real_score, real_targets)
            + F.binary_cross_entropy_with_logits(fake_score_for_discriminator, fake_targets)
        )
        generator_loss = F.binary_cross_entropy_with_logits(
            fake_score_for_generator,
            torch.ones_like(fake_score_for_generator),
        )
        return generator_loss, discriminator_loss

    def _wgan_losses(self, real: torch.Tensor, generated: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        real_score = self.discriminator(real)
        fake_score_for_discriminator = self.discriminator(generated.detach())
        fake_score_for_generator = self._score_generated_with_frozen_discriminator(generated)
        discriminator_loss = fake_score_for_discriminator.mean() - real_score.mean()
        generator_loss = -fake_score_for_generator.mean()
        return generator_loss, discriminator_loss

    def _score_generated_with_frozen_discriminator(self, generated: torch.Tensor) -> torch.Tensor:
        previous = [parameter.requires_grad for parameter in self.discriminator.parameters()]
        for parameter in self.discriminator.parameters():
            parameter.requires_grad_(False)
        try:
            return self.discriminator(generated)
        finally:
            for parameter, requires_grad in zip(self.discriminator.parameters(), previous, strict=True):
                parameter.requires_grad_(requires_grad)


def _resolve_data_dim(config: ModelConfig) -> int:
    if config.output_dim is not None:
        return config.output_dim
    if config.input_dim is not None:
        return config.input_dim
    if config.struct and isinstance(config.struct[-1], int):
        return config.struct[-1]
    raise BuildError(
        "GAN requires a data dimension. Legal options: set output_dim, input_dim, "
        f"or end struct with an integer. Current input was: {config.model_dump(mode='json')!r}."
    )


def _latent_from_batch(reference: torch.Tensor, noise_dim: int) -> torch.Tensor:
    features = _as_feature_matrix(reference, reference.shape[-1], name="batch input")
    if features.shape[1] == noise_dim:
        return features
    return torch.randn(features.shape[0], noise_dim, device=features.device, dtype=features.dtype)


def _maybe_real_batch(batch: Any, data_dim: int) -> torch.Tensor | None:
    candidates = []
    try:
        candidates.append(batch_targets(batch))
    except (TypeError, IndexError):
        pass
    try:
        candidates.append(batch_inputs(batch))
    except (TypeError, IndexError):
        pass
    for candidate in candidates:
        if isinstance(candidate, torch.Tensor) and candidate.shape[-1] == data_dim:
            return _as_feature_matrix(candidate, data_dim, name="real batch")
    return None


def _real_batch(batch: Any, data_dim: int) -> torch.Tensor:
    real = _maybe_real_batch(batch, data_dim)
    if real is None:
        raise BuildError(
            f"GAN compute_loss requires real samples with last dimension {data_dim}. "
            "Legal batch forms are Tensor, (noise, real), (real, target), or dict with a "
            "data/target tensor matching the configured data dimension."
        )
    return real


def _as_feature_matrix(value: torch.Tensor, feature_dim: int, *, name: str) -> torch.Tensor:
    if value.ndim == 2 and value.shape[1] == feature_dim:
        return value
    if value.ndim > 2 and value.shape[-1] == feature_dim:
        return value.reshape(-1, feature_dim)
    raise BuildError(
        f"GAN {name} must have last dimension {feature_dim}. Current shape: {tuple(value.shape)}."
    )


register_model("gan", GAN, aliases=("gan_model",), replace=True)
register_model("wgan", GAN, aliases=("wasserstein_gan",), replace=True)
