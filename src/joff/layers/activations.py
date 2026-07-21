"""Activation registry and safe activation builders."""

from __future__ import annotations

import torch
from torch import nn

from joff.core.errors import BuildError
from joff.core.registry import ACTIVATION_REGISTRY


class Identity(nn.Identity):
    """Identity activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x`` unchanged."""

        return x


class Gaussian(nn.Module):
    """Gaussian radial activation ``exp(-x^2)``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Gaussian activation."""

        return torch.exp(-(x**2))


class Square(nn.Module):
    """Square activation ``x^2``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply elementwise square."""

        return x**2


class SwiGLU(nn.Module):
    """SwiGLU activation that halves the final feature dimension."""

    changes_feature_dim = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Split the last dimension and apply SwiGLU."""

        values, gates = x.chunk(2, dim=-1)
        return values * torch.nn.functional.silu(gates)


class LearnableAffine(nn.Module):
    """Learnable elementwise affine activation."""

    requires_feature_dim = True

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(feature_dim))
        self.bias = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learned scale and bias."""

        return x * self.scale + self.bias


def register_builtin_activations() -> None:
    """Register built-in activation aliases once."""

    if "a" in ACTIVATION_REGISTRY:
        return
    ACTIVATION_REGISTRY.register("identity", Identity, aliases=("a", "id", "linear", "none"))
    ACTIVATION_REGISTRY.register("gaussian", Gaussian, aliases=("g",))
    ACTIVATION_REGISTRY.register("square", Square, aliases=("q",))
    ACTIVATION_REGISTRY.register("relu", nn.ReLU, aliases=("r",))
    ACTIVATION_REGISTRY.register("sigmoid", nn.Sigmoid, aliases=("s",))
    ACTIVATION_REGISTRY.register("tanh", nn.Tanh, aliases=("t",))
    ACTIVATION_REGISTRY.register("gelu", nn.GELU, aliases=("gl",))
    ACTIVATION_REGISTRY.register("leaky_relu", nn.LeakyReLU, aliases=("l", "leaky"))
    ACTIVATION_REGISTRY.register("silu", nn.SiLU, aliases=("si", "swish"))
    ACTIVATION_REGISTRY.register("mish", nn.Mish, aliases=("mi",))
    ACTIVATION_REGISTRY.register("swiglu", SwiGLU, aliases=("sg",))
    ACTIVATION_REGISTRY.register("softplus", nn.Softplus, aliases=("sp",))
    ACTIVATION_REGISTRY.register("learnable_affine", LearnableAffine, aliases=("affine", "af"))


def build_activation(name: str, *, feature_dim: int | None = None) -> nn.Module:
    """Build an activation module by registry name or alias."""

    register_builtin_activations()
    activation_cls = ACTIVATION_REGISTRY.get(name)
    if getattr(activation_cls, "requires_feature_dim", False):
        if feature_dim is None:
            raise BuildError(
                f"Activation {name!r} requires feature_dim. Legal options are: "
                f"{', '.join(ACTIVATION_REGISTRY.keys())}."
            )
        return activation_cls(feature_dim)
    return activation_cls()


def activation_changes_feature_dim(name: str) -> bool:
    """Return whether an activation changes the final feature dimension."""

    register_builtin_activations()
    activation_cls = ACTIVATION_REGISTRY.get(name)
    return bool(getattr(activation_cls, "changes_feature_dim", False))


register_builtin_activations()
