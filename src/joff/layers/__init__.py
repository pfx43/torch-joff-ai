"""Layer builders and activation registries."""

from .activations import (
    Gaussian,
    Identity,
    LearnableAffine,
    Square,
    SwiGLU,
    activation_changes_feature_dim,
    build_activation,
    register_builtin_activations,
)
from .builder import build_mlp, dropout_rate_for_width, resolve_widths

__all__ = [
    "Gaussian",
    "Identity",
    "LearnableAffine",
    "Square",
    "SwiGLU",
    "activation_changes_feature_dim",
    "build_activation",
    "build_mlp",
    "dropout_rate_for_width",
    "register_builtin_activations",
    "resolve_widths",
]

