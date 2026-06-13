"""Central immutable-ish defaults registry."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .errors import ConfigError


class DefaultRegistry:
    """Store package, model, task, and data defaults by namespace and key."""

    def __init__(self) -> None:
        self._items: dict[str, dict[str, Any]] = {}

    def register(self, namespace: str, key: str, value: Any, *, replace: bool = False) -> None:
        """Register a default config value."""

        ns = namespace.strip().lower()
        name = key.strip().lower()
        self._items.setdefault(ns, {})
        if not replace and name in self._items[ns]:
            raise ConfigError(
                f"Default {namespace}.{key} already exists. "
                f"Legal options for {namespace!r}: {', '.join(self.list(namespace))}."
            )
        self._items[ns][name] = deepcopy(value)

    def get(self, namespace: str, key: str) -> Any:
        """Return a deep copy of a registered default value."""

        ns = namespace.strip().lower()
        name = key.strip().lower()
        if ns not in self._items or name not in self._items[ns]:
            legal = ", ".join(self.list(namespace)) or "<empty>"
            raise ConfigError(
                f"Unknown default {namespace}.{key}. Legal options for {namespace!r}: {legal}. "
                f"Current input was: {key!r}."
            )
        return deepcopy(self._items[ns][name])

    def list(self, namespace: str) -> tuple[str, ...]:
        """List registered default keys for a namespace."""

        return tuple(sorted(self._items.get(namespace.strip().lower(), {}).keys()))


DEFAULTS = DefaultRegistry()

DEFAULTS.register(
    "package",
    "experiment",
    {
        "seed": 42,
        "device": "auto",
        "data": {"batch_size": 32},
        "model": {
            "type": "mlp",
            "input_dim": None,
            "output_dim": None,
            "hidden": [],
            "struct": [],
            "act": ["a"],
            "output_act": "a",
            "dropout": "none",
            "batch_norm": False,
            "loss": "mse",
            "noise_std": 0.0,
            "kl_weight": 1.0,
            "coupling_layers": 4,
            "scaling_mode": "last",
            "odd_even_grouping": False,
            "prior_loss_weight": 1.0,
            "flow": None,
            "koopman": None,
            "recurrent_type": "gru",
            "hidden_size": None,
            "num_layers": 1,
            "bidirectional": False,
            "sequence_output": "last",
        },
        "trainer": {
            "max_epochs": 1,
            "batch_size": 32,
            "optimizer": {"type": "adam", "lr": 1e-3, "weight_decay": 0.0},
            "monitor": None,
            "mode": "min",
        },
        "evaluation": {"type": None},
        "artifacts": {"root": "runs", "name": None},
    },
)

DEFAULTS.register(
    "model",
    "mlp",
    {
        "model": {
            "type": "mlp",
            "hidden": [64, 64],
            "act": ["r", "r"],
            "output_act": "a",
            "dropout": "none",
            "batch_norm": False,
            "loss": "mse",
        }
    },
)

DEFAULTS.register(
    "model",
    "dae",
    {
        "model": {
            "type": "dae",
            "act": ["r", "r", "a"],
            "output_act": "a",
            "dropout": "none",
            "batch_norm": False,
            "loss": "mse",
            "noise_std": 0.0,
        }
    },
)

DEFAULTS.register(
    "model",
    "vae",
    {
        "model": {
            "type": "vae",
            "act": ["r", "r"],
            "output_act": "a",
            "dropout": "none",
            "batch_norm": False,
            "loss": "mse",
            "kl_weight": 1.0,
        }
    },
)

DEFAULTS.register(
    "model",
    "nice",
    {
        "model": {
            "type": "nice",
            "hidden": [32, 32],
            "act": ["r", "r"],
            "coupling_layers": 4,
            "scaling_mode": "last",
            "odd_even_grouping": False,
            "prior_loss_weight": 1.0,
        }
    },
)

DEFAULTS.register(
    "model",
    "nkn",
    {
        "model": {
            "type": "nkn",
            "hidden": [32, 32],
            "act": ["r", "r"],
            "coupling_layers": 4,
            "scaling_mode": "last",
            "odd_even_grouping": False,
            "flow": None,
            "koopman": {
                "second_order": False,
                "second_order_ratio": 1.0,
                "fm_rank": 4,
                "prediction_loss_weight": 1.0,
                "nice_loss_weight": 1.0,
                "regularization_weight": 0.0,
                "regularization_norm": "l2",
            },
        }
    },
)

DEFAULTS.register(
    "model",
    "sequence",
    {
        "model": {
            "type": "sequence",
            "recurrent_type": "gru",
            "hidden_size": 32,
            "num_layers": 1,
            "bidirectional": False,
            "sequence_output": "last",
            "loss": "mse",
        }
    },
)


def list_defaults(namespace: str) -> tuple[str, ...]:
    """List default keys in ``namespace``."""

    return DEFAULTS.list(namespace)


def get_default_config(namespace: str, key: str) -> Any:
    """Return a deep copy of a registered default config."""

    return DEFAULTS.get(namespace, key)
