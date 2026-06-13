"""Optimizer factory helpers."""

from __future__ import annotations

from fnmatch import fnmatchcase
from typing import Any

import torch
from torch import nn


def build_optimizer(model: nn.Module, config: dict[str, Any] | None = None) -> torch.optim.Optimizer:
    """Build a PyTorch optimizer from a strict joff optimizer config mapping."""

    optimizer_config = config or {"type": "adam", "lr": 1e-3, "weight_decay": 0.0}
    kind = str(optimizer_config.get("type", "adam")).lower()
    lr = float(optimizer_config.get("lr", 1e-3))
    weight_decay = float(optimizer_config.get("weight_decay", 0.0))
    params = _parameter_groups(model, optimizer_config, lr=lr, weight_decay=weight_decay)
    if kind == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if kind == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if kind == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    if kind == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer {kind!r}. Legal options are: adam, adamw, rmsprop, sgd.")


def _parameter_groups(
    model: nn.Module,
    config: dict[str, Any],
    *,
    lr: float,
    weight_decay: float,
) -> Any:
    explicit_groups = config.get("param_groups") or []
    exclude_bias = bool(config.get("exclude_bias_from_weight_decay", False))
    if not explicit_groups and not exclude_bias:
        return model.parameters()
    named_parameters = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    assigned: set[int] = set()
    groups: list[dict[str, Any]] = []
    for group_config in explicit_groups:
        if not isinstance(group_config, dict):
            raise TypeError(
                f"Optimizer param_groups entries must be mappings. "
                f"Current input: {type(group_config).__name__}."
            )
        patterns = _patterns(group_config)
        matched = [
            parameter
            for name, parameter in named_parameters
            if id(parameter) not in assigned and _matches_any(name, patterns)
        ]
        if not matched:
            legal = ", ".join(name for name, _parameter in named_parameters)
            raise ValueError(
                f"Optimizer param group match {patterns!r} did not match any parameters. "
                f"Legal parameter names include: {legal}."
            )
        assigned.update(id(parameter) for parameter in matched)
        group = _group_options(group_config)
        group["params"] = matched
        groups.append(group)
    remaining = [
        (name, parameter) for name, parameter in named_parameters if id(parameter) not in assigned
    ]
    if exclude_bias:
        decay = [parameter for name, parameter in remaining if not _is_bias(name)]
        no_decay = [parameter for name, parameter in remaining if _is_bias(name)]
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    elif remaining:
        groups.append({"params": [parameter for _name, parameter in remaining], "lr": lr, "weight_decay": weight_decay})
    return groups


def _patterns(group_config: dict[str, Any]) -> list[str]:
    raw = group_config.get("match", group_config.get("matches"))
    if raw is None:
        raise ValueError(
            "Optimizer param group requires 'match' or 'matches'. "
            "Example: {'match': 'second_order.*', 'lr': 0.0001}."
        )
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
        return raw
    raise TypeError(
        f"Optimizer param group match must be a string or list of strings. Current input: {raw!r}."
    )


def _matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatchcase(name, pattern) for pattern in patterns)


def _group_options(group_config: dict[str, Any]) -> dict[str, Any]:
    excluded = {"match", "matches", "name"}
    return {key: value for key, value in group_config.items() if key not in excluded}


def _is_bias(name: str) -> bool:
    return name == "bias" or name.endswith(".bias")
