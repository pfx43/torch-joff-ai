"""Device resolution helpers."""

from __future__ import annotations

import torch

from .errors import ConfigError


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve ``auto``/string/device inputs to a :class:`torch.device`."""

    if isinstance(device, torch.device):
        return device
    normalized = device.strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    try:
        resolved = torch.device(normalized)
    except (RuntimeError, ValueError) as exc:
        raise ConfigError(
            "Unknown device {!r}. Legal options include 'auto', 'cpu', 'cuda', 'cuda:0', "
            "or a valid torch.device string. Current input was: {!r}.".format(device, device)
        ) from exc
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ConfigError(
            f"Device {device!r} requested CUDA, but torch.cuda.is_available() is False. "
            "Legal options on this machine include 'auto' and 'cpu'."
        )
    return resolved

