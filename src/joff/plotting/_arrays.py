"""Array conversion helpers for plotters."""

from __future__ import annotations

from typing import Any

import numpy as np


def as_array(value: Any) -> np.ndarray:
    """Convert tensors or array-like values to a NumPy array."""

    detach = getattr(value, "detach", None)
    if callable(detach):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def as_1d(value: Any) -> np.ndarray:
    """Convert an array-like value to one dimension."""

    return as_array(value).reshape(-1)


def as_2d(value: Any) -> np.ndarray:
    """Convert an array-like value to two dimensions, preserving row count."""

    array = as_array(value)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim >= 2:
        return array.reshape(array.shape[0], -1)
    raise ValueError(f"Expected at least 1D input. Current shape: {array.shape}.")
