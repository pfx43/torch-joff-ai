"""PyTorch-native Jacobian and Hessian utilities."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn


TensorFn = Callable[[torch.Tensor], torch.Tensor]


def jacobian(
    fn: nn.Module | TensorFn,
    x: torch.Tensor,
    *,
    create_graph: bool = False,
) -> torch.Tensor:
    """Compute the Jacobian of ``fn`` at one input tensor."""

    single = _single_input(x)
    return torch.autograd.functional.jacobian(
        lambda value: _as_tensor_output(fn(value)),
        single,
        create_graph=create_graph,
    )


def hessian(
    fn: nn.Module | TensorFn,
    x: torch.Tensor,
    *,
    create_graph: bool = False,
) -> torch.Tensor:
    """Compute the Hessian of a scalar-output function at one input tensor."""

    single = _single_input(x)

    def scalar_fn(value: torch.Tensor) -> torch.Tensor:
        output = _as_tensor_output(fn(value))
        if output.numel() != 1:
            raise ValueError(
                f"hessian requires scalar output. Current output shape: {tuple(output.shape)}."
            )
        return output.reshape(())

    return torch.autograd.functional.hessian(scalar_fn, single, create_graph=create_graph)


def batch_jacobian(
    fn: nn.Module | TensorFn,
    x: torch.Tensor,
    *,
    create_graph: bool = False,
) -> torch.Tensor:
    """Compute per-sample Jacobians for a 2D or higher batch tensor."""

    if x.ndim < 2:
        raise ValueError(f"batch_jacobian requires batched input. Current shape: {tuple(x.shape)}.")
    rows = [
        jacobian(fn, x[idx], create_graph=create_graph)
        for idx in range(x.shape[0])
    ]
    return torch.stack(rows, dim=0)


def _single_input(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor input. Current input: {type(x).__name__}.")
    return x.detach().clone().requires_grad_(True)


def _as_tensor_output(value: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for key in ("prediction", "reconstruction", "output", "y"):
            if key in value:
                return value[key]
    raise TypeError(
        f"Function output must be a Tensor or dict containing one of: "
        f"prediction, reconstruction, output, y. Current output: {type(value).__name__}."
    )
