from __future__ import annotations

import torch
from torch import nn

from joff import batch_jacobian, build_model, hessian, jacobian


def test_jacobian_for_vector_function() -> None:
    result = jacobian(lambda x: torch.stack([x[0] ** 2, x[0] * x[1]]), torch.tensor([3.0, 2.0]))
    assert result.shape == (2, 2)
    assert torch.allclose(result, torch.tensor([[6.0, 0.0], [2.0, 3.0]]))


def test_hessian_for_scalar_function() -> None:
    result = hessian(lambda x: x[0] ** 2 + 3.0 * x[1] ** 2, torch.tensor([1.0, 2.0]))
    assert result.shape == (2, 2)
    assert torch.allclose(result, torch.tensor([[2.0, 0.0], [0.0, 6.0]]))


def test_batch_jacobian_for_module() -> None:
    layer = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[2.0, -1.0]]))
    result = batch_jacobian(layer, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert result.shape == (2, 1, 2)
    assert torch.allclose(result[:, 0, :], torch.tensor([[2.0, -1.0], [2.0, -1.0]]))


def test_jacobian_accepts_model_dict_outputs() -> None:
    model = build_model({"type": "dae", "input_dim": 2, "latent_dim": 1, "hidden": [3]})
    result = jacobian(model, torch.tensor([0.5, -0.25]))
    assert result.shape == (2, 2)
