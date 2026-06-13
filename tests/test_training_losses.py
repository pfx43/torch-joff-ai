from __future__ import annotations

import pytest
import torch

from joff import (
    LiftRegularization,
    NICELoss,
    PredictionLoss,
    SecondOrderPenalty,
    build_model,
)


def test_prediction_loss_modes() -> None:
    prediction = torch.tensor([[[1.0], [2.0], [4.0]]])
    target = torch.tensor([[0.0]])
    assert float(PredictionLoss(mode="all_time")(prediction, target)) == pytest.approx(7.0)
    assert float(
        PredictionLoss(mode="weighted_last", final_weight=3.0)(prediction, target)
    ) == pytest.approx(9.5)
    assert float(PredictionLoss(mode="last_only")(prediction, target)) == pytest.approx(16.0)
    multi = PredictionLoss(mode="multi_lift_last")(
        [prediction, torch.tensor([[[1.0], [1.0], [2.0]]])],
        target,
    )
    assert float(multi) == pytest.approx(10.0)


def test_nice_loss_components_and_transform() -> None:
    z = torch.ones(2, 2)
    log_det = torch.full((2,), 0.5)
    bundle = NICELoss(reconstruction_weight=1.0)(
        z,
        log_det,
        reconstruction=torch.ones(2, 2),
        target=torch.zeros(2, 2),
    )
    assert float(bundle.loss) == pytest.approx(1.5)
    assert float(bundle.losses["prior"]) == pytest.approx(1.0)
    assert float(bundle.losses["log_det"]) == pytest.approx(-0.5)
    assert float(bundle.losses["reconstruction"]) == pytest.approx(1.0)
    transformed = NICELoss(transform="softplus")(z, log_det)
    assert float(transformed.loss) == pytest.approx(
        torch.nn.functional.softplus(torch.tensor(0.5)).item()
    )


def test_lift_regularization_target_norm_and_weight() -> None:
    values = torch.tensor([-1.0, 2.0])
    assert float(LiftRegularization(norm="l1", weight=0.5)(values)) == pytest.approx(0.75)
    assert float(LiftRegularization(norm="l2")(values)) == pytest.approx(2.5)
    mapped = {"second_order": values}
    assert float(LiftRegularization(target="second_order", norm="l1")(mapped)) == pytest.approx(1.5)


def test_second_order_penalty_ratio_and_absolute_caps() -> None:
    first_order = torch.tensor([[1.0, 2.0]])
    second_order = torch.tensor([[2.0, 1.0]])
    ratio_penalty = SecondOrderPenalty(max_ratio=1.0, weight=2.0)(first_order, second_order)
    assert float(ratio_penalty) == pytest.approx(1.0)
    abs_penalty = SecondOrderPenalty(max_abs=1.5)(first_order, second_order)
    assert float(abs_penalty) == pytest.approx(0.25)


def test_nkn_uses_structured_loss_components() -> None:
    model = build_model(
        {
            "type": "nkn",
            "input_dim": 4,
            "output_dim": 2,
            "hidden": [6],
            "coupling_layers": 2,
            "koopman": {
                "second_order": True,
                "fm_rank": 2,
                "prediction_loss_mode": "last_only",
                "nice_loss_transform": "softplus",
                "regularization_weight": 0.1,
                "contribution_max_abs": 0.01,
                "contribution_penalty_weight": 0.1,
            },
        }
    )
    batch = (torch.randn(5, 4), torch.randn(5, 2))
    output = model(batch)
    loss = model.compute_loss(batch, output)
    assert loss["loss"].ndim == 0
    assert {"prediction", "flow", "regularization", "second_order_penalty"} <= set(loss["losses"])
    assert torch.isfinite(loss["loss"])
