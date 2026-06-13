"""Quickstart DAE reconstruction example."""

from __future__ import annotations

import torch

from joff import DataModule, Trainer, build_model
from _reporting import print_data_summary, print_history, print_metrics, print_model_summary, print_section


def main() -> None:
    """Train a small DAE on deterministic synthetic process data."""

    print_section("Quickstart DAE")
    steps = torch.linspace(0.0, 6.0, 72).unsqueeze(1)
    x = torch.cat(
        [
            torch.sin(steps),
            torch.cos(steps),
            steps / 6.0,
            torch.sin(steps * 0.5),
            torch.cos(steps * 0.25),
            torch.sin(steps) * torch.cos(steps),
        ],
        dim=1,
    )
    data = DataModule.from_arrays(
        x[:54],
        x_test=x[54:],
        batch_size=18,
        normalization={"method": "standard"},
        split={"type": "sequential", "test_ratio": 0.25},
    )
    model = build_model(
        {
            "type": "dae",
            "struct": [6, 12, 3],
            "act": ["r", "a"],
            "noise_std": 0.01,
        }
    )
    trainer = Trainer(max_epochs=2, device="auto", seed=123)
    print_data_summary(data)
    print_model_summary(model)
    result = trainer.fit(model, data)
    metrics = trainer.evaluate(model, data)
    print_history(result.history)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
