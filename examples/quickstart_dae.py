"""Quickstart DAE reconstruction example."""

from __future__ import annotations

import torch

from joff import DataModule, Trainer, build_model


def main() -> None:
    """Train a small DAE on deterministic synthetic process data."""

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
    trainer.fit(model, data)
    trainer.evaluate(model, data)


if __name__ == "__main__":
    main()
