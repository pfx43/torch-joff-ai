"""Phase 1 quickstart for a tiny MLP."""

from __future__ import annotations

import torch

from joff import DataModule, Trainer, build_model


def main() -> None:
    """Train a tiny MLP on synthetic data."""

    x = torch.randn(64, 4)
    y = x[:, :2] * 0.5
    data = DataModule.from_arrays(x[:48], y[:48], x[48:], y[48:], batch_size=16)
    model = build_model({"type": "mlp", "input_dim": 4, "output_dim": 2, "hidden": [8], "act": ["r"]})
    trainer = Trainer(max_epochs=1, device="auto", seed=42)
    trainer.fit(model, data)
    trainer.evaluate(model, data)


if __name__ == "__main__":
    main()

