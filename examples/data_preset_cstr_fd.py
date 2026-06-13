"""Load and train a tiny model on the real CSTR fault-diagnosis preset."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Subset

from joff import DataModule, Trainer, build_model


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    data = DataModule.from_preset(
        "cstr_fault_diagnosis",
        root=ROOT / "datasets" / "raw" / "oa" / "CSTR",
        task="fault_diagnosis",
        batch_size=16,
    )
    small = DataModule(
        Subset(data.train_dataset, range(64)),
        Subset(data.test_dataset, range(32)),
        batch_size=16,
        shuffle=False,
    )
    x, y = next(iter(small.loader("train")))
    model = build_model({"type": "mlp", "input_dim": x.shape[-1], "output_dim": y.shape[-1], "hidden": [8]})
    Trainer(max_epochs=1, device="cpu", seed=7).fit(model, small)


if __name__ == "__main__":
    main()
