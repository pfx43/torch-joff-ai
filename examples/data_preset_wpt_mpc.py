"""Load and train a tiny model on the real WPT MPC preset."""

from __future__ import annotations

from torch.utils.data import Subset

from joff import DataModule, Trainer, build_model
from _reporting import print_data_summary, print_history, print_metrics, print_model_summary, print_section


def main() -> None:
    print_section("WPT Preset MPC")
    data = DataModule.from_preset(
        "wpt_mpc",
        root="WPT",
        task="mpc",
        batch_size=16,
    )
    small = DataModule(
        Subset(data.train_dataset, range(64)),
        Subset(data.test_dataset, range(32)),
        batch_size=16,
        shuffle=False,
    )
    x, y = next(iter(small.loader("train")))
    model = build_model(
        {"type": "mlp", "input_dim": x.shape[-1], "output_dim": y.shape[-1], "hidden": [8]}
    )
    trainer = Trainer(max_epochs=1, device="cpu", seed=7)
    print_data_summary(data)
    print_model_summary(model)
    result = trainer.fit(model, small)
    metrics = trainer.evaluate(model, small)
    print_history(result.history)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
