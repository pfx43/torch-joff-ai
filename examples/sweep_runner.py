"""Small ExperimentRunner sweep example."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from joff import ExperimentRunner
from _reporting import print_kv, print_runner_result, print_section


def main() -> None:
    """Run a tiny two-config MLP sweep and write runner summary artifacts."""

    args = _parse_args()
    print_section("Sweep Runner")
    run_root = Path(args.run_root)
    data_path = _write_regression_csv(run_root / "example_data" / "sweep_runner.csv")
    max_epochs = 1 if args.smoke else 3
    print_kv("Sweep Setup", {"data": data_path, "max_epochs": max_epochs, "run_root": run_root})
    configs = []
    for idx, hidden in enumerate(([4], [8])):
        configs.append(
            {
                "seed": 100 + idx,
                "data": {
                    "path": data_path,
                    "batch_size": 8,
                    "target_cols": [-2, -1],
                    "normalization": {"method": "standard"},
                    "split": {"type": "sequential", "test_ratio": 0.25},
                },
                "model": {"type": "mlp", "hidden": hidden, "act": ["r"]},
                "trainer": {"max_epochs": max_epochs, "optimizer": {"lr": 0.01}},
                "artifacts": {"root": run_root, "name": f"sweep_runner_exp_{idx}"},
            }
        )
    result = ExperimentRunner.from_config(
        {
            "name": "sweep_runner",
            "artifacts": {"root": run_root},
            "configs": configs,
        }
    ).run()
    print_runner_result(result)


def _write_regression_csv(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = torch.linspace(0.0, 1.0, 40)
    frame = pd.DataFrame(
        {
            "x1": steps.numpy(),
            "x2": torch.sin(steps * 3.14).numpy(),
            "y1": (0.5 * steps).numpy(),
            "y2": (0.25 * steps + 0.1).numpy(),
        }
    )
    frame.to_csv(path, index=False)
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run the smallest CPU-friendly example.")
    parser.add_argument("--run-root", default="runs", help="Directory for generated run artifacts.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
