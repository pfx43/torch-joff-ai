"""Small Study repeat example."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from joff.experiments import Study
from _reporting import print_kv, print_section, print_study_result


def main() -> None:
    """Run independent repeats for one small regression trial."""

    args = _parse_args()
    print_section("Repeat Study")
    run_root = Path(args.run_root)
    data_path = _write_regression_csv(run_root / "example_data" / "repeat_study.csv")
    print_kv("Study Setup", {"data": data_path, "run_root": run_root, "smoke": args.smoke})
    result = Study.from_config(
        {
            "name": "repeat_study",
            "artifacts": {"root": run_root},
            "base": {
                "data": {
                    "path": data_path,
                    "batch_size": 8,
                    "target_cols": [-2, -1],
                    "normalization": {"method": "standard"},
                    "split": {"type": "sequential", "test_ratio": 0.25},
                },
                "model": {"type": "mlp", "hidden": [6], "act": ["r"]},
                "trainer": {"max_epochs": 1 if args.smoke else 3, "optimizer": {"lr": 0.01}},
            },
            "sweep": {"parameters": {"trainer.optimizer.lr": [0.01]}},
            "repeats": {"strategy": "offset", "base_seed": 200, "n": 2 if args.smoke else 3},
            "ranking": {"metric": "rmse", "mode": "min"},
        }
    ).run()
    print_study_result(result)


def _write_regression_csv(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = torch.linspace(0.0, 1.0, 36)
    frame = pd.DataFrame(
        {
            "x1": steps.numpy(),
            "x2": torch.cos(steps * 3.14).numpy(),
            "y1": (0.4 * steps + 0.05).numpy(),
            "y2": (0.2 * steps).numpy(),
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
