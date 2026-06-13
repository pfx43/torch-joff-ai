"""HM-style NKN smoke experiment with dynamic data splitting."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from joff import Experiment


def main() -> None:
    """Run a compact NICE + Koopman experiment on synthetic HM-like data."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run the smallest CPU smoke setup.")
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    args = parser.parse_args()

    rows = 56 if args.smoke else 160
    max_epochs = 1 if args.smoke else 5
    data_dir = args.run_root / "_example_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / ("hm_nkn_smoke.csv" if args.smoke else "hm_nkn.csv")
    _synthetic_hm_frame(rows).to_csv(csv_path, index=False)

    Experiment.from_config(
        {
            "seed": 19,
            "device": "cpu",
            "data": {
                "path": csv_path,
                "target_cols": -1,
                "batch_size": 8,
                "missing": {"strategy": "interpolate_then_drop"},
                "outliers": {
                    "method": "mad",
                    "feature_scope": "quality",
                    "mad_threshold": 4.0,
                    "use_local_temporal": True,
                    "local_window_radius": 4,
                    "local_force_abs_deviation": 2.0,
                    "max_removal_ratio": 0.1,
                },
                "normalization": {"method": "standard"},
                "window": {"lookback": 3, "future_steps": 1},
                "split": {
                    "type": "dynamic_distribution",
                    "train_ratio": 0.75,
                    "test_ratio": 0.25,
                    "slice_length": 6,
                    "seed": 19,
                },
            },
            "model": {
                "type": "nkn",
                "hidden": [12],
                "act": ["r"],
                "coupling_layers": 2,
                "koopman": {
                    "second_order": True,
                    "fm_rank": 3,
                    "nice_loss_weight": 0.01,
                    "regularization_weight": 0.001,
                },
            },
            "trainer": {"max_epochs": max_epochs, "optimizer": {"lr": 0.005}},
            "artifacts": {
                "root": args.run_root,
                "name": "hm_nkn_smoke" if args.smoke else "hm_nkn",
            },
        }
    ).run()


def _synthetic_hm_frame(rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(19)
    time = np.linspace(0.0, 8.0, rows)
    feed_rate = np.sin(time) + 0.05 * rng.normal(size=rows)
    temperature = np.cos(time * 0.7) + 0.04 * rng.normal(size=rows)
    quality = 0.35 * feed_rate - 0.25 * temperature + 0.1 * np.sin(time * 1.5)
    quality = quality + 0.03 * rng.normal(size=rows)
    frame = pd.DataFrame(
        {
            "feed_rate": feed_rate,
            "temperature": temperature,
            "quality": quality,
        }
    )
    frame.loc[6, "temperature"] = np.nan
    frame.loc[rows // 2, "quality"] += 4.0
    return frame


if __name__ == "__main__":
    main()
