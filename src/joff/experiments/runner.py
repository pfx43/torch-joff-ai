"""ExperimentRunner for executing multiple configs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from joff.artifacts import ArtifactStore

from .experiment import Experiment, ExperimentResult


@dataclass(frozen=True)
class ExperimentRunnerResult:
    """Result returned by :meth:`ExperimentRunner.run`."""

    results: list[ExperimentResult]
    summary: pd.DataFrame
    run_dir: Path | None


class ExperimentRunner:
    """Run one or more experiment configs and collect a summary."""

    def __init__(
        self,
        configs: list[str | Path | dict[str, Any]],
        *,
        root: str | Path | None = None,
        name: str = "runner",
    ) -> None:
        self.configs = configs
        self.root = None if root is None else Path(root)
        self.name = name

    @classmethod
    def from_config(cls, source: str | Path | dict[str, Any] | list[Any]) -> "ExperimentRunner":
        """Build a runner from a YAML path, mapping, or list of configs."""

        if isinstance(source, list):
            return cls(source)
        if isinstance(source, dict):
            if "configs" in source:
                return cls(
                    list(source["configs"]),
                    root=source.get("artifacts", {}).get("root"),
                    name=str(source.get("name", "runner")),
                )
            return cls([source])
        path = Path(source)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(data, dict) and "configs" in data:
            return cls(
                list(data["configs"]),
                root=data.get("artifacts", {}).get("root"),
                name=str(data.get("name", "runner")),
            )
        return cls([data])

    def run(self) -> ExperimentRunnerResult:
        """Run all configured experiments."""

        results: list[ExperimentResult] = []
        rows: list[dict[str, Any]] = []
        for idx, config in enumerate(self.configs):
            result = Experiment.from_config(config).run()
            results.append(result)
            row = {"experiment_id": idx, "run_dir": str(result.run_dir)}
            for key, value in (result.metrics or {}).items():
                row[f"metric.{key}"] = value
            rows.append(row)
        summary = pd.DataFrame(rows)
        run_dir: Path | None = None
        if self.root is not None:
            store = ArtifactStore(self.root, self.name)
            store.save_table("summary/summary.csv", summary)
            run_dir = store.path
        return ExperimentRunnerResult(results=results, summary=summary, run_dir=run_dir)

