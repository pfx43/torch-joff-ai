"""Study, Trial, and Repeat orchestration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import hashlib
import json
import traceback
import numpy as np
import pandas as pd
import yaml

from joff.artifacts import ArtifactStore

from .experiment import Experiment, ExperimentResult


@dataclass(frozen=True)
class TrialSpec:
    """One expanded sweep trial."""

    trial_index: int
    trial_id: str
    overrides: dict[str, Any]


@dataclass(frozen=True)
class RepeatSpec:
    """One repeat execution for a trial."""

    repeat_id: int
    seed: int


@dataclass(frozen=True)
class StudyResult:
    """Result returned by :meth:`Study.run`."""

    results: list[ExperimentResult]
    summary: pd.DataFrame
    leaderboard: pd.DataFrame
    failures: pd.DataFrame
    run_dir: Path


class Study:
    """Expand a local grid sweep and independent repeats over :class:`Experiment`."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = deepcopy(config)

    @classmethod
    def from_config(cls, source: str | Path | dict[str, Any]) -> "Study":
        """Load a study from a YAML path or mapping."""

        if isinstance(source, dict):
            return cls(source)
        path = Path(source)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Study config must be a mapping. Current input: {type(data).__name__}.")
        return cls(data)

    def run(
        self,
        *,
        resume: bool | None = None,
        force: bool | None = None,
        retry_failed: bool | None = None,
    ) -> StudyResult:
        """Run all trial/repeat specs and save summary artifacts."""

        name = str(self.config.get("name", "study"))
        base = deepcopy(self.config.get("base", self.config))
        resume_enabled = bool(self.config.get("resume", False) if resume is None else resume)
        force_enabled = bool(self.config.get("force", False) if force is None else force)
        retry_failed_enabled = bool(
            self.config.get("retry_failed", False) if retry_failed is None else retry_failed
        )
        continue_on_error = bool(self.config.get("continue_on_error", True))
        max_failures = self.config.get("max_failures")
        root = Path(
            self.config.get("artifacts", {}).get(
                "root",
                base.get("artifacts", {}).get("root", "runs"),
            )
        )
        store = ArtifactStore(root, name)
        trials = _expand_trials(self.config.get("sweep", {}))
        repeats = _expand_repeats(self.config.get("repeats", {}), default_seed=base.get("seed", 42))
        results: list[ExperimentResult] = []
        summary_rows: list[dict[str, Any]] = []
        failure_rows: list[dict[str, Any]] = []
        store.save_yaml("study.yaml", _json_ready(self.config))
        store.save_table("expanded_trials.csv", _expanded_trial_rows(trials))
        for trial in trials:
            for repeat in repeats:
                run_config = deepcopy(base)
                for path, value in trial.overrides.items():
                    _set_dot_path(run_config, path, value)
                _set_dot_path(run_config, "seed", repeat.seed)
                _set_dot_path(run_config, "artifacts.root", str(root))
                repeat_name = (
                    f"{name}/trials/trial_{trial.trial_index:03d}_{trial.trial_id}/"
                    f"repeats/repeat_{repeat.repeat_id:03d}_seed{repeat.seed}"
                )
                _set_dot_path(run_config, "artifacts.name", repeat_name)
                run_dir = root / repeat_name
                metrics_path = run_dir / "metrics" / "test_metrics.json"
                failure_path = run_dir / "logs" / "traceback.txt"
                if resume_enabled and not force_enabled and metrics_path.exists():
                    summary_rows.append(
                        _summary_row(
                            trial=trial,
                            repeat=repeat,
                            run_dir=run_dir,
                            metrics=_load_json(metrics_path),
                            status="skipped",
                        )
                    )
                    continue
                if (
                    resume_enabled
                    and not force_enabled
                    and failure_path.exists()
                    and not retry_failed_enabled
                ):
                    failure_rows.append(
                        {
                            "trial_index": trial.trial_index,
                            "trial_id": trial.trial_id,
                            "repeat_id": repeat.repeat_id,
                            "seed": repeat.seed,
                            "status": "skipped_failed",
                            "traceback_path": str(failure_path),
                        }
                    )
                    continue
                try:
                    result = Experiment.from_config(run_config).run()
                    results.append(result)
                    summary_rows.append(
                        _summary_row(
                            trial=trial,
                            repeat=repeat,
                            run_dir=result.run_dir,
                            metrics=result.metrics or {},
                            status="completed",
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - failures are first-class study artifacts.
                    failure_path.parent.mkdir(parents=True, exist_ok=True)
                    failure_path.write_text(traceback.format_exc(), encoding="utf-8")
                    failure_rows.append(
                        {
                            "trial_index": trial.trial_index,
                            "trial_id": trial.trial_id,
                            "repeat_id": repeat.repeat_id,
                            "seed": repeat.seed,
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                            "traceback_path": str(failure_path),
                        }
                    )
                    if not continue_on_error:
                        raise
                    if max_failures is not None and len(failure_rows) >= int(max_failures):
                        break
            if max_failures is not None and len(failure_rows) >= int(max_failures):
                break
        summary = pd.DataFrame(summary_rows)
        failures = pd.DataFrame(failure_rows)
        leaderboard = _leaderboard(summary, self.config.get("ranking", {}), failures=failures)
        store.save_table("summary/summary.csv", summary)
        store.save_table("summary/leaderboard.csv", leaderboard)
        store.save_table("summary/failures.csv", failures)
        _save_best_artifacts(store, leaderboard)
        return StudyResult(
            results=results,
            summary=summary,
            leaderboard=leaderboard,
            failures=failures,
            run_dir=store.path,
        )


def _expand_trials(sweep: dict[str, Any]) -> list[TrialSpec]:
    if not isinstance(sweep, dict):
        return [TrialSpec(trial_index=0, trial_id=_trial_id({}), overrides={})]
    strategy = str(sweep.get("strategy", sweep.get("planner", "grid"))).lower()
    parameters = _sweep_parameters(sweep)
    if not parameters:
        return [TrialSpec(trial_index=0, trial_id=_trial_id({}), overrides={})]
    if strategy == "random":
        return _expand_random_trials(
            parameters,
            num_trials=int(sweep.get("num_trials", sweep.get("n", 1))),
            seed=int(sweep.get("seed", 42)),
        )
    if strategy != "grid":
        raise ValueError(
            f"Unknown sweep strategy {strategy!r}. Legal options are: grid, random."
        )
    normal_parameters: dict[str, Any] = {}
    coupled_groups: list[list[dict[str, Any]]] = []
    for key, value in parameters.items():
        if str(key).startswith("coupled."):
            coupled_groups.append(_coupled_values(value, key=str(key)))
        else:
            normal_parameters[str(key)] = value
    keys = list(normal_parameters)
    value_lists = [_grid_values(normal_parameters[key], key=key) for key in keys]
    normal_products = (
        [dict(zip(keys, values)) for values in product(*value_lists)] if keys else [{}]
    )
    coupled_products = list(product(*coupled_groups)) if coupled_groups else [()]
    trials: list[TrialSpec] = []
    for overrides in normal_products:
        for coupled_values in coupled_products:
            merged = dict(overrides)
            for group in coupled_values:
                merged.update(group)
            trials.append(
                TrialSpec(
                    trial_index=len(trials),
                    trial_id=_trial_id(merged),
                    overrides=merged,
                )
            )
    return trials


def _expand_random_trials(
    parameters: dict[str, Any],
    *,
    num_trials: int,
    seed: int,
) -> list[TrialSpec]:
    if num_trials <= 0:
        raise ValueError(f"random sweep num_trials must be positive. Current input: {num_trials}.")
    rng = np.random.default_rng(seed)
    normal_parameters: dict[str, Any] = {}
    coupled_groups: list[list[dict[str, Any]]] = []
    for key, value in parameters.items():
        if str(key).startswith("coupled."):
            coupled_groups.append(_coupled_values(value, key=str(key)))
        else:
            normal_parameters[str(key)] = value
    trials: list[TrialSpec] = []
    for _ in range(num_trials):
        overrides = {
            key: _sample_sweep_value(value, rng, key=key)
            for key, value in normal_parameters.items()
        }
        for group in coupled_groups:
            overrides.update(group[int(rng.integers(0, len(group)))])
        trials.append(
            TrialSpec(
                trial_index=len(trials),
                trial_id=_trial_id(overrides),
                overrides=overrides,
            )
        )
    return trials


def _expand_repeats(repeats: dict[str, Any], *, default_seed: int) -> list[RepeatSpec]:
    if not isinstance(repeats, dict):
        return [RepeatSpec(repeat_id=0, seed=default_seed)]
    strategy = str(repeats.get("strategy", "offset")).lower()
    if strategy == "list":
        seeds = [int(seed) for seed in repeats.get("seeds", repeats.get("seed_list", []))]
        n = int(repeats.get("n", len(seeds)))
        if len(seeds) != n:
            raise ValueError(
                f"repeats strategy 'list' requires exactly n seeds. Current n={n}, "
                f"seeds={seeds!r}."
            )
        return [RepeatSpec(repeat_id=idx, seed=seed) for idx, seed in enumerate(seeds)]
    n = int(repeats.get("n", 1))
    base_seed = int(repeats.get("base_seed", default_seed))
    if n <= 0:
        raise ValueError(f"repeats.n must be positive. Current input: {n}.")
    if strategy == "offset":
        return [RepeatSpec(repeat_id=idx, seed=base_seed + idx) for idx in range(n)]
    if strategy == "spawn":
        sequence = np.random.SeedSequence(base_seed)
        children = sequence.spawn(n)
        return [
            RepeatSpec(repeat_id=idx, seed=int(child.generate_state(1)[0]))
            for idx, child in enumerate(children)
        ]
    raise ValueError(
        f"Unknown repeat strategy {strategy!r}. Legal options are: offset, list, spawn."
    )


def _leaderboard(
    summary: pd.DataFrame,
    ranking: dict[str, Any],
    *,
    failures: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    metric = (
        str(ranking.get("primary", ranking.get("metric", "rmse"))).lower()
        if isinstance(ranking, dict)
        else "rmse"
    )
    mode = str(ranking.get("mode", "min")).lower() if isinstance(ranking, dict) else "min"
    metric_col = f"metric.{metric}"
    if metric_col not in summary.columns and "metric.loss" in summary.columns:
        metric_col = "metric.loss"
    group_cols = ["trial_index", "trial_id", *[col for col in summary.columns if col.startswith("override.")]]
    if not group_cols:
        group_cols = ["trial_id"]
    numeric_cols = [col for col in summary.columns if col.startswith("metric.")]
    grouped = summary.groupby(group_cols, dropna=False)[numeric_cols].agg(
        ["mean", "std", "min", "max", "median", "count"]
    ).reset_index()
    grouped.columns = [
        ".".join(str(part) for part in column if part != "") if isinstance(column, tuple) else str(column)
        for column in grouped.columns
    ]
    for column in numeric_cols:
        count_col = f"{column}.count"
        std_col = f"{column}.std"
        if count_col in grouped.columns and std_col in grouped.columns:
            grouped[f"{column}.ci95"] = 1.96 * grouped[std_col].fillna(0.0) / np.sqrt(
                grouped[count_col].clip(lower=1)
            )
    grouped["n_completed"] = grouped.get(f"{metric_col}.count", 0)
    grouped["n_failed"] = _failure_counts(grouped, failures)
    grouped = _apply_constraints(grouped, ranking)
    sort_columns, ascending = _ranking_sort(metric_col, mode, ranking, grouped)
    if sort_columns:
        grouped = grouped.sort_values(sort_columns, ascending=ascending)
    return grouped


def _ranking_sort(
    metric_col: str,
    mode: str,
    ranking: dict[str, Any],
    grouped: pd.DataFrame,
) -> tuple[list[str], list[bool]]:
    sort_columns: list[str] = []
    ascending: list[bool] = []
    primary = f"{metric_col}.mean"
    if primary in grouped.columns:
        sort_columns.append(primary)
        ascending.append(mode != "max")
    if isinstance(ranking, dict):
        for item in ranking.get("tie_breakers", []) or []:
            if not isinstance(item, dict):
                continue
            metric = str(item.get("metric", "")).lower()
            column = f"metric.{metric}.mean"
            if column not in grouped.columns:
                continue
            sort_columns.append(column)
            ascending.append(str(item.get("mode", "min")).lower() != "max")
    return sort_columns, ascending


def _failure_counts(grouped: pd.DataFrame, failures: pd.DataFrame | None) -> list[int]:
    if failures is None or failures.empty or "trial_id" not in failures.columns:
        return [0 for _ in range(len(grouped))]
    counts = failures.groupby("trial_id").size().to_dict()
    return [int(counts.get(trial_id, 0)) for trial_id in grouped["trial_id"]]


def _set_dot_path(target: dict[str, Any], dot_path: str, value: Any) -> None:
    parts = dot_path.split(".")
    cursor = target
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
        if not isinstance(cursor, dict):
            raise ValueError(f"Cannot set {dot_path!r}: {part!r} is not a mapping.")
    cursor[parts[-1]] = value


def _sweep_parameters(sweep: dict[str, Any]) -> dict[str, Any]:
    if isinstance(sweep.get("parameters"), dict):
        return dict(sweep["parameters"])
    meta = {"strategy", "planner", "num_trials", "n", "seed"}
    return {str(key): value for key, value in sweep.items() if str(key) not in meta}


def _grid_values(value: Any, *, key: str) -> list[Any]:
    if isinstance(value, dict):
        if "choice" in value:
            return _as_list(value["choice"])
        if "grid" in value:
            return _as_list(value["grid"])
        if "values" in value:
            return _as_list(value["values"])
        if "int_range" in value:
            return _int_range_values(value["int_range"], key=key)
        if "range" in value:
            return _int_range_values(value["range"], key=key)
        raise ValueError(
            f"Unknown grid sweep spec for {key!r}. Legal keys are: choice, grid, values, "
            "int_range, range."
        )
    return _as_list(value)


def _sample_sweep_value(value: Any, rng: np.random.Generator, *, key: str) -> Any:
    if isinstance(value, dict):
        if "choice" in value:
            choices = _as_list(value["choice"])
            return choices[int(rng.integers(0, len(choices)))]
        if "grid" in value or "values" in value:
            choices = _grid_values(value, key=key)
            return choices[int(rng.integers(0, len(choices)))]
        if "int_range" in value or "range" in value:
            choices = _int_range_values(value.get("int_range", value.get("range")), key=key)
            return choices[int(rng.integers(0, len(choices)))]
        if "float_range" in value:
            low, high = _range_bounds(value["float_range"], key=key)
            return float(rng.uniform(low, high))
        if "log_uniform" in value or "log_range" in value:
            low, high = _range_bounds(value.get("log_uniform", value.get("log_range")), key=key)
            if low <= 0 or high <= 0:
                raise ValueError(f"log_uniform for {key!r} requires positive bounds.")
            return float(np.exp(rng.uniform(np.log(low), np.log(high))))
        raise ValueError(
            f"Unknown random sweep spec for {key!r}. Legal keys are: choice, grid, values, "
            "int_range, range, float_range, log_uniform, log_range."
        )
    choices = _as_list(value)
    return choices[int(rng.integers(0, len(choices)))]


def _int_range_values(value: Any, *, key: str) -> list[int]:
    if isinstance(value, dict):
        start = int(value.get("start", value.get("min", 0)))
        stop = int(value.get("stop", value.get("max", start)))
        step = int(value.get("step", 1))
    else:
        raw = _as_list(value)
        if len(raw) not in {2, 3}:
            raise ValueError(
                f"Range sweep for {key!r} requires [start, stop] or [start, stop, step]."
            )
        start = int(raw[0])
        stop = int(raw[1])
        step = int(raw[2]) if len(raw) == 3 else 1
    if step <= 0:
        raise ValueError(f"Range sweep for {key!r} requires a positive step.")
    return list(range(start, stop + 1, step))


def _range_bounds(value: Any, *, key: str) -> tuple[float, float]:
    if isinstance(value, dict):
        low = float(value.get("low", value.get("min")))
        high = float(value.get("high", value.get("max")))
    else:
        raw = _as_list(value)
        if len(raw) != 2:
            raise ValueError(f"Float range sweep for {key!r} requires [low, high].")
        low = float(raw[0])
        high = float(raw[1])
    if high < low:
        raise ValueError(f"Range sweep for {key!r} requires high >= low.")
    return low, high


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def _coupled_values(value: Any, *, key: str) -> list[dict[str, Any]]:
    values = _as_list(value)
    output: list[dict[str, Any]] = []
    for item in values:
        if not isinstance(item, dict):
            raise ValueError(
                f"Coupled sweep {key!r} values must be mappings of dot-path overrides. "
                f"Current input: {item!r}."
            )
        output.append(dict(item))
    return output


def _trial_id(overrides: dict[str, Any]) -> str:
    canonical = json.dumps(_json_ready(overrides), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


def _summary_row(
    *,
    trial: TrialSpec,
    repeat: RepeatSpec,
    run_dir: Path,
    metrics: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    row = {
        "trial_index": trial.trial_index,
        "trial_id": trial.trial_id,
        "repeat_id": repeat.repeat_id,
        "seed": repeat.seed,
        "status": status,
        "run_dir": str(run_dir),
        **{f"override.{key}": _cell_ready(value) for key, value in trial.overrides.items()},
    }
    for key, value in metrics.items():
        row[f"metric.{key}"] = value
    return row


def _expanded_trial_rows(trials: list[TrialSpec]) -> list[dict[str, Any]]:
    return [
        {
            "trial_index": trial.trial_index,
            "trial_id": trial.trial_id,
            **{f"override.{key}": _cell_ready(value) for key, value in trial.overrides.items()},
        }
        for trial in trials
    ]


def _apply_constraints(leaderboard: pd.DataFrame, ranking: dict[str, Any]) -> pd.DataFrame:
    if not isinstance(ranking, dict):
        return leaderboard
    constrained = leaderboard
    for constraint in ranking.get("constraints", []) or []:
        if not isinstance(constraint, dict):
            continue
        metric = str(constraint.get("metric", "")).lower()
        column = f"metric.{metric}.mean"
        if column not in constrained.columns:
            continue
        op = str(constraint.get("op", "<="))
        value = float(constraint.get("value", 0.0))
        if op == "<=":
            constrained = constrained[constrained[column] <= value]
        elif op == "<":
            constrained = constrained[constrained[column] < value]
        elif op == ">=":
            constrained = constrained[constrained[column] >= value]
        elif op == ">":
            constrained = constrained[constrained[column] > value]
        elif op == "==":
            constrained = constrained[constrained[column] == value]
        else:
            raise ValueError(
                f"Unknown ranking constraint op {op!r}. Legal options are: <=, <, >=, >, ==."
            )
    return constrained


def _save_best_artifacts(store: ArtifactStore, leaderboard: pd.DataFrame) -> None:
    if leaderboard.empty:
        return
    best = leaderboard.iloc[0].to_dict()
    config = {
        key.removeprefix("override."): value
        for key, value in best.items()
        if str(key).startswith("override.")
    }
    metrics = {key: value for key, value in best.items() if str(key).startswith("metric.")}
    store.save_yaml("best/config.yaml", config)
    store.save_json("best/metrics.json", _json_ready(metrics))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cell_ready(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, Path)):
        return json.dumps(_json_ready(value), sort_keys=True, ensure_ascii=False)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value
