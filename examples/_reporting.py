"""Small console reporting helpers for example scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_kv(title: str, values: dict[str, Any]) -> None:
    print_section(title)
    for key, value in values.items():
        print(f"{key}: {_format_value(value)}")


def print_data_summary(data: Any, *, title: str = "Data") -> None:
    print_section(title)
    print(f"batch_size: {getattr(data, 'batch_size', '<unknown>')}")
    print(f"shuffle_train: {getattr(data, 'shuffle', '<unknown>')}")
    print(f"train_samples: {len(data.train_dataset)}")
    test_dataset = getattr(data, "test_dataset", None)
    print(f"test_samples: {0 if test_dataset is None else len(test_dataset)}")
    train_batch = next(iter(data.loader("train")))
    print(f"train_batch: {_shape_tree(train_batch)}")
    if test_dataset is not None:
        test_batch = next(iter(data.loader("test")))
        print(f"test_batch: {_shape_tree(test_batch)}")
    _print_named_summary(data.summaries, "preset_summary", ("name", "version", "description"))
    _print_named_summary(data.summaries, "task_summary", ("name", "targets", "label_column"))
    _print_named_summary(data.summaries, "source_summary", ("source_type", "root", "access", "split_rows"))
    _print_named_summary(data.summaries, "pipeline_summary", tuple(data.summaries.get("pipeline_summary", {})))


def print_model_summary(model: Any) -> None:
    print_section("Model")
    print(f"type: {model.__class__.__name__}")
    print(f"parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    config = getattr(model, "config", None)
    if config is not None:
        dumped = config.model_dump(mode="json")
        for key in ("type", "input_dim", "output_dim", "hidden", "struct", "act", "loss"):
            if key in dumped and dumped[key] not in (None, [], {}):
                print(f"{key}: {_format_value(dumped[key])}")


def print_history(history: list[dict[str, float]] | None) -> None:
    print_section("Training History")
    if not history:
        print("<empty>")
        return
    columns = _ordered_columns(history)
    print(" | ".join(columns))
    for row in history:
        print(" | ".join(_format_metric(row.get(column)) for column in columns))


def print_metrics(metrics: dict[str, float] | None, *, title: str = "Evaluation Metrics") -> None:
    print_section(title)
    if not metrics:
        print("<empty>")
        return
    preferred = [
        "loss",
        "mse",
        "rmse",
        "mae",
        "r2",
        "maxabs",
        "FAR",
        "MDR",
        "FDR",
        "false_alarms",
        "missed_detections",
        "detected_faults",
    ]
    printed = set()
    for key in preferred:
        if key in metrics:
            print(f"{key}: {_format_metric(metrics[key])}")
            printed.add(key)
    for key in sorted(metrics):
        if key not in printed:
            print(f"{key}: {_format_metric(metrics[key])}")


def print_artifacts(path: str | Path | None, *, title: str = "Artifacts") -> None:
    print_section(title)
    if path is None:
        print("<none>")
        return
    root = Path(path)
    print(f"run_dir: {root.resolve()}")
    if root.exists():
        interesting = [
            "resolved_config.yaml",
            "provenance.json",
            "metrics/history.csv",
            "metrics/test_metrics.json",
            "metrics/fault_detection_report.json",
            "metrics/test_scores.csv",
            "summary/summary.csv",
            "summary/leaderboard.csv",
            "best/metrics.json",
            "plots/loss.png",
            "plots/fault_scores.png",
            "checkpoints/best.pt",
            "checkpoints/last.pt",
        ]
        for relative in interesting:
            candidate = root / relative
            if candidate.exists():
                print(f"- {candidate.resolve()}")


def print_experiment_result(result: Any, *, title: str = "Experiment Result") -> None:
    print_section(title)
    print(f"run_dir: {Path(result.run_dir).resolve()}")
    if result.data_summaries:
        source = result.data_summaries.get("source_summary", {})
        rows = source.get("split_rows")
        if rows is not None:
            print(f"split_rows: {_format_value(rows)}")
        task = result.data_summaries.get("task_summary", {})
        if task:
            print(f"task: {_format_value(task.get('name'))}")
    print_history(result.history)
    print_metrics(result.metrics)
    if result.checkpoint_paths:
        print_kv(
            "Checkpoints",
            {key: Path(value).resolve() for key, value in result.checkpoint_paths.items()},
        )
    print_artifacts(result.run_dir)


def print_runner_result(result: Any, *, title: str = "Runner Result") -> None:
    print_section(title)
    if result.run_dir is not None:
        print(f"run_dir: {Path(result.run_dir).resolve()}")
    print(result.summary.to_string(index=False))
    for item in result.results:
        print(f"- experiment: {Path(item.run_dir).resolve()}")
    if result.run_dir is not None:
        print_artifacts(result.run_dir)


def print_study_result(result: Any, *, title: str = "Study Result") -> None:
    print_section(title)
    print(f"run_dir: {Path(result.run_dir).resolve()}")
    print("\nSummary")
    print(result.summary.to_string(index=False) if not result.summary.empty else "<empty>")
    print("\nLeaderboard")
    print(result.leaderboard.to_string(index=False) if not result.leaderboard.empty else "<empty>")
    if not result.failures.empty:
        print("\nFailures")
        print(result.failures.to_string(index=False))
    print_artifacts(result.run_dir)


def _print_named_summary(summaries: dict[str, Any], key: str, fields: tuple[str, ...]) -> None:
    summary = summaries.get(key)
    if not isinstance(summary, dict):
        return
    print(f"{key}:")
    for field in fields:
        if field in summary:
            print(f"  {field}: {_format_value(summary[field])}")


def _ordered_columns(rows: list[dict[str, float]]) -> list[str]:
    preferred = ["epoch", "train/loss", "test/loss", "test/mse", "test/rmse", "test/mae", "test/r2"]
    keys = {key for row in rows for key in row}
    return [key for key in preferred if key in keys] + sorted(keys - set(preferred))


def _shape_tree(value: Any) -> Any:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(item) for item in shape)
    if isinstance(value, dict):
        return {key: _shape_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_shape_tree(item) for item in value]
    return type(value).__name__


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return _format_metric(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _format_metric(value: Any) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 10000 or (0 < abs(number) < 1e-4):
        return f"{number:.6e}"
    return f"{number:.6f}"
