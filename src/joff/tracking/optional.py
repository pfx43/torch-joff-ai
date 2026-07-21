"""Optional external tracking adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from matplotlib.figure import Figure

from .base import RunInfo


class TensorBoardTracker:
    """TensorBoard tracker adapter enabled only when tensorboard is installed."""

    def __init__(self, log_dir: str | Path) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise ImportError(
                "TensorBoardTracker requires tensorboard. Install it with "
                'pip install "joff[tracking]".'
            ) from exc
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def start_run(self, run_info: RunInfo) -> None:
        """Record run metadata."""

        self.writer.add_text("run/id", run_info.run_id, 0)
        if run_info.name is not None:
            self.writer.add_text("run/name", run_info.name, 0)

    def log_config(self, config: Any) -> None:
        """Record config as text."""

        self.writer.add_text("config", str(config), 0)

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Log one scalar metric."""

        self.writer.add_scalar(name, float(value), global_step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log multiple scalar metrics."""

        for name, value in metrics.items():
            self.log_metric(name, value, step=step)

    def log_artifact(self, path: str | Path, name: str | None = None) -> None:
        """Record artifact path as text."""

        self.writer.add_text(f"artifact/{name or Path(path).name}", str(path), 0)

    def log_figure(self, figure: Figure, name: str) -> None:
        """Log a matplotlib figure."""

        self.writer.add_figure(name, figure)

    def end_run(self, status: str = "finished") -> None:
        """Close the TensorBoard writer."""

        self.writer.add_text("run/status", status, 0)
        self.writer.close()


class MLflowTracker:
    """MLflow adapter that imports mlflow only when explicitly constructed."""

    def __init__(self, *, experiment_name: str | None = None) -> None:
        try:
            import mlflow
        except ImportError as exc:
            raise ImportError(
                "MLflowTracker requires mlflow. Install it with "
                'pip install "joff[tracking]".'
            ) from exc
        self.mlflow = mlflow
        if experiment_name is not None:
            self.mlflow.set_experiment(experiment_name)

    def start_run(self, run_info: RunInfo) -> None:
        """Start an MLflow run."""

        self.mlflow.start_run(run_name=run_info.name)
        self.mlflow.set_tag("run_id", run_info.run_id)
        for key, value in (run_info.tags or {}).items():
            self.mlflow.set_tag(key, value)

    def log_config(self, config: Any) -> None:
        """Log config text as an MLflow param."""

        self.mlflow.log_param("config", str(config))

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Log one MLflow metric."""

        self.mlflow.log_metric(name, float(value), step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log multiple MLflow metrics."""

        for name, value in metrics.items():
            self.log_metric(name, value, step=step)

    def log_artifact(self, path: str | Path, name: str | None = None) -> None:
        """Log an MLflow artifact."""

        self.mlflow.log_artifact(str(path), artifact_path=name)

    def log_figure(self, figure: Figure, name: str) -> None:
        """Log a matplotlib figure."""

        self.mlflow.log_figure(figure, f"{name}.png")

    def end_run(self, status: str = "finished") -> None:
        """End the MLflow run."""

        self.mlflow.end_run(status=status.upper())


class WandbTracker:
    """Weights & Biases adapter that imports wandb only when explicitly constructed."""

    def __init__(self, *, project: str, entity: str | None = None) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "WandbTracker requires wandb. Install it with "
                'pip install "joff[wandb]".'
            ) from exc
        self.wandb = wandb
        self.project = project
        self.entity = entity
        self.run = None

    def start_run(self, run_info: RunInfo) -> None:
        """Start a W&B run."""

        self.run = self.wandb.init(
            project=self.project,
            entity=self.entity,
            name=run_info.name,
            id=run_info.run_id,
            tags=list((run_info.tags or {}).values()),
        )

    def log_config(self, config: Any) -> None:
        """Update W&B config."""

        if self.run is not None:
            self.run.config.update(config if isinstance(config, dict) else {"config": str(config)})

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Log one W&B metric."""

        payload = {name: float(value)}
        if step is not None:
            payload["step"] = step
        self.wandb.log(payload)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log multiple W&B metrics."""

        payload: dict[str, float | int] = {name: float(value) for name, value in metrics.items()}
        if step is not None:
            payload["step"] = step
        self.wandb.log(payload)

    def log_artifact(self, path: str | Path, name: str | None = None) -> None:
        """Log an artifact path to W&B."""

        artifact = self.wandb.Artifact(name or Path(path).stem, type="artifact")
        artifact.add_file(str(path))
        self.wandb.log_artifact(artifact)

    def log_figure(self, figure: Figure, name: str) -> None:
        """Log a matplotlib figure to W&B."""

        self.wandb.log({name: figure})

    def end_run(self, status: str = "finished") -> None:
        """Finish a W&B run."""

        self.wandb.finish(exit_code=0 if status == "finished" else 1)
