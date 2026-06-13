"""Pydantic configuration objects for the public joff API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, NonNegativeInt, PositiveInt


class StrictConfig(BaseModel):
    """Base class for public configs: strict keys, immutable values."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class DropoutConfig(StrictConfig):
    """Dropout policy used by dense builders."""

    mode: Literal["none", "fixed", "auto"] = "none"
    rate: NonNegativeFloat = 0.0
    threshold: PositiveInt = 100
    scale: float = 100.0
    max_rate: NonNegativeFloat = 0.5


DropoutInput = str | float | DropoutConfig | dict[str, Any]


class ModelConfig(StrictConfig):
    """Model structure configuration."""

    type: str = "mlp"
    input_dim: PositiveInt | None = None
    output_dim: PositiveInt | None = None
    latent_dim: PositiveInt | None = None
    loss: Literal["mse", "mae", "smooth_l1"] = "mse"
    noise_std: NonNegativeFloat = 0.0
    kl_weight: NonNegativeFloat = 1.0
    coupling_layers: PositiveInt = 4
    scaling_mode: Literal["none", "last", "every"] = "last"
    odd_even_grouping: bool = False
    prior_loss_weight: NonNegativeFloat = 1.0
    flow: dict[str, Any] | None = None
    koopman: dict[str, Any] | None = None
    noise_dim: PositiveInt | None = None
    generator_hidden: list[int | str] | str = Field(default_factory=list)
    discriminator_hidden: list[int | str] | str = Field(default_factory=list)
    gan_loss: Literal["bce", "wgan"] = "bce"
    ar_order: PositiveInt = 1
    exogenous_dim: NonNegativeInt = 0
    observer_state_dim: PositiveInt | None = None
    recurrent_type: Literal["rnn", "gru", "lstm"] = "gru"
    embed_dim: PositiveInt | None = None
    num_heads: PositiveInt = 1
    attention_dropout: NonNegativeFloat = 0.0
    attention_mask: Literal[
        "none",
        "temporal",
        "causal",
        "spatial",
        "time_lagged",
        "topological",
        "diagonal",
        "learnable",
    ] = "none"
    attention_lag: PositiveInt | None = None
    attention_topology: list[list[float]] | None = None
    max_sequence_length: PositiveInt = 512
    hidden_size: PositiveInt | None = None
    num_layers: PositiveInt = 1
    bidirectional: bool = False
    sequence_output: Literal["last", "all"] = "last"
    hidden: list[int | str] | str = Field(default_factory=list)
    encoder_hidden: list[int | str] | str = Field(default_factory=list)
    decoder_hidden: list[int | str] | str | Literal["mirror"] = Field(default_factory=list)
    struct: list[int | str] = Field(default_factory=list)
    act: list[str] | str = Field(default_factory=lambda: ["a"])
    output_act: str = "a"
    dropout: DropoutInput = "none"
    batch_norm: bool = False


class DataConfig(StrictConfig):
    """Data loading and lightweight pipeline configuration."""

    preset: str | Path | None = None
    root: Path | None = None
    task: str | None = None
    path: Path | None = None
    target_cols: int | list[int] | None = -1
    batch_size: PositiveInt = 32
    test_ratio: float = 0.2
    seed: int = 42
    shuffle: bool = True
    pipeline: dict[str, Any] | list[Any] | str | Path | None = None
    missing: dict[str, Any] | None = None
    outliers: dict[str, Any] | None = None
    normalization: dict[str, Any] | None = None
    split: dict[str, Any] | None = None
    mask: dict[str, Any] | None = None
    window: dict[str, Any] | None = None
    sequence: dict[str, Any] | None = None
    mpc_window: dict[str, Any] | None = None


class OptimizerConfig(StrictConfig):
    """Optimizer configuration for the minimal trainer."""

    type: Literal["adam", "adamw", "rmsprop", "sgd"] = "adam"
    lr: float = 1e-3
    weight_decay: NonNegativeFloat = 0.0
    param_groups: list[dict[str, Any]] = Field(default_factory=list)
    exclude_bias_from_weight_decay: bool = False


class TrainerConfig(StrictConfig):
    """Training loop configuration."""

    max_epochs: NonNegativeInt = 1
    batch_size: PositiveInt = 32
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    monitor: str | None = None
    mode: Literal["min", "max"] = "min"


class EvaluationConfig(StrictConfig):
    """Evaluation configuration placeholder."""

    type: str | None = None


class ArtifactConfig(StrictConfig):
    """Runtime artifact location."""

    root: Path = Path("runs")
    name: str | None = None


class ExperimentConfig(StrictConfig):
    """Full experiment configuration used by :class:`joff.Experiment`."""

    seed: int = 42
    device: str = "auto"
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)
