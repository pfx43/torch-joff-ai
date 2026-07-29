"""P3 论文正常协议编排、监测分数尺度量与静态 episode 校准对象。

文件用途：
    为五阶段论文流程提供独立编排器、估计段监测分数尺度冻结，以及检测/归因校准段的
    有限样本 episode maximum 分位。当前对象是 P3 简单基线闭环，不是论文动态阈值。
主要职责：
    解析严格配置并保留 16 位 hash/provenance；按固定顺序组合 P2 账本、P3 基线、
    ``Trainer`` checkpoint 与 ``ArtifactStore``；逐命名分数流拟合 RMS 尺度；在 P2
    完整连续 episode 上按有限样本秩产生有限或正无穷静态阈值；验证 checkpoint 重放。
    本文件不修改通用 ``Experiment.run()``，也不计算正式故障指标。
关键输入与输出：
    输入为 ``PaperDataBundle``、严格协议配置、``BaselineScoreBatch`` 与 P2
    ``StageSlice``；输出为运行结果、冻结尺度量、静态阈值、逐 episode 最大分数及
    checkpoint/CSV/JSON 产物路径。
依赖与副作用：
    依赖 NumPy、Pandas、PyYAML、Pydantic、P2 数据协议、P3 基线、``ArtifactStore``。
    ``run_normal`` 会在显式运行目录写产物并推进/冻结访问账本；配置解析和单独校准无文件
    副作用，除非调用方传入 YAML 路径。
重要约束：
    尺度量只表示监测分数尺度，不能替代只能在训练段拟合的模型输入预处理。所有基线必须
    先完成同一阶段再进入下一阶段；校准只接受两个独立正常校准段；正式故障访问仍完全
    委托 P2 冻结与许可门禁。样本分辨率不足时阈值必须为正无穷。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

import hashlib
import json
import math
import numpy as np
import subprocess
import torch
import yaml
from pydantic import Field, field_validator, model_validator

from joff.artifacts import ArtifactStore
from joff.core.config import StrictConfig
from joff.data.paper_protocol import (
    FitPurpose,
    PaperDataBundle,
    ProtocolAccessError,
    StageName,
    StageSlice,
)

from .paper_baselines import (
    BaselineScoreBatch,
    PaperBaseline,
    PaperBaselineConfig,
    build_paper_baseline,
    load_paper_baseline,
)
from .paper_environment import sha256_file

_SCALE_EPSILON = 1e-12
_CALIBRATION_STAGES = frozenset(
    {
        StageName.DETECTION_CALIBRATION,
        StageName.ATTRIBUTION_CALIBRATION,
    }
)


@dataclass(frozen=True)
class MonitoringScoreScaler:
    """估计段上逐分数流拟合的 RMS 监测尺度量。

    参数：
        scales: 分数流名称到严格正有限 RMS 尺度的映射。全零或近零流使用 1，避免除零，
            但不会伪造该分数流中的信息。
    异常：
        名称为空、尺度非有限或非正时抛出 ``ValueError``。
    副作用：
        构造时复制并冻结映射；不访问模型输入、不写文件。
    """

    scales: Mapping[str, float]

    def __post_init__(self) -> None:
        """验证并冻结尺度映射。

        参数：
            无。
        返回：
            无。
        异常：
            映射为空、名称为空或尺度非有限/非正时抛出 ``ValueError``。
        副作用：
            通过 ``object.__setattr__`` 替换为只读映射。
        """

        if not self.scales:
            raise ValueError("Monitoring score scaler requires at least one stream.")
        copied: dict[str, float] = {}
        for name, value in self.scales.items():
            normalized_name = str(name).strip()
            scale = float(value)
            if not normalized_name:
                raise ValueError("Monitoring score scaler stream names cannot be empty.")
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError(
                    f"Monitoring score scale for {normalized_name!r} must be positive and finite."
                )
            copied[normalized_name] = scale
        object.__setattr__(self, "scales", MappingProxyType(copied))

    @classmethod
    def fit(cls, scores: BaselineScoreBatch) -> "MonitoringScoreScaler":
        """从估计段命名分数流拟合 RMS 尺度。

        参数：
            scores: 调用方已通过 P2 账本取得估计段数据并评分后的分数批次。
        返回：
            每条流具有严格正尺度的冻结 ``MonitoringScoreScaler``。
        异常：
            ``BaselineScoreBatch`` 已保证形状和有限性；无额外异常。
        副作用：
            无。
        """

        scales = {}
        for name, values in scores.streams.items():
            rms = float(np.sqrt(np.mean(np.square(values))))
            scales[name] = rms if rms > _SCALE_EPSILON else 1.0
        return cls(scales=scales)

    def transform(self, scores: BaselineScoreBatch) -> BaselineScoreBatch:
        """用冻结尺度变换同名分数流并保留原始索引。

        参数：
            scores: 与拟合时具有完全相同流名称的分数批次。
        返回：
            每条分数除以对应 RMS 的新 ``BaselineScoreBatch``。
        异常：
            流名称缺失或多余时抛出 ``ValueError``。
        副作用：
            无；返回对象复制并冻结数组。
        """

        if set(scores.streams) != set(self.scales):
            missing = sorted(set(self.scales).difference(scores.streams))
            extra = sorted(set(scores.streams).difference(self.scales))
            raise ValueError(
                "Monitoring score streams do not match fitted scaler. "
                f"Missing={missing}, extra={extra}."
            )
        return BaselineScoreBatch(
            raw_indices=scores.raw_indices,
            streams={
                name: values / self.scales[name]
                for name, values in scores.streams.items()
            },
        )

    def manifest(self) -> dict[str, Any]:
        """返回明确限定为监测分数用途的 JSON 摘要。

        参数：
            无。
        返回：
            用途范围、方法、零流策略和逐流尺度。
        异常：
            无。
        副作用：
            无。
        """

        return {
            "scope": "monitoring_scores_only",
            "method": "root_mean_square",
            "near_zero_policy": "unit_scale",
            "scales": dict(self.scales),
        }


@dataclass(frozen=True)
class StaticThreshold:
    """一条分数流的有限样本静态 episode 分位。

    参数：
        stream_name: 分数流名称。
        risk_level: 预先指定的 episode 误报警风险水平 ``alpha``。
        episode_count: 完整校准 episode 数。
        rank: 升序 episode maximum 中使用的一基秩。
        value: 有足够分辨率时的有限分位，否则为正无穷。
        finite: 阈值是否有限；必须与 ``value`` 一致。
        kind: 固定为 ``static_episode_quantile``。
        dynamic: 固定为假，禁止把 P3 基线误标成论文动态阈值。
    异常：
        字段不一致时抛出 ``ValueError``。
    副作用：
        无。
    """

    stream_name: str
    risk_level: float
    episode_count: int
    rank: int
    value: float
    finite: bool
    kind: str = "static_episode_quantile"
    dynamic: bool = False

    def __post_init__(self) -> None:
        """校验阈值元数据与有限性标志一致。

        参数：
            无。
        返回：
            无。
        异常：
            名称、风险、计数、类型或有限性不一致时抛出 ``ValueError``。
        副作用：
            无。
        """

        if not self.stream_name:
            raise ValueError("Static threshold stream_name cannot be empty.")
        if not 0.0 < self.risk_level < 1.0:
            raise ValueError("Static threshold risk_level must be between zero and one.")
        if self.episode_count <= 0 or self.rank <= 0:
            raise ValueError("Static threshold episode_count and rank must be positive.")
        if self.kind != "static_episode_quantile" or self.dynamic:
            raise ValueError("P3 static threshold cannot be labelled as dynamic.")
        if self.finite != math.isfinite(self.value):
            raise ValueError("Static threshold finite flag does not match threshold value.")
        if not self.finite and self.value != math.inf:
            raise ValueError("Unresolved static threshold must be positive infinity.")

    def manifest(self) -> dict[str, Any]:
        """返回不含非标准 JSON Infinity 的阈值摘要。

        参数：
            无。
        返回：
            有限阈值写数值；正无穷写 ``None`` 并用 ``value_state`` 说明分辨率不足。
        异常：
            无。
        副作用：
            无。
        """

        return {
            "stream_name": self.stream_name,
            "risk_level": self.risk_level,
            "episode_count": self.episode_count,
            "rank": self.rank,
            "value": self.value if self.finite else None,
            "value_state": "finite" if self.finite else "positive_infinity",
            "finite": self.finite,
            "kind": self.kind,
            "dynamic": self.dynamic,
        }


@dataclass(frozen=True)
class CalibrationEpisode:
    """一个完整校准 episode 的原始范围与逐流最大分数。

    参数：
        stage: detection 或 attribution 校准阶段。
        episode_id: 在当前阶段内从零开始的稳定序号。
        raw_start/raw_stop: 原始索引半开范围；P2 已保证内部连续。
        maxima: 分数流名到该 episode 内最大分数的映射。
        episode_hash: 对阶段、序号和完整原始索引列表计算的 64 位 SHA-256。
    异常：
        范围、阶段或最大分数非法时抛出 ``ValueError``。
    副作用：
        构造时复制并冻结最大分数映射。
    """

    stage: StageName
    episode_id: int
    raw_start: int
    raw_stop: int
    maxima: Mapping[str, float]
    episode_hash: str

    def __post_init__(self) -> None:
        """验证 episode 边界、最大分数和身份 hash 后冻结映射。

        参数：
            无。
        返回：
            无。
        异常：
            阶段不是校准段、序号/范围非法、最大分数为空/非有限/为负，或
            ``episode_hash`` 不是 64 位小写十六进制时抛出 ``ValueError``。
        副作用：
            复制 ``maxima`` 并替换为只读 ``MappingProxyType``。
        """

        if self.stage not in _CALIBRATION_STAGES:
            raise ValueError("Calibration episode must belong to a calibration stage.")
        if self.episode_id < 0 or self.raw_stop <= self.raw_start:
            raise ValueError("Calibration episode id/range is invalid.")
        copied = {str(name): float(value) for name, value in self.maxima.items()}
        if not copied or any(
            not name or not math.isfinite(value) or value < 0
            for name, value in copied.items()
        ):
            raise ValueError("Calibration episode maxima must be named finite non-negative values.")
        if len(self.episode_hash) != 64 or any(
            character not in "0123456789abcdef" for character in self.episode_hash
        ):
            raise ValueError("Calibration episode_hash must be 64 lowercase hex characters.")
        object.__setattr__(self, "maxima", MappingProxyType(copied))

    def manifest(self) -> dict[str, Any]:
        """返回 JSON 可序列化的 episode 最大分数与边界身份。

        参数：
            无。
        返回：
            阶段、稳定序号、原始半开范围、逐流最大分数和 ``episode_hash`` 的新字典。
        异常：
            无；构造时已经校验所有字段。
        副作用：
            无。
        """

        return {
            "stage": self.stage.value,
            "episode_id": self.episode_id,
            "raw_range": [self.raw_start, self.raw_stop],
            "maxima": dict(self.maxima),
            "episode_hash": self.episode_hash,
        }


@dataclass(frozen=True)
class EpisodeCalibrationResult:
    """一次校准的逐流静态阈值与完整 episode 记录。

    参数：
        stage: 校准阶段。
        thresholds: 分数流名到 ``StaticThreshold`` 的映射。
        episodes: 按时间顺序排列的完整 episode 最大分数。
    重要约束：
        ``thresholds`` 的流集合必须与每个 episode 的 ``maxima`` 完全相同。
    """

    stage: StageName
    thresholds: Mapping[str, StaticThreshold]
    episodes: tuple[CalibrationEpisode, ...]

    def __post_init__(self) -> None:
        """冻结阈值映射并检查每个 episode 的分数流集合一致。

        参数：
            无。
        返回：
            无。
        异常：
            阶段非法、阈值或 episode 为空、任一 episode 的流集合不同，或映射键与
            ``StaticThreshold.stream_name`` 不一致时抛出 ``ValueError``。
        副作用：
            复制阈值映射并冻结 episode 元组，不修改其中的阈值数值。
        """

        copied = dict(self.thresholds)
        if self.stage not in _CALIBRATION_STAGES or not copied or not self.episodes:
            raise ValueError("Episode calibration result requires one calibration stage and data.")
        names = set(copied)
        if any(set(episode.maxima) != names for episode in self.episodes):
            raise ValueError("Calibration episodes and thresholds must use identical score streams.")
        if any(threshold.stream_name != name for name, threshold in copied.items()):
            raise ValueError("Calibration threshold mapping keys must match stream names.")
        object.__setattr__(self, "thresholds", MappingProxyType(copied))
        object.__setattr__(self, "episodes", tuple(self.episodes))

    def manifest(self) -> dict[str, Any]:
        """返回阈值与 episode 最大分数的完整 JSON 摘要。

        参数：
            无。
        返回：
            含静态阈值类型、动态标志、逐流阈值和逐 episode hash/最大分数的新字典。
        异常：
            无；子对象均已在构造时校验。
        副作用：
            无。
        """

        return {
            "stage": self.stage.value,
            "threshold_type": "static_episode_quantile",
            "dynamic_threshold": False,
            "thresholds": {
                name: threshold.manifest()
                for name, threshold in sorted(self.thresholds.items())
            },
            "episodes": [episode.manifest() for episode in self.episodes],
        }


class EpisodeMaximumCalibrator:
    """在完整正常 episode maximum 上拟合有限样本静态分位。"""

    def __init__(self, *, risk_level: float) -> None:
        """保存预先指定的 episode 风险水平。

        参数：
            risk_level: 严格位于 ``(0, 1)`` 的尾部风险 ``alpha``。
        返回：
            无。
        异常：
            风险水平非法时抛出 ``ValueError``。
        副作用：
            无。
        """

        self.risk_level = float(risk_level)
        if not 0.0 < self.risk_level < 1.0:
            raise ValueError(
                f"Episode calibration risk_level must be between zero and one. "
                f"Current input: {risk_level}."
            )

    def fit(
        self,
        scores: BaselineScoreBatch,
        stage_slice: StageSlice,
    ) -> EpisodeCalibrationResult:
        """从 P2 完整校准 episode 拟合逐流静态阈值。

        参数：
            scores: 已经使用冻结监测尺度变换的阶段分数。
            stage_slice: 与分数来自同一阶段的 P2 切片。
        返回：
            逐流有限/正无穷阈值和逐 episode 最大分数。
        异常：
            阶段不是 detection/attribution 校准、没有完整 episode、分数含阶段外索引，
            或任一 episode 对某条流没有可用分数时抛出 ``ValueError``。
        副作用：
            无。
        """

        if stage_slice.stage not in _CALIBRATION_STAGES:
            legal = ", ".join(stage.value for stage in sorted(_CALIBRATION_STAGES, key=str))
            raise ValueError(
                f"Episode calibrator accepts only calibration stages: {legal}. "
                f"Current stage={stage_slice.stage.value!r}."
            )
        if not stage_slice.prepared_episode_ranges:
            raise ValueError(
                f"Calibration stage {stage_slice.stage.value!r} has no complete episodes."
            )
        stage_raw_set = set(stage_slice.raw_indices)
        outside = [int(index) for index in scores.raw_indices if int(index) not in stage_raw_set]
        if outside:
            raise ValueError(
                "Calibration scores contain raw indices outside the declared stage. "
                f"First outside index={outside[0]}."
            )
        stage_start = stage_slice.prepared_row_indices[0]
        episodes: list[CalibrationEpisode] = []
        for episode_id, (prepared_start, prepared_stop) in enumerate(
            stage_slice.prepared_episode_ranges
        ):
            local_start = prepared_start - stage_start
            local_stop = prepared_stop - stage_start
            episode_raw = np.asarray(
                stage_slice.raw_indices[local_start:local_stop],
                dtype=np.int64,
            )
            mask = np.isin(scores.raw_indices, episode_raw)
            if not np.any(mask):
                raise ValueError(
                    f"Calibration episode {episode_id} in {stage_slice.stage.value!r} "
                    "does not contain any scored time."
                )
            episode_maxima = {
                name: float(np.max(values[mask]))
                for name, values in scores.streams.items()
            }
            episodes.append(
                CalibrationEpisode(
                    stage=stage_slice.stage,
                    episode_id=episode_id,
                    raw_start=int(episode_raw[0]),
                    raw_stop=int(episode_raw[-1]) + 1,
                    maxima=episode_maxima,
                    episode_hash=_sha256_json(
                        {
                            "stage": stage_slice.stage.value,
                            "episode_id": episode_id,
                            "raw_indices": episode_raw.tolist(),
                        }
                    ),
                )
            )
        episode_count = len(episodes)
        rank = math.ceil((episode_count + 1) * (1.0 - self.risk_level))
        thresholds: dict[str, StaticThreshold] = {}
        for stream_name in scores.streams:
            ordered_maxima = np.sort(
                np.asarray(
                    [episode.maxima[stream_name] for episode in episodes],
                    dtype=np.float64,
                )
            )
            value = (
                float(ordered_maxima[rank - 1])
                if rank <= episode_count
                else math.inf
            )
            thresholds[stream_name] = StaticThreshold(
                stream_name=stream_name,
                risk_level=self.risk_level,
                episode_count=episode_count,
                rank=rank,
                value=value,
                finite=math.isfinite(value),
            )
        return EpisodeCalibrationResult(
            stage=stage_slice.stage,
            thresholds=thresholds,
            episodes=tuple(episodes),
        )


def _score_complete_calibration_episodes(
    baseline: PaperBaseline,
    data: np.ndarray,
    stage_slice: StageSlice,
    *,
    device: str | torch.device,
) -> BaselineScoreBatch:
    """逐个完整 episode 独立评分，再按原始时间拼接分数。

    参数：
        baseline: 已冻结、只通过公开 ``score`` 接口读取数组的基线。
        data: 与 ``stage_slice`` 行对齐的完整校准阶段副本，形状为
            ``[stage_time, features]``。
        stage_slice: P2 生成的 detection 或 attribution 校准切片；其中
            ``prepared_episode_ranges`` 定义不可跨越的独立 episode 边界。
        device: 基线评分使用的 PyTorch 设备。
    返回：
        保持流名称一致、按原始索引递增拼接的 ``BaselineScoreBatch``。
    异常：
        阶段不是校准阶段、数据行数不匹配、没有完整 episode，或不同 episode 返回不同
        分数流时抛出 ``ValueError``；基线评分异常原样传播。
    副作用：
        神经基线可临时移动到目标设备并进入 eval；不更新权重、不写文件。

    重要约束：
        一步预测器必须在每个 episode 内重新建立相邻样本。若把完整阶段一次性交给它，
        原始索引恰好连续时会错误地用前一 episode 末行预测下一 episode 首行，破坏校准
        单元的独立重置语义。
    """

    if stage_slice.stage not in _CALIBRATION_STAGES:
        raise ValueError("Complete-episode scoring accepts only calibration stages.")
    values = np.asarray(data)
    if values.ndim != 2 or len(values) != len(stage_slice.raw_indices):
        raise ValueError(
            "Calibration data rows must align exactly with the declared stage slice."
        )
    if not stage_slice.prepared_episode_ranges:
        raise ValueError("Calibration stage has no complete episodes to score.")
    stage_start = stage_slice.prepared_row_indices[0]
    batches: list[BaselineScoreBatch] = []
    expected_names: set[str] | None = None
    for prepared_start, prepared_stop in stage_slice.prepared_episode_ranges:
        local_start = prepared_start - stage_start
        local_stop = prepared_stop - stage_start
        batch = baseline.score(
            values[local_start:local_stop],
            np.asarray(stage_slice.raw_indices[local_start:local_stop], dtype=np.int64),
            device=device,
        )
        names = set(batch.streams)
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise ValueError(
                "Baseline score streams changed between calibration episodes."
            )
        batches.append(batch)
    assert expected_names is not None
    return BaselineScoreBatch(
        raw_indices=np.concatenate([batch.raw_indices for batch in batches]),
        streams={
            name: np.concatenate([batch.streams[name] for batch in batches])
            for name in sorted(expected_names)
        },
    )


class PaperProtocolConfig(StrictConfig):
    """P3 独立论文协议的一次正常运行配置。

    参数：
        artifact_root: 所有运行目录的根路径。
        run_name: 当前运行的安全相对目录名。
        mode: ``development`` 只允许正常开发闭环；``frozen`` 才能在正常协议冻结后请求
            P2 的正式故障门禁。
        device: 神经基线训练和重放设备；P3 smoke 使用 ``cpu``。
        detection_risk_level: 调用方显式指定的 detection calibration episode 风险
            ``alpha``，没有理论敏感默认值。
        attribution_risk_level: 调用方显式指定的独立 attribution calibration 静态重放
            风险；它只完成 P3 生命周期，不代表 P9 full-normal attribution 风险。
        baselines: 调用方显式指定的至少三条、名称唯一并覆盖 PCA、DAE、一步 MLP 的配置。
    异常：
        未知字段、非法风险、重复名称、缺少三种基线类型或不安全目录名时由 Pydantic
        抛出 ``ValidationError``。
    副作用：
        无；配置冻结后不可修改。
    """

    artifact_root: Path = Path("runs")
    run_name: str = "paper-p3"
    mode: Literal["development", "frozen"]
    device: str = "cpu"
    detection_risk_level: float = Field(gt=0.0, lt=1.0)
    attribution_risk_level: float = Field(gt=0.0, lt=1.0)
    baselines: tuple[PaperBaselineConfig, ...] = Field(min_length=3)

    @field_validator("run_name")
    @classmethod
    def _validate_run_name(cls, value: str) -> str:
        """校验运行名不能逃逸产物根目录。

        参数：
            value: 用户提供的相对运行名。
        返回：
            去除首尾空白后的名称。
        异常：
            为空、绝对路径、含父目录段或磁盘前缀时抛出 ``ValueError``。
        副作用：
            无。
        """

        normalized = value.strip()
        candidate = Path(normalized)
        if (
            not normalized
            or candidate.is_absolute()
            or candidate.drive
            or ".." in candidate.parts
        ):
            raise ValueError(
                f"Paper protocol run_name must stay below artifact_root. Current input={value!r}."
            )
        return normalized

    @model_validator(mode="after")
    def _validate_baseline_set(self) -> "PaperProtocolConfig":
        """保证一次 P3 运行确实覆盖三类最小基线。

        参数：
            无；校验当前配置的 ``baselines``。
        返回：
            当前冻结配置。
        异常：
            名称重复或缺少 ``pca``/``dae``/``mlp`` 任一类型时抛出 ``ValueError``。
        副作用：
            无。
        """

        names = [baseline.name for baseline in self.baselines]
        if len(set(names)) != len(names):
            raise ValueError(f"Paper protocol baseline names must be unique. Current names={names}.")
        required = {"pca", "dae", "mlp"}
        missing = sorted(required.difference(baseline.type for baseline in self.baselines))
        if missing:
            raise ValueError(
                "P3 paper protocol requires PCA, DAE and one-step MLP baselines. "
                f"Missing types={missing}."
            )
        return self


@dataclass(frozen=True)
class ResolvedPaperProtocolConfig:
    """校验后的 P3 配置、逐字段来源与 16 位内容 hash。

    参数：
        config: 冻结 ``PaperProtocolConfig``。
        resolved_config: 含全部默认值的 JSON 兼容只读映射。
        provenance: 每个叶字段的来源记录。
        config_hash: 对完整解析值计算的 16 位 SHA-256 前缀。
    重要约束：
        hash 包含会改变模型、分数、阈值或产物路径的全部配置字段。
    """

    config: PaperProtocolConfig
    resolved_config: Mapping[str, Any]
    provenance: Mapping[str, tuple[Mapping[str, Any], ...]]
    config_hash: str

    def __post_init__(self) -> None:
        """冻结解析映射并校验 hash 长度。

        参数：
            无。
        返回：
            无。
        异常：
            hash 不是 16 位小写十六进制时抛出 ``ValueError``。
        副作用：
            复制并冻结两个映射。
        """

        if len(self.config_hash) != 16 or any(
            character not in "0123456789abcdef" for character in self.config_hash
        ):
            raise ValueError("Resolved paper protocol config_hash must be 16 hex characters.")
        object.__setattr__(
            self,
            "resolved_config",
            MappingProxyType(dict(self.resolved_config)),
        )
        object.__setattr__(
            self,
            "provenance",
            MappingProxyType(
                {
                    path: tuple(MappingProxyType(dict(record)) for record in records)
                    for path, records in self.provenance.items()
                }
            ),
        )


def resolve_paper_protocol_config(
    source: PaperProtocolConfig | Mapping[str, Any] | str | Path,
) -> ResolvedPaperProtocolConfig:
    """解析 P3 配置并保留默认/显式字段来源与稳定 hash。

    参数：
        source: 已校验配置、配置映射或 UTF-8 YAML 路径。
    返回：
        ``ResolvedPaperProtocolConfig``。
    异常：
        文件不存在、YAML 顶层不是映射或 Pydantic 校验失败时传播 ``OSError``/
        ``ValueError``/``ValidationError``。
    副作用：
        只有传入路径时读取本地 YAML；不写文件、不创建运行目录。
    """

    if isinstance(source, PaperProtocolConfig):
        explicit = source.model_dump(mode="json", exclude_unset=True)
        config = source
        source_label = "api_config"
    elif isinstance(source, Mapping):
        explicit = _plain_json_value(dict(source))
        config = PaperProtocolConfig.model_validate(explicit)
        source_label = "user_config"
    else:
        config_path = Path(source)
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, Mapping):
            raise ValueError(
                f"Paper protocol YAML {config_path} must contain a top-level mapping."
            )
        explicit = _plain_json_value(dict(loaded))
        config = PaperProtocolConfig.model_validate(explicit)
        source_label = f"yaml:{config_path}"
    resolved = config.model_dump(mode="json")
    encoded = json.dumps(resolved, ensure_ascii=False, sort_keys=True).encode("utf-8")
    explicit_paths = set(_flatten_leaves(explicit))
    provenance: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for leaf_path, value in _flatten_leaves(resolved).items():
        provenance[leaf_path] = (
            {
                "source": (
                    source_label
                    if leaf_path in explicit_paths
                    else "paper_default"
                ),
                "value": value,
            },
        )
    return ResolvedPaperProtocolConfig(
        config=config,
        resolved_config=resolved,
        provenance=provenance,
        config_hash=hashlib.sha256(encoded).hexdigest()[:16],
    )


@dataclass(frozen=True)
class PaperProtocolResult:
    """P3 正常协议完成后的运行目录与审计产物路径。

    参数：
        run_dir/config_hash: 本次运行目录与解析配置身份。
        baseline_checkpoint_paths: 基线名到统一可重放 checkpoint。
        score_paths: 基线名到 detection/attribution/frozen-normal 逐时刻 CSV。
        replay_paths: 基线名到 checkpoint 重放一致性 JSON。
        calibration_paths: 基线名到 detection/attribution 校准 JSON。
        protocol_paths: P2 split、fit ledger 和 bundle 三份协议产物。
    重要约束：
        该结果只证明正常数据流水线闭环；不含正式故障指标，也不证明论文方法优越。
    """

    run_dir: Path
    config_hash: str
    baseline_checkpoint_paths: Mapping[str, Path]
    score_paths: Mapping[str, Path]
    replay_paths: Mapping[str, Path]
    calibration_paths: Mapping[str, Mapping[str, Path]]
    protocol_paths: Mapping[str, Path]

    def __post_init__(self) -> None:
        """复制并冻结所有路径映射。

        参数：
            无。
        返回：
            无。
        异常：
            无；路径存在性由 ``run_normal`` 在构造前保证。
        副作用：
            只替换为只读映射。
        """

        for field_name in (
            "baseline_checkpoint_paths",
            "score_paths",
            "replay_paths",
            "protocol_paths",
        ):
            object.__setattr__(
                self,
                field_name,
                MappingProxyType(dict(getattr(self, field_name))),
            )
        object.__setattr__(
            self,
            "calibration_paths",
            MappingProxyType(
                {
                    name: MappingProxyType(dict(paths))
                    for name, paths in self.calibration_paths.items()
                }
            ),
        )


@dataclass
class _BaselineRuntime:
    """编排期间一条基线的内存状态与阶段结果。"""

    baseline: PaperBaseline
    checkpoint_path: Path | None = None
    scaler: MonitoringScoreScaler | None = None
    stage_scores: dict[StageName, BaselineScoreBatch] = field(default_factory=dict)
    detection_calibration: EpisodeCalibrationResult | None = None
    attribution_calibration: EpisodeCalibrationResult | None = None


class PaperProtocolExperiment:
    """独立于通用 ``Experiment.run`` 的 P3 五阶段正常协议编排器。

    编排器按阶段跨全部基线推进，而不是逐基线跑完整生命周期。这样第一条基线不能在
    其他模型尚未冻结时提前查看校准段，也能让 P2 账本形成清晰的全局冻结顺序。
    """

    def __init__(
        self,
        bundle: PaperDataBundle,
        config: PaperProtocolConfig | ResolvedPaperProtocolConfig | Mapping[str, Any] | str | Path,
    ) -> None:
        """绑定一个尚未冻结的 P2 bundle 和解析后的 P3 配置。

        参数：
            bundle: 只含五段正常数据、可选封存故障范围和访问账本的数据 bundle。
            config: 严格配置、已解析配置、映射或 YAML 路径。
        返回：
            无。
        异常：
            配置非法时传播解析异常；bundle 已冻结但不属于本实例时在 ``run_normal`` 拒绝。
        副作用：
            配置路径可能被读取；不访问任何阶段、不创建运行目录。
        """

        self.bundle = bundle
        self.resolved = (
            config
            if isinstance(config, ResolvedPaperProtocolConfig)
            else resolve_paper_protocol_config(config)
        )
        self._normal_result: PaperProtocolResult | None = None
        self._runtimes: dict[str, _BaselineRuntime] = {}

    def run_normal(self) -> PaperProtocolResult:
        """执行三类基线的训练、估计、两次校准和冻结正常重放。

        参数：
            无。
        返回：
            含完整产物路径的 ``PaperProtocolResult``；同一实例重复调用幂等返回首次结果。
        异常：
            bundle 已被其他流程冻结、阶段访问顺序非法、训练/评分/重放失败或 I/O 失败时
            传播 ``ProtocolAccessError``、``ValueError``、PyTorch 或 ``OSError``。
        副作用：
            创建运行目录；训练模型；写配置、checkpoint、历史、校准、分数、重放和 P2
            协议产物；最后冻结账本与正常协议。全程不请求故障数组。
        """

        if self._normal_result is not None:
            return self._normal_result
        if self.bundle.protocol_frozen:
            raise ProtocolAccessError(
                "Paper normal protocol is already frozen by another experiment instance."
            )
        config = self.resolved.config
        store = ArtifactStore(config.artifact_root, config.run_name)
        store.save_yaml("resolved_config.yaml", dict(self.resolved.resolved_config))
        store.save_json(
            "provenance.json",
            {
                path: [dict(record) for record in records]
                for path, records in self.resolved.provenance.items()
            },
        )
        store.save_json(
            "config_identity.json",
            {
                "config_hash": self.resolved.config_hash,
                "protocol": "paper_p3_normal_baselines",
                "mode": config.mode,
                "git_commit": _current_git_commit(),
                "data_hash": self.bundle.split_result.data_hash,
                "source_hash": self.bundle.split_result.source_hash,
                "split_hash": self.bundle.split_result.split_hash,
                "stage_window_hashes": {
                    stage.value: self.bundle.split_result.stage(stage).window_hash
                    for stage in StageName
                },
            },
        )
        checkpoint_paths = self._fit_all_baselines(store)
        self._fit_all_monitoring_scalers(store)
        detection_paths = self._calibrate_all(
            store,
            stage=StageName.DETECTION_CALIBRATION,
            purpose=FitPurpose.DETECTION_QUANTILE,
            risk_level=config.detection_risk_level,
        )
        attribution_paths = self._calibrate_all(
            store,
            stage=StageName.ATTRIBUTION_CALIBRATION,
            purpose=FitPurpose.ATTRIBUTION_QUANTILE,
            risk_level=config.attribution_risk_level,
        )
        score_paths, replay_paths = self._run_all_frozen_normal_replays(store)
        self.bundle.freeze_protocol(self.bundle.split_result.split_hash)
        protocol_paths = self.bundle.save_protocol_artifacts(store.resolve("protocol"))
        calibration_paths = {
            name: {
                StageName.DETECTION_CALIBRATION.value: detection_paths[name],
                StageName.ATTRIBUTION_CALIBRATION.value: attribution_paths[name],
            }
            for name in self._runtimes
        }
        store.save_json(
            "summary.json",
            {
                "protocol": "paper_p3_normal_baselines",
                "config_hash": self.resolved.config_hash,
                "split_hash": self.bundle.split_result.split_hash,
                "protocol_frozen": self.bundle.protocol_frozen,
                "fault_data_accessed": self.bundle.fault_accessed,
                "baseline_names": list(self._runtimes),
                "claims": {
                    "pipeline_operational": True,
                    "dynamic_threshold_implemented": False,
                    "paper_method_implemented": False,
                    "formal_fault_results": False,
                },
            },
        )
        result = PaperProtocolResult(
            run_dir=store.path,
            config_hash=self.resolved.config_hash,
            baseline_checkpoint_paths=checkpoint_paths,
            score_paths=score_paths,
            replay_paths=replay_paths,
            calibration_paths=calibration_paths,
            protocol_paths=protocol_paths,
        )
        self._normal_result = result
        return result

    def request_frozen_fault_test(self) -> np.ndarray:
        """在正常协议完成后委托 P2 门禁返回封存故障数组。

        参数：
            无。
        返回：
            P2 ``PaperDataBundle`` 返回的故障数组副本。
        异常：
            正常运行尚未完成时抛出 ``ProtocolAccessError``；之后仍由 P2 检查是否配置
            故障、normal/fault 来源 hash 和 ``verified`` 许可。
        副作用：
            成功时只把 bundle 的 ``fault_accessed`` 标记为真；P3 不计算或保存故障指标。
        """

        if self._normal_result is None:
            raise ProtocolAccessError(
                "Frozen fault access requires the normal protocol run to complete first."
            )
        if self.resolved.config.mode != "frozen":
            raise ProtocolAccessError(
                "Frozen fault access requires PaperProtocolConfig mode='frozen'. "
                "Development runs are normal-only."
            )
        return self.bundle.request_frozen_fault_test()

    def _fit_all_baselines(self, store: ArtifactStore) -> dict[str, Path]:
        """在任何估计或校准访问前训练并冻结全部基线 checkpoint。

        参数：
            store: 当前有界运行目录。
        返回：
            基线名到统一 checkpoint 路径。
        异常：
            拟合、保存或账本冻结错误原样传播。
        副作用：
            读取 train 阶段副本，训练模型并写 checkpoint/历史。
        """

        stage_slice = self.bundle.split_result.stage(StageName.TRAIN)
        raw_indices = np.asarray(stage_slice.raw_indices, dtype=np.int64)
        paths: dict[str, Path] = {}
        for baseline_config in self.resolved.config.baselines:
            object_id = f"{baseline_config.name}.model"
            train_data = self.bundle.data_for_fit(
                object_id,
                FitPurpose.MODEL_PARAMETERS,
            )
            baseline = build_paper_baseline(baseline_config)
            fit_result = baseline.fit(
                train_data,
                raw_indices,
                checkpoint_dir=store.resolve(
                    Path("baselines") / baseline_config.name / "trainer_checkpoints"
                ),
                device=self.resolved.config.device,
            )
            checkpoint_path = baseline.save_checkpoint(
                store.resolve(Path("baselines") / baseline_config.name / "checkpoint.pt")
            )
            self.bundle.fit_access_ledger.freeze_record(
                object_id,
                sha256_file(checkpoint_path),
            )
            if fit_result.history:
                store.save_table(
                    Path("baselines") / baseline_config.name / "training_history.csv",
                    fit_result.history,
                )
            self._runtimes[baseline_config.name] = _BaselineRuntime(
                baseline=baseline,
                checkpoint_path=checkpoint_path,
            )
            paths[baseline_config.name] = checkpoint_path
        return paths

    def _fit_all_monitoring_scalers(self, store: ArtifactStore) -> None:
        """在 estimate 阶段拟合并冻结全部监测分数 RMS 尺度。

        参数：
            store: 当前有界运行目录。
        返回：
            无。
        异常：
            评分、流名称或账本错误原样传播。
        副作用：
            每条基线登记一次 estimate 访问并写尺度 JSON。
        """

        stage_slice = self.bundle.split_result.stage(StageName.ESTIMATE)
        raw_indices = np.asarray(stage_slice.raw_indices, dtype=np.int64)
        for name, runtime in self._runtimes.items():
            object_id = f"{name}.monitoring_score_scaler"
            data = self.bundle.data_for_fit(
                object_id,
                FitPurpose.MONITORING_SCORE_SCALER,
            )
            raw_scores = runtime.baseline.score(
                data,
                raw_indices,
                device=self.resolved.config.device,
            )
            runtime.scaler = MonitoringScoreScaler.fit(raw_scores)
            path = store.save_json(
                Path("baselines") / name / "monitoring_score_scaler.json",
                runtime.scaler.manifest(),
            )
            self.bundle.fit_access_ledger.freeze_record(object_id, sha256_file(path))

    def _calibrate_all(
        self,
        store: ArtifactStore,
        *,
        stage: StageName,
        purpose: FitPurpose,
        risk_level: float,
    ) -> dict[str, Path]:
        """在一个校准阶段完成全部基线后才允许进入下一阶段。

        参数：
            store: 当前有界运行目录。
            stage: detection 或 attribution 校准阶段。
            purpose: 与阶段严格匹配的账本用途。
            risk_level: 预指定 episode 风险水平。
        返回：
            基线名到校准 JSON 路径。
        异常：
            阶段/用途不匹配、尺度未冻结或校准失败时抛出 ``ValueError``/
            ``RuntimeError``/账本异常。
        副作用：
            每条基线登记当前校准阶段，写校准产物并冻结分位记录。
        """

        expected_purpose = {
            StageName.DETECTION_CALIBRATION: FitPurpose.DETECTION_QUANTILE,
            StageName.ATTRIBUTION_CALIBRATION: FitPurpose.ATTRIBUTION_QUANTILE,
        }.get(stage)
        if expected_purpose is not purpose:
            raise ValueError(
                f"Calibration stage {stage.value!r} does not match purpose {purpose.value!r}."
            )
        stage_slice = self.bundle.split_result.stage(stage)
        paths: dict[str, Path] = {}
        for name, runtime in self._runtimes.items():
            if runtime.scaler is None:
                raise RuntimeError(f"Monitoring score scaler for {name!r} is not frozen.")
            object_id = f"{name}.q_det" if stage is StageName.DETECTION_CALIBRATION else f"{name}.q_attr"
            data = self.bundle.data_for_fit(object_id, purpose)
            scores = runtime.scaler.transform(
                _score_complete_calibration_episodes(
                    runtime.baseline,
                    data,
                    stage_slice,
                    device=self.resolved.config.device,
                )
            )
            calibration = EpisodeMaximumCalibrator(risk_level=risk_level).fit(
                scores,
                stage_slice,
            )
            path = store.save_json(
                Path("baselines") / name / f"{stage.value}.json",
                calibration.manifest(),
            )
            self.bundle.fit_access_ledger.freeze_record(object_id, sha256_file(path))
            runtime.stage_scores[stage] = scores
            if stage is StageName.DETECTION_CALIBRATION:
                runtime.detection_calibration = calibration
            else:
                runtime.attribution_calibration = calibration
            paths[name] = path
        return paths

    def _run_all_frozen_normal_replays(
        self,
        store: ArtifactStore,
    ) -> tuple[dict[str, Path], dict[str, Path]]:
        """在冻结正常段验证 checkpoint 重放并保存逐时刻分数。

        参数：
            store: 当前有界运行目录。
        返回：
            ``(score_paths, replay_paths)``。
        异常：
            checkpoint/尺度/检测分位缺失或重放分数不一致时抛出
            ``RuntimeError``/``ProtocolAccessError``。
        副作用：
            登记 frozen-normal 诊断；读取 checkpoint；写 replay、CSV、诊断 JSON；成功后
            冻结每条诊断账本记录。
        """

        stage = StageName.FROZEN_NORMAL_TEST
        stage_slice = self.bundle.split_result.stage(stage)
        raw_indices = np.asarray(stage_slice.raw_indices, dtype=np.int64)
        score_paths: dict[str, Path] = {}
        replay_paths: dict[str, Path] = {}
        for name, runtime in self._runtimes.items():
            if (
                runtime.checkpoint_path is None
                or runtime.scaler is None
                or runtime.detection_calibration is None
                or runtime.attribution_calibration is None
            ):
                raise RuntimeError(f"Baseline runtime {name!r} is incomplete before frozen replay.")
            object_id = f"{name}.frozen_normal_replay"
            data = self.bundle.data_for_fit(
                object_id,
                FitPurpose.FROZEN_NORMAL_DIAGNOSTIC,
            )
            scores = runtime.scaler.transform(
                runtime.baseline.score(
                    data,
                    raw_indices,
                    device=self.resolved.config.device,
                )
            )
            restored = load_paper_baseline(
                runtime.checkpoint_path,
                device=self.resolved.config.device,
            )
            replay_scores = runtime.scaler.transform(
                restored.score(
                    data,
                    raw_indices,
                    device=self.resolved.config.device,
                )
            )
            matches, maximum_delta = _score_batches_match(scores, replay_scores)
            replay_path = store.save_json(
                Path("baselines") / name / "checkpoint_replay.json",
                {
                    "baseline": name,
                    "checkpoint_sha256": sha256_file(runtime.checkpoint_path),
                    "stage": stage.value,
                    "matches": matches,
                    "absolute_tolerance": 1e-7,
                    "maximum_absolute_delta": maximum_delta,
                    "score_hash": _score_batch_hash(scores),
                    "replay_score_hash": _score_batch_hash(replay_scores),
                },
            )
            if not matches:
                raise ProtocolAccessError(
                    f"Checkpoint replay for baseline {name!r} did not reproduce frozen-normal "
                    f"scores. maximum_absolute_delta={maximum_delta}."
                )
            runtime.stage_scores[stage] = scores
            score_path = store.save_table(
                Path("scores") / f"{name}.csv",
                _score_rows(
                    name,
                    runtime.stage_scores,
                    runtime.detection_calibration,
                ),
            )
            diagnostic_path = store.save_json(
                Path("baselines") / name / "frozen_normal_diagnostic.json",
                {
                    "baseline": name,
                    "stage": stage.value,
                    "score_hash": _score_batch_hash(scores),
                    "checkpoint_replay_path": str(replay_path.relative_to(store.path)),
                    "checkpoint_replay_matches": True,
                    "dynamic_threshold": False,
                    "formal_fault_metrics": False,
                },
            )
            self.bundle.fit_access_ledger.freeze_record(
                object_id,
                sha256_file(diagnostic_path),
            )
            score_paths[name] = score_path
            replay_paths[name] = replay_path
        return score_paths, replay_paths


def _score_rows(
    baseline_name: str,
    stage_scores: Mapping[StageName, BaselineScoreBatch],
    detection_calibration: EpisodeCalibrationResult,
) -> Sequence[Mapping[str, Any]]:
    """展开三个正常评分阶段为机器可读逐时刻长表。

    参数：
        baseline_name: 产物基线名。
        stage_scores: detection、attribution 和 frozen-normal 分数。
        detection_calibration: 冻结检测阈值；报警始终使用它而不是 attribution 阈值。
    返回：
        可交给 ``ArtifactStore.save_table`` 的长表记录。
    异常：
        阶段缺失或分数流与检测阈值不一致时抛出 ``ValueError``。
    副作用：
        无。
    """

    required_stages = (
        StageName.DETECTION_CALIBRATION,
        StageName.ATTRIBUTION_CALIBRATION,
        StageName.FROZEN_NORMAL_TEST,
    )
    missing = [stage.value for stage in required_stages if stage not in stage_scores]
    if missing:
        raise ValueError(f"Score artifact is missing required normal stages: {missing}.")
    rows: list[Mapping[str, Any]] = []
    for stage in required_stages:
        batch = stage_scores[stage]
        if set(batch.streams) != set(detection_calibration.thresholds):
            raise ValueError(
                f"Score streams for stage {stage.value!r} do not match detection thresholds."
            )
        for stream_name in sorted(batch.streams):
            threshold = detection_calibration.thresholds[stream_name]
            for raw_index, score in zip(
                batch.raw_indices,
                batch.streams[stream_name],
                strict=True,
            ):
                rows.append(
                    {
                        "baseline": baseline_name,
                        "stage": stage.value,
                        "raw_index": int(raw_index),
                        "score_name": stream_name,
                        "score": float(score),
                        "threshold": threshold.value,
                        "alarm": bool(score > threshold.value),
                        "threshold_kind": threshold.kind,
                        "dynamic_threshold": threshold.dynamic,
                    }
                )
    return rows


def _score_batches_match(
    expected: BaselineScoreBatch,
    actual: BaselineScoreBatch,
) -> tuple[bool, float]:
    """比较 checkpoint 前后分数与原始索引。

    参数：
        expected/actual: 同一阶段、同一尺度下的两份分数。
    返回：
        ``(是否在绝对 1e-7 内一致, 全流最大绝对差)``。
    异常：
        无；结构不一致直接返回假和正无穷。
    副作用：
        无。
    """

    if (
        not np.array_equal(expected.raw_indices, actual.raw_indices)
        or set(expected.streams) != set(actual.streams)
    ):
        return False, math.inf
    maximum_delta = max(
        float(np.max(np.abs(expected.streams[name] - actual.streams[name])))
        if len(expected.streams[name])
        else 0.0
        for name in expected.streams
    )
    matches = all(
        np.allclose(
            expected.streams[name],
            actual.streams[name],
            rtol=0.0,
            atol=1e-7,
        )
        for name in expected.streams
    )
    return matches, maximum_delta


def _score_batch_hash(scores: BaselineScoreBatch) -> str:
    """计算原始索引、流名称、形状和值共同决定的 SHA-256。

    参数：
        scores: 已冻结分数批次。
    返回：
        64 位小写十六进制摘要。
    异常：
        无。
    副作用：
        无。
    """

    digest = hashlib.sha256()
    digest.update(np.asarray(scores.raw_indices, dtype="<i8").tobytes(order="C"))
    for name in sorted(scores.streams):
        encoded_name = name.encode("utf-8")
        values = np.asarray(scores.streams[name], dtype="<f8")
        digest.update(len(encoded_name).to_bytes(4, "little"))
        digest.update(encoded_name)
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes(order="C"))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    """对 JSON 兼容值计算稳定 SHA-256。

    参数：
        value: 不含凭据或运行时对象的 JSON 兼容结构；映射键会排序。
    返回：
        使用 UTF-8 和紧凑分隔符编码得到的 64 位小写十六进制摘要。
    异常：
        值不能由标准 ``json`` 编码时传播 ``TypeError``。
    副作用：
        无。
    """

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _current_git_commit() -> str:
    """读取当前源码仓库的 40 位 Git commit，供论文运行追溯。

    参数：
        无。
    返回：
        包含本模块的最近 Git 仓库 ``HEAD`` 的 40 位小写十六进制对象名。
    异常：
        找不到仓库根、Git 命令失败或返回值不是完整 commit 时抛出 ``RuntimeError``。
        论文运行不能把未知源码版本静默写成可追溯产物。
    副作用：
        启动一次只读 ``git rev-parse HEAD`` 子进程；不修改索引、分支或工作树。
    """

    repository_root = next(
        (
            parent
            for parent in Path(__file__).resolve().parents
            if (parent / ".git").exists()
        ),
        None,
    )
    if repository_root is None:
        raise RuntimeError("Paper protocol cannot locate the Git repository root.")
    completed = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    commit = completed.stdout.strip().lower()
    if completed.returncode != 0 or len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown Git error"
        raise RuntimeError(
            f"Paper protocol cannot resolve a full Git commit. Git output: {detail}"
        )
    return commit


def _plain_json_value(value: Any) -> Any:
    """递归把配置对象规范化为 JSON 兼容值。

    参数：
        value: ``Path``、Pydantic 配置、映射、非字符串序列或 JSON 标量。
    返回：
        路径转字符串、配置转 ``model_dump(mode='json')``、容器递归复制后的值。
    异常：
        自定义 ``model_dump`` 或容器迭代失败时原样传播对应异常。
    副作用：
        无；不修改原对象。该规范化结果参与配置 hash，分支顺序不可随意改变。
    """

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _plain_json_value(model_dump(mode="json"))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain_json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain_json_value(item) for item in value]
    return value


def _flatten_leaves(value: Any, prefix: str = "") -> dict[str, Any]:
    """把嵌套配置展开为 provenance 使用的稳定叶路径。

    参数：
        value: 已经 JSON 兼容的嵌套映射、非字符串序列或标量。
        prefix: 递归累计的点分路径；顶层调用应为空字符串。
    返回：
        从 ``a.b.0`` 形式叶路径到标量值的新字典；空容器不产生虚构叶子。
    异常：
        映射或序列迭代失败时原样传播异常。
    副作用：
        无。列表顺序与映射遍历只决定路径集合；最终配置 hash 另行使用排序 JSON。
    """

    leaves: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            leaves.update(_flatten_leaves(item, path))
        return leaves
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            path = f"{prefix}.{index}" if prefix else str(index)
            leaves.update(_flatten_leaves(item, path))
        return leaves
    leaves[prefix] = value
    return leaves


__all__ = [
    "CalibrationEpisode",
    "EpisodeCalibrationResult",
    "EpisodeMaximumCalibrator",
    "MonitoringScoreScaler",
    "PaperProtocolConfig",
    "PaperProtocolExperiment",
    "PaperProtocolResult",
    "ResolvedPaperProtocolConfig",
    "StaticThreshold",
    "resolve_paper_protocol_config",
]
