"""正式 P10 checkpoint 可恢复的保守 Attention--Koopman--T--S evaluator。

文件用途：
    将已训练 P4 模型、P5 受保护监视器和冻结 guard/阈值状态组合为无标签的正式运行时，
    并通过 Joff 公共 evaluator registry 构造。
主要职责：
    严格验证纯 JSON evaluator state；从 weights-only checkpoint 的 ``state_dict`` 恢复
    P4 模型；逐 episode 重置 P5 状态；输出完整 P4--P9 来源合同。
关键输入与输出：
    构造输入是 checkpoint mapping 与 ``FrozenProtectedEvaluatorState``；运行输入只含
    ``raw_indices + values``，输出 ``FrozenRuntimeEpisodeEvaluation``，不含故障真值。
依赖与副作用：
    依赖 PyTorch、P4 model factory 和 P5 ``ProtectedMonitor``；构造分配模型并载入参数，
    评价只做 CPU 前向与显式状态推进，不读写文件、不访问网络、不修改 checkpoint。
重要约束：
    feature layout 必须与模型通道维数一致且不重叠。当前仓库没有真实 P6 认证 provider，
    因而该运行时只使用 raw guard 和冻结输入依赖阈值；P6/P9 输出明确为 unavailable/
    Uncertified，绝不授权物理 singleton。该保守路径可验证正式信息合同，但不能冒充论文
    方法已经理论饱和或已有真实故障性能。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import math

import numpy as np
import torch
from pydantic import Field, model_validator

from joff.core.config import StrictConfig
from joff.core.factory import build_model, register_evaluator
from joff.evaluation.protected_reference import (
    AnchorGateConfig,
    MonitorRecord,
    MonitorStage,
    ProtectedMonitor,
)

from .frozen_evaluation import (
    FrozenEpisodeInput,
    FrozenRuntimeEpisodeEvaluation,
    FrozenRuntimePointwiseOutput,
)


class FrozenFeatureLayout(StrictConfig):
    """把 canonical 特征列显式映射到 P4 control/measurement/exogenous 通道。"""

    control_indices: tuple[int, ...] = Field(min_length=1)
    measurement_indices: tuple[int, ...] = Field(min_length=1)
    exogenous_indices: tuple[int, ...]

    @model_validator(mode="after")
    def _validate_indices(self) -> "FrozenFeatureLayout":
        """拒绝负索引、重复列和跨角色重叠。"""

        all_indices = (
            *self.control_indices,
            *self.measurement_indices,
            *self.exogenous_indices,
        )
        if any(
            isinstance(index, bool) or index < 0
            for index in all_indices
        ):
            raise ValueError("Frozen feature indices must be nonnegative integers.")
        if len(set(all_indices)) != len(all_indices):
            raise ValueError("Frozen feature roles cannot reuse the same column.")
        return self


class FrozenGuardThresholdState(StrictConfig):
    """未认证 guard 分支的冻结输入依赖阈值分账。"""

    floor: float = Field(gt=0.0)
    gamma_anc: float = Field(ge=0.0)
    deterministic_intercept: float = Field(ge=0.0)
    input_l1_weight: float = Field(ge=0.0)
    stochastic_quantile: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _validate_finite(self) -> "FrozenGuardThresholdState":
        """所有分账必须有限，不能把 NaN 藏进冻结 state。"""

        if not all(
            math.isfinite(float(value))
            for value in (
                self.floor,
                self.gamma_anc,
                self.deterministic_intercept,
                self.input_l1_weight,
                self.stochastic_quantile,
            )
        ):
            raise ValueError("Frozen guard threshold components must be finite.")
        return self


class FrozenProtectedEvaluatorState(StrictConfig):
    """正式 evaluator 的全部纯 JSON、正常数据冻结状态。"""

    schema_version: Literal[1]
    feature_layout: FrozenFeatureLayout
    anchor_gate: AnchorGateConfig
    eligibility_center: tuple[float, ...] = Field(min_length=1)
    eligibility_scale: tuple[float, ...] = Field(min_length=1)
    score_scale: float = Field(gt=0.0)
    branch_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,127}$")
    mode: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,127}$")
    normal_family_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,127}$")
    unresolved_family_id: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,127}$"
    )
    threshold: FrozenGuardThresholdState

    @model_validator(mode="after")
    def _validate_estimate_state(self) -> "FrozenProtectedEvaluatorState":
        """核对 eligibility 向量维数、正尺度和有限性。"""

        measurement_count = len(self.feature_layout.measurement_indices)
        if (
            len(self.eligibility_center) != measurement_count
            or len(self.eligibility_scale) != measurement_count
        ):
            raise ValueError(
                "Frozen eligibility center/scale must match measurement feature count."
            )
        if not all(math.isfinite(float(value)) for value in self.eligibility_center):
            raise ValueError("Frozen eligibility center must be finite.")
        if not all(
            math.isfinite(float(value)) and float(value) > 0.0
            for value in self.eligibility_scale
        ):
            raise ValueError("Frozen eligibility scale must be finite and positive.")
        if not math.isfinite(self.score_scale):
            raise ValueError("Frozen score_scale must be finite and positive.")
        if self.normal_family_id == self.unresolved_family_id:
            raise ValueError("Normal and unresolved family ids must differ.")
        return self


@register_evaluator("protected_koopman_ts_frozen", replace=True)
class FrozenProtectedKoopmanTSEvaluator:
    """从 checkpoint 恢复 P4/P5，并对 P6--P9 缺失证据保守失败。"""

    # 当前实现没有执行认证算子、完整多步堆叠与结构化隔离，因此正式命令必须在
    # claim 之前读取该能力标记并 fail closed；不能因逐点输出写了 Uncertified 就消费故障。
    formal_pipeline_complete = False

    def __init__(
        self,
        *,
        state: Mapping[str, Any],
        checkpoint: Mapping[str, Any],
    ) -> None:
        """严格恢复模型、feature layout 和正常 estimate 状态。

        参数：
            state: checkpoint evaluator envelope 中的纯 JSON mapping。
            checkpoint: ``torch.load(..., weights_only=True)`` 返回的 mapping。
        返回：
            无。
        异常：
            state/checkpoint/model config/参数或维数不一致时抛出 Pydantic、
            ``TypeError``、``KeyError`` 或 ``ValueError``。
        副作用：
            构造 CPU 模型、载入 ``state_dict`` 并切换为 eval；不读取文件或随机采样。
        """

        self.state = FrozenProtectedEvaluatorState.model_validate(dict(state))
        if not isinstance(checkpoint, Mapping):
            raise TypeError("Frozen protected evaluator checkpoint must be a mapping.")
        raw_model_config = checkpoint.get("config")
        if not isinstance(raw_model_config, Mapping):
            raise ValueError("Frozen protected checkpoint is missing model config.")
        if "model" in raw_model_config:
            raw_model_config = raw_model_config["model"]
        if not isinstance(raw_model_config, Mapping):
            raise ValueError("Frozen protected model config must be a mapping.")
        model = build_model(dict(raw_model_config))
        model_state_dict = checkpoint.get("model_state_dict")
        if not isinstance(model_state_dict, Mapping):
            raise ValueError("Frozen protected checkpoint is missing model_state_dict.")
        model.load_state_dict(dict(model_state_dict), strict=True)
        model.to(torch.device("cpu"))
        model.eval()
        self.model = model
        self.monitor = ProtectedMonitor(model, self.state.anchor_gate)
        self._validate_layout()

    def evaluate_episode(
        self,
        episode: FrozenEpisodeInput,
    ) -> FrozenRuntimeEpisodeEvaluation:
        """在一个无标签 episode 上推进冻结 P4/P5 guard 路径。

        参数：
            episode: 只含连续 raw index 与过程变量的只读特征视图。
        返回：
            每行一个无真值运行时输出；warmup/candidate 阶段阈值为正无穷且不报警。
        异常：
            输入类型、feature count、模型前向或数值状态非法时抛出对应异常。
        副作用：
            仅在当前调用内推进显式不可变 monitor state；不跨 episode 缓存状态。
        """

        if not isinstance(episode, FrozenEpisodeInput):
            raise TypeError("Frozen protected evaluator requires FrozenEpisodeInput.")
        expected_features = self._expected_feature_count()
        if episode.values.shape[1] != expected_features:
            raise ValueError(
                "Frozen evaluator feature count differs from the frozen feature layout."
            )
        monitor_state = self.monitor.initial_state()
        outputs: list[FrozenRuntimePointwiseOutput] = []
        with torch.no_grad():
            for raw_index, row in zip(
                episode.raw_indices.tolist(),
                episode.values,
                strict=True,
            ):
                control = self._select(row, self.state.feature_layout.control_indices)
                measurement = self._select(
                    row,
                    self.state.feature_layout.measurement_indices,
                )
                exogenous = self._select(
                    row,
                    self.state.feature_layout.exogenous_indices,
                )
                eligibility_score = self._eligibility_score(measurement)
                step = self.monitor.step(
                    monitor_state,
                    MonitorRecord(
                        raw_index=int(raw_index),
                        episode_id="runtime-episode",
                        stage=MonitorStage.FROZEN_FAULT_TEST,
                        control=control,
                        measurement=measurement,
                        exogenous=exogenous,
                        anchor_eligibility_score=eligibility_score,
                    ),
                )
                monitor_state = step.state
                rollout = step.output.protected_rollout
                prediction = None if rollout is None else rollout.predicted_measurement
                residual = (
                    ()
                    if prediction is None
                    else tuple(
                        observed - predicted
                        for observed, predicted in zip(
                            measurement,
                            prediction,
                            strict=True,
                        )
                    )
                )
                score = (
                    0.0
                    if not residual
                    else float(np.linalg.norm(residual)) / self.state.score_scale
                )
                gamma_det = (
                    self.state.threshold.deterministic_intercept
                    + self.state.threshold.input_l1_weight
                    * sum(abs(value) for value in control)
                )
                threshold = (
                    math.inf
                    if prediction is None
                    else (
                        self.state.threshold.floor
                        + self.state.threshold.gamma_anc
                        + gamma_det
                        + self.state.threshold.stochastic_quantile
                    )
                )
                alarm = score > threshold
                candidate_ids = (
                    (self.state.unresolved_family_id,)
                    if alarm
                    else (self.state.normal_family_id,)
                )
                isolation_outcome = (
                    "Uncertified" if alarm else "Normal-compatible"
                )
                outputs.append(
                    FrozenRuntimePointwiseOutput(
                        raw_index=int(raw_index),
                        detection_score=score,
                        detection_threshold=threshold,
                        alarm=alarm,
                        branch_id=self.state.branch_id,
                        mode=self.state.mode,
                        normal_family_id=self.state.normal_family_id,
                        candidate_ids=candidate_ids,
                        isolation_outcome=isolation_outcome,
                        reported_family=None,
                        isolation_certified=False,
                        suppression_reason=None,
                        method_outputs={
                            "prediction": {
                                "one_step": (
                                    None
                                    if prediction is None
                                    else list(prediction)
                                ),
                                "multi_step": (
                                    []
                                    if rollout is None
                                    else [
                                        list(item)
                                        for item in rollout.prediction_trajectory
                                    ]
                                ),
                            },
                            "rule_weights": {
                                "status": "not-exposed-by-protected-monitor"
                            },
                            "monitor": step.output.to_dict(),
                            "protected_state": monitor_state.to_dict(),
                            "residual": {"stacked": list(residual)},
                            "operator": {
                                "status": "unavailable",
                                "reason": "no-certified-provider",
                            },
                            "branch_statistics": {
                                self.state.branch_id: score
                            },
                            "threshold": {
                                "floor": self.state.threshold.floor,
                                "gamma_anc": self.state.threshold.gamma_anc,
                                "gamma_det": gamma_det,
                                "stochastic": (
                                    self.state.threshold.stochastic_quantile
                                ),
                                "value": (
                                    "infinity"
                                    if math.isinf(threshold)
                                    else threshold
                                ),
                                "alarm": alarm,
                            },
                            "isolation": {
                                "outcome": isolation_outcome,
                                "certified": False,
                                "reason": "no-certified-operator-or-signature",
                            },
                        },
                    )
                )
        return FrozenRuntimeEpisodeEvaluation(outputs=tuple(outputs))

    def _validate_layout(self) -> None:
        """核对 P4 配置通道数和 feature layout。"""

        model_config = getattr(self.model, "protected_config", None)
        if model_config is None:
            raise TypeError("Frozen evaluator requires a protected P4 model.")
        layout = self.state.feature_layout
        expected = (
            (len(layout.control_indices), int(model_config.control_dim), "control"),
            (
                len(layout.measurement_indices),
                int(model_config.measurement_dim),
                "measurement",
            ),
            (
                len(layout.exogenous_indices),
                int(model_config.exogenous_dim),
                "exogenous",
            ),
        )
        for observed, configured, name in expected:
            if observed != configured:
                raise ValueError(
                    f"Frozen {name} feature count differs from the P4 model config."
                )
        used = sorted(
            (
                *layout.control_indices,
                *layout.measurement_indices,
                *layout.exogenous_indices,
            )
        )
        if used != list(range(len(used))):
            raise ValueError(
                "Frozen feature layout must cover every input column exactly once."
            )

    def _expected_feature_count(self) -> int:
        """返回冻结 layout 覆盖的 canonical feature 数。"""

        layout = self.state.feature_layout
        return (
            len(layout.control_indices)
            + len(layout.measurement_indices)
            + len(layout.exogenous_indices)
        )

    @staticmethod
    def _select(
        row: np.ndarray,
        indices: tuple[int, ...],
    ) -> tuple[float, ...]:
        """按显式角色索引复制一行特征；空外生角色返回空元组。"""

        return tuple(float(row[index]) for index in indices)

    def _eligibility_score(self, measurement: tuple[float, ...]) -> float:
        """按 estimate 冻结中心/正尺度计算无标签 RMS eligibility 分数。"""

        normalized = tuple(
            (value - center) / scale
            for value, center, scale in zip(
                measurement,
                self.state.eligibility_center,
                self.state.eligibility_scale,
                strict=True,
            )
        )
        return math.sqrt(
            sum(value * value for value in normalized) / len(normalized)
        )


__all__ = [
    "FrozenFeatureLayout",
    "FrozenGuardThresholdState",
    "FrozenProtectedEvaluatorState",
    "FrozenProtectedKoopmanTSEvaluator",
]
