"""P5 受保护参考的纯监视状态机与可追溯领域对象。

文件用途：
    在评估层维护论文监视器的严格过去历史、延迟候选锚点、已接受锚点和 episode/stage
    边界，使 P4 正常模型可以在不反馈锚点后真实测量的前提下生成受保护参考。
主要职责：
    定义严格 ``AnchorGateConfig``、不可变在线记录和状态对象，以及
    ``ProtectedMonitor.step`` 纯状态转移；本文件不拟合阈值、不生成最终报警，也不实现
    P6 算子认证、P7 后滤波或 P8 split-conformal 检测。
关键输入与输出：
    输入是一条 ``MonitorRecord`` 和上一时刻 ``MonitorState``；输出
    ``MonitorStepResult``，其中包含新状态与当步公开输出。状态中的历史只保留模型需要
    的固定严格过去窗口，并以原始索引、episode 和五阶段名称维持时间边界。
依赖与副作用：
    依赖 Pydantic 配置校验和一个具有 ``protected_config`` 的 P4 PyTorch 模型。对象构造
    和状态转移不读写文件、不访问网络、不修改模型参数或进程全局状态。
重要约束：
    候选只能在冻结 eligibility gate 下延迟接受；episode、stage 或原始索引连续性变化
    都会重置状态；接受事件只能标为 ``UNVERIFIED`` 或显式声明的覆盖假设，绝不产生
    ``clean`` 字段。最终检测分位不属于本模块的任何状态转移输入。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, cast

import torch
from pydantic import NonNegativeFloat, NonNegativeInt, PositiveInt, model_validator
from torch import nn

from joff.core.config import StrictConfig
from joff.data.paper_protocol import StageName


class MonitorMode(str, Enum):
    """监视器当前可用性模式。

    ``WARMUP`` 表示严格过去历史尚不足，``CANDIDATE`` 表示锚点正在延迟确认，
    ``PROTECTED`` 表示存在可用受保护锚点，``STALE`` 表示锚点已超过冻结寿命。
    枚举不包含报警状态，避免最终检测输出反向改变分数生成路径。
    """

    WARMUP = "warmup"
    CANDIDATE = "candidate"
    PROTECTED = "protected"
    STALE = "stale"


class AnchorCoverageStatus(str, Enum):
    """锚点正常性覆盖声明。

    ``UNVERIFIED`` 是默认且保守的状态；``DECLARED_ASSUMPTION`` 仅表示实验协议另外声明
    了有限延迟覆盖假设。这里故意不提供 ``CLEAN``，因为一段时间无报警不能证明锚点
    一定正常。
    """

    UNVERIFIED = "unverified"
    DECLARED_ASSUMPTION = "declared_assumption"


class MonitorStage(str, Enum):
    """在线监视器允许出现的受控协议范围。

    前五项逐一复用 ``StageName`` 的正常数据语义；``FROZEN_FAULT_TEST`` 是五段之外的
    独立冻结故障范围，不能伪装成 ``FROZEN_NORMAL_TEST``。未知拼写必须在记录进入状态机
    之前失败。
    """

    TRAIN = StageName.TRAIN.value
    ESTIMATE = StageName.ESTIMATE.value
    DETECTION_CALIBRATION = StageName.DETECTION_CALIBRATION.value
    ATTRIBUTION_CALIBRATION = StageName.ATTRIBUTION_CALIBRATION.value
    FROZEN_NORMAL_TEST = StageName.FROZEN_NORMAL_TEST.value
    FROZEN_FAULT_TEST = "frozen_fault_test"

    @classmethod
    def parse(
        cls,
        value: "MonitorStage | StageName | str",
    ) -> "MonitorStage":
        """把正常阶段、监视阶段或精确字符串解析为受控范围。

        参数：
            value: ``MonitorStage``、已有 ``StageName`` 或精确字符串值。
        返回：
            对应的 ``MonitorStage``。
        异常：
            未知拼写时抛出列出全部合法范围的 ``ValueError``。
        副作用：
            无。
        """

        if isinstance(value, cls):
            return value
        normalized = value.value if isinstance(value, StageName) else str(value)
        try:
            return cls(normalized)
        except ValueError as exc:
            legal = ", ".join(stage.value for stage in cls)
            raise ValueError(
                f"Unknown monitor stage {value!r}. Legal options are: {legal}."
            ) from exc


class AnchorGateConfig(StrictConfig):
    """冻结的候选锚点门控参数。

    参数：
        confirmation_delay: 候选创建后必须继续通过 eligibility gate 的记录步数。
        enter_threshold: 创建候选使用的较严格进入阈值。
        exit_threshold: 保持候选使用的退出阈值；必须不小于进入阈值，以形成 hysteresis。
        minimum_reanchor_interval: 两次接受锚点之间允许重新候选的最小记录步数。
        maximum_reference_age: 已接受锚点允许自由展开的最大记录年龄。
    返回：
        冻结、拒绝未知字段的门控配置。
    异常：
        阈值顺序不合法时抛出 Pydantic ``ValidationError``。
    副作用：
        无。配置中故意没有最终检测分位或报警阈值。
    """

    confirmation_delay: PositiveInt
    enter_threshold: NonNegativeFloat
    exit_threshold: NonNegativeFloat
    minimum_reanchor_interval: NonNegativeInt
    maximum_reference_age: PositiveInt

    @model_validator(mode="after")
    def _validate_hysteresis(self) -> "AnchorGateConfig":
        """保证阈值、确认延迟和引用寿命构成可执行的冻结门控。"""

        if self.exit_threshold < self.enter_threshold:
            raise ValueError("exit_threshold must be greater than or equal to enter_threshold.")
        if not math.isfinite(self.enter_threshold) or not math.isfinite(
            self.exit_threshold
        ):
            raise ValueError("Anchor gate thresholds must be finite.")
        if self.confirmation_delay > self.maximum_reference_age:
            raise ValueError(
                "confirmation_delay cannot exceed maximum_reference_age."
            )
        return self


@dataclass(frozen=True)
class MonitorRecord:
    """监视器在一个原始时刻可读取的完整记录。

    参数：
        raw_index: 数据源中的稳定整数索引。
        episode_id/stage: episode 和受控正常/冻结故障协议边界标识。
        control/measurement/exogenous: 当前记录的控制、真实测量和外生工况向量。
        anchor_eligibility_score: 专用于候选锚点门控的冻结分数，不是最终检测统计量。
    返回：
        不可变、可序列化的在线记录。
    异常：
        索引为负、边界标识为空或任一数值非有限时抛出 ``ValueError``。
    副作用：
        无。真实测量只供 data branch 和未来严格过去历史使用，不自动进入 protected branch。
    """

    raw_index: int
    episode_id: str
    stage: MonitorStage
    control: tuple[float, ...]
    measurement: tuple[float, ...]
    exogenous: tuple[float, ...]
    anchor_eligibility_score: float

    def __post_init__(self) -> None:
        """在状态转移前拒绝无法安全重放的记录。"""

        if self.raw_index < 0:
            raise ValueError("raw_index must be non-negative.")
        if (
            not isinstance(self.episode_id, str)
            or not isinstance(self.stage, str)
            or not self.episode_id
            or not self.stage
        ):
            raise ValueError("episode_id and stage must be non-empty.")
        object.__setattr__(
            self,
            "control",
            tuple(float(value) for value in self.control),
        )
        object.__setattr__(
            self,
            "measurement",
            tuple(float(value) for value in self.measurement),
        )
        object.__setattr__(
            self,
            "exogenous",
            tuple(float(value) for value in self.exogenous),
        )
        object.__setattr__(
            self,
            "anchor_eligibility_score",
            float(self.anchor_eligibility_score),
        )
        object.__setattr__(self, "stage", MonitorStage.parse(self.stage))
        values = (
            *self.control,
            *self.measurement,
            *self.exogenous,
            self.anchor_eligibility_score,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("MonitorRecord values must all be finite.")

    def to_dict(self) -> dict[str, Any]:
        """返回只含 JSON 基本类型的确定性表示。"""

        return {
            "raw_index": self.raw_index,
            "episode_id": self.episode_id,
            "stage": self.stage.value,
            "control": list(self.control),
            "measurement": list(self.measurement),
            "exogenous": list(self.exogenous),
            "anchor_eligibility_score": self.anchor_eligibility_score,
        }


@dataclass(frozen=True)
class _RecordedCommand:
    """受保护支路允许保存的记录命令与外生工况。

    该内部对象故意没有 measurement 字段，从数据结构上阻止 post-anchor 真实测量进入
    候选或受保护命令路径。
    """

    raw_index: int
    control: tuple[float, ...]
    exogenous: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        """返回不含真实测量的 JSON 兼容命令表示。"""

        return {
            "raw_index": self.raw_index,
            "control": list(self.control),
            "exogenous": list(self.exogenous),
        }

    @classmethod
    def from_record(cls, record: MonitorRecord) -> "_RecordedCommand":
        """只投影允许进入受保护支路的字段。"""

        return cls(
            raw_index=record.raw_index,
            control=record.control,
            exogenous=record.exogenous,
        )


@dataclass(frozen=True)
class AnchorCandidate:
    """正在延迟确认的候选锚点及其严格过去快照。

    ``age`` 统计创建后继续通过 gate 的记录步数。``history`` 以候选时刻之前结束，因此
    不包含候选时刻的真实测量；后续受保护展开只能从该快照和记录命令恢复。
    """

    raw_index: int
    age: int
    history: tuple[MonitorRecord, ...]
    commands: tuple[_RecordedCommand, ...]

    def to_dict(self) -> dict[str, Any]:
        """返回候选锚点的 JSON 兼容表示。"""

        return {
            "raw_index": self.raw_index,
            "age": self.age,
            "history": [record.to_dict() for record in self.history],
            "commands": [command.to_dict() for command in self.commands],
        }


@dataclass(frozen=True)
class AcceptedAnchor:
    """已通过延迟门控但不声称正常的受保护锚点。"""

    raw_index: int
    history: tuple[MonitorRecord, ...]
    coverage_status: AnchorCoverageStatus = AnchorCoverageStatus.UNVERIFIED

    def to_dict(self) -> dict[str, Any]:
        """返回锚点的 JSON 兼容表示，故意不生成 ``clean`` 字段。"""

        return {
            "raw_index": self.raw_index,
            "history": [record.to_dict() for record in self.history],
            "coverage_status": self.coverage_status.value,
        }


@dataclass(frozen=True)
class MonitorState:
    """一次 ``step`` 调用之间持久化的不可变监视状态。

    参数：
        monitor_identity: 冻结 gate 配置、模型配置和模型 ``state_dict`` 的联合身份哈希。
        episode_id/stage/last_raw_index: 当前时间边界与最后处理的原始索引。
        history: data branch 使用的固定严格过去历史。
        candidate/anchor/anchor_age: 候选、已接受锚点及其记录年龄。
        protected_commands: 从锚点开始且不含 measurement 的控制/外生路径。
        mode/reset_count: 当前可用性模式与累计边界重置次数。
    返回：
        不可变、可序列化并通过 ``content_hash`` 绑定 monitor 身份的状态。
    异常：
        由 ``ProtectedMonitor`` 构造；身份不匹配的状态在 ``step`` 时抛出 ``ValueError``。
    副作用：
        无；边界变化时返回新实例，不原位清空旧状态。
    """

    monitor_identity: str = ""
    episode_id: str | None = None
    stage: MonitorStage | None = None
    last_raw_index: int | None = None
    history: tuple[MonitorRecord, ...] = ()
    candidate: AnchorCandidate | None = None
    anchor: AcceptedAnchor | None = None
    anchor_age: int | None = None
    protected_commands: tuple[_RecordedCommand, ...] = ()
    mode: MonitorMode = MonitorMode.WARMUP
    reset_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """返回可稳定编码为 JSON 的完整状态。"""

        return {
            "monitor_identity": self.monitor_identity,
            "episode_id": self.episode_id,
            "stage": None if self.stage is None else self.stage.value,
            "last_raw_index": self.last_raw_index,
            "history": [record.to_dict() for record in self.history],
            "candidate": None if self.candidate is None else self.candidate.to_dict(),
            "anchor": None if self.anchor is None else self.anchor.to_dict(),
            "anchor_age": self.anchor_age,
            "protected_commands": [
                command.to_dict() for command in self.protected_commands
            ],
            "mode": self.mode.value,
            "reset_count": self.reset_count,
        }

    @property
    def content_hash(self) -> str:
        """返回覆盖全部分数生成状态的 SHA-256 内容哈希。"""

        return _sha256_json(self.to_dict())


@dataclass(frozen=True)
class ProtectedRollout:
    """一个已接受锚点到当前时刻的受保护自由展开。

    参数：
        anchor_raw_index/target_raw_index: 展开的锚点与当前原始索引。
        latent_trajectory: 从 ``z_s`` 到 ``z_t`` 的潜变量序列。
        prediction_trajectory: 从 ``s+1`` 到 ``t`` 的模型输出预测。
        context_trajectory: 各次状态转移实际使用的受保护 context。
        protected_measurement_buffer: 生成当前状态后保留的固定历史测量槽；锚点后的槽位
            只能来自模型解码值。
    返回：
        不可变且可序列化的受保护支路证据。
    异常：
        只由 ``ProtectedMonitor`` 从已校验模型输出构造。
    副作用：
        无；对象不持有 tensor 或模型引用。
    """

    anchor_raw_index: int
    target_raw_index: int
    latent_trajectory: tuple[tuple[float, ...], ...]
    prediction_trajectory: tuple[tuple[float, ...], ...]
    context_trajectory: tuple[tuple[float, ...], ...]
    protected_measurement_buffer: tuple[tuple[float, ...], ...]

    @property
    def latent(self) -> tuple[float, ...]:
        """返回当前目标时刻的受保护潜变量。"""

        return self.latent_trajectory[-1]

    @property
    def predicted_measurement(self) -> tuple[float, ...]:
        """返回当前目标时刻的受保护输出预测。"""

        return self.prediction_trajectory[-1]

    def to_dict(self) -> dict[str, Any]:
        """返回不含 tensor 的 JSON 兼容轨迹。"""

        return {
            "anchor_raw_index": self.anchor_raw_index,
            "target_raw_index": self.target_raw_index,
            "latent_trajectory": [list(row) for row in self.latent_trajectory],
            "prediction_trajectory": [
                list(row) for row in self.prediction_trajectory
            ],
            "context_trajectory": [list(row) for row in self.context_trajectory],
            "protected_measurement_buffer": [
                list(row) for row in self.protected_measurement_buffer
            ],
        }


@dataclass(frozen=True)
class MonitorOutput:
    """单步 data/protected 双支路的公开输出。

    参数：
        monitor_identity: 生成该输出的冻结 monitor 联合身份。
        raw_index/episode_id/stage/mode: 当步时间边界和状态机模式。
        data_latent/data_context: 严格过去真实历史生成的数据支路表示；warmup 时为空。
        protected_rollout: 已接受且未超龄锚点的受保护轨迹，否则为空。
    返回：
        不含 tensor 的不可变单步输出。
    异常：
        episode 为空或 stage 不是受控范围时构造失败。
    副作用：
        无。
    """

    monitor_identity: str
    raw_index: int
    episode_id: str
    stage: MonitorStage
    mode: MonitorMode
    data_latent: tuple[float, ...] | None
    data_context: tuple[float, ...] | None
    protected_rollout: ProtectedRollout | None

    def __post_init__(self) -> None:
        """规范化公开边界字段，阻止 ``replace`` 绕过受控阶段语义。"""

        if not self.episode_id:
            raise ValueError("MonitorOutput episode_id must be non-empty.")
        object.__setattr__(self, "stage", MonitorStage.parse(self.stage))

    def to_dict(self) -> dict[str, Any]:
        """返回单步两个信息支路的 JSON 兼容表示。"""

        return {
            "monitor_identity": self.monitor_identity,
            "raw_index": self.raw_index,
            "episode_id": self.episode_id,
            "stage": self.stage.value,
            "mode": self.mode.value,
            "data_latent": None
            if self.data_latent is None
            else list(self.data_latent),
            "data_context": None
            if self.data_context is None
            else list(self.data_context),
            "protected_rollout": None
            if self.protected_rollout is None
            else self.protected_rollout.to_dict(),
        }


@dataclass(frozen=True)
class MonitorStepResult:
    """``ProtectedMonitor.step`` 的原子返回值。

    参数：
        state: 成功处理当前记录后的新不可变状态。
        output: 同一 monitor、边界和原始索引生成的双支路输出。
    返回：
        供下一次 ``step`` 和 ``MonitorTrace.append`` 使用的配对结果。
    异常：
        对象本身不校验跨字段一致性；trace 追加会 fail closed 核验。
    副作用：
        无。
    """

    state: MonitorState
    output: MonitorOutput


@dataclass(frozen=True)
class MonitorTraceEntry:
    """一条输入记录与其状态转移结果的不可变审计项。

    参数：
        record/state/output: 同一 monitor、原始索引、episode、stage 和 mode 的输入与结果。
    返回：
        可序列化审计项。
    异常：
        应通过 ``MonitorTrace.append`` 构造，矛盾字段会在那里抛出 ``ValueError``。
    副作用：
        无。
    """

    record: MonitorRecord
    state: MonitorState
    output: MonitorOutput

    def to_dict(self) -> dict[str, Any]:
        """返回审计项的 JSON 兼容表示。"""

        return {
            "record": self.record.to_dict(),
            "state": self.state.to_dict(),
            "output": self.output.to_dict(),
        }


@dataclass(frozen=True)
class MonitorTrace:
    """按原始时间顺序累积的不可变受保护监视 trace。

    参数：
        entries: 已完成状态转移的审计项；默认空 trace。
    返回：
        可通过 ``append`` 生成新实例、通过 ``to_dict`` 持久化并用 ``content_hash`` 校验
        重放一致性的轨迹。
    异常：
        追加的记录、输出和状态索引不一致，或同一 trace 内索引不连续时抛出
        ``ValueError``。
    副作用：
        无；``append`` 不修改原 trace。
    """

    monitor_identity: str | None = None
    entries: tuple[MonitorTraceEntry, ...] = ()

    def append(
        self,
        record: MonitorRecord,
        result: MonitorStepResult,
    ) -> "MonitorTrace":
        """追加一条已完成的公开状态转移并返回新 trace。"""

        state = result.state
        output = result.output
        if output.raw_index != record.raw_index or state.last_raw_index != record.raw_index:
            raise ValueError("Trace record, output, and state raw indices must match.")
        if (
            output.episode_id != record.episode_id
            or state.episode_id != record.episode_id
        ):
            raise ValueError("Trace record, output, and state episode_id values must match.")
        if output.stage is not record.stage or state.stage is not record.stage:
            raise ValueError("Trace record, output, and state stage values must match.")
        if output.mode is not state.mode:
            raise ValueError("Trace output mode must match the resulting state mode.")
        entry_identity = state.monitor_identity
        if (
            not entry_identity
            or output.monitor_identity != entry_identity
            or (
                self.monitor_identity is not None
                and self.monitor_identity != entry_identity
            )
        ):
            raise ValueError("Trace entries must share one frozen monitor identity.")
        trace_identity = self.monitor_identity or entry_identity
        if self.entries:
            previous_entry = self.entries[-1]
            previous = previous_entry.record
            same_boundary = (
                previous.episode_id == record.episode_id
                and previous.stage == record.stage
            )
            reset_happened = (
                state.reset_count > previous_entry.state.reset_count
            )
            reset_delta = state.reset_count - previous_entry.state.reset_count
            if reset_delta not in (0, 1):
                raise ValueError("Trace reset_count must be monotonic and increase by at most one.")
            boundary_changed = not same_boundary
            index_gap = record.raw_index != previous.raw_index + 1
            if (boundary_changed or index_gap) != reset_happened:
                raise ValueError(
                    "Trace boundary/index discontinuities must match one state reset."
                )
            if (
                same_boundary
                and not reset_happened
                and record.raw_index != previous.raw_index + 1
            ):
                raise ValueError("Trace raw indices must be consecutive within a boundary.")
        return MonitorTrace(
            monitor_identity=trace_identity,
            entries=(
                *self.entries,
                MonitorTraceEntry(
                    record=record,
                    state=result.state,
                    output=result.output,
                ),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """返回带 schema 版本的 JSON 兼容 trace。"""

        return {
            "schema_version": 1,
            "monitor_identity": self.monitor_identity,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @property
    def content_hash(self) -> str:
        """返回整个 trace 的确定性 SHA-256 内容哈希。"""

        return _sha256_json(self.to_dict())


class ProtectedMonitor:
    """驱动 P4 模型的评估层受保护参考状态机。

    参数：
        model: 具有 ``protected_config`` 的 P4 正常模型；本类不拥有训练生命周期。
        config: 已在正常估计数据上冻结的 anchor gate 配置。
    返回：
        可通过 ``initial_state`` 和 ``step`` 显式推进的无隐藏监视器。
    异常：
        模型缺少维数/历史配置、配置寿命超过模型 rollout 上限或记录维数不符时抛出
        ``TypeError`` 或 ``ValueError``。
    副作用：
        构造和当前门控竖切不修改模型。所有可变生命周期均显式返回在 ``MonitorState``。
    """

    def __init__(self, model: nn.Module, config: AnchorGateConfig) -> None:
        protected_config = getattr(model, "protected_config", None)
        if protected_config is None:
            raise TypeError("ProtectedMonitor requires a model with protected_config.")
        if config.maximum_reference_age > int(protected_config.max_rollout):
            raise ValueError(
                "maximum_reference_age cannot exceed the model max_rollout."
            )
        self.model = model
        self.config = config
        self._model_config = protected_config
        self._monitor_identity = _monitor_identity(
            model=model,
            gate_config=config,
            model_config=protected_config,
        )

    @property
    def identity(self) -> str:
        """返回绑定冻结 gate、模型配置和 checkpoint 参数的联合 SHA-256。"""

        return self._monitor_identity

    def initial_state(self) -> MonitorState:
        """创建没有历史、候选或锚点的确定性初始状态。"""

        return MonitorState(monitor_identity=self.identity)

    def step(self, state: MonitorState, record: MonitorRecord) -> MonitorStepResult:
        """执行一次不接收报警反馈的纯门控状态转移。

        参数：
            state: 上一时刻由本方法返回的不可变状态。
            record: 当前原始记录与专用 eligibility score。
        返回：
            新 ``MonitorState`` 和同一时刻的 ``MonitorOutput``。
        异常：
            状态/记录类型或通道维数不匹配时抛出 ``TypeError`` 或 ``ValueError``。
        副作用：
            无；不原位修改输入状态、记录或模型，也不读取最终检测分位。
        """

        if not isinstance(state, MonitorState):
            raise TypeError("state must be a MonitorState.")
        if not isinstance(record, MonitorRecord):
            raise TypeError("record must be a MonitorRecord.")
        if state.monitor_identity != self.identity:
            raise ValueError(
                "MonitorState identity does not match this frozen monitor identity."
            )
        self._validate_record_dimensions(record)

        boundary_reset = self._crosses_boundary(state, record)
        if boundary_reset:
            state = MonitorState(
                monitor_identity=self.identity,
                reset_count=state.reset_count + 1,
            )

        episode_id = record.episode_id
        stage = record.stage
        history_before = state.history
        history_length = int(self._model_config.history_length)

        candidate = state.candidate
        anchor = state.anchor
        anchor_age = state.anchor_age
        protected_commands = state.protected_commands
        mode = state.mode
        data_latent: tuple[float, ...] | None = None
        data_context: tuple[float, ...] | None = None
        if len(history_before) == history_length:
            data_latent, data_context = self._encode_data_branch(
                history_before,
                record,
            )

        if len(history_before) < history_length:
            candidate = None
            anchor = None
            anchor_age = None
            protected_commands = ()
            mode = MonitorMode.WARMUP
        elif candidate is not None:
            if anchor is not None:
                anchor_age = record.raw_index - anchor.raw_index
            if record.anchor_eligibility_score > self.config.exit_threshold:
                candidate = None
                if anchor is None:
                    mode = MonitorMode.WARMUP
                else:
                    assert anchor_age is not None
                    mode = (
                        MonitorMode.STALE
                        if anchor_age > self.config.maximum_reference_age
                        else MonitorMode.PROTECTED
                    )
            else:
                candidate = replace(candidate, age=candidate.age + 1)
                if candidate.age >= self.config.confirmation_delay:
                    anchor = AcceptedAnchor(
                        raw_index=candidate.raw_index,
                        history=candidate.history,
                    )
                    anchor_age = candidate.age
                    protected_commands = candidate.commands
                    candidate = None
                    mode = MonitorMode.PROTECTED
                else:
                    mode = MonitorMode.CANDIDATE
        elif anchor is None:
            if record.anchor_eligibility_score <= self.config.enter_threshold:
                candidate = AnchorCandidate(
                    raw_index=record.raw_index,
                    age=0,
                    history=history_before,
                    commands=(),
                )
                mode = MonitorMode.CANDIDATE
            else:
                mode = MonitorMode.WARMUP
        else:
            anchor_age = record.raw_index - anchor.raw_index
            can_reanchor = (
                anchor_age >= self.config.minimum_reanchor_interval
                and record.anchor_eligibility_score <= self.config.enter_threshold
            )
            if can_reanchor:
                candidate = AnchorCandidate(
                    raw_index=record.raw_index,
                    age=0,
                    history=history_before,
                    commands=(),
                )
                mode = MonitorMode.CANDIDATE
            else:
                mode = (
                    MonitorMode.STALE
                    if anchor_age > self.config.maximum_reference_age
                    else MonitorMode.PROTECTED
                )

        protected_rollout = None
        if (
            anchor is not None
            and anchor_age is not None
            and 0 < anchor_age <= self.config.maximum_reference_age
        ):
            protected_rollout = self._rollout_protected_branch(
                anchor=anchor,
                commands=protected_commands,
                target_raw_index=record.raw_index,
            )

        current_command = _RecordedCommand.from_record(record)
        if candidate is not None:
            candidate = replace(
                candidate,
                commands=(*candidate.commands, current_command),
            )
        if (
            anchor is not None
            and anchor_age is not None
            and anchor_age < self.config.maximum_reference_age
        ):
            protected_commands = (*protected_commands, current_command)

        history = (*history_before, record)[-history_length:]
        next_state = MonitorState(
            monitor_identity=self.identity,
            episode_id=episode_id,
            stage=stage,
            last_raw_index=record.raw_index,
            history=history,
            candidate=candidate,
            anchor=anchor,
            anchor_age=anchor_age,
            protected_commands=protected_commands,
            mode=mode,
            reset_count=state.reset_count,
        )
        return MonitorStepResult(
            state=next_state,
            output=MonitorOutput(
                monitor_identity=self.identity,
                raw_index=record.raw_index,
                episode_id=record.episode_id,
                stage=record.stage,
                mode=mode,
                data_latent=data_latent,
                data_context=data_context,
                protected_rollout=protected_rollout,
            ),
        )

    def _encode_data_branch(
        self,
        history: tuple[MonitorRecord, ...],
        record: MonitorRecord,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """用严格过去真实历史和当前外生量计算 data branch。

        当前真实 ``measurement`` 不传给 encoder；它只能在下一时刻成为严格过去的一部分。
        显式全保留 mask 和 eval 模式使重复调用不消耗随机数。
        """

        self._require_evaluation_mode()
        past_u, past_y, past_xi = self._history_tensors(history)
        current_xi = self._vector_tensor(record.exogenous)
        keep_mask = torch.ones_like(past_y, dtype=torch.bool)
        model_api = cast(Any, self.model)
        with torch.inference_mode():
            encoded = model_api.encoder(
                past_u,
                past_y,
                past_xi,
                current_xi=current_xi,
                measurement_keep_mask=keep_mask,
            )
        return (
            self._single_vector(encoded["latent"]),
            self._single_vector(encoded["context"]),
        )

    def _rollout_protected_branch(
        self,
        *,
        anchor: AcceptedAnchor,
        commands: tuple[_RecordedCommand, ...],
        target_raw_index: int,
    ) -> ProtectedRollout:
        """从锚点快照和记录命令重算当前受保护轨迹。

        ``commands`` 的结构没有 measurement，因此即使调用方修改锚点后真实测量，也没有
        数据通道可把它传给 P4 ``rollout``。
        """

        self._require_evaluation_mode()
        if len(commands) != target_raw_index - anchor.raw_index:
            raise ValueError(
                "Protected command path must contain one command per transition."
            )
        past_u, past_y, past_xi = self._history_tensors(anchor.history)
        future_u = self._matrix_tensor(tuple(command.control for command in commands))
        future_xi = self._matrix_tensor(
            tuple(command.exogenous for command in commands)
        )
        keep_mask = torch.ones_like(past_y, dtype=torch.bool)
        model_api = cast(Any, self.model)
        with torch.inference_mode():
            rollout = model_api.rollout(
                past_u=past_u,
                past_y=past_y,
                past_xi=past_xi,
                future_u=future_u,
                future_xi=future_xi,
                measurement_keep_mask=keep_mask,
            )
        return ProtectedRollout(
            anchor_raw_index=anchor.raw_index,
            target_raw_index=target_raw_index,
            latent_trajectory=self._single_matrix(rollout["latent_trajectory"]),
            prediction_trajectory=self._single_matrix(rollout["prediction"]),
            context_trajectory=self._single_matrix(rollout["context_trajectory"]),
            protected_measurement_buffer=self._single_matrix(
                rollout["protected_past_y"]
            ),
        )

    def _history_tensors(
        self,
        history: tuple[MonitorRecord, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """把固定历史转换为模型设备和 dtype 上的批大小一 tensor。"""

        return (
            self._matrix_tensor(tuple(record.control for record in history)),
            self._matrix_tensor(tuple(record.measurement for record in history)),
            self._matrix_tensor(tuple(record.exogenous for record in history)),
        )

    def _matrix_tensor(
        self,
        rows: tuple[tuple[float, ...], ...],
    ) -> torch.Tensor:
        """把二维数值元组转换为 ``[1,T,D]`` tensor，包括 ``D=0``。"""

        reference = next(self.model.parameters())
        return torch.tensor(
            rows,
            dtype=reference.dtype,
            device=reference.device,
        ).unsqueeze(0)

    def _vector_tensor(self, values: tuple[float, ...]) -> torch.Tensor:
        """把一维数值元组转换为 ``[1,D]`` tensor。"""

        reference = next(self.model.parameters())
        return torch.tensor(
            values,
            dtype=reference.dtype,
            device=reference.device,
        ).unsqueeze(0)

    @staticmethod
    def _single_vector(value: torch.Tensor) -> tuple[float, ...]:
        """把 ``[1,D]`` tensor 冻结为 Python 数值元组。"""

        return tuple(float(item) for item in value[0].detach().cpu().tolist())

    @staticmethod
    def _single_matrix(value: torch.Tensor) -> tuple[tuple[float, ...], ...]:
        """把 ``[1,T,D]`` tensor 冻结为嵌套 Python 数值元组。"""

        return tuple(
            tuple(float(item) for item in row)
            for row in value[0].detach().cpu().tolist()
        )

    def _require_evaluation_mode(self) -> None:
        """拒绝带 dropout 或训练 mask 随机性的模型状态。"""

        if self.model.training:
            raise ValueError(
                "ProtectedMonitor requires model.eval() for deterministic state transitions."
            )

    def _validate_record_dimensions(self, record: MonitorRecord) -> None:
        """在记录进入历史前核对模型通道宽度，避免错误向量污染后续重放。"""

        expected = (
            (len(record.control), int(self._model_config.control_dim), "control"),
            (
                len(record.measurement),
                int(self._model_config.measurement_dim),
                "measurement",
            ),
            (len(record.exogenous), int(self._model_config.exogenous_dim), "exogenous"),
        )
        for actual, configured, name in expected:
            if actual != configured:
                raise ValueError(
                    f"MonitorRecord {name} width {actual} does not match model width "
                    f"{configured}."
                )

    @staticmethod
    def _crosses_boundary(state: MonitorState, record: MonitorRecord) -> bool:
        """识别会使严格过去历史失效的 episode、stage 或索引断点。"""

        if state.last_raw_index is None:
            return False
        return (
            state.episode_id != record.episode_id
            or state.stage != record.stage
            or record.raw_index != state.last_raw_index + 1
        )


def _sha256_json(value: Any) -> str:
    """以稳定字段顺序和无空白编码计算 JSON 内容哈希。"""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _monitor_identity(
    *,
    model: nn.Module,
    gate_config: AnchorGateConfig,
    model_config: Any,
) -> str:
    """计算冻结 gate、模型配置和模型参数的联合 monitor 身份。

    tensor 按键名、dtype、shape 和原始字节稳定编码；因此不同 checkpoint、buffer 计数或
    配置都会产生不同身份。该计算只在 monitor 构造时执行，不进入每个在线时刻的热路径。
    """

    state_digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        frozen = tensor.detach().cpu().contiguous()
        state_digest.update(name.encode("utf-8"))
        state_digest.update(str(frozen.dtype).encode("ascii"))
        state_digest.update(
            json.dumps(list(frozen.shape), separators=(",", ":")).encode("ascii")
        )
        raw_bytes = (
            frozen.reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
        state_digest.update(raw_bytes)
    model_dump = getattr(model_config, "model_dump", None)
    if not callable(model_dump):
        raise TypeError("Protected model config must provide model_dump().")
    return _sha256_json(
        {
            "gate_config": gate_config.model_dump(mode="json"),
            "model_class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
            "model_config": model_dump(mode="json"),
            "model_state_sha256": state_digest.hexdigest(),
        }
    )
