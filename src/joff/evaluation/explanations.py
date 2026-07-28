"""P9 集合值结构化隔离的 explanation family 与完整观测领域对象。

文件用途：
    定义隔离层使用的正常、传感器侧、动力学侧和 mixed explanation 身份，后续承载
    完整 monitor 前态、raw window、分支/算子证据和 mask 全流水线重算。
主要职责：
    提供不可变、严格验证和可审计哈希的 P9 领域值；本文件不拟合归因分位、不运行
    outer oracle，也不把算法 explanation 自动升级成物理故障标签。
关键输入与输出：
    输入为预声明 explanation 支持、物理证据身份和部署半径；输出为
    ``ExplanationFamily`` 等不可变对象，供 oracle、校准器和候选集共同引用。
依赖与副作用：
    依赖 Python 标准库以及 P5 ``protected_reference``、P6 ``protected_operators`` 领域
    对象；不读取数据、文件或网络，不修改全局状态。
重要约束：
    mixed 支持最多含一个 sensor 与一个 dynamics-side 分量；物理标签必须有外部证据；
    缺少物理证据时只能声明等价类，不能在本层伪造认证。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum

from .protected_operators import OperatorBundle, OperatorStatus
from .protected_reference import MonitorRecord, MonitorState


class DynamicsSide(str, Enum):
    """一个 explanation 中允许出现的唯一动力学侧分量。

    参数：
        value: 受控字符串；区分物理 actuator/process 与较弱 learned/process-side 等价类。
    返回：
        对应的 ``DynamicsSide`` 枚举成员。
    异常：
        未知字符串由 ``Enum`` 抛出 ``ValueError``。
    副作用：
        无。
    """

    ACTUATOR = "actuator"
    PROCESS = "process"
    LEARNED_INPUT = "learned-input-side"
    PROCESS_SIDE = "process-side"


@dataclass(frozen=True)
class ExplanationFamily:
    """一个预声明 normal、fault 或 equivalence explanation family。

    参数：
        family_id/label: 稳定机器身份与读者可见标签。
        sensor_channels: 支持中的 sensor 通道；并发先验限制长度最多为一。
        dynamics_sides: 支持中的动力学侧分量；并发先验限制长度最多为一。
        physical: 是否请求物理故障标签；``True`` 时必须有外部物理证据哈希。
        equivalence_label: 缺少物理可辨识性时允许报告的较弱等价类。
        radius: fault outer family 的预声明非负半径；normal family 的分位由 P9 校准器给出。
        normal: 是否为 nuisance-only Normal explanation。
        physical_evidence_hash: 输入激励、传感器横向覆盖或 process injection map 等外部证据。
    返回：
        不可变 explanation 身份，可由 ``content_hash`` 绑定到 oracle 和隔离报告。
    异常：
        支持超过并发先验、Normal 携带 fault 分量、物理证据缺失或字段非法时抛出
        ``TypeError``/``ValueError``。
    副作用：
        无；构造不会读取正常或故障数据，也不运行物理认证。
    """

    family_id: str
    label: str
    sensor_channels: tuple[str, ...]
    dynamics_sides: tuple[DynamicsSide, ...]
    physical: bool
    equivalence_label: str | None
    radius: float
    normal: bool = False
    physical_evidence_hash: str | None = None

    def __post_init__(self) -> None:
        """验证支持并发先验、物理证据边界和 Normal 特例。"""

        if (
            not isinstance(self.family_id, str)
            or not self.family_id.strip()
            or not isinstance(self.label, str)
            or not self.label.strip()
        ):
            raise ValueError("Explanation family id and label must be non-empty.")
        if not isinstance(self.sensor_channels, tuple) or not all(
            isinstance(channel, str) and channel.strip()
            for channel in self.sensor_channels
        ):
            raise TypeError("Explanation sensor_channels must be a tuple of names.")
        if len(self.sensor_channels) > 1:
            raise ValueError("An explanation support allows at most one sensor.")
        if len(set(self.sensor_channels)) != len(self.sensor_channels):
            raise ValueError("Explanation sensor channels must be unique.")
        if not isinstance(self.dynamics_sides, tuple) or not all(
            isinstance(side, DynamicsSide) for side in self.dynamics_sides
        ):
            raise TypeError("Explanation dynamics_sides must use DynamicsSide values.")
        if len(self.dynamics_sides) > 1:
            raise ValueError("An explanation support allows at most one dynamics-side.")
        if type(self.physical) is not bool or type(self.normal) is not bool:
            raise TypeError("Explanation physical/normal flags must be strict booleans.")
        if isinstance(self.radius, bool) or not isinstance(self.radius, (int, float)):
            raise TypeError("Explanation radius must be numeric.")
        normalized_radius = float(self.radius)
        if normalized_radius < 0.0 or not math.isfinite(normalized_radius):
            raise ValueError("Explanation radius must be finite and non-negative.")
        object.__setattr__(self, "radius", normalized_radius)

        has_fault_support = bool(self.sensor_channels or self.dynamics_sides)
        if self.normal:
            if has_fault_support or self.physical or self.equivalence_label is not None:
                raise ValueError("Normal explanation cannot carry fault support or labels.")
            if self.radius != 0.0 or self.physical_evidence_hash is not None:
                raise ValueError("Normal explanation uses calibrated radius, not physical evidence.")
            return
        if not has_fault_support:
            raise ValueError("A nonnormal explanation requires declared support.")
        if self.physical:
            if self.equivalence_label is not None:
                raise ValueError("Physical explanation cannot also be an equivalence class.")
            if any(
                side in (DynamicsSide.LEARNED_INPUT, DynamicsSide.PROCESS_SIDE)
                for side in self.dynamics_sides
            ):
                raise ValueError(
                    "Physical explanation cannot use learned/process-side equivalence support."
                )
            if not _is_sha256(self.physical_evidence_hash):
                raise ValueError("Physical explanation requires external evidence SHA-256.")
        else:
            if (
                not isinstance(self.equivalence_label, str)
                or not self.equivalence_label.strip()
            ):
                raise ValueError("Nonphysical explanation requires an equivalence label.")
            if self.physical_evidence_hash is not None:
                raise ValueError("Equivalence explanation cannot claim physical evidence.")

    def to_dict(self) -> dict[str, object]:
        """返回 explanation 支持、标签边界和物理证据身份。

        参数：
            无。
        返回：
            只含 JSON 基本类型的稳定字典。
        异常：
            无；字段已在构造时验证。
        副作用：
            无。
        """

        return {
            "family_id": self.family_id,
            "label": self.label,
            "sensor_channels": list(self.sensor_channels),
            "dynamics_sides": [side.value for side in self.dynamics_sides],
            "physical": self.physical,
            "equivalence_label": self.equivalence_label,
            "radius": self.radius,
            "normal": self.normal,
            "physical_evidence_hash": self.physical_evidence_hash,
        }

    @property
    def content_hash(self) -> str:
        """返回绑定全部 family 语义的 SHA-256。

        参数：
            无。
        返回：
            64 位小写十六进制摘要。
        异常：
            无。
        副作用：
            无；只编码内存字段。
        """

        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class DeployedBranchEvidence:
    """完整部署观测中的一个 ``L_b/T_b/Gamma_b`` 与 P6 算子证据。

    参数：
        branch_name/matrix: P7 分支身份与实际部署矩阵 ``L_b``。
        statistic/threshold: 当前窗口统计量 ``T_b`` 与完整动态阈值 ``Gamma_b``。
        operator_bundle: 同一 episode、stage 和路径的完整 P6 算子包及 enclosure。
        score_map_hash/calibration_hash: 冻结 P8 score map 与 detection calibration 身份。
    返回：
        可被完整观测、mask 重算和 oracle 共同引用的不可变分支证据。
    异常：
        名称、矩阵、数值、算子类型或身份哈希非法时抛出 ``TypeError``/``ValueError``。
    副作用：
        无。
    """

    branch_name: str
    matrix: tuple[tuple[float, ...], ...]
    statistic: float
    threshold: float
    operator_bundle: OperatorBundle
    score_map_hash: str
    calibration_hash: str

    def __post_init__(self) -> None:
        """验证部署矩阵、阈值分账身份和完整算子证据。"""

        if not isinstance(self.branch_name, str) or not self.branch_name.strip():
            raise ValueError("Deployed branch name must be non-empty.")
        if (
            not isinstance(self.matrix, tuple)
            or not self.matrix
            or not all(isinstance(row, tuple) and row for row in self.matrix)
        ):
            raise TypeError("Deployed branch matrix must be a non-empty tuple matrix.")
        width = len(self.matrix[0])
        if any(
            len(row) != width
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in row
            )
            for row in self.matrix
        ):
            raise ValueError("Deployed branch matrix must be rectangular and finite.")
        object.__setattr__(
            self,
            "matrix",
            tuple(tuple(float(value) for value in row) for row in self.matrix),
        )
        for name, value in (
            ("statistic", self.statistic),
            ("threshold", self.threshold),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"Deployed branch {name} must be finite and non-negative.")
            object.__setattr__(self, name, float(value))
        if not isinstance(self.operator_bundle, OperatorBundle):
            raise TypeError("Deployed branch requires a complete OperatorBundle.")
        if not _is_sha256(self.score_map_hash) or not _is_sha256(
            self.calibration_hash
        ):
            raise ValueError("Deployed branch score/calibration hashes must be SHA-256.")

    def to_dict(self) -> dict[str, object]:
        """返回分支矩阵、统计量、阈值和完整 P6 算子证据。

        参数：
            无。
        返回：
            只含 JSON 兼容值的稳定字典。
        异常：
            无。
        副作用：
            无。
        """

        return {
            "branch_name": self.branch_name,
            "matrix": [list(row) for row in self.matrix],
            "statistic": self.statistic,
            "threshold": self.threshold,
            "operator_bundle": self.operator_bundle.to_dict(),
            "score_map_hash": self.score_map_hash,
            "calibration_hash": self.calibration_hash,
        }

    @property
    def content_hash(self) -> str:
        """返回覆盖 ``L/T/Gamma`` 和算子包的 SHA-256。

        参数：
            无。
        返回：
            64 位小写十六进制摘要。
        异常：
            无。
        副作用：
            无。
        """

        return _sha256_json(self.to_dict())


@dataclass(frozen=True)
class MaskRecomputation:
    """一个 sensor mask 的完整反事实流水线重算证据。

    参数：
        sensor_channel: 被 mask 的原始测量通道。
        source_state_hash/source_raw_window_hash: 未 mask 观测的完整前态与 raw window 身份。
        pipeline_hash: 冻结 mask-conditioned 模型、monitor、P6--P8 流水线身份。
        masked_raw_window/recomputed_state/recomputed_branches: mask 后重新运行得到的完整输入、
            monitor 状态和 ``L/T/Gamma/operator`` 分支证据。
        measurement_residual: 同次重算产生的测量域残差 ``r^y``。
        exonerated: 是否观察到预声明的单侧回落；``False`` 不能排除 sensor explanation。
    返回：
        不可变 mask 证据；供完整观测和 oracle 使用。
    异常：
        缺失重算 state/branch、身份或数值非法时抛出 ``TypeError``/``ValueError``。
    副作用：
        无；对象只保存已经完成的重算，不在构造时执行模型。
    """

    sensor_channel: str
    source_state_hash: str
    source_raw_window_hash: str
    pipeline_hash: str
    masked_raw_window: tuple[MonitorRecord, ...]
    recomputed_state: MonitorState
    recomputed_branches: tuple[DeployedBranchEvidence, ...]
    measurement_residual: tuple[float, ...]
    exonerated: bool

    def __post_init__(self) -> None:
        """拒绝 residual-column shortcut 和不完整反事实证据。"""

        if not isinstance(self.sensor_channel, str) or not self.sensor_channel.strip():
            raise ValueError("Mask sensor channel must be non-empty.")
        if not all(
            _is_sha256(value)
            for value in (
                self.source_state_hash,
                self.source_raw_window_hash,
                self.pipeline_hash,
            )
        ):
            raise ValueError("Mask source and pipeline identities must be SHA-256.")
        if (
            not isinstance(self.masked_raw_window, tuple)
            or not self.masked_raw_window
            or not all(
                isinstance(record, MonitorRecord)
                for record in self.masked_raw_window
            )
        ):
            raise TypeError("Mask requires a complete masked raw window.")
        if not isinstance(self.recomputed_state, MonitorState):
            raise TypeError("Mask requires a recomputed MonitorState.")
        if (
            not isinstance(self.recomputed_branches, tuple)
            or not self.recomputed_branches
            or not all(
                isinstance(branch, DeployedBranchEvidence)
                for branch in self.recomputed_branches
            )
        ):
            raise ValueError("Mask requires at least one recomputed branch.")
        if (
            not isinstance(self.measurement_residual, tuple)
            or not self.measurement_residual
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in self.measurement_residual
            )
        ):
            raise ValueError("Mask measurement residual must be a finite vector.")
        object.__setattr__(
            self,
            "measurement_residual",
            tuple(float(value) for value in self.measurement_residual),
        )
        if type(self.exonerated) is not bool:
            raise TypeError("Mask exonerated flag must be a strict boolean.")

    def to_dict(self) -> dict[str, object]:
        """返回 source 绑定和完整 mask 流水线输出。

        参数：
            无。
        返回：
            JSON 兼容审计字典。
        异常：
            无。
        副作用：
            无。
        """

        return {
            "sensor_channel": self.sensor_channel,
            "source_state_hash": self.source_state_hash,
            "source_raw_window_hash": self.source_raw_window_hash,
            "pipeline_hash": self.pipeline_hash,
            "masked_raw_window": [
                record.to_dict() for record in self.masked_raw_window
            ],
            "recomputed_state": self.recomputed_state.to_dict(),
            "recomputed_branches": [
                branch.to_dict() for branch in self.recomputed_branches
            ],
            "measurement_residual": list(self.measurement_residual),
            "exonerated": self.exonerated,
        }


@dataclass(frozen=True)
class DeployedObservation:
    """隔离 oracle 唯一允许读取的完整部署观测 ``O_full``。

    参数：
        monitor_state/raw_window: 判决前完整有限 monitor state 与原始 ``(u,y,xi)`` window。
        measurement_channels: raw measurement 向量的冻结通道顺序，用于证明 mask 只改变目标。
        safe_context: 与 raw window 对齐且已证明不受故障影响的外生/调度量。
        branches: 同一时刻全部 active ``L/T/Gamma`` 和 P6 operator enclosure。
        mask_recomputations: 每个已运行 sensor mask 的完整反事实流水线输出。
        feature_schema_hash/linear_features: 线性特例使用的冻结 full-observation 特征模式与值。
        detection_calibration_hash/detection_excess: 已冻结 ``q_det`` 身份与当前检测超额。
    返回：
        绑定全部观测证据的不可变对象；raw-identical/full-state-identical 输入具有相同哈希。
    异常：
        episode、stage、raw index、monitor/operator 身份、mask source 或 shape 不一致时抛出
        ``TypeError``/``ValueError``。
    副作用：
        无；不执行 mask、oracle 或校准。
    """

    monitor_state: MonitorState
    raw_window: tuple[MonitorRecord, ...]
    measurement_channels: tuple[str, ...]
    safe_context: tuple[tuple[float, ...], ...]
    branches: tuple[DeployedBranchEvidence, ...]
    mask_recomputations: tuple[MaskRecomputation, ...]
    feature_schema_hash: str
    linear_features: tuple[float, ...]
    detection_calibration_hash: str
    detection_excess: float

    def __post_init__(self) -> None:
        """交叉核验完整前态、window、分支算子和 mask 重算身份。"""

        if not isinstance(self.monitor_state, MonitorState):
            raise TypeError("Deployed observation requires a MonitorState.")
        if self.monitor_state.episode_id is None or self.monitor_state.stage is None:
            raise ValueError("Deployed observation monitor state must have episode and stage.")
        if (
            not isinstance(self.raw_window, tuple)
            or not self.raw_window
            or not all(isinstance(record, MonitorRecord) for record in self.raw_window)
        ):
            raise TypeError("Deployed observation requires a complete raw window.")
        if (
            not isinstance(self.measurement_channels, tuple)
            or not self.measurement_channels
            or any(
                not isinstance(channel, str) or not channel.strip()
                for channel in self.measurement_channels
            )
            or len(set(self.measurement_channels)) != len(self.measurement_channels)
            or any(
                len(record.measurement) != len(self.measurement_channels)
                for record in self.raw_window
            )
        ):
            raise ValueError(
                "Deployed measurement channels must uniquely match every raw record."
            )
        episode_id = self.monitor_state.episode_id
        stage = self.monitor_state.stage
        raw_indices = tuple(record.raw_index for record in self.raw_window)
        if (
            any(
                record.episode_id != episode_id or record.stage is not stage
                for record in self.raw_window
            )
            or raw_indices != tuple(sorted(raw_indices))
            or len(set(raw_indices)) != len(raw_indices)
        ):
            raise ValueError("Deployed raw window episode, stage, or ordering is invalid.")
        if (
            self.monitor_state.last_raw_index is None
            or self.monitor_state.last_raw_index >= raw_indices[-1]
        ):
            raise ValueError("Deployed monitor state must be the finite pre-decision state.")
        if (
            not isinstance(self.safe_context, tuple)
            or len(self.safe_context) != len(self.raw_window)
            or not all(isinstance(row, tuple) for row in self.safe_context)
            or any(
                not row
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in row
                )
                for row in self.safe_context
            )
        ):
            raise ValueError("Deployed safe context must align with the raw window.")
        object.__setattr__(
            self,
            "safe_context",
            tuple(
                tuple(float(value) for value in row)
                for row in self.safe_context
            ),
        )
        if (
            not isinstance(self.branches, tuple)
            or not self.branches
            or not all(
                isinstance(branch, DeployedBranchEvidence)
                for branch in self.branches
            )
            or len({branch.branch_name for branch in self.branches})
            != len(self.branches)
        ):
            raise ValueError("Deployed observation branches must be non-empty and unique.")
        for branch in self.branches:
            path = branch.operator_bundle.path
            if (
                path.monitor_identity != self.monitor_state.monitor_identity
                or path.episode_id != episode_id
                or path.stage is not stage
                or path.raw_indices[-1] != raw_indices[-1]
            ):
                raise ValueError(
                    "Deployed branch operator path must match monitor state and raw window."
                )
        if not isinstance(self.mask_recomputations, tuple) or not all(
            isinstance(mask, MaskRecomputation)
            for mask in self.mask_recomputations
        ):
            raise TypeError("Deployed mask recomputations must be a tuple.")
        if len({mask.sensor_channel for mask in self.mask_recomputations}) != len(
            self.mask_recomputations
        ):
            raise ValueError("Deployed mask channels must be unique.")
        raw_window_hash = self.hash_raw_window(self.raw_window)
        for mask in self.mask_recomputations:
            if mask.source_state_hash != self.monitor_state.content_hash:
                raise ValueError("Mask source state does not match deployed observation.")
            if mask.source_raw_window_hash != raw_window_hash:
                raise ValueError("Mask source raw window does not match deployed observation.")
            if tuple(record.raw_index for record in mask.masked_raw_window) != raw_indices:
                raise ValueError("Masked raw window must preserve original raw indices.")
            if mask.sensor_channel not in self.measurement_channels:
                raise ValueError(
                    "Masked sensor channel must belong to the deployed measurement schema."
                )
            sensor_index = self.measurement_channels.index(mask.sensor_channel)
            for source_record, masked_record in zip(
                self.raw_window,
                mask.masked_raw_window,
                strict=True,
            ):
                if (
                    source_record.episode_id != masked_record.episode_id
                    or source_record.stage is not masked_record.stage
                    or source_record.control != masked_record.control
                    or source_record.exogenous != masked_record.exogenous
                    or source_record.anchor_eligibility_score
                    != masked_record.anchor_eligibility_score
                    or len(masked_record.measurement) != len(self.measurement_channels)
                    or any(
                        source_value != masked_value
                        for index, (source_value, masked_value) in enumerate(
                            zip(
                                source_record.measurement,
                                masked_record.measurement,
                                strict=True,
                            )
                        )
                        if index != sensor_index
                    )
                ):
                    raise ValueError(
                        "Mask recomputation may change only its declared raw sensor channel."
                    )
            if (
                mask.recomputed_state.episode_id != episode_id
                or mask.recomputed_state.stage is not stage
                or mask.recomputed_state.monitor_identity
                != self.monitor_state.monitor_identity
            ):
                raise ValueError(
                    "Mask recomputed state must preserve monitor, episode, and stage."
                )
            for branch in mask.recomputed_branches:
                path = branch.operator_bundle.path
                if (
                    path.monitor_identity
                    != mask.recomputed_state.monitor_identity
                    or path.episode_id != episode_id
                    or path.stage is not stage
                    or path.raw_indices[-1] != raw_indices[-1]
                ):
                    raise ValueError("Mask recomputed branch path is inconsistent.")
        if not _is_sha256(self.feature_schema_hash) or not _is_sha256(
            self.detection_calibration_hash
        ):
            raise ValueError("Deployed feature/calibration identities must be SHA-256.")
        all_branch_evidence = (
            *self.branches,
            *(
                branch
                for mask in self.mask_recomputations
                for branch in mask.recomputed_branches
            ),
        )
        if any(
            branch.calibration_hash != self.detection_calibration_hash
            for branch in all_branch_evidence
        ):
            raise ValueError(
                "Every deployed and mask-recomputed branch must use the frozen "
                "detection calibration."
            )
        if (
            not isinstance(self.linear_features, tuple)
            or not self.linear_features
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in self.linear_features
            )
        ):
            raise ValueError("Deployed linear features must be a finite vector.")
        object.__setattr__(
            self,
            "linear_features",
            tuple(float(value) for value in self.linear_features),
        )
        if (
            isinstance(self.detection_excess, bool)
            or not isinstance(self.detection_excess, (int, float))
            or not math.isfinite(float(self.detection_excess))
            or float(self.detection_excess) < 0.0
        ):
            raise ValueError("Deployed detection excess must be finite and non-negative.")
        object.__setattr__(self, "detection_excess", float(self.detection_excess))

    @staticmethod
    def hash_raw_window(raw_window: tuple[MonitorRecord, ...]) -> str:
        """计算完整 raw window 的稳定 SHA-256。

        参数：
            raw_window: 按原始索引排序的 ``MonitorRecord`` 元组。
        返回：
            覆盖控制、测量、外生量、stage 和索引的内容哈希。
        异常：
            元组为空或含非 ``MonitorRecord`` 时抛出 ``TypeError``。
        副作用：
            无。
        """

        if (
            not isinstance(raw_window, tuple)
            or not raw_window
            or not all(isinstance(record, MonitorRecord) for record in raw_window)
        ):
            raise TypeError("raw_window must be a non-empty MonitorRecord tuple.")
        return _sha256_json([record.to_dict() for record in raw_window])

    @property
    def all_operators_certified(self) -> bool:
        """返回所有部署分支是否都携带 P6 ``CERTIFIED`` enclosure。

        参数：
            无。
        返回：
            仅全部 operator bundle 均为 ``CERTIFIED`` 时返回 ``True``。
        异常：
            无。
        副作用：
            无。
        """

        return all(
            branch.operator_bundle.status is OperatorStatus.CERTIFIED
            for branch in (
                *self.branches,
                *(
                    recomputed_branch
                    for mask in self.mask_recomputations
                    for recomputed_branch in mask.recomputed_branches
                ),
            )
        )

    def to_dict(self) -> dict[str, object]:
        """返回完整 monitor、raw、branch、operator 和 mask 观测。

        参数：
            无。
        返回：
            JSON 兼容审计字典。
        异常：
            无。
        副作用：
            无。
        """

        return {
            "monitor_state": self.monitor_state.to_dict(),
            "raw_window": [record.to_dict() for record in self.raw_window],
            "measurement_channels": list(self.measurement_channels),
            "safe_context": [list(row) for row in self.safe_context],
            "branches": [branch.to_dict() for branch in self.branches],
            "mask_recomputations": [
                mask.to_dict() for mask in self.mask_recomputations
            ],
            "feature_schema_hash": self.feature_schema_hash,
            "linear_features": list(self.linear_features),
            "detection_calibration_hash": self.detection_calibration_hash,
            "detection_excess": self.detection_excess,
        }

    @property
    def content_hash(self) -> str:
        """返回覆盖完整部署观测的 SHA-256。

        参数：
            无。
        返回：
            64 位小写十六进制摘要。
        异常：
            无。
        副作用：
            无。
        """

        return _sha256_json(self.to_dict())

    def certifies_physical_family(self, family: ExplanationFamily) -> bool:
        """判断完整观测是否覆盖一个物理 explanation 所需的 mask 与命名算子。

        参数：
            family: 请求物理 singleton 的 ``ExplanationFamily``；等价类不会被本方法升级。
        返回：
            仅全部部署/mask operator bundle 已认证，且 sensor family 有同名完整 mask 与
            ``sensor:<channel>`` JVP、actuator/process family 分别有 ``input_response`` /
            ``process_prior`` 时返回 ``True``。
        异常：
            ``family`` 类型非法时抛出 ``TypeError``。
        副作用：
            无；只读取已经冻结的观测证据，不执行新的 mask 或 operator 装配。
        """

        if not isinstance(family, ExplanationFamily):
            raise TypeError("Physical certification requires an ExplanationFamily.")
        if not family.physical or not self.all_operators_certified:
            return False
        masks_by_channel = {
            mask.sensor_channel: mask for mask in self.mask_recomputations
        }
        required_operator_names: list[str] = []
        for sensor_channel in family.sensor_channels:
            if sensor_channel not in masks_by_channel:
                return False
            required_operator_names.append(f"sensor:{sensor_channel}")
        for side in family.dynamics_sides:
            if side is DynamicsSide.ACTUATOR:
                required_operator_names.append("input_response")
            elif side is DynamicsSide.PROCESS:
                required_operator_names.append("process_prior")
            else:
                return False
        return all(
            all(
                operator_name
                in branch.operator_bundle.required_certification_names
                for operator_name in required_operator_names
            )
            for branch in self.branches
        )


def _sha256_json(value: object) -> str:
    """对 JSON 兼容值计算确定性 SHA-256。

    参数：
        value: 只含 JSON 基本类型的对象。
    返回：
        64 位小写十六进制摘要。
    异常：
        值不可 JSON 编码时透传 ``TypeError``。
    副作用：
        无。
    """

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    """判断值是否为 64 位小写十六进制 SHA-256。

    参数：
        value: 待验证对象。
    返回：
        仅严格匹配时返回 ``True``。
    异常：
        无。
    副作用：
        无。
    """

    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
