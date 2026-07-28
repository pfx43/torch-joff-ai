"""
论文专用五阶段正常数据协议、访问账本与冻结故障测试门禁。

文件用途：
    把一条仅正常长序列先按原始时间轴切成五个用途互斥的阶段，并显式保留段间隔离带、
    窗口依赖、episode、哈希和后续拟合访问记录，作为论文实验编排的数据安全边界。
主要职责：
    定义 StageName、FiveStageSplitConfig、StageSlice、FiveStageSplitResult、
    FiveStageNormalSplitter、FitAccessLedger 和 PaperDataBundle；本文件不拟合 scaler、
    不训练模型、不计算阈值或故障性能，也不替代 DatasetAdapter 读取原始文件。
关键输入与输出：
    输入为已规范化的正常 NumPy 数组、历史/rollout/堆叠依赖和风险配置；输出为五段原始
    行索引、合法窗口起点、完整 episode、隔离带、准备后数据 hash 和可序列化 manifest。
依赖与副作用：
    核心切分只依赖 NumPy、hashlib 和标准库且无外部副作用；只有显式保存产物时才写文件。
    不访问网络、不读取数据集路径、不修改随机或绘图库全局状态。
重要约束：
    五段全部只能来自正常数据；检测校准与归因校准不得复用行或 episode；先切原始时间轴，
    再删除隔离带，再保留依赖行完全落在单段内的窗口。真实故障测试是五段之外的封存范围，
    在协议未冻结或许可未核实时必须拒绝访问。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .adapters.base import CanonicalDataset


class StageName(str, Enum):
    """五个仅正常数据阶段的受控名称。

    枚举顺序就是原始时间轴顺序，不能按字母排序或由调用方自由拼写，以免 manifest、
    访问策略和数组切片对同一阶段产生不同解释。
    """

    TRAIN = "train"
    ESTIMATE = "estimate"
    DETECTION_CALIBRATION = "detection_calibration"
    ATTRIBUTION_CALIBRATION = "attribution_calibration"
    FROZEN_NORMAL_TEST = "frozen_normal_test"

    @classmethod
    def parse(cls, value: "StageName | str") -> "StageName":
        """把枚举或字符串解析为受控阶段名。

        参数：
            value: ``StageName`` 或其精确字符串值。
        返回：
            对应的 ``StageName``。
        异常：
            未知名称时抛出 ``ValueError``，错误消息列出全部合法选项。
        副作用：
            无。
        """

        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError as exc:
            legal = ", ".join(stage.value for stage in cls)
            raise ValueError(
                f"Unknown paper data stage {value!r}. Legal options are: {legal}."
            ) from exc


class FaultLicenseStatus(str, Enum):
    """封存故障数据可用性的受控许可状态。

    ``verified`` 是唯一允许正式故障访问的状态；``to_verify`` 表示尚待核实，
    ``restricted`` 表示已知存在使用限制，``not_permitted`` 表示明确不可用于评价。
    自由字符串不能直接参与安全门禁，避免拼写错误被误当成已授权。
    """

    VERIFIED = "verified"
    TO_VERIFY = "to_verify"
    RESTRICTED = "restricted"
    NOT_PERMITTED = "not_permitted"

    @classmethod
    def parse(
        cls,
        value: "FaultLicenseStatus | str",
    ) -> "FaultLicenseStatus":
        """把枚举或字符串解析为受控许可状态。

        参数：
            value: ``FaultLicenseStatus`` 或其精确字符串值；字符串会去除首尾空白并转小写。
        返回：
            对应的 ``FaultLicenseStatus``。
        异常：
            未知状态时抛出 ``ValueError``，错误消息列出全部合法选项。
        副作用：
            无。
        """

        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            legal = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unknown fault license status {value!r}. Legal options are: {legal}."
            ) from exc


_DEFAULT_RATIOS = (0.55, 0.15, 0.10, 0.10, 0.10)


@dataclass(frozen=True)
class FiveStageSplitConfig:
    """五阶段切分、依赖跨度和校准 episode 的冻结配置。

    参数：
        ratios: 按 ``StageName`` 顺序排列的五个比例，必须非负且总和为 1。
        history_length: 单个预测读取的历史行数 ``ell_h``。
        max_rollout: 最大自由预测未来长度 ``N_max``。
        stacked_window: 堆叠多个连续预测/残差时覆盖的预测位置数。
        mask_recompute_span: 传感器 mask 或归因重算额外向后覆盖的行数。
        minimum_gap: 调用方要求的最小隔离带；实际值不会小于计算出的依赖跨度。
        window_stride: 合法窗口起点之间的步长。
        episode_length: 校准 maximum 使用的完整正常 episode 行数。
        target_risk_level: 目标风险水平；用于检查完整校准 episode 数量是否有足够分辨率。
        seed: 写入 manifest 的确定性种子；当前 chronological 策略不随机重排时间轴。
        strategy: 当前只允许 ``chronological``，避免正常时间段被随机打散。
    """

    ratios: tuple[float, float, float, float, float] = _DEFAULT_RATIOS
    history_length: int = 3
    max_rollout: int = 2
    stacked_window: int = 1
    mask_recompute_span: int = 0
    minimum_gap: int = 0
    window_stride: int = 1
    episode_length: int = 64
    target_risk_level: float = 0.1
    seed: int = 42
    strategy: str = "chronological"

    def __post_init__(self) -> None:
        """拒绝会缩短依赖、产生空窗口或破坏风险解释的配置。

        参数：
            无；校验当前冻结 dataclass 的全部字段。
        返回：
            无。
        异常：
            比例非法、任一长度/步长不在允许范围、风险水平不在 ``(0, 1)``，或策略不是
            ``chronological`` 时抛出 ``ValueError``。
        副作用：
            无。冻结 dataclass 不会被修改。
        """

        if len(self.ratios) != len(StageName):
            raise ValueError(
                f"Five-stage ratios must contain {len(StageName)} values. "
                f"Current count: {len(self.ratios)}."
            )
        if any(ratio <= 0 for ratio in self.ratios) or not math.isclose(
            sum(self.ratios),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Five-stage ratios must all be positive and sum to 1. "
                f"Current ratios={self.ratios}, total={sum(self.ratios)}."
            )
        positive_fields = {
            "history_length": self.history_length,
            "max_rollout": self.max_rollout,
            "stacked_window": self.stacked_window,
            "window_stride": self.window_stride,
            "episode_length": self.episode_length,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive. Current input: {value}.")
        if self.mask_recompute_span < 0 or self.minimum_gap < 0:
            raise ValueError(
                "mask_recompute_span and minimum_gap must be non-negative. "
                f"Current inputs: {self.mask_recompute_span}, {self.minimum_gap}."
            )
        if not 0.0 < self.target_risk_level < 1.0:
            raise ValueError(
                "target_risk_level must be strictly between 0 and 1. "
                f"Current input: {self.target_risk_level}."
            )
        if self.strategy != "chronological":
            raise ValueError(
                "Five-stage normal splitting supports only strategy='chronological'. "
                f"Current input: {self.strategy!r}."
            )

    @property
    def dependency_span(self) -> int:
        """返回一个窗口可能读取的保守连续原始行跨度。

        基础跨度是 ``history_length + max_rollout``。堆叠窗口的后续预测位置以及 mask
        重算可能继续向后读取，因此在两者中取更大扩展量，不能只抄论文下限常数。
        参数：
            无。
        返回：
            覆盖历史、最大 rollout 及额外重算范围的正整数位置数。
        异常：
            无；构造时已验证各长度字段。
        副作用：
            无。
        """

        extension = max(self.stacked_window - 1, self.mask_recompute_span)
        return self.history_length + self.max_rollout + extension

    @property
    def effective_gap(self) -> int:
        """返回不小于全部依赖跨度和调用方下限的实际隔离带。

        参数：
            无。
        返回：
            ``minimum_gap`` 与 ``dependency_span`` 的较大值。
        异常：
            无。
        副作用：
            无。
        """

        return max(self.minimum_gap, self.dependency_span)

    @property
    def minimum_calibration_episodes(self) -> int:
        """返回目标风险分辨率至少需要的完整校准 episode 数。

        对 ``m`` 个 exchangeable episode，最小非零尾概率分辨率是 ``1/(m+1)``。
        因此要求 ``m >= ceil(1/alpha)-1``，并至少保留一个完整 episode。
        参数：
            无。
        返回：
            支持 ``target_risk_level`` 所需的最少完整 episode 数。
        异常：
            无；风险水平已在构造时验证。
        副作用：
            无。
        """

        return max(1, math.ceil(1.0 / self.target_risk_level) - 1)

    def manifest(self) -> dict[str, Any]:
        """返回不含运行时对象的 JSON 可序列化配置摘要。

        参数：
            无。
        返回：
            含比例、依赖跨度、隔离带、episode、风险水平、种子和策略的新字典。
        异常：
            无。
        副作用：
            无。
        """

        return {
            "ratios": {
                stage.value: ratio for stage, ratio in zip(StageName, self.ratios, strict=True)
            },
            "history_length": self.history_length,
            "max_rollout": self.max_rollout,
            "stacked_window": self.stacked_window,
            "mask_recompute_span": self.mask_recompute_span,
            "dependency_span": self.dependency_span,
            "minimum_gap": self.minimum_gap,
            "effective_gap": self.effective_gap,
            "window_stride": self.window_stride,
            "episode_length": self.episode_length,
            "target_risk_level": self.target_risk_level,
            "minimum_calibration_episodes": self.minimum_calibration_episodes,
            "seed": self.seed,
            "strategy": self.strategy,
        }


@dataclass(frozen=True)
class StageSlice:
    """一个正常阶段的准备后位置、原始索引、合法窗口、episode 和内容哈希。

    属性：
        stage: 受控阶段名。
        prepared_row_indices: 阶段在准备后正常数组中的全局位置。
        raw_indices: 与准备后位置逐行对齐、由 P1 保留的原始 ``raw_index``。
        prepared_window_starts: 合法窗口在准备后数组中的起点。
        raw_window_starts: 合法窗口对应的原始起点。
        dependency_span: 每个窗口读取的连续位置数。
        prepared_episode_ranges: 完整且原始索引连续的 episode 半开位置范围。
        discarded_prepared_episode_ranges: 因 raw_index 缺口被丢弃的完整候选范围。
        unused_prepared_episode_tail: 不足一个完整 episode 的阶段尾部。
        data_hash/index_hash/window_hash: 数据内容、索引和窗口配置的独立 SHA-256。
    重要约束：
        位置和原始索引不能混用。构造器只保留依赖位置完全落在本阶段、且对应
        ``raw_index`` 严格连续的窗口；校准阶段窗口还必须完全落在一个完整 episode 内。
    """

    stage: StageName
    prepared_row_indices: tuple[int, ...]
    raw_indices: tuple[int, ...]
    prepared_window_starts: tuple[int, ...]
    raw_window_starts: tuple[int, ...]
    dependency_span: int
    prepared_episode_ranges: tuple[tuple[int, int], ...]
    discarded_prepared_episode_ranges: tuple[tuple[int, int], ...]
    unused_prepared_episode_tail: tuple[int, int] | None
    data_hash: str
    index_hash: str
    window_hash: str

    @property
    def row_count(self) -> int:
        """返回本阶段准备后位置数。

        参数：
            无。
        返回：
            ``prepared_row_indices`` 的长度；它也等于对齐的 ``raw_indices`` 长度。
        异常：
            无。
        副作用：
            无。
        """

        return len(self.prepared_row_indices)

    @property
    def window_count(self) -> int:
        """返回依赖位置完全落在本阶段的合法窗口数。

        参数：
            无。
        返回：
            ``prepared_window_starts`` 的长度。
        异常：
            无。
        副作用：
            无。
        """

        return len(self.prepared_window_starts)

    @property
    def prepared_row_range(self) -> tuple[int, int]:
        """返回准备后数组中的半开位置范围 ``[start, stop)``。

        参数：
            无。
        返回：
            由首尾准备后位置得到的半开范围。
        异常：
            阶段没有准备后位置时抛出 ``ValueError``。
        副作用：
            无。
        """

        if not self.prepared_row_indices:
            raise ValueError(f"Stage {self.stage.value!r} does not contain any rows.")
        return self.prepared_row_indices[0], self.prepared_row_indices[-1] + 1

    def dependency_prepared_rows(self, prepared_window_start: int) -> tuple[int, ...]:
        """返回一个已登记窗口读取的全部准备后位置。

        参数：
            prepared_window_start: 必须出现在 ``prepared_window_starts`` 中的全局位置。
        返回：
            长度为 ``dependency_span`` 的连续准备后位置。
        异常：
            起点不属于本阶段合法窗口时抛出 ``ValueError``。
        副作用：
            无。
        """

        if prepared_window_start not in self.prepared_window_starts:
            raise ValueError(
                f"Prepared window start {prepared_window_start} is not legal for stage "
                f"{self.stage.value!r}."
            )
        return tuple(
            range(prepared_window_start, prepared_window_start + self.dependency_span)
        )

    def dependency_raw_rows(self, prepared_window_start: int) -> tuple[int, ...]:
        """返回一个合法窗口实际读取的原始 ``raw_index``。

        参数：
            prepared_window_start: ``prepared_window_starts`` 中的准备后全局位置。
        返回：
            与 ``dependency_span`` 等长且严格连续的原始行索引。
        异常：
            起点不合法或内部 raw_index 映射不完整时抛出 ``ValueError``。
        副作用：
            无。
        """

        self.dependency_prepared_rows(prepared_window_start)
        local_start = prepared_window_start - self.prepared_row_indices[0]
        values = self.raw_indices[local_start : local_start + self.dependency_span]
        if len(values) != self.dependency_span:
            raise ValueError(
                f"Prepared window start {prepared_window_start} does not have a complete "
                "raw-index dependency."
            )
        return values

    def manifest(self) -> dict[str, Any]:
        """返回明确区分准备后位置与原始索引的阶段 manifest。

        参数：
            无。
        返回：
            含位置、原始索引、窗口、episode 和三个 hash 的 JSON 可序列化新字典。
        异常：
            空阶段会由 ``prepared_row_range`` 抛出 ``ValueError``；合法切分不会出现。
        副作用：
            无。
        """

        return {
            "stage": self.stage.value,
            "prepared_row_range": list(self.prepared_row_range),
            "row_count": self.row_count,
            "prepared_row_indices": list(self.prepared_row_indices),
            "raw_indices": list(self.raw_indices),
            "window_count": self.window_count,
            "prepared_window_starts": list(self.prepared_window_starts),
            "raw_window_starts": list(self.raw_window_starts),
            "dependency_span": self.dependency_span,
            "prepared_episode_ranges": [
                list(item) for item in self.prepared_episode_ranges
            ],
            "discarded_prepared_episode_ranges": [
                list(item) for item in self.discarded_prepared_episode_ranges
            ],
            "unused_prepared_episode_tail": (
                None
                if self.unused_prepared_episode_tail is None
                else list(self.unused_prepared_episode_tail)
            ),
            "data_hash": self.data_hash,
            "index_hash": self.index_hash,
            "window_hash": self.window_hash,
        }


@dataclass(frozen=True)
class FiveStageSplitResult:
    """五个正常阶段、准备后隔离带、数据 hash 和确定性 split hash。

    属性：
        config: 冻结切分配置。
        slices: 按 ``StageName`` 索引的只读阶段映射。
        prepared_gap_ranges: 四个隔离带在准备后数组中的半开位置范围。
        effective_gap: 每个隔离带的实际位置数。
        data_hash: 完整准备后正常数组 hash。
        split_hash: 配置、阶段、索引、窗口和来源摘要的确定性 hash。
        source_hash: 可选正常原始文件 SHA-256。
    """

    config: FiveStageSplitConfig
    slices: Mapping[StageName, StageSlice]
    prepared_gap_ranges: tuple[tuple[int, int], ...]
    effective_gap: int
    data_hash: str
    split_hash: str
    source_hash: str | None = None

    def __post_init__(self) -> None:
        """冻结并验证阶段集合，防止调用方事后替换某一段。

        参数：
            无；校验当前结果对象的 ``slices``。
        返回：
            无。
        异常：
            阶段缺失、多余或键与 ``StageSlice.stage`` 不一致时抛出 ``ValueError``。
        副作用：
            通过 ``object.__setattr__`` 把复制后的字典替换为只读 ``MappingProxyType``。
        """

        copied = dict(self.slices)
        if set(copied) != set(StageName):
            missing = [stage.value for stage in StageName if stage not in copied]
            extra = [str(stage) for stage in copied if stage not in set(StageName)]
            raise ValueError(
                f"Five-stage result must contain every StageName exactly once. "
                f"Missing={missing}, extra={extra}."
            )
        for stage, stage_slice in copied.items():
            if stage_slice.stage is not stage:
                raise ValueError(
                    f"Stage mapping key {stage.value!r} does not match slice "
                    f"{stage_slice.stage.value!r}."
                )
        object.__setattr__(self, "slices", MappingProxyType(copied))

    def stage(self, name: StageName | str) -> StageSlice:
        """按受控名称返回一个阶段。

        参数：
            name: ``StageName`` 或其精确字符串值。
        返回：
            对应的只读 ``StageSlice``。
        异常：
            未知拼写时由 ``StageName.parse`` 抛出列出合法选项的 ``ValueError``。
        副作用：
            无。
        """

        return self.slices[StageName.parse(name)]

    def manifest(self) -> dict[str, Any]:
        """返回完整、确定性且可持久化的切分 manifest。

        参数：
            无。
        返回：
            含协议名、配置、准备后隔离带、hash 和五个阶段的 JSON 可序列化新字典。
        异常：
            无。
        副作用：
            无。
        """

        return {
            "protocol": "five_stage_normal_only",
            "config": self.config.manifest(),
            "effective_gap": self.effective_gap,
            "prepared_gap_ranges": [
                list(item) for item in self.prepared_gap_ranges
            ],
            "data_hash": self.data_hash,
            "source_hash": self.source_hash,
            "split_hash": self.split_hash,
            "stages": {
                stage.value: self.stage(stage).manifest() for stage in StageName
            },
        }

    def save_manifest(self, path: str | Path) -> Path:
        """把完整切分 manifest 写入显式 JSON 路径。

        参数：
            path: 目标文件路径；父目录不存在时创建。
        返回：
            写入后的 ``Path``。
        异常：
            目录创建或文件写入失败时传播 ``OSError``。
        副作用：
            创建父目录并以 UTF-8 覆盖目标 JSON；核心 ``split`` 调用本身仍无文件副作用。
        """

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.manifest(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output


class FitPurpose(str, Enum):
    """论文流水线中会读取阶段数据的受控拟合或最终检查用途。"""

    MODEL_PARAMETERS = "model_parameters"
    STRUCTURE_SELECTION = "structure_selection"
    MONITORING_SCORE_SCALER = "monitoring_score_scaler"
    ENVELOPE = "envelope"
    COVARIANCE = "covariance"
    BRANCH_LIBRARY = "branch_library"
    STATE_MACHINE = "state_machine"
    DETECTION_QUANTILE = "detection_quantile"
    ATTRIBUTION_QUANTILE = "attribution_quantile"
    FROZEN_NORMAL_DIAGNOSTIC = "frozen_normal_diagnostic"

    @classmethod
    def parse(cls, value: "FitPurpose | str") -> "FitPurpose":
        """把枚举或字符串解析为受控拟合用途。

        参数：
            value: ``FitPurpose`` 或其精确字符串值。
        返回：
            对应的 ``FitPurpose``。
        异常：
            未知用途时抛出 ``ValueError``，错误消息列出全部合法选项。
        副作用：
            无。
        """

        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError as exc:
            legal = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unknown paper fit purpose {value!r}. Legal options are: {legal}."
            ) from exc


_FIT_STAGE_POLICY: Mapping[FitPurpose, tuple[StageName, ...]] = MappingProxyType(
    {
        FitPurpose.MODEL_PARAMETERS: (StageName.TRAIN,),
        FitPurpose.STRUCTURE_SELECTION: (StageName.ESTIMATE,),
        FitPurpose.MONITORING_SCORE_SCALER: (StageName.ESTIMATE,),
        FitPurpose.ENVELOPE: (StageName.ESTIMATE,),
        FitPurpose.COVARIANCE: (StageName.ESTIMATE,),
        FitPurpose.BRANCH_LIBRARY: (StageName.ESTIMATE,),
        FitPurpose.STATE_MACHINE: (StageName.ESTIMATE,),
        FitPurpose.DETECTION_QUANTILE: (StageName.DETECTION_CALIBRATION,),
        FitPurpose.ATTRIBUTION_QUANTILE: (StageName.ATTRIBUTION_CALIBRATION,),
        FitPurpose.FROZEN_NORMAL_DIAGNOSTIC: (StageName.FROZEN_NORMAL_TEST,),
    }
)

_ESTIMATE_PURPOSES = frozenset(
    {
        FitPurpose.STRUCTURE_SELECTION,
        FitPurpose.MONITORING_SCORE_SCALER,
        FitPurpose.ENVELOPE,
        FitPurpose.COVARIANCE,
        FitPurpose.BRANCH_LIBRARY,
        FitPurpose.STATE_MACHINE,
    }
)
_DESIGN_PURPOSES = frozenset({FitPurpose.MODEL_PARAMETERS, *_ESTIMATE_PURPOSES})


@dataclass(frozen=True)
class FitAccessRecord:
    """记录一个拟合对象读取过的正常阶段及其冻结产物身份。

    参数：
        object_id: 在单份账本内唯一且非空的拟合对象标识，例如模型、监测分数尺度量或
            校准分位；同一对象不能以另一用途重复登记。
        purpose: 受 ``FitPurpose`` 约束的用途。用途决定允许读取的唯一正常阶段，避免
            模型参数、监测设计量和校准分位跨阶段取数。
        stages: 对象实际读取的 ``StageName`` 元组；必须与用途策略完全一致，不能把
            ``frozen_normal_test`` 伪装成可调参数据。
        stage_hashes: 阶段名到访问时数据 SHA-256 的映射，用于冻结后审计“对象看过什么”。
        frozen: 对象是否已停止拟合并冻结；只有冻结对象才能推进后续协议阶段。
        artifact_hash: 冻结对象对应 checkpoint、统计参数或分位产物的 64 位 SHA-256；
            未冻结时必须为 ``None``。
    异常：
        ``artifact_hash`` 格式错误，或未冻结记录携带产物 hash 时抛出 ``ValueError``。
        对象唯一性、用途与阶段权限由 ``FitAccessLedger`` 在构造本记录前校验。
    副作用：
        构造时复制 ``stage_hashes`` 并包装为只读映射；不访问数据、不写文件，也不改变
        账本生命周期。
    """

    object_id: str
    purpose: FitPurpose
    stages: tuple[StageName, ...]
    stage_hashes: Mapping[str, str]
    frozen: bool = False
    artifact_hash: str | None = None

    def __post_init__(self) -> None:
        """复制阶段 hash，并校验冻结状态与拟合产物 hash 一致。

        参数：
            无；校验当前不可变记录。
        返回：
            无。
        异常：
            ``artifact_hash`` 不是 SHA-256，或未冻结记录却带产物 hash 时抛出
            ``ValueError``。
        副作用：
            通过 ``object.__setattr__`` 把阶段 hash 替换为只读副本，并规范化产物 hash。
        """

        object.__setattr__(self, "stage_hashes", MappingProxyType(dict(self.stage_hashes)))
        normalized_hash = _source_hash(
            self.artifact_hash,
            field_name="fit artifact_hash",
        )
        if not self.frozen and normalized_hash is not None:
            raise ValueError("An unfrozen fit access record cannot carry artifact_hash.")
        object.__setattr__(self, "artifact_hash", normalized_hash)

    def manifest(self) -> dict[str, Any]:
        """返回 JSON 可序列化的单条访问记录。

        参数：
            无。
        返回：
            含对象标识、用途、阶段、阶段 hash、冻结状态和产物 hash 的新字典。
        异常：
            无。
        副作用：
            无；返回值不暴露内部只读映射。
        """

        return {
            "object_id": self.object_id,
            "purpose": self.purpose.value,
            "stages": [stage.value for stage in self.stages],
            "stage_hashes": dict(self.stage_hashes),
            "frozen": self.frozen,
            "artifact_hash": self.artifact_hash,
        }


class FitAccessLedger:
    """强制并记录每个拟合对象允许读取的正常数据阶段。

    账本持有 ``FiveStageSplitResult``，因此每条记录不仅写阶段名，还固定当时的阶段内容
    hash。相同 object_id 不能重复登记，避免一次合法拟合后又用其他阶段静默重拟合。
    """

    def __init__(self, split_result: FiveStageSplitResult) -> None:
        """创建空账本并绑定一个确定切分。

        参数：
            split_result: 后续所有记录引用的五阶段切分。
        返回：
            无。
        异常：
            无；``FiveStageSplitResult`` 已在自身构造时验证完整性。
        副作用：
            只初始化内存记录，不访问阶段数组或写文件。
        """

        self.split_result = split_result
        self._records: dict[str, FitAccessRecord] = {}
        self._frozen = False

    @property
    def frozen(self) -> bool:
        """返回账本是否已随协议冻结而禁止新增拟合。

        参数：
            无。
        返回：
            已调用 ``freeze`` 时为真，否则为假。
        异常：
            无。
        副作用：
            无。
        """

        return self._frozen

    @property
    def protocol_ready(self) -> bool:
        """返回账本是否具备冻结正式协议所需的最小证据。

        参数：
            无。
        返回：
            模型、至少一种估计量、检测分位、归因分位和冻结正常诊断均已登记并冻结，
            且所有记录都已冻结时为真。
        异常：
            无。
        副作用：
            无。
        """

        return not self._protocol_readiness_errors()

    def record_fit(
        self,
        object_id: str,
        purpose: FitPurpose | str,
        stages: Sequence[StageName | str],
    ) -> FitAccessRecord:
        """验证阶段策略并登记一次拟合/最终正常检查访问。

        参数：
            object_id: checkpoint、scaler、阈值或诊断对象的稳定标识。
            purpose: 受控拟合用途。
            stages: 对象实际读取的阶段；不能为空或超出用途策略。
        返回：
            已冻结阶段 hash 的 ``FitAccessRecord``。
        异常：
            账本已冻结、标识为空/重复、阶段为空，或用途访问了禁止阶段时抛出
            ``ValueError`` 或 ``RuntimeError``。
        副作用：
            验证成功后把一条记录加入当前内存账本；不读取数组、不写文件。
        """

        if self._frozen:
            raise RuntimeError("Fit access ledger is frozen; no additional fit access is allowed.")
        normalized_id = str(object_id).strip()
        if not normalized_id:
            raise ValueError("Fit access object_id cannot be empty.")
        if normalized_id in self._records:
            raise ValueError(
                f"Fit access object_id {normalized_id!r} is already recorded; "
                "silent refitting is not allowed."
            )
        normalized_purpose = FitPurpose.parse(purpose)
        self._validate_record_order(normalized_purpose)
        normalized_stages = tuple(dict.fromkeys(StageName.parse(stage) for stage in stages))
        if not normalized_stages:
            raise ValueError("Fit access stages cannot be empty.")
        allowed = _FIT_STAGE_POLICY[normalized_purpose]
        illegal = [stage for stage in normalized_stages if stage not in allowed]
        if illegal:
            legal = ", ".join(stage.value for stage in allowed)
            requested = ", ".join(stage.value for stage in normalized_stages)
            raise ValueError(
                f"Fit purpose {normalized_purpose.value!r} cannot access stages: {requested}. "
                f"Legal stages are: {legal}."
            )
        record = FitAccessRecord(
            object_id=normalized_id,
            purpose=normalized_purpose,
            stages=normalized_stages,
            stage_hashes={
                stage.value: self.split_result.stage(stage).data_hash
                for stage in normalized_stages
            },
        )
        self._records[normalized_id] = record
        return record

    def freeze_record(self, object_id: str, artifact_hash: str) -> FitAccessRecord:
        """用拟合产物 SHA-256 冻结一条已登记访问记录。

        参数：
            object_id: 先前 ``record_fit`` 登记的稳定标识。
            artifact_hash: 对 checkpoint、统计参数、分位或诊断产物计算的 SHA-256。
        返回：
            替换后的冻结 ``FitAccessRecord``。
        异常：
            账本已冻结、对象不存在、hash 非法，或同一记录以不同 hash 重复冻结时抛出
            ``RuntimeError``、``KeyError`` 或 ``ValueError``。
        副作用：
            用不可变的冻结记录替换内存账本中的原记录；不读取阶段数据、不写文件。
        """

        if self._frozen:
            raise RuntimeError("Fit access ledger is frozen; records cannot be changed.")
        if object_id not in self._records:
            raise KeyError(f"Unknown fit access object_id {object_id!r}.")
        normalized_hash = _source_hash(
            artifact_hash,
            field_name=f"artifact_hash for {object_id!r}",
        )
        if normalized_hash is None:
            raise ValueError("artifact_hash cannot be empty when freezing a fit record.")
        current = self._records[object_id]
        if current.frozen:
            if current.artifact_hash != normalized_hash:
                raise ValueError(
                    f"Fit access object {object_id!r} is already frozen with a different hash."
                )
            return current
        frozen_record = FitAccessRecord(
            object_id=current.object_id,
            purpose=current.purpose,
            stages=current.stages,
            stage_hashes=current.stage_hashes,
            frozen=True,
            artifact_hash=normalized_hash,
        )
        self._records[object_id] = frozen_record
        return frozen_record

    def freeze(self) -> None:
        """冻结账本，防止协议冻结后继续拟合或重算阈值。

        参数：
            无。
        返回：
            无。
        异常：
            最小协议对象缺失或任一已登记对象尚未冻结时抛出 ``ProtocolAccessError``。
        副作用：
            将内存状态置为只读；现有记录不变。
        """

        errors = self._protocol_readiness_errors()
        if errors:
            raise ProtocolAccessError(
                "Paper access ledger is incomplete and cannot be frozen: "
                + "; ".join(errors)
                + "."
            )
        self._frozen = True

    def manifest(self) -> dict[str, Any]:
        """返回策略、冻结状态、就绪状态和按标识排序的访问记录。

        参数：
            无。
        返回：
            JSON 可序列化的新字典；记录按 ``object_id`` 排序以保证稳定 hash/差异。
        异常：
            无。
        副作用：
            无。
        """

        return {
            "split_hash": self.split_result.split_hash,
            "frozen": self._frozen,
            "protocol_ready": self.protocol_ready,
            "policy": {
                purpose.value: [stage.value for stage in stages]
                for purpose, stages in _FIT_STAGE_POLICY.items()
            },
            "records": [
                self._records[object_id].manifest()
                for object_id in sorted(self._records)
            ],
        }

    def _validate_record_order(self, purpose: FitPurpose) -> None:
        """检查新增访问是否符合模型、估计、两次校准和正常测试顺序。

        参数：
            purpose: 即将登记的受控用途。
        返回：
            无。
        异常：
            设计量在检测校准后重开、归因早于检测分位冻结、正常测试早于归因分位冻结，
            或冻结正常测试后重新拟合时抛出 ``ProtocolAccessError``。
        副作用：
            无。
        """

        records = tuple(self._records.values())
        if any(item.purpose is FitPurpose.FROZEN_NORMAL_DIAGNOSTIC for item in records):
            if purpose is not FitPurpose.FROZEN_NORMAL_DIAGNOSTIC:
                raise ProtocolAccessError(
                    "Paper fitting is forbidden because frozen-normal access has closed "
                    "the design and calibration stages."
                )
        if purpose in _DESIGN_PURPOSES and any(
            item.purpose is FitPurpose.DETECTION_QUANTILE for item in records
        ):
            raise ProtocolAccessError(
                "Model and estimate-stage fitting cannot resume after detection calibration."
            )
        if purpose is FitPurpose.DETECTION_QUANTILE and any(
            item.purpose is FitPurpose.ATTRIBUTION_QUANTILE for item in records
        ):
            raise ProtocolAccessError(
                "Detection quantiles cannot be added after attribution calibration begins."
            )
        if purpose is FitPurpose.ATTRIBUTION_QUANTILE and not self._all_frozen(
            FitPurpose.DETECTION_QUANTILE
        ):
            raise ProtocolAccessError(
                "Attribution calibration requires at least one frozen detection quantile."
            )
        if purpose is FitPurpose.FROZEN_NORMAL_DIAGNOSTIC and not self._all_frozen(
            FitPurpose.ATTRIBUTION_QUANTILE
        ):
            raise ProtocolAccessError(
                "Frozen-normal access requires at least one frozen attribution quantile."
            )

    def _all_frozen(self, purpose: FitPurpose) -> bool:
        """判断某用途是否至少登记一次且该用途的全部记录均已冻结。

        参数：
            purpose: 要检查的受控用途。
        返回：
            至少一条匹配记录且全部 ``frozen=True`` 时为真。
        异常：
            无。
        副作用：
            无。
        """

        records = [item for item in self._records.values() if item.purpose is purpose]
        return bool(records) and all(item.frozen for item in records)

    def _protocol_readiness_errors(self) -> list[str]:
        """列出阻止账本冻结的缺失阶段或未冻结对象。

        参数：
            无。
        返回：
            可直接写入错误消息的稳定字符串列表；空列表表示协议就绪。
        异常：
            无。
        副作用：
            无。
        """

        errors: list[str] = []
        if not self._all_frozen(FitPurpose.MODEL_PARAMETERS):
            errors.append("a frozen model-parameter record is required")
        estimate_records = [
            item for item in self._records.values() if item.purpose in _ESTIMATE_PURPOSES
        ]
        if not estimate_records or not all(item.frozen for item in estimate_records):
            errors.append("at least one frozen estimate-stage record is required")
        for purpose, label in (
            (FitPurpose.DETECTION_QUANTILE, "detection quantile"),
            (FitPurpose.ATTRIBUTION_QUANTILE, "attribution quantile"),
            (FitPurpose.FROZEN_NORMAL_DIAGNOSTIC, "frozen-normal diagnostic"),
        ):
            if not self._all_frozen(purpose):
                errors.append(f"a frozen {label} record is required")
        unfrozen = sorted(item.object_id for item in self._records.values() if not item.frozen)
        if unfrozen:
            errors.append(f"unfrozen records remain: {', '.join(unfrozen)}")
        return errors


class ProtocolAccessError(RuntimeError):
    """表示请求违反论文数据阶段、冻结顺序或许可门禁。

    该异常只描述协议访问被拒绝，不表示数据文件损坏或模型数值失败；调用方应修正
    阶段顺序、补齐 hash/许可或重新建立协议，不能捕获后继续使用被拒绝的数据。
    """


class PaperDataBundle:
    """组合五段正常数据、拟合账本和独立封存故障测试。

    正常数组与故障数组在构造时都复制到内部，调用方只能获得阶段或故障数据的副本。
    因此外部修改返回数组不会改变 split hash、账本或后续评价输入。
    """

    def __init__(
        self,
        normal_data: np.ndarray | Sequence[Any],
        *,
        config: FiveStageSplitConfig | None = None,
        normal_raw_indices: Sequence[int] | np.ndarray | None = None,
        normal_source_hash: str | None = None,
        frozen_fault_test: np.ndarray | Sequence[Any] | None = None,
        fault_source_hash: str | None = None,
        fault_license_status: FaultLicenseStatus | str = FaultLicenseStatus.TO_VERIFY,
    ) -> None:
        """建立正常协议并把可选故障测试保持为未访问状态。

        参数：
            normal_data: 只含正常行的准备后长序列。
            config: 五阶段配置；为空时使用默认值。
            normal_raw_indices: 与正常数组逐行对齐的 P1 原始行号。
            normal_source_hash: 可选正常原始文件 hash。
            frozen_fault_test: 五段之外的独立故障测试数组；构造只封存，不授权评价。
            fault_source_hash: 可选故障原始文件 hash。
            fault_license_status: 受控许可状态；只有 ``verified`` 才允许协议冻结后返回
                故障数组，CSTR 当前必须保持 ``to_verify``。
        异常：
            正常/故障数组非数值或非有限、许可状态为空，或五阶段切分失败时抛出
            ``ValueError``。
        返回：
            无。
        副作用：
            只复制数组、计算 hash 和创建内存账本；不读写文件、不运行故障评价。
        """

        self._normal_data = _normal_array(normal_data)
        self.split_result = FiveStageNormalSplitter(config).split(
            self._normal_data,
            raw_indices=normal_raw_indices,
            source_hash=normal_source_hash,
        )
        self.fit_access_ledger = FitAccessLedger(self.split_result)
        self._fault_data = (
            None
            if frozen_fault_test is None
            else _finite_array(frozen_fault_test, field_name="frozen_fault_test")
        )
        self.fault_license_status = FaultLicenseStatus.parse(fault_license_status)
        self.fault_source_hash = _source_hash(
            fault_source_hash,
            field_name="fault_source_hash",
        )
        self.fault_data_hash = (
            None if self._fault_data is None else _hash_array(self._fault_data)
        )
        self._protocol_frozen = False
        self._fault_accessed = False

    @classmethod
    def from_canonical(
        cls,
        canonical: CanonicalDataset,
        *,
        config: FiveStageSplitConfig | None = None,
        normal_split: str = "train",
        input_roles: tuple[str, ...] = ("control_input", "measured_output"),
        normal_label: int | float = 0,
        normal_source_hash: str | None = None,
    ) -> "PaperDataBundle":
        """从 P1 ``CanonicalDataset`` 的一个正常 split 建立五阶段 bundle。

        参数：
            canonical: 已由适配器修正 schema、逐行标签和 raw_index 的规范数据集。
            config: 五阶段配置。
            normal_split: 只允许读取的正常 split，默认 ``train``。
            input_roles: schema 中进入论文模型的物理角色；追溯列和标签不会被选择。
            normal_label: 该 split 每一行必须具有的正常标签。
            normal_source_hash: 可选显式来源 hash；为空时尝试从 canonical ``files`` 摘要读取。
        返回：
            只包含所选正常 split 的 ``PaperDataBundle``；canonical 的 test/fault split
            不会被复制或配置为 ``frozen_fault_test``。
        异常：
            split/角色/raw_index/标签缺失，标签含故障值，跨 segment raw_index 不严格递增，
            或物理列非数值时抛出 ``ValueError``。
        副作用：
            无文件访问。只读取传入对象的 frame，并把数值与 raw_index 复制到新 bundle。
        """

        segments = canonical.splits.get(normal_split)
        if not segments:
            legal = ", ".join(sorted(canonical.splits))
            raise ValueError(
                f"Canonical dataset does not provide normal split {normal_split!r}. "
                f"Legal options are: {legal}."
            )
        input_columns: list[str] = []
        for role in input_roles:
            columns = canonical.schema.role_columns(role)
            if not columns:
                legal = ", ".join(canonical.schema.roles)
                raise ValueError(
                    f"Canonical schema does not define paper input role {role!r}. "
                    f"Legal roles are: {legal}."
                )
            for column in columns:
                if column not in input_columns:
                    input_columns.append(column)
        raw_columns = canonical.schema.role_columns("raw_index")
        if len(raw_columns) != 1:
            raise ValueError(
                "Paper canonical input requires exactly one raw_index column. "
                f"Current columns: {raw_columns}."
            )
        label_columns = canonical.schema.role_columns("fault_id")
        if len(label_columns) != 1:
            raise ValueError(
                "Paper canonical input requires exactly one fault_id column to prove normal-only "
                f"access. Current columns: {label_columns}."
            )

        value_chunks: list[np.ndarray] = []
        raw_chunks: list[np.ndarray] = []
        for segment in segments:
            required = [*input_columns, raw_columns[0], label_columns[0]]
            missing = [column for column in required if column not in segment.frame.columns]
            if missing:
                raise ValueError(
                    f"Canonical normal segment {segment.meta.segment_id!r} is missing columns: "
                    f"{', '.join(missing)}."
                )
            labels = segment.frame.loc[:, label_columns[0]].to_numpy()
            if not np.equal(labels, normal_label).all():
                observed = sorted({str(value) for value in np.unique(labels)})
                raise ValueError(
                    f"Canonical split {normal_split!r} is not normal-only. "
                    f"Expected label={normal_label}, observed={observed}."
                )
            value_chunks.append(
                segment.frame.loc[:, input_columns].to_numpy(dtype=float)
            )
            raw_chunks.append(
                segment.frame.loc[:, raw_columns[0]].to_numpy()
            )
        values = np.concatenate(value_chunks, axis=0)
        raw_indices = np.concatenate(raw_chunks, axis=0)
        resolved_source_hash = normal_source_hash or _canonical_split_source_hash(
            canonical,
            normal_split,
        )
        return cls(
            values,
            config=config,
            normal_raw_indices=raw_indices,
            normal_source_hash=resolved_source_hash,
        )

    @property
    def protocol_frozen(self) -> bool:
        """返回正常协议和拟合账本是否已冻结。

        参数：
            无。
        返回：
            ``freeze_protocol`` 成功后为真，否则为假。
        异常：
            无。
        副作用：
            无。
        """

        return self._protocol_frozen

    @property
    def fault_accessed(self) -> bool:
        """返回本进程是否已经成功请求过封存故障数组。

        参数：
            无。
        返回：
            ``request_frozen_fault_test`` 至少成功一次时为真。
        异常：
            无。
        副作用：
            无。
        """

        return self._fault_accessed

    def _stage_data(self, stage: StageName | str) -> np.ndarray:
        """仅供已登记访问路径复制一个正常阶段数组。

        参数：
            stage: 受控阶段名。
        返回：
            按 ``StageSlice.prepared_row_indices`` 取得的数组副本。
        异常：
            未知阶段名时由 ``StageName.parse`` 抛出 ``ValueError``。
        副作用：
            无。该内部函数不登记账本，公开调用方只能使用 ``data_for_fit``。
        """

        stage_slice = self.split_result.stage(stage)
        positions = np.asarray(stage_slice.prepared_row_indices, dtype=int)
        return self._normal_data[positions].copy()

    def data_for_fit(
        self,
        object_id: str,
        purpose: FitPurpose | str,
    ) -> np.ndarray:
        """登记拟合用途后返回该用途唯一允许的正常阶段副本。

        参数：
            object_id: 拟合产物的稳定标识。
            purpose: 受控拟合/最终正常检查用途。
        返回：
            用途策略唯一允许阶段的数据副本。
        异常：
            用途非法、object_id 重复或账本已冻结时传播账本错误。
        副作用：
            在 ``fit_access_ledger`` 中新增一条绑定阶段 hash 的记录。
        """

        normalized_purpose = FitPurpose.parse(purpose)
        allowed = _FIT_STAGE_POLICY[normalized_purpose]
        if len(allowed) != 1:
            raise RuntimeError(
                f"Fit purpose {normalized_purpose.value!r} does not resolve one stage."
            )
        self.fit_access_ledger.record_fit(
            object_id,
            normalized_purpose,
            allowed,
        )
        return self._stage_data(allowed[0])

    def freeze_protocol(self, expected_split_hash: str) -> None:
        """核对 split hash 后冻结正常协议与拟合账本。

        参数：
            expected_split_hash: 调用方已经审阅并准备冻结的完整 split manifest hash。
        返回：
            无。
        异常：
            hash 与当前切分不一致时抛出 ``ValueError``；已经冻结时重复调用保持幂等。
        副作用：
            将协议与账本置为冻结状态，后续不能再登记拟合或重算阈值。
        """

        if expected_split_hash != self.split_result.split_hash:
            raise ValueError(
                "Cannot freeze paper protocol with a different split hash. "
                f"expected={expected_split_hash!r}, actual={self.split_result.split_hash!r}."
            )
        if self._protocol_frozen:
            return
        if self._fault_data is not None:
            missing_sources = []
            if self.split_result.source_hash is None:
                missing_sources.append("normal_source_hash")
            if self.fault_source_hash is None:
                missing_sources.append("fault_source_hash")
            if missing_sources:
                raise ProtocolAccessError(
                    "Frozen fault evaluation requires source SHA-256 values for: "
                    + ", ".join(missing_sources)
                    + "."
                )
        self.fit_access_ledger.freeze()
        self._protocol_frozen = True

    def request_frozen_fault_test(self) -> np.ndarray:
        """在协议冻结且许可核实后返回独立故障测试副本。

        参数：
            无。
        返回：
            封存故障数组的副本；调用方修改不会污染内部数据。
        异常：
            协议未冻结、未配置故障范围或许可状态不是 ``verified`` 时抛出
            ``ProtocolAccessError``。
        副作用：
            成功时把 ``fault_accessed`` 标为真；不计算指标、不写产物。
        """

        if not self._protocol_frozen:
            raise ProtocolAccessError(
                "Frozen fault test is not available because the paper protocol is not frozen."
            )
        if self._fault_data is None:
            raise ProtocolAccessError("Frozen fault test was not configured for this bundle.")
        if self.fault_license_status is not FaultLicenseStatus.VERIFIED:
            raise ProtocolAccessError(
                "Frozen fault test license is not verified. "
                f"Current status: {self.fault_license_status.value!r}."
            )
        self._fault_accessed = True
        return self._fault_data.copy()

    def manifest(self) -> dict[str, Any]:
        """返回不含数组值的 bundle、split、账本和故障封存状态摘要。

        参数：
            无。
        返回：
            JSON 可序列化的新字典；只含 hash、行数、状态和嵌套 manifest。
        异常：
            无。
        副作用：
            无。
        """

        return {
            "protocol_frozen": self._protocol_frozen,
            "split": self.split_result.manifest(),
            "fit_access_ledger": self.fit_access_ledger.manifest(),
            "frozen_fault_test": {
                "configured": self._fault_data is not None,
                "rows": 0 if self._fault_data is None else int(self._fault_data.shape[0]),
                "prepared_data_hash": self.fault_data_hash,
                "source_hash": self.fault_source_hash,
                "license_status": self.fault_license_status.value,
                "accessed": self._fault_accessed,
            },
        }

    def save_protocol_artifacts(self, directory: str | Path) -> dict[str, Path]:
        """把 P2 split、访问账本和 bundle 摘要写入显式目录。

        参数：
            directory: 目标目录；不存在时创建。
        返回：
            ``split_manifest``、``fit_access_ledger`` 和 ``paper_data_bundle`` 到实际
            JSON 路径的映射。
        异常：
            目录创建或任一文件写入失败时传播 ``OSError``。
        副作用：
            创建目录并覆盖三个固定名称 JSON；不写正常/故障数组值，不改变冻结状态。
        """

        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "split_manifest": output_dir / "split_manifest.json",
            "fit_access_ledger": output_dir / "fit_access_ledger.json",
            "paper_data_bundle": output_dir / "paper_data_bundle.json",
        }
        _write_json(paths["split_manifest"], self.split_result.manifest())
        _write_json(paths["fit_access_ledger"], self.fit_access_ledger.manifest())
        _write_json(paths["paper_data_bundle"], self.manifest())
        return paths


class FiveStageNormalSplitter:
    """在原始时间轴上执行五段正常数据切分与保守窗口过滤。"""

    def __init__(self, config: FiveStageSplitConfig | None = None) -> None:
        """保存冻结配置；不读取或切分任何数据。

        参数：
            config: 五阶段配置；为空时使用论文当前默认值。
        返回：
            无。
        异常：
            配置非法时由 ``FiveStageSplitConfig`` 构造传播 ``ValueError``。
        副作用：
            无。
        """

        self.config = config or FiveStageSplitConfig()

    def split(
        self,
        normal_data: np.ndarray | Sequence[Any],
        *,
        raw_indices: Sequence[int] | np.ndarray | None = None,
        source_hash: str | None = None,
    ) -> FiveStageSplitResult:
        """切分一条只含正常样本的准备后长序列。

        参数：
            normal_data: 第一维是原始时间行的有限数值数组；不会被修改。
            raw_indices: 与准备后数组逐行对齐的 P1 原始行号；为空时使用 ``0..n-1``。
            source_hash: 可选原始文件 hash，仅用于追溯，不参与阶段分配。
        返回：
            五段行/窗口/episode、四个隔离带及数据与 split hash。
        异常：
            数据非数值、含 NaN/Inf、扣除隔离带后任一阶段没有合法窗口，或两个校准阶段
            的完整 episode 数不足目标风险分辨率时抛出 ``ValueError``。
        副作用：
            无。切分按时间顺序确定，不读取故障数据、不写产物、不改变随机全局状态。
        """

        data = _normal_array(normal_data)
        resolved_source_hash = _source_hash(
            source_hash,
            field_name="normal source_hash",
        )
        resolved_raw_indices = _raw_index_array(
            raw_indices,
            expected_rows=data.shape[0],
        )
        gap = self.config.effective_gap
        available_rows = data.shape[0] - gap * (len(StageName) - 1)
        if available_rows <= 0:
            raise ValueError(
                "Normal sequence is too short after removing four isolation gaps. "
                f"rows={data.shape[0]}, effective_gap={gap}, available={available_rows}."
            )
        counts = _allocate_stage_counts(available_rows, self.config.ratios)
        slices: dict[StageName, StageSlice] = {}
        gaps: list[tuple[int, int]] = []
        cursor = 0
        for index, (stage, count) in enumerate(zip(StageName, counts, strict=True)):
            start = cursor
            stop = start + count
            slices[stage] = _stage_slice(
                stage=stage,
                start=start,
                stop=stop,
                data=data,
                raw_indices=resolved_raw_indices,
                config=self.config,
            )
            cursor = stop
            if index < len(StageName) - 1:
                gaps.append((cursor, cursor + gap))
                cursor += gap
        if cursor != data.shape[0]:
            raise RuntimeError(
                f"Five-stage allocation did not consume the normal timeline. "
                f"cursor={cursor}, rows={data.shape[0]}."
            )
        _validate_disjoint_slices(slices)

        data_hash = _hash_array(data)
        payload = {
            "protocol": "five_stage_normal_only",
            "config": self.config.manifest(),
            "prepared_gap_ranges": [list(item) for item in gaps],
            "data_hash": data_hash,
            "source_hash": resolved_source_hash,
            "stages": {
                stage.value: slices[stage].manifest() for stage in StageName
            },
        }
        return FiveStageSplitResult(
            config=self.config,
            slices=slices,
            prepared_gap_ranges=tuple(gaps),
            effective_gap=gap,
            data_hash=data_hash,
            split_hash=_hash_json(payload),
            source_hash=resolved_source_hash,
        )


def _normal_array(value: np.ndarray | Sequence[Any]) -> np.ndarray:
    """复制并验证准备后正常数组。

    参数：
        value: 只含正常时间行的数值序列或数组。
    返回：
        至少二维、有限、C 连续且独立的浮点数组。
    异常：
        输入非数值、没有时间行或含 NaN/Inf 时传播 ``_finite_array`` 的
        ``ValueError``。
    副作用：
        无。
    """

    return _finite_array(value, field_name="normal_data")


def _source_hash(value: str | None, *, field_name: str) -> str | None:
    """验证可选来源或产物标识是规范 SHA-256 十六进制字符串。

    参数：
        value: 64 位十六进制 SHA-256；``None`` 表示当前范围尚无可核验来源。
        field_name: 错误消息中的字段名。
    返回：
        统一为小写的 SHA-256，或 ``None``。
    异常：
        非空值长度不是 64 或含非十六进制字符时抛出 ``ValueError``。
    副作用：
        无。
    """

    if value is None:
        return None
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(
            f"{field_name} must be a 64-character SHA-256 hex digest. "
            f"Current input: {value!r}."
        )
    return normalized


def _finite_array(
    value: np.ndarray | Sequence[Any],
    *,
    field_name: str,
) -> np.ndarray:
    """把一维/多维数值输入复制为以时间行为第一维的有限连续数组。

    参数：
        value: 数值序列或 NumPy 数组。
        field_name: 错误消息使用的调用字段名。
    返回：
        至少二维、C 连续且与调用方内存独立的浮点数组。
    异常：
        无法转成数值、没有时间行或含 NaN/Inf 时抛出 ``ValueError``。
    副作用：
        无。
    """

    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a numeric array.") from exc
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim < 2 or array.shape[0] == 0:
        raise ValueError(
            f"{field_name} must contain at least one time row. Current shape: {array.shape}."
        )
    if not np.isfinite(array).all():
        raise ValueError(
            f"{field_name} must be finite before paper protocol use; apply allowed "
            "normal-only preprocessing first."
        )
    return np.ascontiguousarray(array.copy())


def _raw_index_array(
    value: Sequence[int] | np.ndarray | None,
    *,
    expected_rows: int,
) -> np.ndarray:
    """验证准备后每行对应的唯一、严格递增原始行号。

    参数：
        value: 一维整数 raw_index；为空时生成连续零基索引。
        expected_rows: 准备后正常数组的行数。
    返回：
        与正常数组逐行对齐的独立 ``int64`` 数组。
    异常：
        维数/长度错误、含非整数、重复或倒序时抛出 ``ValueError``。
    副作用：
        无。
    """

    if value is None:
        return np.arange(expected_rows, dtype=np.int64)
    array = np.asarray(value)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or array.shape[0] != expected_rows:
        raise ValueError(
            "raw_indices must provide one value per normal_data row. "
            f"Current shape: {array.shape}, expected rows={expected_rows}."
        )
    try:
        numeric = array.astype(np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("raw_indices must contain integers.") from exc
    if not np.array_equal(array, numeric):
        raise ValueError("raw_indices must contain exact integers.")
    if numeric.size > 1 and np.any(np.diff(numeric) <= 0):
        raise ValueError("raw_indices must be unique and strictly increasing.")
    return numeric.copy()


def _canonical_split_source_hash(
    canonical: CanonicalDataset,
    split: str,
) -> str | None:
    """从适配器来源摘要中提取指定 split 的已核验原始文件 hash。

    缺少 ``files``、split 条目或 ``sha256`` 时返回 ``None``，因为部分合成/通用适配器
    没有单文件来源；不会根据路径字符串伪造 hash。
    参数：
        canonical: 已解析的规范数据集。
        split: 要查询的 split 名称。
    返回：
        数据集摘要中的 SHA-256 字符串，或无法证明来源时的 ``None``。
    异常：
        无；结构缺失或类型不符均安全降级为 ``None``。
    副作用：
        无。
    """

    files = canonical.metadata.get("files")
    if not isinstance(files, Mapping):
        return None
    item = files.get(split)
    if not isinstance(item, Mapping):
        return None
    value = item.get("sha256")
    return str(value) if value is not None else None


def _is_contiguous_raw_span(
    raw_indices: np.ndarray,
    *,
    start: int,
    stop: int,
) -> bool:
    """判断准备后位置范围是否映射到严格连续的原始行。

    参数：
        raw_indices: 与准备后数组逐行对齐的严格递增原始行号。
        start: 准备后半开范围起点。
        stop: 准备后半开范围终点。
    返回：
        映射长度完整且相邻原始行号差均为 1 时为真。
    异常：
        无；空范围按连续处理。
    副作用：
        无。
    """

    values = raw_indices[start:stop]
    return values.shape[0] == stop - start and bool(
        values.shape[0] <= 1 or np.all(np.diff(values) == 1)
    )


def _allocate_stage_counts(
    available_rows: int,
    ratios: tuple[float, float, float, float, float],
) -> tuple[int, int, int, int, int]:
    """用最大余数法确定性分配扣除隔离带后的行数。

    先取每个比例乘积的下整数，再按小数余数从大到小补齐剩余行；余数相同时保留阶段顺序。
    任一阶段最终为空时立即失败，不能偷偷缩短隔离带。
    参数：
        available_rows: 扣除四个隔离带后的可分配行数。
        ratios: 按 ``StageName`` 顺序排列、和为 1 的五个比例。
    返回：
        总和严格等于 ``available_rows`` 的五个正整数行数。
    异常：
        任一阶段分配结果为空时抛出 ``ValueError``。
    副作用：
        无。
    """

    exact = [available_rows * ratio for ratio in ratios]
    counts = [math.floor(value) for value in exact]
    remaining = available_rows - sum(counts)
    order = sorted(range(len(counts)), key=lambda index: (-(exact[index] - counts[index]), index))
    for index in order[:remaining]:
        counts[index] += 1
    if any(count <= 0 for count in counts):
        raise ValueError(
            "Normal sequence is too short to form five non-empty stages after isolation gaps. "
            f"available_rows={available_rows}, allocated_counts={counts}."
        )
    return tuple(counts)  # type: ignore[return-value]


def _stage_slice(
    *,
    stage: StageName,
    start: int,
    stop: int,
    data: np.ndarray,
    raw_indices: np.ndarray,
    config: FiveStageSplitConfig,
) -> StageSlice:
    """构造单段索引、合法窗口、完整 episode 和三个相互独立的 hash。

    参数：
        stage: 当前阶段的受控名称。
        start: 阶段在准备后数组中的半开起点。
        stop: 阶段在准备后数组中的半开终点。
        data: 完整准备后正常数组。
        raw_indices: 与 ``data`` 逐行对齐的原始行号。
        config: 冻结的切分、窗口与 episode 配置。
    返回：
        只含本阶段合法位置、窗口和完整 episode 的 ``StageSlice``。
    异常：
        阶段没有合法窗口，或校准阶段完整 episode 数不足目标风险分辨率时抛出
        ``ValueError``。
    副作用：
        无；不修改输入数组。
    """

    prepared_row_indices = tuple(range(start, stop))
    stage_raw_indices = tuple(int(value) for value in raw_indices[start:stop])
    candidate_episode_count = (stop - start) // config.episode_length
    candidate_episode_ranges = tuple(
        (
            start + episode * config.episode_length,
            start + (episode + 1) * config.episode_length,
        )
        for episode in range(candidate_episode_count)
    )
    episode_ranges = tuple(
        item
        for item in candidate_episode_ranges
        if _is_contiguous_raw_span(raw_indices, start=item[0], stop=item[1])
    )
    discarded_episode_ranges = tuple(
        item for item in candidate_episode_ranges if item not in episode_ranges
    )
    complete_episode_count = len(episode_ranges)
    tail_start = start + candidate_episode_count * config.episode_length
    unused_tail = None if tail_start == stop else (tail_start, stop)
    if stage in {
        StageName.DETECTION_CALIBRATION,
        StageName.ATTRIBUTION_CALIBRATION,
    } and complete_episode_count < config.minimum_calibration_episodes:
        raise ValueError(
            f"Stage {stage.value!r} has only {complete_episode_count} complete calibration "
            f"episodes, but target_risk_level={config.target_risk_level} requires at least "
            f"{config.minimum_calibration_episodes}. Increase normal data, reduce episode_length, "
            "or revisit the target risk level; the isolation gap will not be shortened."
        )

    final_start = stop - config.dependency_span
    candidate_starts = (
        range(start, final_start + 1, config.window_stride)
        if final_start >= start
        else ()
    )
    prepared_window_starts = tuple(
        prepared_window_start
        for prepared_window_start in candidate_starts
        if _is_contiguous_raw_span(
            raw_indices,
            start=prepared_window_start,
            stop=prepared_window_start + config.dependency_span,
        )
        and (
            stage
            not in {
                StageName.DETECTION_CALIBRATION,
                StageName.ATTRIBUTION_CALIBRATION,
            }
            or any(
                prepared_window_start >= episode_start
                and prepared_window_start + config.dependency_span <= episode_stop
                for episode_start, episode_stop in episode_ranges
            )
        )
    )
    if not prepared_window_starts:
        raise ValueError(
            f"Stage {stage.value!r} has no legal windows after applying dependency_span="
            f"{config.dependency_span}. row_range=({start}, {stop})."
        )
    return StageSlice(
        stage=stage,
        prepared_row_indices=prepared_row_indices,
        raw_indices=stage_raw_indices,
        prepared_window_starts=prepared_window_starts,
        raw_window_starts=tuple(
            int(raw_indices[start]) for start in prepared_window_starts
        ),
        dependency_span=config.dependency_span,
        prepared_episode_ranges=episode_ranges,
        discarded_prepared_episode_ranges=discarded_episode_ranges,
        unused_prepared_episode_tail=unused_tail,
        data_hash=_hash_array(data[start:stop]),
        index_hash=_hash_json(
            {
                "prepared_row_indices": prepared_row_indices,
                "raw_indices": stage_raw_indices,
            }
        ),
        window_hash=_hash_json(
            {
                "prepared_window_starts": prepared_window_starts,
                "dependency_span": config.dependency_span,
            }
        ),
    )


def _validate_disjoint_slices(slices: Mapping[StageName, StageSlice]) -> None:
    """验证任意两段准备后位置、原始行、窗口依赖和完整 episode 均不重叠。

    参数：
        slices: 含五个阶段的映射。
    返回：
        无。
    异常：
        任意两段共享准备后位置、原始行号、窗口依赖原始行或 episode 位置时抛出
        ``RuntimeError``。
    副作用：
        无。
    """

    stages = list(StageName)
    for index, left_name in enumerate(stages):
        left = slices[left_name]
        left_rows = set(left.prepared_row_indices)
        left_raw_rows = set(left.raw_indices)
        left_dependencies = {
            row
            for start in left.prepared_window_starts
            for row in left.dependency_raw_rows(start)
        }
        left_episode_rows = {
            row
            for episode_start, episode_stop in left.prepared_episode_ranges
            for row in range(episode_start, episode_stop)
        }
        for right_name in stages[index + 1 :]:
            right = slices[right_name]
            if left_rows.intersection(right.prepared_row_indices):
                raise RuntimeError(
                    f"Paper stages {left_name.value!r} and {right_name.value!r} "
                    "overlap prepared row positions."
                )
            if left_raw_rows.intersection(right.raw_indices):
                raise RuntimeError(
                    f"Paper stages {left_name.value!r} and {right_name.value!r} "
                    "overlap original raw_index values."
                )
            right_dependencies = {
                row
                for start in right.prepared_window_starts
                for row in right.dependency_raw_rows(start)
            }
            if left_dependencies.intersection(right_dependencies):
                raise RuntimeError(
                    f"Paper stages {left_name.value!r} and {right_name.value!r} "
                    "overlap window dependency rows."
                )
            right_episode_rows = {
                row
                for episode_start, episode_stop in right.prepared_episode_ranges
                for row in range(episode_start, episode_stop)
            }
            if left_episode_rows.intersection(right_episode_rows):
                raise RuntimeError(
                    f"Paper stages {left_name.value!r} and {right_name.value!r} "
                    "reuse calibration episode rows."
                )


def _hash_array(array: np.ndarray) -> str:
    """对 dtype、shape 和连续字节做 SHA-256，避免只比较行数。

    参数：
        array: 要追溯的 NumPy 数组。
    返回：
        64 位小写 SHA-256 十六进制字符串。
    异常：
        无；输入会转换为 C 连续视图。
    副作用：
        无。
    """

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _hash_json(value: Mapping[str, Any]) -> str:
    """对排序后的紧凑 JSON 做稳定 SHA-256。

    参数：
        value: 只含 JSON 可序列化值的映射。
    返回：
        64 位小写 SHA-256 十六进制字符串。
    异常：
        值不可 JSON 序列化时传播 ``TypeError``。
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


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    """以稳定格式覆盖 UTF-8 JSON；目录必须由调用方显式准备。

    参数：
        path: 已有父目录中的目标文件。
        value: JSON 可序列化映射。
    返回：
        无。
    异常：
        父目录不存在、文件不可写或值不可序列化时传播 ``OSError``/``TypeError``。
    副作用：
        以 UTF-8 和尾随换行覆盖目标文件。
    """

    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
