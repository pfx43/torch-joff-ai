"""P8 输入调度确定性包络、有限 episode 校准与动态阈值。

文件用途：
    把 P7 冻结支路上的可观测统计量拆成锚点半径、输入/路径相关确定性半径和独立正常
    detection-calibration episode 的随机校准余量，形成论文定义的动态阈值。
主要职责：
    在正常 estimate 段拟合输入描述符与 reference-age 包络；通过同一个 P7 ``L_b``
    传播 P6 ``G_nu`` block columns；冻结有限 episode maximum 的 conformal 分位；输出
    分项可审计阈值。本文件不训练模型、不改变 anchor/mode/window 状态，也不执行 P9
    归因或故障类别判决。
关键输入与输出：
    输入为受控协议阶段、``(u, delta_u, delta_xi, region, age)``、estimate mismatch/
    drift 幅值、P6/P7 冻结几何和 detection-calibration episode 分数；输出为不可变包络、
    校准证据及 ``ThresholdResult``。
依赖与副作用：
    依赖 NumPy、SciPy 线性规划和本地 evaluation 类型；所有计算仅在内存中完成，不读写
    文件、不访问网络、不修改随机数或全局数值状态。
重要约束：
    包络、尺度、正 floor、score map 和 monitor reset 身份必须在 estimate 段冻结；
    ``q_det`` 只能读取独立 detection calibration episode。归因校准和故障数据不得进入
    P8 拟合。描述符未覆盖、有限秩不足、数值失败或证据不完整时必须返回正无穷并禁用
    决定，不能静默外推或给出无限首报警保证。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence, cast

import numpy as np
# SciPy 当前未随本项目提供完整 py.typed 元数据；只抑制这一处第三方导入。
from scipy.optimize import linprog  # type: ignore[import-untyped]

from .postfilter import BranchBank, BranchOperator
from .protected_operators import OperatorBundle, OperatorStatus
from .protected_reference import MonitorStage


@dataclass(frozen=True)
class InputDescriptor:
    """输入调度包络使用的一个运行时描述符。

    参数：
        region: estimate 阶段冻结的 fuzzy/operating region 稳定名称。
        u/delta_u/delta_xi: 当前记录输入、相邻输入变化和外生条件变化向量。
    返回：
        不可变描述符；``features`` 返回论文式 ``(1, ||u||, ||delta_u||,
        ||delta_xi||)``。
    异常：
        region 为空、向量不是非空 tuple 或包含非有限值时抛出
        ``TypeError``/``ValueError``。
    副作用：
        无；不会保存原始数组引用。
    """

    region: str
    u: tuple[float, ...]
    delta_u: tuple[float, ...]
    delta_xi: tuple[float, ...]

    def __post_init__(self) -> None:
        """验证受控区域名称和三个有限向量。"""

        if not isinstance(self.region, str) or not self.region.strip():
            raise ValueError("Input descriptor region must be a non-empty string.")
        for name, values in (
            ("u", self.u),
            ("delta_u", self.delta_u),
            ("delta_xi", self.delta_xi),
        ):
            if not isinstance(values, tuple) or not values:
                raise TypeError(f"Input descriptor {name} must be a non-empty tuple.")
            if any(not _is_real_number(value) for value in values):
                raise TypeError(
                    f"Input descriptor {name} must contain only numeric values."
                )
            array = np.asarray(
                tuple(
                    _coerce_finite_float(
                        value,
                        name=f"Input descriptor {name} item",
                    )
                    for value in values
                ),
                dtype=np.float64,
            )
            if array.ndim != 1 or not np.isfinite(array).all():
                raise ValueError(
                    f"Input descriptor {name} must be a finite one-dimensional vector."
                )

    def features(self) -> tuple[float, float, float, float]:
        """返回带常数项的三个尺度安全 Euclidean 范数。

        参数：
            无。
        返回：
            ``(1, ||u||, ||delta_u||, ||delta_xi||)``。
        异常：
            向量已在构造时验证，不额外抛出。
        副作用：
            无。
        """

        return (
            1.0,
            _stable_vector_norm(self.u),
            _stable_vector_norm(self.delta_u),
            _stable_vector_norm(self.delta_xi),
        )

    def to_dict(self) -> dict[str, object]:
        """返回原始描述符向量的标准 JSON 字典。"""

        return {
            "region": self.region,
            "u": list(self.u),
            "delta_u": list(self.delta_u),
            "delta_xi": list(self.delta_xi),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "InputDescriptor":
        """从严格 JSON 映射恢复描述符并重新验证向量有限性。"""

        _require_exact_keys(
            value,
            {"region", "u", "delta_u", "delta_xi"},
            name="InputDescriptor",
        )
        return cls(
            region=_strict_string(value["region"], name="input descriptor region"),
            u=_strict_float_tuple(value["u"], name="input descriptor u"),
            delta_u=_strict_float_tuple(
                value["delta_u"],
                name="input descriptor delta_u",
            ),
            delta_xi=_strict_float_tuple(
                value["delta_xi"],
                name="input descriptor delta_xi",
            ),
        )


@dataclass(frozen=True)
class EnvelopeEvaluation:
    """一次包络查询的值与 descriptor coverage 状态。

    ``supported=False`` 时 ``value`` 必须为 ``+infinity``，使下游阈值自然禁用；调用方
    仍可从 ``reason`` 区分未知 region 与已知 region 的范围外描述符。
    """

    value: float
    supported: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        """阻止有限性、支持状态和原因互相矛盾。"""

        if type(self.supported) is not bool or np.isnan(self.value) or self.value < 0.0:
            raise ValueError("Envelope evaluation value/status is invalid.")
        if self.supported:
            if not np.isfinite(self.value) or self.reason is not None:
                raise ValueError("Supported envelope evaluation must be finite without a reason.")
        elif self.value != float("inf") or not self.reason:
            raise ValueError(
                "Unsupported envelope evaluation must be +infinity with a reason."
            )


@dataclass(frozen=True)
class _RegionEnvelope:
    """一个 estimate region 的非负系数和经验 descriptor 支持范围。"""

    region: str
    coefficients: tuple[float, float, float, float]
    feature_minima: tuple[float, float, float]
    feature_maxima: tuple[float, float, float]
    sample_count: int

    def __post_init__(self) -> None:
        """校验系数非负、范围有序且样本计数为正整数。"""

        if not self.region or type(self.sample_count) is not int or self.sample_count <= 0:
            raise ValueError("Region envelope identity/sample_count is invalid.")
        numeric = (*self.coefficients, *self.feature_minima, *self.feature_maxima)
        if any(value < 0.0 or not np.isfinite(value) for value in numeric):
            raise ValueError("Region envelope values must be finite and non-negative.")
        if any(
            lower > upper
            for lower, upper in zip(self.feature_minima, self.feature_maxima, strict=True)
        ):
            raise ValueError("Region envelope feature ranges must be ordered.")


@dataclass(frozen=True)
class InputDependentEnvelope:
    """仅正常 estimate 段拟合的 region-wise 非负分位包络。

    参数：
        stage: 固定为 ``MonitorStage.ESTIMATE``。
        quantile/minimum_region_samples: 经验 pinball-loss 分位与每个 region 样本门槛。
        source_hash: 输入 estimate 记录的 SHA-256，由上游数据协议提供。
        regions: 按名称稳定排序的非负系数及经验 descriptor 支持范围。
    返回：
        ``fit`` 产生冻结包络，``evaluate`` 返回有限支持值或失败关闭的 ``+infinity``。
    异常：
        阶段越权、样本不足、输入 shape/有限性非法或线性规划失败时抛出
        ``TypeError``/``ValueError``。
    副作用：
        无；SciPy HiGHS 求解器只消费传入数组。
    """

    stage: MonitorStage
    quantile: float
    minimum_region_samples: int
    source_hash: str
    regions: tuple[_RegionEnvelope, ...]

    def __post_init__(self) -> None:
        """验证 estimate 身份、哈希、分位配置和 region 唯一性。"""

        if self.stage is not MonitorStage.ESTIMATE:
            raise ValueError("Input envelope may only be fitted on the estimate stage.")
        if not _is_real_number(self.quantile):
            raise TypeError("Input envelope quantile must be numeric.")
        if not 0.0 < self.quantile < 1.0 or not np.isfinite(self.quantile):
            raise ValueError("Input envelope quantile must be finite and strictly between 0 and 1.")
        if (
            type(self.minimum_region_samples) is not int
            or self.minimum_region_samples < 4
        ):
            raise ValueError(
                "Input envelope minimum_region_samples must be an integer at least 4."
            )
        if not _is_sha256(self.source_hash):
            raise ValueError("Input envelope source_hash must be a 64-character SHA-256.")
        if not isinstance(self.regions, tuple) or not self.regions:
            raise ValueError("Input envelope requires at least one fitted region.")
        if not all(isinstance(region, _RegionEnvelope) for region in self.regions):
            raise TypeError("Input envelope regions must contain fitted region objects.")
        names = tuple(region.region for region in self.regions)
        if names != tuple(sorted(names)) or len(set(names)) != len(names):
            raise ValueError("Input envelope regions must be unique and stably sorted.")
        if any(
            region.sample_count < self.minimum_region_samples for region in self.regions
        ):
            raise ValueError("Every input envelope region must satisfy the sample gate.")

    @classmethod
    def fit(
        cls,
        descriptors: Sequence[InputDescriptor],
        mismatch_magnitudes: Sequence[float] | np.ndarray,
        *,
        stage: MonitorStage | str,
        quantile: float,
        minimum_region_samples: int,
        source_hash: str,
    ) -> "InputDependentEnvelope":
        """以非负线性规划拟合每个 region 的经验分位包络。

        参数：
            descriptors: estimate 记录的 ``InputDescriptor`` 序列。
            mismatch_magnitudes: 同顺序的非负 mismatch 范数。
            stage: 只允许 ``estimate``。
            quantile: pinball loss 的目标分位，严格位于 ``(0,1)``。
            minimum_region_samples: 每个出现 region 的最小样本数，至少为特征数 4。
            source_hash: 上游 estimate 输入 SHA-256。
        返回：
            冻结的 ``InputDependentEnvelope``。
        异常：
            越权阶段、空输入、长度/类型/有限性错误、region 样本不足或优化失败时抛出
            ``TypeError``/``ValueError``。
        副作用：
            无；不读取其他协议阶段。
        """

        normalized_stage = MonitorStage.parse(stage)
        if normalized_stage is not MonitorStage.ESTIMATE:
            raise ValueError("Input envelope fit may only use the estimate stage.")
        if not descriptors or not all(
            isinstance(item, InputDescriptor) for item in descriptors
        ):
            raise TypeError("Input envelope descriptors must be a non-empty descriptor sequence.")
        raw_targets = np.asarray(mismatch_magnitudes)
        if raw_targets.dtype.kind not in "iuf":
            raise TypeError(
                "Input envelope mismatch_magnitudes must contain only numeric values."
            )
        targets = np.asarray(raw_targets, dtype=np.float64)
        if (
            targets.ndim != 1
            or targets.shape[0] != len(descriptors)
            or not np.isfinite(targets).all()
            or np.any(targets < 0.0)
        ):
            raise ValueError(
                "Input envelope mismatch_magnitudes must be finite, non-negative, and aligned."
            )
        normalized_quantile = _coerce_finite_float(
            quantile,
            name="Input envelope quantile",
        )
        if not 0.0 < normalized_quantile < 1.0:
            raise ValueError("Input envelope quantile must be finite and strictly between 0 and 1.")
        if type(minimum_region_samples) is not int or minimum_region_samples < 4:
            raise ValueError(
                "Input envelope minimum_region_samples must be an integer at least 4."
            )

        fitted_regions: list[_RegionEnvelope] = []
        region_names = sorted({item.region for item in descriptors})
        for region_name in region_names:
            indices = [
                index
                for index, descriptor in enumerate(descriptors)
                if descriptor.region == region_name
            ]
            if len(indices) < minimum_region_samples:
                raise ValueError(
                    f"Input envelope region {region_name!r} has {len(indices)} samples; "
                    f"at least {minimum_region_samples} are required."
                )
            design = np.asarray(
                [descriptors[index].features() for index in indices],
                dtype=np.float64,
            )
            response = targets[indices]
            coefficients = _fit_nonnegative_quantile(
                design,
                response,
                quantile=normalized_quantile,
            )
            scheduled_features = design[:, 1:]
            feature_minima = np.min(scheduled_features, axis=0)
            feature_maxima = np.max(scheduled_features, axis=0)
            fitted_regions.append(
                _RegionEnvelope(
                    region=region_name,
                    coefficients=(
                        float(coefficients[0]),
                        float(coefficients[1]),
                        float(coefficients[2]),
                        float(coefficients[3]),
                    ),
                    feature_minima=(
                        float(feature_minima[0]),
                        float(feature_minima[1]),
                        float(feature_minima[2]),
                    ),
                    feature_maxima=(
                        float(feature_maxima[0]),
                        float(feature_maxima[1]),
                        float(feature_maxima[2]),
                    ),
                    sample_count=len(indices),
                )
            )
        return cls(
            stage=normalized_stage,
            quantile=normalized_quantile,
            minimum_region_samples=minimum_region_samples,
            source_hash=source_hash,
            regions=tuple(fitted_regions),
        )

    def evaluate(self, descriptor: InputDescriptor) -> EnvelopeEvaluation:
        """查询已冻结 region 包络，不对范围外 descriptor 做无证据外推。

        参数：
            descriptor: 当前在线输入/变化/region。
        返回：
            region 存在且三个范数都落在 estimate 经验范围内时返回有限非负包络；否则
            返回 ``supported=False, value=+infinity``。
        异常：
            descriptor 类型非法时抛出 ``TypeError``。
        副作用：
            无。
        """

        if not isinstance(descriptor, InputDescriptor):
            raise TypeError("descriptor must be an InputDescriptor.")
        region = next(
            (item for item in self.regions if item.region == descriptor.region),
            None,
        )
        if region is None:
            return EnvelopeEvaluation(
                value=float("inf"),
                supported=False,
                reason=f"Input region {descriptor.region!r} was not covered on estimate.",
            )
        features = descriptor.features()
        for feature_name, value, lower, upper in zip(
            ("input_norm", "input_delta_norm", "exogenous_delta_norm"),
            features[1:],
            region.feature_minima,
            region.feature_maxima,
            strict=True,
        ):
            tolerance = 1e-12 * max(1.0, abs(lower), abs(upper))
            if value < lower - tolerance or value > upper + tolerance:
                return EnvelopeEvaluation(
                    value=float("inf"),
                    supported=False,
                    reason=(
                        f"Descriptor {feature_name}={value:.17g} is outside estimate support "
                        f"[{lower:.17g}, {upper:.17g}] for region {region.region!r}."
                    ),
                )
        with np.errstate(over="ignore", invalid="ignore"):
            value = float(
                np.dot(
                    np.asarray(region.coefficients),
                    np.asarray(features),
                )
            )
        if not np.isfinite(value) or value < 0.0:
            return EnvelopeEvaluation(
                value=float("inf"),
                supported=False,
                reason="Numerical input-envelope evaluation is not finite.",
            )
        return EnvelopeEvaluation(value=value, supported=True)

    def to_dict(self) -> dict[str, object]:
        """返回可直接标准 JSON 编码的冻结 input-envelope 字典。"""

        return _input_envelope_payload(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "InputDependentEnvelope":
        """从严格 JSON 映射恢复包络并重验 region、范围和样本门禁。"""

        _require_exact_keys(
            value,
            {
                "stage",
                "quantile",
                "minimum_region_samples",
                "source_hash",
                "regions",
            },
            name="InputDependentEnvelope",
        )
        raw_regions = _strict_list(value["regions"], name="input envelope regions")
        regions: list[_RegionEnvelope] = []
        for raw_region in raw_regions:
            region = _strict_mapping(raw_region, name="input envelope region")
            _require_exact_keys(
                region,
                {
                    "region",
                    "coefficients",
                    "feature_minima",
                    "feature_maxima",
                    "sample_count",
                },
                name="input envelope region",
            )
            coefficients = _strict_float_tuple(
                region["coefficients"],
                name="input envelope coefficients",
                length=4,
            )
            minima = _strict_float_tuple(
                region["feature_minima"],
                name="input envelope feature_minima",
                length=3,
            )
            maxima = _strict_float_tuple(
                region["feature_maxima"],
                name="input envelope feature_maxima",
                length=3,
            )
            regions.append(
                _RegionEnvelope(
                    region=_strict_string(region["region"], name="input envelope region"),
                    coefficients=(
                        coefficients[0],
                        coefficients[1],
                        coefficients[2],
                        coefficients[3],
                    ),
                    feature_minima=(minima[0], minima[1], minima[2]),
                    feature_maxima=(maxima[0], maxima[1], maxima[2]),
                    sample_count=_strict_int(
                        region["sample_count"],
                        name="input envelope sample_count",
                    ),
                )
            )
        return cls(
            stage=MonitorStage.parse(
                _strict_string(value["stage"], name="input envelope stage")
            ),
            quantile=_strict_float(value["quantile"], name="input envelope quantile"),
            minimum_region_samples=_strict_int(
                value["minimum_region_samples"],
                name="input envelope minimum_region_samples",
            ),
            source_hash=_strict_string(
                value["source_hash"],
                name="input envelope source_hash",
            ),
            regions=tuple(regions),
        )


@dataclass(frozen=True)
class ContextAgeEnvelope:
    """仅正常 estimate 段拟合的单调 reference-age 漂移包络。

    参数：
        stage: 固定为 ``estimate``。
        quantile/minimum_samples_per_age: 每个已声明 age 的经验分位与样本门槛。
        source_hash: estimate drift 记录的上游 SHA-256。
        values: 从 age 0 开始连续保存的累计最大经验分位。
        sample_counts: 每个 age 的独立样本计数，与 ``values`` 等长。
    返回：
        ``fit`` 产生冻结年龄曲线；``evaluate`` 对 ``0..maximum_age`` 返回有限值，越界
        返回 ``+infinity``。
    异常：
        阶段越权、age 不连续、样本不足或数值非法时抛出 ``TypeError``/``ValueError``。
    副作用：
        无。
    """

    stage: MonitorStage
    quantile: float
    minimum_samples_per_age: int
    source_hash: str
    values: tuple[float, ...]
    sample_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        """验证 estimate 身份、曲线单调性和逐 age 样本证据。"""

        if self.stage is not MonitorStage.ESTIMATE:
            raise ValueError("Context-age envelope may only be fitted on the estimate stage.")
        if not _is_real_number(self.quantile):
            raise TypeError("Context-age envelope quantile must be numeric.")
        if not 0.0 < self.quantile < 1.0 or not np.isfinite(self.quantile):
            raise ValueError(
                "Context-age envelope quantile must be finite and strictly between 0 and 1."
            )
        if (
            type(self.minimum_samples_per_age) is not int
            or self.minimum_samples_per_age < 1
        ):
            raise ValueError(
                "Context-age minimum_samples_per_age must be a positive integer."
            )
        if not _is_sha256(self.source_hash):
            raise ValueError("Context-age source_hash must be a 64-character SHA-256.")
        if (
            not isinstance(self.values, tuple)
            or not self.values
            or not isinstance(self.sample_counts, tuple)
            or len(self.values) != len(self.sample_counts)
        ):
            raise ValueError("Context-age curve and sample counts must be non-empty and aligned.")
        if any(value < 0.0 or not np.isfinite(value) for value in self.values):
            raise ValueError("Context-age values must be finite and non-negative.")
        if any(
            later < earlier
            for earlier, later in zip(self.values, self.values[1:], strict=False)
        ):
            raise ValueError("Context-age envelope values must be monotone non-decreasing.")
        if any(
            type(count) is not int or count < self.minimum_samples_per_age
            for count in self.sample_counts
        ):
            raise ValueError("Every context age must satisfy the frozen sample gate.")

    @property
    def maximum_age(self) -> int:
        """返回 estimate 证据覆盖的最大整数 reference age。"""

        return len(self.values) - 1

    @classmethod
    def fit(
        cls,
        *,
        reference_ages: Sequence[int],
        drift_magnitudes: Sequence[float] | np.ndarray,
        stage: MonitorStage | str,
        quantile: float,
        minimum_samples_per_age: int,
        source_hash: str,
    ) -> "ContextAgeEnvelope":
        """逐 age 拟合经验分位并取累计最大，冻结单调漂移曲线。

        参数：
            reference_ages: estimate 记录的非负整数 age，必须从 0 连续覆盖到最大值。
            drift_magnitudes: 同顺序的非负 context-drift 范数。
            stage: 只允许 ``estimate``。
            quantile/minimum_samples_per_age/source_hash: 分位、逐 age 门槛和来源身份。
        返回：
            冻结 ``ContextAgeEnvelope``。
        异常：
            阶段、类型、连续覆盖、样本数或有限性不满足时抛出
            ``TypeError``/``ValueError``。
        副作用：
            无；不访问 detection/attribution/fault 数据。
        """

        normalized_stage = MonitorStage.parse(stage)
        if normalized_stage is not MonitorStage.ESTIMATE:
            raise ValueError("Context-age fit may only use the estimate stage.")
        if not isinstance(reference_ages, Sequence) or not reference_ages:
            raise TypeError("reference_ages must be a non-empty sequence.")
        normalized_ages: list[int] = []
        for age in reference_ages:
            if type(age) is not int or age < 0:
                raise ValueError("reference_ages must contain non-negative integers.")
            normalized_ages.append(age)
        raw_magnitudes = np.asarray(drift_magnitudes)
        if raw_magnitudes.dtype.kind not in "iuf":
            raise TypeError("drift_magnitudes must contain only numeric values.")
        magnitudes = np.asarray(raw_magnitudes, dtype=np.float64)
        if (
            magnitudes.ndim != 1
            or magnitudes.shape[0] != len(normalized_ages)
            or not np.isfinite(magnitudes).all()
            or np.any(magnitudes < 0.0)
        ):
            raise ValueError(
                "drift_magnitudes must be finite, non-negative, and aligned with ages."
            )
        normalized_quantile = _coerce_finite_float(
            quantile,
            name="Context-age quantile",
        )
        if not 0.0 < normalized_quantile < 1.0:
            raise ValueError(
                "Context-age quantile must be finite and strictly between 0 and 1."
            )
        if type(minimum_samples_per_age) is not int or minimum_samples_per_age < 1:
            raise ValueError(
                "minimum_samples_per_age must be a positive integer."
            )
        maximum_age = max(normalized_ages)
        expected_ages = set(range(maximum_age + 1))
        if set(normalized_ages) != expected_ages:
            raise ValueError(
                "Context-age estimate evidence must continuously cover age 0 through maximum."
            )

        raw_quantiles: list[float] = []
        sample_counts: list[int] = []
        for age in range(maximum_age + 1):
            samples = np.sort(
                magnitudes[
                    np.asarray([item == age for item in normalized_ages], dtype=bool)
                ]
            )
            if samples.size < minimum_samples_per_age:
                raise ValueError(
                    f"Context age {age} has {samples.size} samples; "
                    f"at least {minimum_samples_per_age} are required."
                )
            order_index = int(np.ceil(normalized_quantile * (samples.size - 1)))
            raw_quantiles.append(float(samples[order_index]))
            sample_counts.append(int(samples.size))
        monotone_values = tuple(float(value) for value in np.maximum.accumulate(raw_quantiles))
        return cls(
            stage=normalized_stage,
            quantile=normalized_quantile,
            minimum_samples_per_age=minimum_samples_per_age,
            source_hash=source_hash,
            values=monotone_values,
            sample_counts=tuple(sample_counts),
        )

    def evaluate(self, reference_age: int) -> EnvelopeEvaluation:
        """查询冻结 age 曲线；负数、非整数或超过最大 age 时失败关闭。

        参数：
            reference_age: 当前 anchor/reference 的非负整数年龄。
        返回：
            支持范围内的有限值，或带原因的 ``+infinity``。
        异常：
            无；非法 age 作为在线 coverage 失败返回，不使监视循环崩溃。
        副作用：
            无。
        """

        if (
            type(reference_age) is not int
            or reference_age < 0
            or reference_age > self.maximum_age
        ):
            return EnvelopeEvaluation(
                value=float("inf"),
                supported=False,
                reason=(
                    f"Reference age {reference_age!r} is outside frozen maximum "
                    f"{self.maximum_age}."
                ),
            )
        return EnvelopeEvaluation(value=self.values[reference_age], supported=True)

    def to_dict(self) -> dict[str, object]:
        """返回可直接标准 JSON 编码的冻结 reference-age 曲线。"""

        return _age_envelope_payload(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ContextAgeEnvelope":
        """从严格 JSON 映射恢复年龄曲线并重验单调性与样本门禁。"""

        _require_exact_keys(
            value,
            {
                "stage",
                "quantile",
                "minimum_samples_per_age",
                "source_hash",
                "values",
                "sample_counts",
            },
            name="ContextAgeEnvelope",
        )
        raw_counts = _strict_list(value["sample_counts"], name="context-age sample_counts")
        return cls(
            stage=MonitorStage.parse(
                _strict_string(value["stage"], name="context-age stage")
            ),
            quantile=_strict_float(value["quantile"], name="context-age quantile"),
            minimum_samples_per_age=_strict_int(
                value["minimum_samples_per_age"],
                name="context-age minimum_samples_per_age",
            ),
            source_hash=_strict_string(
                value["source_hash"],
                name="context-age source_hash",
            ),
            values=_strict_float_tuple(
                value["values"],
                name="context-age values",
            ),
            sample_counts=tuple(
                _strict_int(item, name="context-age sample_count") for item in raw_counts
            ),
        )


@dataclass(frozen=True)
class DeterministicRadius:
    """一个窗口、一条 P7 支路的确定性半径分账。

    参数：
        branch_name/episode_id/start_raw_index/raw_indices: 支路与 P6 路径身份。
        operator_status: P6 算子包的证据级别；不会因本地 SVD 自动升级。
        gamma_anchor: P7 用同一 ``L_b`` 计算的锚点半径。
        gamma_deterministic: 输入 mismatch 与 context age 的 block-column 上界和。
        source_envelopes/block_norms: 每一步 ``mu_bar+delta_bar`` 与
            ``||L_b G_nu E_j||_2``，供逐项审计。
        supported/reason: descriptor、shape 和数值证据是否支持有限结果。
    返回：
        不可变半径记录；失败关闭时 ``gamma_deterministic=+infinity``。
    异常：
        直接构造字段互相矛盾时抛出 ``TypeError``/``ValueError``。
    副作用：
        无。
    """

    branch_name: str
    episode_id: str
    start_raw_index: int
    raw_indices: tuple[int, ...]
    operator_status: OperatorStatus
    gamma_anchor: float
    gamma_deterministic: float
    source_envelopes: tuple[float, ...]
    block_norms: tuple[float, ...]
    supported: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        """验证路径、分账数组和失败关闭状态。"""

        if not self.branch_name or not self.episode_id:
            raise ValueError("Deterministic radius branch/path identity cannot be empty.")
        if (
            type(self.start_raw_index) is not int
            or self.start_raw_index < 0
            or not isinstance(self.raw_indices, tuple)
            or not self.raw_indices
        ):
            raise ValueError("Deterministic radius path indices are invalid.")
        if not isinstance(self.operator_status, OperatorStatus):
            raise TypeError("Deterministic radius requires a controlled OperatorStatus.")
        if (
            self.gamma_anchor < 0.0
            or not np.isfinite(self.gamma_anchor)
            or type(self.supported) is not bool
        ):
            raise ValueError("Deterministic radius anchor/status is invalid.")
        if len(self.source_envelopes) != len(self.raw_indices) or len(
            self.block_norms
        ) != len(self.raw_indices):
            raise ValueError("Deterministic radius per-step arrays must match the path length.")
        if any(value < 0.0 or np.isnan(value) for value in self.source_envelopes):
            raise ValueError("Deterministic radius source envelopes must be non-negative.")
        if any(value < 0.0 or not np.isfinite(value) for value in self.block_norms):
            raise ValueError("Deterministic radius block norms must be finite and non-negative.")
        if self.supported:
            if not np.isfinite(self.gamma_deterministic) or self.reason is not None:
                raise ValueError("Supported deterministic radius must be finite without a reason.")
        elif self.gamma_deterministic != float("inf") or not self.reason:
            raise ValueError(
                "Unsupported deterministic radius must be +infinity with a reason."
            )

    def to_dict(self) -> dict[str, object]:
        """返回路径身份、逐步分账和 coverage 状态的标准 JSON 字典。"""

        return {
            "branch_name": self.branch_name,
            "episode_id": self.episode_id,
            "start_raw_index": self.start_raw_index,
            "raw_indices": list(self.raw_indices),
            "operator_status": self.operator_status.value,
            "gamma_anchor": self.gamma_anchor,
            "gamma_deterministic": _encode_finite_or_infinity(
                self.gamma_deterministic
            ),
            "source_envelopes": [
                _encode_finite_or_infinity(value)
                for value in self.source_envelopes
            ],
            "block_norms": list(self.block_norms),
            "supported": self.supported,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "DeterministicRadius":
        """从严格 JSON 映射恢复半径，并重验路径长度和失败关闭状态。"""

        _require_exact_keys(
            value,
            {
                "branch_name",
                "episode_id",
                "start_raw_index",
                "raw_indices",
                "operator_status",
                "gamma_anchor",
                "gamma_deterministic",
                "source_envelopes",
                "block_norms",
                "supported",
                "reason",
            },
            name="DeterministicRadius",
        )
        raw_indices = _strict_list(
            value["raw_indices"],
            name="deterministic radius raw_indices",
        )
        raw_source_envelopes = _strict_list(
            value["source_envelopes"],
            name="deterministic radius source_envelopes",
        )
        reason = value["reason"]
        return cls(
            branch_name=_strict_string(
                value["branch_name"],
                name="deterministic radius branch_name",
            ),
            episode_id=_strict_string(
                value["episode_id"],
                name="deterministic radius episode_id",
            ),
            start_raw_index=_strict_int(
                value["start_raw_index"],
                name="deterministic radius start_raw_index",
            ),
            raw_indices=tuple(
                _strict_int(item, name="deterministic radius raw index")
                for item in raw_indices
            ),
            operator_status=OperatorStatus(
                _strict_string(
                    value["operator_status"],
                    name="deterministic radius operator_status",
                )
            ),
            gamma_anchor=_strict_float(
                value["gamma_anchor"],
                name="deterministic radius gamma_anchor",
            ),
            gamma_deterministic=_strict_finite_or_infinity(
                value["gamma_deterministic"],
                name="deterministic radius gamma_deterministic",
            ),
            source_envelopes=tuple(
                _strict_finite_or_infinity(
                    item,
                    name="deterministic radius source envelope",
                )
                for item in raw_source_envelopes
            ),
            block_norms=_strict_float_tuple(
                value["block_norms"],
                name="deterministic radius block_norms",
            ),
            supported=_strict_bool(
                value["supported"],
                name="deterministic radius supported",
            ),
            reason=None
            if reason is None
            else _strict_string(reason, name="deterministic radius reason"),
        )


@dataclass(frozen=True)
class DeterministicRadiusGenerator:
    """用冻结 estimate 包络和 P7/P6 公共几何生成 branch-wise ``gamma_det``。

    参数：
        input_envelope/context_age_envelope: 两个只读 estimate 拟合对象。
    返回：
        ``compute`` 对一个 P6 窗口和一个启用 P7 branch 返回 ``DeterministicRadius``。
    异常：
        类型、路径长度、block shape 或分支输入宽度矛盾时抛出
        ``TypeError``/``ValueError``；在线 descriptor coverage 失败则不抛异常，而返回
        ``+infinity``。
    副作用：
        无；不会改写 operator bundle、branch 或 monitor state。
    """

    input_envelope: InputDependentEnvelope
    context_age_envelope: ContextAgeEnvelope

    def __post_init__(self) -> None:
        """要求两个包络均为已冻结 estimate 对象。"""

        if not isinstance(self.input_envelope, InputDependentEnvelope) or not isinstance(
            self.context_age_envelope,
            ContextAgeEnvelope,
        ):
            raise TypeError("Deterministic radius generator requires both envelope objects.")

    def compute(
        self,
        *,
        branch: BranchOperator,
        operator_bundle: OperatorBundle,
        descriptors: Sequence[InputDescriptor],
        reference_ages: Sequence[int],
    ) -> DeterministicRadius:
        """按 ``sum_j (mu_bar_j+delta_bar_j)||L_b G_nu E_j||`` 计算半径。

        参数：
            branch: P7 已冻结且启用的 ``BranchOperator``。
            operator_bundle: 同一窗口的 P6 ``OperatorBundle``；使用其完整 ``g_nu``。
            descriptors/reference_ages: 与 P6 ``raw_indices`` 一一对应的在线描述符和年龄。
        返回：
            有限分账，或 descriptor/数值不支持时的 ``+infinity`` 记录。
        异常：
            对象类型、禁用支路、窗口长度、``g_nu`` block shape 或输入宽度矛盾时抛出
            ``TypeError``/``ValueError``。
        副作用：
            无；每个 block 必须调用 ``branch.transform_operator``，保证与统计量共用
            同一个 ``L_b``。
        """

        if not isinstance(branch, BranchOperator):
            raise TypeError("branch must be a BranchOperator.")
        if not branch.enabled:
            raise ValueError("Disabled post-filter branches cannot receive a dynamic radius.")
        if not isinstance(operator_bundle, OperatorBundle):
            raise TypeError("operator_bundle must be an OperatorBundle.")
        horizon = len(operator_bundle.path.raw_indices)
        if len(descriptors) != horizon or len(reference_ages) != horizon:
            raise ValueError(
                "Descriptors and reference ages must match the operator path horizon."
            )
        if not all(isinstance(item, InputDescriptor) for item in descriptors):
            raise TypeError("All deterministic-radius descriptors must be InputDescriptor values.")
        g_nu = np.asarray(operator_bundle.g_nu, dtype=np.float64)
        if g_nu.ndim != 2 or g_nu.shape[0] != g_nu.shape[1]:
            raise ValueError("OperatorBundle g_nu must be square.")
        if g_nu.shape[0] % horizon != 0:
            raise ValueError("OperatorBundle g_nu rows must split evenly across the horizon.")
        if branch.input_dim != g_nu.shape[0]:
            raise ValueError("Branch input width must match OperatorBundle g_nu rows.")
        block_width = g_nu.shape[1] // horizon

        source_envelopes: list[float] = []
        block_norms: list[float] = []
        unsupported_reasons: list[str] = []
        for index, (descriptor, reference_age) in enumerate(
            zip(descriptors, reference_ages, strict=True)
        ):
            input_value = self.input_envelope.evaluate(descriptor)
            age_value = self.context_age_envelope.evaluate(reference_age)
            if not input_value.supported:
                unsupported_reasons.append(
                    f"step {index} input descriptor: {input_value.reason}"
                )
            if not age_value.supported:
                unsupported_reasons.append(
                    f"step {index} reference age: {age_value.reason}"
                )
            if input_value.supported and age_value.supported:
                source_envelopes.append(input_value.value + age_value.value)
            else:
                source_envelopes.append(float("inf"))
            raw_block = g_nu[:, index * block_width : (index + 1) * block_width]
            # P7 已验证支路与本地 block 的 shape；这里剩余的可预期失败来自极端有限
            # 数值在矩阵乘法或 SVD 中溢出。此类失败不能终止在线监视循环，也不能被
            # 误当作零半径，因此保留一个最大有限审计占位并让整条决定失败关闭。
            with np.errstate(over="ignore", invalid="ignore"):
                transformed = np.asarray(
                    branch.transform_operator(raw_block),
                    dtype=np.float64,
                )
                try:
                    block_norm = _scaled_matrix_spectral_norm(transformed)
                except ValueError as exc:
                    unsupported_reasons.append(
                        f"step {index} numerical branch propagation: {exc}"
                    )
                    block_norm = float(np.finfo(np.float64).max)
            block_norms.append(block_norm)

        if unsupported_reasons:
            return DeterministicRadius(
                branch_name=branch.name,
                episode_id=operator_bundle.path.episode_id,
                start_raw_index=operator_bundle.path.start_raw_index,
                raw_indices=operator_bundle.path.raw_indices,
                operator_status=operator_bundle.status,
                gamma_anchor=branch.anchor_radius,
                gamma_deterministic=float("inf"),
                source_envelopes=tuple(source_envelopes),
                block_norms=tuple(block_norms),
                supported=False,
                reason="; ".join(unsupported_reasons),
            )
        with np.errstate(over="ignore", invalid="ignore"):
            gamma_deterministic = float(
                np.dot(
                    np.asarray(source_envelopes, dtype=np.float64),
                    np.asarray(block_norms, dtype=np.float64),
                )
            )
        if not np.isfinite(gamma_deterministic):
            return DeterministicRadius(
                branch_name=branch.name,
                episode_id=operator_bundle.path.episode_id,
                start_raw_index=operator_bundle.path.start_raw_index,
                raw_indices=operator_bundle.path.raw_indices,
                operator_status=operator_bundle.status,
                gamma_anchor=branch.anchor_radius,
                gamma_deterministic=float("inf"),
                source_envelopes=tuple(source_envelopes),
                block_norms=tuple(block_norms),
                supported=False,
                reason=(
                    "Numerical failure: deterministic radius sum is not representable "
                    "in float64."
                ),
            )
        return DeterministicRadius(
            branch_name=branch.name,
            episode_id=operator_bundle.path.episode_id,
            start_raw_index=operator_bundle.path.start_raw_index,
            raw_indices=operator_bundle.path.raw_indices,
            operator_status=operator_bundle.status,
            gamma_anchor=branch.anchor_radius,
            gamma_deterministic=gamma_deterministic,
            source_envelopes=tuple(source_envelopes),
            block_norms=tuple(block_norms),
            supported=True,
        )


@dataclass(frozen=True, order=True)
class ScoreCoordinate:
    """有限 detection family 中一个可选择的时间、mode 和 branch 坐标。

    三元组必须在 calibration 前完整冻结；``EpisodeMaxCalibrator`` 要求每个 episode
    恰好出现一次全部坐标，避免遗漏高分时间或动态选择后再做边际校准。
    """

    time_index: int
    mode: str
    branch_name: str

    def __post_init__(self) -> None:
        """验证非负时间索引与稳定非空名称。"""

        if type(self.time_index) is not int or self.time_index < 0:
            raise ValueError("Score coordinate time_index must be a non-negative integer.")
        if (
            not isinstance(self.mode, str)
            or not self.mode.strip()
            or not isinstance(self.branch_name, str)
            or not self.branch_name.strip()
        ):
            raise ValueError("Score coordinate mode and branch_name cannot be empty.")

    def to_dict(self) -> dict[str, object]:
        """返回时间、mode 和 branch 的标准 JSON 字典。"""

        return {
            "time_index": self.time_index,
            "mode": self.mode,
            "branch_name": self.branch_name,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ScoreCoordinate":
        """从严格 JSON 映射恢复一个 score-family 坐标。"""

        _require_exact_keys(
            value,
            {"time_index", "mode", "branch_name"},
            name="ScoreCoordinate",
        )
        return cls(
            time_index=_strict_int(value["time_index"], name="score time_index"),
            mode=_strict_string(value["mode"], name="score mode"),
            branch_name=_strict_string(value["branch_name"], name="score branch_name"),
        )


@dataclass(frozen=True)
class DetectionScore:
    """冻结 score map 在一个 detection 坐标上的可观测超额分数。

    参数：
        score_map_hash: candidate、branch family、包络、尺度、floor 和 monitor path 的
            联合 SHA-256；最终分位不属于该 hash，避免反馈循环。
        episode_id/coordinate: 独立 calibration episode 与预声明 family 坐标。
        statistic/gamma_anchor/gamma_deterministic/scale: ``T_b``、两项确定性半径和
            estimate-only 正尺度。
        normalized_excess: 严格派生的
            ``max(0,T_b-gamma_anchor-gamma_deterministic)/scale``。
        supported/reason: descriptor/半径不支持时分数为 ``+infinity``，使校准失败关闭。
    返回：
        ``from_components`` 生成不可变分数。
    异常：
        身份、有限性、正尺度或派生分数矛盾时抛出 ``TypeError``/``ValueError``。
    副作用：
        无。
    """

    score_map_hash: str
    episode_id: str
    coordinate: ScoreCoordinate
    statistic: float
    gamma_anchor: float
    gamma_deterministic: float
    scale: float
    normalized_excess: float
    supported: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        """重算 normalized excess 并验证失败关闭状态。"""

        if not _is_sha256(self.score_map_hash) or not self.episode_id:
            raise ValueError("Detection score requires a score-map hash and episode identity.")
        if not isinstance(self.coordinate, ScoreCoordinate):
            raise TypeError("Detection score coordinate must be a ScoreCoordinate.")
        if (
            self.statistic < 0.0
            or not np.isfinite(self.statistic)
            or self.gamma_anchor < 0.0
            or not np.isfinite(self.gamma_anchor)
            or self.scale <= 0.0
            or not np.isfinite(self.scale)
            or type(self.supported) is not bool
        ):
            raise ValueError("Detection score components must be finite with a positive scale.")
        if self.supported:
            if not np.isfinite(self.gamma_deterministic) or self.gamma_deterministic < 0.0:
                raise ValueError("Supported detection score needs a finite deterministic radius.")
            expected = max(
                0.0,
                self.statistic - self.gamma_anchor - self.gamma_deterministic,
            ) / self.scale
            if self.reason is not None or not np.isclose(
                self.normalized_excess,
                expected,
                atol=1e-12,
                rtol=1e-10,
            ):
                raise ValueError("Detection normalized_excess must match its frozen components.")
        elif (
            self.gamma_deterministic != float("inf")
            or self.normalized_excess != float("inf")
            or not self.reason
        ):
            raise ValueError(
                "Unsupported detection score must carry infinite radius/score and a reason."
            )

    @classmethod
    def from_components(
        cls,
        *,
        score_map_hash: str,
        episode_id: str,
        coordinate: ScoreCoordinate,
        statistic: float,
        gamma_anchor: float,
        gamma_deterministic: float,
        scale: float,
        unsupported_reason: str | None = None,
    ) -> "DetectionScore":
        """从阈值分账计算可观测超额分数。

        参数：
            score_map_hash/episode_id/coordinate: 冻结 score map 和 family 身份。
            statistic/gamma_anchor/gamma_deterministic/scale: 分数公式四项。
            unsupported_reason: ``gamma_deterministic=+infinity`` 时必须提供的原因。
        返回：
            可供 episode maximum 校准的 ``DetectionScore``。
        异常：
            字段不满足构造不变量时抛出 ``TypeError``/``ValueError``。
        副作用：
            无。
        """

        supported = np.isfinite(gamma_deterministic)
        normalized_excess = (
            max(0.0, statistic - gamma_anchor - gamma_deterministic) / scale
            if supported
            else float("inf")
        )
        return cls(
            score_map_hash=score_map_hash,
            episode_id=episode_id,
            coordinate=coordinate,
            statistic=float(statistic),
            gamma_anchor=float(gamma_anchor),
            gamma_deterministic=float(gamma_deterministic),
            scale=float(scale),
            normalized_excess=float(normalized_excess),
            supported=bool(supported),
            reason=None if supported else unsupported_reason,
        )

    def to_dict(self) -> dict[str, object]:
        """返回 score map 身份、分账和派生超额分数的标准 JSON 字典。"""

        return {
            "score_map_hash": self.score_map_hash,
            "episode_id": self.episode_id,
            "coordinate": self.coordinate.to_dict(),
            "statistic": self.statistic,
            "gamma_anchor": self.gamma_anchor,
            "gamma_deterministic": _encode_finite_or_infinity(
                self.gamma_deterministic
            ),
            "scale": self.scale,
            "normalized_excess": _encode_finite_or_infinity(
                self.normalized_excess
            ),
            "supported": self.supported,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "DetectionScore":
        """从严格 JSON 恢复分数，并重算 normalized excess。"""

        _require_exact_keys(
            value,
            {
                "score_map_hash",
                "episode_id",
                "coordinate",
                "statistic",
                "gamma_anchor",
                "gamma_deterministic",
                "scale",
                "normalized_excess",
                "supported",
                "reason",
            },
            name="DetectionScore",
        )
        reason = value["reason"]
        return cls(
            score_map_hash=_strict_string(
                value["score_map_hash"],
                name="detection score_map_hash",
            ),
            episode_id=_strict_string(
                value["episode_id"],
                name="detection episode_id",
            ),
            coordinate=ScoreCoordinate.from_dict(
                _strict_mapping(value["coordinate"], name="detection coordinate")
            ),
            statistic=_strict_float(
                value["statistic"],
                name="detection statistic",
            ),
            gamma_anchor=_strict_float(
                value["gamma_anchor"],
                name="detection gamma_anchor",
            ),
            gamma_deterministic=_strict_finite_or_infinity(
                value["gamma_deterministic"],
                name="detection gamma_deterministic",
            ),
            scale=_strict_float(value["scale"], name="detection scale"),
            normalized_excess=_strict_finite_or_infinity(
                value["normalized_excess"],
                name="detection normalized_excess",
            ),
            supported=_strict_bool(
                value["supported"],
                name="detection supported",
            ),
            reason=None
            if reason is None
            else _strict_string(reason, name="detection reason"),
        )


class CalibrationStatus(str, Enum):
    """有限 episode detection calibration 的受控结果状态。"""

    READY = "ready"
    INSUFFICIENT_RESOLUTION = "insufficient_resolution"
    INCOMPLETE_EVIDENCE = "incomplete_evidence"


@dataclass(frozen=True)
class EpisodeMaxCalibrator:
    """冻结 detection-calibration episode maximum 的有限秩 conformal 分位。

    参数：
        stage: 固定为 ``DETECTION_CALIBRATION``。
        error_rate/rank/quantile: 请求 ``alpha``、有限秩
            ``ceil((n_cal+1)(1-alpha))`` 与对应顺序统计量。
        score_map_hash/reset_state_hash: 校准前冻结的 score/状态身份。
        episode_definition_hash/exchangeability_assumption_hash/source_hash: 有限 episode
            定义、可交换性条件声明和独立 detection-calibration 数据身份。
        expected_coordinates: 每个 episode 必须恰好覆盖一次的完整 family。
        episode_ids/episode_scores/episode_maxima: 稳定顺序的独立 episode、完整坐标分数
            与其全 family 最大分数；严格重放从 ``episode_scores`` 重算 maximum。
        status/reason/strict_exceedance: 有限、分辨率不足或证据不全，以及固定的严格超限
            规则。
    返回：
        ``fit`` 产生可审计校准；分辨率不足或证据不全时 ``quantile=+infinity``。
    异常：
        阶段越权、hash/episode 身份、风险水平或 family 类型非法时抛出
        ``TypeError``/``ValueError``。
    副作用：
        无；不更新 anchor、mode、window 或任何 monitor state。
    """

    stage: MonitorStage
    error_rate: float
    rank: int
    quantile: float
    score_map_hash: str
    reset_state_hash: str
    episode_definition_hash: str
    exchangeability_assumption_hash: str
    source_hash: str
    expected_coordinates: tuple[ScoreCoordinate, ...]
    episode_ids: tuple[str, ...]
    episode_scores: tuple[tuple[DetectionScore, ...], ...]
    episode_maxima: tuple[float, ...]
    status: CalibrationStatus
    reason: str | None
    strict_exceedance: bool = True

    def __post_init__(self) -> None:
        """验证有限秩、episode 最大值和失败关闭状态。"""

        if self.stage is not MonitorStage.DETECTION_CALIBRATION:
            raise ValueError("Episode maximum calibration requires detection calibration.")
        if not _is_real_number(self.error_rate):
            raise TypeError("Calibration error_rate must be numeric.")
        if not 0.0 < self.error_rate < 1.0 or not np.isfinite(self.error_rate):
            raise ValueError("Calibration error_rate must be finite and strictly between 0 and 1.")
        if not all(
            _is_sha256(value)
            for value in (
                self.score_map_hash,
                self.reset_state_hash,
                self.episode_definition_hash,
                self.exchangeability_assumption_hash,
                self.source_hash,
            )
        ):
            raise ValueError("Calibration identities must be 64-character SHA-256 values.")
        if (
            not isinstance(self.expected_coordinates, tuple)
            or not self.expected_coordinates
            or not all(
                isinstance(item, ScoreCoordinate) for item in self.expected_coordinates
            )
            or len(set(self.expected_coordinates)) != len(self.expected_coordinates)
        ):
            raise ValueError("Calibration expected_coordinates must be non-empty and unique.")
        if (
            not isinstance(self.episode_ids, tuple)
            or not self.episode_ids
            or len(set(self.episode_ids)) != len(self.episode_ids)
            or not isinstance(self.episode_scores, tuple)
            or len(self.episode_ids) != len(self.episode_scores)
            or len(self.episode_ids) != len(self.episode_maxima)
        ):
            raise ValueError(
                "Calibration episode identities/scores/maxima must be unique and aligned."
            )
        if self.episode_ids != tuple(sorted(self.episode_ids)):
            raise ValueError("Calibration episode identities must be stably sorted.")
        if any(value < 0.0 or np.isnan(value) for value in self.episode_maxima):
            raise ValueError("Calibration episode maxima must be non-negative.")
        expected_coordinate_set = set(self.expected_coordinates)
        incomplete_evidence = False
        for episode_id, scores, stored_maximum in zip(
            self.episode_ids,
            self.episode_scores,
            self.episode_maxima,
            strict=True,
        ):
            if (
                not isinstance(scores, tuple)
                or not scores
                or not all(isinstance(score, DetectionScore) for score in scores)
                or any(score.episode_id != episode_id for score in scores)
                or any(score.score_map_hash != self.score_map_hash for score in scores)
            ):
                raise ValueError(
                    "Calibration stored scores must match episode and score-map identities."
                )
            coordinates = tuple(score.coordinate for score in scores)
            family_complete = (
                len(set(coordinates)) == len(coordinates)
                and set(coordinates) == expected_coordinate_set
            )
            computed_maximum = max(score.normalized_excess for score in scores)
            if not (
                (computed_maximum == stored_maximum)
                or np.isclose(
                    computed_maximum,
                    stored_maximum,
                    atol=1e-12,
                    rtol=1e-10,
                )
            ):
                raise ValueError(
                    "Calibration episode maximum must be derived from its stored scores."
                )
            if not family_complete or not np.isfinite(computed_maximum):
                incomplete_evidence = True
        expected_rank = math.ceil(
            (len(self.episode_maxima) + 1) * (1.0 - self.error_rate)
        )
        if type(self.rank) is not int or self.rank != expected_rank:
            raise ValueError("Calibration rank must follow the finite-sample conformal rule.")
        if not isinstance(self.status, CalibrationStatus) or self.strict_exceedance is not True:
            raise ValueError("Calibration status/strict-exceedance contract is invalid.")
        if self.status is CalibrationStatus.READY:
            if (
                self.rank > len(self.episode_maxima)
                or incomplete_evidence
                or not np.isfinite(self.quantile)
                or self.reason is not None
            ):
                raise ValueError("READY calibration requires a finite in-sample rank quantile.")
            expected_quantile = sorted(self.episode_maxima)[self.rank - 1]
            if not np.isclose(self.quantile, expected_quantile, atol=1e-12, rtol=1e-10):
                raise ValueError("Calibration quantile must equal its episode order statistic.")
        elif self.status is CalibrationStatus.INSUFFICIENT_RESOLUTION:
            if (
                self.rank != len(self.episode_maxima) + 1
                or incomplete_evidence
                or self.quantile != float("inf")
                or not self.reason
            ):
                raise ValueError(
                    "Insufficient-resolution calibration requires complete evidence and rank n+1."
                )
        elif (
            not incomplete_evidence
            or self.quantile != float("inf")
            or not self.reason
        ):
            raise ValueError("Non-ready calibration must use +infinity with an audit reason.")

    @property
    def risk_resolution(self) -> float:
        """返回当前 ``n_cal`` 可实现的最小风险水平 ``1/(n_cal+1)``。"""

        return 1.0 / (len(self.episode_maxima) + 1)

    @classmethod
    def fit(
        cls,
        episodes: Mapping[str, Sequence[DetectionScore]],
        *,
        expected_coordinates: Sequence[ScoreCoordinate],
        stage: MonitorStage | str,
        error_rate: float,
        score_map_hash: str,
        reset_state_hash: str,
        episode_definition_hash: str,
        exchangeability_assumption_hash: str,
        source_hash: str,
    ) -> "EpisodeMaxCalibrator":
        """对每个完整 episode 取全时间/mode/branch maximum，再应用有限秩规则。

        参数：
            episodes: ``episode_id -> DetectionScore 序列``；每个序列必须恰好覆盖完整
                ``expected_coordinates``。
            expected_coordinates: 校准前冻结的可选择 family。
            stage: 只允许 ``detection_calibration``。
            error_rate: episode family-wise 风险预算 ``alpha``。
            score_map_hash/reset_state_hash: 校准前冻结的 score/state 身份。
            episode_definition_hash/exchangeability_assumption_hash/source_hash: 有限 episode
                定义、可交换性条件声明和 detection-calibration 数据来源。
        返回：
            READY、INSUFFICIENT_RESOLUTION 或 INCOMPLETE_EVIDENCE 校准对象。
        异常：
            阶段、容器、风险、hash 或基础身份非法时抛出 ``TypeError``/``ValueError``；
            family 遗漏和不支持分数返回 ``INCOMPLETE_EVIDENCE``，不伪造有限分位。
        副作用：
            无；不读取 attribution calibration 或 fault 范围。
        """

        normalized_stage = MonitorStage.parse(stage)
        if normalized_stage is not MonitorStage.DETECTION_CALIBRATION:
            raise ValueError(
                "Episode maximum fit may only use the detection calibration stage."
            )
        if not isinstance(episodes, Mapping) or not episodes:
            raise TypeError("Calibration episodes must be a non-empty mapping.")
        coordinates = tuple(expected_coordinates)
        if (
            not coordinates
            or not all(isinstance(item, ScoreCoordinate) for item in coordinates)
            or len(set(coordinates)) != len(coordinates)
        ):
            raise ValueError("Expected calibration coordinates must be non-empty and unique.")
        normalized_error_rate = _coerce_finite_float(
            error_rate,
            name="Calibration error_rate",
        )
        if not 0.0 < normalized_error_rate < 1.0:
            raise ValueError("Calibration error_rate must be finite and strictly between 0 and 1.")
        if not all(
            _is_sha256(value)
            for value in (
                score_map_hash,
                reset_state_hash,
                episode_definition_hash,
                exchangeability_assumption_hash,
                source_hash,
            )
        ):
            raise ValueError("Calibration hashes must be 64-character SHA-256 values.")

        expected_set = set(coordinates)
        episode_ids = tuple(sorted(episodes))
        maxima: list[float] = []
        stored_scores: list[tuple[DetectionScore, ...]] = []
        incomplete_reasons: list[str] = []
        for episode_id in episode_ids:
            if not isinstance(episode_id, str) or not episode_id:
                raise ValueError("Calibration episode ids must be non-empty strings.")
            scores = tuple(episodes[episode_id])
            if not scores or not all(isinstance(score, DetectionScore) for score in scores):
                raise TypeError("Every calibration episode must contain DetectionScore values.")
            if any(score.episode_id != episode_id for score in scores):
                raise ValueError("DetectionScore episode identity must match its mapping key.")
            if any(score.score_map_hash != score_map_hash for score in scores):
                raise ValueError("All calibration scores must use the frozen score_map_hash.")
            scores = tuple(sorted(scores, key=lambda score: score.coordinate))
            actual_coordinates = tuple(score.coordinate for score in scores)
            if len(set(actual_coordinates)) != len(actual_coordinates):
                incomplete_reasons.append(f"episode {episode_id!r} repeats score coordinates")
            if set(actual_coordinates) != expected_set:
                incomplete_reasons.append(
                    f"episode {episode_id!r} does not cover the complete score family"
                )
            maximum = max(score.normalized_excess for score in scores)
            if not np.isfinite(maximum):
                incomplete_reasons.append(
                    f"episode {episode_id!r} contains unsupported threshold evidence"
                )
            maxima.append(float(maximum))
            stored_scores.append(scores)

        rank = math.ceil((len(maxima) + 1) * (1.0 - normalized_error_rate))
        if incomplete_reasons:
            status = CalibrationStatus.INCOMPLETE_EVIDENCE
            quantile = float("inf")
            reason = "; ".join(dict.fromkeys(incomplete_reasons))
        elif rank == len(maxima) + 1:
            status = CalibrationStatus.INSUFFICIENT_RESOLUTION
            quantile = float("inf")
            reason = (
                f"Requested error_rate {normalized_error_rate:.17g} is below finite resolution "
                f"{1.0 / (len(maxima) + 1):.17g} for {len(maxima)} episodes."
            )
        else:
            status = CalibrationStatus.READY
            quantile = sorted(maxima)[rank - 1]
            reason = None
        return cls(
            stage=normalized_stage,
            error_rate=normalized_error_rate,
            rank=rank,
            quantile=float(quantile),
            score_map_hash=score_map_hash,
            reset_state_hash=reset_state_hash,
            episode_definition_hash=episode_definition_hash,
            exchangeability_assumption_hash=exchangeability_assumption_hash,
            source_hash=source_hash,
            expected_coordinates=coordinates,
            episode_ids=episode_ids,
            episode_scores=tuple(stored_scores),
            episode_maxima=tuple(maxima),
            status=status,
            reason=reason,
        )

    def to_dict(self) -> dict[str, object]:
        """返回有限秩、完整 family 和 episode maxima 的标准 JSON 证据。"""

        return {
            "stage": self.stage.value,
            "error_rate": self.error_rate,
            "rank": self.rank,
            "quantile": _encode_finite_or_infinity(self.quantile),
            "score_map_hash": self.score_map_hash,
            "reset_state_hash": self.reset_state_hash,
            "episode_definition_hash": self.episode_definition_hash,
            "exchangeability_assumption_hash": self.exchangeability_assumption_hash,
            "source_hash": self.source_hash,
            "expected_coordinates": [
                coordinate.to_dict() for coordinate in self.expected_coordinates
            ],
            "episode_ids": list(self.episode_ids),
            "episode_scores": [
                [score.to_dict() for score in scores]
                for scores in self.episode_scores
            ],
            "episode_maxima": [
                _encode_finite_or_infinity(value) for value in self.episode_maxima
            ],
            "status": self.status.value,
            "reason": self.reason,
            "strict_exceedance": self.strict_exceedance,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "EpisodeMaxCalibrator":
        """从严格 JSON 映射恢复校准，并重算有限秩和 READY 顺序统计量。"""

        _require_exact_keys(
            value,
            {
                "stage",
                "error_rate",
                "rank",
                "quantile",
                "score_map_hash",
                "reset_state_hash",
                "episode_definition_hash",
                "exchangeability_assumption_hash",
                "source_hash",
                "expected_coordinates",
                "episode_ids",
                "episode_scores",
                "episode_maxima",
                "status",
                "reason",
                "strict_exceedance",
            },
            name="EpisodeMaxCalibrator",
        )
        raw_coordinates = _strict_list(
            value["expected_coordinates"],
            name="calibration expected_coordinates",
        )
        raw_episode_ids = _strict_list(
            value["episode_ids"],
            name="calibration episode_ids",
        )
        raw_maxima = _strict_list(
            value["episode_maxima"],
            name="calibration episode_maxima",
        )
        raw_episode_scores = _strict_list(
            value["episode_scores"],
            name="calibration episode_scores",
        )
        reason = value["reason"]
        return cls(
            stage=MonitorStage.parse(
                _strict_string(value["stage"], name="calibration stage")
            ),
            error_rate=_strict_float(
                value["error_rate"],
                name="calibration error_rate",
            ),
            rank=_strict_int(value["rank"], name="calibration rank"),
            quantile=_strict_finite_or_infinity(
                value["quantile"],
                name="calibration quantile",
            ),
            score_map_hash=_strict_string(
                value["score_map_hash"],
                name="calibration score_map_hash",
            ),
            reset_state_hash=_strict_string(
                value["reset_state_hash"],
                name="calibration reset_state_hash",
            ),
            episode_definition_hash=_strict_string(
                value["episode_definition_hash"],
                name="calibration episode_definition_hash",
            ),
            exchangeability_assumption_hash=_strict_string(
                value["exchangeability_assumption_hash"],
                name="calibration exchangeability_assumption_hash",
            ),
            source_hash=_strict_string(
                value["source_hash"],
                name="calibration source_hash",
            ),
            expected_coordinates=tuple(
                ScoreCoordinate.from_dict(
                    _strict_mapping(item, name="calibration score coordinate")
                )
                for item in raw_coordinates
            ),
            episode_ids=tuple(
                _strict_string(item, name="calibration episode_id")
                for item in raw_episode_ids
            ),
            episode_scores=tuple(
                tuple(
                    DetectionScore.from_dict(
                        _strict_mapping(
                            score,
                            name="calibration stored detection score",
                        )
                    )
                    for score in _strict_list(
                        episode,
                        name="calibration episode score list",
                    )
                )
                for episode in raw_episode_scores
            ),
            episode_maxima=tuple(
                _strict_finite_or_infinity(
                    item,
                    name="calibration episode maximum",
                )
                for item in raw_maxima
            ),
            status=CalibrationStatus(
                _strict_string(value["status"], name="calibration status")
            ),
            reason=None
            if reason is None
            else _strict_string(reason, name="calibration reason"),
            strict_exceedance=_strict_bool(
                value["strict_exceedance"],
                name="calibration strict_exceedance",
            ),
        )


class ThresholdStatus(str, Enum):
    """在线动态阈值是否具有有限、身份一致的完整证据。"""

    READY = "ready"
    DISABLED = "disabled"


@dataclass(frozen=True)
class ThresholdResult:
    """一个在线 branch 坐标的动态阈值分账与严格超限决定。

    参数：
        score_map_hash/episode_definition_hash/episode_id/coordinate/operator_status: 冻结
            score map、目标 episode 定义、目标路径和 P6 证据身份。
        statistic/gamma_anchor/gamma_deterministic/scale: 统计量及 estimate-only 三项。
        calibration_quantile/calibration_component: detection-calibration ``q_det`` 与
            ``scale*q_det``。
        threshold_floor/threshold: 校准前冻结正 floor 与最终
            ``max(floor, gamma_anchor+gamma_det+scale*q_det)``。
        normalized_excess/alarm: 可观测超额分数与严格 ``statistic > threshold`` 判决。
        status/calibration_status/reason/strict_exceedance: 有限证据、校准状态和禁用原因。
    返回：
        不可变审计记录；任何证据缺失时阈值为 ``+infinity`` 且 ``alarm=False``。
    异常：
        字段分账或判决不一致时抛出 ``TypeError``/``ValueError``。
    副作用：
        无；不会反馈 monitor state。
    """

    score_map_hash: str
    episode_definition_hash: str
    episode_id: str
    coordinate: ScoreCoordinate
    operator_status: OperatorStatus
    statistic: float
    gamma_anchor: float
    gamma_deterministic: float
    scale: float
    calibration_quantile: float
    calibration_component: float
    threshold_floor: float
    threshold: float
    normalized_excess: float
    alarm: bool
    status: ThresholdStatus
    calibration_status: CalibrationStatus | None
    reason: str | None = None
    strict_exceedance: bool = True

    def __post_init__(self) -> None:
        """验证三项分账、正 floor、严格 tie 规则和禁用状态。"""

        if (
            not _is_sha256(self.score_map_hash)
            or not _is_sha256(self.episode_definition_hash)
            or not self.episode_id
            or not isinstance(self.coordinate, ScoreCoordinate)
            or not isinstance(self.operator_status, OperatorStatus)
        ):
            raise ValueError("Threshold result identities are invalid.")
        finite_positive_components = (
            self.statistic,
            self.gamma_anchor,
            self.scale,
            self.threshold_floor,
        )
        if (
            any(not np.isfinite(value) for value in finite_positive_components)
            or self.statistic < 0.0
            or self.gamma_anchor < 0.0
            or self.scale <= 0.0
            or self.threshold_floor <= 0.0
            or type(self.alarm) is not bool
            or self.strict_exceedance is not True
            or not isinstance(self.status, ThresholdStatus)
        ):
            raise ValueError("Threshold result numeric/status fields are invalid.")
        if self.status is ThresholdStatus.READY:
            if (
                self.calibration_status is not CalibrationStatus.READY
                or any(
                    not np.isfinite(value)
                    for value in (
                        self.gamma_deterministic,
                        self.calibration_quantile,
                        self.calibration_component,
                        self.threshold,
                        self.normalized_excess,
                    )
                )
                or self.reason is not None
            ):
                raise ValueError("READY threshold requires complete finite calibration evidence.")
            expected_component = self.scale * self.calibration_quantile
            expected_threshold = max(
                self.threshold_floor,
                self.gamma_anchor
                + self.gamma_deterministic
                + expected_component,
            )
            expected_excess = max(
                0.0,
                self.statistic - self.gamma_anchor - self.gamma_deterministic,
            ) / self.scale
            if (
                not np.isclose(
                    self.calibration_component,
                    expected_component,
                    atol=1e-12,
                    rtol=1e-10,
                )
                or not np.isclose(
                    self.threshold,
                    expected_threshold,
                    atol=1e-12,
                    rtol=1e-10,
                )
                or not np.isclose(
                    self.normalized_excess,
                    expected_excess,
                    atol=1e-12,
                    rtol=1e-10,
                )
                or self.alarm is not (self.statistic > self.threshold)
            ):
                raise ValueError("Threshold result breakdown or strict alarm decision is inconsistent.")
        elif (
            self.threshold != float("inf")
            or self.calibration_component != float("inf")
            or self.alarm
            or not self.reason
        ):
            raise ValueError("Disabled threshold must be infinite, silent, and auditable.")

    def to_dict(self) -> dict[str, object]:
        """返回各阈值分项、状态和严格报警位的标准 JSON 字典。"""

        return {
            "score_map_hash": self.score_map_hash,
            "episode_definition_hash": self.episode_definition_hash,
            "episode_id": self.episode_id,
            "coordinate": self.coordinate.to_dict(),
            "operator_status": self.operator_status.value,
            "statistic": self.statistic,
            "gamma_anchor": self.gamma_anchor,
            "gamma_deterministic": _encode_finite_or_infinity(
                self.gamma_deterministic
            ),
            "scale": self.scale,
            "calibration_quantile": _encode_finite_or_infinity(
                self.calibration_quantile
            ),
            "calibration_component": _encode_finite_or_infinity(
                self.calibration_component
            ),
            "threshold_floor": self.threshold_floor,
            "threshold": _encode_finite_or_infinity(self.threshold),
            "normalized_excess": _encode_finite_or_infinity(
                self.normalized_excess
            ),
            "alarm": self.alarm,
            "status": self.status.value,
            "calibration_status": None
            if self.calibration_status is None
            else self.calibration_status.value,
            "reason": self.reason,
            "strict_exceedance": self.strict_exceedance,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ThresholdResult":
        """从严格 JSON 映射恢复结果并重算分账和严格超限决定。"""

        _require_exact_keys(
            value,
            {
                "score_map_hash",
                "episode_definition_hash",
                "episode_id",
                "coordinate",
                "operator_status",
                "statistic",
                "gamma_anchor",
                "gamma_deterministic",
                "scale",
                "calibration_quantile",
                "calibration_component",
                "threshold_floor",
                "threshold",
                "normalized_excess",
                "alarm",
                "status",
                "calibration_status",
                "reason",
                "strict_exceedance",
            },
            name="ThresholdResult",
        )
        raw_calibration_status = value["calibration_status"]
        reason = value["reason"]
        return cls(
            score_map_hash=_strict_string(
                value["score_map_hash"],
                name="threshold score_map_hash",
            ),
            episode_definition_hash=_strict_string(
                value["episode_definition_hash"],
                name="threshold episode_definition_hash",
            ),
            episode_id=_strict_string(
                value["episode_id"],
                name="threshold episode_id",
            ),
            coordinate=ScoreCoordinate.from_dict(
                _strict_mapping(value["coordinate"], name="threshold coordinate")
            ),
            operator_status=OperatorStatus(
                _strict_string(value["operator_status"], name="threshold operator_status")
            ),
            statistic=_strict_float(value["statistic"], name="threshold statistic"),
            gamma_anchor=_strict_float(
                value["gamma_anchor"],
                name="threshold gamma_anchor",
            ),
            gamma_deterministic=_strict_finite_or_infinity(
                value["gamma_deterministic"],
                name="threshold gamma_deterministic",
            ),
            scale=_strict_float(value["scale"], name="threshold scale"),
            calibration_quantile=_strict_finite_or_infinity(
                value["calibration_quantile"],
                name="threshold calibration_quantile",
            ),
            calibration_component=_strict_finite_or_infinity(
                value["calibration_component"],
                name="threshold calibration_component",
            ),
            threshold_floor=_strict_float(
                value["threshold_floor"],
                name="threshold floor",
            ),
            threshold=_strict_finite_or_infinity(
                value["threshold"],
                name="threshold value",
            ),
            normalized_excess=_strict_finite_or_infinity(
                value["normalized_excess"],
                name="threshold normalized_excess",
            ),
            alarm=_strict_bool(value["alarm"], name="threshold alarm"),
            status=ThresholdStatus(
                _strict_string(value["status"], name="threshold status")
            ),
            calibration_status=None
            if raw_calibration_status is None
            else CalibrationStatus(
                _strict_string(
                    raw_calibration_status,
                    name="threshold calibration_status",
                )
            ),
            reason=None
            if reason is None
            else _strict_string(reason, name="threshold reason"),
            strict_exceedance=_strict_bool(
                value["strict_exceedance"],
                name="threshold strict_exceedance",
            ),
        )


@dataclass(frozen=True)
class DynamicThresholdGenerator:
    """校准前冻结的 P8 score map 与在线阈值生成器。

    参数：
        stage/candidate_hash/branch_bank: estimate 阶段、P7 candidate 内容身份和严格重放
            的完整 branch family。
        input_envelope/context_age_envelope: 两个 estimate-only 包络。
        branch_names/branch_scales: 完整启用 branch family 与同顺序正尺度。
        threshold_floor/normalization_source_hash: 正 floor，以及 branch scale/floor 读取
            的 estimate 记录 SHA-256。
        mode_names/reset_state_hash: 有限 mode family 和确定性 monitor reset 身份。
    返回：
        ``freeze`` 构造 score map；``score`` 生成校准分数；``evaluate`` 只读地组合独立
        ``EpisodeMaxCalibrator`` 和在线半径。
    异常：
        阶段、branch family、尺度、mode 或身份非法时抛出 ``TypeError``/``ValueError``。
    副作用：
        无；``content_hash`` 明确不含最终 calibration quantile，因此后者不能改变生成
        自身分数的状态路径。
    """

    stage: MonitorStage
    candidate_hash: str
    branch_bank: BranchBank
    input_envelope: InputDependentEnvelope
    context_age_envelope: ContextAgeEnvelope
    branch_names: tuple[str, ...]
    branch_scales: tuple[float, ...]
    threshold_floor: float
    normalization_source_hash: str
    mode_names: tuple[str, ...]
    reset_state_hash: str

    def __post_init__(self) -> None:
        """验证 estimate-only score map 的冻结字段。"""

        if self.stage is not MonitorStage.ESTIMATE:
            raise ValueError("Dynamic threshold score map must be frozen on estimate.")
        if not all(
            _is_sha256(value)
            for value in (
                self.candidate_hash,
                self.normalization_source_hash,
                self.reset_state_hash,
            )
        ):
            raise ValueError(
                "Dynamic threshold candidate/normalization/reset identities must be SHA-256."
            )
        if not isinstance(self.input_envelope, InputDependentEnvelope) or not isinstance(
            self.context_age_envelope,
            ContextAgeEnvelope,
        ):
            raise TypeError("Dynamic threshold generator requires both frozen envelopes.")
        if not isinstance(self.branch_bank, BranchBank):
            raise TypeError("Dynamic threshold generator requires a frozen BranchBank.")
        enabled_names = tuple(
            branch.name for branch in self.branch_bank.branches if branch.enabled
        )
        if (
            not isinstance(self.branch_names, tuple)
            or not self.branch_names
            or len(set(self.branch_names)) != len(self.branch_names)
            or any(
                not isinstance(name, str) or not name.strip()
                for name in self.branch_names
            )
            or not isinstance(self.branch_scales, tuple)
            or len(self.branch_names) != len(self.branch_scales)
        ):
            raise ValueError("Dynamic threshold branch family/scales are invalid.")
        if any(not _is_real_number(scale) for scale in self.branch_scales):
            raise TypeError("Dynamic threshold branch scales must be numeric.")
        if any(
            scale <= 0.0 or not np.isfinite(scale)
            for scale in self.branch_scales
        ):
            raise ValueError("Dynamic threshold branch family/scales are invalid.")
        if self.branch_names != enabled_names:
            raise ValueError(
                "Dynamic threshold branch names must match the embedded enabled BranchBank."
            )
        if not _is_real_number(self.threshold_floor):
            raise TypeError("Dynamic threshold floor must be numeric.")
        if self.threshold_floor <= 0.0 or not np.isfinite(self.threshold_floor):
            raise ValueError("Dynamic threshold floor must be finite and positive.")
        if (
            not isinstance(self.mode_names, tuple)
            or not self.mode_names
            or len(set(self.mode_names)) != len(self.mode_names)
        ):
            raise ValueError("Dynamic threshold mode family must be non-empty and unique.")
        if any(
            not isinstance(name, str) or not name.strip() for name in self.mode_names
        ):
            raise TypeError("Dynamic threshold mode names must be non-empty strings.")

    @classmethod
    def freeze(
        cls,
        *,
        branch_bank: BranchBank,
        candidate_hash: str,
        input_envelope: InputDependentEnvelope,
        context_age_envelope: ContextAgeEnvelope,
        branch_scales: Mapping[str, float],
        threshold_floor: float,
        normalization_source_hash: str,
        mode_names: Sequence[str],
        reset_state_hash: str,
        stage: MonitorStage | str,
    ) -> "DynamicThresholdGenerator":
        """在 estimate 阶段冻结完整 detection score map。

        参数：
            branch_bank/candidate_hash: P7 完整 family 与候选内容 hash。
            input_envelope/context_age_envelope: estimate-only 包络。
            branch_scales: 每条启用 branch 的严格正尺度；键必须恰好匹配 family。
            threshold_floor/normalization_source_hash: 正 floor 与 scale/floor 的 estimate
                来源 hash。
            mode_names/reset_state_hash: 有限 mode 与 monitor reset。
            stage: 只允许 ``estimate``。
        返回：
            不包含 ``q_det`` 的冻结 ``DynamicThresholdGenerator``。
        异常：
            越权阶段、branch/scale 集合不一致或配置非法时抛出
            ``TypeError``/``ValueError``。
        副作用：
            无。
        """

        normalized_stage = MonitorStage.parse(stage)
        if normalized_stage is not MonitorStage.ESTIMATE:
            raise ValueError("Dynamic threshold score map may only be frozen on estimate.")
        if not isinstance(branch_bank, BranchBank):
            raise TypeError("branch_bank must be a BranchBank.")
        if not isinstance(branch_scales, Mapping):
            raise TypeError("branch_scales must be a mapping.")
        enabled_names = tuple(
            branch.name for branch in branch_bank.branches if branch.enabled
        )
        if set(branch_scales) != set(enabled_names):
            raise ValueError(
                "branch_scales keys must exactly match every enabled post-filter branch."
            )
        normalized_scales: list[float] = []
        for name in enabled_names:
            normalized_scales.append(
                _coerce_finite_float(
                    branch_scales[name],
                    name=f"Branch scale for {name!r}",
                )
            )
        normalized_floor = _coerce_finite_float(
            threshold_floor,
            name="Dynamic threshold floor",
        )
        if isinstance(mode_names, (str, bytes)) or not isinstance(
            mode_names,
            Sequence,
        ):
            raise TypeError("Dynamic threshold mode_names must be a sequence of strings.")
        normalized_modes = tuple(mode_names)
        if any(
            not isinstance(name, str) or not name.strip() for name in normalized_modes
        ):
            raise TypeError("Dynamic threshold mode names must be non-empty strings.")
        return cls(
            stage=normalized_stage,
            candidate_hash=candidate_hash,
            branch_bank=branch_bank,
            input_envelope=input_envelope,
            context_age_envelope=context_age_envelope,
            branch_names=enabled_names,
            branch_scales=tuple(normalized_scales),
            threshold_floor=normalized_floor,
            normalization_source_hash=normalization_source_hash,
            mode_names=normalized_modes,
            reset_state_hash=reset_state_hash,
        )

    @property
    def content_hash(self) -> str:
        """返回校准前 score map 的确定性 SHA-256；不包含任何 calibration quantile。"""

        payload = {
            "stage": self.stage.value,
            "candidate_hash": self.candidate_hash,
            "branch_bank": self.branch_bank.to_dict(),
            "input_envelope": _input_envelope_payload(self.input_envelope),
            "context_age_envelope": _age_envelope_payload(self.context_age_envelope),
            "branch_names": list(self.branch_names),
            "branch_scales": list(self.branch_scales),
            "threshold_floor": self.threshold_floor,
            "normalization_source_hash": self.normalization_source_hash,
            "mode_names": list(self.mode_names),
            "reset_state_hash": self.reset_state_hash,
        }
        return _content_hash(payload)

    def scale_for(self, branch_name: str) -> float:
        """按冻结 branch 名返回正尺度；未知或禁用 branch 时抛出 ``KeyError``。"""

        try:
            index = self.branch_names.index(branch_name)
        except ValueError as exc:
            legal = ", ".join(self.branch_names)
            raise KeyError(
                f"Unknown dynamic-threshold branch {branch_name!r}; legal names: {legal}."
            ) from exc
        return self.branch_scales[index]

    def score(
        self,
        *,
        statistic: float,
        radius: DeterministicRadius,
        time_index: int,
        mode: str,
    ) -> DetectionScore:
        """用冻结 estimate score map 生成一个 calibration/online 共同分数。

        参数：
            statistic/radius: 同一 branch/path 的可观测统计量与确定性半径。
            time_index/mode: 预声明 finite family 坐标。
        返回：
            ``DetectionScore``；不支持半径映射为无限分数。
        异常：
            branch、mode、索引或类型不属于冻结 family 时抛出 ``TypeError``/``ValueError``。
        副作用：
            无。
        """

        if not isinstance(radius, DeterministicRadius):
            raise TypeError("radius must be a DeterministicRadius.")
        if type(time_index) is not int or time_index < 0:
            raise ValueError("Dynamic threshold time_index must be a non-negative integer.")
        if mode not in self.mode_names:
            raise ValueError(f"Dynamic threshold mode {mode!r} is not frozen.")
        scale = self.scale_for(radius.branch_name)
        return DetectionScore.from_components(
            score_map_hash=self.content_hash,
            episode_id=radius.episode_id,
            coordinate=ScoreCoordinate(
                time_index=time_index,
                mode=mode,
                branch_name=radius.branch_name,
            ),
            statistic=statistic,
            gamma_anchor=radius.gamma_anchor,
            gamma_deterministic=radius.gamma_deterministic,
            scale=scale,
            unsupported_reason=radius.reason,
        )

    def evaluate(
        self,
        *,
        statistic: float,
        radius: DeterministicRadius,
        time_index: int,
        mode: str,
        calibration: EpisodeMaxCalibrator | None,
        episode_definition_hash: str,
    ) -> ThresholdResult:
        """组合独立检测校准并执行严格超限；证据不全时返回无限阈值。

        参数：
            statistic/radius/time_index/mode: 与 ``score`` 相同。
            calibration: 可选 detection episode 校准；必须 READY，且 score-map/reset
                hash 与本 generator 完全一致。
            episode_definition_hash: 当前目标有限 episode 的冻结定义；必须与校准一致。
        返回：
            分项 ``ThresholdResult``。
        异常：
            基础 score 输入非法时传播 ``score`` 的错误；缺失或错配 calibration 不抛
            异常，而失败关闭为无限阈值。
        副作用：
            无；最终 quantile 不写回 generator 或 monitor state。
        """

        score = self.score(
            statistic=statistic,
            radius=radius,
            time_index=time_index,
            mode=mode,
        )
        if not _is_sha256(episode_definition_hash):
            raise ValueError("Target episode_definition_hash must be a 64-character SHA-256.")
        disabled_reason: str | None = None
        if not score.supported:
            disabled_reason = score.reason or "Deterministic radius is unsupported."
        elif calibration is None:
            disabled_reason = "Detection calibration evidence is missing."
        elif calibration.status is not CalibrationStatus.READY:
            disabled_reason = calibration.reason or "Detection calibration is not ready."
        elif calibration.score_map_hash != self.content_hash:
            disabled_reason = "Detection calibration score_map_hash does not match."
        elif calibration.reset_state_hash != self.reset_state_hash:
            disabled_reason = "Detection calibration reset_state_hash does not match."
        elif calibration.episode_definition_hash != episode_definition_hash:
            disabled_reason = "Detection calibration episode_definition_hash does not match."
        elif score.coordinate not in calibration.expected_coordinates:
            disabled_reason = "Online score coordinate was not included in calibration."

        if disabled_reason is not None:
            return ThresholdResult(
                score_map_hash=self.content_hash,
                episode_definition_hash=episode_definition_hash,
                episode_id=radius.episode_id,
                coordinate=score.coordinate,
                operator_status=radius.operator_status,
                statistic=float(statistic),
                gamma_anchor=radius.gamma_anchor,
                gamma_deterministic=radius.gamma_deterministic,
                scale=score.scale,
                calibration_quantile=float("inf"),
                calibration_component=float("inf"),
                threshold_floor=self.threshold_floor,
                threshold=float("inf"),
                normalized_excess=score.normalized_excess,
                alarm=False,
                status=ThresholdStatus.DISABLED,
                calibration_status=None if calibration is None else calibration.status,
                reason=disabled_reason,
            )
        if calibration is None:
            raise RuntimeError("Validated calibration unexpectedly missing.")
        with np.errstate(over="ignore", invalid="ignore"):
            calibration_component = float(
                np.multiply(score.scale, calibration.quantile)
            )
            threshold_sum = float(
                np.sum(
                    np.asarray(
                        (
                            radius.gamma_anchor,
                            radius.gamma_deterministic,
                            calibration_component,
                        ),
                        dtype=np.float64,
                    )
                )
            )
        threshold = max(self.threshold_floor, threshold_sum)
        if not np.isfinite(threshold):
            return ThresholdResult(
                score_map_hash=self.content_hash,
                episode_definition_hash=episode_definition_hash,
                episode_id=radius.episode_id,
                coordinate=score.coordinate,
                operator_status=radius.operator_status,
                statistic=float(statistic),
                gamma_anchor=radius.gamma_anchor,
                gamma_deterministic=radius.gamma_deterministic,
                scale=score.scale,
                calibration_quantile=calibration.quantile,
                calibration_component=float("inf"),
                threshold_floor=self.threshold_floor,
                threshold=float("inf"),
                normalized_excess=score.normalized_excess,
                alarm=False,
                status=ThresholdStatus.DISABLED,
                calibration_status=calibration.status,
                reason=(
                    "Numerical failure: dynamic threshold sum is not representable "
                    "in float64."
                ),
            )
        return ThresholdResult(
            score_map_hash=self.content_hash,
            episode_definition_hash=episode_definition_hash,
            episode_id=radius.episode_id,
            coordinate=score.coordinate,
            operator_status=radius.operator_status,
            statistic=float(statistic),
            gamma_anchor=radius.gamma_anchor,
            gamma_deterministic=radius.gamma_deterministic,
            scale=score.scale,
            calibration_quantile=calibration.quantile,
            calibration_component=calibration_component,
            threshold_floor=self.threshold_floor,
            threshold=threshold,
            normalized_excess=score.normalized_excess,
            alarm=bool(float(statistic) > threshold),
            status=ThresholdStatus.READY,
            calibration_status=calibration.status,
        )

    def to_dict(self) -> dict[str, object]:
        """返回不含最终 ``q_det`` 的冻结 score-map 标准 JSON 字典。"""

        return {
            "stage": self.stage.value,
            "candidate_hash": self.candidate_hash,
            "branch_bank": self.branch_bank.to_dict(),
            "input_envelope": self.input_envelope.to_dict(),
            "context_age_envelope": self.context_age_envelope.to_dict(),
            "branch_names": list(self.branch_names),
            "branch_scales": list(self.branch_scales),
            "threshold_floor": self.threshold_floor,
            "normalization_source_hash": self.normalization_source_hash,
            "mode_names": list(self.mode_names),
            "reset_state_hash": self.reset_state_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "DynamicThresholdGenerator":
        """从严格 JSON 恢复 score map；额外 ``q_det`` 字段会被拒绝。"""

        _require_exact_keys(
            value,
            {
                "stage",
                "candidate_hash",
                "branch_bank",
                "input_envelope",
                "context_age_envelope",
                "branch_names",
                "branch_scales",
                "threshold_floor",
                "normalization_source_hash",
                "mode_names",
                "reset_state_hash",
            },
            name="DynamicThresholdGenerator",
        )
        raw_branch_names = _strict_list(
            value["branch_names"],
            name="dynamic threshold branch_names",
        )
        raw_mode_names = _strict_list(
            value["mode_names"],
            name="dynamic threshold mode_names",
        )
        return cls(
            stage=MonitorStage.parse(
                _strict_string(value["stage"], name="dynamic threshold stage")
            ),
            candidate_hash=_strict_string(
                value["candidate_hash"],
                name="dynamic threshold candidate_hash",
            ),
            branch_bank=BranchBank.from_dict(
                _strict_mapping(
                    value["branch_bank"],
                    name="dynamic threshold branch_bank",
                )
            ),
            input_envelope=InputDependentEnvelope.from_dict(
                _strict_mapping(
                    value["input_envelope"],
                    name="dynamic threshold input_envelope",
                )
            ),
            context_age_envelope=ContextAgeEnvelope.from_dict(
                _strict_mapping(
                    value["context_age_envelope"],
                    name="dynamic threshold context_age_envelope",
                )
            ),
            branch_names=tuple(
                _strict_string(item, name="dynamic threshold branch_name")
                for item in raw_branch_names
            ),
            branch_scales=_strict_float_tuple(
                value["branch_scales"],
                name="dynamic threshold branch_scales",
            ),
            threshold_floor=_strict_float(
                value["threshold_floor"],
                name="dynamic threshold floor",
            ),
            normalization_source_hash=_strict_string(
                value["normalization_source_hash"],
                name="dynamic threshold normalization_source_hash",
            ),
            mode_names=tuple(
                _strict_string(item, name="dynamic threshold mode_name")
                for item in raw_mode_names
            ),
            reset_state_hash=_strict_string(
                value["reset_state_hash"],
                name="dynamic threshold reset_state_hash",
            ),
        )


def _fit_nonnegative_quantile(
    design: np.ndarray,
    response: np.ndarray,
    *,
    quantile: float,
) -> np.ndarray:
    """用 pinball-loss 线性规划求非负系数，不引入额外截距。"""

    sample_count, feature_count = design.shape
    objective = np.concatenate(
        (
            np.zeros(feature_count, dtype=np.float64),
            np.full(sample_count, quantile, dtype=np.float64),
            np.full(sample_count, 1.0 - quantile, dtype=np.float64),
        )
    )
    equality = np.concatenate(
        (
            design,
            np.eye(sample_count, dtype=np.float64),
            -np.eye(sample_count, dtype=np.float64),
        ),
        axis=1,
    )
    result = linprog(
        objective,
        A_eq=equality,
        b_eq=response,
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success or result.x is None:
        raise ValueError(
            "Non-negative input-envelope quantile fit failed: "
            f"{result.message or 'unknown solver failure'}"
        )
    coefficients = np.asarray(result.x[:feature_count], dtype=np.float64)
    if not np.isfinite(coefficients).all():
        raise ValueError("Non-negative input-envelope fit returned non-finite coefficients.")
    return np.maximum(coefficients, 0.0)


def _is_real_number(value: object) -> bool:
    """返回值是否为非布尔的实数标量，拒绝字符串和复数的隐式转换。"""

    return not isinstance(value, (bool, np.bool_)) and isinstance(
        value,
        (int, float, np.integer, np.floating),
    )


def _coerce_finite_float(value: object, *, name: str) -> float:
    """把公开运行时实数转成有限 float，不接受 bool 或数字字符串。"""

    if not _is_real_number(value):
        raise TypeError(f"{name} must be numeric.")
    try:
        parsed = float(cast(float, value))
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} is not representable as float64.") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite.")
    return parsed


def _stable_vector_norm(values: Sequence[float]) -> float:
    """用幅值缩放计算向量 2-范数，避免大有限输入在平方时溢出。"""

    array = np.abs(np.asarray(values, dtype=np.float64))
    scale = float(np.max(array, initial=0.0))
    if scale == 0.0:
        return 0.0
    value = float(np.linalg.norm(array / scale, ord=2) * scale)
    if not np.isfinite(value):
        raise ValueError("Input descriptor norm is not representable in float64.")
    return value


def _scaled_matrix_spectral_norm(matrix: np.ndarray) -> float:
    """缩放后计算二维矩阵谱范数，数值失败时由调用方得到明确异常。"""

    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Transformed nuisance block must be a finite matrix.")
    if matrix.size == 0:
        return 0.0
    scale = float(np.max(np.abs(matrix), initial=0.0))
    if scale == 0.0:
        return 0.0
    try:
        singular_values = np.linalg.svd(matrix / scale, compute_uv=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Transformed nuisance block SVD failed.") from exc
    value = float(singular_values[0] * scale)
    if not np.isfinite(value):
        raise ValueError("Transformed nuisance block norm is not representable.")
    return value


def _is_sha256(value: object) -> bool:
    """返回值是否为 64 位小写十六进制 SHA-256。"""

    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _input_envelope_payload(envelope: InputDependentEnvelope) -> dict[str, object]:
    """返回 input envelope 的稳定 hash payload，不包含运行时缓存。"""

    return {
        "stage": envelope.stage.value,
        "quantile": envelope.quantile,
        "minimum_region_samples": envelope.minimum_region_samples,
        "source_hash": envelope.source_hash,
        "regions": [
            {
                "region": region.region,
                "coefficients": list(region.coefficients),
                "feature_minima": list(region.feature_minima),
                "feature_maxima": list(region.feature_maxima),
                "sample_count": region.sample_count,
            }
            for region in envelope.regions
        ],
    }


def _age_envelope_payload(envelope: ContextAgeEnvelope) -> dict[str, object]:
    """返回 context-age envelope 的稳定 hash payload。"""

    return {
        "stage": envelope.stage.value,
        "quantile": envelope.quantile,
        "minimum_samples_per_age": envelope.minimum_samples_per_age,
        "source_hash": envelope.source_hash,
        "values": list(envelope.values),
        "sample_counts": list(envelope.sample_counts),
    }


def _content_hash(value: object) -> str:
    """对无 NaN 的 JSON 值计算确定性 SHA-256。"""

    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    name: str,
) -> None:
    """要求 JSON 对象字段与 schema 完全一致，拒绝静默忽略扩展。"""

    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} fields differ; missing={missing}, extra={extra}.")


def _strict_mapping(value: object, *, name: str) -> Mapping[str, object]:
    """把值验证为字符串键 JSON 对象。"""

    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise TypeError(f"{name} must be a string-keyed mapping.")
    return value


def _strict_list(value: object, *, name: str) -> list[object]:
    """把值验证为 JSON list；tuple 不作为严格重放输入接受。"""

    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON list.")
    return value


def _strict_string(value: object, *, name: str) -> str:
    """解析非空字符串，拒绝用数字或枚举对象隐式转换。"""

    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string.")
    return value


def _strict_bool(value: object, *, name: str) -> bool:
    """解析真正布尔值，拒绝整数 0/1。"""

    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool.")
    return value


def _strict_int(value: object, *, name: str) -> int:
    """解析真正整数，拒绝 bool 和整值浮点数。"""

    if type(value) is not int:
        raise TypeError(f"{name} must be an integer.")
    return value


def _strict_float(value: object, *, name: str) -> float:
    """解析有限 JSON 数值，拒绝 bool、字符串、NaN 和无穷。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite.")
    return parsed


def _strict_float_tuple(
    value: object,
    *,
    name: str,
    length: int | None = None,
) -> tuple[float, ...]:
    """解析 JSON 数值列表为有限浮点元组，并可要求固定长度。"""

    raw = _strict_list(value, name=name)
    if length is not None and len(raw) != length:
        raise ValueError(f"{name} must contain exactly {length} values.")
    return tuple(
        _strict_float(item, name=f"{name} item")
        for item in raw
    )


def _encode_finite_or_infinity(value: float) -> float | str:
    """把正无穷编码为受控字符串，避免非标准 JSON ``Infinity``。"""

    if value == float("inf"):
        return "+infinity"
    if not np.isfinite(value):
        raise ValueError("Only finite values or positive infinity can be serialized.")
    return float(value)


def _strict_finite_or_infinity(value: object, *, name: str) -> float:
    """解析有限数值或唯一受控 ``+infinity`` 字符串。"""

    if value == "+infinity":
        return float("inf")
    return _strict_float(value, name=name)
