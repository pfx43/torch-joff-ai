"""P7 white-space 堆叠残差白化、锚点谱商空间与检测支路库。

文件用途：
    把 P5 形成的堆叠受保护残差和 P6 装配的锚点传播算子 ``G_0`` 组合为论文部署所需
    的后滤波候选，统一实现跨时间白化、white-space 谱选择、omnibus/guard/matched
    branches。
主要职责：
    仅用正常 estimate 段拟合 Ledoit--Wolf 协方差并冻结 ``W_0``；在
    ``K_anc=W_0 G_0 R_eta^(1/2)`` 的白空间中整簇选择锚点方向；构造
    ``L_0=Q_w^T W_0``、``L_c=P_c L_0`` 与 ``L_g=I``。本文件不拟合 P8 检测分位、
    不读取故障数据，也不执行 P9 物理类别判决。
关键输入与输出：
    输入为 ``[n_estimate,d]`` 正常堆叠残差、``G_0=[d,m]``、锚点椭球平方根和可选
    retained-white-space 投影器；输出为不可变白化估计、候选诊断和共享同一部署矩阵
    的 ``BranchBank``。
依赖与副作用：
    依赖 NumPy、scikit-learn ``LedoitWolf`` 和 ``MonitorStage``；计算只在内存中进行，
    不读写文件、不访问网络、不修改随机数或全局数值配置。
重要约束：
    拟合和支路选择只接受 ``estimate`` 阶段。奇异值阈值必须平方后与 Gram 特征值比较；
    投影必须在 white space 选择后通过 ``Q_w^T W_0`` 映回 raw residual；样本或数值证据
    不足时 fail closed 到 guard，而不能借用 calibration/fault 数据重选候选。严格重放
    要求 ``W_0`` 是协方差的主对称正定逆平方根；超出 float64 可表示范围的锚点谱不得
    进入 quotient 选择。
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np
# scikit-learn 当前未声明 py.typed；只抑制这一个第三方导入的缺失类型元数据。
from sklearn.covariance import LedoitWolf  # type: ignore[import-untyped]

from .protected_reference import MonitorStage


MatrixTuple = tuple[tuple[float, ...], ...]


class SpectralMode(str, Enum):
    """锚点谱选择后实际部署的受控模式。

    ``NO_QUOTIENT`` 保留完整白空间，``PARTIAL_QUOTIENT`` 只去除有稳定谱隙的锚点方向，
    ``FULL_QUOTIENT`` 去除全部白空间方向，``HYBRID`` 明示阈值与谱簇碰撞，
    ``GUARD_ONLY`` 表示白化或谱证据不足时只保留 raw residual guard。
    """

    NO_QUOTIENT = "no_quotient"
    PARTIAL_QUOTIENT = "partial_quotient"
    FULL_QUOTIENT = "full_quotient"
    HYBRID = "hybrid"
    GUARD_ONLY = "guard_only"


class BranchKind(str, Enum):
    """P7 branch bank 中允许出现的三种受控支路类型。

    稳定候选必须恰有一个 ``OMNIBUS``、一个 ``GUARD``，其余支路只能是
    ``MATCHED``；guard-only fallback 只含一个 ``GUARD``。受控枚举避免严格重放时
    通过自由字符串注入第二个 omnibus 或 guard。
    """

    OMNIBUS = "omnibus"
    GUARD = "guard"
    MATCHED = "matched"


@dataclass(frozen=True)
class WhiteningEstimate:
    """仅正常 estimate 段拟合的跨时间白化估计。

    参数：
        stage: 拟合数据所属协议阶段，必须是 ``MonitorStage.ESTIMATE``。
        sample_count/feature_dim: estimate 矩阵的样本数和堆叠残差宽度。
        sample_mean: Ledoit--Wolf 中心化使用的 estimate 均值，仅供审计；部署线性算子
            不减该均值，以保持论文中的 ``L_b e`` 定义。
        covariance/operator: 加入 ridge 和特征值 floor 后的正定协方差及其逆平方根。
        shrinkage/ridge/eigenvalue_floor: 有限样本正则化分账。
        eigenvalues/effective_rank/condition_number: 最终矩阵的数值诊断。
        stable/fallback_reason: 当前白化能否授权谱候选；不稳定时 ``operator`` 是单位阵，
            只供 guard-only 确定性回退。
        minimum_samples/required_sample_count/max_condition_number/source_hash: 用户下限、
            随残差维数收紧后的实际样本下限、条件数上限与输入内容身份。
    返回：
        不可变、可严格 JSON 重放的白化证据；``stable=False`` 时只授权 guard-only。
    异常：
        直接构造或反序列化时，阶段、shape、谱、条件数、正定性或逆平方根关系矛盾会
        抛出 ``TypeError``/``ValueError``。
    副作用：
        对象不可变；构造和 ``fit`` 都不修改输入数组。
    """

    stage: MonitorStage
    sample_count: int
    feature_dim: int
    sample_mean: tuple[float, ...]
    covariance: MatrixTuple
    operator: MatrixTuple
    shrinkage: float | None
    ridge: float
    eigenvalue_floor: float
    eigenvalues: tuple[float, ...]
    effective_rank: int
    condition_number: float
    stable: bool
    fallback_reason: str | None
    minimum_samples: int
    required_sample_count: int
    max_condition_number: float
    source_hash: str

    def __post_init__(self) -> None:
        """校验白化矩阵、诊断量和 fallback 状态彼此一致。"""

        if self.stage is not MonitorStage.ESTIMATE:
            raise ValueError("Whitening estimate stage must be estimate.")
        if (
            type(self.sample_count) is not int
            or type(self.feature_dim) is not int
            or self.sample_count <= 0
            or self.feature_dim <= 0
        ):
            raise ValueError("Whitening sample_count and feature_dim must be positive integers.")
        if len(self.sample_mean) != self.feature_dim or not _all_finite(self.sample_mean):
            raise ValueError("Whitening sample_mean must match feature_dim and be finite.")
        _validate_matrix_tuple(
            self.covariance,
            name="Whitening covariance",
            rows=self.feature_dim,
            columns=self.feature_dim,
        )
        _validate_matrix_tuple(
            self.operator,
            name="Whitening operator",
            rows=self.feature_dim,
            columns=self.feature_dim,
        )
        covariance = np.asarray(self.covariance, dtype=np.float64)
        operator = np.asarray(self.operator, dtype=np.float64)
        if not np.allclose(covariance, covariance.T, atol=1e-10, rtol=0.0):
            raise ValueError("Whitening covariance must be symmetric.")
        if not np.allclose(operator, operator.T, atol=1e-10, rtol=0.0):
            raise ValueError("Whitening operator must be symmetric.")
        if self.shrinkage is not None and (
            not np.isfinite(self.shrinkage) or not 0.0 <= self.shrinkage <= 1.0
        ):
            raise ValueError("Whitening shrinkage must be None or lie in [0, 1].")
        if self.ridge < 0.0 or not np.isfinite(self.ridge):
            raise ValueError("Whitening ridge must be finite and non-negative.")
        if self.eigenvalue_floor <= 0.0 or not np.isfinite(self.eigenvalue_floor):
            raise ValueError("Whitening eigenvalue_floor must be finite and positive.")
        if len(self.eigenvalues) != self.feature_dim or any(
            value <= 0.0 or not np.isfinite(value) for value in self.eigenvalues
        ):
            raise ValueError("Whitening eigenvalues must be positive and match feature_dim.")
        if (
            type(self.effective_rank) is not int
            or not 0 <= self.effective_rank <= self.feature_dim
        ):
            raise ValueError("Whitening effective_rank must lie in [0, feature_dim].")
        if self.condition_number < 1.0 or not np.isfinite(self.condition_number):
            raise ValueError("Whitening condition_number must be finite and at least 1.")
        if type(self.stable) is not bool:
            raise TypeError("Whitening stable must be a bool.")
        if self.stable is (self.fallback_reason is not None):
            raise ValueError(
                "Stable whitening cannot have a fallback reason, and unstable whitening must."
            )
        if self.fallback_reason is not None and (
            not isinstance(self.fallback_reason, str) or not self.fallback_reason.strip()
        ):
            raise ValueError("Whitening fallback_reason must be a non-empty string.")
        if (
            type(self.minimum_samples) is not int
            or self.minimum_samples < 2
            or type(self.required_sample_count) is not int
            or self.required_sample_count
            != max(self.minimum_samples, self.feature_dim + 1)
            or self.max_condition_number <= 1.0
            or not np.isfinite(self.max_condition_number)
        ):
            raise ValueError("Whitening stopping-rule parameters are invalid.")
        if not _is_sha256(self.source_hash):
            raise ValueError("Whitening source_hash must be a 64-character SHA-256.")

        actual_eigenvalues, covariance_eigenvectors = np.linalg.eigh(covariance)
        if actual_eigenvalues[0] <= 0.0:
            raise ValueError("Whitening covariance must be positive definite.")
        expected_eigenvalues = actual_eigenvalues[::-1]
        if not np.allclose(
            np.asarray(self.eigenvalues),
            expected_eigenvalues,
            atol=1e-12,
            rtol=1e-10,
        ):
            raise ValueError("Whitening eigenvalues must match its covariance spectrum.")
        expected_condition_number = float(
            expected_eigenvalues[0] / expected_eigenvalues[-1]
        )
        if not np.isclose(
            self.condition_number,
            expected_condition_number,
            atol=1e-12,
            rtol=1e-10,
        ):
            raise ValueError("Whitening condition_number must match its covariance spectrum.")
        expected_rank = int(
            np.count_nonzero(
                actual_eigenvalues > self.eigenvalue_floor * (1.0 + 1e-12)
            )
        )
        if self.effective_rank != expected_rank:
            raise ValueError(
                "Whitening effective_rank must match eigenvalues above the frozen floor."
            )
        if self.stable:
            if (
                self.sample_count < self.required_sample_count
                or self.shrinkage is None
                or self.condition_number > self.max_condition_number
            ):
                raise ValueError(
                    "Stable whitening must satisfy sample, shrinkage, and condition gates."
                )
            if actual_eigenvalues[0] < self.eigenvalue_floor:
                raise ValueError(
                    "Stable whitening covariance must stay above its frozen eigenvalue floor."
                )
            expected_operator = (
                covariance_eigenvectors
                * np.power(actual_eigenvalues, -0.5)[None, :]
            ) @ covariance_eigenvectors.T
            comparison_scale = float(np.max(np.abs(expected_operator)))
            scaled_operator = operator / comparison_scale
            scaled_expected_operator = expected_operator / comparison_scale
            operator_eigenvalues = np.linalg.eigvalsh(
                0.5 * (operator + operator.T)
            )
            if (
                not np.allclose(
                    operator,
                    operator.T,
                    atol=comparison_scale * 1e-10,
                    rtol=1e-8,
                )
                or operator_eigenvalues[0] <= 0.0
                or not np.allclose(
                    scaled_operator,
                    scaled_expected_operator,
                    atol=1e-10,
                    rtol=1e-8,
                )
            ):
                raise ValueError(
                    "Whitening operator must be the principal symmetric positive-definite "
                    "covariance inverse square root."
                )
            if not np.allclose(
                operator @ covariance @ operator,
                np.eye(self.feature_dim),
                atol=1e-9,
                rtol=1e-8,
            ):
                raise ValueError("Whitening operator must satisfy W_0 covariance W_0 = I.")
        elif not np.allclose(
            operator,
            np.eye(self.feature_dim),
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError("Unstable whitening must fail closed to the identity operator.")

    @classmethod
    def fit(
        cls,
        residuals: Sequence[Sequence[float]] | np.ndarray,
        *,
        stage: MonitorStage | str,
        minimum_samples: int = 4,
        ridge: float = 1e-8,
        eigenvalue_floor: float = 1e-10,
        max_condition_number: float = 1e12,
    ) -> "WhiteningEstimate":
        """从正常 estimate 堆叠残差拟合冻结白化矩阵。

        参数：
            residuals: 二维有限矩阵 ``[n_estimate,d]``。
            stage: 数据所属阶段；任何 calibration、正常测试或故障测试值都会被拒绝。
            minimum_samples: 启用协方差候选所需的最少完整窗口数。
            ridge/eigenvalue_floor: Ledoit--Wolf 后叠加的绝对正则项。
            max_condition_number: 超过该最终条件数时回退 guard-only。
        返回：
            稳定时返回 Ledoit--Wolf 正则化白化；样本不足时返回带原因的单位阵回退。
        异常：
            阶段、shape、有限性或配置非法时抛出 ``ValueError``。
        副作用：
            无；scikit-learn 只拟合当前内存副本。
        """

        normalized_stage = MonitorStage.parse(stage)
        if normalized_stage is not MonitorStage.ESTIMATE:
            raise ValueError("Whitening may only fit the normal estimate stage.")
        matrix = _finite_matrix(residuals, name="estimate residuals")
        sample_count, feature_dim = matrix.shape
        if minimum_samples < 2:
            raise ValueError("Whitening minimum_samples must be at least 2.")
        if ridge < 0.0 or not np.isfinite(ridge):
            raise ValueError("Whitening ridge must be finite and non-negative.")
        if eigenvalue_floor <= 0.0 or not np.isfinite(eigenvalue_floor):
            raise ValueError("Whitening eigenvalue_floor must be finite and positive.")
        if max_condition_number <= 1.0 or not np.isfinite(max_condition_number):
            raise ValueError("Whitening max_condition_number must be finite and greater than 1.")

        source_hash = _array_hash(matrix)
        required_sample_count = max(minimum_samples, feature_dim + 1)
        if sample_count < required_sample_count:
            return _identity_whitening_fallback(
                matrix=matrix,
                stage=normalized_stage,
                minimum_samples=minimum_samples,
                required_sample_count=required_sample_count,
                ridge=float(ridge),
                eigenvalue_floor=float(eigenvalue_floor),
                max_condition_number=float(max_condition_number),
                source_hash=source_hash,
                reason=(
                    f"Estimate sample count {sample_count} is below dimension-aware minimum "
                    f"{required_sample_count}."
                ),
            )

        estimator = LedoitWolf(assume_centered=False, store_precision=False)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                estimator.fit(matrix)
                covariance = np.asarray(estimator.covariance_, dtype=np.float64)
                location = np.asarray(estimator.location_, dtype=np.float64)
                if not np.isfinite(covariance).all() or not np.isfinite(location).all():
                    raise FloatingPointError("Ledoit-Wolf covariance is non-finite.")
                covariance = 0.5 * (covariance + covariance.T)
                covariance = covariance + float(ridge) * np.eye(
                    feature_dim,
                    dtype=np.float64,
                )
                raw_eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                if not np.isfinite(raw_eigenvalues).all():
                    raise FloatingPointError("Whitening covariance spectrum is non-finite.")
                floored_eigenvalues = np.maximum(
                    raw_eigenvalues,
                    float(eigenvalue_floor),
                )
                covariance = (eigenvectors * floored_eigenvalues) @ eigenvectors.T
                inverse_roots = 1.0 / np.sqrt(floored_eigenvalues)
                operator = (eigenvectors * inverse_roots) @ eigenvectors.T
        except (FloatingPointError, RuntimeWarning, ValueError, np.linalg.LinAlgError) as exc:
            return _identity_whitening_fallback(
                matrix=matrix,
                stage=normalized_stage,
                minimum_samples=minimum_samples,
                required_sample_count=required_sample_count,
                ridge=float(ridge),
                eigenvalue_floor=float(eigenvalue_floor),
                max_condition_number=float(max_condition_number),
                source_hash=source_hash,
                reason=(
                    "Ledoit-Wolf covariance estimation failed closed "
                    f"({type(exc).__name__})."
                ),
            )

        condition_number = float(
            floored_eigenvalues.max() / floored_eigenvalues.min()
        )
        effective_rank = int(
            np.count_nonzero(
                floored_eigenvalues > eigenvalue_floor * (1.0 + 1e-12)
            )
        )
        stable = bool(
            np.isfinite(condition_number)
            and condition_number <= max_condition_number
            and np.isfinite(operator).all()
        )
        fallback_reason = None
        if not stable:
            fallback_reason = (
                "Regularized estimate covariance is numerically unstable: "
                f"condition_number={condition_number:.17g}, "
                f"limit={max_condition_number:.17g}."
            )
            # guard-only 的执行矩阵可以退回单位阵，但审计证据必须保留触发停止条件的
            # 实际协方差、谱和条件数；否则产物会掩盖为何不能授权 white-space 候选。
            operator = np.eye(feature_dim, dtype=np.float64)

        return cls(
            stage=normalized_stage,
            sample_count=sample_count,
            feature_dim=feature_dim,
            sample_mean=tuple(float(value) for value in location),
            covariance=_matrix_tuple(covariance),
            operator=_matrix_tuple(operator),
            shrinkage=float(estimator.shrinkage_),
            ridge=float(ridge),
            eigenvalue_floor=float(eigenvalue_floor),
            eigenvalues=tuple(float(value) for value in floored_eigenvalues[::-1]),
            effective_rank=effective_rank,
            condition_number=condition_number,
            stable=stable,
            fallback_reason=fallback_reason,
            minimum_samples=minimum_samples,
            required_sample_count=required_sample_count,
            max_condition_number=float(max_condition_number),
            source_hash=source_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        """返回包含正则化、数值稳定性和输入身份的 JSON 表示。

        参数：
            无。
        返回：
            可直接交给 ``json.dumps`` 的新字典。
        异常：
            无；对象在构造时已经通过全部不变量。
        副作用：
            无；返回值不暴露内部元组。
        """

        return {
            "stage": self.stage.value,
            "sample_count": self.sample_count,
            "feature_dim": self.feature_dim,
            "sample_mean": list(self.sample_mean),
            "covariance": _matrix_list(self.covariance),
            "operator": _matrix_list(self.operator),
            "shrinkage": self.shrinkage,
            "ridge": self.ridge,
            "eigenvalue_floor": self.eigenvalue_floor,
            "eigenvalues": list(self.eigenvalues),
            "effective_rank": self.effective_rank,
            "condition_number": self.condition_number,
            "stable": self.stable,
            "fallback_reason": self.fallback_reason,
            "minimum_samples": self.minimum_samples,
            "required_sample_count": self.required_sample_count,
            "max_condition_number": self.max_condition_number,
            "source_hash": self.source_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WhiteningEstimate":
        """从严格 JSON 映射恢复白化估计并重新执行全部数值不变量。

        参数：
            value: ``to_dict`` 产生并可能落盘后的 JSON 对象。
        返回：
            重新验证正定协方差、谱、条件数和 ``W_0 Sigma W_0=I`` 的白化证据。
        异常：
            字段缺失/额外、类型宽松、shape 错误或派生几何不一致时抛出
            ``TypeError``/``ValueError``。
        副作用：
            无；不读写文件。
        """

        _require_exact_keys(
            value,
            {
                "stage",
                "sample_count",
                "feature_dim",
                "sample_mean",
                "covariance",
                "operator",
                "shrinkage",
                "ridge",
                "eigenvalue_floor",
                "eigenvalues",
                "effective_rank",
                "condition_number",
                "stable",
                "fallback_reason",
                "minimum_samples",
                "required_sample_count",
                "max_condition_number",
                "source_hash",
            },
            name="WhiteningEstimate",
        )
        shrinkage_value = value["shrinkage"]
        fallback_value = value["fallback_reason"]
        return cls(
            stage=MonitorStage.parse(value["stage"]),
            sample_count=_strict_int(value["sample_count"], name="sample_count"),
            feature_dim=_strict_int(value["feature_dim"], name="feature_dim"),
            sample_mean=_float_tuple(value["sample_mean"], name="sample_mean"),
            covariance=_coerce_matrix_tuple(value["covariance"], name="covariance"),
            operator=_coerce_matrix_tuple(value["operator"], name="operator"),
            shrinkage=None
            if shrinkage_value is None
            else _strict_float(shrinkage_value, name="shrinkage"),
            ridge=_strict_float(value["ridge"], name="ridge"),
            eigenvalue_floor=_strict_float(
                value["eigenvalue_floor"],
                name="eigenvalue_floor",
            ),
            eigenvalues=_float_tuple(value["eigenvalues"], name="eigenvalues"),
            effective_rank=_strict_int(value["effective_rank"], name="effective_rank"),
            condition_number=_strict_float(
                value["condition_number"],
                name="condition_number",
            ),
            stable=_strict_bool(value["stable"], name="stable"),
            fallback_reason=None
            if fallback_value is None
            else _strict_string(fallback_value, name="fallback_reason"),
            minimum_samples=_strict_int(value["minimum_samples"], name="minimum_samples"),
            required_sample_count=_strict_int(
                value["required_sample_count"],
                name="required_sample_count",
            ),
            max_condition_number=_strict_float(
                value["max_condition_number"],
                name="max_condition_number",
            ),
            source_hash=_strict_string(value["source_hash"], name="source_hash"),
        )


def _identity_whitening_fallback(
    *,
    matrix: np.ndarray,
    stage: MonitorStage,
    minimum_samples: int,
    required_sample_count: int,
    ridge: float,
    eigenvalue_floor: float,
    max_condition_number: float,
    source_hash: str,
    reason: str,
) -> WhiteningEstimate:
    """构造不授权谱选择的单位执行算子，同时保留停止规则和输入身份。

    参数：
        matrix/stage: 已验证的 estimate 残差及其受控阶段。
        minimum_samples/required_sample_count: 用户下限与维数感知实际下限。
        ridge/eigenvalue_floor/max_condition_number: 原拟合请求的冻结数值配置。
        source_hash/reason: 输入内容身份与不授权白化的明确原因。
    返回：
        ``stable=False``、``operator=I`` 的 ``WhiteningEstimate``。
    异常：
        参数矛盾时由 ``WhiteningEstimate`` 不变量抛出 ``ValueError``。
    副作用：
        无；不修改 ``matrix``。
    """

    feature_dim = int(matrix.shape[1])
    identity = np.eye(feature_dim, dtype=np.float64)
    covariance_scale = max(1.0, eigenvalue_floor)
    covariance = covariance_scale * identity
    effective_rank = int(
        np.count_nonzero(
            np.full(feature_dim, covariance_scale)
            > eigenvalue_floor * (1.0 + 1e-12)
        )
    )
    return WhiteningEstimate(
        stage=stage,
        sample_count=int(matrix.shape[0]),
        feature_dim=feature_dim,
        sample_mean=_scaled_column_mean(matrix),
        covariance=_matrix_tuple(covariance),
        operator=_matrix_tuple(identity),
        shrinkage=None,
        ridge=ridge,
        eigenvalue_floor=eigenvalue_floor,
        eigenvalues=tuple(covariance_scale for _ in range(feature_dim)),
        effective_rank=effective_rank,
        condition_number=1.0,
        stable=False,
        fallback_reason=reason,
        minimum_samples=minimum_samples,
        required_sample_count=required_sample_count,
        max_condition_number=max_condition_number,
        source_hash=source_hash,
    )


@dataclass(frozen=True)
class BranchOperator:
    """一个检测统计量及后续阈值/签名共同使用的部署矩阵。

    ``matrix`` 始终从 raw stacked residual 映到当前支路坐标。``anchor_radius`` 由同一
    矩阵传播 ``G_0 R_eta^(1/2)`` 得到，避免统计量与确定性阈值使用不同坐标。
    参数：
        name/kind: 稳定支路名及受控 ``BranchKind``。
        input_dim/matrix: raw residual 宽度和 ``L_b``；允许禁用的零行输出矩阵。
        anchor_radius: 同一个 ``L_b`` 得到的锚点椭球谱范数半径。
        enabled/disabled_reason: 是否进入 bank maximum 及禁用依据。
    返回：
        不可变、可严格 JSON 重放的部署支路。
    异常：
        名称与种类、shape、半径或启用状态矛盾时抛出 ``TypeError``/``ValueError``。
    副作用：
        无。
    """

    name: str
    kind: BranchKind
    input_dim: int
    matrix: MatrixTuple
    anchor_radius: float
    enabled: bool = True
    disabled_reason: str | None = None

    def __post_init__(self) -> None:
        """校验分支名称、shape、半径与启用状态。"""

        if not self.name or not isinstance(self.kind, BranchKind):
            raise ValueError("Branch name must be non-empty and kind must be controlled.")
        if (
            (self.kind is BranchKind.OMNIBUS and self.name != "omnibus")
            or (self.kind is BranchKind.GUARD and self.name != "guard")
            or (
                self.kind is BranchKind.MATCHED
                and self.name in {"omnibus", "guard"}
            )
        ):
            raise ValueError("Branch name must agree with its controlled kind.")
        if type(self.input_dim) is not int or self.input_dim <= 0:
            raise ValueError("Branch input_dim must be a positive integer.")
        _validate_matrix_tuple(
            self.matrix,
            name=f"Branch {self.name} matrix",
            rows=len(self.matrix),
            columns=self.input_dim,
            allow_zero_rows=True,
        )
        if self.anchor_radius < 0.0 or not np.isfinite(self.anchor_radius):
            raise ValueError("Branch anchor_radius must be finite and non-negative.")
        if type(self.enabled) is not bool:
            raise TypeError("Branch enabled must be a bool.")
        if self.enabled is (self.disabled_reason is not None):
            raise ValueError(
                "Enabled branch cannot have a disabled reason, and disabled branch must."
            )

    def transform(
        self,
        residual: Sequence[float] | np.ndarray,
    ) -> tuple[float, ...]:
        """把一个 raw stacked residual 映到当前支路坐标。

        参数：
            residual: 长度为 ``input_dim`` 的有限一维 raw stacked residual。
        返回：
            ``L_b e`` 的不可变浮点元组。
        异常：
            支路禁用时抛出 ``RuntimeError``；shape 或有限性非法时抛出 ``ValueError``。
        副作用：
            无；不修改输入。
        """

        if not self.enabled:
            raise RuntimeError(
                f"Post-filter branch {self.name!r} is disabled: {self.disabled_reason}"
            )
        vector = np.asarray(residual, dtype=np.float64)
        matrix = np.asarray(self.matrix, dtype=np.float64).reshape(
            len(self.matrix),
            self.input_dim,
        )
        if vector.ndim != 1 or vector.shape[0] != self.input_dim:
            raise ValueError("Branch residual width must match its deployed operator.")
        if not np.isfinite(vector).all():
            raise ValueError("Branch residual must contain only finite values.")
        return tuple(float(value) for value in matrix @ vector)

    def statistic(self, residual: Sequence[float] | np.ndarray) -> float:
        """返回 ``||L_b e||_2``，不执行 P8 阈值归一化或报警判决。

        参数：
            residual: 长度为 ``input_dim`` 的有限 raw stacked residual。
        返回：
            当前支路非负 Euclidean 统计量。
        异常：
            与 ``transform`` 相同。
        副作用：
            无。
        """

        return float(np.linalg.norm(self.transform(residual), ord=2))

    def transform_operator(
        self,
        raw_operator: Sequence[Sequence[float]] | np.ndarray,
    ) -> MatrixTuple:
        """用同一个 ``L_b`` 传播 nuisance 或 signature 的列算子。

        参数：
            raw_operator: 行数等于 raw stacked residual 宽度的二维列算子。
        返回：
            ``L_b @ raw_operator`` 的不可变矩阵；P8 确定性半径和 P9 signature 应只消费
            这个入口，避免另行重建投影坐标。
        异常：
            支路已禁用、输入 shape 不匹配或含非有限值时抛出 ``RuntimeError`` 或
            ``ValueError``。
        副作用：
            无；不修改输入矩阵。
        """

        if not self.enabled:
            raise RuntimeError(
                f"Post-filter branch {self.name!r} is disabled: {self.disabled_reason}"
            )
        operator = _finite_matrix(raw_operator, name="raw branch operator")
        if operator.shape[0] != self.input_dim:
            raise ValueError("Raw operator rows must match the branch input dimension.")
        matrix = np.asarray(self.matrix, dtype=np.float64).reshape(
            len(self.matrix),
            self.input_dim,
        )
        return _matrix_tuple(matrix @ operator)

    def to_dict(self) -> dict[str, Any]:
        """返回支路部署矩阵、锚点半径和禁用原因的 JSON 表示。

        参数：
            无。
        返回：
            可直接 JSON 编码的新字典。
        异常：
            无。
        副作用：
            无。
        """

        return {
            "name": self.name,
            "kind": self.kind.value,
            "input_dim": self.input_dim,
            "matrix": _matrix_list(self.matrix),
            "anchor_radius": self.anchor_radius,
            "enabled": self.enabled,
            "disabled_reason": self.disabled_reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BranchOperator":
        """从严格 JSON 映射恢复一个分支并验证 shape/状态。

        参数：
            value: 单个 branch 的 JSON 对象。
        返回：
            已验证名称、种类、shape、半径和启用状态的 ``BranchOperator``。
        异常：
            字段、类型、受控种类或对象不变量非法时抛出 ``TypeError``/``ValueError``。
        副作用：
            无。
        """

        _require_exact_keys(
            value,
            {
                "name",
                "kind",
                "input_dim",
                "matrix",
                "anchor_radius",
                "enabled",
                "disabled_reason",
            },
            name="BranchOperator",
        )
        reason = value["disabled_reason"]
        return cls(
            name=_strict_string(value["name"], name="branch name"),
            kind=BranchKind(value["kind"]),
            input_dim=_strict_int(value["input_dim"], name="branch input_dim"),
            matrix=_coerce_matrix_tuple(
                value["matrix"],
                name="branch matrix",
                allow_zero_rows=True,
            ),
            anchor_radius=_strict_float(
                value["anchor_radius"],
                name="branch anchor_radius",
            ),
            enabled=_strict_bool(value["enabled"], name="branch enabled"),
            disabled_reason=None
            if reason is None
            else _strict_string(reason, name="branch disabled_reason"),
        )


@dataclass(frozen=True)
class BranchBank:
    """冻结候选中的 omnibus、guard 和可选 matched branches。

    新增或改变任何支路都会改变 family-wise maximum 的正常分布，因此
    ``requires_recalibration`` 在 P7 固定为真；P8 必须在独立 detection calibration
    episode 上为整个 bank 重新定价。
    参数：
        branches: 稳定顺序的支路元组；恰有一个 guard，最多一个 omnibus。
        requires_recalibration: P7 固定为真，禁止复用旧 maximum 分位。
    返回：
        可按名称查询、无条件评价并严格重放的不可变 bank。
    异常：
        分支为空、名称重复、种类注入、输入宽度不一或校准标志为假时抛出
        ``TypeError``/``ValueError``。
    副作用：
        无。
    """

    branches: tuple[BranchOperator, ...]
    requires_recalibration: bool = True

    def __post_init__(self) -> None:
        """要求分支名称唯一、输入维度一致且 guard 恰好一个。"""

        if not isinstance(self.branches, tuple) or not self.branches:
            raise ValueError("BranchBank branches must be a non-empty tuple.")
        if not all(isinstance(branch, BranchOperator) for branch in self.branches):
            raise TypeError("BranchBank entries must be BranchOperator values.")
        names = tuple(branch.name for branch in self.branches)
        if len(set(names)) != len(names):
            raise ValueError("BranchBank branch names must be unique.")
        if names.count("guard") != 1:
            raise ValueError("BranchBank must contain exactly one guard branch.")
        if sum(branch.kind is BranchKind.GUARD for branch in self.branches) != 1:
            raise ValueError("BranchBank must contain exactly one guard kind.")
        if sum(branch.kind is BranchKind.OMNIBUS for branch in self.branches) > 1:
            raise ValueError("BranchBank cannot contain multiple omnibus branches.")
        if len({branch.input_dim for branch in self.branches}) != 1:
            raise ValueError("BranchBank branches must share one raw residual input dimension.")
        if self.requires_recalibration is not True:
            raise ValueError("A P7 BranchBank must require independent recalibration.")

    def branch(self, name: str) -> BranchOperator:
        """按稳定名称返回支路；未知名称时列出合法选项。

        参数：
            name: 精确支路名。
        返回：
            bank 中原有的不可变 ``BranchOperator``。
        异常：
            名称不存在时抛出 ``KeyError``。
        副作用：
            无。
        """

        for branch in self.branches:
            if branch.name == name:
                return branch
        legal = ", ".join(branch.name for branch in self.branches)
        raise KeyError(f"Unknown post-filter branch {name!r}. Legal names are: {legal}.")

    def evaluate(
        self,
        residual: Sequence[float] | np.ndarray,
    ) -> dict[str, float]:
        """无条件计算全部启用支路统计量，供后续统一校准。

        参数：
            residual: 与 bank 公共输入宽度一致的 raw stacked residual。
        返回：
            按稳定支路顺序生成的 ``name -> ||L_b e||_2`` 新字典；禁用支路不进入。
        异常：
            residual shape/有限性非法时传播 ``BranchOperator.statistic`` 的错误。
        副作用：
            无；不会根据报警状态条件执行支路。
        """

        return {
            branch.name: branch.statistic(residual)
            for branch in self.branches
            if branch.enabled
        }

    def to_dict(self) -> dict[str, Any]:
        """返回分支稳定顺序和重新校准义务的 JSON 表示。

        参数：
            无。
        返回：
            可直接 JSON 编码的新字典。
        异常：
            无。
        副作用：
            无。
        """

        return {
            "branches": [branch.to_dict() for branch in self.branches],
            "requires_recalibration": self.requires_recalibration,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BranchBank":
        """从严格 JSON 映射恢复完整 branch bank。

        参数：
            value: ``to_dict`` 产生的 JSON 对象。
        返回：
            重新验证分支身份、顺序无关基础不变量和校准义务的 bank。
        异常：
            字段、列表类型或嵌套 branch 非法时抛出 ``TypeError``/``ValueError``。
        副作用：
            无。
        """

        _require_exact_keys(
            value,
            {"branches", "requires_recalibration"},
            name="BranchBank",
        )
        branches = value["branches"]
        if not isinstance(branches, list):
            raise TypeError("BranchBank branches must be a JSON list.")
        return cls(
            branches=tuple(
                BranchOperator.from_dict(_strict_mapping(item, name="branch"))
                for item in branches
            ),
            requires_recalibration=_strict_bool(
                value["requires_recalibration"],
                name="requires_recalibration",
            ),
        )


@dataclass(frozen=True)
class PostFilterCandidate:
    """由一个冻结白化估计和一个 white-space 谱模式定义的 P7 候选。

    候选保存完整 ``Q_w`` 和 ``L_0``，使 P8/P9 能从同一矩阵传播确定性 nuisance 与
    signature。阈值碰撞标为 ``HYBRID``；白化不稳定时只构造 guard。
    参数：
        candidate_id/stage/mode: 候选身份、estimate 阶段和谱模式。
        whitening: 同一候选冻结的白化证据。
        singular_value_threshold/gram_threshold: ``tau`` 与严格派生的 ``tau^2``。
        minimum_projector_gap/spectral_cluster_tolerance/branch_scale_floor: 谱隙、整簇和
            零尺度支路停止规则。
        gram_eigenvalues/projector_gap/selected_rank/retained_rank: 谱选择诊断。
        retained_white_basis/common_operator: ``Q_w`` 与 ``L_0=Q_w^TW_0``。
        anchor_response/anchor_covariance_sqrt/branch_bank: ``G_0``、锚点椭球和完整 bank。
        fallback_reason: guard-only 时不授权谱候选的原因。
    返回：
        不可变、可严格 JSON 重放且带内容 hash 的后滤波候选。
    异常：
        阶段、公式、谱簇、分支结构或派生半径矛盾时抛出 ``TypeError``/``ValueError``。
    副作用：
        无。
    """

    candidate_id: str
    stage: MonitorStage
    mode: SpectralMode
    whitening: WhiteningEstimate
    singular_value_threshold: float
    gram_threshold: float
    minimum_projector_gap: float
    spectral_cluster_tolerance: float
    branch_scale_floor: float
    gram_eigenvalues: tuple[float, ...]
    projector_gap: float
    selected_rank: int
    retained_rank: int
    retained_white_basis: MatrixTuple
    common_operator: MatrixTuple
    anchor_response: MatrixTuple
    anchor_covariance_sqrt: MatrixTuple
    branch_bank: BranchBank
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        """校验谱秩、矩阵 shape、模式和 fallback 状态的一致性。"""

        if not self.candidate_id or self.stage is not MonitorStage.ESTIMATE:
            raise ValueError("Post-filter candidate identity and estimate stage are required.")
        if not isinstance(self.mode, SpectralMode) or not isinstance(
            self.whitening,
            WhiteningEstimate,
        ):
            raise TypeError("Post-filter mode and whitening estimate are required.")
        if self.whitening.stage is not self.stage:
            raise ValueError("Post-filter candidate and whitening stages must match.")
        numeric_parameters = (
            self.singular_value_threshold,
            self.gram_threshold,
            self.minimum_projector_gap,
            self.spectral_cluster_tolerance,
            self.projector_gap,
        )
        if any(value < 0.0 or not np.isfinite(value) for value in numeric_parameters):
            raise ValueError("Candidate spectral parameters must be finite and non-negative.")
        if not np.isclose(
            self.gram_threshold,
            self.singular_value_threshold**2,
            atol=1e-14,
            rtol=1e-12,
        ):
            raise ValueError("Candidate gram_threshold must equal singular_value_threshold squared.")
        if self.branch_scale_floor <= 0.0 or not np.isfinite(self.branch_scale_floor):
            raise ValueError("Candidate branch_scale_floor must be finite and positive.")
        raw_dim = self.whitening.feature_dim
        _validate_matrix_tuple(
            self.anchor_response,
            name="Candidate anchor_response",
            rows=raw_dim,
        )
        anchor_width = len(self.anchor_response[0])
        _validate_matrix_tuple(
            self.anchor_covariance_sqrt,
            name="Candidate anchor_covariance_sqrt",
            rows=anchor_width,
        )
        if (
            type(self.selected_rank) is not int
            or type(self.retained_rank) is not int
            or self.selected_rank < 0
            or self.retained_rank < 0
        ):
            raise ValueError("Candidate spectral ranks must be non-negative.")
        if self.mode is not SpectralMode.GUARD_ONLY:
            if self.selected_rank + self.retained_rank != raw_dim:
                raise ValueError("Candidate selected and retained ranks must partition white space.")
            if len(self.gram_eigenvalues) != raw_dim:
                raise ValueError("Candidate Gram spectrum must match whitening feature_dim.")
        _validate_matrix_tuple(
            self.retained_white_basis,
            name="Candidate retained_white_basis",
            rows=raw_dim,
            columns=self.retained_rank,
        )
        _validate_matrix_tuple(
            self.common_operator,
            name="Candidate common_operator",
            rows=self.retained_rank,
            columns=raw_dim,
            allow_zero_rows=True,
        )
        if not isinstance(self.branch_bank, BranchBank):
            raise TypeError("Candidate branch_bank must be a BranchBank.")
        if any(branch.input_dim != raw_dim for branch in self.branch_bank.branches):
            raise ValueError("Candidate branches must consume its raw residual width.")
        g_0 = np.asarray(self.anchor_response, dtype=np.float64)
        r_sqrt = np.asarray(self.anchor_covariance_sqrt, dtype=np.float64)
        w_0 = np.asarray(self.whitening.operator, dtype=np.float64)
        anchor_effect, white_anchor_effect, spectral_fallback_reason = (
            _prepare_anchor_spectrum(
                w_0=w_0,
                anchor_response=g_0,
                anchor_covariance_sqrt=r_sqrt,
            )
        )
        for branch in self.branch_bank.branches:
            matrix = np.asarray(branch.matrix, dtype=np.float64).reshape(
                len(branch.matrix),
                raw_dim,
            )
            expected_radius = (
                _propagated_spectral_radius(matrix, anchor_effect)
                if matrix.shape[0] > 0
                else 0.0
            )
            if not np.isclose(
                branch.anchor_radius,
                expected_radius,
                atol=1e-12,
                rtol=1e-10,
            ):
                raise ValueError(
                    f"Branch {branch.name!r} anchor_radius must use its deployed matrix."
                )
        guard = self.branch_bank.branch("guard")
        guard_matrix = np.asarray(guard.matrix, dtype=np.float64).reshape(raw_dim, raw_dim)
        if (
            guard.kind is not BranchKind.GUARD
            or not guard.enabled
            or not np.allclose(guard_matrix, np.eye(raw_dim), atol=1e-12, rtol=0.0)
        ):
            raise ValueError("Candidate guard must be the enabled raw-coordinate identity.")
        if self.mode is SpectralMode.GUARD_ONLY:
            if (
                self.fallback_reason is None
                or len(self.branch_bank.branches) != 1
                or self.selected_rank != 0
                or self.retained_rank != 0
                or self.gram_eigenvalues
            ):
                raise ValueError("GUARD_ONLY candidate requires one guard and a fallback reason.")
            expected_fallback_reason = (
                spectral_fallback_reason
                if self.whitening.stable
                else self.whitening.fallback_reason
            )
            if expected_fallback_reason is None or self.fallback_reason != expected_fallback_reason:
                raise ValueError(
                    "GUARD_ONLY fallback reason must match the replayed whitening or "
                    "anchor-spectrum stopping rule."
                )
            return
        if (
            not self.whitening.stable
            or self.fallback_reason is not None
            or spectral_fallback_reason is not None
            or anchor_effect is None
            or white_anchor_effect is None
        ):
            raise ValueError("Non-fallback candidate requires stable whitening and no fallback.")

        gram = white_anchor_effect @ white_anchor_effect.T
        expected_eigenvalues, expected_eigenvectors = np.linalg.eigh(
            0.5 * (gram + gram.T)
        )
        order = np.argsort(expected_eigenvalues)[::-1]
        expected_eigenvalues = np.maximum(expected_eigenvalues[order], 0.0)
        expected_eigenvectors = expected_eigenvectors[:, order]
        if not np.allclose(
            np.asarray(self.gram_eigenvalues),
            expected_eigenvalues,
            atol=1e-11,
            rtol=1e-10,
        ):
            raise ValueError("Candidate Gram spectrum must match W_0 G_0 R_eta^(1/2).")
        expected_gap = float(
            np.min(np.abs(expected_eigenvalues - self.gram_threshold))
        )
        if not np.isclose(self.projector_gap, expected_gap, atol=1e-12, rtol=1e-10):
            raise ValueError("Candidate projector_gap must match its Gram spectrum.")
        cluster_scale = max(float(expected_eigenvalues.max(initial=0.0)), 1.0)
        cluster_tolerance = self.spectral_cluster_tolerance * cluster_scale
        expected_selected, collision_rank = _stable_cluster_selection(
            expected_eigenvalues,
            threshold=self.gram_threshold,
            cluster_tolerance=cluster_tolerance,
            collision_tolerance=max(self.minimum_projector_gap, cluster_tolerance),
        )
        expected_selected_rank = int(np.count_nonzero(expected_selected))
        expected_retained_rank = raw_dim - expected_selected_rank
        if (
            self.selected_rank != expected_selected_rank
            or self.retained_rank != expected_retained_rank
        ):
            raise ValueError("Candidate ranks must match whole-cluster Gram selection.")
        retained_basis = np.asarray(self.retained_white_basis, dtype=np.float64).reshape(
            raw_dim,
            self.retained_rank,
        )
        expected_basis = expected_eigenvectors[:, ~expected_selected]
        if not np.allclose(
            retained_basis @ retained_basis.T,
            expected_basis @ expected_basis.T,
            atol=1e-10,
            rtol=0.0,
        ):
            raise ValueError("Candidate retained basis must span the selected white-space quotient.")
        common_operator = np.asarray(self.common_operator, dtype=np.float64).reshape(
            self.retained_rank,
            raw_dim,
        )
        if not np.allclose(
            common_operator,
            retained_basis.T @ w_0,
            atol=1e-11,
            rtol=1e-10,
        ):
            raise ValueError("Candidate common_operator must equal Q_w.T @ W_0.")
        if collision_rank > 0:
            expected_mode = SpectralMode.HYBRID
        elif expected_selected_rank == 0:
            expected_mode = SpectralMode.NO_QUOTIENT
        elif expected_retained_rank == 0:
            expected_mode = SpectralMode.FULL_QUOTIENT
        else:
            expected_mode = SpectralMode.PARTIAL_QUOTIENT
        if self.mode is not expected_mode:
            raise ValueError("Candidate mode must match its ranks and projector gap.")

        if len(self.branch_bank.branches) < 2:
            raise ValueError("Stable candidate must retain omnibus and guard branches.")
        omnibus = self.branch_bank.branches[0]
        if (
            omnibus.name != "omnibus"
            or omnibus.kind is not BranchKind.OMNIBUS
            or not np.allclose(
                np.asarray(omnibus.matrix, dtype=np.float64).reshape(
                    len(omnibus.matrix),
                    raw_dim,
                ),
                common_operator,
                atol=1e-11,
                rtol=1e-10,
            )
        ):
            raise ValueError("Candidate omnibus branch must use the common operator.")
        if self.branch_bank.branches[1].name != "guard":
            raise ValueError("Candidate branch order must place guard after omnibus.")
        if any(
            branch.kind is not BranchKind.MATCHED
            for branch in self.branch_bank.branches[2:]
        ):
            raise ValueError("Every branch after omnibus and guard must be matched.")
        for branch in self.branch_bank.branches:
            matrix = np.asarray(branch.matrix, dtype=np.float64).reshape(
                len(branch.matrix),
                raw_dim,
            )
            scale = float(np.linalg.norm(matrix, ord=2)) if matrix.size else 0.0
            expected_enabled = (
                branch.kind is BranchKind.GUARD or scale >= self.branch_scale_floor
            )
            if branch.enabled is not expected_enabled:
                raise ValueError(
                    f"Branch {branch.name!r} enabled state must use branch_scale_floor."
                )
            if branch.kind is BranchKind.MATCHED:
                projector = matrix @ np.linalg.pinv(common_operator)
                if not np.allclose(
                    matrix,
                    projector @ common_operator,
                    atol=1e-10,
                    rtol=1e-10,
                ) or not np.allclose(
                    projector,
                    projector.T,
                    atol=1e-10,
                    rtol=0.0,
                ) or not np.allclose(
                    projector @ projector,
                    projector,
                    atol=1e-10,
                    rtol=0.0,
                ):
                    raise ValueError(
                        f"Matched branch {branch.name!r} must equal P_c @ L_0 "
                        "for an orthogonal projector."
                    )

    @classmethod
    def fit(
        cls,
        *,
        candidate_id: str,
        whitening: WhiteningEstimate,
        anchor_response: Sequence[Sequence[float]] | np.ndarray,
        anchor_covariance_sqrt: Sequence[Sequence[float]] | np.ndarray,
        singular_value_threshold: float,
        stage: MonitorStage | str,
        matched_projectors: Mapping[
            str,
            Sequence[Sequence[float]] | np.ndarray,
        ]
        | None = None,
        minimum_projector_gap: float = 1e-8,
        spectral_cluster_tolerance: float = 1e-8,
        branch_scale_floor: float = 1e-12,
    ) -> "PostFilterCandidate":
        """在冻结白化坐标中选择锚点谱簇并构造完整分支库。

        参数：
            candidate_id: 运行内稳定且非空的候选标识。
            whitening: 同一 estimate 数据上已经冻结的白化估计。
            anchor_response: P6 ``G_0``，shape 为 ``[d,m]``。
            anchor_covariance_sqrt: ``R_eta^(1/2)``，shape 为 ``[m,r]``。
            singular_value_threshold: ``K_anc`` 奇异值阈值 ``tau``。
            stage: 分支选择阶段，只允许 estimate。
            matched_projectors: retained white space 中命名正交投影器 ``P_c``。
            minimum_projector_gap/spectral_cluster_tolerance: 谱稳定与整簇规则。
            branch_scale_floor: 禁用零尺度 matched/omnibus 支路的冻结正 floor。
        返回：
            包含 ``L_0``、guard 和 matched ``L_c`` 的不可变候选。
        异常：
            阶段、shape、有限性、投影器或阈值配置非法时抛出 ``ValueError``。
        副作用：
            无；不访问任何 calibration 或 fault 数据。
        """

        normalized_id = str(candidate_id).strip()
        if not normalized_id:
            raise ValueError("Post-filter candidate_id cannot be empty.")
        normalized_stage = MonitorStage.parse(stage)
        if normalized_stage is not MonitorStage.ESTIMATE:
            raise ValueError("Post-filter branch selection may only use the estimate stage.")
        if whitening.stage is not MonitorStage.ESTIMATE:
            raise ValueError("Post-filter whitening must originate from the estimate stage.")
        if (
            singular_value_threshold < 0.0
            or not np.isfinite(singular_value_threshold)
        ):
            raise ValueError("Post-filter singular_value_threshold must be finite and non-negative.")
        if minimum_projector_gap < 0.0 or not np.isfinite(minimum_projector_gap):
            raise ValueError("Post-filter minimum_projector_gap must be finite and non-negative.")
        if (
            spectral_cluster_tolerance < 0.0
            or not np.isfinite(spectral_cluster_tolerance)
        ):
            raise ValueError(
                "Post-filter spectral_cluster_tolerance must be finite and non-negative."
            )
        if branch_scale_floor <= 0.0 or not np.isfinite(branch_scale_floor):
            raise ValueError("Post-filter branch_scale_floor must be finite and positive.")

        g_0 = _finite_matrix(anchor_response, name="anchor response")
        r_sqrt = _finite_matrix(
            anchor_covariance_sqrt,
            name="anchor covariance square root",
        )
        if g_0.shape[0] != whitening.feature_dim:
            raise ValueError("Anchor response rows must match whitening feature_dim.")
        if g_0.shape[1] != r_sqrt.shape[0]:
            raise ValueError(
                "Anchor response columns must match anchor covariance square-root rows."
            )
        w_0 = np.asarray(whitening.operator, dtype=np.float64)
        raw_dim = whitening.feature_dim
        guard_matrix = np.eye(raw_dim, dtype=np.float64)
        anchor_effect, white_anchor_effect, spectral_fallback_reason = (
            _prepare_anchor_spectrum(
                w_0=w_0,
                anchor_response=g_0,
                anchor_covariance_sqrt=r_sqrt,
            )
        )
        guard = _branch(
            name="guard",
            kind=BranchKind.GUARD,
            matrix=guard_matrix,
            anchor_effect=anchor_effect,
            scale_floor=branch_scale_floor,
            force_enabled=True,
        )

        def guard_only(reason: str) -> "PostFilterCandidate":
            """构造不授权 quotient 的可审计 raw-guard 候选。"""

            empty_basis = np.empty((raw_dim, 0), dtype=np.float64)
            empty_operator = np.empty((0, raw_dim), dtype=np.float64)
            return cls(
                candidate_id=normalized_id,
                stage=normalized_stage,
                mode=SpectralMode.GUARD_ONLY,
                whitening=whitening,
                singular_value_threshold=float(singular_value_threshold),
                gram_threshold=float(singular_value_threshold**2),
                minimum_projector_gap=float(minimum_projector_gap),
                spectral_cluster_tolerance=float(spectral_cluster_tolerance),
                branch_scale_floor=float(branch_scale_floor),
                gram_eigenvalues=(),
                projector_gap=0.0,
                selected_rank=0,
                retained_rank=0,
                retained_white_basis=_matrix_tuple(empty_basis),
                common_operator=_matrix_tuple(empty_operator),
                anchor_response=_matrix_tuple(g_0),
                anchor_covariance_sqrt=_matrix_tuple(r_sqrt),
                branch_bank=BranchBank((guard,)),
                fallback_reason=reason,
            )

        if not whitening.stable:
            if whitening.fallback_reason is None:
                raise ValueError("Unstable whitening must provide a fallback reason.")
            return guard_only(whitening.fallback_reason)
        if spectral_fallback_reason is not None:
            return guard_only(spectral_fallback_reason)
        if anchor_effect is None or white_anchor_effect is None:
            raise RuntimeError("Validated anchor spectrum unexpectedly lacks finite matrices.")

        with np.errstate(over="ignore", invalid="ignore"):
            gram = white_anchor_effect @ white_anchor_effect.T
        if not np.isfinite(gram).all():
            return guard_only(
                "Anchor Gram spectrum is not representable in float64; quotient is disabled."
            )
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (gram + gram.T))
        except np.linalg.LinAlgError:
            return guard_only(
                "Anchor Gram spectrum decomposition failed; quotient is disabled."
            )
        if not np.isfinite(eigenvalues).all() or not np.isfinite(eigenvectors).all():
            return guard_only(
                "Anchor Gram spectrum decomposition is non-finite; quotient is disabled."
            )
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = eigenvectors[:, order]
        gram_threshold = float(singular_value_threshold**2)
        cluster_scale = max(float(eigenvalues.max(initial=0.0)), 1.0)
        collision_tolerance = max(
            float(minimum_projector_gap),
            float(spectral_cluster_tolerance) * cluster_scale,
        )
        projector_gap = float(np.min(np.abs(eigenvalues - gram_threshold)))

        selected, collision_rank = _stable_cluster_selection(
            eigenvalues,
            threshold=gram_threshold,
            cluster_tolerance=float(spectral_cluster_tolerance) * cluster_scale,
            collision_tolerance=collision_tolerance,
        )
        selected_rank = int(np.count_nonzero(selected))
        retained_basis = eigenvectors[:, ~selected]
        common_operator = retained_basis.T @ w_0
        retained_rank = int(common_operator.shape[0])

        if collision_rank > 0:
            mode = SpectralMode.HYBRID
        elif selected_rank == 0:
            mode = SpectralMode.NO_QUOTIENT
        elif retained_rank == 0:
            mode = SpectralMode.FULL_QUOTIENT
        else:
            mode = SpectralMode.PARTIAL_QUOTIENT

        omnibus = _branch(
            name="omnibus",
            kind=BranchKind.OMNIBUS,
            matrix=common_operator,
            anchor_effect=anchor_effect,
            scale_floor=branch_scale_floor,
        )
        branches = [omnibus, guard]
        for name, projector_value in sorted((matched_projectors or {}).items()):
            if not str(name).strip() or name in {"omnibus", "guard"}:
                raise ValueError("Matched branch names must be non-empty and non-reserved.")
            projector = _finite_matrix(
                projector_value,
                name=f"matched projector {name}",
            )
            expected_shape = (retained_rank, retained_rank)
            if projector.shape != expected_shape:
                raise ValueError(
                    f"Matched projector {name!r} must have shape {expected_shape}."
                )
            if not np.allclose(projector, projector.T, atol=1e-10, rtol=0.0):
                raise ValueError(f"Matched projector {name!r} must be symmetric.")
            if not np.allclose(
                projector @ projector,
                projector,
                atol=1e-10,
                rtol=0.0,
            ):
                raise ValueError(f"Matched projector {name!r} must be idempotent.")
            branches.append(
                _branch(
                    name=str(name),
                    kind=BranchKind.MATCHED,
                    matrix=projector @ common_operator,
                    anchor_effect=anchor_effect,
                    scale_floor=branch_scale_floor,
                )
            )

        return cls(
            candidate_id=normalized_id,
            stage=normalized_stage,
            mode=mode,
            whitening=whitening,
            singular_value_threshold=float(singular_value_threshold),
            gram_threshold=gram_threshold,
            minimum_projector_gap=float(minimum_projector_gap),
            spectral_cluster_tolerance=float(spectral_cluster_tolerance),
            branch_scale_floor=float(branch_scale_floor),
            gram_eigenvalues=tuple(float(value) for value in eigenvalues),
            projector_gap=projector_gap,
            selected_rank=selected_rank,
            retained_rank=retained_rank,
            retained_white_basis=_matrix_tuple(retained_basis),
            common_operator=_matrix_tuple(common_operator),
            anchor_response=_matrix_tuple(g_0),
            anchor_covariance_sqrt=_matrix_tuple(r_sqrt),
            branch_bank=BranchBank(tuple(branches)),
            fallback_reason=None,
        )

    def to_dict(self) -> dict[str, Any]:
        """返回可审计重放的候选、谱诊断、白化和完整 branch bank。

        参数：
            无。
        返回：
            可直接 JSON 编码的完整候选新字典。
        异常：
            无。
        副作用：
            无。
        """

        return {
            "candidate_id": self.candidate_id,
            "stage": self.stage.value,
            "mode": self.mode.value,
            "whitening": self.whitening.to_dict(),
            "singular_value_threshold": self.singular_value_threshold,
            "gram_threshold": self.gram_threshold,
            "minimum_projector_gap": self.minimum_projector_gap,
            "spectral_cluster_tolerance": self.spectral_cluster_tolerance,
            "branch_scale_floor": self.branch_scale_floor,
            "gram_eigenvalues": list(self.gram_eigenvalues),
            "projector_gap": self.projector_gap,
            "selected_rank": self.selected_rank,
            "retained_rank": self.retained_rank,
            "retained_white_basis": _matrix_list(self.retained_white_basis),
            "common_operator": _matrix_list(self.common_operator),
            "anchor_response": _matrix_list(self.anchor_response),
            "anchor_covariance_sqrt": _matrix_list(self.anchor_covariance_sqrt),
            "branch_bank": self.branch_bank.to_dict(),
            "fallback_reason": self.fallback_reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PostFilterCandidate":
        """从严格 JSON 映射恢复候选并重新执行谱/shape/fallback 不变量。

        参数：
            value: ``to_dict`` 产生并可能落盘后的 JSON 对象。
        返回：
            重算白化关系、Gram 谱、``Q_w``、``L_0``、分支半径和种类的候选。
        异常：
            字段、类型或任何派生几何不一致时抛出 ``TypeError``/``ValueError``。
        副作用：
            无；不读取文件或重新拟合数据。
        """

        _require_exact_keys(
            value,
            {
                "candidate_id",
                "stage",
                "mode",
                "whitening",
                "singular_value_threshold",
                "gram_threshold",
                "minimum_projector_gap",
                "spectral_cluster_tolerance",
                "branch_scale_floor",
                "gram_eigenvalues",
                "projector_gap",
                "selected_rank",
                "retained_rank",
                "retained_white_basis",
                "common_operator",
                "anchor_response",
                "anchor_covariance_sqrt",
                "branch_bank",
                "fallback_reason",
            },
            name="PostFilterCandidate",
        )
        fallback_value = value["fallback_reason"]
        return cls(
            candidate_id=_strict_string(value["candidate_id"], name="candidate_id"),
            stage=MonitorStage.parse(value["stage"]),
            mode=SpectralMode(value["mode"]),
            whitening=WhiteningEstimate.from_dict(
                _strict_mapping(value["whitening"], name="whitening")
            ),
            singular_value_threshold=_strict_float(
                value["singular_value_threshold"],
                name="singular_value_threshold",
            ),
            gram_threshold=_strict_float(value["gram_threshold"], name="gram_threshold"),
            minimum_projector_gap=_strict_float(
                value["minimum_projector_gap"],
                name="minimum_projector_gap",
            ),
            spectral_cluster_tolerance=_strict_float(
                value["spectral_cluster_tolerance"],
                name="spectral_cluster_tolerance",
            ),
            branch_scale_floor=_strict_float(
                value["branch_scale_floor"],
                name="branch_scale_floor",
            ),
            gram_eigenvalues=_float_tuple(
                value["gram_eigenvalues"],
                name="gram_eigenvalues",
            ),
            projector_gap=_strict_float(value["projector_gap"], name="projector_gap"),
            selected_rank=_strict_int(value["selected_rank"], name="selected_rank"),
            retained_rank=_strict_int(value["retained_rank"], name="retained_rank"),
            retained_white_basis=_coerce_matrix_tuple(
                value["retained_white_basis"],
                name="retained_white_basis",
            ),
            common_operator=_coerce_matrix_tuple(
                value["common_operator"],
                name="common_operator",
                allow_zero_rows=True,
            ),
            anchor_response=_coerce_matrix_tuple(
                value["anchor_response"],
                name="anchor_response",
            ),
            anchor_covariance_sqrt=_coerce_matrix_tuple(
                value["anchor_covariance_sqrt"],
                name="anchor_covariance_sqrt",
            ),
            branch_bank=BranchBank.from_dict(
                _strict_mapping(value["branch_bank"], name="branch_bank")
            ),
            fallback_reason=None
            if fallback_value is None
            else _strict_string(fallback_value, name="fallback_reason"),
        )

    @property
    def content_hash(self) -> str:
        """返回完整候选 JSON 的确定性 SHA-256，供访问账本和产物重放使用。

        参数：
            无。
        返回：
            64 位小写十六进制 SHA-256。
        异常：
            无；对象已经拒绝 NaN/Inf。
        副作用：
            无。
        """

        encoded = json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()


def _prepare_anchor_spectrum(
    *,
    w_0: np.ndarray,
    anchor_response: np.ndarray,
    anchor_covariance_sqrt: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """准备锚点谱输入，并在乘积或 Gram 谱不可表示时返回确定性停止原因。

    ``G_0 R_eta^(1/2)`` 仍可能在所有输入元素有限时发生乘法溢出。即使该乘积有限，
    其最大奇异值的平方也可能超过 float64，令 Gram 矩阵无法作为可追溯谱保存。因此
    这里先用缩放后的 SVD 检查可表示性；调用方在失败时只能部署 raw guard。
    """

    with np.errstate(over="ignore", invalid="ignore"):
        anchor_effect = anchor_response @ anchor_covariance_sqrt
    if not np.isfinite(anchor_effect).all():
        return (
            None,
            None,
            "Anchor spectrum construction exceeds float64 range; quotient is disabled.",
        )
    with np.errstate(over="ignore", invalid="ignore"):
        white_anchor_effect = w_0 @ anchor_effect
    if not np.isfinite(white_anchor_effect).all():
        return (
            anchor_effect,
            None,
            "Whitened anchor spectrum exceeds float64 range; quotient is disabled.",
        )

    scale = float(np.max(np.abs(white_anchor_effect), initial=0.0))
    if scale == 0.0:
        largest_singular_value = 0.0
    else:
        try:
            scaled_singular_values = np.linalg.svd(
                white_anchor_effect / scale,
                compute_uv=False,
            )
        except np.linalg.LinAlgError:
            return (
                anchor_effect,
                white_anchor_effect,
                "Anchor spectrum decomposition failed; quotient is disabled.",
            )
        largest_singular_value = float(scaled_singular_values[0] * scale)
    if (
        not np.isfinite(largest_singular_value)
        or largest_singular_value > np.sqrt(np.finfo(np.float64).max)
    ):
        return (
            anchor_effect,
            white_anchor_effect,
            "Anchor Gram spectrum is not representable in float64; quotient is disabled.",
        )
    return anchor_effect, white_anchor_effect, None


def _scaled_spectral_norm(matrix: np.ndarray) -> float:
    """用幅值缩放计算二维矩阵谱范数，避免 ``A.T @ A`` 式平方溢出。"""

    if matrix.size == 0:
        return 0.0
    scale = float(np.max(np.abs(matrix), initial=0.0))
    if scale == 0.0:
        return 0.0
    singular_values = np.linalg.svd(matrix / scale, compute_uv=False)
    value = float(singular_values[0] * scale)
    if not np.isfinite(value):
        return float(np.finfo(np.float64).max)
    return value


def _propagated_spectral_radius(
    matrix: np.ndarray,
    anchor_effect: np.ndarray | None,
) -> float:
    """保守计算 ``||L_b G_0 R_eta^(1/2)||_2``，溢出时饱和到最大有限值。"""

    if anchor_effect is None:
        return float(np.finfo(np.float64).max)
    with np.errstate(over="ignore", invalid="ignore"):
        propagated = matrix @ anchor_effect
    if not np.isfinite(propagated).all():
        return float(np.finfo(np.float64).max)
    return _scaled_spectral_norm(propagated)


def _branch(
    *,
    name: str,
    kind: BranchKind,
    matrix: np.ndarray,
    anchor_effect: np.ndarray | None,
    scale_floor: float,
    force_enabled: bool = False,
) -> BranchOperator:
    """从同一个部署矩阵计算支路尺度和锚点半径。"""

    scale = _scaled_spectral_norm(matrix)
    enabled = force_enabled or scale >= scale_floor
    reason = None if enabled else f"Branch scale {scale:.17g} is below floor {scale_floor:.17g}."
    anchor_radius = (
        _propagated_spectral_radius(matrix, anchor_effect)
        if matrix.shape[0] > 0
        else 0.0
    )
    return BranchOperator(
        name=name,
        kind=kind,
        input_dim=int(matrix.shape[1]),
        matrix=_matrix_tuple(matrix),
        anchor_radius=anchor_radius,
        enabled=enabled,
        disabled_reason=reason,
    )


def _stable_cluster_selection(
    eigenvalues: np.ndarray,
    *,
    threshold: float,
    cluster_tolerance: float,
    collision_tolerance: float,
) -> tuple[np.ndarray, int]:
    """保证近重复 Gram 特征值整簇决策，并保守保留碰撞簇。

    阈值与某个簇的距离不大于冻结碰撞容差时，该簇全部进入 retained white space；
    这使从阈值两侧逼近的浮点扰动得到同一个商空间。调用方把候选标成 ``HYBRID``，
    后续仍须作为独立模式共同校准。
    """

    result = np.zeros(eigenvalues.shape, dtype=bool)
    collision_rank = 0
    start = 0
    while start < eigenvalues.size:
        end = start + 1
        while (
            end < eigenvalues.size
            and abs(float(eigenvalues[end - 1] - eigenvalues[end]))
            <= cluster_tolerance
        ):
            end += 1
        cluster = eigenvalues[start:end]
        cluster_collides = bool(
            np.min(np.abs(cluster - threshold)) <= collision_tolerance
            or (float(cluster[-1]) <= threshold <= float(cluster[0]))
        )
        if cluster_collides:
            collision_rank += end - start
            result[start:end] = False
        else:
            result[start:end] = bool(np.min(cluster) > threshold)
        start = end
    return result, collision_rank


def _finite_matrix(
    value: Sequence[Sequence[float]] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """复制并验证一个非空二维有限 float64 矩阵。"""

    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values.")
    return matrix.copy()


def _matrix_tuple(matrix: np.ndarray) -> MatrixTuple:
    """把二维数组复制为不可变、JSON 兼容的行元组。"""

    return tuple(tuple(float(value) for value in row) for row in matrix)


def _matrix_list(matrix: MatrixTuple) -> list[list[float]]:
    """把不可变矩阵复制为 JSON 行列表。"""

    return [list(row) for row in matrix]


def _coerce_matrix_tuple(
    value: Any,
    *,
    name: str,
    allow_zero_rows: bool = False,
) -> MatrixTuple:
    """从 JSON 行列表恢复有限矩阵；空行数只用于零维输出算子。"""

    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON list of rows.")
    if not value:
        if allow_zero_rows:
            return ()
        raise ValueError(f"{name} must contain at least one row.")
    rows: list[tuple[float, ...]] = []
    width: int | None = None
    for row in value:
        if not isinstance(row, list):
            raise TypeError(f"{name} rows must be JSON lists.")
        converted = tuple(_strict_float(item, name=f"{name} value") for item in row)
        if width is None:
            width = len(converted)
        elif len(converted) != width:
            raise ValueError(f"{name} rows must have a constant width.")
        rows.append(converted)
    return tuple(rows)


def _validate_matrix_tuple(
    value: MatrixTuple,
    *,
    name: str,
    rows: int | None = None,
    columns: int | None = None,
    allow_zero_rows: bool = False,
) -> None:
    """校验不可变矩阵的行列数、定宽和有限性。"""

    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple of row tuples.")
    if not value and not allow_zero_rows:
        raise ValueError(f"{name} must contain at least one row.")
    if rows is not None and len(value) != rows:
        raise ValueError(f"{name} must have {rows} rows.")
    observed_columns = columns
    if observed_columns is None and value:
        observed_columns = len(value[0])
    for row in value:
        if not isinstance(row, tuple):
            raise TypeError(f"{name} rows must be tuples.")
        if observed_columns is not None and len(row) != observed_columns:
            raise ValueError(f"{name} rows must have {observed_columns} columns.")
        if not _all_finite(row):
            raise ValueError(f"{name} must contain only finite values.")


def _all_finite(values: Sequence[float]) -> bool:
    """返回一维数值序列是否全部有限且不含 bool。"""

    return all(
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, float, np.integer, np.floating))
        and np.isfinite(value)
        for value in values
    )


def _strict_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    """要求 JSON 对象映射，拒绝数组和自定义标量。"""

    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object.")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    name: str,
) -> None:
    """拒绝缺失和额外字段，保持论文产物 schema 严格。"""

    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} keys mismatch: missing={missing}, extra={extra}.")


def _strict_string(value: Any, *, name: str) -> str:
    """要求非空字符串。"""

    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string.")
    return value


def _strict_bool(value: Any, *, name: str) -> bool:
    """要求真正的 JSON bool，拒绝整数 0/1。"""

    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool.")
    return value


def _strict_int(value: Any, *, name: str) -> int:
    """要求真正的整数，拒绝 bool 和浮点伪整数。"""

    if type(value) is not int:
        raise TypeError(f"{name} must be an integer.")
    return value


def _strict_float(value: Any, *, name: str) -> float:
    """把 JSON int/float 规范为有限 float，同时拒绝 bool。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number.")
    converted = float(value)
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be finite.")
    return converted


def _float_tuple(value: Any, *, name: str) -> tuple[float, ...]:
    """从 JSON 数值列表恢复有限浮点元组。"""

    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON list.")
    return tuple(_strict_float(item, name=f"{name} value") for item in value)


def _scaled_column_mean(matrix: np.ndarray) -> tuple[float, ...]:
    """用逐列缩放避免极大有限输入在 fallback 均值审计中再次溢出。"""

    scales = np.max(np.abs(matrix), axis=0)
    normalized = np.divide(
        matrix,
        scales,
        out=np.zeros_like(matrix),
        where=scales > 0.0,
    )
    means = normalized.mean(axis=0) * scales
    if not np.isfinite(means).all():
        raise ValueError("Estimate residual mean is non-finite after scaled accumulation.")
    return tuple(float(value) for value in means)


def _is_sha256(value: str) -> bool:
    """检查小写或大写十六进制 SHA-256 形状。"""

    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _array_hash(matrix: np.ndarray) -> str:
    """计算包含 shape 和 float64 字节序列的确定性 SHA-256。"""

    digest = hashlib.sha256()
    digest.update(str(tuple(int(value) for value in matrix.shape)).encode("ascii"))
    digest.update(np.ascontiguousarray(matrix, dtype="<f8").tobytes())
    return digest.hexdigest()
