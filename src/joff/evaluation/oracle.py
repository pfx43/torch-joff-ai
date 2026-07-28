"""P9 explanation outer family 的线性 oracle 公开接口。

文件用途：
    为集合值结构化隔离提供可手算、可审计的线性 outer-family 特例，使调用方能够把
    完整部署观测映射为 explanation score，并按右闭半径判断某个解释是否仍然可行。
主要职责：
    定义冻结线性 cell、精确 oracle 结果、单调 cell-refinement 缓存和线性
    ``OuterExplanationOracle``；本文件不拟合 normal attribution 分位、不运行 full
    nonlinear 全局优化，也不输出最终类别。
关键输入与输出：
    输入为 ``DeployedObservation.linear_features``、匹配的 feature schema、cell 中心、
    正尺度和 explanation family；输出为规范化 ``L-infinity`` 距离及可行/认证标志。
依赖与副作用：
    只依赖 Python 标准库和 P9 领域对象；不读取文件或网络，不使用随机数，不修改
    monitor、模型或全局状态。``MonotoneRefinementCache.record`` 会在内存中保存
    已认证 cell bounds，生命周期由调用方显式持有。
重要约束：
    线性 cell 必须绑定 explanation 与 feature schema 的 SHA-256 身份；边界采用
    ``score <= radius`` 的右闭语义；当前线性入口只接受完整匹配的精确 cell，不能被
    解释为 full nonlinear oracle 已获认证。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Sequence

from .explanations import DeployedObservation, ExplanationFamily


@dataclass(frozen=True)
class LinearExplanationCell:
    """一个冻结 explanation family 的线性 outer cell。

    参数：
        cell_id: cell 在冻结 refinement hierarchy 中的稳定身份。
        family_hash: 目标 ``ExplanationFamily.content_hash``。
        feature_schema_hash: 该 cell 允许读取的部署特征模式身份。
        center/scale: 等长向量；score 为各坐标 ``abs(x-center)/scale`` 的最大值。
        coverage_evidence_hash: 生成该 cell 的 signature/nuisance coverage 证据身份。
    返回：
        不可变的精确线性 cell，可由 ``OuterExplanationOracle.linear`` 组合。
    异常：
        名称、哈希、形状、有限性或正尺度非法时抛出 ``TypeError``/``ValueError``。
    副作用：
        无；构造不会读取训练、校准或故障数据。
    """

    cell_id: str
    family_hash: str
    feature_schema_hash: str
    center: tuple[float, ...]
    scale: tuple[float, ...]
    coverage_evidence_hash: str

    def __post_init__(self) -> None:
        """验证 cell 身份、特征形状和数值尺度。"""

        if not isinstance(self.cell_id, str) or not self.cell_id.strip():
            raise ValueError("Linear explanation cell id must be non-empty.")
        if not all(
            _is_sha256(value)
            for value in (
                self.family_hash,
                self.feature_schema_hash,
                self.coverage_evidence_hash,
            )
        ):
            raise ValueError("Linear explanation cell identities must be SHA-256.")
        if (
            not isinstance(self.center, tuple)
            or not self.center
            or not isinstance(self.scale, tuple)
            or len(self.center) != len(self.scale)
        ):
            raise ValueError("Linear explanation cell center and scale must align.")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in self.center
        ):
            raise ValueError("Linear explanation cell center must be finite.")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in self.scale
        ):
            raise ValueError("Linear explanation cell scale must be finite and positive.")
        object.__setattr__(self, "center", tuple(float(value) for value in self.center))
        object.__setattr__(self, "scale", tuple(float(value) for value in self.scale))

    def score(self, features: Sequence[float]) -> float:
        """计算观测到该 cell 的精确规范化 ``L-infinity`` 距离。

        参数：
            features: 与冻结 feature schema 对齐的一维有限特征序列。
        返回：
            非负有限 score；中心点返回零。
        异常：
            特征维数或数值非法时抛出 ``TypeError``/``ValueError``。
        副作用：
            无。
        """

        if isinstance(features, (str, bytes)) or not isinstance(features, Sequence):
            raise TypeError("Linear explanation features must be a sequence.")
        if len(features) != len(self.center):
            raise ValueError("Linear explanation feature dimension does not match its cell.")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in features
        ):
            raise ValueError("Linear explanation features must be finite.")
        return max(
            abs(float(value) - center) / scale
            for value, center, scale in zip(
                features,
                self.center,
                self.scale,
                strict=True,
            )
        )

    def to_dict(self) -> dict[str, object]:
        """返回 cell 身份、几何参数与 coverage 证据。

        参数：
            无。
        返回：
            只含 JSON 基本类型的稳定字典。
        异常：
            无。
        副作用：
            无。
        """

        return {
            "cell_id": self.cell_id,
            "family_hash": self.family_hash,
            "feature_schema_hash": self.feature_schema_hash,
            "center": list(self.center),
            "scale": list(self.scale),
            "coverage_evidence_hash": self.coverage_evidence_hash,
        }


@dataclass(frozen=True)
class OracleEvaluation:
    """一次 certified 线性或显式未认证 outer-family 查询结果。

    参数：
        oracle_hash/family_id/family_hash/observation_hash: 冻结 oracle、查询 explanation 与
            完整观测身份。
        radius: 本次查询的部署半径，允许 ``+infinity``。
        outer_score: 所有匹配 cell 精确 score 的最小值。
        feasible: 是否满足右闭条件 ``outer_score <= radius``。
        certified: 当前结果是否来自本文件支持的认证 outer 计算。
        reason: 未认证原因；认证结果必须为空，拒识结果必须明确说明缺失项。
        unresolved_cell_ids: 当前半径处下界已进入、上界尚未进入的保守未决 cell。
    返回：
        不可变、可交叉核验的 oracle 结果。
    异常：
        身份、数值或派生可行性不一致时抛出 ``TypeError``/``ValueError``。
    副作用：
        无。
    """

    oracle_hash: str
    family_id: str
    family_hash: str
    observation_hash: str
    radius: float
    outer_score: float
    feasible: bool
    certified: bool
    reason: str | None
    unresolved_cell_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """拒绝被调用方伪造的边界判定或认证状态。"""

        if not isinstance(self.family_id, str) or not self.family_id.strip():
            raise ValueError("Oracle evaluation family id must be non-empty.")
        if not all(
            _is_sha256(value)
            for value in (
                self.oracle_hash,
                self.family_hash,
                self.observation_hash,
            )
        ):
            raise ValueError("Oracle evaluation identities must be SHA-256.")
        normalized_radius = _nonnegative_finite_or_infinity(
            self.radius,
            name="Oracle evaluation radius",
        )
        normalized_score = _nonnegative_finite_or_infinity(
            self.outer_score,
            name="Oracle evaluation outer_score",
        )
        object.__setattr__(self, "radius", normalized_radius)
        object.__setattr__(self, "outer_score", normalized_score)
        if type(self.feasible) is not bool or self.feasible is not (
            normalized_score <= normalized_radius
        ):
            raise ValueError("Oracle feasibility must use right-closed score <= radius.")
        if (
            not isinstance(self.unresolved_cell_ids, tuple)
            or any(
                not isinstance(cell_id, str) or not cell_id.strip()
                for cell_id in self.unresolved_cell_ids
            )
            or len(set(self.unresolved_cell_ids)) != len(self.unresolved_cell_ids)
            or self.unresolved_cell_ids != tuple(sorted(self.unresolved_cell_ids))
        ):
            raise ValueError("Oracle unresolved cell ids must be unique and sorted.")
        if type(self.certified) is not bool:
            raise TypeError("Oracle certified flag must be a strict boolean.")
        if self.certified:
            if self.reason is not None:
                raise ValueError("Certified oracle evaluation cannot carry a failure reason.")
        elif (
            not self.feasible
            or not isinstance(self.reason, str)
            or not self.reason.strip()
            or not self.unresolved_cell_ids
        ):
            raise ValueError(
                "Uncertified oracle evaluation must retain an unresolved family with a reason."
            )


@dataclass(frozen=True)
class OracleCellRefinement:
    """一个观测特定 cell 的认证 score 区间。

    参数：
        oracle_hash/family_hash/observation_hash: 冻结 oracle、解释和完整观测身份。
        cell_id/refinement_level: hierarchy 中的稳定 cell 与严格递增细化层级。
        lower_score/upper_score: 认证 score 下、上界；允许正无穷且必须满足下界不大于上界。
        coverage_evidence_hash: 产生 bounds 的 signature/nuisance coverage 证据身份。
    返回：
        可写入 ``MonotoneRefinementCache`` 的不可变 refinement。
    异常：
        身份、层级、数值或区间非法时抛出 ``TypeError``/``ValueError``。
    副作用：
        无；对象本身不执行 refinement。
    """

    oracle_hash: str
    family_hash: str
    observation_hash: str
    cell_id: str
    refinement_level: int
    lower_score: float
    upper_score: float
    coverage_evidence_hash: str

    def __post_init__(self) -> None:
        """验证缓存键、层级和认证 score 区间。"""

        if not all(
            _is_sha256(value)
            for value in (
                self.oracle_hash,
                self.family_hash,
                self.observation_hash,
                self.coverage_evidence_hash,
            )
        ):
            raise ValueError("Oracle cell refinement identities must be SHA-256.")
        if not isinstance(self.cell_id, str) or not self.cell_id.strip():
            raise ValueError("Oracle cell refinement id must be non-empty.")
        if type(self.refinement_level) is not int or self.refinement_level < 0:
            raise ValueError("Oracle refinement level must be a non-negative integer.")
        normalized_lower = _nonnegative_finite_or_infinity(
            self.lower_score,
            name="Oracle refinement lower_score",
        )
        normalized_upper = _nonnegative_finite_or_infinity(
            self.upper_score,
            name="Oracle refinement upper_score",
        )
        if normalized_lower > normalized_upper:
            raise ValueError("Oracle refinement lower_score cannot exceed upper_score.")
        object.__setattr__(self, "lower_score", normalized_lower)
        object.__setattr__(self, "upper_score", normalized_upper)


class MonotoneRefinementCache:
    """跨查询半径共享 cell hierarchy 的单调内存缓存。

    参数：
        无。
    返回：
        空缓存；``record`` 收紧 cell bounds，``evaluate`` 生成保守 outer 查询。
    异常：
        后续层级未递增、下界下降或上界上升时抛出 ``ValueError``，防止较大预算反而
        放宽旧证据或破坏跨半径嵌套。
    副作用：
        ``record`` 修改本对象的内存字典；不写文件，不访问网络，也不修改观测对象。
    """

    def __init__(self) -> None:
        """创建不含任何观测或 cell 的显式生命周期缓存。"""

        self._entries: dict[
            tuple[str, str, str, str],
            OracleCellRefinement,
        ] = {}

    def record(self, refinement: OracleCellRefinement) -> None:
        """记录首次 bounds 或严格更深且单调收紧的 refinement。

        参数：
            refinement: 已绑定 oracle/family/observation/cell 的认证 score 区间。
        返回：
            无。
        异常：
            类型错误、层级未严格递增或区间相对旧记录放宽时抛出
            ``TypeError``/``ValueError``。
        副作用：
            成功时替换该缓存键的内存记录。
        """

        if not isinstance(refinement, OracleCellRefinement):
            raise TypeError("Monotone cache accepts OracleCellRefinement values.")
        key = (
            refinement.oracle_hash,
            refinement.family_hash,
            refinement.observation_hash,
            refinement.cell_id,
        )
        previous = self._entries.get(key)
        if previous is not None:
            if refinement.refinement_level <= previous.refinement_level:
                raise ValueError(
                    "Oracle refinement level must strictly increase in the monotone cache."
                )
            if (
                refinement.lower_score < previous.lower_score
                or refinement.upper_score > previous.upper_score
            ):
                raise ValueError(
                    "Oracle refinement must preserve monotone nested score bounds."
                )
        self._entries[key] = refinement

    def evaluate(
        self,
        *,
        oracle_hash: str,
        family: ExplanationFamily,
        observation: DeployedObservation,
        radius: float,
    ) -> OracleEvaluation:
        """用当前所有 cell 下界执行保守且右闭的 outer-family 查询。

        参数：
            oracle_hash: 生成这些 refinement 的冻结 oracle 身份。
            family/observation: 待查询 explanation 与同一完整部署观测。
            radius: 当前查询半径，允许 ``+infinity``。
        返回：
            ``outer_score=min(lower_score)``；任何下界不大于半径的 cell 都保留。
            对满足 ``lower <= radius < upper`` 的 cell 显式列入 ``unresolved_cell_ids``。
        异常：
            对象、身份、半径非法或缓存中没有匹配 cell 时抛出
            ``TypeError``/``ValueError``。
        副作用：
            无；查询不改变已缓存 hierarchy。
        """

        if not _is_sha256(oracle_hash):
            raise ValueError("Monotone cache oracle_hash must be SHA-256.")
        if not isinstance(family, ExplanationFamily):
            raise TypeError("Monotone cache requires an ExplanationFamily.")
        if not isinstance(observation, DeployedObservation):
            raise TypeError("Monotone cache requires a DeployedObservation.")
        normalized_radius = _nonnegative_finite_or_infinity(
            radius,
            name="Monotone cache radius",
        )
        refinements = tuple(
            refinement
            for key, refinement in self._entries.items()
            if key[:3]
            == (
                oracle_hash,
                family.content_hash,
                observation.content_hash,
            )
        )
        if not refinements:
            raise ValueError("Monotone cache has no cell for this oracle query.")
        outer_score = min(item.lower_score for item in refinements)
        unresolved_cell_ids = tuple(
            sorted(
                item.cell_id
                for item in refinements
                if item.lower_score <= normalized_radius < item.upper_score
            )
        )
        return OracleEvaluation(
            oracle_hash=oracle_hash,
            family_id=family.family_id,
            family_hash=family.content_hash,
            observation_hash=observation.content_hash,
            radius=normalized_radius,
            outer_score=outer_score,
            feasible=outer_score <= normalized_radius,
            certified=True,
            reason=None,
            unresolved_cell_ids=unresolved_cell_ids,
        )


@dataclass(frozen=True)
class _UncertifiedOracleFamily:
    """full nonlinear 后端尚未认证时的保守 family 声明。

    参数：
        family_hash/feature_schema_hash: 被保留 explanation 与允许观测特征模式的身份。
        reason: signature、nuisance、operator 或后端认证缺失的明确原因。
    返回：
        只能由 ``OuterExplanationOracle.uncertified`` 持有的不可变拒识声明。
    异常：
        哈希或原因非法时抛出 ``ValueError``。
    副作用：
        无。
    """

    family_hash: str
    feature_schema_hash: str
    reason: str

    def __post_init__(self) -> None:
        """验证拒识声明仍绑定稳定 family/schema 身份和明确原因。

        参数：
            无。
        返回：
            无。
        异常：
            family/schema 不是 SHA-256 或 reason 为空时抛出 ``ValueError``。
        副作用：
            无；只规范化构造不变量。
        """

        if not _is_sha256(self.family_hash) or not _is_sha256(
            self.feature_schema_hash
        ):
            raise ValueError("Uncertified oracle family identities must be SHA-256.")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("Uncertified oracle family requires an explicit reason.")

    def to_dict(self) -> dict[str, object]:
        """返回拒识 family 的稳定 JSON 字段。

        参数：
            无。
        返回：
            包含 family/schema 身份和拒识原因的 JSON 兼容字典。
        异常：
            无。
        副作用：
            无。
        """

        return {
            "family_hash": self.family_hash,
            "feature_schema_hash": self.feature_schema_hash,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class OuterExplanationOracle:
    """组合精确线性 cell 与显式未认证 family 的 explanation outer oracle。

    参数：
        cells: cell_id 唯一的冻结线性 cell；纯未认证路径允许为空。
        uncertified_families: full nonlinear 后端缺失时保守保留的 family 声明。
    返回：
        由 ``linear``、``uncertified`` 或 ``mixed`` 构造并通过 ``evaluate`` 查询的无状态
        oracle；至少包含一种 family 声明。
    异常：
        cell 容器、身份或查询绑定不一致时抛出 ``TypeError``/``ValueError``。
    副作用：
        无；同一完整观测和冻结 cell 总是得到相同结果。
    """

    cells: tuple[LinearExplanationCell, ...]
    uncertified_families: tuple[_UncertifiedOracleFamily, ...] = ()

    def __post_init__(self) -> None:
        """验证 cell family 非空、类型正确且身份唯一。"""

        if not isinstance(self.cells, tuple) or not all(
            isinstance(cell, LinearExplanationCell) for cell in self.cells
        ):
            raise TypeError("Outer explanation oracle linear cells must be a tuple.")
        if not isinstance(self.uncertified_families, tuple) or not all(
            isinstance(family, _UncertifiedOracleFamily)
            for family in self.uncertified_families
        ):
            raise TypeError("Outer explanation oracle uncertified families must be a tuple.")
        if not self.cells and not self.uncertified_families:
            raise ValueError("Outer explanation oracle requires at least one family.")
        if len({cell.cell_id for cell in self.cells}) != len(self.cells):
            raise ValueError("Outer explanation oracle cell ids must be unique.")
        uncertified_hashes = tuple(
            family.family_hash for family in self.uncertified_families
        )
        if len(set(uncertified_hashes)) != len(uncertified_hashes):
            raise ValueError("Uncertified outer oracle family identities must be unique.")
        if {cell.family_hash for cell in self.cells} & set(uncertified_hashes):
            raise ValueError(
                "One outer oracle family cannot be both linear-certified and uncertified."
            )

    @classmethod
    def linear(
        cls,
        *,
        cells: Sequence[LinearExplanationCell],
    ) -> "OuterExplanationOracle":
        """冻结一组精确线性 cell。

        参数：
            cells: 可迭代的 ``LinearExplanationCell``；顺序会按 cell_id 固定。
        返回：
            可重复查询的线性 ``OuterExplanationOracle``。
        异常：
            字符串等非法容器或非法 cell 由本方法/构造器拒绝。
        副作用：
            无。
        """

        if isinstance(cells, (str, bytes)) or not isinstance(cells, Sequence):
            raise TypeError("Linear outer oracle cells must be a sequence.")
        return cls(cells=tuple(sorted(cells, key=lambda cell: cell.cell_id)))

    @classmethod
    def uncertified(
        cls,
        *,
        family_hashes: Sequence[str],
        feature_schema_hash: str,
        reason: str,
    ) -> "OuterExplanationOracle":
        """为尚无认证后端的 full nonlinear family 创建保守拒识 oracle。

        参数：
            family_hashes: 必须保留的 explanation family 内容身份。
            feature_schema_hash: 这些声明允许接收的完整观测特征模式身份。
            reason: 后端、signature、nuisance 或 certification 缺失的明确原因。
        返回：
            查询时始终保留匹配 family 且返回 ``certified=False`` 的 oracle。
        异常：
            容器、哈希、重复 family 或原因非法时抛出 ``TypeError``/``ValueError``。
        副作用：
            无。
        """

        if isinstance(family_hashes, (str, bytes)) or not isinstance(
            family_hashes,
            Sequence,
        ):
            raise TypeError("Uncertified oracle family hashes must be a sequence.")
        normalized_hashes = tuple(sorted(family_hashes))
        if (
            not normalized_hashes
            or len(set(normalized_hashes)) != len(normalized_hashes)
            or not all(_is_sha256(value) for value in normalized_hashes)
        ):
            raise ValueError(
                "Uncertified oracle family hashes must be non-empty and unique SHA-256."
            )
        return cls(
            cells=(),
            uncertified_families=tuple(
                _UncertifiedOracleFamily(
                    family_hash=family_hash,
                    feature_schema_hash=feature_schema_hash,
                    reason=reason,
                )
                for family_hash in normalized_hashes
            ),
        )

    @classmethod
    def mixed(
        cls,
        *,
        cells: Sequence[LinearExplanationCell],
        uncertified_family_hashes: Sequence[str],
        feature_schema_hash: str,
        reason: str,
    ) -> "OuterExplanationOracle":
        """组合 certified 线性 family 与显式未认证 full nonlinear family。

        参数：
            cells: normal 或已认证线性 fault family 的精确 cell。
            uncertified_family_hashes: 必须保守保留、但当前没有认证后端的 family 身份。
            feature_schema_hash/reason: 未认证 family 允许的观测模式与统一拒识原因。
        返回：
            ``content_hash`` 和 ``family_hashes`` 同时覆盖两类 family 的冻结 oracle。
        异常：
            任一子构造参数非法、family 重复或同一 family 同时声明为认证/未认证时抛出
            ``TypeError``/``ValueError``。
        副作用：
            无；组合不会运行 refinement、mask 或模型。
        """

        linear_oracle = cls.linear(cells=cells)
        uncertified_oracle = cls.uncertified(
            family_hashes=uncertified_family_hashes,
            feature_schema_hash=feature_schema_hash,
            reason=reason,
        )
        return cls(
            cells=linear_oracle.cells,
            uncertified_families=uncertified_oracle.uncertified_families,
        )

    def evaluate(
        self,
        observation: DeployedObservation,
        family: ExplanationFamily,
        *,
        radius: float | None = None,
    ) -> OracleEvaluation:
        """计算一个 explanation family 对完整部署观测的线性 outer score。

        参数：
            observation: 同时包含 monitor、raw window、branch/operator 与 mask 的完整观测。
            family: 待查询 explanation；cell 必须绑定其完整内容哈希。
            radius: 可选查询半径；省略时使用 family 的冻结部署半径。
        返回：
            线性 cell 返回精确 score、右闭可行性和 ``certified=True``；未认证 family
            保守返回可行、``certified=False`` 及明确拒识原因。
        异常：
            对象类型、family/schema 绑定、维数或半径非法时抛出
            ``TypeError``/``ValueError``。
        副作用：
            无；不会更新 refinement cache 或执行 mask。
        """

        if not isinstance(observation, DeployedObservation):
            raise TypeError("Outer oracle requires a DeployedObservation.")
        if not isinstance(family, ExplanationFamily):
            raise TypeError("Outer oracle requires an ExplanationFamily.")
        query_radius = family.radius if radius is None else radius
        normalized_radius = _nonnegative_finite_or_infinity(
            query_radius,
            name="Outer oracle radius",
        )
        matching_cells = tuple(
            cell for cell in self.cells if cell.family_hash == family.content_hash
        )
        if matching_cells:
            if any(
                cell.feature_schema_hash != observation.feature_schema_hash
                for cell in matching_cells
            ):
                raise ValueError(
                    "Outer oracle cell feature schema does not match the observation."
                )
            outer_score = min(
                cell.score(observation.linear_features) for cell in matching_cells
            )
            return OracleEvaluation(
                oracle_hash=self.content_hash,
                family_id=family.family_id,
                family_hash=family.content_hash,
                observation_hash=observation.content_hash,
                radius=normalized_radius,
                outer_score=outer_score,
                feasible=outer_score <= normalized_radius,
                certified=True,
                reason=None,
            )
        matching_uncertified = tuple(
            item
            for item in self.uncertified_families
            if item.family_hash == family.content_hash
        )
        if not matching_uncertified:
            raise ValueError("Outer oracle has no cell for the requested explanation family.")
        (uncertified_family,) = matching_uncertified
        if uncertified_family.feature_schema_hash != observation.feature_schema_hash:
            raise ValueError(
                "Uncertified outer oracle feature schema does not match the observation."
            )
        return OracleEvaluation(
            oracle_hash=self.content_hash,
            family_id=family.family_id,
            family_hash=family.content_hash,
            observation_hash=observation.content_hash,
            radius=normalized_radius,
            outer_score=0.0,
            feasible=True,
            certified=False,
            reason=uncertified_family.reason,
            unresolved_cell_ids=(f"uncertified:{family.family_id}",),
        )

    @property
    def content_hash(self) -> str:
        """返回覆盖全部冻结 cell 的 SHA-256。

        参数：
            无。
        返回：
            64 位小写十六进制摘要。
        异常：
            无。
        副作用：
            无。
        """

        payload = json.dumps(
            {
                "linear_cells": [cell.to_dict() for cell in self.cells],
                "uncertified_families": [
                    family.to_dict() for family in self.uncertified_families
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def family_hashes(self) -> tuple[str, ...]:
        """返回 oracle 冻结的完整 explanation family 内容身份。

        参数：
            无。
        返回：
            合并精确线性 cell 与未认证声明后去重、排序的 SHA-256 元组。
        异常：
            无。
        副作用：
            无；该集合已包含在 ``content_hash`` 中。
        """

        return tuple(
            sorted(
                {
                    *(cell.family_hash for cell in self.cells),
                    *(
                        family.family_hash
                        for family in self.uncertified_families
                    ),
                }
            )
        )


def _nonnegative_finite_or_infinity(value: object, *, name: str) -> float:
    """把数值规范化为非负有限值或正无穷。

    参数：
        value: 待校验对象。
        name: 错误消息中的字段名。
    返回：
        Python ``float``。
    异常：
        bool、非数值、负值、NaN 或负无穷时抛出 ``TypeError``/``ValueError``。
    副作用：
        无。
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    normalized = float(value)
    if normalized < 0.0 or math.isnan(normalized) or normalized == float("-inf"):
        raise ValueError(f"{name} must be non-negative and not NaN.")
    return normalized


def _is_sha256(value: object) -> bool:
    """判断对象是否为 64 位小写十六进制 SHA-256。

    参数：
        value: 待检查对象。
    返回：
        严格匹配时返回 ``True``。
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
