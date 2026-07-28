"""P6 受保护堆叠算子的名义装配与认证状态边界。

文件用途：
    从冻结模型沿共同时间路径得到的转移 Jacobian/JVP 装配 ``G_nu``、``G_0`` 和故障
    响应算子，并把解析恒等式、名义在线数值和认证 outer enclosure 明确分层。
主要职责：
    定义受控 Jacobian 语义、算子状态、共同路径和 ``OperatorBundle``；实现
    ``NominalJVPAssembler`` 的块下三角装配。本文件不拟合 P7 后滤波、不计算 P8 阈值，
    也不执行 P9 物理类别排除。
关键输入与输出：
    输入为 ``[N,m_z,m_z]`` 转移 Jacobian、可选局部输入响应和同一 episode/stage 的连续
    路径；输出为 ``G_nu=[Nm_z,Nm_z]``、``G_0=[Nm_z,m_z]``、传播后的输入响应及状态。
依赖与副作用：
    依赖 PyTorch、P5 ``MonitorStage`` 和标准库。装配只在 CPU float64 临时 tensor 上
    运算，返回不含 tensor 的不可变对象；不读写文件、不访问网络、不修改模型或 RNG。
重要约束：
    ``SEGMENT_AVERAGED_EXACT`` 只声明调用方提供了 exact segment average，并不证明其
    数值计算已认证；没有完整认证 provider 时状态只能是 ``NOMINAL``。所有算子必须共享
    同一时间路径，路径 gap、非有限值或 shape 不匹配时 fail closed。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

import torch

from .protected_reference import MonitorStage


class JacobianSemantics(str, Enum):
    """调用方提供的转移 Jacobian 数值语义。

    ``SEGMENT_AVERAGED_EXACT`` 对应论文解析恒等式中的线段平均 Jacobian；
    ``NOMINAL_POINTWISE`` 对应部署时沿名义轨迹计算的点值 JVP。两者都不自动构成认证。
    """

    SEGMENT_AVERAGED_EXACT = "segment_averaged_exact"
    NOMINAL_POINTWISE = "nominal_pointwise"


class OperatorStatus(str, Enum):
    """算子证据状态。

    ``NOMINAL`` 表示数值矩阵可用于诊断/消融但不能安全排除；``CERTIFIED`` 表示所需算子
    具有共享 outer enclosure；``UNAVAILABLE`` 表示请求的认证后端未能给出完整证据。
    """

    NOMINAL = "nominal"
    CERTIFIED = "certified"
    UNAVAILABLE = "unavailable"


class OperatorNorm(str, Enum):
    """联合认证误差元素半径使用的受控范数。

    ``ELEMENTWISE_INF`` 的对偶 support 使用一范数；``SPECTRAL_L2`` 与 ``FROBENIUS``
    在当前有限共享系数向量表示中都使用二范数。枚举值会写入产物，禁止调用方隐式猜测。
    """

    SPECTRAL_L2 = "spectral_l2"
    FROBENIUS = "frobenius"
    ELEMENTWISE_INF = "elementwise_inf"


class UncertifiedOperatorError(RuntimeError):
    """认证消费者请求未认证算子时的 fail-closed 异常。

    参数：
        message: 缺失状态、算子名或共享证据的可读原因。
    返回：
        不返回值；调用方必须降级为未认证/拒识路径。
    副作用：
        无。
    """


@dataclass(frozen=True)
class OperatorAssemblyBudget:
    """P6 稠密算子装配的显式资源上限。

    参数：
        max_workspace_elements: 单次装配允许创建的 tensor 标量元素总量上限，包含
            ``G_nu``、块对角临时量和传播结果。
        max_persisted_elements: ``OperatorBundle`` 及认证 enclosure 允许持久化的标量元素
            总量上限。
    返回：
        可复用、不可变的资源预算；预算单位是可精确预估的标量元素，而不是依赖 Python
        实现细节的近似字节数。
    异常：
        上限不是严格正整数（包括误传 ``bool``）时抛出 ``ValueError``。
    副作用：
        无。调用方必须显式选择预算，装配器不会猜测机器可用内存。
    """

    max_workspace_elements: int
    max_persisted_elements: int

    def __post_init__(self) -> None:
        """拒绝隐式布尔/浮点转换，保持预算配置可审计。"""

        for name, value in (
            ("max_workspace_elements", self.max_workspace_elements),
            ("max_persisted_elements", self.max_persisted_elements),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a strictly positive integer.")

    def ensure_within(
        self,
        *,
        workspace_elements: int,
        persisted_elements: int,
    ) -> None:
        """验证一次装配的预估资源量不超过显式预算。

        参数：
            workspace_elements: 本次调用将创建的 tensor 标量元素总量上界。
            persisted_elements: 返回对象将持有的标量元素总量。
        返回：
            两项均未超限时返回 ``None``。
        异常：
            任一计数为负或超过对应上限时抛出 ``ValueError``；检查应发生在稠密分配前。
        副作用：
            无。
        """

        if workspace_elements < 0 or persisted_elements < 0:
            raise ValueError("Operator resource estimates must be non-negative.")
        if (
            workspace_elements > self.max_workspace_elements
            or persisted_elements > self.max_persisted_elements
        ):
            raise ValueError(
                "Operator assembly resource budget exceeded: "
                f"workspace={workspace_elements}/{self.max_workspace_elements}, "
                f"persisted={persisted_elements}/{self.max_persisted_elements}."
            )

    def to_dict(self) -> dict[str, int]:
        """返回 JSON 兼容预算。

        返回：
            含 workspace 与 persisted 元素上限的字典。
        异常：
            无。
        副作用：
            无。
        """

        return {
            "max_workspace_elements": self.max_workspace_elements,
            "max_persisted_elements": self.max_persisted_elements,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OperatorAssemblyBudget:
        """从严格 JSON 映射恢复资源预算。

        参数：
            value: ``to_dict`` 产生的映射。
        返回：
            经构造器再次验证的 ``OperatorAssemblyBudget``。
        异常：
            字段缺失、额外或类型非法时抛出 ``TypeError``/``ValueError``。
        副作用：
            无。
        """

        _require_exact_keys(
            value,
            {"max_workspace_elements", "max_persisted_elements"},
            name="OperatorAssemblyBudget",
        )
        return cls(
            max_workspace_elements=value["max_workspace_elements"],
            max_persisted_elements=value["max_persisted_elements"],
        )


@dataclass(frozen=True)
class OperatorPath:
    """所有堆叠算子共同使用的时间与 monitor 身份。

    参数：
        monitor_identity: P5 gate、模型配置和 checkpoint 的联合身份，或线性验证夹具身份。
        episode_id/stage: 共同 episode 和受控正常/冻结故障范围。
        start_raw_index: 堆叠递推开始时刻 ``s``。
        raw_indices: 依次对应 ``s+1,...,s+N`` 的连续输出索引。
    返回：
        不可变、可序列化的公共路径。
    异常：
        身份/episode 为空、索引为负或不严格连续时抛出 ``ValueError``。
    副作用：
        无。
    """

    monitor_identity: str
    episode_id: str
    stage: MonitorStage
    start_raw_index: int
    raw_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        """严格规范化阶段并验证路径从 ``s+1`` 开始逐步连续。"""

        _strict_string(self.monitor_identity, name="OperatorPath monitor_identity")
        _strict_string(self.episode_id, name="OperatorPath episode_id")
        start_raw_index = _strict_int(
            self.start_raw_index,
            name="OperatorPath start_raw_index",
        )
        if not isinstance(self.raw_indices, (list, tuple)):
            raise TypeError("OperatorPath raw_indices must be a list or tuple.")
        object.__setattr__(
            self,
            "raw_indices",
            tuple(
                _strict_int(index, name="OperatorPath raw index")
                for index in self.raw_indices
            ),
        )
        if start_raw_index < 0 or not self.raw_indices:
            raise ValueError("Operator path requires a non-negative start and non-empty rows.")
        object.__setattr__(self, "stage", MonitorStage.parse(self.stage))
        expected = tuple(
            range(
                self.start_raw_index + 1,
                self.start_raw_index + 1 + len(self.raw_indices),
            )
        )
        if self.raw_indices != expected:
            raise ValueError("Operator path raw_indices must be consecutive from start + 1.")

    def to_dict(self) -> dict[str, Any]:
        """返回 JSON 兼容的公共路径。

        返回：
            含 monitor、episode、受控阶段和连续索引的字典。
        异常：
            无。
        副作用：
            无。
        """

        return {
            "monitor_identity": self.monitor_identity,
            "episode_id": self.episode_id,
            "stage": self.stage.value,
            "start_raw_index": self.start_raw_index,
            "raw_indices": list(self.raw_indices),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OperatorPath:
        """从严格 JSON 映射恢复公共路径。

        参数：
            value: ``to_dict`` 产生的映射。
        返回：
            经连续性与阶段校验的 ``OperatorPath``。
        异常：
            字段缺失、额外或取值非法时抛出 ``TypeError``/``ValueError``。
        副作用：
            无。
        """

        _require_exact_keys(
            value,
            {
                "monitor_identity",
                "episode_id",
                "stage",
                "start_raw_index",
                "raw_indices",
            },
            name="OperatorPath",
        )
        raw_indices = value["raw_indices"]
        if not isinstance(raw_indices, list):
            raise TypeError("OperatorPath raw_indices must be a JSON list.")
        return cls(
            monitor_identity=_strict_string(
                value["monitor_identity"],
                name="OperatorPath monitor_identity",
            ),
            episode_id=_strict_string(
                value["episode_id"],
                name="OperatorPath episode_id",
            ),
            stage=MonitorStage.parse(value["stage"]),
            start_raw_index=_strict_int(
                value["start_raw_index"],
                name="OperatorPath start_raw_index",
            ),
            raw_indices=tuple(
                _strict_int(item, name="OperatorPath raw index")
                for item in raw_indices
            ),
        )


@dataclass(frozen=True)
class OperatorAffineImage:
    """一个命名算子在共享不确定系数下的仿射 image。

    参数：
        operator_name: 与 ``OperatorCertificationRequest`` 一致的稳定名称。
        center: 名义或认证中心矩阵。
        generators: 共享系数向量每个坐标在本算子上的矩阵 image；第 ``j`` 个生成元必须
            与其他命名算子的第 ``j`` 个生成元使用同一个系数。
    返回：
        不可变、可序列化的命名仿射 image。
    异常：
        名称为空、矩阵非法、生成元为空或 shape 不一致时抛出 ``ValueError``。
    副作用：
        无。
    """

    operator_name: str
    center: tuple[tuple[float, ...], ...]
    generators: tuple[tuple[tuple[float, ...], ...], ...]

    def __post_init__(self) -> None:
        """冻结数值并验证所有生成元与中心 shape 相同。"""

        _strict_string(
            self.operator_name,
            name="OperatorAffineImage operator_name",
        )
        center = _coerce_matrix_tuple(self.center, name=f"{self.operator_name} center")
        generators = tuple(
            _coerce_matrix_tuple(
                generator,
                name=f"{self.operator_name} generator {index}",
            )
            for index, generator in enumerate(self.generators)
        )
        if not generators:
            raise ValueError("Operator affine image requires at least one shared generator.")
        shape = (len(center), len(center[0]))
        if any(
            (len(generator), len(generator[0])) != shape
            for generator in generators
        ):
            raise ValueError("Operator affine image generators must match the center shape.")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "generators", generators)

    @property
    def generator_count(self) -> int:
        """返回共享系数向量维数。

        返回：
            ``generators`` 的长度。
        异常：
            无。
        副作用：
            无。
        """

        return len(self.generators)

    @property
    def persisted_elements(self) -> int:
        """返回中心与生成元持有的标量元素总数。

        返回：
            用于资源预算检查的精确标量计数。
        异常：
            无。
        副作用：
            无。
        """

        matrix_size = len(self.center) * len(self.center[0])
        return matrix_size * (1 + self.generator_count)

    def to_dict(self) -> dict[str, Any]:
        """返回 JSON 兼容的命名仿射 image。

        返回：
            含名称、中心和按共享坐标排序生成元的字典。
        异常：
            无。
        副作用：
            无。
        """

        return {
            "operator_name": self.operator_name,
            "center": [list(row) for row in self.center],
            "generators": [
                [list(row) for row in generator]
                for generator in self.generators
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OperatorAffineImage:
        """从严格 JSON 映射恢复命名仿射 image。

        参数：
            value: ``to_dict`` 产生的映射。
        返回：
            经矩阵和 shape 校验的 ``OperatorAffineImage``。
        异常：
            字段或矩阵非法时抛出 ``TypeError``/``ValueError``。
        副作用：
            无。
        """

        _require_exact_keys(
            value,
            {"operator_name", "center", "generators"},
            name="OperatorAffineImage",
        )
        generators = value["generators"]
        if not isinstance(generators, list):
            raise TypeError("OperatorAffineImage generators must be a JSON list.")
        return cls(
            operator_name=_strict_string(
                value["operator_name"],
                name="OperatorAffineImage operator_name",
            ),
            center=_coerce_matrix_tuple(
                value["center"],
                name="OperatorAffineImage center",
            ),
            generators=tuple(
                _coerce_matrix_tuple(
                    generator,
                    name=f"OperatorAffineImage generator {index}",
                )
                for index, generator in enumerate(generators)
            ),
        )


@dataclass(frozen=True)
class OperatorEnclosure:
    """所有命名算子共享同一误差系数的联合认证 outer enclosure。

    数学语义为 ``M_i(theta)=C_i+sum_j theta_j D_{i,j}``，其中每个命名算子都使用同一个
    ``theta``，且 ``||theta|| <= error_radius``。这种联合参数化阻止不同检测/解释支路为
    同一时窗分别挑选互相矛盾的误差。

    参数：
        images: 完整命名算子集合；每个 image 必须具有相同生成元数量。
        error_radius/norm: 共享系数向量的有限非负半径与受控范数。
        shared_uncertainty_id: 由公共路径和全部名义算子派生的误差元素身份。
        source/certificate_id: 可追溯认证后端与联合证据标识。
        verified_remainder: provider 是否包含经验证的积分/数值余项，必须是真正的 ``bool``。
    返回：
        一个不可拆分的联合 enclosure；安全消费者通过 ``support_upper`` 同步查询多算子。
    异常：
        证据、矩阵、共享生成元维数或严格布尔字段非法时抛出 ``ValueError``。
    副作用：
        无。小半径或共享字符串本身都不会使未验证证据升级。
    """

    images: tuple[OperatorAffineImage, ...]
    error_radius: float
    norm: OperatorNorm
    shared_uncertainty_id: str
    source: str
    certificate_id: str
    verified_remainder: bool

    def __post_init__(self) -> None:
        """验证 enclosure 只有一个共享系数空间和一份联合证据。"""

        images = tuple(self.images)
        if not images:
            raise ValueError("Operator enclosure images must be non-empty.")
        _strict_string(
            self.shared_uncertainty_id,
            name="OperatorEnclosure shared_uncertainty_id",
        )
        _strict_string(self.source, name="OperatorEnclosure source")
        _strict_string(
            self.certificate_id,
            name="OperatorEnclosure certificate_id",
        )
        if not isinstance(self.norm, OperatorNorm):
            raise ValueError("Operator enclosure norm must be an OperatorNorm.")
        if type(self.verified_remainder) is not bool:
            raise ValueError("Operator enclosure verified_remainder must be a bool.")
        if any(not isinstance(image, OperatorAffineImage) for image in images):
            raise ValueError("Operator enclosure images must be OperatorAffineImage values.")
        names = tuple(image.operator_name for image in images)
        if len(names) != len(set(names)):
            raise ValueError("Operator enclosure image names must be unique.")
        generator_counts = {image.generator_count for image in images}
        if len(generator_counts) != 1:
            raise ValueError(
                "Operator enclosure images must share one coefficient-vector dimension."
            )
        radius = _strict_float(
            self.error_radius,
            name="OperatorEnclosure error_radius",
        )
        if radius < 0.0:
            raise ValueError("Operator enclosure error_radius must be finite and non-negative.")
        object.__setattr__(self, "images", images)
        object.__setattr__(self, "error_radius", radius)

    @property
    def certified(self) -> bool:
        """只有 provider 明确验证余项时才承认联合 enclosure 已认证。

        返回：
            严格布尔认证标志。
        异常：
            无。
        副作用：
            无。
        """

        return self.verified_remainder

    @property
    def operator_names(self) -> tuple[str, ...]:
        """返回联合 enclosure 覆盖的稳定算子名。

        返回：
            与 ``images`` 顺序一致的名称元组。
        异常：
            无。
        副作用：
            无。
        """

        return tuple(image.operator_name for image in self.images)

    @property
    def persisted_elements(self) -> int:
        """返回全部中心与共享生成元 image 的标量总数。

        返回：
            用于装配资源预算的精确标量计数。
        异常：
            无。
        副作用：
            无。
        """

        return sum(image.persisted_elements for image in self.images)

    def image(self, operator_name: str) -> OperatorAffineImage:
        """在不拆分联合证据的前提下读取一个命名 image。

        参数：
            operator_name: 请求中的稳定算子名。
        返回：
            属于当前联合 enclosure 的 ``OperatorAffineImage``。
        异常：
            名称不在覆盖集合时抛出 ``KeyError``。
        副作用：
            无；返回对象仍通过其创建来源关联到当前 enclosure，安全优化应优先使用
            ``support_upper`` 做联合查询。
        """

        for image in self.images:
            if image.operator_name == operator_name:
                return image
        legal = ", ".join(self.operator_names)
        raise KeyError(f"Unknown enclosed operator {operator_name!r}. Legal names are: {legal}.")

    def support_upper(
        self,
        linear_functionals: Mapping[
            str,
            tuple[tuple[float, ...], ...],
        ],
    ) -> float:
        """计算多个算子线性函数之和在联合 enclosure 上的 support 上界。

        参数：
            linear_functionals: 算子名到同 shape 矩阵的映射，计算
                ``sum_i <H_i,M_i(theta)>`` 的最大值。
        返回：
            在同一个共享 ``theta`` 上得到的有限 support 上界。
        异常：
            映射为空、名称未知、shape/数值非法时抛出 ``TypeError``/``ValueError``。
        副作用：
            无；不会为不同命名 image 独立优化不确定量。
        """

        if not isinstance(linear_functionals, Mapping) or not linear_functionals:
            raise TypeError("linear_functionals must be a non-empty mapping.")
        coefficients = [0.0] * self.images[0].generator_count
        center_value = 0.0
        for name, raw_functional in linear_functionals.items():
            if not isinstance(name, str):
                raise TypeError("Operator support names must be strings.")
            image = self.image(name)
            functional = _coerce_matrix_tuple(
                raw_functional,
                name=f"Support functional {name}",
            )
            if (
                len(functional) != len(image.center)
                or len(functional[0]) != len(image.center[0])
            ):
                raise ValueError("Support functional shape must match its operator image.")
            center_value += _frobenius_inner(functional, image.center)
            for index, generator in enumerate(image.generators):
                coefficients[index] += _frobenius_inner(functional, generator)
        if self.norm is OperatorNorm.ELEMENTWISE_INF:
            dual_value = sum(abs(value) for value in coefficients)
        else:
            dual_value = math.sqrt(sum(value * value for value in coefficients))
        return center_value + self.error_radius * dual_value

    def to_dict(self) -> dict[str, Any]:
        """返回 JSON 兼容的联合认证证据。

        返回：
            含全部命名 image、共享半径/范数和可追溯证据的字典。
        异常：
            无。
        副作用：
            无。
        """

        return {
            "images": [image.to_dict() for image in self.images],
            "error_radius": self.error_radius,
            "norm": self.norm.value,
            "shared_uncertainty_id": self.shared_uncertainty_id,
            "source": self.source,
            "certificate_id": self.certificate_id,
            "verified_remainder": self.verified_remainder,
            "certified": self.certified,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OperatorEnclosure:
        """从严格 JSON 映射恢复联合认证 enclosure。

        参数：
            value: ``to_dict`` 产生的映射。
        返回：
            经共享系数维数和严格证据类型复验的 ``OperatorEnclosure``。
        异常：
            字段缺失、额外、派生 ``certified`` 不一致或证据非法时抛出
            ``TypeError``/``ValueError``。
        副作用：
            无。
        """

        _require_exact_keys(
            value,
            {
                "images",
                "error_radius",
                "norm",
                "shared_uncertainty_id",
                "source",
                "certificate_id",
                "verified_remainder",
                "certified",
            },
            name="OperatorEnclosure",
        )
        images = value["images"]
        if not isinstance(images, list):
            raise TypeError("OperatorEnclosure images must be a JSON list.")
        verified = _strict_bool(
            value["verified_remainder"],
            name="OperatorEnclosure verified_remainder",
        )
        if _strict_bool(
            value["certified"],
            name="OperatorEnclosure certified",
        ) is not verified:
            raise ValueError("OperatorEnclosure certified must match verified_remainder.")
        return cls(
            images=tuple(
                OperatorAffineImage.from_dict(_strict_mapping(item, name="operator image"))
                for item in images
            ),
            error_radius=_strict_float(
                value["error_radius"],
                name="OperatorEnclosure error_radius",
            ),
            norm=OperatorNorm(value["norm"]),
            shared_uncertainty_id=_strict_string(
                value["shared_uncertainty_id"],
                name="OperatorEnclosure shared_uncertainty_id",
            ),
            source=_strict_string(value["source"], name="OperatorEnclosure source"),
            certificate_id=_strict_string(
                value["certificate_id"],
                name="OperatorEnclosure certificate_id",
            ),
            verified_remainder=verified,
        )


@dataclass(frozen=True)
class OperatorCertificationRequest:
    """传给认证后端的完整名义算子与共同不确定元素请求。

    参数：
        semantics/path: Jacobian 数值语义和公共时间身份。
        nominal_operators: 稳定排序的 ``(name,matrix)`` 集合。
        required_operator_names: 本次 bundle 升级为 certified 必须全部覆盖的名称。
        shared_uncertainty_id: provider 必须原样回传的联合误差元素身份。
        resource_budget/nominal_persisted_elements: provider 可用于限制生成元数量的总预算与
            已被名义矩阵占用的元素数。
    返回：
        认证 provider 的不可变输入，可用 ``nominal(name)`` 读取中心。
    异常：
        名称重复、必需名称缺失或共享身份为空时抛出 ``ValueError``。
    副作用：
        无。
    """

    semantics: JacobianSemantics
    path: OperatorPath
    nominal_operators: tuple[
        tuple[str, tuple[tuple[float, ...], ...]],
        ...,
    ]
    required_operator_names: tuple[str, ...]
    shared_uncertainty_id: str
    resource_budget: OperatorAssemblyBudget
    nominal_persisted_elements: int

    def __post_init__(self) -> None:
        """保证 provider 看到的是完整且无歧义的命名算子集合。"""

        if not isinstance(self.semantics, JacobianSemantics):
            raise ValueError("Certification request semantics must be a JacobianSemantics.")
        if not isinstance(self.path, OperatorPath):
            raise ValueError("Certification request path must be an OperatorPath.")
        _strict_string(
            self.shared_uncertainty_id,
            name="Certification request shared_uncertainty_id",
        )
        if not isinstance(self.nominal_operators, tuple):
            raise TypeError("Certification request nominal_operators must be a tuple.")
        if not isinstance(self.required_operator_names, tuple):
            raise TypeError("Certification request required_operator_names must be a tuple.")
        names = tuple(name for name, _ in self.nominal_operators)
        if (
            len(names) != len(set(names))
            or names != self.required_operator_names
        ):
            raise ValueError(
                "Certification request names must be unique, complete, and ordered."
            )
        if not isinstance(self.resource_budget, OperatorAssemblyBudget):
            raise ValueError("Certification request requires an OperatorAssemblyBudget.")
        if (
            type(self.nominal_persisted_elements) is not int
            or self.nominal_persisted_elements <= 0
        ):
            raise ValueError(
                "Certification request nominal_persisted_elements must be a positive integer."
            )
        self.resource_budget.ensure_within(
            workspace_elements=0,
            persisted_elements=self.nominal_persisted_elements,
        )
        for name, matrix in self.nominal_operators:
            _strict_string(name, name="Certification operator name")
            _validate_matrix_tuple(matrix, name=f"Nominal operator {name}")

    def nominal(self, name: str) -> tuple[tuple[float, ...], ...]:
        """按稳定名称返回名义中心矩阵。

        参数：
            name: ``required_operator_names`` 中的名称。
        返回：
            不可变二维矩阵。
        异常：
            名称未知时抛出带合法名称列表的 ``KeyError``。
        副作用：
            无。
        """

        for candidate, matrix in self.nominal_operators:
            if candidate == name:
                return matrix
        legal = ", ".join(self.required_operator_names)
        raise KeyError(f"Unknown nominal operator {name!r}. Legal names are: {legal}.")

    def to_dict(self) -> dict[str, Any]:
        """返回 provider 请求的 JSON 兼容审计表示。

        返回：
            含语义、公共路径、全部中心、共享身份和剩余资源契约的字典。
        异常：
            无。
        副作用：
            无。
        """

        return {
            "semantics": self.semantics.value,
            "path": self.path.to_dict(),
            "nominal_operators": {
                name: [list(row) for row in matrix]
                for name, matrix in self.nominal_operators
            },
            "required_operator_names": list(self.required_operator_names),
            "shared_uncertainty_id": self.shared_uncertainty_id,
            "resource_budget": self.resource_budget.to_dict(),
            "nominal_persisted_elements": self.nominal_persisted_elements,
        }


class CertifiedEnclosureProvider(Protocol):
    """认证 outer enclosure 后端协议。

    实现必须返回一个覆盖所有必需名称的联合 enclosure：每个命名 image 使用同一个
    共享系数向量，并回传请求给定的 ``shared_uncertainty_id``。返回 ``None`` 表示当前
    路径无法认证；未验证余项、缺项、独立误差元素或超出请求资源预算会由装配器降为
    ``UNAVAILABLE``。
    """

    def enclose(
        self,
        request: OperatorCertificationRequest,
    ) -> OperatorEnclosure | None:
        """返回完整联合 enclosure，或显式表示认证不可用。

        参数：
            request: 含全部名义中心、必需名称、公共路径和共享误差身份的不可变请求。
        返回：
            一个完整联合 ``OperatorEnclosure``；当前路径不能认证时返回 ``None``。
        异常：
            后端自身不可恢复的配置/实现错误可以抛出；预期的无认证结果应返回 ``None``。
        副作用：
            由具体 provider 声明；实现不得修改 ``request``。
        """

        ...


@dataclass(frozen=True)
class OperatorBundle:
    """一个公共路径上的完整名义/认证算子包。

    参数：
        status/status_reason: 当前证据级别和不能升级时的明确原因。
        semantics/path: Jacobian 数值语义和全部算子的公共时间身份。
        g_nu/g_0: 创新传播与 anchor-error 传播矩阵。
        input_response: 可选 ``G_nu blkdiag(B^u_k)``。
        sensor_jvps/process_prior: 传感器 JVP 响应和传播后的过程先验。
        required_certification_names/shared_uncertainty_id/enclosure: 联合认证覆盖范围、共享
            operator-error 身份及不可拆分的完整证据。
        resource_budget/workspace_elements/persisted_elements: 调用方预算与本次装配的预估
            资源占用，供产物审计和重放。
    返回：
        不可变、可序列化且带内容哈希的算子包。
    异常：
        只应由装配器从已校验输入构造。
    副作用：
        无。``certified`` 只是状态派生值，不会从小的名义误差猜测认证。
    """

    status: OperatorStatus
    status_reason: str
    semantics: JacobianSemantics
    path: OperatorPath
    g_nu: tuple[tuple[float, ...], ...]
    g_0: tuple[tuple[float, ...], ...]
    required_certification_names: tuple[str, ...]
    shared_uncertainty_id: str
    resource_budget: OperatorAssemblyBudget
    workspace_elements: int
    persisted_elements: int
    enclosure: OperatorEnclosure | None = None
    input_response: tuple[tuple[float, ...], ...] | None = None
    sensor_jvps: tuple[tuple[str, tuple[tuple[float, ...], ...]], ...] = ()
    process_prior: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        """阻止状态枚举、算子集合和认证证据互相矛盾。"""

        if (
            not isinstance(self.status, OperatorStatus)
            or not isinstance(self.semantics, JacobianSemantics)
            or not isinstance(self.path, OperatorPath)
            or not isinstance(self.resource_budget, OperatorAssemblyBudget)
            or type(self.workspace_elements) is not int
            or type(self.persisted_elements) is not int
        ):
            raise ValueError("Operator bundle status, semantics, and identities are required.")
        _strict_string(self.status_reason, name="OperatorBundle status_reason")
        _strict_string(
            self.shared_uncertainty_id,
            name="OperatorBundle shared_uncertainty_id",
        )
        self.resource_budget.ensure_within(
            workspace_elements=self.workspace_elements,
            persisted_elements=self.persisted_elements,
        )
        _validate_matrix_tuple(self.g_nu, name="OperatorBundle g_nu")
        _validate_matrix_tuple(self.g_0, name="OperatorBundle g_0")
        row_count = len(self.g_nu)
        if len(self.g_nu[0]) != row_count or len(self.g_0) != row_count:
            raise ValueError("OperatorBundle g_nu must be square and share rows with g_0.")

        expected_names = ["g_nu", "g_0"]
        nominal_matrices = {
            "g_nu": self.g_nu,
            "g_0": self.g_0,
        }
        if self.input_response is not None:
            _validate_matrix_tuple(
                self.input_response,
                name="OperatorBundle input_response",
            )
            if len(self.input_response) != row_count:
                raise ValueError("OperatorBundle input_response row count must match g_nu.")
            expected_names.append("input_response")
            nominal_matrices["input_response"] = self.input_response
        if self.process_prior is not None:
            _validate_matrix_tuple(
                self.process_prior,
                name="OperatorBundle process_prior",
            )
            if len(self.process_prior) != row_count:
                raise ValueError("OperatorBundle process_prior row count must match g_nu.")
            expected_names.append("process_prior")
            nominal_matrices["process_prior"] = self.process_prior
        sensor_names: set[str] = set()
        for name, matrix in self.sensor_jvps:
            if not name or name in sensor_names:
                raise ValueError("OperatorBundle sensor JVP names must be unique and non-empty.")
            _validate_matrix_tuple(matrix, name=f"OperatorBundle sensor JVP {name}")
            if len(matrix) != row_count:
                raise ValueError("OperatorBundle sensor JVP row count must match g_nu.")
            sensor_names.add(name)
            operator_name = f"sensor:{name}"
            expected_names.append(operator_name)
            nominal_matrices[operator_name] = matrix
        if self.required_certification_names != tuple(expected_names):
            raise ValueError(
                "OperatorBundle required certification names must match its operators."
            )

        if self.status is OperatorStatus.CERTIFIED:
            if self.enclosure is None:
                raise ValueError(
                    "CERTIFIED OperatorBundle requires one joint enclosure."
                )
            if (
                not self.enclosure.certified
                or self.enclosure.shared_uncertainty_id != self.shared_uncertainty_id
                or self.enclosure.operator_names != tuple(expected_names)
            ):
                raise ValueError(
                    "CERTIFIED OperatorBundle requires one complete shared verified enclosure."
                )
            for name in expected_names:
                image = self.enclosure.image(name)
                nominal = nominal_matrices[name]
                if (
                    len(image.center) != len(nominal)
                    or len(image.center[0]) != len(nominal[0])
                ):
                    raise ValueError(
                        "CERTIFIED OperatorBundle enclosure names/shapes must match operators."
                    )
        elif self.enclosure is not None:
            raise ValueError(
                "Only a CERTIFIED OperatorBundle may retain a certification enclosure."
            )

        actual_persisted_elements = sum(
            len(matrix) * len(matrix[0])
            for matrix in nominal_matrices.values()
        )
        if self.enclosure is not None:
            actual_persisted_elements += self.enclosure.persisted_elements
        if self.persisted_elements != actual_persisted_elements:
            raise ValueError(
                "OperatorBundle persisted_elements must match its matrices and enclosure."
            )

    @property
    def certified(self) -> bool:
        """仅在状态明确为 ``CERTIFIED`` 时允许认证消费者继续。

        返回：
            状态枚举是否为 ``CERTIFIED``。
        异常：
            无。
        副作用：
            无。
        """

        return self.status is OperatorStatus.CERTIFIED

    def require_certified(self, *operator_names: str) -> OperatorEnclosure:
        """为安全排除消费者返回同一个联合认证 enclosure，否则立即失败。

        参数：
            operator_names: 当前消费者实际依赖的一个或多个稳定算子名。
        返回：
            覆盖全部请求名称的单一 ``OperatorEnclosure``；调用方必须通过其联合
            ``support_upper`` 接口同步处理不确定系数。
        异常：
            bundle 未认证、名称不在完整认证集合或证据缺失时抛出
            ``UncertifiedOperatorError``。
        副作用：
            无。
        """

        if not operator_names:
            raise ValueError("At least one certified operator name is required.")
        if not self.certified:
            raise UncertifiedOperatorError(
                f"Safe exclusion requires certified operators: {self.status_reason}"
            )
        if self.enclosure is None:
            raise UncertifiedOperatorError(
                "Certified bundle is missing its joint operator enclosure."
            )
        missing = tuple(
            name for name in operator_names if name not in self.enclosure.operator_names
        )
        if missing:
            raise UncertifiedOperatorError(
                "Certified bundle does not cover requested operators: "
                + ", ".join(missing)
            )
        if self.enclosure.shared_uncertainty_id != self.shared_uncertainty_id:
            raise UncertifiedOperatorError(
                "Certified operators do not share the bundle uncertainty element."
            )
        return self.enclosure

    def to_dict(self) -> dict[str, Any]:
        """返回包含状态、联合证据、资源预算和全部矩阵的 JSON 表示。

        返回：
            可直接写入论文运行产物或最终报告的完整字典。
        异常：
            无。
        副作用：
            无。
        """

        return {
            "status": self.status.value,
            "status_reason": self.status_reason,
            "certified": self.certified,
            "semantics": self.semantics.value,
            "path": self.path.to_dict(),
            "required_certification_names": list(self.required_certification_names),
            "shared_uncertainty_id": self.shared_uncertainty_id,
            "resource_budget": self.resource_budget.to_dict(),
            "workspace_elements": self.workspace_elements,
            "persisted_elements": self.persisted_elements,
            "enclosure": None if self.enclosure is None else self.enclosure.to_dict(),
            "g_nu": [list(row) for row in self.g_nu],
            "g_0": [list(row) for row in self.g_0],
            "input_response": None
            if self.input_response is None
            else [list(row) for row in self.input_response],
            "sensor_jvps": {
                name: [list(row) for row in matrix]
                for name, matrix in self.sensor_jvps
            },
            "process_prior": None
            if self.process_prior is None
            else [list(row) for row in self.process_prior],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OperatorBundle:
        """从严格 JSON 映射恢复可重放的算子报告。

        参数：
            value: ``to_dict`` 产生并可能经 ``ArtifactStore`` 落盘的映射。
        返回：
            重新执行状态、shape、联合证据和资源预算不变量后的 ``OperatorBundle``。
        异常：
            字段缺失/额外、派生认证标志矛盾或数值/证据非法时抛出
            ``TypeError``/``ValueError``。
        副作用：
            无；不读取文件，调用方负责 JSON I/O。
        """

        _require_exact_keys(
            value,
            {
                "status",
                "status_reason",
                "certified",
                "semantics",
                "path",
                "required_certification_names",
                "shared_uncertainty_id",
                "resource_budget",
                "workspace_elements",
                "persisted_elements",
                "enclosure",
                "g_nu",
                "g_0",
                "input_response",
                "sensor_jvps",
                "process_prior",
            },
            name="OperatorBundle",
        )
        status = OperatorStatus(value["status"])
        certified = _strict_bool(value["certified"], name="OperatorBundle certified")
        if certified is not (status is OperatorStatus.CERTIFIED):
            raise ValueError("OperatorBundle certified must match status.")
        required_names = value["required_certification_names"]
        if not isinstance(required_names, list):
            raise TypeError(
                "OperatorBundle required_certification_names must be a JSON list."
            )
        sensor_jvps = _strict_mapping(
            value["sensor_jvps"],
            name="OperatorBundle sensor_jvps",
        )
        enclosure_value = value["enclosure"]
        return cls(
            status=status,
            status_reason=_strict_string(
                value["status_reason"],
                name="OperatorBundle status_reason",
            ),
            semantics=JacobianSemantics(value["semantics"]),
            path=OperatorPath.from_dict(
                _strict_mapping(value["path"], name="OperatorBundle path")
            ),
            g_nu=_coerce_matrix_tuple(value["g_nu"], name="OperatorBundle g_nu"),
            g_0=_coerce_matrix_tuple(value["g_0"], name="OperatorBundle g_0"),
            required_certification_names=tuple(
                _strict_string(name, name="OperatorBundle required operator name")
                for name in required_names
            ),
            shared_uncertainty_id=_strict_string(
                value["shared_uncertainty_id"],
                name="OperatorBundle shared_uncertainty_id",
            ),
            resource_budget=OperatorAssemblyBudget.from_dict(
                _strict_mapping(
                    value["resource_budget"],
                    name="OperatorBundle resource_budget",
                )
            ),
            workspace_elements=_strict_int(
                value["workspace_elements"],
                name="OperatorBundle workspace_elements",
            ),
            persisted_elements=_strict_int(
                value["persisted_elements"],
                name="OperatorBundle persisted_elements",
            ),
            enclosure=None
            if enclosure_value is None
            else OperatorEnclosure.from_dict(
                _strict_mapping(
                    enclosure_value,
                    name="OperatorBundle enclosure",
                )
            ),
            input_response=None
            if value["input_response"] is None
            else _coerce_matrix_tuple(
                value["input_response"],
                name="OperatorBundle input_response",
            ),
            sensor_jvps=tuple(
                (
                    _strict_string(name, name="OperatorBundle sensor name"),
                    _coerce_matrix_tuple(
                        matrix,
                        name=f"OperatorBundle sensor {name}",
                    ),
                )
                for name, matrix in sorted(sensor_jvps.items())
            ),
            process_prior=None
            if value["process_prior"] is None
            else _coerce_matrix_tuple(
                value["process_prior"],
                name="OperatorBundle process_prior",
            ),
        )

    @property
    def content_hash(self) -> str:
        """返回算子状态、路径、资源预算和联合证据的确定性 SHA-256。

        返回：
            64 位小写十六进制 SHA-256。
        异常：
            理论上无；对象构造时已拒绝 NaN/Inf。
        副作用：
            无。
        """

        encoded = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class NominalJVPAssembler:
    """从共同路径上的转移 Jacobian/JVP 装配堆叠算子。

    参数：
        resource_budget: 调用方显式选择的 workspace 与持久化元素预算；没有默认值，避免
            根据窗口和 latent 维数静默分配无上限的 ``(N*m_z)^2`` 矩阵。
    返回：
        调用 ``assemble`` 得到 ``OperatorBundle``。
    异常：
        预算缺失/非法、horizon/shape/path 不匹配、资源超限或输入含非有限值时抛出
        ``TypeError``/``ValueError``。
    副作用：
        无；不修改输入 tensor，未提供认证后端时不创建 enclosure。
    """

    def __init__(self, *, resource_budget: OperatorAssemblyBudget) -> None:
        """绑定所有后续装配调用共享的显式资源预算。

        参数：
            resource_budget: 经严格正整数验证的 ``OperatorAssemblyBudget``。
        返回：
            无。
        异常：
            类型错误时抛出 ``TypeError``。
        副作用：
            无；只保存不可变预算引用。
        """

        if not isinstance(resource_budget, OperatorAssemblyBudget):
            raise TypeError("resource_budget must be an OperatorAssemblyBudget.")
        self.resource_budget = resource_budget

    def assemble(
        self,
        *,
        transition_jacobians: torch.Tensor,
        semantics: JacobianSemantics,
        path: OperatorPath,
        input_jacobians: torch.Tensor | None = None,
        sensor_jvps: Mapping[str, torch.Tensor] | None = None,
        process_prior: torch.Tensor | None = None,
        enclosure_provider: CertifiedEnclosureProvider | None = None,
    ) -> OperatorBundle:
        """装配单位块下三角 ``G_nu``、``G_0`` 和可选输入响应。

        参数：
            transition_jacobians: ``[N,m_z,m_z]`` 的线段平均或点值 Jacobian。
            semantics: 调用方对上述 Jacobian 的受控语义声明。
            path: 长度同 ``N`` 的公共连续路径。
            input_jacobians: 可选 ``[N,m_z,m_u]`` 局部输入响应。
            sensor_jvps: 可选命名 ``[N*m_z,q]`` 传感器历史响应。
            process_prior: 可选 ``[N,m_z,p]`` 局部过程故障先验。
            enclosure_provider: 可选联合认证后端；省略时状态严格保持 ``NOMINAL``。
        返回：
            没有认证 provider 的 ``NOMINAL`` 算子包。
        异常：
            类型、shape、路径长度或数值有限性不满足时抛出 ``TypeError``/``ValueError``。
        副作用：
            无。
        """

        if not isinstance(semantics, JacobianSemantics):
            raise TypeError("semantics must be a JacobianSemantics value.")
        if not isinstance(path, OperatorPath):
            raise TypeError("path must be an OperatorPath.")
        transition_shape = _tensor_shape(
            transition_jacobians,
            name="transition_jacobians",
        )
        if (
            len(transition_shape) != 3
            or transition_shape[0] <= 0
            or transition_shape[1] <= 0
            or transition_shape[1] != transition_shape[2]
        ):
            raise ValueError(
                "transition_jacobians must have shape [N, latent_dim, latent_dim]."
            )
        horizon = transition_shape[0]
        latent_dim = transition_shape[1]
        if len(path.raw_indices) != horizon:
            raise ValueError("Operator path length must match the Jacobian horizon.")

        input_shape: tuple[int, ...] | None = None
        if input_jacobians is not None:
            input_shape = _tensor_shape(input_jacobians, name="input_jacobians")
            if (
                len(input_shape) != 3
                or input_shape[0] != horizon
                or input_shape[1] != latent_dim
                or input_shape[2] <= 0
            ):
                raise ValueError(
                    "input_jacobians must have shape [N, latent_dim, input_dim]."
                )

        sensor_shapes: tuple[tuple[str, tuple[int, ...]], ...] = ()
        if sensor_jvps is not None:
            if not isinstance(sensor_jvps, Mapping):
                raise TypeError("sensor_jvps must be a mapping of named tensors.")
            inspected_sensors: list[tuple[str, tuple[int, ...]]] = []
            for name in sorted(sensor_jvps):
                if not isinstance(name, str) or not name:
                    raise ValueError("Sensor JVP names must be non-empty strings.")
                shape = _tensor_shape(
                    sensor_jvps[name],
                    name=f"sensor_jvps[{name!r}]",
                )
                if (
                    len(shape) != 2
                    or shape[0] != horizon * latent_dim
                    or shape[1] <= 0
                ):
                    raise ValueError(
                        "Each sensor JVP must have shape [N * latent_dim, response_dim]."
                    )
                inspected_sensors.append((name, shape))
            sensor_shapes = tuple(inspected_sensors)

        process_shape: tuple[int, ...] | None = None
        if process_prior is not None:
            process_shape = _tensor_shape(process_prior, name="process_prior")
            if (
                len(process_shape) != 3
                or process_shape[0] != horizon
                or process_shape[1] != latent_dim
                or process_shape[2] <= 0
            ):
                raise ValueError(
                    "process_prior must have shape [N, latent_dim, process_dim]."
                )

        workspace_elements, nominal_persisted_elements = _estimate_resource_elements(
            horizon=horizon,
            latent_dim=latent_dim,
            input_shape=input_shape,
            sensor_shapes=sensor_shapes,
            process_shape=process_shape,
        )
        # 预算检查必须先于 float64 clone、G_nu 和 block_diag 分配，避免误配直接 OOM。
        self.resource_budget.ensure_within(
            workspace_elements=workspace_elements,
            persisted_elements=nominal_persisted_elements,
        )

        transitions = _finite_float64_tensor(
            transition_jacobians,
            name="transition_jacobians",
        )
        g_nu = _assemble_g_nu(transitions)
        g_0 = _assemble_g_0(transitions)
        input_response = None
        if input_jacobians is not None:
            inputs = _finite_float64_tensor(input_jacobians, name="input_jacobians")
            input_response = g_nu @ torch.block_diag(*inputs.unbind(dim=0))

        sensor_responses: list[
            tuple[str, tuple[tuple[float, ...], ...]]
        ] = []
        if sensor_jvps is not None:
            for name, _ in sensor_shapes:
                response = _finite_float64_tensor(
                    sensor_jvps[name],
                    name=f"sensor_jvps[{name!r}]",
                )
                sensor_responses.append((name, _matrix_tuple(response)))

        propagated_process = None
        if process_prior is not None:
            process = _finite_float64_tensor(process_prior, name="process_prior")
            propagated_process = g_nu @ torch.block_diag(*process.unbind(dim=0))

        nominal_items: list[
            tuple[str, tuple[tuple[float, ...], ...]]
        ] = [
            ("g_nu", _matrix_tuple(g_nu)),
            ("g_0", _matrix_tuple(g_0)),
        ]
        if input_response is not None:
            nominal_items.append(("input_response", _matrix_tuple(input_response)))
        if propagated_process is not None:
            nominal_items.append(("process_prior", _matrix_tuple(propagated_process)))
        nominal_items.extend(
            (f"sensor:{name}", matrix)
            for name, matrix in sensor_responses
        )
        required_names = tuple(name for name, _ in nominal_items)
        shared_uncertainty_id = _shared_uncertainty_id(
            semantics=semantics,
            path=path,
            nominal_items=tuple(nominal_items),
        )
        status = OperatorStatus.NOMINAL
        status_reason = "No certified enclosure provider was supplied."
        accepted_enclosure: OperatorEnclosure | None = None
        persisted_elements = nominal_persisted_elements
        if enclosure_provider is not None:
            request = OperatorCertificationRequest(
                semantics=semantics,
                path=path,
                nominal_operators=tuple(nominal_items),
                required_operator_names=required_names,
                shared_uncertainty_id=shared_uncertainty_id,
                resource_budget=self.resource_budget,
                nominal_persisted_elements=nominal_persisted_elements,
            )
            provided = enclosure_provider.enclose(request)
            accepted_enclosure, unavailable_reason = _validated_enclosure(
                provided,
                request=request,
            )
            if unavailable_reason is None:
                assert accepted_enclosure is not None
                persisted_elements += accepted_enclosure.persisted_elements
                status = OperatorStatus.CERTIFIED
                status_reason = "Complete shared certified enclosure is available."
            else:
                status = OperatorStatus.UNAVAILABLE
                status_reason = unavailable_reason

        return OperatorBundle(
            status=status,
            status_reason=status_reason,
            semantics=semantics,
            path=path,
            g_nu=nominal_items[0][1],
            g_0=nominal_items[1][1],
            required_certification_names=required_names,
            shared_uncertainty_id=shared_uncertainty_id,
            resource_budget=self.resource_budget,
            workspace_elements=workspace_elements,
            persisted_elements=persisted_elements,
            enclosure=accepted_enclosure,
            input_response=None
            if input_response is None
            else _matrix_tuple(input_response),
            sensor_jvps=tuple(sensor_responses),
            process_prior=None
            if propagated_process is None
            else _matrix_tuple(propagated_process),
        )


def _assemble_g_nu(transitions: torch.Tensor) -> torch.Tensor:
    """按 exact propagation 索引装配单位块下三角创新传播算子。"""

    horizon, latent_dim, _ = transitions.shape
    result = transitions.new_zeros(
        horizon * latent_dim,
        horizon * latent_dim,
    )
    identity = torch.eye(latent_dim, dtype=transitions.dtype)
    for row in range(horizon):
        row_slice = slice(row * latent_dim, (row + 1) * latent_dim)
        diagonal_slice = slice(row * latent_dim, (row + 1) * latent_dim)
        result[row_slice, diagonal_slice] = identity
        product = identity
        # 固定 row 后从对角块向左递推，既保持 A_row ... A_{column+1} 的顺序，也避免
        # 每个块重新计算相同的转移前缀。
        for column in range(row - 1, -1, -1):
            product = product @ transitions[column + 1]
            column_slice = slice(
                column * latent_dim,
                (column + 1) * latent_dim,
            )
            result[row_slice, column_slice] = product
    return result


def _assemble_g_0(transitions: torch.Tensor) -> torch.Tensor:
    """堆叠从起点误差到每个未来时刻的转移乘积。"""

    products: list[torch.Tensor] = []
    product = torch.eye(transitions.shape[1], dtype=transitions.dtype)
    for transition in transitions:
        product = transition @ product
        products.append(product)
    return torch.cat(products, dim=0)


def _tensor_shape(value: torch.Tensor, *, name: str) -> tuple[int, ...]:
    """在复制或分配前读取 tensor shape，并拒绝其他对象。"""

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    return tuple(int(size) for size in value.shape)


def _estimate_resource_elements(
    *,
    horizon: int,
    latent_dim: int,
    input_shape: tuple[int, ...] | None,
    sensor_shapes: tuple[tuple[str, tuple[int, ...]], ...],
    process_shape: tuple[int, ...] | None,
) -> tuple[int, int]:
    """在任何稠密分配前估算峰值 workspace 与持久化标量元素。

    ``workspace`` 保守计入输入 float64 clone、``G_nu/G_0``、矩阵乘法临时量、块对角
    临时量和传播结果；``persisted`` 精确计入 ``OperatorBundle`` 的名义矩阵。认证
    provider 的联合生成元在返回后单独检查。
    """

    stacked_dim = horizon * latent_dim
    g_nu_elements = stacked_dim * stacked_dim
    g_0_elements = stacked_dim * latent_dim
    transition_elements = horizon * latent_dim * latent_dim
    # G_nu 内层乘法最多同时保留输入 product 与一个新结果。
    workspace = (
        transition_elements
        + g_nu_elements
        + g_0_elements
        + 2 * latent_dim * latent_dim
    )
    persisted = g_nu_elements + g_0_elements

    if input_shape is not None:
        input_elements = math.prod(input_shape)
        response_elements = stacked_dim * (horizon * input_shape[2])
        workspace += input_elements + 2 * response_elements
        persisted += response_elements
    for _, shape in sensor_shapes:
        sensor_elements = math.prod(shape)
        workspace += sensor_elements
        persisted += sensor_elements
    if process_shape is not None:
        process_elements = math.prod(process_shape)
        response_elements = stacked_dim * (horizon * process_shape[2])
        workspace += process_elements + 2 * response_elements
        persisted += response_elements
    return workspace, persisted


def _finite_float64_tensor(value: torch.Tensor, *, name: str) -> torch.Tensor:
    """复制到 CPU float64，并拒绝非 tensor 或非有限值。"""

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    result = value.detach().to(device="cpu", dtype=torch.float64).clone()
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _matrix_tuple(value: torch.Tensor) -> tuple[tuple[float, ...], ...]:
    """把二维 tensor 冻结为 JSON 可序列化元组。"""

    if value.ndim != 2 or not math.prod(value.shape):
        raise ValueError("Operator matrices must be non-empty and two-dimensional.")
    return tuple(
        tuple(float(item) for item in row)
        for row in value.detach().cpu().tolist()
    )


def _coerce_matrix_tuple(
    value: Any,
    *,
    name: str,
) -> tuple[tuple[float, ...], ...]:
    """把嵌套 JSON/tuple 数值严格冻结为有限矩阵。"""

    if not isinstance(value, (list, tuple)) or not value:
        raise TypeError(f"{name} must be a non-empty matrix sequence.")
    rows: list[tuple[float, ...]] = []
    for row in value:
        if not isinstance(row, (list, tuple)) or not row:
            raise TypeError(f"{name} rows must be non-empty sequences.")
        converted: list[float] = []
        for item in row:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise TypeError(f"{name} entries must be real numbers, not bool.")
            number = float(item)
            if not math.isfinite(number):
                raise ValueError(f"{name} must contain only finite values.")
            converted.append(number)
        rows.append(tuple(converted))
    result = tuple(rows)
    _validate_matrix_tuple(result, name=name)
    return result


def _validate_matrix_tuple(
    value: tuple[tuple[float, ...], ...],
    *,
    name: str,
) -> None:
    """验证持久化矩阵非空、矩形且只含有限值。"""

    if not value or not value[0]:
        raise ValueError(f"{name} must be a non-empty matrix.")
    width = len(value[0])
    if any(len(row) != width for row in value):
        raise ValueError(f"{name} must be rectangular.")
    if not all(math.isfinite(float(item)) for row in value for item in row):
        raise ValueError(f"{name} must contain only finite values.")


def _frobenius_inner(
    left: tuple[tuple[float, ...], ...],
    right: tuple[tuple[float, ...], ...],
) -> float:
    """计算两个已校验同 shape 矩阵的 Frobenius 内积。"""

    return sum(
        left_value * right_value
        for left_row, right_row in zip(left, right, strict=True)
        for left_value, right_value in zip(left_row, right_row, strict=True)
    )


def _strict_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    """严格接受字符串键映射，供持久化证据反序列化使用。"""

    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise TypeError(f"{name} must be a string-keyed mapping.")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    name: str,
) -> None:
    """拒绝缺失或未知 JSON 字段，避免证据 schema 静默漂移。"""

    mapping = _strict_mapping(value, name=name)
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} fields mismatch; missing={missing}, extra={extra}.")


def _strict_string(value: Any, *, name: str) -> str:
    """严格读取非空字符串。"""

    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string.")
    return value


def _strict_int(value: Any, *, name: str) -> int:
    """严格读取整数并拒绝 Python 中作为 int 子类的 bool。"""

    if type(value) is not int:
        raise TypeError(f"{name} must be an integer.")
    return value


def _strict_bool(value: Any, *, name: str) -> bool:
    """严格读取布尔证据，禁止字符串和 0/1 伪装。"""

    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool.")
    return value


def _strict_float(value: Any, *, name: str) -> float:
    """严格读取有限实数并拒绝 bool。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _shared_uncertainty_id(
    *,
    semantics: JacobianSemantics,
    path: OperatorPath,
    nominal_items: tuple[
        tuple[str, tuple[tuple[float, ...], ...]],
        ...,
    ],
) -> str:
    """从共同路径和全部名义算子生成联合误差元素身份。"""

    payload = {
        "semantics": semantics.value,
        "path": path.to_dict(),
        "operators": {
            name: [list(row) for row in matrix]
            for name, matrix in nominal_items
        },
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"operator-error:{hashlib.sha256(encoded).hexdigest()}"


def _validated_enclosure(
    provided: OperatorEnclosure | None,
    *,
    request: OperatorCertificationRequest,
) -> tuple[OperatorEnclosure | None, str | None]:
    """验证 provider 返回一个完整、同参数化且未超预算的联合 enclosure。"""

    if provided is None:
        return None, "Certified enclosure provider returned no enclosure."
    if not isinstance(provided, OperatorEnclosure):
        return None, "Certified enclosure provider returned an invalid joint enclosure."
    if provided.operator_names != request.required_operator_names:
        return None, "Certified enclosure provider did not cover every required operator."
    if not provided.certified:
        return None, "Certified enclosure lacks a verified remainder."
    if provided.shared_uncertainty_id != request.shared_uncertainty_id:
        return None, "Certified enclosure must use the requested operator-error element."

    for name in request.required_operator_names:
        image = provided.image(name)
        nominal = request.nominal(name)
        if (
            len(image.center) != len(nominal)
            or len(image.center[0]) != len(nominal[0])
        ):
            return None, f"Certified enclosure center shape mismatch for {name}."
    try:
        request.resource_budget.ensure_within(
            workspace_elements=0,
            persisted_elements=(
                request.nominal_persisted_elements + provided.persisted_elements
            ),
        )
    except ValueError:
        return None, "Certified enclosure exceeds the requested persisted resource budget."
    return provided, None
