"""
仓库内真实过程数据集的规范化适配器。

文件用途：
    把 TE、CSTR、TTS、HY、多相流和 WPT 等异构原始文件转换为统一的
    ``CanonicalDataset``，供 DataModule 和实验编排层使用。
主要职责：
    解析各数据集的官方 split、物理 schema、任务语义和 segment 来源信息；
    本文件不负责训练模型、拟合预处理器、选择阈值或计算论文故障性能。
关键输入与输出：
    输入为显式传入的公开或私有数据根目录及任务名称；输出为带 DataSchema、
    SegmentInfo 和来源摘要的 CanonicalDataset。未提供根目录时只返回确定性 smoke fallback。
依赖与副作用：
    依赖 NumPy、Pandas 和 joff.data.sources 的 MAT/NPZ 读取器；只有调用 ``read`` 时
    才读取文件，模块导入本身不访问磁盘、不创建目录、不改变随机状态。
重要约束：
    schema 决定列语义，答案列和 episode 元数据不得进入模型输入；真实故障数据只用于
    冻结后的评价或协议回归，不能用于模型、阈值和结构选择；每个数据变体的许可状态和
    权威证据必须原样进入名称解析、卡片解析与真实读取的来源摘要。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from joff.data.schema import ColumnSpec, DataSchema, SegmentInfo, TaskSchema
from joff.data.sources import read_mat_arrays, read_npz_arrays

from .base import CanonicalDataset, Segment
from .builtin import SyntheticCSTRFaultAdapter, SyntheticProcessAdapter


_OA_ACCESS = {"tag": "oa", "disclosure": "open_access", "license": "to_verify"}
_CSTR_CLOSED_LOOP_ACCESS = {
    "tag": "oa",
    "disclosure": "open_access",
    "license": "to_verify",
    "license_status": "to_verify",
    "license_reason": "local_mat_generation_chain_not_documented",
    "upstream_model_license": "BSD-3-Clause",
    "upstream_model_license_status": "verified",
    "upstream_model_source_version": "1.1.0.1",
    "upstream_model_source_url": (
        "https://www.mathworks.com/matlabcentral/fileexchange/"
        "66189-feedback-controlled-cstr-process-for-fault-simulation"
    ),
    "upstream_model_license_url": (
        "https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/"
        "89e57c22-64a7-4cba-8aa3-da4279b09619/"
        "cb37e495-cbbf-47ab-90eb-605e5328de59/license/license.txt"
    ),
    "upstream_model_license_text_sha256": (
        "8c89de130c1e25815100e4dd5dcc3a9b602a74ee9b94f3eebf3513c53945b39e"
    ),
    "upstream_model_local_sha256": (
        "4555be2fa4c93ab43d2f24ab26e2bf6511ec25d701b231fc7d57f6657b523a81"
    ),
    "upstream_model_notice_file": (
        "datasets/cards/oa/cstr_closed_loop_fd/THIRD_PARTY_NOTICE.txt"
    ),
}
_PRIVATE_ACCESS = {
    "tag": "private",
    "disclosure": "non_public",
    "reason": "hydrocracking_proprietary",
}


@dataclass(frozen=True)
class FaultDatasetProtocol:
    """描述一个过程故障数据变体中不会从数组数值猜测的物理协议。

    参数：
        feature_names: 原始矩阵各列按存储顺序对应的物理变量名。
        feature_roles: 与 ``feature_names`` 一一对应的 schema 角色。
        fault_onset: episode 内的故障开始采样索引；该值必须来自数据说明。
        fault_families: ``(fault_id, family)`` 对，用于场景元数据而非模型特征。
        description_file: 保存变量、onset 和来源说明的相对文件名。
        stored_feature_indices: 从原始矩阵选择过程变量的零基列号；为空时要求原始矩阵
            宽度与 ``feature_names`` 完全一致。该字段用于排除随数据一同保存、但属于
            答案信息的故障输出通道，绝不能根据故障性能动态选择。
        normal_stored_width/fault_stored_width: 可选的正常/故障原始矩阵精确列宽。发布版
            同时携带答案列时，必须先按 normal/fault 身份核对完整宽度再做静态列选择。
    副作用：
        无。对象只保存已核验的静态协议，不能读取数据或自动推断列语义。
    """

    feature_names: tuple[str, ...]
    feature_roles: tuple[str, ...]
    fault_onset: int
    fault_families: tuple[tuple[int, str], ...]
    description_file: str
    stored_feature_indices: tuple[int, ...] | None = None
    normal_stored_width: int | None = None
    fault_stored_width: int | None = None

    def __post_init__(self) -> None:
        """在适配器读取任何数据前拒绝不完整或自相矛盾的协议。

        异常：
            变量名与角色/存储列数量不等，存储列重复或为负数，声明宽度不能容纳选中列，
            或 ``fault_onset`` 为负数时抛出 ``ValueError``。
        副作用：
            无。只检查冻结字段，不读取数据，也不修改对象。
        """

        if len(self.feature_names) != len(self.feature_roles):
            raise ValueError(
                "Fault dataset protocol feature_names and feature_roles must have equal length. "
                f"Current lengths: {len(self.feature_names)} and {len(self.feature_roles)}."
            )
        if self.fault_onset < 0:
            raise ValueError(
                "Fault dataset protocol fault_onset must be non-negative. "
                f"Current input: {self.fault_onset}."
            )
        indices = self.stored_feature_indices
        widths = (self.normal_stored_width, self.fault_stored_width)
        if any(width is not None and width <= 0 for width in widths):
            raise ValueError(
                "Fault dataset protocol stored widths must be positive integers."
            )
        if indices is not None:
            if len(indices) != len(self.feature_names):
                raise ValueError(
                    "Fault dataset protocol stored_feature_indices must match feature_names. "
                    f"Current lengths: {len(indices)} and {len(self.feature_names)}."
                )
            if any(index < 0 for index in indices) or len(set(indices)) != len(indices):
                raise ValueError(
                    "Fault dataset protocol stored_feature_indices must be unique "
                    "non-negative integers."
                )
            if any(
                width is not None and indices and max(indices) >= width
                for width in widths
            ):
                raise ValueError(
                    "Fault dataset protocol stored width cannot contain all selected columns."
                )

    def input_roles(self) -> tuple[str, ...]:
        """按首次出现顺序返回任务允许读取的物理输入角色。

        返回：
            去重但保持原始列顺序的角色元组，例如闭环 CSTR 返回
            ``("control_input", "measured_output")``。
        异常：
            正常情况下不抛异常；角色长度错误已在协议构造阶段被拒绝。
        副作用：
            无。结果只约束 TaskView 选列，不读取数据或修改协议。
        """

        return tuple(dict.fromkeys(self.feature_roles))

    def role_indices(self, role: str) -> tuple[int, ...]:
        """返回某一物理 schema 角色在发布过程变量中的固定列号。

        参数：
            role: 例如 ``control_input`` 或 ``measured_output`` 的 schema 角色名。
        返回：
            按原始发布列序排列的零基索引；协议不含该角色时返回空元组。
        异常：
            无；角色字符串只做相等比较，不触发数据或注册表访问。
        副作用：
            无。配置层可据此派生 feature layout，避免再次硬编码同一物理合同。
        """

        return tuple(
            index
            for index, declared_role in enumerate(self.feature_roles)
            if declared_role == role
        )

    def fault_family(self, fault_id: int) -> str:
        """把数据说明中的故障编号映射为结构化故障族。

        参数：
            fault_id: episode 的整数故障编号；0 专门表示正常状态。
        返回：
            ``normal``、``process``、``actuator`` 或 ``sensor`` 等协议声明名称。
        异常：
            编号既不是 0 也不在 ``fault_families`` 中时抛出 ``ValueError``，
            防止未知场景被静默并入正常类。
        副作用：
            无。返回值只写入追溯元数据，不参与特征构造。
        """

        if fault_id == 0:
            return "normal"
        families = dict(self.fault_families)
        if fault_id not in families:
            legal = ", ".join(str(item) for item in sorted(families))
            raise ValueError(
                f"Unknown fault dataset protocol fault_id {fault_id}. "
                f"Legal options are: 0, {legal}."
            )
        return families[fault_id]

    def select_stored_features(
        self,
        values: np.ndarray,
        *,
        normal: bool,
    ) -> np.ndarray:
        """按静态发布协议选择可作为过程变量的存储列。

        参数：
            values: 已清理的二维原始数组。它可能额外包含故障输出等受保护答案通道。
            normal: 当前 episode 是否来自正常发布文件；只用于选择协议声明的精确列宽，
                不读取标签或数值表现。
        返回：
            只含 ``feature_names`` 对应列的新数组视图；列顺序严格等于协议声明顺序。
        异常：
            原始宽度不等于 normal/fault 发布协议、未声明选择时宽度不等于物理变量数，
            或声明列号超出矩阵宽度时抛出 ``ValueError``，防止静默截断或把答案列送入
            模型。
        副作用：
            无。不修改输入数组、不读取文件，也不根据数值内容推断列语义。
        """

        expected_width = (
            self.normal_stored_width if normal else self.fault_stored_width
        )
        if expected_width is not None and values.shape[1] != expected_width:
            split_name = "normal" if normal else "fault"
            raise ValueError(
                f"Fault dataset {split_name} raw width differs from the published protocol. "
                f"Current columns={values.shape[1]}, declared={expected_width}."
            )
        indices = self.stored_feature_indices
        if indices is None:
            if values.shape[1] != len(self.feature_names):
                raise ValueError(
                    "Fault dataset raw width differs from the declared physical variables. "
                    f"Current columns={values.shape[1]}, declared={len(self.feature_names)}."
                )
            return values
        if indices and max(indices) >= values.shape[1]:
            raise ValueError(
                "Fault dataset stored feature index lies outside the raw matrix width. "
                f"Maximum index={max(indices)}, columns={values.shape[1]}."
            )
        return values[:, indices]

    def summary(self) -> dict[str, Any]:
        """生成可直接写入来源 manifest 的静态协议摘要。

        返回：
            包含按角色分组的 ``variables``、``fault_onset`` 与字符串键
            ``fault_families`` 的 JSON 可序列化字典。
        异常：
            正常情况下不抛异常；名称与角色错位已在构造阶段被拒绝。
        副作用：
            无。每次调用都新建字典，调用方修改返回值不会改变协议对象。
        """

        variables: dict[str, list[str]] = {}
        for name, role in zip(self.feature_names, self.feature_roles, strict=True):
            variables.setdefault(role, []).append(name)
        summary = {
            "variables": variables,
            "fault_onset": self.fault_onset,
            "fault_families": {
                str(fault_id): family for fault_id, family in self.fault_families
            },
        }
        if self.stored_feature_indices is not None:
            summary["stored_feature_indices"] = list(self.stored_feature_indices)
        if self.normal_stored_width is not None or self.fault_stored_width is not None:
            summary["stored_widths"] = {
                "normal": self.normal_stored_width,
                "fault": self.fault_stored_width,
            }
        return summary


# P1 已在内部使用过 CSTR 专名；保留别名可避免外部研究脚本在通用化后失效。
CSTRFaultProtocol = FaultDatasetProtocol


_CSTR_CLOSED_LOOP_PROTOCOL = FaultDatasetProtocol(
    feature_names=("Ci", "Ti", "Tci", "C", "T", "Tc", "Qc"),
    feature_roles=(
        "control_input",
        "control_input",
        "control_input",
        "measured_output",
        "measured_output",
        "measured_output",
        "measured_output",
    ),
    fault_onset=200,
    fault_families=(
        (1, "process"),
        (2, "process"),
        (3, "actuator"),
        (4, "actuator"),
        (5, "actuator"),
        (6, "sensor"),
        (7, "sensor"),
        (8, "sensor"),
    ),
    description_file="7v 正序, 8c 故障.txt",
    normal_stored_width=7,
    fault_stored_width=7,
)

TTS_SIX_FAULT_PROTOCOL = FaultDatasetProtocol(
    feature_names=("Q1", "Q2", "Q1s", "Q2s", "h1", "h2", "h3"),
    feature_roles=(
        "control_input",
        "control_input",
        "measured_output",
        "measured_output",
        "measured_output",
        "measured_output",
        "measured_output",
    ),
    fault_onset=200,
    fault_families=(
        (1, "actuator"),
        (2, "actuator"),
        (3, "actuator"),
        (4, "sensor"),
        (5, "sensor"),
        (6, "sensor"),
    ),
    description_file="7v+6f 正序, 6c 故障.txt",
    stored_feature_indices=tuple(range(7)),
    normal_stored_width=7,
    fault_stored_width=13,
)


class TEFaultDiagnosisAdapter:
    """Read the real Tennessee Eastman fault-diagnosis files."""

    name = "te_fault_diagnosis"
    version = "real-v1"
    description = "Tennessee Eastman fault-diagnosis dataset."

    def __init__(self) -> None:
        self._fallback = SyntheticProcessAdapter(
            name=self.name,
            task_name="fault_diagnosis",
            description="Deterministic TE-style fault-diagnosis smoke dataset.",
            domain="tennessee_eastman",
        )

    def read(self, *, root: str | Path | None = None, task: str | None = None) -> CanonicalDataset:
        if root is None:
            return self._fallback.read(root=None, task=task)
        root_path = _resolve_dataset_root(root, "fd", required=("train/Mode1_Normal.mat",))
        train_array = _first_array(read_mat_arrays(root_path / "train" / "Mode1_Normal.mat"), "Normal")
        train = (
            _segment(
                _feature_frame(train_array, label=0, segment_id="Normal"),
                split="train",
                source=root_path / "train" / "Mode1_Normal.mat",
                segment_id="Normal",
                metadata={"fault_id": 0},
            ),
        )
        test_segments: list[Segment] = []
        npy_paths = sorted((root_path / "test").glob("Fault*.npy"))
        if npy_paths:
            for path in npy_paths:
                fault_id = _fault_id_from_name(path.stem)
                test_segments.append(
                    _segment(
                        _feature_frame(np.load(path), label=fault_id, segment_id=path.stem),
                        split="test",
                        source=path,
                        segment_id=path.stem,
                        metadata={"fault_id": fault_id},
                    )
                )
        else:
            for key, array in sorted(read_mat_arrays(root_path / "test" / "Mode1_Faulty.mat").items()):
                fault_id = _fault_id_from_name(key)
                test_segments.append(
                    _segment(
                        _feature_frame(array, label=fault_id, segment_id=key),
                        split="test",
                        source=root_path / "test" / "Mode1_Faulty.mat",
                        segment_id=key,
                        metadata={"fault_id": fault_id, "mat_key": key},
                    )
                )
        return _canonical(
            name=self.name,
            version=self.version,
            root=root_path,
            splits={"train": train, "test": tuple(test_segments)},
            schema=_fault_schema(train_array.shape[1], domain="tennessee_eastman"),
            access=_OA_ACCESS,
        )

    def schema(self) -> DataSchema:
        return _fault_schema(33, domain="tennessee_eastman")

    def default_task(self, task: str | None = None) -> TaskSchema:
        return _fault_task(task)

    def default_pipeline(self, task: str | None = None) -> dict[str, Any]:
        self.default_task(task)
        return _standard_official_pipeline()

    def summary(self, task: str | None = None) -> dict[str, Any]:
        return _summary(self, self.default_task(task), access=_OA_ACCESS, files={"root": "TE/fd"})


class TEClassificationAdapter:
    """Read the real Tennessee Eastman DAT classification files."""

    name = "te_classification"
    version = "real-v1"
    description = "Tennessee Eastman classification dataset."

    def __init__(self) -> None:
        self._fallback = SyntheticProcessAdapter(
            name=self.name,
            task_name="classification",
            description="Deterministic TE-style classification smoke dataset.",
            domain="tennessee_eastman",
        )

    def read(self, *, root: str | Path | None = None, task: str | None = None) -> CanonicalDataset:
        if root is None:
            return self._fallback.read(root=None, task=task)
        root_path = _resolve_dataset_root(root, "cls", required=("train/d00.dat",))
        train_segments = tuple(_te_dat_segment(path, "train") for path in sorted((root_path / "train").glob("d*.dat")))
        test_segments = tuple(_te_dat_segment(path, "test") for path in sorted((root_path / "test").glob("d*_te.dat")))
        if not train_segments or not test_segments:
            raise FileNotFoundError(f"TE classification files were not found under {root_path}.")
        feature_count = int(train_segments[0].frame.filter(regex=r"^x").shape[1])
        return _canonical(
            name=self.name,
            version=self.version,
            root=root_path,
            splits={"train": train_segments, "test": test_segments},
            schema=_classification_schema(feature_count, domain="tennessee_eastman"),
            access=_OA_ACCESS,
        )

    def schema(self) -> DataSchema:
        return _classification_schema(52, domain="tennessee_eastman")

    def default_task(self, task: str | None = None) -> TaskSchema:
        task_name = task or "classification"
        if task_name != "classification":
            raise ValueError(f"TE classification supports only task 'classification', got {task_name!r}.")
        return TaskSchema(
            name="classification",
            inputs=("input",),
            targets=("label",),
            label_column="label",
            normal_label=0,
        )

    def default_pipeline(self, task: str | None = None) -> dict[str, Any]:
        self.default_task(task)
        return _standard_official_pipeline()

    def summary(self, task: str | None = None) -> dict[str, Any]:
        return _summary(self, self.default_task(task), access=_OA_ACCESS, files={"root": "TE/cls"})


@dataclass(frozen=True)
class CSTRFaultAdapter:
    """读取 CSTR 故障诊断 MAT 文件并保留变体特有的物理协议。

    ``protocol=None`` 时维持历史通用 CSTR 行为；闭环 7 变量 preset 显式传入
    :class:`CSTRFaultProtocol`，从而避免根据矩阵宽度猜测变量名、onset 或故障族。
    适配器只负责数据语义和来源，不计算检测分数或使用故障性能调节任何参数。
    """

    name: str = "cstr_fault_diagnosis"
    subdir: str = "fd"
    feature_count: int = 10
    description: str = "Feedback-controlled CSTR fault-diagnosis dataset."
    version: str = "real-v1"
    protocol: CSTRFaultProtocol | None = None
    _fallback: SyntheticCSTRFaultAdapter | SyntheticProcessAdapter = field(
        init=False,
        repr=False,
        compare=False,
    )

    @classmethod
    def closed_loop(cls) -> "CSTRFaultAdapter":
        """构造使用仓库闭环 CSTR 协议常量的适配器。

        返回：
            尚未读取磁盘的 ``CSTRFaultAdapter``；七个物理变量与 onset=200 来自
            ``_CSTR_CLOSED_LOOP_PROTOCOL``。
        异常：
            协议列数与 ``feature_count=7`` 不一致时，构造阶段抛出 ``ValueError``。
        副作用：
            无。不读取数据、不检查许可、不创建目录。
        """

        return cls(
            name="cstr_closed_loop_fd",
            subdir="fd_close",
            feature_count=7,
            description="Closed-loop CSTR fault-diagnosis dataset.",
            protocol=_CSTR_CLOSED_LOOP_PROTOCOL,
        )

    def __post_init__(self) -> None:
        """安装与变体匹配的确定性 smoke fallback。

        返回：
            无。
        异常：
            fallback 构造失败时传播其配置异常；本方法不吞掉错误。
        副作用：
            仅通过 ``object.__setattr__`` 写入冻结 dataclass 的内部 ``_fallback`` 引用。
            不读取文件、不生成数据，也不注册全局状态。
        """

        fallback = (
            SyntheticCSTRFaultAdapter()
            if self.name == "cstr_fault_diagnosis"
            else SyntheticProcessAdapter(
                name=self.name,
                task_name="fault_diagnosis",
                description="Deterministic closed-loop CSTR fault-diagnosis smoke dataset.",
                domain="process_control",
            )
        )
        object.__setattr__(self, "_fallback", fallback)

    def _access_metadata(self) -> dict[str, str]:
        """返回与当前 CSTR 变体一致的独立访问元数据副本。

        返回：
            闭环七变量变体把本地 MAT 的 ``to_verify`` 状态与已核验的上游模型
            BSD-3-Clause 证据分开返回；其他历史 CSTR 变体继续只返回 ``to_verify``，
            不能借用闭环模型的证据。
        异常：
            无。
        副作用：
            只复制模块级字典，不访问网络或原始数据；调用方修改结果不会污染后续摘要。
        """

        access = (
            _CSTR_CLOSED_LOOP_ACCESS
            if self.name == "cstr_closed_loop_fd"
            and self.protocol == _CSTR_CLOSED_LOOP_PROTOCOL
            else _OA_ACCESS
        )
        return dict(access)

    def read(self, *, root: str | Path | None = None, task: str | None = None) -> CanonicalDataset:
        """读取官方 train/test MAT，并按可选物理协议生成标签与来源摘要。

        参数：
            root: CSTR 数据家族根目录；省略时返回不读取真实文件的 smoke fallback。
            task: 仅接受 ``fault_diagnosis``。
        返回：
            train/test segment、schema 和来源信息组成的 ``CanonicalDataset``。
        异常：
            文件缺失时抛出 ``FileNotFoundError``；矩阵列数、fault key、任务名称或
            协议边界不一致时抛出 ``ValueError``；MAT 读取错误按原类型传播。
        副作用：
            ``root`` 非空时只读 MAT 与说明文件并计算 SHA-256；不写文件、不访问网络，
            不使用故障值拟合预处理或选择模型。
        """

        if root is None and self.protocol is not None:
            return _closed_loop_cstr_smoke_dataset(self)
        if root is None:
            return self._fallback.read(root=None, task=task)
        root_path = _resolve_dataset_root(root, self.subdir, required=("train/model1[train].mat",))
        train_path = root_path / "train" / "model1[train].mat"
        test_path = root_path / "test" / "model1[test].mat"
        protocol_metadata: dict[str, Any] | None = None
        train_sha256: str | None = None
        test_sha256: str | None = None
        if self.protocol is not None:
            description_path = root_path / self.protocol.description_file
            files = {
                "train": _file_summary(train_path),
                "test": _file_summary(test_path),
                "description": _file_summary(description_path),
            }
            train_sha256 = str(files["train"]["sha256"])
            test_sha256 = str(files["test"]["sha256"])
            protocol_metadata = {**self.protocol.summary(), "files": files}
        train_segments = _mat_fault_segments(
            train_path,
            split="train",
            normal=True,
            protocol=self.protocol,
            source_sha256=train_sha256,
        )
        test_segments = _mat_fault_segments(
            test_path,
            split="test",
            normal=False,
            protocol=self.protocol,
            source_sha256=test_sha256,
        )
        return _canonical(
            name=self.name,
            version=self.version,
            root=root_path,
            splits={"train": train_segments, "test": test_segments},
            schema=self.schema(),
            access=self._access_metadata(),
            metadata=protocol_metadata,
        )

    def schema(self) -> DataSchema:
        """返回 CSTR 原始矩阵对应的规范列角色。

        返回：
            ``DataSchema``。闭环变体额外含 ``raw_index``，并把七列明确区分为
            ``control_input`` 与 ``measured_output``；旧变体维持通用 input 列。
        异常：
            物理变量名、角色数量与矩阵宽度不一致时由 ``_fault_schema`` 抛出
            ``ValueError``。
        副作用：
            无。方法不读取数据，返回的新 schema 只描述列语义。
        """

        if self.protocol is None:
            return _fault_schema(self.feature_count, domain="process_control")
        return _fault_schema(
            self.feature_count,
            domain="process_control",
            feature_names=self.protocol.feature_names,
            feature_roles=self.protocol.feature_roles,
            include_raw_index=True,
        )

    def default_task(self, task: str | None = None) -> TaskSchema:
        """返回适配器支持的故障诊断任务及输入角色约束。

        参数：
            task: 任务名称或 ``None``；本适配器只接受 ``fault_diagnosis``。
        返回：
            以 ``fault_id`` 为标签的 ``TaskSchema``。闭环协议同时写入物理输入角色
            与 ``fault_switch``，排除 raw_index、segment 和标签。
        异常：
            任务名不受支持时由 ``_fault_task`` 抛出 ``ValueError``。
        副作用：
            无。不读取数据，也不访问故障 split。
        """

        if self.protocol is None:
            return _fault_task(task)
        return _fault_task(
            task,
            inputs=self.protocol.input_roles(),
            fault_switch=self.protocol.fault_onset,
        )

    def default_pipeline(self, task: str | None = None) -> dict[str, Any]:
        self.default_task(task)
        return _standard_official_pipeline()

    def summary(self, task: str | None = None) -> dict[str, Any]:
        """返回 CSTR preset、任务、访问状态和预期文件位置摘要。

        参数：
            task: 可选任务名，必须符合 ``default_task`` 约束。
        返回：
            JSON 可序列化字典；真实文件 hash 与逐 episode 统计在 ``read`` 后的
            ``CanonicalDataset.source_summary`` 中提供。
        异常：
            任务名不受支持时抛出 ``ValueError``。
        副作用：
            无。不检查文件存在性、不访问网络；闭环变体只重放已核验的上游模型
            BSD-3-Clause 证据，同时对本地 MAT 保持 ``to_verify``。其他 CSTR 变体
            继续只保留 ``to_verify``。
        """

        return _summary(
            self,
            self.default_task(task),
            access=self._access_metadata(),
            files={"root": f"CSTR/{self.subdir}"},
        )


class TTSFaultDiagnosisAdapter:
    """读取论文 P11 使用的三容水箱七变量、六故障 MAT 发布变体。

    该适配器把 ``fe`` 目录中前七列过程变量映射为两个控制输入和五个测量输出，并在
    onset=200 后赋故障标签。测试 MAT 随附的后六列故障输出是答案信息，读取边界会按
    静态列协议排除它们；适配器不训练模型、不拟合阈值，也不比较 CSTR/TTS 性能。
    """

    name = "tts_fault_diagnosis"
    version = "real-v1"
    description = "Three-tank-system seven-variable, six-fault diagnosis dataset."
    protocol = TTS_SIX_FAULT_PROTOCOL

    def read(self, *, root: str | Path | None = None, task: str | None = None) -> CanonicalDataset:
        """读取 TTS 正常训练与六个故障 episode，并恢复物理语义和来源追溯。

        参数：
            root: TTS 家族根目录或直接的 ``fe`` 目录；为空时只返回 smoke fallback。
            task: 仅接受 ``fault_diagnosis``。
        返回：
            具有物理 schema、raw_index、onset 标签和文件 SHA-256 的规范数据集。
        异常：
            任务错误、文件缺失、MAT 结构/列宽/故障编号不符时抛出
            ``ValueError``、``FileNotFoundError`` 或底层读取异常。
        副作用：
            ``root`` 非空时读取两个 MAT 与说明文件并计算 hash；不写文件、不访问网络，
            不把故障值用于任何拟合、校准或结构选择。
        """

        if root is None:
            self.default_task(task)
            return _tts_six_fault_smoke_dataset(self)
        self.default_task(task)
        root_path = _resolve_dataset_root(root, "fe", required=("train/[train].mat",))
        train_path = root_path / "train" / "[train].mat"
        test_path = root_path / "test" / "[test].mat"
        description_path = root_path / self.protocol.description_file
        files = {
            "train": _file_summary(train_path),
            "test": _file_summary(test_path),
            "description": _file_summary(description_path),
        }
        train_segments = _mat_fault_segments(
            train_path,
            split="train",
            normal=True,
            protocol=self.protocol,
            source_sha256=str(files["train"]["sha256"]),
        )
        test_segments = _mat_fault_segments(
            test_path,
            split="test",
            normal=False,
            protocol=self.protocol,
            source_sha256=str(files["test"]["sha256"]),
        )
        return _canonical(
            name=self.name,
            version=self.version,
            root=root_path,
            splits={"train": train_segments, "test": test_segments},
            schema=self.schema(),
            access=_OA_ACCESS,
            metadata={**self.protocol.summary(), "files": files},
        )

    def schema(self) -> DataSchema:
        """返回七个物理变量与不可作为模型输入的追溯/标签列。

        返回：
            两列 ``control_input``、五列 ``measured_output``，以及 time、raw_index、
            segment、fault_id 的 ``DataSchema``。
        异常：
            协议名称、角色和列数不一致时由 ``_fault_schema`` 抛出 ``ValueError``。
        副作用：
            无。不读取 MAT，也不暴露后六列故障输出。
        """

        return _fault_schema(
            len(self.protocol.feature_names),
            domain="process_control",
            feature_names=self.protocol.feature_names,
            feature_roles=self.protocol.feature_roles,
            include_raw_index=True,
        )

    def default_task(self, task: str | None = None) -> TaskSchema:
        """返回只允许七个物理过程变量的故障诊断任务。

        参数：
            task: ``fault_diagnosis`` 或 ``None``。
        返回：
            写明输入角色、标签列、正常标签和 onset=200 的 ``TaskSchema``。
        异常：
            其他任务名由 ``_fault_task`` 拒绝。
        副作用：
            无。
        """

        return _fault_task(
            task,
            inputs=self.protocol.input_roles(),
            fault_switch=self.protocol.fault_onset,
        )

    def default_pipeline(self, task: str | None = None) -> dict[str, Any]:
        """返回官方 split 与仅正常训练拟合的标准预处理声明。"""

        self.default_task(task)
        return _standard_official_pipeline()

    def summary(self, task: str | None = None) -> dict[str, Any]:
        """返回不访问文件的 TTS preset、任务、许可与预期目录摘要。"""

        return _summary(
            self,
            self.default_task(task),
            access=_OA_ACCESS,
            files={"root": "TTS/fe"},
        )


@dataclass(frozen=True)
class NpyReconstructionAdapter:
    """Read Normal/Fault*.npy reconstruction datasets."""

    name: str
    subdir: str
    raw_folder: str
    description: str
    domain: str = "process_control"
    feature_count: int = 5
    version: str = "real-v1"

    def read(self, *, root: str | Path | None = None, task: str | None = None) -> CanonicalDataset:
        if root is None:
            return SyntheticProcessAdapter(
                name=self.name,
                task_name="prediction",
                description=self.description,
                domain=self.domain,
            ).read(root=None, task="prediction")
        root_path = _resolve_dataset_root(root, self.subdir, required=("train/Normal.npy",))
        train_segments = (
            _segment(
                _feature_frame(np.load(root_path / "train" / "Normal.npy"), segment_id="Normal"),
                split="train",
                source=root_path / "train" / "Normal.npy",
                segment_id="Normal",
            ),
        )
        test_segments = tuple(
            _segment(
                _feature_frame(np.load(path), segment_id=path.stem),
                split="test",
                source=path,
                segment_id=path.stem,
            )
            for path in sorted((root_path / "test").glob("Fault*.npy"))
        )
        if not test_segments:
            raise FileNotFoundError(f"No Fault*.npy files were found under {root_path / 'test'}.")
        return _canonical(
            name=self.name,
            version=self.version,
            root=root_path,
            splits={"train": train_segments, "test": test_segments},
            schema=_reconstruction_schema(self.feature_count, domain=self.domain),
            access=_OA_ACCESS,
        )

    def schema(self) -> DataSchema:
        return _reconstruction_schema(self.feature_count, domain=self.domain)

    def default_task(self, task: str | None = None) -> TaskSchema:
        task_name = task or "reconstruction"
        if task_name != "reconstruction":
            raise ValueError(f"{self.name} supports only task 'reconstruction', got {task_name!r}.")
        return TaskSchema(name="reconstruction")

    def default_pipeline(self, task: str | None = None) -> dict[str, Any]:
        self.default_task(task)
        return _standard_official_pipeline()

    def summary(self, task: str | None = None) -> dict[str, Any]:
        return _summary(
            self,
            self.default_task(task),
            access=_OA_ACCESS,
            files={"root": self.raw_folder},
        )


class MultiphaseFaultAdapter:
    """Read the Multiphase Flow Facility normal and faulty MAT files."""

    name = "multiphase_fd"
    version = "real-v1"
    description = "Multiphase Flow Facility fault-diagnosis dataset."

    def __init__(self) -> None:
        self._fallback = SyntheticProcessAdapter(
            name=self.name,
            task_name="fault_diagnosis",
            description="Deterministic multiphase-flow fault-diagnosis smoke dataset.",
            domain="multiphase_flow",
        )

    def read(self, *, root: str | Path | None = None, task: str | None = None) -> CanonicalDataset:
        if root is None:
            return self._fallback.read(root=None, task=task)
        root_path = Path(root)
        if not (root_path / "train" / "Training.mat").exists():
            root_path = root_path / "Multiphase_Flow_Facility"
        _require_file(root_path / "train" / "Training.mat")
        train_segments = tuple(
            _segment(
                _feature_frame(array, label=0, segment_id=key),
                split="train",
                source=root_path / "train" / "Training.mat",
                segment_id=key,
                metadata={"fault_id": 0, "mat_key": key},
            )
            for key, array in sorted(read_mat_arrays(root_path / "train" / "Training.mat").items())
            if key.lower().startswith("normal")
        )
        test_segments: list[Segment] = []
        for path in sorted((root_path / "test").glob("FaultyCase*.mat")):
            arrays = read_mat_arrays(path)
            case_id = _fault_id_from_name(path.stem)
            for key, array in sorted(arrays.items()):
                if not key.startswith("Set"):
                    continue
                suffix = key.removeprefix("Set")
                label_key = f"EvoFault{suffix}"
                labels = arrays.get(label_key)
                if labels is None:
                    label = np.full(_as_2d(array).shape[0], case_id)
                else:
                    label = np.asarray(labels).reshape(-1)
                    label = np.where(label > 0, case_id, 0)
                segment_id = f"{path.stem}:{key}"
                test_segments.append(
                    _segment(
                        _feature_frame(array, label=label, segment_id=segment_id),
                        split="test",
                        source=path,
                        segment_id=segment_id,
                        metadata={"fault_case": case_id, "mat_key": key, "label_key": label_key},
                    )
                )
        if not train_segments or not test_segments:
            raise FileNotFoundError(f"Multiphase train/test segments were not found under {root_path}.")
        return _canonical(
            name=self.name,
            version=self.version,
            root=root_path,
            splits={"train": train_segments, "test": tuple(test_segments)},
            schema=_fault_schema(24, domain="multiphase_flow"),
            access=_OA_ACCESS,
        )

    def schema(self) -> DataSchema:
        return _fault_schema(24, domain="multiphase_flow")

    def default_task(self, task: str | None = None) -> TaskSchema:
        return _fault_task(task)

    def default_pipeline(self, task: str | None = None) -> dict[str, Any]:
        self.default_task(task)
        return _standard_official_pipeline()

    def summary(self, task: str | None = None) -> dict[str, Any]:
        return _summary(self, self.default_task(task), access=_OA_ACCESS, files={"root": "Multiphase_Flow_Facility"})


class WPTMPCAdapter:
    """Read WPT PRBS input/output data for MPC windows."""

    name = "wpt_mpc"
    version = "real-v1"
    description = "WPT PRBS MPC dataset."

    def __init__(self) -> None:
        self._fallback = SyntheticProcessAdapter(
            name=self.name,
            task_name="mpc",
            description="Deterministic WPT-style MPC smoke dataset.",
            domain="model_predictive_control",
        )

    def read(self, *, root: str | Path | None = None, task: str | None = None) -> CanonicalDataset:
        if root is None:
            return self._fallback.read(root=None, task=task)
        root_path = Path(root)
        if not (root_path / "PRBS_1024_u.mat").exists():
            root_path = root_path / "WPT"
        _require_file(root_path / "PRBS_1024_u.mat")
        u = _first_array(read_mat_arrays(root_path / "PRBS_1024_u.mat"), "uk")
        y = _first_array(read_mat_arrays(root_path / "PRBS_1024_y.mat"), "yk")
        frame = _wpt_frame(u, y)
        split_at = max(16, int(frame.shape[0] * 0.8))
        train_frame = frame.iloc[:split_at].reset_index(drop=True)
        test_frame = frame.iloc[split_at:].reset_index(drop=True)
        return _canonical(
            name=self.name,
            version=self.version,
            root=root_path,
            splits={
                "train": (
                    _segment(train_frame, split="train", source=root_path / "PRBS_1024_u.mat", segment_id="prbs_train"),
                ),
                "test": (
                    _segment(test_frame, split="test", source=root_path / "PRBS_1024_y.mat", segment_id="prbs_test"),
                ),
            },
            schema=self.schema(),
            access=_OA_ACCESS,
        )

    def schema(self) -> DataSchema:
        return DataSchema(
            columns=(
                ColumnSpec("episode", role="episode"),
                ColumnSpec("state1", role="state"),
                ColumnSpec("control1", role="control"),
                ColumnSpec("output1", role="output"),
                ColumnSpec("reference1", role="reference"),
            ),
            sample_rate=1.0,
            metadata={"domain": "model_predictive_control"},
        )

    def default_task(self, task: str | None = None) -> TaskSchema:
        task_name = task or "mpc"
        if task_name != "mpc":
            raise ValueError(f"WPT adapter supports only task 'mpc', got {task_name!r}.")
        return TaskSchema(name="mpc", inputs=("state", "output", "control", "reference"), targets=("output",))

    def default_pipeline(self, task: str | None = None) -> dict[str, Any]:
        self.default_task(task)
        return {
            "split": {"type": "official"},
            "normalization": {"method": "standard"},
            "mpc_window": {
                "past_horizon": 2,
                "prediction_horizon": 2,
                "control_horizon": 2,
                "return_mode": "tuple",
            },
        }

    def summary(self, task: str | None = None) -> dict[str, Any]:
        return _summary(self, self.default_task(task), access=_OA_ACCESS, files={"root": "WPT"})


class HYFaultAdapter:
    """Read private hydrocracking NPZ fault-diagnosis splits."""

    name = "hy_fault_diagnosis"
    version = "real-v1"
    description = "Private hydrocracking fault-diagnosis dataset."

    def __init__(self) -> None:
        self._fallback = SyntheticProcessAdapter(
            name=self.name,
            task_name="fault_diagnosis",
            description="Deterministic HY-style fault-diagnosis smoke dataset.",
            domain="hydrocracking",
        )

    def read(self, *, root: str | Path | None = None, task: str | None = None) -> CanonicalDataset:
        if root is None:
            return self._fallback.read(root=None, task=task)
        root_path = _resolve_dataset_root(root, "fd", required=("train/train_X.npz",))
        train_x = read_npz_arrays(root_path / "train" / "train_X.npz")
        test_x = read_npz_arrays(root_path / "test" / "test_X.npz")
        test_y = read_npz_arrays(root_path / "test" / "test_Y.npz")
        train_segments = tuple(
            _segment(
                _feature_frame(array, label=0, segment_id=key),
                split="train",
                source=root_path / "train" / "train_X.npz",
                segment_id=key,
                metadata={"fault_id": 0, "npz_key": key},
            )
            for key, array in sorted(train_x.items())
        )
        test_segments = []
        for key, array in sorted(test_x.items()):
            label_key = key.replace("_X", "_Y")
            labels = test_y.get(label_key)
            if labels is None:
                raise KeyError(f"Missing HY label key {label_key!r} for feature key {key!r}.")
            test_segments.append(
                _segment(
                    _feature_frame(array, label=np.asarray(labels).reshape(-1), segment_id=key),
                    split="test",
                    source=root_path / "test" / "test_X.npz",
                    segment_id=key,
                    metadata={"npz_key": key, "label_key": label_key},
                )
            )
        feature_count = int(next(iter(train_x.values())).shape[1])
        return _canonical(
            name=self.name,
            version=self.version,
            root=root_path,
            splits={"train": train_segments, "test": tuple(test_segments)},
            schema=_fault_schema(feature_count, domain="hydrocracking", access=_PRIVATE_ACCESS),
            access=_PRIVATE_ACCESS,
        )

    def schema(self) -> DataSchema:
        return _fault_schema(52, domain="hydrocracking", access=_PRIVATE_ACCESS)

    def default_task(self, task: str | None = None) -> TaskSchema:
        return _fault_task(task)

    def default_pipeline(self, task: str | None = None) -> dict[str, Any]:
        self.default_task(task)
        return _standard_official_pipeline()

    def summary(self, task: str | None = None) -> dict[str, Any]:
        return _summary(self, self.default_task(task), access=_PRIVATE_ACCESS, files={"root": "HY/fd"})


class HYQualityPredictionAdapter:
    """Read private hydrocracking product-quality CSV files."""

    name = "hy_quality_prediction"
    version = "real-v1"
    description = "Private hydrocracking product-quality prediction dataset."

    def __init__(self) -> None:
        self._fallback = SyntheticProcessAdapter(
            name=self.name,
            task_name="prediction",
            description="Deterministic HY quality-prediction smoke dataset.",
            domain="hydrocracking",
        )

    def read(self, *, root: str | Path | None = None, task: str | None = None) -> CanonicalDataset:
        if root is None:
            return self._fallback.read(root=None, task=task)
        root_path = Path(root)
        if not (root_path / "郑迪_prd" / "2017年后数据").exists():
            root_path = root_path / "HY_PRD"
        csv_paths = sorted((root_path / "郑迪_prd" / "2017年后数据").glob("*.csv"))
        if not csv_paths:
            raise FileNotFoundError(f"No HY_PRD CSV files were found under {root_path}.")
        prepared = [_hy_prd_arrays(path) for path in csv_paths]
        feature_count = min(item[0].shape[1] for item in prepared)
        train_segments: list[Segment] = []
        test_segments: list[Segment] = []
        for path, (x, y, target_name) in zip(csv_paths, prepared, strict=True):
            x = x[:, :feature_count]
            split_at = max(1, int(x.shape[0] * 0.8))
            split_at = min(split_at, x.shape[0] - 1)
            train_segments.append(
                _segment(
                    _prediction_frame(x[:split_at], y[:split_at], segment_id=path.stem),
                    split="train",
                    source=path,
                    segment_id=path.stem,
                    metadata={"target_name": target_name},
                )
            )
            test_segments.append(
                _segment(
                    _prediction_frame(x[split_at:], y[split_at:], segment_id=path.stem),
                    split="test",
                    source=path,
                    segment_id=path.stem,
                    metadata={"target_name": target_name},
                )
            )
        return _canonical(
            name=self.name,
            version=self.version,
            root=root_path,
            splits={"train": tuple(train_segments), "test": tuple(test_segments)},
            schema=_prediction_schema(feature_count, domain="hydrocracking", access=_PRIVATE_ACCESS),
            access=_PRIVATE_ACCESS,
        )

    def schema(self) -> DataSchema:
        return _prediction_schema(42, domain="hydrocracking", access=_PRIVATE_ACCESS)

    def default_task(self, task: str | None = None) -> TaskSchema:
        task_name = task or "prediction"
        if task_name == "imputation":
            return TaskSchema(name="imputation")
        if task_name != "prediction":
            raise ValueError(f"HY_PRD adapter supports prediction/imputation, got {task_name!r}.")
        return TaskSchema(name="prediction", targets=("quality",))

    def default_pipeline(self, task: str | None = None) -> dict[str, Any]:
        task_schema = self.default_task(task)
        pipeline = _standard_official_pipeline()
        if task_schema.name == "imputation":
            pipeline["mask"] = {"strategy": "random", "missing_rate": 0.2, "seed": 42}
        return pipeline

    def summary(self, task: str | None = None) -> dict[str, Any]:
        return _summary(self, self.default_task(task), access=_PRIVATE_ACCESS, files={"root": "HY_PRD"})


def _canonical(
    *,
    name: str,
    version: str,
    root: Path,
    splits: dict[str, tuple[Segment, ...]],
    schema: DataSchema,
    access: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> CanonicalDataset:
    """组装真实数据集的公共来源字段和适配器专属元数据。

    参数：
        name/version/root: preset 标识、数据版本和已解析的原始数据根目录。
        splits: 已完成列规范化的 train/test ``Segment`` 映射。
        schema: 与所有 segment frame 一致的 ``DataSchema``。
        access: 许可、公开性和访问条件；这里只复制，不做网络核验。
        metadata: 可选协议、文件 hash 或统计摘要，必须可序列化。
    返回：
        带来源元数据的 ``CanonicalDataset``。
    异常：
        不吞掉异常；splits/schema 不一致等问题由规范数据集或后续校验入口报告。
    副作用：
        无文件或网络访问。元数据映射复制到新字典。
    """

    source_metadata: dict[str, Any] = {
        "source_type": "real_dataset",
        "preset": name,
        "version": version,
        "root": str(root),
        "access": dict(access),
    }
    if metadata:
        source_metadata.update(metadata)
    return CanonicalDataset(
        splits=splits,
        schema=schema,
        metadata=source_metadata,
    )


def _closed_loop_cstr_smoke_dataset(adapter: CSTRFaultAdapter) -> CanonicalDataset:
    """生成遵守真实物理 schema 的确定性闭环 CSTR smoke 数据。

    参数：
        adapter: 必须携带闭环七变量协议的 CSTR 适配器。
    返回：
        含一个正常 train episode 和一个 fault01 test episode 的 ``CanonicalDataset``。
    异常：
        适配器没有物理协议时抛出 ``ValueError``。
    副作用：
        无文件或随机访问；只在内存生成解析信号。它保留 onset、schema 与 segment 边界，
        但数值不是物理仿真结果，绝不能作为论文基线或故障性能证据。
    """

    protocol = adapter.protocol
    if protocol is None:
        raise ValueError("Closed-loop CSTR smoke data requires an explicit physical protocol.")

    def _frame(*, rows: int, episode: str, fault_id: int) -> pd.DataFrame:
        """构造一个确定性 episode 的七变量 frame。

        参数：
            rows/episode/fault_id: episode 行数、追溯标识和协议故障编号。
        返回：
            含物理列、time/raw_index、segment 和逐行标签的规范 frame。
        异常：
            行数或 onset 不合法时由 ``_episode_fault_labels`` 抛出 ``ValueError``。
        副作用：
            无。``Ci/Ti/Tci`` 使用不同周期的小幅正弦信号，``C`` 与 ``Qc`` 再由输入/
            温度的解析关系生成，只为让 smoke 数据稳定且非退化；公式不代表 CSTR 机理。
        """

        time = np.arange(rows, dtype=float)
        ci = 1.0 + 0.02 * np.sin(time / 17.0)
        ti = 350.0 + 0.5 * np.cos(time / 23.0)
        tci = 300.0 + 0.8 * np.sin(time / 29.0)
        concentration = 0.7 * ci + 0.001 * (ti - 350.0)
        temperature = 365.0 + 0.4 * np.sin(time / 19.0)
        coolant_temperature = 310.0 + 0.2 * np.cos(time / 13.0)
        coolant_flow = 100.0 + 0.3 * (temperature - 365.0)
        values = np.column_stack(
            (
                ci,
                ti,
                tci,
                concentration,
                temperature,
                coolant_temperature,
                coolant_flow,
            )
        )
        labels = _episode_fault_labels(
            row_count=rows,
            fault_id=fault_id,
            fault_onset=protocol.fault_onset,
        )
        return _feature_frame(
            values,
            label=labels,
            segment_id=episode,
            feature_names=protocol.feature_names,
            include_raw_index=True,
        )

    train_rows = max(256, protocol.fault_onset + 1)
    test_rows = max(240, protocol.fault_onset + 1)
    train_frame = _frame(rows=train_rows, episode="normal", fault_id=0)
    test_frame = _frame(rows=test_rows, episode="fault01", fault_id=1)
    return CanonicalDataset(
        splits={
            "train": (
                _segment(
                    train_frame,
                    split="train",
                    source=f"synthetic:{adapter.name}/train",
                    segment_id="normal",
                    metadata={
                        "episode": "normal",
                        "episode_fault_id": 0,
                        "fault_family": "normal",
                        "fault_onset": None,
                    },
                ),
            ),
            "test": (
                _segment(
                    test_frame,
                    split="test",
                    source=f"synthetic:{adapter.name}/test",
                    segment_id="fault01",
                    metadata={
                        "episode": "fault01",
                        "episode_fault_id": 1,
                        "fault_family": protocol.fault_family(1),
                        "fault_onset": protocol.fault_onset,
                    },
                ),
            ),
        },
        schema=adapter.schema(),
        metadata={
            "source_type": "builtin_synthetic",
            "preset": adapter.name,
            "version": "synthetic-smoke-v1",
            **protocol.summary(),
        },
    )


def _tts_six_fault_smoke_dataset(
    adapter: TTSFaultDiagnosisAdapter,
) -> CanonicalDataset:
    """生成与七变量 TTS 物理 schema 一致的确定性 smoke 数据。

    参数：
        adapter: 使用 ``TTS_SIX_FAULT_PROTOCOL`` 的公开 TTS 适配器。
    返回：
        一个正常 train episode 和一个 fault01 test episode；列角色、raw_index 与
        onset=200 均与真实适配器一致，但不含真实故障输出通道。
    异常：
        协议列数与生成矩阵不一致时由 ``_feature_frame`` 抛出 ``ValueError``。
    副作用：
        无文件/网络/随机访问；解析信号只验证公共管线，绝不能作为 TTS 性能或机理证据。
    """

    protocol = adapter.protocol

    def _frame(*, rows: int, episode: str, fault_id: int) -> pd.DataFrame:
        """用非退化解析信号构造单个 TTS smoke episode。"""

        time = np.arange(rows, dtype=float)
        q1 = 1.0 + 0.1 * np.sin(time / 17.0)
        q2 = 0.8 + 0.08 * np.cos(time / 19.0)
        q1s = q1 + 0.01 * np.sin(time / 7.0)
        q2s = q2 + 0.01 * np.cos(time / 11.0)
        h1 = 0.6 * q1 + 0.2 * q2
        h2 = 0.3 * q1 + 0.5 * q2
        h3 = 0.4 * h1 + 0.5 * h2
        values = np.column_stack((q1, q2, q1s, q2s, h1, h2, h3))
        labels = _episode_fault_labels(
            row_count=rows,
            fault_id=fault_id,
            fault_onset=protocol.fault_onset,
        )
        return _feature_frame(
            values,
            label=labels,
            segment_id=episode,
            feature_names=protocol.feature_names,
            include_raw_index=True,
        )

    train_rows = max(256, protocol.fault_onset + 1)
    test_rows = max(240, protocol.fault_onset + 1)
    return CanonicalDataset(
        splits={
            "train": (
                _segment(
                    _frame(rows=train_rows, episode="normal", fault_id=0),
                    split="train",
                    source=f"synthetic:{adapter.name}/train",
                    segment_id="normal",
                    metadata={
                        "episode": "normal",
                        "episode_fault_id": 0,
                        "fault_family": "normal",
                        "fault_onset": None,
                    },
                ),
            ),
            "test": (
                _segment(
                    _frame(rows=test_rows, episode="fault01", fault_id=1),
                    split="test",
                    source=f"synthetic:{adapter.name}/test",
                    segment_id="fault01",
                    metadata={
                        "episode": "fault01",
                        "episode_fault_id": 1,
                        "fault_family": protocol.fault_family(1),
                        "fault_onset": protocol.fault_onset,
                    },
                ),
            ),
        },
        schema=adapter.schema(),
        metadata={
            "source_type": "builtin_synthetic",
            "preset": adapter.name,
            "version": "synthetic-smoke-v1",
            **protocol.summary(),
        },
    )


def _summary(
    adapter: Any,
    task: TaskSchema,
    *,
    access: dict[str, Any],
    files: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": adapter.name,
        "version": adapter.version,
        "description": adapter.description,
        "task": task.summary(),
        "tasks": [task.name],
        "files": files,
        "access": dict(access),
        "source": "real dataset when root is supplied; synthetic smoke fallback when root is omitted",
    }


def _standard_official_pipeline() -> dict[str, Any]:
    return {"split": {"type": "official"}, "normalization": {"method": "standard"}}


def _fault_task(
    task: str | None,
    *,
    inputs: tuple[str, ...] = (),
    fault_switch: int | None = None,
) -> TaskSchema:
    """构造故障诊断任务，必要时显式冻结物理输入角色和 onset。

    参数：
        task: 请求任务名或 ``None``；只允许 ``fault_diagnosis``。
        inputs: 可选物理角色选择器，空元组保持旧适配器的 schema 回退。
        fault_switch: 可选 episode 内故障开始索引，只作任务元数据。
    返回：
        以 ``fault_id`` 为标签、正常编号为 0 的 ``TaskSchema``。
    异常：
        任务名称不受支持时抛出 ``ValueError``。
    副作用：
        无。不读取正常或故障数据。
    """

    task_name = task or "fault_diagnosis"
    if task_name != "fault_diagnosis":
        raise ValueError(f"Fault adapters support only task 'fault_diagnosis', got {task_name!r}.")
    return TaskSchema(
        name="fault_diagnosis",
        inputs=inputs,
        targets=("fault_id",),
        label_column="fault_id",
        normal_label=0,
        fault_switch=fault_switch,
    )


def _resolve_dataset_root(root: str | Path, child: str, *, required: tuple[str, ...]) -> Path:
    root_path = Path(root)
    if all((root_path / item).exists() for item in required):
        return root_path
    child_path = root_path / child
    if all((child_path / item).exists() for item in required):
        return child_path
    missing = ", ".join(required)
    raise FileNotFoundError(f"Dataset root {root_path} does not contain required files: {missing}.")


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required dataset file does not exist: {path}")


def _mat_fault_segments(
    path: Path,
    *,
    split: str,
    normal: bool,
    protocol: FaultDatasetProtocol | None = None,
    source_sha256: str | None = None,
) -> tuple[Segment, ...]:
    """把 MAT key 转成互不跨越的 episode，并按协议生成逐行标签。

    参数：
        path: 包含一个或多个 episode key 的 MAT 文件。
        split: 写入每个 ``SegmentInfo`` 的官方 split 名称。
        normal: 为真时强制所有 key 视为正常 episode。
        protocol: 可选物理协议；为空时保留旧数据集的整段标签行为。协议可声明静态
            存储列选择，以在 frame 构造前排除故障输出等答案通道。
        source_sha256: 已由调用方计算的原始文件哈希，只写入来源元数据。
    返回：
        按 MAT key 排序的 ``Segment`` 元组；每个 key 都是独立窗口边界。
    异常：
        文件读取错误由 ``read_mat_arrays`` 传播；数组形状、未知故障编号、onset 越界
        或协议列宽不匹配时抛出 ``ValueError``。
    副作用：
        读取 ``path``，不写文件。声明协议后只有 onset 及其后的行使用非零标签；
        hash、episode 和标签统计不进入模型特征。
    """

    arrays = read_mat_arrays(path)
    segments: list[Segment] = []
    for key, array in sorted(arrays.items()):
        fault_id = 0 if normal or "normal" in key.lower() else _fault_id_from_name(key)
        values = _clean_array(array)
        labels: int | np.ndarray = fault_id
        metadata: dict[str, Any] = {"fault_id": fault_id, "mat_key": key}
        feature_names: tuple[str, ...] | None = None
        include_raw_index = False
        if protocol is not None:
            # 列选择必须发生在 schema/frame 边界之前；否则随数据发布的故障输出会被通用
            # 特征命名悄悄送入模型。选择只依赖静态说明，不查看数值或评价表现。
            values = protocol.select_stored_features(values, normal=normal)
            labels = _episode_fault_labels(
                row_count=values.shape[0],
                fault_id=fault_id,
                fault_onset=protocol.fault_onset,
            )
            feature_names = protocol.feature_names
            include_raw_index = True
            unique_labels, counts = np.unique(labels, return_counts=True)
            metadata.update(
                {
                    "episode": key,
                    "episode_fault_id": fault_id,
                    "fault_family": protocol.fault_family(fault_id),
                    "fault_onset": None if fault_id == 0 else protocol.fault_onset,
                    "label_counts": {
                        str(int(label)): int(count)
                        for label, count in zip(unique_labels, counts, strict=True)
                    },
                    "source_sha256": source_sha256,
                }
            )
        segments.append(
            _segment(
                _feature_frame(
                    values,
                    label=labels,
                    segment_id=key,
                    feature_names=feature_names,
                    include_raw_index=include_raw_index,
                ),
                split=split,
                source=path,
                segment_id=key,
                metadata=metadata,
            )
        )
    return tuple(segments)


def _te_dat_segment(path: Path, split: str) -> Segment:
    label = _fault_id_from_name(path.stem)
    array = pd.read_csv(path, sep=r"\s+", header=None).to_numpy(dtype=float)
    return _segment(
        _feature_frame(array, label=label, label_column="label", segment_id=path.stem),
        split=split,
        source=path,
        segment_id=path.stem,
        metadata={"label": label},
    )


def _feature_frame(
    array: Any,
    *,
    label: int | np.ndarray | None = None,
    label_column: str = "fault_id",
    segment_id: str | None = None,
    feature_names: tuple[str, ...] | None = None,
    include_raw_index: bool = False,
) -> pd.DataFrame:
    """把二维数值数组转换为适配器公共规范 frame。

    参数：
        array: 原始二维特征矩阵；经 ``_clean_array`` 验证并转成数值数组。
        label/label_column: 可选标量或逐行标签及其列名。
        segment_id: 可选 episode 标识，写入 ``segment`` 列。
        feature_names: 可选物理列名；为空时生成通用特征名。
        include_raw_index: 是否在时间列后保留原始零基行号。
    返回：
        列顺序稳定的 ``pandas.DataFrame``，包含 time、可选追溯列、特征和标签。
    异常：
        数组不是有效二维矩阵、物理列名数量不等于矩阵宽度，或逐行标签长度不等于
        行数时抛出 ``ValueError``。
    副作用：
        只在内存中新建 frame；不修改调用方数组，不读写磁盘。
    """

    values = _clean_array(array)
    names = feature_names or _feature_names(values.shape[1])
    if len(names) != values.shape[1]:
        raise ValueError(
            "Feature names must match the raw array width. "
            f"Current names={len(names)}, columns={values.shape[1]}."
        )
    frame = pd.DataFrame(values, columns=names)
    frame.insert(0, "time", np.arange(values.shape[0], dtype=float))
    if include_raw_index:
        frame.insert(1, "raw_index", np.arange(values.shape[0], dtype=int))
    if segment_id is not None:
        frame["segment"] = segment_id
    if label is not None:
        labels = np.asarray(label)
        if labels.ndim == 0:
            labels = np.full(values.shape[0], float(labels))
        labels = labels.reshape(-1)
        if labels.shape[0] != values.shape[0]:
            raise ValueError(
                f"Label length must match rows. Current labels={labels.shape[0]}, rows={values.shape[0]}."
            )
        frame[label_column] = labels.astype(float)
    return frame


def _episode_fault_labels(
    *,
    row_count: int,
    fault_id: int,
    fault_onset: int,
) -> np.ndarray:
    """按 episode 内零基采样索引生成逐行故障标签。

    参数：
        row_count: episode 总行数，必须为正。
        fault_id: episode 故障编号；0 表示全段正常。
        fault_onset: 第一条故障样本的零基索引。
    返回：
        长度为 ``row_count`` 的整数数组；故障 episode 在 onset 前为 0，之后为
        ``fault_id``，正常 episode 全为 0。
    异常：
        行数非正，或非正常 episode 的 onset 不在数组内时抛出 ``ValueError``。
    副作用：
        无。函数只分配新数组，不访问真实数据文件。
    """

    if row_count <= 0:
        raise ValueError(f"CSTR episode must contain at least one row. Current rows: {row_count}.")
    if fault_id == 0:
        return np.zeros(row_count, dtype=int)
    if fault_onset >= row_count:
        raise ValueError(
            "CSTR fault_onset must fall inside every fault episode. "
            f"Current onset={fault_onset}, rows={row_count}, fault_id={fault_id}."
        )
    labels = np.zeros(row_count, dtype=int)
    labels[fault_onset:] = fault_id
    return labels


def _file_summary(path: Path) -> dict[str, Any]:
    """计算原始协议文件的可追溯摘要。

    参数：
        path: 必须存在且为普通文件的本地路径。
    返回：
        含字符串路径、字节数和小写 SHA-256 十六进制摘要的字典。
    异常：
        文件缺失时抛出 ``FileNotFoundError``；读取失败时传播底层 ``OSError``。
    副作用：
        以只读方式分块扫描完整文件；不修改文件，也不在摘要中保存文件内容。
    """

    if not path.is_file():
        raise FileNotFoundError(f"Required CSTR protocol source file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _prediction_frame(x: np.ndarray, y: np.ndarray, *, segment_id: str) -> pd.DataFrame:
    values = _clean_array(x)
    target = _as_2d(y)
    frame = pd.DataFrame(values, columns=_feature_names(values.shape[1]))
    frame.insert(0, "time", np.arange(values.shape[0], dtype=float))
    frame["segment"] = segment_id
    frame["quality"] = target.reshape(-1).astype(float)
    return frame


def _wpt_frame(u: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    u = _clean_array(u)
    y = _clean_array(y)
    rows = min(u.shape[0], y.shape[0])
    u = u[:rows]
    y = y[:rows]
    time = u[:, 0] if u.shape[1] > 1 else np.arange(rows, dtype=float)
    control = u[:, -1]
    output = y[:, -1]
    return pd.DataFrame(
        {
            "episode": np.zeros(rows, dtype=int),
            "time": time,
            "state1": output,
            "control1": control,
            "output1": output,
            "reference1": np.zeros(rows, dtype=float),
        }
    )


def _hy_prd_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    frame = pd.read_csv(path)
    excluded = {"日期", "序号"}
    candidate_columns = [column for column in frame.columns if str(column) not in excluded]
    if len(candidate_columns) < 2:
        raise ValueError(f"HY_PRD CSV must contain target and feature columns: {path}")
    target_column = str(candidate_columns[0])
    feature_columns = [str(column) for column in candidate_columns[1:]]
    numeric = frame.loc[:, [target_column, *feature_columns]].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=0, how="any")
    if numeric.empty:
        raise ValueError(f"HY_PRD CSV produced no numeric rows after cleanup: {path}")
    y = numeric[target_column].to_numpy(dtype=float)[:, None]
    x = numeric.loc[:, feature_columns].to_numpy(dtype=float)
    return x, y, target_column


def _segment(
    frame: pd.DataFrame,
    *,
    split: str,
    source: str | Path,
    segment_id: str,
    metadata: dict[str, Any] | None = None,
) -> Segment:
    return Segment(
        frame=frame,
        meta=SegmentInfo(
            split=split,
            source=str(source),
            rows=int(frame.shape[0]),
            segment_id=segment_id,
            metadata=metadata or {},
        ),
    )


def _fault_schema(
    feature_count: int,
    *,
    domain: str,
    access: dict[str, Any] | None = None,
    feature_names: tuple[str, ...] | None = None,
    feature_roles: tuple[str, ...] | None = None,
    include_raw_index: bool = False,
) -> DataSchema:
    """构造故障数据的列级语义 schema。

    参数：
        feature_count: 原始矩阵特征宽度。
        domain/access: 数据领域与可选访问/许可元数据。
        feature_names/feature_roles: 可选物理列名和一一对应角色；缺省时生成通用 input 列。
        include_raw_index: 是否声明仅用于追溯的原始行号。
    返回：
        依次含 time、可选 raw_index、segment、特征和 fault_id 的 ``DataSchema``。
    异常：
        特征名或角色数量与 ``feature_count`` 不一致时抛出 ``ValueError``。
    副作用：
        无。函数只创建 schema，不根据数据数值猜测列语义。
    """

    names = feature_names or _feature_names(feature_count)
    roles = feature_roles or tuple("input" for _ in names)
    if len(names) != feature_count or len(roles) != feature_count:
        raise ValueError(
            "Fault schema physical columns must match feature_count. "
            f"Current feature_count={feature_count}, names={len(names)}, roles={len(roles)}."
        )
    return DataSchema(
        columns=(
            ColumnSpec("time", role="time"),
            *((ColumnSpec("raw_index", role="raw_index"),) if include_raw_index else ()),
            ColumnSpec("segment", role="segment"),
            *[
                ColumnSpec(name, role=role)
                for name, role in zip(names, roles, strict=True)
            ],
            ColumnSpec("fault_id", role="fault_id"),
        ),
        sample_rate=1.0,
        metadata={"domain": domain, "access": access} if access else {"domain": domain},
    )


def _classification_schema(feature_count: int, *, domain: str) -> DataSchema:
    return DataSchema(
        columns=(
            ColumnSpec("time", role="time"),
            ColumnSpec("segment", role="segment"),
            *[ColumnSpec(name, role="input") for name in _feature_names(feature_count)],
            ColumnSpec("label", role="label"),
        ),
        sample_rate=1.0,
        metadata={"domain": domain},
    )


def _reconstruction_schema(feature_count: int, *, domain: str) -> DataSchema:
    return DataSchema(
        columns=(
            ColumnSpec("time", role="time"),
            ColumnSpec("segment", role="segment"),
            *[ColumnSpec(name, role="input") for name in _feature_names(feature_count)],
        ),
        sample_rate=1.0,
        metadata={"domain": domain},
    )


def _prediction_schema(feature_count: int, *, domain: str, access: dict[str, Any] | None = None) -> DataSchema:
    return DataSchema(
        columns=(
            ColumnSpec("time", role="time"),
            ColumnSpec("segment", role="segment"),
            *[ColumnSpec(name, role="input") for name in _feature_names(feature_count)],
            ColumnSpec("quality", role="quality"),
        ),
        sample_rate=1.0,
        metadata={"domain": domain, "access": access} if access else {"domain": domain},
    )


def _feature_names(feature_count: int) -> tuple[str, ...]:
    return tuple(f"x{idx:02d}" for idx in range(int(feature_count)))


def _fault_id_from_name(name: str) -> int:
    digits = "".join(ch for ch in str(name) if ch.isdigit())
    return int(digits[-2:] if len(digits) >= 2 else digits or 0)


def _first_array(arrays: dict[str, np.ndarray], preferred: str) -> np.ndarray:
    if preferred in arrays:
        return arrays[preferred]
    if not arrays:
        raise ValueError("No numeric arrays were found in dataset file.")
    return arrays[sorted(arrays)[0]]


def _clean_array(array: Any) -> np.ndarray:
    values = _as_2d(array).astype(float)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _as_2d(array: Any) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim == 1:
        return values[:, None]
    if values.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D array. Current shape: {values.shape}.")
    return values


__all__ = [
    "CSTRFaultAdapter",
    "CSTRFaultProtocol",
    "FaultDatasetProtocol",
    "HYFaultAdapter",
    "HYQualityPredictionAdapter",
    "MultiphaseFaultAdapter",
    "NpyReconstructionAdapter",
    "TEClassificationAdapter",
    "TEFaultDiagnosisAdapter",
    "TTSFaultDiagnosisAdapter",
    "WPTMPCAdapter",
]
