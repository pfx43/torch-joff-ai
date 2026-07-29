"""闭环 CSTR frozen evaluation 的元数据检查与 claim 后数值 loader。

文件用途：
    在不读取 MAT 数值 payload 的前提下，用文件 SHA-256 和 MAT key/shape 冻结八个故障
    episode 身份；正式 claim 后再通过 P1 ``CSTRFaultAdapter`` 读取数值并重建逐行标签。
主要职责：
    定义 ``CSTRArchiveInspection``、``inspect_closed_loop_cstr_archive``、
    ``CSTRClosedLoopEpisodeLoader`` 和跨进程可用的 ``ManifestBoundCSTRFaultSource``。
    本文件不训练/拟合模型、不计算阈值或指标，也不决定数据许可。
关键输入与输出：
    输入为闭环 CSTR 根目录、onset、期望特征数和已经冻结的 P10 manifest；输出为原始文件
    hash、八个 ``FrozenFaultEpisodeManifest`` 及 claim 后的只读 ``FrozenFaultEpisode``。
依赖与副作用：
    inspection 只读 MAT header/shape 和文件字节 hash，不把数组值载入 NumPy；loader 才
    调用适配器读取 train/test MAT。读取 MATLAB v7.3 header 时可选依赖 h5py；不写文件、
    不访问网络、不修改随机状态。
重要约束：
    fault01--fault08 必须完整、列数一致、onset 位于每段内；正式 source 只接受
    ``verified``，并在调用 loader 前先消耗一次访问状态。raw 文件任一 hash 改变、manifest
    ledger 未冻结或 episode 身份不一致都 fail closed；加载失败也不能重开同一 source。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import re

import numpy as np
from scipy.io import whosmat  # type: ignore[import-untyped]

from joff.data import ProtocolAccessError
from joff.data.adapters.real_process import CSTRFaultAdapter

from .frozen_evaluation import (
    FrozenEvaluationClaim,
    FrozenFaultEpisode,
    FrozenFaultEpisodeManifest,
    FrozenProtocolIntegrityError,
    FrozenProtocolManifest,
)
from .paper_environment import sha256_file


_FAMILIES = (
    "process",
    "process",
    "actuator",
    "actuator",
    "actuator",
    "sensor",
    "sensor",
    "sensor",
)


@dataclass(frozen=True)
class CSTRArchiveInspection:
    """只含文件身份和 MAT header 几何、不含故障数值的检查结果。

    参数：
        root/normal_path/fault_path/description_path: 归一化后的官方文件位置。
        normal_source_hash/fault_source_hash: 两个 MAT 文件的内容身份。
        normal_rows/feature_count/fault_episodes: normal 几何与完整八故障 episode 身份。
    返回：
        路径绝对化、episode 元组化后的不可变检查结果。
    异常：
        hash、行列数、fault id 集合或 episode source hash 不一致时抛出 ``ValueError``。
    副作用：
        只规范化内存值，不读取文件。
    """

    root: Path
    normal_path: Path
    fault_path: Path
    description_path: Path
    normal_source_hash: str
    fault_source_hash: str
    normal_rows: int
    feature_count: int
    fault_episodes: tuple[FrozenFaultEpisodeManifest, ...]

    def __post_init__(self) -> None:
        """冻结绝对路径并验证完整八 episode 几何。"""

        for name in ("root", "normal_path", "fault_path", "description_path"):
            path = Path(getattr(self, name)).resolve()
            if not path.is_absolute():
                raise ValueError(f"CSTR archive {name} must resolve to an absolute path.")
            object.__setattr__(self, name, path)
        for name in ("normal_source_hash", "fault_source_hash"):
            value = str(getattr(self, name))
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"CSTR archive {name} must be a lowercase SHA-256.")
        if self.normal_rows <= 0 or self.feature_count <= 0:
            raise ValueError("CSTR normal rows and feature_count must be positive.")
        episodes = tuple(self.fault_episodes)
        if len(episodes) != 8 or {item.fault_id for item in episodes} != set(range(1, 9)):
            raise ValueError("CSTR archive inspection must contain fault ids 1..8 exactly once.")
        if any(item.source_hash != self.fault_source_hash for item in episodes):
            raise ValueError("CSTR episode source hashes must equal the inspected fault file.")
        object.__setattr__(self, "fault_episodes", episodes)


def inspect_closed_loop_cstr_archive(
    root: str | Path,
    *,
    fault_onset: int,
    expected_feature_count: int = 7,
    normal_file: str | Path = Path("train") / "model1[train].mat",
    fault_file: str | Path = Path("test") / "model1[test].mat",
    description_file: str | Path = "7v 正序, 8c 故障.txt",
) -> CSTRArchiveInspection:
    """只读文件 hash 和 MAT header，生成冻结前可审阅的 episode manifest。

    参数：
        root: 已解析的闭环 CSTR 数据根。
        fault_onset/expected_feature_count: 协议固定 onset 和七个物理变量。
        normal_file/fault_file/description_file: 相对根目录的官方文件位置。
    返回：
        不含数组值的 ``CSTRArchiveInspection``。
    异常：
        文件缺失、MAT header 无法解析、normal/fault key/shape 不完整或 onset 越界时抛出
        ``FileNotFoundError``/``ValueError``。
    副作用：
        读取三个文件元数据和两个 MAT 文件字节以计算 SHA-256；不调用 ``loadmat``，不把
        numeric payload 载入内存。
    """

    root_path = Path(root).expanduser().resolve()
    normal_path = _resolve_below(root_path, Path(normal_file))
    fault_path = _resolve_below(root_path, Path(fault_file))
    description_path = _resolve_below(root_path, Path(description_file))
    for path in (normal_path, fault_path, description_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required closed-loop CSTR file does not exist: {path}")
    if fault_onset < 0:
        raise ValueError("CSTR fault_onset must be nonnegative.")
    if expected_feature_count <= 0:
        raise ValueError("CSTR expected_feature_count must be positive.")
    normal_shapes = _mat_array_shapes(normal_path)
    if len(normal_shapes) != 1:
        raise ValueError("Closed-loop CSTR normal MAT must contain exactly one 2D array.")
    _, normal_shape = next(iter(sorted(normal_shapes.items())))
    if normal_shape[1] != expected_feature_count:
        raise ValueError(
            "Closed-loop CSTR normal feature count differs from the declared protocol."
        )
    fault_shapes = _mat_array_shapes(fault_path)
    indexed_faults: dict[int, tuple[str, tuple[int, int]]] = {}
    for key, shape in fault_shapes.items():
        fault_id = _fault_id(key)
        if fault_id in indexed_faults:
            raise ValueError(f"Duplicate CSTR fault id {fault_id} in MAT header.")
        indexed_faults[fault_id] = (key, shape)
    if set(indexed_faults) != set(range(1, 9)):
        raise ValueError("Closed-loop CSTR fault MAT must contain fault01 through fault08.")
    fault_hash = sha256_file(fault_path)
    episodes: list[FrozenFaultEpisodeManifest] = []
    for fault_id in range(1, 9):
        key, shape = indexed_faults[fault_id]
        rows, features = shape
        if features != expected_feature_count:
            raise ValueError(
                f"CSTR fault {fault_id} feature count differs from the declared protocol."
            )
        if fault_onset >= rows:
            raise ValueError(f"CSTR fault onset lies outside episode {key!r}.")
        episodes.append(
            FrozenFaultEpisodeManifest(
                episode_id=key,
                fault_id=fault_id,
                fault_family=cast(
                    Literal["process", "actuator", "sensor"],
                    _FAMILIES[fault_id - 1],
                ),
                onset=fault_onset,
                row_count=rows,
                raw_index_start=0,
                raw_index_end=rows - 1,
                source_hash=fault_hash,
            )
        )
    return CSTRArchiveInspection(
        root=root_path,
        normal_path=normal_path,
        fault_path=fault_path,
        description_path=description_path,
        normal_source_hash=sha256_file(normal_path),
        fault_source_hash=fault_hash,
        normal_rows=normal_shape[0],
        feature_count=normal_shape[1],
        fault_episodes=tuple(episodes),
    )


@dataclass(frozen=True)
class CSTRClosedLoopEpisodeLoader:
    """claim 后读取被 inspection 固定的闭环 CSTR 数值并重建 episode。

    参数：
        inspection: 冻结前只读 MAT header/hash 得到的静态身份。
    返回：
        可调用 loader；调用时必须同时给出正式 manifest 和已经持久消费 fault access 的 claim。
    异常：
        claim、许可、raw hash、episode 字典、schema 或逐行标签不一致时抛出
        ``FrozenProtocolIntegrityError``/``ProtocolAccessError``。
    副作用：
        构造无副作用；调用会读取 normal/fault MAT 数值，但不写文件、不拟合或校准。
    """

    inspection: CSTRArchiveInspection

    def __call__(
        self,
        *,
        manifest: FrozenProtocolManifest,
        claim: FrozenEvaluationClaim,
    ) -> tuple[FrozenFaultEpisode, ...]:
        """复验持久授权、许可和 raw hash 后加载逐行数值。

        参数：
            manifest: 许可、raw hash、episode 字典与 fit ledger 均已冻结的正式 manifest。
            claim: 与 manifest 绑定且已有持久 fault-access 记录的一次性 token。
        返回：
            顺序严格等于 manifest 的八份只读 ``FrozenFaultEpisode``。
        异常：
            授权、许可、hash、schema、segment 或逐行标签不一致时抛出
            ``FrozenProtocolIntegrityError``/``ProtocolAccessError``。
        副作用：
            读取 normal/fault MAT 数值并构造数组；不写文件、不拟合、不重新校准。loader
            不提供零参数捷径，因此公开重导出本身不能绕过正式门禁。
        """

        claim.verify_fault_access_consumed(manifest)
        dataset = manifest.resolved_config.get("dataset")
        if not isinstance(dataset, Mapping):
            raise FrozenProtocolIntegrityError(
                "Frozen manifest dataset config is missing before CSTR loading."
            )
        if (
            manifest.resolved_config.get("mode") != "frozen"
            or dataset.get("license_status") != "verified"
        ):
            raise ProtocolAccessError(
                "CSTR fault values require a formal frozen manifest with verified license."
            )
        if manifest.raw_data_hashes.get("normal") != self.inspection.normal_source_hash:
            raise FrozenProtocolIntegrityError(
                "Frozen manifest normal hash differs from the inspected CSTR archive."
            )
        if manifest.raw_data_hashes.get("fault") != self.inspection.fault_source_hash:
            raise FrozenProtocolIntegrityError(
                "Frozen manifest fault hash differs from the inspected CSTR archive."
            )
        if manifest.fault_episode_manifest != self.inspection.fault_episodes:
            raise FrozenProtocolIntegrityError(
                "Frozen manifest episode identities differ from the inspected CSTR archive."
            )
        if sha256_file(self.inspection.normal_path) != self.inspection.normal_source_hash:
            raise FrozenProtocolIntegrityError(
                "Closed-loop CSTR normal file changed after archive inspection."
            )
        if sha256_file(self.inspection.fault_path) != self.inspection.fault_source_hash:
            raise FrozenProtocolIntegrityError(
                "Closed-loop CSTR fault file changed after archive inspection."
            )
        canonical = CSTRFaultAdapter.closed_loop().read(root=self.inspection.root)
        test_segments = canonical.splits.get("test", ())
        by_id = {
            segment.meta.segment_id: segment
            for segment in test_segments
            if segment.meta.segment_id is not None
        }
        expected_ids = {item.episode_id for item in self.inspection.fault_episodes}
        if set(by_id) != expected_ids:
            raise FrozenProtocolIntegrityError(
                "Loaded CSTR test segment ids differ from the inspected MAT header."
            )
        input_columns = [
            *canonical.schema.role_columns("control_input"),
            *canonical.schema.role_columns("measured_output"),
        ]
        raw_columns = canonical.schema.role_columns("raw_index")
        label_columns = canonical.schema.role_columns("fault_id")
        if (
            len(input_columns) != self.inspection.feature_count
            or len(raw_columns) != 1
            or len(label_columns) != 1
        ):
            raise FrozenProtocolIntegrityError(
                "Loaded CSTR schema no longer matches the inspected seven-variable protocol."
            )
        episodes: list[FrozenFaultEpisode] = []
        for episode_manifest in self.inspection.fault_episodes:
            segment = by_id[episode_manifest.episode_id]
            metadata = segment.meta.metadata
            if metadata.get("source_sha256") != self.inspection.fault_source_hash:
                raise FrozenProtocolIntegrityError(
                    "Loaded CSTR segment source hash differs from the inspection."
                )
            episodes.append(
                FrozenFaultEpisode(
                    manifest=episode_manifest,
                    raw_indices=_integer_column(
                        segment.frame.loc[:, raw_columns[0]].to_numpy(),
                        name="raw_index",
                    ),
                    values=segment.frame.loc[:, input_columns].to_numpy(dtype=float),
                    labels=_integer_column(
                        segment.frame.loc[:, label_columns[0]].to_numpy(),
                        name="fault_id",
                    ),
                )
            )
        return tuple(episodes)


@dataclass
class ManifestBoundCSTRFaultSource:
    """跨进程正式运行使用的 manifest/许可绑定一次性 source。

    参数：
        loader: 自身要求 manifest/claim 的 ``CSTRClosedLoopEpisodeLoader``。
        normal_source_hash/fault_source_hash: inspection 固定的两个 raw 内容身份。
    返回：
        可跨进程重建的正式 source；每个实例和共享 registry 都只允许一次读取。
    异常：
        loader/hash 类型非法时构造失败；请求时许可、ledger、hash 或持久授权不一致则拒绝。
    副作用：
        构造无副作用；请求会 exclusive-create fault-access 记录并调用 loader。

    与 ``LazyFrozenCSTRFaultSource`` 的同进程 P2 bundle 门禁不同，本 source 直接重放
    manifest 内已经冻结的 fit ledger，因此可由新的 CLI 进程执行。
    """

    loader: CSTRClosedLoopEpisodeLoader
    normal_source_hash: str
    fault_source_hash: str
    _requested: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """校验 loader 与 hash，不读取配置、MAT 或创建访问记录。

        异常：
            loader 类型不是受门禁的 ``CSTRClosedLoopEpisodeLoader``，或 hash 不是小写
            SHA-256 时抛出 ``TypeError``/``ValueError``。
        副作用：
            无。
        """

        if not isinstance(self.loader, CSTRClosedLoopEpisodeLoader):
            raise TypeError(
                "Manifest-bound source requires CSTRClosedLoopEpisodeLoader."
            )
        for name in ("normal_source_hash", "fault_source_hash"):
            value = str(getattr(self, name))
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"Manifest-bound {name} must be a lowercase SHA-256.")

    def request_episodes(
        self,
        manifest: FrozenProtocolManifest,
        *,
        claim: FrozenEvaluationClaim,
    ) -> tuple[FrozenFaultEpisode, ...]:
        """验证 manifest ledger/raw hash/许可并在调用 loader 前消耗一次访问。

        参数：
            manifest: 已复验且自身冻结 ``license_status='verified'`` 的 formal manifest。
            claim: 与该 manifest 和唯一 registry 绑定的持久 token。
        返回：
            loader 返回的八份只读故障 episode。
        异常：
            实例重复请求、许可/ledger/hash 不一致时抛出 ``ProtocolAccessError``；共享访问记录
            已存在时抛出 ``FrozenEvaluationAlreadyClaimedError``；loader 异常原样传播。
        副作用：
            在共享 registry 写一次 fault-access 记录，然后读取 MAT 数值；失败不回滚。
        """

        claim.verify(manifest)
        if self._requested:
            raise ProtocolAccessError(
                "Manifest-bound CSTR fault source was already accessed."
            )
        dataset = manifest.resolved_config.get("dataset")
        if (
            not isinstance(dataset, Mapping)
            or manifest.resolved_config.get("mode") != "frozen"
            or dataset.get("license_status") != "verified"
        ):
            raise ProtocolAccessError(
                "Manifest-bound CSTR source requires the manifest itself to freeze a "
                "verified dataset license."
            )
        ledger = manifest.fit_access_ledger
        if ledger.get("frozen") is not True or ledger.get("protocol_ready") is not True:
            raise ProtocolAccessError(
                "Manifest-bound CSTR source requires a frozen, protocol-ready fit ledger."
            )
        if manifest.raw_data_hashes.get("normal") != self.normal_source_hash:
            raise ProtocolAccessError(
                "Manifest normal raw hash differs from the CSTR archive inspection."
            )
        if manifest.raw_data_hashes.get("fault") != self.fault_source_hash:
            raise ProtocolAccessError(
                "Manifest fault raw hash differs from the CSTR archive inspection."
            )
        if any(
            episode.source_hash != self.fault_source_hash
            for episode in manifest.fault_episode_manifest
        ):
            raise ProtocolAccessError(
                "Manifest episode source hashes differ from the CSTR archive inspection."
            )
        self._requested = True
        claim.consume_fault_access(manifest)
        return self.loader(manifest=manifest, claim=claim)


def _mat_array_shapes(path: Path) -> dict[str, tuple[int, int]]:
    """只读 MAT header 的二维 array key/shape；v7.3 回退 h5py dataset metadata。"""

    try:
        entries = whosmat(path)
        return {
            str(name): (int(shape[0]), int(shape[1]))
            for name, shape, _matlab_class in entries
            if len(shape) == 2
        }
    except (NotImplementedError, ValueError):
        return _hdf5_array_shapes(path)


def _hdf5_array_shapes(path: Path) -> dict[str, tuple[int, int]]:
    """读取 HDF5 dataset ``shape`` 属性，不调用 ``np.asarray`` 载入数值。"""

    try:
        import h5py  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "Inspecting MATLAB v7.3 CSTR files requires joff[hdf5]."
        ) from exc
    shapes: dict[str, tuple[int, int]] = {}
    with h5py.File(path, "r") as handle:

        def collect(name: str, value: Any) -> None:
            shape = getattr(value, "shape", None)
            if shape is not None and len(shape) == 2:
                shapes[str(name)] = (int(shape[0]), int(shape[1]))

        handle.visititems(collect)
    return shapes


def _fault_id(name: str) -> int:
    """从 fault01 等 key 取最后一组十进制数字。"""

    match = re.search(r"(\d+)(?!.*\d)", name)
    if match is None:
        raise ValueError(f"CSTR fault MAT key has no numeric id: {name!r}.")
    value = int(match.group(1))
    if value not in range(1, 9):
        raise ValueError(f"CSTR fault MAT key id is outside 1..8: {name!r}.")
    return value


def _integer_column(value: Any, *, name: str) -> np.ndarray:
    """验证 canonical 数值列逐项为整数后转换，拒绝静默截断。"""

    array = np.asarray(value)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise FrozenProtocolIntegrityError(
            f"Loaded CSTR {name} column must be a finite vector."
        )
    rounded = np.rint(array)
    if not np.equal(array, rounded).all():
        raise FrozenProtocolIntegrityError(
            f"Loaded CSTR {name} column contains non-integer values."
        )
    return rounded.astype(np.int64)


def _resolve_below(root: Path, relative: Path) -> Path:
    """解析声明文件并阻止绝对路径或 ``..`` 逃逸数据根。"""

    if relative.is_absolute() or relative.drive:
        raise ValueError("CSTR archive member paths must be relative.")
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("CSTR archive member path escapes the dataset root.") from exc
    return target


__all__ = [
    "CSTRArchiveInspection",
    "CSTRClosedLoopEpisodeLoader",
    "ManifestBoundCSTRFaultSource",
    "inspect_closed_loop_cstr_archive",
]
