"""
数据集 preset 与声明式 dataset card 的安全解析注册表。

文件用途：
    把用户提供的 preset 名称、别名或本地 ``dataset_card.yaml`` 路径解析为显式
    ``DatasetAdapter``，供 DataModule 构造规范数据集。
主要职责：
    维护已注册适配器及别名；解析普通声明式卡片；当卡片声明一个已注册专用适配器时，
    以卡片根目录调用该适配器。这里不读取原始数据、不执行预处理、不动态导入任意类。
关键输入与输出：
    输入为适配器实例、preset 名称或 YAML 路径；输出为满足 DatasetAdapter 协议的对象。
依赖与副作用：
    注册会修改当前 ``DatasetRegistry`` 实例内的映射；解析卡片会只读 YAML，
    模块导入本身不扫描数据目录、不访问网络、不创建文件。
重要约束：
    只有卡片 ``name`` 与 ``adapter`` 声明都精确匹配已注册实例时才启用专用路由；
    历史卡片中不匹配的 adapter 元数据继续走通用读取。禁止根据 YAML 字符串做任意
    动态导入，避免绕过 schema、许可和数据泄漏边界。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from joff.data.schema import DataSchema, TaskSchema

from .base import CanonicalDataset, DatasetAdapter, DatasetCardAdapter


class DatasetRegistry:
    """Register and resolve dataset adapters by preset name."""

    def __init__(self) -> None:
        self._adapters: dict[str, DatasetAdapter] = {}
        self._aliases: dict[str, str] = {}

    def register(
        self,
        adapter: DatasetAdapter,
        *,
        aliases: tuple[str, ...] | list[str] = (),
        replace: bool = False,
    ) -> None:
        """Register an adapter and optional aliases."""

        key = _normalize(adapter.name)
        if not replace and key in self._adapters:
            raise ValueError(
                f"Dataset preset {adapter.name!r} is already registered. "
                f"Legal presets are: {', '.join(self.list())}."
            )
        self._adapters[key] = adapter
        for alias in aliases:
            self._aliases[_normalize(alias)] = key

    def get(self, name: str) -> DatasetAdapter:
        """Return an adapter by preset name or alias."""

        key = _normalize(name)
        resolved = self._aliases.get(key, key)
        if resolved not in self._adapters:
            legal = ", ".join(self.list()) or "<empty>"
            raise ValueError(
                f"Unknown dataset preset {name!r}. Legal presets are: {legal}. "
                f"Current input was: {name!r}."
            )
        return self._adapters[resolved]

    def resolve(self, preset: str | Path) -> DatasetAdapter:
        """解析注册名称或本地 dataset card 路径。

        参数：
            preset: 已注册的 preset/别名，或存在的 YAML 文件路径。
        返回：
            普通卡片返回 ``DatasetCardAdapter``；包含 ``adapter`` 声明的卡片返回
            ``DeclaredDatasetCardAdapter``，由已注册专用实现读取数据。
        异常：
            YAML 路径不存在时抛出 ``FileNotFoundError``；名称形式的 preset 未注册时
            抛出 ``ValueError``。卡片中的历史 adapter 元数据不匹配时保持通用卡片行为。
        副作用：
            解析卡片时只读 YAML；不读取原始数据，也不修改注册表。
        """

        path = Path(preset)
        if path.exists():
            card = DatasetCardAdapter.from_yaml(path)
            declared_name = card.raw_card.get("adapter")
            if declared_name is None:
                return card
            try:
                delegate = self.get(card.name)
            except ValueError:
                return card
            legal_declarations = {delegate.name, type(delegate).__name__}
            if str(declared_name) not in legal_declarations:
                # 旧卡片曾把 adapter 当作说明字段；只有精确匹配才升级为专用路由，
                # 否则维持 P1 之前的通用 DatasetCardAdapter 行为。
                return card
            return DeclaredDatasetCardAdapter(card=card, delegate=delegate)
        if path.suffix.lower() in {".yaml", ".yml"}:
            raise FileNotFoundError(f"Dataset card does not exist: {path}")
        return self.get(str(preset))

    def list(self) -> tuple[str, ...]:
        """List registered preset names."""

        return tuple(sorted(adapter.name for adapter in self._adapters.values()))


class DeclaredDatasetCardAdapter:
    """把 dataset card 的路径配置绑定到一个显式注册的专用适配器。

    该包装器只解决“卡片声明专用适配器”的安全路由问题。schema、任务与真实读取逻辑仍由
    ``delegate`` 拥有；卡片只提供默认根目录、默认流水线和可审计描述，不能覆盖专用协议。
    """

    def __init__(self, *, card: DatasetCardAdapter, delegate: DatasetAdapter) -> None:
        """保存已验证的卡片与专用适配器，不立即访问原始数据。

        参数：
            card: 已解析的本地 dataset card。
            delegate: 名称和类声明都已由 ``DatasetRegistry.resolve`` 验证的注册适配器。
        异常：
            本构造器不额外抛异常；调用前的匹配验证由注册表负责。
        副作用：
            无。只保存对象引用和读者可见的名称、版本、描述。
        """

        self.card = card
        self.delegate = delegate
        self.name = card.name
        self.version = card.version
        self.description = card.description

    def read(
        self,
        *,
        root: str | Path | None = None,
        task: str | None = None,
    ) -> CanonicalDataset:
        """使用显式根目录或卡片默认根目录调用专用读取器。

        参数：
            root: 可选的调用方数据根目录；为空时解析卡片 ``files.root``。
            task: 交给专用适配器校验的任务名称。
        返回：
            专用适配器产生的 ``CanonicalDataset``，因此保留其逐行标签和来源追溯。
        异常：
            卡片缺少可解析根目录时抛出 ``ValueError``；文件和协议错误由专用适配器传播。
        副作用：
            调用时只读卡片指定的原始数据；不写文件、不修改注册表。
        """

        resolved_root = Path(root) if root is not None else self._default_root()
        return self.delegate.read(root=resolved_root, task=task)

    def schema(self) -> DataSchema:
        """返回专用适配器的物理 schema，不允许通用卡片读取器猜列。

        返回：
            委托适配器生成的 ``DataSchema``。
        异常：
            schema 声明内部不一致时由委托适配器以 ``ValueError`` 拒绝。
        副作用：
            无。不读取数据。
        """

        return self.delegate.schema()

    def default_task(self, task: str | None = None) -> TaskSchema:
        """返回专用适配器验证后的任务定义。

        参数：
            task: 可选任务名称。
        返回：
            委托适配器的 ``TaskSchema``。
        异常：
            不支持的任务名由委托适配器以 ``ValueError`` 拒绝。
        副作用：
            无。
        """

        return self.delegate.default_task(task)

    def default_pipeline(self, task: str | None = None) -> dict[str, Any]:
        """返回卡片声明的流水线，并先由专用适配器验证任务。

        参数：
            task: 可选任务名称，必须同时受专用适配器与卡片支持。
        返回：
            卡片级与任务级流水线合并后的新字典。
        异常：
            任务不受专用适配器或卡片支持时抛出 ``ValueError``。
        副作用：
            无。不读取原始数据，不修改卡片配置。

        卡片是可移植配置的权威入口；专用适配器仍先验证任务名称，避免卡片声明超出其
        支持范围的任务。
        """

        self.delegate.default_task(task)
        return self.card.default_pipeline(task)

    def summary(self, task: str | None = None) -> dict[str, Any]:
        """返回卡片摘要并记录实际使用的显式注册适配器。

        参数：
            task: 要写入摘要的任务名称。
        返回：
            JSON 可序列化的卡片摘要，额外含适配器类名与注册 preset。
        异常：
            任务不在卡片声明中时由 ``DatasetCardAdapter`` 抛出 ``ValueError``。
        副作用：
            无。每次调用都修改新返回的摘要字典，不改变卡片对象。
        """

        summary = self.card.summary(task)
        summary["adapter"] = type(self.delegate).__name__
        summary["adapter_preset"] = self.delegate.name
        return summary

    def _default_root(self) -> Path:
        """解析相对卡片位置的默认数据根目录。

        返回：
            绝对或仍可由当前进程解析的 ``Path``；相对 ``files.root`` 以卡片目录为基准。
        异常：
            卡片未声明 ``files.root`` 且自身没有路径时抛出 ``ValueError``。
        副作用：
            无。不要求目标路径此时存在，存在性由专用适配器给出更具体的错误。
        """

        raw_root = self.card.preset.files.get("root")
        if raw_root is None:
            if self.card.card_path is None:
                raise ValueError(
                    f"Dataset card {self.card.name!r} must declare files.root for adapter routing."
                )
            return self.card.card_path.parent
        root_path = Path(raw_root)
        if root_path.is_absolute():
            return root_path
        if self.card.card_path is None:
            return root_path
        return (self.card.card_path.parent / root_path).resolve()


def _normalize(name: str) -> str:
    return name.strip().lower()
