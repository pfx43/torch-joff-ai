"""内存 PyTorch dataset 的最小数据层适配器。

文件用途：
    为已经由上层合法准备好的内存 ``Dataset`` 提供 Joff ``Trainer`` 所需的
    ``train_dataloader``/``test_dataloader`` 接口，避免实验编排层直接拥有 DataLoader。
主要职责：
    校验训练/测试 dataset 与批大小；在调用时构造 PyTorch ``DataLoader``。本文件不读取
    原始文件、不推断 schema、不拟合预处理器，也不决定论文阶段访问权限。
关键输入与输出：
    输入是非空训练 ``Dataset``、可选测试 ``Dataset``、正批大小和是否打乱；输出是对应
    的 DataLoader，或在没有测试集时返回 ``None``。
依赖与副作用：
    只依赖 ``torch.utils.data``。构造对象没有文件或随机副作用；迭代启用 shuffle 的
    DataLoader 时会使用 PyTorch 当前随机状态，调用方必须在 Trainer 层固定种子。
重要约束：
    该适配器只负责已准备 tensor 的加载边界，不能替代 ``DataModule`` 的原始数据读取、
    schema/task 选列、训练段预处理或五阶段访问账本。
"""

from __future__ import annotations

from collections.abc import Sized
from typing import Any, cast

from torch.utils.data import DataLoader, Dataset


class InMemoryDataModule:
    """把已准备的内存 dataset 暴露为 Trainer 数据接口。

    参数：
        train_dataset: 至少包含一个样本的训练 dataset。
        batch_size: 每批样本数，必须为正整数。
        test_dataset: 可选测试 dataset；为空时 ``test_dataloader`` 返回 ``None``。
        shuffle: 是否在每次训练迭代时打乱样本。时序论文基线必须传 ``False``。
    异常：
        dataset 不实现 ``__len__``、训练集为空、测试集为空但已提供，或批大小不是正整数
        时抛出 ``TypeError``/``ValueError``。
    副作用：
        构造时只保存引用；不会迭代数据、创建目录或修改随机状态。
    """

    def __init__(
        self,
        train_dataset: Dataset[Any],
        *,
        batch_size: int,
        test_dataset: Dataset[Any] | None = None,
        shuffle: bool = False,
    ) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(
                f"InMemoryDataModule batch_size must be a positive integer. "
                f"Current input: {batch_size!r}."
        )
        try:
            train_count = len(cast(Sized, train_dataset))
        except TypeError as exc:
            raise TypeError("InMemoryDataModule train_dataset must implement __len__.") from exc
        if train_count <= 0:
            raise ValueError("InMemoryDataModule train_dataset cannot be empty.")
        if test_dataset is not None:
            try:
                test_count = len(cast(Sized, test_dataset))
            except TypeError as exc:
                raise TypeError(
                    "InMemoryDataModule test_dataset must implement __len__."
                ) from exc
            if test_count <= 0:
                raise ValueError(
                    "InMemoryDataModule test_dataset cannot be empty when provided."
                )
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.shuffle = bool(shuffle)

    def train_dataloader(self) -> DataLoader[Any]:
        """构造训练 DataLoader。

        参数：
            无。
        返回：
            使用冻结 ``batch_size`` 和 ``shuffle`` 设置的 DataLoader。
        异常：
            PyTorch 在 sampler 或 dataset 访问期间产生的异常原样传播。
        副作用：
            构造 DataLoader；真正迭代且 ``shuffle=True`` 时消耗 PyTorch 随机状态。
        """

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )

    def test_dataloader(self) -> DataLoader[Any] | None:
        """构造可选测试 DataLoader。

        参数：
            无。
        返回：
            未提供测试集时为 ``None``；否则返回不打乱顺序的 DataLoader。
        异常：
            PyTorch 在 dataset 访问期间产生的异常原样传播。
        副作用：
            有测试集时构造 DataLoader；不消耗随机状态。
        """

        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )


__all__ = ["InMemoryDataModule"]
