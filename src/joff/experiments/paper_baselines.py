"""P3 正常数据基线的统一拟合、评分与 checkpoint 重放接口。

文件用途：
    为论文五阶段协议提供 PCA、去噪自编码器（DAE）和一步 MLP 三类最小基线，使编排层
    可以用同一接口训练模型、生成逐时刻分数并验证 checkpoint 恢复一致性。
主要职责：
    定义严格基线配置与分数对象；在训练数据上拟合模型输入标准化；复用 ``Trainer``
    训练神经基线；通过 ``CheckpointManager`` 保存仅含配置与状态的 checkpoint。
    本文件不负责阶段授权、检测/归因分位、报警、正式故障指标或运行目录编排。
关键输入与输出：
    输入为二维有限正常数组及逐行 ``raw_index``；输出为带原始索引的命名分数流。
    PCA 同时输出 Hotelling ``T²`` 与 SPE/Q，DAE 输出重构误差，MLP 输出一步预测误差。
依赖与副作用：
    依赖 NumPy、PyTorch、Joff 模型工厂、``Trainer`` 和 ``CheckpointManager``。拟合会固定
    随机种子并在显式目录写训练 checkpoint；评分无随机性且不写文件。
重要约束：
    所有模型输入标准化参数只能从调用方传入的训练阶段拟合。该模块不知道也不能读取
    ``PaperDataBundle`` 的其他阶段；阶段访问必须由论文编排层通过 ``FitAccessLedger``
    先登记。checkpoint 重放只证明代码路径和统计状态一致，不代表论文方法或正式结果。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

import numpy as np
import torch
from pydantic import Field, PositiveFloat, PositiveInt, field_validator, model_validator
from torch.utils.data import TensorDataset

from joff.core.config import ModelConfig, StrictConfig
from joff.data import InMemoryDataModule
from joff.core.device import resolve_device
from joff.core.factory import build_model
from joff.training import CheckpointManager, Trainer

_CHECKPOINT_FORMAT = "joff.paper_baseline.v1"
_NUMERICAL_EPSILON = 1e-12


class PaperBaselineConfig(StrictConfig):
    """一条 P3 正常数据基线的严格配置。

    参数：
        name: 当前论文运行内唯一的产物名，只允许字母、数字、连字符和下划线。
        type: ``pca``、``dae`` 或 ``mlp``。
        pca_components: PCA 保留主成分数；只对 ``pca`` 生效，为空时按输入维度保守选择。
        latent_dim: DAE 潜变量维数；只对 ``dae`` 生效，为空时取输入维度的一半。
        hidden: DAE 编码器或 MLP 的隐藏层宽度。
        noise_std: DAE 训练时显式指定的正输入噪声标准差；其他基线必须留空，评分时始终
            关闭噪声。
        max_epochs/batch_size/learning_rate: 神经基线训练参数。
        seed: 模型初始化、DataLoader 和优化器使用的确定性种子。
    异常：
        未知字段由 ``StrictConfig`` 拒绝；名称为空或含路径字符、DAE 未显式提供正噪声，
        或非 DAE 错配噪声时抛出 ``ValueError``。
    副作用：
        无；配置冻结后不可修改。
    """

    name: str
    type: Literal["pca", "dae", "mlp"]
    pca_components: PositiveInt | None = None
    latent_dim: PositiveInt | None = None
    hidden: tuple[PositiveInt, ...] = Field(default=(16,))
    noise_std: PositiveFloat | None = None
    max_epochs: PositiveInt = 1
    batch_size: PositiveInt = 32
    learning_rate: PositiveFloat = 1e-3
    seed: int = 42

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        """规范化并校验可安全用于相对产物路径的基线名。

        参数：
            value: 用户配置的名称。
        返回：
            去除首尾空白后的名称。
        异常：
            名称为空或含非字母数字、连字符、下划线字符时抛出 ``ValueError``。
        副作用：
            无。
        """

        normalized = value.strip()
        if not normalized or any(
            not (character.isalnum() or character in {"-", "_"})
            for character in normalized
        ):
            raise ValueError(
                "Paper baseline name must contain only letters, numbers, '-' or '_'. "
                f"Current input: {value!r}."
            )
        return normalized

    @model_validator(mode="after")
    def _validate_dae_noise(self) -> "PaperBaselineConfig":
        """保证 DAE 名称对应真实去噪训练，而不是零噪声普通 AE。

        参数：
            无；校验当前配置的 ``type`` 与 ``noise_std``。
        返回：
            校验后的冻结配置。
        异常：
            DAE 未显式提供正噪声，或 PCA/MLP 携带无效 DAE 噪声时抛出 ``ValueError``。
        副作用：
            无。
        """

        if self.type == "dae" and self.noise_std is None:
            raise ValueError("DAE baseline requires an explicit positive noise_std.")
        if self.type != "dae" and self.noise_std is not None:
            raise ValueError("noise_std is only valid for a DAE baseline.")
        return self


@dataclass(frozen=True)
class BaselineFitResult:
    """一次基线拟合产生的训练历史和底层 Trainer checkpoint。

    参数：
        history: 每个 epoch 的标量训练记录；PCA 没有迭代训练，因此为空元组。
        trainer_checkpoint_paths: 神经基线由 ``Trainer`` 写出的 last/best 路径；PCA 为空。
    重要约束：
        这些路径只描述训练过程。论文协议账本冻结的是随后写出的统一基线 checkpoint hash。
    """

    history: tuple[Mapping[str, float], ...]
    trainer_checkpoint_paths: Mapping[str, Path]

    def __post_init__(self) -> None:
        """复制并冻结可变容器，防止拟合后外部修改审计信息。

        参数：
            无。
        返回：
            无。
        异常：
            无。
        副作用：
            通过 ``object.__setattr__`` 替换为不可变副本。
        """

        object.__setattr__(
            self,
            "history",
            tuple(MappingProxyType(dict(row)) for row in self.history),
        )
        object.__setattr__(
            self,
            "trainer_checkpoint_paths",
            MappingProxyType(dict(self.trainer_checkpoint_paths)),
        )


@dataclass(frozen=True)
class BaselineScoreBatch:
    """一条基线在若干原始时刻上的命名非负分数流。

    参数：
        raw_indices: 每个分数对应的原始行号。一步预测分数对应目标时刻，因此通常比输入
            行数少一；PCA 和 DAE 与输入逐行对齐。
        streams: 分数名到一维有限非负数组的映射，各数组长度必须等于 ``raw_indices``。
    异常：
        原始索引不是一维、分数流为空、长度不一致，或分数含 NaN/Inf/负值时抛出
        ``ValueError``。
    副作用：
        构造时复制数组并设为只读；不写文件、不改变模型。
    """

    raw_indices: np.ndarray
    streams: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        """验证并冻结原始索引及每条分数流。

        参数：
            无。
        返回：
            无。
        异常：
            数据形状、有限性或非负约束不满足时抛出 ``ValueError``。
        副作用：
            用只读数组和 ``MappingProxyType`` 替换构造参数副本。
        """

        raw = np.asarray(self.raw_indices)
        if raw.ndim != 1:
            raise ValueError(
                f"Baseline score raw_indices must be one-dimensional. Shape={raw.shape}."
            )
        if not np.equal(raw, raw.astype(np.int64)).all():
            raise ValueError("Baseline score raw_indices must contain integer row identifiers.")
        frozen_raw = np.asarray(raw, dtype=np.int64).copy()
        frozen_raw.setflags(write=False)
        if not self.streams:
            raise ValueError("Baseline score streams cannot be empty.")
        frozen_streams: dict[str, np.ndarray] = {}
        for name, values in self.streams.items():
            normalized_name = str(name).strip()
            score = np.asarray(values, dtype=np.float64)
            if not normalized_name:
                raise ValueError("Baseline score stream names cannot be empty.")
            if score.ndim != 1 or len(score) != len(frozen_raw):
                raise ValueError(
                    f"Baseline score stream {normalized_name!r} must be one-dimensional and "
                    f"match raw_indices length {len(frozen_raw)}. Shape={score.shape}."
                )
            if not np.isfinite(score).all() or np.any(score < -_NUMERICAL_EPSILON):
                raise ValueError(
                    f"Baseline score stream {normalized_name!r} must be finite and non-negative."
                )
            frozen_score = np.maximum(score, 0.0).copy()
            frozen_score.setflags(write=False)
            frozen_streams[normalized_name] = frozen_score
        object.__setattr__(self, "raw_indices", frozen_raw)
        object.__setattr__(self, "streams", MappingProxyType(frozen_streams))


class PaperBaseline(ABC):
    """P3 基线的公共拟合、评分与统一 checkpoint 接口。

    子类只处理传入数组，不持有数据 bundle 或阶段名。调用方必须先通过 P2 账本取得合法
    阶段副本，才能调用 ``fit`` 或 ``score``。
    """

    def __init__(self, config: PaperBaselineConfig) -> None:
        """保存冻结配置并初始化未拟合状态。

        参数：
            config: 已通过严格校验的基线配置。
        返回：
            无。
        异常：
            无。
        副作用：
            只初始化内存状态。
        """

        self.config = config
        self._fitted = False

    @property
    def fitted(self) -> bool:
        """返回基线是否已具备可评分和可保存状态。

        参数：
            无。
        返回：
            成功拟合或恢复 checkpoint 后为真。
        异常：
            无。
        副作用：
            无。
        """

        return self._fitted

    @abstractmethod
    def fit(
        self,
        data: np.ndarray,
        raw_indices: np.ndarray,
        *,
        checkpoint_dir: str | Path,
        device: str | torch.device = "cpu",
    ) -> BaselineFitResult:
        """只用传入训练阶段拟合基线并返回训练审计信息。

        参数：
            data: 形状 ``[time, features]`` 的有限二维正常训练数组。
            raw_indices: 与 ``data`` 逐行对齐、严格递增的原始整数索引。
            checkpoint_dir: 神经基线训练 checkpoint 的有界目标目录；PCA 可忽略内容但仍
                接受统一参数。
            device: 神经基线使用的 Joff/PyTorch 设备；PCA 不移动设备。
        返回：
            训练历史和底层 Trainer checkpoint 路径；PCA 返回空历史/路径。
        异常：
            输入形状、连续一步样本、模型配置或文件写入非法时传播 ``ValueError``、
            ``RuntimeError``、PyTorch 或 ``OSError``。
        副作用：
            只从传入训练数组拟合统计量/权重；神经基线固定种子并在显式目录写 checkpoint。
        """

    @abstractmethod
    def score(
        self,
        data: np.ndarray,
        raw_indices: np.ndarray,
        *,
        device: str | torch.device = "cpu",
    ) -> BaselineScoreBatch:
        """对传入阶段生成带原始索引的命名分数流。

        参数：
            data: 形状 ``[time, fitted_features]`` 的有限二维阶段数组。
            raw_indices: 与数组逐行对齐的严格递增原始整数索引。
            device: 神经基线推理设备；PCA 不使用设备。
        返回：
            逐时刻非负有限分数流；一步预测分数对齐目标行，因而不含每个独立序列首行。
        异常：
            未拟合、特征维数不一致、没有连续一步样本或设备非法时传播
            ``RuntimeError``/``ValueError``/PyTorch 异常。
        副作用：
            神经模型临时移动到目标设备并进入 eval；不更新参数、不写文件。
        """

    def save_checkpoint(self, path: str | Path) -> Path:
        """保存可由 ``load_paper_baseline`` 恢复的统一基线 checkpoint。

        参数：
            path: 目标 ``.pt`` 文件。父目录不存在时创建。
        返回：
            实际写入路径。
        异常：
            基线尚未拟合时抛出 ``RuntimeError``；写文件失败时传播 I/O 异常。
        副作用：
            使用 ``CheckpointManager.save`` 覆盖显式目标文件。
        """

        self._require_fitted()
        output = Path(path)
        record = {
            "format": _CHECKPOINT_FORMAT,
            "baseline_type": self.config.type,
            "config": self.config.model_dump(mode="json"),
            "state": self._checkpoint_state(),
        }
        return CheckpointManager(
            output.parent,
            save_last=False,
            save_best=False,
        ).save(record, output)

    def restore_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        """从已校验 checkpoint 的状态映射恢复当前基线。

        参数：
            state: 与当前基线类型匹配的状态映射。
        返回：
            无。
        异常：
            状态缺字段、形状不兼容或模型权重不匹配时传播 ``KeyError``/``ValueError``/
            ``RuntimeError``。
        副作用：
            替换当前统计量或模型权重，并把对象标记为已拟合。
        """

        self._restore_checkpoint_state(state)
        self._fitted = True

    @abstractmethod
    def _checkpoint_state(self) -> Mapping[str, Any]:
        """返回不含整个模型对象的 checkpoint 状态。

        参数：
            无。
        返回：
            当前类型所需的统计数组、模型配置和/或 ``state_dict`` 映射。
        异常：
            拟合状态不完整时抛出 ``RuntimeError``。
        副作用：
            复制数组或把 tensor 移到 CPU；不写文件。
        """

    @abstractmethod
    def _restore_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        """把统一 checkpoint 状态恢复到当前实例。

        参数：
            state: 与当前基线类型和严格配置一致的 checkpoint 状态。
        返回：
            无。
        异常：
            缺字段、数组非有限、形状或模型配置不匹配时传播 ``KeyError``/
            ``ValueError``/``RuntimeError``。
        副作用：
            替换当前统计量或模型权重；不读取阶段数据、不写文件。
        """

    def _require_fitted(self) -> None:
        """拒绝在拟合或恢复前评分和保存。

        参数：
            无。
        返回：
            无。
        异常：
            当前对象未拟合时抛出 ``RuntimeError``。
        副作用：
            无。
        """

        if not self._fitted:
            raise RuntimeError(
                f"Paper baseline {self.config.name!r} must be fitted or restored before use."
            )


class PCABaseline(PaperBaseline):
    """训练段标准化后的 PCA Hotelling ``T²`` 与 SPE/Q 基线。"""

    def __init__(self, config: PaperBaselineConfig) -> None:
        """初始化未拟合的 PCA 统计量。

        参数：
            config: ``type='pca'`` 的严格配置。
        返回：
            无。
        异常：
            类型不匹配时抛出 ``ValueError``。
        副作用：
            无。
        """

        if config.type != "pca":
            raise ValueError(f"PCABaseline requires type='pca'. Current type={config.type!r}.")
        super().__init__(config)
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._components: np.ndarray | None = None
        self._eigenvalues: np.ndarray | None = None

    def fit(
        self,
        data: np.ndarray,
        raw_indices: np.ndarray,
        *,
        checkpoint_dir: str | Path,
        device: str | torch.device = "cpu",
    ) -> BaselineFitResult:
        """在训练段拟合标准化参数、主成分和保留方向方差。

        参数：
            data/raw_indices: 二维有限训练数组及逐行原始索引。
            checkpoint_dir: 为统一接口保留；PCA 无 epoch checkpoint，不会写入该目录。
            device: 为统一接口保留；PCA 始终在 CPU NumPy 上计算。
        返回：
            空训练历史与空 Trainer checkpoint 映射。
        异常：
            样本不足、主成分数越界或输入非法时抛出 ``ValueError``。
        副作用：
            只更新当前对象的 NumPy 统计量；不写文件。
        """

        del checkpoint_dir, device
        values, _ = _validated_data_and_raw_indices(data, raw_indices)
        if len(values) < 2:
            raise ValueError("PCA baseline requires at least two training rows.")
        standardized, mean, scale = _fit_input_standardizer(values)
        _, singular_values, right_vectors = np.linalg.svd(
            standardized,
            full_matrices=False,
        )
        maximum_components = min(values.shape[1], len(values) - 1)
        requested = self.config.pca_components
        component_count = (
            max(1, min(maximum_components, (values.shape[1] + 1) // 2))
            if requested is None
            else int(requested)
        )
        if component_count > maximum_components:
            raise ValueError(
                f"PCA components must not exceed min(features, rows-1)={maximum_components}. "
                f"Current input: {component_count}."
            )
        eigenvalues = np.square(singular_values[:component_count]) / float(len(values) - 1)
        self._mean = mean
        self._scale = scale
        self._components = right_vectors[:component_count]
        self._eigenvalues = np.maximum(eigenvalues, _NUMERICAL_EPSILON)
        self._fitted = True
        return BaselineFitResult(history=(), trainer_checkpoint_paths={})

    def score(
        self,
        data: np.ndarray,
        raw_indices: np.ndarray,
        *,
        device: str | torch.device = "cpu",
    ) -> BaselineScoreBatch:
        """计算逐行 Hotelling ``T²`` 和平方预测误差 SPE/Q。

        参数：
            data/raw_indices: 要评分的二维有限阶段数组与逐行原始索引。
            device: 为统一接口保留；PCA 始终在 CPU 上计算。
        返回：
            与输入逐行对齐的两条非负分数流。
        异常：
            未拟合、输入维度不匹配或数据非法时抛出 ``RuntimeError``/``ValueError``。
        副作用：
            无。
        """

        del device
        self._require_fitted()
        values, raw = _validated_data_and_raw_indices(data, raw_indices)
        mean, scale, components, eigenvalues = self._statistics()
        if values.shape[1] != len(mean):
            raise ValueError(
                f"PCA score feature count {values.shape[1]} does not match fitted count "
                f"{len(mean)}."
            )
        standardized = (values - mean) / scale
        latent = standardized @ components.T
        reconstructed = latent @ components
        hotelling_t2 = np.sum(np.square(latent) / eigenvalues, axis=1)
        spe_q = np.sum(np.square(standardized - reconstructed), axis=1)
        return BaselineScoreBatch(
            raw_indices=raw,
            streams={"hotelling_t2": hotelling_t2, "spe_q": spe_q},
        )

    def _checkpoint_state(self) -> Mapping[str, Any]:
        """返回 PCA 标准化、方向和方差张量。"""

        mean, scale, components, eigenvalues = self._statistics()
        return {
            "mean": torch.as_tensor(mean.copy(), dtype=torch.float64),
            "scale": torch.as_tensor(scale.copy(), dtype=torch.float64),
            "components": torch.as_tensor(components.copy(), dtype=torch.float64),
            "eigenvalues": torch.as_tensor(eigenvalues.copy(), dtype=torch.float64),
        }

    def _restore_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        """恢复并验证 PCA 统计量形状。"""

        mean = _state_array(state, "mean")
        scale = _state_array(state, "scale")
        components = _state_array(state, "components")
        eigenvalues = _state_array(state, "eigenvalues")
        if mean.ndim != 1 or scale.shape != mean.shape:
            raise ValueError("PCA checkpoint mean and scale must be matching one-dimensional arrays.")
        if components.ndim != 2 or components.shape[1] != len(mean):
            raise ValueError("PCA checkpoint components do not match fitted feature count.")
        if eigenvalues.ndim != 1 or len(eigenvalues) != components.shape[0]:
            raise ValueError("PCA checkpoint eigenvalues do not match component count.")
        if np.any(scale <= 0) or np.any(eigenvalues <= 0):
            raise ValueError("PCA checkpoint scale and eigenvalues must be strictly positive.")
        self._mean = mean
        self._scale = scale
        self._components = components
        self._eigenvalues = eigenvalues

    def _statistics(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """返回完整 PCA 统计量，缺失时拒绝继续。"""

        if any(
            value is None
            for value in (self._mean, self._scale, self._components, self._eigenvalues)
        ):
            raise RuntimeError("PCA baseline statistics are incomplete.")
        assert self._mean is not None
        assert self._scale is not None
        assert self._components is not None
        assert self._eigenvalues is not None
        return self._mean, self._scale, self._components, self._eigenvalues


class _TorchPaperBaseline(PaperBaseline):
    """复用 Joff ``Trainer`` 的 DAE/MLP 基线公共实现。"""

    def __init__(self, config: PaperBaselineConfig) -> None:
        super().__init__(config)
        self._input_mean: np.ndarray | None = None
        self._input_scale: np.ndarray | None = None
        self._model: torch.nn.Module | None = None
        self._input_dim: int | None = None

    def fit(
        self,
        data: np.ndarray,
        raw_indices: np.ndarray,
        *,
        checkpoint_dir: str | Path,
        device: str | torch.device = "cpu",
    ) -> BaselineFitResult:
        """在训练段拟合输入标准化并用 ``Trainer`` 训练神经基线。

        参数：
            data/raw_indices: 二维有限训练数组和逐行原始索引。
            checkpoint_dir: ``Trainer`` 写 last/best checkpoint 的目录。
            device: Joff 设备字符串或 PyTorch 设备；P3 smoke 使用 CPU。
        返回：
            epoch 历史与 Trainer checkpoint 路径。
        异常：
            连续一步样本不足、输入非法或训练失败时传播 ``ValueError``/PyTorch 异常。
        副作用：
            固定随机种子、更新模型权重，并在 ``checkpoint_dir`` 写训练 checkpoint。
        """

        values, raw = _validated_data_and_raw_indices(data, raw_indices)
        standardized, mean, scale = _fit_input_standardizer(values)
        self._input_mean = mean
        self._input_scale = scale
        self._input_dim = values.shape[1]
        self._model = build_model(self._model_config(values.shape[1]))
        training_data = self._training_data(standardized, raw)
        trainer = Trainer(
            max_epochs=int(self.config.max_epochs),
            device=device,
            optimizer={
                "type": "adam",
                "lr": float(self.config.learning_rate),
                "weight_decay": 0.0,
            },
            seed=self.config.seed,
            monitor="train/loss",
            mode="min",
            checkpoint_dir=checkpoint_dir,
            checkpoint_config=self._model_config(values.shape[1]).model_dump(mode="json"),
        )
        result = trainer.fit(self._model, training_data)
        best = result.checkpoint_paths.get("best")
        if best is not None:
            CheckpointManager(Path(checkpoint_dir)).load(
                best,
                model=self._model,
                map_location=resolve_device(device),
            )
        self._model.eval()
        self._fitted = True
        return BaselineFitResult(
            history=tuple(result.history),
            trainer_checkpoint_paths=result.checkpoint_paths,
        )

    def _checkpoint_state(self) -> Mapping[str, Any]:
        """返回输入标准化、模型配置和 state_dict。"""

        model, mean, scale, input_dim = self._torch_state()
        return {
            "input_dim": input_dim,
            "input_mean": torch.as_tensor(mean.copy(), dtype=torch.float64),
            "input_scale": torch.as_tensor(scale.copy(), dtype=torch.float64),
            "model_config": self._model_config(input_dim).model_dump(mode="json"),
            "model_state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
        }

    def _restore_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        """恢复输入标准化和模型 state_dict。"""

        input_dim = int(state["input_dim"])
        mean = _state_array(state, "input_mean")
        scale = _state_array(state, "input_scale")
        if mean.shape != (input_dim,) or scale.shape != (input_dim,) or np.any(scale <= 0):
            raise ValueError("Neural baseline checkpoint standardizer has incompatible shape.")
        stored_config = ModelConfig.model_validate(state["model_config"])
        expected_config = self._model_config(input_dim)
        if stored_config != expected_config:
            raise ValueError("Neural baseline checkpoint model config does not match baseline config.")
        model = build_model(stored_config)
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        self._input_dim = input_dim
        self._input_mean = mean
        self._input_scale = scale
        self._model = model

    @abstractmethod
    def _model_config(self, input_dim: int) -> ModelConfig:
        """构建与基线类型匹配的 Joff 模型配置。"""

    @abstractmethod
    def _training_data(
        self,
        standardized: np.ndarray,
        raw_indices: np.ndarray,
    ) -> InMemoryDataModule:
        """把标准化训练阶段变为 Trainer 可读取的数据对象。"""

    def _torch_state(self) -> tuple[torch.nn.Module, np.ndarray, np.ndarray, int]:
        """返回完整神经状态，缺失时拒绝评分或保存。"""

        if (
            self._model is None
            or self._input_mean is None
            or self._input_scale is None
            or self._input_dim is None
        ):
            raise RuntimeError(f"Neural baseline {self.config.name!r} state is incomplete.")
        return self._model, self._input_mean, self._input_scale, self._input_dim


class DAEBaseline(_TorchPaperBaseline):
    """训练段标准化空间中的 DAE 逐行重构误差基线。"""

    def __init__(self, config: PaperBaselineConfig) -> None:
        """校验 DAE 类型并初始化未拟合神经状态。

        参数：
            config: ``type='dae'`` 且含显式正 ``noise_std`` 的严格配置。
        返回：
            无。
        异常：
            类型不匹配时抛出 ``ValueError``；噪声约束已由配置构造器提前校验。
        副作用：
            只初始化内存字段，不构造模型、不固定随机种子、不写 checkpoint。
        """

        if config.type != "dae":
            raise ValueError(f"DAEBaseline requires type='dae'. Current type={config.type!r}.")
        super().__init__(config)

    def score(
        self,
        data: np.ndarray,
        raw_indices: np.ndarray,
        *,
        device: str | torch.device = "cpu",
    ) -> BaselineScoreBatch:
        """在冻结标准化空间中计算逐行平均平方重构误差。

        参数：
            data/raw_indices: 要评分的阶段数组及逐行原始索引。
            device: 推理设备。
        返回：
            名为 ``reconstruction_error`` 的逐行非负分数。
        异常：
            未拟合、特征数不匹配或输入非法时抛出 ``RuntimeError``/``ValueError``。
        副作用：
            临时把模型移动到目标设备并进入 eval；不更新权重、不写文件。
        """

        self._require_fitted()
        values, raw = _validated_data_and_raw_indices(data, raw_indices)
        model, mean, scale, input_dim = self._torch_state()
        if values.shape[1] != input_dim:
            raise ValueError("DAE score feature count does not match fitted input dimension.")
        target_device = resolve_device(device)
        tensor = torch.as_tensor(
            (values - mean) / scale,
            dtype=torch.float32,
            device=target_device,
        )
        model.to(target_device)
        model.eval()
        with torch.no_grad():
            reconstruction = model(tensor)["reconstruction"]
            error = torch.mean((reconstruction - tensor).pow(2), dim=1)
        return BaselineScoreBatch(
            raw_indices=raw,
            streams={"reconstruction_error": error.detach().cpu().numpy()},
        )

    def _model_config(self, input_dim: int) -> ModelConfig:
        latent_dim = self.config.latent_dim or max(1, input_dim // 2)
        if self.config.noise_std is None:
            raise RuntimeError("Validated DAE configuration is missing noise_std.")
        return ModelConfig(
            type="dae",
            input_dim=input_dim,
            latent_dim=latent_dim,
            encoder_hidden=list(self.config.hidden),
            decoder_hidden="mirror",
            noise_std=float(self.config.noise_std),
            loss="mse",
        )

    def _training_data(
        self,
        standardized: np.ndarray,
        raw_indices: np.ndarray,
    ) -> InMemoryDataModule:
        del raw_indices
        inputs = torch.as_tensor(standardized, dtype=torch.float32)
        return InMemoryDataModule(
            TensorDataset(inputs),
            batch_size=int(self.config.batch_size),
            shuffle=False,
        )


class MLPOneStepBaseline(_TorchPaperBaseline):
    """只在原始索引连续相邻行上训练和评分的一步 MLP 基线。"""

    def __init__(self, config: PaperBaselineConfig) -> None:
        """校验一步 MLP 类型并初始化未拟合神经状态。

        参数：
            config: ``type='mlp'`` 且不携带 DAE 噪声字段的严格配置。
        返回：
            无。
        异常：
            类型不匹配时抛出 ``ValueError``。
        副作用：
            只初始化内存字段，不构造模型、不读取数据、不写 checkpoint。
        """

        if config.type != "mlp":
            raise ValueError(
                f"MLPOneStepBaseline requires type='mlp'. Current type={config.type!r}."
            )
        super().__init__(config)

    def score(
        self,
        data: np.ndarray,
        raw_indices: np.ndarray,
        *,
        device: str | torch.device = "cpu",
    ) -> BaselineScoreBatch:
        """计算连续 ``raw_index`` 相邻时刻的一步平均平方预测误差。

        参数：
            data/raw_indices: 要评分的阶段数组和逐行原始索引。
            device: 推理设备。
        返回：
            与目标时刻原始索引对齐的 ``prediction_error`` 分数。
        异常：
            阶段没有连续相邻行、未拟合或维度不匹配时抛出 ``ValueError``/``RuntimeError``。
        副作用：
            临时把模型移动到目标设备并进入 eval；不更新权重、不写文件。
        """

        self._require_fitted()
        values, raw = _validated_data_and_raw_indices(data, raw_indices)
        model, mean, scale, input_dim = self._torch_state()
        if values.shape[1] != input_dim:
            raise ValueError("MLP score feature count does not match fitted input dimension.")
        inputs, targets, target_raw = _one_step_pairs((values - mean) / scale, raw)
        target_device = resolve_device(device)
        x_tensor = torch.as_tensor(inputs, dtype=torch.float32, device=target_device)
        y_tensor = torch.as_tensor(targets, dtype=torch.float32, device=target_device)
        model.to(target_device)
        model.eval()
        with torch.no_grad():
            prediction = model(x_tensor)
            error = torch.mean((prediction - y_tensor).pow(2), dim=1)
        return BaselineScoreBatch(
            raw_indices=target_raw,
            streams={"prediction_error": error.detach().cpu().numpy()},
        )

    def _model_config(self, input_dim: int) -> ModelConfig:
        return ModelConfig(
            type="mlp",
            input_dim=input_dim,
            output_dim=input_dim,
            hidden=list(self.config.hidden),
            loss="mse",
        )

    def _training_data(
        self,
        standardized: np.ndarray,
        raw_indices: np.ndarray,
    ) -> InMemoryDataModule:
        inputs, targets, _ = _one_step_pairs(standardized, raw_indices)
        return InMemoryDataModule(
            TensorDataset(
                torch.as_tensor(inputs, dtype=torch.float32),
                torch.as_tensor(targets, dtype=torch.float32),
            ),
            batch_size=int(self.config.batch_size),
            shuffle=False,
        )


def build_paper_baseline(config: PaperBaselineConfig | Mapping[str, Any]) -> PaperBaseline:
    """按受控类型构建一条未拟合 P3 基线。

    参数：
        config: 严格配置对象或待校验映射。
    返回：
        ``PCABaseline``、``DAEBaseline`` 或 ``MLPOneStepBaseline``。
    异常：
        未知/多余字段由 Pydantic 拒绝；内部映射遗漏时抛出 ``ValueError``。
    副作用：
        只构造内存对象，不拟合、不写文件。
    """

    resolved = (
        config
        if isinstance(config, PaperBaselineConfig)
        else PaperBaselineConfig.model_validate(dict(config))
    )
    if resolved.type == "pca":
        return PCABaseline(resolved)
    if resolved.type == "dae":
        return DAEBaseline(resolved)
    if resolved.type == "mlp":
        return MLPOneStepBaseline(resolved)
    raise ValueError(f"Unknown paper baseline type {resolved.type!r}.")


def load_paper_baseline(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> PaperBaseline:
    """从统一 checkpoint 构建并恢复一条 P3 基线。

    参数：
        path: ``PaperBaseline.save_checkpoint`` 写出的文件。
        device: checkpoint 读取与后续模型初始放置设备；默认 CPU。
    返回：
        已拟合、可直接评分的基线对象。
    异常：
        文件不存在、格式版本错误、配置非法或状态不兼容时传播相应异常。
    副作用：
        从本地读取 checkpoint；不写文件、不访问数据 bundle。
    """

    checkpoint = torch.load(
        Path(path),
        map_location=resolve_device(device),
        weights_only=False,
    )
    if checkpoint.get("format") != _CHECKPOINT_FORMAT:
        raise ValueError(
            f"Unknown paper baseline checkpoint format {checkpoint.get('format')!r}."
        )
    config = PaperBaselineConfig.model_validate(checkpoint["config"])
    if checkpoint.get("baseline_type") != config.type:
        raise ValueError("Paper baseline checkpoint type and config disagree.")
    baseline = build_paper_baseline(config)
    baseline.restore_checkpoint_state(checkpoint["state"])
    return baseline


def _validated_data_and_raw_indices(
    data: np.ndarray,
    raw_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """复制并验证二维有限数组及严格递增整数原始索引。

    参数：
        data: 预期形状 ``[time, features]`` 的数值数组。
        raw_indices: 与每行对齐的原始行号。
    返回：
        ``float64`` 数据副本和 ``int64`` 原始索引副本。
    异常：
        维度、长度、有限性、整数性或严格递增条件不满足时抛出 ``ValueError``。
    副作用：
        无。
    """

    values = np.asarray(data, dtype=np.float64)
    raw = np.asarray(raw_indices)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(
            f"Paper baseline data must be non-empty and two-dimensional. Shape={values.shape}."
        )
    if not np.isfinite(values).all():
        raise ValueError("Paper baseline data must contain only finite values.")
    if raw.ndim != 1 or len(raw) != len(values):
        raise ValueError(
            "Paper baseline raw_indices must be one-dimensional and match data rows."
        )
    if not np.equal(raw, raw.astype(np.int64)).all():
        raise ValueError("Paper baseline raw_indices must contain integers.")
    normalized_raw = raw.astype(np.int64)
    if len(normalized_raw) > 1 and np.any(np.diff(normalized_raw) <= 0):
        raise ValueError("Paper baseline raw_indices must be strictly increasing.")
    return values.copy(), normalized_raw.copy()


def _fit_input_standardizer(
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """只用训练数组拟合逐特征均值和安全标准差。

    参数：
        data: 已验证二维训练数组。
    返回：
        标准化数组、均值和将近常数列替换为 1 的标准差。
    异常：
        无；调用方已验证有限非空数据。
    副作用：
        无。
    """

    mean = np.mean(data, axis=0)
    scale = np.std(data, axis=0, ddof=0)
    safe_scale = np.where(scale > _NUMERICAL_EPSILON, scale, 1.0)
    return (data - mean) / safe_scale, mean, safe_scale


def _one_step_pairs(
    data: np.ndarray,
    raw_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """仅保留 ``raw_index[t+1] = raw_index[t] + 1`` 的一步样本。

    参数：
        data/raw_indices: 已验证且逐行对齐的阶段数组与原始索引。
    返回：
        当前时刻输入、下一时刻目标及目标原始索引。
    异常：
        没有任何连续相邻样本时抛出 ``ValueError``。
    副作用：
        无。
    """

    if len(data) < 2:
        raise ValueError("One-step MLP baseline requires at least two rows.")
    valid = np.diff(raw_indices) == 1
    if not np.any(valid):
        raise ValueError(
            "One-step MLP baseline requires at least one raw-index-contiguous pair."
        )
    return data[:-1][valid], data[1:][valid], raw_indices[1:][valid]


def _state_array(state: Mapping[str, Any], key: str) -> np.ndarray:
    """把 checkpoint 中的 tensor/array 字段复制为有限 ``float64`` 数组。

    参数：
        state: checkpoint 状态映射。
        key: 必须存在的字段名。
    返回：
        有限 ``float64`` 数组副本。
    异常：
        字段缺失或含 NaN/Inf 时抛出 ``KeyError``/``ValueError``。
    副作用：
        无。
    """

    value = state[key]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64).copy()
    if not np.isfinite(array).all():
        raise ValueError(f"Paper baseline checkpoint field {key!r} must be finite.")
    return array


__all__ = [
    "BaselineFitResult",
    "BaselineScoreBatch",
    "DAEBaseline",
    "MLPOneStepBaseline",
    "PCABaseline",
    "PaperBaseline",
    "PaperBaselineConfig",
    "build_paper_baseline",
    "load_paper_baseline",
]
