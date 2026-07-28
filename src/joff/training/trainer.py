"""Joff 的轻量 PyTorch 训练、评估与 checkpoint 协调器。

文件用途：
    为通用模型和论文模型执行统一的 epoch/batch 训练与评估循环，把模型计算、数据加载、
    优化器和 checkpoint 职责组合起来。
主要职责：
    递归搬运具名 batch 到设备；调用 ``forward`` 与 ``compute_loss``；聚合总损失和命名
    分量；选择回归/重构评估器；协调 callback 与 ``CheckpointManager``。
关键输入与输出：
    输入是实现 ``train_dataloader``/可选 ``test_dataloader`` 的数据对象、PyTorch 模型及
    训练配置；输出 ``TrainingResult``、逐 epoch 指标和可选 checkpoint 路径。
依赖与副作用：
    依赖 PyTorch、Joff 设备/随机种子、优化器、评估器和 checkpoint 模块。训练会更新
    模型/优化器状态并可写 checkpoint；评估不反向传播。
重要约束：
    Trainer 不读取原始文件、不推断列语义；具名监督目标必须先于通用输入 fallback 解析，
    以支持 ``target_future`` 等论文批协议；模型专属损失由模型拥有。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import torch
from torch import nn
from torch.nn import functional as F

from joff.core.device import resolve_device
from joff.core.seed import seed_everything
from joff.evaluation import ReconstructionEvaluator, RegressionEvaluator
from joff.training.callbacks import Callback
from joff.training.checkpoint import CheckpointManager
from joff.training.optim import build_optimizer


@dataclass(frozen=True)
class TrainingResult:
    """训练循环的不可变返回对象。

    参数：
        history: 每个 epoch 的扁平浮点指标。
        checkpoint_paths: 实际写出的 ``last``/``best`` 路径。
    副作用：
        无。
    """

    history: list[dict[str, float]]
    checkpoint_paths: dict[str, Path]


class _LossInfo(TypedDict):
    """Trainer 内部已规范化的总损失与命名分量。"""

    loss: torch.Tensor
    losses: dict[str, torch.Tensor]


class Trainer:
    """保持模型、数据和产物职责分离的轻量训练器。

    参数：
        max_epochs/device/optimizer/seed: 训练轮数、目标设备、优化器配置和可选随机种子。
        monitor/mode: best checkpoint 的监控指标与优化方向。
        checkpoint_dir/save_last/save_best: checkpoint 目录与保存策略。
        callbacks: 生命周期回调。
        checkpoint_config/resolved_config/checkpoint_extra_state: 写入 checkpoint 的可追溯配置
            与扩展状态。
    异常：
        设备、优化器、监控方向或模型损失非法时由相应组件抛出异常。
    副作用：
        ``fit`` 更新模型参数、优化器、callback 和可选 checkpoint 文件。
    """

    def __init__(
        self,
        max_epochs: int = 1,
        *,
        device: str | torch.device = "auto",
        optimizer: dict[str, Any] | None = None,
        seed: int | None = None,
        monitor: str | None = None,
        mode: str = "min",
        checkpoint_dir: str | Path | None = None,
        save_last: bool = True,
        save_best: bool = True,
        callbacks: list[Callback] | None = None,
        checkpoint_config: dict[str, Any] | None = None,
        resolved_config: dict[str, Any] | None = None,
        checkpoint_extra_state: dict[str, Any] | None = None,
    ) -> None:
        """冻结一次 Trainer 运行所需的协调配置，不立即访问数据或模型。"""

        self.max_epochs = max_epochs
        self.device = resolve_device(device)
        self.optimizer_config = optimizer or {"type": "adam", "lr": 1e-3, "weight_decay": 0.0}
        self.seed = seed
        self.monitor = monitor
        self.mode = mode
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.save_last = save_last
        self.save_best = save_best
        self.callbacks = callbacks or []
        self.checkpoint_config = checkpoint_config
        self.resolved_config = resolved_config
        self.checkpoint_extra_state = checkpoint_extra_state

    def fit(self, model: nn.Module, data: Any) -> TrainingResult:
        """训练模型并返回历史与 checkpoint 路径。

        参数：
            model: 实现 ``forward``，并通常实现 ``compute_loss`` 的 PyTorch 模型。
            data: 提供训练 DataLoader 和可选测试 DataLoader 的数据层对象。
        返回：
            每轮聚合指标与实际 checkpoint 路径。
        异常：
            空/非法 batch、非标量损失、反向传播、评估或 checkpoint 错误原样传播。
        副作用：
            可固定随机种子、更新参数/优化器/callback，并写 checkpoint。
        """

        if self.seed is not None:
            seed_everything(self.seed)
        model.to(self.device)
        optimizer = build_optimizer(model, self.optimizer_config)
        checkpoint_manager = self._checkpoint_manager()
        history: list[dict[str, float]] = []
        checkpoint_paths: dict[str, Path] = {}
        train_loader = data.train_dataloader()
        context: dict[str, Any] = {"model": model, "trainer": self, "optimizer": optimizer}
        for callback in self.callbacks:
            callback.on_fit_start(context)
        for epoch in range(self.max_epochs):
            model.train()
            total_loss = 0.0
            count = 0
            component_totals: dict[str, float] = {}
            for batch_idx, batch in enumerate(train_loader):
                batch = _move_to_device(batch, self.device)
                optimizer.zero_grad(set_to_none=True)
                if hasattr(model, "training_step"):
                    training_step = cast(
                        Callable[[Any, int], Any],
                        getattr(model, "training_step"),
                    )
                    step = training_step(batch, batch_idx)
                    loss_info = _normalize_loss_output(step)
                else:
                    output = model(batch)
                    loss_info = _normalize_loss_output(_compute_loss(model, batch, output))
                loss = loss_info["loss"]
                loss.backward()
                optimizer.step()
                batch_size = _batch_size(batch)
                total_loss += float(loss.detach().cpu()) * batch_size
                for name, component in loss_info["losses"].items():
                    component_totals[name] = (
                        component_totals.get(name, 0.0) + float(component.detach().cpu()) * batch_size
                    )
                count += batch_size
            row = {"epoch": float(epoch), "train/loss": total_loss / max(count, 1)}
            for name, value in component_totals.items():
                row[f"train/{name}_loss"] = value / max(count, 1)
            test_loader_factory = getattr(data, "test_dataloader", None)
            if callable(test_loader_factory):
                test_loader = test_loader_factory()
                if test_loader is not None:
                    test_metrics = _evaluate_loader(model, test_loader, self.device)
                    row.update({f"test/{key}": value for key, value in test_metrics.items()})
            history.append(row)
            context.update({"epoch": epoch, "epoch_metrics": row})
            if checkpoint_manager is not None:
                saved = checkpoint_manager.save_epoch(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=row,
                    config=self.checkpoint_config,
                    resolved_config=self.resolved_config,
                    extra_state=self.checkpoint_extra_state,
                )
                if saved.last_path is not None:
                    checkpoint_paths["last"] = saved.last_path
                if saved.best_path is not None:
                    checkpoint_paths["best"] = saved.best_path
            for callback in self.callbacks:
                callback.on_epoch_end(context)
        context.update({"history": history, "checkpoint_paths": checkpoint_paths})
        for callback in self.callbacks:
            callback.on_fit_end(context)
        return TrainingResult(history=history, checkpoint_paths=checkpoint_paths)

    @torch.no_grad()
    def evaluate(self, model: nn.Module, data: Any) -> dict[str, float]:
        """在测试集（若有）或训练集上无梯度评估模型。

        参数：
            model/data: 与 ``fit`` 相同；优先使用非空测试 DataLoader。
        返回：
            聚合损失、命名分量和回归/重构指标。
        异常：
            batch 或模型输出不满足通用评估契约时抛出 ``TypeError``/``ValueError``。
        副作用：
            把模型移至目标设备并切换为 eval 模式；不更新参数。
        """

        model.to(self.device)
        model.eval()
        loader = data.test_dataloader() or data.train_dataloader()
        return _evaluate_loader(model, loader, self.device)

    def _checkpoint_manager(self) -> CheckpointManager | None:
        if self.checkpoint_dir is None:
            return None
        return CheckpointManager(
            self.checkpoint_dir,
            monitor=self.monitor or "train/loss",
            mode=self.mode,
            save_last=self.save_last,
            save_best=self.save_best,
        )


@torch.no_grad()
def _evaluate_loader(model: nn.Module, loader: Any, device: torch.device) -> dict[str, float]:
    """遍历一个 DataLoader，聚合损失并按输出语义选择评估器。

    参数：
        model/loader/device: 已构造模型、可迭代批次和目标设备。
    返回：
        全样本加权的损失分量与回归/重构指标。
    异常：
        预测和目标无法提取或拼接时抛出异常。
    副作用：
        将模型设为 eval；由 ``torch.no_grad`` 禁止梯度记录。
    """

    model.eval()
    total_loss = 0.0
    count = 0
    component_totals: dict[str, float] = {}
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    reconstruction_mode = False
    for batch in loader:
        batch = _move_to_device(batch, device)
        output = model(batch)
        loss_info = _normalize_loss_output(_compute_loss(model, batch, output))
        loss = loss_info["loss"]
        prediction, target, is_reconstruction = _prediction_and_target(batch, output)
        predictions.append(prediction.detach().cpu())
        targets.append(target.detach().cpu())
        reconstruction_mode = reconstruction_mode or is_reconstruction
        batch_size = _batch_size(batch)
        total_loss += float(loss.detach().cpu()) * batch_size
        for name, component in loss_info["losses"].items():
            component_totals[name] = (
                component_totals.get(name, 0.0) + float(component.detach().cpu()) * batch_size
            )
        count += batch_size
    metrics = {"loss": total_loss / max(count, 1)}
    for name, value in component_totals.items():
        metrics[f"{name}_loss"] = value / max(count, 1)
    if predictions and targets:
        y_pred = torch.cat(predictions, dim=0)
        y_true = torch.cat(targets, dim=0)
        evaluator = ReconstructionEvaluator() if reconstruction_mode else RegressionEvaluator()
        report = evaluator.evaluate(y_true, y_pred)
        metrics.update(report.to_flat_dict())
    return metrics


def _compute_loss(model: nn.Module, batch: Any, output: Any) -> Any:
    """计算一个 batch 的训练损失。

    参数：
        model/batch/output: 当前模型、设备上的 batch 和对应前向输出。
    返回：
        模型专属损失对象；无 ``compute_loss`` 时返回均方误差标量。
    异常：
        动态损失接口不可调用、通用输出/目标不是 tensor 或 shape 不匹配时抛出异常。
    副作用：
        无；返回值保留 autograd 图。
    """

    if hasattr(model, "compute_loss"):
        compute_loss = cast(
            Callable[[Any, Any], Any],
            getattr(model, "compute_loss"),
        )
        return compute_loss(batch, output)
    target = batch[1] if isinstance(batch, (tuple, list)) and len(batch) > 1 else batch[0]
    if isinstance(output, dict) and "reconstruction" in output:
        output = output["reconstruction"]
    return F.mse_loss(output, target)


def _prediction_and_target(batch: Any, output: Any) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """从模型输出与 batch 提取可评估预测、目标和重构标志。

    具名目标先独立解析；只有目标不存在时才解析通用输入。这一顺序避免论文 batch 已含
    ``target_future`` 却因没有 ``x``/``history`` 别名而提前失败。
    """

    if isinstance(output, dict):
        if "prediction" in output:
            prediction = output["prediction"]
            target = _target_from_batch(batch)
            if target is None:
                target = _input_from_batch(batch)
            return prediction, _align_target(target, prediction), False
        if "reconstruction" in output:
            prediction = output["reconstruction"]
            target = _input_from_batch(batch)
            return prediction, _align_target(target, prediction), True
    prediction = output
    target = _target_from_batch(batch)
    if target is None:
        target = _input_from_batch(batch)
    return prediction, _align_target(target, prediction), False


def _input_from_batch(batch: Any) -> torch.Tensor:
    """从通用 batch 中提取模型输入 fallback。

    返回：
        Tensor 本身、序列首项或 mapping 的标准输入字段。
    异常：
        没有任何合法输入字段时抛出 ``TypeError``。
    副作用：
        无。
    """

    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, dict):
        for key in ("x", "input", "inputs", "features", "history", "past"):
            if key in batch:
                return batch[key]
    if isinstance(batch, (tuple, list)) and batch:
        return batch[0]
    raise TypeError("Cannot extract input tensor from batch.")


def _target_from_batch(batch: Any) -> torch.Tensor | None:
    """提取显式监督目标。

    返回：
        mapping 的标准目标字段、序列第二项，或在不存在时返回 ``None``。
    重要约束：
        不在本函数解析输入 fallback，避免 eager fallback 遮蔽已有 ``target_future``。
    """

    if isinstance(batch, dict):
        for key in ("y", "target", "targets", "target_future", "future", "label", "labels"):
            if key in batch:
                return batch[key]
    if isinstance(batch, (tuple, list)) and len(batch) > 1:
        return batch[1]
    return None


def _align_target(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """对齐评估目标与预测 shape。

    仅支持 shape 已相同，或把 ``[B,D]`` 目标显式扩展为 ``[B,N,D]``；其他情况原样返回，
    由后续评估器报告真实 shape 错误，不做可能掩盖语义的 reshape。
    """

    if target.shape == prediction.shape:
        return target
    if prediction.ndim == 3 and target.ndim == 2 and target.shape[0] == prediction.shape[0]:
        return target.unsqueeze(1).expand(-1, prediction.shape[1], -1)
    return target


def _normalize_loss_output(value: Any) -> _LossInfo:
    """把多种模型损失返回形式规范化为 Trainer 契约。

    参数：
        value: Tensor、含 ``loss``/可选 ``losses`` 的字典，或同名属性对象。
    返回：
        ``_LossInfo``，其中总损失为 Tensor、命名分量只保留 Tensor。
    异常：
        缺总损失、总损失非 Tensor 或分量容器非字典时抛出 ``ValueError``/``TypeError``。
    副作用：
        无。
    """

    if isinstance(value, torch.Tensor):
        return {"loss": value, "losses": {}}
    if isinstance(value, dict):
        if "loss" not in value:
            raise ValueError("Loss dict must contain key 'loss'. Legal keys include 'loss' and 'losses'.")
        loss = value["loss"]
        if not isinstance(loss, torch.Tensor):
            raise TypeError(f"Loss value must be a torch.Tensor. Current input: {type(loss).__name__}.")
        raw_losses = value.get("losses", {})
        if raw_losses is None:
            raw_losses = {}
        if not isinstance(raw_losses, dict):
            raise TypeError(
                f"Loss components must be a mapping. Current input: {type(raw_losses).__name__}."
            )
        losses: dict[str, torch.Tensor] = {}
        for name, component in raw_losses.items():
            if isinstance(component, torch.Tensor):
                losses[str(name)] = component
        return {"loss": loss, "losses": losses}
    if hasattr(value, "loss"):
        return _normalize_loss_output({"loss": value.loss, "losses": getattr(value, "losses", {})})
    raise TypeError(
        f"Unsupported loss output {type(value).__name__}. Legal options are Tensor, dict, "
        "or object with a loss attribute."
    )


def _move_to_device(value: Any, device: torch.device) -> Any:
    """递归搬运嵌套 tensor。

    参数：
        value/device: 任意嵌套 batch 和目标设备。
    返回：
        保持 tuple/list/dict 形态的同构对象；非 tensor 元数据原样返回。
    副作用：
        Tensor ``to`` 可能分配设备内存；不修改原容器。
    """

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _batch_size(batch: Any) -> int:
    """从嵌套 batch 推断样本数。

    返回：
        Tensor 首维；序列/字典递归使用第一项；未知标量对象保守返回 1。
    异常：
        空字典会由 ``next`` 抛出 ``StopIteration``，提示 batch 契约非法。
    副作用：
        无。
    """

    if isinstance(batch, torch.Tensor):
        return int(batch.shape[0])
    if isinstance(batch, (tuple, list)) and batch:
        return _batch_size(batch[0])
    if isinstance(batch, dict):
        first = next(iter(batch.values()))
        return _batch_size(first)
    return 1
