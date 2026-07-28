"""Joff 训练层的稳定公共导入入口。

文件用途：
    汇总 Trainer、checkpoint、优化器、回调、通用损失和论文 P4 自由多视野损失。
主要职责：
    只做显式再导出并维护 ``__all__``；不启动训练、不创建 checkpoint、不固定随机种子。
关键输入与输出：
    本文件没有运行时输入；输出是 ``joff.training`` 命名空间下受支持的训练对象。
依赖与副作用：
    导入 PyTorch 训练定义和 Pydantic 配置；导入本身不访问数据或文件。
重要约束：
    Trainer 不读取原始数据；论文损失只读取模型前向结果与监督目标，不拥有五阶段协议、
    在线监视状态或校准分位。
"""

from .callbacks import Callback, HistoryCallback, NoOpCallback
from .checkpoint import CheckpointManager, CheckpointSaveResult
from .losses import LiftRegularization, LossBundle, NICELoss, PredictionLoss, SecondOrderPenalty
from .optim import build_optimizer
from .protected_losses import (
    ProtectedDiagnosticResult,
    ProtectedDiagnosticsConfig,
    ProtectedLossConfig,
    ProtectedModelDiagnostics,
    ProtectedMultiHorizonLoss,
)
from .trainer import Trainer, TrainingResult

__all__ = [
    "Callback",
    "CheckpointManager",
    "CheckpointSaveResult",
    "HistoryCallback",
    "LiftRegularization",
    "LossBundle",
    "NICELoss",
    "NoOpCallback",
    "PredictionLoss",
    "ProtectedDiagnosticResult",
    "ProtectedDiagnosticsConfig",
    "ProtectedLossConfig",
    "ProtectedModelDiagnostics",
    "ProtectedMultiHorizonLoss",
    "SecondOrderPenalty",
    "Trainer",
    "TrainingResult",
    "build_optimizer",
]
