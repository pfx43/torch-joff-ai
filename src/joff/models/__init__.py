"""Joff 内置模型的稳定公共导入与显式注册入口。

文件用途：
    汇总通用基线模型和论文 P4 的受保护 Attention--Koopman--T--S 正常模型，使
    ``joff.core.factory.build_model`` 在按需导入后完成显式注册。
主要职责：
    只做模型类/配置再导出并触发各模型模块底部的注册语句；不构造模型、不读取数据。
关键输入与输出：
    本文件无运行时输入；输出为 ``joff.models`` 命名空间中的稳定公共对象。
依赖与副作用：
    导入模型定义会更新进程内模型注册表；不会创建目录、固定随机种子或启动训练。
重要约束：
    注册名称必须稳定且不使用任意动态导入；论文模型不得在导入时创建在线状态或产物。
"""

from .attention import Attention, AttentionMaskFactory, MaskedMultiheadAttention
from .autoencoder import DAE
from .base import BaseModel
from .control import ARX, Observer
from .flow import NICE
from .gan import GAN, Discriminator, Generator
from .koopman import NKN
from .mlp import MLP
from .protected_koopman_ts import ProtectedKoopmanTS, ProtectedKoopmanTSConfig
from .rnn import SequenceRegressor
from .vae import VAE

__all__ = [
    "Attention",
    "AttentionMaskFactory",
    "ARX",
    "BaseModel",
    "DAE",
    "Discriminator",
    "GAN",
    "Generator",
    "MLP",
    "MaskedMultiheadAttention",
    "NICE",
    "NKN",
    "Observer",
    "ProtectedKoopmanTS",
    "ProtectedKoopmanTSConfig",
    "SequenceRegressor",
    "VAE",
]
