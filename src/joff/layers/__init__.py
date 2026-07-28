"""Joff 可复用神经层的稳定公共导入入口。

文件用途：
    汇总激活函数、MLP builder，以及论文 P4 的严格过去 causal attention 和 channel mask
    对象，供模型层通过稳定路径导入。
主要职责：
    只做显式再导出并维护 ``__all__``；不构造网络、不采样 mask、不修改注册表。
关键输入与输出：
    本文件没有运行时输入；输出是 ``joff.layers`` 命名空间下受支持的配置、层和 builder。
依赖与副作用：
    导入 PyTorch 层定义与 Pydantic 配置定义；模块导入本身不固定种子、不创建 tensor 或
    文件。
重要约束：
    严格过去编码器不得扩展出当前测量参数；新增层应保持通用，不拥有实验或监视状态。
"""

from .activations import (
    Gaussian,
    Identity,
    LearnableAffine,
    Square,
    SwiGLU,
    activation_changes_feature_dim,
    build_activation,
    register_builtin_activations,
)
from .builder import build_mlp, dropout_rate_for_width, resolve_widths
from .causal_attention import (
    CausalAttentionConfig,
    CausalAttentionEncoder,
    ChannelMaskConfig,
    ChannelMaskSampler,
)
from .fuzzy_koopman import FuzzyKoopmanConfig, FuzzyKoopmanTransition

__all__ = [
    "Gaussian",
    "Identity",
    "LearnableAffine",
    "CausalAttentionConfig",
    "CausalAttentionEncoder",
    "ChannelMaskConfig",
    "ChannelMaskSampler",
    "FuzzyKoopmanConfig",
    "FuzzyKoopmanTransition",
    "Square",
    "SwiGLU",
    "activation_changes_feature_dim",
    "build_activation",
    "build_mlp",
    "dropout_rate_for_width",
    "register_builtin_activations",
    "resolve_widths",
]
