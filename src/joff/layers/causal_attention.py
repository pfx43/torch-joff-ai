"""严格过去的因果注意力编码与可复现测量通道掩码。

文件用途：
    为 P4 正常模型把 ``k`` 时刻之前的控制/测量历史与截至 ``k`` 的外生工况编码为
    潜变量和 context，并提供训练期固定分布的 channel mask。公开接口根本不接收当前
    ``y_k``，从结构上保留严格过去测量边界。
主要职责：
    定义严格配置；按 all-pass、单通道、独立多通道三种模式采样可重放 mask；把逐时刻
    token 送入 causal multi-head self-attention；返回锚点潜变量、context 和注意力权重。
    本文件不实现 Koopman 动力学、自由 rollout、告警、在线 anchor 状态机或数据切分。
关键输入与输出：
    输入形状为 ``past_u=[B,H,m_u]``、``past_y=[B,H,m_y]``、
    ``past_xi=[B,H,m_xi]``、``current_xi=[B,m_xi]`` 和可选布尔 keep mask；输出
    潜变量 ``[B,m_z]``、context ``[B,m_chi]``、逐头权重与实际 mask。
依赖与副作用：
    依赖 PyTorch 和 Joff ``StrictConfig``。编码前向只操作 tensor；mask sampler 维护一个
    持久化 draw counter，使相同配置和 checkpoint 能继续同一随机序列，不读写文件。
重要约束：
    attention 的任一 query 只能读取同一时刻及更早 token；确定性位置编码必须保留时间
    顺序；当前测量和真实未来输出不能进入本接口。mask 在完整历史上按通道一致应用。
"""

from __future__ import annotations

import math

import torch
from pydantic import Field, PositiveInt, model_validator
from torch import nn

from joff.core.config import StrictConfig


class ChannelMaskConfig(StrictConfig):
    """训练期测量通道 mask 的冻结概率与随机种子。

    参数：
        all_pass_probability: 完全保留全部测量通道的样本概率。
        single_channel_probability: 恰好屏蔽一个随机测量通道的样本概率。
        independent_drop_probability: 剩余概率质量对应的独立多通道模式中，每个通道被
            屏蔽的概率。
        seed: stateless draw 序列的基准种子；持久化 draw counter 决定后续子种子。
    异常：
        概率不在 ``[0,1]`` 或前两种互斥模式概率之和超过 1 时由 Pydantic 抛出
        ``ValidationError``。
    副作用：
        无；配置冻结。
    """

    all_pass_probability: float = Field(ge=0.0, le=1.0)
    single_channel_probability: float = Field(ge=0.0, le=1.0)
    independent_drop_probability: float = Field(ge=0.0, le=1.0)
    seed: int

    @model_validator(mode="after")
    def _validate_mode_probabilities(self) -> "ChannelMaskConfig":
        """校验三个互斥采样模式具有合法概率质量。

        参数：
            无。
        返回：
            当前冻结配置。
        异常：
            all-pass 与 single-channel 概率和超过 1 时抛出 ``ValueError``。
        副作用：
            无。
        """

        if self.all_pass_probability + self.single_channel_probability > 1.0:
            raise ValueError(
                "Channel mask all-pass and single-channel probabilities must sum to at most one."
            )
        return self


class CausalAttentionConfig(StrictConfig):
    """严格过去编码器的注意力结构配置。

    参数：
        embed_dim: 每个历史 token 的嵌入宽度。
        num_heads: multi-head attention 头数，必须整除 ``embed_dim``。
        dropout: attention 权重 dropout；确定性验证通常显式设为 0。
    异常：
        宽度/头数非正、dropout 超界或宽度不能整除头数时由 Pydantic 抛出
        ``ValidationError``。
    副作用：
        无；配置冻结。
    """

    embed_dim: PositiveInt
    num_heads: PositiveInt
    dropout: float = Field(ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _validate_head_width(self) -> "CausalAttentionConfig":
        """保证每个 attention head 得到整数特征宽度。

        参数：
            无。
        返回：
            当前冻结配置。
        异常：
            ``embed_dim`` 不能被 ``num_heads`` 整除时抛出 ``ValueError``。
        副作用：
            无。
        """

        if self.embed_dim % self.num_heads != 0:
            raise ValueError("Causal attention embed_dim must be divisible by num_heads.")
        return self


class ChannelMaskSampler(nn.Module):
    """从冻结分布采样历史测量 keep mask。

    参数：
        measurement_dim: 测量通道数，必须为正。
        config: 严格概率配置和基准随机种子。
    异常：
        测量维数非法时抛出 ``ValueError``。
    副作用：
        注册持久化 ``draw_count`` buffer；每次 ``sample`` 成功后加一。buffer 会进入
        ``state_dict``，checkpoint 恢复后继续同一 mask 序列。
    """

    draw_count: torch.Tensor

    def __init__(self, *, measurement_dim: int, config: ChannelMaskConfig) -> None:
        super().__init__()
        if measurement_dim <= 0:
            raise ValueError("ChannelMaskSampler measurement_dim must be positive.")
        self.measurement_dim = int(measurement_dim)
        self.config = config
        self.register_buffer(
            "draw_count",
            torch.zeros((), dtype=torch.int64),
            persistent=True,
        )

    def sample(
        self,
        *,
        batch_size: int,
        history_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """采样并扩展为完整历史上的布尔 keep mask。

        参数：
            batch_size: 独立样本数。
            history_length: 每个样本的严格过去 token 数。
            device: 返回 mask 的目标设备；随机选择先在 CPU stateless generator 上完成。
        返回：
            形状 ``[B,H,m_y]`` 的布尔 tensor；真表示保留，假表示以零替代该通道。
        异常：
            batch 或历史长度非正时抛出 ``ValueError``。
        副作用：
            成功后递增持久化 ``draw_count``；不修改 PyTorch 全局 RNG。
        """

        if batch_size <= 0 or history_length <= 0:
            raise ValueError("Channel mask batch_size and history_length must be positive.")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.seed + int(self.draw_count))
        mode_draw = torch.rand(batch_size, generator=generator)
        keep_channels = torch.ones(
            batch_size,
            self.measurement_dim,
            dtype=torch.bool,
        )
        all_pass_end = self.config.all_pass_probability
        single_end = all_pass_end + self.config.single_channel_probability
        single_rows = (mode_draw >= all_pass_end) & (mode_draw < single_end)
        if torch.any(single_rows):
            row_ids = torch.nonzero(single_rows, as_tuple=False).flatten()
            channel_ids = torch.randint(
                self.measurement_dim,
                (len(row_ids),),
                generator=generator,
            )
            keep_channels[row_ids, channel_ids] = False
        independent_rows = mode_draw >= single_end
        if torch.any(independent_rows):
            row_ids = torch.nonzero(independent_rows, as_tuple=False).flatten()
            drops = torch.rand(
                len(row_ids),
                self.measurement_dim,
                generator=generator,
            ) < self.config.independent_drop_probability
            keep_channels[row_ids] = ~drops
        self.draw_count += 1
        return (
            keep_channels.unsqueeze(1)
            .expand(-1, history_length, -1)
            .clone()
            .to(device=device)
        )


class CausalAttentionEncoder(nn.Module):
    """把严格过去 token 编码为锚点潜变量和 attention context。

    参数：
        control_dim/measurement_dim/exogenous_dim: 每个 token 的物理通道维数；前两者必须
            为正，外生维数可以为零。
        latent_dim/context_dim: 两个输出向量的正维数。
        config: attention 嵌入、头数和 dropout 的严格配置。
    异常：
        任一维数或输入 tensor 形状不合法时抛出 ``ValueError``。
    副作用：
        构造可训练线性层、attention 和 LayerNorm；不创建文件、不修改全局随机状态。
    """

    def __init__(
        self,
        *,
        control_dim: int,
        measurement_dim: int,
        exogenous_dim: int,
        latent_dim: int,
        context_dim: int,
        config: CausalAttentionConfig,
    ) -> None:
        super().__init__()
        positive = {
            "control_dim": control_dim,
            "measurement_dim": measurement_dim,
            "latent_dim": latent_dim,
            "context_dim": context_dim,
        }
        if any(value <= 0 for value in positive.values()) or exogenous_dim < 0:
            raise ValueError(
                "CausalAttentionEncoder requires positive control/measurement/latent/context "
                "dimensions and non-negative exogenous_dim."
            )
        self.control_dim = int(control_dim)
        self.measurement_dim = int(measurement_dim)
        self.exogenous_dim = int(exogenous_dim)
        self.latent_dim = int(latent_dim)
        self.context_dim = int(context_dim)
        self.config = config
        token_dim = self.control_dim + self.measurement_dim + self.exogenous_dim
        self.token_projection = nn.Linear(token_dim, config.embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self.latent_head = nn.Linear(config.embed_dim, self.latent_dim)
        self.context_head = nn.Linear(config.embed_dim, self.context_dim)
        self.current_exogenous_projection = (
            nn.Linear(self.exogenous_dim, config.embed_dim, bias=False)
            if self.exogenous_dim > 0
            else None
        )

    def forward(
        self,
        past_u: torch.Tensor,
        past_y: torch.Tensor,
        past_xi: torch.Tensor | None = None,
        *,
        current_xi: torch.Tensor | None = None,
        measurement_keep_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """编码严格过去历史。

        参数：
            past_u/past_y/past_xi: 共享 ``[B,H]`` 前两维、末维分别匹配配置的 tensor；
                三项历史都结束于 ``k-1``。
            current_xi: ``[B,m_xi]`` 的 ``k`` 时刻已知外生工况；外生维数为正时必需，
                为零时可省略。该字段不是当前测量 ``y_k``。
            measurement_keep_mask: 可选 ``[B,H,m_y]`` 布尔 mask；省略时全部保留。
        返回：
            ``latent``、``context``、完整 sequence、逐头 attention 权重、实际 mask 和
            masked measurement 的字典。
        异常：
            dtype/设备交由 PyTorch检查；rank、批/历史/特征形状或 mask dtype 不匹配时
            抛出 ``ValueError``。
        副作用：
            训练模式下 attention dropout 可消耗 PyTorch RNG；不修改输入 tensor。
        """

        batch_size, history_length = self._validate_histories(past_u, past_y, past_xi)
        current_condition = self._validated_current_exogenous(
            past_u,
            current_xi,
        )
        if measurement_keep_mask is None:
            keep_mask = torch.ones_like(past_y, dtype=torch.bool)
        else:
            if (
                measurement_keep_mask.dtype is not torch.bool
                or measurement_keep_mask.shape != past_y.shape
            ):
                raise ValueError(
                    "measurement_keep_mask must be bool and match past_y shape."
                )
            keep_mask = measurement_keep_mask.to(device=past_y.device)
        masked_y = torch.where(keep_mask, past_y, torch.zeros_like(past_y))
        if self.exogenous_dim == 0:
            xi = past_u.new_empty(batch_size, history_length, 0)
        else:
            assert past_xi is not None
            xi = past_xi
        tokens = torch.cat((past_u, masked_y, xi), dim=-1)
        embedded = self.token_projection(tokens)
        embedded = embedded + self._position_encoding(
            history_length,
            dtype=embedded.dtype,
            device=embedded.device,
        ).unsqueeze(0)
        blocked_future = torch.triu(
            torch.ones(
                history_length,
                history_length,
                dtype=torch.bool,
                device=embedded.device,
            ),
            diagonal=1,
        )
        attended, weights = self.attention(
            embedded,
            embedded,
            embedded,
            attn_mask=blocked_future,
            need_weights=True,
            average_attn_weights=False,
        )
        sequence = torch.tanh(self.norm(embedded + attended))
        final_token = sequence[:, -1, :]
        if self.current_exogenous_projection is not None:
            final_token = torch.tanh(
                final_token + self.current_exogenous_projection(current_condition)
            )
        return {
            "latent": self.latent_head(final_token),
            "context": self.context_head(final_token),
            "sequence": sequence,
            "attention_weights": weights,
            "measurement_keep_mask": keep_mask,
            "masked_past_y": masked_y,
        }

    def _validate_histories(
        self,
        past_u: torch.Tensor,
        past_y: torch.Tensor,
        past_xi: torch.Tensor | None,
    ) -> tuple[int, int]:
        """校验三个历史 tensor 的 rank、共享轴和通道维数。

        参数：
            past_u/past_y/past_xi: ``forward`` 收到的原始 tensor。
        返回：
            ``(batch_size, history_length)``。
        异常：
            rank、批/历史轴、通道维数或必需外生历史不匹配时抛出 ``ValueError``。
        副作用：
            无。
        """

        if past_u.ndim != 3 or past_y.ndim != 3:
            raise ValueError("Strict-past control and measurement histories must be 3D.")
        if past_u.shape[:2] != past_y.shape[:2] or past_u.shape[0] <= 0 or past_u.shape[1] <= 0:
            raise ValueError("Strict-past histories must share non-empty batch/time axes.")
        if past_u.shape[-1] != self.control_dim or past_y.shape[-1] != self.measurement_dim:
            raise ValueError("Strict-past history feature dimensions do not match encoder config.")
        if self.exogenous_dim == 0:
            if past_xi is not None and (
                past_xi.ndim != 3
                or past_xi.shape[:2] != past_u.shape[:2]
                or past_xi.shape[-1] != 0
            ):
                raise ValueError("Zero-dimensional past_xi must be omitted or have last dim zero.")
        elif (
            past_xi is None
            or past_xi.ndim != 3
            or past_xi.shape[:2] != past_u.shape[:2]
            or past_xi.shape[-1] != self.exogenous_dim
        ):
            raise ValueError("past_xi shape does not match encoder exogenous_dim.")
        return int(past_u.shape[0]), int(past_u.shape[1])

    def _validated_current_exogenous(
        self,
        reference: torch.Tensor,
        current_xi: torch.Tensor | None,
    ) -> torch.Tensor:
        """校验并规范化当前外生工况。

        参数：
            reference: 提供 batch、device 和 dtype 的历史控制 tensor。
            current_xi: ``[B,m_xi]`` 当前工况；零维时可省略。
        返回：
            总是存在的 ``[B,m_xi]`` tensor。
        异常：
            外生维数为正但缺失，或 batch/通道维不匹配时抛出 ``ValueError``。
        副作用：
            无。
        """

        batch_size = reference.shape[0]
        if self.exogenous_dim == 0:
            if current_xi is not None and current_xi.shape != (batch_size, 0):
                raise ValueError(
                    "Zero-dimensional current_xi must be omitted or have last dim zero."
                )
            return reference.new_empty(batch_size, 0)
        if current_xi is None or current_xi.shape != (
            batch_size,
            self.exogenous_dim,
        ):
            raise ValueError("current_xi shape does not match encoder exogenous_dim.")
        return current_xi

    def _position_encoding(
        self,
        history_length: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """生成无额外超参数的确定性正弦位置编码。

        参数：
            history_length/dtype/device: token 数及目标数值环境。
        返回：
            ``[H,embed_dim]`` 常量；不同位置具有不同相位，使 token 置换可被网络识别。
        异常：
            历史正长度已由调用方校验。
        副作用：
            无；不创建持久状态、不消耗随机数。
        """

        width = self.config.embed_dim
        positions = torch.arange(
            history_length,
            dtype=dtype,
            device=device,
        ).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, width, 2, dtype=dtype, device=device)
            * (-math.log(10000.0) / width)
        )
        angles = positions * frequencies.unsqueeze(0)
        encoding = torch.zeros(
            history_length,
            width,
            dtype=dtype,
            device=device,
        )
        encoding[:, 0::2] = torch.sin(angles)
        encoding[:, 1::2] = torch.cos(angles[:, : encoding[:, 1::2].shape[1]])
        return encoding


__all__ = [
    "CausalAttentionConfig",
    "CausalAttentionEncoder",
    "ChannelMaskConfig",
    "ChannelMaskSampler",
]
