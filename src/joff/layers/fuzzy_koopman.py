"""可学习 T--S 隶属度、局部 Koopman 动力学与完整解析 Jacobian。

文件用途：
    为 P4 正常模型实现论文中的 learned premise、softmax T--S 权重和局部
    ``A_i z + B_i u + o_i``，并显式装配包含 membership variation 的完整 Jacobian。
主要职责：
    定义严格配置；用 smooth bottleneck MLP 生成 premise；用有界正对角度量计算二次规则
    分数；对局部 Koopman 做凸组合；返回局部/组合算子、完整状态/输入/context Jacobian。
    本文件不编码历史、不解码测量、不执行多步 rollout 或任何监测决策。
关键输入与输出：
    输入为 ``z=[B,m_z]``、``u=[B,m_u]``、``xi=[B,m_xi]``、context
    ``[B,m_chi]``；输出包括 ``next_latent``、premise、规则分数/权重、局部预测、局部和
    组合算子，以及 ``jacobian_z/u/context/xi``。
依赖与副作用：
    只依赖 PyTorch 和 ``StrictConfig``。构造时按 PyTorch 当前 RNG 初始化参数；前向
    完全是可微 tensor 运算，不读写文件、不修改全局状态。
重要约束：
    权重必须非负且逐样本和为 1；单规则必须精确退化；度量谱位于显式区间；局部 A 使用
    Frobenius 有界的平滑参数化，从而保守限制谱范数；完整 Jacobian 不得只返回
    ``sum_i omega_i A_i``，必须包含规则权重随 premise/状态变化的项。
"""

from __future__ import annotations

import torch
from pydantic import PositiveFloat, PositiveInt, model_validator
from torch import nn

from joff.core.config import StrictConfig


class FuzzyKoopmanConfig(StrictConfig):
    """T--S premise、度量和局部动力学的严格配置。

    参数：
        rule_count: 局部 Koopman 规则数。
        premise_dim: 二次隶属度所在的 premise 维数。
        premise_hidden_dim: smooth bottleneck MLP 隐藏宽度。
        metric_eigenvalue_min/max: 正对角规则度量允许的闭区间。
        spectral_cap: 每个局部 ``A_i`` 的严格正 Frobenius 上界，因而也是谱范数上界。
    异常：
        正数字段非法或度量上界不大于下界时由 Pydantic 抛出 ``ValidationError``。
    副作用：
        无；配置冻结。
    """

    rule_count: PositiveInt
    premise_dim: PositiveInt
    premise_hidden_dim: PositiveInt
    metric_eigenvalue_min: PositiveFloat
    metric_eigenvalue_max: PositiveFloat
    spectral_cap: PositiveFloat

    @model_validator(mode="after")
    def _validate_metric_interval(self) -> "FuzzyKoopmanConfig":
        """保证度量谱区间具有正宽度。

        参数：
            无。
        返回：
            当前冻结配置。
        异常：
            ``metric_eigenvalue_max <= metric_eigenvalue_min`` 时抛出 ``ValueError``。
        副作用：
            无。
        """

        if self.metric_eigenvalue_max <= self.metric_eigenvalue_min:
            raise ValueError(
                "Fuzzy metric_eigenvalue_max must exceed metric_eigenvalue_min."
            )
        return self


class _SmoothPremiseNetwork(nn.Module):
    """一层 tanh bottleneck premise 网络及其输入解析 Jacobian。"""

    def __init__(self, input_dim: int, hidden_dim: int, premise_dim: int) -> None:
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, premise_dim)

    def forward_with_jacobian(
        self,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """同时计算 premise 与对拼接输入的解析 Jacobian。

        参数：
            value: 形状 ``[B,input_dim]`` 的拼接 ``(z,context,u,xi)``。
        返回：
            ``rho=[B,m_rho]`` 和 ``J=[B,m_rho,input_dim]``。
        异常：
            形状或 dtype 非法时由 ``nn.Linear``/``einsum`` 抛出 PyTorch 异常。
        副作用：
            无；所有运算保留 autograd 图。
        """

        hidden = torch.tanh(self.input_layer(value))
        premise = self.output_layer(hidden)
        activation_derivative = 1.0 - hidden.square()
        jacobian = torch.einsum(
            "ph,bh,hi->bpi",
            self.output_layer.weight,
            activation_derivative,
            self.input_layer.weight,
        )
        return premise, jacobian


class FuzzyKoopmanTransition(nn.Module):
    """执行一步可学习 T--S 局部 Koopman 转移。

    参数：
        latent_dim/control_dim/context_dim: 潜变量、记录控制和 attention context 的正维数。
        exogenous_dim: premise 使用的外生条件维数，可以为零。
        config: 规则数、premise 网络、度量边界和局部 A 上界。
    异常：
        维数非法时抛出 ``ValueError``。
    副作用：
        创建可训练 premise、prototype、度量、规则偏置和局部 ``A/B/o`` 参数；不写文件。
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        control_dim: int,
        exogenous_dim: int,
        context_dim: int,
        config: FuzzyKoopmanConfig,
    ) -> None:
        super().__init__()
        if (
            latent_dim <= 0
            or control_dim <= 0
            or context_dim <= 0
            or exogenous_dim < 0
        ):
            raise ValueError(
                "FuzzyKoopmanTransition requires positive latent/control/context dimensions "
                "and non-negative exogenous_dim."
            )
        self.latent_dim = int(latent_dim)
        self.control_dim = int(control_dim)
        self.exogenous_dim = int(exogenous_dim)
        self.context_dim = int(context_dim)
        self.config = config
        premise_input_dim = (
            self.latent_dim
            + self.context_dim
            + self.control_dim
            + self.exogenous_dim
        )
        self.premise_network = _SmoothPremiseNetwork(
            premise_input_dim,
            config.premise_hidden_dim,
            config.premise_dim,
        )
        self.prototypes = nn.Parameter(
            torch.empty(config.rule_count, config.premise_dim)
        )
        self.raw_metric_diagonal = nn.Parameter(
            torch.zeros(config.rule_count, config.premise_dim)
        )
        self.rule_bias = nn.Parameter(torch.zeros(config.rule_count))
        identity = torch.eye(self.latent_dim).unsqueeze(0).repeat(
            config.rule_count,
            1,
            1,
        )
        self.raw_A = nn.Parameter(0.5 * identity)
        self.local_B = nn.Parameter(
            torch.empty(config.rule_count, self.latent_dim, self.control_dim)
        )
        self.local_offset = nn.Parameter(
            torch.zeros(config.rule_count, self.latent_dim)
        )
        nn.init.normal_(self.prototypes, mean=0.0, std=0.25)
        nn.init.xavier_uniform_(self.local_B)

    def forward(
        self,
        latent: torch.Tensor,
        control: torch.Tensor,
        exogenous: torch.Tensor | None,
        context: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """计算一步局部预测、T--S 组合和完整 Jacobian。

        参数：
            latent/control/context: 形状分别为 ``[B,m_z]``、``[B,m_u]``、
                ``[B,m_chi]`` 的同批 tensor。
            exogenous: ``[B,m_xi]``；当 ``m_xi=0`` 时可省略。
        返回：
            premise、rule scores/weights、局部预测与算子、组合算子、
            ``next_latent``，以及对 latent/control/context/exogenous 的完整 Jacobian。
        异常：
            rank、批轴或特征维数不匹配时抛出 ``ValueError``。
        副作用：
            无。
        """

        xi = self._validated_exogenous(latent, control, exogenous, context)
        premise_input = torch.cat((latent, context, control, xi), dim=-1)
        premise, premise_jacobian = self.premise_network.forward_with_jacobian(
            premise_input
        )
        metric = self.metric_diagonal()
        premise_delta = premise.unsqueeze(1) - self.prototypes.unsqueeze(0)
        rule_scores = (
            -0.5 * torch.sum(
                metric.unsqueeze(0) * premise_delta.square(),
                dim=-1,
            )
            + self.rule_bias.unsqueeze(0)
        )
        weights = torch.softmax(rule_scores, dim=-1)
        local_A = self.bounded_local_A()
        local_next = (
            torch.einsum("rij,bj->bri", local_A, latent)
            + torch.einsum("rij,bj->bri", self.local_B, control)
            + self.local_offset.unsqueeze(0)
        )
        next_latent = torch.einsum("br,bri->bi", weights, local_next)
        combined_A = torch.einsum("br,rij->bij", weights, local_A)
        combined_B = torch.einsum("br,rij->bij", weights, self.local_B)
        combined_offset = torch.einsum("br,ri->bi", weights, self.local_offset)

        score_gradients = -metric.unsqueeze(0) * premise_delta
        mean_score_gradient = torch.einsum(
            "br,brp->bp",
            weights,
            score_gradients,
        )
        centered_next = local_next - next_latent.unsqueeze(1)
        centered_score_gradients = (
            score_gradients - mean_score_gradient.unsqueeze(1)
        )
        membership_dispersion = torch.einsum(
            "br,brz,brp->bzp",
            weights,
            centered_next,
            centered_score_gradients,
        )
        z_stop = self.latent_dim
        context_stop = z_stop + self.context_dim
        control_stop = context_stop + self.control_dim
        premise_jacobian_z = premise_jacobian[:, :, :z_stop]
        premise_jacobian_context = premise_jacobian[:, :, z_stop:context_stop]
        premise_jacobian_u = premise_jacobian[:, :, context_stop:control_stop]
        premise_jacobian_xi = premise_jacobian[:, :, control_stop:]
        membership_z = torch.matmul(
            membership_dispersion,
            premise_jacobian_z,
        )
        membership_u = torch.matmul(
            membership_dispersion,
            premise_jacobian_u,
        )
        membership_context = torch.matmul(
            membership_dispersion,
            premise_jacobian_context,
        )
        membership_xi = torch.matmul(
            membership_dispersion,
            premise_jacobian_xi,
        )
        return {
            "next_latent": next_latent,
            "premise": premise,
            "rule_scores": rule_scores,
            "rule_weights": weights,
            "local_next_latent": local_next,
            "local_A": local_A,
            "local_B": self.local_B,
            "local_offset": self.local_offset,
            "combined_A": combined_A,
            "combined_B": combined_B,
            "combined_offset": combined_offset,
            "metric_diagonal": metric,
            "membership_dispersion": membership_dispersion,
            "jacobian_z": combined_A + membership_z,
            "jacobian_u": combined_B + membership_u,
            "jacobian_context": membership_context,
            "jacobian_xi": membership_xi,
        }

    def metric_diagonal(self) -> torch.Tensor:
        """返回位于配置闭区间内的正对角度量特征值。

        参数：
            无。
        返回：
            ``[rules,premise_dim]`` 正 tensor。
        异常：
            无。
        副作用：
            无；返回值保留对 raw 参数的梯度。
        """

        lower = self.config.metric_eigenvalue_min
        width = self.config.metric_eigenvalue_max - lower
        return lower + width * torch.sigmoid(self.raw_metric_diagonal)

    def bounded_local_A(self) -> torch.Tensor:
        """用平滑 Frobenius 缩放返回谱范数受限的局部 A。

        参数：
            无。
        返回：
            ``[rules,m_z,m_z]``；每条规则的 Frobenius 范数严格小于 ``spectral_cap``。
        异常：
            无。
        副作用：
            无；参数化保留梯度，且无需不可微的硬 clipping。
        """

        squared_norm = self.raw_A.square().sum(dim=(-2, -1), keepdim=True)
        cap = self.config.spectral_cap
        scale = cap / torch.sqrt(self.raw_A.new_tensor(cap * cap) + squared_norm)
        return self.raw_A * scale

    def _validated_exogenous(
        self,
        latent: torch.Tensor,
        control: torch.Tensor,
        exogenous: torch.Tensor | None,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """校验一步输入并规范化零维外生 tensor。

        参数：
            latent/control/exogenous/context: ``forward`` 原始输入。
        返回：
            总是存在的 ``[B,m_xi]`` tensor；零维时返回同设备空末维。
        异常：
            rank、批轴或特征维数不匹配时抛出 ``ValueError``。
        副作用：
            无。
        """

        tensors = (latent, control, context)
        if any(value.ndim != 2 for value in tensors):
            raise ValueError("Fuzzy Koopman latent, control and context must be 2D.")
        batch_size = latent.shape[0]
        if (
            batch_size <= 0
            or control.shape[0] != batch_size
            or context.shape[0] != batch_size
            or latent.shape[-1] != self.latent_dim
            or control.shape[-1] != self.control_dim
            or context.shape[-1] != self.context_dim
        ):
            raise ValueError("Fuzzy Koopman batch or feature dimensions do not match config.")
        if self.exogenous_dim == 0:
            if exogenous is not None and (
                exogenous.ndim != 2
                or exogenous.shape != (batch_size, 0)
            ):
                raise ValueError(
                    "Zero-dimensional exogenous input must be omitted or have last dim zero."
                )
            return latent.new_empty(batch_size, 0)
        if (
            exogenous is None
            or exogenous.ndim != 2
            or exogenous.shape != (batch_size, self.exogenous_dim)
        ):
            raise ValueError("Fuzzy Koopman exogenous shape does not match config.")
        return exogenous


__all__ = ["FuzzyKoopmanConfig", "FuzzyKoopmanTransition"]
