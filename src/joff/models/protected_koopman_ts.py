"""严格过去 Attention--Koopman--T--S 正常模型与自由多步展开。

文件用途：
    实现论文 P4 的公开正常模型：用 ``k`` 时刻之前的控制、测量和外生历史编码
    ``z_k`` 与 attention context，再通过可学习 T--S 局部 Koopman 动力学执行自由多步
    预测。
主要职责：
    定义嵌套严格模型配置；组合 causal attention、channel mask、fuzzy Koopman 和线性
    输出解码器；提供 ``forward``、可复用 ``rollout`` 与 Trainer 兼容的 ``compute_loss``；
    返回预测、潜变量、context、规则权重、局部/组合算子及完整 Jacobian 轨迹。
关键输入与输出：
    锚点输入为 ``past_*=[B,H,*]``，记录的未来控制/外生量为
    ``future_*=[B,N,*]``；训练时可额外提供 ``target_past_*=[B,N,H,*]`` 和输出目标。
    输出 ``prediction=[B,N,m_y]``、``latent_trajectory=[B,N+1,m_z]`` 及逐步算子字典。
依赖与副作用：
    依赖 PyTorch、Pydantic、Joff 层与训练损失。构造会初始化可训练参数；训练模式下
    ``rollout`` 会推进持久化 channel-mask 抽样计数。模型不读写数据、目录或 checkpoint，
    不修改 Matplotlib 或进程级全局状态。
重要约束：
    当前真实 ``y_k``、真实未来输出和目标历史只允许进入监督路径，绝不能反馈到自由
    rollout。受保护历史在锚点后只写入模型解码值；未来控制和外生量视为记录命令，不由
    模型猜测。P4 不实现告警、anchor 状态机、阈值、后滤波或隔离。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import torch
from pydantic import NonNegativeInt, PositiveInt, model_validator
from torch import nn

from joff.core.config import StrictConfig
from joff.core.factory import register_model
from joff.layers import (
    CausalAttentionConfig,
    CausalAttentionEncoder,
    ChannelMaskConfig,
    ChannelMaskSampler,
    FuzzyKoopmanConfig,
    FuzzyKoopmanTransition,
)
from joff.training import ProtectedLossConfig, ProtectedMultiHorizonLoss

from .base import BaseModel, LossOutput


class ProtectedKoopmanTSConfig(StrictConfig):
    """P4 模型的全部结构、随机掩码和损失配置。

    参数：
        type: 注册表稳定名称，只允许 ``protected_koopman_ts``。
        control_dim/measurement_dim: 记录控制与测量的正通道数。
        exogenous_dim: 外生运行条件通道数，可以为零。
        history_length: 每次严格过去编码使用的固定 token 数。
        latent_dim/context_dim: 潜变量和 attention context 宽度。
        max_rollout: 一次自由展开允许的最大预测步数。
        horizon_seed: 训练期均匀视野抽样的显式基准种子。
        attention/channel_mask/fuzzy/loss: 分别控制编码、训练掩码、局部动力学和分项损失。
    返回：
        冻结且拒绝未知字段的 Pydantic 配置。
    异常：
        维数符号、嵌套字段或 horizon 权重覆盖不足时抛出 Pydantic ``ValidationError``。
    副作用：
        无。
    """

    type: Literal["protected_koopman_ts"]
    control_dim: PositiveInt
    measurement_dim: PositiveInt
    exogenous_dim: NonNegativeInt
    history_length: PositiveInt
    latent_dim: PositiveInt
    context_dim: PositiveInt
    max_rollout: PositiveInt
    horizon_seed: int
    attention: CausalAttentionConfig
    channel_mask: ChannelMaskConfig
    fuzzy: FuzzyKoopmanConfig
    loss: ProtectedLossConfig

    @model_validator(mode="after")
    def _validate_loss_horizon(self) -> "ProtectedKoopmanTSConfig":
        """保证损失权重覆盖模型允许的全部自由视野。

        参数：
            无。
        返回：
            当前冻结配置。
        异常：
            ``loss.horizon_weights`` 少于 ``max_rollout`` 时抛出 ``ValueError``。
        副作用：
            无。
        """

        if len(self.loss.horizon_weights) < self.max_rollout:
            raise ValueError(
                "Protected loss horizon_weights must cover every configured rollout step."
            )
        return self


class ProtectedKoopmanTS(BaseModel):
    """组合严格过去编码器、T--S Koopman 转移和输出解码器。

    参数：
        config: 已通过严格校验的 ``ProtectedKoopmanTSConfig``。
    返回：
        可由 ``build_model`` 构造的 PyTorch 模型。
    异常：
        输入批缺字段、目标历史只给一部分、shape 不符或 horizon 越界时抛出
        ``KeyError``、``TypeError`` 或 ``ValueError``。
    副作用：
        构造可训练层；训练模式下前向会推进 ``channel_mask_sampler.draw_count``。
    """

    model_type = "protected_koopman_ts"
    config_model = ProtectedKoopmanTSConfig
    horizon_draw_count: torch.Tensor

    def __init__(self, config: ProtectedKoopmanTSConfig) -> None:
        super().__init__(config)  # type: ignore[arg-type]
        self.protected_config = config
        self.encoder = CausalAttentionEncoder(
            control_dim=config.control_dim,
            measurement_dim=config.measurement_dim,
            exogenous_dim=config.exogenous_dim,
            latent_dim=config.latent_dim,
            context_dim=config.context_dim,
            config=config.attention,
        )
        self.channel_mask_sampler = ChannelMaskSampler(
            measurement_dim=config.measurement_dim,
            config=config.channel_mask,
        )
        self.transition = FuzzyKoopmanTransition(
            latent_dim=config.latent_dim,
            control_dim=config.control_dim,
            exogenous_dim=config.exogenous_dim,
            context_dim=config.context_dim,
            config=config.fuzzy,
        )
        self.output_decoder = nn.Linear(config.latent_dim, config.measurement_dim)
        self.loss_function = ProtectedMultiHorizonLoss(config.loss)
        self.register_buffer(
            "horizon_draw_count",
            torch.zeros((), dtype=torch.int64),
            persistent=True,
        )

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """执行训练和推理共用的自由多步路径。

        参数：
            batch: 必含 ``past_u``、``past_y``、``future_u``；外生维数为正时还需
                ``past_xi``、``future_xi``。训练潜变量监督可提供完整的三项
                ``target_past_u/y/xi``。
        返回：
            ``rollout`` 的全部轨迹；若目标历史完整，额外返回只用于损失的
            ``target_latent=[B,N,m_z]``。
        异常：
            batch 不是映射、必需 tensor 缺失、目标历史不完整或形状非法时抛出
            ``TypeError``、``KeyError`` 或 ``ValueError``。
        副作用：
            训练模式下采样一次 channel mask 并推进可持久化计数；不修改 batch。
        """

        if not isinstance(batch, Mapping):
            raise TypeError("ProtectedKoopmanTS batch must be a mapping of named tensors.")
        future_u = self._required_tensor(batch, "future_u")
        future_xi = self._optional_tensor(batch, "future_xi")
        if self.training:
            horizon = self._sample_training_horizon(future_u)
            future_u = future_u[:, :horizon, :]
            if future_xi is not None:
                future_xi = future_xi[:, :horizon, :]
        rollout = self.rollout(
            past_u=self._required_tensor(batch, "past_u"),
            past_y=self._required_tensor(batch, "past_y"),
            past_xi=self._optional_tensor(batch, "past_xi"),
            future_u=future_u,
            future_xi=future_xi,
        )
        target_keys = self._target_history_keys()
        present = tuple(name for name in target_keys if name in batch)
        if present and len(present) != len(target_keys):
            missing = ", ".join(name for name in target_keys if name not in batch)
            raise ValueError(
                "Protected target histories are all-or-none. "
                f"Missing fields: {missing}."
            )
        if len(present) == len(target_keys):
            rollout["target_latent"] = self._encode_target_latents(
                target_past_u=self._required_tensor(batch, "target_past_u"),
                target_past_y=self._required_tensor(batch, "target_past_y"),
                target_past_xi=self._optional_tensor(batch, "target_past_xi"),
                target_current_xi=self._optional_tensor(
                    batch,
                    "target_current_xi",
                ),
                expected_horizon=rollout["prediction"].shape[1],
            )
        return rollout

    def rollout(
        self,
        *,
        past_u: torch.Tensor,
        past_y: torch.Tensor,
        future_u: torch.Tensor,
        past_xi: torch.Tensor | None = None,
        future_xi: torch.Tensor | None = None,
        measurement_keep_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """从严格过去锚点执行不读取未来真实测量的自由展开。

        参数：
            past_u/past_y/past_xi: 以 ``k-1`` 结束的 ``[B,H,*]`` 历史。
            future_u/future_xi: 从 ``k`` 开始的 ``[B,N,*]`` 记录命令和外生量。
            measurement_keep_mask: 可选固定历史 keep mask；省略时训练模式按配置采样，
                评估模式保留全部通道。
        返回：
            锚点解码、未来输出预测、潜变量/context、规则、局部/组合算子、完整 Jacobian
            以及最终只含预测测量的受保护历史缓冲。
        异常：
            历史长度不是配置值、未来视野越界、shape 或外生输入不匹配时抛出
            ``ValueError``。
        副作用：
            训练模式且未显式给 mask 时推进 channel-mask sampler；不修改输入 tensor。
        """

        batch_size, horizon, xi_history, xi_future = self._validate_rollout_inputs(
            past_u=past_u,
            past_y=past_y,
            past_xi=past_xi,
            future_u=future_u,
            future_xi=future_xi,
        )
        keep_mask = measurement_keep_mask
        if keep_mask is None and self.training:
            keep_mask = self.channel_mask_sampler.sample(
                batch_size=batch_size,
                history_length=self.protected_config.history_length,
                device=past_y.device,
            )
        encoded = self.encoder(
            past_u,
            past_y,
            xi_history,
            current_xi=xi_future[:, 0, :],
            measurement_keep_mask=keep_mask,
        )
        actual_keep_mask = encoded["measurement_keep_mask"]
        latent = encoded["latent"]
        context = encoded["context"]
        decoded_anchor = self.output_decoder(latent)
        protected_u = past_u
        protected_y = encoded["masked_past_y"]
        protected_xi = xi_history

        latent_steps = [latent]
        prediction_steps: list[torch.Tensor] = []
        context_steps: list[torch.Tensor] = []
        transition_steps: dict[str, list[torch.Tensor]] = {
            name: []
            for name in (
                "premise",
                "rule_scores",
                "rule_weights",
                "local_next_latent",
                "local_A",
                "local_B",
                "local_offset",
                "combined_A",
                "combined_B",
                "combined_offset",
                "jacobian_z",
                "jacobian_u",
                "jacobian_context",
                "jacobian_xi",
            )
        }
        decoded_current = decoded_anchor
        for step in range(horizon):
            context_steps.append(context)
            transition = self.transition(
                latent,
                future_u[:, step, :],
                xi_future[:, step, :],
                context,
            )
            for name, values in transition_steps.items():
                value = transition[name]
                if name in {"local_A", "local_B", "local_offset"}:
                    value = value.unsqueeze(0).expand(batch_size, *value.shape)
                values.append(value)
            latent = transition["next_latent"]
            latent_steps.append(latent)
            prediction_steps.append(self.output_decoder(latent))

            # context 的历史 measurement slot 只能写入展开前已有的模型解码，
            # 不能写入 current_y、target_future 或任一真实 post-anchor 测量。
            protected_u = torch.cat(
                (protected_u[:, 1:, :], future_u[:, step : step + 1, :]),
                dim=1,
            )
            protected_y = torch.cat(
                (protected_y[:, 1:, :], decoded_current.unsqueeze(1)),
                dim=1,
            )
            protected_xi = torch.cat(
                (protected_xi[:, 1:, :], xi_future[:, step : step + 1, :]),
                dim=1,
            )
            decoded_current = prediction_steps[-1]
            if step + 1 < horizon:
                protected_context = self.encoder(
                    protected_u,
                    protected_y,
                    protected_xi,
                    current_xi=xi_future[:, step + 1, :],
                    measurement_keep_mask=actual_keep_mask,
                )
                context = protected_context["context"]

        result = {
            "decoded_anchor": decoded_anchor,
            "prediction": torch.stack(prediction_steps, dim=1),
            "latent_trajectory": torch.stack(latent_steps, dim=1),
            "context_trajectory": torch.stack(context_steps, dim=1),
            "measurement_keep_mask": actual_keep_mask,
            "protected_past_u": protected_u,
            "protected_past_y": protected_y,
            "protected_past_xi": protected_xi,
        }
        result.update(
            {
                name: torch.stack(values, dim=1)
                for name, values in transition_steps.items()
            }
        )
        return result

    def compute_loss(
        self,
        batch: Mapping[str, torch.Tensor],
        output: Mapping[str, torch.Tensor],
        loss_context: dict[str, object] | None = None,
    ) -> LossOutput:
        """计算自由多视野分项损失。

        参数：
            batch: 含真实 ``current_y``、``target_future`` 和完整目标历史的训练批。
            output: 当前模型 ``forward`` 的结果。
            loss_context: 为 BaseModel 协议保留；P4 不读取该字段。
        返回：
            ``loss`` 总标量和 ``losses`` 命名分量，供 Trainer 反向传播和记录。
        异常：
            监督字段缺失或 shape 不匹配时由 ``ProtectedMultiHorizonLoss`` 抛出异常。
        副作用：
            无；保留所有可训练映射的 autograd 图。
        """

        del loss_context
        return self.loss_function(batch, output)

    def _validate_rollout_inputs(
        self,
        *,
        past_u: torch.Tensor,
        past_y: torch.Tensor,
        past_xi: torch.Tensor | None,
        future_u: torch.Tensor,
        future_xi: torch.Tensor | None,
    ) -> tuple[int, int, torch.Tensor, torch.Tensor]:
        """校验自由展开公共输入并规范化零维外生量。

        参数：
            past_u/past_y/past_xi: 锚点严格过去历史。
            future_u/future_xi: 请求视野内的记录控制和当前外生工况序列。
        返回：
            batch 数、视野，以及总是存在的过去/未来外生 tensor；零维时末维为 0。
        异常：
            rank、固定历史长度、特征宽度、batch 或 ``max_rollout`` 不匹配时抛出
            ``ValueError``。
        副作用：
            无；不修改输入。
        """

        if past_u.ndim != 3 or past_y.ndim != 3 or future_u.ndim != 3:
            raise ValueError("Protected histories and future_u must be 3D tensors.")
        batch_size = int(past_u.shape[0])
        horizon = int(future_u.shape[1])
        if (
            batch_size <= 0
            or past_u.shape[:2] != past_y.shape[:2]
            or past_u.shape[1] != self.protected_config.history_length
            or past_u.shape[-1] != self.protected_config.control_dim
            or past_y.shape[-1] != self.protected_config.measurement_dim
            or future_u.shape[0] != batch_size
            or future_u.shape[-1] != self.protected_config.control_dim
        ):
            raise ValueError("Protected history/future dimensions do not match model config.")
        if horizon <= 0 or horizon > self.protected_config.max_rollout:
            raise ValueError(
                "Protected rollout horizon must be positive and not exceed max_rollout."
            )
        if self.protected_config.exogenous_dim == 0:
            if past_xi is not None and past_xi.shape != (
                batch_size,
                self.protected_config.history_length,
                0,
            ):
                raise ValueError("Zero-dimensional past_xi must be omitted or end in zero.")
            if future_xi is not None and future_xi.shape != (batch_size, horizon, 0):
                raise ValueError("Zero-dimensional future_xi must be omitted or end in zero.")
            return (
                batch_size,
                horizon,
                past_u.new_empty(
                    batch_size,
                    self.protected_config.history_length,
                    0,
                ),
                future_u.new_empty(batch_size, horizon, 0),
            )
        if (
            past_xi is None
            or past_xi.shape
            != (
                batch_size,
                self.protected_config.history_length,
                self.protected_config.exogenous_dim,
            )
            or future_xi is None
            or future_xi.shape
            != (batch_size, horizon, self.protected_config.exogenous_dim)
        ):
            raise ValueError("Protected exogenous histories do not match model config.")
        return batch_size, horizon, past_xi, future_xi

    def _encode_target_latents(
        self,
        *,
        target_past_u: torch.Tensor,
        target_past_y: torch.Tensor,
        target_past_xi: torch.Tensor | None,
        target_current_xi: torch.Tensor | None,
        expected_horizon: int,
    ) -> torch.Tensor:
        """只在监督支路编码真实目标历史。

        参数：
            target_past_u/y/xi: 每个未来锚点的有序严格过去历史。
            target_current_xi: 每个未来锚点的当前外生工况。
            expected_horizon: 本次训练抽样后的实际监督视野。
        返回：
            ``[B,N,m_z]`` 目标潜变量；只供 latent prediction loss 使用。
        异常：
            目标历史未覆盖抽样视野，或 rank/历史/特征维不匹配时抛出 ``ValueError``。
        副作用：
            调用共享 encoder 并保留 autograd 图；结果绝不写回 rollout recurrence。
        """

        if target_past_u.ndim != 4 or target_past_y.ndim != 4:
            raise ValueError("Protected target histories must be 4D tensors.")
        batch_size, available_horizon, history_length, _ = target_past_u.shape
        if (
            available_horizon < expected_horizon
            or history_length != self.protected_config.history_length
            or target_past_y.shape[:3] != target_past_u.shape[:3]
            or target_past_u.shape[-1] != self.protected_config.control_dim
            or target_past_y.shape[-1] != self.protected_config.measurement_dim
        ):
            raise ValueError("Protected target histories do not match rollout/config dimensions.")
        horizon = expected_horizon
        target_past_u = target_past_u[:, :horizon, :, :]
        target_past_y = target_past_y[:, :horizon, :, :]
        flat_u = target_past_u.reshape(
            batch_size * horizon,
            history_length,
            self.protected_config.control_dim,
        )
        flat_y = target_past_y.reshape(
            batch_size * horizon,
            history_length,
            self.protected_config.measurement_dim,
        )
        if self.protected_config.exogenous_dim == 0:
            if target_past_xi is not None and target_past_xi.shape != (
                batch_size,
                available_horizon,
                history_length,
                0,
            ):
                raise ValueError(
                    "Zero-dimensional target_past_xi must be omitted or end in zero."
                )
            flat_xi = flat_u.new_empty(batch_size * horizon, history_length, 0)
            if target_current_xi is not None and target_current_xi.shape != (
                batch_size,
                available_horizon,
                0,
            ):
                raise ValueError(
                    "Zero-dimensional target_current_xi must be omitted or end in zero."
                )
            flat_current_xi = flat_u.new_empty(batch_size * horizon, 0)
        else:
            if target_past_xi is None or target_past_xi.shape != (
                batch_size,
                available_horizon,
                history_length,
                self.protected_config.exogenous_dim,
            ):
                raise ValueError("Protected target_past_xi does not match model config.")
            flat_xi = target_past_xi[:, :horizon, :, :].reshape(
                batch_size * horizon,
                history_length,
                self.protected_config.exogenous_dim,
            )
            if target_current_xi is None or target_current_xi.shape != (
                batch_size,
                available_horizon,
                self.protected_config.exogenous_dim,
            ):
                raise ValueError("Protected target_current_xi does not match model config.")
            flat_current_xi = target_current_xi[:, :horizon, :].reshape(
                batch_size * horizon,
                self.protected_config.exogenous_dim,
            )
        encoded = self.encoder(
            flat_u,
            flat_y,
            flat_xi,
            current_xi=flat_current_xi,
        )
        return encoded["latent"].reshape(
            batch_size,
            horizon,
            self.protected_config.latent_dim,
        )

    def _target_history_keys(self) -> tuple[str, ...]:
        """返回必须成组出现的潜变量监督字段。

        返回：
            外生维数为零时只含 ``target_past_u/y``；否则还含过去和当前外生字段。
        异常与副作用：
            无。
        """

        if self.protected_config.exogenous_dim == 0:
            return ("target_past_u", "target_past_y")
        return (
            "target_past_u",
            "target_past_y",
            "target_past_xi",
            "target_current_xi",
        )

    def _sample_training_horizon(self, future_u: torch.Tensor) -> int:
        """按论文协议从 ``1..N_max`` 均匀抽取一次训练视野。

        参数：
            future_u: 调用方准备的未来控制，必须至少覆盖 ``max_rollout``。
        返回：
            闭区间 ``[1,max_rollout]`` 内的整数。
        异常：
            future_u 不是三维或可用视野不足时抛出 ``ValueError``，避免悄悄改变分布支持。
        副作用：
            成功后推进持久化 ``horizon_draw_count``；不修改 PyTorch 全局 RNG。
        """

        if (
            future_u.ndim != 3
            or future_u.shape[1] != self.protected_config.max_rollout
        ):
            raise ValueError(
                "Protected training batches must contain exactly max_rollout future steps."
            )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            self.protected_config.horizon_seed + int(self.horizon_draw_count)
        )
        horizon = int(
            torch.randint(
                1,
                self.protected_config.max_rollout + 1,
                (1,),
                generator=generator,
            )
        )
        self.horizon_draw_count += 1
        return horizon

    @staticmethod
    def _required_tensor(
        batch: Mapping[str, torch.Tensor],
        name: str,
    ) -> torch.Tensor:
        """从具名 batch 读取必需 tensor。

        参数：
            batch/name: batch 映射与稳定字段名。
        返回：
            对应 tensor。
        异常：
            字段缺失抛出 ``KeyError``；存在但不是 tensor 抛出 ``TypeError``。
        副作用：
            无。
        """

        if name not in batch:
            raise KeyError(f"Protected batch is missing required tensor {name!r}.")
        value = batch[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Protected batch field {name!r} must be a torch.Tensor.")
        return value

    @staticmethod
    def _optional_tensor(
        batch: Mapping[str, torch.Tensor],
        name: str,
    ) -> torch.Tensor | None:
        """从具名 batch 读取可选 tensor。

        参数：
            batch/name: batch 映射与稳定字段名。
        返回：
            字段不存在时 ``None``，否则返回 tensor。
        异常：
            存在但不是 tensor 时抛出 ``TypeError``，不做隐式转换。
        副作用：
            无。
        """

        if name not in batch:
            return None
        value = batch[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Protected batch field {name!r} must be a torch.Tensor.")
        return value


register_model("protected_koopman_ts", ProtectedKoopmanTS, replace=True)

__all__ = ["ProtectedKoopmanTS", "ProtectedKoopmanTSConfig"]
