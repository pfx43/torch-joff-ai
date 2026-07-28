"""P4 自由多视野正常建模的分项损失与防退化正则。

文件用途：
    为严格过去 Attention--Koopman--T--S 模型组合自由多步潜变量/输出误差、当前解码
    一致性、潜变量方差、规则平衡和完整 Jacobian 乘积惩罚。
主要职责：
    定义无隐藏理论默认值的严格损失配置；校验模型输出和监督目标形状；按显式 horizon
    权重汇总损失；返回 Joff Trainer 兼容的 ``loss``/``losses`` 字典。
关键输入与输出：
    输入 batch 至少含 ``target_future``、``current_y``，模型输出含预测、目标潜变量、
    anchor 解码、规则权重和逐步完整 ``jacobian_z``；输出总标量和六个命名标量分量。
依赖与副作用：
    只依赖 PyTorch、Pydantic 和 ``StrictConfig``。计算保留 autograd 图，不读写文件、
    不采样随机数、不修改模型状态。
重要约束：
    所有监督目标只进入损失，不能反馈到自由 rollout；Jacobian 产品必须使用包含
    membership variation 的 ``jacobian_z``；方差与规则平衡只防退化，不得被解释为
    认证或故障性能。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import torch
from pydantic import Field, NonNegativeFloat, PositiveFloat, model_validator

from joff.core.config import StrictConfig


class ProtectedLossConfig(StrictConfig):
    """P4 多视野损失的全部显式权重和安全尺度。

    参数：
        horizon_weights: 从第一步开始的正权重；长度至少覆盖模型 ``max_rollout``。
        latent/output/decoding/variance/rule_balance/jacobian_product_weight: 六个非负分量
            系数；至少一个预测或解码系数必须为正。
        minimum_latent_std: variance penalty 希望每个潜变量维度达到的正批内标准差。
        maximum_jacobian_product_norm: 有限视野完整 Jacobian 累积 Frobenius 范数软上限。
    异常：
        字段符号非法、horizon 为空或所有预测/解码权重为零时由 Pydantic 抛出
        ``ValidationError``。
    副作用：
        无；配置冻结。
    """

    horizon_weights: tuple[PositiveFloat, ...] = Field(min_length=1)
    latent_weight: NonNegativeFloat
    output_weight: NonNegativeFloat
    decoding_weight: NonNegativeFloat
    variance_weight: NonNegativeFloat
    rule_balance_weight: NonNegativeFloat
    jacobian_product_weight: NonNegativeFloat
    minimum_latent_std: PositiveFloat
    maximum_jacobian_product_norm: PositiveFloat

    @model_validator(mode="after")
    def _validate_supervised_signal(self) -> "ProtectedLossConfig":
        """拒绝完全没有预测/解码监督的配置。

        参数：
            无。
        返回：
            当前冻结配置。
        异常：
            latent、output 和 decoding 三个权重全为零时抛出 ``ValueError``。
        副作用：
            无。
        """

        if (
            self.latent_weight == 0
            and self.output_weight == 0
            and self.decoding_weight == 0
        ):
            raise ValueError(
                "Protected loss requires a positive latent, output or decoding weight."
            )
        return self


class ProtectedDiagnosticsConfig(StrictConfig):
    """P4 正常验证停止条件的显式阈值。

    参数：
        maximum_rmse_ratio: 全视野最大 RMSE 相对首步 RMSE 的允许上限。
        rmse_floor: 首步 RMSE 接近零时使用的正分母下限。
        minimum_rule_usage: 每条规则在样本/视野上的最低平均隶属度。
        minimum_latent_std: 锚点潜变量每一维允许的最低样本标准差。
    异常：
        非正比例/floor/std 或规则占用不在 ``[0,1]`` 时由 Pydantic 拒绝。
    副作用：
        无；配置冻结。
    """

    maximum_rmse_ratio: PositiveFloat
    rmse_floor: PositiveFloat
    minimum_rule_usage: float = Field(ge=0.0, le=1.0)
    minimum_latent_std: PositiveFloat


@dataclass(frozen=True)
class ProtectedDiagnosticResult:
    """一次正常验证对三项 P4 停止条件的不可变判断。

    参数：
        horizon_rmse: 每个自由视野的输出均方根误差。
        maximum_rmse_ratio: 观测到的最大/首步（带 floor）误差比。
        rule_usage: 每条 T--S 规则的平均隶属度。
        minimum_latent_std: 锚点潜变量各维标准差的最小值。
        rollout_stable/rules_active/latent_not_collapsed: 三项独立 gate。
    副作用：
        无。
    """

    horizon_rmse: tuple[float, ...]
    maximum_rmse_ratio: float
    rule_usage: tuple[float, ...]
    minimum_latent_std: float
    rollout_stable: bool
    rules_active: bool
    latent_not_collapsed: bool

    @property
    def ready_for_protected_reference(self) -> bool:
        """仅当三项正常验证 gate 全部通过时返回真。"""

        return (
            self.rollout_stable
            and self.rules_active
            and self.latent_not_collapsed
        )

    def to_dict(self) -> dict[str, object]:
        """返回可直接写入 JSON 产物的普通 Python 字典。

        返回：
            包含原始诊断量、各 gate 和总 gate 的字典。
        异常与副作用：
            无。
        """

        return {
            "horizon_rmse": list(self.horizon_rmse),
            "maximum_rmse_ratio": self.maximum_rmse_ratio,
            "rule_usage": list(self.rule_usage),
            "minimum_latent_std": self.minimum_latent_std,
            "rollout_stable": self.rollout_stable,
            "rules_active": self.rules_active,
            "latent_not_collapsed": self.latent_not_collapsed,
            "ready_for_protected_reference": self.ready_for_protected_reference,
        }


class ProtectedModelDiagnostics:
    """在只含正常数据的验证批上执行 P4 fail-closed 停止条件。

    参数：
        config: 误差增长、规则占用和潜变量方差阈值。
    异常：
        输入字段缺失、shape 不一致、规则占用阈值对当前规则数不可能满足时抛出异常。
    副作用：
        无；计算会 detach，不进入训练梯度或改变模型状态。
    """

    def __init__(self, config: ProtectedDiagnosticsConfig) -> None:
        self.config = config

    def evaluate(
        self,
        batch: Mapping[str, torch.Tensor],
        output: Mapping[str, torch.Tensor],
    ) -> ProtectedDiagnosticResult:
        """计算误差视野曲线、规则占用和锚点潜变量离散程度。

        参数：
            batch: 含 ``target_future=[B,N,m_y]`` 的正常验证批。
            output: 模型输出，含 prediction、latent_trajectory 和 rule_weights。
        返回：
            三项独立 gate 和总 gate 的 ``ProtectedDiagnosticResult``。
        异常：
            rank、batch、horizon、通道或规则数不匹配时抛出 ``ValueError``。
        副作用：
            无；结果转换为 CPU Python 标量，便于保存审计产物。
        """

        prediction = output["prediction"]
        latent_trajectory = output["latent_trajectory"]
        rule_weights = output["rule_weights"]
        target = batch["target_future"]
        if (
            prediction.ndim != 3
            or target.ndim != 3
            or target.shape[0] != prediction.shape[0]
            or target.shape[1] < prediction.shape[1]
            or target.shape[2] != prediction.shape[2]
        ):
            raise ValueError(
                "Protected diagnostics prediction/target shapes are incompatible."
            )
        if (
            latent_trajectory.ndim != 3
            or latent_trajectory.shape[0] != prediction.shape[0]
            or latent_trajectory.shape[1] != prediction.shape[1] + 1
            or rule_weights.ndim != 3
            or rule_weights.shape[:2] != prediction.shape[:2]
        ):
            raise ValueError(
                "Protected diagnostics latent/rule trajectories are incompatible."
            )
        rule_count = int(rule_weights.shape[-1])
        if self.config.minimum_rule_usage * rule_count > 1.0 + 1e-12:
            raise ValueError(
                "minimum_rule_usage cannot be simultaneously satisfied by every rule."
            )
        aligned_target = target[:, : prediction.shape[1], :].to(prediction)
        horizon_rmse_tensor = torch.sqrt(
            torch.mean((prediction - aligned_target).square(), dim=(0, 2))
        )
        denominator = torch.clamp(
            horizon_rmse_tensor[0],
            min=self.config.rmse_floor,
        )
        maximum_ratio_tensor = torch.max(horizon_rmse_tensor) / denominator
        rule_usage_tensor = torch.mean(rule_weights, dim=(0, 1))
        latent_std_tensor = torch.std(
            latent_trajectory[:, 0, :],
            dim=0,
            correction=0,
        )
        minimum_latent_std_tensor = torch.min(latent_std_tensor)
        return ProtectedDiagnosticResult(
            horizon_rmse=tuple(
                float(value)
                for value in horizon_rmse_tensor.detach().cpu().tolist()
            ),
            maximum_rmse_ratio=float(maximum_ratio_tensor.detach().cpu()),
            rule_usage=tuple(
                float(value)
                for value in rule_usage_tensor.detach().cpu().tolist()
            ),
            minimum_latent_std=float(minimum_latent_std_tensor.detach().cpu()),
            rollout_stable=bool(
                maximum_ratio_tensor <= self.config.maximum_rmse_ratio
            ),
            rules_active=bool(
                torch.all(rule_usage_tensor >= self.config.minimum_rule_usage)
            ),
            latent_not_collapsed=bool(
                minimum_latent_std_tensor >= self.config.minimum_latent_std
            ),
        )


class ProtectedMultiHorizonLoss:
    """计算 P4 自由多视野训练目标。

    参数：
        config: 冻结分量权重、horizon 权重和防退化尺度。
    异常：
        构造本身无额外异常；配置已严格校验。
    副作用：
        无。
    """

    def __init__(self, config: ProtectedLossConfig) -> None:
        self.config = config

    def __call__(
        self,
        batch: Mapping[str, torch.Tensor],
        output: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """计算总损失和命名分量。

        参数：
            batch: 含 ``target_future=[B,N,m_y]`` 与 ``current_y=[B,m_y]`` 的训练批。
            output: 模型前向结果，至少含 ``prediction``、``target_latent``、
                ``latent_trajectory``、``decoded_anchor``、``rule_weights`` 和
                ``jacobian_z``。
        返回：
            Trainer 兼容字典：``loss`` 为加权总标量，``losses`` 含六个未加权分量。
        异常：
            字段缺失时抛出 ``KeyError``；horizon 超过配置或任一形状不匹配时抛出
            ``ValueError``。
        副作用：
            无；所有分量保留 autograd 图。
        """

        prediction = output["prediction"]
        target_future = batch["target_future"][:, : prediction.shape[1], :].to(
            prediction
        )
        predicted_latent = output["latent_trajectory"][:, 1:, :]
        target_latent = output["target_latent"].to(predicted_latent)
        decoded_anchor = output["decoded_anchor"]
        current_y = batch["current_y"].to(decoded_anchor)
        rule_weights = output["rule_weights"]
        jacobians = output["jacobian_z"]
        if prediction.ndim != 3:
            raise ValueError("Protected prediction must have shape [B,N,m_y].")
        horizon = prediction.shape[1]
        if horizon > len(self.config.horizon_weights):
            raise ValueError("Protected loss horizon exceeds configured horizon_weights.")
        expected_pairs = (
            (target_future.shape, prediction.shape, "target_future"),
            (target_latent.shape, predicted_latent.shape, "target_latent"),
            (current_y.shape, decoded_anchor.shape, "current_y"),
        )
        for actual, expected, name in expected_pairs:
            if actual != expected:
                raise ValueError(
                    f"Protected {name} shape {tuple(actual)} does not match "
                    f"{tuple(expected)}."
                )
        if (
            rule_weights.ndim != 3
            or rule_weights.shape[:2] != prediction.shape[:2]
            or jacobians.ndim != 4
            or jacobians.shape[:2] != prediction.shape[:2]
            or jacobians.shape[-1] != jacobians.shape[-2]
        ):
            raise ValueError("Protected rule/Jacobian trajectories have incompatible shapes.")
        horizon_weight = prediction.new_tensor(
            self.config.horizon_weights[:horizon]
        )
        horizon_weight = horizon_weight / horizon_weight.mean()
        view = horizon_weight.view(1, horizon, 1)
        output_loss = ((prediction - target_future).square() * view).mean()
        latent_loss = ((predicted_latent - target_latent).square() * view).mean()
        decoding_loss = torch.mean((decoded_anchor - current_y).square())
        anchor_latent = output["latent_trajectory"][:, 0, :]
        latent_std = torch.std(anchor_latent, dim=0, correction=0)
        variance_loss = torch.relu(
            anchor_latent.new_tensor(self.config.minimum_latent_std) - latent_std
        ).square().mean()
        mean_rule_usage = rule_weights.mean(dim=(0, 1))
        uniform_usage = torch.full_like(
            mean_rule_usage,
            1.0 / mean_rule_usage.numel(),
        )
        rule_balance_loss = torch.mean((mean_rule_usage - uniform_usage).square())
        jacobian_product_loss = self._jacobian_product_penalty(jacobians)
        components = {
            "latent_prediction": latent_loss,
            "output_prediction": output_loss,
            "decoding_consistency": decoding_loss,
            "latent_variance": variance_loss,
            "rule_balance": rule_balance_loss,
            "jacobian_product": jacobian_product_loss,
        }
        total = (
            self.config.latent_weight * latent_loss
            + self.config.output_weight * output_loss
            + self.config.decoding_weight * decoding_loss
            + self.config.variance_weight * variance_loss
            + self.config.rule_balance_weight * rule_balance_loss
            + self.config.jacobian_product_weight * jacobian_product_loss
        )
        return {"loss": total, "losses": components}

    def _jacobian_product_penalty(self, jacobians: torch.Tensor) -> torch.Tensor:
        """惩罚每个自由视野前缀的完整 Jacobian 累积范数超额。

        参数：
            jacobians: ``[B,N,m_z,m_z]`` 的完整逐步状态 Jacobian。
        返回：
            所有样本和 horizon 前缀的平方超额均值。
        异常：
            方阵/形状约束由调用方提前检查。
        副作用：
            无。
        """

        batch_size, horizon, latent_dim, _ = jacobians.shape
        product = torch.eye(
            latent_dim,
            dtype=jacobians.dtype,
            device=jacobians.device,
        ).expand(batch_size, -1, -1)
        penalties: list[torch.Tensor] = []
        limit = jacobians.new_tensor(self.config.maximum_jacobian_product_norm)
        for step in range(horizon):
            product = torch.matmul(jacobians[:, step, :, :], product)
            norm = torch.linalg.matrix_norm(product, ord="fro", dim=(-2, -1))
            penalties.append(torch.relu(norm - limit).square())
        return torch.stack(penalties, dim=1).mean()


__all__ = [
    "ProtectedDiagnosticResult",
    "ProtectedDiagnosticsConfig",
    "ProtectedLossConfig",
    "ProtectedModelDiagnostics",
    "ProtectedMultiHorizonLoss",
]
