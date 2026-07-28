"""P4 严格过去 Attention--Koopman--T--S 正常模型的公开行为测试。

文件用途：
    验证 P4 正常建模阶段的 causal attention、可复现 channel mask、T--S 局部 Koopman、
    自由多步 rollout、完整 Jacobian、训练损失和 checkpoint 重放契约。
主要职责：
    只通过公开配置、层、模型工厂和 Trainer 接口观察行为；不测试私有参数名，也不提前
    覆盖 P5 告警、锚点状态机、P6 算子认证或 P8 动态阈值。
关键输入与输出：
    输入是小型确定性 CPU tensor，明确区分严格过去历史、当前解码目标和未来控制/输出；
    输出是潜变量、预测、context、规则权重、算子、Jacobian 与分项损失。
依赖与副作用：
    依赖 PyTorch 和 pytest；测试只在 ``tmp_path`` 写 checkpoint，不读取真实数据或故障
    范围，不产生可作为论文结果的数值。
重要约束：
    当前 ``y_k`` 与真实未来输出只能作为训练目标，不能进入锚点编码或第二步后的自由
    rollout。完整 Jacobian 必须包含 membership variation，并分别通过解析式、autograd
    和中心有限差分验证。
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset

from joff.core.errors import ConfigError
from joff.core.factory import build_model
from joff.data.in_memory import InMemoryDataModule
from joff.layers import (
    CausalAttentionConfig,
    CausalAttentionEncoder,
    ChannelMaskConfig,
    ChannelMaskSampler,
    FuzzyKoopmanConfig,
    FuzzyKoopmanTransition,
)
from joff.models import ProtectedKoopmanTSConfig
from joff.training import (
    CheckpointManager,
    ProtectedDiagnosticsConfig,
    ProtectedLossConfig,
    ProtectedModelDiagnostics,
    Trainer,
)


def test_causal_attention_encoder_masks_future_tokens_and_replays_channel_masks() -> None:
    """因果权重不得看未来 token，同种子 mask 必须逐样本完全一致且保持可微。"""

    mask_config = ChannelMaskConfig(
        all_pass_probability=0.0,
        single_channel_probability=1.0,
        independent_drop_probability=0.5,
        seed=17,
    )
    first_sampler = ChannelMaskSampler(measurement_dim=3, config=mask_config)
    second_sampler = ChannelMaskSampler(measurement_dim=3, config=mask_config)
    first_mask = first_sampler.sample(
        batch_size=4,
        history_length=5,
        device=torch.device("cpu"),
    )
    second_mask = second_sampler.sample(
        batch_size=4,
        history_length=5,
        device=torch.device("cpu"),
    )

    assert torch.equal(first_mask, second_mask)
    assert first_mask.shape == (4, 5, 3)
    assert torch.all(first_mask[:, 0, :] == first_mask[:, -1, :])
    assert torch.all((~first_mask[:, 0, :]).sum(dim=-1) == 1)

    encoder = CausalAttentionEncoder(
        control_dim=2,
        measurement_dim=3,
        exogenous_dim=1,
        latent_dim=4,
        context_dim=5,
        config=CausalAttentionConfig(
            embed_dim=8,
            num_heads=2,
            dropout=0.0,
        ),
    )
    past_u = torch.randn(4, 5, 2)
    past_y = torch.randn(4, 5, 3, requires_grad=True)
    past_xi = torch.randn(4, 5, 1)
    current_xi = torch.randn(4, 1)
    output = encoder(
        past_u,
        past_y,
        past_xi,
        current_xi=current_xi,
        measurement_keep_mask=first_mask,
    )

    assert output["latent"].shape == (4, 4)
    assert output["context"].shape == (4, 5)
    assert output["attention_weights"].shape == (4, 2, 5, 5)
    blocked_future = torch.triu(
        output["attention_weights"],
        diagonal=1,
    )
    assert torch.count_nonzero(blocked_future) == 0
    (output["latent"].sum() + output["context"].sum()).backward()
    assert past_y.grad is not None
    assert torch.isfinite(past_y.grad).all()


def test_causal_attention_encoder_preserves_history_order_and_current_exogenous_state() -> None:
    """有序历史和当前工况必须进入编码，但公开接口仍不得接收当前测量。"""

    torch.manual_seed(19)
    encoder = CausalAttentionEncoder(
        control_dim=1,
        measurement_dim=1,
        exogenous_dim=1,
        latent_dim=3,
        context_dim=2,
        config=CausalAttentionConfig(
            embed_dim=8,
            num_heads=2,
            dropout=0.0,
        ),
    )
    encoder.eval()
    past_u = torch.randn(2, 4, 1)
    past_y = torch.randn(2, 4, 1)
    past_xi = torch.randn(2, 4, 1)
    current_xi = torch.randn(2, 1)
    original = encoder(
        past_u,
        past_y,
        past_xi,
        current_xi=current_xi,
    )
    order = torch.tensor([2, 0, 1, 3])
    reordered = encoder(
        past_u[:, order, :],
        past_y[:, order, :],
        past_xi[:, order, :],
        current_xi=current_xi,
    )
    changed_condition = encoder(
        past_u,
        past_y,
        past_xi,
        current_xi=current_xi + 5.0,
    )

    assert not torch.allclose(original["latent"], reordered["latent"])
    assert not torch.allclose(original["context"], reordered["context"])
    assert not torch.allclose(original["latent"], changed_condition["latent"])
    assert not torch.allclose(original["context"], changed_condition["context"])


def test_single_rule_fuzzy_koopman_is_exact_local_model_and_weights_sum_to_one() -> None:
    """单规则必须精确退化为该局部 Koopman，所有样本隶属度都非负且和为一。"""

    transition = FuzzyKoopmanTransition(
        latent_dim=3,
        control_dim=2,
        exogenous_dim=1,
        context_dim=4,
        config=FuzzyKoopmanConfig(
            rule_count=1,
            premise_dim=2,
            premise_hidden_dim=5,
            metric_eigenvalue_min=0.1,
            metric_eigenvalue_max=3.0,
            spectral_cap=1.2,
        ),
    )
    output = transition(
        torch.randn(6, 3),
        torch.randn(6, 2),
        torch.randn(6, 1),
        torch.randn(6, 4),
    )

    assert output["rule_weights"].shape == (6, 1)
    assert torch.all(output["rule_weights"] >= 0)
    assert torch.allclose(
        output["rule_weights"].sum(dim=-1),
        torch.ones(6),
    )
    assert torch.allclose(output["rule_weights"], torch.ones(6, 1))
    assert torch.allclose(output["next_latent"], output["local_next_latent"][:, 0, :])
    assert torch.allclose(
        output["combined_A"],
        output["local_A"].unsqueeze(0).expand(6, -1, -1, -1)[:, 0, :, :],
    )


def test_complete_fuzzy_jacobian_matches_autograd_and_central_difference() -> None:
    """完整 Jacobian 必须三方一致，并与忽略 membership variation 的组合 A 不同。"""

    torch.manual_seed(23)
    transition = FuzzyKoopmanTransition(
        latent_dim=2,
        control_dim=1,
        exogenous_dim=1,
        context_dim=2,
        config=FuzzyKoopmanConfig(
            rule_count=2,
            premise_dim=2,
            premise_hidden_dim=4,
            metric_eigenvalue_min=0.2,
            metric_eigenvalue_max=2.5,
            spectral_cap=1.1,
        ),
    ).double()
    latent = torch.tensor([[0.35, -0.4]], dtype=torch.float64)
    control = torch.tensor([[0.2]], dtype=torch.float64)
    exogenous = torch.tensor([[-0.1]], dtype=torch.float64)
    context = torch.tensor([[0.5, -0.3]], dtype=torch.float64)

    output = transition(latent, control, exogenous, context)
    named_inputs = {
        "z": latent,
        "u": control,
        "xi": exogenous,
        "context": context,
    }

    def next_latent(name: str, candidate: torch.Tensor) -> torch.Tensor:
        """替换一个输入并保持另外三项固定，作为独立数值参照。"""

        values = {key: value for key, value in named_inputs.items()}
        values[name] = candidate.unsqueeze(0)
        return transition(
            values["z"],
            values["u"],
            values["xi"],
            values["context"],
        )["next_latent"].squeeze(0)

    epsilon = 1e-6
    for input_name, input_value in named_inputs.items():
        analytic = output[f"jacobian_{input_name}"][0]
        autograd = torch.autograd.functional.jacobian(
            lambda value, name=input_name: next_latent(name, value),
            input_value.squeeze(0),
        )
        columns = []
        for index in range(input_value.shape[-1]):
            perturbation = torch.zeros_like(input_value.squeeze(0))
            perturbation[index] = epsilon
            positive = next_latent(
                input_name,
                input_value.squeeze(0) + perturbation,
            )
            negative = next_latent(
                input_name,
                input_value.squeeze(0) - perturbation,
            )
            columns.append((positive - negative) / (2.0 * epsilon))
        finite_difference = torch.stack(columns, dim=-1)

        assert torch.allclose(analytic, autograd, rtol=1e-8, atol=1e-9)
        assert torch.allclose(analytic, finite_difference, rtol=1e-6, atol=1e-7)
    assert not torch.allclose(
        output["jacobian_z"][0],
        output["combined_A"][0],
        rtol=1e-8,
        atol=1e-10,
    )


def _protected_model_config() -> ProtectedKoopmanTSConfig:
    """构造所有理论敏感字段都显式给出的两规则小模型配置。"""

    return ProtectedKoopmanTSConfig(
        type="protected_koopman_ts",
        control_dim=2,
        measurement_dim=2,
        exogenous_dim=1,
        history_length=3,
        latent_dim=4,
        context_dim=3,
        max_rollout=3,
        horizon_seed=59,
        attention=CausalAttentionConfig(
            embed_dim=8,
            num_heads=2,
            dropout=0.0,
        ),
        channel_mask=ChannelMaskConfig(
            all_pass_probability=0.5,
            single_channel_probability=0.25,
            independent_drop_probability=0.4,
            seed=31,
        ),
        fuzzy=FuzzyKoopmanConfig(
            rule_count=2,
            premise_dim=3,
            premise_hidden_dim=5,
            metric_eigenvalue_min=0.1,
            metric_eigenvalue_max=2.0,
            spectral_cap=1.1,
        ),
        loss=ProtectedLossConfig(
            horizon_weights=(1.0, 1.5, 2.0),
            latent_weight=1.0,
            output_weight=1.0,
            decoding_weight=0.5,
            variance_weight=0.1,
            rule_balance_weight=0.1,
            jacobian_product_weight=0.1,
            minimum_latent_std=0.1,
            maximum_jacobian_product_norm=2.0,
        ),
    )


def _protected_batch(*, batch_size: int = 3) -> dict[str, torch.Tensor]:
    """构造严格区分历史、当前解码目标和未来监督目标的小批。"""

    history = 3
    horizon = 3
    return {
        "past_u": torch.randn(batch_size, history, 2),
        "past_y": torch.randn(batch_size, history, 2),
        "past_xi": torch.randn(batch_size, history, 1),
        "future_u": torch.randn(batch_size, horizon, 2),
        "future_xi": torch.randn(batch_size, horizon, 1),
        "current_y": torch.randn(batch_size, 2),
        "target_future": torch.randn(batch_size, horizon, 2),
        "target_past_u": torch.randn(batch_size, horizon, history, 2),
        "target_past_y": torch.randn(batch_size, horizon, history, 2),
        "target_past_xi": torch.randn(batch_size, horizon, history, 1),
        "target_current_xi": torch.randn(batch_size, horizon, 1),
    }


class _ProtectedDataset(Dataset[dict[str, torch.Tensor]]):
    """把一批具名 P4 tensor 暴露为 Trainer 可批处理的样本 dataset。"""

    def __init__(self, batch: dict[str, torch.Tensor]) -> None:
        self.batch = batch
        sizes = {int(value.shape[0]) for value in batch.values()}
        if len(sizes) != 1:
            raise ValueError("Protected test tensors must share one sample axis.")
        self.sample_count = sizes.pop()

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {name: value[index] for name, value in self.batch.items()}


def test_protected_model_free_rollout_ignores_current_and_future_measurement_targets() -> None:
    """当前/未来真实测量只能监督损失，不能改变锚点或自由 rollout 预测。"""

    torch.manual_seed(37)
    model = build_model(_protected_model_config())
    model.eval()
    batch = _protected_batch()
    changed_targets = {
        name: value.clone()
        for name, value in batch.items()
    }
    changed_targets["current_y"].add_(1000.0)
    changed_targets["target_future"].mul_(-500.0)
    changed_targets["target_past_y"].add_(250.0)
    changed_targets["target_current_xi"].sub_(125.0)

    first = model(batch)
    second = model(changed_targets)

    assert first["prediction"].shape == (3, 3, 2)
    assert first["latent_trajectory"].shape == (3, 4, 4)
    assert first["context_trajectory"].shape == (3, 3, 3)
    assert first["rule_weights"].shape == (3, 3, 2)
    assert first["combined_A"].shape == (3, 3, 4, 4)
    assert first["jacobian_z"].shape == (3, 3, 4, 4)
    assert first["target_latent"].shape == (3, 3, 4)
    assert torch.allclose(first["prediction"], second["prediction"])
    assert torch.allclose(first["latent_trajectory"], second["latent_trajectory"])
    assert not torch.allclose(first["target_latent"], second["target_latent"])

    changed_condition = {
        name: value.clone()
        for name, value in batch.items()
    }
    changed_condition["future_xi"][:, 0, :].add_(7.0)
    third = model(changed_condition)
    assert not torch.allclose(
        first["latent_trajectory"][:, 0, :],
        third["latent_trajectory"][:, 0, :],
    )


def test_protected_model_rejects_unknown_nested_config_and_incomplete_target_histories() -> None:
    """严格配置和目标历史成组约束必须在进入损失前给出明确错误。"""

    raw_config = _protected_model_config().model_dump(mode="python")
    raw_config["attention"]["future_leak"] = True
    with pytest.raises(ConfigError, match="future_leak"):
        build_model(raw_config)

    model = build_model(_protected_model_config())
    incomplete = _protected_batch()
    incomplete.pop("target_past_xi")
    with pytest.raises(ValueError, match="all-or-none"):
        model(incomplete)


def test_protected_rollout_rejects_horizon_beyond_frozen_model_limit() -> None:
    """在线和训练共用的 rollout 都不得越过配置冻结的最大视野。"""

    model = build_model(_protected_model_config())
    batch = _protected_batch()
    batch["future_u"] = torch.randn(3, 4, 2)
    batch["future_xi"] = torch.randn(3, 4, 1)

    with pytest.raises(ValueError, match="max_rollout"):
        model(batch)


def test_training_horizon_sampling_is_reproducible_and_covers_frozen_support() -> None:
    """同一显式种子必须重放视野序列，且训练抽样覆盖 ``1..N_max``。"""

    first = build_model(_protected_model_config())
    second = build_model(_protected_model_config())
    first.train()
    second.train()
    batch = _protected_batch(batch_size=2)

    first_horizons = [first(batch)["prediction"].shape[1] for _ in range(12)]
    second_horizons = [second(batch)["prediction"].shape[1] for _ in range(12)]

    assert first_horizons == second_horizons
    assert set(first_horizons) == {1, 2, 3}


def test_protected_multihorizon_loss_is_finite_and_trains_every_model_map() -> None:
    """多视野损失必须分项可追踪，并把有限梯度送到四类核心可学习映射。"""

    torch.manual_seed(41)
    model = build_model(_protected_model_config())
    model.train()
    batch = _protected_batch(batch_size=5)
    output = model(batch)
    loss_info = model.compute_loss(batch, output)

    assert loss_info["loss"].ndim == 0
    assert torch.isfinite(loss_info["loss"])
    assert set(loss_info["losses"]) == {
        "latent_prediction",
        "output_prediction",
        "decoding_consistency",
        "latent_variance",
        "rule_balance",
        "jacobian_product",
    }
    assert all(
        component.ndim == 0 and torch.isfinite(component)
        for component in loss_info["losses"].values()
    )

    loss_info["loss"].backward()
    named_parameters = dict(model.named_parameters())
    required_gradient_paths = (
        "encoder.token_projection.weight",
        "transition.premise_network.input_layer.weight",
        "transition.raw_A",
        "transition.local_B",
        "output_decoder.weight",
    )
    for name in required_gradient_paths:
        gradient = named_parameters[name].grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


def test_protected_diagnostics_fail_closed_on_all_three_p4_stop_conditions() -> None:
    """误差爆炸、规则单支化或 latent collapse 任一出现都不得放行 P5 参考。"""

    diagnostics = ProtectedModelDiagnostics(
        ProtectedDiagnosticsConfig(
            maximum_rmse_ratio=2.0,
            rmse_floor=0.01,
            minimum_rule_usage=0.1,
            minimum_latent_std=0.1,
        )
    )
    target = torch.zeros(4, 3, 2)
    healthy_prediction = torch.tensor(
        [
            [[0.10, 0.10], [0.12, 0.12], [0.15, 0.15]],
        ]
    ).expand(4, -1, -1)
    healthy_latent = torch.tensor(
        [
            [-1.0, -0.5],
            [-0.3, 0.2],
            [0.4, 0.7],
            [1.1, 1.3],
        ]
    )
    healthy_output = {
        "prediction": healthy_prediction,
        "latent_trajectory": torch.cat(
            (
                healthy_latent.unsqueeze(1),
                torch.zeros(4, 3, 2),
            ),
            dim=1,
        ),
        "rule_weights": torch.full((4, 3, 2), 0.5),
    }
    healthy = diagnostics.evaluate(
        {"target_future": target},
        healthy_output,
    )

    assert healthy.rollout_stable
    assert healthy.rules_active
    assert healthy.latent_not_collapsed
    assert healthy.ready_for_protected_reference

    failed_output = {
        "prediction": healthy_prediction.clone(),
        "latent_trajectory": torch.zeros(4, 4, 2),
        "rule_weights": torch.tensor([1.0, 0.0]).expand(4, 3, -1),
    }
    failed_output["prediction"][:, -1, :] = 20.0
    failed = diagnostics.evaluate(
        {"target_future": target},
        failed_output,
    )

    assert not failed.rollout_stable
    assert not failed.rules_active
    assert not failed.latent_not_collapsed
    assert not failed.ready_for_protected_reference


def test_protected_model_trains_on_cpu_and_checkpoint_replays_outputs(tmp_path) -> None:
    """Trainer 保存的严格配置与完整 state_dict 必须精确重放自由展开。"""

    torch.manual_seed(43)
    config = _protected_model_config()
    model = build_model(config)
    data = InMemoryDataModule(
        _ProtectedDataset(_protected_batch(batch_size=8)),
        batch_size=4,
        test_dataset=_ProtectedDataset(_protected_batch(batch_size=4)),
        shuffle=False,
    )
    checkpoint_dir = tmp_path / "protected_checkpoints"
    result = Trainer(
        max_epochs=1,
        device="cpu",
        seed=47,
        checkpoint_dir=checkpoint_dir,
    ).fit(model, data)

    assert result.history[0]["train/loss"] >= 0
    assert result.history[0]["test/loss"] >= 0
    assert "train/jacobian_product_loss" in result.history[0]
    manager = CheckpointManager(checkpoint_dir)
    checkpoint = manager.load_last()
    restored = build_model(checkpoint["config"])
    manager.load_last(model=restored)

    replay_batch = _protected_batch(batch_size=2)
    model.eval()
    restored.eval()
    original_output = model(replay_batch)
    restored_output = restored(replay_batch)
    for name in (
        "prediction",
        "latent_trajectory",
        "context_trajectory",
        "rule_weights",
        "combined_A",
        "jacobian_z",
        "target_latent",
    ):
        assert torch.equal(original_output[name], restored_output[name])
    assert checkpoint["config"] == config.model_dump(mode="json")
    assert (
        checkpoint["model_state_dict"]["channel_mask_sampler.draw_count"].item()
        == model.channel_mask_sampler.draw_count.item()
    )


def test_channel_mask_sampler_checkpoint_state_continues_exact_draw_sequence() -> None:
    """恢复持久化 draw_count 后，下一批 mask 必须与未中断序列完全相同。"""

    config = ChannelMaskConfig(
        all_pass_probability=0.2,
        single_channel_probability=0.3,
        independent_drop_probability=0.4,
        seed=53,
    )
    original = ChannelMaskSampler(measurement_dim=4, config=config)
    original.sample(
        batch_size=5,
        history_length=3,
        device=torch.device("cpu"),
    )
    checkpoint_state = original.state_dict()
    restored = ChannelMaskSampler(measurement_dim=4, config=config)
    restored.load_state_dict(checkpoint_state)

    expected_next = original.sample(
        batch_size=5,
        history_length=3,
        device=torch.device("cpu"),
    )
    restored_next = restored.sample(
        batch_size=5,
        history_length=3,
        device=torch.device("cpu"),
    )
    assert torch.equal(expected_next, restored_next)
