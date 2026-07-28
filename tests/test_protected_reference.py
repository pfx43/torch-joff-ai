"""P5 受保护参考、监视状态、trace 与堆叠残差的公开行为测试。

文件用途：
    验证延迟 anchor gate、hysteresis、episode/stage reset、记录命令条件下的直接非污染、
    受保护预测缓冲、trace/hash 和残差窗口边界。
主要职责：
    只通过 ``ProtectedMonitor.step``、不可变状态/记录对象和残差 builder 观察行为；不读取
    故障数据，不测试 P6 算子认证、P7 后滤波或 P8 检测分位。
关键输入与输出：
    输入为小型确定性 CPU 记录序列；输出为 anchor/candidate 状态、data/protected latent、
    预测测量、逐步 trace 和堆叠潜变量残差。
依赖与副作用：
    依赖 PyTorch、Joff P4 模型和 P5 evaluation 接口；不写文件、不访问网络。
重要约束：
    anchor 后真实测量不得进入 protected measurement slot；检测报警不得反馈 anchor/mode；
    无报警延迟只形成声明覆盖事件，绝不把锚点标为 ``clean``。
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from joff.core.factory import build_model
from joff.evaluation import (
    AnchorCoverageStatus,
    AnchorGateConfig,
    MonitorMode,
    MonitorRecord,
    MonitorStage,
    MonitorTrace,
    ProtectedMonitor,
    StackedProtectedResidual,
)


def _protected_model():
    """构造 P5 状态机测试使用的确定性小型 P4 模型。"""

    model = build_model(
        {
            "type": "protected_koopman_ts",
            "control_dim": 1,
            "measurement_dim": 1,
            "exogenous_dim": 1,
            "history_length": 2,
            "latent_dim": 2,
            "context_dim": 2,
            "max_rollout": 3,
            "horizon_seed": 7,
            "attention": {
                "embed_dim": 4,
                "num_heads": 1,
                "dropout": 0.0,
            },
            "channel_mask": {
                "all_pass_probability": 1.0,
                "single_channel_probability": 0.0,
                "independent_drop_probability": 0.0,
                "seed": 11,
            },
            "fuzzy": {
                "rule_count": 2,
                "premise_dim": 2,
                "premise_hidden_dim": 3,
                "metric_eigenvalue_min": 0.1,
                "metric_eigenvalue_max": 2.0,
                "spectral_cap": 1.1,
            },
            "loss": {
                "horizon_weights": [1.0, 1.0, 1.0],
                "latent_weight": 1.0,
                "output_weight": 1.0,
                "decoding_weight": 0.5,
                "variance_weight": 0.1,
                "rule_balance_weight": 0.1,
                "jacobian_product_weight": 0.1,
                "minimum_latent_std": 0.1,
                "maximum_jacobian_product_norm": 2.0,
            },
        }
    )
    model.eval()
    return model


def _record(
    index: int,
    *,
    score: float = 0.1,
    episode: str = "episode-a",
    stage: str = "detection_calibration",
    control: float | None = None,
    measurement: float | None = None,
) -> MonitorRecord:
    """构造一条有稳定原始索引和明确分支输入的在线记录。"""

    return MonitorRecord(
        raw_index=index,
        episode_id=episode,
        stage=MonitorStage.parse(stage),
        control=(float(index) if control is None else control,),
        measurement=(float(index) * 0.2 if measurement is None else measurement,),
        exogenous=(0.5 + float(index) * 0.01,),
        anchor_eligibility_score=score,
    )


def test_anchor_gate_delays_acceptance_resets_boundaries_and_never_claims_clean() -> None:
    """候选需连续延迟确认；episode 边界重置；接受后覆盖状态仍不得写成 clean。"""

    monitor = ProtectedMonitor(
        _protected_model(),
        AnchorGateConfig(
            confirmation_delay=2,
            enter_threshold=0.2,
            exit_threshold=0.4,
            minimum_reanchor_interval=100,
            maximum_reference_age=3,
        ),
    )
    state = monitor.initial_state()
    outputs = []
    for index in range(5):
        result = monitor.step(state, _record(index))
        state = result.state
        outputs.append(result.output)

    assert state.anchor is not None
    assert state.anchor.raw_index == 2
    assert state.anchor_age == 2
    assert state.candidate is None
    assert state.mode is MonitorMode.PROTECTED
    assert state.anchor.coverage_status is AnchorCoverageStatus.UNVERIFIED
    assert "clean" not in state.to_dict()["anchor"]

    reset_result = monitor.step(
        state,
        _record(20, episode="episode-b"),
    )
    assert reset_result.state.episode_id == "episode-b"
    assert reset_result.state.reset_count == state.reset_count + 1
    assert reset_result.state.anchor is None
    assert reset_result.state.mode is MonitorMode.WARMUP


def test_fixed_recorded_commands_block_post_anchor_measurements_from_protected_rollout() -> None:
    """固定记录命令时，锚点后测量只改变 data branch，不能改变 protected branch。"""

    monitor = ProtectedMonitor(
        _protected_model(),
        AnchorGateConfig(
            confirmation_delay=2,
            enter_threshold=0.2,
            exit_threshold=0.4,
            minimum_reanchor_interval=100,
            maximum_reference_age=3,
        ),
    )

    def run(post_anchor_measurement: float):
        state = monitor.initial_state()
        result = None
        for index in range(5):
            measurement = None if index <= 2 else post_anchor_measurement + index
            result = monitor.step(
                state,
                _record(index, measurement=measurement),
            )
            state = result.state
        assert result is not None
        return result.output

    baseline = run(10.0)
    changed = run(10_000.0)

    assert baseline.data_latent != pytest.approx(changed.data_latent)
    assert baseline.protected_rollout is not None
    assert changed.protected_rollout is not None
    assert baseline.protected_rollout.to_dict() == changed.protected_rollout.to_dict()
    assert all(
        abs(value) < 1_000.0
        for row in changed.protected_rollout.protected_measurement_buffer
        for value in row
    )


def test_identical_episode_replay_has_serializable_identical_state_and_trace_hashes() -> None:
    """显式 reset 后重放同一 episode，状态与 trace 的内容哈希必须完全一致。"""

    monitor = ProtectedMonitor(
        _protected_model(),
        AnchorGateConfig(
            confirmation_delay=2,
            enter_threshold=0.2,
            exit_threshold=0.4,
            minimum_reanchor_interval=100,
            maximum_reference_age=3,
        ),
    )

    def run():
        state = monitor.initial_state()
        trace = MonitorTrace()
        for index in range(6):
            record = _record(index)
            result = monitor.step(state, record)
            trace = trace.append(record, result)
            state = result.state
        return state, trace

    first_state, first_trace = run()
    replay_state, replay_trace = run()

    assert first_state.content_hash == replay_state.content_hash
    assert first_trace.content_hash == replay_trace.content_hash
    assert json.loads(json.dumps(first_trace.to_dict())) == first_trace.to_dict()


def test_changed_recorded_command_can_change_protected_total_path() -> None:
    """真实测量若改变后续记录命令，总受保护路径允许变化，不能宣称因果独立。"""

    monitor = ProtectedMonitor(
        _protected_model(),
        AnchorGateConfig(
            confirmation_delay=2,
            enter_threshold=0.2,
            exit_threshold=0.4,
            minimum_reanchor_interval=100,
            maximum_reference_age=3,
        ),
    )

    def run(changed_control: float):
        state = monitor.initial_state()
        result = None
        for index in range(5):
            control = changed_control if index == 3 else None
            result = monitor.step(state, _record(index, control=control))
            state = result.state
        assert result is not None
        assert result.output.protected_rollout is not None
        return result.output.protected_rollout.latent

    assert run(3.0) != pytest.approx(run(300.0))


def test_stacked_residual_accepts_only_consecutive_same_boundary_same_anchor_outputs() -> None:
    """堆叠残差接受合法连续窗口，并拒绝跨 episode、stage、索引或锚点的窗口。"""

    monitor = ProtectedMonitor(
        _protected_model(),
        AnchorGateConfig(
            confirmation_delay=2,
            enter_threshold=0.2,
            exit_threshold=0.4,
            minimum_reanchor_interval=100,
            maximum_reference_age=3,
        ),
    )
    state = monitor.initial_state()
    outputs = []
    for index in range(6):
        result = monitor.step(state, _record(index))
        state = result.state
        if result.output.protected_rollout is not None:
            outputs.append(result.output)

    residual = StackedProtectedResidual.from_outputs(outputs)
    assert residual.raw_indices == (4, 5)
    assert residual.anchor_raw_index == 2
    assert len(residual.vector) == 4

    second_rollout = outputs[1].protected_rollout
    assert second_rollout is not None
    altered = (
        replace(outputs[1], episode_id="episode-b"),
        replace(outputs[1], stage=MonitorStage.ATTRIBUTION_CALIBRATION),
        replace(outputs[1], raw_index=7),
        replace(
            outputs[1],
            protected_rollout=replace(
                second_rollout,
                anchor_raw_index=3,
            ),
        ),
    )
    for invalid_second in altered:
        with pytest.raises(ValueError):
            StackedProtectedResidual.from_outputs((outputs[0], invalid_second))


def test_reanchor_waits_for_interval_and_keeps_old_reference_until_new_acceptance() -> None:
    """重锚候选只在冻结间隔后启动，并在延迟确认期间继续使用旧受保护参考。"""

    monitor = ProtectedMonitor(
        _protected_model(),
        AnchorGateConfig(
            confirmation_delay=1,
            enter_threshold=0.2,
            exit_threshold=0.4,
            minimum_reanchor_interval=2,
            maximum_reference_age=3,
        ),
    )
    state = monitor.initial_state()
    results = []
    for index in range(6):
        result = monitor.step(state, _record(index))
        state = result.state
        results.append(result)

    assert results[3].state.anchor is not None
    assert results[3].state.anchor.raw_index == 2
    assert results[4].state.candidate is not None
    assert results[4].state.candidate.raw_index == 4
    assert results[4].output.protected_rollout is not None
    assert results[4].output.protected_rollout.anchor_raw_index == 2
    assert results[5].state.anchor is not None
    assert results[5].state.anchor.raw_index == 4
    assert results[5].state.anchor_age == 1


def test_candidate_hysteresis_and_stale_mode_fail_closed() -> None:
    """候选可停留在 hysteresis 带内；越过退出阈值会取消；旧锚超龄后显式失效。"""

    model = _protected_model()
    accepting_monitor = ProtectedMonitor(
        model,
        AnchorGateConfig(
            confirmation_delay=2,
            enter_threshold=0.2,
            exit_threshold=0.4,
            minimum_reanchor_interval=100,
            maximum_reference_age=2,
        ),
    )
    state = accepting_monitor.initial_state()
    scores = (0.1, 0.1, 0.1, 0.3, 0.3, 0.3)
    results = []
    for index, score in enumerate(scores):
        result = accepting_monitor.step(state, _record(index, score=score))
        state = result.state
        results.append(result)

    assert results[3].state.candidate is not None
    assert results[4].state.anchor is not None
    assert results[5].state.mode is MonitorMode.STALE
    assert results[5].output.protected_rollout is None

    rejecting_monitor = ProtectedMonitor(
        model,
        AnchorGateConfig(
            confirmation_delay=2,
            enter_threshold=0.2,
            exit_threshold=0.4,
            minimum_reanchor_interval=100,
            maximum_reference_age=2,
        ),
    )
    state = rejecting_monitor.initial_state()
    for index, score in enumerate((0.1, 0.1, 0.1, 0.5)):
        result = rejecting_monitor.step(state, _record(index, score=score))
        state = result.state
    assert state.candidate is None
    assert state.anchor is None
    assert state.mode is MonitorMode.WARMUP


def test_final_detection_quantile_changes_alarms_not_anchor_or_mode_trace() -> None:
    """最终检测阈值只消费残差输出，不得反馈 score-generating state path。"""

    monitor = ProtectedMonitor(
        _protected_model(),
        AnchorGateConfig(
            confirmation_delay=1,
            enter_threshold=0.2,
            exit_threshold=0.4,
            minimum_reanchor_interval=100,
            maximum_reference_age=3,
        ),
    )

    def run(final_quantile: float):
        state = monitor.initial_state()
        path = []
        alarms = []
        for index in range(6):
            result = monitor.step(state, _record(index))
            state = result.state
            anchor_index = None if state.anchor is None else state.anchor.raw_index
            path.append((state.mode, anchor_index, state.content_hash))
            rollout = result.output.protected_rollout
            score = (
                0.0
                if result.output.data_latent is None or rollout is None
                else sum(
                    abs(data - protected)
                    for data, protected in zip(
                        result.output.data_latent,
                        rollout.latent,
                        strict=True,
                    )
                )
            )
            alarms.append(score > final_quantile)
        return path, alarms

    low_path, low_alarms = run(-1.0)
    high_path, high_alarms = run(float("inf"))

    assert low_path == high_path
    assert low_alarms != high_alarms


def test_monitor_stage_is_controlled_and_models_fault_test_outside_five_normal_stages() -> None:
    """阶段拼写错误必须失败；冻结故障测试使用独立受控范围，不能冒充正常第五段。"""

    with pytest.raises(ValueError, match="Legal options"):
        _record(0, stage="detection_calibraton")

    fault_record = _record(0, stage="frozen_fault_test")
    assert fault_record.stage is MonitorStage.FROZEN_FAULT_TEST


def test_trace_rejects_inconsistent_record_state_output_and_monitor_identity() -> None:
    """trace 必须拒绝边界、mode 或冻结 monitor 身份互相矛盾的审计项。"""

    model = _protected_model()
    first_monitor = ProtectedMonitor(
        model,
        AnchorGateConfig(
            confirmation_delay=1,
            enter_threshold=0.2,
            exit_threshold=0.4,
            minimum_reanchor_interval=100,
            maximum_reference_age=3,
        ),
    )
    second_monitor = ProtectedMonitor(
        model,
        AnchorGateConfig(
            confirmation_delay=1,
            enter_threshold=0.1,
            exit_threshold=0.4,
            minimum_reanchor_interval=100,
            maximum_reference_age=3,
        ),
    )
    record = _record(0)
    result = first_monitor.step(first_monitor.initial_state(), record)

    with pytest.raises(ValueError, match="episode"):
        MonitorTrace().append(
            record,
            replace(
                result,
                output=replace(result.output, episode_id="episode-b"),
            ),
        )
    with pytest.raises(ValueError, match="mode"):
        MonitorTrace().append(
            record,
            replace(
                result,
                output=replace(result.output, mode=MonitorMode.PROTECTED),
            ),
        )
    assert first_monitor.initial_state().content_hash != (
        second_monitor.initial_state().content_hash
    )
    with pytest.raises(ValueError, match="identity"):
        second_monitor.step(first_monitor.initial_state(), record)
