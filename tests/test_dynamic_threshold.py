"""P8 输入调度确定性半径与有限 episode 动态阈值的公开行为测试。

文件用途：
    通过可手算的小矩阵和有限 episode 验证输入包络、reference-age 包络、P7/P6 同坐标
    传播、family-wise conformal 校准及最终阈值分账。
主要职责：
    只从 ``joff.evaluation`` 公开入口测试 P8 行为；不测试 P9 归因/隔离、真实认证后端或
    CSTR 故障性能。
关键输入与输出：
    输入为仅正常 estimate/detection-calibration 合成记录、冻结分支和小型算子包；输出
    为包络值、``gamma_anc``、``gamma_det``、episode maximum、``q_det`` 和报警判决。
依赖与副作用：
    依赖 NumPy、pytest 与 Joff 公开评估接口；不读写文件、不访问网络、不修改随机状态。
重要约束：
    包络和尺度只能读 estimate；``q_det`` 只能读 detection calibration；归因校准和故障
    数据不得进入任何 P8 拟合。所有数值均为代码验证夹具，不是论文实验结果。
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from typing import cast

import numpy as np
import pytest
import torch

from joff.evaluation import (
    BranchBank,
    BranchKind,
    BranchOperator,
    CalibrationStatus,
    ContextAgeEnvelope,
    DetectionScore,
    DeterministicRadius,
    DeterministicRadiusGenerator,
    DynamicThresholdGenerator,
    EpisodeMaxCalibrator,
    InputDependentEnvelope,
    InputDescriptor,
    JacobianSemantics,
    MonitorStage,
    NominalJVPAssembler,
    OperatorAssemblyBudget,
    OperatorPath,
    OperatorStatus,
    ScoreCoordinate,
    ThresholdStatus,
)


def _descriptor(
    *,
    region: str = "nominal",
    u: float = 0.0,
    delta_u: float = 0.0,
    delta_xi: float = 0.0,
) -> InputDescriptor:
    """构造一维向量描述符，使三个范数可直接手算。"""

    return InputDescriptor(
        region=region,
        u=(u,),
        delta_u=(delta_u,),
        delta_xi=(delta_xi,),
    )


@pytest.mark.parametrize("value", [True, cast(float, "1.0")])
def test_input_descriptor_rejects_implicit_numeric_coercion(value: float) -> None:
    """运行时描述符不得把 bool 或数字字符串悄悄转换为浮点输入。"""

    with pytest.raises(TypeError, match="numeric"):
        InputDescriptor(
            region="nominal",
            u=(value,),
            delta_u=(0.0,),
            delta_xi=(0.0,),
        )


def test_input_envelope_uses_all_scheduled_features_and_estimate_only() -> None:
    """输入、输入变化和外生变化都必须改变 estimate-only 非负分位包络。"""

    descriptors = (
        _descriptor(),
        _descriptor(u=1.0),
        _descriptor(u=2.0),
        _descriptor(delta_u=1.0),
        _descriptor(delta_u=2.0),
        _descriptor(delta_xi=1.0),
        _descriptor(delta_xi=2.0),
        _descriptor(u=1.0, delta_u=1.0, delta_xi=1.0),
    )
    magnitudes = np.asarray(
        [
            1.0
            + 2.0 * abs(item.u[0])
            + 3.0 * abs(item.delta_u[0])
            + 4.0 * abs(item.delta_xi[0])
            for item in descriptors
        ],
        dtype=np.float64,
    )

    envelope = InputDependentEnvelope.fit(
        descriptors,
        magnitudes,
        stage=MonitorStage.ESTIMATE,
        quantile=0.5,
        minimum_region_samples=4,
        source_hash="a" * 64,
    )

    baseline = envelope.evaluate(_descriptor())
    changed_u = envelope.evaluate(_descriptor(u=1.0))
    changed_delta_u = envelope.evaluate(_descriptor(delta_u=1.0))
    changed_delta_xi = envelope.evaluate(_descriptor(delta_xi=1.0))
    assert baseline.supported
    assert baseline.value == pytest.approx(1.0)
    assert changed_u.value == pytest.approx(3.0)
    assert changed_delta_u.value == pytest.approx(4.0)
    assert changed_delta_xi.value == pytest.approx(5.0)

    with pytest.raises(ValueError, match="estimate"):
        InputDependentEnvelope.fit(
            descriptors,
            magnitudes,
            stage=MonitorStage.DETECTION_CALIBRATION,
            quantile=0.5,
            minimum_region_samples=4,
            source_hash="b" * 64,
        )


def test_input_envelope_fails_closed_outside_estimate_descriptor_support() -> None:
    """未知 region 或超出 estimate 范围时不得静默线性外推。"""

    descriptors = tuple(_descriptor(u=float(index)) for index in range(4))
    envelope = InputDependentEnvelope.fit(
        descriptors,
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        stage=MonitorStage.ESTIMATE,
        quantile=0.5,
        minimum_region_samples=4,
        source_hash="c" * 64,
    )

    unknown_region = envelope.evaluate(_descriptor(region="unseen"))
    outside_range = envelope.evaluate(_descriptor(u=5.0))

    assert not unknown_region.supported
    assert unknown_region.value == float("inf")
    assert "region" in (unknown_region.reason or "").lower()
    assert not outside_range.supported
    assert outside_range.value == float("inf")
    assert "outside estimate support" in (outside_range.reason or "")


def test_context_age_envelope_is_monotone_and_fails_closed_beyond_frozen_age() -> None:
    """reference-age 包络不能随年龄下降，未覆盖年龄必须禁用阈值决定。"""

    envelope = ContextAgeEnvelope.fit(
        reference_ages=(0, 0, 1, 1, 2, 2),
        drift_magnitudes=(1.0, 2.0, 3.0, 3.0, 1.0, 1.0),
        stage=MonitorStage.ESTIMATE,
        quantile=0.5,
        minimum_samples_per_age=2,
        source_hash="d" * 64,
    )

    assert envelope.evaluate(0).value == pytest.approx(2.0)
    assert envelope.evaluate(1).value == pytest.approx(3.0)
    assert envelope.evaluate(2).value == pytest.approx(3.0)
    unsupported = envelope.evaluate(3)
    assert not unsupported.supported
    assert unsupported.value == float("inf")
    assert "maximum" in (unsupported.reason or "").lower()

    with pytest.raises(ValueError, match="estimate"):
        ContextAgeEnvelope.fit(
            reference_ages=(0, 0),
            drift_magnitudes=(1.0, 2.0),
            stage=MonitorStage.ATTRIBUTION_CALIBRATION,
            quantile=0.5,
            minimum_samples_per_age=2,
            source_hash="e" * 64,
        )


def test_deterministic_radius_uses_the_same_branch_operator_for_each_block() -> None:
    """``gamma_det`` 必须由同一个 ``L_b`` 传播每个 ``G_nu E_j`` block column。"""

    input_descriptors = tuple(_descriptor(u=float(index)) for index in range(4))
    input_envelope = InputDependentEnvelope.fit(
        input_descriptors,
        np.ones(4),
        stage=MonitorStage.ESTIMATE,
        quantile=0.5,
        minimum_region_samples=4,
        source_hash="f" * 64,
    )
    age_envelope = ContextAgeEnvelope.fit(
        reference_ages=(0, 0, 1, 1),
        drift_magnitudes=(0.5, 0.5, 1.0, 1.0),
        stage=MonitorStage.ESTIMATE,
        quantile=0.5,
        minimum_samples_per_age=2,
        source_hash="1" * 64,
    )
    branch = BranchOperator(
        name="scaled",
        kind=BranchKind.MATCHED,
        input_dim=2,
        matrix=((2.0, 0.0), (0.0, 3.0)),
        anchor_radius=4.0,
    )
    path = OperatorPath(
        monitor_identity="monitor-v1",
        episode_id="calibration-episode-1",
        stage=MonitorStage.DETECTION_CALIBRATION,
        start_raw_index=0,
        raw_indices=(1, 2),
    )
    bundle = NominalJVPAssembler(
        resource_budget=OperatorAssemblyBudget(
            max_workspace_elements=100,
            max_persisted_elements=100,
        )
    ).assemble(
        transition_jacobians=torch.zeros((2, 1, 1), dtype=torch.float64),
        semantics=JacobianSemantics.NOMINAL_POINTWISE,
        path=path,
    )

    result = DeterministicRadiusGenerator(
        input_envelope=input_envelope,
        context_age_envelope=age_envelope,
    ).compute(
        branch=branch,
        operator_bundle=bundle,
        descriptors=(_descriptor(u=0.0), _descriptor(u=1.0)),
        reference_ages=(0, 1),
    )
    unsupported = DeterministicRadiusGenerator(
        input_envelope=input_envelope,
        context_age_envelope=age_envelope,
    ).compute(
        branch=branch,
        operator_bundle=bundle,
        descriptors=(_descriptor(u=0.0), _descriptor(u=4.0)),
        reference_ages=(0, 1),
    )
    overflow_bundle = replace(
        bundle,
        g_nu=((2.0, 0.0), (0.0, 1.0)),
    )
    overflowing_branch = BranchOperator(
        name="overflowing",
        kind=BranchKind.MATCHED,
        input_dim=2,
        matrix=((1e308, 1e308),),
        anchor_radius=0.0,
    )
    numerical_fallback = DeterministicRadiusGenerator(
        input_envelope=input_envelope,
        context_age_envelope=age_envelope,
    ).compute(
        branch=overflowing_branch,
        operator_bundle=overflow_bundle,
        descriptors=(_descriptor(u=0.0), _descriptor(u=1.0)),
        reference_ages=(0, 1),
    )

    assert result.supported
    assert result.operator_status is OperatorStatus.NOMINAL
    assert result.gamma_anchor == pytest.approx(4.0)
    assert result.source_envelopes == pytest.approx((1.5, 2.0))
    assert result.block_norms == pytest.approx((2.0, 3.0))
    assert result.gamma_deterministic == pytest.approx(9.0)
    assert not unsupported.supported
    assert unsupported.gamma_deterministic == float("inf")
    assert "outside estimate support" in (unsupported.reason or "")
    assert not numerical_fallback.supported
    assert numerical_fallback.gamma_deterministic == float("inf")
    assert "numerical" in (numerical_fallback.reason or "").lower()
    assert DeterministicRadius.from_dict(
        json.loads(json.dumps(result.to_dict(), allow_nan=False))
    ).to_dict() == result.to_dict()
    assert DeterministicRadius.from_dict(
        json.loads(json.dumps(unsupported.to_dict(), allow_nan=False))
    ).to_dict() == unsupported.to_dict()


def test_episode_max_calibration_uses_finite_rank_and_returns_infinity_below_resolution() -> None:
    """校准必须先取 family-wise episode maximum，再应用有限样本秩规则。"""

    coordinates = tuple(
        ScoreCoordinate(time_index=time_index, mode=mode, branch_name=branch)
        for time_index in (0, 1)
        for mode in ("steady", "hybrid")
        for branch in ("guard", "omnibus")
    )
    episodes: dict[str, tuple[DetectionScore, ...]] = {}
    for episode_index, episode_maximum in enumerate((1.0, 2.0, 3.0, 4.0), start=1):
        episode_id = f"det-cal-{episode_index}"
        scores = []
        for coordinate_index, coordinate in enumerate(coordinates):
            normalized_value = (
                episode_maximum
                if coordinate_index == len(coordinates) - 1
                else 0.1 * episode_index
            )
            scores.append(
                DetectionScore.from_components(
                    score_map_hash="2" * 64,
                    episode_id=episode_id,
                    coordinate=coordinate,
                    statistic=normalized_value,
                    gamma_anchor=0.0,
                    gamma_deterministic=0.0,
                    scale=1.0,
                )
            )
        episodes[episode_id] = tuple(scores)

    finite = EpisodeMaxCalibrator.fit(
        episodes,
        expected_coordinates=coordinates,
        stage=MonitorStage.DETECTION_CALIBRATION,
        error_rate=0.25,
        score_map_hash="2" * 64,
        reset_state_hash="3" * 64,
        episode_definition_hash="a" * 64,
        exchangeability_assumption_hash="b" * 64,
        source_hash="4" * 64,
    )
    below_resolution = EpisodeMaxCalibrator.fit(
        episodes,
        expected_coordinates=coordinates,
        stage=MonitorStage.DETECTION_CALIBRATION,
        error_rate=0.1,
        score_map_hash="2" * 64,
        reset_state_hash="3" * 64,
        episode_definition_hash="a" * 64,
        exchangeability_assumption_hash="b" * 64,
        source_hash="4" * 64,
    )

    assert finite.status is CalibrationStatus.READY
    assert finite.episode_maxima == pytest.approx((1.0, 2.0, 3.0, 4.0))
    assert finite.rank == 4
    assert finite.quantile == pytest.approx(4.0)
    assert finite.risk_resolution == pytest.approx(0.2)
    assert below_resolution.status is CalibrationStatus.INSUFFICIENT_RESOLUTION
    assert below_resolution.rank == 5
    assert below_resolution.quantile == float("inf")
    replayed_infinite = EpisodeMaxCalibrator.from_dict(
        json.loads(json.dumps(below_resolution.to_dict(), allow_nan=False))
    )
    assert replayed_infinite.to_dict() == below_resolution.to_dict()
    first_score = episodes["det-cal-1"][0]
    assert DetectionScore.from_dict(
        json.loads(json.dumps(first_score.to_dict(), allow_nan=False))
    ).to_dict() == first_score.to_dict()

    with pytest.raises(ValueError, match="detection calibration"):
        EpisodeMaxCalibrator.fit(
            episodes,
            expected_coordinates=coordinates,
            stage=MonitorStage.ATTRIBUTION_CALIBRATION,
            error_rate=0.25,
            score_map_hash="2" * 64,
            reset_state_hash="3" * 64,
            episode_definition_hash="a" * 64,
            exchangeability_assumption_hash="b" * 64,
            source_hash="4" * 64,
        )


def test_dynamic_threshold_separates_components_and_uses_strict_exceedance() -> None:
    """最终阈值必须分账，等于阈值不报警，缺失校准证据时返回正无穷。"""

    descriptors = tuple(_descriptor(u=float(index)) for index in range(4))
    input_envelope = InputDependentEnvelope.fit(
        descriptors,
        np.ones(4),
        stage=MonitorStage.ESTIMATE,
        quantile=0.5,
        minimum_region_samples=4,
        source_hash="5" * 64,
    )
    age_envelope = ContextAgeEnvelope.fit(
        reference_ages=(0, 0),
        drift_magnitudes=(0.5, 0.5),
        stage=MonitorStage.ESTIMATE,
        quantile=0.5,
        minimum_samples_per_age=2,
        source_hash="6" * 64,
    )
    guard = BranchOperator(
        name="guard",
        kind=BranchKind.GUARD,
        input_dim=1,
        matrix=((1.0,),),
        anchor_radius=1.0,
    )
    generator = DynamicThresholdGenerator.freeze(
        branch_bank=BranchBank((guard,)),
        candidate_hash="7" * 64,
        input_envelope=input_envelope,
        context_age_envelope=age_envelope,
        branch_scales={"guard": 2.0},
        threshold_floor=12.0,
        normalization_source_hash="0" * 64,
        mode_names=("steady",),
        reset_state_hash="8" * 64,
        stage=MonitorStage.ESTIMATE,
    )
    coordinate = ScoreCoordinate(
        time_index=0,
        mode="steady",
        branch_name="guard",
    )
    calibration_episodes = {
        f"episode-{index}": (
            DetectionScore.from_components(
                score_map_hash=generator.content_hash,
                episode_id=f"episode-{index}",
                coordinate=coordinate,
                statistic=float(index),
                gamma_anchor=0.0,
                gamma_deterministic=0.0,
                scale=1.0,
            ),
        )
        for index in range(1, 5)
    }
    calibration = EpisodeMaxCalibrator.fit(
        calibration_episodes,
        expected_coordinates=(coordinate,),
        stage=MonitorStage.DETECTION_CALIBRATION,
        error_rate=0.25,
        score_map_hash=generator.content_hash,
        reset_state_hash=generator.reset_state_hash,
        episode_definition_hash="a" * 64,
        exchangeability_assumption_hash="b" * 64,
        source_hash="9" * 64,
    )
    radius = DeterministicRadius(
        branch_name="guard",
        episode_id="target-episode",
        start_raw_index=0,
        raw_indices=(1,),
        operator_status=OperatorStatus.NOMINAL,
        gamma_anchor=1.0,
        gamma_deterministic=2.0,
        source_envelopes=(2.0,),
        block_norms=(1.0,),
        supported=True,
    )

    tie = generator.evaluate(
        statistic=12.0,
        radius=radius,
        time_index=0,
        mode="steady",
        calibration=calibration,
        episode_definition_hash="a" * 64,
    )
    strict_exceedance = generator.evaluate(
        statistic=12.0001,
        radius=radius,
        time_index=0,
        mode="steady",
        calibration=calibration,
        episode_definition_hash="a" * 64,
    )
    missing_calibration = generator.evaluate(
        statistic=100.0,
        radius=radius,
        time_index=0,
        mode="steady",
        calibration=None,
        episode_definition_hash="a" * 64,
    )
    wrong_score_map_hash = "d" * 64
    wrong_calibration = EpisodeMaxCalibrator.fit(
        {
            f"other-{index}": (
                DetectionScore.from_components(
                    score_map_hash=wrong_score_map_hash,
                    episode_id=f"other-{index}",
                    coordinate=coordinate,
                    statistic=float(index),
                    gamma_anchor=0.0,
                    gamma_deterministic=0.0,
                    scale=1.0,
                ),
            )
            for index in range(1, 5)
        },
        expected_coordinates=(coordinate,),
        stage=MonitorStage.DETECTION_CALIBRATION,
        error_rate=0.25,
        score_map_hash=wrong_score_map_hash,
        reset_state_hash=generator.reset_state_hash,
        episode_definition_hash="a" * 64,
        exchangeability_assumption_hash="b" * 64,
        source_hash="e" * 64,
    )
    wrong_identity = generator.evaluate(
        statistic=100.0,
        radius=radius,
        time_index=0,
        mode="steady",
        calibration=wrong_calibration,
        episode_definition_hash="a" * 64,
    )

    assert tie.status is ThresholdStatus.READY
    assert generator.normalization_source_hash == "0" * 64
    assert tie.gamma_anchor == pytest.approx(1.0)
    assert tie.gamma_deterministic == pytest.approx(2.0)
    assert tie.scale == pytest.approx(2.0)
    assert tie.calibration_quantile == pytest.approx(4.0)
    assert tie.calibration_component == pytest.approx(8.0)
    assert tie.threshold == pytest.approx(12.0)
    assert not tie.alarm
    assert strict_exceedance.alarm
    assert missing_calibration.status is ThresholdStatus.DISABLED
    assert missing_calibration.threshold == float("inf")
    assert not missing_calibration.alarm
    assert wrong_identity.status is ThresholdStatus.DISABLED
    assert wrong_identity.threshold == float("inf")

    replayed_generator = DynamicThresholdGenerator.from_dict(
        json.loads(json.dumps(generator.to_dict()))
    )
    replayed_calibration = EpisodeMaxCalibrator.from_dict(
        json.loads(json.dumps(calibration.to_dict()))
    )
    replayed_result = type(tie).from_dict(json.loads(json.dumps(tie.to_dict())))
    replayed_disabled = type(missing_calibration).from_dict(
        json.loads(
            json.dumps(missing_calibration.to_dict(), allow_nan=False)
        )
    )
    assert replayed_generator.to_dict() == generator.to_dict()
    assert replayed_generator.content_hash == generator.content_hash
    assert replayed_calibration.to_dict() == calibration.to_dict()
    assert replayed_result.to_dict() == tie.to_dict()
    assert replayed_disabled.to_dict() == missing_calibration.to_dict()

    tampered_calibration = copy.deepcopy(calibration.to_dict())
    tampered_calibration["quantile"] = 3.5
    tampered_result = copy.deepcopy(tie.to_dict())
    tampered_result["alarm"] = True
    unknown_generator_field = copy.deepcopy(generator.to_dict())
    unknown_generator_field["calibration_quantile"] = 4.0
    for owner, payload in (
        (EpisodeMaxCalibrator, tampered_calibration),
        (type(tie), tampered_result),
        (DynamicThresholdGenerator, unknown_generator_field),
    ):
        with pytest.raises(ValueError):
            owner.from_dict(payload)


def test_incomplete_episode_family_fails_closed_instead_of_using_marginal_scores() -> None:
    """漏掉任一可选择坐标时不得把剩余边际分数当作 family-wise 校准。"""

    first = ScoreCoordinate(time_index=0, mode="steady", branch_name="guard")
    second = ScoreCoordinate(time_index=1, mode="steady", branch_name="guard")
    incomplete = {
        "episode-1": (
            DetectionScore.from_components(
                score_map_hash="a" * 64,
                episode_id="episode-1",
                coordinate=first,
                statistic=1.0,
                gamma_anchor=0.0,
                gamma_deterministic=0.0,
                scale=1.0,
            ),
        )
    }

    calibration = EpisodeMaxCalibrator.fit(
        incomplete,
        expected_coordinates=(first, second),
        stage=MonitorStage.DETECTION_CALIBRATION,
        error_rate=0.5,
        score_map_hash="a" * 64,
        reset_state_hash="b" * 64,
        episode_definition_hash="d" * 64,
        exchangeability_assumption_hash="e" * 64,
        source_hash="c" * 64,
    )

    assert calibration.status is CalibrationStatus.INCOMPLETE_EVIDENCE
    assert calibration.quantile == float("inf")
    assert "complete score family" in (calibration.reason or "")


def test_finite_rank_boundary_is_attainable_without_rounding_to_infinity() -> None:
    """19 个 episode 的 ``alpha=0.05`` 应有限，略低于 0.05 才返回正无穷。"""

    coordinate = ScoreCoordinate(
        time_index=0,
        mode="steady",
        branch_name="guard",
    )
    episodes = {
        f"episode-{index:02d}": (
            DetectionScore.from_components(
                score_map_hash="d" * 64,
                episode_id=f"episode-{index:02d}",
                coordinate=coordinate,
                statistic=float(index),
                gamma_anchor=0.0,
                gamma_deterministic=0.0,
                scale=1.0,
            ),
        )
        for index in range(1, 20)
    }

    attainable = EpisodeMaxCalibrator.fit(
        episodes,
        expected_coordinates=(coordinate,),
        stage=MonitorStage.DETECTION_CALIBRATION,
        error_rate=0.05,
        score_map_hash="d" * 64,
        reset_state_hash="e" * 64,
        episode_definition_hash="a" * 64,
        exchangeability_assumption_hash="b" * 64,
        source_hash="f" * 64,
    )
    unattainable = EpisodeMaxCalibrator.fit(
        episodes,
        expected_coordinates=(coordinate,),
        stage=MonitorStage.DETECTION_CALIBRATION,
        error_rate=0.049,
        score_map_hash="d" * 64,
        reset_state_hash="e" * 64,
        episode_definition_hash="a" * 64,
        exchangeability_assumption_hash="b" * 64,
        source_hash="f" * 64,
    )

    assert attainable.status is CalibrationStatus.READY
    assert attainable.rank == 19
    assert attainable.quantile == pytest.approx(19.0)
    assert unattainable.status is CalibrationStatus.INSUFFICIENT_RESOLUTION
    assert unattainable.rank == 20
    assert unattainable.quantile == float("inf")


def test_score_map_freeze_requires_exact_positive_estimate_normalization() -> None:
    """启用 branch 必须恰有一个 estimate 来源的正尺度，floor 也必须严格为正。"""

    descriptors = tuple(_descriptor(u=float(index)) for index in range(4))
    input_envelope = InputDependentEnvelope.fit(
        descriptors,
        np.ones(4),
        stage=MonitorStage.ESTIMATE,
        quantile=0.5,
        minimum_region_samples=4,
        source_hash="1" * 64,
    )
    age_envelope = ContextAgeEnvelope.fit(
        reference_ages=(0,),
        drift_magnitudes=(0.0,),
        stage=MonitorStage.ESTIMATE,
        quantile=0.5,
        minimum_samples_per_age=1,
        source_hash="2" * 64,
    )
    guard = BranchOperator(
        name="guard",
        kind=BranchKind.GUARD,
        input_dim=1,
        matrix=((1.0,),),
        anchor_radius=0.0,
    )
    branch_bank = BranchBank((guard,))

    def freeze(
        branch_scales: dict[str, float],
        threshold_floor: float,
    ) -> DynamicThresholdGenerator:
        """用共同 estimate 证据冻结待验证的尺度/floor 配置。"""

        return DynamicThresholdGenerator.freeze(
            branch_bank=branch_bank,
            candidate_hash="3" * 64,
            input_envelope=input_envelope,
            context_age_envelope=age_envelope,
            branch_scales=branch_scales,
            threshold_floor=threshold_floor,
            normalization_source_hash="4" * 64,
            mode_names=("steady",),
            reset_state_hash="5" * 64,
            stage=MonitorStage.ESTIMATE,
        )

    with pytest.raises(ValueError):
        freeze({"guard": 0.0}, 1.0)
    with pytest.raises(ValueError, match="exactly match"):
        freeze({"guard": 1.0, "injected": 1.0}, 1.0)
    with pytest.raises(ValueError, match="positive"):
        freeze({"guard": 1.0}, 0.0)
    with pytest.raises(TypeError, match="floor"):
        freeze({"guard": 1.0}, True)
    with pytest.raises(TypeError, match="mode"):
        DynamicThresholdGenerator.freeze(
            branch_bank=branch_bank,
            candidate_hash="3" * 64,
            input_envelope=input_envelope,
            context_age_envelope=age_envelope,
            branch_scales={"guard": 1.0},
            threshold_floor=1.0,
            normalization_source_hash="4" * 64,
            mode_names=("steady", cast(str, 1)),
            reset_state_hash="5" * 64,
            stage=MonitorStage.ESTIMATE,
        )
