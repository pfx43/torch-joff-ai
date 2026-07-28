"""P9 full-normal 归因与集合值结构化隔离的公开行为测试。

文件用途：
    通过可手算的线性 explanation family 和有限正常 episode，验证 P9 的完整观测、
    单调 outer oracle、独立归因校准、集合值输出和失败关闭语义。
主要职责：
    只从 ``joff.evaluation`` 公开入口测试调用者可观察行为；不测试真实非线性认证后端、
    CSTR 故障识别率或论文实验数值；直接构造/替换公开结果对象也必须经过同一门禁。
关键输入与输出：
    输入为冻结 monitor 快照、raw window、mask 全流水线证据、线性 oracle 和仅正常归因
    episode；输出为 full-normal 分位、候选 explanation 集与受控隔离报告。
依赖与副作用：
    依赖 pytest、PyTorch 和 Joff 公开评估接口；不读写文件、不访问网络、不修改随机状态。
重要约束：
    detection 与 attribution episode 不得重用；未决 oracle 必须保留解释；没有 certified
    operator/物理证据时不得输出物理 singleton；所有夹具仅用于代码验证，不是论文结果。
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from joff.evaluation import (
    AttributionCalibrationStatus,
    CertifiedEnclosureProvider,
    DeployedBranchEvidence,
    DeployedObservation,
    DynamicsSide,
    ExplanationFamily,
    FullNormalCalibrator,
    IsolationCandidateSet,
    IsolationOutcome,
    IsolationReport,
    JacobianSemantics,
    LinearExplanationCell,
    MaskRecomputation,
    MonotoneRefinementCache,
    MonitorRecord,
    MonitorStage,
    MonitorState,
    NominalJVPAssembler,
    OperatorAssemblyBudget,
    OperatorAffineImage,
    OperatorBundle,
    OperatorCertificationRequest,
    OperatorEnclosure,
    OperatorNorm,
    OracleCellRefinement,
    OperatorPath,
    OuterExplanationOracle,
)


class _CertifiedFixtureProvider(CertifiedEnclosureProvider):
    """为 P9 认证门禁测试生成共享零误差 enclosure。"""

    def enclose(
        self,
        request: OperatorCertificationRequest,
    ) -> OperatorEnclosure:
        """覆盖请求中的全部现存算子，但不凭空增加 sensor JVP。"""

        return OperatorEnclosure(
            images=tuple(
                OperatorAffineImage(
                    operator_name=name,
                    center=request.nominal(name),
                    generators=(
                        tuple(
                            tuple(0.0 for _ in row)
                            for row in request.nominal(name)
                        ),
                    ),
                )
                for name in request.required_operator_names
            ),
            error_radius=0.0,
            norm=OperatorNorm.SPECTRAL_L2,
            shared_uncertainty_id=request.shared_uncertainty_id,
            source="p9-test-verified-enclosure",
            certificate_id="p9-test-certificate",
            verified_remainder=True,
        )


def _operator_bundle(
    *,
    episode_id: str,
    stage: MonitorStage,
    raw_index: int,
    certified: bool = False,
    sensor_channel: str | None = None,
) -> OperatorBundle:
    """构造一维单步名义算子包，作为 P9 观测的可审计 P6 证据。"""

    return NominalJVPAssembler(
        resource_budget=OperatorAssemblyBudget(
            max_workspace_elements=100,
            max_persisted_elements=100,
        )
    ).assemble(
        transition_jacobians=torch.zeros((1, 1, 1), dtype=torch.float64),
        semantics=JacobianSemantics.NOMINAL_POINTWISE,
        path=OperatorPath(
            monitor_identity="p9-test-monitor",
            episode_id=episode_id,
            stage=stage,
            start_raw_index=raw_index - 1,
            raw_indices=(raw_index,),
        ),
        sensor_jvps=(
            None
            if sensor_channel is None
            else {
                sensor_channel: torch.ones((1, 1), dtype=torch.float64),
            }
        ),
        enclosure_provider=_CertifiedFixtureProvider() if certified else None,
    )


def _record(
    *,
    episode_id: str,
    stage: MonitorStage,
    raw_index: int,
    measurement: float,
) -> MonitorRecord:
    """构造包含控制、测量和安全外生量的最小 raw-window 记录。"""

    return MonitorRecord(
        raw_index=raw_index,
        episode_id=episode_id,
        stage=stage,
        control=(0.0,),
        measurement=(measurement,),
        exogenous=(0.0,),
        anchor_eligibility_score=0.0,
    )


def _branch_evidence(
    *,
    episode_id: str,
    stage: MonitorStage,
    raw_index: int,
    calibration_hash: str = "5" * 64,
    certified_operator: bool = False,
    sensor_channel: str | None = None,
) -> DeployedBranchEvidence:
    """构造带完整 P6 bundle 的一维 ``L/T/Gamma`` 分支证据。"""

    return DeployedBranchEvidence(
        branch_name="guard",
        matrix=((1.0,),),
        statistic=2.0,
        threshold=1.0,
        operator_bundle=_operator_bundle(
            episode_id=episode_id,
            stage=stage,
            raw_index=raw_index,
            certified=certified_operator,
            sensor_channel=sensor_channel,
        ),
        score_map_hash="1" * 64,
        calibration_hash=calibration_hash,
    )


def _linear_observation(
    *,
    episode_id: str,
    linear_feature: float = 2.0,
    detection_calibration_hash: str = "5" * 64,
    stage: MonitorStage = MonitorStage.ATTRIBUTION_CALIBRATION,
    certified_operator: bool = False,
) -> DeployedObservation:
    """构造 feature score 可手算的完整 attribution 观测。"""

    raw_window = (
        _record(
            episode_id=episode_id,
            stage=stage,
            raw_index=1,
            measurement=2.0,
        ),
    )
    state = MonitorState(
        monitor_identity="p9-test-monitor",
        episode_id=episode_id,
        stage=stage,
        last_raw_index=0,
    )
    return DeployedObservation(
        monitor_state=state,
        raw_window=raw_window,
        measurement_channels=("sensor-1",),
        safe_context=((0.0,),),
        branches=(
            _branch_evidence(
                episode_id=episode_id,
                stage=stage,
                raw_index=1,
                calibration_hash=detection_calibration_hash,
                certified_operator=certified_operator,
            ),
        ),
        mask_recomputations=(),
        feature_schema_hash="4" * 64,
        linear_features=(linear_feature,),
        detection_calibration_hash=detection_calibration_hash,
        detection_excess=2.0,
    )


def _ready_isolation_context(
    fault_centers: tuple[tuple[ExplanationFamily, float], ...],
    *,
    uncertified_faults: tuple[ExplanationFamily, ...] = (),
) -> tuple[FullNormalCalibrator, OuterExplanationOracle]:
    """构造有限 ``q_attr=0`` 的可手算线性隔离上下文。"""

    detection_calibration_hash = "a" * 64
    normal_family = ExplanationFamily(
        family_id="normal",
        label="Normal",
        sensor_channels=(),
        dynamics_sides=(),
        physical=False,
        equivalence_label=None,
        radius=0.0,
        normal=True,
    )
    cells = [
        LinearExplanationCell(
            cell_id="normal-cell",
            family_hash=normal_family.content_hash,
            feature_schema_hash="4" * 64,
            center=(0.0,),
            scale=(1.0,),
            coverage_evidence_hash="b" * 64,
        )
    ]
    cells.extend(
        LinearExplanationCell(
            cell_id=f"{family.family_id}-cell",
            family_hash=family.content_hash,
            feature_schema_hash="4" * 64,
            center=(center,),
            scale=(1.0,),
            coverage_evidence_hash=f"{index:x}" * 64,
        )
        for index, (family, center) in enumerate(fault_centers, start=1)
    )
    if uncertified_faults:
        oracle = OuterExplanationOracle.mixed(
            cells=tuple(cells),
            uncertified_family_hashes=tuple(
                family.content_hash for family in uncertified_faults
            ),
            feature_schema_hash="4" * 64,
            reason="Full nonlinear oracle backend is not certified.",
        )
    else:
        oracle = OuterExplanationOracle.linear(cells=tuple(cells))
    calibration = FullNormalCalibrator.fit(
        {
            f"attr-cal-{index}": (
                _linear_observation(
                    episode_id=f"attr-cal-{index}",
                    linear_feature=0.0,
                    detection_calibration_hash=detection_calibration_hash,
                ),
            )
            for index in range(1, 5)
        },
        normal_family=normal_family,
        oracle=oracle,
        stage=MonitorStage.ATTRIBUTION_CALIBRATION,
        beta=0.25,
        detection_alpha=0.3,
        detection_quantile=1.0,
        detection_calibration_hash=detection_calibration_hash,
        detection_source_hash="c" * 64,
        detection_episode_ids=("det-cal-1", "det-cal-2"),
        attribution_source_hash="d" * 64,
        episode_definition_hash="e" * 64,
        exchangeability_assumption_hash="f" * 64,
    )
    return calibration, oracle


def test_explanation_family_enforces_the_declared_mixed_concurrency_prior() -> None:
    """一个支持最多包含一个 sensor 和一个 dynamics-side 分量。"""

    learned_input_side = ExplanationFamily(
        family_id="learned-input-side",
        label="Learned-input-side",
        sensor_channels=(),
        dynamics_sides=(DynamicsSide.LEARNED_INPUT,),
        physical=False,
        equivalence_label="learned-input-side",
        radius=1.0,
    )

    assert learned_input_side.equivalence_label == "learned-input-side"
    with pytest.raises(ValueError, match="at most one sensor"):
        ExplanationFamily(
            family_id="two-sensors",
            label="Two sensors",
            sensor_channels=("sensor-1", "sensor-2"),
            dynamics_sides=(),
            physical=False,
            equivalence_label="sensor-side",
            radius=1.0,
        )


def test_deployed_observation_requires_full_mask_pipeline_recomputation() -> None:
    """mask 输出必须绑定源输入并包含重算 state、branch 和 operator 证据。"""

    episode_id = "attr-episode-1"
    stage = MonitorStage.ATTRIBUTION_CALIBRATION
    raw_window = (
        _record(
            episode_id=episode_id,
            stage=stage,
            raw_index=1,
            measurement=3.0,
        ),
    )
    state = MonitorState(
        monitor_identity="p9-test-monitor",
        episode_id=episode_id,
        stage=stage,
        last_raw_index=0,
    )
    branch = _branch_evidence(
        episode_id=episode_id,
        stage=stage,
        raw_index=1,
    )
    masked_window = (
        _record(
            episode_id=episode_id,
            stage=stage,
            raw_index=1,
            measurement=0.0,
        ),
    )
    mask = MaskRecomputation(
        sensor_channel="sensor-1",
        source_state_hash=state.content_hash,
        source_raw_window_hash=DeployedObservation.hash_raw_window(raw_window),
        pipeline_hash="3" * 64,
        masked_raw_window=masked_window,
        recomputed_state=state,
        recomputed_branches=(branch,),
        measurement_residual=(0.0,),
        exonerated=False,
    )
    observation = DeployedObservation(
        monitor_state=state,
        raw_window=raw_window,
        measurement_channels=("sensor-1",),
        safe_context=((0.0,),),
        branches=(branch,),
        mask_recomputations=(mask,),
        feature_schema_hash="4" * 64,
        linear_features=(2.0,),
        detection_calibration_hash="5" * 64,
        detection_excess=2.0,
    )

    assert observation.mask_recomputations[0].recomputed_branches == (branch,)
    sensor_family = ExplanationFamily(
        family_id="sensor-1",
        label="Sensor 1",
        sensor_channels=("sensor-1",),
        dynamics_sides=(),
        physical=False,
        equivalence_label="sensor-side",
        radius=0.0,
    )
    sensor_oracle = OuterExplanationOracle.linear(
        cells=(
            LinearExplanationCell(
                cell_id="sensor-1-cell",
                family_hash=sensor_family.content_hash,
                feature_schema_hash=observation.feature_schema_hash,
                center=(2.0,),
                scale=(1.0,),
                coverage_evidence_hash="6" * 64,
            ),
        ),
    )
    assert not observation.mask_recomputations[0].exonerated
    assert sensor_oracle.evaluate(observation, sensor_family).feasible
    with pytest.raises(ValueError, match="calibration"):
        replace(observation, detection_calibration_hash="9" * 64)
    with pytest.raises(ValueError, match="source raw window"):
        replace(
            observation,
            mask_recomputations=(
                replace(mask, source_raw_window_hash="0" * 64),
            ),
        )
    with pytest.raises(ValueError, match="recomputed branch"):
        replace(mask, recomputed_branches=())
    with pytest.raises(ValueError, match="at most one dynamics-side"):
        ExplanationFamily(
            family_id="two-dynamics",
            label="Two dynamics",
            sensor_channels=(),
            dynamics_sides=(DynamicsSide.ACTUATOR, DynamicsSide.PROCESS),
            physical=False,
            equivalence_label="dynamics-side",
            radius=1.0,
        )


def test_linear_outer_oracle_uses_right_closed_radius_membership() -> None:
    """精确落在半径边界的观测必须以 ``score <= radius`` 保留。"""

    episode_id = "attr-episode-linear"
    stage = MonitorStage.ATTRIBUTION_CALIBRATION
    raw_window = (
        _record(
            episode_id=episode_id,
            stage=stage,
            raw_index=1,
            measurement=2.0,
        ),
    )
    state = MonitorState(
        monitor_identity="p9-test-monitor",
        episode_id=episode_id,
        stage=stage,
        last_raw_index=0,
    )
    observation = DeployedObservation(
        monitor_state=state,
        raw_window=raw_window,
        measurement_channels=("sensor-1",),
        safe_context=((0.0,),),
        branches=(
            _branch_evidence(
                episode_id=episode_id,
                stage=stage,
                raw_index=1,
            ),
        ),
        mask_recomputations=(),
        feature_schema_hash="4" * 64,
        linear_features=(2.0,),
        detection_calibration_hash="5" * 64,
        detection_excess=2.0,
    )
    family = ExplanationFamily(
        family_id="process-side",
        label="Process-side",
        sensor_channels=(),
        dynamics_sides=(DynamicsSide.PROCESS_SIDE,),
        physical=False,
        equivalence_label="process-side",
        radius=1.0,
    )
    oracle = OuterExplanationOracle.linear(
        cells=(
            LinearExplanationCell(
                cell_id="process-side-cell",
                family_hash=family.content_hash,
                feature_schema_hash=observation.feature_schema_hash,
                center=(0.0,),
                scale=(2.0,),
                coverage_evidence_hash="6" * 64,
            ),
        ),
    )

    boundary = oracle.evaluate(observation, family, radius=1.0)
    just_below = oracle.evaluate(observation, family, radius=0.999)

    assert boundary.outer_score == pytest.approx(1.0)
    assert boundary.feasible
    assert boundary.certified
    assert not just_below.feasible
    assert just_below.certified


def test_monotone_refinement_cache_retains_unresolved_cells_across_radii() -> None:
    """未决 cell 保持可行，且后续 refinement 只能收紧 score 区间。"""

    observation = _linear_observation(episode_id="attr-episode-cache")
    family = ExplanationFamily(
        family_id="process-side",
        label="Process-side",
        sensor_channels=(),
        dynamics_sides=(DynamicsSide.PROCESS_SIDE,),
        physical=False,
        equivalence_label="process-side",
        radius=1.0,
    )
    oracle_hash = "7" * 64
    cache = MonotoneRefinementCache()
    cache.record(
        OracleCellRefinement(
            oracle_hash=oracle_hash,
            family_hash=family.content_hash,
            observation_hash=observation.content_hash,
            cell_id="unresolved-process-cell",
            refinement_level=0,
            lower_score=1.0,
            upper_score=2.0,
            coverage_evidence_hash="8" * 64,
        )
    )

    boundary = cache.evaluate(
        oracle_hash=oracle_hash,
        family=family,
        observation=observation,
        radius=1.0,
    )
    larger = cache.evaluate(
        oracle_hash=oracle_hash,
        family=family,
        observation=observation,
        radius=2.0,
    )

    assert boundary.feasible
    assert boundary.unresolved_cell_ids == ("unresolved-process-cell",)
    assert larger.feasible
    assert larger.unresolved_cell_ids == ()
    with pytest.raises(ValueError, match="monotone"):
        cache.record(
            OracleCellRefinement(
                oracle_hash=oracle_hash,
                family_hash=family.content_hash,
                observation_hash=observation.content_hash,
                cell_id="unresolved-process-cell",
                refinement_level=1,
                lower_score=0.5,
                upper_score=2.5,
                coverage_evidence_hash="9" * 64,
            )
        )


def test_full_normal_calibration_uses_episode_rank_and_infinite_resolution_guard() -> None:
    """归因校准取完整观测 episode maximum，分辨率不足时禁用 singleton。"""

    detection_calibration_hash = "a" * 64
    normal_family = ExplanationFamily(
        family_id="normal",
        label="Normal",
        sensor_channels=(),
        dynamics_sides=(),
        physical=False,
        equivalence_label=None,
        radius=0.0,
        normal=True,
    )
    observations = {
        f"attr-cal-{index}": (
            _linear_observation(
                episode_id=f"attr-cal-{index}",
                linear_feature=float(index),
                detection_calibration_hash=detection_calibration_hash,
            ),
        )
        for index in range(1, 5)
    }
    oracle = OuterExplanationOracle.linear(
        cells=(
            LinearExplanationCell(
                cell_id="normal-cell",
                family_hash=normal_family.content_hash,
                feature_schema_hash="4" * 64,
                center=(0.0,),
                scale=(1.0,),
                coverage_evidence_hash="b" * 64,
            ),
        ),
    )
    common = {
        "normal_family": normal_family,
        "oracle": oracle,
        "stage": MonitorStage.ATTRIBUTION_CALIBRATION,
        "detection_alpha": 0.3,
        "detection_quantile": 3.0,
        "detection_calibration_hash": detection_calibration_hash,
        "detection_source_hash": "c" * 64,
        "detection_episode_ids": ("det-cal-1", "det-cal-2"),
        "attribution_source_hash": "d" * 64,
        "episode_definition_hash": "e" * 64,
        "exchangeability_assumption_hash": "f" * 64,
    }

    finite = FullNormalCalibrator.fit(observations, beta=0.25, **common)
    below_resolution = FullNormalCalibrator.fit(
        observations,
        beta=0.1,
        **common,
    )

    assert finite.status is AttributionCalibrationStatus.READY
    assert finite.episode_maxima == pytest.approx((1.0, 2.0, 3.0, 4.0))
    assert finite.rank == 4
    assert finite.quantile == pytest.approx(4.0)
    assert finite.risk_resolution == pytest.approx(0.2)
    assert finite.nonnormal_singleton_enabled
    assert (
        below_resolution.status
        is AttributionCalibrationStatus.INSUFFICIENT_RESOLUTION
    )
    assert below_resolution.quantile == float("inf")
    assert not below_resolution.nonnormal_singleton_enabled
    with pytest.raises(ValueError, match="must not overlap"):
        FullNormalCalibrator.fit(
            observations,
            beta=0.25,
            **{
                **common,
                "detection_episode_ids": ("attr-cal-1",),
            },
        )
    with pytest.raises(ValueError, match="sources must be independent"):
        FullNormalCalibrator.fit(
            observations,
            beta=0.25,
            **{
                **common,
                "attribution_source_hash": common["detection_source_hash"],
            },
        )
    tampered_scores = (
        (99.0,),
        *finite.episode_scores[1:],
    )
    with pytest.raises(ValueError, match="derived from"):
        replace(finite, episode_scores=tampered_scores)


def test_infinite_attribution_quantile_forces_normal_compatible_output() -> None:
    """分辨率不足时 Normal 永远保留，不能输出非正常 singleton。"""

    detection_calibration_hash = "a" * 64
    normal_family = ExplanationFamily(
        family_id="normal",
        label="Normal",
        sensor_channels=(),
        dynamics_sides=(),
        physical=False,
        equivalence_label=None,
        radius=0.0,
        normal=True,
    )
    fault_family = ExplanationFamily(
        family_id="process-side",
        label="Process-side",
        sensor_channels=(),
        dynamics_sides=(DynamicsSide.PROCESS_SIDE,),
        physical=False,
        equivalence_label="process-side",
        radius=0.5,
    )
    oracle = OuterExplanationOracle.linear(
        cells=(
            LinearExplanationCell(
                cell_id="normal-cell",
                family_hash=normal_family.content_hash,
                feature_schema_hash="4" * 64,
                center=(0.0,),
                scale=(1.0,),
                coverage_evidence_hash="b" * 64,
            ),
            LinearExplanationCell(
                cell_id="process-cell",
                family_hash=fault_family.content_hash,
                feature_schema_hash="4" * 64,
                center=(10.0,),
                scale=(1.0,),
                coverage_evidence_hash="c" * 64,
            ),
        ),
    )
    calibration = FullNormalCalibrator.fit(
        {
            "attr-cal-1": (
                _linear_observation(
                    episode_id="attr-cal-1",
                    linear_feature=0.0,
                    detection_calibration_hash=detection_calibration_hash,
                ),
            )
        },
        normal_family=normal_family,
        oracle=oracle,
        stage=MonitorStage.ATTRIBUTION_CALIBRATION,
        beta=0.2,
        detection_alpha=0.3,
        detection_quantile=1.0,
        detection_calibration_hash=detection_calibration_hash,
        detection_source_hash="d" * 64,
        detection_episode_ids=("det-cal-1",),
        attribution_source_hash="e" * 64,
        episode_definition_hash="f" * 64,
        exchangeability_assumption_hash="1" * 64,
    )
    final_observation = _linear_observation(
        episode_id="fault-test-1",
        linear_feature=10.0,
        detection_calibration_hash=detection_calibration_hash,
        stage=MonitorStage.FROZEN_FAULT_TEST,
    )

    candidates = IsolationCandidateSet.evaluate(
        final_observation,
        normal_calibration=calibration,
        fault_families=(fault_family,),
        oracle=oracle,
    )
    report = IsolationReport.from_candidate_set(
        candidates,
        observation=final_observation,
    )

    assert candidates.candidate_family_ids == ("normal", "process-side")
    assert report.outcome is IsolationOutcome.NORMAL_COMPATIBLE
    assert report.display_label == "Normal-compatible"
    assert report.reported_family_id is None


def test_full_nonlinear_oracle_is_explicitly_uncertified_and_retains_family() -> None:
    """未实现的 nonlinear certification 必须保守保留解释并显式拒绝认证。"""

    observation = _linear_observation(
        episode_id="fault-test-uncertified",
        stage=MonitorStage.FROZEN_FAULT_TEST,
    )
    family = ExplanationFamily(
        family_id="physical-process",
        label="Physical process",
        sensor_channels=(),
        dynamics_sides=(DynamicsSide.PROCESS,),
        physical=True,
        equivalence_label=None,
        radius=1.0,
        physical_evidence_hash="a" * 64,
    )
    oracle = OuterExplanationOracle.uncertified(
        family_hashes=(family.content_hash,),
        feature_schema_hash=observation.feature_schema_hash,
        reason="Full nonlinear oracle backend is not certified.",
    )

    evaluation = oracle.evaluate(observation, family)

    assert evaluation.feasible
    assert not evaluation.certified
    assert "not certified" in (evaluation.reason or "").lower()


def test_physical_singleton_requires_certified_operator_evidence() -> None:
    """算法候选唯一但 P6 operator 仅 nominal 时必须输出 Uncertified。"""

    physical_process = ExplanationFamily(
        family_id="physical-process",
        label="Physical process",
        sensor_channels=(),
        dynamics_sides=(DynamicsSide.PROCESS,),
        physical=True,
        equivalence_label=None,
        radius=0.5,
        physical_evidence_hash="1" * 64,
    )
    calibration, oracle = _ready_isolation_context(
        ((physical_process, 10.0),)
    )
    observation = _linear_observation(
        episode_id="fault-test-physical",
        linear_feature=10.0,
        detection_calibration_hash=calibration.detection_calibration_hash,
        stage=MonitorStage.FROZEN_FAULT_TEST,
    )

    candidates = IsolationCandidateSet.evaluate(
        observation,
        normal_calibration=calibration,
        fault_families=(physical_process,),
        oracle=oracle,
    )
    report = IsolationReport.from_candidate_set(
        candidates,
        observation=observation,
    )

    assert candidates.candidate_family_ids == ("physical-process",)
    assert report.outcome is IsolationOutcome.UNCERTIFIED
    assert report.display_label == "Uncertified"
    assert report.reported_family_id is None


def test_uncertified_equivalence_singleton_is_reported_as_uncertified() -> None:
    """equivalence family 的 oracle 未认证时也不能输出 singleton。"""

    process_side = ExplanationFamily(
        family_id="process-side",
        label="Process side",
        sensor_channels=(),
        dynamics_sides=(DynamicsSide.PROCESS_SIDE,),
        physical=False,
        equivalence_label="process-side",
        radius=0.5,
    )
    calibration, oracle = _ready_isolation_context(
        (),
        uncertified_faults=(process_side,),
    )
    observation = _linear_observation(
        episode_id="fault-test-uncertified-equivalence",
        linear_feature=10.0,
        detection_calibration_hash=calibration.detection_calibration_hash,
        stage=MonitorStage.FROZEN_FAULT_TEST,
    )
    candidates = IsolationCandidateSet.evaluate(
        observation,
        normal_calibration=calibration,
        fault_families=(process_side,),
        oracle=oracle,
    )
    report = IsolationReport.from_candidate_set(
        candidates,
        observation=observation,
    )

    assert report.outcome is IsolationOutcome.UNCERTIFIED
    assert report.display_label == "Uncertified"
    assert report.reported_family_id is None


def test_physical_sensor_singleton_requires_matching_mask_and_sensor_jvp() -> None:
    """只有 certified 通用算子、没有目标 mask/JVP 时不得确认物理 sensor。"""

    physical_sensor = ExplanationFamily(
        family_id="physical-sensor-1",
        label="Physical sensor 1",
        sensor_channels=("sensor-1",),
        dynamics_sides=(),
        physical=True,
        equivalence_label=None,
        radius=0.5,
        physical_evidence_hash="1" * 64,
    )
    calibration, oracle = _ready_isolation_context(
        ((physical_sensor, 10.0),)
    )
    observation = _linear_observation(
        episode_id="fault-test-physical-sensor",
        linear_feature=10.0,
        detection_calibration_hash=calibration.detection_calibration_hash,
        stage=MonitorStage.FROZEN_FAULT_TEST,
        certified_operator=True,
    )
    candidates = IsolationCandidateSet.evaluate(
        observation,
        normal_calibration=calibration,
        fault_families=(physical_sensor,),
        oracle=oracle,
    )

    report = IsolationReport.from_candidate_set(
        candidates,
        observation=observation,
    )

    assert observation.all_operators_certified
    assert report.outcome is IsolationOutcome.UNCERTIFIED
    assert report.reported_family_id is None


def test_physical_sensor_singleton_accepts_matching_mask_and_certified_jvp() -> None:
    """完整目标 mask 与 certified sensor JVP 齐全时允许物理 sensor singleton。"""

    physical_sensor = ExplanationFamily(
        family_id="physical-sensor-1",
        label="Physical sensor 1",
        sensor_channels=("sensor-1",),
        dynamics_sides=(),
        physical=True,
        equivalence_label=None,
        radius=0.5,
        physical_evidence_hash="1" * 64,
    )
    calibration, oracle = _ready_isolation_context(
        ((physical_sensor, 10.0),)
    )
    episode_id = "fault-test-certified-sensor"
    stage = MonitorStage.FROZEN_FAULT_TEST
    state = MonitorState(
        monitor_identity="p9-test-monitor",
        episode_id=episode_id,
        stage=stage,
        last_raw_index=0,
    )
    raw_window = (
        _record(
            episode_id=episode_id,
            stage=stage,
            raw_index=1,
            measurement=10.0,
        ),
    )
    branch = _branch_evidence(
        episode_id=episode_id,
        stage=stage,
        raw_index=1,
        calibration_hash=calibration.detection_calibration_hash,
        certified_operator=True,
        sensor_channel="sensor-1",
    )
    mask = MaskRecomputation(
        sensor_channel="sensor-1",
        source_state_hash=state.content_hash,
        source_raw_window_hash=DeployedObservation.hash_raw_window(raw_window),
        pipeline_hash="2" * 64,
        masked_raw_window=(
            _record(
                episode_id=episode_id,
                stage=stage,
                raw_index=1,
                measurement=0.0,
            ),
        ),
        recomputed_state=state,
        recomputed_branches=(branch,),
        measurement_residual=(0.0,),
        exonerated=False,
    )
    observation = DeployedObservation(
        monitor_state=state,
        raw_window=raw_window,
        measurement_channels=("sensor-1",),
        safe_context=((0.0,),),
        branches=(branch,),
        mask_recomputations=(mask,),
        feature_schema_hash="4" * 64,
        linear_features=(10.0,),
        detection_calibration_hash=calibration.detection_calibration_hash,
        detection_excess=2.0,
    )
    candidates = IsolationCandidateSet.evaluate(
        observation,
        normal_calibration=calibration,
        fault_families=(physical_sensor,),
        oracle=oracle,
    )

    report = IsolationReport.from_candidate_set(
        candidates,
        observation=observation,
    )

    assert observation.certifies_physical_family(physical_sensor)
    assert report.outcome is IsolationOutcome.SINGLETON
    assert report.display_label == "Physical sensor 1"
    assert report.reported_family_id == "physical-sensor-1"


def test_nonphysical_singleton_reports_only_its_equivalence_class() -> None:
    """缺少物理映射时 singleton 只能使用较弱 equivalence label。"""

    process_side = ExplanationFamily(
        family_id="process-side",
        label="Process-side model response",
        sensor_channels=(),
        dynamics_sides=(DynamicsSide.PROCESS_SIDE,),
        physical=False,
        equivalence_label="process-side",
        radius=0.5,
    )
    calibration, oracle = _ready_isolation_context(((process_side, 10.0),))
    observation = _linear_observation(
        episode_id="fault-test-equivalence",
        linear_feature=10.0,
        detection_calibration_hash=calibration.detection_calibration_hash,
        stage=MonitorStage.FROZEN_FAULT_TEST,
    )

    candidates = IsolationCandidateSet.evaluate(
        observation,
        normal_calibration=calibration,
        fault_families=(process_side,),
        oracle=oracle,
    )
    report = IsolationReport.from_candidate_set(
        candidates,
        observation=observation,
    )

    assert report.outcome is IsolationOutcome.SINGLETON
    assert report.display_label == "process-side"
    assert report.reported_family_id == "process-side"
    assert all(
        forbidden not in report.risk_statement.lower()
        for forbidden in ("ppv", "fdr", "posterior", "confidence")
    )


def test_candidate_set_rejects_tampered_fault_radius_or_oracle_identity() -> None:
    """公开结果对象不得用放宽半径或伪造 oracle 来源制造 singleton。"""

    process_side = ExplanationFamily(
        family_id="process-side",
        label="Process-side model response",
        sensor_channels=(),
        dynamics_sides=(DynamicsSide.PROCESS_SIDE,),
        physical=False,
        equivalence_label="process-side",
        radius=0.5,
    )
    calibration, oracle = _ready_isolation_context(((process_side, 10.0),))
    observation = _linear_observation(
        episode_id="fault-test-tampered-evidence",
        linear_feature=10.0,
        detection_calibration_hash=calibration.detection_calibration_hash,
        stage=MonitorStage.FROZEN_FAULT_TEST,
    )
    candidates = IsolationCandidateSet.evaluate(
        observation,
        normal_calibration=calibration,
        fault_families=(process_side,),
        oracle=oracle,
    )
    fault_evaluation = next(
        evaluation
        for evaluation in candidates.evaluations
        if evaluation.family_id == process_side.family_id
    )

    tampered_radius = tuple(
        replace(evaluation, radius=999.0)
        if evaluation.family_id == process_side.family_id
        else evaluation
        for evaluation in candidates.evaluations
    )
    with pytest.raises(ValueError, match="frozen family radius"):
        replace(candidates, evaluations=tampered_radius)

    with pytest.raises(ValueError, match="frozen oracle"):
        replace(
            candidates,
            evaluations=tuple(
                replace(evaluation, oracle_hash="0" * 64)
                if evaluation is fault_evaluation
                else evaluation
                for evaluation in candidates.evaluations
            ),
        )


def test_candidate_set_replays_frozen_normal_quantile_and_detection_gate() -> None:
    """直接构造不得缩小 ``q_attr`` 或篡改 detection gate 来排除 Normal。"""

    detection_calibration_hash = "a" * 64
    normal_family = ExplanationFamily(
        family_id="normal",
        label="Normal",
        sensor_channels=(),
        dynamics_sides=(),
        physical=False,
        equivalence_label=None,
        radius=0.0,
        normal=True,
    )
    process_side = ExplanationFamily(
        family_id="process-side",
        label="Process-side model response",
        sensor_channels=(),
        dynamics_sides=(DynamicsSide.PROCESS_SIDE,),
        physical=False,
        equivalence_label="process-side",
        radius=0.5,
    )
    oracle = OuterExplanationOracle.linear(
        cells=(
            LinearExplanationCell(
                cell_id="normal-cell",
                family_hash=normal_family.content_hash,
                feature_schema_hash="4" * 64,
                center=(0.0,),
                scale=(1.0,),
                coverage_evidence_hash="b" * 64,
            ),
            LinearExplanationCell(
                cell_id="process-cell",
                family_hash=process_side.content_hash,
                feature_schema_hash="4" * 64,
                center=(4.0,),
                scale=(1.0,),
                coverage_evidence_hash="c" * 64,
            ),
        )
    )
    calibration = FullNormalCalibrator.fit(
        {
            f"attr-cal-{index}": (
                _linear_observation(
                    episode_id=f"attr-cal-{index}",
                    linear_feature=float(index),
                    detection_calibration_hash=detection_calibration_hash,
                ),
            )
            for index in range(1, 5)
        },
        normal_family=normal_family,
        oracle=oracle,
        stage=MonitorStage.ATTRIBUTION_CALIBRATION,
        beta=0.25,
        detection_alpha=0.3,
        detection_quantile=1.0,
        detection_calibration_hash=detection_calibration_hash,
        detection_source_hash="d" * 64,
        detection_episode_ids=("det-cal-1", "det-cal-2"),
        attribution_source_hash="e" * 64,
        episode_definition_hash="f" * 64,
        exchangeability_assumption_hash="1" * 64,
    )
    observation = _linear_observation(
        episode_id="fault-test-normal-radius",
        linear_feature=4.0,
        detection_calibration_hash=detection_calibration_hash,
        stage=MonitorStage.FROZEN_FAULT_TEST,
    )
    candidates = IsolationCandidateSet.evaluate(
        observation,
        normal_calibration=calibration,
        fault_families=(process_side,),
        oracle=oracle,
    )
    normal_evaluation = next(
        evaluation
        for evaluation in candidates.evaluations
        if evaluation.family_id == normal_family.family_id
    )
    tampered_evaluations = tuple(
        replace(evaluation, radius=0.0, feasible=False)
        if evaluation is normal_evaluation
        else evaluation
        for evaluation in candidates.evaluations
    )

    with pytest.raises(ValueError, match="replay the frozen calibration"):
        replace(
            candidates,
            candidate_families=(process_side,),
            evaluations=tampered_evaluations,
            detection_normal_compatible=False,
        )


@pytest.mark.parametrize(
    "risk_statement",
    (
        "PPV is 99%.",
        "FDR is controlled.",
        "Posterior probability is 99%.",
        "Confidence is 99%.",
        "Positive predictive value is 99%.",
        "This label is correct with probability 0.99.",
    ),
)
def test_isolation_report_rejects_posterior_or_classification_risk_claims(
    risk_statement: str,
) -> None:
    """直接替换报告措辞也不能冒充类别后验或分类性能保证。"""

    process_side = ExplanationFamily(
        family_id="process-side",
        label="Process-side model response",
        sensor_channels=(),
        dynamics_sides=(DynamicsSide.PROCESS_SIDE,),
        physical=False,
        equivalence_label="process-side",
        radius=0.5,
    )
    calibration, oracle = _ready_isolation_context(((process_side, 10.0),))
    observation = _linear_observation(
        episode_id="fault-test-risk-language",
        linear_feature=10.0,
        detection_calibration_hash=calibration.detection_calibration_hash,
        stage=MonitorStage.FROZEN_FAULT_TEST,
    )
    candidates = IsolationCandidateSet.evaluate(
        observation,
        normal_calibration=calibration,
        fault_families=(process_side,),
        oracle=oracle,
    )
    report = IsolationReport.from_candidate_set(candidates, observation=observation)

    with pytest.raises(ValueError, match="posterior or classification"):
        replace(report, risk_statement=risk_statement)


@pytest.mark.parametrize(
    ("outcome", "display_label"),
    (
        (IsolationOutcome.OUT_OF_MODEL, "Out-of-model"),
        (IsolationOutcome.NORMAL_COMPATIBLE, "Normal-compatible"),
        (IsolationOutcome.NONUNIQUE, "Nonunique"),
    ),
)
def test_isolation_report_rejects_forged_set_semantics(
    outcome: IsolationOutcome,
    display_label: str,
) -> None:
    """非 singleton 结果也必须由候选集合基数和 Normal 包含关系唯一决定。"""

    process_side = ExplanationFamily(
        family_id="process-side",
        label="Process-side model response",
        sensor_channels=(),
        dynamics_sides=(DynamicsSide.PROCESS_SIDE,),
        physical=False,
        equivalence_label="process-side",
        radius=0.5,
    )
    calibration, oracle = _ready_isolation_context(((process_side, 10.0),))
    observation = _linear_observation(
        episode_id="fault-test-forged-report",
        linear_feature=10.0,
        detection_calibration_hash=calibration.detection_calibration_hash,
        stage=MonitorStage.FROZEN_FAULT_TEST,
    )
    candidates = IsolationCandidateSet.evaluate(
        observation,
        normal_calibration=calibration,
        fault_families=(process_side,),
        oracle=oracle,
    )
    report = IsolationReport.from_candidate_set(candidates, observation=observation)

    with pytest.raises(ValueError, match="candidate-set semantics"):
        replace(
            report,
            outcome=outcome,
            display_label=display_label,
            reported_family_id=None,
        )
    with pytest.raises(ValueError, match="candidate-set semantics"):
        replace(report, normal_family_id="forged-normal")


def test_multiple_fault_explanations_are_reported_as_nonunique() -> None:
    """两个 fault outer family 同时覆盖时必须保留整个候选集。"""

    learned_input = ExplanationFamily(
        family_id="learned-input",
        label="Learned input",
        sensor_channels=(),
        dynamics_sides=(DynamicsSide.LEARNED_INPUT,),
        physical=False,
        equivalence_label="learned-input-side",
        radius=0.5,
    )
    process_side = ExplanationFamily(
        family_id="process-side",
        label="Process side",
        sensor_channels=(),
        dynamics_sides=(DynamicsSide.PROCESS_SIDE,),
        physical=False,
        equivalence_label="process-side",
        radius=0.5,
    )
    calibration, oracle = _ready_isolation_context(
        ((learned_input, 10.0), (process_side, 10.0))
    )
    observation = _linear_observation(
        episode_id="fault-test-nonunique",
        linear_feature=10.0,
        detection_calibration_hash=calibration.detection_calibration_hash,
        stage=MonitorStage.FROZEN_FAULT_TEST,
    )

    candidates = IsolationCandidateSet.evaluate(
        observation,
        normal_calibration=calibration,
        fault_families=(learned_input, process_side),
        oracle=oracle,
    )
    report = IsolationReport.from_candidate_set(
        candidates,
        observation=observation,
    )

    assert candidates.candidate_family_ids == ("learned-input", "process-side")
    assert report.outcome is IsolationOutcome.NONUNIQUE
    assert report.display_label == "Nonunique"
    with pytest.raises(ValueError, match="complete frozen fault dictionary"):
        IsolationCandidateSet.evaluate(
            observation,
            normal_calibration=calibration,
            fault_families=(learned_input,),
            oracle=oracle,
        )


def test_no_declared_explanation_covering_observation_is_out_of_model() -> None:
    """Normal 与全部 fault family 都排除时返回 Out-of-model。"""

    process_side = ExplanationFamily(
        family_id="process-side",
        label="Process side",
        sensor_channels=(),
        dynamics_sides=(DynamicsSide.PROCESS_SIDE,),
        physical=False,
        equivalence_label="process-side",
        radius=0.5,
    )
    calibration, oracle = _ready_isolation_context(((process_side, 20.0),))
    observation = _linear_observation(
        episode_id="fault-test-out-of-model",
        linear_feature=10.0,
        detection_calibration_hash=calibration.detection_calibration_hash,
        stage=MonitorStage.FROZEN_FAULT_TEST,
    )

    candidates = IsolationCandidateSet.evaluate(
        observation,
        normal_calibration=calibration,
        fault_families=(process_side,),
        oracle=oracle,
    )
    report = IsolationReport.from_candidate_set(
        candidates,
        observation=observation,
    )

    assert candidates.candidate_family_ids == ()
    assert report.outcome is IsolationOutcome.OUT_OF_MODEL
    assert report.display_label == "Out-of-model"
