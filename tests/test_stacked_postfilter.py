"""P7 white-space 堆叠残差后滤波的公开行为测试。

文件用途：
    通过手算小矩阵验证白化、锚点谱商空间、guard 与 matched branches 的部署坐标，
    防止把 white-space 投影器错误地直接用于原始残差。
主要职责：
    只测试 ``WhiteningEstimate``、``PostFilterCandidate`` 和 ``BranchBank`` 的公开接口；
    不测试 P8 动态阈值、P9 集合值隔离或真实 CSTR 故障性能。
关键输入与输出：
    输入为合成的仅正常 estimate 残差、可手算 ``G_0``、锚点椭球与固定投影器；输出为
    冻结白化矩阵、分支算子、统计量、诊断字段和 JSON 往返结果。
依赖与副作用：
    依赖 NumPy、pytest 和 Joff evaluation 公共入口；不读写文件、不访问网络。
重要约束：
    所有拟合/选择测试只允许 ``estimate`` 阶段；校准和故障数据不得改变候选、支路或
    fallback 决策。测试中的合成故障方向仅用于验证几何盲向，不构成论文实验结果。
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from joff.evaluation import (
    BranchKind,
    MonitorStage,
    PostFilterCandidate,
    SpectralMode,
    WhiteningEstimate,
)


def _estimate_residuals() -> np.ndarray:
    """返回带跨坐标相关性的确定性正常 estimate 残差。"""

    return np.asarray(
        [
            [-2.0, -1.0],
            [-1.0, -2.0],
            [-0.5, 0.5],
            [0.5, -0.5],
            [1.0, 2.0],
            [2.0, 1.0],
        ],
        dtype=np.float64,
    )


def _partial_candidate(
    *,
    matched_projectors: dict[str, np.ndarray] | None = None,
) -> PostFilterCandidate:
    """构造一个选择一维、保留一维的确定性 partial quotient 候选。"""

    whitening = WhiteningEstimate.fit(
        _estimate_residuals(),
        stage=MonitorStage.ESTIMATE,
        minimum_samples=4,
        ridge=1e-8,
        eigenvalue_floor=1e-10,
    )
    w_0 = np.asarray(whitening.operator)
    white_anchor = np.diag([3.0, 1.0])
    raw_anchor = np.linalg.solve(w_0, white_anchor)
    return PostFilterCandidate.fit(
        candidate_id="partial-fixture",
        whitening=whitening,
        anchor_response=raw_anchor,
        anchor_covariance_sqrt=np.eye(2, dtype=np.float64),
        singular_value_threshold=2.0,
        stage=MonitorStage.ESTIMATE,
        matched_projectors=matched_projectors,
    )


def test_white_space_quotient_builds_deployed_operator_in_raw_coordinates() -> None:
    """白空间选出的方向必须通过 ``Q_w.T @ W_0`` 映回原残差坐标。"""

    whitening = WhiteningEstimate.fit(
        _estimate_residuals(),
        stage=MonitorStage.ESTIMATE,
        minimum_samples=4,
        ridge=1e-8,
        eigenvalue_floor=1e-10,
    )
    w_0 = np.asarray(whitening.operator)
    white_anchor = np.asarray([[3.0], [0.0]], dtype=np.float64)
    raw_anchor = np.linalg.solve(w_0, white_anchor)

    candidate = PostFilterCandidate.fit(
        candidate_id="white-coordinate-fixture",
        whitening=whitening,
        anchor_response=raw_anchor,
        anchor_covariance_sqrt=np.eye(1, dtype=np.float64),
        singular_value_threshold=2.0,
        stage=MonitorStage.ESTIMATE,
    )

    l_0 = np.asarray(candidate.common_operator)
    assert candidate.selected_rank == 1
    assert candidate.retained_rank == 1
    assert np.allclose(l_0 @ raw_anchor, 0.0, atol=1e-12, rtol=0.0)
    assert np.allclose(
        l_0,
        np.asarray(candidate.retained_white_basis).T @ w_0,
        atol=1e-12,
        rtol=0.0,
    )


def test_frozen_candidate_round_trips_through_json_and_replays_statistics() -> None:
    """JSON 重放必须恢复同一分支矩阵、诊断字段、hash 和可观测统计量。"""

    candidate = _partial_candidate(
        matched_projectors={"retained-axis": np.eye(1, dtype=np.float64)}
    )
    residual = np.asarray([0.25, -0.75], dtype=np.float64)
    expected_statistics = candidate.branch_bank.evaluate(residual)

    payload = json.loads(json.dumps(candidate.to_dict()))
    replayed = PostFilterCandidate.from_dict(payload)

    assert replayed.to_dict() == payload
    assert replayed.content_hash == candidate.content_hash
    assert replayed.branch_bank.evaluate(residual) == expected_statistics


def test_unstable_covariance_fallback_preserves_the_observed_condition_number() -> None:
    """guard-only 可使用单位阵，但审计字段必须保留触发停止条件的实际条件数。"""

    whitening = WhiteningEstimate.fit(
        _estimate_residuals(),
        stage=MonitorStage.ESTIMATE,
        minimum_samples=4,
        ridge=0.0,
        eigenvalue_floor=1e-12,
        max_condition_number=1.01,
    )

    assert not whitening.stable
    assert whitening.condition_number > whitening.max_condition_number
    assert "condition_number" in (whitening.fallback_reason or "")
    assert np.array_equal(np.asarray(whitening.operator), np.eye(2))


def test_singular_value_threshold_is_squared_before_gram_spectrum_selection() -> None:
    """``tau=1.5`` 不得错误选择 Gram 特征值 2.0，因为正确阈值是 2.25。"""

    whitening = WhiteningEstimate.fit(
        _estimate_residuals(),
        stage=MonitorStage.ESTIMATE,
    )
    w_0 = np.asarray(whitening.operator)
    white_anchor = np.diag([np.sqrt(2.0), 0.5])
    candidate = PostFilterCandidate.fit(
        candidate_id="tau-square-fixture",
        whitening=whitening,
        anchor_response=np.linalg.solve(w_0, white_anchor),
        anchor_covariance_sqrt=np.eye(2),
        singular_value_threshold=1.5,
        stage=MonitorStage.ESTIMATE,
        minimum_projector_gap=0.0,
        spectral_cluster_tolerance=0.0,
    )

    assert candidate.gram_threshold == pytest.approx(2.25)
    assert candidate.gram_eigenvalues == pytest.approx((2.0, 0.25))
    assert candidate.selected_rank == 0
    assert candidate.mode is SpectralMode.NO_QUOTIENT


def test_repeated_spectral_cluster_is_never_split_and_collision_is_hybrid() -> None:
    """阈值穿过近重复谱簇时必须整簇决策，并显式记录独立 hybrid mode。"""

    whitening = WhiteningEstimate.fit(
        np.column_stack((_estimate_residuals(), np.asarray([-1, 1, -1, 1, -1, 1]))),
        stage=MonitorStage.ESTIMATE,
    )
    w_0 = np.asarray(whitening.operator)
    white_anchor = np.diag(
        [
            np.sqrt(4.0 + 1e-10),
            np.sqrt(4.0 - 1e-10),
            0.25,
        ]
    )
    candidate = PostFilterCandidate.fit(
        candidate_id="cluster-collision-fixture",
        whitening=whitening,
        anchor_response=np.linalg.solve(w_0, white_anchor),
        anchor_covariance_sqrt=np.eye(3),
        singular_value_threshold=2.0,
        stage=MonitorStage.ESTIMATE,
        minimum_projector_gap=0.0,
        spectral_cluster_tolerance=1e-8,
    )

    assert candidate.mode is SpectralMode.HYBRID
    assert candidate.selected_rank in {0, 2}
    assert candidate.selected_rank != 1
    assert candidate.projector_gap <= 1e-8


def test_guard_and_matched_branches_share_the_frozen_deployed_geometry() -> None:
    """guard 保留 quotient 盲向，matched branch 必须严格等于 ``P_c @ L_0``。"""

    projector = np.eye(1, dtype=np.float64)
    candidate = _partial_candidate(matched_projectors={"retained-axis": projector})
    l_0 = np.asarray(candidate.common_operator)
    omnibus = candidate.branch_bank.branch("omnibus")
    guard = candidate.branch_bank.branch("guard")
    matched = candidate.branch_bank.branch("retained-axis")
    w_0 = np.asarray(candidate.whitening.operator)
    removed_raw_direction = np.linalg.solve(w_0, np.asarray([1.0, 0.0]))

    assert np.allclose(np.asarray(omnibus.matrix), l_0)
    assert np.allclose(np.asarray(matched.matrix), projector @ l_0)
    assert omnibus.anchor_radius == pytest.approx(1.0)
    assert matched.anchor_radius == pytest.approx(1.0)
    assert omnibus.statistic(removed_raw_direction) == pytest.approx(0.0, abs=1e-12)
    assert guard.statistic(removed_raw_direction) > 0.0
    assert np.array_equal(np.asarray(guard.matrix), np.eye(2))
    assert candidate.branch_bank.requires_recalibration


def test_zero_scale_matched_branch_is_disabled_by_the_frozen_positive_floor() -> None:
    """零投影支路不能进入 max-score；禁用决定及正 floor 必须写入候选产物。"""

    candidate = _partial_candidate(
        matched_projectors={"zero-scale": np.zeros((1, 1), dtype=np.float64)}
    )
    zero_branch = candidate.branch_bank.branch("zero-scale")

    assert not zero_branch.enabled
    assert "below floor" in (zero_branch.disabled_reason or "")
    with pytest.raises(RuntimeError, match="disabled"):
        zero_branch.statistic(np.asarray([1.0, 1.0]))
    assert candidate.to_dict()["branch_scale_floor"] > 0.0


def test_fit_and_selection_reject_every_stage_except_normal_estimate() -> None:
    """检测校准、归因校准、正常测试和故障测试都不能重拟合 P7 候选。"""

    with pytest.raises(ValueError, match="estimate"):
        WhiteningEstimate.fit(
            _estimate_residuals(),
            stage=MonitorStage.DETECTION_CALIBRATION,
        )

    whitening = WhiteningEstimate.fit(
        _estimate_residuals(),
        stage=MonitorStage.ESTIMATE,
    )
    with pytest.raises(ValueError, match="estimate"):
        PostFilterCandidate.fit(
            candidate_id="illegal-calibration-selection",
            whitening=whitening,
            anchor_response=np.eye(2),
            anchor_covariance_sqrt=np.eye(2),
            singular_value_threshold=1.0,
            stage=MonitorStage.ATTRIBUTION_CALIBRATION,
        )


def test_insufficient_estimate_samples_fail_closed_to_guard_only() -> None:
    """估计窗口不足时不得借用校准数据，应确定性退回唯一 raw guard。"""

    whitening = WhiteningEstimate.fit(
        _estimate_residuals()[:2],
        stage=MonitorStage.ESTIMATE,
        minimum_samples=4,
    )
    candidate = PostFilterCandidate.fit(
        candidate_id="insufficient-estimate-fixture",
        whitening=whitening,
        anchor_response=np.eye(2),
        anchor_covariance_sqrt=np.eye(2),
        singular_value_threshold=1.0,
        stage=MonitorStage.ESTIMATE,
    )

    assert not whitening.stable
    assert candidate.mode is SpectralMode.GUARD_ONLY
    assert tuple(branch.name for branch in candidate.branch_bank.branches) == ("guard",)
    assert candidate.fallback_reason == whitening.fallback_reason
    assert candidate.branch_bank.evaluate(np.asarray([3.0, 4.0])) == {"guard": 5.0}


def test_candidate_json_rejects_unknown_fields() -> None:
    """严格产物 schema 必须拒绝无法解释的额外字段。"""

    payload = _partial_candidate().to_dict()
    payload["unexpected"] = "cannot-be-silently-ignored"

    with pytest.raises(ValueError, match="extra"):
        PostFilterCandidate.from_dict(payload)


def test_candidate_json_rejects_tampered_derived_geometry() -> None:
    """``tau^2``、``L_0`` 和锚点半径不能脱离来源矩阵被单独改写。"""

    original = _partial_candidate().to_dict()
    tampered_threshold = copy.deepcopy(original)
    tampered_threshold["gram_threshold"] = 99.0
    tampered_common_operator = copy.deepcopy(original)
    tampered_common_operator["common_operator"][0][0] += 1.0
    tampered_anchor_radius = copy.deepcopy(original)
    tampered_anchor_radius["branch_bank"]["branches"][1]["anchor_radius"] += 1.0

    for payload in (
        tampered_threshold,
        tampered_common_operator,
        tampered_anchor_radius,
    ):
        with pytest.raises(ValueError):
            PostFilterCandidate.from_dict(payload)


def test_branch_transforms_nuisance_and_signature_columns_with_the_same_l_b() -> None:
    """P8/P9 列算子必须经分支公开接口使用与统计量完全相同的 ``L_b``。"""

    candidate = _partial_candidate(
        matched_projectors={"retained-axis": np.eye(1, dtype=np.float64)}
    )
    branch = candidate.branch_bank.branch("retained-axis")
    raw_columns = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    transformed = np.asarray(branch.transform_operator(raw_columns))

    assert np.allclose(transformed, np.asarray(branch.matrix) @ raw_columns)


def test_hybrid_collision_retains_the_whole_cluster_across_tiny_perturbations() -> None:
    """碰撞簇从阈值两侧逼近时都应整簇保留，不能由浮点单边决定 quotient。"""

    whitening = WhiteningEstimate.fit(
        _estimate_residuals(),
        stage=MonitorStage.ESTIMATE,
    )
    w_0 = np.asarray(whitening.operator)
    candidates = []
    for perturbation in (-1e-10, 1e-10):
        white_anchor = np.diag([np.sqrt(4.0 + perturbation), 0.25])
        candidates.append(
            PostFilterCandidate.fit(
                candidate_id=f"hybrid-{perturbation:+.1e}",
                whitening=whitening,
                anchor_response=np.linalg.solve(w_0, white_anchor),
                anchor_covariance_sqrt=np.eye(2),
                singular_value_threshold=2.0,
                stage=MonitorStage.ESTIMATE,
                minimum_projector_gap=0.0,
                spectral_cluster_tolerance=1e-8,
            )
        )

    assert all(candidate.mode is SpectralMode.HYBRID for candidate in candidates)
    assert all(candidate.selected_rank == 0 for candidate in candidates)
    assert all(candidate.retained_rank == 2 for candidate in candidates)


def test_high_dimensional_estimate_requires_dimension_aware_sample_evidence() -> None:
    """4 个窗口不能授权 100 维谱子空间，即使 Ledoit--Wolf 能给出满秩矩阵。"""

    residuals = np.arange(400, dtype=np.float64).reshape(4, 100)
    whitening = WhiteningEstimate.fit(
        residuals,
        stage=MonitorStage.ESTIMATE,
        minimum_samples=4,
    )

    assert whitening.required_sample_count == 101
    assert not whitening.stable
    assert "101" in (whitening.fallback_reason or "")


def test_whitening_json_rejects_operator_covariance_and_spectrum_tampering() -> None:
    """稳定重放必须验证正定协方差、存储谱和 ``W_0 Sigma W_0=I``。"""

    original = WhiteningEstimate.fit(
        _estimate_residuals(),
        stage=MonitorStage.ESTIMATE,
    ).to_dict()
    negative_covariance = copy.deepcopy(original)
    negative_covariance["covariance"] = [[-1.0, 0.0], [0.0, -1.0]]
    independent_operator = copy.deepcopy(original)
    independent_operator["operator"][0][0] *= 2.0
    negative_operator = copy.deepcopy(original)
    negative_operator["operator"] = (
        -np.asarray(negative_operator["operator"], dtype=np.float64)
    ).tolist()
    tiny_negative_operator = copy.deepcopy(original)
    tiny_negative_operator["covariance"] = [[1e200, 0.0], [0.0, 1e200]]
    tiny_negative_operator["operator"] = [[-1e-100, 0.0], [0.0, -1e-100]]
    tiny_negative_operator["eigenvalues"] = [1e200, 1e200]
    tiny_negative_operator["condition_number"] = 1.0
    tiny_negative_operator["effective_rank"] = 2
    below_frozen_floor = copy.deepcopy(original)
    below_frozen_floor["covariance"] = [[1e-20, 0.0], [0.0, 1e-20]]
    below_frozen_floor["operator"] = [[1e10, 0.0], [0.0, 1e10]]
    below_frozen_floor["eigenvalues"] = [1e-20, 1e-20]
    below_frozen_floor["condition_number"] = 1.0
    below_frozen_floor["effective_rank"] = 0
    false_spectrum = copy.deepcopy(original)
    false_spectrum["eigenvalues"][0] *= 2.0

    for payload in (
        negative_covariance,
        independent_operator,
        negative_operator,
        tiny_negative_operator,
        below_frozen_floor,
        false_spectrum,
    ):
        with pytest.raises(ValueError):
            WhiteningEstimate.from_dict(payload)


def test_candidate_json_rejects_injected_non_matched_extra_branches() -> None:
    """前两支路以外只能是 matched；额外 omnibus/guard 不得改变 max-score family。"""

    payload = _partial_candidate().to_dict()
    injected = copy.deepcopy(payload["branch_bank"]["branches"][0])
    injected["name"] = "injected-omnibus"
    injected["kind"] = BranchKind.OMNIBUS.value
    payload["branch_bank"]["branches"].append(injected)

    with pytest.raises(ValueError):
        PostFilterCandidate.from_dict(payload)


def test_finite_overflowing_estimate_values_fail_closed_instead_of_escaping() -> None:
    """有限但会使协方差中间运算溢出的输入必须产生可审计 guard-only 回退。"""

    residuals = np.asarray(
        [
            [1e200, -1e200],
            [-1e200, 1e200],
            [1e200, 1e200],
            [-1e200, -1e200],
            [5e199, -5e199],
        ],
        dtype=np.float64,
    )

    whitening = WhiteningEstimate.fit(
        residuals,
        stage=MonitorStage.ESTIMATE,
        minimum_samples=4,
    )
    candidate = PostFilterCandidate.fit(
        candidate_id="overflow-fallback",
        whitening=whitening,
        anchor_response=np.eye(2),
        anchor_covariance_sqrt=np.eye(2),
        singular_value_threshold=1.0,
        stage=MonitorStage.ESTIMATE,
    )

    assert not whitening.stable
    assert "failed" in (whitening.fallback_reason or "").lower()
    assert candidate.mode is SpectralMode.GUARD_ONLY


def test_finite_overflowing_anchor_spectrum_fails_closed_to_raw_guard() -> None:
    """Gram 谱超出 float64 时必须禁用 quotient，而不是让线性代数异常逃逸。"""

    whitening = WhiteningEstimate.fit(
        _estimate_residuals(),
        stage=MonitorStage.ESTIMATE,
    )

    candidate = PostFilterCandidate.fit(
        candidate_id="anchor-spectrum-overflow",
        whitening=whitening,
        anchor_response=1e200 * np.eye(2),
        anchor_covariance_sqrt=np.eye(2),
        singular_value_threshold=1.0,
        stage=MonitorStage.ESTIMATE,
    )

    assert candidate.mode is SpectralMode.GUARD_ONLY
    assert tuple(branch.name for branch in candidate.branch_bank.branches) == ("guard",)
    assert candidate.branch_bank.branch("guard").anchor_radius == pytest.approx(1e200)
    assert "spectrum" in (candidate.fallback_reason or "").lower()
