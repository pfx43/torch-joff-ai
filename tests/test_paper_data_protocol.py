"""
论文五阶段正常数据协议的 P2 行为测试。

文件用途：
    通过 paper_protocol 公共对象验证五段正常数据、隔离带、窗口依赖、哈希、访问账本
    和冻结故障测试门禁，防止训练、校准和最终测试之间发生数据泄漏。
主要职责：
    固定 55/15/10/10/10 分配、确定性 manifest、校准 episode 独立性和协议冻结规则；
    不训练论文模型、不计算真实故障指标，也不核实 CSTR 数据许可。
关键输入与输出：
    输入为小型确定性 NumPy 数组和显式 FiveStageSplitConfig；输出为对 StageSlice、
    FiveStageSplitResult、FitAccessLedger 和 PaperDataBundle 的公开行为断言。
依赖与副作用：
    主要测试只在内存运行；产物测试仅写入 pytest 临时目录，不访问网络或真实故障数据。
重要约束：
    所有训练与校准输入均为合成正常数据；故障数组只用于验证门禁，不能据此形成论文结论。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from joff.data import (
    DATASET_REGISTRY,
    FaultLicenseStatus,
    FitAccessLedger,
    FitPurpose,
    FiveStageNormalSplitter,
    FiveStageSplitConfig,
    PaperDataBundle,
    ProtocolAccessError,
    StageName,
)

_ARTIFACT_HASH = "a" * 64
_NORMAL_SOURCE_HASH = "b" * 64
_FAULT_SOURCE_HASH = "c" * 64


def _complete_minimal_protocol(bundle: PaperDataBundle) -> None:
    """按模型、估计、两次校准和冻结正常检查的顺序完成最小协议。

    参数：
        bundle: 尚未冻结且未访问故障范围的论文数据 bundle。
    返回：
        无。
    异常：
        任一阶段违反访问顺序、对象重复或产物 hash 非法时传播协议异常。
    副作用：
        在 bundle 账本中登记并冻结五个最小对象，但不访问故障数组、不写文件。
    """

    bundle.data_for_fit("paper_model", FitPurpose.MODEL_PARAMETERS)
    bundle.fit_access_ledger.freeze_record("paper_model", _ARTIFACT_HASH)
    bundle.data_for_fit("monitoring_score_scaler", FitPurpose.MONITORING_SCORE_SCALER)
    bundle.fit_access_ledger.freeze_record("monitoring_score_scaler", _ARTIFACT_HASH)
    bundle.data_for_fit("q_det", FitPurpose.DETECTION_QUANTILE)
    bundle.fit_access_ledger.freeze_record("q_det", _ARTIFACT_HASH)
    bundle.data_for_fit("q_attr", FitPurpose.ATTRIBUTION_QUANTILE)
    bundle.fit_access_ledger.freeze_record("q_attr", _ARTIFACT_HASH)
    bundle.data_for_fit("normal_diagnostic", FitPurpose.FROZEN_NORMAL_DIAGNOSTIC)
    bundle.fit_access_ledger.freeze_record("normal_diagnostic", _ARTIFACT_HASH)


def test_five_stage_split_allocates_ratios_after_removing_four_gaps() -> None:
    """1000 行扣除四个 10 行隔离带后，应严格分成 528/144/96/96/96。"""

    normal = np.arange(2000, dtype=float).reshape(1000, 2)
    config = FiveStageSplitConfig(
        history_length=3,
        max_rollout=2,
        minimum_gap=10,
        episode_length=24,
        target_risk_level=0.5,
    )

    result = FiveStageNormalSplitter(config).split(normal)

    assert result.effective_gap == 10
    assert [result.stage(stage).row_count for stage in StageName] == [528, 144, 96, 96, 96]
    assert result.prepared_gap_ranges == (
        (528, 538),
        (682, 692),
        (788, 798),
        (894, 904),
    )
    assert result.stage(StageName.TRAIN).prepared_row_indices == tuple(range(528))
    assert result.stage(StageName.FROZEN_NORMAL_TEST).prepared_row_indices == tuple(
        range(904, 1000)
    )


def test_fit_access_ledger_rejects_stage_use_outside_each_fitting_purpose() -> None:
    """模型、检测分位和归因分位必须分别只访问 train、检测校准和归因校准。"""

    normal = np.arange(2400, dtype=float).reshape(1200, 2)
    result = FiveStageNormalSplitter(
        FiveStageSplitConfig(
            history_length=3,
            max_rollout=2,
            episode_length=20,
            target_risk_level=0.5,
        )
    ).split(normal)
    ledger = FitAccessLedger(result)

    model_record = ledger.record_fit(
        "paper_model",
        FitPurpose.MODEL_PARAMETERS,
        [StageName.TRAIN],
    )
    with pytest.raises(ValueError, match="Legal stages.*estimate"):
        ledger.record_fit(
            "scaler_wrong_stage",
            FitPurpose.MONITORING_SCORE_SCALER,
            [StageName.DETECTION_CALIBRATION],
        )
    detection_record = ledger.record_fit(
        "q_det",
        FitPurpose.DETECTION_QUANTILE,
        [StageName.DETECTION_CALIBRATION],
    )
    ledger.freeze_record("q_det", _ARTIFACT_HASH)
    attribution_record = ledger.record_fit(
        "q_attr",
        FitPurpose.ATTRIBUTION_QUANTILE,
        [StageName.ATTRIBUTION_CALIBRATION],
    )

    assert model_record.stage_hashes == {
        "train": result.stage(StageName.TRAIN).data_hash,
    }
    assert detection_record.stages != attribution_record.stages


def test_fault_test_requires_frozen_protocol_and_verified_license() -> None:
    """真实故障范围在协议未冻结或许可未核实时必须拒绝返回数组。"""

    normal = np.arange(3000, dtype=float).reshape(1500, 2)
    fault = np.arange(400, dtype=float).reshape(200, 2)
    config = FiveStageSplitConfig(
        history_length=3,
        max_rollout=2,
        episode_length=20,
        target_risk_level=0.5,
    )
    bundle = PaperDataBundle(
        normal,
        config=config,
        frozen_fault_test=fault,
        normal_source_hash=_NORMAL_SOURCE_HASH,
        fault_source_hash=_FAULT_SOURCE_HASH,
        fault_license_status=FaultLicenseStatus.VERIFIED,
    )

    with pytest.raises(ProtocolAccessError, match="not frozen"):
        bundle.request_frozen_fault_test()
    with pytest.raises(ValueError, match="split hash"):
        bundle.freeze_protocol("wrong-split-hash")

    _complete_minimal_protocol(bundle)
    bundle.freeze_protocol(bundle.split_result.split_hash)
    returned = bundle.request_frozen_fault_test()
    assert np.array_equal(returned, fault)
    returned[0, 0] = -999.0
    assert bundle.request_frozen_fault_test()[0, 0] == fault[0, 0]

    unlicensed = PaperDataBundle(
        normal,
        config=config,
        frozen_fault_test=fault,
        normal_source_hash=_NORMAL_SOURCE_HASH,
        fault_source_hash=_FAULT_SOURCE_HASH,
        fault_license_status=FaultLicenseStatus.TO_VERIFY,
    )
    _complete_minimal_protocol(unlicensed)
    unlicensed.freeze_protocol(unlicensed.split_result.split_hash)
    with pytest.raises(ProtocolAccessError, match="license"):
        unlicensed.request_frozen_fault_test()


def test_all_stage_data_access_is_ledgered_and_calibration_order_is_enforced() -> None:
    """正常阶段数组不得绕过账本，归因和正常测试必须等待上游分位冻结。"""

    normal = np.arange(4800, dtype=float).reshape(2400, 2)
    bundle = PaperDataBundle(
        normal,
        config=FiveStageSplitConfig(
            history_length=3,
            max_rollout=2,
            episode_length=20,
            target_risk_level=0.1,
        ),
    )

    assert not hasattr(bundle, "stage_data")
    with pytest.raises(ProtocolAccessError, match="frozen detection quantile"):
        bundle.data_for_fit("q_attr_too_early", FitPurpose.ATTRIBUTION_QUANTILE)

    bundle.data_for_fit("q_det", FitPurpose.DETECTION_QUANTILE)
    with pytest.raises(ProtocolAccessError, match="frozen detection quantile"):
        bundle.data_for_fit("q_attr_still_early", FitPurpose.ATTRIBUTION_QUANTILE)
    bundle.fit_access_ledger.freeze_record("q_det", _ARTIFACT_HASH)

    bundle.data_for_fit("q_attr", FitPurpose.ATTRIBUTION_QUANTILE)
    with pytest.raises(ProtocolAccessError, match="frozen attribution quantile"):
        bundle.data_for_fit("normal_too_early", FitPurpose.FROZEN_NORMAL_DIAGNOSTIC)
    bundle.fit_access_ledger.freeze_record("q_attr", _ARTIFACT_HASH)

    bundle.data_for_fit("normal_diagnostic", FitPurpose.FROZEN_NORMAL_DIAGNOSTIC)
    bundle.fit_access_ledger.freeze_record("normal_diagnostic", _ARTIFACT_HASH)
    with pytest.raises(ProtocolAccessError, match="frozen-normal access has closed"):
        bundle.data_for_fit("late_model", FitPurpose.MODEL_PARAMETERS)

    records = bundle.fit_access_ledger.manifest()["records"]
    assert [item["object_id"] for item in records] == [
        "normal_diagnostic",
        "q_attr",
        "q_det",
    ]
    assert all(item["frozen"] for item in records)


def test_fault_gate_requires_complete_ledger_and_both_source_hashes() -> None:
    """正式故障范围必须同时具备完整冻结账本、正常来源 hash 和故障来源 hash。"""

    normal = np.arange(4800, dtype=float).reshape(2400, 2)
    fault = np.arange(400, dtype=float).reshape(200, 2)
    config = FiveStageSplitConfig(
        history_length=3,
        max_rollout=2,
        episode_length=20,
        target_risk_level=0.1,
    )
    missing_hashes = PaperDataBundle(
        normal,
        config=config,
        frozen_fault_test=fault,
        fault_license_status=FaultLicenseStatus.VERIFIED,
    )
    _complete_minimal_protocol(missing_hashes)
    with pytest.raises(ProtocolAccessError, match="source SHA-256"):
        missing_hashes.freeze_protocol(missing_hashes.split_result.split_hash)

    incomplete_ledger = PaperDataBundle(
        normal,
        config=config,
        frozen_fault_test=fault,
        normal_source_hash=_NORMAL_SOURCE_HASH,
        fault_source_hash=_FAULT_SOURCE_HASH,
        fault_license_status=FaultLicenseStatus.VERIFIED,
    )
    with pytest.raises(ProtocolAccessError, match="access ledger is incomplete"):
        incomplete_ledger.freeze_protocol(incomplete_ledger.split_result.split_hash)


def test_split_is_deterministic_and_all_dependency_scopes_are_disjoint() -> None:
    """相同配置应产生同一 manifest，且行、窗口依赖和校准 episode 均不得复用。"""

    normal = np.arange(6000, dtype=float).reshape(2000, 3)
    config = FiveStageSplitConfig(
        history_length=4,
        max_rollout=3,
        stacked_window=4,
        mask_recompute_span=5,
        episode_length=20,
        target_risk_level=0.1,
        seed=17,
    )

    first = FiveStageNormalSplitter(config).split(normal, source_hash=_NORMAL_SOURCE_HASH)
    second = FiveStageNormalSplitter(config).split(normal, source_hash=_NORMAL_SOURCE_HASH)

    assert config.dependency_span == 12
    assert first.effective_gap == 12
    assert first.manifest() == second.manifest()
    assert first.split_hash == second.split_hash

    row_sets: list[set[int]] = []
    dependency_sets: list[set[int]] = []
    for stage in StageName:
        stage_slice = first.stage(stage)
        row_sets.append(set(stage_slice.prepared_row_indices))
        dependency_sets.append(
            {
                row
                for start in stage_slice.prepared_window_starts
                for row in stage_slice.dependency_prepared_rows(start)
            }
        )
    for left in range(len(StageName)):
        for right in range(left + 1, len(StageName)):
            assert row_sets[left].isdisjoint(row_sets[right])
            assert dependency_sets[left].isdisjoint(dependency_sets[right])

    detection_episodes = {
        row
        for start, stop in first.stage(
            StageName.DETECTION_CALIBRATION
        ).prepared_episode_ranges
        for row in range(start, stop)
    }
    attribution_episodes = {
        row
        for start, stop in first.stage(
            StageName.ATTRIBUTION_CALIBRATION
        ).prepared_episode_ranges
        for row in range(start, stop)
    }
    assert detection_episodes.isdisjoint(attribution_episodes)
    for stage_name in (
        StageName.DETECTION_CALIBRATION,
        StageName.ATTRIBUTION_CALIBRATION,
    ):
        stage_slice = first.stage(stage_name)
        episode_sets = [
            set(range(start, stop))
            for start, stop in stage_slice.prepared_episode_ranges
        ]
        for window_start in stage_slice.prepared_window_starts:
            dependencies = set(stage_slice.dependency_prepared_rows(window_start))
            assert any(dependencies.issubset(episode) for episode in episode_sets)
    with pytest.raises(ValueError, match="Legal options.*frozen_normal_test"):
        first.stage("calibration")


def test_calibration_or_fault_changes_cannot_change_train_or_estimate_hashes() -> None:
    """只改检测校准或故障范围时，模型与估计阶段的数据内容和 hash 必须保持不变。"""

    normal = np.arange(4000, dtype=float).reshape(2000, 2)
    config = FiveStageSplitConfig(
        history_length=3,
        max_rollout=2,
        episode_length=20,
        target_risk_level=0.1,
    )
    baseline = FiveStageNormalSplitter(config).split(normal)
    modified = normal.copy()
    detection_rows = baseline.stage(
        StageName.DETECTION_CALIBRATION
    ).prepared_row_indices
    modified[np.asarray(detection_rows, dtype=int)] += 10_000.0
    changed = FiveStageNormalSplitter(config).split(modified)

    assert baseline.stage(StageName.TRAIN).data_hash == changed.stage(StageName.TRAIN).data_hash
    assert baseline.stage(StageName.ESTIMATE).data_hash == changed.stage(StageName.ESTIMATE).data_hash
    assert baseline.stage(StageName.DETECTION_CALIBRATION).data_hash != changed.stage(
        StageName.DETECTION_CALIBRATION
    ).data_hash
    assert baseline.data_hash != changed.data_hash
    assert baseline.split_hash != changed.split_hash

    first_bundle = PaperDataBundle(
        normal,
        config=config,
        frozen_fault_test=np.ones((100, 2)),
    )
    second_bundle = PaperDataBundle(
        normal,
        config=config,
        frozen_fault_test=np.full((100, 2), 99.0),
    )
    assert first_bundle.split_result.split_hash == second_bundle.split_result.split_hash
    assert np.array_equal(
        first_bundle.data_for_fit("first_model", FitPurpose.MODEL_PARAMETERS),
        second_bundle.data_for_fit("second_model", FitPurpose.MODEL_PARAMETERS),
    )


def test_split_refuses_insufficient_complete_calibration_episodes() -> None:
    """校准 episode 数不足目标风险分辨率时必须停止，不能缩短隔离带。"""

    normal = np.arange(600, dtype=float).reshape(300, 2)
    config = FiveStageSplitConfig(
        history_length=3,
        max_rollout=2,
        episode_length=100,
        target_risk_level=0.1,
    )

    with pytest.raises(ValueError, match="requires at least 9.*will not be shortened"):
        FiveStageNormalSplitter(config).split(normal)


def test_bundle_saves_split_indices_hashes_and_fit_access_ledger(tmp_path) -> None:
    """显式保存应生成可独立审计的 split、ledger 和 bundle 三份 JSON。"""

    normal = np.arange(4800, dtype=float).reshape(2400, 2)
    bundle = PaperDataBundle(
        normal,
        config=FiveStageSplitConfig(
            history_length=3,
            max_rollout=2,
            episode_length=20,
            target_risk_level=0.1,
        ),
        normal_source_hash=_NORMAL_SOURCE_HASH,
    )
    bundle.data_for_fit("paper_model", FitPurpose.MODEL_PARAMETERS)
    bundle.data_for_fit("monitoring_score_scaler", FitPurpose.MONITORING_SCORE_SCALER)
    bundle.data_for_fit("q_det", FitPurpose.DETECTION_QUANTILE)

    paths = bundle.save_protocol_artifacts(tmp_path / "protocol")

    assert set(paths) == {"split_manifest", "fit_access_ledger", "paper_data_bundle"}
    split_manifest = json.loads(paths["split_manifest"].read_text(encoding="utf-8"))
    ledger_manifest = json.loads(paths["fit_access_ledger"].read_text(encoding="utf-8"))
    bundle_manifest = json.loads(paths["paper_data_bundle"].read_text(encoding="utf-8"))
    assert split_manifest["split_hash"] == bundle.split_result.split_hash
    assert split_manifest["stages"]["train"]["prepared_row_indices"]
    assert split_manifest["stages"]["train"]["prepared_window_starts"]
    assert split_manifest["prepared_gap_ranges"]
    assert split_manifest["source_hash"] == _NORMAL_SOURCE_HASH
    assert [item["object_id"] for item in ledger_manifest["records"]] == [
        "monitoring_score_scaler",
        "paper_model",
        "q_det",
    ]
    assert bundle_manifest["frozen_fault_test"]["configured"] is False


def test_split_preserves_raw_indices_and_drops_windows_crossing_removed_rows() -> None:
    """显式 raw_index 有跳号时，任何依赖跨越缺口的窗口都不得保留。"""

    normal = np.arange(2000, dtype=float).reshape(1000, 2)
    raw_indices = np.arange(1000)
    raw_indices[100:] += 1
    result = FiveStageNormalSplitter(
        FiveStageSplitConfig(
            history_length=3,
            max_rollout=2,
            minimum_gap=10,
            episode_length=24,
            target_risk_level=0.5,
        )
    ).split(normal, raw_indices=raw_indices)

    train = result.stage(StageName.TRAIN)
    assert train.raw_indices[99:102] == (99, 101, 102)
    assert 96 not in train.prepared_window_starts
    for window_start in train.prepared_window_starts:
        dependencies = train.dependency_raw_rows(window_start)
        assert np.diff(dependencies).tolist() == [1] * (train.dependency_span - 1)


def test_bundle_from_canonical_uses_only_normal_split_and_physical_schema_roles() -> None:
    """P1 CanonicalDataset 入口必须排除标签/追溯列，且不自动接入其中的 test split。"""

    canonical = DATASET_REGISTRY.resolve("cstr_closed_loop_fd").read(root=None)
    bundle = PaperDataBundle.from_canonical(
        canonical,
        config=FiveStageSplitConfig(
            history_length=3,
            max_rollout=2,
            episode_length=5,
            target_risk_level=0.5,
        ),
    )

    train_data = bundle.data_for_fit("canonical_model", FitPurpose.MODEL_PARAMETERS)
    assert train_data.shape[1] == 7
    assert bundle.split_result.stage(StageName.TRAIN).raw_indices[0] == 0
    assert bundle.manifest()["frozen_fault_test"]["configured"] is False
