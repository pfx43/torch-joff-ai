"""P3 论文协议编排与正常数据基线的公开行为测试。

文件用途：
    验证 PCA、DAE、一步 MLP 三类正常数据基线，以及五阶段论文编排在不读取故障数据时
    能形成可冻结、可重放、可审计的最小闭环。
主要职责：
    通过公开 ``fit/score/save/load`` 和 ``PaperProtocolExperiment`` 接口检查分数、阈值、
    报警、原始索引、episode 校准与 checkpoint 重放；不测试神经网络内部层实现。
关键输入与输出：
    输入是小型确定性正常序列和 P2 ``PaperDataBundle``；输出是内存分数对象与临时目录
    中的 checkpoint、CSV 和 JSON 产物。
依赖与副作用：
    测试只使用 CPU，并仅在 pytest ``tmp_path`` 下写文件；随机神经基线显式固定种子。
重要约束：
    所有拟合与校准只可读取对应正常阶段。合成故障仅用于验证冻结门禁，任何 smoke 数值
    都不能作为论文结果；本文件也不覆盖 P4-P9 的论文方法或 P10 正式故障指标。
"""

from __future__ import annotations

from pathlib import Path

import json
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from joff.data import (
    FiveStageNormalSplitter,
    FiveStageSplitConfig,
    FitPurpose,
    PaperDataBundle,
    ProtocolAccessError,
    StageName,
)
from joff.experiments.paper_baselines import (
    BaselineScoreBatch,
    PaperBaselineConfig,
    build_paper_baseline,
    load_paper_baseline,
)
from joff.experiments.paper_protocol import (
    EpisodeMaximumCalibrator,
    MonitoringScoreScaler,
    PaperProtocolConfig,
    PaperProtocolExperiment,
    StaticThreshold,
    resolve_paper_protocol_config,
)


@pytest.mark.parametrize(
    ("config", "expected_streams"),
    [
        (
            PaperBaselineConfig(name="pca", type="pca", pca_components=2),
            {"hotelling_t2", "spe_q"},
        ),
        (
            PaperBaselineConfig(
                name="dae",
                type="dae",
                latent_dim=2,
                hidden=(6,),
                noise_std=0.05,
                max_epochs=1,
                batch_size=16,
                seed=7,
            ),
            {"reconstruction_error"},
        ),
        (
            PaperBaselineConfig(
                name="mlp",
                type="mlp",
                hidden=(6,),
                max_epochs=1,
                batch_size=16,
                seed=7,
            ),
            {"prediction_error"},
        ),
    ],
)
def test_normal_baseline_checkpoint_round_trip_reproduces_scores(
    tmp_path: Path,
    config: PaperBaselineConfig,
    expected_streams: set[str],
) -> None:
    """三类基线从 checkpoint 恢复后必须复现相同分数与原始时刻对齐。"""

    time = np.linspace(0.0, 6.0, 96)
    data = np.column_stack(
        (
            np.sin(time),
            np.cos(time),
            0.25 * time + 0.05 * np.sin(2.0 * time),
        )
    )
    raw_indices = np.arange(1000, 1000 + len(data), dtype=np.int64)
    baseline = build_paper_baseline(config)

    baseline.fit(
        data,
        raw_indices,
        checkpoint_dir=tmp_path / "trainer",
        device="cpu",
    )
    expected = baseline.score(data, raw_indices, device="cpu")
    checkpoint_path = baseline.save_checkpoint(tmp_path / f"{config.name}.pt")
    restored = load_paper_baseline(checkpoint_path, device="cpu")
    actual = restored.score(data, raw_indices, device="cpu")

    assert set(expected.streams) == expected_streams
    assert set(actual.streams) == expected_streams
    assert np.array_equal(actual.raw_indices, expected.raw_indices)
    for stream_name in expected_streams:
        assert np.allclose(
            actual.streams[stream_name],
            expected.streams[stream_name],
            rtol=0.0,
            atol=1e-7,
        )


def test_monitoring_score_scaler_uses_named_estimate_streams_without_changing_indices() -> None:
    """监测分数尺度量应逐流冻结 RMS，不能充当模型输入预处理器。"""

    batch = BaselineScoreBatch(
        raw_indices=np.array([10, 11, 12], dtype=np.int64),
        streams={
            "primary": np.array([0.0, 3.0, 4.0]),
            "constant_zero": np.zeros(3),
        },
    )

    scaler = MonitoringScoreScaler.fit(batch)
    transformed = scaler.transform(batch)

    assert np.array_equal(transformed.raw_indices, batch.raw_indices)
    assert scaler.scales["primary"] == pytest.approx(np.sqrt(25.0 / 3.0))
    assert scaler.scales["constant_zero"] == 1.0
    assert transformed.streams["primary"][-1] == pytest.approx(
        4.0 / np.sqrt(25.0 / 3.0)
    )
    assert scaler.manifest()["scope"] == "monitoring_scores_only"


def test_episode_maximum_calibrator_marks_static_threshold_and_infinite_resolution() -> None:
    """完整 episode 分位应使用有限样本秩，分辨率不足时返回正无穷。"""

    normal = np.arange(1200, dtype=float).reshape(600, 2)
    split = FiveStageNormalSplitter(
        FiveStageSplitConfig(
            history_length=2,
            max_rollout=1,
            episode_length=8,
            target_risk_level=0.5,
        )
    ).split(normal)
    stage = split.stage(StageName.DETECTION_CALIBRATION)
    raw_chunks: list[np.ndarray] = []
    score_chunks: list[np.ndarray] = []
    stage_start = stage.prepared_row_indices[0]
    for episode_number, (start, stop) in enumerate(
        stage.prepared_episode_ranges,
        start=1,
    ):
        local_start = start - stage_start
        local_stop = stop - stage_start
        episode_raw = np.asarray(stage.raw_indices[local_start:local_stop], dtype=np.int64)
        raw_chunks.append(episode_raw)
        score_chunks.append(np.full(len(episode_raw), float(episode_number)))
    scores = BaselineScoreBatch(
        raw_indices=np.concatenate(raw_chunks),
        streams={"score": np.concatenate(score_chunks)},
    )

    finite = EpisodeMaximumCalibrator(risk_level=0.5).fit(scores, stage)
    unresolved = EpisodeMaximumCalibrator(risk_level=0.01).fit(scores, stage)

    episode_count = len(stage.prepared_episode_ranges)
    expected_rank = int(np.ceil((episode_count + 1) * 0.5))
    assert finite.thresholds["score"].value == float(expected_rank)
    assert finite.thresholds["score"].dynamic is False
    assert finite.thresholds["score"].kind == "static_episode_quantile"
    assert len(finite.episodes) == episode_count
    assert np.isinf(unresolved.thresholds["score"].value)
    assert unresolved.thresholds["score"].finite is False


def test_static_threshold_rejects_negative_infinity_for_unresolved_resolution() -> None:
    """分辨率不足只能用正无穷禁用决定，负无穷会错误地让所有分数报警。"""

    with pytest.raises(ValueError, match="positive infinity"):
        StaticThreshold(
            stream_name="score",
            risk_level=0.01,
            episode_count=19,
            rank=20,
            value=-np.inf,
            finite=False,
        )


def _paper_protocol_config(tmp_path: Path) -> PaperProtocolConfig:
    """构造覆盖三类基线、只运行一个 epoch 的确定性 CPU 配置。"""

    return PaperProtocolConfig(
        artifact_root=tmp_path,
        run_name="paper-p3-smoke",
        mode="development",
        device="cpu",
        detection_risk_level=0.5,
        attribution_risk_level=0.5,
        baselines=(
            PaperBaselineConfig(name="pca", type="pca", pca_components=2),
            PaperBaselineConfig(
                name="dae",
                type="dae",
                latent_dim=2,
                hidden=(8,),
                noise_std=0.05,
                max_epochs=1,
                batch_size=32,
                seed=11,
            ),
            PaperBaselineConfig(
                name="mlp",
                type="mlp",
                hidden=(8,),
                max_epochs=1,
                batch_size=32,
                seed=11,
            ),
        ),
    )


def test_paper_protocol_config_is_strict_and_has_stable_hash(tmp_path: Path) -> None:
    """论文协议配置应拒绝未知字段，并为相同解析值生成稳定 16 位 hash。"""

    config = _paper_protocol_config(tmp_path)
    first = resolve_paper_protocol_config(config)
    second = resolve_paper_protocol_config(config.model_dump(mode="json"))

    assert len(first.config_hash) == 16
    assert first.config_hash == second.config_hash
    assert first.resolved_config == second.resolved_config
    assert first.provenance
    with pytest.raises(ValidationError, match="extra_forbidden"):
        PaperProtocolConfig.model_validate(
            {
                **config.model_dump(mode="json"),
                "dynamic_threshold": True,
            }
        )
    for required_field in (
        "mode",
        "detection_risk_level",
        "attribution_risk_level",
        "baselines",
    ):
        assert required_field in PaperProtocolConfig.model_fields
        assert PaperProtocolConfig.model_fields[required_field].is_required()


def test_dae_baseline_requires_explicit_positive_training_noise() -> None:
    """DAE 必须显式加入正训练噪声，零噪声模型只能称为普通 AE。"""

    with pytest.raises(ValidationError, match="noise_std"):
        PaperBaselineConfig(
            name="dae",
            type="dae",
            latent_dim=2,
            noise_std=0.0,
        )


def test_paper_protocol_runs_three_normal_baselines_and_saves_replay_artifacts(
    tmp_path: Path,
) -> None:
    """无故障数据时应完成三类基线、两次校准、冻结正常重放和协议冻结。"""

    time = np.linspace(0.0, 24.0, 800)
    normal = np.column_stack(
        (
            np.sin(time),
            np.cos(0.5 * time),
            0.03 * time + 0.1 * np.sin(0.25 * time),
        )
    )
    raw_indices = np.arange(20_000, 20_000 + len(normal), dtype=np.int64)
    bundle = PaperDataBundle(
        normal,
        normal_raw_indices=raw_indices,
        normal_source_hash="1" * 64,
        config=FiveStageSplitConfig(
            history_length=2,
            max_rollout=1,
            episode_length=8,
            target_risk_level=0.5,
        ),
    )
    experiment = PaperProtocolExperiment(
        bundle,
        _paper_protocol_config(tmp_path),
    )

    with pytest.raises(ProtocolAccessError, match="normal protocol run"):
        experiment.request_frozen_fault_test()
    result = experiment.run_normal()

    assert bundle.protocol_frozen is True
    assert result.run_dir == (tmp_path / "paper-p3-smoke").resolve()
    assert set(result.baseline_checkpoint_paths) == {"pca", "dae", "mlp"}
    assert set(result.score_paths) == {"pca", "dae", "mlp"}
    assert set(result.replay_paths) == {"pca", "dae", "mlp"}
    assert set(result.calibration_paths["pca"]) == {
        "detection_calibration",
        "attribution_calibration",
    }
    identity = json.loads(
        (result.run_dir / "config_identity.json").read_text(encoding="utf-8")
    )
    assert len(identity["git_commit"]) == 40
    assert set(identity["git_commit"]) <= set("0123456789abcdef")
    assert identity["mode"] == "development"
    assert identity["data_hash"] == bundle.split_result.data_hash
    assert identity["split_hash"] == bundle.split_result.split_hash
    assert identity["source_hash"] == bundle.split_result.source_hash
    assert identity["stage_window_hashes"] == {
        stage.value: bundle.split_result.stage(stage).window_hash
        for stage in StageName
    }
    for baseline_name, replay_path in result.replay_paths.items():
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        assert replay["baseline"] == baseline_name
        assert replay["matches"] is True
        score_table = pd.read_csv(result.score_paths[baseline_name])
        assert {
            "baseline",
            "stage",
            "raw_index",
            "score_name",
            "score",
            "threshold",
            "alarm",
            "threshold_kind",
            "dynamic_threshold",
        } <= set(score_table.columns)
        assert set(score_table["stage"]) == {
            "detection_calibration",
            "attribution_calibration",
            "frozen_normal_test",
        }
        assert not score_table["dynamic_threshold"].astype(bool).any()
        assert set(score_table["threshold_kind"]) == {"static_episode_quantile"}
        if baseline_name == "mlp":
            for stage_name in (
                StageName.DETECTION_CALIBRATION,
                StageName.ATTRIBUTION_CALIBRATION,
            ):
                stage_slice = bundle.split_result.stage(stage_name)
                stage_offset = stage_slice.prepared_row_indices[0]
                episode_starts = {
                    int(stage_slice.raw_indices[start - stage_offset])
                    for start, _ in stage_slice.prepared_episode_ranges
                }
                scored_indices = set(
                    score_table.loc[
                        score_table["stage"] == stage_name.value,
                        "raw_index",
                    ].astype(int)
                )
                assert episode_starts.isdisjoint(scored_indices)
        for calibration_path in result.calibration_paths[baseline_name].values():
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            episode_hashes = [
                episode["episode_hash"] for episode in calibration["episodes"]
            ]
            assert episode_hashes
            assert all(
                len(value) == 64 and set(value) <= set("0123456789abcdef")
                for value in episode_hashes
            )
            assert len(set(episode_hashes)) == len(episode_hashes)

    ledger = bundle.fit_access_ledger.manifest()
    records = ledger["records"]
    expected_stages = {
        FitPurpose.MODEL_PARAMETERS.value: {"train"},
        FitPurpose.MONITORING_SCORE_SCALER.value: {"estimate"},
        FitPurpose.DETECTION_QUANTILE.value: {"detection_calibration"},
        FitPurpose.ATTRIBUTION_QUANTILE.value: {"attribution_calibration"},
        FitPurpose.FROZEN_NORMAL_DIAGNOSTIC.value: {"frozen_normal_test"},
    }
    assert len(records) == 15
    assert all(record["frozen"] for record in records)
    for record in records:
        assert set(record["stages"]) == expected_stages[record["purpose"]]
    assert all(path.exists() for path in result.protocol_paths.values())
    with pytest.raises(ProtocolAccessError, match="mode='frozen'"):
        experiment.request_frozen_fault_test()


def test_paper_protocol_public_api_is_exported_from_experiments_package() -> None:
    """P3 调用方应从 experiments 包获得配置、编排器和基线构建接口。"""

    import joff.experiments as experiments

    assert experiments.PaperProtocolConfig is PaperProtocolConfig
    assert experiments.PaperProtocolExperiment is PaperProtocolExperiment
    assert experiments.PaperBaselineConfig is PaperBaselineConfig
    assert experiments.build_paper_baseline is build_paper_baseline
