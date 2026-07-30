"""P10 冻结 CSTR 评价的公开协议测试。

文件用途：
    通过公开 API 验证冻结 manifest、一次性故障访问和机器可读评价产物，防止正式故障
    数据被用于协议选择或同一 evaluation ID 被重复运行。
主要职责：
    覆盖 manifest 的深层不可变性、往返重放、防篡改、八 episode 完整性、逐时刻集合
    语义、持久化一次性 fault authorization、CLI readiness、许可证证据和 artifact
    integrity。
关键输入与输出：
    测试使用固定 SHA-256、最小五段/校准/认证摘要和 pytest 临时目录；输出只是在临时
    目录中的 manifest 与评价产物，不读取仓库真实 CSTR 故障文件。
依赖与副作用：
    依赖 pytest、NumPy、SciPy 和 ``joff.experiments`` 公共接口；只写 ``tmp_path``，
    不访问网络、不训练模型、不修改全局随机状态。
重要约束：
    所有 fixture 都是合成审计证据，不能作为论文数值结果；预期值来自规格中的字段与
    手工固定字面量，不能通过重复实现生产 hash 逻辑形成同义反复测试。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any, Literal, cast

import csv
import hashlib
import json
import math
import subprocess

import numpy as np
import pytest
import torch
import yaml
from scipy.io import savemat  # type: ignore[import-untyped]

from joff.experiments import (
    CSTRClosedLoopEpisodeLoader,
    FrozenEpisodeInput,
    FrozenEvaluationAlreadyClaimedError,
    FrozenEvaluationClaim,
    FrozenEvaluationEntryConfig,
    FrozenEvaluationWorkflow,
    FrozenFaultEpisode,
    FrozenFaultEpisodeManifest,
    FrozenNormalArtifactBundle,
    LazyFrozenCSTRFaultSource,
    FrozenPointwiseOutput,
    FrozenProtocolIntegrityError,
    FrozenProtocolManifest,
    FrozenRiskCalibration,
    FrozenRuntimeEpisodeEvaluation,
    FrozenRuntimePointwiseOutput,
    ManifestBoundCSTRFaultSource,
    freeze_cstr_protocol_from_artifacts,
    inspect_closed_loop_cstr_archive,
    resolve_frozen_evaluation_config,
    verify_frozen_evaluation_artifacts,
)
from joff.experiments import frozen_cli as frozen_cli_module
import joff.experiments.paper_freeze as paper_freeze_module
from joff.data import (
    FaultLicenseStatus,
    FitPurpose,
    FiveStageSplitConfig,
    PaperDataBundle,
    ProtocolAccessError,
)
from joff.core.config import StrictConfig
from joff.core.factory import (
    build_evaluator,
    build_model,
    register_evaluator,
)
from joff.experiments.frozen_cli import (
    ProtectedKoopmanTSCheckpointDriver,
    main as frozen_cli_main,
)
from joff.experiments.frozen_artifacts import (
    validate_protected_evaluator_artifact_bindings,
)
from joff.experiments.paper_environment import current_clean_paper_git_commit
from joff.experiments.paper_development import run_cstr_normal_development


_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64
_SYNTHETIC_CHECKPOINT_HASH = (
    "a42a2335521b58b64ab42e9f6baad712d717977ffb3ced24b3b11ab018c63e43"
)
_ROOT = Path(__file__).resolve().parents[1]


def _write_verified_dataset_card_fixture(
    card_path: Path,
    *,
    data_root: Path,
    normal_file: str,
    fault_file: str,
    normal_sha256: str,
    fault_sha256: str,
) -> tuple[Path, str]:
    """写入只供合成测试使用的完整数据卡证据链。

    参数：
        card_path/data_root/normal_file/fault_file: 临时卡片及其声明的 synthetic MAT 路径。
        normal_sha256/fault_sha256: 由测试独立创建的 fixture 文件身份。
    返回：
        卡片路径及其 SHA-256，供严格入口配置绑定。
    异常：
        临时目录不可写或 YAML 序列化失败时传播原异常。
    副作用：
        在 ``tmp_path`` 下创建生成记录、许可说明和数据卡；内容明确标记 test-only，绝不
        作为真实 CSTR 数据许可证据或论文结果。
    """

    card_path.parent.mkdir(parents=True, exist_ok=True)
    generation_record = card_path.with_suffix(".generation.txt")
    license_evidence = card_path.with_suffix(".license.txt")
    generation_record.write_text(
        "test-only synthetic MAT generation record\n",
        encoding="utf-8",
    )
    license_evidence.write_text(
        "test-only permission for synthetic pytest fixture\n",
        encoding="utf-8",
    )
    card = {
        "name": "cstr_closed_loop_fd",
        "access": {
            "tag": "test",
            "disclosure": "synthetic_fixture",
            "license": "test-only-fixture-license",
            "license_status": "verified",
        },
        "files": {
            "root": str(data_root),
            "train": normal_file,
            "test": fault_file,
        },
        "source": {
            "local_mat": {
                "provenance_status": "verified",
                "license_status": "verified",
                "normal_sha256": normal_sha256,
                "fault_sha256": fault_sha256,
                "generation_record": generation_record.name,
                "generation_record_sha256": hashlib.sha256(
                    generation_record.read_bytes()
                ).hexdigest(),
                "license_evidence": license_evidence.name,
                "license_evidence_sha256": hashlib.sha256(
                    license_evidence.read_bytes()
                ).hexdigest(),
            }
        },
    }
    card_path.write_text(
        yaml.safe_dump(card, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return card_path, hashlib.sha256(card_path.read_bytes()).hexdigest()


def _risk_calibration(
    name: Literal["detection", "attribution"],
    *,
    quantile: float,
    source_hash: str,
    stage_manifest: Mapping[str, Any],
) -> FrozenRiskCalibration:
    """从 P2 完整 episode ranges 构造有限样本校准摘要。

    episode ID 直接编码半开区间，便于 manifest 不读取数组也能独立重放数量和身份。
    """

    episode_ranges = tuple(
        (int(start), int(end))
        for start, end in stage_manifest["prepared_episode_ranges"]
    )
    episode_count = len(episode_ranges)
    return FrozenRiskCalibration(
        name=name,
        requested_risk=0.3 if name == "detection" else 0.2,
        quantile=quantile,
        episode_count=episode_count,
        score_count=int(stage_manifest["window_count"]),
        attainable_risk_resolution=1.0 / (episode_count + 1),
        status="finite",
        source_hash=source_hash,
        episode_ids=tuple(
            f"{name}-{start}-{end}"
            for start, end in episode_ranges
        ),
    )


def _fault_episode_manifest() -> tuple[FrozenFaultEpisodeManifest, ...]:
    """返回八个小型合成 CSTR fault episode 的冻结身份。

    smoke 使用 onset=2 以缩短 CPU 路径；正式闭环 CSTR 配置会把同一字段冻结为 200。
    """

    families = (
        "process",
        "process",
        "actuator",
        "actuator",
        "actuator",
        "sensor",
        "sensor",
        "sensor",
    )
    return tuple(
        FrozenFaultEpisodeManifest(
            episode_id=f"fault-{fault_id}",
            fault_id=fault_id,
            fault_family=cast(
                Literal["process", "actuator", "sensor"],
                families[fault_id - 1],
            ),
            onset=2,
            row_count=5,
            raw_index_start=0,
            raw_index_end=4,
            source_hash=_B,
        )
        for fault_id in range(1, 9)
    )


def _normal_protocol_evidence(
    *,
    normal_source_hash: str,
    fault_source_hash: str,
    artifact_hashes: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """通过 P2 公开对象生成完整五段 manifest 和冻结拟合账本，禁止手写空 stage。"""

    values = np.arange(800 * 2, dtype=float).reshape(800, 2) / 100.0
    bundle = PaperDataBundle(
        values,
        config=FiveStageSplitConfig(
            history_length=2,
            max_rollout=1,
            stacked_window=1,
            episode_length=4,
            target_risk_level=0.2,
            seed=19,
        ),
        normal_source_hash=normal_source_hash,
        fault_source_hash=fault_source_hash,
        fault_license_status=FaultLicenseStatus.TO_VERIFY,
    )
    _complete_bundle_protocol(bundle, artifact_hashes=artifact_hashes)
    return (
        bundle.split_result.manifest(),
        bundle.fit_access_ledger.manifest(),
    )


def _normal_method_config_fixture() -> dict[str, Any]:
    """返回字段完整的小型 P4 配置；只用于验证 formal 产物合同，不代表论文超参数。"""

    return {
        "model": {
            "type": "protected_koopman_ts",
            "control_dim": 2,
            "measurement_dim": 4,
            "exogenous_dim": 1,
            "history_length": 3,
            "latent_dim": 4,
            "context_dim": 3,
            "max_rollout": 3,
            "horizon_seed": 59,
            "attention": {
                "embed_dim": 8,
                "num_heads": 2,
                "dropout": 0.0,
            },
            "channel_mask": {
                "all_pass_probability": 0.5,
                "single_channel_probability": 0.25,
                "independent_drop_probability": 0.4,
                "seed": 31,
            },
            "fuzzy": {
                "rule_count": 2,
                "premise_dim": 3,
                "premise_hidden_dim": 5,
                "metric_eigenvalue_min": 0.1,
                "metric_eigenvalue_max": 2.0,
                "spectral_cap": 1.1,
            },
            "loss": {
                "horizon_weights": [1.0, 1.5, 2.0],
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
    }


def _provenance_fixture(
    value: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, list[dict[str, Any]]]:
    """为测试配置的每个 mapping 叶值生成独立且值匹配的来源记录。"""

    records: dict[str, list[dict[str, Any]]] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping) and item:
            records.update(_provenance_fixture(item, prefix=path))
        else:
            records[path] = [{"source": "unit_test", "value": item}]
    return records


def _freeze_manifest(
    path: Path,
    *,
    normal_source_hash: str = _A,
    fault_source_hash: str = _B,
    fault_episode_manifest: tuple[FrozenFaultEpisodeManifest, ...] | None = None,
) -> FrozenProtocolManifest:
    """把最小但字段完整的 P10 manifest 写到测试目录。"""

    checkpoint_path = path.parent / "synthetic_checkpoint.bin"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint_path.exists():
        checkpoint_path.write_bytes(b"synthetic checkpoint\n")
    episodes = fault_episode_manifest or _fault_episode_manifest()
    base = resolve_frozen_evaluation_config(_ROOT / "configs" / "paper" / "smoke.yaml")
    entry = base.config.model_copy(
        update={
            "evaluation_id": "cstr-smoke-eval-001",
            "claim_registry": path.parent / "claim-registry",
            "detection_risk": 0.3,
            "attribution_risk": 0.2,
        }
    )
    resolved = resolve_frozen_evaluation_config(entry)
    split_manifest, fit_access_ledger = _normal_protocol_evidence(
        normal_source_hash=normal_source_hash,
        fault_source_hash=fault_source_hash,
    )
    detection_source_hash = split_manifest["stages"]["detection_calibration"][
        "data_hash"
    ]
    attribution_source_hash = split_manifest["stages"]["attribution_calibration"][
        "data_hash"
    ]
    return FrozenProtocolManifest.freeze(
        path,
        protocol_version="cstr-smoke-v1",
        evaluation_id="cstr-smoke-eval-001",
        git_commit="1" * 40,
        resolved_config=resolved.resolved_config,
        config_provenance={
            path: [dict(record) for record in records]
            for path, records in resolved.provenance.items()
        },
        config_hash=resolved.config_hash,
        claim_registry_path=path.parent / "claim-registry",
        dependency_versions={
            "python": "3.12.0",
            "torch": "2.7.0",
            "numpy": "2.2.0",
            "pandas": "2.2.0",
            "scipy": "1.15.0",
            "scikit-learn": "1.6.0",
            "pydantic": "2.11.0",
        },
        raw_data_hashes={
            "normal": normal_source_hash,
            "fault": fault_source_hash,
        },
        split_manifest=split_manifest,
        fault_episode_manifest=episodes,
        seeds={"python": 7, "numpy": 11, "torch": 13, "dataloader": 17},
        checkpoint_paths={"protected_koopman_ts": checkpoint_path},
        checkpoint_hashes={
            "protected_koopman_ts": _SYNTHETIC_CHECKPOINT_HASH
        },
        checkpoint_replay={
            "status": "synthetic_contract",
            "checkpoint_hashes": {
                "protected_koopman_ts": _SYNTHETIC_CHECKPOINT_HASH
            },
            "output_hashes": {"contract": _A},
        },
        fit_access_ledger=fit_access_ledger,
        normal_artifacts=None,
        postfilter_library={
            "candidate_id": "candidate-001",
            "mode": "regular",
            "branches": ["guard", "omnibus"],
        },
        monitor_policy={
            "anchor_gate": {"gate_hash": _A},
            "hysteresis": {"enter": 2, "exit": 3},
            "reset_state": {"state_hash": _C},
        },
        detection_calibration=_risk_calibration(
            "detection",
            quantile=3.5,
            source_hash=detection_source_hash,
            stage_manifest=split_manifest["stages"]["detection_calibration"],
        ),
        attribution_calibration=_risk_calibration(
            "attribution",
            quantile=4.5,
            source_hash=attribution_source_hash,
            stage_manifest=split_manifest["stages"]["attribution_calibration"],
        ),
        certification_status={
            "operator": {"status": "unavailable", "reason": "no certified provider"},
            "signature": {"status": "uncertified"},
            "nuisance": {"status": "uncertified"},
        },
    )


def _resign_manifest_file(path: Path, payload: dict[str, Any]) -> None:
    """模拟攻击者改写嵌套证据后重算顶层 hash，验证内部合同仍能独立拒绝篡改。"""

    unsigned = dict(payload)
    unsigned.pop("manifest_hash", None)
    encoded = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    payload["manifest_hash"] = hashlib.sha256(encoded).hexdigest()
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _rehash_resolved_config(payload: dict[str, Any]) -> None:
    """重算被主动改写的 resolved config 身份，确保测试能越过外层完整性校验。"""

    encoded = json.dumps(
        payload["resolved_config"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["config_hash"] = hashlib.sha256(encoded).hexdigest()[:16]


def test_frozen_protocol_manifest_is_deeply_immutable_and_detects_tampering(
    tmp_path: Path,
) -> None:
    """冻结证据可重放，但内存或磁盘任一层被改写都不能静默通过。"""

    path = tmp_path / "frozen_protocol_manifest.json"
    manifest = _freeze_manifest(path)

    assert manifest.status == "frozen"
    assert manifest.evaluation_id == "cstr-smoke-eval-001"
    assert manifest.detection_calibration.quantile == 3.5
    assert manifest.attribution_calibration.quantile == 4.5
    assert manifest.detection_calibration.attainable_risk_resolution == 0.05
    assert not math.isinf(manifest.detection_calibration.quantile)
    assert FrozenProtocolManifest.load(path) == manifest

    with pytest.raises(FrozenInstanceError):
        manifest.evaluation_id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        manifest.resolved_config["mode"] = "frozen"  # type: ignore[index]
    with pytest.raises(TypeError):
        manifest.resolved_config["dataset"]["fault_onset"] = 200  # type: ignore[index]

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["resolved_config"]["dataset"]["fault_onset"] = 200
    path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(FrozenProtocolIntegrityError, match="manifest hash"):
        FrozenProtocolManifest.load(path)


def test_frozen_manifest_requires_value_matching_provenance_for_every_leaf(
    tmp_path: Path,
) -> None:
    """来源记录必须完整覆盖 resolved config，且最终 value 与叶字段一致。"""

    missing_path = tmp_path / "missing" / "frozen_protocol_manifest.json"
    _freeze_manifest(missing_path)
    missing = json.loads(missing_path.read_text(encoding="utf-8"))
    del missing["config_provenance"]["dataset.feature_count"]
    _resign_manifest_file(missing_path, missing)
    with pytest.raises(FrozenProtocolIntegrityError, match="provenance leaf coverage"):
        FrozenProtocolManifest.load(missing_path)

    wrong_value_path = tmp_path / "wrong" / "frozen_protocol_manifest.json"
    _freeze_manifest(wrong_value_path)
    wrong = json.loads(wrong_value_path.read_text(encoding="utf-8"))
    wrong["config_provenance"]["dataset.feature_count"][-1]["value"] = 99
    _resign_manifest_file(wrong_value_path, wrong)
    with pytest.raises(FrozenProtocolIntegrityError, match="provenance value"):
        FrozenProtocolManifest.load(wrong_value_path)


def test_frozen_manifest_rejects_incomplete_or_rehashed_split_evidence(
    tmp_path: Path,
) -> None:
    """顶层 hash 合法也不能掩盖空五阶段证据或失配的 P2 split hash。"""

    empty_stage_path = tmp_path / "empty-stage" / "frozen_protocol_manifest.json"
    _freeze_manifest(empty_stage_path)
    empty_stage = json.loads(empty_stage_path.read_text(encoding="utf-8"))
    empty_stage["split_manifest"]["stages"]["train"] = {}
    _resign_manifest_file(empty_stage_path, empty_stage)
    with pytest.raises(
        FrozenProtocolIntegrityError,
        match="split stage.*fields",
    ):
        FrozenProtocolManifest.load(empty_stage_path)

    changed_stage_path = tmp_path / "changed-stage" / "frozen_protocol_manifest.json"
    _freeze_manifest(changed_stage_path)
    changed_stage = json.loads(changed_stage_path.read_text(encoding="utf-8"))
    changed_stage["split_manifest"]["stages"]["train"]["data_hash"] = _B
    _resign_manifest_file(changed_stage_path, changed_stage)
    with pytest.raises(
        FrozenProtocolIntegrityError,
        match="split_hash.*content",
    ):
        FrozenProtocolManifest.load(changed_stage_path)


def test_frozen_manifest_rejects_incomplete_or_forged_fit_access_ledger(
    tmp_path: Path,
) -> None:
    """伪造 ready 标志不能掩盖缺失估计用途或与 P2 阶段不符的访问 hash。"""

    missing_estimate_path = tmp_path / "missing-estimate" / "frozen_protocol_manifest.json"
    _freeze_manifest(missing_estimate_path)
    missing_estimate = json.loads(missing_estimate_path.read_text(encoding="utf-8"))
    missing_estimate["fit_access_ledger"]["records"] = [
        record
        for record in missing_estimate["fit_access_ledger"]["records"]
        if record["purpose"]
        not in {
            FitPurpose.STRUCTURE_SELECTION.value,
            FitPurpose.MONITORING_SCORE_SCALER.value,
            FitPurpose.ENVELOPE.value,
            FitPurpose.COVARIANCE.value,
            FitPurpose.BRANCH_LIBRARY.value,
            FitPurpose.STATE_MACHINE.value,
        }
    ]
    _resign_manifest_file(missing_estimate_path, missing_estimate)
    with pytest.raises(
        FrozenProtocolIntegrityError,
        match="estimate-stage",
    ):
        FrozenProtocolManifest.load(missing_estimate_path)

    forged_stage_path = tmp_path / "forged-stage" / "frozen_protocol_manifest.json"
    _freeze_manifest(forged_stage_path)
    forged_stage = json.loads(forged_stage_path.read_text(encoding="utf-8"))
    detection_record = next(
        record
        for record in forged_stage["fit_access_ledger"]["records"]
        if record["purpose"] == FitPurpose.DETECTION_QUANTILE.value
    )
    detection_record["stage_hashes"]["detection_calibration"] = "f" * 64
    _resign_manifest_file(forged_stage_path, forged_stage)
    with pytest.raises(
        FrozenProtocolIntegrityError,
        match="stage_hashes.*split stage",
    ):
        FrozenProtocolManifest.load(forged_stage_path)


def test_frozen_manifest_binds_calibrations_to_their_p2_stages(tmp_path: Path) -> None:
    """两次风险摘要必须分别来自 detection/attribution calibration 阶段。"""

    path = tmp_path / "frozen_protocol_manifest.json"
    _freeze_manifest(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["detection_calibration"]["source_hash"] = payload["split_manifest"]["stages"][
        "train"
    ]["data_hash"]
    _resign_manifest_file(path, payload)

    with pytest.raises(
        FrozenProtocolIntegrityError,
        match="detection calibration source_hash",
    ):
        FrozenProtocolManifest.load(path)


def test_frozen_manifest_replays_calibration_episode_ids_from_p2_ranges(
    tmp_path: Path,
) -> None:
    """校准 episode 数量和 ID 必须由对应 P2 半开区间唯一导出。"""

    path = tmp_path / "frozen_protocol_manifest.json"
    _freeze_manifest(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["detection_calibration"]["episode_ids"][0] = "forged-episode"
    _resign_manifest_file(path, payload)

    with pytest.raises(
        FrozenProtocolIntegrityError,
        match="detection calibration episode_ids",
    ):
        FrozenProtocolManifest.load(path)


def test_frozen_manifest_strictly_replays_normal_method_config(tmp_path: Path) -> None:
    """重签后的未知 P4 模型字段也不能绕过严格 normal method schema。"""

    path = tmp_path / "frozen_protocol_manifest.json"
    _freeze_manifest(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["resolved_config"]["normal_method_config"] = {
        "model": {
            "type": "protected_koopman_ts",
            "unknown_field": "must-fail",
        }
    }
    _rehash_resolved_config(payload)
    _resign_manifest_file(path, payload)

    with pytest.raises(
        FrozenProtocolIntegrityError,
        match="normal_method_config",
    ):
        FrozenProtocolManifest.load(path)


def test_frozen_entry_and_manifest_bind_nested_risk_budget(tmp_path: Path) -> None:
    """归因风险 beta 必须严格小于检测风险 alpha，且校准产物必须逐项匹配配置。"""

    base = resolve_frozen_evaluation_config(_ROOT / "configs" / "paper" / "smoke.yaml")
    invalid = base.config.model_dump(mode="json")
    invalid["detection_risk"] = 0.2
    invalid["attribution_risk"] = 0.2
    with pytest.raises(ValueError, match="attribution_risk.*detection_risk"):
        FrozenEvaluationEntryConfig.model_validate(invalid)

    entry = base.config.model_copy(
        update={
            "detection_risk": 0.3,
            "attribution_risk": 0.2,
        }
    )
    resolved = resolve_frozen_evaluation_config(entry)
    checkpoint_path = tmp_path / "synthetic_checkpoint.bin"
    checkpoint_path.write_bytes(b"synthetic checkpoint\n")
    split_manifest, fit_access_ledger = _normal_protocol_evidence(
        normal_source_hash=_A,
        fault_source_hash=_B,
    )
    with pytest.raises(ValueError, match="detection calibration risk"):
        FrozenProtocolManifest.freeze(
            tmp_path / "unused-risk-manifest.json",
            protocol_version="cstr-smoke-v1",
            evaluation_id="cstr-smoke-risk-mismatch",
            git_commit="1" * 40,
            resolved_config=resolved.resolved_config,
            config_provenance={
                path: [dict(record) for record in records]
                for path, records in resolved.provenance.items()
            },
            config_hash=resolved.config_hash,
            claim_registry_path=tmp_path / "unused-claim-registry",
            dependency_versions={
                "python": "3.12.0",
                "torch": "2.7.0",
                "numpy": "2.2.0",
                "pandas": "2.2.0",
                "scipy": "1.15.0",
                "scikit-learn": "1.6.0",
                "pydantic": "2.11.0",
            },
            raw_data_hashes={"normal": _A, "fault": _B},
            split_manifest=split_manifest,
            fault_episode_manifest=_fault_episode_manifest(),
            seeds={"python": 7, "numpy": 11, "torch": 13, "dataloader": 17},
            checkpoint_paths={"protected_koopman_ts": checkpoint_path},
            checkpoint_hashes={
                "protected_koopman_ts": _SYNTHETIC_CHECKPOINT_HASH
            },
            checkpoint_replay={
                "status": "synthetic_contract",
                "checkpoint_hashes": {
                    "protected_koopman_ts": _SYNTHETIC_CHECKPOINT_HASH
                },
                "output_hashes": {"contract": _A},
            },
            fit_access_ledger=fit_access_ledger,
            normal_artifacts=None,
            postfilter_library={
                "candidate_id": "candidate-001",
                "mode": "regular",
                "branches": ["guard"],
            },
            monitor_policy={
                "anchor_gate": {"gate_hash": _A},
                "hysteresis": {"enter": 2, "exit": 3},
                "reset_state": {"state_hash": _C},
            },
            detection_calibration=replace(
                _risk_calibration(
                    "detection",
                    quantile=3.5,
                    source_hash=split_manifest["stages"]["detection_calibration"][
                        "data_hash"
                    ],
                    stage_manifest=split_manifest["stages"][
                        "detection_calibration"
                    ],
                ),
                requested_risk=0.4,
            ),
            attribution_calibration=_risk_calibration(
                "attribution",
                quantile=4.5,
                source_hash=split_manifest["stages"]["attribution_calibration"][
                    "data_hash"
                ],
                stage_manifest=split_manifest["stages"][
                    "attribution_calibration"
                ],
            ),
            certification_status={
                "operator": {"status": "unavailable"},
                "signature": {"status": "uncertified"},
                "nuisance": {"status": "uncertified"},
            },
        )


def test_formal_git_identity_rejects_a_dirty_worktree(tmp_path: Path) -> None:
    """formal manifest 只能绑定 clean HEAD；未跟踪文件也属于不可重放状态。"""

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "paper-test@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Paper Test"],
        cwd=repo,
        check=True,
    )
    tracked = repo / "tracked.txt"
    tracked.write_text("frozen\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "fixture"],
        cwd=repo,
        check=True,
        capture_output=True,
    )

    assert len(current_clean_paper_git_commit(repo)) == 40
    (repo / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="dirty"):
        current_clean_paper_git_commit(repo)


def test_existing_manifest_reuse_requires_the_same_clean_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """跨进程复用也必须核对当前 clean HEAD，不能只比较入口配置。"""

    manifest_path = tmp_path / "frozen_protocol_manifest.json"
    manifest = _freeze_manifest(manifest_path)
    base = resolve_frozen_evaluation_config(
        _ROOT / "configs" / "paper" / "smoke.yaml"
    )
    resolved = resolve_frozen_evaluation_config(
        base.config.model_copy(
            update={
                "evaluation_id": manifest.evaluation_id,
                "claim_registry": manifest_path.parent / "claim-registry",
                "detection_risk": 0.3,
                "attribution_risk": 0.2,
            }
        )
    )
    monkeypatch.setattr(
        frozen_cli_module,
        "current_clean_paper_git_commit",
        lambda _root: "2" * 40,
    )

    with pytest.raises(FrozenProtocolIntegrityError, match="clean HEAD"):
        frozen_cli_module._load_or_build_manifest(
            resolved,
            repo_root=_ROOT,
            manifest_path=manifest_path,
        )


def test_existing_manifest_reuse_requires_identical_dependency_versions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """跨进程依赖版本变化必须在 claim 和故障访问前阻塞。"""

    manifest_path = tmp_path / "frozen_protocol_manifest.json"
    manifest = _freeze_manifest(manifest_path)
    base = resolve_frozen_evaluation_config(
        _ROOT / "configs" / "paper" / "smoke.yaml"
    )
    resolved = resolve_frozen_evaluation_config(
        base.config.model_copy(
            update={
                "evaluation_id": manifest.evaluation_id,
                "claim_registry": manifest_path.parent / "claim-registry",
                "detection_risk": 0.3,
                "attribution_risk": 0.2,
            }
        )
    )
    monkeypatch.setattr(
        frozen_cli_module,
        "current_clean_paper_git_commit",
        lambda _root: manifest.git_commit,
    )
    changed_versions = dict(manifest.dependency_versions)
    changed_versions["torch"] = "99.0.0"
    monkeypatch.setattr(
        frozen_cli_module,
        "collect_paper_dependency_versions",
        lambda: changed_versions,
    )

    with pytest.raises(FrozenProtocolIntegrityError, match="dependency_versions"):
        frozen_cli_module._load_or_build_manifest(
            resolved,
            repo_root=_ROOT,
            manifest_path=manifest_path,
        )


def test_frozen_normal_artifact_bundle_replays_training_and_output_files(
    tmp_path: Path,
) -> None:
    """训练历史和 checkpoint 输出必须绑定实际文件，摘要 JSON 不能自报 hash。"""

    training_history = tmp_path / "training_history.json"
    replay_output = tmp_path / "frozen_normal_outputs.jsonl"
    training_history.write_text('[{"epoch": 1, "loss": 0.5}]\n', encoding="utf-8")
    replay_output.write_text('{"raw_index": 10, "score": 0.25}\n', encoding="utf-8")
    bundle = FrozenNormalArtifactBundle.build(
        artifact_paths={
            "training_history": training_history,
            "checkpoint_replay_output.frozen_normal": replay_output,
        },
        ledger_bindings={
            "model": "training_history",
            "normal-diagnostic": "checkpoint_replay_output.frozen_normal",
        },
        replay_outputs={
            "frozen_normal": "checkpoint_replay_output.frozen_normal",
        },
        runtime_evaluator={
            "type": "protected_koopman_ts_frozen",
            "state_hash": _A,
            "checkpoint_name": "protected_koopman_ts",
        },
    )

    assert FrozenNormalArtifactBundle.from_dict(bundle.to_dict()) == bundle
    replay_output.write_text('{"raw_index": 10, "score": 9.0}\n', encoding="utf-8")
    with pytest.raises(FrozenProtocolIntegrityError, match="normal artifact hash"):
        FrozenNormalArtifactBundle.from_dict(bundle.to_dict())


def test_registered_protected_runtime_restores_p4_model_without_fault_truth() -> None:
    """默认正式 evaluator 由公共 factory 恢复，并在未认证路径保守输出。"""

    model_config = _normal_method_config_fixture()["model"]
    model = build_model(model_config)
    evaluator = build_evaluator(
        {
            "type": "protected_koopman_ts_frozen",
            "state": {
                "schema_version": 1,
                "feature_layout": {
                    "control_indices": [0, 1],
                    "measurement_indices": [2, 3, 4, 5],
                    "exogenous_indices": [6],
                },
                "anchor_gate": {
                    "confirmation_delay": 1,
                    "enter_threshold": 10.0,
                    "exit_threshold": 10.0,
                    "minimum_reanchor_interval": 1,
                    "maximum_reference_age": 3,
                },
                "eligibility_center": [0.0, 0.0, 0.0, 0.0],
                "eligibility_scale": [1.0, 1.0, 1.0, 1.0],
                "score_scale": 1.0,
                "branch_id": "guard",
                "mode": "regular",
                "normal_family_id": "normal",
                "unresolved_family_id": "unresolved-fault",
                "threshold": {
                    "floor": 0.1,
                    "gamma_anc": 0.2,
                    "deterministic_intercept": 0.3,
                    "input_l1_weight": 0.01,
                    "stochastic_quantile": 0.4,
                },
            },
            "checkpoint": {
                "config": model_config,
                "model_state_dict": model.state_dict(),
            },
        }
    )
    runtime_input = FrozenEpisodeInput(
        raw_indices=np.arange(8, dtype=np.int64),
        values=np.zeros((8, 7), dtype=float),
    )

    result = evaluator.evaluate_episode(runtime_input)

    assert evaluator.formal_pipeline_complete is False
    assert len(result.outputs) == 8
    assert all(output.isolation_certified is False for output in result.outputs)
    assert all(
        output.method_outputs["operator"]["status"] == "unavailable"
        for output in result.outputs
    )
    assert not hasattr(result, "fault_id")


def test_normal_artifact_bundle_recomputes_evaluator_identity_from_checkpoint(
    tmp_path: Path,
) -> None:
    """type/state_hash 必须来自 checkpoint envelope，不能由 manifest 任意声明。"""

    checkpoint = tmp_path / "protected.pt"
    torch.save(
        {
            "extra_state": {
                "frozen_episode_evaluator": {
                    "schema_version": 1,
                    "type": "protected_koopman_ts_frozen",
                    "state": {"schema_version": 1},
                }
            }
        },
        checkpoint,
    )

    with pytest.raises(
        FrozenProtocolIntegrityError,
        match="runtime evaluator identity",
    ):
        FrozenNormalArtifactBundle.build(
            artifact_paths={
                "checkpoint_files.protected_koopman_ts": checkpoint,
            },
            ledger_bindings={},
            replay_outputs={},
            runtime_evaluator={
                "type": "protected_koopman_ts_frozen",
                "state_hash": _A,
                "checkpoint_name": "protected_koopman_ts",
            },
        )


def test_protected_runtime_state_is_cross_bound_to_frozen_method_artifacts(
    tmp_path: Path,
) -> None:
    """checkpoint 内 gate/branch/mode/family/q_det 必须与独立正常产物逐字段一致。"""

    checkpoint = tmp_path / "protected.pt"
    anchor_gate = {
        "confirmation_delay": 1,
        "enter_threshold": 10.0,
        "exit_threshold": 10.0,
        "minimum_reanchor_interval": 1,
        "maximum_reference_age": 3,
    }
    state = {
        "schema_version": 1,
        "feature_layout": {
            "control_indices": [0, 1],
            "measurement_indices": [2, 3, 4, 5],
            "exogenous_indices": [6],
        },
        "anchor_gate": anchor_gate,
        "eligibility_center": [0.0, 0.0, 0.0, 0.0],
        "eligibility_scale": [1.0, 1.0, 1.0, 1.0],
        "score_scale": 1.0,
        "branch_id": "guard",
        "mode": "regular",
        "normal_family_id": "normal",
        "unresolved_family_id": "unresolved-fault",
        "threshold": {
            "floor": 0.1,
            "gamma_anc": 0.2,
            "deterministic_intercept": 0.3,
            "input_l1_weight": 0.01,
            "stochastic_quantile": 0.4,
        },
    }
    torch.save(
        {
            "extra_state": {
                "frozen_episode_evaluator": {
                    "schema_version": 1,
                    "type": "protected_koopman_ts_frozen",
                    "state": state,
                }
            }
        },
        checkpoint,
    )
    gate_hash = hashlib.sha256(
        json.dumps(
            anchor_gate,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    postfilter = {
        "candidate_id": "candidate-001",
        "mode": "regular",
        "branches": ["guard", "omnibus"],
    }
    monitor = {
        "anchor_gate": {"source": "estimate", "gate_hash": gate_hash},
        "hysteresis": {"enter": 2, "exit": 3},
        "reset_state": {"kind": "episode-boundary"},
    }
    isolation = {
        "schema_version": 1,
        "normal_family": "normal",
        "fault_families": ["unresolved-fault"],
        "certified": True,
    }

    validate_protected_evaluator_artifact_bindings(
        checkpoint,
        postfilter_library=postfilter,
        monitor_policy=monitor,
        monitoring_score_scaler={"schema_version": 1, "rms_scale": 1.0},
        isolation_library=isolation,
        detection_quantile=0.4,
    )

    invalid_bindings = (
        ({**postfilter, "mode": "guard-only"}, monitor, 0.4),
        ({**postfilter, "branches": ["omnibus"]}, monitor, 0.4),
        (
            postfilter,
            {
                **monitor,
                "anchor_gate": {"source": "estimate", "gate_hash": _B},
            },
            0.4,
        ),
        (postfilter, monitor, 9.0),
    )
    for invalid_postfilter, invalid_monitor, invalid_quantile in invalid_bindings:
        with pytest.raises(FrozenProtocolIntegrityError):
            validate_protected_evaluator_artifact_bindings(
                checkpoint,
                postfilter_library=invalid_postfilter,
                monitor_policy=invalid_monitor,
                monitoring_score_scaler={"schema_version": 1, "rms_scale": 1.0},
                isolation_library=isolation,
                detection_quantile=invalid_quantile,
            )
    with pytest.raises(FrozenProtocolIntegrityError, match="score_scale"):
        validate_protected_evaluator_artifact_bindings(
            checkpoint,
            postfilter_library=postfilter,
            monitor_policy=monitor,
            monitoring_score_scaler={"schema_version": 1, "rms_scale": 2.0},
            isolation_library=isolation,
            detection_quantile=0.4,
        )
    for invalid_isolation in (
        {**isolation, "normal_family": "baseline"},
        {**isolation, "fault_families": ["process"]},
    ):
        with pytest.raises(FrozenProtocolIntegrityError, match="family"):
            validate_protected_evaluator_artifact_bindings(
                checkpoint,
                postfilter_library=postfilter,
                monitor_policy=monitor,
                monitoring_score_scaler={"schema_version": 1, "rms_scale": 1.0},
                isolation_library=invalid_isolation,
                detection_quantile=0.4,
            )


def test_pointwise_isolation_outcomes_preserve_set_semantics() -> None:
    """空集、含 Normal 集、非唯一集和 singleton 必须与顶层报告字段严格一致。"""

    out_of_model = FrozenPointwiseOutput(
        episode_id="fault-1",
        raw_index=0,
        fault_id=1,
        true_label=0,
        detection_score=0.0,
        detection_threshold=1.0,
        alarm=False,
        branch_id="guard",
        mode="regular",
        normal_family_id="normal",
        candidate_ids=(),
        isolation_outcome="Out-of-model",
        reported_family=None,
        isolation_certified=True,
        suppression_reason=None,
        method_outputs=_complete_method_outputs(raw_index=0, alarm=False),
    )
    assert out_of_model.candidate_ids == ()

    invalid_cases = (
        {
            "candidate_ids": (),
            "isolation_outcome": "Normal-compatible",
            "reported_family": None,
            "isolation_certified": True,
        },
        {
            "candidate_ids": ("normal", "sensor-1"),
            "isolation_outcome": "Nonunique",
            "reported_family": None,
            "isolation_certified": True,
        },
        {
            "candidate_ids": ("sensor-1",),
            "isolation_outcome": "singleton",
            "reported_family": None,
            "isolation_certified": True,
        },
    )
    for invalid in invalid_cases:
        with pytest.raises(ValueError, match="isolation"):
            replace(out_of_model, **invalid)


def test_claim_registry_is_bound_by_manifest_before_any_write(tmp_path: Path) -> None:
    """调用方换 registry 不能为同一 manifest 创建第二套一次性 ID 空间。"""

    manifest = _freeze_manifest(tmp_path / "frozen_protocol_manifest.json")
    wrong_registry = tmp_path / "alternate-claim-registry"
    with pytest.raises(FrozenProtocolIntegrityError, match="Claim registry differs"):
        FrozenEvaluationClaim.create(
            manifest=manifest,
            claim_registry=wrong_registry,
            artifact_dir=tmp_path / "evaluation",
        )
    assert not wrong_registry.exists()


def _fault_episodes(
    manifests: tuple[FrozenFaultEpisodeManifest, ...],
) -> tuple[FrozenFaultEpisode, ...]:
    """生成与八份静态身份逐行对齐的合成数值、标签和 raw index。"""

    episodes: list[FrozenFaultEpisode] = []
    for episode_manifest in manifests:
        values = np.column_stack(
            (
                np.arange(episode_manifest.row_count, dtype=float),
                np.full(episode_manifest.row_count, episode_manifest.fault_id, dtype=float),
            )
        )
        labels = np.zeros(episode_manifest.row_count, dtype=int)
        labels[episode_manifest.onset :] = episode_manifest.fault_id
        episodes.append(
            FrozenFaultEpisode(
                manifest=episode_manifest,
                raw_indices=np.arange(
                    episode_manifest.raw_index_start,
                    episode_manifest.raw_index_end + 1,
                    dtype=int,
                ),
                values=values,
                labels=labels,
            )
        )
    return tuple(episodes)


def test_frozen_runtime_input_exposes_features_without_fault_truth() -> None:
    """evaluator 输入只含特征与行号，不能直接读取故障答案。"""

    episode = _fault_episodes((_fault_episode_manifest()[0],))[0]
    runtime_input = FrozenEpisodeInput.from_fault_episode(episode)

    assert np.array_equal(runtime_input.raw_indices, episode.raw_indices)
    assert np.array_equal(runtime_input.values, episode.values)
    for forbidden_name in (
        "labels",
        "manifest",
        "episode_id",
        "onset",
        "fault_id",
        "fault_family",
    ):
        assert not hasattr(runtime_input, forbidden_name)


class _ClaimCheckingSource:
    """只在测试中使用的 source，用调用次数证明 claim 先于故障访问。"""

    def __init__(
        self,
        episodes: tuple[FrozenFaultEpisode, ...],
        claim_path: Path,
    ) -> None:
        self.episodes = episodes
        self.claim_path = claim_path
        self.calls = 0

    def request_episodes(
        self,
        manifest: FrozenProtocolManifest,
        *,
        claim: FrozenEvaluationClaim,
    ) -> tuple[FrozenFaultEpisode, ...]:
        """确认全局 claim 已落盘，再返回一次合成故障副本。"""

        assert manifest.status == "frozen"
        claim.verify(manifest)
        assert claim.claim_path == self.claim_path.resolve()
        assert self.claim_path.exists()
        self.calls += 1
        claim.consume_fault_access(manifest)
        return self.episodes


class _DeterministicPointwiseEvaluator:
    """把每一输入行映射为字段完整、结果可手算的 synthetic 输出。"""

    formal_pipeline_complete = True

    def __init__(self) -> None:
        self.episode_calls: list[tuple[int, int]] = []

    def evaluate_episode(
        self,
        episode: FrozenEpisodeInput,
    ) -> FrozenRuntimeEpisodeEvaluation:
        """只根据特征视图构造固定输出；没有接口可读取真实故障答案。"""

        for forbidden_name in (
            "labels",
            "manifest",
            "episode_id",
            "onset",
            "fault_id",
            "fault_family",
        ):
            assert not hasattr(episode, forbidden_name)
        self.episode_calls.append(
            (int(episode.raw_indices[0]), int(episode.raw_indices[-1]))
        )
        outputs: list[FrozenRuntimePointwiseOutput] = []
        threshold = 1.5
        for raw_index in episode.raw_indices.tolist():
            alarm = float(raw_index) > threshold
            outputs.append(
                FrozenRuntimePointwiseOutput(
                    raw_index=raw_index,
                    detection_score=float(raw_index),
                    detection_threshold=threshold,
                    alarm=alarm,
                    branch_id="guard",
                    mode="regular",
                    normal_family_id="normal",
                    candidate_ids=(
                        ("unresolved-fault",)
                        if alarm
                        else ("normal",)
                    ),
                    isolation_outcome=(
                        "Uncertified" if alarm else "Normal-compatible"
                    ),
                    reported_family=None,
                    isolation_certified=False,
                    suppression_reason=None,
                    method_outputs=_complete_method_outputs(
                        raw_index=raw_index,
                        alarm=alarm,
                    ),
                )
            )
        return FrozenRuntimeEpisodeEvaluation(outputs=tuple(outputs))


class _DeterministicFixtureEvaluatorState(StrictConfig):
    """仅测试用的严格纯数据 evaluator state。"""

    output_contract: Literal["deterministic-pointwise-test-v1"]


class _RegisteredDeterministicFixtureEvaluator(_DeterministicPointwiseEvaluator):
    """通过公共 evaluator registry 恢复的严格测试运行时。"""

    def __init__(
        self,
        *,
        state: Mapping[str, Any],
        checkpoint: Mapping[str, Any],
    ) -> None:
        """严格验证纯数据 state；checkpoint 只验证 weights-only mapping 边界。"""

        config = _DeterministicFixtureEvaluatorState.model_validate(state)
        if config.model_dump(mode="json") != {
            "output_contract": "deterministic-pointwise-test-v1"
        }:
            raise ValueError("Unexpected deterministic fixture evaluator state.")
        if not isinstance(checkpoint, Mapping):
            raise TypeError("Fixture checkpoint must be a weights-only mapping.")
        super().__init__()


register_evaluator(
    "deterministic_pointwise_fixture",
    _RegisteredDeterministicFixtureEvaluator,
    replace=True,
)


class _WrongBranchEvaluator(_DeterministicPointwiseEvaluator):
    """故意把首行输出改为冻结 library 外分支，验证工作流交叉门禁。"""

    def evaluate_episode(
        self,
        episode: FrozenEpisodeInput,
    ) -> FrozenRuntimeEpisodeEvaluation:
        """复用合法输出后只篡改 branch_id，其他字段保持可通过单对象校验。"""

        result = super().evaluate_episode(episode)
        return FrozenRuntimeEpisodeEvaluation(
            outputs=(
                replace(result.outputs[0], branch_id="late-added"),
                *result.outputs[1:],
            ),
        )


def _complete_method_outputs(*, raw_index: int, alarm: bool) -> dict[str, Any]:
    """提供 P22 要求的九类逐时刻来源；值小而完整，便于独立检查。"""

    return {
        "prediction": {"raw_index": raw_index, "one_step": [0.0], "multi_step": [0.0]},
        "rule_weights": {"weights": [1.0]},
        "monitor": {"mode": "regular", "anchor_raw_index": 0},
        "protected_state": {"context_age": raw_index},
        "residual": {"stacked": [float(raw_index)]},
        "operator": {"status": "unavailable"},
        "branch_statistics": {"guard": float(raw_index)},
        "threshold": {
            "floor": 0.5,
            "gamma_anc": 0.25,
            "gamma_det": 0.25,
            "stochastic": 0.5,
            "alarm": alarm,
        },
        "isolation": {"outcome": "Uncertified" if alarm else "Normal-compatible"},
    }


def _complete_bundle_protocol(
    bundle: PaperDataBundle,
    *,
    artifact_hashes: Mapping[str, str] | None = None,
) -> None:
    """按 P2 顺序冻结一个最小正常协议，供 lazy source 门禁测试使用。"""

    for object_id, purpose in (
        ("model", FitPurpose.MODEL_PARAMETERS),
        ("structure", FitPurpose.STRUCTURE_SELECTION),
        ("score-scaler", FitPurpose.MONITORING_SCORE_SCALER),
        ("envelope", FitPurpose.ENVELOPE),
        ("covariance", FitPurpose.COVARIANCE),
        ("branch-library", FitPurpose.BRANCH_LIBRARY),
        ("state-machine", FitPurpose.STATE_MACHINE),
        ("q-det", FitPurpose.DETECTION_QUANTILE),
        ("q-attr", FitPurpose.ATTRIBUTION_QUANTILE),
        ("normal-diagnostic", FitPurpose.FROZEN_NORMAL_DIAGNOSTIC),
    ):
        bundle.data_for_fit(object_id, purpose)
        bundle.fit_access_ledger.freeze_record(
            object_id,
            (artifact_hashes or {}).get(object_id, _D),
        )
    bundle.freeze_protocol(bundle.split_result.split_hash)


def test_lazy_cstr_source_opens_only_after_normal_protocol_freeze_and_only_once(
    tmp_path: Path,
) -> None:
    """具体 source 必须复用 P2 门禁，构造时不能提前调用故障 loader。"""

    manifest_path = tmp_path / "frozen_protocol_manifest.json"
    manifest = _freeze_manifest(manifest_path)
    episodes = _fault_episodes(manifest.fault_episode_manifest)
    loader_calls = 0

    def load_faults() -> tuple[FrozenFaultEpisode, ...]:
        nonlocal loader_calls
        loader_calls += 1
        return episodes

    split_config = FiveStageSplitConfig(
        history_length=1,
        max_rollout=1,
        episode_length=2,
        target_risk_level=0.5,
    )
    bundle = PaperDataBundle(
        np.arange(240, dtype=float).reshape(120, 2),
        config=split_config,
        normal_source_hash=_A,
        fault_source_hash=_B,
        fault_license_status=FaultLicenseStatus.VERIFIED,
    )
    source = LazyFrozenCSTRFaultSource(
        bundle=bundle,
        loader=load_faults,
        fault_source_hash=_B,
    )
    assert loader_calls == 0

    _complete_bundle_protocol(bundle)
    claim = FrozenEvaluationClaim.create(
        manifest=manifest,
        claim_registry=tmp_path / "claim-registry",
        artifact_dir=tmp_path / "evaluation",
    )
    assert source.request_episodes(manifest, claim=claim) == episodes
    assert loader_calls == 1
    assert bundle.fault_accessed

    with pytest.raises(ProtocolAccessError, match="already accessed"):
        source.request_episodes(manifest, claim=claim)
    assert loader_calls == 1


def test_frozen_evaluation_claims_once_and_writes_complete_machine_readable_sources(
    tmp_path: Path,
) -> None:
    """一次 ID 覆盖八个 episode、全部行和来源表；重复运行在访问故障前失败。"""

    manifest_path = tmp_path / "frozen_protocol_manifest.json"
    manifest = _freeze_manifest(manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    registry = tmp_path / "claim-registry"
    artifact_dir = tmp_path / "evaluation"
    claim_path = registry / f"{manifest.evaluation_id}.claim.json"
    source = _ClaimCheckingSource(
        _fault_episodes(manifest.fault_episode_manifest),
        claim_path,
    )
    evaluator = _DeterministicPointwiseEvaluator()

    result = FrozenEvaluationWorkflow(
        manifest_path=manifest_path,
        claim_registry=registry,
        artifact_dir=artifact_dir,
        fault_source=source,
        evaluator=evaluator,
    ).run()

    assert source.calls == 1
    assert evaluator.episode_calls == [(0, 4)] * 8
    assert result.evaluation_id == manifest.evaluation_id
    assert result.pointwise_row_count == 40
    assert manifest_path.read_bytes() == manifest_bytes
    assert result.pointwise_path.exists()
    assert result.detection_source_path.exists()
    assert result.isolation_source_path.exists()
    assert result.artifact_index_path.exists()
    assert result.receipt_path.exists()
    assert verify_frozen_evaluation_artifacts(
        manifest_path=manifest_path,
        receipt_path=result.receipt_path,
    ) == result
    with result.detection_source_path.open(encoding="utf-8", newline="") as stream:
        detection_rows = list(csv.DictReader(stream))
    assert len(detection_rows) == 8
    assert {int(row["fault_id"]) for row in detection_rows} == set(range(1, 9))
    assert {int(row["pre_fault_alarm_count"]) for row in detection_rows} == {0}
    assert {int(row["post_onset_alarm_count"]) for row in detection_rows} == {3}

    duplicate_source = _ClaimCheckingSource(
        _fault_episodes(manifest.fault_episode_manifest),
        claim_path,
    )
    with pytest.raises(
        FrozenEvaluationAlreadyClaimedError,
        match=manifest.evaluation_id,
    ):
        FrozenEvaluationWorkflow(
            manifest_path=manifest_path,
            claim_registry=registry,
            artifact_dir=tmp_path / "duplicate-evaluation",
            fault_source=duplicate_source,
            evaluator=_DeterministicPointwiseEvaluator(),
        ).run()
    assert duplicate_source.calls == 0


def test_incomplete_fault_library_burns_claim_without_calling_evaluator(
    tmp_path: Path,
) -> None:
    """fault source 已打开后发现少 episode 时必须保留 claim，不能修补后用原 ID 重跑。"""

    manifest_path = tmp_path / "frozen_protocol_manifest.json"
    manifest = _freeze_manifest(manifest_path)
    registry = tmp_path / "claim-registry"
    claim_path = registry / f"{manifest.evaluation_id}.claim.json"
    incomplete_source = _ClaimCheckingSource(
        _fault_episodes(manifest.fault_episode_manifest)[:-1],
        claim_path,
    )
    evaluator = _DeterministicPointwiseEvaluator()

    with pytest.raises(
        FrozenProtocolIntegrityError,
        match="all eight",
    ):
        FrozenEvaluationWorkflow(
            manifest_path=manifest_path,
            claim_registry=registry,
            artifact_dir=tmp_path / "incomplete-evaluation",
            fault_source=incomplete_source,
            evaluator=evaluator,
        ).run()

    assert incomplete_source.calls == 1
    assert evaluator.episode_calls == []
    assert claim_path.exists()
    with pytest.raises(FrozenEvaluationAlreadyClaimedError):
        FrozenEvaluationWorkflow(
            manifest_path=manifest_path,
            claim_registry=registry,
            artifact_dir=tmp_path / "retry-evaluation",
            fault_source=_ClaimCheckingSource(
                _fault_episodes(manifest.fault_episode_manifest),
                claim_path,
            ),
            evaluator=_DeterministicPointwiseEvaluator(),
        ).run()


def test_evaluator_cannot_use_branch_outside_frozen_postfilter_library(
    tmp_path: Path,
) -> None:
    """在线输出出现未冻结 branch 时整次评价失败，claim 保留且不写 completion receipt。"""

    manifest_path = tmp_path / "frozen_protocol_manifest.json"
    manifest = _freeze_manifest(manifest_path)
    registry = tmp_path / "claim-registry"
    artifact_dir = tmp_path / "evaluation"
    source = _ClaimCheckingSource(
        _fault_episodes(manifest.fault_episode_manifest),
        registry / f"{manifest.evaluation_id}.claim.json",
    )

    with pytest.raises(FrozenProtocolIntegrityError, match="branch library"):
        FrozenEvaluationWorkflow(
            manifest_path=manifest_path,
            claim_registry=registry,
            artifact_dir=artifact_dir,
            fault_source=source,
            evaluator=_WrongBranchEvaluator(),
        ).run()

    assert (registry / f"{manifest.evaluation_id}.claim.json").exists()
    assert not (artifact_dir / "evaluation_receipt.json").exists()


def test_artifact_verification_rejects_changed_pointwise_output(
    tmp_path: Path,
) -> None:
    """receipt 完成后改动一行逐时刻来源，独立 verifier 必须按文件 hash 拒绝。"""

    manifest_path = tmp_path / "frozen_protocol_manifest.json"
    manifest = _freeze_manifest(manifest_path)
    registry = tmp_path / "claim-registry"
    source = _ClaimCheckingSource(
        _fault_episodes(manifest.fault_episode_manifest),
        registry / f"{manifest.evaluation_id}.claim.json",
    )
    result = FrozenEvaluationWorkflow(
        manifest_path=manifest_path,
        claim_registry=registry,
        artifact_dir=tmp_path / "evaluation",
        fault_source=source,
        evaluator=_DeterministicPointwiseEvaluator(),
    ).run()

    with result.pointwise_path.open("a", encoding="utf-8") as stream:
        stream.write("{}\n")

    with pytest.raises(FrozenProtocolIntegrityError, match="hash does not match"):
        verify_frozen_evaluation_artifacts(
            manifest_path=manifest_path,
            receipt_path=result.receipt_path,
        )


def test_changed_checkpoint_is_rejected_before_claim_or_fault_access(
    tmp_path: Path,
) -> None:
    """manifest 后替换 checkpoint 必须在 claim 前失败，不能消耗或读取故障 source。"""

    manifest_path = tmp_path / "frozen_protocol_manifest.json"
    manifest = _freeze_manifest(manifest_path)
    checkpoint_path = Path(manifest.checkpoint_paths["protected_koopman_ts"])
    checkpoint_path.write_bytes(b"changed checkpoint\n")
    registry = tmp_path / "claim-registry"
    claim_path = registry / f"{manifest.evaluation_id}.claim.json"
    source = _ClaimCheckingSource(
        _fault_episodes(manifest.fault_episode_manifest),
        claim_path,
    )

    with pytest.raises(FrozenProtocolIntegrityError, match="checkpoint hash"):
        FrozenEvaluationWorkflow(
            manifest_path=manifest_path,
            claim_registry=registry,
            artifact_dir=tmp_path / "evaluation",
            fault_source=source,
            evaluator=_DeterministicPointwiseEvaluator(),
        ).run()

    assert not claim_path.exists()
    assert source.calls == 0


def test_three_paper_entry_configs_are_strict_and_expose_frozen_readiness_gates() -> None:
    """三条入口都可严格解析；本地 MAT 许可未核实时 frozen 必须最先阻塞。"""

    smoke_path = _ROOT / "configs" / "paper" / "smoke.yaml"
    development_path = _ROOT / "configs" / "paper" / "cstr_development.yaml"
    frozen_path = _ROOT / "configs" / "paper" / "cstr_frozen.yaml"
    smoke = resolve_frozen_evaluation_config(smoke_path)
    development = resolve_frozen_evaluation_config(development_path)
    frozen = resolve_frozen_evaluation_config(frozen_path)

    assert smoke.config.mode == "smoke"
    assert smoke.config.runtime == "synthetic_contract_smoke"
    assert development.config.mode == "development"
    assert development.config.development is not None
    assert development.config.normal_artifacts is not None
    assert frozen.config.mode == "frozen"
    assert frozen.config.evaluation_id == "cstr-frozen-pending-mat-license-and-certified-runtime"
    assert len(smoke.config_hash) == 16
    assert resolve_frozen_evaluation_config(smoke_path).config_hash == smoke.config_hash
    assert smoke.provenance["mode"][0]["source"].startswith("yaml:")
    assert set(frozen.config.normal_artifacts.ledger_bindings) == {
        "model",
        "structure",
        "score-scaler",
        "envelope",
        "covariance",
        "branch-library",
        "state-machine",
        "q-det",
        "q-attr",
        "normal-diagnostic",
    }

    readiness_errors = frozen.config.frozen_readiness_errors(repo_root=_ROOT)
    assert development.config.dataset.license_status == "to_verify"
    assert frozen.config.dataset.license_status == "to_verify"
    assert any("license" in error for error in readiness_errors)
    # 干净克隆还会报告缺失的 ignored 正常产物；即使本地已有 development 产物，
    # MAT 生成链未核实也必须在 runtime/manifest/claim 之前独立关闭。
    assert all(
        "license" in error or "normal_artifacts" in error
        for error in readiness_errors
    )

    invalid = smoke.config.model_dump(mode="json")
    invalid["unknown_paper_key"] = True
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        FrozenEvaluationEntryConfig.model_validate(invalid)

    invalid_floor = development.config.model_dump(mode="json")
    invalid_floor["development"]["threshold_floor"] = 0.0
    with pytest.raises(ValueError, match="greater than 0"):
        FrozenEvaluationEntryConfig.model_validate(invalid_floor)

    invalid_reference_age = development.config.model_dump(mode="json")
    invalid_reference_age["development"]["anchor_gate"]["maximum_reference_age"] = (
        development.config.development.method.model.max_rollout + 1
    )
    with pytest.raises(ValueError, match="maximum_reference_age"):
        FrozenEvaluationEntryConfig.model_validate(invalid_reference_age)


def test_closed_loop_cstr_model_license_does_not_overstate_local_mat_readiness() -> None:
    """卡片保留上游模型 BSD 证据，但本地 MAT 来源链未闭合时正式配置必须关闭。"""

    card_path = (
        _ROOT
        / "datasets"
        / "cards"
        / "oa"
        / "cstr_closed_loop_fd"
        / "dataset_card.yaml"
    )
    card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
    model_source = card["source"]["upstream_model"]
    notice_path = card_path.parent / model_source["notice_file"]

    assert card["access"]["license"] == "to_verify"
    assert card["access"]["license_status"] == "to_verify"
    assert card["access"]["license_reason"] == "local_mat_generation_chain_not_documented"
    assert model_source["license_status"] == "verified"
    assert model_source["license_id"] == "BSD-3-Clause"
    assert model_source["version"] == "1.1.0.1"
    assert model_source["license_url"] == (
        "https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/"
        "89e57c22-64a7-4cba-8aa3-da4279b09619/"
        "cb37e495-cbbf-47ab-90eb-605e5328de59/license/license.txt"
    )
    assert model_source["license_text_sha256"] == (
        "8c89de130c1e25815100e4dd5dcc3a9b602a74ee9b94f3eebf3513c53945b39e"
    )
    assert model_source["local_model_sha256"] == (
        "4555be2fa4c93ab43d2f24ab26e2bf6511ec25d701b231fc7d57f6657b523a81"
    )
    assert notice_path.is_file()
    assert hashlib.sha256(notice_path.read_bytes()).hexdigest() == model_source["notice_sha256"]


def test_normal_development_entry_generates_all_artifacts_without_fault_access(
    tmp_path: Path,
) -> None:
    """development 命令不覆盖协议摘要，并从正常 MAT 生成完整无故障访问文件合同。"""

    data_root = tmp_path / "data"
    data_root.mkdir()
    normal_path = data_root / "normal.mat"
    rng = np.random.default_rng(17)
    values = rng.normal(scale=0.05, size=(800, 7))
    savemat(normal_path, {"normal": values})
    normal_hash = hashlib.sha256(normal_path.read_bytes()).hexdigest()
    card_path, card_hash = _write_verified_dataset_card_fixture(
        data_root / ".paper-evidence" / "dataset_card.yaml",
        data_root=data_root,
        normal_file="normal.mat",
        fault_file="must-not-be-opened.mat",
        normal_sha256=normal_hash,
        fault_sha256=_B,
    )
    run_dir = tmp_path / "runs" / "development"
    artifact_paths = {
        "resolved_config": run_dir / "resolved_config.yaml",
        "provenance": run_dir / "provenance.json",
        "split_manifest": run_dir / "protocol" / "split_manifest.json",
        "fit_access_ledger": run_dir / "protocol" / "fit_access_ledger.json",
        "training_history": run_dir / "training" / "protected_koopman_ts.json",
        "training_checkpoint": run_dir / "training" / "protected_koopman_ts.pt",
        "checkpoint_replay": run_dir / "replay" / "protected_koopman_ts.json",
        "structure_selection": run_dir / "method" / "structure_selection.json",
        "monitoring_score_scaler": run_dir / "method" / "monitoring_score_scaler.json",
        "deterministic_envelope": run_dir / "method" / "deterministic_envelope.json",
        "innovation_covariance": run_dir / "method" / "innovation_covariance.json",
        "postfilter_library": run_dir / "method" / "postfilter_library.json",
        "monitor_policy": run_dir / "method" / "monitor_policy.json",
        "operator_bundle": run_dir / "method" / "operator_bundle.json",
        "detection_calibration": run_dir / "calibration" / "detection.json",
        "attribution_calibration": run_dir / "calibration" / "attribution.json",
        "isolation_library": run_dir / "method" / "isolation_library.json",
        "certification_status": run_dir / "method" / "certification_status.json",
    }
    base = resolve_frozen_evaluation_config(
        _ROOT / "configs" / "paper" / "cstr_development.yaml"
    ).config.model_dump(mode="json")
    base.update(
        {
            "artifact_root": str(tmp_path / "runs"),
            "run_name": "development",
            "claim_registry": str(tmp_path / "claim-registry"),
            "detection_risk": 0.3,
            "attribution_risk": 0.2,
        }
    )
    base["dataset"].update(
        {
            "root": str(data_root),
            "normal_file": "normal.mat",
            "fault_file": "must-not-be-opened.mat",
            "dataset_card": str(card_path),
            "dataset_card_sha256": card_hash,
            "license_status": "verified",
            "normal_rows": 800,
            "normal_source_hash": normal_hash,
            "fault_source_hash": _B,
        }
    )
    base["development"]["method"] = _normal_method_config_fixture()
    base["development"]["method"]["model"].update(
        {
            "control_dim": 2,
            "measurement_dim": 4,
            "exogenous_dim": 1,
            "history_length": 3,
            "max_rollout": 3,
        }
    )
    base["development"]["feature_layout"] = {
        "control_indices": [0, 1],
        "measurement_indices": [2, 3, 4, 5],
        "exogenous_indices": [6],
    }
    base["development"]["split"].update(
        {
            "episode_length": 8,
            "target_risk_level": 0.2,
            "seed": 19,
        }
    )
    base["development"]["training"] = {
        "epochs": 1,
        "batch_size": 128,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
    }
    base["normal_artifacts"] = {
        **{name: str(path) for name, path in artifact_paths.items()},
        "checkpoint_files": {
            "protected_koopman_ts": str(
                run_dir / "checkpoints" / "protected_koopman_ts.pt"
            )
        },
        "checkpoint_replay_outputs": {
            "frozen_normal": str(
                run_dir / "replay" / "frozen_normal_outputs.jsonl"
            )
        },
        "ledger_bindings": {
            "model": "training_checkpoint",
            "structure": "structure_selection",
            "score-scaler": "monitoring_score_scaler",
            "envelope": "deterministic_envelope",
            "covariance": "innovation_covariance",
            "branch-library": "postfilter_library",
            "state-machine": "monitor_policy",
            "q-det": "detection_calibration",
            "q-attr": "attribution_calibration",
            "normal-diagnostic": "checkpoint_replay_outputs.frozen_normal",
        },
    }
    resolved = resolve_frozen_evaluation_config(base)

    protocol_bundle = run_dir / "protocol" / "paper_data_bundle.json"
    protocol_bundle.parent.mkdir(parents=True)
    protocol_bundle.write_text('{"owner":"previous-run"}\n', encoding="utf-8")
    with pytest.raises(FileExistsError, match="paper_data_bundle"):
        run_cstr_normal_development(resolved, repo_root=_ROOT)
    assert protocol_bundle.read_text(encoding="utf-8") == '{"owner":"previous-run"}\n'
    protocol_bundle.unlink()

    result = run_cstr_normal_development(resolved, repo_root=_ROOT)

    assert len(result.artifact_paths) == 20
    assert all(path.is_file() for path in result.artifact_paths.values())
    assert not (data_root / "must-not-be-opened.mat").exists()
    checkpoint = torch.load(
        result.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    evaluator = checkpoint["extra_state"]["frozen_episode_evaluator"]
    assert evaluator["type"] == "protected_koopman_ts_frozen"
    assert checkpoint["extra_state"]["formal_pipeline_complete"] is False
    certification = json.loads(
        artifact_paths["certification_status"].read_text(encoding="utf-8")
    )
    assert certification["operator"]["status"] == "unavailable"
    ledger = json.loads(
        artifact_paths["fit_access_ledger"].read_text(encoding="utf-8")
    )
    assert ledger["protocol_ready"] is True
    assert len(ledger["records"]) == 10


def test_real_cstr_archive_inspection_cannot_load_values_with_smoke_manifest(
    tmp_path: Path,
) -> None:
    """MAT header 可冻结身份，但 synthetic manifest 不能授权读取真实 CSTR 数值。"""

    root = tmp_path / "fd_close"
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir(parents=True)
    (root / "7v 正序, 8c 故障.txt").write_text("synthetic fixture\n", encoding="utf-8")
    savemat(root / "train" / "model1[train].mat", {"normal": np.zeros((12, 7))})
    savemat(
        root / "test" / "model1[test].mat",
        {
            f"fault{fault_id:02d}": np.full((205, 7), fault_id, dtype=float)
            for fault_id in range(1, 9)
        },
    )
    inspection = inspect_closed_loop_cstr_archive(
        root,
        fault_onset=200,
        expected_feature_count=7,
    )
    assert inspection.normal_rows == 12
    assert [episode.episode_id for episode in inspection.fault_episodes] == [
        f"fault{fault_id:02d}" for fault_id in range(1, 9)
    ]
    assert {episode.row_count for episode in inspection.fault_episodes} == {205}

    manifest_path = tmp_path / "frozen_protocol_manifest.json"
    manifest = _freeze_manifest(
        manifest_path,
        normal_source_hash=inspection.normal_source_hash,
        fault_source_hash=inspection.fault_source_hash,
        fault_episode_manifest=inspection.fault_episodes,
    )
    loader = CSTRClosedLoopEpisodeLoader(inspection)
    source = ManifestBoundCSTRFaultSource(
        loader=loader,
        normal_source_hash=inspection.normal_source_hash,
        fault_source_hash=inspection.fault_source_hash,
    )
    claim = FrozenEvaluationClaim.create(
        manifest=manifest,
        claim_registry=tmp_path / "claim-registry",
        artifact_dir=tmp_path / "evaluation",
    )
    with pytest.raises(ProtocolAccessError, match="manifest itself"):
        source.request_episodes(manifest, claim=claim)
    assert not claim.claim_path.with_name(
        f"{manifest.evaluation_id}.fault-access.json"
    ).exists()


def test_formal_manifest_builder_binds_reviewed_normal_artifacts_before_fault_claim(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """formal fixture 可由 builder 冻结，再由公开 CLI 执行八 episode 且不能重读。"""

    data_root = tmp_path / "datasets" / "fd_close"
    (data_root / "train").mkdir(parents=True)
    (data_root / "test").mkdir(parents=True)
    (data_root / "7v 正序, 8c 故障.txt").write_text("fixture\n", encoding="utf-8")
    savemat(data_root / "train" / "model1[train].mat", {"normal": np.zeros((12, 7))})
    savemat(
        data_root / "test" / "model1[test].mat",
        {
            f"fault{fault_id:02d}": np.full((205, 7), fault_id, dtype=float)
            for fault_id in range(1, 9)
        },
    )
    inspection = inspect_closed_loop_cstr_archive(data_root, fault_onset=200)
    card_path, card_hash = _write_verified_dataset_card_fixture(
        data_root / ".paper-evidence" / "dataset_card.yaml",
        data_root=data_root,
        normal_file="train/model1[train].mat",
        fault_file="test/model1[test].mat",
        normal_sha256=inspection.normal_source_hash,
        fault_sha256=inspection.fault_source_hash,
    )
    artifacts = tmp_path / "normal-artifacts"
    checkpoint = artifacts / "protected.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(
        {
            "config": {"type": "deterministic_pointwise_fixture"},
            "model_state_dict": {"weight": torch.tensor([1.0])},
            "extra_state": {
                "frozen_episode_evaluator": {
                    "schema_version": 1,
                    "type": "deterministic_pointwise_fixture",
                    "state": {
                        "output_contract": "deterministic-pointwise-test-v1",
                    },
                },
            }
        },
        checkpoint,
    )
    monkeypatch.setattr(
        frozen_cli_module,
        "_RUNTIME_DRIVERS",
        {
            "protected_koopman_ts": ProtectedKoopmanTSCheckpointDriver(
                allowed_evaluator_types=("deterministic_pointwise_fixture",)
            )
        },
    )
    monkeypatch.setattr(
        paper_freeze_module,
        "current_clean_paper_git_commit",
        lambda _root: "1" * 40,
    )
    monkeypatch.setattr(
        frozen_cli_module,
        "current_clean_paper_git_commit",
        lambda _root: "1" * 40,
    )
    formal_checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    split_manifest, _ = _normal_protocol_evidence(
        normal_source_hash=inspection.normal_source_hash,
        fault_source_hash=inspection.fault_source_hash,
    )
    replay_output = artifacts / "frozen_normal_outputs.jsonl"
    paths = {
        "resolved_config": artifacts / "resolved_config.json",
        "provenance": artifacts / "provenance.json",
        "split_manifest": artifacts / "split_manifest.json",
        "fit_access_ledger": artifacts / "fit_access_ledger.json",
        "training_history": artifacts / "training_history.json",
        "training_checkpoint": artifacts / "training_checkpoint.pt",
        "checkpoint_replay": artifacts / "checkpoint_replay.json",
        "structure_selection": artifacts / "structure_selection.json",
        "monitoring_score_scaler": artifacts / "monitoring_score_scaler.json",
        "deterministic_envelope": artifacts / "deterministic_envelope.json",
        "innovation_covariance": artifacts / "innovation_covariance.json",
        "postfilter_library": artifacts / "postfilter_library.json",
        "monitor_policy": artifacts / "monitor_policy.json",
        "operator_bundle": artifacts / "operator_bundle.json",
        "detection_calibration": artifacts / "detection_calibration.json",
        "attribution_calibration": artifacts / "attribution_calibration.json",
        "isolation_library": artifacts / "isolation_library.json",
        "certification_status": artifacts / "certification_status.json",
    }
    _write_json_fixture(paths["resolved_config"], _normal_method_config_fixture())
    _write_json_fixture(
        paths["provenance"],
        _provenance_fixture(_normal_method_config_fixture()),
    )
    _write_json_fixture(
        paths["split_manifest"],
        split_manifest,
    )
    _write_json_fixture(paths["training_history"], {"epochs": [{"loss": 0.5}]})
    paths["training_checkpoint"].write_bytes(checkpoint.read_bytes())
    _write_json_fixture(
        paths["structure_selection"],
        {"candidate_id": "candidate-001", "source": "normal-only"},
    )
    _write_json_fixture(
        paths["monitoring_score_scaler"],
        {"schema_version": 1, "rms_scale": 1.0},
    )
    _write_json_fixture(
        paths["deterministic_envelope"],
        {
            "schema_version": 1,
            "status": "certified",
            "source": "fixture",
            "envelope": {"radius": 0.5},
        },
    )
    _write_json_fixture(
        paths["innovation_covariance"],
        {"schema_version": 1, "status": "frozen", "diagonal": [1.0]},
    )
    replay_output.write_text(
        '{"raw_index": 0, "score": 0.25}\n',
        encoding="utf-8",
    )
    _write_json_fixture(
        paths["postfilter_library"],
        {
            "candidate_id": "candidate-001",
            "mode": "regular",
            "branches": ["guard", "omnibus"],
        },
    )
    _write_json_fixture(
        paths["monitor_policy"],
        {
            "anchor_gate": {"source": "estimate", "gate_hash": _A},
            "hysteresis": {"enter": 2, "exit": 3},
            "reset_state": {"kind": "episode-boundary"},
        },
    )
    _write_json_fixture(
        paths["operator_bundle"],
        {"schema_version": 1, "status": "certified", "source": "fixture"},
    )
    _write_json_fixture(
        paths["detection_calibration"],
        _risk_calibration(
            "detection",
            quantile=3.5,
            source_hash=split_manifest["stages"]["detection_calibration"]["data_hash"],
            stage_manifest=split_manifest["stages"]["detection_calibration"],
        ).to_dict(),
    )
    _write_json_fixture(
        paths["attribution_calibration"],
        _risk_calibration(
            "attribution",
            quantile=4.5,
            source_hash=split_manifest["stages"]["attribution_calibration"]["data_hash"],
            stage_manifest=split_manifest["stages"]["attribution_calibration"],
        ).to_dict(),
    )
    _write_json_fixture(
        paths["isolation_library"],
        {
            "schema_version": 1,
            "normal_family": "normal",
            "fault_families": ["unresolved-fault"],
            "certified": True,
        },
    )
    _write_json_fixture(
        paths["certification_status"],
        {
            "operator": {
                "status": "certified",
                "artifact_hash": hashlib.sha256(
                    paths["operator_bundle"].read_bytes()
                ).hexdigest(),
            },
            "signature": {
                "status": "certified",
                "artifact_hash": hashlib.sha256(
                    paths["isolation_library"].read_bytes()
                ).hexdigest(),
            },
            "nuisance": {
                "status": "certified",
                "artifact_hash": hashlib.sha256(
                    paths["deterministic_envelope"].read_bytes()
                ).hexdigest(),
            },
        },
    )
    ledger_bindings = {
        "model": "training_checkpoint",
        "structure": "structure_selection",
        "score-scaler": "monitoring_score_scaler",
        "envelope": "deterministic_envelope",
        "covariance": "innovation_covariance",
        "branch-library": "postfilter_library",
        "state-machine": "monitor_policy",
        "q-det": "detection_calibration",
        "q-attr": "attribution_calibration",
        "normal-diagnostic": "checkpoint_replay_outputs.frozen_normal",
    }
    ledger_hashes = {
        object_id: (
            formal_checkpoint_hash
            if artifact_name == "checkpoint_files.protected_koopman_ts"
            else hashlib.sha256(replay_output.read_bytes()).hexdigest()
            if artifact_name == "checkpoint_replay_outputs.frozen_normal"
            else hashlib.sha256(paths[artifact_name].read_bytes()).hexdigest()
        )
        for object_id, artifact_name in ledger_bindings.items()
    }
    _, fit_access_ledger = _normal_protocol_evidence(
        normal_source_hash=inspection.normal_source_hash,
        fault_source_hash=inspection.fault_source_hash,
        artifact_hashes=ledger_hashes,
    )
    _write_json_fixture(paths["fit_access_ledger"], fit_access_ledger)
    _write_json_fixture(
        paths["checkpoint_replay"],
        {
            "status": "passed",
            "checkpoint_hashes": {
                "protected_koopman_ts": formal_checkpoint_hash
            },
            "output_hashes": {
                "frozen_normal": hashlib.sha256(
                    replay_output.read_bytes()
                ).hexdigest()
            },
        },
    )
    entry = FrozenEvaluationEntryConfig.model_validate(
        {
            "mode": "frozen",
            "protocol_version": "cstr-fixture-v1",
            "evaluation_id": "cstr-fixture-eval-001",
            "artifact_root": str(tmp_path / "runs"),
            "run_name": "cstr-fixture",
            "claim_registry": str(tmp_path / "claim-registry"),
            "device": "cpu",
            "runtime": "protected_koopman_ts",
            "detection_risk": 0.3,
            "attribution_risk": 0.2,
            "seeds": {"python": 1, "numpy": 2, "torch": 3, "dataloader": 4},
            "dataset": {
                "name": "cstr_closed_loop_fd",
                "root": str(data_root),
                "normal_file": "train/model1[train].mat",
                "fault_file": "test/model1[test].mat",
                "dataset_card": str(card_path),
                "dataset_card_sha256": card_hash,
                "license_status": "verified",
                "feature_count": 7,
                "normal_rows": 12,
                "fault_episode_count": 8,
                "fault_episode_rows": 205,
                "fault_onset": 200,
                "normal_source_hash": inspection.normal_source_hash,
                "fault_source_hash": inspection.fault_source_hash,
            },
            "normal_artifacts": {
                **{name: str(path) for name, path in paths.items()},
                "checkpoint_files": {"protected_koopman_ts": str(checkpoint)},
                "checkpoint_replay_outputs": {
                    "frozen_normal": str(replay_output)
                },
                "ledger_bindings": ledger_bindings,
            },
        }
    )
    resolved = resolve_frozen_evaluation_config(entry)
    assert resolved.config.frozen_readiness_errors(repo_root=_ROOT) == ()
    run_dir = tmp_path / "runs" / "cstr-fixture"
    manifest_path = run_dir / "frozen_protocol_manifest.json"
    config_path = tmp_path / "formal-config.json"
    config_path.write_text(
        json.dumps(entry.model_dump(mode="json"), ensure_ascii=False),
        encoding="utf-8",
    )

    with monkeypatch.context() as capability_patch:
        capability_patch.setattr(
            _RegisteredDeterministicFixtureEvaluator,
            "formal_pipeline_complete",
            False,
        )
        assert (
            frozen_cli_main(
                [
                    "--config",
                    str(config_path),
                    "--repo-root",
                    str(_ROOT),
                    "--preflight-only",
                ]
            )
            != 0
        )
        incomplete_runtime = json.loads(capsys.readouterr().out)
        assert incomplete_runtime["claim_created"] is False
        assert incomplete_runtime["fault_data_accessed"] is False
        assert any(
            "complete P4--P9" in error
            for error in incomplete_runtime["errors"]
        )

    original_checkpoint = checkpoint.read_bytes()
    original_replay = json.loads(paths["checkpoint_replay"].read_text(encoding="utf-8"))
    invalid_checkpoints = (
        {
            "extra_state": {
                "frozen_episode_evaluator": _DeterministicPointwiseEvaluator(),
            }
        },
        {
            "extra_state": {
                "frozen_episode_evaluator": {
                    "schema_version": 1,
                    "type": "not_in_formal_whitelist",
                    "state": {},
                }
            }
        },
        {
            "config": {"type": "deterministic_pointwise_fixture"},
            "model_state_dict": {"weight": torch.tensor([2.0])},
            "extra_state": {
                "frozen_episode_evaluator": {
                    "schema_version": 1,
                    "type": "deterministic_pointwise_fixture",
                    "state": {
                        "output_contract": "deterministic-pointwise-test-v1",
                    },
                }
            },
        },
    )
    for invalid_checkpoint in invalid_checkpoints:
        torch.save(invalid_checkpoint, checkpoint)
        invalid_replay = dict(original_replay)
        invalid_replay["checkpoint_hashes"] = {
            "protected_koopman_ts": hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        }
        _write_json_fixture(paths["checkpoint_replay"], invalid_replay)
        assert (
            frozen_cli_main(
                [
                    "--config",
                    str(config_path),
                    "--repo-root",
                    str(_ROOT),
                    "--preflight-only",
                ]
            )
            != 0
        )
        blocked_checkpoint = json.loads(capsys.readouterr().out)
        assert blocked_checkpoint["claim_created"] is False
        assert blocked_checkpoint["fault_data_accessed"] is False
        assert not manifest_path.exists()
        checkpoint.write_bytes(original_checkpoint)
        _write_json_fixture(paths["checkpoint_replay"], original_replay)

    original_envelope = json.loads(
        paths["deterministic_envelope"].read_text(encoding="utf-8")
    )
    original_certification = json.loads(
        paths["certification_status"].read_text(encoding="utf-8")
    )
    original_nuisance_ledger = json.loads(
        paths["fit_access_ledger"].read_text(encoding="utf-8")
    )
    contradictory_envelope = {
        **original_envelope,
        "status": "nominal",
    }
    _write_json_fixture(paths["deterministic_envelope"], contradictory_envelope)
    contradictory_certification = json.loads(json.dumps(original_certification))
    contradictory_certification["nuisance"]["artifact_hash"] = hashlib.sha256(
        paths["deterministic_envelope"].read_bytes()
    ).hexdigest()
    _write_json_fixture(paths["certification_status"], contradictory_certification)
    contradictory_nuisance_ledger = json.loads(
        json.dumps(original_nuisance_ledger)
    )
    for record in contradictory_nuisance_ledger["records"]:
        if record["object_id"] == "envelope":
            record["artifact_hash"] = hashlib.sha256(
                paths["deterministic_envelope"].read_bytes()
            ).hexdigest()
            break
    else:  # pragma: no cover - fixture 构造失败时给出明确原因
        raise AssertionError("envelope ledger fixture is missing")
    _write_json_fixture(
        paths["fit_access_ledger"],
        contradictory_nuisance_ledger,
    )
    assert (
        frozen_cli_main(
            [
                "--config",
                str(config_path),
                "--repo-root",
                str(_ROOT),
                "--preflight-only",
            ]
        )
        != 0
    )
    contradictory_nuisance = json.loads(capsys.readouterr().out)
    assert contradictory_nuisance["claim_created"] is False
    assert contradictory_nuisance["fault_data_accessed"] is False
    assert not manifest_path.exists()
    _write_json_fixture(paths["deterministic_envelope"], original_envelope)
    _write_json_fixture(paths["certification_status"], original_certification)
    _write_json_fixture(paths["fit_access_ledger"], original_nuisance_ledger)

    invalid_artifacts = (
        (
            paths["detection_calibration"],
            {
                **_risk_calibration(
                    "detection",
                    quantile=3.5,
                    source_hash=split_manifest["stages"]["train"]["data_hash"],
                    stage_manifest=split_manifest["stages"][
                        "detection_calibration"
                    ],
                ).to_dict(),
            },
        ),
        (
            paths["checkpoint_replay"],
            {
                "status": "synthetic_contract",
                "checkpoint_hashes": {
                    "protected_koopman_ts": formal_checkpoint_hash
                },
                "output_hashes": {"frozen_normal": _A},
            },
        ),
        (
            paths["certification_status"],
            {
                "operator": {
                    "status": "certified",
                    "artifact_hash": _A,
                },
                "signature": {
                    "status": "certified",
                    "artifact_hash": hashlib.sha256(
                        paths["isolation_library"].read_bytes()
                    ).hexdigest(),
                },
                "nuisance": {
                    "status": "certified",
                    "artifact_hash": hashlib.sha256(
                        paths["deterministic_envelope"].read_bytes()
                    ).hexdigest(),
                },
            },
        ),
        (
            paths["certification_status"],
            {
                "operator": {"status": "unavailable"},
                "signature": {"status": "uncertified"},
                "nuisance": {"status": "uncertified"},
                "unknown_field": "must-fail",
            },
        ),
    )
    for artifact_path, invalid_payload in invalid_artifacts:
        original_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        _write_json_fixture(artifact_path, invalid_payload)
        assert (
            frozen_cli_main(
                [
                    "--config",
                    str(config_path),
                    "--repo-root",
                    str(_ROOT),
                    "--preflight-only",
                ]
            )
            != 0
        )
        blocked = json.loads(capsys.readouterr().out)
        assert blocked["claim_created"] is False
        assert blocked["fault_data_accessed"] is False
        assert not manifest_path.exists()
        _write_json_fixture(artifact_path, original_payload)

    original_ledger = json.loads(paths["fit_access_ledger"].read_text(encoding="utf-8"))
    misbound_ledger = json.loads(json.dumps(original_ledger))
    for record in misbound_ledger["records"]:
        if record["object_id"] == "score-scaler":
            record["artifact_hash"] = hashlib.sha256(
                paths["detection_calibration"].read_bytes()
            ).hexdigest()
            break
    else:  # pragma: no cover - fixture 构造失败时给出明确原因
        raise AssertionError("score-scaler ledger fixture is missing")
    misbound_entry = entry.model_dump(mode="json")
    misbound_entry["normal_artifacts"]["ledger_bindings"][
        "score-scaler"
    ] = "detection_calibration"
    _write_json_fixture(paths["fit_access_ledger"], misbound_ledger)
    _write_json_fixture(config_path, misbound_entry)
    assert (
        frozen_cli_main(
            [
                "--config",
                str(config_path),
                "--repo-root",
                str(_ROOT),
                "--preflight-only",
            ]
        )
        != 0
    )
    semantic_mismatch = json.loads(capsys.readouterr().out)
    assert semantic_mismatch["claim_created"] is False
    assert semantic_mismatch["fault_data_accessed"] is False
    assert any(
        "monitoring_score_scaler" in error
        for error in semantic_mismatch["errors"]
    )
    _write_json_fixture(paths["fit_access_ledger"], original_ledger)
    _write_json_fixture(config_path, entry.model_dump(mode="json"))

    assert (
        frozen_cli_main(
            [
                "--config",
                str(config_path),
                "--repo-root",
                str(_ROOT),
                "--preflight-only",
            ]
        )
        == 0
    )
    ready = json.loads(capsys.readouterr().out)
    assert ready["status"] == "ready"
    assert not manifest_path.exists()
    assert not (tmp_path / "claim-registry").exists()

    manifest = freeze_cstr_protocol_from_artifacts(
        resolved,
        repo_root=_ROOT,
        manifest_path=manifest_path,
    )

    assert manifest.status == "frozen"
    assert manifest.resolved_config["mode"] == "frozen"
    assert manifest.checkpoint_replay["status"] == "passed"
    assert manifest.fault_episode_manifest == inspection.fault_episodes
    assert not (tmp_path / "claim-registry").exists()

    direct_registry = tmp_path / "direct-workflow-claims"
    direct_source = ManifestBoundCSTRFaultSource(
        loader=CSTRClosedLoopEpisodeLoader(inspection),
        normal_source_hash=inspection.normal_source_hash,
        fault_source_hash=inspection.fault_source_hash,
    )
    with monkeypatch.context() as capability_patch:
        capability_patch.setattr(
            _DeterministicPointwiseEvaluator,
            "formal_pipeline_complete",
            False,
        )
        with pytest.raises(FrozenProtocolIntegrityError, match="complete P4--P9"):
            FrozenEvaluationWorkflow(
                manifest_path=manifest_path,
                claim_registry=direct_registry,
                artifact_dir=tmp_path / "direct-incomplete",
                fault_source=direct_source,
                evaluator=_DeterministicPointwiseEvaluator(),
            ).run()
    assert not direct_registry.exists()

    with pytest.raises(FrozenProtocolIntegrityError, match="runtime identity"):
        FrozenEvaluationWorkflow(
            manifest_path=manifest_path,
            claim_registry=direct_registry,
            artifact_dir=tmp_path / "direct-unbound",
            fault_source=direct_source,
            evaluator=_DeterministicPointwiseEvaluator(),
        ).run()
    assert not direct_registry.exists()

    assert (
        frozen_cli_main(
            [
                "--config",
                str(config_path),
                "--repo-root",
                str(_ROOT),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed"
    assert payload["fault_data_accessed"] is True
    result = verify_frozen_evaluation_artifacts(
        manifest_path=manifest_path,
        receipt_path=run_dir / "frozen_evaluation" / "evaluation_receipt.json",
    )
    assert result.pointwise_row_count == 8 * 205
    claim_path = Path(manifest.claim_registry_path) / (
        f"{manifest.evaluation_id}.claim.json"
    )
    claim_payload = json.loads(claim_path.read_text(encoding="utf-8"))
    claim = FrozenEvaluationClaim(
        evaluation_id=manifest.evaluation_id,
        protocol_version=manifest.protocol_version,
        manifest_hash=manifest.manifest_hash,
        artifact_dir=Path(claim_payload["artifact_dir"]),
        claimed_at_utc=claim_payload["claimed_at_utc"],
        claim_path=claim_path.resolve(),
        claim_hash=hashlib.sha256(claim_path.read_bytes()).hexdigest(),
    )
    reconstructed_source = ManifestBoundCSTRFaultSource(
        loader=CSTRClosedLoopEpisodeLoader(inspection),
        normal_source_hash=inspection.normal_source_hash,
        fault_source_hash=inspection.fault_source_hash,
    )
    with pytest.raises(
        FrozenEvaluationAlreadyClaimedError,
        match="fault-data access",
    ):
        reconstructed_source.request_episodes(manifest, claim=claim)


def _write_json_fixture(path: Path, value: Mapping[str, Any]) -> None:
    """在 tmp_path 写测试 normal artifact；不用于生产代码或仓库固定产物。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
