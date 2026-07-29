"""P10/P11 CSTR 与 TTS 正常-only 开发产物生成入口。

文件用途：
    消除 CSTR/TTS development YAML 与正常产物之间的手工拼装步骤；只使用已核验许可
    的正常 MAT，按 P2 五阶段边界训练 P4 模型并生成 P5--P9 正常证据。
主要职责：
    严格解析开发配置、验证正常文件身份、构造滑窗训练批、执行确定性 CPU 训练、拟合
    estimate-only 缩放/包络/协方差、两次独立校准、checkpoint evaluator envelope 和
    frozen-normal 重放，并冻结 fit access ledger；TTS 运行开始前还要复验 CSTR 主配置锁。
关键输入与输出：
    输入是 ``ResolvedFrozenEvaluationConfig`` 的 development 模式和单个正常 MAT；
    输出是 ``PaperNormalArtifactsConfig`` 指定的 20 个文件，以及同目录 P2 bundle 摘要。
依赖与副作用：
    依赖 NumPy、PyTorch、PyYAML、MAT reader、P2 协议和 P4 模型注册表。运行会在
    ``artifact_root/run_name`` 内独占创建文件并执行 CPU 训练；不读取 fault MAT，不创建
    frozen manifest/claim，也不访问网络。
重要约束：
    数据许可和正常源 hash 必须先核实；所有拟合访问都通过 ``PaperDataBundle.data_for_fit``
    登记。TTS 必须先由真实 P10 formal manifest 证明 CSTR 正式冻结评价已完成，否则在
    数据/输出 I/O 前关闭。TTS/TE 结果不得回写 CSTR 选择。当前没有认证 provider，因此
    operator/signature/nuisance 明确写为不可用，checkpoint evaluator 标为
    development-only；该运行不能作为论文故障性能结果。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import argparse
import hashlib
import json
import math
import subprocess

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]

from joff.core.factory import build_evaluator, build_model
from joff.data import FaultLicenseStatus, FitPurpose, PaperDataBundle, StageName
from joff.data.sources.readers import read_mat_arrays

from .frozen_evaluation import (
    FrozenProtocolIntegrityError,
    FrozenProtocolManifest,
    FrozenRiskCalibration,
    verify_frozen_evaluation_artifacts,
)
from .paper_entrypoints import (
    PaperDevelopmentConfig,
    ResolvedFrozenEvaluationConfig,
    resolve_frozen_evaluation_config,
)
from .paper_environment import sha256_file


_PURPOSE_STAGES = {
    FitPurpose.MODEL_PARAMETERS: StageName.TRAIN,
    FitPurpose.STRUCTURE_SELECTION: StageName.TRAIN,
    FitPurpose.MONITORING_SCORE_SCALER: StageName.ESTIMATE,
    FitPurpose.ENVELOPE: StageName.ESTIMATE,
    FitPurpose.COVARIANCE: StageName.ESTIMATE,
    FitPurpose.BRANCH_LIBRARY: StageName.ESTIMATE,
    FitPurpose.STATE_MACHINE: StageName.ESTIMATE,
    FitPurpose.DETECTION_QUANTILE: StageName.DETECTION_CALIBRATION,
    FitPurpose.ATTRIBUTION_QUANTILE: StageName.ATTRIBUTION_CALIBRATION,
    FitPurpose.FROZEN_NORMAL_DIAGNOSTIC: StageName.FROZEN_NORMAL_TEST,
}


@dataclass(frozen=True)
class PaperDevelopmentResult:
    """返回正常开发运行的目录、checkpoint 和协议身份。"""

    run_dir: Path
    checkpoint_path: Path
    split_hash: str
    checkpoint_hash: str
    artifact_paths: Mapping[str, Path]


def run_paper_normal_development(
    resolved: ResolvedFrozenEvaluationConfig,
    *,
    repo_root: str | Path,
) -> PaperDevelopmentResult:
    """执行一次全新、正常-only 的 CSTR 或 TTS 开发运行。

    参数：
        resolved: development 模式严格配置。
        repo_root: 配置相对路径的解析根。
    返回：
        已写完并冻结账本的 ``PaperDevelopmentResult``。
    异常：
        模式/许可/hash/shape/训练/校准或目标文件边界非法时抛出 ``ValueError``、
        ``FrozenProtocolIntegrityError``、``FileExistsError`` 或底层 I/O 异常。
    副作用：
        读取一个正常 MAT；在受限运行目录中独占写产物并执行 CPU 训练。不读取 fault
        文件、不创建 manifest/claim。TTS 路径先复验 CSTR frozen 配置、Git 提交和真实
        formal manifest；阻塞态不会读取 TTS 数据或创建输出。
    """

    config = resolved.config
    if config.mode != "development":
        raise ValueError("Normal development runner requires mode='development'.")
    development = config.development
    artifacts = config.normal_artifacts
    if development is None or artifacts is None:
        raise ValueError("Development parameters and normal_artifacts are required.")
    root = Path(repo_root).expanduser().resolve()
    _validate_primary_protocol_lock(resolved, root=root)
    if config.dataset.license_status != "verified":
        raise FrozenProtocolIntegrityError(
            "Normal paper development requires dataset license_status='verified'."
        )

    run_dir = _resolve(root, config.artifact_root) / config.run_name
    artifact_paths = {
        name: _resolve(root, path)
        for name, path in artifacts.paths().items()
    }
    protocol_dir = artifact_paths["split_manifest"].parent
    protocol_paths = {
        "split_manifest": protocol_dir / "split_manifest.json",
        "fit_access_ledger": protocol_dir / "fit_access_ledger.json",
        "paper_data_bundle": protocol_dir / "paper_data_bundle.json",
    }
    if (
        protocol_paths["split_manifest"].resolve()
        != artifact_paths["split_manifest"]
        or protocol_paths["fit_access_ledger"].resolve()
        != artifact_paths["fit_access_ledger"]
    ):
        raise FrozenProtocolIntegrityError(
            "Configured protocol artifact names must be split_manifest.json and "
            "fit_access_ledger.json in one directory."
        )
    _validate_output_paths(
        run_dir,
        {**artifact_paths, "paper_data_bundle": protocol_paths["paper_data_bundle"]},
    )
    normal_path = _resolve(root, config.dataset.root) / str(config.dataset.normal_file)
    values = _load_normal_matrix(
        normal_path,
        expected_rows=config.dataset.normal_rows,
        expected_features=config.dataset.feature_count,
        expected_hash=config.dataset.normal_source_hash,
    )
    development_identity = {
        "dataset_name": config.dataset.name,
        "entry_config_hash": resolved.config_hash,
        "primary_protocol_lock": (
            None
            if config.primary_protocol_lock is None
            else config.primary_protocol_lock.model_dump(mode="json")
        ),
        "fault_data_accessed": False,
    }

    bundle = PaperDataBundle(
        values,
        config=development.split,
        normal_raw_indices=np.arange(values.shape[0], dtype=np.int64),
        normal_source_hash=sha256_file(normal_path),
        fault_license_status=FaultLicenseStatus.TO_VERIFY,
    )
    fit_data = {
        purpose: bundle.data_for_fit(object_id, purpose)
        for object_id, purpose in (
            ("model", FitPurpose.MODEL_PARAMETERS),
            ("structure", FitPurpose.STRUCTURE_SELECTION),
            ("score-scaler", FitPurpose.MONITORING_SCORE_SCALER),
            ("envelope", FitPurpose.ENVELOPE),
            ("covariance", FitPurpose.COVARIANCE),
            ("branch-library", FitPurpose.BRANCH_LIBRARY),
            ("state-machine", FitPurpose.STATE_MACHINE),
        )
    }

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seeds.torch)
        model = build_model(development.method.model.model_dump(mode="python"))
        history = _train_model(
            model,
            values=fit_data[FitPurpose.MODEL_PARAMETERS],
            stage=bundle.split_result.stage(StageName.TRAIN),
            development=development,
            dataloader_seed=config.seeds.dataloader,
        )

    estimate_rows = _score_stage(
        model,
        values=fit_data[FitPurpose.MONITORING_SCORE_SCALER],
        stage=bundle.split_result.stage(StageName.ESTIMATE),
        development=development,
    )
    score_scale = _rms_scale(row["raw_score"] for row in estimate_rows)
    _normalize_scores(estimate_rows, score_scale)

    _write_yaml_new(
        artifact_paths["resolved_config"],
        development.method.model_dump(mode="json"),
    )
    _write_json_new(
        artifact_paths["provenance"],
        _provenance(development.method.model_dump(mode="json")),
    )
    _write_json_new(
        artifact_paths["training_history"],
        {
            "schema_version": 1,
            "status": "development_only",
            **development_identity,
            "epochs": history,
        },
    )
    training_checkpoint = artifact_paths["training_checkpoint"]
    _prepare_new_file(training_checkpoint)
    with training_checkpoint.open("xb") as stream:
        torch.save(
            {
                "config": development.method.model.model_dump(mode="json"),
                "model_state_dict": model.state_dict(),
                "extra_state": {
                    "stage": "normal_train_only",
                    "development_identity": development_identity,
                    "fault_data_accessed": False,
                },
            },
            stream,
        )
    bundle.fit_access_ledger.freeze_record(
        "model",
        sha256_file(training_checkpoint),
    )
    _write_and_freeze(
        bundle,
        object_id="structure",
        path=artifact_paths["structure_selection"],
        payload={
            "schema_version": 1,
            "candidate_id": "configured-protected-koopman-ts",
            "selection_source": "normal_train_only",
            **development_identity,
            "model_config": development.method.model.model_dump(mode="json"),
        },
    )
    _write_and_freeze(
        bundle,
        object_id="score-scaler",
        path=artifact_paths["monitoring_score_scaler"],
        payload={
            "schema_version": 1,
            "kind": "residual-rms",
            "source": "normal_estimate_only",
            "rms_scale": score_scale,
            "score_count": len(estimate_rows),
        },
    )
    _write_and_freeze(
        bundle,
        object_id="envelope",
        path=artifact_paths["deterministic_envelope"],
        payload={
            "schema_version": 1,
            "status": "nominal",
            "source": "normal_estimate_only",
            "maximum_normal_score": max(row["score"] for row in estimate_rows),
        },
    )
    residuals = np.asarray([row["residual"] for row in estimate_rows], dtype=float)
    covariance = np.atleast_2d(np.cov(residuals, rowvar=False, ddof=1))
    _write_and_freeze(
        bundle,
        object_id="covariance",
        path=artifact_paths["innovation_covariance"],
        payload={
            "schema_version": 1,
            "status": "nominal",
            "source": "normal_estimate_only",
            "matrix": covariance.tolist(),
        },
    )
    _write_and_freeze(
        bundle,
        object_id="branch-library",
        path=artifact_paths["postfilter_library"],
        payload={
            "candidate_id": "configured-guard-branch",
            "mode": development.mode,
            "branches": [development.branch_id],
            "paper_method_implemented": False,
        },
    )
    gate_payload = development.anchor_gate.model_dump(mode="json")
    _write_and_freeze(
        bundle,
        object_id="state-machine",
        path=artifact_paths["monitor_policy"],
        payload={
            "anchor_gate": {
                "source": "normal_estimate_only",
                "gate_hash": _hash_json(gate_payload),
            },
            "hysteresis": {
                "enter": development.anchor_gate.confirmation_delay,
                "exit": development.anchor_gate.confirmation_delay,
            },
            "reset_state": {"kind": "episode-boundary", "age": 0},
        },
    )
    detection_values = bundle.data_for_fit(
        "q-det",
        FitPurpose.DETECTION_QUANTILE,
    )
    detection_rows = _score_stage(
        model,
        values=detection_values,
        stage=bundle.split_result.stage(StageName.DETECTION_CALIBRATION),
        development=development,
    )
    _normalize_scores(detection_rows, score_scale)
    detection_calibration = _calibrate(
        "detection",
        requested_risk=config.detection_risk,
        rows=detection_rows,
        stage=bundle.split_result.stage(StageName.DETECTION_CALIBRATION),
    )
    if not math.isfinite(detection_calibration.quantile):
        raise FrozenProtocolIntegrityError(
            "Development detection calibration lacks finite episode resolution."
        )
    _write_and_freeze(
        bundle,
        object_id="q-det",
        path=artifact_paths["detection_calibration"],
        payload=detection_calibration.to_dict(),
    )

    attribution_values = bundle.data_for_fit(
        "q-attr",
        FitPurpose.ATTRIBUTION_QUANTILE,
    )
    attribution_rows = _score_stage(
        model,
        values=attribution_values,
        stage=bundle.split_result.stage(StageName.ATTRIBUTION_CALIBRATION),
        development=development,
    )
    _normalize_scores(attribution_rows, score_scale)
    attribution_calibration = _calibrate(
        "attribution",
        requested_risk=config.attribution_risk,
        rows=attribution_rows,
        stage=bundle.split_result.stage(StageName.ATTRIBUTION_CALIBRATION),
    )
    _write_and_freeze(
        bundle,
        object_id="q-attr",
        path=artifact_paths["attribution_calibration"],
        payload=attribution_calibration.to_dict(),
    )

    replay_values = bundle.data_for_fit(
        "normal-diagnostic",
        FitPurpose.FROZEN_NORMAL_DIAGNOSTIC,
    )
    replay_rows = _score_stage(
        model,
        values=replay_values,
        stage=bundle.split_result.stage(StageName.FROZEN_NORMAL_TEST),
        development=development,
    )
    _normalize_scores(replay_rows, score_scale)
    _write_json_new(
        artifact_paths["operator_bundle"],
        {
            "schema_version": 1,
            "status": "unavailable",
            "reason": "no-certified-provider",
            "source": "normal-only-development",
        },
    )
    _write_json_new(
        artifact_paths["isolation_library"],
        {
            "schema_version": 1,
            "normal_family": development.normal_family_id,
            "fault_families": [development.unresolved_family_id],
            "certified": False,
        },
    )
    _write_json_new(
        artifact_paths["certification_status"],
        {
            "operator": {
                "status": "unavailable",
                "reason": "no certified interval or verified-quadrature provider",
            },
            "signature": {"status": "uncertified"},
            "nuisance": {"status": "uncertified"},
        },
    )

    layout = development.feature_layout
    estimate_measurements = fit_data[FitPurpose.MONITORING_SCORE_SCALER][
        :, layout.measurement_indices
    ]
    eligibility_center = np.mean(estimate_measurements, axis=0)
    eligibility_scale = np.maximum(np.std(estimate_measurements, axis=0), 1e-12)
    evaluator_state = {
        "schema_version": 1,
        "feature_layout": layout.model_dump(mode="json"),
        "anchor_gate": gate_payload,
        "eligibility_center": eligibility_center.tolist(),
        "eligibility_scale": eligibility_scale.tolist(),
        "score_scale": score_scale,
        "branch_id": development.branch_id,
        "mode": development.mode,
        "normal_family_id": development.normal_family_id,
        "unresolved_family_id": development.unresolved_family_id,
        "threshold": {
            "floor": development.threshold_floor,
            "gamma_anc": development.gamma_anc,
            "deterministic_intercept": development.deterministic_intercept,
            "input_l1_weight": development.input_l1_weight,
            "stochastic_quantile": detection_calibration.quantile,
        },
    }
    checkpoint_path = artifact_paths["checkpoint_files.protected_koopman_ts"]
    _prepare_new_file(checkpoint_path)
    with checkpoint_path.open("xb") as stream:
        torch.save(
            {
                "config": development.method.model.model_dump(mode="json"),
                "model_state_dict": model.state_dict(),
                "extra_state": {
                    "frozen_episode_evaluator": {
                        "schema_version": 1,
                        "type": "protected_koopman_ts_frozen",
                        "state": evaluator_state,
                    },
                    "development_identity": development_identity,
                    "formal_pipeline_complete": False,
                    "fault_data_accessed": False,
                },
            },
            stream,
        )
    checkpoint_hash = sha256_file(checkpoint_path)

    replay_evaluator = _restore_development_evaluator(checkpoint_path)
    replay_model = getattr(replay_evaluator, "model", None)
    if replay_model is None:
        raise FrozenProtocolIntegrityError(
            "Restored development evaluator does not expose its checkpoint model."
        )
    restored_replay_rows = _score_stage(
        replay_model,
        values=replay_values,
        stage=bundle.split_result.stage(StageName.FROZEN_NORMAL_TEST),
        development=development,
    )
    _normalize_scores(restored_replay_rows, score_scale)
    if restored_replay_rows != replay_rows:
        raise FrozenProtocolIntegrityError(
            "Checkpoint-restored frozen-normal outputs differ from the in-memory model."
        )

    replay_output = artifact_paths["checkpoint_replay_outputs.frozen_normal"]
    _write_jsonl_new(replay_output, restored_replay_rows)
    bundle.fit_access_ledger.freeze_record(
        "normal-diagnostic",
        sha256_file(replay_output),
    )
    _write_json_new(
        artifact_paths["checkpoint_replay"],
        {
            "status": "passed",
            **development_identity,
            "checkpoint_hashes": {"protected_koopman_ts": checkpoint_hash},
            "output_hashes": {"frozen_normal": sha256_file(replay_output)},
            "model_reloaded_from_checkpoint": True,
            "comparison": "exact",
        },
    )

    bundle.freeze_protocol(bundle.split_result.split_hash)
    # P2 的通用保存入口允许覆盖；development 产物必须更严格，三份协议证据都独占创建，
    # 防止残留摘要或并发运行被静默改写。
    _write_json_new(protocol_paths["split_manifest"], bundle.split_result.manifest())
    _write_json_new(
        protocol_paths["fit_access_ledger"],
        bundle.fit_access_ledger.manifest(),
    )
    _write_json_new(protocol_paths["paper_data_bundle"], bundle.manifest())
    missing = [
        name for name, path in artifact_paths.items() if not path.is_file()
    ]
    if missing:
        raise FrozenProtocolIntegrityError(
            "Normal development did not create required artifacts: "
            + ", ".join(sorted(missing))
        )
    return PaperDevelopmentResult(
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        split_hash=bundle.split_result.split_hash,
        checkpoint_hash=checkpoint_hash,
        artifact_paths=artifact_paths,
    )


def run_cstr_normal_development(
    resolved: ResolvedFrozenEvaluationConfig,
    *,
    repo_root: str | Path,
) -> PaperDevelopmentResult:
    """兼容 P10 调用名，并委托给数据集无关的正常开发入口。

    参数：
        resolved/repo_root: 与 ``run_paper_normal_development`` 相同。
    返回：
        通用入口生成的 ``PaperDevelopmentResult``。
    异常：
        原样传播通用入口的配置、完整性、训练和文件异常。
    副作用：
        与通用入口相同；不会额外读写任何文件。保留该名称只为已有 P10 脚本兼容。
    """

    return run_paper_normal_development(resolved, repo_root=repo_root)


def _validate_primary_protocol_lock(
    resolved: ResolvedFrozenEvaluationConfig,
    *,
    root: Path,
) -> None:
    """在次级数据集读数前复验 CSTR 配置、提交及正式 manifest 身份。

    参数：
        resolved: 已通过模式校验的 development 配置。
        root: 规范化仓库根。
    返回：
        无；CSTR 主开发，或 TTS 完成态全部证据完全匹配时静默返回。
    异常：
        TTS 缺锁、锁路径逃出仓库、文件/提交/manifest 缺失，或任一 SHA-256、协议、
        评价、正常产物 bundle 或 completion receipt/完整输出身份不一致时抛出
        ``FrozenProtocolIntegrityError`` 或 ``FrozenEvaluationArtifactError``。
    副作用：
        只读 CSTR YAML、Git object 数据库和完成态 manifest 引用的冻结产物；不读取任何
        TTS normal/fault MAT，不修改主配置、claim 或次级运行目录。
    """

    config = resolved.config
    if config.dataset.name != "tts_fault_diagnosis":
        return
    lock = config.primary_protocol_lock
    if lock is None:
        # Pydantic 模式校验应更早拒绝；此防御分支避免绕过严格解析的内部调用失守。
        raise FrozenProtocolIntegrityError(
            "TTS development requires a primary CSTR protocol lock."
        )
    frozen_config = _resolve(root, lock.frozen_config)
    try:
        frozen_config.relative_to(root)
    except ValueError as exc:
        raise FrozenProtocolIntegrityError(
            "TTS primary CSTR frozen config must remain inside the repository root."
        ) from exc
    if not frozen_config.is_file():
        raise FrozenProtocolIntegrityError(
            f"TTS primary CSTR frozen config is missing: {frozen_config}"
        )
    try:
        frozen_bytes = frozen_config.read_bytes()
    except OSError as exc:
        raise FrozenProtocolIntegrityError(
            "TTS primary CSTR frozen config cannot be read."
        ) from exc
    if hashlib.sha256(frozen_bytes).hexdigest() != lock.frozen_config_sha256:
        raise FrozenProtocolIntegrityError(
            "TTS primary CSTR frozen config SHA-256 differs from the declared lock."
        )
    try:
        frozen_value = yaml.safe_load(frozen_bytes.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as exc:
        raise FrozenProtocolIntegrityError(
            "TTS primary CSTR frozen config cannot be parsed after hash verification."
        ) from exc
    if not isinstance(frozen_value, Mapping):
        raise FrozenProtocolIntegrityError(
            "TTS primary CSTR frozen config must contain a top-level mapping."
        )
    dataset_value = frozen_value.get("dataset")
    if (
        frozen_value.get("protocol_version") != lock.protocol_version
        or not isinstance(dataset_value, Mapping)
        or dataset_value.get("name") != lock.dataset_name
    ):
        raise FrozenProtocolIntegrityError(
            "TTS primary CSTR frozen config identity differs from the declared lock."
        )
    _validate_primary_implementation_commit(
        root=root,
        implementation_commit=lock.implementation_commit,
        protected_frozen_config=frozen_config,
        protected_frozen_config_sha256=lock.frozen_config_sha256,
    )
    if lock.selection_status != "formal_cstr_frozen_evaluation_completed":
        raise FrozenProtocolIntegrityError(
            "P11 TTS development is stage-gated because the formal CSTR frozen evaluation "
            "has not completed."
        )

    # Pydantic 已保证完成态五项均非空；这里保留显式检查，避免内部构造或未来 schema
    # 迁移绕过文件 I/O 前的 fail-closed 边界。
    if (
        lock.evaluation_id is None
        or lock.manifest_path is None
        or lock.manifest_sha256 is None
        or lock.manifest_hash is None
        or lock.normal_artifact_bundle_hash is None
        or lock.receipt_path is None
        or lock.receipt_sha256 is None
    ):
        raise FrozenProtocolIntegrityError(
            "Completed TTS primary protocol lock lacks formal manifest/receipt evidence."
        )
    manifest_path = _resolve(root, lock.manifest_path)
    try:
        manifest_path.relative_to(root)
    except ValueError as exc:
        raise FrozenProtocolIntegrityError(
            "TTS primary CSTR manifest must remain inside the repository root."
        ) from exc
    if not manifest_path.is_file():
        raise FrozenProtocolIntegrityError(
            f"TTS primary CSTR formal manifest is missing: {manifest_path}"
        )
    if sha256_file(manifest_path) != lock.manifest_sha256:
        raise FrozenProtocolIntegrityError(
            "TTS primary CSTR formal manifest SHA-256 differs from the declared lock."
        )
    manifest = FrozenProtocolManifest.load(manifest_path)
    normal_artifacts = manifest.normal_artifacts
    resolved_dataset = manifest.resolved_config.get("dataset")
    if (
        manifest.protocol_version != lock.protocol_version
        or manifest.evaluation_id != lock.evaluation_id
        or manifest.git_commit != lock.implementation_commit
        or manifest.manifest_hash != lock.manifest_hash
        or not isinstance(resolved_dataset, Mapping)
        or resolved_dataset.get("name") != lock.dataset_name
    ):
        raise FrozenProtocolIntegrityError(
            "TTS primary CSTR formal manifest identity differs from the declared lock."
        )
    if (
        normal_artifacts is None
        or normal_artifacts.bundle_hash != lock.normal_artifact_bundle_hash
    ):
        raise FrozenProtocolIntegrityError(
            "TTS primary CSTR normal artifact bundle differs from the declared lock."
        )
    checkpoint_name = normal_artifacts.runtime_evaluator["checkpoint_name"]
    required_artifacts = {
        "training_checkpoint",
        "structure_selection",
        f"checkpoint_files.{checkpoint_name}",
    }
    if not required_artifacts.issubset(normal_artifacts.artifact_paths):
        raise FrozenProtocolIntegrityError(
            "TTS primary CSTR formal manifest does not bind training, structure and "
            "runtime checkpoint artifacts."
        )
    receipt_path = _resolve(root, lock.receipt_path)
    try:
        receipt_path.relative_to(root)
    except ValueError as exc:
        raise FrozenProtocolIntegrityError(
            "TTS primary CSTR evaluation receipt must remain inside the repository root."
        ) from exc
    if not receipt_path.is_file():
        raise FrozenProtocolIntegrityError(
            f"TTS primary CSTR evaluation receipt is missing: {receipt_path}"
        )
    if sha256_file(receipt_path) != lock.receipt_sha256:
        raise FrozenProtocolIntegrityError(
            "TTS primary CSTR evaluation receipt SHA-256 differs from the declared lock."
        )
    verified_evaluation = verify_frozen_evaluation_artifacts(
        manifest_path=manifest_path,
        receipt_path=receipt_path,
    )
    if (
        verified_evaluation.evaluation_id != lock.evaluation_id
        or verified_evaluation.manifest_hash != lock.manifest_hash
        or verified_evaluation.receipt_path.resolve() != receipt_path
    ):
        raise FrozenProtocolIntegrityError(
            "TTS primary CSTR completed evaluation identity differs from the declared lock."
        )


def _validate_primary_implementation_commit(
    *,
    root: Path,
    implementation_commit: str,
    protected_frozen_config: Path,
    protected_frozen_config_sha256: str,
) -> None:
    """确认主协议提交及其受保护 frozen 配置是真实、可达且内容一致。

    参数：
        root: 已规范化的仓库根。
        implementation_commit: P10 锁声明的完整 40 位提交。
        protected_frozen_config/protected_frozen_config_sha256: 当前已核验的主 YAML 路径和
            工作树文件 hash；提交中的同一路径必须产生完全相同的 blob。
    返回：
        无；对象存在、类型为 commit、为当前 HEAD 祖先，且配置 blob 相同时静默返回。
    异常：
        Git 不可执行、命令超时、对象缺失/类型错误、提交不在当前历史，或受保护配置
        不是该提交中的同一内容时抛出 ``FrozenProtocolIntegrityError``。
    副作用：
        只读 ``.git`` 对象和引用；不修改索引、工作树或分支。
    """

    commands = (
        (
            "resolve",
            [
                "git",
                "-C",
                str(root),
                "cat-file",
                "-e",
                f"{implementation_commit}^{{commit}}",
            ],
        ),
        (
            "ancestry",
            [
                "git",
                "-C",
                str(root),
                "merge-base",
                "--is-ancestor",
                implementation_commit,
                "HEAD",
            ],
        ),
    )
    for check_name, command in commands:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise FrozenProtocolIntegrityError(
                "Primary implementation commit cannot be verified by Git."
            ) from exc
        if completed.returncode != 0:
            raise FrozenProtocolIntegrityError(
                "Primary implementation commit failed Git "
                f"{check_name} verification."
            )
    try:
        protected_relative = protected_frozen_config.relative_to(root).as_posix()
    except ValueError as exc:
        raise FrozenProtocolIntegrityError(
            "Primary protected frozen config must remain inside the Git repository."
        ) from exc
    try:
        committed_config = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "show",
                f"{implementation_commit}:{protected_relative}",
            ],
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FrozenProtocolIntegrityError(
            "Primary protected frozen config cannot be verified by Git."
        ) from exc
    if (
        committed_config.returncode != 0
        or hashlib.sha256(committed_config.stdout).hexdigest()
        != protected_frozen_config_sha256
    ):
        raise FrozenProtocolIntegrityError(
            "Primary protected frozen config differs from the declared Git commit."
        )


def _train_model(
    model: Any,
    *,
    values: np.ndarray,
    stage: Any,
    development: PaperDevelopmentConfig,
    dataloader_seed: int,
) -> list[dict[str, Any]]:
    """用 train 阶段合法窗口执行显式预算的确定性 CPU 优化。"""

    tensors, _, _ = _window_tensors(values, stage=stage, development=development)
    sample_count = int(next(iter(tensors.values())).shape[0])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=development.training.learning_rate,
        weight_decay=development.training.weight_decay,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(dataloader_seed)
    history: list[dict[str, Any]] = []
    model.train()
    for epoch in range(development.training.epochs):
        order = torch.randperm(sample_count, generator=generator)
        total_loss = 0.0
        batches = 0
        for start in range(0, sample_count, development.training.batch_size):
            positions = order[start : start + development.training.batch_size]
            batch = {name: value[positions] for name, value in tensors.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss_output = model.compute_loss(batch, output)
            if isinstance(loss_output, Mapping):
                loss = loss_output["loss"]
            else:
                loss = loss_output
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                raise FrozenProtocolIntegrityError(
                    "Protected model training loss must be a scalar tensor."
                )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        history.append(
            {
                "epoch": epoch + 1,
                "mean_loss": total_loss / max(batches, 1),
                "batch_count": batches,
            }
        )
    return history


def _score_stage(
    model: Any,
    *,
    values: np.ndarray,
    stage: Any,
    development: PaperDevelopmentConfig,
) -> list[dict[str, Any]]:
    """按阶段合法窗口计算无标签预测残差和原始分数。"""

    tensors, raw_indices, episode_ids = _window_tensors(
        values,
        stage=stage,
        development=development,
    )
    rows: list[dict[str, Any]] = []
    model.eval()
    batch_size = development.training.batch_size
    with torch.no_grad():
        sample_count = int(raw_indices.shape[0])
        for start in range(0, sample_count, batch_size):
            stop = min(sample_count, start + batch_size)
            batch = {name: value[start:stop] for name, value in tensors.items()}
            output = model(batch)
            prediction = output["prediction"][:, 0, :]
            residual = batch["target_future"][:, 0, :] - prediction
            scores = torch.linalg.vector_norm(residual, dim=1)
            for offset in range(stop - start):
                index = start + offset
                rows.append(
                    {
                        "episode_id": episode_ids[index],
                        "raw_index": int(raw_indices[index]),
                        "raw_score": float(scores[offset].cpu()),
                        "residual": residual[offset].cpu().tolist(),
                    }
                )
    if not rows:
        raise FrozenProtocolIntegrityError(
            f"Stage {stage.stage.value!r} produced no legal development windows."
        )
    return rows


def _window_tensors(
    values: np.ndarray,
    *,
    stage: Any,
    development: PaperDevelopmentConfig,
) -> tuple[dict[str, torch.Tensor], np.ndarray, list[str]]:
    """把 P2 合法窗口转成 P4 具名 tensor，不跨越 gap 或 episode。"""

    model = development.method.model
    layout = development.feature_layout
    history = model.history_length
    horizon = model.max_rollout
    base = int(stage.prepared_row_indices[0])
    raw_by_local = np.asarray(stage.raw_indices, dtype=np.int64)
    episode_ranges = tuple(stage.prepared_episode_ranges)
    samples: dict[str, list[np.ndarray]] = {
        "past_u": [],
        "past_y": [],
        "future_u": [],
        "current_y": [],
        "target_future": [],
        "target_past_u": [],
        "target_past_y": [],
    }
    if model.exogenous_dim > 0:
        samples.update(
            {
                "past_xi": [],
                "future_xi": [],
                "target_past_xi": [],
                "target_current_xi": [],
            }
        )
    raw_indices: list[int] = []
    episode_ids: list[str] = []
    for prepared_start in stage.prepared_window_starts:
        local_start = int(prepared_start) - base
        anchor = local_start + history
        stop = anchor + horizon
        if local_start < 0 or stop > values.shape[0]:
            continue
        block = values[local_start:stop]
        past = block[:history]
        future = block[history:]
        samples["past_u"].append(past[:, layout.control_indices])
        samples["past_y"].append(past[:, layout.measurement_indices])
        samples["future_u"].append(future[:, layout.control_indices])
        samples["current_y"].append(past[-1, layout.measurement_indices])
        samples["target_future"].append(future[:, layout.measurement_indices])
        target_u = []
        target_y = []
        target_xi = []
        current_xi = []
        for step in range(horizon):
            target_history = values[
                local_start + step : local_start + step + history
            ]
            target_u.append(target_history[:, layout.control_indices])
            target_y.append(target_history[:, layout.measurement_indices])
            if model.exogenous_dim > 0:
                target_xi.append(target_history[:, layout.exogenous_indices])
                current_xi.append(
                    values[anchor + step, layout.exogenous_indices]
                )
        samples["target_past_u"].append(np.stack(target_u))
        samples["target_past_y"].append(np.stack(target_y))
        if model.exogenous_dim > 0:
            samples["past_xi"].append(past[:, layout.exogenous_indices])
            samples["future_xi"].append(future[:, layout.exogenous_indices])
            samples["target_past_xi"].append(np.stack(target_xi))
            samples["target_current_xi"].append(np.stack(current_xi))
        raw_indices.append(int(raw_by_local[anchor]))
        episode_ids.append(
            _episode_id(stage.stage, int(prepared_start), episode_ranges)
        )
    if not raw_indices:
        raise FrozenProtocolIntegrityError(
            f"Stage {stage.stage.value!r} has no P4-compatible legal windows."
        )
    tensors = {
        name: torch.as_tensor(np.stack(items), dtype=torch.float32)
        for name, items in samples.items()
    }
    return tensors, np.asarray(raw_indices, dtype=np.int64), episode_ids


def _episode_id(
    stage: StageName,
    prepared_start: int,
    ranges: tuple[tuple[int, int], ...],
) -> str:
    """把合法窗口绑定到一个完整 P2 episode 身份。"""

    for start, stop in ranges:
        if start <= prepared_start < stop:
            return f"{stage.value}-{start}-{stop}"
    return f"{stage.value}-window-{prepared_start}"


def _calibrate(
    name: str,
    *,
    requested_risk: float,
    rows: list[dict[str, Any]],
    stage: Any,
) -> FrozenRiskCalibration:
    """按完整 episode 最大分数执行有限样本上分位校准。"""

    maxima: dict[str, float] = {}
    for row in rows:
        episode_id = str(row["episode_id"])
        maxima[episode_id] = max(maxima.get(episode_id, 0.0), float(row["score"]))
    episode_ids = tuple(
        f"{name}-{start}-{stop}"
        for start, stop in stage.prepared_episode_ranges
    )
    observed = []
    for source_id, target_id in zip(
        tuple(
            f"{stage.stage.value}-{start}-{stop}"
            for start, stop in stage.prepared_episode_ranges
        ),
        episode_ids,
        strict=True,
    ):
        if source_id not in maxima:
            raise FrozenProtocolIntegrityError(
                f"Calibration episode {source_id!r} has no legal score."
            )
        observed.append((target_id, maxima[source_id]))
    rank = math.ceil((len(observed) + 1) * (1.0 - requested_risk))
    finite = rank <= len(observed)
    quantile = (
        sorted(value for _, value in observed)[rank - 1]
        if finite
        else math.inf
    )
    return FrozenRiskCalibration(
        name=name,  # type: ignore[arg-type]
        requested_risk=requested_risk,
        quantile=quantile,
        episode_count=len(observed),
        score_count=len(rows),
        attainable_risk_resolution=1.0 / (len(observed) + 1),
        status="finite" if finite else "unattainable",
        source_hash=stage.data_hash,
        episode_ids=tuple(item[0] for item in observed),
    )


def _load_normal_matrix(
    path: Path,
    *,
    expected_rows: int,
    expected_features: int,
    expected_hash: str | None,
) -> np.ndarray:
    """只读并合并论文开发 normal MAT 的有限二维数组，核对声明 geometry/hash。"""

    if not path.is_file():
        raise FileNotFoundError(f"Paper development normal MAT is missing: {path}")
    observed_hash = sha256_file(path)
    if expected_hash is None or observed_hash != expected_hash:
        raise FrozenProtocolIntegrityError(
            "Paper development normal MAT SHA-256 differs from the config."
        )
    arrays = read_mat_arrays(path)
    if not arrays:
        raise ValueError("Paper development normal MAT contains no numeric arrays.")
    values = np.concatenate(
        [np.asarray(array, dtype=float) for _, array in sorted(arrays.items())],
        axis=0,
    )
    if values.shape != (expected_rows, expected_features):
        raise ValueError(
            "Paper development normal matrix geometry differs from config: "
            f"observed={values.shape}, expected={(expected_rows, expected_features)}."
        )
    if not np.isfinite(values).all():
        raise ValueError("Paper development normal matrix must contain only finite values.")
    return values


def _validate_output_paths(
    run_dir: Path,
    artifact_paths: Mapping[str, Path],
) -> None:
    """要求全部目标位于同一新运行目录且不存在，避免覆盖或越界写入。"""

    resolved_run = run_dir.resolve()
    for name, path in artifact_paths.items():
        if not path.is_relative_to(resolved_run):
            raise ValueError(
                f"Development artifact {name!r} escapes run directory {resolved_run}."
            )
        if path.exists():
            raise FileExistsError(f"Development artifact already exists: {path}")


def _write_and_freeze(
    bundle: PaperDataBundle,
    *,
    object_id: str,
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    """独占写一个拟合产物，并把实际文件 hash 冻结回 P2 ledger。"""

    _write_json_new(path, payload)
    bundle.fit_access_ledger.freeze_record(object_id, sha256_file(path))


def _normalize_scores(rows: list[dict[str, Any]], scale: float) -> None:
    """原地补充运行时实际使用的归一化 score；输入只在当前函数链内部持有。"""

    for row in rows:
        row["score"] = float(row["raw_score"]) / scale


def _restore_development_evaluator(checkpoint_path: Path) -> Any:
    """从刚写入的 weights-only checkpoint 通过公共 registry 恢复 development evaluator。

    参数：
        checkpoint_path: 已独占写入且尚未登记为 replay 通过的 runtime checkpoint。
    返回：
        公共 evaluator registry 构造的对象；调用方继续用其实际恢复模型重算正常输出。
    异常：
        checkpoint/envelope 不是严格 mapping、缺少字段或注册 evaluator 无法恢复时，
        抛出 ``FrozenProtocolIntegrityError`` 或底层严格配置异常。
    副作用：
        只读 checkpoint 并在 CPU 构造模型；不读取故障数据、不写产物。
    """

    # 该 evaluator 属于 experiments 的正式运行时，不在通用 joff.evaluation 内。按需导入
    # 保证直接调用 development API/CLI 时也完成显式注册，而不依赖其他测试先导入 frozen CLI。
    from . import frozen_runtime as _frozen_runtime  # noqa: F401

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise FrozenProtocolIntegrityError(
            "Development runtime checkpoint must contain a mapping."
        )
    extra_state = checkpoint.get("extra_state")
    if not isinstance(extra_state, Mapping):
        raise FrozenProtocolIntegrityError(
            "Development runtime checkpoint is missing extra_state."
        )
    envelope = extra_state.get("frozen_episode_evaluator")
    if not isinstance(envelope, Mapping) or set(envelope) != {
        "schema_version",
        "type",
        "state",
    }:
        raise FrozenProtocolIntegrityError(
            "Development evaluator envelope must contain schema_version, type and state."
        )
    if envelope["schema_version"] != 1:
        raise FrozenProtocolIntegrityError(
            "Development evaluator envelope schema_version must be 1."
        )
    evaluator_type = envelope["type"]
    state = envelope["state"]
    if not isinstance(evaluator_type, str) or not isinstance(state, Mapping):
        raise FrozenProtocolIntegrityError(
            "Development evaluator type/state must be a string and mapping."
        )
    return build_evaluator(
        {
            "type": evaluator_type,
            "state": dict(state),
            "checkpoint": checkpoint,
        }
    )


def _rms_scale(values: Sequence[float] | Any) -> float:
    """返回正的有限 RMS；零残差时使用机器 epsilon，避免除零。"""

    array = np.asarray(tuple(float(value) for value in values), dtype=float)
    scale = float(np.sqrt(np.mean(np.square(array))))
    if not math.isfinite(scale):
        raise FrozenProtocolIntegrityError("Estimate residual RMS is not finite.")
    return max(scale, float(np.finfo(np.float64).eps))


def _provenance(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """为 normal method resolved config 的每个叶字段生成值匹配来源。"""

    records: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping) and item:
            records.update(_provenance(item, path))
        else:
            records[path] = [{"source": "development_config", "value": item}]
    return records


def _hash_json(value: Mapping[str, Any]) -> str:
    """计算与 manifest/evaluator state 相同的稳定 JSON SHA-256。"""

    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _prepare_new_file(path: Path) -> None:
    """创建父目录并拒绝复用已有目标。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite development artifact: {path}")


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    """以 exclusive-create 写严格 JSON object。"""

    _prepare_new_file(path)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    # ``Path.write_text`` 会截断并覆盖竞态期间出现的同名文件；``x`` 模式把拒绝覆盖
    # 下沉到操作系统的独占创建语义，即使两个 development 进程同时启动也只允许一个成功。
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def _write_yaml_new(path: Path, value: Mapping[str, Any]) -> None:
    """以 exclusive-create 写 UTF-8 YAML。"""

    _prepare_new_file(path)
    payload = yaml.safe_dump(dict(value), allow_unicode=True, sort_keys=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def _write_jsonl_new(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """以 exclusive-create 写 frozen-normal 逐窗口重放来源。"""

    _prepare_new_file(path)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    {
                        "episode_id": row["episode_id"],
                        "raw_index": row["raw_index"],
                        "score": row["score"],
                        "residual": row["residual"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            stream.write("\n")


def _resolve(root: Path, value: Path | None) -> Path:
    """把配置相对路径解析到仓库根。"""

    if value is None:
        return root
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    """公开 ``joff-paper-development`` 命令；失败时输出无故障访问的 JSON。"""

    parser = argparse.ArgumentParser(
        description="Generate normal-only P2-P9 artifacts for the frozen paper workflow."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        resolved = resolve_frozen_evaluation_config(args.config)
        result = run_paper_normal_development(
            resolved,
            repo_root=args.repo_root,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "errors": [str(exc)],
                    "fault_data_accessed": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": "completed_development_only",
                "run_dir": str(result.run_dir),
                "split_hash": result.split_hash,
                "checkpoint_hash": result.checkpoint_hash,
                "artifact_count": len(result.artifact_paths),
                "formal_fault_ready": False,
                "fault_data_accessed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "PaperDevelopmentResult",
    "main",
    "run_cstr_normal_development",
    "run_paper_normal_development",
]
