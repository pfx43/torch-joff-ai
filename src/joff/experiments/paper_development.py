"""P10 CSTR 正常-only 开发产物生成入口。

文件用途：
    消除 ``cstr_development.yaml`` 与正式冻结产物之间的手工拼装步骤；只使用已核验许可
    的正常 MAT 数据，按 P2 五阶段边界训练 P4 模型并生成 P5--P9 正常证据。
主要职责：
    严格解析开发配置、验证正常文件身份、构造滑窗训练批、执行确定性 CPU 训练、拟合
    estimate-only 缩放/包络/协方差、两次独立校准、checkpoint evaluator envelope 和
    frozen-normal 重放，并冻结 fit access ledger。
关键输入与输出：
    输入是 ``ResolvedFrozenEvaluationConfig`` 的 development 模式和单个正常 MAT；
    输出是 ``PaperNormalArtifactsConfig`` 指定的 20 个文件，以及同目录 P2 bundle 摘要。
依赖与副作用：
    依赖 NumPy、PyTorch、PyYAML、MAT reader、P2 协议和 P4 模型注册表。运行会在
    ``artifact_root/run_name`` 内独占创建文件并执行 CPU 训练；不读取 fault MAT，不创建
    frozen manifest/claim，也不访问网络。
重要约束：
    数据许可和正常源 hash 必须先核实；所有拟合访问都通过 ``PaperDataBundle.data_for_fit``
    登记。当前没有认证 provider，因此 operator/signature/nuisance 明确写为不可用，
    checkpoint evaluator 标为 development-only，正式 CLI 会在 claim 前拒绝它。该运行
    不能作为论文故障性能结果。
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

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]

from joff.core.factory import build_evaluator, build_model
from joff.data import FaultLicenseStatus, FitPurpose, PaperDataBundle, StageName
from joff.data.sources.readers import read_mat_arrays

from .frozen_evaluation import FrozenProtocolIntegrityError, FrozenRiskCalibration
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


def run_cstr_normal_development(
    resolved: ResolvedFrozenEvaluationConfig,
    *,
    repo_root: str | Path,
) -> PaperDevelopmentResult:
    """执行一次全新、正常-only 的 CSTR 开发运行。

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
        文件、不创建 manifest/claim。
    """

    config = resolved.config
    if config.mode != "development":
        raise ValueError("Normal development runner requires mode='development'.")
    development = config.development
    artifacts = config.normal_artifacts
    if development is None or artifacts is None:
        raise ValueError("Development parameters and normal_artifacts are required.")
    if config.dataset.license_status != "verified":
        raise FrozenProtocolIntegrityError(
            "Normal CSTR development requires dataset license_status='verified'."
        )

    root = Path(repo_root).expanduser().resolve()
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
            "epochs": history,
            "fault_data_accessed": False,
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
            "checkpoint_hashes": {"protected_koopman_ts": checkpoint_hash},
            "output_hashes": {"frozen_normal": sha256_file(replay_output)},
            "model_reloaded_from_checkpoint": True,
            "comparison": "exact",
            "fault_data_accessed": False,
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
    """只读并合并正常 MAT 的有限二维数组，核对声明 geometry/hash。"""

    if not path.is_file():
        raise FileNotFoundError(f"Normal CSTR MAT is missing: {path}")
    observed_hash = sha256_file(path)
    if expected_hash is None or observed_hash != expected_hash:
        raise FrozenProtocolIntegrityError(
            "Normal CSTR MAT SHA-256 differs from the development config."
        )
    arrays = read_mat_arrays(path)
    if not arrays:
        raise ValueError("Normal CSTR MAT contains no numeric arrays.")
    values = np.concatenate(
        [np.asarray(array, dtype=float) for _, array in sorted(arrays.items())],
        axis=0,
    )
    if values.shape != (expected_rows, expected_features):
        raise ValueError(
            "Normal CSTR matrix geometry differs from config: "
            f"observed={values.shape}, expected={(expected_rows, expected_features)}."
        )
    if not np.isfinite(values).all():
        raise ValueError("Normal CSTR matrix must contain only finite values.")
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
        result = run_cstr_normal_development(
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
]
