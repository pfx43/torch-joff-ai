"""P10 synthetic contract smoke 的可运行端到端编排。

文件用途：
    在 CPU 上用确定性合成正常/故障 episode 演练 P2 五段账本、P10 manifest、lazy fault
    gate、八 episode 逐时刻评价、机器来源和完整性重放，作为正式 CSTR 入口的最小验收。
主要职责：
    生成不含真实数据的合成 recipe，完成最小正常拟合账本并写审计占位产物；构造明确标为
    ``synthetic_contract_smoke`` 的 evaluator；调用 ``FrozenEvaluationWorkflow`` 并再次
    复验 receipt。本文件不训练论文模型、不加载 CSTR MAT、不提供正式性能结论。
关键输入与输出：
    输入为解析后的 ``configs/paper/smoke.yaml``；输出位于 ``artifact_root/run_name``，
    包含 resolved config、provenance、合成 checkpoint、P2 protocol、冻结 manifest 和
    P10 评价产物。
依赖与副作用：
    依赖 NumPy、PyTorch 版本元数据、PyYAML 和 Joff P2/P10 公共对象；创建一个新运行目录
    及共享 claim。使用局部 NumPy generator，不改变全局随机状态，不访问网络。
重要约束：
    只接受 ``mode='smoke'`` 与 ``runtime='synthetic_contract_smoke'``。所有输出都只是
    接口/产物合同证据，operator/signature/nuisance 固定为 unavailable/uncertified，
    不得写入论文主结果或用来解除真实数据许可阻塞。
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import hashlib
import json

import numpy as np
import yaml  # type: ignore[import-untyped]

from joff.data import (
    FaultLicenseStatus,
    FitPurpose,
    FiveStageSplitConfig,
    PaperDataBundle,
    StageName,
)

from .frozen_evaluation import (
    FrozenEpisodeInput,
    FrozenEvaluationResult,
    FrozenEvaluationWorkflow,
    FrozenFaultEpisode,
    FrozenFaultEpisodeManifest,
    FrozenProtocolManifest,
    FrozenRiskCalibration,
    FrozenRuntimeEpisodeEvaluation,
    FrozenRuntimePointwiseOutput,
    LazyFrozenCSTRFaultSource,
    verify_frozen_evaluation_artifacts,
)
from .paper_entrypoints import ResolvedFrozenEvaluationConfig
from .paper_environment import (
    collect_paper_dependency_versions,
    current_paper_git_commit,
    sha256_file,
)


_CSTR_FAMILIES = (
    "process",
    "process",
    "actuator",
    "actuator",
    "actuator",
    "sensor",
    "sensor",
    "sensor",
)


def run_paper_smoke(
    resolved: ResolvedFrozenEvaluationConfig,
) -> FrozenEvaluationResult:
    """执行一次全新 synthetic P10 smoke 并返回已复验产物。

    参数：
        resolved: 已解析且可追溯的 smoke 入口配置。
    返回：
        ``FrozenEvaluationResult``；返回前已经重新读取 manifest/receipt 并核对所有 hash。
    异常：
        模式错误、运行目录已有内容、配置几何不满足 P2、manifest/评价不一致或 I/O 失败时
        传播 ``ValueError``/``FileExistsError``/协议异常。
    副作用：
        创建运行目录、P2 正常审计文件、冻结 manifest、claim 和 synthetic 评价产物。
    """

    config = resolved.config
    if config.mode != "smoke" or config.runtime != "synthetic_contract_smoke":
        raise ValueError("run_paper_smoke accepts only the synthetic smoke entry config.")
    artifact_root = config.artifact_root.expanduser().resolve()
    run_dir = (artifact_root / config.run_name).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Paper smoke run directory already contains artifacts: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_new_yaml(run_dir / "resolved_config.yaml", resolved.resolved_config)
    _write_new_json(
        run_dir / "provenance.json",
        {
            path: [dict(record) for record in records]
            for path, records in resolved.provenance.items()
        },
    )

    dataset = config.dataset
    normal_source_hash = _recipe_hash(
        {
            "kind": "synthetic_normal",
            "rows": dataset.normal_rows,
            "features": dataset.feature_count,
            "seed": config.seeds.numpy,
        }
    )
    fault_source_hash = _recipe_hash(
        {
            "kind": "synthetic_fault_library",
            "episodes": dataset.fault_episode_count,
            "rows": dataset.fault_episode_rows,
            "features": dataset.feature_count,
            "onset": dataset.fault_onset,
            "seed": config.seeds.numpy,
        }
    )
    normal_values = _synthetic_normal_values(
        rows=dataset.normal_rows,
        features=dataset.feature_count,
        seed=config.seeds.numpy,
    )
    split_config = FiveStageSplitConfig(
        history_length=2,
        max_rollout=1,
        stacked_window=1,
        mask_recompute_span=0,
        episode_length=4,
        target_risk_level=min(config.detection_risk, config.attribution_risk),
        seed=config.seeds.numpy,
    )
    bundle = PaperDataBundle(
        normal_values,
        config=split_config,
        normal_source_hash=normal_source_hash,
        fault_source_hash=fault_source_hash,
        fault_license_status=FaultLicenseStatus.VERIFIED,
    )
    normal_artifact_dir = run_dir / "normal_artifacts"
    checkpoint_path = normal_artifact_dir / "synthetic_contract_checkpoint.json"
    checkpoint_hash = _write_audit_artifact(
        checkpoint_path,
        {
            "runtime": "synthetic_contract_smoke",
            "paper_method_implemented": False,
            "formal_fault_results": False,
            "seed": config.seeds.torch,
        },
    )
    _record_fit(
        bundle,
        object_id="synthetic-contract-runtime",
        purpose=FitPurpose.MODEL_PARAMETERS,
        artifact_hash=checkpoint_hash,
    )
    for object_id, purpose, payload in (
        (
            "synthetic-score-scaler",
            FitPurpose.MONITORING_SCORE_SCALER,
            {"kind": "identity", "source": "estimate"},
        ),
        (
            "synthetic-q-det",
            FitPurpose.DETECTION_QUANTILE,
            {"kind": "finite-episode", "risk": config.detection_risk},
        ),
        (
            "synthetic-q-attr",
            FitPurpose.ATTRIBUTION_QUANTILE,
            {"kind": "full-normal", "risk": config.attribution_risk},
        ),
        (
            "synthetic-normal-diagnostic",
            FitPurpose.FROZEN_NORMAL_DIAGNOSTIC,
            {"kind": "contract-replay", "paper_result": False},
        ),
    ):
        artifact_hash = _write_audit_artifact(
            normal_artifact_dir / f"{object_id}.json",
            payload,
        )
        _record_fit(
            bundle,
            object_id=object_id,
            purpose=purpose,
            artifact_hash=artifact_hash,
        )
    bundle.freeze_protocol(bundle.split_result.split_hash)
    bundle.save_protocol_artifacts(run_dir / "normal_protocol")

    detection_stage = bundle.split_result.stage(StageName.DETECTION_CALIBRATION)
    attribution_stage = bundle.split_result.stage(StageName.ATTRIBUTION_CALIBRATION)
    detection_calibration = _smoke_risk_calibration(
        name="detection",
        requested_risk=config.detection_risk,
        quantile=1.0,
        stage_hash=detection_stage.data_hash,
        episode_ranges=detection_stage.prepared_episode_ranges,
    )
    attribution_calibration = _smoke_risk_calibration(
        name="attribution",
        requested_risk=config.attribution_risk,
        quantile=1.5,
        stage_hash=attribution_stage.data_hash,
        episode_ranges=attribution_stage.prepared_episode_ranges,
    )
    fault_manifests = _fault_manifests(
        rows=dataset.fault_episode_rows,
        onset=dataset.fault_onset,
        source_hash=fault_source_hash,
    )
    manifest_path = run_dir / "frozen_protocol_manifest.json"
    manifest = FrozenProtocolManifest.freeze(
        manifest_path,
        protocol_version=config.protocol_version,
        evaluation_id=config.evaluation_id,
        git_commit=current_paper_git_commit(),
        resolved_config=resolved.resolved_config,
        config_provenance={
            path: [dict(record) for record in records]
            for path, records in resolved.provenance.items()
        },
        config_hash=resolved.config_hash,
        claim_registry_path=config.claim_registry,
        dependency_versions=collect_paper_dependency_versions(),
        raw_data_hashes={
            "normal": normal_source_hash,
            "fault": fault_source_hash,
        },
        split_manifest=bundle.split_result.manifest(),
        fault_episode_manifest=fault_manifests,
        seeds=config.seeds.model_dump(),
        checkpoint_paths={"synthetic_contract_smoke": checkpoint_path},
        checkpoint_hashes={"synthetic_contract_smoke": checkpoint_hash},
        checkpoint_replay={
            "status": "synthetic_contract",
            "checkpoint_hashes": {
                "synthetic_contract_smoke": checkpoint_hash
            },
            "output_hashes": {
                "contract_replay": _recipe_hash(
                    {
                        "runtime": "synthetic_contract_smoke",
                        "checkpoint_hash": checkpoint_hash,
                        "replay": "identity-contract",
                    }
                )
            },
        },
        fit_access_ledger=bundle.fit_access_ledger.manifest(),
        normal_artifacts=None,
        postfilter_library={
            "candidate_id": "synthetic-guard-candidate",
            "mode": "guard-only",
            "branches": ["guard"],
            "runtime": "synthetic_contract_smoke",
            "paper_method_implemented": False,
        },
        monitor_policy={
            "anchor_gate": {"kind": "synthetic-fixed", "source": "estimate"},
            "hysteresis": {"enter": 1, "exit": 1},
            "reset_state": {"kind": "episode-boundary", "age": 0},
        },
        detection_calibration=detection_calibration,
        attribution_calibration=attribution_calibration,
        certification_status={
            "operator": {
                "status": "unavailable",
                "reason": "synthetic contract smoke has no certified provider",
            },
            "signature": {"status": "uncertified"},
            "nuisance": {"status": "uncertified"},
        },
    )

    def load_fault_episodes() -> tuple[FrozenFaultEpisode, ...]:
        """在 lazy gate 之后按冻结 recipe 生成 synthetic fault episode。"""

        return _synthetic_fault_episodes(
            manifests=manifest.fault_episode_manifest,
            features=dataset.feature_count,
            seed=config.seeds.numpy,
        )

    source = LazyFrozenCSTRFaultSource(
        bundle=bundle,
        loader=load_fault_episodes,
        fault_source_hash=fault_source_hash,
    )
    result = FrozenEvaluationWorkflow(
        manifest_path=manifest_path,
        claim_registry=config.claim_registry.expanduser().resolve(),
        artifact_dir=run_dir / "frozen_evaluation",
        fault_source=source,
        evaluator=_SyntheticContractEvaluator(),
    ).run()
    verified = verify_frozen_evaluation_artifacts(
        manifest_path=manifest_path,
        receipt_path=result.receipt_path,
    )
    if verified != result:
        raise RuntimeError("Synthetic paper smoke receipt replay changed the result identity.")
    return result


class _SyntheticContractEvaluator:
    """只验证 P10 输出合同的确定性 evaluator，不实现论文模型。"""

    def evaluate_episode(
        self,
        episode: FrozenEpisodeInput,
    ) -> FrozenRuntimeEpisodeEvaluation:
        """只按无标签行号生成 contract 输出，不读取 onset 或故障身份。"""

        threshold = 1.0
        outputs: list[FrozenRuntimePointwiseOutput] = []
        for raw_index in episode.raw_indices.tolist():
            score = float(raw_index)
            alarm = score > threshold
            outputs.append(
                FrozenRuntimePointwiseOutput(
                    raw_index=int(raw_index),
                    detection_score=score,
                    detection_threshold=threshold,
                    alarm=alarm,
                    branch_id="guard",
                    mode="guard-only",
                    normal_family_id="normal",
                    candidate_ids=(
                        ("unresolved-fault",) if alarm else ("normal",)
                    ),
                    isolation_outcome=(
                        "Uncertified" if alarm else "Normal-compatible"
                    ),
                    reported_family=None,
                    isolation_certified=False,
                    suppression_reason=None,
                    method_outputs=_synthetic_method_outputs(
                        raw_index=int(raw_index),
                        threshold=threshold,
                        alarm=alarm,
                    ),
                )
            )
        return FrozenRuntimeEpisodeEvaluation(outputs=tuple(outputs))


def _synthetic_method_outputs(
    *,
    raw_index: int,
    threshold: float,
    alarm: bool,
) -> dict[str, Any]:
    """构造字段完整但明确不含真实 P4--P9 数值的逐时刻来源。"""

    return {
        "prediction": {
            "runtime": "synthetic_contract_smoke",
            "one_step": [0.0],
            "multi_step": [0.0],
        },
        "rule_weights": {"weights": [1.0], "learned": False},
        "monitor": {"mode": "guard-only", "anchor_raw_index": 0},
        "protected_state": {"context_age": raw_index, "synthetic": True},
        "residual": {"stacked": [float(raw_index)], "synthetic": True},
        "operator": {"status": "unavailable", "certified": False},
        "branch_statistics": {"guard": float(raw_index)},
        "threshold": {
            "floor": threshold,
            "gamma_anc": 0.0,
            "gamma_det": 0.0,
            "stochastic": 0.0,
            "alarm": alarm,
        },
        "isolation": {
            "outcome": "Uncertified" if alarm else "Normal-compatible",
            "certified": False,
        },
    }


def _record_fit(
    bundle: PaperDataBundle,
    *,
    object_id: str,
    purpose: FitPurpose,
    artifact_hash: str,
) -> None:
    """通过 P2 公开访问路径登记并冻结一项 synthetic 正常产物。"""

    bundle.data_for_fit(object_id, purpose)
    bundle.fit_access_ledger.freeze_record(object_id, artifact_hash)


def _smoke_risk_calibration(
    *,
    name: str,
    requested_risk: float,
    quantile: float,
    stage_hash: str,
    episode_ranges: tuple[tuple[int, int], ...],
) -> FrozenRiskCalibration:
    """从 P2 完整 episode 半开区间构造可重放 smoke 风险摘要。"""

    if name not in {"detection", "attribution"}:
        raise ValueError("Synthetic risk name must be detection or attribution.")
    episode_count = len(episode_ranges)
    return FrozenRiskCalibration(
        name=name,  # type: ignore[arg-type]
        requested_risk=requested_risk,
        quantile=quantile,
        episode_count=episode_count,
        score_count=sum(end - start for start, end in episode_ranges),
        attainable_risk_resolution=1.0 / (episode_count + 1),
        status="finite",
        source_hash=stage_hash,
        episode_ids=tuple(
            f"{name}-{start}-{end}"
            for start, end in episode_ranges
        ),
    )


def _fault_manifests(
    *,
    rows: int,
    onset: int,
    source_hash: str,
) -> tuple[FrozenFaultEpisodeManifest, ...]:
    """冻结八个 synthetic CSTR episode 的身份和物理族。"""

    return tuple(
        FrozenFaultEpisodeManifest(
            episode_id=f"fault-{fault_id}",
            fault_id=fault_id,
            fault_family=_CSTR_FAMILIES[fault_id - 1],  # type: ignore[arg-type]
            onset=onset,
            row_count=rows,
            raw_index_start=0,
            raw_index_end=rows - 1,
            source_hash=source_hash,
        )
        for fault_id in range(1, 9)
    )


def _synthetic_normal_values(
    *,
    rows: int,
    features: int,
    seed: int,
) -> np.ndarray:
    """使用局部 RNG 生成只含正常、数值稳定的二维序列。"""

    generator = np.random.default_rng(seed)
    time = np.arange(rows, dtype=float)[:, None]
    frequencies = np.arange(1, features + 1, dtype=float)[None, :]
    noise = generator.normal(loc=0.0, scale=0.001, size=(rows, features))
    return np.sin(time * frequencies / 37.0) + noise


def _synthetic_fault_episodes(
    *,
    manifests: tuple[FrozenFaultEpisodeManifest, ...],
    features: int,
    seed: int,
) -> tuple[FrozenFaultEpisode, ...]:
    """在门禁后生成 onset 前正常、onset 后带确定性偏移的八 episode。"""

    episodes: list[FrozenFaultEpisode] = []
    for episode_manifest in manifests:
        generator = np.random.default_rng(seed + episode_manifest.fault_id)
        time = np.arange(episode_manifest.row_count, dtype=float)[:, None]
        frequencies = np.arange(1, features + 1, dtype=float)[None, :]
        values = np.sin(time * frequencies / 13.0)
        values += generator.normal(
            loc=0.0,
            scale=0.001,
            size=values.shape,
        )
        values[episode_manifest.onset :] += episode_manifest.fault_id / 10.0
        labels = np.zeros(episode_manifest.row_count, dtype=np.int64)
        labels[episode_manifest.onset :] = episode_manifest.fault_id
        episodes.append(
            FrozenFaultEpisode(
                manifest=episode_manifest,
                raw_indices=np.arange(episode_manifest.row_count, dtype=np.int64),
                values=values,
                labels=labels,
            )
        )
    return tuple(episodes)


def _recipe_hash(value: Mapping[str, Any]) -> str:
    """对 synthetic 生成 recipe 计算来源身份，而不是伪装成真实文件 hash。"""

    encoded = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_audit_artifact(path: Path, value: Mapping[str, Any]) -> str:
    """只创建一个 JSON 审计产物并返回其文件 SHA-256。"""

    _write_new_json(path, value)
    return sha256_file(path)


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    """以 exclusive-create 写 UTF-8 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        dict(value),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.write("\n")


def _write_new_yaml(path: Path, value: Mapping[str, Any]) -> None:
    """以 exclusive-create 写 resolved config YAML。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(_plain_json(value), allow_unicode=True, sort_keys=False)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def _plain_json(value: Any) -> Any:
    """把 mapping proxy/tuple 递归复制为 YAML/JSON 可表示的普通容器。"""

    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    return value


__all__ = ["run_paper_smoke"]
