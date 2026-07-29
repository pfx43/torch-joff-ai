"""从已审阅 P2--P9 正常产物构造 formal CSTR P10 manifest。

文件用途：
    把严格 frozen 入口配置、闭环 CSTR raw archive inspection 和正常运行目录中的配置、
    split、fit ledger、checkpoint/replay、post-filter、monitor、两次校准及认证摘要组合成
    一份不可变 ``FrozenProtocolManifest``。
主要职责：
    在写 manifest 前运行只读 readiness、复验 raw header/hash 和 normal artifact 文件，
    计算包含入口与完整方法配置的 16 位 hash，并委托 ``FrozenProtocolManifest.freeze``
    独占写入。本文件不加载 fault 数值、不运行 evaluator、不创建 claim。
关键输入与输出：
    输入为 ``ResolvedFrozenEvaluationConfig``、仓库根和新 manifest 路径；输出为 formal
    frozen manifest。所有 normal artifact 路径来自严格 ``PaperNormalArtifactsConfig``。
依赖与副作用：
    读取 YAML/JSON 正常产物、checkpoint 和 CSTR MAT header/hash；调用 Git/依赖版本采集；
    只写目标 manifest 及其父目录。不读取故障数组 payload，不访问网络。
重要约束：
    只接受 mode=frozen、license=verified 且 readiness 无错误；配置声明的 rows/features/
    onset/hash 必须与 archive inspection 一致。checkpoint replay 必须 ``passed``；detection
    和 attribution 校准仍由 manifest 对象检查来源独立性与有限 episode 分辨率。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import hashlib
import json

import yaml  # type: ignore[import-untyped]

from .cstr_frozen_source import inspect_closed_loop_cstr_archive
from .frozen_artifacts import (
    FrozenNormalArtifactBundle,
    runtime_evaluator_identity,
)
from .frozen_evaluation import FrozenProtocolManifest, FrozenRiskCalibration
from .paper_entrypoints import PaperNormalMethodConfig, ResolvedFrozenEvaluationConfig
from .paper_environment import (
    collect_paper_dependency_versions,
    current_clean_paper_git_commit,
    sha256_file,
)


def build_cstr_protocol_from_artifacts(
    resolved: ResolvedFrozenEvaluationConfig,
    *,
    repo_root: str | Path,
) -> FrozenProtocolManifest:
    """复验 formal 前置证据并在内存构造 manifest，不写运行文件。

    参数：
        resolved: mode=frozen 的严格入口配置。
        repo_root: 解析仓库相对数据/normal artifact 路径，并用于取得 Git commit。
    返回：
        深层只读、全部正常证据已重放但尚未持久化的 ``FrozenProtocolManifest``。
    异常：
        readiness、archive 几何、artifact schema/hash 或校准不一致时抛出 ``ValueError``；
        I/O、Git 和 Pydantic 错误按原类型传播。
    副作用：
        只读声明文件、MAT header 和 checkpoint；不写 manifest、不创建 claim、不调用
        fault loader。
    """

    config = resolved.config
    errors = config.frozen_readiness_errors(repo_root=repo_root)
    if errors:
        raise ValueError(
            "Frozen CSTR protocol is not ready: " + "; ".join(errors) + "."
        )
    if config.mode != "frozen" or config.normal_artifacts is None:
        raise ValueError("Formal manifest builder requires mode='frozen' normal_artifacts.")
    root = Path(repo_root).expanduser().resolve()
    dataset_root = _resolve_path(root, config.dataset.root)
    inspection = inspect_closed_loop_cstr_archive(
        dataset_root,
        fault_onset=config.dataset.fault_onset,
        expected_feature_count=config.dataset.feature_count,
        normal_file=config.dataset.normal_file or "",
        fault_file=config.dataset.fault_file or "",
    )
    _validate_inspection_against_config(resolved, inspection)
    artifacts = config.normal_artifacts
    method_config = PaperNormalMethodConfig.model_validate(
        _read_mapping(
            _resolve_path(root, artifacts.resolved_config),
            name="normal resolved_config",
        )
    ).model_dump(mode="json")
    method_provenance = _read_mapping(
        _resolve_path(root, artifacts.provenance),
        name="normal provenance",
    )
    split_manifest = _read_mapping(
        _resolve_path(root, artifacts.split_manifest),
        name="split manifest",
    )
    fit_access_ledger = _read_mapping(
        _resolve_path(root, artifacts.fit_access_ledger),
        name="fit access ledger",
    )
    checkpoint_paths = {
        name: _resolve_path(root, path)
        for name, path in artifacts.checkpoint_files.items()
    }
    checkpoint_hashes = {
        name: sha256_file(path) for name, path in checkpoint_paths.items()
    }
    artifact_paths = {
        name: _resolve_path(root, path)
        for name, path in artifacts.paths().items()
    }
    runtime_evaluator = runtime_evaluator_identity(
        checkpoint_paths["protected_koopman_ts"],
        checkpoint_name="protected_koopman_ts",
    )
    normal_artifact_bundle = FrozenNormalArtifactBundle.build(
        artifact_paths=artifact_paths,
        ledger_bindings=artifacts.ledger_bindings,
        replay_outputs={
            name: f"checkpoint_replay_outputs.{name}"
            for name in artifacts.checkpoint_replay_outputs
        },
        runtime_evaluator=runtime_evaluator,
    )
    checkpoint_replay = _read_mapping(
        _resolve_path(root, artifacts.checkpoint_replay),
        name="checkpoint replay",
    )
    postfilter_library = _read_mapping(
        _resolve_path(root, artifacts.postfilter_library),
        name="postfilter library",
    )
    monitor_policy = _read_mapping(
        _resolve_path(root, artifacts.monitor_policy),
        name="monitor policy",
    )
    detection_calibration = FrozenRiskCalibration.from_dict(
        _read_mapping(
            _resolve_path(root, artifacts.detection_calibration),
            name="detection calibration",
        )
    )
    attribution_calibration = FrozenRiskCalibration.from_dict(
        _read_mapping(
            _resolve_path(root, artifacts.attribution_calibration),
            name="attribution calibration",
        )
    )
    certification_status = _read_mapping(
        _resolve_path(root, artifacts.certification_status),
        name="certification status",
    )
    combined_config = _plain_json(resolved.resolved_config)
    combined_config["normal_method_config"] = _plain_json(method_config)
    config_hash = hashlib.sha256(
        json.dumps(
            combined_config,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    combined_provenance = {
        path: [dict(record) for record in records]
        for path, records in resolved.provenance.items()
    }
    combined_provenance.update(
        {
            f"normal_method_config.{path}": _plain_json(records)
            for path, records in method_provenance.items()
        }
    )
    return FrozenProtocolManifest.build(
        protocol_version=config.protocol_version,
        evaluation_id=config.evaluation_id,
        git_commit=current_clean_paper_git_commit(root),
        resolved_config=combined_config,
        config_provenance=combined_provenance,
        config_hash=config_hash,
        claim_registry_path=_resolve_path(root, config.claim_registry),
        dependency_versions=collect_paper_dependency_versions(),
        raw_data_hashes={
            "normal": inspection.normal_source_hash,
            "fault": inspection.fault_source_hash,
        },
        split_manifest=split_manifest,
        fault_episode_manifest=inspection.fault_episodes,
        seeds=config.seeds.model_dump(),
        checkpoint_paths=checkpoint_paths,
        checkpoint_hashes=checkpoint_hashes,
        checkpoint_replay=checkpoint_replay,
        fit_access_ledger=fit_access_ledger,
        normal_artifacts=normal_artifact_bundle,
        postfilter_library=postfilter_library,
        monitor_policy=monitor_policy,
        detection_calibration=detection_calibration,
        attribution_calibration=attribution_calibration,
        certification_status=certification_status,
    )


def freeze_cstr_protocol_from_artifacts(
    resolved: ResolvedFrozenEvaluationConfig,
    *,
    repo_root: str | Path,
    manifest_path: str | Path,
) -> FrozenProtocolManifest:
    """构建 formal manifest，并在全部校验通过后独占写入目标路径。

    参数：
        resolved/repo_root: 与 :func:`build_cstr_protocol_from_artifacts` 相同。
        manifest_path: 新 ``frozen_protocol_manifest.json``；已存在时拒绝覆盖。
    返回：
        与磁盘内容一致的深层只读 manifest。
    异常：
        build 的校验错误原样传播；目标存在或写入失败时传播 ``FileExistsError``/``OSError``。
    副作用：
        只在纯内存 build 成功后创建 manifest；不创建 claim 或故障访问记录。
    """

    return build_cstr_protocol_from_artifacts(
        resolved,
        repo_root=repo_root,
    ).save(manifest_path)


def _validate_inspection_against_config(
    resolved: ResolvedFrozenEvaluationConfig,
    inspection: Any,
) -> None:
    """核对配置声明和无 payload archive inspection 的全部几何/hash。"""

    dataset = resolved.config.dataset
    errors: list[str] = []
    if inspection.normal_source_hash != dataset.normal_source_hash:
        errors.append("normal source hash")
    if inspection.fault_source_hash != dataset.fault_source_hash:
        errors.append("fault source hash")
    if inspection.normal_rows != dataset.normal_rows:
        errors.append("normal row count")
    if inspection.feature_count != dataset.feature_count:
        errors.append("feature count")
    if len(inspection.fault_episodes) != dataset.fault_episode_count:
        errors.append("fault episode count")
    if any(
        episode.row_count != dataset.fault_episode_rows
        for episode in inspection.fault_episodes
    ):
        errors.append("fault episode row count")
    if any(
        episode.onset != dataset.fault_onset
        for episode in inspection.fault_episodes
    ):
        errors.append("fault onset")
    if errors:
        raise ValueError(
            "Frozen CSTR archive differs from the entry config: "
            + ", ".join(errors)
            + "."
        )


def _read_mapping(path: Path, *, name: str) -> Mapping[str, Any]:
    """读取 JSON/YAML object；顶层标量/数组不能冒充审计产物。"""

    text = path.read_text(encoding="utf-8")
    value = (
        yaml.safe_load(text)
        if path.suffix.lower() in {".yaml", ".yml"}
        else json.loads(text)
    )
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} at {path} must contain a mapping.")
    return value


def _resolve_path(root: Path, value: str | Path | None) -> Path:
    """把配置路径解析为绝对路径；``None`` 在调用前应被严格配置拒绝。"""

    if value is None:
        raise ValueError("Frozen artifact/data path cannot be None.")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _plain_json(value: Any) -> Any:
    """递归复制 mapping proxy/tuple/Path 为普通 JSON 容器。"""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_plain_json(item) for item in value]
    return value


__all__ = [
    "build_cstr_protocol_from_artifacts",
    "freeze_cstr_protocol_from_artifacts",
]
