"""正式 CSTR frozen evaluation 的显式命令、runtime 绑定与 fail-closed preflight。

文件用途：
    提供 ``python -m joff.experiments.frozen_cli`` / ``joff-paper-frozen``，在任何 manifest
    写入、claim 或故障读取之前集中检查 frozen 模式、许可、raw hash 和 P2--P9 正常产物。
主要职责：
    解析严格入口配置，输出机器可读 readiness JSON；许可或正常产物不完整时返回 2。
    就绪后用显式 ``protected_koopman_ts`` driver 复验 checkpoint hash，以
    ``weights_only=True`` 读取纯数据 evaluator state，再经公共 evaluator registry 和当前
    driver 的严格允许类型集合重建 evaluator；
    随后检查 CSTR header/hash、冻结或复用 manifest，并执行一次八 episode 工作流。
关键输入与输出：
    输入为 ``--config`` 与仓库路径 ``--repo-root``；stdout 输出 status、evaluation ID、
    config hash 和错误列表。``--preflight-only`` 在就绪时返回 0，但仍不创建产物。
依赖与副作用：
    依赖 PyTorch checkpoint、``paper_entrypoints``、manifest builder 和 CSTR source。当前
    ``to_verify`` 状态下只读 YAML 和正常产物路径元数据，不读取 fault MAT 内容、不创建
    运行目录、不写 manifest/claim；只有全部 readiness 与 driver 预检通过后才可能写入。
重要约束：
    evaluator state 必须随受 hash/replay 证据保护的 checkpoint 一起冻结，且 type 必须
    同时命中 ``register_evaluator`` 注册表和 driver 允许集合；不能反序列化任意 Python
    对象或由命令行动态导入。driver 预检不读取 fault 数值；manifest 写入后任何失败都
    保留 claim/access 记录。命令不回退 synthetic smoke，也不把 readiness、失败运行或
    未认证输出写成正式性能结论。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence

import argparse
import json

import torch
from pydantic import Field

from joff.core.config import StrictConfig
from joff.core.factory import build_evaluator

from .cstr_frozen_source import (
    CSTRClosedLoopEpisodeLoader,
    ManifestBoundCSTRFaultSource,
    inspect_closed_loop_cstr_archive,
)
from .frozen_evaluation import (
    FrozenEpisodeInput,
    FrozenEpisodeEvaluator,
    FrozenEvaluationResult,
    FrozenEvaluationWorkflow,
    FrozenFaultEpisodeSource,
    FrozenProtocolIntegrityError,
    FrozenProtocolManifest,
    FrozenRuntimeEpisodeEvaluation,
)
from .paper_entrypoints import (
    ResolvedFrozenEvaluationConfig,
    resolve_frozen_evaluation_config,
)
from .paper_environment import (
    collect_paper_dependency_versions,
    current_clean_paper_git_commit,
    sha256_file,
)
from .paper_freeze import build_cstr_protocol_from_artifacts
from . import frozen_runtime as _frozen_runtime  # noqa: F401


@dataclass(frozen=True)
class FrozenRuntimeBinding:
    """driver 在 claim 前准备好的 source/evaluator 组合。

    参数：
        fault_source: 仅在工作流创建 claim 后才允许读取数值的正式 CSTR source。
        evaluator: 已由正常数据阶段冻结并随 checkpoint hash 保护的 evaluator；正式
            frozen 模式还要求它显式声明完整 P4--P9 执行能力。
    返回：
        不可变绑定对象，供 ``FrozenEvaluationWorkflow`` 使用。
    异常：
        字段缺少所需公开方法时抛出 ``TypeError``。
    副作用：
        构造无副作用。
    """

    fault_source: FrozenFaultEpisodeSource
    evaluator: FrozenEpisodeEvaluator

    def __post_init__(self) -> None:
        """拒绝缺少正式公开协议方法的伪 source/evaluator。"""

        if not callable(getattr(self.fault_source, "request_episodes", None)):
            raise TypeError("Frozen runtime source must implement request_episodes().")
        if not callable(getattr(self.evaluator, "evaluate_episode", None)):
            raise TypeError("Frozen runtime evaluator must implement evaluate_episode().")


@dataclass(frozen=True)
class _ManifestBoundFrozenEvaluator:
    """给恢复后的 evaluator 附加 manifest 可独立核对的 runtime 身份。

    参数：
        evaluator: 公共 registry 从 weights-only checkpoint 恢复的实际 evaluator。
        formal_pipeline_complete: evaluator 是否实现完整 P4--P9 正式流水线。
        frozen_evaluator_type/frozen_checkpoint_name/frozen_checkpoint_hash: evaluator
            envelope 类型、逻辑 checkpoint 名与当前文件 SHA-256。
    返回：
        仍实现 ``FrozenEpisodeEvaluator`` 的只读委托对象。
    异常：
        构造阶段无额外 I/O；协议缺口会由 ``FrozenEvaluationWorkflow`` 在 claim 前拒绝。
    副作用：
        ``evaluate_episode`` 仅委托底层 evaluator；不拟合、不读文件。
    """

    evaluator: FrozenEpisodeEvaluator
    formal_pipeline_complete: bool
    frozen_evaluator_type: str
    frozen_checkpoint_name: str
    frozen_checkpoint_hash: str

    def evaluate_episode(
        self,
        episode: FrozenEpisodeInput,
    ) -> FrozenRuntimeEpisodeEvaluation:
        """把无标签 episode 原样交给已恢复 evaluator。"""

        return self.evaluator.evaluate_episode(episode)


class FrozenRuntimeDriver(Protocol):
    """在不读取 fault 数值的前提下准备一个 frozen runtime 绑定。"""

    def prepare(
        self,
        resolved: ResolvedFrozenEvaluationConfig,
        *,
        repo_root: Path,
    ) -> FrozenRuntimeBinding:
        """复验正常产物并返回 source/evaluator。

        参数：
            resolved: mode=frozen 且 readiness 已通过的严格入口配置。
            repo_root: 解析正常产物和数据 archive 路径的绝对仓库根。
        返回：
            尚未读取 fault 数值的 ``FrozenRuntimeBinding``。
        异常：
            checkpoint、evaluator state、archive header/hash 或注册类型不一致时按具体
            driver 契约抛出。
        副作用：
            允许只读正常产物、checkpoint 和 MAT header；不得创建 claim、manifest 或调用
            fault value loader。
        """


class _FrozenEpisodeEvaluatorEnvelope(StrictConfig):
    """checkpoint 中 evaluator 纯数据 envelope 的严格内部 schema。

    参数：
        schema_version/type/state: 固定版本、公共 evaluator registry 键和纯数据状态。
    返回：
        拒绝未知字段的不可变 Pydantic 对象。
    异常：
        版本、类型名、state 类型或未知字段非法时抛出 ``ValidationError``。
    副作用：
        无；解析不构造 evaluator 或读取其他文件。
    """

    schema_version: Literal[1]
    type: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,127}$")
    state: dict[str, Any]


class ProtectedKoopmanTSCheckpointDriver:
    """从受 hash 保护的纯数据 checkpoint state 恢复候选正式 evaluator。

    checkpoint 的 ``extra_state.frozen_episode_evaluator`` 必须是包含 schema version、
    白名单 type 和严格 state 的纯数据 mapping。命令不接受 Python 对象、模块路径或任意
    import 字符串；checkpoint 自身必须先与 ``checkpoint_replay`` 的 SHA-256 一致，随后
    才允许 ``torch.load(..., weights_only=True)``。恢复后还要检查
    ``formal_pipeline_complete``；缺少完整 P6--P9 的 evaluator 只能用于开发测试，不能
    获得一次性故障访问权。
    """

    def __init__(
        self,
        *,
        allowed_evaluator_types: Sequence[str] = (
            "protected_koopman_ts_frozen",
        ),
    ) -> None:
        """冻结当前进程允许从公共 evaluator registry 恢复的类型集合。

        参数：
            allowed_evaluator_types: 已通过 ``register_evaluator`` 源码注册的稳定类型名。
                默认只允许正式 ``protected_koopman_ts_frozen``；测试可显式收窄到 fixture。
        返回：
            无。
        异常：
            类型名为空、重复或集合为空时抛出 ``ValueError``。
        副作用：
            只复制为不可变元组；不导入 evaluator、不读取 checkpoint/manifest/故障数据。
        """

        copied: list[str] = []
        for raw_name in allowed_evaluator_types:
            name = str(raw_name).strip()
            if not name:
                raise ValueError("Frozen evaluator type cannot be empty.")
            if name in copied:
                raise ValueError(f"Duplicate frozen evaluator builder type {name!r}.")
            copied.append(name)
        if not copied:
            raise ValueError("Frozen evaluator allowed type set cannot be empty.")
        self._allowed_evaluator_types = tuple(copied)

    def prepare(
        self,
        resolved: ResolvedFrozenEvaluationConfig,
        *,
        repo_root: Path,
    ) -> FrozenRuntimeBinding:
        """绑定 hash 已核验的 checkpoint evaluator 和只读 header inspection source。

        参数：
            resolved: frozen 模式严格入口配置。
            repo_root: 所有相对数据/正常产物路径的解析根。
        返回：
            ``FrozenRuntimeBinding``；此时尚未读取 fault 数值。
        异常：
            checkpoint/replay 结构、hash、evaluator 协议或 CSTR header 不一致时抛出
            ``FrozenProtocolIntegrityError``/``ValueError``。
        副作用：
            读取 checkpoint、replay JSON、MAT header 与文件 hash；不创建 manifest/claim，
            不调用 ``CSTRClosedLoopEpisodeLoader``。
        """

        config = resolved.config
        artifacts = config.normal_artifacts
        if artifacts is None:
            raise FrozenProtocolIntegrityError(
                "protected_koopman_ts driver requires normal_artifacts."
            )
        checkpoint_path = _resolve_repo_path(
            repo_root,
            artifacts.checkpoint_files["protected_koopman_ts"],
        )
        replay_path = _resolve_repo_path(repo_root, artifacts.checkpoint_replay)
        replay = _read_json_mapping(replay_path, name="checkpoint replay")
        replay_hashes = replay.get("checkpoint_hashes")
        if not isinstance(replay_hashes, Mapping):
            raise FrozenProtocolIntegrityError(
                "Checkpoint replay must contain checkpoint_hashes."
            )
        expected_hash = replay_hashes.get("protected_koopman_ts")
        observed_hash = sha256_file(checkpoint_path)
        if expected_hash != observed_hash:
            raise FrozenProtocolIntegrityError(
                "protected_koopman_ts checkpoint hash differs from replay evidence."
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, Mapping):
            raise FrozenProtocolIntegrityError(
                "protected_koopman_ts checkpoint must contain a mapping."
            )
        extra_state = checkpoint.get("extra_state")
        if not isinstance(extra_state, Mapping):
            raise FrozenProtocolIntegrityError(
                "Checkpoint extra_state is missing frozen_episode_evaluator."
            )
        envelope = _FrozenEpisodeEvaluatorEnvelope.model_validate(
            extra_state.get("frozen_episode_evaluator")
        )
        if envelope.type not in self._allowed_evaluator_types:
            legal = ", ".join(sorted(self._allowed_evaluator_types))
            raise FrozenProtocolIntegrityError(
                "Checkpoint frozen evaluator type is not in the formal runtime whitelist: "
                f"{envelope.type!r}. Legal types: {legal}."
            )
        evaluator = build_evaluator(
            {
                "type": envelope.type,
                "state": dict(envelope.state),
                "checkpoint": checkpoint,
            }
        )
        if not callable(getattr(evaluator, "evaluate_episode", None)):
            raise FrozenProtocolIntegrityError(
                "Registered frozen evaluator must implement evaluate_episode()."
            )
        if getattr(evaluator, "formal_pipeline_complete", False) is not True:
            raise FrozenProtocolIntegrityError(
                "Frozen evaluator must implement the complete P4--P9 pipeline before "
                "formal fault access; the restored evaluator is development-only."
            )

        dataset = config.dataset
        if dataset.root is None or dataset.normal_file is None or dataset.fault_file is None:
            raise FrozenProtocolIntegrityError(
                "Formal CSTR driver requires dataset root, normal_file and fault_file."
            )
        inspection = inspect_closed_loop_cstr_archive(
            _resolve_repo_path(repo_root, dataset.root),
            fault_onset=dataset.fault_onset,
            expected_feature_count=dataset.feature_count,
            normal_file=dataset.normal_file,
            fault_file=dataset.fault_file,
        )
        source = ManifestBoundCSTRFaultSource(
            loader=CSTRClosedLoopEpisodeLoader(inspection),
            normal_source_hash=inspection.normal_source_hash,
            fault_source_hash=inspection.fault_source_hash,
        )
        return FrozenRuntimeBinding(
            fault_source=source,
            evaluator=_ManifestBoundFrozenEvaluator(
                evaluator=evaluator,
                formal_pipeline_complete=(
                    getattr(evaluator, "formal_pipeline_complete", False) is True
                ),
                frozen_evaluator_type=envelope.type,
                frozen_checkpoint_name="protected_koopman_ts",
                frozen_checkpoint_hash=observed_hash,
            ),
        )


_RUNTIME_DRIVERS: Mapping[str, FrozenRuntimeDriver] = {
    "protected_koopman_ts": ProtectedKoopmanTSCheckpointDriver(),
}


def main(argv: Sequence[str] | None = None) -> int:
    """执行 frozen preflight 或一次正式评价并返回进程状态码。

    参数：
        argv: 可选命令参数；省略时由 ``argparse`` 读取进程参数。
    返回：
        0 表示 preflight ready 或评价完成；2 表示许可/产物 readiness 阻塞；3 表示 driver
        checkpoint/header 预检阻塞；4 表示 manifest 已冻结后执行失败。
    异常：
        参数格式错误由 ``argparse`` 按标准 ``SystemExit`` 处理；预期运行失败会转为稳定
        JSON 和上述非零状态，不把 traceback 当机器接口。
    副作用：
        ``--preflight-only`` 只读；正式路径可写 manifest、claim、fault-access 和评价产物，
        并在 claim 后读取恰好八个 fault episode。
    """

    parser = argparse.ArgumentParser(
        description="Preflight the one-shot protected CSTR frozen evaluation."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Return success when all prerequisites are ready without creating a claim.",
    )
    args = parser.parse_args(argv)
    resolved = resolve_frozen_evaluation_config(args.config)
    errors = resolved.config.frozen_readiness_errors(repo_root=args.repo_root)
    if errors:
        _print_payload(
            {
                "status": "blocked",
                "evaluation_id": resolved.config.evaluation_id,
                "config_hash": resolved.config_hash,
                "errors": list(errors),
                "claim_created": False,
                "fault_data_accessed": False,
            }
        )
        return 2
    driver = _RUNTIME_DRIVERS.get(resolved.config.runtime)
    if driver is None:
        return _print_driver_error(
            resolved,
            f"No frozen runtime driver is registered for {resolved.config.runtime!r}.",
        )
    try:
        binding = driver.prepare(
            resolved,
            repo_root=args.repo_root.expanduser().resolve(),
        )
    except Exception as exc:
        return _print_driver_error(resolved, str(exc))
    repo_root = args.repo_root.expanduser().resolve()
    run_dir = _resolve_repo_path(
        repo_root,
        resolved.config.artifact_root,
    ) / resolved.config.run_name
    manifest_path = run_dir / "frozen_protocol_manifest.json"
    try:
        manifest = _load_or_build_manifest(
            resolved,
            repo_root=repo_root,
            manifest_path=manifest_path,
        )
    except Exception as exc:
        return _print_driver_error(resolved, str(exc))
    execution_errors = _formal_execution_readiness_errors(manifest)
    if execution_errors:
        _print_payload(
            {
                "status": "blocked_formal_certification",
                "evaluation_id": resolved.config.evaluation_id,
                "config_hash": resolved.config_hash,
                "errors": list(execution_errors),
                "claim_created": False,
                "fault_data_accessed": False,
            }
        )
        return 3
    if args.preflight_only:
        _print_payload(
            {
                "status": "ready",
                "evaluation_id": resolved.config.evaluation_id,
                "config_hash": resolved.config_hash,
                "manifest_hash": manifest.manifest_hash,
                "errors": [],
                "claim_created": False,
                "fault_data_accessed": False,
            }
        )
        return 0
    try:
        if not manifest_path.exists():
            manifest.save(manifest_path)
        result = FrozenEvaluationWorkflow(
            manifest_path=manifest_path,
            claim_registry=manifest.claim_registry_path,
            artifact_dir=run_dir / "frozen_evaluation",
            fault_source=binding.fault_source,
            evaluator=binding.evaluator,
        ).run()
    except Exception as exc:
        claim_path = Path(
            manifest.claim_registry_path
            if "manifest" in locals()
            else _resolve_repo_path(repo_root, resolved.config.claim_registry)
        ) / f"{resolved.config.evaluation_id}.claim.json"
        access_path = claim_path.with_name(
            f"{resolved.config.evaluation_id}.fault-access.json"
        )
        _print_payload(
            {
                "status": "failed_after_freeze",
                "evaluation_id": resolved.config.evaluation_id,
                "config_hash": resolved.config_hash,
                "errors": [str(exc)],
                "claim_created": claim_path.exists(),
                "fault_data_accessed": access_path.exists(),
            }
        )
        return 4
    _print_completed_payload(result, config_hash=resolved.config_hash)
    return 0


def _formal_execution_readiness_errors(
    manifest: FrozenProtocolManifest,
) -> tuple[str, ...]:
    """在 manifest 已冻结、claim 尚未创建时核对三类认证状态。

    参数：
        manifest: 已完成配置、文件 hash、ledger 和跨产物校验的正式 manifest。
    返回：
        每个未达到 ``certified`` 的 operator/signature/nuisance 错误；空元组表示认证门
        已通过。
    异常：
        无；manifest 构造时已严格验证字段结构。
    副作用：
        无；不读取故障文件、不写 manifest/claim。
    """

    errors: list[str] = []
    for name in ("operator", "signature", "nuisance"):
        entry = manifest.certification_status[name]
        status = entry["status"]
        if status != "certified":
            errors.append(
                f"Formal fault access requires {name} certification status "
                f"'certified', observed {status!r}."
            )
    return tuple(errors)


def _load_or_build_manifest(
    resolved: ResolvedFrozenEvaluationConfig,
    *,
    repo_root: Path,
    manifest_path: Path,
) -> FrozenProtocolManifest:
    """复用身份一致的 manifest，否则只在内存构造并完整复验。

    已有 manifest 只允许与当前 clean HEAD、evaluation/protocol 和入口配置逐字段一致；
    不存在时调用 formal 纯内存 builder。两条路径都不写 manifest、claim 或故障访问记录。

    参数：
        resolved: 当前命令重新解析得到的严格入口配置。
        repo_root: 用于核对 clean HEAD 和解析仓库相对路径的根目录。
        manifest_path: 可能已存在的 frozen manifest 位置。
    返回：
        新建路径返回纯内存 manifest；复用路径返回经环境与入口身份复验的磁盘 manifest。
    异常：
        Git 工作树不干净、HEAD/配置/协议身份变化或嵌套 manifest 非法时抛出
        ``RuntimeError``/``FrozenProtocolIntegrityError``。
    副作用：
        只读 Git、入口配置、manifest 及其引用产物；不写文件、不创建 claim。
    """

    if not manifest_path.exists():
        return build_cstr_protocol_from_artifacts(
            resolved,
            repo_root=repo_root,
        )
    manifest = FrozenProtocolManifest.load(manifest_path)
    current_head = current_clean_paper_git_commit(repo_root)
    if current_head != manifest.git_commit:
        raise FrozenProtocolIntegrityError(
            "Existing frozen manifest was created from a different clean HEAD."
        )
    if collect_paper_dependency_versions() != dict(manifest.dependency_versions):
        raise FrozenProtocolIntegrityError(
            "Existing frozen manifest dependency_versions differ from the current "
            "runtime environment."
        )
    config = resolved.config
    if (
        manifest.evaluation_id != config.evaluation_id
        or manifest.protocol_version != config.protocol_version
        or any(
            manifest.resolved_config.get(name) != value
            for name, value in resolved.resolved_config.items()
        )
    ):
        raise FrozenProtocolIntegrityError(
            "Existing frozen manifest does not match the requested entry config."
        )
    return manifest


def _print_driver_error(
    resolved: ResolvedFrozenEvaluationConfig,
    message: str,
) -> int:
    """输出 claim 前 driver 阻塞。

    参数：
        resolved: 用于稳定输出 evaluation/config 身份的当前入口配置。
        message: 不含 traceback 的可审计阻塞原因。
    返回：
        固定进程状态码 3。
    异常：
        仅 ``print`` 的底层输出异常可能传播。
    副作用：
        向 stdout 写一行 JSON；不创建 manifest、claim 或故障访问记录。
    """

    _print_payload(
        {
            "status": "blocked_runtime_driver",
            "evaluation_id": resolved.config.evaluation_id,
            "config_hash": resolved.config_hash,
            "errors": [message],
            "claim_created": False,
            "fault_data_accessed": False,
        }
    )
    return 3


def _print_completed_payload(
    result: FrozenEvaluationResult,
    *,
    config_hash: str,
) -> None:
    """输出一次完成运行的最小稳定摘要，不解释为论文性能。

    参数：
        result: 已通过 receipt/index 校验的一次评价结果。
        config_hash: 当前入口解析后的 16 位配置身份。
    返回：
        无。
    异常：
        仅 ``print`` 的底层输出异常可能传播。
    副作用：
        向 stdout 写一行 JSON；不修改任何运行产物。
    """

    _print_payload(
        {
            "status": "completed",
            "evaluation_id": result.evaluation_id,
            "config_hash": config_hash,
            "manifest_hash": result.manifest_hash,
            "pointwise_row_count": result.pointwise_row_count,
            "receipt": str(result.receipt_path),
            "claim_created": True,
            "fault_data_accessed": True,
        }
    )


def _resolve_repo_path(root: Path, value: Path) -> Path:
    """将配置路径解析为绝对路径；绝对输入保持不变。

    参数：
        root: 已解析仓库根。
        value: 配置中的绝对或仓库相对路径。
    返回：
        规范化绝对 ``Path``。
    异常：
        路径解析的 ``OSError`` 按平台行为传播。
    副作用：
        无；不检查、不创建目标。
    """

    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _read_json_mapping(path: Path, *, name: str) -> Mapping[str, Any]:
    """读取 UTF-8 JSON 对象；非对象输入 fail closed。

    参数：
        path: 正常产物 JSON 文件。
        name: 错误消息中的业务名称。
    返回：
        顶层 mapping。
    异常：
        文件/JSON 错误按原类型传播；顶层不是 object 时抛出
        ``FrozenProtocolIntegrityError``。
    副作用：
        只读一个文件；不缓存或改写内容。
    """

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise FrozenProtocolIntegrityError(f"{name} must contain a JSON object.")
    return value


def _print_payload(value: dict[str, object]) -> None:
    """把命令状态写成一行稳定 JSON，便于 CI 和人工审计。

    参数：
        value: 只含机器接口字段的 JSON 兼容字典。
    返回：
        无。
    异常：
        非 JSON 值或 stdout 写入失败时传播 ``TypeError``/``OSError``。
    副作用：
        向 stdout 写一行按键排序 UTF-8 JSON。
    """

    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
