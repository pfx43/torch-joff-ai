"""P10 正常产物文件、账本绑定和 evaluator 身份的可重放证据。

文件用途：
    把训练历史、checkpoint 重放输出及 P5--P9 方法产物从“JSON 中自报一个 hash”提升为
    manifest 可逐文件复验的路径/hash 集合。
主要职责：
    构建和严格重放 ``FrozenNormalArtifactBundle``；绑定 ledger object ID、replay 输出名
    和 evaluator type/state 到明确文件；对正式 protected evaluator 的 gate、branch、
    mode 与检测分位执行跨产物一致性校验；核对训练与 runtime checkpoint 的模型配置和
    权重连续性。本文不运行算法，也不读取故障数据。
关键输入与输出：
    输入为逻辑产物名到普通文件路径、ledger/replay 绑定和纯数据 evaluator 身份；输出为
    深层只读、可 JSON 往返的 artifact bundle。
依赖与副作用：
    依赖 ``paper_environment.sha256_file``；构建和重放会只读列出的正常产物文件，不写
    文件、不访问网络、不创建 manifest/claim。
重要约束：
    路径必须绝对且指向普通文件，路径和 hash 键必须完全一致；任何文件内容变化都
    fail closed。绑定只能引用 bundle 内已有逻辑名，不能用未落盘的摘要替代实际产物。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import hashlib
import json
import math
import re

import torch

from .paper_environment import sha256_file


_LOGICAL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{1,191}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FrozenProtocolIntegrityError(ValueError):
    """冻结协议或其引用产物无法按持久化证据重放。"""


@dataclass(frozen=True)
class FrozenNormalArtifactBundle:
    """保存正式评价之前全部正常产物的文件级身份。

    参数：
        schema_version: 当前固定为 1。
        artifact_paths/artifact_hashes: 同键的逻辑产物名、绝对路径和文件 SHA-256。
        ledger_bindings: P2 ``object_id`` 到其真实拟合产物逻辑名。
        replay_outputs: checkpoint replay 输出名到实际输出文件逻辑名。
        runtime_evaluator: evaluator ``type``、纯数据 ``state_hash`` 和承载它的 checkpoint 名。
        bundle_hash: 除自身外全部字段的稳定 SHA-256。
    返回：
        深层只读且每次构造都会重读文件核验 hash 的 schema-v1 bundle。
    异常：
        字段、路径、hash、绑定或当前文件内容不一致时抛出
        ``FrozenProtocolIntegrityError``。
    副作用：
        只读每个正常产物文件以计算 SHA-256。
    """

    schema_version: int
    artifact_paths: Mapping[str, str]
    artifact_hashes: Mapping[str, str]
    ledger_bindings: Mapping[str, str]
    replay_outputs: Mapping[str, str]
    runtime_evaluator: Mapping[str, str]
    bundle_hash: str

    def __post_init__(self) -> None:
        """严格冻结映射并重放每个正常产物文件。"""

        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise FrozenProtocolIntegrityError(
                "Unsupported frozen normal artifact schema_version."
            )
        paths = _string_mapping(self.artifact_paths, name="normal artifact paths")
        hashes = _string_mapping(self.artifact_hashes, name="normal artifact hashes")
        if not paths or set(paths) != set(hashes):
            raise FrozenProtocolIntegrityError(
                "Frozen normal artifact paths and hashes must use identical nonempty names."
            )
        for logical_name, path_text in paths.items():
            _logical_name(logical_name, name="normal artifact logical name")
            path = Path(path_text)
            if not path.is_absolute() or not path.is_file():
                raise FrozenProtocolIntegrityError(
                    f"Frozen normal artifact {logical_name!r} must be an existing "
                    "absolute file."
                )
            expected_hash = _sha256(
                hashes[logical_name],
                name=f"normal artifact hash {logical_name}",
            )
            if sha256_file(path) != expected_hash:
                raise FrozenProtocolIntegrityError(
                    f"Frozen normal artifact hash changed for {logical_name!r}."
                )

        ledger_bindings = _binding_mapping(
            self.ledger_bindings,
            targets=paths,
            name="ledger bindings",
        )
        replay_outputs = _binding_mapping(
            self.replay_outputs,
            targets=paths,
            name="checkpoint replay outputs",
        )
        runtime_evaluator = _string_mapping(
            self.runtime_evaluator,
            name="runtime evaluator",
        )
        expected_runtime_keys = {"type", "state_hash", "checkpoint_name"}
        if set(runtime_evaluator) != expected_runtime_keys:
            raise FrozenProtocolIntegrityError(
                "Frozen runtime evaluator must contain exactly type, state_hash and "
                "checkpoint_name."
            )
        _logical_name(runtime_evaluator["type"], name="runtime evaluator type")
        _sha256(runtime_evaluator["state_hash"], name="runtime evaluator state_hash")
        _logical_name(
            runtime_evaluator["checkpoint_name"],
            name="runtime evaluator checkpoint_name",
        )
        checkpoint_artifact_name = (
            "checkpoint_files." + runtime_evaluator["checkpoint_name"]
        )
        if checkpoint_artifact_name in paths:
            observed_runtime = runtime_evaluator_identity(
                Path(paths[checkpoint_artifact_name]),
                checkpoint_name=runtime_evaluator["checkpoint_name"],
            )
            if dict(runtime_evaluator) != observed_runtime:
                raise FrozenProtocolIntegrityError(
                    "Frozen runtime evaluator identity differs from its checkpoint "
                    "envelope."
                )
        _sha256(self.bundle_hash, name="normal artifact bundle_hash")
        expected_bundle_hash = _hash_payload(
            {
                "schema_version": self.schema_version,
                "artifact_paths": dict(paths),
                "artifact_hashes": dict(hashes),
                "ledger_bindings": dict(ledger_bindings),
                "replay_outputs": dict(replay_outputs),
                "runtime_evaluator": dict(runtime_evaluator),
            }
        )
        if self.bundle_hash != expected_bundle_hash:
            raise FrozenProtocolIntegrityError(
                "Frozen normal artifact bundle_hash does not match its content."
            )
        object.__setattr__(self, "artifact_paths", paths)
        object.__setattr__(self, "artifact_hashes", hashes)
        object.__setattr__(self, "ledger_bindings", ledger_bindings)
        object.__setattr__(self, "replay_outputs", replay_outputs)
        object.__setattr__(self, "runtime_evaluator", runtime_evaluator)

    @classmethod
    def build(
        cls,
        *,
        artifact_paths: Mapping[str, str | Path],
        ledger_bindings: Mapping[str, str],
        replay_outputs: Mapping[str, str],
        runtime_evaluator: Mapping[str, str],
    ) -> "FrozenNormalArtifactBundle":
        """从当前实际文件构建 bundle。

        参数：
            artifact_paths: 逻辑名到正常产物文件；相对路径会被解析为当前绝对路径。
            ledger_bindings/replay_outputs/runtime_evaluator: 与类字段同义。
        返回：
            已逐文件计算 hash 并完整复验的不可变 bundle。
        异常：
            文件不可读、绑定或 evaluator 身份非法时抛出 ``OSError``/
            ``FrozenProtocolIntegrityError``。
        副作用：
            只读列出的文件；不创建或修改产物。
        """

        paths = {
            str(name): str(Path(path).expanduser().resolve())
            for name, path in artifact_paths.items()
        }
        hashes = {name: sha256_file(path) for name, path in paths.items()}
        payload = {
            "schema_version": 1,
            "artifact_paths": paths,
            "artifact_hashes": hashes,
            "ledger_bindings": dict(ledger_bindings),
            "replay_outputs": dict(replay_outputs),
            "runtime_evaluator": dict(runtime_evaluator),
        }
        return cls(
            schema_version=1,
            artifact_paths=paths,
            artifact_hashes=hashes,
            ledger_bindings=dict(ledger_bindings),
            replay_outputs=dict(replay_outputs),
            runtime_evaluator=dict(runtime_evaluator),
            bundle_hash=_hash_payload(payload),
        )

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "FrozenNormalArtifactBundle":
        """从 JSON mapping 严格重放 bundle 和全部引用文件。"""

        expected_keys = {
            "schema_version",
            "artifact_paths",
            "artifact_hashes",
            "ledger_bindings",
            "replay_outputs",
            "runtime_evaluator",
            "bundle_hash",
        }
        if set(value) != expected_keys:
            raise FrozenProtocolIntegrityError(
                "Frozen normal artifact bundle has missing or unknown fields."
            )
        schema_version = value["schema_version"]
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise FrozenProtocolIntegrityError(
                "Frozen normal artifact schema_version must be an integer."
            )
        return cls(
            schema_version=schema_version,
            artifact_paths=_mapping(value["artifact_paths"], name="artifact_paths"),
            artifact_hashes=_mapping(value["artifact_hashes"], name="artifact_hashes"),
            ledger_bindings=_mapping(value["ledger_bindings"], name="ledger_bindings"),
            replay_outputs=_mapping(value["replay_outputs"], name="replay_outputs"),
            runtime_evaluator=_mapping(
                value["runtime_evaluator"],
                name="runtime_evaluator",
            ),
            bundle_hash=_string(value["bundle_hash"], name="bundle_hash"),
        )

    def to_dict(self) -> dict[str, Any]:
        """返回可写入 manifest 的完整 JSON 副本。"""

        return {
            "schema_version": self.schema_version,
            "artifact_paths": dict(self.artifact_paths),
            "artifact_hashes": dict(self.artifact_hashes),
            "ledger_bindings": dict(self.ledger_bindings),
            "replay_outputs": dict(self.replay_outputs),
            "runtime_evaluator": dict(self.runtime_evaluator),
            "bundle_hash": self.bundle_hash,
        }


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    """要求 JSON object，拒绝列表或标量。"""

    if not isinstance(value, Mapping):
        raise FrozenProtocolIntegrityError(f"Frozen {name} must be a mapping.")
    return value


def runtime_evaluator_identity(
    checkpoint_path: str | Path,
    *,
    checkpoint_name: str,
) -> dict[str, str]:
    """从 weights-only checkpoint 重算 evaluator type 和纯 JSON state hash。

    参数：
        checkpoint_path: 已受文件 hash 保护的 checkpoint。
        checkpoint_name: bundle/manifest 使用的 checkpoint 逻辑名。
    返回：
        ``type/state_hash/checkpoint_name`` 三字段身份。
    异常：
        checkpoint/envelope/state 结构不严格或 state 不是有限 JSON 时抛出
        ``FrozenProtocolIntegrityError``。
    副作用：
        以 CPU ``weights_only=True`` 读取 checkpoint；不构造 evaluator 或运行模型。
    """

    envelope = _runtime_evaluator_envelope(checkpoint_path)
    evaluator_type = envelope["type"]
    state = _mapping(envelope["state"], name="runtime evaluator state")
    state_hash = _hash_payload(state)
    return {
        "type": evaluator_type,
        "state_hash": state_hash,
        "checkpoint_name": checkpoint_name,
    }


def validate_protected_evaluator_artifact_bindings(
    checkpoint_path: str | Path,
    *,
    postfilter_library: Mapping[str, Any],
    monitor_policy: Mapping[str, Any],
    monitoring_score_scaler: Mapping[str, Any],
    isolation_library: Mapping[str, Any],
    detection_quantile: float,
) -> None:
    """把 protected evaluator state 与 P5/P7/P8/P9 独立冻结文件逐字段配对。

    参数：
        checkpoint_path: 已由 bundle 和主 manifest 做文件 hash 保护的 weights-only
            checkpoint。
        postfilter_library: P7 冻结的 mode 与允许 branch 集合。
        monitor_policy: P5 冻结的 anchor gate 证据；正式 evaluator 必须提供
            ``gate_hash``，其值是 checkpoint ``anchor_gate`` 纯 JSON 的稳定 SHA-256。
        monitoring_score_scaler: estimate-only 缩放器产物，必须含运行时实际使用的
            ``rms_scale``。
        isolation_library: P9 冻结的 Normal 与 fault family 字典；运行时输出使用的
            family ID 必须来自该字典。
        detection_quantile: P8 独立 detection calibration 的 ``q_det``。
    返回：
        无；非 ``protected_koopman_ts_frozen`` evaluator 不使用这组专用字段，直接返回。
    异常：
        checkpoint state 与任一独立产物的 branch、mode、gate 或分位不一致时抛出
        ``FrozenProtocolIntegrityError``。
    副作用：
        以 CPU ``weights_only=True`` 只读 checkpoint；不构造模型、不写文件。
    """

    envelope = _runtime_evaluator_envelope(checkpoint_path)
    if envelope["type"] != "protected_koopman_ts_frozen":
        return
    state = _mapping(envelope["state"], name="protected evaluator state")
    branch_id = _string(state.get("branch_id"), name="protected evaluator branch_id")
    mode = _string(state.get("mode"), name="protected evaluator mode")

    postfilter = _mapping(postfilter_library, name="postfilter library")
    frozen_mode = _string(postfilter.get("mode"), name="postfilter mode")
    raw_branches = postfilter.get("branches")
    if not isinstance(raw_branches, (list, tuple)):
        raise FrozenProtocolIntegrityError(
            "Frozen postfilter branches must be a sequence."
        )
    frozen_branches = tuple(
        _string(value, name="postfilter branch") for value in raw_branches
    )
    if mode != frozen_mode:
        raise FrozenProtocolIntegrityError(
            "Protected evaluator mode differs from the frozen postfilter library."
        )
    if branch_id not in frozen_branches:
        raise FrozenProtocolIntegrityError(
            "Protected evaluator branch_id is absent from the frozen postfilter library."
        )

    anchor_gate = _mapping(state.get("anchor_gate"), name="protected anchor_gate")
    monitor = _mapping(monitor_policy, name="monitor policy")
    gate_evidence = _mapping(
        monitor.get("anchor_gate"),
        name="monitor policy anchor_gate",
    )
    gate_hash = _string(
        gate_evidence.get("gate_hash"),
        name="monitor policy anchor gate_hash",
    )
    _sha256(gate_hash, name="monitor policy anchor gate_hash")
    if gate_hash != _hash_payload(anchor_gate):
        raise FrozenProtocolIntegrityError(
            "Protected evaluator anchor_gate differs from the frozen monitor policy."
        )

    scaler = _mapping(
        monitoring_score_scaler,
        name="monitoring score scaler",
    )
    state_scale = _finite_float(
        state.get("score_scale"),
        name="protected score_scale",
    )
    artifact_scale = _finite_float(
        scaler.get("rms_scale"),
        name="monitoring score scaler rms_scale",
    )
    if state_scale <= 0.0 or artifact_scale <= 0.0 or state_scale != artifact_scale:
        raise FrozenProtocolIntegrityError(
            "Protected evaluator score_scale differs from the frozen monitoring scaler."
        )

    isolation = _mapping(isolation_library, name="isolation library")
    normal_family = _string(
        isolation.get("normal_family"),
        name="isolation library normal_family",
    )
    raw_fault_families = isolation.get("fault_families")
    if not isinstance(raw_fault_families, (list, tuple)):
        raise FrozenProtocolIntegrityError(
            "Frozen isolation library fault_families must be a sequence."
        )
    fault_families = tuple(
        _string(value, name="isolation library fault family")
        for value in raw_fault_families
    )
    if not fault_families or len(set(fault_families)) != len(fault_families):
        raise FrozenProtocolIntegrityError(
            "Frozen isolation library fault families must be nonempty and unique."
        )
    state_normal_family = _string(
        state.get("normal_family_id"),
        name="protected normal_family_id",
    )
    state_unresolved_family = _string(
        state.get("unresolved_family_id"),
        name="protected unresolved_family_id",
    )
    if state_normal_family != normal_family:
        raise FrozenProtocolIntegrityError(
            "Protected evaluator normal family differs from the frozen isolation library."
        )
    if (
        state_unresolved_family not in fault_families
        or normal_family in fault_families
    ):
        raise FrozenProtocolIntegrityError(
            "Protected evaluator fault family differs from the frozen isolation library."
        )

    threshold = _mapping(state.get("threshold"), name="protected threshold")
    stochastic_quantile = _finite_float(
        threshold.get("stochastic_quantile"),
        name="protected stochastic_quantile",
    )
    frozen_quantile = _finite_float(
        detection_quantile,
        name="detection calibration quantile",
    )
    if stochastic_quantile != frozen_quantile:
        raise FrozenProtocolIntegrityError(
            "Protected evaluator stochastic_quantile differs from frozen q_det."
        )


def validate_training_runtime_checkpoint_continuity(
    training_checkpoint_path: str | Path,
    runtime_checkpoint_path: str | Path,
) -> None:
    """逐字段核对训练 checkpoint 与最终 runtime checkpoint 的模型 payload。

    参数：
        training_checkpoint_path: P2 ``model_parameters`` 账本实际绑定的训练产物。
        runtime_checkpoint_path: 正式 evaluator 和主 manifest 实际引用的 checkpoint。
    返回：
        无；成功表示二者的模型配置、参数键、tensor dtype/shape/value 完全一致。
    异常：
        checkpoint 不是 weights-only mapping、模型配置/参数缺失或任一参数不一致时抛出
        ``FrozenProtocolIntegrityError``。
    副作用：
        以 CPU ``weights_only=True`` 只读两个 checkpoint；不构造模型、不写文件。
    """

    training = _weights_only_checkpoint(
        training_checkpoint_path,
        name="training checkpoint",
    )
    runtime = _weights_only_checkpoint(
        runtime_checkpoint_path,
        name="runtime checkpoint",
    )
    training_config = _mapping(
        training.get("config"),
        name="training checkpoint model config",
    )
    runtime_config = _mapping(
        runtime.get("config"),
        name="runtime checkpoint model config",
    )
    if _json_mapping_copy(training_config) != _json_mapping_copy(runtime_config):
        raise FrozenProtocolIntegrityError(
            "Runtime checkpoint model config differs from the training checkpoint."
        )

    training_state = _mapping(
        training.get("model_state_dict"),
        name="training checkpoint model_state_dict",
    )
    runtime_state = _mapping(
        runtime.get("model_state_dict"),
        name="runtime checkpoint model_state_dict",
    )
    if set(training_state) != set(runtime_state):
        raise FrozenProtocolIntegrityError(
            "Runtime checkpoint parameter keys differ from the training checkpoint."
        )
    for name in sorted(training_state):
        training_value = training_state[name]
        runtime_value = runtime_state[name]
        if (
            not isinstance(training_value, torch.Tensor)
            or not isinstance(runtime_value, torch.Tensor)
        ):
            raise FrozenProtocolIntegrityError(
                f"Checkpoint model parameter {name!r} must be a tensor."
            )
        if (
            training_value.dtype != runtime_value.dtype
            or training_value.shape != runtime_value.shape
            or not torch.equal(training_value, runtime_value)
        ):
            raise FrozenProtocolIntegrityError(
                f"Runtime checkpoint model parameter {name!r} differs from training."
            )


def _weights_only_checkpoint(
    path: str | Path,
    *,
    name: str,
) -> Mapping[str, Any]:
    """以 CPU weights-only 模式读取 checkpoint mapping，并统一失败语义。"""

    try:
        value = torch.load(Path(path), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise FrozenProtocolIntegrityError(f"Frozen {name} cannot be restored.") from exc
    if not isinstance(value, Mapping):
        raise FrozenProtocolIntegrityError(f"Frozen {name} must contain a mapping.")
    return value


def _json_mapping_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    """通过严格 JSON 往返复制模型配置，拒绝 NaN 和非数据对象。"""

    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise FrozenProtocolIntegrityError(
            "Frozen checkpoint model config must contain finite JSON data."
        ) from exc


def _runtime_evaluator_envelope(
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """从 checkpoint 复制并严格验证 evaluator type/state envelope。"""

    checkpoint = torch.load(
        Path(checkpoint_path),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise FrozenProtocolIntegrityError(
            "Protected checkpoint must contain a mapping."
        )
    extra_state = checkpoint.get("extra_state")
    if not isinstance(extra_state, Mapping):
        raise FrozenProtocolIntegrityError(
            "Protected checkpoint is missing extra_state."
        )
    envelope = extra_state.get("frozen_episode_evaluator")
    if not isinstance(envelope, Mapping) or set(envelope) != {
        "schema_version",
        "type",
        "state",
    }:
        raise FrozenProtocolIntegrityError(
            "Frozen evaluator envelope must contain exactly schema_version, type and state."
        )
    if envelope["schema_version"] != 1:
        raise FrozenProtocolIntegrityError(
            "Frozen evaluator envelope schema_version must be 1."
        )
    evaluator_type = envelope["type"]
    state = envelope["state"]
    if not isinstance(evaluator_type, str) or not evaluator_type:
        raise FrozenProtocolIntegrityError(
            "Frozen evaluator type must be a nonempty string."
        )
    if not isinstance(state, Mapping):
        raise FrozenProtocolIntegrityError(
            "Frozen evaluator state must be a mapping."
        )
    try:
        encoded_state = json.dumps(
            state,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        copied_state = json.loads(encoded_state)
    except (TypeError, ValueError) as exc:
        raise FrozenProtocolIntegrityError(
            "Frozen evaluator state must contain finite JSON data."
        ) from exc
    return {
        "type": evaluator_type,
        "state": copied_state,
    }


def _string_mapping(value: Mapping[str, Any], *, name: str) -> Mapping[str, str]:
    """复制字符串到字符串映射并冻结。"""

    copied: dict[str, str] = {}
    for key, item in value.items():
        copied[_string(key, name=f"{name} key")] = _string(
            item,
            name=f"{name} value",
        )
    return MappingProxyType(copied)


def _binding_mapping(
    value: Mapping[str, Any],
    *,
    targets: Mapping[str, str],
    name: str,
) -> Mapping[str, str]:
    """验证 object/output 到已有产物逻辑名的绑定。"""

    copied = _string_mapping(value, name=name)
    for source, target in copied.items():
        _logical_name(source, name=f"{name} source")
        if target not in targets:
            raise FrozenProtocolIntegrityError(
                f"Frozen {name} target {target!r} is not a declared normal artifact."
            )
    return copied


def _string(value: Any, *, name: str) -> str:
    """要求非空字符串，不做隐式 ``str()`` 转换。"""

    if not isinstance(value, str) or not value:
        raise FrozenProtocolIntegrityError(f"Frozen {name} must be a nonempty string.")
    return value


def _logical_name(value: str, *, name: str) -> str:
    """限制进入绑定和 evaluator registry 的稳定逻辑名。"""

    if not _LOGICAL_NAME_RE.fullmatch(value):
        raise FrozenProtocolIntegrityError(f"Frozen {name} is invalid: {value!r}.")
    return value


def _sha256(value: str, *, name: str) -> str:
    """要求小写 64 位 SHA-256。"""

    if not _SHA256_RE.fullmatch(value):
        raise FrozenProtocolIntegrityError(f"Frozen {name} must be a SHA-256.")
    return value


def _finite_float(value: Any, *, name: str) -> float:
    """要求真实有限数值，拒绝 bool、字符串及无穷占位。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FrozenProtocolIntegrityError(f"Frozen {name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise FrozenProtocolIntegrityError(f"Frozen {name} must be finite.")
    return result


def _hash_payload(value: Mapping[str, Any]) -> str:
    """按稳定 JSON 编码计算内容 SHA-256。"""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "FrozenNormalArtifactBundle",
    "FrozenProtocolIntegrityError",
    "runtime_evaluator_identity",
    "validate_protected_evaluator_artifact_bindings",
    "validate_training_runtime_checkpoint_continuity",
]
