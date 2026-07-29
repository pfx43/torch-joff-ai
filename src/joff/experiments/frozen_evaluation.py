"""P10 冻结协议与一次性 CSTR 故障评价边界。

文件用途：
    为论文正式评价提供独立于模型训练的冻结协议对象。该模块先把配置、数据、拟合访问、
    后滤波、监视器、有限 episode 校准和认证状态绑定为一个可防篡改 manifest；后续评价
    工作流只能消费这个 manifest，不能根据故障结果回写或重选方法。
主要职责：
    实现 ``FrozenProtocolManifest``、持久化 claim、故障 episode 身份、有限样本风险摘要、
    八 episode 一次性工作流、逐时刻集合输出、机器可读来源、完整性索引和 receipt 复验。
关键输入与输出：
    输入是已经由 P2--P9 正常数据流程冻结的 JSON 兼容摘要、SHA-256、随机种子和八个
    CSTR fault episode 的静态身份；输出是不可变 Python 对象及
    ``frozen_protocol_manifest.json``。manifest 不包含故障数组或任何由故障表现得到的
    选择结果。
依赖与副作用：
    依赖 Python 标准库、NumPy、P10 严格入口配置和 P2 数据协议。``freeze``、claim 和
    工作流会以 exclusive-create 语义写 JSON/JSONL/CSV；``load`` 与 verifier 只读文件。
    模块导入不读取 Git、环境、数据或文件系统，也不修改随机状态。
重要约束：
    detection 与 attribution 校准必须来自互异的正常数据来源；八个 CSTR episode 的
    fault id/族映射必须完整；所有嵌套映射在内存中深层冻结。文件一旦存在不得覆盖，
    内容或自带 hash 被篡改时 fail closed。这里记录 ``certified``/``uncertified`` 状态，
    但绝不把名义算子或 smoke 证据升级为确定性认证。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol

import csv
import hashlib
import json
import math
import re

import numpy as np
from pydantic import Field, ValidationError, field_validator, model_validator

from joff.core.config import StrictConfig
from joff.data.paper_protocol import (
    FitPurpose,
    FiveStageSplitConfig,
    PaperDataBundle,
    ProtocolAccessError,
    StageName,
    fit_stage_policy_manifest,
)

from .frozen_artifacts import (
    FrozenNormalArtifactBundle,
    FrozenProtocolIntegrityError,
    validate_protected_evaluator_artifact_bindings,
    validate_training_runtime_checkpoint_continuity,
)
from .paper_entrypoints import FrozenEvaluationEntryConfig, PaperNormalMethodConfig
from .paper_environment import sha256_file


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONFIG_HASH_RE = re.compile(r"^[0-9a-f]{16}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
_REQUIRED_DEPENDENCIES = frozenset(
    {
        "python",
        "torch",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "pydantic",
    }
)
_REQUIRED_NORMAL_ARTIFACTS = frozenset(
    {
        "resolved_config",
        "provenance",
        "split_manifest",
        "fit_access_ledger",
        "training_history",
        "training_checkpoint",
        "checkpoint_replay",
        "structure_selection",
        "monitoring_score_scaler",
        "deterministic_envelope",
        "innovation_covariance",
        "monitor_policy",
        "operator_bundle",
        "postfilter_library",
        "detection_calibration",
        "attribution_calibration",
        "isolation_library",
        "certification_status",
    }
)
_FIT_PURPOSE_ARTIFACT_NAMES = {
    FitPurpose.MODEL_PARAMETERS.value: "training_checkpoint",
    FitPurpose.STRUCTURE_SELECTION.value: "structure_selection",
    FitPurpose.MONITORING_SCORE_SCALER.value: "monitoring_score_scaler",
    FitPurpose.ENVELOPE.value: "deterministic_envelope",
    FitPurpose.COVARIANCE.value: "innovation_covariance",
    FitPurpose.BRANCH_LIBRARY.value: "postfilter_library",
    FitPurpose.STATE_MACHINE.value: "monitor_policy",
    FitPurpose.DETECTION_QUANTILE.value: "detection_calibration",
    FitPurpose.ATTRIBUTION_QUANTILE.value: "attribution_calibration",
}
_REQUIRED_SPLIT_STAGES = frozenset(
    {
        "train",
        "estimate",
        "detection_calibration",
        "attribution_calibration",
        "frozen_normal_test",
    }
)
_CSTR_FAULT_FAMILIES = {
    1: "process",
    2: "process",
    3: "actuator",
    4: "actuator",
    5: "actuator",
    6: "sensor",
    7: "sensor",
    8: "sensor",
}
_POINTWISE_OUTPUT_KEYS = frozenset(
    {
        "prediction",
        "rule_weights",
        "monitor",
        "protected_state",
        "residual",
        "operator",
        "branch_statistics",
        "threshold",
        "isolation",
    }
)
_ISOLATION_OUTCOMES = frozenset(
    {
        "Normal-compatible",
        "Nonunique",
        "Out-of-model",
        "singleton",
        "Uncertified",
    }
)
_ARTIFACT_RELATIVE_PATHS = {
    "pointwise": Path("pointwise") / "all_outputs.jsonl",
    "score_trajectory": Path("sources") / "score_trajectories.csv",
    "detection": Path("sources") / "detection_by_episode.csv",
    "isolation": Path("sources") / "isolation_by_episode.csv",
}


class _FrozenPostfilterLibraryConfig(StrictConfig):
    """严格限定 manifest 中允许出现的 P7 candidate/mode/branch 摘要。"""

    candidate_id: str
    mode: str
    branches: tuple[str, ...]
    runtime: Literal["synthetic_contract_smoke", "protected_koopman_ts"] | None = None
    paper_method_implemented: bool | None = None

    @field_validator("candidate_id", "mode")
    @classmethod
    def _validate_identifier(cls, value: str) -> str:
        """规范化 candidate 与 mode 标识。"""

        return _require_identifier(value, name="postfilter identifier")

    @field_validator("branches")
    @classmethod
    def _validate_branches(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """规范化 branch 标识并要求集合非空且唯一。"""

        branches = tuple(
            _require_identifier(branch, name="postfilter branch")
            for branch in value
        )
        if not branches or len(set(branches)) != len(branches):
            raise ValueError("Frozen postfilter branches must be nonempty and unique.")
        return branches


class _FrozenAnchorGateEvidenceConfig(StrictConfig):
    """严格描述 estimate-only anchor gate 的一种稳定身份。"""

    source: str | None = None
    kind: str | None = None
    gate_hash: str | None = None

    @model_validator(mode="after")
    def _require_identity(self) -> "_FrozenAnchorGateEvidenceConfig":
        """要求 source、kind 或 gate_hash 至少存在一个。"""

        if self.source is None and self.kind is None and self.gate_hash is None:
            raise ValueError("Frozen anchor_gate must retain an identity field.")
        if self.gate_hash is not None:
            _require_sha256(self.gate_hash, name="anchor gate_hash")
        return self


class _FrozenHysteresisConfig(StrictConfig):
    """严格保存进入/退出持续步数，不接受隐藏默认。"""

    enter: int = Field(gt=0)
    exit: int = Field(gt=0)


class _FrozenResetStateConfig(StrictConfig):
    """严格保存 episode reset 的类型、hash 或初始 context age。"""

    kind: str | None = None
    state_hash: str | None = None
    age: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _require_identity(self) -> "_FrozenResetStateConfig":
        """要求 reset state 至少保留一种可审计身份。"""

        if self.kind is None and self.state_hash is None and self.age is None:
            raise ValueError("Frozen reset_state must retain an identity field.")
        if self.state_hash is not None:
            _require_sha256(self.state_hash, name="reset state_hash")
        return self


class _FrozenMonitorPolicyConfig(StrictConfig):
    """严格组合 P5 anchor gate、hysteresis 和 reset state。"""

    anchor_gate: _FrozenAnchorGateEvidenceConfig
    hysteresis: _FrozenHysteresisConfig
    reset_state: _FrozenResetStateConfig


class _FrozenCertificationEntryConfig(StrictConfig):
    """严格保存 operator/signature/nuisance 的认证状态与可选原因。"""

    status: Literal["certified", "nominal", "unavailable", "uncertified"]
    reason: str | None = None
    artifact_hash: str | None = None

    @field_validator("artifact_hash")
    @classmethod
    def _validate_artifact_hash(cls, value: str | None) -> str | None:
        """存在 artifact hash 时要求完整小写 SHA-256。"""

        if value is not None:
            _require_sha256(value, name="certification artifact_hash")
        return value


class _FrozenCertificationStatusConfig(StrictConfig):
    """严格组合 P6/P9 三类认证边界。"""

    operator: _FrozenCertificationEntryConfig
    signature: _FrozenCertificationEntryConfig
    nuisance: _FrozenCertificationEntryConfig


class _FrozenCertifiedNuisanceEnvelopeConfig(StrictConfig):
    """formal 模式可接受的确定性 nuisance envelope 顶层 schema。

    参数：
        schema_version/status/source/envelope: 固定版本、必须为 certified 的内部状态、
            认证来源和 provider 专属但非空的包络 payload。
    返回：
        拒绝未知顶层字段的不可变配置对象。
    异常：
        nominal/frozen 状态、空来源、空 payload 或未知顶层字段由 Pydantic 拒绝。
    副作用：
        无；只验证已经由文件 hash 和拟合账本绑定的 JSON 数据。
    """

    schema_version: Literal[1]
    status: Literal["certified"]
    source: str = Field(min_length=1)
    envelope: dict[str, Any] = Field(min_length=1)


class FrozenEvaluationAlreadyClaimedError(RuntimeError):
    """表示同一 evaluation ID 已在共享 registry 中占用。

    claim 在故障 source 被调用前写入，且失败运行也不会释放。方法或运行环境需要改变时，
    必须建立新协议版本和新 evaluation ID，不能删除 claim 后重跑同一正式评价。
    """


class FrozenEvaluationArtifactError(FrozenProtocolIntegrityError):
    """表示冻结评价的 receipt、artifact index 或任一来源文件无法通过 hash 复验。"""


@dataclass(frozen=True)
class FrozenEvaluationClaim:
    """共享 registry 中一次性 evaluation ID 的不可变证明。

    参数：
        evaluation_id/protocol_version/manifest_hash: 唯一评价与协议内容身份。
        artifact_dir/claimed_at_utc/claim_path/claim_hash: 输出目录、占用时间与持久文件身份。
    返回：
        ``create`` exclusive-create 后得到的不可变 token。
    异常：
        身份、内容或文件 hash 不一致时抛出 ``FrozenProtocolIntegrityError``；重复 ID 抛出
        ``FrozenEvaluationAlreadyClaimedError``。
    副作用：
        直接构造无副作用；``create`` 与 ``consume_fault_access`` 分别写 claim/access 文件。

    source 必须接收并复验该 token，不能只相信调用顺序。token 绑定 manifest hash、协议版本、
    绝对 artifact 目录和实际 claim 文件 hash；文件被删除/修改或拿给另一 manifest 时拒绝。
    """

    evaluation_id: str
    protocol_version: str
    manifest_hash: str
    artifact_dir: Path
    claimed_at_utc: str
    claim_path: Path
    claim_hash: str

    @classmethod
    def create(
        cls,
        *,
        manifest: "FrozenProtocolManifest",
        claim_registry: str | Path,
        artifact_dir: str | Path,
    ) -> "FrozenEvaluationClaim":
        """在共享 registry exclusive-create 一个 claim 并返回可复验证明。

        参数：
            manifest: 已复验并冻结 registry 绝对路径的协议。
            claim_registry: 必须与 manifest 中的绝对路径相同。
            artifact_dir: 当前 ID 的唯一评价输出目录。
        返回：
            与新 claim 文件 hash 绑定的 ``FrozenEvaluationClaim``。
        异常：
            registry 不一致时抛出 ``FrozenProtocolIntegrityError``；ID 已存在时抛出
            ``FrozenEvaluationAlreadyClaimedError``；I/O 失败传播 ``OSError``。
        副作用：
            创建 registry 目录，并以 ``x`` 模式写一份 UTF-8 claim JSON。
        """

        registry = Path(claim_registry).expanduser().resolve()
        if registry != Path(manifest.claim_registry_path):
            raise FrozenProtocolIntegrityError(
                "Claim registry differs from the absolute path frozen in the manifest."
            )
        registry.mkdir(parents=True, exist_ok=True)
        target = registry / f"{manifest.evaluation_id}.claim.json"
        artifact_path = Path(artifact_dir).expanduser().resolve()
        payload = {
            "schema_version": 1,
            "evaluation_id": manifest.evaluation_id,
            "protocol_version": manifest.protocol_version,
            "manifest_hash": manifest.manifest_hash,
            "artifact_dir": str(artifact_path),
            "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            _write_new_json(target, payload)
        except FileExistsError as exc:
            raise FrozenEvaluationAlreadyClaimedError(
                f"Frozen evaluation ID {manifest.evaluation_id!r} is already claimed at "
                f"{target}."
            ) from exc
        return cls(
            evaluation_id=manifest.evaluation_id,
            protocol_version=manifest.protocol_version,
            manifest_hash=manifest.manifest_hash,
            artifact_dir=artifact_path,
            claimed_at_utc=str(payload["claimed_at_utc"]),
            claim_path=target.resolve(),
            claim_hash=sha256_file(target),
        )

    def verify(self, manifest: "FrozenProtocolManifest") -> None:
        """复验 claim 文件内容/hash 及其与当前 manifest 的绑定。

        参数：
            manifest: 当前调用链正在消费的冻结协议。
        返回：
            无；全部一致时正常返回。
        异常：
            manifest 身份、claim 文件存在性、hash 或字段任一不一致时抛出
            ``FrozenProtocolIntegrityError``。
        副作用：
            只读 claim 文件并计算 SHA-256。
        """

        if (
            self.evaluation_id != manifest.evaluation_id
            or self.protocol_version != manifest.protocol_version
            or self.manifest_hash != manifest.manifest_hash
        ):
            raise FrozenProtocolIntegrityError(
                "Frozen evaluation claim does not belong to the supplied manifest."
            )
        try:
            observed_hash = sha256_file(self.claim_path)
        except OSError as exc:
            raise FrozenProtocolIntegrityError(
                "Frozen evaluation claim file is missing or unreadable."
            ) from exc
        if observed_hash != self.claim_hash:
            raise FrozenProtocolIntegrityError(
                "Frozen evaluation claim file changed after creation."
            )
        loaded = _read_json_mapping(self.claim_path, name="evaluation claim")
        if loaded != self.to_dict():
            raise FrozenProtocolIntegrityError(
                "Frozen evaluation claim content differs from its token."
            )

    def consume_fault_access(self, manifest: "FrozenProtocolManifest") -> Path:
        """持久化消费当前 claim 的唯一故障数值访问机会。

        参数：
            manifest: 与 claim 绑定且已完整复验的冻结协议。
        返回：
            新建的 ``*.fault-access.json`` 绝对路径。
        异常：
            claim/manifest 不一致时抛出 ``FrozenProtocolIntegrityError``；同一 claim 已经
            消费过故障访问时抛出 ``FrozenEvaluationAlreadyClaimedError``。
        副作用：
            在 manifest 冻结的共享 registry 中 exclusive-create 一个访问记录。即使后续
            loader 或 evaluator 失败也不会删除，因此重新构造 source 或换进程不能重读。
        """

        self.verify(manifest)
        access_path = self.claim_path.with_name(
            f"{self.evaluation_id}.fault-access.json"
        )
        payload = {
            "schema_version": 1,
            "evaluation_id": self.evaluation_id,
            "protocol_version": self.protocol_version,
            "manifest_hash": self.manifest_hash,
            "claim_hash": self.claim_hash,
            "claim_path": str(self.claim_path),
            "consumed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            _write_new_json(access_path, payload)
        except FileExistsError as exc:
            raise FrozenEvaluationAlreadyClaimedError(
                f"Frozen evaluation ID {self.evaluation_id!r} already consumed its "
                f"fault-data access at {access_path}."
            ) from exc
        return access_path.resolve()

    def verify_fault_access_consumed(
        self,
        manifest: "FrozenProtocolManifest",
    ) -> Path:
        """验证当前 claim 的持久 fault-access 记录存在且身份完全匹配。

        参数：
            manifest: 当前 source 返回 episode 时声称使用的冻结协议，必须与 claim 绑定。
        返回：
            已存在且通过身份复验的 ``*.fault-access.json`` 绝对路径。
        异常：
            claim 无效、访问记录缺失/不可读、身份字段或消费时间戳类型不符时抛出
            ``FrozenProtocolIntegrityError``。
        副作用：
            只读 claim 与 fault-access 文件；不创建授权、不重新开放 loader，也不修改记录。

        记录中的时间戳不参与预期值比较，其余身份字段必须与当前 token 完全一致。
        """

        self.verify(manifest)
        access_path = self.claim_path.with_name(
            f"{self.evaluation_id}.fault-access.json"
        )
        try:
            loaded = _read_json_mapping(access_path, name="fault access record")
        except OSError as exc:
            raise FrozenProtocolIntegrityError(
                "Frozen fault source returned without consuming persistent fault access."
            ) from exc
        expected = {
            "schema_version": 1,
            "evaluation_id": self.evaluation_id,
            "protocol_version": self.protocol_version,
            "manifest_hash": self.manifest_hash,
            "claim_hash": self.claim_hash,
            "claim_path": str(self.claim_path),
        }
        observed = {key: loaded.get(key) for key in expected}
        if observed != expected or not isinstance(loaded.get("consumed_at_utc"), str):
            raise FrozenProtocolIntegrityError(
                "Frozen fault-access record differs from the current claim."
            )
        return access_path.resolve()

    def to_dict(self) -> dict[str, Any]:
        """返回与 registry/local claim 文件完全一致的 JSON payload。

        返回：
            新分配的 schema-v1 JSON 兼容字典，不包含 token 自身的 ``claim_hash``。
        异常：
            无。
        副作用：
            无。
        """

        return {
            "schema_version": 1,
            "evaluation_id": self.evaluation_id,
            "protocol_version": self.protocol_version,
            "manifest_hash": self.manifest_hash,
            "artifact_dir": str(self.artifact_dir),
            "claimed_at_utc": self.claimed_at_utc,
        }


@dataclass(frozen=True)
class FrozenRiskCalibration:
    """记录 detection 或 attribution 的有限正常 episode 校准结论。

    参数：
        name: 只能是 ``detection`` 或 ``attribution``。
        requested_risk: 调用方冻结的超额风险水平，位于开区间 ``(0, 1)``。
        quantile: 有限可实现时为非负有限 ``q_det``/``q_attr``；分辨率不足或未认证时
            必须是正无穷，不能放入一个看似保守的有限占位值。
        episode_count: 相互独立的完整正常校准 episode 数。
        score_count: 这些 episode 内被校准流程实际覆盖的逐时刻/分支分数总数。
        attainable_risk_resolution: 必须等于 ``1 / (episode_count + 1)``。
        status: ``finite``、``unattainable`` 或 ``uncertified``。
        source_hash: 该校准分区的 SHA-256，用于证明两类校准不复用来源。
        episode_ids: 长度与 ``episode_count`` 相同的唯一 episode 身份。
    异常：
        数值、状态、hash 或 episode 身份违反上述约束时抛出 ``ValueError``。
    副作用：
        无；构造后不可变。
    """

    name: Literal["detection", "attribution"]
    requested_risk: float
    quantile: float
    episode_count: int
    score_count: int
    attainable_risk_resolution: float
    status: Literal["finite", "unattainable", "uncertified"]
    source_hash: str
    episode_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """校验有限秩分辨率与分位状态，避免不可实现风险被写成有限阈值。"""

        if self.name not in {"detection", "attribution"}:
            raise ValueError("Frozen risk calibration name must be detection or attribution.")
        if not math.isfinite(self.requested_risk) or not 0.0 < self.requested_risk < 1.0:
            raise ValueError("Frozen requested_risk must be finite and lie in (0, 1).")
        if isinstance(self.episode_count, bool) or self.episode_count <= 0:
            raise ValueError("Frozen calibration episode_count must be a positive integer.")
        if isinstance(self.score_count, bool) or self.score_count < self.episode_count:
            raise ValueError(
                "Frozen calibration score_count must cover at least one score per episode."
            )
        expected_resolution = 1.0 / (self.episode_count + 1)
        if not math.isclose(
            self.attainable_risk_resolution,
            expected_resolution,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "Frozen attainable_risk_resolution must equal 1 / (episode_count + 1)."
            )
        if self.status == "finite":
            if not math.isfinite(self.quantile) or self.quantile < 0.0:
                raise ValueError("A finite frozen calibration requires a nonnegative quantile.")
            if self.requested_risk + 1e-15 < expected_resolution:
                raise ValueError(
                    "A requested risk below finite-episode resolution cannot have finite status."
                )
        elif self.status in {"unattainable", "uncertified"}:
            if not math.isinf(self.quantile) or self.quantile < 0.0:
                raise ValueError(
                    "Unattainable or uncertified calibration must use positive infinity."
                )
        else:
            raise ValueError(f"Unknown frozen calibration status {self.status!r}.")
        _require_sha256(self.source_hash, name=f"{self.name} source_hash")
        normalized_ids = tuple(_require_identifier(value, name="episode_id") for value in self.episode_ids)
        if len(normalized_ids) != self.episode_count:
            raise ValueError(
                "Frozen calibration episode_ids length must equal episode_count."
            )
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("Frozen calibration episode_ids must be unique.")
        object.__setattr__(self, "episode_ids", normalized_ids)

    def to_dict(self) -> dict[str, Any]:
        """把风险校准摘要编码为 manifest 使用的 JSON payload。

        参数：
            无。
        返回：
            新分配字典；episode 元组被复制为列表，正无穷分位编码为字符串
            ``"infinity"``，因此输出可由标准 JSON 严格写入。
        异常：
            无；对象在构造时已验证全部数值与身份。
        副作用：
            无；不暴露内部元组，也不读取校准分数。
        """

        return {
            "name": self.name,
            "requested_risk": self.requested_risk,
            "quantile": _encode_finite_or_infinity(self.quantile),
            "episode_count": self.episode_count,
            "score_count": self.score_count,
            "attainable_risk_resolution": self.attainable_risk_resolution,
            "status": self.status,
            "source_hash": self.source_hash,
            "episode_ids": list(self.episode_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenRiskCalibration":
        """从 JSON mapping 严格重建风险摘要。

        参数：
            value: 必须精确包含 schema-v1 风险字段的映射。
        返回：
            已复验风险分辨率、分位状态、来源 hash 与 episode 身份的不可变摘要。
        异常：
            缺失/未知字段、类型错误、非法无穷编码或跨字段约束不成立时抛出
            ``FrozenProtocolIntegrityError``/``ValueError``。
        副作用：
            无；不读取文件或原始校准分数。
        """

        _require_exact_keys(
            value,
            {
                "name",
                "requested_risk",
                "quantile",
                "episode_count",
                "score_count",
                "attainable_risk_resolution",
                "status",
                "source_hash",
                "episode_ids",
            },
            name="frozen risk calibration",
        )
        return cls(
            name=_strict_string(value["name"], name="name"),  # type: ignore[arg-type]
            requested_risk=_strict_float(
                value["requested_risk"],
                name="requested_risk",
            ),
            quantile=_decode_finite_or_infinity(value["quantile"], name="quantile"),
            episode_count=_strict_int(value["episode_count"], name="episode_count"),
            score_count=_strict_int(value["score_count"], name="score_count"),
            attainable_risk_resolution=_strict_float(
                value["attainable_risk_resolution"],
                name="attainable_risk_resolution",
            ),
            status=_strict_string(value["status"], name="status"),  # type: ignore[arg-type]
            source_hash=_strict_string(value["source_hash"], name="source_hash"),
            episode_ids=tuple(
                _strict_string(item, name="episode_id")
                for item in _strict_sequence(value["episode_ids"], name="episode_ids")
            ),
        )


@dataclass(frozen=True)
class FrozenFaultEpisodeManifest:
    """记录一个 CSTR fault episode 在读取数值前即可审阅的静态身份。

    参数：
        episode_id/fault_id/fault_family: 唯一 episode 名、1--8 故障号和声明故障族。
        onset: 该 episode 第一条故障标签的零基位置；正式闭环 CSTR 为 200。
        row_count: episode 总行数。
        raw_index_start/raw_index_end: 连续原始行号闭区间。
        source_hash: 承载该 episode 的原始故障文件 SHA-256。
    重要约束：
        fault id 到 process/actuator/sensor 的映射固定为 P1 已修复的 CSTR 协议；不能在
        看到评价结果后重命名故障族。
    返回：
        构造后不可变、可严格 JSON 往返的 episode 身份。
    异常：
        标识、故障族映射、onset、行号区间或 SHA-256 非法时抛出 ``ValueError``。
    副作用：
        无。
    """

    episode_id: str
    fault_id: int
    fault_family: Literal["process", "actuator", "sensor"]
    onset: int
    row_count: int
    raw_index_start: int
    raw_index_end: int
    source_hash: str

    def __post_init__(self) -> None:
        """验证故障身份、onset 和连续原始行号。"""

        object.__setattr__(
            self,
            "episode_id",
            _require_identifier(self.episode_id, name="fault episode_id"),
        )
        if isinstance(self.fault_id, bool) or self.fault_id not in _CSTR_FAULT_FAMILIES:
            raise ValueError("Frozen CSTR fault_id must be one of 1..8.")
        expected_family = _CSTR_FAULT_FAMILIES[self.fault_id]
        if self.fault_family != expected_family:
            raise ValueError(
                f"Frozen CSTR fault_id {self.fault_id} belongs to {expected_family!r}, "
                f"not {self.fault_family!r}."
            )
        for name, value in (
            ("onset", self.onset),
            ("row_count", self.row_count),
            ("raw_index_start", self.raw_index_start),
            ("raw_index_end", self.raw_index_end),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"Frozen fault {name} must be an integer.")
        if self.row_count <= 0:
            raise ValueError("Frozen fault row_count must be positive.")
        if self.onset < 0 or self.onset >= self.row_count:
            raise ValueError("Frozen fault onset must identify a row inside the episode.")
        if self.raw_index_start < 0 or self.raw_index_end < self.raw_index_start:
            raise ValueError("Frozen fault raw index interval is invalid.")
        if self.raw_index_end - self.raw_index_start + 1 != self.row_count:
            raise ValueError(
                "Frozen fault raw index interval length must equal row_count."
            )
        _require_sha256(self.source_hash, name="fault episode source_hash")

    def to_dict(self) -> dict[str, Any]:
        """把一个故障 episode 的静态身份复制为 JSON payload。

        参数：
            无。
        返回：
            包含故障号/族、onset、行区间和来源 SHA-256 的新字典。
        异常：
            无；对象已在构造时验证。
        副作用：
            无；不读取对应 MAT 数值。
        """

        return {
            "episode_id": self.episode_id,
            "fault_id": self.fault_id,
            "fault_family": self.fault_family,
            "onset": self.onset,
            "row_count": self.row_count,
            "raw_index_start": self.raw_index_start,
            "raw_index_end": self.raw_index_end,
            "source_hash": self.source_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenFaultEpisodeManifest":
        """从 JSON mapping 严格重建一个 fault episode 身份。

        参数：
            value: 必须精确包含 episode 静态身份字段的映射。
        返回：
            已核对故障号到故障族、onset、连续行区间和来源 hash 的不可变对象。
        异常：
            缺失/未知字段、类型、故障族映射、行几何或 SHA-256 非法时抛出
            ``FrozenProtocolIntegrityError``/``ValueError``。
        副作用：
            无；只解析内存值，不打开故障文件。
        """

        _require_exact_keys(
            value,
            {
                "episode_id",
                "fault_id",
                "fault_family",
                "onset",
                "row_count",
                "raw_index_start",
                "raw_index_end",
                "source_hash",
            },
            name="frozen fault episode manifest",
        )
        return cls(
            episode_id=_strict_string(value["episode_id"], name="episode_id"),
            fault_id=_strict_int(value["fault_id"], name="fault_id"),
            fault_family=_strict_string(  # type: ignore[arg-type]
                value["fault_family"],
                name="fault_family",
            ),
            onset=_strict_int(value["onset"], name="onset"),
            row_count=_strict_int(value["row_count"], name="row_count"),
            raw_index_start=_strict_int(
                value["raw_index_start"],
                name="raw_index_start",
            ),
            raw_index_end=_strict_int(value["raw_index_end"], name="raw_index_end"),
            source_hash=_strict_string(value["source_hash"], name="source_hash"),
        )


@dataclass(frozen=True)
class FrozenFaultEpisode:
    """保存 claim 成功后才允许出现的一份故障 episode 数值。

    参数：
        manifest: 冻结前已经写入主 manifest 的静态 episode 身份。
        raw_indices: 与数值逐行对齐、连续且等于 manifest 闭区间的原始行号。
        values: 二维有限浮点输入矩阵 ``[row_count, feature_count]``。
        labels: onset 前为 0、onset 起为 ``fault_id`` 的逐行标签。
    返回：
        构造后四个字段都不可替换，三个数组是独立只读副本。
    异常：
        行数、shape、raw index、标签或有限性不匹配时抛出 ``ValueError``。
    副作用：
        只复制内存数组并关闭其写标志；不读取文件、不登记 claim。
    """

    manifest: FrozenFaultEpisodeManifest
    raw_indices: np.ndarray = field(repr=False, compare=False)
    values: np.ndarray = field(repr=False, compare=False)
    labels: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        """复制并严格核对逐行身份，防止错位 episode 进入评价器。"""

        raw_indices = np.asarray(self.raw_indices)
        values = np.asarray(self.values, dtype=float)
        labels = np.asarray(self.labels)
        if raw_indices.ndim != 1:
            raise ValueError("Frozen fault raw_indices must be one-dimensional.")
        if values.ndim != 2 or values.shape[1] <= 0:
            raise ValueError("Frozen fault values must have shape [rows, positive features].")
        if labels.ndim != 1:
            raise ValueError("Frozen fault labels must be one-dimensional.")
        row_count = self.manifest.row_count
        if (
            raw_indices.shape[0] != row_count
            or values.shape[0] != row_count
            or labels.shape[0] != row_count
        ):
            raise ValueError("Frozen fault arrays must all match manifest row_count.")
        if not np.issubdtype(raw_indices.dtype, np.integer):
            raise ValueError("Frozen fault raw_indices must contain integers.")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Frozen fault labels must contain integers.")
        expected_indices = np.arange(
            self.manifest.raw_index_start,
            self.manifest.raw_index_end + 1,
            dtype=np.int64,
        )
        if not np.array_equal(raw_indices.astype(np.int64, copy=False), expected_indices):
            raise ValueError(
                "Frozen fault raw_indices do not match the manifest interval."
            )
        if not np.isfinite(values).all():
            raise ValueError("Frozen fault values must be finite.")
        expected_labels = np.zeros(row_count, dtype=np.int64)
        expected_labels[self.manifest.onset :] = self.manifest.fault_id
        if not np.array_equal(labels.astype(np.int64, copy=False), expected_labels):
            raise ValueError(
                "Frozen fault labels must be normal before onset and equal fault_id afterward."
            )
        raw_copy = raw_indices.astype(np.int64, copy=True)
        value_copy = values.astype(float, copy=True)
        label_copy = labels.astype(np.int64, copy=True)
        for array in (raw_copy, value_copy, label_copy):
            array.setflags(write=False)
        object.__setattr__(self, "raw_indices", raw_copy)
        object.__setattr__(self, "values", value_copy)
        object.__setattr__(self, "labels", label_copy)


@dataclass(frozen=True)
class FrozenEpisodeInput:
    """正式 evaluator 可见的无标签只读特征视图。

    文件工作流在故障 source 返回完整 ``FrozenFaultEpisode`` 后立即构造该视图；evaluator
    只能看到逐行原始索引和过程变量，不能直接读取 episode 名、onset、故障编号、故障族或
    真实标签。真实答案只由 ``FrozenEvaluationWorkflow`` 在算法返回后用于评价汇总。

    参数：
        raw_indices: 与输入矩阵逐行对齐的稳定原始行号。
        values: 二维有限过程变量矩阵 ``[rows, features]``。
    返回：
        两个数组均为独立只读副本的不可变输入对象。
    异常：
        shape、长度、索引整数性、连续性或数值有限性非法时抛出 ``ValueError``。
    副作用：
        只复制内存数组并关闭写标志；不读取文件、不持有完整故障 episode。
    """

    raw_indices: np.ndarray = field(repr=False, compare=False)
    values: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        """复制并校验 evaluator 唯一可见的行号与过程变量。"""

        raw_indices = np.asarray(self.raw_indices)
        values = np.asarray(self.values, dtype=float)
        if raw_indices.ndim != 1 or not np.issubdtype(
            raw_indices.dtype,
            np.integer,
        ):
            raise ValueError(
                "Frozen evaluator raw_indices must be a one-dimensional integer array."
            )
        if values.ndim != 2 or values.shape[1] <= 0:
            raise ValueError(
                "Frozen evaluator values must have shape [rows, positive features]."
            )
        if values.shape[0] != raw_indices.shape[0] or values.shape[0] == 0:
            raise ValueError(
                "Frozen evaluator values and raw_indices must have the same positive row count."
            )
        normalized_indices = raw_indices.astype(np.int64, copy=True)
        if normalized_indices[0] < 0 or not np.array_equal(
            np.diff(normalized_indices),
            np.ones(normalized_indices.shape[0] - 1, dtype=np.int64),
        ):
            raise ValueError(
                "Frozen evaluator raw_indices must be nonnegative and contiguous."
            )
        if not np.isfinite(values).all():
            raise ValueError("Frozen evaluator values must all be finite.")
        normalized_values = values.astype(float, copy=True)
        normalized_indices.setflags(write=False)
        normalized_values.setflags(write=False)
        object.__setattr__(self, "raw_indices", normalized_indices)
        object.__setattr__(self, "values", normalized_values)

    @classmethod
    def from_fault_episode(
        cls,
        episode: FrozenFaultEpisode,
    ) -> "FrozenEpisodeInput":
        """从已校验完整 episode 投影无真值视图。

        参数：
            episode: claim 后由受控 source 返回的完整故障 episode。
        返回：
            不保留 ``episode`` 或其 manifest/labels 引用的只读特征副本。
        异常：
            输入类型错误时抛出 ``TypeError``；数组错误由构造校验传播。
        副作用：
            复制行号和过程变量；不读取、修改或缓存真实标签。
        """

        if not isinstance(episode, FrozenFaultEpisode):
            raise TypeError("Frozen evaluator input requires a FrozenFaultEpisode.")
        return cls(
            raw_indices=episode.raw_indices,
            values=episode.values,
        )


@dataclass(frozen=True)
class FrozenPointwiseOutput:
    """记录 P4--P9 在一个故障时刻产生的完整可审计输出。

    ``method_outputs`` 必须同时包含预测/规则权重、monitor、protected state、堆叠残差、
    operator 状态、支路统计、阈值分账和集合值隔离九类来源。``normal_family_id`` 把
    ``Normal-compatible`` 与任意命名的 Normal explanation 显式绑定；空候选只对应
    ``Out-of-model``。顶层字段只是为主表和主图提供稳定列，不替代原始来源。

    参数：
        episode/raw/fault/label 与 detection 字段: 当前行身份、分数、阈值和告警。
        branch_id/mode: 必须属于 manifest 冻结的 P7 library。
        normal_family_id/candidate_ids/isolation_*: P9 集合语义及可选 singleton 报告。
        suppression_reason/method_outputs: 抑制原因与完整 P4--P9 原始来源。
    返回：
        深层不可变、可写 JSONL/CSV 的逐时刻记录。
    异常：
        判决、候选基数、Normal 成员、认证、字段集合或数值不一致时抛出 ``ValueError``。
    副作用：
        只复制并冻结内存映射。
    """

    episode_id: str
    raw_index: int
    fault_id: int
    true_label: int
    detection_score: float
    detection_threshold: float
    alarm: bool
    branch_id: str
    mode: str
    normal_family_id: str
    candidate_ids: tuple[str, ...]
    isolation_outcome: str
    reported_family: str | None
    isolation_certified: bool
    suppression_reason: str | None
    method_outputs: Mapping[str, Any]

    def __post_init__(self) -> None:
        """验证判决一致性并深层冻结完整方法输出。"""

        object.__setattr__(
            self,
            "episode_id",
            _require_identifier(self.episode_id, name="pointwise episode_id"),
        )
        for name, value in (
            ("raw_index", self.raw_index),
            ("fault_id", self.fault_id),
            ("true_label", self.true_label),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"Frozen pointwise {name} must be an integer.")
        if self.fault_id not in _CSTR_FAULT_FAMILIES:
            raise ValueError("Frozen pointwise fault_id must be one of 1..8.")
        if self.true_label not in {0, self.fault_id}:
            raise ValueError("Frozen pointwise true_label must be 0 or its episode fault_id.")
        if not math.isfinite(self.detection_score) or self.detection_score < 0.0:
            raise ValueError("Frozen detection_score must be finite and nonnegative.")
        if (
            math.isnan(self.detection_threshold)
            or self.detection_threshold < 0.0
            or self.detection_threshold == -math.inf
        ):
            raise ValueError(
                "Frozen detection_threshold must be nonnegative finite or positive infinity."
            )
        if not isinstance(self.alarm, bool):
            raise ValueError("Frozen alarm must be a bool.")
        exceeds = self.detection_score > self.detection_threshold
        if self.alarm and not exceeds:
            raise ValueError("A frozen alarm requires strict score > threshold.")
        if not self.alarm and exceeds and not self.suppression_reason:
            raise ValueError(
                "A suppressed strict exceedance must retain a suppression_reason."
            )
        object.__setattr__(
            self,
            "branch_id",
            _require_identifier(self.branch_id, name="branch_id"),
        )
        object.__setattr__(
            self,
            "mode",
            _require_identifier(self.mode, name="monitor mode"),
        )
        object.__setattr__(
            self,
            "normal_family_id",
            _require_identifier(self.normal_family_id, name="normal_family_id"),
        )
        candidate_ids = tuple(
            _require_identifier(value, name="candidate_id")
            for value in self.candidate_ids
        )
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("Frozen candidate_ids must be unique.")
        object.__setattr__(self, "candidate_ids", candidate_ids)
        if self.isolation_outcome not in _ISOLATION_OUTCOMES:
            raise ValueError(
                f"Unknown frozen isolation_outcome {self.isolation_outcome!r}."
            )
        if self.reported_family is not None:
            object.__setattr__(
                self,
                "reported_family",
                _require_identifier(self.reported_family, name="reported_family"),
            )
        if not isinstance(self.isolation_certified, bool):
            raise ValueError("Frozen isolation_certified must be a bool.")
        normal_present = self.normal_family_id in candidate_ids
        if self.isolation_outcome == "Normal-compatible":
            valid_isolation = normal_present and self.reported_family is None
        elif self.isolation_outcome == "Out-of-model":
            valid_isolation = not candidate_ids and self.reported_family is None
        elif self.isolation_outcome == "Nonunique":
            valid_isolation = (
                len(candidate_ids) > 1
                and not normal_present
                and self.reported_family is None
            )
        elif self.isolation_outcome == "singleton":
            valid_isolation = (
                len(candidate_ids) == 1
                and not normal_present
                and self.reported_family == candidate_ids[0]
                and self.isolation_certified
            )
        else:
            valid_isolation = (
                len(candidate_ids) == 1
                and not normal_present
                and self.reported_family is None
                and not self.isolation_certified
            )
        if not valid_isolation:
            raise ValueError(
                "Frozen isolation outcome, Normal membership, candidate cardinality, "
                "reported_family and certification are inconsistent."
            )
        if self.suppression_reason is not None:
            reason = self.suppression_reason.strip()
            if not reason:
                raise ValueError("Frozen suppression_reason cannot be blank.")
            object.__setattr__(self, "suppression_reason", reason)
        method_outputs = _deep_freeze_json(
            self.method_outputs,
            name="pointwise method_outputs",
        )
        actual_keys = set(method_outputs)
        if actual_keys != _POINTWISE_OUTPUT_KEYS:
            missing = sorted(_POINTWISE_OUTPUT_KEYS.difference(actual_keys))
            extra = sorted(actual_keys.difference(_POINTWISE_OUTPUT_KEYS))
            raise ValueError(
                "Frozen pointwise method_outputs must contain the complete P4-P9 source set. "
                f"missing={missing}, unknown={extra}."
            )
        object.__setattr__(self, "method_outputs", method_outputs)

    def to_dict(self) -> dict[str, Any]:
        """返回可写入逐时刻 JSONL 的完整输出副本。

        参数：
            无。
        返回：
            包含检测、branch、集合隔离结论和完整 P4--P9 ``method_outputs`` 的新字典；
            正无穷阈值被显式编码为 ``"infinity"``。
        异常：
            无；字段与集合语义已在构造时验证。
        副作用：
            无；深复制嵌套方法输出，不修改 evaluator 状态。
        """

        return {
            "episode_id": self.episode_id,
            "raw_index": self.raw_index,
            "fault_id": self.fault_id,
            "true_label": self.true_label,
            "detection_score": self.detection_score,
            "detection_threshold": _encode_finite_or_infinity(
                self.detection_threshold
            ),
            "alarm": self.alarm,
            "branch_id": self.branch_id,
            "mode": self.mode,
            "normal_family_id": self.normal_family_id,
            "candidate_ids": list(self.candidate_ids),
            "isolation_outcome": self.isolation_outcome,
            "reported_family": self.reported_family,
            "isolation_certified": self.isolation_certified,
            "suppression_reason": self.suppression_reason,
            "method_outputs": _json_copy(self.method_outputs),
        }


@dataclass(frozen=True)
class FrozenRuntimePointwiseOutput:
    """evaluator 返回的无真值逐时刻方法输出。

    该对象刻意不含 episode ID、故障编号或真实标签。其余检测、支路、阈值和集合隔离字段
    与最终 ``FrozenPointwiseOutput`` 使用同一校验；工作流随后通过 :meth:`bind_truth`
    添加只用于评价的真值列。

    参数：
        raw_index 至 method_outputs: 已冻结 P4--P9 方法在一个输入行上的运行结果。
    返回：
        深层不可变且尚未绑定任何故障答案的运行时输出。
    异常：
        数值、判决、候选集合或完整方法来源不一致时抛出 ``ValueError``。
    副作用：
        只复制和冻结 JSON 方法来源；不读取 manifest 或标签。
    """

    raw_index: int
    detection_score: float
    detection_threshold: float
    alarm: bool
    branch_id: str
    mode: str
    normal_family_id: str
    candidate_ids: tuple[str, ...]
    isolation_outcome: str
    reported_family: str | None
    isolation_certified: bool
    suppression_reason: str | None
    method_outputs: Mapping[str, Any]

    def __post_init__(self) -> None:
        """复用最终输出校验，同时保持占位真值完全不进入当前对象。"""

        # 最终对象的全部方法字段校验与故障真值无关。这里用固定合法占位身份执行同一
        # 公共校验，再只复制方法字段；占位值不会被保存，也不会暴露给 evaluator。
        validated = FrozenPointwiseOutput(
            episode_id="runtime-output",
            raw_index=self.raw_index,
            fault_id=1,
            true_label=0,
            detection_score=self.detection_score,
            detection_threshold=self.detection_threshold,
            alarm=self.alarm,
            branch_id=self.branch_id,
            mode=self.mode,
            normal_family_id=self.normal_family_id,
            candidate_ids=self.candidate_ids,
            isolation_outcome=self.isolation_outcome,
            reported_family=self.reported_family,
            isolation_certified=self.isolation_certified,
            suppression_reason=self.suppression_reason,
            method_outputs=self.method_outputs,
        )
        for name in (
            "raw_index",
            "detection_score",
            "detection_threshold",
            "alarm",
            "branch_id",
            "mode",
            "normal_family_id",
            "candidate_ids",
            "isolation_outcome",
            "reported_family",
            "isolation_certified",
            "suppression_reason",
            "method_outputs",
        ):
            object.__setattr__(self, name, getattr(validated, name))

    def bind_truth(
        self,
        *,
        episode_id: str,
        fault_id: int,
        true_label: int,
    ) -> FrozenPointwiseOutput:
        """由工作流把 source 真值绑定到已经完成的方法输出。

        参数：
            episode_id/fault_id/true_label: 来自已校验 ``FrozenFaultEpisode`` 的评价身份。
        返回：
            可进入 JSONL、主表和主图来源的完整最终输出。
        异常：
            真值与最终输出合同不一致时由 ``FrozenPointwiseOutput`` 抛出 ``ValueError``。
        副作用：
            无；不修改当前运行时输出。
        """

        return FrozenPointwiseOutput(
            episode_id=episode_id,
            raw_index=self.raw_index,
            fault_id=fault_id,
            true_label=true_label,
            detection_score=self.detection_score,
            detection_threshold=self.detection_threshold,
            alarm=self.alarm,
            branch_id=self.branch_id,
            mode=self.mode,
            normal_family_id=self.normal_family_id,
            candidate_ids=self.candidate_ids,
            isolation_outcome=self.isolation_outcome,
            reported_family=self.reported_family,
            isolation_certified=self.isolation_certified,
            suppression_reason=self.suppression_reason,
            method_outputs=self.method_outputs,
        )


@dataclass(frozen=True)
class FrozenRuntimeEpisodeEvaluation:
    """一个无真值 evaluator 调用返回的逐行结果。

    参数：
        outputs: 与输入特征视图逐行对应的非空运行时输出。
    返回：
        不含 episode/fault 身份且 raw index 唯一的不可变结果。
    异常：
        输出为空、类型错误或 raw index 重复时抛出 ``TypeError``/``ValueError``。
    副作用：
        无。
    """

    outputs: tuple[FrozenRuntimePointwiseOutput, ...]

    def __post_init__(self) -> None:
        """冻结输出顺序并拒绝空结果、错误类型和重复行。"""

        outputs = tuple(self.outputs)
        if not outputs:
            raise ValueError("Frozen runtime evaluation must contain pointwise outputs.")
        if any(
            not isinstance(output, FrozenRuntimePointwiseOutput)
            for output in outputs
        ):
            raise TypeError(
                "Frozen runtime evaluation outputs must be FrozenRuntimePointwiseOutput."
            )
        raw_indices = [output.raw_index for output in outputs]
        if len(raw_indices) != len(set(raw_indices)):
            raise ValueError(
                "Frozen runtime raw_indices must be unique within an evaluator call."
            )
        object.__setattr__(self, "outputs", outputs)


@dataclass(frozen=True)
class FrozenEpisodeEvaluation:
    """一份 episode 的逐行输出集合；跨输入对齐由工作流复验。

    参数：
        episode_id/fault_id: 与输入 episode 相同的身份。
        outputs: 非空、raw index 唯一的逐时刻输出元组。
    返回：
        不可变 episode 结果；完整行序与标签随后由工作流交叉核验。
    异常：
        episode/fault 身份不一致、输出为空或 raw index 重复时抛出 ``ValueError``。
    副作用：
        无。
    """

    episode_id: str
    fault_id: int
    outputs: tuple[FrozenPointwiseOutput, ...]

    def __post_init__(self) -> None:
        """保证输出非空、同 episode、同 fault 且 raw index 不重复。"""

        object.__setattr__(
            self,
            "episode_id",
            _require_identifier(self.episode_id, name="evaluation episode_id"),
        )
        if self.fault_id not in _CSTR_FAULT_FAMILIES:
            raise ValueError("Frozen episode evaluation fault_id must be one of 1..8.")
        outputs = tuple(self.outputs)
        if not outputs:
            raise ValueError("Frozen episode evaluation must contain pointwise outputs.")
        if any(output.episode_id != self.episode_id for output in outputs):
            raise ValueError("Frozen pointwise outputs changed episode_id.")
        if any(output.fault_id != self.fault_id for output in outputs):
            raise ValueError("Frozen pointwise outputs changed fault_id.")
        raw_indices = [output.raw_index for output in outputs]
        if len(raw_indices) != len(set(raw_indices)):
            raise ValueError("Frozen pointwise raw_indices must be unique within an episode.")
        object.__setattr__(self, "outputs", outputs)


class FrozenFaultEpisodeSource(Protocol):
    """协议冻结并被 claim 后，返回八份封存 CSTR fault episode 的数据源接口。

    实现必须持久消费 claim 的 fault-access 机会；工作流会在返回后再次验证记录。构造与
    preflight 阶段不得读取故障数值。
    """

    def request_episodes(
        self,
        manifest: "FrozenProtocolManifest",
        *,
        claim: FrozenEvaluationClaim,
    ) -> Sequence[FrozenFaultEpisode]:
        """复验 claim 并持久消费一次访问后返回八 episode。

        参数：
            manifest/claim: 已冻结协议及其一次性 token。
        返回：
            顺序与 manifest 完全一致的八份 episode。
        异常：
            许可、hash、账本、访问记录或 loader 失败时按实现契约抛出。
        副作用：
            必须写 fault-access 记录并读取一次故障数值；不得拟合或回写 manifest。
        """


@dataclass
class LazyFrozenCSTRFaultSource:
    """把 P2 正常协议门禁与一个延迟执行的 CSTR episode loader 绑定。

    参数：
        bundle: 已声明 normal/fault 原始 hash 和许可、但没有预装故障数组的
            ``PaperDataBundle``。
        loader: 零参数可调用对象；只有所有 hash/账本/许可检查通过且 P2 门禁被消费后才
            调用。它应读取原始故障文件并返回八个 ``FrozenFaultEpisode``。
        fault_source_hash: loader 将读取的原始故障文件 SHA-256。
    返回：
        同进程 synthetic/P2 bundle 使用的一次性 source。
    异常：
        bundle、hash、许可或协议状态非法时抛出 ``TypeError``/``ValueError``/
        ``ProtocolAccessError``。
    副作用：
        构造无副作用；请求会消费 bundle/source/共享 registry 三层状态并调用 loader。
    重要约束：
        构造不调用 loader。授权成功后先把 source 与 bundle 标为已访问，再执行 loader；
        因此异常也不会产生同一 source 的第二次读取机会。
    """

    bundle: PaperDataBundle
    loader: Callable[[], Sequence[FrozenFaultEpisode]]
    fault_source_hash: str
    _requested: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """只校验静态依赖，不触发故障加载或协议状态变化。"""

        _require_sha256(self.fault_source_hash, name="lazy fault_source_hash")
        if not callable(self.loader):
            raise TypeError("Lazy frozen CSTR loader must be callable.")

    def request_episodes(
        self,
        manifest: "FrozenProtocolManifest",
        *,
        claim: FrozenEvaluationClaim,
    ) -> tuple[FrozenFaultEpisode, ...]:
        """通过正常协议、来源 hash 和许可门禁后恰好调用一次 loader。

        参数：
            manifest: 已从磁盘复验的 P10 manifest。
            claim: 与 manifest 绑定的持久 token。
        返回：
            loader 返回值的元组副本；身份和逐行覆盖随后由工作流再次核对。
        异常：
            source/bundle 已访问、协议未冻结、账本不完整、hash/许可不一致时抛出
            ``ProtocolAccessError``；loader 自身异常按原类型传播。
        副作用：
            成功越过门禁时立即消费 bundle 与 source 的一次访问状态，然后读取故障数据。
        """

        claim.verify(manifest)
        if self._requested or self.bundle.fault_accessed:
            raise ProtocolAccessError(
                "Frozen CSTR fault episode source was already accessed."
            )
        normal_source_hash = self.bundle.split_result.source_hash
        if normal_source_hash is None:
            raise ProtocolAccessError(
                "Frozen CSTR source requires a declared normal source hash."
            )
        if manifest.raw_data_hashes.get("normal") != normal_source_hash:
            raise ProtocolAccessError(
                "Frozen manifest normal raw hash differs from the normal data bundle."
            )
        if manifest.raw_data_hashes.get("fault") != self.fault_source_hash:
            raise ProtocolAccessError(
                "Frozen manifest fault raw hash differs from the lazy source."
            )
        if any(
            episode.source_hash != self.fault_source_hash
            for episode in manifest.fault_episode_manifest
        ):
            raise ProtocolAccessError(
                "Frozen episode manifest source hashes differ from the lazy source."
            )
        self.bundle.authorize_frozen_fault_episode_source(self.fault_source_hash)
        self._requested = True
        claim.consume_fault_access(manifest)
        return tuple(self.loader())


class FrozenEpisodeEvaluator(Protocol):
    """消费无标签只读特征视图，并返回每行 P4--P9 方法输出的运行时接口。

    evaluator 必须在正常数据阶段冻结；正式调用只能前向计算，不得拟合、重校准、选择 branch
    或修改 checkpoint。协议不提供 manifest、onset、故障 ID/族或真实标签。
    """

    def evaluate_episode(
        self,
        episode: FrozenEpisodeInput,
    ) -> FrozenRuntimeEpisodeEvaluation:
        """对当前 episode 执行已冻结方法。

        参数：
            episode: 只含原始行号和过程变量的无真值输入。
        返回：
            每个输入行恰好一个无真值输出的 ``FrozenRuntimeEpisodeEvaluation``。
        异常：
            runtime、shape、数值或冻结证据错误按具体实现传播。
        副作用：
            只允许模型前向和进程内状态推进；禁止拟合或重校准。
        """


@dataclass(frozen=True)
class FrozenEvaluationResult:
    """一次完成评价的固定产物位置与行数摘要。

    参数：
        evaluation_id/manifest_hash/artifact_dir/pointwise_row_count: 完成运行身份与规模。
        其余 ``*_path``: JSONL、三份 CSV、index 与 receipt 的绝对路径。
    返回：
        工作流或独立 verifier 产生的不可变摘要。
    异常：
        直接构造不做 I/O 校验；调用方应使用工作流/verifier。
    副作用：
        无。
    """

    evaluation_id: str
    manifest_hash: str
    artifact_dir: Path
    pointwise_row_count: int
    pointwise_path: Path
    score_trajectory_path: Path
    detection_source_path: Path
    isolation_source_path: Path
    artifact_index_path: Path
    receipt_path: Path


@dataclass(frozen=True)
class FrozenProtocolManifest:
    """完整绑定一次 P10 正式评价之前的全部正常数据证据。

    公开调用方应使用 :meth:`freeze` 创建并独占写入文件，或用 :meth:`load` 读取并复验。
    直接构造仍会执行同样的 hash 和字段校验，不能借助 dataclass 构造绕过冻结条件。

    ``resolved_config``、split/ledger、P7 library、P5 monitor policy 与认证摘要会递归转换为
    ``MappingProxyType`` 和元组，因此 ``frozen=True`` 不只是禁止替换顶层属性，也禁止
    修改嵌套配置。

    参数：
        schema/status/protocol/evaluation/git/config/registry/dependency/raw/split/episode/seed:
            一次评价的协议、环境与数据身份。
        checkpoint/replay/ledger/normal_artifacts/postfilter/monitor/calibrations/certification:
            P2--P9 正常阶段冻结的全部方法证据；normal_artifacts 逐文件绑定训练历史、
            checkpoint replay 输出和 P5--P9 产物。
        manifest_hash: 除自身外全部字段的 SHA-256。
    返回：
        深层不可变且自验证的 schema-v1 manifest。
    异常：
        字段、严格嵌套配置、hash、许可、风险或跨阶段身份不一致时抛出
        ``ValueError``/``FrozenProtocolIntegrityError``。
    副作用：
        直接构造会复验 checkpoint 文件 hash；``freeze`` 还会独占写 JSON。
    """

    schema_version: int
    status: Literal["frozen"]
    protocol_version: str
    evaluation_id: str
    git_commit: str
    resolved_config: Mapping[str, Any]
    config_provenance: Mapping[str, Any]
    config_hash: str
    claim_registry_path: str
    dependency_versions: Mapping[str, str]
    raw_data_hashes: Mapping[str, str]
    split_manifest: Mapping[str, Any]
    fault_episode_manifest: tuple[FrozenFaultEpisodeManifest, ...]
    seeds: Mapping[str, int]
    checkpoint_paths: Mapping[str, str]
    checkpoint_hashes: Mapping[str, str]
    checkpoint_replay: Mapping[str, Any]
    fit_access_ledger: Mapping[str, Any]
    normal_artifacts: FrozenNormalArtifactBundle | None
    postfilter_library: Mapping[str, Any]
    monitor_policy: Mapping[str, Any]
    detection_calibration: FrozenRiskCalibration
    attribution_calibration: FrozenRiskCalibration
    certification_status: Mapping[str, Any]
    manifest_hash: str

    def __post_init__(self) -> None:
        """深层冻结全部映射并核对跨字段不变量与 manifest hash。"""

        if self.schema_version != 1:
            raise FrozenProtocolIntegrityError("Unsupported frozen manifest schema_version.")
        if self.status != "frozen":
            raise FrozenProtocolIntegrityError("Frozen protocol status must be 'frozen'.")
        object.__setattr__(
            self,
            "protocol_version",
            _require_identifier(self.protocol_version, name="protocol_version"),
        )
        object.__setattr__(
            self,
            "evaluation_id",
            _require_identifier(self.evaluation_id, name="evaluation_id"),
        )
        if not _GIT_COMMIT_RE.fullmatch(self.git_commit):
            raise ValueError("Frozen git_commit must be a 40- or 64-character lowercase hex id.")
        if not _CONFIG_HASH_RE.fullmatch(self.config_hash):
            raise ValueError("Frozen config_hash must be 16 lowercase hex characters.")
        _require_sha256(self.manifest_hash, name="manifest_hash")
        expected_hash = _sha256_json(self._payload())
        if self.manifest_hash != expected_hash:
            raise FrozenProtocolIntegrityError(
                "Frozen protocol manifest hash does not match its content."
            )

        resolved_config_json = _json_copy(self.resolved_config)
        entry_fields = set(FrozenEvaluationEntryConfig.model_fields)
        extra_fields = set(resolved_config_json).difference(entry_fields)
        if not extra_fields.issubset({"normal_method_config"}):
            raise FrozenProtocolIntegrityError(
                "Frozen resolved_config contains unknown top-level fields: "
                f"{sorted(extra_fields)}."
            )
        entry_config = FrozenEvaluationEntryConfig.model_validate(
            {
                name: resolved_config_json[name]
                for name in entry_fields
                if name in resolved_config_json
            }
        )
        resolved_config = _deep_freeze_json(
            resolved_config_json,
            name="resolved_config",
        )
        config_provenance = _deep_freeze_json(
            self.config_provenance,
            name="config_provenance",
        )
        if not config_provenance:
            raise ValueError("Frozen config_provenance cannot be empty.")
        _validate_config_provenance(
            resolved_config_json,
            config_provenance,
        )
        run_mode = entry_config.mode
        if run_mode not in {"smoke", "frozen"}:
            raise ValueError(
                "Frozen protocol manifest requires resolved_config mode smoke or frozen."
            )
        normal_method_config = resolved_config_json.get("normal_method_config")
        if run_mode == "frozen" and normal_method_config is None:
            raise FrozenProtocolIntegrityError(
                "Formal frozen resolved_config requires normal_method_config."
            )
        if normal_method_config is not None:
            try:
                PaperNormalMethodConfig.model_validate(normal_method_config)
            except ValidationError as exc:
                raise FrozenProtocolIntegrityError(
                    "Frozen normal_method_config does not match the strict P4 schema."
                ) from exc
        encoded_config = json.dumps(
            resolved_config_json,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if hashlib.sha256(encoded_config).hexdigest()[:16] != self.config_hash:
            raise FrozenProtocolIntegrityError(
                "Frozen config_hash does not match the strict resolved_config."
            )
        claim_registry_path = Path(self.claim_registry_path)
        if not claim_registry_path.is_absolute():
            raise ValueError("Frozen claim_registry_path must be absolute.")
        claim_registry_path = claim_registry_path.resolve()
        configured_registry = entry_config.claim_registry
        if configured_registry.is_absolute() and configured_registry.resolve() != claim_registry_path:
            raise FrozenProtocolIntegrityError(
                "Frozen claim_registry_path differs from the absolute resolved config path."
            )
        if run_mode == "frozen" and entry_config.dataset.license_status != "verified":
            raise FrozenProtocolIntegrityError(
                "A formal frozen manifest requires dataset license_status='verified'."
            )
        if run_mode == "smoke" and entry_config.dataset.license_status != "synthetic_only":
            raise FrozenProtocolIntegrityError(
                "A smoke manifest requires dataset license_status='synthetic_only'."
            )
        dependency_versions = _frozen_string_mapping(
            self.dependency_versions,
            name="dependency_versions",
        )
        missing_dependencies = sorted(
            _REQUIRED_DEPENDENCIES.difference(dependency_versions)
        )
        if missing_dependencies:
            raise ValueError(
                "Frozen dependency_versions is missing: "
                + ", ".join(missing_dependencies)
                + "."
            )
        raw_data_hashes = _frozen_hash_mapping(
            self.raw_data_hashes,
            name="raw_data_hashes",
        )
        for required_hash in ("normal", "fault"):
            if required_hash not in raw_data_hashes:
                raise ValueError(
                    f"Frozen raw_data_hashes must include {required_hash!r}."
                )
        split_manifest = _deep_freeze_json(
            self.split_manifest,
            name="split_manifest",
        )
        _validate_split_manifest(split_manifest)
        fault_episode_manifest = tuple(self.fault_episode_manifest)
        _validate_fault_episode_library(fault_episode_manifest)
        seeds = _frozen_seed_mapping(self.seeds)
        checkpoint_paths = _frozen_string_mapping(
            self.checkpoint_paths,
            name="checkpoint_paths",
        )
        checkpoint_hashes = _frozen_hash_mapping(
            self.checkpoint_hashes,
            name="checkpoint_hashes",
        )
        if not checkpoint_hashes:
            raise ValueError("Frozen checkpoint_hashes cannot be empty.")
        if set(checkpoint_paths) != set(checkpoint_hashes):
            raise ValueError(
                "Frozen checkpoint_paths and checkpoint_hashes must use identical names."
            )
        for checkpoint_name, checkpoint_path in checkpoint_paths.items():
            path = Path(checkpoint_path)
            if not path.is_absolute():
                raise ValueError("Frozen checkpoint paths must be absolute.")
            try:
                observed_hash = sha256_file(path)
            except OSError as exc:
                raise FrozenProtocolIntegrityError(
                    f"Frozen checkpoint {checkpoint_name!r} cannot be read at {path}."
                ) from exc
            if observed_hash != checkpoint_hashes[checkpoint_name]:
                raise FrozenProtocolIntegrityError(
                    f"Frozen checkpoint hash does not match for {checkpoint_name!r}."
                )
        checkpoint_replay = _deep_freeze_json(
            self.checkpoint_replay,
            name="checkpoint_replay",
        )
        _validate_checkpoint_replay(
            checkpoint_replay,
            checkpoint_hashes=checkpoint_hashes,
            run_mode=str(run_mode),
            normal_artifacts=self.normal_artifacts,
        )
        fit_access_ledger = _deep_freeze_json(
            self.fit_access_ledger,
            name="fit_access_ledger",
        )
        _validate_fit_access_ledger(
            fit_access_ledger,
            split_manifest=split_manifest,
            normal_artifacts=self.normal_artifacts,
        )
        normal_artifacts = self.normal_artifacts
        if run_mode == "frozen" and normal_artifacts is None:
            raise FrozenProtocolIntegrityError(
                "Formal frozen manifest requires file-bound normal_artifacts."
            )
        if normal_artifacts is not None and not isinstance(
            normal_artifacts,
            FrozenNormalArtifactBundle,
        ):
            raise TypeError(
                "Frozen normal_artifacts must be a FrozenNormalArtifactBundle."
            )
        if normal_artifacts is not None:
            _validate_normal_artifact_bundle(
                normal_artifacts,
                checkpoint_hashes=checkpoint_hashes,
            )
        postfilter_config = _FrozenPostfilterLibraryConfig.model_validate(
            _json_copy(self.postfilter_library)
        )
        postfilter_library = _deep_freeze_json(
            postfilter_config.model_dump(mode="json", exclude_none=True),
            name="postfilter_library",
        )
        monitor_config = _FrozenMonitorPolicyConfig.model_validate(
            _json_copy(self.monitor_policy)
        )
        monitor_policy = _deep_freeze_json(
            monitor_config.model_dump(mode="json", exclude_none=True),
            name="monitor_policy",
        )
        certification_config = _FrozenCertificationStatusConfig.model_validate(
            _json_copy(self.certification_status)
        )
        certification_status = _deep_freeze_json(
            certification_config.model_dump(mode="json", exclude_none=True),
            name="certification_status",
        )
        if run_mode == "frozen":
            assert normal_artifacts is not None
            _validate_formal_certification_artifact_bindings(
                normal_artifacts,
                certification_status=certification_status,
            )
        if self.detection_calibration.name != "detection":
            raise ValueError("detection_calibration must have name='detection'.")
        if self.attribution_calibration.name != "attribution":
            raise ValueError("attribution_calibration must have name='attribution'.")
        if not math.isclose(
            self.detection_calibration.requested_risk,
            entry_config.detection_risk,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "Frozen detection calibration risk must match resolved config detection_risk."
            )
        if not math.isclose(
            self.attribution_calibration.requested_risk,
            entry_config.attribution_risk,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "Frozen attribution calibration risk must match resolved config "
                "attribution_risk."
            )
        split_stages = _strict_mapping(
            split_manifest["stages"],
            name="split stages",
        )
        expected_detection_hash = _strict_mapping(
            split_stages[StageName.DETECTION_CALIBRATION.value],
            name="detection calibration split stage",
        )["data_hash"]
        if self.detection_calibration.source_hash != expected_detection_hash:
            raise FrozenProtocolIntegrityError(
                "Frozen detection calibration source_hash differs from its P2 split stage."
            )
        _validate_calibration_episode_binding(
            self.detection_calibration,
            stage=_strict_mapping(
                split_stages[StageName.DETECTION_CALIBRATION.value],
                name="detection calibration split stage",
            ),
        )
        expected_attribution_hash = _strict_mapping(
            split_stages[StageName.ATTRIBUTION_CALIBRATION.value],
            name="attribution calibration split stage",
        )["data_hash"]
        if self.attribution_calibration.source_hash != expected_attribution_hash:
            raise FrozenProtocolIntegrityError(
                "Frozen attribution calibration source_hash differs from its P2 split stage."
            )
        _validate_calibration_episode_binding(
            self.attribution_calibration,
            stage=_strict_mapping(
                split_stages[StageName.ATTRIBUTION_CALIBRATION.value],
                name="attribution calibration split stage",
            ),
        )
        if (
            self.detection_calibration.source_hash
            == self.attribution_calibration.source_hash
        ):
            raise ValueError(
                "Detection and attribution calibration must use independent source hashes."
            )
        if normal_artifacts is not None:
            checkpoint_name = normal_artifacts.runtime_evaluator["checkpoint_name"]
            validate_protected_evaluator_artifact_bindings(
                normal_artifacts.artifact_paths[
                    f"checkpoint_files.{checkpoint_name}"
                ],
                postfilter_library=postfilter_library,
                monitor_policy=monitor_policy,
                monitoring_score_scaler=_read_bound_json_artifact(
                    normal_artifacts,
                    "monitoring_score_scaler",
                ),
                isolation_library=_read_bound_json_artifact(
                    normal_artifacts,
                    "isolation_library",
                ),
                detection_quantile=self.detection_calibration.quantile,
            )

        object.__setattr__(self, "resolved_config", resolved_config)
        object.__setattr__(self, "claim_registry_path", str(claim_registry_path))
        object.__setattr__(self, "config_provenance", config_provenance)
        object.__setattr__(self, "dependency_versions", dependency_versions)
        object.__setattr__(self, "raw_data_hashes", raw_data_hashes)
        object.__setattr__(self, "split_manifest", split_manifest)
        object.__setattr__(self, "fault_episode_manifest", fault_episode_manifest)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "checkpoint_paths", checkpoint_paths)
        object.__setattr__(self, "checkpoint_hashes", checkpoint_hashes)
        object.__setattr__(self, "checkpoint_replay", checkpoint_replay)
        object.__setattr__(self, "fit_access_ledger", fit_access_ledger)
        object.__setattr__(self, "normal_artifacts", normal_artifacts)
        object.__setattr__(self, "postfilter_library", postfilter_library)
        object.__setattr__(self, "monitor_policy", monitor_policy)
        object.__setattr__(self, "certification_status", certification_status)

    @classmethod
    def build(
        cls,
        *,
        protocol_version: str,
        evaluation_id: str,
        git_commit: str,
        resolved_config: Mapping[str, Any],
        config_provenance: Mapping[str, Any],
        config_hash: str,
        claim_registry_path: str | Path,
        dependency_versions: Mapping[str, str],
        raw_data_hashes: Mapping[str, str],
        split_manifest: Mapping[str, Any],
        fault_episode_manifest: Sequence[FrozenFaultEpisodeManifest],
        seeds: Mapping[str, int],
        checkpoint_paths: Mapping[str, str | Path],
        checkpoint_hashes: Mapping[str, str],
        checkpoint_replay: Mapping[str, Any],
        fit_access_ledger: Mapping[str, Any],
        normal_artifacts: FrozenNormalArtifactBundle | None,
        postfilter_library: Mapping[str, Any],
        monitor_policy: Mapping[str, Any],
        detection_calibration: FrozenRiskCalibration,
        attribution_calibration: FrozenRiskCalibration,
        certification_status: Mapping[str, Any],
    ) -> "FrozenProtocolManifest":
        """在内存中创建并完整校验一份 manifest，不写任何运行文件。

        参数：
            所有参数: P10 规格要求的冻结前证据。映射必须是 JSON 兼容值。
        返回：
            深层不可变且已重放全部 P2--P9 证据的 ``FrozenProtocolManifest``。
        异常：
            字段、引用 checkpoint 或跨阶段证据不完整时抛出 ``ValueError`` 或
            ``FrozenProtocolIntegrityError``。
        副作用：
            只读引用的 checkpoint 以复验 hash；不写 manifest、claim 或故障访问记录。
        """

        normalized_checkpoint_paths = {
            str(name): str(Path(checkpoint_path).resolve())
            for name, checkpoint_path in checkpoint_paths.items()
        }
        raw_payload = {
            "schema_version": 1,
            "status": "frozen",
            "protocol_version": protocol_version,
            "evaluation_id": evaluation_id,
            "git_commit": git_commit,
            "resolved_config": _json_copy(resolved_config),
            "config_provenance": _json_copy(config_provenance),
            "config_hash": config_hash,
            "claim_registry_path": str(Path(claim_registry_path).resolve()),
            "dependency_versions": dict(dependency_versions),
            "raw_data_hashes": dict(raw_data_hashes),
            "split_manifest": _json_copy(split_manifest),
            "fault_episode_manifest": [
                episode.to_dict() for episode in fault_episode_manifest
            ],
            "seeds": dict(seeds),
            "checkpoint_paths": normalized_checkpoint_paths,
            "checkpoint_hashes": dict(checkpoint_hashes),
            "checkpoint_replay": _json_copy(checkpoint_replay),
            "fit_access_ledger": _json_copy(fit_access_ledger),
            "normal_artifacts": (
                None if normal_artifacts is None else normal_artifacts.to_dict()
            ),
            "postfilter_library": _json_copy(postfilter_library),
            "monitor_policy": _json_copy(monitor_policy),
            "detection_calibration": detection_calibration.to_dict(),
            "attribution_calibration": attribution_calibration.to_dict(),
            "certification_status": _json_copy(certification_status),
        }
        return cls(
            schema_version=1,
            status="frozen",
            protocol_version=protocol_version,
            evaluation_id=evaluation_id,
            git_commit=git_commit,
            resolved_config=resolved_config,
            config_provenance=config_provenance,
            config_hash=config_hash,
            claim_registry_path=str(Path(claim_registry_path).resolve()),
            dependency_versions=dependency_versions,
            raw_data_hashes=raw_data_hashes,
            split_manifest=split_manifest,
            fault_episode_manifest=tuple(fault_episode_manifest),
            seeds=seeds,
            checkpoint_paths=normalized_checkpoint_paths,
            checkpoint_hashes=checkpoint_hashes,
            checkpoint_replay=checkpoint_replay,
            fit_access_ledger=fit_access_ledger,
            normal_artifacts=normal_artifacts,
            postfilter_library=postfilter_library,
            monitor_policy=monitor_policy,
            detection_calibration=detection_calibration,
            attribution_calibration=attribution_calibration,
            certification_status=certification_status,
            manifest_hash=_sha256_json(raw_payload),
        )

    @classmethod
    def freeze(
        cls,
        path: str | Path,
        *,
        protocol_version: str,
        evaluation_id: str,
        git_commit: str,
        resolved_config: Mapping[str, Any],
        config_provenance: Mapping[str, Any],
        config_hash: str,
        claim_registry_path: str | Path,
        dependency_versions: Mapping[str, str],
        raw_data_hashes: Mapping[str, str],
        split_manifest: Mapping[str, Any],
        fault_episode_manifest: Sequence[FrozenFaultEpisodeManifest],
        seeds: Mapping[str, int],
        checkpoint_paths: Mapping[str, str | Path],
        checkpoint_hashes: Mapping[str, str],
        checkpoint_replay: Mapping[str, Any],
        fit_access_ledger: Mapping[str, Any],
        normal_artifacts: FrozenNormalArtifactBundle | None,
        postfilter_library: Mapping[str, Any],
        monitor_policy: Mapping[str, Any],
        detection_calibration: FrozenRiskCalibration,
        attribution_calibration: FrozenRiskCalibration,
        certification_status: Mapping[str, Any],
    ) -> "FrozenProtocolManifest":
        """构建并以只创建不覆盖语义持久化一份新 manifest。

        参数：
            path: 新 manifest 文件路径；已存在时抛出 ``FileExistsError``。
            其余参数: 与 :meth:`build` 相同的 P10 冻结前证据。
        返回：
            与磁盘内容一致的深层不可变 manifest。
        异常：
            证据校验异常由 :meth:`build` 传播；目标已存在或写入失败时传播
            ``FileExistsError``/``OSError``。
        副作用：
            只在所有内存校验通过后创建父目录和一个 UTF-8 JSON 文件；不创建 claim。
        """

        manifest = cls.build(
            protocol_version=protocol_version,
            evaluation_id=evaluation_id,
            git_commit=git_commit,
            resolved_config=resolved_config,
            config_provenance=config_provenance,
            config_hash=config_hash,
            claim_registry_path=claim_registry_path,
            dependency_versions=dependency_versions,
            raw_data_hashes=raw_data_hashes,
            split_manifest=split_manifest,
            fault_episode_manifest=fault_episode_manifest,
            seeds=seeds,
            checkpoint_paths=checkpoint_paths,
            checkpoint_hashes=checkpoint_hashes,
            checkpoint_replay=checkpoint_replay,
            fit_access_ledger=fit_access_ledger,
            normal_artifacts=normal_artifacts,
            postfilter_library=postfilter_library,
            monitor_policy=monitor_policy,
            detection_calibration=detection_calibration,
            attribution_calibration=attribution_calibration,
            certification_status=certification_status,
        )
        return manifest.save(path)

    def save(self, path: str | Path) -> "FrozenProtocolManifest":
        """以 exclusive-create 语义保存已经完整校验的 manifest。

        参数：
            path: 新 JSON 文件路径；父目录可不存在，但目标文件必须不存在。
        返回：
            当前不可变对象，便于 builder 直接把保存结果交给工作流。
        异常：
            目标已存在或文件系统失败时传播 ``FileExistsError``/``OSError``。
        副作用：
            创建父目录并写一份带尾换行的 UTF-8 JSON；不修改对象或其他运行产物。
        """

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.write("\n")
        return self

    @classmethod
    def load(cls, path: str | Path) -> "FrozenProtocolManifest":
        """从 JSON 读取并完整复验 manifest。

        参数：
            path: 已冻结 JSON 文件。
        返回：
            深层不可变且 checkpoint/hash 已重放的 manifest。
        异常：
            文件内容缺失、含未知字段、自带 hash 不匹配或嵌套证据非法时传播
            ``OSError``/``JSONDecodeError``/``FrozenProtocolIntegrityError``。
        副作用：
            读取 manifest 和其引用的 checkpoint；不尝试修复、迁移或覆盖文件。
        """

        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise FrozenProtocolIntegrityError(
                "Frozen protocol manifest must contain a JSON object."
            )
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenProtocolManifest":
        """严格重建 manifest，拒绝未知字段和隐式默认值。

        参数：
            value: schema-v1 JSON 兼容映射。
        返回：
            经 dataclass 全部跨字段校验的 manifest。
        异常：
            缺失/未知字段、类型、hash 或引用 checkpoint 不一致时 fail closed。
        副作用：
            会读取 checkpoint 复验 hash；不写文件。
        """

        expected_keys = {
            "schema_version",
            "status",
            "protocol_version",
            "evaluation_id",
            "git_commit",
            "resolved_config",
            "config_provenance",
            "config_hash",
            "claim_registry_path",
            "dependency_versions",
            "raw_data_hashes",
            "split_manifest",
            "fault_episode_manifest",
            "seeds",
            "checkpoint_paths",
            "checkpoint_hashes",
            "checkpoint_replay",
            "fit_access_ledger",
            "normal_artifacts",
            "postfilter_library",
            "monitor_policy",
            "detection_calibration",
            "attribution_calibration",
            "certification_status",
            "manifest_hash",
        }
        _require_exact_keys(value, expected_keys, name="frozen protocol manifest")
        return cls(
            schema_version=_strict_int(
                value["schema_version"],
                name="schema_version",
            ),
            status=_strict_string(value["status"], name="status"),  # type: ignore[arg-type]
            protocol_version=_strict_string(
                value["protocol_version"],
                name="protocol_version",
            ),
            evaluation_id=_strict_string(
                value["evaluation_id"],
                name="evaluation_id",
            ),
            git_commit=_strict_string(value["git_commit"], name="git_commit"),
            resolved_config=_strict_mapping(
                value["resolved_config"],
                name="resolved_config",
            ),
            config_provenance=_strict_mapping(
                value["config_provenance"],
                name="config_provenance",
            ),
            config_hash=_strict_string(value["config_hash"], name="config_hash"),
            claim_registry_path=_strict_string(
                value["claim_registry_path"],
                name="claim_registry_path",
            ),
            dependency_versions={
                _strict_string(key, name="dependency name"): _strict_string(
                    item,
                    name="dependency version",
                )
                for key, item in _strict_mapping(
                    value["dependency_versions"],
                    name="dependency_versions",
                ).items()
            },
            raw_data_hashes={
                _strict_string(key, name="raw data name"): _strict_string(
                    item,
                    name="raw data hash",
                )
                for key, item in _strict_mapping(
                    value["raw_data_hashes"],
                    name="raw_data_hashes",
                ).items()
            },
            split_manifest=_strict_mapping(
                value["split_manifest"],
                name="split_manifest",
            ),
            fault_episode_manifest=tuple(
                FrozenFaultEpisodeManifest.from_dict(
                    _strict_mapping(item, name="fault episode")
                )
                for item in _strict_sequence(
                    value["fault_episode_manifest"],
                    name="fault_episode_manifest",
                )
            ),
            seeds={
                _strict_string(key, name="seed name"): _strict_int(
                    item,
                    name="seed value",
                )
                for key, item in _strict_mapping(
                    value["seeds"],
                    name="seeds",
                ).items()
            },
            checkpoint_paths={
                _strict_string(key, name="checkpoint name"): _strict_string(
                    item,
                    name="checkpoint path",
                )
                for key, item in _strict_mapping(
                    value["checkpoint_paths"],
                    name="checkpoint_paths",
                ).items()
            },
            checkpoint_hashes={
                _strict_string(key, name="checkpoint name"): _strict_string(
                    item,
                    name="checkpoint hash",
                )
                for key, item in _strict_mapping(
                    value["checkpoint_hashes"],
                    name="checkpoint_hashes",
                ).items()
            },
            checkpoint_replay=_strict_mapping(
                value["checkpoint_replay"],
                name="checkpoint_replay",
            ),
            fit_access_ledger=_strict_mapping(
                value["fit_access_ledger"],
                name="fit_access_ledger",
            ),
            normal_artifacts=(
                None
                if value["normal_artifacts"] is None
                else FrozenNormalArtifactBundle.from_dict(
                    _strict_mapping(
                        value["normal_artifacts"],
                        name="normal_artifacts",
                    )
                )
            ),
            postfilter_library=_strict_mapping(
                value["postfilter_library"],
                name="postfilter_library",
            ),
            monitor_policy=_strict_mapping(
                value["monitor_policy"],
                name="monitor_policy",
            ),
            detection_calibration=FrozenRiskCalibration.from_dict(
                _strict_mapping(
                    value["detection_calibration"],
                    name="detection_calibration",
                )
            ),
            attribution_calibration=FrozenRiskCalibration.from_dict(
                _strict_mapping(
                    value["attribution_calibration"],
                    name="attribution_calibration",
                )
            ),
            certification_status=_strict_mapping(
                value["certification_status"],
                name="certification_status",
            ),
            manifest_hash=_strict_string(
                value["manifest_hash"],
                name="manifest_hash",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """返回完整 JSON 兼容副本，包括自验证 ``manifest_hash``。

        返回：
            新分配的 schema-v1 字典；嵌套只读映射被复制为普通 JSON 容器。
        异常：
            无。
        副作用：
            无。
        """

        return {**self._payload(), "manifest_hash": self.manifest_hash}

    def _payload(self) -> dict[str, Any]:
        """构造参与内容 hash 的稳定 payload，不含 ``manifest_hash`` 本身。"""

        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "protocol_version": self.protocol_version,
            "evaluation_id": self.evaluation_id,
            "git_commit": self.git_commit,
            "resolved_config": _json_copy(self.resolved_config),
            "config_provenance": _json_copy(self.config_provenance),
            "config_hash": self.config_hash,
            "claim_registry_path": self.claim_registry_path,
            "dependency_versions": dict(self.dependency_versions),
            "raw_data_hashes": dict(self.raw_data_hashes),
            "split_manifest": _json_copy(self.split_manifest),
            "fault_episode_manifest": [
                episode.to_dict() for episode in self.fault_episode_manifest
            ],
            "seeds": dict(self.seeds),
            "checkpoint_paths": dict(self.checkpoint_paths),
            "checkpoint_hashes": dict(self.checkpoint_hashes),
            "checkpoint_replay": _json_copy(self.checkpoint_replay),
            "fit_access_ledger": _json_copy(self.fit_access_ledger),
            "normal_artifacts": (
                None
                if self.normal_artifacts is None
                else self.normal_artifacts.to_dict()
            ),
            "postfilter_library": _json_copy(self.postfilter_library),
            "monitor_policy": _json_copy(self.monitor_policy),
            "detection_calibration": self.detection_calibration.to_dict(),
            "attribution_calibration": self.attribution_calibration.to_dict(),
            "certification_status": _json_copy(self.certification_status),
        }


class FrozenEvaluationWorkflow:
    """在共享 claim registry 约束下执行一次完整八 episode 评价。

    生命周期严格为 ``load/verify manifest -> exclusive claim -> request fault episodes ->
    evaluate every row -> write sources -> write integrity index/receipt``。任何阶段失败都会
    保留 claim，因此同一 ID 不可借失败重跑；调用方只能冻结新协议和新 ID。
    """

    def __init__(
        self,
        *,
        manifest_path: str | Path,
        claim_registry: str | Path,
        artifact_dir: str | Path,
        fault_source: FrozenFaultEpisodeSource,
        evaluator: FrozenEpisodeEvaluator,
    ) -> None:
        """绑定路径与运行时实现，但不读取 manifest、故障数据或创建目录。

        参数：
            manifest_path: 已由 ``FrozenProtocolManifest.freeze`` 写入的只读逻辑文件。
            claim_registry: 同一研究项目所有正式评价共享的 ID registry。
            artifact_dir: 当前 ID 的空白输出目录；已有固定产物会被拒绝而非覆盖。
            fault_source: 只有 ``run`` 完成 claim 后才会调用的数据源。
            evaluator: 已冻结的 P4--P9 运行时；每个 episode 调用一次。
        返回：
            无；构造完成后得到尚未开始 I/O 的工作流对象。
        异常：
            构造阶段不主动校验路径或协议实现；参数类型/协议错误会在 ``run`` 的对应边界
            fail closed。
        副作用：
            仅把路径转换为 ``Path`` 并保存依赖；不读 manifest、不建目录、不创建 claim。
        """

        self.manifest_path = Path(manifest_path)
        self.claim_registry = Path(claim_registry)
        self.artifact_dir = Path(artifact_dir)
        self.fault_source = fault_source
        self.evaluator = evaluator

    def run(self) -> FrozenEvaluationResult:
        """执行一次性评价并返回可独立复验的 receipt 摘要。

        参数：
            无；使用构造时绑定的 manifest、registry、artifact 目录、source 与 evaluator。
        返回：
            含全部机器可读来源路径、receipt 路径和逐时刻行数的
            ``FrozenEvaluationResult``。
        异常：
            manifest/episode/output/产物不一致时抛出 ``FrozenProtocolIntegrityError`` 或
            ``FrozenEvaluationArtifactError``；ID 已存在时抛出
            ``FrozenEvaluationAlreadyClaimedError``；运行时与 I/O 错误按原类型传播。
        副作用：
            写共享 claim、当前运行的逐时刻 JSONL、三份 CSV、artifact index 和 receipt；
            不修改 manifest，不删除失败运行，也不释放 claim。
        """

        manifest_bytes = self.manifest_path.read_bytes()
        manifest = FrozenProtocolManifest.load(self.manifest_path)
        self._preflight_artifact_paths()
        self._validate_formal_runtime_identity(manifest)
        claim = FrozenEvaluationClaim.create(
            manifest=manifest,
            claim_registry=self.claim_registry,
            artifact_dir=self.artifact_dir,
        )
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        local_claim_path = self.artifact_dir / "evaluation_claim.json"
        _write_new_json(local_claim_path, claim.to_dict())
        episodes = tuple(
            self.fault_source.request_episodes(manifest, claim=claim)
        )
        claim.verify_fault_access_consumed(manifest)
        self._validate_episodes(manifest, episodes)
        runtime_evaluations = tuple(
            self.evaluator.evaluate_episode(
                FrozenEpisodeInput.from_fault_episode(episode)
            )
            for episode in episodes
        )
        evaluations = tuple(
            self._bind_evaluation_truth(episode, evaluation)
            for episode, evaluation in zip(
                episodes,
                runtime_evaluations,
                strict=True,
            )
        )
        self._validate_evaluations(manifest, episodes, evaluations)
        outputs = tuple(
            output for evaluation in evaluations for output in evaluation.outputs
        )
        paths = {
            name: self.artifact_dir / relative
            for name, relative in _ARTIFACT_RELATIVE_PATHS.items()
        }
        _write_pointwise_jsonl(paths["pointwise"], outputs)
        _write_csv(
            paths["score_trajectory"],
            _score_trajectory_rows(outputs),
            fieldnames=(
                "episode_id",
                "fault_id",
                "raw_index",
                "true_label",
                "detection_score",
                "detection_threshold",
                "alarm",
                "branch_id",
                "mode",
                "normal_family_id",
                "isolation_outcome",
                "candidate_ids",
                "reported_family",
                "isolation_certified",
                "suppression_reason",
            ),
        )
        _write_csv(
            paths["detection"],
            _detection_summary_rows(episodes, evaluations),
            fieldnames=(
                "episode_id",
                "fault_id",
                "fault_family",
                "onset",
                "row_count",
                "pre_fault_row_count",
                "post_onset_row_count",
                "pre_fault_alarm_count",
                "post_onset_alarm_count",
                "pre_fault_alarm_rate",
                "post_onset_alarm_rate",
                "first_post_onset_alarm_raw_index",
                "detection_delay_steps",
            ),
        )
        _write_csv(
            paths["isolation"],
            _isolation_summary_rows(episodes, evaluations),
            fieldnames=(
                "episode_id",
                "fault_id",
                "fault_family",
                "Normal-compatible",
                "Nonunique",
                "Out-of-model",
                "singleton",
                "Uncertified",
                "certified_row_count",
                "reported_families",
                "candidate_ids",
            ),
        )
        if self.manifest_path.read_bytes() != manifest_bytes:
            raise FrozenProtocolIntegrityError(
                "Frozen protocol manifest changed during evaluation."
            )

        artifact_index_path = self.artifact_dir / "artifact_index.json"
        indexed_paths = {"evaluation_claim": local_claim_path, **paths}
        artifact_index = {
            "schema_version": 1,
            "evaluation_id": manifest.evaluation_id,
            "manifest_hash": manifest.manifest_hash,
            "artifacts": {
                path.relative_to(self.artifact_dir).as_posix(): sha256_file(path)
                for path in indexed_paths.values()
            },
        }
        _write_new_json(artifact_index_path, artifact_index)
        receipt_path = self.artifact_dir / "evaluation_receipt.json"
        receipt = {
            "schema_version": 1,
            "status": "completed",
            "evaluation_id": manifest.evaluation_id,
            "manifest_hash": manifest.manifest_hash,
            "pointwise_row_count": len(outputs),
            "artifact_index": artifact_index_path.name,
            "artifact_index_hash": sha256_file(artifact_index_path),
        }
        _write_new_json(receipt_path, receipt)
        return FrozenEvaluationResult(
            evaluation_id=manifest.evaluation_id,
            manifest_hash=manifest.manifest_hash,
            artifact_dir=self.artifact_dir,
            pointwise_row_count=len(outputs),
            pointwise_path=paths["pointwise"],
            score_trajectory_path=paths["score_trajectory"],
            detection_source_path=paths["detection"],
            isolation_source_path=paths["isolation"],
            artifact_index_path=artifact_index_path,
            receipt_path=receipt_path,
        )

    def _preflight_artifact_paths(self) -> None:
        """在占用 ID 前发现可避免的固定文件冲突；不清理或覆盖旧产物。"""

        fixed_paths = [
            self.artifact_dir / "evaluation_claim.json",
            self.artifact_dir / "artifact_index.json",
            self.artifact_dir / "evaluation_receipt.json",
            *(
                self.artifact_dir / relative
                for relative in _ARTIFACT_RELATIVE_PATHS.values()
            ),
        ]
        collisions = [path for path in fixed_paths if path.exists()]
        if collisions:
            formatted = ", ".join(str(path) for path in collisions)
            raise FileExistsError(
                "Frozen evaluation artifact paths already exist and will not be overwritten: "
                + formatted
            )

    def _validate_formal_runtime_identity(
        self,
        manifest: FrozenProtocolManifest,
    ) -> None:
        """在 claim 前复验正式 evaluator 能力、checkpoint 身份和认证门。

        参数：
            manifest: 已从磁盘完整重放的冻结协议。
        返回：
            无；smoke/development manifest 不使用正式故障入口，直接返回。
        异常：
            formal evaluator 未声明完整 P4--P9、不是 manifest 绑定的 type/checkpoint，
            或任一认证状态不是 ``certified`` 时抛出 ``FrozenProtocolIntegrityError``。
        副作用：
            无；只读内存字段，不访问故障 source、不创建 claim。
        """

        if manifest.resolved_config["mode"] != "frozen":
            return
        if getattr(self.evaluator, "formal_pipeline_complete", False) is not True:
            raise FrozenProtocolIntegrityError(
                "Frozen evaluator must implement the complete P4--P9 pipeline before "
                "formal fault access."
            )
        normal_artifacts = manifest.normal_artifacts
        if normal_artifacts is None:
            raise FrozenProtocolIntegrityError(
                "Formal evaluator runtime identity requires normal_artifacts."
            )
        checkpoint_name = normal_artifacts.runtime_evaluator["checkpoint_name"]
        expected_identity = {
            "frozen_evaluator_type": normal_artifacts.runtime_evaluator["type"],
            "frozen_checkpoint_name": checkpoint_name,
            "frozen_checkpoint_hash": manifest.checkpoint_hashes[checkpoint_name],
        }
        observed_identity = {
            name: getattr(self.evaluator, name, None)
            for name in expected_identity
        }
        if observed_identity != expected_identity:
            raise FrozenProtocolIntegrityError(
                "Frozen evaluator runtime identity differs from the manifest-bound "
                "type/checkpoint."
            )
        for name in ("operator", "signature", "nuisance"):
            if manifest.certification_status[name]["status"] != "certified":
                raise FrozenProtocolIntegrityError(
                    f"Formal evaluator requires certified {name} evidence before claim."
                )

    @staticmethod
    def _validate_episodes(
        manifest: FrozenProtocolManifest,
        episodes: tuple[FrozenFaultEpisode, ...],
    ) -> None:
        """要求 source 返回与冻结顺序和身份完全一致的八个 episode。"""

        if len(episodes) != len(manifest.fault_episode_manifest):
            raise FrozenProtocolIntegrityError(
                "Frozen fault source did not return all eight manifest episodes."
            )
        for expected, episode in zip(
            manifest.fault_episode_manifest,
            episodes,
            strict=True,
        ):
            if episode.manifest != expected:
                raise FrozenProtocolIntegrityError(
                    "Frozen fault source episode identity/order differs from the manifest."
                )

    @staticmethod
    def _validate_evaluations(
        manifest: FrozenProtocolManifest,
        episodes: tuple[FrozenFaultEpisode, ...],
        evaluations: tuple[FrozenEpisodeEvaluation, ...],
    ) -> None:
        """核对每个输入行恰有一个同序输出，不允许跳行、补行或重排。"""

        frozen_branches = _postfilter_branch_ids(manifest.postfilter_library)
        frozen_mode = _strict_string(
            manifest.postfilter_library["mode"],
            name="postfilter mode",
        )
        operator_status = _strict_mapping(
            manifest.certification_status["operator"],
            name="operator certification status",
        ).get("status")
        if len(evaluations) != len(episodes):
            raise FrozenProtocolIntegrityError(
                "Frozen evaluator did not return one result per episode."
            )
        for episode, evaluation in zip(episodes, evaluations, strict=True):
            if (
                evaluation.episode_id != episode.manifest.episode_id
                or evaluation.fault_id != episode.manifest.fault_id
            ):
                raise FrozenProtocolIntegrityError(
                    "Frozen evaluator changed episode or fault identity."
                )
            output_indices = np.asarray(
                [output.raw_index for output in evaluation.outputs],
                dtype=np.int64,
            )
            output_labels = np.asarray(
                [output.true_label for output in evaluation.outputs],
                dtype=np.int64,
            )
            if not np.array_equal(output_indices, episode.raw_indices):
                raise FrozenProtocolIntegrityError(
                    f"Frozen evaluator output coverage/order is incomplete for "
                    f"{evaluation.episode_id!r}."
                )
            if not np.array_equal(output_labels, episode.labels):
                raise FrozenProtocolIntegrityError(
                    f"Frozen evaluator changed true labels for {evaluation.episode_id!r}."
                )
            for output in evaluation.outputs:
                if output.branch_id not in frozen_branches:
                    raise FrozenProtocolIntegrityError(
                        f"Frozen evaluator used branch {output.branch_id!r} outside the "
                        "post-filter branch library."
                    )
                if output.mode != frozen_mode:
                    raise FrozenProtocolIntegrityError(
                        f"Frozen evaluator used mode {output.mode!r}, but the manifest "
                        f"froze mode {frozen_mode!r}."
                    )
                if output.isolation_certified and operator_status != "certified":
                    raise FrozenProtocolIntegrityError(
                        "Frozen evaluator reported certified isolation without a certified "
                        "operator status."
                    )

    @staticmethod
    def _bind_evaluation_truth(
        episode: FrozenFaultEpisode,
        evaluation: FrozenRuntimeEpisodeEvaluation,
    ) -> FrozenEpisodeEvaluation:
        """在 evaluator 返回后绑定 source 真值，算法调用期间不可见。

        参数：
            episode: 带 manifest/labels 的完整 source 结果，仅供评价层使用。
            evaluation: 只含方法输出的 runtime 结果。
        返回：
            episode 身份和每行真实标签均来自 source 的最终评价对象。
        异常：
            runtime 输出覆盖、顺序或行号与 source 不一致时抛出
            ``FrozenProtocolIntegrityError``。
        副作用：
            无；只构造新的最终输出对象。
        """

        output_indices = np.asarray(
            [output.raw_index for output in evaluation.outputs],
            dtype=np.int64,
        )
        if not np.array_equal(output_indices, episode.raw_indices):
            raise FrozenProtocolIntegrityError(
                "Frozen evaluator runtime output coverage/order is incomplete."
            )
        outputs = tuple(
            output.bind_truth(
                episode_id=episode.manifest.episode_id,
                fault_id=episode.manifest.fault_id,
                true_label=int(label),
            )
            for output, label in zip(
                evaluation.outputs,
                episode.labels,
                strict=True,
            )
        )
        return FrozenEpisodeEvaluation(
            episode_id=episode.manifest.episode_id,
            fault_id=episode.manifest.fault_id,
            outputs=outputs,
        )


def verify_frozen_evaluation_artifacts(
    *,
    manifest_path: str | Path,
    receipt_path: str | Path,
) -> FrozenEvaluationResult:
    """从 manifest、receipt 和 artifact index 独立复验一次评价。

    参数：
        manifest_path: 原始冻结 manifest。
        receipt_path: 工作流最后写入的 completion receipt。
    返回：
        与 ``FrozenEvaluationWorkflow.run`` 相同的路径/行数摘要。
    异常：
        任一字段、路径边界、SHA-256 或 JSONL 行数不一致时抛出
        ``FrozenEvaluationArtifactError``。
    副作用：
        只读文件并计算 hash；不修复或覆盖产物。
    """

    manifest = FrozenProtocolManifest.load(manifest_path)
    receipt = _read_json_mapping(Path(receipt_path), name="evaluation receipt")
    _require_exact_keys(
        receipt,
        {
            "schema_version",
            "status",
            "evaluation_id",
            "manifest_hash",
            "pointwise_row_count",
            "artifact_index",
            "artifact_index_hash",
        },
        name="evaluation receipt",
    )
    if receipt["schema_version"] != 1 or receipt["status"] != "completed":
        raise FrozenEvaluationArtifactError(
            "Frozen evaluation receipt is not a completed schema-v1 receipt."
        )
    if (
        receipt["evaluation_id"] != manifest.evaluation_id
        or receipt["manifest_hash"] != manifest.manifest_hash
    ):
        raise FrozenEvaluationArtifactError(
            "Frozen evaluation receipt does not belong to the supplied manifest."
        )
    output_dir = Path(receipt_path).parent.resolve()
    index_name = _strict_string(receipt["artifact_index"], name="artifact_index")
    artifact_index_path = _resolve_inside(output_dir, Path(index_name))
    expected_index_hash = _strict_string(
        receipt["artifact_index_hash"],
        name="artifact_index_hash",
    )
    _require_sha256(expected_index_hash, name="artifact_index_hash")
    if sha256_file(artifact_index_path) != expected_index_hash:
        raise FrozenEvaluationArtifactError("Frozen artifact index hash does not match.")
    index = _read_json_mapping(artifact_index_path, name="artifact index")
    _require_exact_keys(
        index,
        {"schema_version", "evaluation_id", "manifest_hash", "artifacts"},
        name="artifact index",
    )
    if (
        index["schema_version"] != 1
        or index["evaluation_id"] != manifest.evaluation_id
        or index["manifest_hash"] != manifest.manifest_hash
    ):
        raise FrozenEvaluationArtifactError(
            "Frozen artifact index identity does not match the manifest."
        )
    artifacts = _strict_mapping(index["artifacts"], name="indexed artifacts")
    expected_relative_paths = {
        "evaluation_claim.json",
        *(
            relative.as_posix()
            for relative in _ARTIFACT_RELATIVE_PATHS.values()
        ),
    }
    if set(artifacts) != expected_relative_paths:
        raise FrozenEvaluationArtifactError(
            "Frozen artifact index does not contain the exact required source set."
        )
    for relative_name, expected_hash_value in artifacts.items():
        relative = Path(_strict_string(relative_name, name="artifact relative path"))
        artifact_path = _resolve_inside(output_dir, relative)
        expected_hash = _strict_string(
            expected_hash_value,
            name=f"artifact hash {relative_name}",
        )
        _require_sha256(expected_hash, name=f"artifact hash {relative_name}")
        if sha256_file(artifact_path) != expected_hash:
            raise FrozenEvaluationArtifactError(
                f"Frozen artifact hash does not match for {relative_name!r}."
            )
    pointwise_row_count = _strict_int(
        receipt["pointwise_row_count"],
        name="pointwise_row_count",
    )
    pointwise_path = output_dir / _ARTIFACT_RELATIVE_PATHS["pointwise"]
    with pointwise_path.open(encoding="utf-8") as stream:
        observed_rows = sum(1 for line in stream if line.strip())
    if observed_rows != pointwise_row_count:
        raise FrozenEvaluationArtifactError(
            "Frozen pointwise JSONL row count does not match the receipt."
        )
    return FrozenEvaluationResult(
        evaluation_id=manifest.evaluation_id,
        manifest_hash=manifest.manifest_hash,
        artifact_dir=output_dir,
        pointwise_row_count=pointwise_row_count,
        pointwise_path=pointwise_path,
        score_trajectory_path=output_dir
        / _ARTIFACT_RELATIVE_PATHS["score_trajectory"],
        detection_source_path=output_dir / _ARTIFACT_RELATIVE_PATHS["detection"],
        isolation_source_path=output_dir / _ARTIFACT_RELATIVE_PATHS["isolation"],
        artifact_index_path=artifact_index_path,
        receipt_path=Path(receipt_path),
    )


def _postfilter_branch_ids(library: Mapping[str, Any]) -> frozenset[str]:
    """从简表或 P7 branch dict 列表恢复冻结 branch id 集合。"""

    raw_branches = _strict_sequence(
        library["branches"],
        name="postfilter branches",
    )
    branch_ids: list[str] = []
    for branch in raw_branches:
        if isinstance(branch, str):
            branch_id = branch
        elif isinstance(branch, Mapping):
            branch_id = _strict_string(
                branch.get("branch_id"),
                name="postfilter branch_id",
            )
        else:
            raise FrozenProtocolIntegrityError(
                "Post-filter branches must be ids or mappings with branch_id."
            )
        branch_ids.append(_require_identifier(branch_id, name="postfilter branch_id"))
    if not branch_ids or len(set(branch_ids)) != len(branch_ids):
        raise FrozenProtocolIntegrityError(
            "Post-filter branch library must be nonempty and unique."
        )
    return frozenset(branch_ids)


def _score_trajectory_rows(
    outputs: Sequence[FrozenPointwiseOutput],
) -> list[dict[str, Any]]:
    """把固定顶层字段转换为主图可直接消费的逐行长表。"""

    return [
        {
            "episode_id": output.episode_id,
            "fault_id": output.fault_id,
            "raw_index": output.raw_index,
            "true_label": output.true_label,
            "detection_score": output.detection_score,
            "detection_threshold": (
                "infinity"
                if math.isinf(output.detection_threshold)
                else output.detection_threshold
            ),
            "alarm": output.alarm,
            "branch_id": output.branch_id,
            "mode": output.mode,
            "normal_family_id": output.normal_family_id,
            "isolation_outcome": output.isolation_outcome,
            "candidate_ids": json.dumps(list(output.candidate_ids), separators=(",", ":")),
            "reported_family": output.reported_family,
            "isolation_certified": output.isolation_certified,
            "suppression_reason": output.suppression_reason,
        }
        for output in outputs
    ]


def _detection_summary_rows(
    episodes: Sequence[FrozenFaultEpisode],
    evaluations: Sequence[FrozenEpisodeEvaluation],
) -> list[dict[str, Any]]:
    """从逐行 alarm 独立派生 episode 检测表，不接受手工指标输入。"""

    rows: list[dict[str, Any]] = []
    for episode, evaluation in zip(episodes, evaluations, strict=True):
        onset = episode.manifest.onset
        pre_outputs = evaluation.outputs[:onset]
        post_outputs = evaluation.outputs[onset:]
        pre_alarm_count = sum(output.alarm for output in pre_outputs)
        post_alarm_count = sum(output.alarm for output in post_outputs)
        first_alarm_position = next(
            (
                position
                for position, output in enumerate(evaluation.outputs)
                if position >= onset and output.alarm
            ),
            None,
        )
        first_alarm_raw_index = (
            None
            if first_alarm_position is None
            else evaluation.outputs[first_alarm_position].raw_index
        )
        rows.append(
            {
                "episode_id": episode.manifest.episode_id,
                "fault_id": episode.manifest.fault_id,
                "fault_family": episode.manifest.fault_family,
                "onset": onset,
                "row_count": episode.manifest.row_count,
                "pre_fault_row_count": len(pre_outputs),
                "post_onset_row_count": len(post_outputs),
                "pre_fault_alarm_count": pre_alarm_count,
                "post_onset_alarm_count": post_alarm_count,
                "pre_fault_alarm_rate": (
                    pre_alarm_count / len(pre_outputs) if pre_outputs else 0.0
                ),
                "post_onset_alarm_rate": post_alarm_count / len(post_outputs),
                "first_post_onset_alarm_raw_index": first_alarm_raw_index,
                "detection_delay_steps": (
                    None
                    if first_alarm_position is None
                    else first_alarm_position - onset
                ),
            }
        )
    return rows


def _isolation_summary_rows(
    episodes: Sequence[FrozenFaultEpisode],
    evaluations: Sequence[FrozenEpisodeEvaluation],
) -> list[dict[str, Any]]:
    """从逐行集合输出派生 episode 隔离来源；保留拒识而不强制单标签。"""

    rows: list[dict[str, Any]] = []
    for episode, evaluation in zip(episodes, evaluations, strict=True):
        outcome_counts = {
            outcome: sum(
                output.isolation_outcome == outcome for output in evaluation.outputs
            )
            for outcome in sorted(_ISOLATION_OUTCOMES)
        }
        reported_families = sorted(
            {
                output.reported_family
                for output in evaluation.outputs
                if output.reported_family is not None
            }
        )
        candidate_ids = sorted(
            {
                candidate_id
                for output in evaluation.outputs
                for candidate_id in output.candidate_ids
            }
        )
        rows.append(
            {
                "episode_id": episode.manifest.episode_id,
                "fault_id": episode.manifest.fault_id,
                "fault_family": episode.manifest.fault_family,
                **outcome_counts,
                "certified_row_count": sum(
                    output.isolation_certified for output in evaluation.outputs
                ),
                "reported_families": json.dumps(
                    reported_families,
                    separators=(",", ":"),
                ),
                "candidate_ids": json.dumps(candidate_ids, separators=(",", ":")),
            }
        )
    return rows


def _write_pointwise_jsonl(
    path: Path,
    outputs: Sequence[FrozenPointwiseOutput],
) -> None:
    """以 exclusive-create 写完整逐时刻 JSONL；失败时不覆盖旧评价。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for output in outputs:
            stream.write(
                json.dumps(
                    output.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
            stream.write("\n")


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
) -> None:
    """以固定列顺序写机器来源 CSV，拒绝覆盖。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    """以 exclusive-create 写标准 JSON，作为 manifest/claim/receipt 的共同语义。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        _json_copy(value),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.write("\n")


def _read_json_mapping(path: Path, *, name: str) -> Mapping[str, Any]:
    """读取一个 JSON object；解析失败或顶层非 object 时 fail closed。"""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenEvaluationArtifactError(f"Cannot read {name} at {path}.") from exc
    if not isinstance(value, Mapping):
        raise FrozenEvaluationArtifactError(f"{name} must contain a JSON object.")
    return value


def _resolve_inside(root: Path, relative: Path) -> Path:
    """解析产物相对路径并阻止绝对路径、磁盘前缀和 ``..`` 逃逸。"""

    if relative.is_absolute() or relative.drive:
        raise FrozenEvaluationArtifactError("Artifact paths must be relative.")
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise FrozenEvaluationArtifactError(
            "Artifact path escapes the frozen evaluation directory."
        ) from exc
    return target


def _validate_split_manifest(value: Mapping[str, Any]) -> None:
    """完整重放 P2 五阶段 manifest，并核对其确定性 ``split_hash``。

    参数：
        value: 已从 P10 manifest 深层复制的 P2 JSON 快照。
    返回：
        无；校验成功表示配置派生量、五个非空阶段及 split hash 相互一致。
    异常：
        缺失/未知字段、空阶段、计数或区间矛盾、非法 SHA-256，以及重算
        ``split_hash`` 不一致时抛出 ``FrozenProtocolIntegrityError``。
    副作用：
        无；不会重新读取正常数组，也不会改写传入映射。
    """

    expected_top_keys = {
        "protocol",
        "config",
        "effective_gap",
        "prepared_gap_ranges",
        "data_hash",
        "source_hash",
        "split_hash",
        "stages",
    }
    _require_exact_keys(value, expected_top_keys, name="split manifest")
    if _strict_string(value["protocol"], name="split protocol") != "five_stage_normal_only":
        raise FrozenProtocolIntegrityError(
            "Frozen split manifest must use protocol='five_stage_normal_only'."
        )
    config = _validate_split_config(
        _strict_mapping(value["config"], name="split config")
    )
    effective_gap = _strict_int(value["effective_gap"], name="split effective_gap")
    if effective_gap != config.effective_gap:
        raise FrozenProtocolIntegrityError(
            "Frozen split effective_gap does not match its P2 configuration."
        )
    prepared_gap_ranges = _strict_range_sequence(
        value["prepared_gap_ranges"],
        name="split prepared_gap_ranges",
    )
    if len(prepared_gap_ranges) != len(StageName) - 1 or any(
        stop - start != effective_gap for start, stop in prepared_gap_ranges
    ):
        raise FrozenProtocolIntegrityError(
            "Frozen split must contain four prepared gaps of exactly effective_gap rows."
        )
    _require_sha256(
        _strict_string(value["data_hash"], name="split data_hash"),
        name="split data_hash",
    )
    _require_sha256(
        _strict_string(value["source_hash"], name="split source_hash"),
        name="split source_hash",
    )
    split_hash = _strict_string(value["split_hash"], name="split_hash")
    _require_sha256(split_hash, name="split_hash")

    stages = _strict_mapping(value["stages"], name="split stages")
    _require_exact_keys(stages, set(_REQUIRED_SPLIT_STAGES), name="split stages")
    for stage in StageName:
        _validate_split_stage(
            _strict_mapping(
                stages[stage.value],
                name=f"split stage {stage.value}",
            ),
            stage=stage,
            dependency_span=config.dependency_span,
        )

    expected_hash = _sha256_json(
        {
            "protocol": value["protocol"],
            "config": value["config"],
            "prepared_gap_ranges": value["prepared_gap_ranges"],
            "data_hash": value["data_hash"],
            "source_hash": value["source_hash"],
            "stages": value["stages"],
        }
    )
    if split_hash != expected_hash:
        raise FrozenProtocolIntegrityError(
            "Frozen split_hash does not match the complete P2 split content."
        )


def _validate_config_provenance(
    resolved_config: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    """核对每个 resolved config 叶字段的来源路径、结构和最终值。

    参数：
        resolved_config: 已规范化为 JSON 的完整入口 + normal method 配置。
        provenance: 路径到按优先级排列的 ``source/value`` 记录。
    返回：
        无。
    异常：
        路径缺失/多余、记录为空/字段非法，或最后记录值不等于 resolved 叶值时抛出
        ``FrozenProtocolIntegrityError``。
    副作用：
        无。
    """

    expected_leaves = _flatten_config_leaves(resolved_config)
    if set(provenance) != set(expected_leaves):
        missing = sorted(set(expected_leaves).difference(provenance))
        extra = sorted(set(provenance).difference(expected_leaves))
        raise FrozenProtocolIntegrityError(
            "Frozen config provenance leaf coverage differs from resolved_config. "
            f"missing={missing}, extra={extra}."
        )
    for path, expected_value in expected_leaves.items():
        records = _strict_sequence(
            provenance[path],
            name=f"config provenance {path}",
        )
        if not records:
            raise FrozenProtocolIntegrityError(
                f"Frozen config provenance {path!r} must contain at least one record."
            )
        for index, raw_record in enumerate(records):
            record = _strict_mapping(
                raw_record,
                name=f"config provenance {path} record {index}",
            )
            _require_exact_keys(
                record,
                {"source", "value"},
                name=f"config provenance {path} record {index}",
            )
            _strict_string(
                record["source"],
                name=f"config provenance {path} source",
            )
        final_record = _strict_mapping(
            records[-1],
            name=f"config provenance {path} final record",
        )
        if _json_copy(final_record["value"]) != expected_value:
            raise FrozenProtocolIntegrityError(
                f"Frozen config provenance value for {path!r} differs from "
                "resolved_config."
            )


def _flatten_config_leaves(
    value: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    """把嵌套 mapping 展开为 dot-path；列表/空 mapping 作为一个配置叶值。"""

    leaves: dict[str, Any] = {}
    for raw_key, item in value.items():
        key = _strict_string(raw_key, name="resolved config key")
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, Mapping) and item:
            leaves.update(_flatten_config_leaves(item, prefix=path))
        else:
            leaves[path] = _json_copy(item)
    return leaves


def _validate_split_config(value: Mapping[str, Any]) -> FiveStageSplitConfig:
    """严格解析 P2 切分配置，并核对所有由基础字段计算得到的审计量。"""

    expected_keys = {
        "ratios",
        "history_length",
        "max_rollout",
        "stacked_window",
        "mask_recompute_span",
        "dependency_span",
        "minimum_gap",
        "effective_gap",
        "window_stride",
        "episode_length",
        "target_risk_level",
        "minimum_calibration_episodes",
        "seed",
        "strategy",
    }
    _require_exact_keys(value, expected_keys, name="split config")
    ratios = _strict_mapping(value["ratios"], name="split ratios")
    _require_exact_keys(ratios, set(_REQUIRED_SPLIT_STAGES), name="split ratios")
    config = FiveStageSplitConfig(
        ratios=tuple(
            _strict_float(ratios[stage.value], name=f"split ratio {stage.value}")
            for stage in StageName
        ),  # type: ignore[arg-type]
        history_length=_strict_int(
            value["history_length"],
            name="split history_length",
        ),
        max_rollout=_strict_int(value["max_rollout"], name="split max_rollout"),
        stacked_window=_strict_int(
            value["stacked_window"],
            name="split stacked_window",
        ),
        mask_recompute_span=_strict_int(
            value["mask_recompute_span"],
            name="split mask_recompute_span",
        ),
        minimum_gap=_strict_int(value["minimum_gap"], name="split minimum_gap"),
        window_stride=_strict_int(
            value["window_stride"],
            name="split window_stride",
        ),
        episode_length=_strict_int(
            value["episode_length"],
            name="split episode_length",
        ),
        target_risk_level=_strict_float(
            value["target_risk_level"],
            name="split target_risk_level",
        ),
        seed=_strict_int(value["seed"], name="split seed"),
        strategy=_strict_string(value["strategy"], name="split strategy"),
    )
    derived = {
        "dependency_span": config.dependency_span,
        "effective_gap": config.effective_gap,
        "minimum_calibration_episodes": config.minimum_calibration_episodes,
    }
    for field_name, expected in derived.items():
        observed = _strict_int(value[field_name], name=f"split {field_name}")
        if observed != expected:
            raise FrozenProtocolIntegrityError(
                f"Frozen split {field_name} does not match its base configuration."
            )
    return config


def _validate_split_stage(
    value: Mapping[str, Any],
    *,
    stage: StageName,
    dependency_span: int,
) -> None:
    """验证一个非空 P2 阶段的索引、窗口、episode 和三个内容摘要。"""

    expected_keys = {
        "stage",
        "prepared_row_range",
        "row_count",
        "prepared_row_indices",
        "raw_indices",
        "window_count",
        "prepared_window_starts",
        "raw_window_starts",
        "dependency_span",
        "prepared_episode_ranges",
        "discarded_prepared_episode_ranges",
        "unused_prepared_episode_tail",
        "data_hash",
        "index_hash",
        "window_hash",
    }
    _require_exact_keys(value, expected_keys, name=f"split stage {stage.value}")
    if _strict_string(value["stage"], name=f"split stage {stage.value} name") != stage.value:
        raise FrozenProtocolIntegrityError(
            f"Frozen split stage {stage.value!r} carries a mismatched stage name."
        )
    prepared_indices = _strict_int_sequence(
        value["prepared_row_indices"],
        name=f"split stage {stage.value} prepared_row_indices",
    )
    raw_indices = _strict_int_sequence(
        value["raw_indices"],
        name=f"split stage {stage.value} raw_indices",
    )
    row_count = _strict_int(value["row_count"], name=f"split stage {stage.value} row_count")
    if (
        row_count <= 0
        or len(prepared_indices) != row_count
        or len(raw_indices) != row_count
    ):
        raise FrozenProtocolIntegrityError(
            f"Frozen split stage {stage.value!r} must contain aligned, nonempty rows."
        )
    if any(
        current != previous + 1
        for previous, current in zip(prepared_indices, prepared_indices[1:])
    ):
        raise FrozenProtocolIntegrityError(
            f"Frozen split stage {stage.value!r} prepared rows must be contiguous."
        )
    prepared_range = _strict_single_range(
        value["prepared_row_range"],
        name=f"split stage {stage.value} prepared_row_range",
    )
    if prepared_range != (prepared_indices[0], prepared_indices[-1] + 1):
        raise FrozenProtocolIntegrityError(
            f"Frozen split stage {stage.value!r} row range disagrees with its indices."
        )

    prepared_windows = _strict_int_sequence(
        value["prepared_window_starts"],
        name=f"split stage {stage.value} prepared_window_starts",
    )
    raw_windows = _strict_int_sequence(
        value["raw_window_starts"],
        name=f"split stage {stage.value} raw_window_starts",
    )
    window_count = _strict_int(
        value["window_count"],
        name=f"split stage {stage.value} window_count",
    )
    if (
        window_count <= 0
        or len(prepared_windows) != window_count
        or len(raw_windows) != window_count
    ):
        raise FrozenProtocolIntegrityError(
            f"Frozen split stage {stage.value!r} must contain aligned, nonempty windows."
        )
    observed_span = _strict_int(
        value["dependency_span"],
        name=f"split stage {stage.value} dependency_span",
    )
    if observed_span != dependency_span:
        raise FrozenProtocolIntegrityError(
            f"Frozen split stage {stage.value!r} uses a mismatched dependency span."
        )
    _strict_range_sequence(
        value["prepared_episode_ranges"],
        name=f"split stage {stage.value} prepared_episode_ranges",
    )
    _strict_range_sequence(
        value["discarded_prepared_episode_ranges"],
        name=f"split stage {stage.value} discarded_prepared_episode_ranges",
    )
    unused_tail = value["unused_prepared_episode_tail"]
    if unused_tail is not None:
        _strict_single_range(
            unused_tail,
            name=f"split stage {stage.value} unused_prepared_episode_tail",
        )
    for hash_name in ("data_hash", "index_hash", "window_hash"):
        _require_sha256(
            _strict_string(
                value[hash_name],
                name=f"split stage {stage.value} {hash_name}",
            ),
            name=f"split stage {stage.value} {hash_name}",
        )


def _strict_int_sequence(value: Any, *, name: str) -> tuple[int, ...]:
    """把 JSON sequence 严格读取为整数元组，拒绝布尔值伪装成索引。"""

    return tuple(
        _strict_int(item, name=f"{name} item")
        for item in _strict_sequence(value, name=name)
    )


def _strict_single_range(value: Any, *, name: str) -> tuple[int, int]:
    """读取一个非空半开整数区间 ``[start, stop)``。"""

    items = _strict_int_sequence(value, name=name)
    if len(items) != 2 or items[0] >= items[1]:
        raise FrozenProtocolIntegrityError(
            f"{name} must be a two-item increasing half-open range."
        )
    return items[0], items[1]


def _strict_range_sequence(
    value: Any,
    *,
    name: str,
) -> tuple[tuple[int, int], ...]:
    """读取半开整数区间序列；空序列仅用于没有被丢弃 episode 的合法场景。"""

    return tuple(
        _strict_single_range(item, name=f"{name} item")
        for item in _strict_sequence(value, name=name)
    )


def _validate_fault_episode_library(
    episodes: tuple[FrozenFaultEpisodeManifest, ...],
) -> None:
    """要求一次冻结评价完整覆盖 CSTR fault 1--8，且身份唯一。"""

    if len(episodes) != 8:
        raise ValueError("Frozen CSTR evaluation requires exactly eight fault episodes.")
    fault_ids = [episode.fault_id for episode in episodes]
    if set(fault_ids) != set(_CSTR_FAULT_FAMILIES):
        raise ValueError("Frozen CSTR episode manifest must cover fault ids 1..8 exactly once.")
    if len(fault_ids) != len(set(fault_ids)):
        raise ValueError("Frozen CSTR fault ids must be unique.")
    episode_ids = [episode.episode_id for episode in episodes]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("Frozen CSTR episode ids must be unique.")


def _validate_calibration_episode_binding(
    calibration: FrozenRiskCalibration,
    *,
    stage: Mapping[str, Any],
) -> None:
    """从 P2 完整 episode ranges 重算校准 episode 数量和稳定 ID。

    参数：
        calibration: detection 或 attribution 的冻结有限样本摘要。
        stage: 对应 P2 stage 的已校验 manifest。
    返回：
        无。
    异常：
        episode_count 或 episode_ids 不能由 ``prepared_episode_ranges`` 唯一重放时抛出
        ``FrozenProtocolIntegrityError``。
    副作用：
        无。
    """

    ranges = _strict_range_sequence(
        stage["prepared_episode_ranges"],
        name=f"{calibration.name} calibration prepared_episode_ranges",
    )
    expected_ids = tuple(
        f"{calibration.name}-{start}-{end}"
        for start, end in ranges
    )
    if calibration.episode_count != len(ranges):
        raise FrozenProtocolIntegrityError(
            f"Frozen {calibration.name} calibration episode_count differs from "
            "its P2 prepared episode ranges."
        )
    if calibration.episode_ids != expected_ids:
        raise FrozenProtocolIntegrityError(
            f"Frozen {calibration.name} calibration episode_ids differ from "
            "its P2 prepared episode ranges."
        )


def _validate_normal_artifact_bundle(
    bundle: FrozenNormalArtifactBundle,
    *,
    checkpoint_hashes: Mapping[str, str],
) -> None:
    """要求 formal bundle 完整覆盖训练、P5--P9、replay 和 evaluator 身份。

    参数：
        bundle: 已逐文件重放的正常产物集合。
        checkpoint_hashes: 主 manifest 的 checkpoint 内容身份。
    返回：
        无。
    异常：
        缺少固定方法产物、checkpoint 文件/hash 或 evaluator checkpoint 身份不一致时
        抛出 ``FrozenProtocolIntegrityError``。
    副作用：
        无；bundle 已在自身构造时完成文件读取。
    """

    missing = sorted(
        _REQUIRED_NORMAL_ARTIFACTS.difference(bundle.artifact_paths)
    )
    if missing:
        raise FrozenProtocolIntegrityError(
            "Frozen normal artifact bundle is missing required files: "
            + ", ".join(missing)
            + "."
        )
    checkpoint_name = bundle.runtime_evaluator["checkpoint_name"]
    if checkpoint_name not in checkpoint_hashes:
        raise FrozenProtocolIntegrityError(
            "Frozen runtime evaluator references an unknown checkpoint."
        )
    checkpoint_artifact_name = f"checkpoint_files.{checkpoint_name}"
    if checkpoint_artifact_name not in bundle.artifact_hashes:
        raise FrozenProtocolIntegrityError(
            "Frozen normal artifact bundle is missing the runtime evaluator checkpoint."
        )
    if (
        bundle.artifact_hashes[checkpoint_artifact_name]
        != checkpoint_hashes[checkpoint_name]
    ):
        raise FrozenProtocolIntegrityError(
            "Frozen runtime evaluator checkpoint hash differs from the main manifest."
        )
    validate_training_runtime_checkpoint_continuity(
        bundle.artifact_paths["training_checkpoint"],
        bundle.artifact_paths[checkpoint_artifact_name],
    )
    if not bundle.replay_outputs:
        raise FrozenProtocolIntegrityError(
            "Frozen normal artifact bundle requires checkpoint replay output files."
        )


def _validate_formal_certification_artifact_bindings(
    bundle: FrozenNormalArtifactBundle,
    *,
    certification_status: Mapping[str, Any],
) -> None:
    """把 formal 认证声明绑定到实际 operator/signature/nuisance 文件。

    参数：
        bundle: 已逐文件复验的正常产物集合。
        certification_status: 三类认证状态及其声明的 ``artifact_hash``。
    返回：
        无。
    异常：
        任一状态不是 certified、hash 缺失/错配，或 operator/isolation 文件内部未声明
        已认证时抛出 ``FrozenProtocolIntegrityError``。
    副作用：
        只读 operator bundle 与 isolation library 两个已受 bundle hash 保护的 JSON 文件。
    """

    artifact_bindings = {
        "operator": "operator_bundle",
        "signature": "isolation_library",
        "nuisance": "deterministic_envelope",
    }
    for certification_name, artifact_name in artifact_bindings.items():
        entry = _strict_mapping(
            certification_status[certification_name],
            name=f"{certification_name} certification",
        )
        if entry["status"] != "certified":
            raise FrozenProtocolIntegrityError(
                "Formal frozen evaluation requires operator, signature and nuisance "
                "certification statuses to be 'certified'."
            )
        artifact_hash = _strict_string(
            entry.get("artifact_hash"),
            name=f"{certification_name} certification artifact_hash",
        )
        if artifact_hash != bundle.artifact_hashes[artifact_name]:
            raise FrozenProtocolIntegrityError(
                f"Frozen {certification_name} certification artifact_hash differs "
                f"from {artifact_name}."
            )

    operator_bundle = _read_bound_json_artifact(bundle, "operator_bundle")
    if operator_bundle.get("status") != "certified":
        raise FrozenProtocolIntegrityError(
            "Formal operator bundle must declare status='certified'."
        )
    isolation_library = _read_bound_json_artifact(bundle, "isolation_library")
    if (
        _strict_bool(
            isolation_library.get("certified"),
            name="isolation library certified",
        )
        is not True
    ):
        raise FrozenProtocolIntegrityError(
            "Formal isolation library must declare certified=true."
        )
    nuisance_envelope = _read_bound_json_artifact(
        bundle,
        "deterministic_envelope",
    )
    try:
        _FrozenCertifiedNuisanceEnvelopeConfig.model_validate(
            _json_copy(nuisance_envelope)
        )
    except ValidationError as exc:
        raise FrozenProtocolIntegrityError(
            "Formal nuisance envelope must use the strict certified schema."
        ) from exc


def _read_bound_json_artifact(
    bundle: FrozenNormalArtifactBundle,
    logical_name: str,
) -> Mapping[str, Any]:
    """读取已由 bundle hash 保护的 JSON object，用于跨产物字段配对。"""

    path = Path(bundle.artifact_paths[logical_name])
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FrozenProtocolIntegrityError(
            f"Frozen normal artifact {logical_name!r} is not readable JSON."
        ) from exc
    return _strict_mapping(value, name=f"normal artifact {logical_name}")


def _validate_fit_access_ledger(
    value: Mapping[str, Any],
    *,
    split_manifest: Mapping[str, Any],
    normal_artifacts: FrozenNormalArtifactBundle | None,
) -> None:
    """独立重放 P2 拟合账本，不能只相信持久化的 ``protocol_ready`` 标志。

    参数：
        value: P2 ``FitAccessLedger.manifest()`` 的 JSON 快照。
        split_manifest: 已通过完整校验的五阶段 manifest，用于绑定 split 与阶段内容 hash。
        normal_artifacts: formal 模式的文件级正常产物；存在时每条 artifact_hash 必须绑定
            到实际文件。
    返回：
        无；成功表示模型、估计、两次校准和冻结正常诊断记录均存在且全部冻结。
    异常：
        策略、用途、阶段、hash、冻结状态或最小用途集合不一致时抛出
        ``FrozenProtocolIntegrityError``。
    副作用：
        无；不重开正常数组，也不修改账本文件。
    """

    expected_keys = {
        "split_hash",
        "frozen",
        "protocol_ready",
        "policy",
        "records",
    }
    _require_exact_keys(value, expected_keys, name="fit access ledger")
    split_hash = _strict_string(value["split_hash"], name="fit access split_hash")
    _require_sha256(split_hash, name="fit access split_hash")
    if split_hash != split_manifest["split_hash"]:
        raise FrozenProtocolIntegrityError(
            "Frozen fit access ledger split_hash differs from the P2 split manifest."
        )
    if (
        _strict_bool(value["frozen"], name="fit access frozen") is not True
        or _strict_bool(
            value["protocol_ready"],
            name="fit access protocol_ready",
        )
        is not True
    ):
        raise FrozenProtocolIntegrityError(
            "Frozen evaluation requires a frozen and protocol-ready fit access ledger."
        )

    expected_policy = fit_stage_policy_manifest()
    policy = _strict_mapping(value["policy"], name="fit access policy")
    _require_exact_keys(policy, set(expected_policy), name="fit access policy")
    for purpose, expected_stages in expected_policy.items():
        observed_stages = tuple(
            _strict_string(stage, name=f"fit access policy {purpose} stage")
            for stage in _strict_sequence(
                policy[purpose],
                name=f"fit access policy {purpose}",
            )
        )
        if observed_stages != expected_stages:
            raise FrozenProtocolIntegrityError(
                f"Frozen fit access policy for {purpose!r} differs from P2."
            )

    records = _strict_sequence(value["records"], name="fit access records")
    if not records:
        raise FrozenProtocolIntegrityError(
            "Frozen fit access ledger must contain at least one record."
        )
    observed_ids: set[str] = set()
    observed_purposes: set[str] = set()
    ledger_bindings = (
        {} if normal_artifacts is None else dict(normal_artifacts.ledger_bindings)
    )
    split_stages = _strict_mapping(split_manifest["stages"], name="split stages")
    record_keys = {
        "object_id",
        "purpose",
        "stages",
        "stage_hashes",
        "frozen",
        "artifact_hash",
    }
    for index, raw_record in enumerate(records):
        record = _strict_mapping(raw_record, name=f"fit access record {index}")
        _require_exact_keys(record, record_keys, name=f"fit access record {index}")
        object_id = _require_identifier(
            _strict_string(record["object_id"], name="fit access object_id"),
            name="fit access object_id",
        )
        if object_id in observed_ids:
            raise FrozenProtocolIntegrityError(
                f"Frozen fit access object_id {object_id!r} is duplicated."
            )
        observed_ids.add(object_id)
        purpose = _strict_string(record["purpose"], name="fit access purpose")
        try:
            FitPurpose(purpose)
        except ValueError as exc:
            raise FrozenProtocolIntegrityError(
                f"Frozen fit access purpose {purpose!r} is unknown."
            ) from exc
        expected_stages = expected_policy[purpose]
        stages = tuple(
            _strict_string(stage, name=f"fit access record {object_id} stage")
            for stage in _strict_sequence(
                record["stages"],
                name=f"fit access record {object_id} stages",
            )
        )
        if stages != expected_stages:
            raise FrozenProtocolIntegrityError(
                f"Frozen fit access record {object_id!r} violates the P2 stage policy."
            )
        stage_hashes = _strict_mapping(
            record["stage_hashes"],
            name=f"fit access record {object_id} stage_hashes",
        )
        _require_exact_keys(
            stage_hashes,
            set(stages),
            name=f"fit access record {object_id} stage_hashes",
        )
        for stage in stages:
            observed_hash = _strict_string(
                stage_hashes[stage],
                name=f"fit access record {object_id} stage hash",
            )
            _require_sha256(
                observed_hash,
                name=f"fit access record {object_id} stage hash",
            )
            expected_hash = _strict_mapping(
                split_stages[stage],
                name=f"split stage {stage}",
            )["data_hash"]
            if observed_hash != expected_hash:
                raise FrozenProtocolIntegrityError(
                    f"Frozen fit access stage_hashes for {object_id!r} "
                    "do not match the corresponding split stage."
                )
        if _strict_bool(
            record["frozen"],
            name=f"fit access record {object_id} frozen",
        ) is not True:
            raise FrozenProtocolIntegrityError(
                f"Frozen fit access record {object_id!r} is not frozen."
            )
        artifact_hash = _require_sha256(
            _strict_string(
                record["artifact_hash"],
                name=f"fit access record {object_id} artifact_hash",
            ),
            name=f"fit access record {object_id} artifact_hash",
        )
        if normal_artifacts is not None:
            target_name = ledger_bindings.get(object_id)
            if target_name is None:
                raise FrozenProtocolIntegrityError(
                    f"Frozen fit access record {object_id!r} has no normal artifact binding."
                )
            expected_target = _FIT_PURPOSE_ARTIFACT_NAMES.get(purpose)
            if (
                expected_target is not None
                and target_name != expected_target
            ):
                raise FrozenProtocolIntegrityError(
                    f"Frozen fit purpose {purpose!r} must bind to "
                    f"{expected_target!r}, not {target_name!r}."
                )
            if (
                purpose == FitPurpose.FROZEN_NORMAL_DIAGNOSTIC.value
                and not target_name.startswith("checkpoint_replay_outputs.")
            ):
                raise FrozenProtocolIntegrityError(
                    "Frozen fit purpose 'frozen_normal_diagnostic' must bind to a "
                    "'checkpoint_replay_outputs.<name>' artifact."
                )
            expected_artifact_hash = normal_artifacts.artifact_hashes[target_name]
            if artifact_hash != expected_artifact_hash:
                raise FrozenProtocolIntegrityError(
                    f"Frozen fit access record {object_id!r} artifact_hash differs from "
                    f"the actual {target_name!r} file."
                )
        observed_purposes.add(purpose)

    if normal_artifacts is not None and set(ledger_bindings) != observed_ids:
        raise FrozenProtocolIntegrityError(
            "Frozen ledger_bindings must cover exactly every fit access object_id."
        )

    required_purposes = {
        FitPurpose.MODEL_PARAMETERS.value,
        FitPurpose.DETECTION_QUANTILE.value,
        FitPurpose.ATTRIBUTION_QUANTILE.value,
        FitPurpose.FROZEN_NORMAL_DIAGNOSTIC.value,
    }
    if normal_artifacts is not None:
        required_purposes.update(
            {
                FitPurpose.MONITORING_SCORE_SCALER.value,
                FitPurpose.ENVELOPE.value,
                FitPurpose.COVARIANCE.value,
                FitPurpose.BRANCH_LIBRARY.value,
                FitPurpose.STATE_MACHINE.value,
            }
        )
    missing_purposes = sorted(required_purposes.difference(observed_purposes))
    if missing_purposes:
        raise FrozenProtocolIntegrityError(
            "Frozen fit access ledger is missing required purposes: "
            + ", ".join(missing_purposes)
            + "."
        )
    estimate_purposes = {
        purpose
        for purpose, stages in expected_policy.items()
        if stages == (StageName.ESTIMATE.value,)
    }
    if not observed_purposes.intersection(estimate_purposes):
        raise FrozenProtocolIntegrityError(
            "Frozen fit access ledger requires at least one estimate-stage purpose."
        )


def _validate_checkpoint_replay(
    value: Mapping[str, Any],
    *,
    checkpoint_hashes: Mapping[str, str],
    run_mode: str,
    normal_artifacts: FrozenNormalArtifactBundle | None,
) -> None:
    """绑定 checkpoint 重放证据；formal frozen 模式只接受真正 passed 状态。"""

    _require_mapping_keys(
        value,
        {"status", "checkpoint_hashes", "output_hashes"},
        name="checkpoint_replay",
    )
    status = _strict_string(value["status"], name="checkpoint replay status")
    allowed_statuses = {"passed"} if run_mode == "frozen" else {
        "passed",
        "synthetic_contract",
    }
    if status not in allowed_statuses:
        raise ValueError(
            f"Checkpoint replay status {status!r} is not allowed for mode {run_mode!r}."
        )
    replay_checkpoint_hashes = _strict_mapping(
        value["checkpoint_hashes"],
        name="checkpoint replay hashes",
    )
    if dict(replay_checkpoint_hashes) != dict(checkpoint_hashes):
        raise ValueError(
            "Checkpoint replay hashes must exactly match the frozen checkpoint hashes."
        )
    output_hashes = _strict_mapping(
        value["output_hashes"],
        name="checkpoint replay output_hashes",
    )
    if not output_hashes:
        raise ValueError("Checkpoint replay must retain at least one output hash.")
    for output_name, output_hash in output_hashes.items():
        _require_sha256(
            _strict_string(output_hash, name=f"replay output hash {output_name}"),
            name=f"replay output hash {output_name}",
        )
    if normal_artifacts is not None:
        replay_bindings = normal_artifacts.replay_outputs
        if set(output_hashes) != set(replay_bindings):
            raise FrozenProtocolIntegrityError(
                "Checkpoint replay output_hashes must exactly match file-bound outputs."
            )
        for output_name, artifact_name in replay_bindings.items():
            if output_hashes[output_name] != normal_artifacts.artifact_hashes[
                artifact_name
            ]:
                raise FrozenProtocolIntegrityError(
                    f"Checkpoint replay output {output_name!r} hash differs from its "
                    "actual output file."
                )


def _deep_freeze_json(value: Any, *, name: str) -> Any:
    """复制并递归冻结 JSON 值；拒绝非字符串键和非有限浮点数。"""

    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{name} mapping keys must be nonempty strings.")
            frozen[key] = _deep_freeze_json(item, name=f"{name}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return tuple(
            _deep_freeze_json(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        )
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must not contain NaN or infinity.")
        return value
    raise TypeError(
        f"{name} contains non-JSON value of type {type(value).__name__!r}."
    )


def _json_copy(value: Any) -> Any:
    """把只读映射/元组还原为全新的 JSON 兼容结构。"""

    if isinstance(value, Mapping):
        return {str(key): _json_copy(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_json_copy(item) for item in value]
    return value


def _frozen_string_mapping(
    value: Mapping[str, str],
    *,
    name: str,
) -> Mapping[str, str]:
    """校验非空字符串键值并返回只读副本。"""

    result: dict[str, str] = {}
    for key, item in value.items():
        normalized_key = _strict_string(key, name=f"{name} key").strip()
        normalized_value = _strict_string(item, name=f"{name}.{normalized_key}").strip()
        if not normalized_key or not normalized_value:
            raise ValueError(f"{name} keys and values cannot be empty.")
        result[normalized_key] = normalized_value
    return MappingProxyType(result)


def _frozen_hash_mapping(
    value: Mapping[str, str],
    *,
    name: str,
) -> Mapping[str, str]:
    """校验逻辑名称到 SHA-256 的映射并返回只读副本。"""

    result = dict(_frozen_string_mapping(value, name=name))
    for key, digest in result.items():
        _require_sha256(digest, name=f"{name}.{key}")
    return MappingProxyType(result)


def _frozen_seed_mapping(value: Mapping[str, int]) -> Mapping[str, int]:
    """冻结所有已声明随机源；空映射会失去实验可重现性，因此拒绝。"""

    if not value:
        raise ValueError("Frozen seeds cannot be empty.")
    result: dict[str, int] = {}
    for key, item in value.items():
        normalized_key = _strict_string(key, name="seed name").strip()
        if not normalized_key:
            raise ValueError("Frozen seed names cannot be empty.")
        result[normalized_key] = _strict_int(item, name=f"seed {normalized_key}")
    return MappingProxyType(result)


def _require_mapping_keys(
    value: Mapping[str, Any],
    required: set[str],
    *,
    name: str,
) -> None:
    """要求映射至少含指定字段；P4--P9 自身快照可保留额外可审计字段。"""

    missing = sorted(required.difference(value))
    if missing:
        raise ValueError(f"{name} is missing required fields: {', '.join(missing)}.")


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    name: str,
) -> None:
    """严格拒绝缺失和未知字段，避免加载时隐式采用未来默认值。"""

    actual = set(value)
    missing = sorted(expected.difference(actual))
    extra = sorted(actual.difference(expected))
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing=" + ", ".join(missing))
        if extra:
            details.append("unknown=" + ", ".join(extra))
        raise FrozenProtocolIntegrityError(f"{name} has invalid fields: {'; '.join(details)}.")


def _strict_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    """要求值为映射；JSON 数组或标量不能冒充结构化证据。"""

    if not isinstance(value, Mapping):
        raise FrozenProtocolIntegrityError(f"{name} must be a mapping.")
    return value


def _strict_sequence(value: Any, *, name: str) -> tuple[Any, ...]:
    """要求值为非字符串序列并返回元组副本。"""

    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        raise FrozenProtocolIntegrityError(f"{name} must be a sequence.")
    return tuple(value)


def _strict_string(value: Any, *, name: str) -> str:
    """要求精确字符串类型。"""

    if not isinstance(value, str):
        raise FrozenProtocolIntegrityError(f"{name} must be a string.")
    return value


def _strict_int(value: Any, *, name: str) -> int:
    """要求精确整数，拒绝 ``bool`` 作为 0/1。"""

    if isinstance(value, bool) or not isinstance(value, int):
        raise FrozenProtocolIntegrityError(f"{name} must be an integer.")
    return value


def _strict_bool(value: Any, *, name: str) -> bool:
    """要求精确布尔值，拒绝整数或字符串冒充冻结状态。"""

    if not isinstance(value, bool):
        raise FrozenProtocolIntegrityError(f"{name} must be a boolean.")
    return value


def _strict_float(value: Any, *, name: str) -> float:
    """读取 JSON 数值为 float，拒绝布尔和非有限值。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FrozenProtocolIntegrityError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise FrozenProtocolIntegrityError(f"{name} must be finite.")
    return result


def _encode_finite_or_infinity(value: float) -> float | str:
    """用字符串编码正无穷，保持 JSON 标准兼容且显式可审计。"""

    return "infinity" if math.isinf(value) else value


def _decode_finite_or_infinity(value: Any, *, name: str) -> float:
    """读取有限数或唯一受控的正无穷字符串。"""

    if value == "infinity":
        return math.inf
    return _strict_float(value, name=name)


def _require_sha256(value: str, *, name: str) -> str:
    """校验 64 位小写 SHA-256。"""

    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a 64-character lowercase SHA-256.")
    return value


def _require_identifier(value: str, *, name: str) -> str:
    """校验可安全用于产物名和 claim registry 的稳定标识。"""

    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(
            f"{name} must be 3-128 characters using letters, digits, '.', '_' or '-'."
        )
    return value


def _sha256_json(value: Any) -> str:
    """对标准化 JSON 计算稳定 SHA-256。"""

    encoded = json.dumps(
        _json_copy(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "FrozenEpisodeEvaluation",
    "FrozenEpisodeEvaluator",
    "FrozenEpisodeInput",
    "FrozenEvaluationAlreadyClaimedError",
    "FrozenEvaluationArtifactError",
    "FrozenEvaluationClaim",
    "FrozenEvaluationResult",
    "FrozenEvaluationWorkflow",
    "FrozenFaultEpisode",
    "FrozenFaultEpisodeManifest",
    "FrozenFaultEpisodeSource",
    "FrozenNormalArtifactBundle",
    "FrozenPointwiseOutput",
    "FrozenProtocolIntegrityError",
    "FrozenProtocolManifest",
    "FrozenRiskCalibration",
    "FrozenRuntimeEpisodeEvaluation",
    "FrozenRuntimePointwiseOutput",
    "LazyFrozenCSTRFaultSource",
    "verify_frozen_evaluation_artifacts",
]
