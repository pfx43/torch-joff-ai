"""论文 P10/P11 运行入口的严格配置、解析身份与冻结就绪检查。

文件用途：
    把 synthetic CPU smoke、CSTR/TTS 正常开发和 CSTR frozen evaluation 明确区分，
    防止次级数据集开发、未核实许可或宽松字典误入正式故障入口。
主要职责：
    定义嵌套 Pydantic 严格配置、保留 resolved config/provenance/16 位 hash，并以只读检查
    报告 frozen 模式的许可、原始文件 hash 和 P2--P9 正常产物是否就绪；P11 次级开发
    额外声明保存 CSTR frozen 配置 blob 的提交、配置和正式 manifest/bundle 身份。本文件
    不训练模型、不加载 MAT 数值、不创建 manifest。
关键输入与输出：
    输入为 ``configs/paper/*.yaml``、等价映射或已经校验的配置；输出为
    ``ResolvedFrozenEvaluationConfig`` 和稳定的 readiness error 列表。
依赖与副作用：
    依赖 PyYAML、Pydantic、Joff ``StrictConfig`` 和标准库。解析 YAML 只读一个配置文件；
    readiness 在许可为 ``verified`` 时才可流式核对声明的 raw file SHA-256，并检查正常
    产物路径是否存在。模块导入和普通解析均不读数据、写文件或修改随机状态。
重要约束：
    所有未知字段都拒绝；理论敏感风险、种子、episode 长度和 onset 没有隐藏默认值。
    ``to_verify`` 必须作为阻塞事实保留，不能被布尔转换误当成授权。readiness 只报告状态，
    绝不自动修复路径、改许可、创建 claim 或回退为 smoke runtime。TTS 只允许
    development，其物理索引从数据层唯一协议对象派生；阻塞态不得执行，完成态也不得把
    次级结果回写由 frozen-config 提交、配置 blob 和正式 manifest 共同固定的 CSTR 选择。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import hashlib
import json
import re

import yaml  # type: ignore[import-untyped]
from pydantic import Field, field_validator, model_validator

from joff.core.config import StrictConfig
from joff.data.adapters import TTS_SIX_FAULT_PROTOCOL
from joff.data.paper_protocol import FiveStageSplitConfig
from joff.evaluation.protected_reference import AnchorGateConfig
from joff.models.protected_koopman_ts import ProtectedKoopmanTSConfig

from .paper_environment import sha256_file


_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SECONDARY_PHYSICAL_FEATURE_LAYOUTS = MappingProxyType(
    {
        "tts_fault_diagnosis": (
            TTS_SIX_FAULT_PROTOCOL.role_indices("control_input"),
            TTS_SIX_FAULT_PROTOCOL.role_indices("measured_output"),
            TTS_SIX_FAULT_PROTOCOL.role_indices("exogenous_input"),
        ),
    }
)


class PaperEvaluationSeedsConfig(StrictConfig):
    """P10 必须显式冻结的全部常用随机源。

    参数：
        python/numpy/torch/dataloader: 四条独立随机源的整数种子，均无隐藏默认。
    返回：
        冻结且拒绝未知字段的配置。
    异常：
        缺字段、类型错误或未知字段时抛出 Pydantic ``ValidationError``。
    副作用：
        无；这里只记录种子，不修改任何全局随机状态。
    """

    python: int
    numpy: int
    torch: int
    dataloader: int


class PaperEvaluationDatasetConfig(StrictConfig):
    """描述 smoke、闭环 CSTR 或 TTS 数据身份，但不读取数据值。

    参数：
        name/root/normal_file/fault_file: synthetic、真实闭环 CSTR 或 TTS 的路径身份。
        license_status: 许可事实；只有 ``verified`` 可进入正式 manifest。
        feature_count/normal_rows/fault_episode_count/fault_episode_rows/fault_onset: 冻结几何。
        normal_source_hash/fault_source_hash: 可选 raw SHA-256；正式 readiness 必须存在。
    返回：
        冻结且拒绝未知字段的配置；相对文件始终相对 ``root``。
    异常：
        episode 数与数据集协议不符、onset 越界、路径/许可模式冲突或 hash 非法时抛出
        Pydantic ``ValidationError``。
    副作用：
        无；构造不访问路径或 MAT 内容。
    """

    name: Literal["synthetic_cstr", "cstr_closed_loop_fd", "tts_fault_diagnosis"]
    root: Path | None
    normal_file: Path | None
    fault_file: Path | None
    license_status: Literal[
        "synthetic_only",
        "to_verify",
        "verified",
        "restricted",
        "not_permitted",
    ]
    feature_count: int = Field(gt=0)
    normal_rows: int = Field(gt=0)
    fault_episode_count: int = Field(gt=0)
    fault_episode_rows: int = Field(gt=0)
    fault_onset: int = Field(ge=0)
    normal_source_hash: str | None
    fault_source_hash: str | None

    @field_validator("normal_source_hash", "fault_source_hash")
    @classmethod
    def _validate_optional_sha256(cls, value: str | None) -> str | None:
        """校验显式 raw hash；``None`` 只允许留给 smoke/development readiness。"""

        if value is not None and not _SHA256_RE.fullmatch(value):
            raise ValueError("Raw source hashes must be 64 lowercase SHA-256 characters.")
        return value

    @model_validator(mode="after")
    def _validate_episode_geometry(self) -> "PaperEvaluationDatasetConfig":
        """固定各数据集 episode 字典，并保证 onset 位于 episode 内。"""

        expected_episode_count = 6 if self.name == "tts_fault_diagnosis" else 8
        if self.fault_episode_count != expected_episode_count:
            raise ValueError(
                f"{self.name} entry configs require exactly "
                f"{expected_episode_count} fault episodes."
            )
        if self.fault_onset >= self.fault_episode_rows:
            raise ValueError("fault_onset must identify a row inside each fault episode.")
        if self.name == "synthetic_cstr":
            if self.license_status != "synthetic_only":
                raise ValueError("Synthetic CSTR must use license_status='synthetic_only'.")
            if self.root is not None or self.normal_file is not None or self.fault_file is not None:
                raise ValueError("Synthetic CSTR must not declare real data paths.")
        else:
            if self.root is None or self.normal_file is None or self.fault_file is None:
                raise ValueError(
                    "Real paper dataset config requires root, normal_file and fault_file."
                )
        return self


class PaperPrimaryProtocolLockConfig(StrictConfig):
    """P11 次级数据集开发必须复验的 CSTR 主协议完成状态与文件身份。

    参数：
        dataset_name/protocol_version: 主数据集与 P10 frozen 协议的稳定名称。
        frozen_config_commit: 保存当前受保护 CSTR frozen 配置 blob 的完整 Git 提交身份；
            完成态还必须与 formal manifest 的执行提交一致。
        frozen_config/frozen_config_sha256: 只读 CSTR frozen YAML 及其内容 SHA-256。
        selection_status: 区分“只冻结软件配置、正式评价仍阻塞”和“正式冻结评价已完成”。
        evaluation_id/manifest_path/manifest_sha256/manifest_hash/normal_artifact_bundle_hash:
            仅完成态必填，逐层绑定正式 P10 manifest、其自哈希和 P2--P9 正常产物 bundle。
        receipt_path/receipt_sha256: 仅完成态必填，绑定 P10 运行后 completion receipt；独立
            verifier 将从该 receipt 继续复验 claim、artifact index 和全部逐时刻/表图来源。
        secondary_results_may_modify_primary/fault_results_accessed: 必须恒为假，分别禁止
            TTS/TE 回调 CSTR 选择，并证明创建本锁不曾使用故障结果。
    返回：
        冻结且拒绝未知字段的配置对象。
    异常：
        标识、提交/hash 格式、路径后缀或状态与证据组合非法时抛出 Pydantic
        ``ValidationError``。
    副作用：
        无；构造只验证文本字段，真实文件 hash 在 development runner 开始时复验。
    """

    dataset_name: Literal["cstr_closed_loop_fd"]
    protocol_version: str
    frozen_config_commit: str
    frozen_config: Path
    frozen_config_sha256: str
    selection_status: Literal[
        "configuration_frozen_fault_evaluation_blocked",
        "formal_cstr_frozen_evaluation_completed",
    ]
    evaluation_id: str | None = None
    manifest_path: Path | None = None
    manifest_sha256: str | None = None
    manifest_hash: str | None = None
    normal_artifact_bundle_hash: str | None = None
    receipt_path: Path | None = None
    receipt_sha256: str | None = None
    secondary_results_may_modify_primary: Literal[False]
    fault_results_accessed: Literal[False]

    @field_validator("protocol_version", "evaluation_id")
    @classmethod
    def _validate_protocol_identifier(cls, value: str | None) -> str | None:
        """要求可选主协议/评价名称可安全写入产物与日志。"""

        if value is None:
            return None
        normalized = value.strip()
        if not _IDENTIFIER_RE.fullmatch(normalized):
            raise ValueError("Primary protocol identifiers must use safe identifier syntax.")
        return normalized

    @field_validator("frozen_config_commit")
    @classmethod
    def _validate_commit(cls, value: str) -> str:
        """要求完整小写 Git object id，避免短 hash 歧义。"""

        if not _COMMIT_RE.fullmatch(value):
            raise ValueError(
                "Primary frozen_config_commit must be 40 lowercase hex characters."
            )
        return value

    @field_validator(
        "frozen_config_sha256",
        "manifest_sha256",
        "manifest_hash",
        "normal_artifact_bundle_hash",
        "receipt_sha256",
    )
    @classmethod
    def _validate_primary_hash(cls, value: str | None) -> str | None:
        """要求主配置与完成态证据使用完整小写 SHA-256。"""

        if value is not None and not _SHA256_RE.fullmatch(value):
            raise ValueError("Primary protocol hashes must be lowercase SHA-256 values.")
        return value

    @field_validator("frozen_config")
    @classmethod
    def _validate_frozen_config_path(cls, value: Path) -> Path:
        """限制主配置引用为 YAML 文件；存在性和仓库边界留给 runner 复验。"""

        if value.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError("Primary frozen_config must reference a YAML file.")
        return value

    @field_validator("manifest_path", "receipt_path")
    @classmethod
    def _validate_evidence_path(cls, value: Path | None) -> Path | None:
        """限制完成态 manifest/receipt 为 JSON；存在性和内容留给 runner 复验。"""

        if value is not None and value.suffix.lower() != ".json":
            raise ValueError("Primary manifest/receipt paths must reference JSON files.")
        return value

    @model_validator(mode="after")
    def _validate_completion_evidence(self) -> "PaperPrimaryProtocolLockConfig":
        """阻塞态禁止伪造完成证据，完成态则要求五项文件级身份全部存在。"""

        completion_evidence = (
            self.evaluation_id,
            self.manifest_path,
            self.manifest_sha256,
            self.manifest_hash,
            self.normal_artifact_bundle_hash,
            self.receipt_path,
            self.receipt_sha256,
        )
        completed = self.selection_status == "formal_cstr_frozen_evaluation_completed"
        if completed and any(value is None for value in completion_evidence):
            raise ValueError(
                "A completed primary protocol lock requires evaluation_id, manifest_path, "
                "manifest_sha256, manifest_hash, normal_artifact_bundle_hash, receipt_path "
                "and receipt_sha256."
            )
        if not completed and any(value is not None for value in completion_evidence):
            raise ValueError(
                "A blocked primary protocol lock must not carry formal completion evidence."
            )
        return self


class PaperNormalArtifactsConfig(StrictConfig):
    """P2--P9 正常数据冻结产物的显式路径集合。

    参数：
        resolved_config/provenance/split/ledger/training/checkpoint/replay/monitor/operator/
        structure/scaler/envelope/covariance/postfilter/calibration/isolation/certification
        字段：P2--P9 正常阶段显式产物路径。拟合语义不同的对象必须各自落盘，不能用同一
        风险校准摘要冒充分数缩放器、确定性包络或创新协方差。
        ledger_bindings: P2 object ID 到上述实际产物逻辑名的冻结映射。
    返回：
        冻结且拒绝未知字段的路径配置。
    异常：
        缺字段、checkpoint 集为空或出现未知字段时抛出 Pydantic ``ValidationError``。
    副作用：
        构造不读取、创建、补全或迁移文件；只在 readiness/builder 中使用这些路径。
    """

    resolved_config: Path
    provenance: Path
    split_manifest: Path
    fit_access_ledger: Path
    training_history: Path
    training_checkpoint: Path
    checkpoint_files: dict[str, Path] = Field(min_length=1)
    checkpoint_replay: Path
    checkpoint_replay_outputs: dict[str, Path] = Field(min_length=1)
    structure_selection: Path
    monitoring_score_scaler: Path
    deterministic_envelope: Path
    innovation_covariance: Path
    postfilter_library: Path
    monitor_policy: Path
    operator_bundle: Path
    detection_calibration: Path
    attribution_calibration: Path
    isolation_library: Path
    certification_status: Path
    ledger_bindings: dict[str, str] = Field(min_length=1)

    def paths(self) -> Mapping[str, Path]:
        """返回逻辑名称到路径的新只读映射，供 readiness 统一遍历。

        返回：
            包含固定产物和展开后 ``checkpoint_files.<name>`` 的 ``MappingProxyType``。
        异常：
            无。
        副作用：
            只分配新字典并冻结；不检查路径是否存在。
        """

        paths: dict[str, Path] = {
            "resolved_config": self.resolved_config,
            "provenance": self.provenance,
            "split_manifest": self.split_manifest,
            "fit_access_ledger": self.fit_access_ledger,
            "training_history": self.training_history,
            "training_checkpoint": self.training_checkpoint,
            "checkpoint_replay": self.checkpoint_replay,
            "structure_selection": self.structure_selection,
            "monitoring_score_scaler": self.monitoring_score_scaler,
            "deterministic_envelope": self.deterministic_envelope,
            "innovation_covariance": self.innovation_covariance,
            "postfilter_library": self.postfilter_library,
            "monitor_policy": self.monitor_policy,
            "operator_bundle": self.operator_bundle,
            "detection_calibration": self.detection_calibration,
            "attribution_calibration": self.attribution_calibration,
            "isolation_library": self.isolation_library,
            "certification_status": self.certification_status,
        }
        paths.update(
            {
                f"checkpoint_files.{name}": path
                for name, path in self.checkpoint_files.items()
            }
        )
        paths.update(
            {
                f"checkpoint_replay_outputs.{name}": path
                for name, path in self.checkpoint_replay_outputs.items()
            }
        )
        return MappingProxyType(paths)


class PaperNormalMethodConfig(StrictConfig):
    """P10 可接受的完整 P4 正常方法配置。

    参数：
        model: 已由 ``ProtectedKoopmanTSConfig`` 严格验证的模型结构、掩码、模糊规则和损失
            配置；不能只提供注册表 ``type``，也不能携带未知字段。
    返回：
        冻结的正常方法配置，供 formal builder 与 manifest 重放共同使用。
    异常：
        缺少任何 P4 理论敏感字段、嵌套数值非法或出现未知字段时由 Pydantic 抛出
        ``ValidationError``。
    副作用：
        无；不构造模型、不读取 checkpoint，也不修改随机状态。
    """

    model: ProtectedKoopmanTSConfig


class PaperDevelopmentTrainingConfig(StrictConfig):
    """正常开发训练的显式优化预算，不提供会被误当成论文结论的隐藏默认。"""

    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)


class PaperDevelopmentFeatureLayoutConfig(StrictConfig):
    """把原始过程列显式分成控制、测量和外生量索引。"""

    control_indices: tuple[int, ...] = Field(min_length=1)
    measurement_indices: tuple[int, ...] = Field(min_length=1)
    exogenous_indices: tuple[int, ...] = ()

    @model_validator(mode="after")
    def _validate_disjoint_indices(self) -> "PaperDevelopmentFeatureLayoutConfig":
        """要求三组索引非负、各自唯一且彼此不重叠。"""

        groups = (
            self.control_indices,
            self.measurement_indices,
            self.exogenous_indices,
        )
        flat = tuple(index for group in groups for index in group)
        if any(isinstance(index, bool) or index < 0 for index in flat):
            raise ValueError("Paper development feature indices must be nonnegative integers.")
        if any(len(set(group)) != len(group) for group in groups):
            raise ValueError("Paper development feature indices must be unique per role.")
        if len(set(flat)) != len(flat):
            raise ValueError("Paper development feature roles must not overlap.")
        return self


class PaperDevelopmentConfig(StrictConfig):
    """把正常-only P2--P9 开发运行所需的全部敏感参数显式冻结。"""

    method: PaperNormalMethodConfig
    split: FiveStageSplitConfig
    training: PaperDevelopmentTrainingConfig
    feature_layout: PaperDevelopmentFeatureLayoutConfig
    anchor_gate: AnchorGateConfig
    branch_id: str
    mode: str
    normal_family_id: str
    unresolved_family_id: str
    threshold_floor: float = Field(gt=0.0)
    gamma_anc: float = Field(ge=0.0)
    deterministic_intercept: float = Field(ge=0.0)
    input_l1_weight: float = Field(ge=0.0)

    @field_validator(
        "branch_id",
        "mode",
        "normal_family_id",
        "unresolved_family_id",
    )
    @classmethod
    def _validate_development_identifier(cls, value: str) -> str:
        """限制会进入 checkpoint 与方法产物的稳定名称。"""

        normalized = value.strip()
        if not _IDENTIFIER_RE.fullmatch(normalized):
            raise ValueError("Paper development identifiers must use safe identifier syntax.")
        return normalized

    @model_validator(mode="after")
    def _validate_model_layout(self) -> "PaperDevelopmentConfig":
        """把 P2 依赖跨度、P4 维数和原始列角色锁成同一合同。"""

        model = self.method.model
        layout = self.feature_layout
        if len(layout.control_indices) != model.control_dim:
            raise ValueError("Development control indices must match model.control_dim.")
        if len(layout.measurement_indices) != model.measurement_dim:
            raise ValueError(
                "Development measurement indices must match model.measurement_dim."
            )
        if len(layout.exogenous_indices) != model.exogenous_dim:
            raise ValueError("Development exogenous indices must match model.exogenous_dim.")
        if self.split.history_length != model.history_length:
            raise ValueError("Development split/model history_length must match.")
        if self.split.max_rollout != model.max_rollout:
            raise ValueError("Development split/model max_rollout must match.")
        if self.anchor_gate.maximum_reference_age > model.max_rollout:
            raise ValueError(
                "Development anchor_gate.maximum_reference_age cannot exceed "
                "model.max_rollout."
            )
        if self.normal_family_id == self.unresolved_family_id:
            raise ValueError("Development normal and unresolved family ids must differ.")
        return self


class FrozenEvaluationEntryConfig(StrictConfig):
    """P10/P11 smoke/development/frozen 的统一且受模式约束的入口配置。

    参数：
        mode/protocol_version/evaluation_id/runtime: 协议与运行时身份。
        artifact_root/run_name/claim_registry/device: 产物、全局一次性 ID 空间和 CPU 设备。
        detection_risk/attribution_risk: 检测 ``alpha`` 与严格更小的归因 ``beta``。
        seeds/dataset/development/normal_artifacts: 嵌套随机源、数据身份、正常开发参数和
            正常产物输入/输出路径。
        primary_protocol_lock: 仅 TTS development 必须提供的 CSTR 主配置只读身份。
    返回：
        冻结且拒绝未知字段的 P10 入口。
    异常：
        标识、模式组合、风险嵌套或嵌套配置非法时抛出 Pydantic ``ValidationError``。
    副作用：
        无；构造不读数据、产物或 claim registry。
    """

    mode: Literal["smoke", "development", "frozen"]
    protocol_version: str
    evaluation_id: str
    artifact_root: Path
    run_name: str
    claim_registry: Path
    device: Literal["cpu"]
    runtime: Literal["synthetic_contract_smoke", "protected_koopman_ts"]
    detection_risk: float = Field(gt=0.0, lt=1.0)
    attribution_risk: float = Field(gt=0.0, lt=1.0)
    seeds: PaperEvaluationSeedsConfig
    dataset: PaperEvaluationDatasetConfig
    primary_protocol_lock: PaperPrimaryProtocolLockConfig | None = None
    development: PaperDevelopmentConfig | None = None
    normal_artifacts: PaperNormalArtifactsConfig | None

    @field_validator("protocol_version", "evaluation_id", "run_name")
    @classmethod
    def _validate_identifier(cls, value: str) -> str:
        """限制会进入 manifest、claim 名和运行目录的稳定标识。"""

        normalized = value.strip()
        if not _IDENTIFIER_RE.fullmatch(normalized):
            raise ValueError(
                "Paper identifiers must use 3-128 letters, digits, '.', '_' or '-'."
            )
        return normalized

    @model_validator(mode="after")
    def _validate_mode_boundary(self) -> "FrozenEvaluationEntryConfig":
        """防止模式交叉复用，并强制归因风险预算严格嵌套在检测预算内。

        返回：
            校验后的当前配置。
        异常：
            runtime/数据集与模式不一致、development 冒充 frozen 产物，或
            ``attribution_risk >= detection_risk`` 时抛出 ``ValueError``。
        副作用：
            无。
        """

        if self.mode == "smoke":
            if (
                self.runtime != "synthetic_contract_smoke"
                or self.dataset.name != "synthetic_cstr"
                or self.primary_protocol_lock is not None
                or self.development is not None
                or self.normal_artifacts is not None
            ):
                raise ValueError(
                    "Smoke mode requires synthetic_contract_smoke, synthetic_cstr and no "
                    "primary_protocol_lock or normal_artifacts."
                )
        elif self.mode == "frozen":
            if (
                self.runtime != "protected_koopman_ts"
                or self.dataset.name != "cstr_closed_loop_fd"
                or self.primary_protocol_lock is not None
            ):
                raise ValueError(
                    "Frozen mode requires protected_koopman_ts, cstr_closed_loop_fd and no "
                    "secondary primary_protocol_lock."
                )
        elif self.runtime != "protected_koopman_ts" or self.dataset.name not in {
            "cstr_closed_loop_fd",
            "tts_fault_diagnosis",
        }:
            raise ValueError(
                "Development mode requires protected_koopman_ts and a supported real "
                "CSTR/TTS dataset."
            )
        if self.mode == "development":
            if self.development is None or self.normal_artifacts is None:
                raise ValueError(
                    "Development entry config requires explicit development parameters and "
                    "normal_artifacts output paths."
                )
            if (
                self.dataset.name == "tts_fault_diagnosis"
                and self.primary_protocol_lock is None
            ):
                raise ValueError(
                    "TTS development requires primary_protocol_lock so secondary results "
                    "cannot change the CSTR protocol."
                )
            if (
                self.dataset.name == "cstr_closed_loop_fd"
                and self.primary_protocol_lock is not None
            ):
                raise ValueError(
                    "Primary CSTR development must not declare a secondary protocol lock."
                )
        elif self.development is not None:
            raise ValueError("Only development mode may declare development parameters.")
        if self.development is not None:
            layout = self.development.feature_layout
            covered = {
                *layout.control_indices,
                *layout.measurement_indices,
                *layout.exogenous_indices,
            }
            if covered != set(range(self.dataset.feature_count)):
                raise ValueError(
                    "Development feature roles must cover every dataset feature exactly once."
                )
            # P11 的 TTS 发布只有这一套已核验物理列序；CSTR 的既有开发 fixture 仍允许
            # 显式替换模型布局，避免次级扩展反向改变 P10 的兼容边界。
            expected_layout = _SECONDARY_PHYSICAL_FEATURE_LAYOUTS.get(
                self.dataset.name
            )
            observed_layout = (
                layout.control_indices,
                layout.measurement_indices,
                layout.exogenous_indices,
            )
            if expected_layout is not None and observed_layout != expected_layout:
                raise ValueError(
                    f"{self.dataset.name} development physical feature layout must match "
                    "the public dataset adapter schema."
                )
        if self.attribution_risk >= self.detection_risk:
            raise ValueError(
                "attribution_risk (beta) must be strictly smaller than "
                "detection_risk (alpha)."
            )
        return self

    def frozen_readiness_errors(self, *, repo_root: str | Path) -> tuple[str, ...]:
        """只读列出阻止正式 frozen evaluation 的全部当前前置条件。

        参数：
            repo_root: 配置中仓库相对路径的解析根目录。
        返回：
            稳定错误元组；空元组才允许后续建立 manifest/claim。
        异常：
            文件 hash 读取的 ``OSError`` 被转为可理解的 readiness error，不向外抛出。
        副作用：
            只检查路径。只有许可已是 ``verified`` 时才读取 normal/fault 原始文件计算 hash；
            ``to_verify`` 当前状态不会触碰 fault 文件内容。
        """

        errors: list[str] = []
        if self.mode != "frozen":
            errors.append(
                f"mode is {self.mode!r}; explicit frozen evaluation requires mode='frozen'"
            )
        if self.dataset.license_status != "verified":
            errors.append(
                "dataset license status is "
                f"{self.dataset.license_status!r}; frozen fault access requires 'verified'"
            )
        if self.dataset.normal_source_hash is None:
            errors.append("dataset normal_source_hash is not frozen")
        if self.dataset.fault_source_hash is None:
            errors.append("dataset fault_source_hash is not frozen")
        root = Path(repo_root)
        if self.normal_artifacts is None:
            errors.append("normal_artifacts are not declared")
        else:
            for name, configured_path in self.normal_artifacts.paths().items():
                path = _resolve_repo_path(root, configured_path)
                if not path.is_file():
                    errors.append(f"normal_artifacts.{name} is missing: {path}")
        if self.dataset.license_status == "verified":
            data_root = _resolve_repo_path(root, self.dataset.root)
            for name, relative, expected_hash in (
                (
                    "normal",
                    self.dataset.normal_file,
                    self.dataset.normal_source_hash,
                ),
                (
                    "fault",
                    self.dataset.fault_file,
                    self.dataset.fault_source_hash,
                ),
            ):
                if relative is None:
                    errors.append(f"dataset {name}_file is not declared")
                    continue
                path = (data_root / relative).resolve()
                if not path.is_file():
                    errors.append(f"dataset {name} file is missing: {path}")
                    continue
                if expected_hash is None:
                    continue
                try:
                    observed_hash = sha256_file(path)
                except OSError as exc:
                    errors.append(f"dataset {name} hash cannot be read: {exc}")
                    continue
                if observed_hash != expected_hash:
                    errors.append(
                        f"dataset {name} SHA-256 differs from the frozen config"
                    )
        return tuple(errors)


@dataclass(frozen=True)
class ResolvedFrozenEvaluationConfig:
    """配置、JSON 解析值、逐叶来源和 16 位内容 hash 的不可变组合。

    参数：
        config: 严格 ``FrozenEvaluationEntryConfig``。
        resolved_config: 包含所有显式值的 JSON 兼容配置。
        provenance: 每个叶字段的来源记录。
        config_hash: resolved JSON 的 16 位 SHA-256 前缀。
    返回：
        深层只读的解析结果。
    异常：
        hash 格式非法时抛出 ``ValueError``。
    副作用：
        只复制并冻结内存映射。
    """

    config: FrozenEvaluationEntryConfig
    resolved_config: Mapping[str, Any]
    provenance: Mapping[str, tuple[Mapping[str, Any], ...]]
    config_hash: str

    def __post_init__(self) -> None:
        """复制并冻结映射，校验 16 位配置身份。"""

        if not re.fullmatch(r"[0-9a-f]{16}", self.config_hash):
            raise ValueError("Resolved frozen evaluation config_hash must be 16 hex characters.")
        object.__setattr__(
            self,
            "resolved_config",
            _deep_freeze_json(self.resolved_config),
        )
        object.__setattr__(
            self,
            "provenance",
            MappingProxyType(
                {
                    path: tuple(
                        MappingProxyType(dict(record)) for record in records
                    )
                    for path, records in self.provenance.items()
                }
            ),
        )


def resolve_frozen_evaluation_config(
    source: FrozenEvaluationEntryConfig | Mapping[str, Any] | str | Path,
) -> ResolvedFrozenEvaluationConfig:
    """严格解析 P10 入口配置并保留默认/显式字段来源。

    参数：
        source: 已校验对象、映射或 UTF-8 YAML 路径。
    返回：
        ``ResolvedFrozenEvaluationConfig``。
    异常：
        YAML 顶层非映射、未知字段或模式组合非法时传播 ``ValueError``/Pydantic 错误。
    副作用：
        传入路径时只读该 YAML；不检查数据/产物 readiness，不创建运行目录。
    """

    if isinstance(source, FrozenEvaluationEntryConfig):
        explicit = source.model_dump(mode="json", exclude_unset=True)
        config = source
        source_label = "api_config"
    elif isinstance(source, Mapping):
        explicit = _json_copy(dict(source))
        config = FrozenEvaluationEntryConfig.model_validate(explicit)
        source_label = "user_config"
    else:
        config_path = Path(source)
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, Mapping):
            raise ValueError(
                f"Frozen evaluation YAML {config_path} must contain a top-level mapping."
            )
        explicit = _json_copy(dict(loaded))
        config = FrozenEvaluationEntryConfig.model_validate(explicit)
        source_label = f"yaml:{config_path}"
    # P11 的次级锁只在被显式声明时参与配置身份。若把新字段的默认 ``None`` 注入旧 CSTR
    # resolved JSON，会在没有任何 P10 行为变化时改写其已冻结 hash，违反不回调主协议。
    excluded = (
        {"primary_protocol_lock"}
        if config.primary_protocol_lock is None
        else set()
    )
    resolved = config.model_dump(mode="json", exclude=excluded)
    encoded = json.dumps(
        resolved,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    explicit_paths = set(_flatten_leaves(explicit))
    provenance: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for leaf_path, value in _flatten_leaves(resolved).items():
        provenance[leaf_path] = (
            {
                "source": source_label if leaf_path in explicit_paths else "paper_default",
                "value": value,
            },
        )
    return ResolvedFrozenEvaluationConfig(
        config=config,
        resolved_config=resolved,
        provenance=provenance,
        config_hash=hashlib.sha256(encoded).hexdigest()[:16],
    )


def _resolve_repo_path(root: Path, value: Path | None) -> Path:
    """把配置路径解析到仓库根；绝对路径保持绝对。"""

    if value is None:
        return root.resolve()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _json_copy(value: Any) -> Any:
    """复制 Path/Pydantic 已解析后的 JSON 兼容值。"""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_copy(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_json_copy(item) for item in value]
    return value


def _deep_freeze_json(value: Any) -> Any:
    """递归把 JSON 映射/列表转换为只读映射/元组。"""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return tuple(_deep_freeze_json(item) for item in value)
    return value


def _flatten_leaves(value: Any, prefix: str = "") -> dict[str, Any]:
    """把嵌套配置展开成 provenance 使用的叶路径。"""

    if isinstance(value, Mapping):
        leaves: dict[str, Any] = {}
        if not value and prefix:
            leaves[prefix] = {}
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            leaves.update(_flatten_leaves(item, child))
        return leaves
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        leaves = {}
        if not value and prefix:
            leaves[prefix] = []
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            leaves.update(_flatten_leaves(item, child))
        return leaves
    return {prefix: value}


__all__ = [
    "FrozenEvaluationEntryConfig",
    "PaperDevelopmentConfig",
    "PaperDevelopmentFeatureLayoutConfig",
    "PaperDevelopmentTrainingConfig",
    "PaperEvaluationDatasetConfig",
    "PaperEvaluationSeedsConfig",
    "PaperNormalArtifactsConfig",
    "PaperNormalMethodConfig",
    "PaperPrimaryProtocolLockConfig",
    "ResolvedFrozenEvaluationConfig",
    "resolve_frozen_evaluation_config",
]
