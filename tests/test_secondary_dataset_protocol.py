"""
论文 P11 次级数据集协议的公开接口回归测试。

文件用途：
    验证三容水箱（TTS）数据能够沿用 P1 的 CanonicalDataset 与 P2 的五段正常数据入口，
    同时保留七个过程变量、六个故障 episode、onset=200 和故障族的明确语义。
主要职责：
    固定 TTS 公开适配器的目录、列选择、物理角色、逐行标签和来源摘要；证明
    PaperDataBundle.from_canonical 只从正常 train split 建立五段协议。
关键输入与输出：
    输入为测试临时目录中的小型合成 MAT 与说明文件；输出为对公开 registry、schema、
    CanonicalDataset、TaskSchema 和 PaperDataBundle manifest 的行为断言。
依赖与副作用：
    依赖 SciPy 写入临时 MAT，并用系统 Git 在 tmp_path 下建立一次性隔离仓库；不访问网络，
    不修改主仓库，不读取仓库内真实 TTS/TE 故障数值，也不生成或声称任何论文性能结果。
重要约束：
    合成故障矩阵的后六列模拟随数据发布的故障输出通道，它们是答案信息，必须在适配器
    边界被排除；次级数据集不得反向改变已经冻结的 CSTR 模型、阈值或结构选择。主协议锁
    必须跟随已提交的 CSTR 许可证据和 frozen 配置身份更新，不能绑定未提交工作树。
"""

from __future__ import annotations

import copy
import hashlib
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from pydantic import ValidationError
from scipy.io import savemat

from joff.data.adapters import DATASET_REGISTRY
from joff.data.paper_protocol import (
    FitPurpose,
    FiveStageSplitConfig,
    PaperDataBundle,
    StageName,
)
from joff.experiments import FrozenProtocolIntegrityError, resolve_frozen_evaluation_config


ROOT = Path(__file__).resolve().parents[1]
TTS_DEVELOPMENT_CONFIG = ROOT / "configs" / "paper" / "tts_development.yaml"
TTS_CARD_PATH = (
    ROOT / "datasets" / "cards" / "oa" / "tts_fault_diagnosis" / "dataset_card.yaml"
)


def _write_synthetic_tts_release(
    root: Path,
    *,
    normal_width: int = 7,
    fault_width: int = 13,
) -> Path:
    """创建最小 TTS 发布结构，模拟七个过程变量加六个不可作为输入的故障输出。

    参数：
        root: pytest 管理的临时根目录。
        normal_width/fault_width: 合成正常与故障 MAT 的原始列宽；默认分别复现
            官方 7 个过程变量和“7 个过程变量 + 6 个受保护故障输出”。
    返回：
        可直接传给 ``TTSFaultDiagnosisAdapter.read`` 的数据家族根目录。
    异常：
        临时目录或 MAT 写入失败时传播原始文件系统/SciPy 异常。
    副作用：
        在 ``root/fe`` 下创建 train、test、说明文件；不触碰仓库真实数据。
    """

    release_root = root / "fe"
    (release_root / "train").mkdir(parents=True)
    (release_root / "test").mkdir()

    normal = np.arange(480 * normal_width, dtype=float).reshape(480, normal_width)
    fault_payload: dict[str, np.ndarray] = {}
    for fault_id in range(1, 7):
        fault_payload[f"Fault{fault_id:02d}"] = np.full(
            (205, fault_width),
            float(fault_id),
        )

    savemat(release_root / "train" / "[train].mat", {"normal": normal})
    savemat(release_root / "test" / "[test].mat", fault_payload)
    (release_root / "7v+6f 正序, 6c 故障.txt").write_text(
        "u=(Q1,Q2); y=(Q1s,Q2s,h1,h2,h3); fault_onset=200",
        encoding="utf-8",
    )
    return root


def _rebase_tts_artifact_paths(value: Any, *, run_dir: Path) -> Any:
    """把正式 TTS YAML 中的运行路径递归重定向到 pytest 临时目录。

    参数：
        value: ``normal_artifacts`` 中的字符串、映射或列表。
        run_dir: 临时 TTS 运行目录。
    返回：
        保持原结构、只替换 ``runs/tts_development`` 前缀后的 JSON 兼容值。
    异常：
        输入结构不受支持时按原值返回；路径解析错误由 ``Path.relative_to`` 传播。
    副作用：
        无。不创建目录、不修改原始 YAML 映射。
    """

    if isinstance(value, dict):
        return {
            key: _rebase_tts_artifact_paths(item, run_dir=run_dir)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rebase_tts_artifact_paths(item, run_dir=run_dir) for item in value]
    if isinstance(value, str) and value.startswith("runs/tts_development"):
        relative = Path(value).relative_to(Path("runs") / "tts_development")
        return str(run_dir / relative)
    return value


def _run_git(repo: Path, *arguments: str) -> str:
    """在测试隔离仓库中执行只读或最小建库 Git 命令。

    参数：
        repo: ``tmp_path`` 下的隔离仓库，不得指向项目主仓库。
        arguments: 传给 Git 的参数。
    返回：
        去除首尾空白的标准输出。
    异常：
        Git 不可用或命令失败时由 ``subprocess.run(check=True)`` 抛出。
    副作用：
        仅对隔离仓库执行调用方明确传入的 init/config/add/commit/rev-parse；不修改主仓库。
    """

    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    return completed.stdout.strip()


def test_tts_adapter_preserves_seven_physical_variables_and_six_fault_episodes(
    tmp_path: Path,
) -> None:
    """TTS 适配器必须排除六个答案通道，并从 onset=200 开始逐行赋故障标签。"""

    root = _write_synthetic_tts_release(tmp_path)
    adapter = DATASET_REGISTRY.resolve("tts_fault_diagnosis")
    assert DATASET_REGISTRY.resolve("TTS/fe") is adapter
    dataset = adapter.read(root=root, task="fault_diagnosis")
    task = adapter.default_task("fault_diagnosis")

    assert adapter.schema().role_columns("control_input") == ("Q1", "Q2")
    assert adapter.schema().role_columns("measured_output") == (
        "Q1s",
        "Q2s",
        "h1",
        "h2",
        "h3",
    )
    assert adapter.schema().role_columns("raw_index") == ("raw_index",)
    assert task.inputs == ("control_input", "measured_output")
    assert task.fault_switch == 200
    assert dataset.split_rows() == {"train": 480, "test": 6 * 205}

    expected_families = {
        1: "actuator",
        2: "actuator",
        3: "actuator",
        4: "sensor",
        5: "sensor",
        6: "sensor",
    }
    for fault_id, segment in enumerate(dataset.splits["test"], start=1):
        assert segment.frame.columns.tolist() == [
            "time",
            "raw_index",
            "Q1",
            "Q2",
            "Q1s",
            "Q2s",
            "h1",
            "h2",
            "h3",
            "segment",
            "fault_id",
        ]
        assert segment.frame.loc[:199, "fault_id"].eq(0).all()
        assert segment.frame.loc[200:, "fault_id"].eq(fault_id).all()
        assert segment.meta.metadata["fault_family"] == expected_families[fault_id]
        assert segment.meta.metadata["label_counts"] == {
            "0": 200,
            str(fault_id): 5,
        }

    summary = dataset.source_summary()
    assert summary["variables"] == {
        "control_input": ["Q1", "Q2"],
        "measured_output": ["Q1s", "Q2s", "h1", "h2", "h3"],
    }
    assert summary["fault_onset"] == 200
    assert summary["fault_families"] == {
        str(fault_id): family for fault_id, family in expected_families.items()
    }
    assert summary["stored_feature_indices"] == list(range(7))
    assert summary["files"]["description"]["path"].endswith("7v+6f 正序, 6c 故障.txt")


@pytest.mark.parametrize(
    ("normal_width", "fault_width"),
    (
        (8, 13),
        (7, 12),
        (7, 14),
    ),
)
def test_tts_adapter_rejects_raw_widths_outside_the_published_7_plus_6_layout(
    tmp_path: Path,
    normal_width: int,
    fault_width: int,
) -> None:
    """TTS 发布矩阵不能凭“至少七列”通过；正常/故障列宽必须精确为 7/13。"""

    root = _write_synthetic_tts_release(
        tmp_path,
        normal_width=normal_width,
        fault_width=fault_width,
    )
    adapter = DATASET_REGISTRY.resolve("tts_fault_diagnosis")

    with pytest.raises(ValueError, match="raw width"):
        adapter.read(root=root, task="fault_diagnosis")


def test_tts_canonical_normal_split_builds_five_stages_without_attaching_fault_test(
    tmp_path: Path,
) -> None:
    """P2 入口只能使用 TTS 正常 train，不能自动把 adapter 的 test 作为冻结故障输入。"""

    root = _write_synthetic_tts_release(tmp_path)
    canonical = DATASET_REGISTRY.resolve("tts_fault_diagnosis").read(
        root=root,
        task="fault_diagnosis",
    )
    bundle = PaperDataBundle.from_canonical(
        canonical,
        config=FiveStageSplitConfig(
            history_length=3,
            max_rollout=2,
            minimum_gap=5,
            episode_length=24,
            target_risk_level=0.5,
        ),
    )

    train = bundle.data_for_fit("tts_model", FitPurpose.MODEL_PARAMETERS)
    assert train.shape[1] == 7
    assert bundle.split_result.stage(StageName.TRAIN).raw_indices[0] == 0
    assert bundle.manifest()["frozen_fault_test"]["configured"] is False


def test_tts_dataset_card_routes_to_the_physical_protocol_adapter(tmp_path: Path) -> None:
    """卡片路径必须复现七变量协议，而不是退回通用 MAT 卡片读取器。"""

    root = _write_synthetic_tts_release(tmp_path)
    adapter = DATASET_REGISTRY.resolve(TTS_CARD_PATH)
    dataset = adapter.read(root=root, task="fault_diagnosis")

    assert adapter.schema().role_columns("control_input") == ("Q1", "Q2")
    assert adapter.default_task().fault_switch == 200
    assert dataset.split_rows() == {"train": 480, "test": 6 * 205}
    assert dataset.splits["test"][0].frame.columns.tolist()[2:9] == [
        "Q1",
        "Q2",
        "Q1s",
        "Q2s",
        "h1",
        "h2",
        "h3",
    ]


def test_tts_development_config_is_strict_and_binds_the_primary_cstr_protocol() -> None:
    """次级开发配置必须固定 CSTR 配置身份，并且不能被改成 TTS frozen 入口。"""

    assert (
        resolve_frozen_evaluation_config(
            ROOT / "configs" / "paper" / "cstr_frozen.yaml"
        ).config_hash
        == "b29bc1aae5a3751a"
    )
    resolved = resolve_frozen_evaluation_config(TTS_DEVELOPMENT_CONFIG)
    config = resolved.config

    assert config.mode == "development"
    assert config.dataset.name == "tts_fault_diagnosis"
    assert config.dataset.fault_episode_count == 6
    assert config.development is not None
    assert config.development.feature_layout.control_indices == (0, 1)
    assert config.development.feature_layout.measurement_indices == (2, 3, 4, 5, 6)
    assert config.primary_protocol_lock is not None
    assert config.primary_protocol_lock.dataset_name == "cstr_closed_loop_fd"
    assert (
        config.primary_protocol_lock.frozen_config_commit
        == "06f5abd0985e04615f78f1ebe3906d1dfe8c64ec"
    )
    assert (
        config.primary_protocol_lock.frozen_config_sha256
        == "5b5e4c48cfe9831124aa42d6a3e7eea1986ca37668f00f77f7efc12ff709d3ae"
    )
    assert (
        config.primary_protocol_lock.selection_status
        == "configuration_frozen_fault_evaluation_blocked"
    )
    assert config.primary_protocol_lock.evaluation_id is None
    assert config.primary_protocol_lock.manifest_path is None
    assert config.primary_protocol_lock.manifest_sha256 is None
    assert config.primary_protocol_lock.manifest_hash is None
    assert config.primary_protocol_lock.normal_artifact_bundle_hash is None
    assert config.primary_protocol_lock.receipt_path is None
    assert config.primary_protocol_lock.receipt_sha256 is None
    assert config.primary_protocol_lock.secondary_results_may_modify_primary is False
    assert config.primary_protocol_lock.fault_results_accessed is False

    raw = yaml.safe_load(TTS_DEVELOPMENT_CONFIG.read_text(encoding="utf-8"))
    assert "implementation_commit" not in raw["primary_protocol_lock"]
    without_lock = copy.deepcopy(raw)
    without_lock.pop("primary_protocol_lock")
    with pytest.raises(ValidationError, match="primary_protocol_lock"):
        resolve_frozen_evaluation_config(without_lock)

    frozen_tts = copy.deepcopy(raw)
    frozen_tts["mode"] = "frozen"
    frozen_tts["development"] = None
    with pytest.raises(ValidationError, match="frozen"):
        resolve_frozen_evaluation_config(frozen_tts)

    wrong_layout = copy.deepcopy(raw)
    wrong_layout["development"]["feature_layout"] = {
        "control_indices": [0, 2],
        "measurement_indices": [1, 3, 4, 5, 6],
        "exogenous_indices": [],
    }
    with pytest.raises(ValidationError, match="physical feature layout"):
        resolve_frozen_evaluation_config(wrong_layout)

    incomplete_completion = copy.deepcopy(raw)
    incomplete_completion["primary_protocol_lock"].update(
        {
            "selection_status": "formal_cstr_frozen_evaluation_completed",
            "evaluation_id": "formal-cstr-evaluation",
            "manifest_path": "runs/formal-cstr/manifest.json",
            "manifest_sha256": "a" * 64,
            "manifest_hash": "b" * 64,
            "normal_artifact_bundle_hash": "c" * 64,
        }
    )
    with pytest.raises(ValidationError, match="completed primary protocol lock"):
        resolve_frozen_evaluation_config(incomplete_completion)


def test_tts_development_validates_primary_lock_then_enforces_the_p11_stage_gate(
    tmp_path: Path,
) -> None:
    """runner 必须先复验 CSTR 锁与 Git 身份，再因正式 P10 尚未完成而关闭。"""

    from joff.experiments.paper_development import run_paper_normal_development

    raw = yaml.safe_load(TTS_DEVELOPMENT_CONFIG.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(raw)
    tampered["primary_protocol_lock"]["frozen_config_sha256"] = "0" * 64
    tampered_resolved = resolve_frozen_evaluation_config(tampered)
    with pytest.raises(FrozenProtocolIntegrityError, match="primary CSTR.*SHA-256"):
        run_paper_normal_development(tampered_resolved, repo_root=ROOT)

    wrong_identity = copy.deepcopy(raw)
    wrong_identity["primary_protocol_lock"]["protocol_version"] = "wrong-cstr-v1"
    with pytest.raises(FrozenProtocolIntegrityError, match="primary CSTR.*identity"):
        run_paper_normal_development(
            resolve_frozen_evaluation_config(wrong_identity),
            repo_root=ROOT,
        )

    nonexistent_commit = copy.deepcopy(raw)
    nonexistent_commit["primary_protocol_lock"]["frozen_config_commit"] = "0" * 40
    with pytest.raises(FrozenProtocolIntegrityError, match="frozen-config commit.*Git"):
        run_paper_normal_development(
            resolve_frozen_evaluation_config(nonexistent_commit),
            repo_root=ROOT,
        )

    outside_repository = copy.deepcopy(raw)
    outside_repository["primary_protocol_lock"]["frozen_config"] = str(
        ROOT.parent / "outside-cstr-frozen.yaml"
    )
    with pytest.raises(
        FrozenProtocolIntegrityError,
        match="must remain inside the repository root",
    ):
        run_paper_normal_development(
            resolve_frozen_evaluation_config(outside_repository),
            repo_root=ROOT,
        )

    isolated_repo = tmp_path / "primary-repo"
    frozen_config = isolated_repo / "configs" / "paper" / "cstr_frozen.yaml"
    frozen_config.parent.mkdir(parents=True)
    frozen_config.write_bytes(
        (ROOT / "configs" / "paper" / "cstr_frozen.yaml").read_bytes()
    )
    _run_git(isolated_repo, "init")
    _run_git(isolated_repo, "config", "user.name", "P11 Test")
    _run_git(isolated_repo, "config", "user.email", "p11-test@example.invalid")
    _run_git(isolated_repo, "add", "configs/paper/cstr_frozen.yaml")
    _run_git(isolated_repo, "commit", "-m", "test: freeze primary config")
    primary_commit = _run_git(isolated_repo, "rev-parse", "HEAD")
    frozen_config.write_bytes(
        frozen_config.read_bytes() + b"\n# test-only uncommitted protocol change\n"
    )

    uncommitted_config = copy.deepcopy(raw)
    uncommitted_config["primary_protocol_lock"]["frozen_config_commit"] = (
        primary_commit
    )
    uncommitted_config["primary_protocol_lock"]["frozen_config"] = (
        "configs/paper/cstr_frozen.yaml"
    )
    uncommitted_config["primary_protocol_lock"]["frozen_config_sha256"] = (
        hashlib.sha256(frozen_config.read_bytes()).hexdigest()
    )
    with pytest.raises(
        FrozenProtocolIntegrityError,
        match="protected frozen config.*Git commit",
    ):
        run_paper_normal_development(
            resolve_frozen_evaluation_config(uncommitted_config),
            repo_root=isolated_repo,
        )

    resolved = resolve_frozen_evaluation_config(TTS_DEVELOPMENT_CONFIG)
    with pytest.raises(
        FrozenProtocolIntegrityError,
        match="formal CSTR frozen evaluation has not completed",
    ):
        run_paper_normal_development(resolved, repo_root=ROOT)


def test_blocked_p11_tts_development_never_reads_data_or_creates_outputs(
    tmp_path: Path,
) -> None:
    """即使 TTS 许可与 normal 文件就绪，P10 未正式完成时也必须在 I/O 前关闭。"""

    from joff.experiments.paper_development import run_paper_normal_development

    data_root = tmp_path / "data"
    data_root.mkdir()
    normal_path = data_root / "normal.mat"
    normal = np.arange(800 * 7, dtype=float).reshape(800, 7)
    savemat(normal_path, {"normal": normal})
    normal_hash = hashlib.sha256(normal_path.read_bytes()).hexdigest()
    sealed_fault_path = data_root / "sealed-fault.mat"

    raw = yaml.safe_load(TTS_DEVELOPMENT_CONFIG.read_text(encoding="utf-8"))
    run_dir = tmp_path / "tts_run"
    raw.update(
        {
            "artifact_root": str(tmp_path),
            "run_name": "tts_run",
            "claim_registry": str(tmp_path / "claims"),
            "detection_risk": 0.3,
            "attribution_risk": 0.2,
        }
    )
    raw["dataset"].update(
        {
            "root": str(data_root),
            "normal_file": "normal.mat",
            "fault_file": sealed_fault_path.name,
            "license_status": "verified",
            "normal_rows": 800,
            "normal_source_hash": normal_hash,
            "fault_source_hash": "b" * 64,
        }
    )
    raw["development"]["split"].update(
        {
            "episode_length": 8,
            "target_risk_level": 0.2,
            "seed": 19,
        }
    )
    raw["development"]["training"] = {
        "epochs": 1,
        "batch_size": 128,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
    }
    raw["normal_artifacts"] = _rebase_tts_artifact_paths(
        raw["normal_artifacts"],
        run_dir=run_dir,
    )

    cstr_config = ROOT / "configs" / "paper" / "cstr_frozen.yaml"
    cstr_hash_before = hashlib.sha256(cstr_config.read_bytes()).hexdigest()
    with pytest.raises(
        FrozenProtocolIntegrityError,
        match="formal CSTR frozen evaluation has not completed",
    ):
        run_paper_normal_development(
            resolve_frozen_evaluation_config(raw),
            repo_root=ROOT,
        )
    cstr_hash_after = hashlib.sha256(cstr_config.read_bytes()).hexdigest()

    assert normal_path.is_file()
    assert not sealed_fault_path.exists()
    assert not run_dir.exists()
    assert cstr_hash_after == cstr_hash_before
