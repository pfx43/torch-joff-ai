"""论文数据集卡、原始路径和许可证据边界的公开回归测试。

文件用途：
    验证 paper development/frozen 入口不能把 hash 正确但位于授权边界外的卡片、raw 子文件
    或证据文件当成正式许可链，防止配置通过绝对路径、``..`` 或符号链接重定向数据访问。
主要职责：
    覆盖卡片 hash、配置/卡片许可交叉核对、缺失证据、raw 路径漂移、development 直调，
    以及卡片逃离仓库/数据根、normal 文件逃离数据根和证据文件逃离授权根；不重复测试
    P10 manifest、claim 或模型运行。
关键输入与输出：
    输入是 pytest 临时目录、严格 CSTR frozen 配置和 test-only 证据文本；输出是稳定的
    readiness error 元组，不读取真实 CSTR MAT，也不生成论文结果。
依赖与副作用：
    依赖 PyYAML 与 ``joff.experiments`` 公共配置入口；只在 ``tmp_path`` 写合成卡片和证据，
    不访问网络、不修改 Git、不创建正式运行目录。
重要约束：
    fixture 中的 ``verified`` 只表示测试自建文件彼此一致，绝不证明真实数据许可。路径检查
    必须基于解析后的路径，因而同样阻止 ``..`` 和可解析符号链接逃逸。
"""

from __future__ import annotations

from pathlib import Path

import hashlib

import pytest
import yaml

from joff.experiments import (
    FrozenProtocolIntegrityError,
    resolve_frozen_evaluation_config,
)
from joff.experiments.paper_development import run_cstr_normal_development


_ROOT = Path(__file__).resolve().parents[1]
_NORMAL_HASH = "a" * 64
_FAULT_HASH = "b" * 64


def _write_verified_card(
    card_path: Path,
    *,
    data_root: Path,
    normal_file: str = "train/normal.mat",
    fault_file: str = "test/fault.mat",
    evidence_root: Path | None = None,
) -> str:
    """写入 test-only verified 卡片并返回 SHA-256。

    ``evidence_root`` 可故意放在许可路径边界外，用来证明内容 hash 不能替代路径授权。
    """

    card_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_evidence_root = evidence_root or card_path.parent
    resolved_evidence_root.mkdir(parents=True, exist_ok=True)
    generation_record = resolved_evidence_root / "generation.txt"
    license_evidence = resolved_evidence_root / "license.txt"
    generation_record.write_text("test-only generation\n", encoding="utf-8")
    license_evidence.write_text("test-only permission\n", encoding="utf-8")
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
                "normal_sha256": _NORMAL_HASH,
                "fault_sha256": _FAULT_HASH,
                "generation_record": str(generation_record),
                "generation_record_sha256": hashlib.sha256(
                    generation_record.read_bytes()
                ).hexdigest(),
                "license_evidence": str(license_evidence),
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
    return hashlib.sha256(card_path.read_bytes()).hexdigest()


def _write_current_to_verify_card(
    card_path: Path,
    *,
    data_root: Path,
) -> str:
    """复制当前真实 CSTR 卡片并仅把 raw root 指向不存在的合成封存目录。"""

    source = (
        _ROOT
        / "datasets"
        / "cards"
        / "oa"
        / "cstr_closed_loop_fd"
        / "dataset_card.yaml"
    )
    card = yaml.safe_load(source.read_text(encoding="utf-8"))
    card["files"]["root"] = str(data_root)
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(
        yaml.safe_dump(card, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return hashlib.sha256(card_path.read_bytes()).hexdigest()


def _dataset_errors(
    *,
    repo_root: Path,
    card_path: Path,
    card_hash: str,
    data_root: Path,
    normal_file: str = "train/normal.mat",
    fault_file: str = "test/fault.mat",
) -> tuple[str, ...]:
    """从严格 frozen 配置生成数据集卡错误，不运行其余 P10 readiness。"""

    raw = yaml.safe_load(
        (_ROOT / "configs" / "paper" / "cstr_frozen.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["dataset"].update(
        {
            "root": str(data_root),
            "normal_file": normal_file,
            "fault_file": fault_file,
            "dataset_card": str(card_path),
            "dataset_card_sha256": card_hash,
            "license_status": "verified",
            "normal_source_hash": _NORMAL_HASH,
            "fault_source_hash": _FAULT_HASH,
        }
    )
    return resolve_frozen_evaluation_config(
        raw
    ).config.dataset.dataset_card_readiness_errors(repo_root=repo_root)


def test_frozen_readiness_requires_exact_dataset_card_identity_before_raw_files(
    tmp_path: Path,
) -> None:
    """正式预检必须先复验数据卡 hash，不能只相信同一配置内自报的许可状态。"""

    repo_root = tmp_path / "repo"
    sealed_root = repo_root / "sealed-raw"
    card_path = repo_root / "datasets" / "cards" / "oa" / "fixture" / "card.yaml"
    _write_current_to_verify_card(card_path, data_root=sealed_root)
    raw = yaml.safe_load(
        (_ROOT / "configs" / "paper" / "cstr_frozen.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["dataset"].update(
        {
            "root": str(sealed_root),
            "dataset_card": str(card_path),
            "dataset_card_sha256": "0" * 64,
        }
    )

    errors = resolve_frozen_evaluation_config(
        raw
    ).config.frozen_readiness_errors(repo_root=repo_root)

    assert "dataset card SHA-256 differs from the frozen config" in errors
    assert not any("dataset normal file" in error for error in errors)
    assert not any("dataset fault file" in error for error in errors)


def test_frozen_readiness_rejects_config_license_that_overstates_dataset_card(
    tmp_path: Path,
) -> None:
    """配置不能单独把数据许可改为 verified，也不能因此触碰封存 raw 文件。"""

    repo_root = tmp_path / "repo"
    sealed_root = repo_root / "sealed-raw"
    card_path = repo_root / "datasets" / "cards" / "oa" / "fixture" / "card.yaml"
    card_hash = _write_current_to_verify_card(card_path, data_root=sealed_root)
    raw = yaml.safe_load(
        (_ROOT / "configs" / "paper" / "cstr_frozen.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["dataset"].update(
        {
            "root": str(sealed_root),
            "dataset_card": str(card_path),
            "dataset_card_sha256": card_hash,
            "license_status": "verified",
        }
    )

    errors = resolve_frozen_evaluation_config(
        raw
    ).config.frozen_readiness_errors(repo_root=repo_root)

    assert (
        "dataset card license_status is 'to_verify'; "
        "entry config declares 'verified'"
    ) in errors
    assert not any("dataset normal file" in error for error in errors)
    assert not any("dataset fault file" in error for error in errors)


def test_verified_dataset_card_requires_hashed_mat_evidence_files(
    tmp_path: Path,
) -> None:
    """只翻转卡片状态不构成证据；生成记录和许可文件必须存在且被 hash 绑定。"""

    repo_root = tmp_path / "repo"
    data_root = repo_root / "data"
    card_path = repo_root / "datasets" / "cards" / "oa" / "fixture" / "card.yaml"
    _write_current_to_verify_card(card_path, data_root=data_root)
    card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
    card["access"].update(
        {
            "license": "BSD-3-Clause",
            "license_status": "verified",
        }
    )
    card["source"]["local_mat"].update(
        {
            "provenance_status": "verified",
            "license_status": "verified",
        }
    )
    card_path.write_text(
        yaml.safe_dump(card, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    raw = yaml.safe_load(
        (_ROOT / "configs" / "paper" / "cstr_frozen.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["dataset"].update(
        {
            "root": str(data_root),
            "dataset_card": str(card_path),
            "dataset_card_sha256": hashlib.sha256(card_path.read_bytes()).hexdigest(),
            "license_status": "verified",
        }
    )

    errors = resolve_frozen_evaluation_config(
        raw
    ).config.frozen_readiness_errors(repo_root=repo_root)

    assert "verified dataset card requires generation_record and its SHA-256" in errors
    assert "verified dataset card requires license_evidence and its SHA-256" in errors


def test_frozen_readiness_rejects_raw_paths_that_differ_from_dataset_card() -> None:
    """入口不能在保留数据卡 hash 的同时把 normal/fault 路径重定向到别的文件。"""

    frozen_path = _ROOT / "configs" / "paper" / "cstr_frozen.yaml"
    raw = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    raw["dataset"]["normal_file"] = "train/not-the-card-file.mat"

    errors = resolve_frozen_evaluation_config(
        raw
    ).config.frozen_readiness_errors(repo_root=_ROOT)

    assert "dataset normal_file differs from the dataset card" in errors


def test_normal_development_rejects_overstated_license_before_data_and_outputs(
    tmp_path: Path,
) -> None:
    """normal-only 入口也必须先复验数据卡，不能靠配置自报 verified 绕过。"""

    repo_root = tmp_path / "repo"
    sealed_root = repo_root / "sealed-raw"
    card_path = repo_root / "datasets" / "cards" / "oa" / "fixture" / "card.yaml"
    card_hash = _write_current_to_verify_card(card_path, data_root=sealed_root)
    config_path = _ROOT / "configs" / "paper" / "cstr_development.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw.update(
        {
            "artifact_root": str(tmp_path / "runs"),
            "run_name": "must-not-exist",
        }
    )
    raw["dataset"].update(
        {
            "root": str(sealed_root),
            "dataset_card": str(card_path),
            "dataset_card_sha256": card_hash,
            "license_status": "verified",
        }
    )

    with pytest.raises(
        FrozenProtocolIntegrityError,
        match="dataset card license_status.*entry config",
    ):
        run_cstr_normal_development(
            resolve_frozen_evaluation_config(raw),
            repo_root=repo_root,
        )

    assert not (tmp_path / "runs").exists()
    assert not sealed_root.exists()


def test_dataset_card_cannot_escape_repo_or_declared_data_root(tmp_path: Path) -> None:
    """绝对卡片路径即使 hash 正确，也不能指向仓库和声明数据根之外。"""

    repo_root = tmp_path / "repo"
    data_root = tmp_path / "data"
    card_path = tmp_path / "external-evidence" / "dataset_card.yaml"
    card_hash = _write_verified_card(card_path, data_root=data_root)

    errors = _dataset_errors(
        repo_root=repo_root,
        card_path=card_path,
        card_hash=card_hash,
        data_root=data_root,
    )

    assert "dataset card escapes the repository and dataset root" in errors


def test_existing_dataset_card_symlink_cannot_escape_allowed_roots(
    tmp_path: Path,
) -> None:
    """平台允许创建 symlink 时，边界内链接也不能把卡片解析到边界外。"""

    repo_root = tmp_path / "repo"
    data_root = tmp_path / "data"
    external_card = tmp_path / "external-evidence" / "dataset_card.yaml"
    card_hash = _write_verified_card(external_card, data_root=data_root)
    linked_card = (
        repo_root
        / "datasets"
        / "cards"
        / "oa"
        / "fixture"
        / "linked-dataset-card.yaml"
    )
    linked_card.parent.mkdir(parents=True, exist_ok=True)
    try:
        linked_card.symlink_to(external_card)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"当前文件系统不允许创建测试 symlink: {exc}")

    errors = _dataset_errors(
        repo_root=repo_root,
        card_path=linked_card,
        card_hash=card_hash,
        data_root=data_root,
    )

    assert "dataset card escapes the repository and dataset root" in errors


def test_raw_child_path_cannot_escape_declared_data_root(tmp_path: Path) -> None:
    """卡片与配置一致也不能把 normal 子文件通过 ``..`` 指到 root 外。"""

    repo_root = tmp_path / "repo"
    data_root = tmp_path / "data"
    card_path = repo_root / "datasets" / "cards" / "oa" / "fixture" / "card.yaml"
    normal_file = "../outside.mat"
    card_hash = _write_verified_card(
        card_path,
        data_root=data_root,
        normal_file=normal_file,
    )

    errors = _dataset_errors(
        repo_root=repo_root,
        card_path=card_path,
        card_hash=card_hash,
        data_root=data_root,
        normal_file=normal_file,
    )

    assert "dataset normal_file escapes the dataset root" in errors


def test_hashed_license_evidence_cannot_escape_repo_or_data_root(
    tmp_path: Path,
) -> None:
    """证据内容 hash 正确仍不足够；其真实路径必须留在审计边界内。"""

    repo_root = tmp_path / "repo"
    data_root = tmp_path / "data"
    card_path = repo_root / "datasets" / "cards" / "oa" / "fixture" / "card.yaml"
    external_evidence = tmp_path / "external-evidence"
    card_hash = _write_verified_card(
        card_path,
        data_root=data_root,
        evidence_root=external_evidence,
    )

    errors = _dataset_errors(
        repo_root=repo_root,
        card_path=card_path,
        card_hash=card_hash,
        data_root=data_root,
    )

    assert "verified dataset card generation_record escapes allowed roots" in errors
    assert "verified dataset card license_evidence escapes allowed roots" in errors
