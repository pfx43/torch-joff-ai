"""仓库公开示例脚本的子进程 smoke 测试。

文件用途：
    从仓库外工作目录启动示例，验证安装前 ``PYTHONPATH=src`` 路径、CPU 最小运行和约定
    产物确实可用，避免示例只在仓库当前目录偶然成功。
主要职责：
    覆盖通用模型示例、数据/Study 示例以及论文 P10 synthetic contract smoke；只检查公开
    命令的退出码与关键产物，并确认真实 CSTR 配置会在未核实的本地 MAT 许可处阻断；
    不复算模型内部单元逻辑。
关键输入与输出：
    输入为 pytest 临时运行根和显式 ``--smoke``/``--run-root`` 参数；输出全部位于
    ``tmp_path``，测试断言配置、checkpoint、图、manifest、逐时刻来源和 receipt。
依赖与副作用：
    通过 ``subprocess`` 启动当前 Python，最长等待 90 秒；不访问网络、不读取真实 CSTR
    故障文件，不修改仓库内 ``runs``。
重要约束：
    smoke 数值只能证明代码路径可运行，不能作为论文性能、认证或数据许可证据。
"""

from __future__ import annotations

import os
import subprocess
import sys
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_quickstart_dae_example_runs(tmp_path) -> None:
    completed = _run_example("quickstart_dae.py", cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr


def test_quickstart_mlp_example_runs(tmp_path) -> None:
    completed = _run_example("quickstart_mlp.py", cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr


def test_hm_nkn_smoke_example_runs_and_writes_artifacts(tmp_path) -> None:
    run_root = tmp_path / "runs"
    completed = _run_example(
        "hm_nkn.py",
        "--smoke",
        "--run-root",
        str(run_root),
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    run_dir = run_root / "hm_nkn_smoke"
    assert (run_dir / "resolved_config.yaml").exists()
    assert (run_dir / "provenance.json").exists()
    assert (run_dir / "checkpoints" / "last.pt").exists()
    assert (run_dir / "checkpoints" / "best.pt").exists()
    assert (run_dir / "data" / "outlier_summary.json").exists()
    assert (run_dir / "data" / "normalization_summary.json").exists()
    assert (run_dir / "data" / "dynamic_slice_summary.csv").exists()
    assert (run_dir / "data" / "dynamic_split_summary.csv").exists()
    assert (run_dir / "data" / "prepared_dataset_hash.json").exists()
    assert (run_dir / "plots" / "loss.png").exists()


def test_fd_cstr_smoke_example_runs_and_writes_fault_artifacts(tmp_path) -> None:
    run_root = tmp_path / "runs"
    completed = _run_example(
        "fd_cstr.py",
        "--smoke",
        "--run-root",
        str(run_root),
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    run_dir = run_root / "fd_cstr_smoke"
    assert (run_dir / "resolved_config.yaml").exists()
    assert (run_dir / "provenance.json").exists()
    assert (run_dir / "checkpoints" / "last.pt").exists()
    assert (run_dir / "checkpoints" / "best.pt").exists()
    assert (run_dir / "metrics" / "fault_detection_report.json").exists()
    assert (run_dir / "metrics" / "test_scores.csv").exists()
    assert (run_dir / "plots" / "fault_scores.png").exists()


def test_sweep_runner_smoke_example_runs_and_writes_summary(tmp_path) -> None:
    run_root = tmp_path / "runs"
    completed = _run_example(
        "sweep_runner.py",
        "--smoke",
        "--run-root",
        str(run_root),
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    assert (run_root / "sweep_runner" / "summary" / "summary.csv").exists()
    assert (run_root / "sweep_runner_exp_0" / "resolved_config.yaml").exists()
    assert (run_root / "sweep_runner_exp_1" / "metrics" / "test_metrics.json").exists()


def test_repeat_study_smoke_example_runs_and_writes_leaderboard(tmp_path) -> None:
    run_root = tmp_path / "runs"
    completed = _run_example(
        "repeat_study.py",
        "--smoke",
        "--run-root",
        str(run_root),
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    run_dir = run_root / "repeat_study"
    assert (run_dir / "summary" / "summary.csv").exists()
    assert (run_dir / "summary" / "leaderboard.csv").exists()
    assert (run_dir / "best" / "metrics.json").exists()


def test_paper_smoke_runs_frozen_contract_without_real_fault_data(tmp_path) -> None:
    """P10 synthetic 入口覆盖 manifest、一次性评价和机器来源，不触碰真实 CSTR。"""

    run_root = tmp_path / "runs"
    completed = _run_example(
        "paper_smoke.py",
        "--run-root",
        str(run_root),
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    run_dir = run_root / "paper_smoke"
    assert (run_dir / "resolved_config.yaml").exists()
    assert (run_dir / "provenance.json").exists()
    assert (run_dir / "frozen_protocol_manifest.json").exists()
    evaluation_dir = run_dir / "frozen_evaluation"
    assert (evaluation_dir / "pointwise" / "all_outputs.jsonl").exists()
    assert (evaluation_dir / "sources" / "score_trajectories.csv").exists()
    assert (evaluation_dir / "sources" / "detection_by_episode.csv").exists()
    assert (evaluation_dir / "sources" / "isolation_by_episode.csv").exists()
    assert (evaluation_dir / "artifact_index.json").exists()
    assert (evaluation_dir / "evaluation_receipt.json").exists()


def test_frozen_evaluation_command_blocks_before_claim_while_mat_license_is_unverified(
    tmp_path: Path,
) -> None:
    """本地 MAT 生成链未核实，必须在 runtime、manifest 与 claim 之前关闭。"""

    env = os.environ.copy()
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "joff.experiments.frozen_cli",
            "--config",
            str(ROOT / "configs" / "paper" / "cstr_frozen.yaml"),
            "--repo-root",
            str(ROOT),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 2, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "blocked"
    assert any("license" in error for error in payload["errors"])
    assert payload["claim_created"] is False
    assert payload["fault_data_accessed"] is False
    assert not (tmp_path / "runs").exists()


def _run_example(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, str(ROOT / "examples" / args[0]), *args[1:]],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
