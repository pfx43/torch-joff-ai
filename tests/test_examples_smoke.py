from __future__ import annotations

import os
import subprocess
import sys
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
