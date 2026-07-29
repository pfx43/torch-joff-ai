"""论文运行 manifest 使用的环境、Git 和文件身份采集。

文件用途：
    集中提供 P10 smoke 与 formal freeze 共用的 Python/依赖版本、当前 Git commit 和流式
    SHA-256，避免不同入口对同一实验身份采用不同算法。
主要职责：
    只读收集环境事实；不解析论文配置、不访问数据语义、不创建运行目录或 claim。
关键输入与输出：
    可选输入为仓库根或文件路径；输出为版本字符串映射、完整 Git 对象名或小写 SHA-256。
依赖与副作用：
    使用 ``importlib.metadata``、``platform``、``subprocess git rev-parse`` 和文件读取；
    不访问网络、不修改 Git、不写文件。
重要约束：
    Git/包版本无法解析时 fail closed，不能写 ``unknown`` 占位；文件 hash 使用固定 1 MiB
    分块，对内容而非路径/mtime 建立身份。
"""

from __future__ import annotations

from importlib import metadata
from pathlib import Path

import hashlib
import platform
import subprocess


def collect_paper_dependency_versions() -> dict[str, str]:
    """返回 manifest 要求的 Python、PyTorch 和关键依赖版本。

    参数：
        无。
    返回：
        包含 Python、torch、NumPy、pandas、SciPy、scikit-learn 和 Pydantic 的版本映射。
    异常：
        任一发行包元数据不存在时传播 ``importlib.metadata.PackageNotFoundError``，不能用
        ``unknown`` 占位。
    副作用：
        只读当前解释器与已安装发行包元数据；不访问网络、不导入训练 runtime。
    """

    return {
        "python": platform.python_version(),
        "torch": metadata.version("torch"),
        "numpy": metadata.version("numpy"),
        "pandas": metadata.version("pandas"),
        "scipy": metadata.version("scipy"),
        "scikit-learn": metadata.version("scikit-learn"),
        "pydantic": metadata.version("pydantic"),
    }


def current_paper_git_commit(repo_root: str | Path | None = None) -> str:
    """只读当前完整 Git commit；失败时拒绝生成无身份 manifest。

    参数：
        repo_root: 可选仓库根；省略时使用当前模块向上三级的项目根。
    返回：
        40 位 SHA-1 或 64 位 SHA-256 小写完整对象名。
    异常：
        Git 命令失败、超时或输出不是完整对象名时抛出 ``RuntimeError``/
        ``subprocess.TimeoutExpired``。
    副作用：
        启动一次只读 ``git rev-parse HEAD`` 子进程；不修改工作树、索引或引用。
    """

    root = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[3]
    )
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    commit = completed.stdout.strip().lower()
    if completed.returncode != 0 or not _is_full_git_commit(commit):
        raise RuntimeError(f"Cannot resolve the Git commit below {root}.")
    return commit


def current_clean_paper_git_commit(repo_root: str | Path | None = None) -> str:
    """仅在工作树和索引完全干净时返回当前完整 Git commit。

    参数：
        repo_root: 可选仓库根；省略时使用项目根。
    返回：
        clean HEAD 的 40 位 SHA-1 或 64 位 SHA-256 小写对象名。
    异常：
        ``git status`` 失败、超时或发现已跟踪/暂存/未跟踪变化时抛出 ``RuntimeError``；
        HEAD 身份非法时传播 :func:`current_paper_git_commit` 的异常。
    副作用：
        启动只读 Git 子进程；不修改索引、工作树、引用或忽略规则。
    """

    root = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[3]
    )
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Cannot inspect the Git worktree below {root}.")
    if completed.stdout:
        raise RuntimeError(
            "Formal paper manifest requires a clean Git worktree and index; "
            f"the repository below {root} is dirty."
        )
    return current_paper_git_commit(root)


def sha256_file(path: str | Path) -> str:
    """流式计算文件小写 SHA-256。

    参数：
        path: 必须存在且可读的普通文件路径。
    返回：
        64 位小写 SHA-256。
    异常：
        文件缺失、不可读或读取中断时传播 ``OSError``。
    副作用：
        以 1 MiB 分块读取文件内容；不写文件、不缓存结果。
    """

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_full_git_commit(value: str) -> bool:
    """接受 Git SHA-1/SHA-256 的小写完整对象名。"""

    return (
        len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "collect_paper_dependency_versions",
    "current_clean_paper_git_commit",
    "current_paper_git_commit",
    "sha256_file",
]
