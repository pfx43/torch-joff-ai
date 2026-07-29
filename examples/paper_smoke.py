"""运行论文 P10 synthetic contract CPU smoke。

文件用途：
    提供 ``configs/paper/smoke.yaml`` 的公开命令行入口，方便用户在不读取真实 CSTR 故障
    数据的前提下验证五段正常协议、冻结 manifest、一次性评价和机器来源产物。
主要职责：
    解析 ``--config`` 和可选 ``--run-root``，调用 ``run_paper_smoke``，并把 evaluation ID、
    逐时刻行数和 receipt 路径打印为 JSON；不在脚本中复制模型或协议实现。
关键输入与输出：
    输入为严格 YAML 和可选输出根；输出位于 ``<run-root>/paper_smoke``，stdout 只含简短
    JSON 摘要。
依赖与副作用：
    依赖已安装或 ``PYTHONPATH=src`` 的 Joff；创建 smoke 运行目录和 claim，不访问网络或
    真实 MAT 文件。
重要约束：
    本脚本只接受 smoke 配置。生成数值、Uncertified 隔离结果和 checkpoint 均为合成合同
    证据，不能作为论文实验结果、确定性认证或许可证明。
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import argparse
import json

from joff.experiments import resolve_frozen_evaluation_config, run_paper_smoke


ROOT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    """解析参数并执行一次新的 synthetic paper smoke。

    参数：
        argv: 可选 ``--config`` 与 ``--run-root`` 参数；省略时读取进程参数。
    返回：
        成功固定返回 0，并向 stdout 写一行不含论文性能声明的 JSON 摘要。
    异常：
        配置、manifest、claim、合成 evaluator 或 I/O 错误按原类型传播。
    副作用：
        在 run root 下创建 synthetic 正常协议、manifest、claim 与评价产物；不读取真实 MAT。
    """

    parser = argparse.ArgumentParser(description="Run the P10 synthetic paper contract smoke.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "paper" / "smoke.yaml",
    )
    parser.add_argument("--run-root", type=Path, default=None)
    args = parser.parse_args(argv)
    resolved = resolve_frozen_evaluation_config(args.config)
    if args.run_root is not None:
        run_root = args.run_root.expanduser().resolve()
        updated = resolved.config.model_copy(
            update={
                "artifact_root": run_root,
                "claim_registry": run_root / ".frozen-evaluation-claims",
            }
        )
        resolved = resolve_frozen_evaluation_config(updated)
    result = run_paper_smoke(resolved)
    print(
        json.dumps(
            {
                "evaluation_id": result.evaluation_id,
                "pointwise_row_count": result.pointwise_row_count,
                "receipt": str(result.receipt_path),
                "claims": {
                    "paper_method_implemented": False,
                    "formal_fault_results": False,
                    "certified": False,
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
