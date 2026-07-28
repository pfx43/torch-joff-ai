"""P5 堆叠受保护潜变量残差及严格窗口边界校验。

文件用途：
    把连续 monitor 输出中的 data-branch 潜变量与 protected-branch 潜变量配对，形成论文
    P5/P6 共用的堆叠受保护残差，同时阻止窗口跨越任何会改变时间语义的边界。
主要职责：
    定义不可变 ``StackedProtectedResidual``，从公开 ``MonitorOutput`` 序列计算
    ``z^E_k - zhat_{k|s}``；校验原始索引连续、episode/stage 相同且全窗使用同一锚点。
    本文件不计算 P6 响应矩阵、算子认证、检测统计量或隔离分数。
关键输入与输出：
    输入是一个或多个已有 data/protected latent 的连续 monitor 输出；输出包含二维残差、
    展平向量、窗口原始索引、协议边界和锚点索引的可序列化对象。
依赖与副作用：
    只依赖 P5 ``MonitorOutput`` 和 Python 标准库；不读写文件、不访问网络、不修改输入。
重要约束：
    缺失任一支路、潜变量宽度不一致、索引有 gap、跨 episode/五阶段边界或中途换锚时
    必须 fail closed，不能静默截断或拼接窗口。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .protected_reference import MonitorOutput, MonitorStage


@dataclass(frozen=True)
class StackedProtectedResidual:
    """同一锚点和协议边界内的堆叠潜变量残差。

    参数：
        raw_indices: 窗口内连续原始索引。
        episode_id/stage: 全窗口共享的 episode 和五阶段名称。
        anchor_raw_index: 全窗口共享的已接受锚点。
        residuals: 每个时刻的 ``z^E_k - zhat_{k|s}``，形状语义为 ``[L,m_z]``。
    返回：
        不可变、可序列化的窗口对象；``vector`` 按时间优先展平为 ``L*m_z`` 元组。
    异常：
        应通过 ``from_outputs`` 构造，由该方法对所有时间边界和 shape 执行 fail-closed
        校验。
    副作用：
        无。
    """

    raw_indices: tuple[int, ...]
    episode_id: str
    stage: MonitorStage
    anchor_raw_index: int
    residuals: tuple[tuple[float, ...], ...]

    @classmethod
    def from_outputs(
        cls,
        outputs: Sequence[MonitorOutput],
    ) -> "StackedProtectedResidual":
        """从连续公开 monitor 输出构建受保护残差窗口。

        参数：
            outputs: 时间升序的非空输出序列，每项都必须同时具有 data latent 和
                protected rollout。
        返回：
            通过边界校验的 ``StackedProtectedResidual``。
        异常：
            输入为空、支路不可用、潜变量宽度不匹配、索引不连续、episode/stage 不同或
            anchor 改变时抛出 ``ValueError``。
        副作用：
            无；不修改输出序列。
        """

        window = tuple(outputs)
        if not window:
            raise ValueError("Stacked protected residual requires at least one output.")

        first = window[0]
        first_rollout = first.protected_rollout
        if first.data_latent is None or first_rollout is None:
            raise ValueError(
                "Every stacked residual output requires data and protected latent branches."
            )
        anchor_raw_index = first_rollout.anchor_raw_index
        latent_width = len(first.data_latent)
        if latent_width == 0 or len(first_rollout.latent) != latent_width:
            raise ValueError("Data and protected latent widths must match and be non-empty.")

        rows: list[tuple[float, ...]] = []
        previous_raw_index: int | None = None
        for output in window:
            rollout = output.protected_rollout
            if output.data_latent is None or rollout is None:
                raise ValueError(
                    "Every stacked residual output requires data and protected latent branches."
                )
            if output.episode_id != first.episode_id:
                raise ValueError("Stacked residual window cannot cross an episode boundary.")
            if output.stage != first.stage:
                raise ValueError("Stacked residual window cannot cross a five-stage boundary.")
            if previous_raw_index is not None and output.raw_index != previous_raw_index + 1:
                raise ValueError("Stacked residual raw indices must be consecutive.")
            if rollout.target_raw_index != output.raw_index:
                raise ValueError(
                    "Protected rollout target index must match its monitor output."
                )
            if rollout.anchor_raw_index != anchor_raw_index:
                raise ValueError("Stacked residual window cannot cross a re-anchor event.")
            if (
                len(output.data_latent) != latent_width
                or len(rollout.latent) != latent_width
            ):
                raise ValueError("Stacked residual latent width must be constant.")
            rows.append(
                tuple(
                    data_value - protected_value
                    for data_value, protected_value in zip(
                        output.data_latent,
                        rollout.latent,
                        strict=True,
                    )
                )
            )
            previous_raw_index = output.raw_index

        return cls(
            raw_indices=tuple(output.raw_index for output in window),
            episode_id=first.episode_id,
            stage=first.stage,
            anchor_raw_index=anchor_raw_index,
            residuals=tuple(rows),
        )

    @property
    def vector(self) -> tuple[float, ...]:
        """按时间优先顺序展平 ``[L,m_z]`` 残差。"""

        return tuple(value for row in self.residuals for value in row)

    def to_dict(self) -> dict[str, Any]:
        """返回 JSON 兼容的窗口、边界与残差表示。"""

        return {
            "raw_indices": list(self.raw_indices),
            "episode_id": self.episode_id,
            "stage": self.stage.value,
            "anchor_raw_index": self.anchor_raw_index,
            "residuals": [list(row) for row in self.residuals],
            "vector": list(self.vector),
        }
