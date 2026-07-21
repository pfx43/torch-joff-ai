![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.1-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-alpha-orange)

这是一个面向仿真与过程工业数据的 spec-first PyTorch 实验工具包。它把严格配置、可复用模型构建器、训练集域内数据流水线、实验编排、故障诊断评估和论文级绘图能力收束到一个安静、可复现实验的 Python 包里。

这个项目适合需要在 TE、CSTR、TTS、NE、多相流、WPT 等过程数据集上做可重复实验的研究者和工程师，同时明确把非 OA/私有工业数据排除在公开发布之外。

## 亮点

- **Spec-first 实验**：Pydantic 配置、显式注册表、确定性随机种子和可复现 artifact store。
- **过程数据流水线**：缺失值处理、仅基于训练集的归一化、异常值规则、顺序/分层/分组划分、动态窗口和 MPC 窗口。
- **模型组件**：MLP、DAE、VAE、NICE、NKN、RNN、Attention、GAN/WGAN 风格组件、ARX、Observer 和序列回归模型。
- **评估体系**：回归、分类、重构、Koopman 贡献分析，以及 `re-T2-kde`、`re-Q-ineq`、`lv-T2-pdf` 等故障诊断过程。
- **实验工作流**：runner、网格/耦合 sweep、重复实验、checkpoint、best trial 导出和 CI 友好的 smoke 路径。
- **科研绘图**：预测、数据、故障诊断、流模型、Koopman 和训练曲线绘图，支持 PDF/SVG/PNG 输出。

## 安装

```bash
python -m pip install -e .
```

常用可选依赖：

```bash
python -m pip install -e ".[excel,hdf5,paper,tracking,hpo,dev]"
```

## 快速开始

```python
from joff import DataModule, DataPipeline, build_model

pipeline = DataPipeline.from_config([
    {"split": {"type": "sequential", "test_ratio": 0.25}},
    {"scaler": {"method": "standard"}},
])

data = DataModule.from_preset(
    "cstr_fd",
    root="CSTR",
    task="fd",
    pipeline=pipeline,
    batch_size=32,
)

x, y = next(iter(data.loader("train")))
model = build_model({
    "type": "mlp",
    "input_dim": int(x.shape[-1]),
    "output_dim": int(y.shape[-1]),
    "hidden": ["*2", "/2"],
    "act": ["relu", "sigmoid"],
})
```

运行快速示例：

```bash
python examples/quickstart_dae.py
python examples/hm_nkn.py --smoke
python examples/fd_cstr.py --smoke
python examples/sweep_runner.py --smoke
python examples/repeat_study.py --smoke
```

## 数据集边界

本仓库按公开发布要求设置了严格的数据边界：

- OA 原始数据可以放在 `datasets/raw/oa/**`。
- OA 数据集卡片放在 `datasets/cards/oa/**`。
- 公开版 manifest 为 `datasets/manifest.public.yaml`。
- 非 OA/private 原始数据、private 数据集卡片和 private 示例脚本已被忽略，不能推送到 GitHub。

当前公开 OA presets：

| Preset | 任务 | 原始数据目录 |
| --- | --- | --- |
| `cstr_fault_diagnosis` | 故障诊断 | `datasets/raw/oa/CSTR` |
| `cstr_closed_loop_fd` | 故障诊断 | `datasets/raw/oa/CSTR` |
| `te_fault_diagnosis` | 故障诊断 | `datasets/raw/oa/TE` |
| `te_classification` | 分类 | `datasets/raw/oa/TE` |
| `tts_fault_diagnosis` | 故障诊断 | `datasets/raw/oa/TTS` |
| `tts_sui_fault_estimation` | 重构 | `datasets/raw/oa/TTS` |
| `ne_fault_estimation` | 重构 | `datasets/raw/oa/NE` |
| `multiphase_fd` | 故障诊断 | `datasets/raw/oa/Multiphase_Flow_Facility` |
| `wpt_mpc` | MPC | `datasets/raw/oa/WPT` |

私有工业数据仍可在本地通过 adapter 使用，但需要用户自行提供本地 root，这些文件不会进入公开仓库。

### 短名写法

`DataModule.from_preset(...)` 支持常用数据集的 preset、task 和 root 短名：

```python
data = DataModule.from_preset("cstr_fd", root="CSTR", task="fd")
te = DataModule.from_preset("te_cls", root="TE", task="cls")
wpt = DataModule.from_preset("wpt", root="WPT", task="mpc")
```

`root="CSTR"` 默认解析为 `datasets/raw/oa/CSTR`。private 数据必须显式标记，例如 `root="*HY"` 或 `root="private:HY"`。

## 默认配置

项目没有单独的默认 YAML 文件。运行时默认值由 `src/joff/core/defaults.py` 中的 `DefaultRegistry` 注册；配置 schema 与校验在 `src/joff/core/config.py`；`src/joff/core/resolver.py` 负责把 package/model defaults 与用户 YAML、API 参数、方法覆盖和 CLI 覆盖合并，并记录 provenance。

`configs/example.yaml` 只是可编辑示例配置，不是权威默认配置源。

## 项目结构

```text
src/joff/              核心包
examples/              smoke 与 quickstart 示例
tests/                 单元测试与集成测试
configs/               示例实验配置
datasets/cards/oa/     公开数据集卡片
datasets/raw/oa/       OA 原始数据
datasets/manifest.public.yaml
```

## 开发

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
```

`joff` 在 import 时保持安静：不会读取数据、创建运行目录、修改 Matplotlib 全局状态或启动 tracking。

## 路线图

- 扩展 dataset-card 校验和 license 元数据。
- 补充故障诊断、重构、MPC 与质量预测 benchmark 配方。
- 发布基于 OA presets 的可复现实验表格。
- 改进 MLflow、TensorBoard、W&B、Optuna 和 Hydra 等可选集成。

## 引用

如果 Joff 对你的研究有帮助，目前可以先引用本仓库：

```bibtex
@software{joff2026,
  title = {Joff: A Spec-First PyTorch Experiment Toolkit for Process Data},
  author = {Joff contributors},
  year = {2026},
  url = {https://github.com/zhuofupan/torch-joff-ai}
}
```

## 许可证

Joff 使用 MIT License。数据集许可证由各自的 dataset card 单独记录，重新分发前请进一步确认。
