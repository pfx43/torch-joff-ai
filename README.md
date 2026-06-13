# Joff

[简体中文](README.zh-CN.md) | English

![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.1-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-alpha-orange)

Joff is a spec-first PyTorch experiment toolkit for simulation and process data. It brings together strict configuration, reusable model builders, train-scoped data pipelines, experiment orchestration, fault-diagnosis evaluation, and publication-friendly plotting in one quiet Python package.

The project is designed for researchers and engineers who need repeatable experiments across process datasets such as TE, CSTR, TTS, NE, multiphase flow, and WPT, while keeping private industrial datasets out of public releases.

## Highlights

- **Spec-first experiments**: Pydantic-backed configs, explicit registries, deterministic seeds, and reproducible artifact stores.
- **Process-data pipelines**: missing-value handling, train-only scaling, outlier rules, sequential/stratified/group splits, dynamic windows, and MPC windows.
- **Model zoo**: MLP, DAE, VAE, NICE, NKN, RNN, Attention, GAN/WGAN-style components, ARX, Observer, and sequence regressors.
- **Evaluation batteries**: regression, classification, reconstruction, Koopman contribution analysis, and fault-detection procedures such as `re-T2-kde`, `re-Q-ineq`, and `lv-T2-pdf`.
- **Experiment workflows**: reusable runners, grid/coupled sweeps, repeated studies, checkpointing, best-trial exports, and CI-friendly smoke paths.
- **Research plots**: prediction, data, fault-detection, flow, Koopman, and training plotters with PDF/SVG/PNG output.

## Installation

```bash
python -m pip install -e .
```

Optional extras are available for common research workflows:

```bash
python -m pip install -e ".[excel,hdf5,paper,tracking,hpo,dev]"
```

## Quick Start

```python
from joff import DataModule, DataPipeline, build_model

pipeline = DataPipeline.from_config([
    {"split": {"type": "sequential", "test_ratio": 0.25}},
    {"scaler": {"method": "standard"}},
])

data = DataModule.from_preset(
    "cstr_fault_diagnosis",
    task="fault_diagnosis",
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

Run the fast examples:

```bash
python examples/quickstart_dae.py
python examples/hm_nkn.py --smoke
python examples/fd_cstr.py --smoke
python examples/sweep_runner.py --smoke
python examples/repeat_study.py --smoke
```

## Datasets

This repository is prepared for public release with a strict data boundary:

- Open-access data may live under `datasets/raw/oa/**`.
- Open-access dataset cards live under `datasets/cards/oa/**`.
- The public manifest is `datasets/manifest.public.yaml`.
- Non-OA/private raw data, private dataset cards, and private example scripts are intentionally ignored and must not be pushed.

The included OA presets cover:

| Preset | Task | Raw root |
| --- | --- | --- |
| `cstr_fault_diagnosis` | fault diagnosis | `datasets/raw/oa/CSTR` |
| `cstr_closed_loop_fd` | fault diagnosis | `datasets/raw/oa/CSTR` |
| `te_fault_diagnosis` | fault diagnosis | `datasets/raw/oa/TE` |
| `te_classification` | classification | `datasets/raw/oa/TE` |
| `tts_fault_diagnosis` | fault diagnosis | `datasets/raw/oa/TTS` |
| `tts_sui_fault_estimation` | reconstruction | `datasets/raw/oa/TTS` |
| `ne_fault_estimation` | reconstruction | `datasets/raw/oa/NE` |
| `multiphase_fd` | fault diagnosis | `datasets/raw/oa/Multiphase_Flow_Facility` |
| `wpt_mpc` | MPC | `datasets/raw/oa/WPT` |

Private industrial datasets can still be used locally through adapters when the user supplies a local root, but those files are outside the public repository.

## Project Layout

```text
src/joff/              Core package
examples/              Smoke and quickstart scripts
tests/                 Unit and integration tests
configs/               Example experiment config
datasets/cards/oa/     Public dataset cards
datasets/raw/oa/       Open-access raw datasets
datasets/manifest.public.yaml
```

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
```

The package is intentionally quiet on import: importing `joff` does not read data, create run directories, modify Matplotlib state, or start trackers.

## Roadmap

- Expand dataset-card validation and license metadata.
- Add more benchmark recipes for fault diagnosis, reconstruction, MPC, and quality prediction.
- Publish reproducible experiment tables for the bundled OA presets.
- Improve optional integrations for MLflow, TensorBoard, W&B, Optuna, and Hydra.

## Citation

If Joff helps your research, please cite the repository for now:

```bibtex
@software{joff2026,
  title = {Joff: A Spec-First PyTorch Experiment Toolkit for Process Data},
  author = {Joff contributors},
  year = {2026},
  url = {https://github.com/zhuofupan/torch-joff-ai}
}
```

## License

Joff is released under the MIT License. Dataset licenses are tracked separately in dataset cards and should be verified before redistribution.
