from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Subset

from joff import DataModule, DatasetCardAdapter, Trainer, build_model
from joff.artifacts import ArtifactStore


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "datasets" / "raw"
CARD_ROOT = ROOT / "datasets" / "cards"


OA_CASES = [
    ("te_fault_diagnosis", RAW_ROOT / "oa" / "TE", "fault_diagnosis"),
    ("te_classification", RAW_ROOT / "oa" / "TE", "classification"),
    ("cstr_fault_diagnosis", RAW_ROOT / "oa" / "CSTR", "fault_diagnosis"),
    ("cstr_closed_loop_fd", RAW_ROOT / "oa" / "CSTR", "fault_diagnosis"),
    ("tts_fault_diagnosis", RAW_ROOT / "oa" / "TTS", "fault_diagnosis"),
    ("tts_sui_fault_estimation", RAW_ROOT / "oa" / "TTS", "reconstruction"),
    ("ne_fault_estimation", RAW_ROOT / "oa" / "NE", "reconstruction"),
    ("multiphase_fd", RAW_ROOT / "oa" / "Multiphase_Flow_Facility", "fault_diagnosis"),
    ("wpt_mpc", RAW_ROOT / "oa" / "WPT", "mpc"),
]

PRIVATE_CASES = [
    ("hy_fault_diagnosis", RAW_ROOT / "private" / "HY", "fault_diagnosis"),
    ("hy_quality_prediction", RAW_ROOT / "private" / "HY_PRD", "prediction"),
]

CASES = OA_CASES + PRIVATE_CASES


def _require_real_data() -> None:
    if not RAW_ROOT.exists():
        pytest.skip("Real legacy datasets are not available in this checkout.")


def _require_dataset_root(root: Path) -> None:
    _require_real_data()
    if not root.exists():
        pytest.skip(f"Dataset root is not available in this checkout: {root}")


def test_real_dataset_cards_parse_and_carry_access_metadata() -> None:
    _require_real_data()
    cards = sorted(CARD_ROOT.rglob("dataset_card.yaml"))
    assert cards
    for path in cards:
        adapter = DatasetCardAdapter.from_yaml(path)
        summary = adapter.summary()
        expected_tag = "private" if "private" in path.parts else "oa"
        assert summary["access"]["tag"] == expected_tag


def test_real_dataset_short_root_and_task_aliases_load_default_oa_data() -> None:
    _require_dataset_root(RAW_ROOT / "oa" / "CSTR")
    data = DataModule.from_preset("cstr_fd", root="CSTR", task="fd", batch_size=8)
    assert data.summaries["preset_summary"]["name"] == "cstr_fault_diagnosis"
    assert data.summaries["task_summary"]["name"] == "fault_diagnosis"
    assert data.summaries["source_summary"]["access"]["tag"] == "oa"
    assert Path(data.summaries["source_summary"]["root"]).parts[-5:] == (
        "datasets",
        "raw",
        "oa",
        "CSTR",
        "fd",
    )


def test_private_root_alias_requires_explicit_marker() -> None:
    _require_dataset_root(RAW_ROOT / "private" / "HY")
    data = DataModule.from_preset("hy_fd", root="*HY", task="fd", batch_size=8)
    assert data.summaries["preset_summary"]["name"] == "hy_fault_diagnosis"
    assert data.summaries["source_summary"]["access"]["tag"] == "private"


@pytest.mark.parametrize(("preset", "root", "task"), CASES)
def test_real_dataset_presets_smoke_load_and_save_summaries(
    preset: str,
    root: Path,
    task: str,
    tmp_path: Path,
) -> None:
    _require_dataset_root(root)
    data = DataModule.from_preset(preset, root=root, task=task, batch_size=8)
    train_batch = next(iter(data.loader("train")))
    test_batch = next(iter(data.loader("test")))
    assert len(data.train_dataset) > 0
    assert len(data.test_dataset) > 0
    assert _batch_size(train_batch) > 0
    assert _batch_size(test_batch) > 0
    assert data.summaries["source_summary"]["source_type"] == "real_dataset"
    assert "access" in data.summaries["source_summary"]

    paths = data.save_summaries(ArtifactStore(tmp_path, preset))
    assert paths["prepared_dataset_hash"].exists()
    assert paths["source_summary"].exists()


@pytest.mark.parametrize(("preset", "root", "task"), CASES)
def test_real_dataset_presets_minimal_training_smoke(preset: str, root: Path, task: str) -> None:
    _require_dataset_root(root)
    data = DataModule.from_preset(preset, root=root, task=task, batch_size=16)
    small = DataModule(
        Subset(data.train_dataset, range(min(64, len(data.train_dataset)))),
        Subset(data.test_dataset, range(min(32, len(data.test_dataset)))),
        batch_size=16,
        shuffle=False,
        summaries=data.summaries,
    )
    batch = next(iter(small.loader("train")))
    model = _model_for_batch(batch)
    result = Trainer(max_epochs=1, device="cpu", seed=7).fit(model, small)
    assert torch.isfinite(torch.tensor(result.history[-1]["train/loss"]))


def _model_for_batch(batch: Any) -> nn.Module:
    if isinstance(batch, dict):
        return _MPCSmokeModel(
            past_shape=tuple(batch["past"].shape[1:]),
            target_shape=tuple(batch["target_future"].shape[1:]),
        )
    x, y = batch
    return build_model(
        {
            "type": "mlp",
            "input_dim": int(x.shape[-1]),
            "output_dim": int(y.shape[-1]),
            "hidden": [8],
            "act": "relu",
        }
    )


class _MPCSmokeModel(nn.Module):
    def __init__(self, *, past_shape: tuple[int, ...], target_shape: tuple[int, ...]) -> None:
        super().__init__()
        self.target_shape = target_shape
        input_dim = 1
        for value in past_shape:
            input_dim *= int(value)
        output_dim = 1
        for value in target_shape:
            output_dim *= int(value)
        self.net = nn.Linear(input_dim, output_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.net(batch["past"].reshape(batch["past"].shape[0], -1))
        return output.reshape(batch["past"].shape[0], *self.target_shape)

    def compute_loss(self, batch: dict[str, torch.Tensor], output: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(output, batch["target_future"])


def _batch_size(batch: Any) -> int:
    if isinstance(batch, dict):
        return int(next(iter(batch.values())).shape[0])
    if isinstance(batch, (tuple, list)):
        return int(batch[0].shape[0])
    return int(batch.shape[0])
