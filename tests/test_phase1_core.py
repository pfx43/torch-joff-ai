from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path

import pytest
import torch
from torch import nn

from joff import DataModule, Experiment, Trainer, build_model
from joff.core import ConfigManager, ConfigResolver
from joff.core.errors import ConfigError
from joff.layers import build_activation, build_mlp, dropout_rate_for_width, resolve_widths


def test_resolve_widths_supports_relative_expressions_and_chinese_symbols() -> None:
    assert resolve_widths(128, 16, ["/2", "/2", "o*2"]) == [64, 32, 32]
    assert resolve_widths(10, None, ["＊10", "／2", "prev＋5"]) == [100, 50, 55]
    assert resolve_widths(20, 5, "i/2，o*3") == [10, 15]


def test_activation_aliases_build_expected_modules() -> None:
    x = torch.tensor([0.0, 1.0])
    gaussian = build_activation("g")
    assert torch.allclose(gaussian(x), torch.exp(-(x**2)))
    assert isinstance(build_activation("r"), nn.ReLU)
    assert isinstance(build_activation("a"), nn.Identity)
    assert isinstance(build_activation("swish"), nn.SiLU)


def test_auto_dropout_policy_and_builder() -> None:
    assert dropout_rate_for_width(100, threshold=100) == 0.0
    assert dropout_rate_for_width(1000, threshold=100, scale=100.0) == pytest.approx(0.1)
    net = build_mlp(4, 1, [50, 1000], act=["r", "r"], dropout="auto")
    dropouts = [module for module in net if isinstance(module, nn.Dropout)]
    assert len(dropouts) == 1
    assert dropouts[0].p == pytest.approx(0.1)


def test_mlp_forward_from_builder_and_model_factory() -> None:
    net = build_mlp(20, 3, [16, 8], act=["r", "s"])
    assert net(torch.randn(4, 20)).shape == (4, 3)

    model = build_model(
        {"type": "mlp", "input_dim": 20, "output_dim": 3, "hidden": [16, 8], "act": ["r", "s"]}
    )
    assert model(torch.randn(4, 20)).shape == (4, 3)


def test_legacy_dae_dsl_builds_without_eval_exec() -> None:
    model = build_model({"type": "dae", "struct": [10, "*10", "/2"], "act": ["g", "s", "a"]})
    output = model(torch.randn(2, 10))
    assert output["reconstruction"].shape == (2, 10)
    assert output["latent"].shape == (2, 50)


def test_unknown_config_key_raises_helpful_error() -> None:
    with pytest.raises(ConfigError) as excinfo:
        ConfigManager().validate({"trianer": {"max_epochs": 1}})
    message = str(excinfo.value)
    assert "Unknown config key 'trianer'" in message
    assert "trainer" in message
    assert "Legal top-level keys" in message


def test_config_precedence_and_provenance_smoke() -> None:
    resolved = ConfigResolver().resolve(
        (
            "user_config",
            {
                "model": {"type": "mlp", "input_dim": 4, "output_dim": 2, "hidden": [8]},
                "trainer": {"max_epochs": 5},
            },
        ),
        ("study_trial", {"trainer": {"optimizer": {"lr": 0.01}}}),
        ("api_kwargs", {"trainer": {"max_epochs": 7}}),
    )
    assert resolved.config.trainer.max_epochs == 7
    assert resolved.config.trainer.optimizer.lr == pytest.approx(0.01)
    provenance = resolved.provenance.to_dict()
    assert [entry["source"] for entry in provenance["trainer.max_epochs"]][-2:] == [
        "user_config",
        "api_kwargs",
    ]
    assert provenance["trainer.optimizer.lr"][-1]["source"] == "study_trial"


def test_config_legacy_aliases_hash_and_frozen_resolved_config() -> None:
    manager = ConfigManager()
    base = {
        "model": {"type": "mlp", "input_dim": 4, "output_dim": 2, "hidden": [8]},
        "trainer": {"max_epochs": 2},
    }
    resolved = manager.resolve(base, method_overrides={"lr": 0.02, "e": 3, "b": 6})
    same = manager.resolve(base, method_overrides={"lr": 0.02, "e": 3, "b": 6})

    assert resolved.config.trainer.optimizer.lr == pytest.approx(0.02)
    assert resolved.config.trainer.max_epochs == 3
    assert resolved.config.trainer.batch_size == 6
    assert resolved.config_hash == same.config_hash
    with pytest.raises(Exception):
        resolved.config.trainer.max_epochs = 5  # type: ignore[misc]

    mpc = manager.validate({"data": {"mpc_window": {"past_horizon": 2}}})
    assert mpc.data.mpc_window == {"past_horizon": 2}


def test_public_api_smoke_and_import_is_quiet(tmp_path) -> None:
    env = os.environ.copy()
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-c", "import joff"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.stdout == ""
    assert completed.stderr == ""

    x = torch.randn(16, 4)
    y = torch.randn(16, 2)
    data = DataModule.from_arrays(x[:12], y[:12], x[12:], y[12:], batch_size=4)
    model = build_model({"type": "mlp", "input_dim": 4, "output_dim": 2, "hidden": [8]})
    trainer = Trainer(max_epochs=1, device="cpu", seed=42)
    result = trainer.fit(model, data)
    metrics = trainer.evaluate(model, data)
    assert len(result.history) == 1
    assert metrics["loss"] >= 0

    exp = Experiment.from_config(
        {
            "model": {"type": "mlp", "input_dim": 4, "output_dim": 2, "hidden": [8]},
            "artifacts": {"root": tmp_path, "name": "phase1"},
        }
    )
    exp_result = exp.run()
    assert (exp_result.run_dir / "resolved_config.yaml").exists()
    assert (exp_result.run_dir / "provenance.json").exists()
