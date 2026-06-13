from __future__ import annotations

import torch
import pytest

from joff import CheckpointManager, DataModule, Trainer, build_model, build_optimizer
from joff.evaluation import ReconstructionEvaluator, RegressionEvaluator
from joff.models.attention import AttentionMaskFactory


def _roundtrip_state_dict(spec: dict, batch: torch.Tensor) -> None:
    model = build_model(spec)
    model.eval()
    output = model(batch)
    clone = build_model(spec)
    clone.load_state_dict(model.state_dict())
    clone.eval()
    clone_output = clone(batch)
    if isinstance(output, dict):
        key = "reconstruction" if "reconstruction" in output else "prediction"
        assert output[key].shape == clone_output[key].shape
    else:
        assert output.shape == clone_output.shape


def test_mlp_loss_and_state_dict_roundtrip() -> None:
    spec = {"type": "mlp", "input_dim": 4, "output_dim": 2, "hidden": [8], "loss": "mae"}
    model = build_model(spec)
    batch = (torch.randn(5, 4), torch.randn(5, 2))
    output = model(batch)
    loss = model.compute_loss(batch, output)
    assert loss["loss"].ndim == 0
    assert "mae" in loss["losses"]
    _roundtrip_state_dict(spec, torch.randn(3, 4))


def test_dae_loss_and_state_dict_roundtrip() -> None:
    spec = {
        "type": "dae",
        "input_dim": 6,
        "latent_dim": 2,
        "encoder_hidden": [4],
        "noise_std": 0.01,
    }
    model = build_model(spec)
    batch = torch.randn(5, 6)
    output = model(batch)
    loss = model.compute_loss(batch, output)
    assert output["reconstruction"].shape == (5, 6)
    assert output["latent"].shape == (5, 2)
    assert loss["loss"].ndim == 0
    assert "reconstruction" in loss["losses"]
    _roundtrip_state_dict(spec, torch.randn(3, 6))


def test_vae_forward_loss_and_state_dict_roundtrip() -> None:
    spec = {
        "type": "vae",
        "input_dim": 6,
        "latent_dim": 2,
        "encoder_hidden": [5],
        "kl_weight": 0.25,
    }
    model = build_model(spec)
    batch = torch.randn(4, 6)
    output = model(batch)
    assert output["reconstruction"].shape == (4, 6)
    assert output["mu"].shape == (4, 2)
    assert output["logvar"].shape == (4, 2)
    loss = model.compute_loss(batch, output)
    assert loss["loss"].ndim == 0
    assert {"reconstruction", "kl"} <= set(loss["losses"])
    _roundtrip_state_dict(spec, torch.randn(3, 6))


def test_nice_inverse_loss_and_state_dict_roundtrip() -> None:
    spec = {
        "type": "nice",
        "input_dim": 6,
        "hidden": [8, 8],
        "act": ["r", "r"],
        "coupling_layers": 3,
        "scaling_mode": "last",
    }
    model = build_model(spec)
    x = torch.randn(5, 6)
    output = model(x)
    assert output["z"].shape == (5, 6)
    assert output["log_det"].shape == (5,)
    assert torch.allclose(output["reconstruction"], x, atol=1e-5)
    loss = model.compute_loss(x, output)
    assert loss["loss"].ndim == 0
    assert {"prior", "log_det", "reconstruction"} <= set(loss["losses"])
    _roundtrip_state_dict(spec, torch.randn(3, 6))


def test_nkn_forward_loss_and_state_dict_roundtrip() -> None:
    spec = {
        "type": "nkn",
        "input_dim": 4,
        "output_dim": 2,
        "hidden": [8],
        "act": ["r"],
        "coupling_layers": 2,
        "koopman": {
            "second_order": True,
            "fm_rank": 2,
            "regularization_weight": 0.01,
            "regularization_norm": "l1",
            "nice_loss_weight": 0.1,
        },
    }
    model = build_model(spec)
    batch = (torch.randn(5, 4), torch.randn(5, 2))
    output = model(batch)
    assert output["prediction"].shape == (5, 2)
    assert output["latent"].shape == (5, 4)
    assert output["second_order"].shape == (5, 4)
    loss = model.compute_loss(batch, output)
    assert loss["loss"].ndim == 0
    assert {"prediction", "flow", "regularization"} <= set(loss["losses"])
    _roundtrip_state_dict(spec, torch.randn(3, 4))


def test_sequence_regressor_rnn_gru_lstm_smoke_and_state_dict() -> None:
    for model_type in ("rnn", "gru", "lstm"):
        spec = {
            "type": model_type,
            "input_dim": 3,
            "output_dim": 2,
            "hidden_size": 5,
            "num_layers": 1,
        }
        model = build_model(spec)
        batch = (torch.randn(4, 6, 3), torch.randn(4, 2))
        output = model(batch)
        assert output.shape == (4, 2)
        loss = model.compute_loss(batch, output)
        assert loss["loss"].ndim == 0
        _roundtrip_state_dict(spec, torch.randn(2, 6, 3))


def test_attention_masks_cover_supported_core_modes() -> None:
    temporal = AttentionMaskFactory("temporal").build(3)
    assert temporal is not None
    assert torch.isneginf(temporal[0, 1])
    assert temporal[1, 0] == 0

    lagged = AttentionMaskFactory("time_lagged", lag=1).build(4)
    assert lagged is not None
    assert torch.isneginf(lagged[3, 1])
    assert lagged[3, 2] == 0

    diagonal = AttentionMaskFactory("diagonal").build(3)
    assert diagonal is not None
    assert torch.isneginf(diagonal[1, 1])
    assert diagonal[1, 0] == 0

    topological = AttentionMaskFactory(
        "topological",
        topology=[[1.0, 0.0], [1.0, 1.0]],
    ).build(2)
    assert topological is not None
    assert torch.isneginf(topological[0, 1])
    assert topological[1, 0] == 0


def test_attention_forward_loss_and_state_dict_roundtrip() -> None:
    spec = {
        "type": "attention",
        "input_dim": 3,
        "output_dim": 2,
        "embed_dim": 4,
        "num_heads": 2,
        "attention_mask": "time_lagged",
        "attention_lag": 2,
    }
    model = build_model(spec)
    batch = (torch.randn(5, 6, 3), torch.randn(5, 2))
    output = model(batch)
    assert output["prediction"].shape == (5, 2)
    assert output["sequence"].shape == (5, 6, 4)
    assert output["attention_weights"].shape == (5, 2, 6, 6)
    loss = model.compute_loss(batch, output)
    assert loss["loss"].ndim == 0
    _roundtrip_state_dict(spec, torch.randn(3, 6, 3))


def test_gan_forward_loss_and_state_dict_roundtrip() -> None:
    spec = {
        "type": "gan",
        "input_dim": 4,
        "noise_dim": 3,
        "generator_hidden": [7],
        "discriminator_hidden": [6],
        "act": ["r"],
    }
    model = build_model(spec)
    batch = torch.randn(5, 4)
    output = model(batch)
    assert output["generated"].shape == (5, 4)
    assert output["fake_score"].shape == (5, 1)
    assert output["real_score"].shape == (5, 1)
    loss = model.compute_loss(batch, output)
    assert loss["loss"].ndim == 0
    assert {"generator", "discriminator"} <= set(loss["losses"])
    generated = model.generate(3)
    assert generated.shape == (3, 4)
    _roundtrip_state_dict(spec, torch.randn(3, 4))


def test_wgan_forward_loss_and_state_dict_roundtrip() -> None:
    spec = {
        "type": "wgan",
        "input_dim": 4,
        "latent_dim": 2,
        "hidden": [5],
        "act": ["r"],
    }
    model = build_model(spec)
    batch = torch.randn(6, 4)
    output = model(batch)
    assert output["generated"].shape == (6, 4)
    loss = model.compute_loss(batch, output)
    assert loss["loss"].ndim == 0
    assert torch.isfinite(loss["loss"])
    _roundtrip_state_dict(spec, torch.randn(3, 4))


def test_arx_forward_loss_and_state_dict_roundtrip() -> None:
    spec = {
        "type": "arx",
        "input_dim": 2,
        "output_dim": 1,
        "ar_order": 3,
    }
    model = build_model(spec)
    batch = (torch.randn(5, 3, 2), torch.randn(5, 1))
    output = model(batch)
    assert output["regressor"].shape == (5, 6)
    assert output["prediction"].shape == (5, 1)
    loss = model.compute_loss(batch, output)
    assert loss["loss"].ndim == 0
    _roundtrip_state_dict(spec, torch.randn(3, 3, 2))


def test_observer_forward_loss_and_state_dict_roundtrip() -> None:
    spec = {
        "type": "observer",
        "input_dim": 3,
        "output_dim": 2,
        "observer_state_dim": 4,
    }
    model = build_model(spec)
    batch = (torch.randn(5, 6, 3), torch.randn(5, 2))
    output = model(batch)
    assert output["prediction"].shape == (5, 2)
    assert output["sequence"].shape == (5, 6, 2)
    assert output["state"].shape == (5, 4)
    loss = model.compute_loss(batch, output)
    assert loss["loss"].ndim == 0
    _roundtrip_state_dict(spec, torch.randn(3, 6, 3))


def test_evaluators_return_regression_and_reconstruction_metrics() -> None:
    y_true = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y_pred = torch.tensor([[1.0, 1.0], [4.0, 4.0]])
    regression = RegressionEvaluator().evaluate(y_true, y_pred)
    assert {"MSE", "RMSE", "MAE", "R2"} <= set(regression.overall)
    assert len(regression.per_target) == 2
    reconstruction = ReconstructionEvaluator().evaluate(y_true, y_pred)
    assert "MaxAbs" in reconstruction.overall


def test_trainer_saves_last_and_best_checkpoints(tmp_path) -> None:
    x = torch.randn(24, 4)
    y = x[:, :2]
    data = DataModule.from_arrays(x[:18], y[:18], x[18:], y[18:], batch_size=6)
    model = build_model({"type": "mlp", "input_dim": 4, "output_dim": 2, "hidden": [8]})
    checkpoint_dir = tmp_path / "checkpoints"
    trainer = Trainer(
        max_epochs=2,
        device="cpu",
        seed=7,
        checkpoint_dir=checkpoint_dir,
        monitor="test/rmse",
    )
    result = trainer.fit(model, data)
    assert (checkpoint_dir / "last.pt").exists()
    assert (checkpoint_dir / "best.pt").exists()
    assert result.checkpoint_paths["last"] == checkpoint_dir / "last.pt"
    assert "test/rmse" in result.history[-1]
    assert "test/mse_loss" in result.history[-1]
    manager = CheckpointManager(checkpoint_dir)
    restored = build_model({"type": "mlp", "input_dim": 4, "output_dim": 2, "hidden": [8]})
    restored_optimizer = build_optimizer(restored, {"type": "adam", "lr": 1e-3})
    checkpoint = manager.load(
        checkpoint_dir / "best.pt",
        model=restored,
        optimizer=restored_optimizer,
    )
    assert checkpoint["model_class"] == "MLP"
    assert "model_state_dict" in checkpoint
    assert checkpoint["optimizer_state_dict"] is not None
    assert restored_optimizer.state_dict()["state"]
    assert checkpoint["config"]["type"] == "mlp"
    assert checkpoint["resolved_config"]["type"] == "mlp"
    assert {"python", "numpy", "torch"} <= set(checkpoint["rng_state"])
    assert "train/loss" in checkpoint["metrics"]
    last_checkpoint = manager.load_last()
    best_checkpoint = manager.load_best()
    assert last_checkpoint["model_class"] == "MLP"
    assert best_checkpoint["model_class"] == "MLP"


def test_trainer_evaluate_returns_metrics() -> None:
    x = torch.randn(20, 4)
    y = x[:, :2]
    data = DataModule.from_arrays(x[:16], y[:16], x[16:], y[16:], batch_size=4)
    model = build_model({"type": "mlp", "input_dim": 4, "output_dim": 2, "hidden": [6]})
    trainer = Trainer(max_epochs=1, device="cpu", seed=11)
    trainer.fit(model, data)
    metrics = trainer.evaluate(model, data)
    assert {"loss", "mse", "rmse", "mae", "r2"} <= set(metrics)


def test_model_compatibility_wrappers_delegate_to_trainer(tmp_path) -> None:
    x = torch.randn(18, 3)
    y = x[:, :1] * 0.25
    data = DataModule.from_arrays(x[:12], y[:12], x[12:], y[12:], batch_size=6)
    model = build_model({"type": "mlp", "input_dim": 3, "output_dim": 1, "hidden": [5]})
    checkpoint_dir = tmp_path / "compat_checkpoints"
    with pytest.warns(DeprecationWarning, match="model.fit is deprecated"):
        result = model.fit(data, max_epochs=1, device="cpu", checkpoint_dir=checkpoint_dir)
    assert result.history
    assert (checkpoint_dir / "last.pt").exists()
    clone = build_model({"type": "mlp", "input_dim": 3, "output_dim": 1, "hidden": [5]})
    with pytest.warns(DeprecationWarning, match="model.load is deprecated"):
        returned = clone.load(checkpoint_dir / "last.pt")
    assert returned is clone
    checkpoint = torch.load(checkpoint_dir / "last.pt", map_location="cpu", weights_only=False)
    for name, value in clone.state_dict().items():
        assert torch.allclose(value, checkpoint["model_state_dict"][name])
    with pytest.warns(DeprecationWarning, match="model.test is deprecated"):
        metrics = clone.test(data, device="cpu")
    assert metrics["loss"] >= 0
    with pytest.warns(DeprecationWarning, match="model.run is deprecated"):
        run_result = clone.run(data, max_epochs=1, device="cpu")
    assert run_result.history
