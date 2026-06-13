from __future__ import annotations

import builtins
import io
from pathlib import Path
import pandas as pd
import pytest
import torch
from matplotlib import pyplot as plt

from joff import (
    BestResultMonitor,
    ConfigTable,
    DataModule,
    FaultDetectionEvaluator,
    HistoryCallback,
    JoffConsole,
    KoopmanContributionEvaluator,
    MetricTable,
    Study,
    Trainer,
    build_evaluator,
    build_model,
    build_optimizer,
)
from joff.artifacts import ArtifactStore
from joff.evaluation import reconstruction_scores
from joff.plotting import (
    DataPlotter,
    FaultDetectionPlotter,
    FigureSpec,
    FlowPlotter,
    KoopmanPlotter,
    Palette,
    PaletteRegistry,
    PredictionPlotter,
    PlotTheme,
    TrainingPlotter,
)
from joff.tracking import LocalTracker, RunInfo
from joff.tracking.optional import MLflowTracker, TensorBoardTracker, WandbTracker
from joff.experiments.study import _expand_repeats, _expand_trials, _leaderboard


@pytest.fixture(autouse=True)
def _close_figures_after_test() -> None:
    yield
    plt.close("all")


def test_optimizer_factory_and_history_callback() -> None:
    model = build_model({"type": "mlp", "input_dim": 3, "output_dim": 1, "hidden": [4]})
    optimizer = build_optimizer(model, {"type": "adamw", "lr": 0.01, "weight_decay": 0.001})
    assert optimizer.__class__.__name__ == "AdamW"
    x = torch.randn(12, 3)
    y = torch.randn(12, 1)
    data = DataModule.from_arrays(x[:8], y[:8], x[8:], y[8:], batch_size=4)
    callback = HistoryCallback()
    trainer = Trainer(max_epochs=2, device="cpu", callbacks=[callback])
    result = trainer.fit(model, data)
    assert len(result.history) == 2
    assert len(callback.history) == 2


def test_optimizer_param_groups_and_bias_weight_decay_split() -> None:
    model = build_model({"type": "mlp", "input_dim": 3, "output_dim": 1, "hidden": [4]})
    optimizer = build_optimizer(
        model,
        {
            "type": "adamw",
            "lr": 0.01,
            "weight_decay": 0.1,
            "exclude_bias_from_weight_decay": True,
            "param_groups": [{"match": "net.0.weight", "lr": 0.001, "weight_decay": 0.0}],
        },
    )
    first_weight = model.net[0].weight
    first_bias = model.net[0].bias
    weight_group = next(
        group for group in optimizer.param_groups if any(parameter is first_weight for parameter in group["params"])
    )
    bias_group = next(
        group for group in optimizer.param_groups if any(parameter is first_bias for parameter in group["params"])
    )
    assert weight_group["lr"] == pytest.approx(0.001)
    assert weight_group["weight_decay"] == pytest.approx(0.0)
    assert bias_group["weight_decay"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="did not match any parameters"):
        build_optimizer(model, {"param_groups": [{"match": "missing.*", "lr": 0.1}]})


def test_fault_detection_evaluator_and_reconstruction_scores() -> None:
    normal_scores = torch.tensor([0.1, 0.2, 0.15, 0.18, 0.12])
    test_scores = torch.tensor([0.1, 0.25, 2.0, 3.0])
    labels = torch.tensor([0, 0, 1, 1])
    report = FaultDetectionEvaluator(expected_far=0.2).fit_evaluate(
        normal_scores,
        test_scores,
        labels,
    )
    assert report.threshold > 0
    assert report.metrics["FDR"] == pytest.approx(1.0)
    scores = reconstruction_scores(torch.zeros(2, 3), torch.ones(2, 3))
    assert scores.tolist() == pytest.approx([1.0, 1.0])


def test_fault_detection_evaluator_supports_re_lv_procedures() -> None:
    normal_true = torch.zeros(8, 2)
    normal_pred = normal_true + 0.05
    normal_latent = torch.randn(8, 2) * 0.1
    test_true = torch.cat([torch.zeros(4, 2), torch.ones(4, 2) * 2.0], dim=0)
    test_pred = torch.cat([torch.zeros(4, 2) + 0.05, torch.zeros(4, 2)], dim=0)
    test_latent = torch.cat([torch.zeros(4, 2), torch.ones(4, 2) * 3.0], dim=0)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    evaluator = FaultDetectionEvaluator(
        expected_far=0.05,
        procedures=["re-T2-kde", "re-Q-ineq", "lv-T2-pdf"],
    )
    evaluator.fit({"x_true": normal_true, "x_pred": normal_pred, "latent": normal_latent})
    report = evaluator.evaluate(
        {"x_true": test_true, "x_pred": test_pred, "latent": test_latent},
        labels,
    )
    assert {"re-T2-kde", "re-Q-ineq", "lv-T2-pdf"} <= set(report.thresholds)
    assert {"AFAR", "AMDR", "AFDR"} <= set(report.metrics)
    assert report.metrics["AFDR"] > 0.0
    assert report.procedure_metrics["re-Q-ineq"]["FDR"] == pytest.approx(1.0)


def test_classification_evaluator_and_factory() -> None:
    evaluator = build_evaluator("classification")
    report = evaluator.evaluate(
        torch.tensor([0, 1, 1, 0]),
        torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.8, 0.2]]),
    )
    assert report.overall["Accuracy"] == pytest.approx(0.75)
    assert len(report.confusion_matrix) == 2
    fault = build_evaluator({"type": "fault_detection", "expected_far": 0.1})
    assert fault.expected_far == pytest.approx(0.1)
    koopman = build_evaluator("koopman_contribution")
    assert isinstance(koopman, KoopmanContributionEvaluator)


def test_koopman_contribution_evaluator_reports_ratio_sparsity_and_dimensions() -> None:
    output = {
        "first_order": torch.tensor([[1.0, -2.0], [3.0, -4.0]]),
        "second_order": torch.tensor([[0.0, 1.0], [0.5, 0.0]]),
        "second_order_diagonal": torch.tensor([[0.2, 0.0], [0.2, 0.0]]),
        "second_order_fm": torch.tensor([[0.3, 0.4], [0.0, 0.0]]),
    }
    report = KoopmanContributionEvaluator(sparsity_threshold=0.0).evaluate(output)
    assert report.overall["first_order_mean_abs"] == pytest.approx(2.5)
    assert report.overall["second_order_mean_abs"] == pytest.approx(0.375)
    assert report.overall["second_to_first_ratio"] == pytest.approx(0.15)
    assert report.overall["second_order_sparsity"] == pytest.approx(0.5)
    assert report.overall["diagonal_mean_abs"] == pytest.approx(0.1)
    assert report.overall["fm_mean_abs"] == pytest.approx(0.175)
    assert len(report.per_dimension) == 2
    assert "first_order_mean_abs" in report.to_flat_dict()


def test_best_result_monitor_uses_rmse_then_r2(tmp_path) -> None:
    monitor = BestResultMonitor(tmp_path / "best.json")
    first = monitor.update_if_better({"rmse": 1.0, "r2": 0.5}, {"model": "a"}, "a.pt")
    assert first.updated
    worse = monitor.update_if_better({"rmse": 1.1, "r2": 0.9}, {"model": "b"}, "b.pt")
    assert not worse.updated
    tie_better = monitor.update_if_better({"rmse": 1.0, "r2": 0.7}, {"model": "c"}, "c.pt")
    assert tie_better.updated
    assert monitor.load()["checkpoint_path"] == "c.pt"


def test_artifact_store_rejects_paths_outside_run_directory(tmp_path) -> None:
    store = ArtifactStore(tmp_path, "safe_store")
    with pytest.raises(ValueError, match="inside the run directory"):
        store.save_json("../escape.json", {"bad": True})
    with pytest.raises(ValueError, match="inside the run directory"):
        store.save_json(tmp_path / "absolute.json", {"bad": True})


def test_study_expands_grid_repeats_and_writes_summary(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "x": torch.linspace(0, 1, 24).numpy(),
            "y": torch.linspace(0, 1, 24).numpy() * 2.0,
        }
    )
    csv_path = tmp_path / "study.csv"
    frame.to_csv(csv_path, index=False)
    study = Study.from_config(
        {
            "name": "study_smoke",
            "artifacts": {"root": tmp_path},
            "base": {
                "seed": 10,
                "data": {
                    "path": csv_path,
                    "target_cols": -1,
                    "batch_size": 4,
                    "split": {"type": "sequential", "test_ratio": 0.25},
                },
                "model": {"type": "mlp", "hidden": [4]},
                "trainer": {"max_epochs": 1, "optimizer": {"lr": 0.01}},
                "artifacts": {"root": tmp_path},
            },
            "sweep": {"trainer.optimizer.lr": [0.01, 0.005]},
            "repeats": {"n": 2, "base_seed": 100},
            "ranking": {"metric": "rmse", "mode": "min"},
        }
    )
    result = study.run()
    assert len(result.results) == 4
    assert len(result.summary) == 4
    assert result.failures.empty
    assert (result.run_dir / "summary" / "summary.csv").exists()
    assert (result.run_dir / "summary" / "leaderboard.csv").exists()
    assert (result.run_dir / "summary" / "failures.csv").exists()
    assert (result.run_dir / "study.yaml").exists()
    assert (result.run_dir / "expanded_trials.csv").exists()
    assert (result.run_dir / "best" / "config.yaml").exists()
    assert (result.run_dir / "best" / "metrics.json").exists()
    assert "metric.rmse.ci95" in result.leaderboard.columns


def test_study_coupled_trials_and_repeat_seed_policies_are_deterministic() -> None:
    sweep = {
        "model.type": ["dae"],
        "coupled.arch": [
            {"model.struct": [4, 8, 2], "model.act": ["r", "a"]},
            {"model.struct": [4, 6, 2], "model.act": ["g", "a"]},
        ],
    }
    first = _expand_trials(sweep)
    second = _expand_trials(sweep)
    assert len(first) == 2
    assert [trial.trial_id for trial in first] == [trial.trial_id for trial in second]
    assert all("model.struct" in trial.overrides and "model.act" in trial.overrides for trial in first)
    assert [repeat.seed for repeat in _expand_repeats({"strategy": "list", "seeds": [7, 11]}, default_seed=1)] == [
        7,
        11,
    ]
    spawned = _expand_repeats({"strategy": "spawn", "n": 3, "base_seed": 123}, default_seed=1)
    assert len({repeat.seed for repeat in spawned}) == 3


def test_study_random_planner_is_seeded_and_supports_range_specs() -> None:
    sweep = {
        "strategy": "random",
        "num_trials": 4,
        "seed": 17,
        "parameters": {
            "model.hidden": {"choice": [[4], [8], [12]]},
            "trainer.batch_size": {"int_range": {"start": 4, "stop": 8, "step": 2}},
            "trainer.optimizer.lr": {"log_uniform": [1e-4, 1e-2]},
        },
    }
    first = _expand_trials(sweep)
    second = _expand_trials(sweep)
    assert [trial.overrides for trial in first] == [trial.overrides for trial in second]
    assert len(first) == 4
    assert all(trial.overrides["trainer.batch_size"] in {4, 6, 8} for trial in first)
    assert all(1e-4 <= trial.overrides["trainer.optimizer.lr"] <= 1e-2 for trial in first)


def test_study_leaderboard_supports_tie_breakers_and_failure_counts() -> None:
    summary = pd.DataFrame(
        [
            {"trial_index": 0, "trial_id": "a", "metric.rmse": 1.0, "metric.r2": 0.5},
            {"trial_index": 1, "trial_id": "b", "metric.rmse": 1.0, "metric.r2": 0.8},
            {"trial_index": 2, "trial_id": "c", "metric.rmse": 1.2, "metric.r2": 0.9},
        ]
    )
    failures = pd.DataFrame([{"trial_index": 1, "trial_id": "b", "status": "failed"}])
    leaderboard = _leaderboard(
        summary,
        {"primary": "rmse", "mode": "min", "tie_breakers": [{"metric": "r2", "mode": "max"}]},
        failures=failures,
    )
    assert leaderboard.iloc[0]["trial_id"] == "b"
    assert leaderboard.iloc[0]["n_failed"] == 1
    assert leaderboard.iloc[1]["trial_id"] == "a"


def test_study_resume_skips_completed_repeat(tmp_path) -> None:
    frame = pd.DataFrame({"x": torch.linspace(0, 1, 16).numpy(), "y": torch.linspace(0, 1, 16).numpy()})
    csv_path = tmp_path / "resume.csv"
    frame.to_csv(csv_path, index=False)
    config = {
        "name": "resume_study",
        "artifacts": {"root": tmp_path},
        "base": {
            "seed": 3,
            "data": {
                "path": csv_path,
                "target_cols": -1,
                "batch_size": 4,
                "split": {"type": "sequential", "test_ratio": 0.25},
            },
            "model": {"type": "mlp", "hidden": [3]},
            "trainer": {"max_epochs": 1},
        },
        "repeats": {"n": 1, "base_seed": 50},
    }
    first = Study.from_config(config).run()
    second = Study.from_config(config).run(resume=True)
    assert len(first.results) == 1
    assert len(second.results) == 0
    assert second.summary.iloc[0]["status"] == "skipped"


def test_study_records_failure_traceback_and_continues(tmp_path) -> None:
    result = Study.from_config(
        {
            "name": "failure_study",
            "artifacts": {"root": tmp_path},
            "base": {
                "model": {"type": "missing_model", "input_dim": 2, "output_dim": 1},
                "artifacts": {"root": tmp_path},
            },
            "continue_on_error": True,
        }
    ).run()
    assert result.summary.empty
    assert len(result.failures) == 1
    traceback_path = result.failures.iloc[0]["traceback_path"]
    assert traceback_path
    assert Path(traceback_path).exists()


def test_training_plotter_returns_loss_figure() -> None:
    figure = TrainingPlotter().loss_curve(
        [{"epoch": 0.0, "train/loss": 2.0}, {"epoch": 1.0, "train/loss": 1.0}]
    )
    assert figure.axes


def test_paper_training_plotter_saves_pdf_svg_png_without_global_rc_mutation(tmp_path) -> None:
    before_font_size = plt.rcParams["font.size"]
    plotter = TrainingPlotter(theme="paper", figure=FigureSpec.paper_double_short())
    figure = plotter.loss_curve(
        [{"epoch": 0.0, "train/loss": 2.0}, {"epoch": 1.0, "train/loss": 1.0}]
    )
    width, height = figure.get_size_inches()
    assert width > height
    assert width == pytest.approx(7.16)
    assert height == pytest.approx(1.80)
    assert figure.axes[0].xaxis.label.get_fontsize() >= 13
    paths = plotter.save(figure, ArtifactStore(tmp_path, "paper"), "loss")
    assert {path.suffix for path in paths} == {".pdf", ".svg", ".png"}
    assert all(path.exists() for path in paths)
    assert plt.rcParams["font.size"] == before_font_size


def test_specialized_plotters_return_figures() -> None:
    y_true = torch.tensor([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    y_pred = y_true + 0.1
    assert PredictionPlotter().series(y_true, y_pred).axes
    assert PredictionPlotter().scatter_true_pred(y_true, y_pred).axes
    assert DataPlotter().split_distribution(["train", "train", "test"]).axes
    assert DataPlotter().outlier_marks([1.0, 4.0, 2.0], [1]).axes
    assert FaultDetectionPlotter().stat_curve([0.1, 0.2, 1.5], threshold=0.5, labels=[0, 0, 1]).axes
    assert FaultDetectionPlotter().far_mdr_bar({"FAR": 0.1, "MDR": 0.2, "FDR": 0.8}).axes
    assert FlowPlotter().z_distribution(torch.randn(8, 3)).axes
    assert KoopmanPlotter().contribution([0.2, -0.4, 0.1], feature_names=["a", "b", "c"]).axes


def test_palette_registry_imports_user_coolors_url_without_trending_scrape() -> None:
    palette = Palette.from_coolors_url(
        "https://coolors.co/264653-2a9d8f-e9c46a",
        name="earthy_process",
    )
    assert palette.colors == ("#264653", "#2A9D8F", "#E9C46A")
    assert palette.source == "coolors:user_imported"
    registry = PaletteRegistry()
    assert "joff_colorblind" in registry.list()
    registry.register("earthy_process", palette)
    assert registry.get("earthy_process") == palette
    assert not hasattr(registry, "download_trending")
    with pytest.raises(ValueError, match="Trending pages are not imported"):
        Palette.from_coolors_url("https://coolors.co/palettes/trending", name="bad")


def test_plot_theme_uses_palette_color_cycle_without_global_rc_mutation() -> None:
    before_cycle = plt.rcParams["axes.prop_cycle"]
    palette = Palette.from_hex(["#111111", "#222222"], name="gray_pair")
    theme = PlotTheme.paper(palette=palette)
    figure = TrainingPlotter(theme=theme).loss_curve(
        [{"epoch": 0.0, "train/loss": 2.0}, {"epoch": 1.0, "train/loss": 1.0}]
    )
    assert figure.axes[0].lines[0].get_color() == "#111111"
    assert plt.rcParams["axes.prop_cycle"] == before_cycle


def test_joff_console_renders_tables_metrics_and_progress_with_plain_fallback() -> None:
    stream = io.StringIO()
    console = JoffConsole(file=stream, verbose=2, force_plain=True)
    console.rule("Training")
    console.metric("test/rmse", 0.123456, style="good")
    console.table({"rmse": 0.12, "r2": 0.9}, title="Metrics")
    with console.progress(total=2, description="fit") as progress:
        progress.advance()
        progress.advance()
    output = stream.getvalue()
    assert "Training" in output
    assert "test/rmse: 0.123456" in output
    assert "Metrics" in output
    assert "rmse" in output
    assert "fit progress=2/2" in output


def test_console_tables_normalize_config_and_records() -> None:
    table = MetricTable.from_data([{"a": 1, "b": 2.0}, {"a": 3, "b": 4.0}])
    assert table.headers == ["a", "b"]
    assert table.rows[0] == ["1", "2"]
    config = ConfigTable.from_config({"model": {"type": "dae"}, "trainer": {"max_epochs": 1}})
    plain = config.render_plain()
    assert "model.type" in plain
    assert "trainer.max_epochs" in plain


def test_joff_console_quiet_suppresses_output() -> None:
    stream = io.StringIO()
    console = JoffConsole(file=stream, quiet=True, force_plain=True)
    console.info("hidden")
    console.table({"a": 1})
    assert stream.getvalue() == ""


def test_local_tracker_writes_events_config_metrics_and_figures(tmp_path) -> None:
    store = ArtifactStore(tmp_path, "tracked")
    tracker = LocalTracker(store)
    tracker.start_run(RunInfo(run_id="abc123", name="local_smoke", tags={"model": "dae"}))
    tracker.log_config({"model": {"type": "dae"}})
    tracker.log_metric("loss", 0.25, step=1)
    tracker.log_metrics({"rmse": 0.5, "r2": 0.8}, step=2)
    artifact = store.save_json("metrics/example.json", {"ok": True})
    tracker.log_artifact(artifact, name="example")
    figure = TrainingPlotter().loss_curve([{"epoch": 0.0, "train/loss": 1.0}])
    tracker.log_figure(figure, "loss")
    tracker.end_run()

    events_path = store.path / "tracking" / "events.jsonl"
    text = events_path.read_text(encoding="utf-8")
    assert "run_start" in text
    assert "metric" in text
    assert "run_end" in text
    assert (store.path / "tracking" / "config.json").exists()
    assert (store.path / "tracking" / "figures" / "loss.png").exists()


def test_optional_trackers_raise_install_hints_when_dependencies_are_missing(
    tmp_path,
    monkeypatch,
) -> None:
    original_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name in {"mlflow", "wandb", "torch.utils.tensorboard"}:
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match='pip install "joff\\[tracking\\]"'):
        TensorBoardTracker(tmp_path / "tb")
    with pytest.raises(ImportError, match='pip install "joff\\[tracking\\]"'):
        MLflowTracker()
    with pytest.raises(ImportError, match='pip install "joff\\[wandb\\]"'):
        WandbTracker(project="joff")
