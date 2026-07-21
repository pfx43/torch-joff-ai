from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from joff.artifacts import ArtifactStore
from joff.data import (
    ColumnSpec,
    DataPipeline,
    DataModule,
    DataSchema,
    DynamicDistributionSplitter,
    DynamicWindowDataset,
    MissingValueProcessor,
    MPCWindowDataset,
    Normalizer,
    OutlierConfig,
    OutlierProcessor,
    SequentialSplitter,
    SequenceDataset,
    TabularSeries,
    TaskSchema,
    TaskView,
    list_dataset_presets,
)


def test_missing_value_processor_interpolates_then_drops() -> None:
    data = TabularSeries(x=np.array([[1.0], [np.nan], [3.0], [np.inf]]), y=np.arange(4.0))
    result = MissingValueProcessor().fit_transform(data)
    assert result.data.row_count == 4
    assert np.isfinite(result.data.x).all()
    assert result.summary["removed_rows"] == 0


def test_outlier_methods_and_local_temporal_detection() -> None:
    values = np.ones((30, 1))
    values[10] = 20.0
    data = TabularSeries(x=values, quality=values)
    result = OutlierProcessor(
        OutlierConfig(method="mad", feature_scope="quality", max_removal_ratio=0.2)
    ).fit_transform(data)
    assert 10 in result.removed_indices

    smooth = np.linspace(0, 1, 40)[:, None]
    spike = smooth.copy()
    spike[20] += 5.0
    local = OutlierProcessor(
        OutlierConfig(
            method="zscore",
            feature_scope="quality",
            z_threshold=99.0,
            use_local_temporal=True,
            local_window_radius=4,
            local_mad_threshold=6.0,
            local_min_side_neighbors=2,
            local_min_abs_deviation=0.5,
            max_removal_ratio=0.1,
        )
    ).fit_transform(TabularSeries(x=spike, quality=spike))
    assert 20 in local.removed_indices


def test_outlier_max_removal_ratio_caps_deletions() -> None:
    values = np.arange(20.0)[:, None]
    result = OutlierProcessor(
        OutlierConfig(method="tail_percent", tail_percent=40, feature_scope="input", max_removal_ratio=0.1)
    ).fit_transform(TabularSeries(x=values))
    assert len(result.removed_indices) == 2


def test_normalizer_fits_train_only_and_inverse_transforms() -> None:
    train = np.array([[0.0, 2.0], [2.0, 2.0], [4.0, 2.0]])
    test = np.array([[6.0, 2.0]])
    normalizer = Normalizer(method="minmax")
    train_scaled, test_scaled = normalizer.fit_transform_train_test(train, test)
    assert train_scaled[:, 0].min() == pytest.approx(0.0)
    assert train_scaled[:, 0].max() == pytest.approx(1.0)
    assert test_scaled[0, 0] == pytest.approx(1.5)
    assert normalizer.summary()["constant_columns"] == 1
    assert np.allclose(normalizer.inverse_transform(train_scaled), train)


def test_task_view_resolves_columns_and_reconstruction_target() -> None:
    frame = pd.DataFrame(
        {
            "time": np.arange(5),
            "u": np.linspace(0.0, 1.0, 5),
            "y": np.linspace(1.0, 2.0, 5),
            "label": [0, 0, 1, 1, 1],
        }
    )
    schema = DataSchema(
        columns=(
            ColumnSpec("time", role="time"),
            ColumnSpec("u", role="control"),
            ColumnSpec("y", role="output"),
            ColumnSpec("label", role="label"),
        )
    )

    prediction = TaskView.from_schema(
        frame,
        schema,
        TaskSchema(name="prediction", inputs=("control", "output"), targets=("output",)),
    )
    x, y = prediction.arrays(frame)
    assert prediction.input_columns == ("u", "y")
    assert prediction.target_columns == ("y",)
    assert x.shape == (5, 2)
    assert y.shape == (5, 1)
    assert prediction.summary()["kind"] == "prediction"

    reconstruction = TaskView.from_schema(
        frame,
        schema,
        TaskSchema(name="reconstruction", inputs=("control", "output")),
    )
    rec_x, rec_y = reconstruction.arrays(frame)
    assert reconstruction.target_policy == "reconstruction_input"
    assert np.allclose(rec_x, rec_y)


def test_dynamic_window_dataset_indices_and_segments() -> None:
    u = np.arange(10.0)[:, None]
    y = (np.arange(10.0) * 10)[:, None]
    segments = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    dataset = DynamicWindowDataset(u, y, lookback=2, future_steps=1, segment_ids=segments)
    assert dataset.sample_starts.tolist() == [0, 1, 2, 5, 6, 7]
    first = dataset[0]
    assert first["index"].item() == 0
    assert first["history"].shape[0] == 4
    assert first["future"].shape[0] == 2


def test_sequence_dataset_continuous_modes_and_segment_boundaries() -> None:
    x = np.arange(12.0)[:, None]
    y = (np.arange(12.0) + 100.0)[:, None]
    segments = np.array([0] * 6 + [1] * 6)

    n_to_1 = SequenceDataset(x, y, input_length=3, task="n_to_1", segment_ids=segments)
    assert n_to_1.sample_starts.tolist() == [0, 1, 2, 6, 7, 8]
    first = n_to_1[0]
    assert first["x"].shape == (3, 1)
    assert first["target"].shape == (1,)
    assert first["target"].item() == pytest.approx(103.0)

    n_to_n = SequenceDataset(x, y, input_length=3, task="n_to_n", segment_ids=segments)
    assert n_to_n.sample_starts.tolist() == [0, 1, 2, 3, 6, 7, 8, 9]
    assert n_to_n[0]["target"].shape == (3, 1)

    n_to_m = SequenceDataset(x, y, input_length=3, target_length=2, task="n_to_m", segment_ids=segments)
    assert n_to_m.sample_starts.tolist() == [0, 1, 6, 7]
    assert n_to_m[0]["target"].shape == (2, 1)
    assert n_to_m.summary()["task"] == "n_to_m"


def test_mpc_window_dataset_uses_horizons_and_episode_boundaries() -> None:
    state = np.arange(12.0)[:, None]
    output = (np.arange(12.0) + 20.0)[:, None]
    control = (np.arange(12.0) + 40.0)[:, None]
    reference = (np.arange(12.0) + 60.0)[:, None]
    episodes = np.array([0] * 6 + [1] * 6)
    dataset = MPCWindowDataset(
        state=state,
        output=output,
        control=control,
        reference=reference,
        target=output,
        past_horizon=2,
        prediction_horizon=2,
        control_horizon=1,
        episode_ids=episodes,
    )

    assert dataset.sample_starts.tolist() == [0, 1, 2, 6, 7, 8]
    first = dataset[0]
    assert first["past"].shape == (2, 4)
    assert first["target_future"].shape == (2, 1)
    assert first["control_future"].shape == (1, 1)
    assert first["reference_future"].shape == (2, 1)


def test_sequential_and_dynamic_distribution_splitters() -> None:
    u = np.arange(80.0)[:, None]
    y = np.sin(np.arange(80.0) / 5.0)[:, None]
    dataset = DynamicWindowDataset(u, y, lookback=3, future_steps=2)

    sequential = SequentialSplitter(train_ratio=0.7, eval_ratio=0.1).split(dataset)
    assert len(sequential.train) > 0
    assert len(sequential.test) > 0
    assert sequential.summary["type"] == "sequential"

    dynamic = DynamicDistributionSplitter(
        train_ratio=0.75,
        eval_ratio=0.0,
        test_ratio=0.25,
        slice_length=8,
        seed=1,
    ).split(dataset)
    assert len(dynamic.train) > 0
    assert len(dynamic.test) > 0
    assert dynamic.summary["type"] == "dynamic_distribution"
    assert "slice_summary" in dynamic.summary
    assert {"train", "eval", "test"} <= set(dynamic.split_indices)


def test_datamodule_saves_outlier_and_dynamic_split_csv_artifacts(tmp_path) -> None:
    u = np.arange(40.0)[:, None]
    y = np.sin(np.arange(40.0) / 4.0)[:, None]
    y[12] = 10.0
    from joff import DataModule

    data = DataModule.from_arrays(
        u,
        y,
        outliers={
            "method": "mad",
            "feature_scope": "target",
            "max_removal_ratio": 0.1,
        },
        window={"lookback": 2, "future_steps": 1},
        split={
            "type": "dynamic_distribution",
            "train_ratio": 0.7,
            "test_ratio": 0.3,
            "slice_length": 5,
        },
    )
    paths = data.save_summaries(ArtifactStore(tmp_path, "csv_artifacts"))
    assert paths["outlier_removed_indices_csv"].exists()
    assert paths["split_summary_csv"].exists()
    assert paths["dynamic_slice_summary_csv"].exists()
    assert paths["dynamic_split_summary_csv"].exists()
    assert paths["dynamic_distribution_summary_csv"].exists()
    assert paths["outlier_summary_csv"].exists()
    assert paths["window_summary_csv"].exists()
    assert paths["prepared_dataset_hash"].exists()


def test_datamodule_sequence_pipeline_returns_rnn_ready_batches_and_artifacts(tmp_path) -> None:
    x = np.arange(40.0).reshape(20, 2)
    y = np.linspace(0.0, 1.0, 20)[:, None]
    data = DataModule.from_arrays(
        x,
        y,
        batch_size=2,
        shuffle=False,
        split={"type": "sequential", "test_ratio": 0.25},
        normalization={"method": "standard"},
        sequence={"input_length": 4, "target_length": 1, "task": "n_to_1"},
    )

    batch = next(iter(data.loader("train")))
    assert batch["x"].shape == (2, 4, 2)
    assert batch["target"].shape == (2, 1)
    assert data.shuffle is False
    assert data.summaries["split_summary"]["type"] == "sequential"
    assert data.summaries["sequence_summary"]["train_samples"] == len(data.train_dataset)
    assert data.summaries["sequence_summary"]["test_samples"] == len(data.test_dataset)
    paths = data.save_summaries(ArtifactStore(tmp_path, "sequence_artifacts"))
    assert paths["sequence_summary_csv"].exists()


def test_data_pipeline_from_config_normalizes_aliases_and_rejects_unknown_steps(tmp_path) -> None:
    pipeline = DataPipeline.from_config(
        [
            "validate_schema",
            {"scaler": "standard"},
            {"window": {"history": 2, "horizon": 1}},
            {"sequence": {"input_length": 4, "task": "n_to_1"}},
            {"mpc_window": {"past_horizon": 2, "prediction_horizon": 3, "control_horizon": 1}},
            {"to_torch": {"batch_size": 4}},
        ]
    )
    assert pipeline.to_config()["normalization"] == {"method": "standard"}
    assert pipeline.to_config()["window"] == {"history": 2, "horizon": 1}
    assert pipeline.to_config()["sequence"] == {"input_length": 4, "task": "n_to_1"}
    assert pipeline.to_config()["mpc_window"]["prediction_horizon"] == 3

    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text(
        """
pipeline:
  - missing: {strategy: interpolate_then_drop}
  - outlier: {method: mad, feature_scope: target}
  - mask: {strategy: random, missing_rate: 0.2, seed: 3}
""".strip(),
        encoding="utf-8",
    )
    loaded = DataPipeline.from_config(yaml_path)
    assert loaded.to_config()["outliers"]["method"] == "mad"
    assert loaded.to_config()["mask"]["seed"] == 3

    with pytest.raises(ValueError, match="Legal options"):
        DataPipeline.from_config({"unknown_step": {}})


def test_datamodule_from_file_accepts_data_pipeline_object(tmp_path) -> None:
    csv_path = tmp_path / "pipeline_file.csv"
    pd.DataFrame(
        {
            "u": np.linspace(0.0, 1.0, 18),
            "y": np.linspace(1.0, 2.0, 18),
        }
    ).to_csv(csv_path, index=False)
    pipeline = DataPipeline.from_config(
        {
            "split": {"type": "sequential", "test_ratio": 0.25},
            "scaler": {"method": "standard"},
        }
    )

    data = DataModule.from_file(csv_path, batch_size=4, pipeline=pipeline)
    assert data.summaries["normalization_summary"]["method"] == "standard"
    assert data.summaries["split_summary"]["type"] == "sequential"


def test_stratified_split_keeps_classes_in_train_and_test() -> None:
    x = np.arange(48.0).reshape(24, 2)
    labels = np.array([0] * 12 + [1] * 12)
    data = DataModule.from_arrays(
        x,
        labels,
        batch_size=4,
        split={"method": "stratified", "test_ratio": 0.25, "seed": 3},
    )
    train_labels = data.train_dataset.tensors[1].numpy().reshape(-1)
    test_labels = data.test_dataset.tensors[1].numpy().reshape(-1)
    assert set(train_labels.tolist()) == {0.0, 1.0}
    assert set(test_labels.tolist()) == {0.0, 1.0}
    assert data.summaries["split_summary"]["type"] == "stratified"
    assert data.summaries["split_summary"]["class_counts"]["0.0"] == {"train": 9, "test": 3}


def test_group_split_keeps_whole_groups_together() -> None:
    x = np.arange(60.0).reshape(30, 2)
    y = np.arange(30.0)
    groups = np.repeat(np.arange(5), 6)
    data = DataModule.from_arrays(
        x,
        y,
        groups=groups,
        batch_size=4,
        split={"method": "group", "test_ratio": 0.4, "seed": 4},
    )
    summary = data.summaries["split_summary"]
    assert summary["type"] == "group"
    assert summary["train_groups"] + summary["test_groups"] == 5
    test_groups = {
        int(group)
        for group, info in summary["group_counts"].items()
        if info["split"] == "test"
    }
    train_groups = {
        int(group)
        for group, info in summary["group_counts"].items()
        if info["split"] == "train"
    }
    assert test_groups
    assert train_groups
    assert test_groups.isdisjoint(train_groups)
    assert summary["test_samples"] == sum(info["rows"] for info in summary["group_counts"].values() if info["split"] == "test")


def test_statistical_outliers_remove_train_only_and_preserve_test_faults() -> None:
    x = np.zeros((20, 1), dtype=float)
    labels = np.zeros(20, dtype=float)
    x[4, 0] = 50.0
    x[-5:, 0] = 100.0
    labels[-5:] = 1.0

    data = DataModule.from_arrays(
        x,
        labels,
        batch_size=4,
        split={"method": "sequential", "test_ratio": 0.25},
        outliers={
            "method": "zscore",
            "scope": "input",
            "z_threshold": 2.0,
            "max_removal_ratio": 0.2,
            "mark_test": True,
        },
    )

    train_labels = data.train_dataset.tensors[1].numpy().reshape(-1)
    test_labels = data.test_dataset.tensors[1].numpy().reshape(-1)
    summary = data.summaries["outlier_summary"]
    assert 1.0 not in set(train_labels.tolist())
    assert set(test_labels.tolist()) == {1.0}
    assert len(data.test_dataset) == 5
    assert summary["policy"]["fit_scope"] == "train"
    assert summary["train"]["removed_rows"] == 1
    assert summary["test"]["marked_rows"] == 5
    assert summary["test"]["removed_rows"] == 0
    assert data.summaries["split_summary"]["test_samples_after_outliers"] == 5


def test_datamodule_from_file_reads_csv_npz_and_mat(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "x1": np.linspace(0.0, 1.0, 12),
            "x2": np.linspace(1.0, 2.0, 12),
            "y": np.linspace(2.0, 3.0, 12),
        }
    )
    csv_path = tmp_path / "data.csv"
    frame.to_csv(csv_path, index=False)
    csv_data = DataModule.from_file(csv_path, batch_size=4, split={"type": "sequential", "test_ratio": 0.25})
    csv_batch = next(iter(csv_data.loader("train")))
    assert csv_batch[0].shape[1] == 2
    assert csv_batch[1].shape[1] == 1
    assert csv_data.summaries["source_summary"]["format"] == "csv"

    x = np.arange(24.0).reshape(12, 2)
    y = np.arange(12.0).reshape(12, 1)
    npz_path = tmp_path / "data.npz"
    np.savez(npz_path, x=x, y=y)
    npz_data = DataModule.from_file(npz_path, batch_size=4)
    npz_batch = next(iter(npz_data.loader("train")))
    assert npz_batch[0].shape[1] == 2
    assert npz_batch[1].shape[1] == 1
    assert npz_data.summaries["source_summary"]["format"] == "npz"

    mat_path = tmp_path / "data.mat"
    savemat(mat_path, {"x": x, "y": y})
    mat_data = DataModule.from_file(mat_path, batch_size=4)
    mat_batch = next(iter(mat_data.loader("train")))
    assert mat_batch[0].shape[1] == 2
    assert mat_batch[1].shape[1] == 1
    assert mat_data.summaries["source_summary"]["format"] == "mat"


def test_datamodule_from_file_reads_xlsx_when_excel_dependency_is_available(tmp_path) -> None:
    pytest.importorskip("openpyxl")
    xlsx_path = tmp_path / "data.xlsx"
    pd.DataFrame(
        {
            "x": np.linspace(0.0, 1.0, 10),
            "y": np.linspace(1.0, 2.0, 10),
        }
    ).to_excel(xlsx_path, index=False)

    data = DataModule.from_file(xlsx_path, batch_size=4, split={"type": "sequential", "test_ratio": 0.3})
    batch = next(iter(data.loader("train")))
    assert batch[0].shape[1] == 1
    assert batch[1].shape[1] == 1
    assert data.summaries["source_summary"]["format"] == "excel"


def test_imputation_mask_is_generated_after_split_and_returns_masks() -> None:
    x = np.arange(60.0).reshape(20, 3)
    data = DataModule.from_arrays(
        x,
        batch_size=5,
        split={"type": "sequential", "test_ratio": 0.25},
        normalization={"method": "standard"},
        mask={"strategy": "block_missing", "missing_rate": 0.2, "seed": 7, "block_length": 2},
    )
    sample = data.train_dataset[0]
    assert set(sample) == {"x", "target", "corrupted", "observed_mask", "eval_mask"}
    assert sample["x"].shape[0] == 6
    assert sample["target"].shape[0] == 3
    assert data.summaries["split_summary"]["train_samples"] == 15
    assert data.summaries["split_summary"]["test_samples"] == 5
    assert data.summaries["mask_summary"]["train"]["rows"] == 15
    assert data.summaries["mask_summary"]["test"]["rows"] == 5
    assert data.summaries["mask_summary"]["train"]["seed"] != data.summaries["mask_summary"]["test"]["seed"]
    assert data.summaries["mask_summary"]["train"]["masked_entries"] > 0


def test_from_file_imputation_requires_no_supervised_target(tmp_path) -> None:
    csv_path = tmp_path / "impute.csv"
    pd.DataFrame({"x1": np.arange(10.0), "x2": np.arange(10.0) + 1.0}).to_csv(
        csv_path,
        index=False,
    )
    with pytest.raises(ValueError, match="target_cols=None"):
        DataModule.from_file(csv_path, mask={"missing_rate": 0.2})

    data = DataModule.from_file(csv_path, target_cols=None, mask={"missing_rate": 0.2, "seed": 9})
    sample = data.train_dataset[0]
    assert sample["target"].shape[0] == 2
    assert data.summaries["mask_summary"]["train"]["features"] == 2


def test_datamodule_from_dataset_card_preset_and_summary_artifacts(tmp_path) -> None:
    train = pd.DataFrame(
        {
            "time": np.arange(16),
            "u": np.linspace(0.0, 1.0, 16),
            "y": np.linspace(1.0, 2.0, 16),
        }
    )
    test = pd.DataFrame(
        {
            "time": np.arange(8),
            "u": np.linspace(1.0, 1.5, 8),
            "y": np.linspace(2.0, 2.5, 8),
        }
    )
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    card_path = tmp_path / "dataset_card.yaml"
    card_path.write_text(
        """
name: toy_process
version: 1
description: Toy process prediction dataset
files:
  root: .
  train: train.csv
  test: test.csv
schema:
  sample_rate: 1.0
  columns:
    - {name: time, role: time}
    - {name: u, role: control}
    - {name: y, role: output}
tasks:
  prediction:
    inputs: [control, output]
    targets: [output]
""".strip(),
        encoding="utf-8",
    )

    data = DataModule.from_preset(
        card_path,
        task="prediction",
        batch_size=4,
        normalization={"method": "standard"},
    )
    batch = next(iter(data.loader("train")))
    assert batch[0].shape[1] == 2
    assert batch[1].shape[1] == 1
    assert data.loader("test") is not None
    assert {"schema_summary", "preset_summary", "task_view_summary", "source_summary", "pipeline_summary"} <= set(
        data.summaries
    )
    assert data.summaries["task_view_summary"]["input_columns"] == ["u", "y"]
    assert data.summaries["task_view_summary"]["target_columns"] == ["y"]
    paths = data.save_summaries(ArtifactStore(tmp_path, "card_artifacts"))
    assert paths["schema_yaml"].exists()
    assert paths["preset_yaml"].exists()
    assert paths["pipeline_yaml"].exists()
    assert paths["scaler_summary"].exists()
    assert paths["normalization_summary_csv"].exists()
    assert paths["prepared_dataset_hash"].exists()


def test_dataset_card_reads_npz_via_shared_source_reader(tmp_path) -> None:
    x_train = np.arange(24.0).reshape(12, 2)
    y_train = np.arange(12.0).reshape(12, 1)
    x_test = np.arange(12.0).reshape(6, 2)
    y_test = np.arange(6.0).reshape(6, 1)
    np.savez(tmp_path / "train.npz", x=x_train, y=y_train)
    np.savez(tmp_path / "test.npz", x=x_test, y=y_test)
    card_path = tmp_path / "dataset_card.yaml"
    card_path.write_text(
        """
name: npz_process
version: 1
files:
  root: .
  train: train.npz
  test: test.npz
schema:
  columns:
    - {name: x0, role: control}
    - {name: x1, role: output}
    - {name: y0, role: quality}
tasks:
  prediction:
    inputs: [control, output]
    targets: [quality]
""".strip(),
        encoding="utf-8",
    )

    data = DataModule.from_preset(card_path, task="prediction", batch_size=4)
    batch = next(iter(data.loader("train")))
    assert batch[0].shape[1] == 2
    assert batch[1].shape[1] == 1
    assert data.summaries["source_summary"]["split_rows"] == {"train": 12, "test": 6}


def test_dataset_card_preprocessing_drops_columns_and_label_categories(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "x1": np.linspace(0.0, 1.0, 24),
            "x2": np.linspace(1.0, 2.0, 24),
            "bad_sensor": np.linspace(2.0, 3.0, 24),
            "status": ["normal"] * 10 + ["fault"] * 10 + ["ignore"] * 4,
        }
    )
    frame.to_csv(tmp_path / "preprocess.csv", index=False)
    card_path = tmp_path / "dataset_card.yaml"
    card_path.write_text(
        """
name: preprocessing_process
version: 1
files:
  root: .
  data: preprocess.csv
preprocessing:
  drop_columns: [bad_sensor]
  drop_label_values:
    status: [ignore]
schema:
  columns:
    - {name: x1, role: input}
    - {name: x2, role: input}
    - {name: bad_sensor, role: input}
    - {name: status, role: label}
tasks:
  classification:
    inputs: [input]
    label_column: status
    normal_label: normal
    pipeline:
      split: {method: stratified, test_ratio: 0.2, seed: 3}
""".strip(),
        encoding="utf-8",
    )

    data = DataModule.from_preset(card_path, task="classification", batch_size=4)
    batch = next(iter(data.loader("train")))
    preprocessing = data.summaries["preprocessing_summary"]
    labels = data.train_dataset.tensors[1].numpy().reshape(-1)
    assert batch[0].shape[1] == 2
    assert data.summaries["task_view_summary"]["input_columns"] == ["x1", "x2"]
    assert data.summaries["source_summary"]["split_rows"] == {"train": 20}
    assert preprocessing["dropped_rows"] == 4
    assert preprocessing["dropped_columns"] == ["bad_sensor"]
    assert set(labels.tolist()) == {0.0, 1.0}
    paths = data.save_summaries(ArtifactStore(tmp_path, "preprocess_artifacts"))
    assert paths["preprocessing_summary_csv"].exists()


def test_dataset_card_imputation_task_uses_mask_pipeline(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "x1": np.linspace(0.0, 1.0, 18),
            "x2": np.linspace(1.0, 2.0, 18),
        }
    )
    frame.to_csv(tmp_path / "train.csv", index=False)
    card_path = tmp_path / "dataset_card.yaml"
    card_path.write_text(
        """
name: imputation_process
version: 1
files:
  root: .
  data: train.csv
schema:
  columns:
    - {name: x1, role: input}
    - {name: x2, role: input}
tasks:
  imputation:
    inputs: [input]
    pipeline:
      split: {type: sequential, test_ratio: 0.25}
      mask: {strategy: random, missing_rate: 0.25, seed: 11}
""".strip(),
        encoding="utf-8",
    )

    data = DataModule.from_preset(card_path, task="imputation", batch_size=4)
    sample = data.train_dataset[0]
    assert sample["x"].shape[0] == 4
    assert sample["target"].shape[0] == 2
    assert data.summaries["task_summary"]["name"] == "imputation"
    assert data.summaries["pipeline_summary"]["mask"]["missing_rate"] == 0.25
    assert data.summaries["mask_summary"]["test"]["rows"] > 0


def test_dataset_card_accepts_data_pipeline_object_override(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "u": np.linspace(0.0, 1.0, 24),
            "y": np.linspace(1.0, 2.0, 24),
        }
    )
    frame.to_csv(tmp_path / "train.csv", index=False)
    card_path = tmp_path / "dataset_card.yaml"
    card_path.write_text(
        """
name: pipeline_process
version: 1
files:
  root: .
  data: train.csv
schema:
  columns:
    - {name: u, role: control}
    - {name: y, role: output}
tasks:
  prediction:
    inputs: [control]
    targets: [output]
""".strip(),
        encoding="utf-8",
    )
    pipeline = DataPipeline.from_config(
        [
            {"split": {"type": "sequential", "test_ratio": 0.25}},
            {"scaler": "standard"},
        ]
    )

    data = DataModule.from_preset(card_path, task="prediction", pipeline=pipeline, batch_size=4)
    assert data.summaries["pipeline_summary"]["normalization"]["method"] == "standard"
    assert data.summaries["split_summary"]["type"] == "sequential"


def test_dataset_card_classification_task_uses_stratified_split(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "x1": np.linspace(0.0, 1.0, 24),
            "x2": np.linspace(1.0, 2.0, 24),
            "label": [0] * 12 + [1] * 12,
        }
    )
    frame.to_csv(tmp_path / "classification.csv", index=False)
    card_path = tmp_path / "dataset_card.yaml"
    card_path.write_text(
        """
name: classification_process
version: 1
files:
  root: .
  data: classification.csv
schema:
  columns:
    - {name: x1, role: input}
    - {name: x2, role: input}
    - {name: label, role: label}
tasks:
  classification:
    inputs: [input]
    label_column: label
    pipeline:
      split: {method: stratified, test_ratio: 0.25, seed: 5}
""".strip(),
        encoding="utf-8",
    )

    data = DataModule.from_preset(card_path, task="classification", batch_size=4)
    train_labels = data.train_dataset.tensors[1].numpy().reshape(-1)
    test_labels = data.test_dataset.tensors[1].numpy().reshape(-1)
    assert set(train_labels.tolist()) == {0.0, 1.0}
    assert set(test_labels.tolist()) == {0.0, 1.0}
    assert data.summaries["task_summary"]["name"] == "classification"
    assert data.summaries["split_summary"]["type"] == "stratified"


def test_dataset_card_string_labels_are_encoded_and_summarized(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "x1": np.linspace(0.0, 1.0, 24),
            "x2": np.linspace(1.0, 2.0, 24),
            "status": ["normal"] * 12 + ["fault"] * 12,
        }
    )
    frame.to_csv(tmp_path / "string_labels.csv", index=False)
    card_path = tmp_path / "dataset_card.yaml"
    card_path.write_text(
        """
name: string_label_process
version: 1
files:
  root: .
  data: string_labels.csv
schema:
  columns:
    - {name: x1, role: input}
    - {name: x2, role: input}
    - {name: status, role: label}
tasks:
  classification:
    inputs: [input]
    label_column: status
    normal_label: normal
    pipeline:
      split: {method: stratified, test_ratio: 0.25, seed: 5}
""".strip(),
        encoding="utf-8",
    )

    data = DataModule.from_preset(card_path, task="classification", batch_size=4)
    train_labels = data.train_dataset.tensors[1].numpy().reshape(-1)
    test_labels = data.test_dataset.tensors[1].numpy().reshape(-1)
    summary = data.summaries["label_mapping_summary"]
    assert set(train_labels.tolist()) == {0.0, 1.0}
    assert set(test_labels.tolist()) == {0.0, 1.0}
    assert summary["mapping"] == {"normal": 0, "fault": 1}
    assert summary["counts"]["train"] == {"fault": 12, "normal": 12}
    paths = data.save_summaries(ArtifactStore(tmp_path, "label_artifacts"))
    assert paths["label_mapping_summary_csv"].exists()


def test_dataset_card_episode_split_keeps_episode_rows_together(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "episode": np.repeat([10, 11, 12, 13], 5),
            "state": np.linspace(0.0, 1.0, 20),
            "control": np.linspace(1.0, 2.0, 20),
            "target": np.linspace(2.0, 3.0, 20),
        }
    )
    frame.to_csv(tmp_path / "mpc.csv", index=False)
    card_path = tmp_path / "dataset_card.yaml"
    card_path.write_text(
        """
name: mpc_process
version: 1
files:
  root: .
  data: mpc.csv
schema:
  columns:
    - {name: episode, role: episode}
    - {name: state, role: state}
    - {name: control, role: control}
    - {name: target, role: target}
tasks:
  mpc:
    inputs: [state, control]
    targets: [target]
    pipeline:
      split: {method: episode, test_ratio: 0.5, seed: 2}
""".strip(),
        encoding="utf-8",
    )

    data = DataModule.from_preset(card_path, task="mpc", batch_size=4)
    summary = data.summaries["split_summary"]
    assert summary["type"] == "episode"
    assert summary["train_groups"] + summary["test_groups"] == 4
    assert summary["train_samples"] + summary["test_samples"] == 20
    assert all(info["rows"] == 5 for info in summary["group_counts"].values())
    batch = next(iter(data.loader("train")))
    assert batch[0].shape[1] == 2
    assert batch[1].shape[1] == 1


def test_dataset_card_mpc_window_builds_role_horizons_and_reports(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "episode": np.repeat([0, 1, 2, 3], 8),
            "state": np.linspace(0.0, 1.0, 32),
            "control": np.linspace(1.0, 2.0, 32),
            "output": np.linspace(2.0, 3.0, 32),
            "reference": np.linspace(3.0, 4.0, 32),
        }
    )
    frame.to_csv(tmp_path / "mpc_window.csv", index=False)
    card_path = tmp_path / "dataset_card.yaml"
    card_path.write_text(
        """
name: mpc_window_process
version: 1
files:
  root: .
  data: mpc_window.csv
schema:
  columns:
    - {name: episode, role: episode}
    - {name: state, role: state}
    - {name: control, role: control}
    - {name: output, role: output}
    - {name: reference, role: reference}
tasks:
  mpc:
    inputs: [state, output, control, reference]
    targets: [output]
    pipeline:
      split: {method: episode, test_ratio: 0.5, seed: 4}
      scaler: {method: standard}
      mpc_window: {past_horizon: 2, prediction_horizon: 3, control_horizon: 2}
""".strip(),
        encoding="utf-8",
    )

    data = DataModule.from_preset(card_path, task="mpc", batch_size=4)
    summary = data.summaries["mpc_window_summary"]
    assert data.summaries["split_summary"]["type"] == "episode"
    assert summary["past_horizon"] == 2
    assert summary["prediction_horizon"] == 3
    assert summary["control_horizon"] == 2
    assert summary["train_samples"] == len(data.train_dataset)
    sample = next(iter(data.loader("train")))
    assert sample["past"].shape[1:] == (2, 4)
    assert sample["target_future"].shape[1:] == (3, 1)
    assert sample["control_future"].shape[1:] == (2, 1)
    span = summary["past_horizon"] + max(summary["prediction_horizon"], summary["control_horizon"])
    for start in data.train_dataset.sample_starts:
        episode_slice = data.train_dataset.episode_ids[start : start + span]
        assert np.unique(episode_slice).shape[0] == 1
    paths = data.save_summaries(ArtifactStore(tmp_path, "mpc_artifacts"))
    assert paths["mpc_window_summary_csv"].exists()


def test_dataset_preset_errors_and_legacy_special_mapping_are_helpful() -> None:
    with pytest.raises(ValueError) as excinfo:
        DataModule.from_preset("does_not_exist")
    message = str(excinfo.value)
    assert "Legal presets" in message
    assert "cstr_fault_diagnosis" in message

    data = DataModule.from_legacy_special("CSTR/fd", batch_size=8)
    assert data.prepare() is data
    assert next(iter(data.loader("train")))[0].shape[1] == 5
    assert data.summaries["preset_summary"]["name"] == "cstr_fault_diagnosis"
    assert data.summaries["source_summary"]["source_type"] == "builtin_synthetic"


def test_dataset_preset_short_aliases_resolve_for_synthetic_fallback() -> None:
    data = DataModule.from_preset("cstr_fd", task="fd", batch_size=8)
    assert data.summaries["preset_summary"]["name"] == "cstr_fault_diagnosis"
    assert data.summaries["task_summary"]["name"] == "fault_diagnosis"
    assert data.summaries["source_summary"]["source_type"] == "builtin_synthetic"


def test_legacy_special_presets_are_registered_and_smoke_load() -> None:
    expected_presets = {
        "te_fault_diagnosis",
        "te_classification",
        "cstr_fault_diagnosis",
        "cstr_closed_loop_fd",
        "tts_fault_diagnosis",
        "hy_fault_diagnosis",
        "hy_quality_prediction",
        "multiphase_fd",
        "wpt_mpc",
    }
    assert expected_presets <= set(list_dataset_presets())

    cases = {
        "TE/fd": "fault_diagnosis",
        "TE/cls": "classification",
        "CSTR/fd": "fault_diagnosis",
        "CSTR/fd_close": "fault_diagnosis",
        "TTS/fd": "fault_diagnosis",
        "HY/fd": "fault_diagnosis",
        "HY_PRD": "prediction",
        "Multiphase_Flow_Facility": "fault_diagnosis",
        "WPT": "mpc",
    }
    for special, task_name in cases.items():
        data = DataModule.from_legacy_special(special, batch_size=8)
        batch = next(iter(data.loader("train")))
        assert len(data.train_dataset) > 0
        assert len(data.test_dataset) > 0
        assert data.summaries["task_summary"]["name"] == task_name
        assert data.summaries["source_summary"]["source_type"] == "builtin_synthetic"
        assert batch[0].shape[0] > 0

    imputation = DataModule.from_preset("hy_quality_prediction", task="imputation", batch_size=8)
    sample = imputation.train_dataset[0]
    assert set(sample) == {"x", "target", "corrupted", "observed_mask", "eval_mask"}
