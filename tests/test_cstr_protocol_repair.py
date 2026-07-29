"""
闭环 CSTR 论文协议元数据的 P1 回归测试。

文件用途：
    通过公开 dataset preset、CanonicalDataset 和 DataModule 接口验证
    `cstr_closed_loop_fd` 的物理列语义、逐行故障标签、episode 边界与来源追溯。
主要职责：
    固定随原始数据发布的 7 变量顺序和 onset=200 边界，并防止答案元数据进入模型输入；
    不测试 P2 五阶段切分、论文模型、检测性能或结构化隔离。
关键输入与输出：
    输入为仓库内公开 CSTR MAT 文件、数据说明和一个模拟删行的小数组；输出为对
    schema、frame、summary 与 DataModule 序列样本的行为断言，不生成论文实验产物。
依赖与副作用：
    依赖 SciPy MAT 读取能力和仓库内 OA 数据；测试只读文件，不访问网络、不修改数据，
    也不使用真实故障表现选择任何参数。
重要约束：
    真实 fault episode 仅用于 shape、onset、标签和追溯回归；上游模型的 BSD-3-Clause
    许可证不能替代本地 MAT 生成链证明，数据许可必须保持 ``to_verify``。测试通过仍不能
    把结构回归写成故障性能或论文结果。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from joff import DataModule
from joff.data.adapters import DATASET_REGISTRY


ROOT = Path(__file__).resolve().parents[1]
CSTR_RAW_ROOT = ROOT / "datasets" / "raw" / "oa" / "CSTR"
CSTR_CARD_PATH = (
    ROOT / "datasets" / "cards" / "oa" / "cstr_closed_loop_fd" / "dataset_card.yaml"
)


def test_closed_loop_cstr_schema_exposes_physical_roles_and_explicit_task_inputs() -> None:
    """公开 schema 必须区分控制输入、测量输出和不可进入模型的追溯列。"""

    adapter = DATASET_REGISTRY.resolve("cstr_closed_loop_fd")
    schema = adapter.schema()
    task = adapter.default_task("fault_diagnosis")

    assert schema.role_columns("control_input") == ("Ci", "Ti", "Tci")
    assert schema.role_columns("measured_output") == ("C", "T", "Tc", "Qc")
    assert schema.role_columns("raw_index") == ("raw_index",)
    assert task.inputs == ("control_input", "measured_output")
    assert task.targets == ("fault_id",)
    assert task.fault_switch == 200
    assert DATASET_REGISTRY.resolve("CSTR/fd_close") is adapter
    summary = adapter.summary()
    assert summary["access"]["license"] == "to_verify"
    assert summary["access"]["license_status"] == "to_verify"
    assert summary["access"]["license_reason"] == "local_mat_generation_chain_not_documented"
    assert summary["access"]["upstream_model_license"] == "BSD-3-Clause"
    assert summary["access"]["upstream_model_license_status"] == "verified"
    assert summary["access"]["upstream_model_source_version"] == "1.1.0.1"
    assert summary["access"]["upstream_model_local_sha256"] == (
        "4555be2fa4c93ab43d2f24ab26e2bf6511ec25d701b231fc7d57f6657b523a81"
    )


def test_closed_loop_cstr_read_applies_onset_labels_and_serializable_provenance() -> None:
    """真实 MAT 读取必须逐行标注 onset，并把物理答案留在 segment 元数据中。"""

    adapter = DATASET_REGISTRY.resolve("cstr_closed_loop_fd")
    dataset = adapter.read(root=CSTR_RAW_ROOT, task="fault_diagnosis")

    assert dataset.split_rows() == {"train": 16814, "test": 8 * 1201}
    normal_segment = dataset.splits["train"][0]
    assert normal_segment.meta.segment_id == "normal"
    assert normal_segment.frame["fault_id"].value_counts().to_dict() == {0.0: 16814}
    assert normal_segment.frame["raw_index"].tolist() == list(range(16814))

    expected_families = {
        1: "process",
        2: "process",
        3: "actuator",
        4: "actuator",
        5: "actuator",
        6: "sensor",
        7: "sensor",
        8: "sensor",
    }
    for fault_id, segment in enumerate(dataset.splits["test"], start=1):
        frame = segment.frame
        assert segment.meta.segment_id == f"fault{fault_id:02d}"
        assert frame["raw_index"].tolist() == list(range(1201))
        assert frame.loc[frame["time"] < 200, "fault_id"].eq(0).all()
        assert frame.loc[frame["time"] >= 200, "fault_id"].eq(fault_id).all()
        assert segment.meta.metadata == {
            "fault_id": fault_id,
            "mat_key": f"fault{fault_id:02d}",
            "episode": f"fault{fault_id:02d}",
            "episode_fault_id": fault_id,
            "fault_family": expected_families[fault_id],
            "fault_onset": 200,
            "label_counts": {"0": 200, str(fault_id): 1001},
            "source_sha256": "83e438244bc30a76daf2873f20adc058c5096b67e26237205daf5965c2dba956",
        }

    summary = dataset.source_summary()
    assert summary["variables"] == {
        "control_input": ["Ci", "Ti", "Tci"],
        "measured_output": ["C", "T", "Tc", "Qc"],
    }
    assert summary["fault_onset"] == 200
    assert summary["fault_families"] == {str(key): value for key, value in expected_families.items()}
    assert summary["files"]["train"]["sha256"] == (
        "ef889b69b5ffa270b96a7ded86595b066a8b278df0cb39ddbb3ad18dacdd9839"
    )
    assert summary["files"]["test"]["sha256"] == (
        "83e438244bc30a76daf2873f20adc058c5096b67e26237205daf5965c2dba956"
    )
    assert summary["files"]["description"]["sha256"] == (
        "59d1b8ad014c853215cde873857365a95ed7e38538b2f4f355b423b49fdf7a92"
    )
    assert summary["access"]["license"] == "to_verify"
    assert summary["access"]["license_status"] == "to_verify"
    assert summary["access"]["upstream_model_license"] == "BSD-3-Clause"
    assert summary["access"]["upstream_model_license_status"] == "verified"
    assert summary["access"]["upstream_model_local_sha256"] == (
        "4555be2fa4c93ab43d2f24ab26e2bf6511ec25d701b231fc7d57f6657b523a81"
    )


def test_closed_loop_cstr_datamodule_keeps_model_inputs_and_sequence_episodes_separate() -> None:
    """DataModule 必须保留 episode 边界和可由预测索引查询的原始位置。"""

    data = DataModule.from_preset(
        "cstr_closed_loop_fd",
        root=CSTR_RAW_ROOT,
        task="fault_diagnosis",
        batch_size=16,
        shuffle=False,
        sequence={
            "input_length": 3,
            "target_length": 1,
            "task": "n_to_1",
            "target_offset": 0,
            "stride": 1,
        },
    )

    assert data.summaries["task_view_summary"]["input_columns"] == [
        "Ci",
        "Ti",
        "Tci",
        "C",
        "T",
        "Tc",
        "Qc",
    ]
    test_dataset = data.test_dataset
    assert test_dataset is not None
    assert np.unique(test_dataset.segment_ids).tolist() == [
        "fault01",
        "fault02",
        "fault03",
        "fault04",
        "fault05",
        "fault06",
        "fault07",
        "fault08",
    ]
    for start in test_dataset.sample_starts:
        span_start = int(start)
        span_end = (
            span_start
            + test_dataset.input_length
            + test_dataset.target_offset
            + test_dataset.target_length
        )
        assert np.unique(test_dataset.segment_ids[span_start:span_end]).size == 1

    provenance = data.summaries["sequence_provenance"]
    first_sample = test_dataset[0]
    target_index = int(first_sample["target_index"])
    assert provenance["index_semantics"] == (
        "SequenceDataset index and target_index address processed split rows in these columns."
    )
    assert provenance["test"]["source"][target_index].endswith("model1[test].mat")
    assert provenance["test"]["episode"][target_index] == "fault01"
    assert provenance["test"]["time"][target_index] == 3.0
    assert provenance["test"]["raw_index"][target_index] == 3


def test_closed_loop_cstr_dataset_card_path_routes_to_declared_protocol_adapter() -> None:
    """卡片路径必须复现注册 preset 的 onset、schema 和真实来源，而非走通用 MAT 读取。"""

    adapter = DATASET_REGISTRY.resolve(CSTR_CARD_PATH)
    dataset = adapter.read(task="fault_diagnosis")
    card_summary = adapter.summary()

    assert adapter.schema().role_columns("control_input") == ("Ci", "Ti", "Tci")
    assert card_summary["access"]["license"] == "to_verify"
    assert card_summary["access"]["license_status"] == "to_verify"
    assert card_summary["access"]["upstream_model_license"] == "BSD-3-Clause"
    assert card_summary["access"]["upstream_model_license_status"] == "verified"
    assert card_summary["access"]["upstream_model_local_sha256"] == (
        "4555be2fa4c93ab43d2f24ab26e2bf6511ec25d701b231fc7d57f6657b523a81"
    )
    assert dataset.split_rows() == {"train": 16814, "test": 8 * 1201}
    assert dataset.metadata["source_type"] == "real_dataset"
    assert Path(dataset.metadata["root"]).resolve() == (CSTR_RAW_ROOT / "fd_close").resolve()
    assert dataset.source_summary()["access"]["license"] == "to_verify"
    assert dataset.source_summary()["access"]["license_status"] == "to_verify"
    assert dataset.splits["test"][0].frame.loc[199, "fault_id"] == 0
    assert dataset.splits["test"][0].frame.loc[200, "fault_id"] == 1


def test_specialized_cstr_card_routing_does_not_break_other_dataset_cards() -> None:
    """专用 CSTR 路由不得把其他卡片的历史 adapter 元数据变成严格类名要求。"""

    card_paths = sorted((ROOT / "datasets" / "cards" / "oa").glob("*/dataset_card.yaml"))
    other_cards = [path for path in card_paths if path != CSTR_CARD_PATH]

    assert other_cards
    for card_path in other_cards:
        adapter = DATASET_REGISTRY.resolve(card_path)
        assert adapter.name == card_path.parent.name


def test_sequence_provenance_keeps_original_row_ids_after_missing_value_drop() -> None:
    """删行预处理后，局部 target_index 必须仍能查到跳号的原始 raw_index。"""

    train_x = np.arange(16, dtype=float).reshape(8, 2)
    test_x = train_x + 100.0
    train_x[1, 0] = np.nan
    test_x[1, 0] = np.nan
    segment_ids = np.asarray(["short", "short", "kept", "kept", "kept", "kept", "kept", "kept"])

    data = DataModule.from_arrays(
        train_x,
        train_x.copy(),
        test_x,
        test_x.copy(),
        shuffle=False,
        groups=segment_ids,
        test_segment_ids=segment_ids,
        train_row_provenance={
            "source": ["train.mat"] * 8,
            "episode": segment_ids,
            "time": np.arange(8, dtype=float),
            "raw_index": np.arange(100, 108),
        },
        test_row_provenance={
            "source": ["test.mat"] * 8,
            "episode": segment_ids,
            "time": np.arange(8, dtype=float),
            "raw_index": np.arange(200, 208),
        },
        missing={"strategy": "drop"},
        sequence={
            "input_length": 2,
            "target_length": 1,
            "task": "n_to_1",
        },
    )

    provenance = data.summaries["sequence_provenance"]
    assert provenance["train"]["raw_index"] == [100, 102, 103, 104, 105, 106, 107]
    assert provenance["test"]["raw_index"] == [200, 202, 203, 204, 205, 206, 207]
    assert provenance["train"]["episode"] == [
        "short",
        "kept",
        "kept",
        "kept",
        "kept",
        "kept",
        "kept",
    ]
