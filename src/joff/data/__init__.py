"""
Joff 数据层的稳定公共导入入口。

文件用途：
    汇总数据适配器、schema、通用流水线、DataModule 和论文五阶段协议，使实验编排不依赖
    子模块内部路径。
主要职责：
    只做显式符号再导出并维护 ``__all__``；不读取数据、不构造 DataModule、不执行切分，
    也不自动访问论文故障测试。
关键输入与输出：
    本文件没有运行时输入；输出是 ``joff.data`` 命名空间下受支持的类、函数和注册表。
依赖与副作用：
    导入依赖 NumPy/Pandas/PyTorch 数据模块和读取器的定义，但模块导入本身不读文件、
    不建目录、不修改随机性或绘图库全局状态。
重要约束：
    新数据集仍须显式注册；schema/task 决定列语义；论文协议的故障范围只有通过
    ``PaperDataBundle`` 门禁才能访问，不能因再导出而绕过冻结与许可检查。
"""

from .adapters import (
    DATASET_REGISTRY,
    CanonicalDataset,
    DatasetAdapter,
    DatasetCardAdapter,
    DatasetPreset,
    DatasetRegistry,
    Segment,
    list_dataset_presets,
    register_dataset_adapter,
)
from .datamodule import DataModule
from .in_memory import InMemoryDataModule
from .paper_protocol import (
    FaultLicenseStatus,
    FitAccessLedger,
    FitAccessRecord,
    FitPurpose,
    FiveStageNormalSplitter,
    FiveStageSplitConfig,
    FiveStageSplitResult,
    PaperDataBundle,
    ProtocolAccessError,
    StageName,
    StageSlice,
)
from .pipeline import (
    DataPipeline,
    DynamicDistributionSplitter,
    DynamicWindowDataset,
    ImputationDataset,
    ImputationMaskConfig,
    ImputationMaskResult,
    ImputationMasker,
    MPCWindowDataset,
    MPCWindowSample,
    MissingValueProcessor,
    Normalizer,
    OutlierConfig,
    OutlierProcessor,
    PipelineStep,
    SequentialSplitter,
    SequenceDataset,
    SequenceSample,
    SplitResult,
    TabularSeries,
    merge_pipeline_configs,
    normalize_pipeline_config,
)
from .schema import ColumnSpec, DataSchema, SegmentInfo, TaskSchema
from .sources import SUPPORTED_SOURCE_SUFFIXES, SourceData, read_source, read_source_frame, split_source_xy
from .sources import read_mat_arrays, read_npz_arrays
from .tasks import TaskView

__all__ = [
    "CanonicalDataset",
    "ColumnSpec",
    "DATASET_REGISTRY",
    "DataModule",
    "DataPipeline",
    "DataSchema",
    "DatasetAdapter",
    "DatasetCardAdapter",
    "DatasetPreset",
    "DatasetRegistry",
    "DynamicDistributionSplitter",
    "DynamicWindowDataset",
    "FaultLicenseStatus",
    "FitAccessLedger",
    "FitAccessRecord",
    "FitPurpose",
    "FiveStageNormalSplitter",
    "FiveStageSplitConfig",
    "FiveStageSplitResult",
    "ImputationDataset",
    "ImputationMaskConfig",
    "ImputationMaskResult",
    "ImputationMasker",
    "InMemoryDataModule",
    "MPCWindowDataset",
    "MPCWindowSample",
    "MissingValueProcessor",
    "Normalizer",
    "OutlierConfig",
    "OutlierProcessor",
    "PaperDataBundle",
    "PipelineStep",
    "ProtocolAccessError",
    "SequentialSplitter",
    "SequenceDataset",
    "SequenceSample",
    "Segment",
    "SegmentInfo",
    "SUPPORTED_SOURCE_SUFFIXES",
    "SourceData",
    "SplitResult",
    "StageName",
    "StageSlice",
    "TabularSeries",
    "TaskSchema",
    "TaskView",
    "list_dataset_presets",
    "merge_pipeline_configs",
    "normalize_pipeline_config",
    "read_mat_arrays",
    "read_npz_arrays",
    "read_source",
    "read_source_frame",
    "register_dataset_adapter",
    "split_source_xy",
]
