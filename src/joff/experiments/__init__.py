"""实验编排层的公共 API。

文件用途：
    汇总通用单次实验、Study/Runner、P3 五阶段正常协议，以及 P10 CSTR manifest、一次性
    source/workflow、严格入口配置和 synthetic smoke，让调用方无需依赖包内文件布局。
主要职责：
    只重导出已经实现的配置、编排器、结果对象、P3 基线和 P10 冻结边界；不拥有实验状态，
    不替代 ``Experiment``、``PaperProtocolExperiment`` 或
    ``FrozenEvaluationWorkflow`` 的生命周期。
关键输入与输出：
    本模块没有运行时输入输出；导出的对象分别接收通用配置、P2 ``PaperDataBundle``、
    正常产物路径或冻结 manifest，返回运行结果、checkpoint、逐时刻来源和完整性 receipt。
依赖与副作用：
    导入实验、P3 和 P10 模块。导入本身不读配置/数据、不检查 MAT、不创建运行目录、不
    占用 evaluation ID、不训练模型，也不修改 Matplotlib 或追踪后端全局状态。
重要约束：
    通用 ``Experiment.run()`` 保持 train/test 语义；论文五阶段流程只能通过独立
    ``PaperProtocolExperiment`` 表达，不能把校准段伪装成普通测试集。正式故障数据只能
    经已冻结 manifest、持久化 claim 和 verified source 进入 P10 evaluator；smoke 接口
    不能被当作论文方法或正式结果。
"""

from .cstr_frozen_source import (
    CSTRArchiveInspection,
    CSTRClosedLoopEpisodeLoader,
    ManifestBoundCSTRFaultSource,
    inspect_closed_loop_cstr_archive,
)
from .experiment import Experiment, ExperimentResult
from .frozen_evaluation import (
    FrozenEpisodeEvaluation,
    FrozenEpisodeEvaluator,
    FrozenEpisodeInput,
    FrozenEvaluationAlreadyClaimedError,
    FrozenEvaluationArtifactError,
    FrozenEvaluationClaim,
    FrozenEvaluationResult,
    FrozenEvaluationWorkflow,
    FrozenFaultEpisode,
    FrozenFaultEpisodeManifest,
    FrozenFaultEpisodeSource,
    FrozenNormalArtifactBundle,
    FrozenPointwiseOutput,
    FrozenProtocolIntegrityError,
    FrozenProtocolManifest,
    FrozenRiskCalibration,
    FrozenRuntimeEpisodeEvaluation,
    FrozenRuntimePointwiseOutput,
    LazyFrozenCSTRFaultSource,
    verify_frozen_evaluation_artifacts,
)
from .monitor import BestResultMonitor, MonitorDecision
from .paper_baselines import (
    BaselineFitResult,
    BaselineScoreBatch,
    PaperBaseline,
    PaperBaselineConfig,
    build_paper_baseline,
    load_paper_baseline,
)
from .paper_entrypoints import (
    FrozenEvaluationEntryConfig,
    PaperDevelopmentConfig,
    PaperDevelopmentFeatureLayoutConfig,
    PaperDevelopmentTrainingConfig,
    PaperEvaluationDatasetConfig,
    PaperEvaluationSeedsConfig,
    PaperNormalArtifactsConfig,
    PaperNormalMethodConfig,
    ResolvedFrozenEvaluationConfig,
    resolve_frozen_evaluation_config,
)
from .paper_development import (
    PaperDevelopmentResult,
    run_cstr_normal_development,
)
from .paper_freeze import (
    build_cstr_protocol_from_artifacts,
    freeze_cstr_protocol_from_artifacts,
)
from .paper_protocol import (
    EpisodeCalibrationResult,
    EpisodeMaximumCalibrator,
    MonitoringScoreScaler,
    PaperProtocolConfig,
    PaperProtocolExperiment,
    PaperProtocolResult,
    ResolvedPaperProtocolConfig,
    StaticThreshold,
    resolve_paper_protocol_config,
)
from .paper_smoke import run_paper_smoke
from .runner import ExperimentRunner, ExperimentRunnerResult
from .study import Study, StudyResult

__all__ = [
    "BaselineFitResult",
    "BaselineScoreBatch",
    "BestResultMonitor",
    "CSTRArchiveInspection",
    "CSTRClosedLoopEpisodeLoader",
    "EpisodeCalibrationResult",
    "EpisodeMaximumCalibrator",
    "Experiment",
    "ExperimentResult",
    "ExperimentRunner",
    "ExperimentRunnerResult",
    "FrozenEpisodeEvaluation",
    "FrozenEpisodeEvaluator",
    "FrozenEpisodeInput",
    "FrozenEvaluationAlreadyClaimedError",
    "FrozenEvaluationArtifactError",
    "FrozenEvaluationClaim",
    "FrozenEvaluationEntryConfig",
    "PaperDevelopmentConfig",
    "PaperDevelopmentFeatureLayoutConfig",
    "PaperDevelopmentTrainingConfig",
    "PaperDevelopmentResult",
    "FrozenEvaluationResult",
    "FrozenEvaluationWorkflow",
    "FrozenFaultEpisode",
    "FrozenFaultEpisodeManifest",
    "FrozenFaultEpisodeSource",
    "FrozenNormalArtifactBundle",
    "FrozenPointwiseOutput",
    "FrozenProtocolIntegrityError",
    "FrozenProtocolManifest",
    "FrozenRiskCalibration",
    "FrozenRuntimeEpisodeEvaluation",
    "FrozenRuntimePointwiseOutput",
    "LazyFrozenCSTRFaultSource",
    "MonitoringScoreScaler",
    "ManifestBoundCSTRFaultSource",
    "MonitorDecision",
    "PaperBaseline",
    "PaperBaselineConfig",
    "PaperEvaluationDatasetConfig",
    "PaperEvaluationSeedsConfig",
    "PaperNormalArtifactsConfig",
    "PaperNormalMethodConfig",
    "PaperProtocolConfig",
    "PaperProtocolExperiment",
    "PaperProtocolResult",
    "ResolvedPaperProtocolConfig",
    "ResolvedFrozenEvaluationConfig",
    "StaticThreshold",
    "Study",
    "StudyResult",
    "build_paper_baseline",
    "build_cstr_protocol_from_artifacts",
    "freeze_cstr_protocol_from_artifacts",
    "load_paper_baseline",
    "inspect_closed_loop_cstr_archive",
    "resolve_paper_protocol_config",
    "resolve_frozen_evaluation_config",
    "run_cstr_normal_development",
    "run_paper_smoke",
    "verify_frozen_evaluation_artifacts",
]
