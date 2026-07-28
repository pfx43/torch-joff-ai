"""Joff 评估层的公开注册表、报告与论文监视接口。

文件用途：
    汇总通用回归、分类、故障检测、Koopman 贡献评估器，以及论文 P5 受保护参考对象和
    P6 联合认证堆叠算子、P7 white-space 后滤波、P8 输入调度有限 episode 动态阈值和
    P9 集合值 explanation family，为 ``joff.evaluation`` 提供稳定导入入口。
主要职责：
    注册内置 evaluator 并重导出公开配置、状态和报告；本文件不执行评估、不创建运行
    目录，也不拥有实验或训练状态。
关键输入与输出：
    导入时向 ``EVALUATOR_REGISTRY`` 写入固定内置类型；使用者从本模块获得 evaluator
    类、报告类、受保护监视状态机、资源预算、联合 operator enclosure、冻结分支库、
    estimate 包络、有限 episode 校准和阈值分账对象。
依赖与副作用：
    依赖核心注册表和各评估子模块。唯一副作用是幂等覆盖式注册内置 evaluator；不读取
    数据、网络或文件系统。
重要约束：
    注册名称保持向后兼容；论文监视器不得把最终检测报警反馈到 anchor/mode 状态路径；
    名义算子不能越权授权安全排除，认证算子必须共享一个联合不确定系数空间；P8 最终
    detection quantile 不得进入生成自身 calibration scores 的冻结 score map。
"""

from joff.core.registry import EVALUATOR_REGISTRY

from .classification import ClassificationEvaluator, ClassificationReport
from .dynamic_threshold import (
    CalibrationStatus,
    ContextAgeEnvelope,
    DetectionScore,
    DeterministicRadius,
    DeterministicRadiusGenerator,
    DynamicThresholdGenerator,
    EnvelopeEvaluation,
    EpisodeMaxCalibrator,
    InputDependentEnvelope,
    InputDescriptor,
    ScoreCoordinate,
    ThresholdResult,
    ThresholdStatus,
)
from .fault_detection import FaultDetectionEvaluator, FaultDetectionReport, reconstruction_scores
from .explanations import (
    DeployedBranchEvidence,
    DeployedObservation,
    DynamicsSide,
    ExplanationFamily,
    MaskRecomputation,
)
from .koopman import KoopmanContributionEvaluator, KoopmanContributionReport
from .metrics import MetricReport, ReconstructionEvaluator, RegressionEvaluator
from .oracle import (
    LinearExplanationCell,
    MonotoneRefinementCache,
    OracleCellRefinement,
    OracleEvaluation,
    OuterExplanationOracle,
)
from .postfilter import (
    BranchBank,
    BranchKind,
    BranchOperator,
    PostFilterCandidate,
    SpectralMode,
    WhiteningEstimate,
)
from .protected_operators import (
    CertifiedEnclosureProvider,
    JacobianSemantics,
    NominalJVPAssembler,
    OperatorAffineImage,
    OperatorAssemblyBudget,
    OperatorBundle,
    OperatorCertificationRequest,
    OperatorEnclosure,
    OperatorNorm,
    OperatorPath,
    OperatorStatus,
    UncertifiedOperatorError,
)
from .protected_reference import (
    AnchorCoverageStatus,
    AnchorGateConfig,
    MonitorMode,
    MonitorRecord,
    MonitorStage,
    MonitorState,
    MonitorTrace,
    MonitorTraceEntry,
    ProtectedRollout,
    ProtectedMonitor,
)
from .residuals import StackedProtectedResidual
from .structured_isolation import (
    AttributionCalibrationStatus,
    FullNormalCalibrator,
    IsolationCandidateSet,
    IsolationOutcome,
    IsolationReport,
)

EVALUATOR_REGISTRY.register("regression", RegressionEvaluator, replace=True)
EVALUATOR_REGISTRY.register("reconstruction", ReconstructionEvaluator, replace=True)
EVALUATOR_REGISTRY.register("classification", ClassificationEvaluator, replace=True)
EVALUATOR_REGISTRY.register("fault_detection", FaultDetectionEvaluator, aliases=("fd",), replace=True)
EVALUATOR_REGISTRY.register(
    "koopman_contribution",
    KoopmanContributionEvaluator,
    aliases=("koopman", "nkn_contribution"),
    replace=True,
)

__all__ = [
    "AnchorCoverageStatus",
    "AnchorGateConfig",
    "AttributionCalibrationStatus",
    "BranchBank",
    "BranchKind",
    "BranchOperator",
    "CalibrationStatus",
    "ClassificationEvaluator",
    "ClassificationReport",
    "CertifiedEnclosureProvider",
    "ContextAgeEnvelope",
    "DetectionScore",
    "DeployedBranchEvidence",
    "DeployedObservation",
    "DeterministicRadius",
    "DeterministicRadiusGenerator",
    "DynamicThresholdGenerator",
    "DynamicsSide",
    "EnvelopeEvaluation",
    "EpisodeMaxCalibrator",
    "ExplanationFamily",
    "FaultDetectionEvaluator",
    "FaultDetectionReport",
    "FullNormalCalibrator",
    "JacobianSemantics",
    "InputDependentEnvelope",
    "InputDescriptor",
    "IsolationCandidateSet",
    "IsolationOutcome",
    "IsolationReport",
    "KoopmanContributionEvaluator",
    "KoopmanContributionReport",
    "LinearExplanationCell",
    "MetricReport",
    "MaskRecomputation",
    "MonotoneRefinementCache",
    "MonitorMode",
    "MonitorRecord",
    "MonitorStage",
    "MonitorState",
    "MonitorTrace",
    "MonitorTraceEntry",
    "NominalJVPAssembler",
    "OperatorAffineImage",
    "OperatorAssemblyBudget",
    "OperatorBundle",
    "OperatorCertificationRequest",
    "OperatorEnclosure",
    "OperatorNorm",
    "OperatorPath",
    "OperatorStatus",
    "OracleCellRefinement",
    "OracleEvaluation",
    "OuterExplanationOracle",
    "PostFilterCandidate",
    "ProtectedRollout",
    "ProtectedMonitor",
    "ReconstructionEvaluator",
    "RegressionEvaluator",
    "StackedProtectedResidual",
    "SpectralMode",
    "ScoreCoordinate",
    "ThresholdResult",
    "ThresholdStatus",
    "UncertifiedOperatorError",
    "WhiteningEstimate",
    "reconstruction_scores",
]
