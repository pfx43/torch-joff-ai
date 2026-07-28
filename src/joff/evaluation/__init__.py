"""Joff 评估层的公开注册表、报告与论文监视接口。

文件用途：
    汇总通用回归、分类、故障检测、Koopman 贡献评估器，以及论文 P5 受保护参考对象和
    P6 联合认证堆叠算子和 P7 white-space 后滤波，为 ``joff.evaluation`` 提供稳定导入
    入口。
主要职责：
    注册内置 evaluator 并重导出公开配置、状态和报告；本文件不执行评估、不创建运行
    目录，也不拥有实验或训练状态。
关键输入与输出：
    导入时向 ``EVALUATOR_REGISTRY`` 写入固定内置类型；使用者从本模块获得 evaluator
    类、报告类、受保护监视状态机、资源预算、联合 operator enclosure 和冻结分支库。
依赖与副作用：
    依赖核心注册表和各评估子模块。唯一副作用是幂等覆盖式注册内置 evaluator；不读取
    数据、网络或文件系统。
重要约束：
    注册名称保持向后兼容；论文监视器不得把最终检测报警反馈到 anchor/mode 状态路径；
    名义算子不能越权授权安全排除，认证算子必须共享一个联合不确定系数空间。
"""

from joff.core.registry import EVALUATOR_REGISTRY

from .classification import ClassificationEvaluator, ClassificationReport
from .fault_detection import FaultDetectionEvaluator, FaultDetectionReport, reconstruction_scores
from .koopman import KoopmanContributionEvaluator, KoopmanContributionReport
from .metrics import MetricReport, ReconstructionEvaluator, RegressionEvaluator
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
    "BranchBank",
    "BranchKind",
    "BranchOperator",
    "ClassificationEvaluator",
    "ClassificationReport",
    "CertifiedEnclosureProvider",
    "FaultDetectionEvaluator",
    "FaultDetectionReport",
    "JacobianSemantics",
    "KoopmanContributionEvaluator",
    "KoopmanContributionReport",
    "MetricReport",
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
    "PostFilterCandidate",
    "ProtectedRollout",
    "ProtectedMonitor",
    "ReconstructionEvaluator",
    "RegressionEvaluator",
    "StackedProtectedResidual",
    "SpectralMode",
    "UncertifiedOperatorError",
    "WhiteningEstimate",
    "reconstruction_scores",
]
