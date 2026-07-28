"""实验编排层的公共 API。

文件用途：
    汇总通用单次实验、Study/Runner、结果监视器，以及独立的论文五阶段正常协议接口，
    让调用方无需依赖包内文件布局。
主要职责：
    只重导出已经实现的配置、编排器、结果对象和 P3 基线构建/恢复函数；不拥有实验状态，
    不替代 ``Experiment`` 或 ``PaperProtocolExperiment`` 的生命周期。
关键输入与输出：
    本模块没有运行时输入输出；导出的对象分别接收通用配置或 P2 ``PaperDataBundle``，
    返回运行结果、checkpoint 和审计产物。
依赖与副作用：
    导入实验与 P3 基线模块。导入本身不读配置/数据、不创建运行目录、不训练模型，也不
    修改 Matplotlib 或追踪后端全局状态。
重要约束：
    通用 ``Experiment.run()`` 保持 train/test 语义；论文五阶段流程只能通过独立
    ``PaperProtocolExperiment`` 表达，不能把校准段伪装成普通测试集。
"""

from .experiment import Experiment, ExperimentResult
from .monitor import BestResultMonitor, MonitorDecision
from .paper_baselines import (
    BaselineFitResult,
    BaselineScoreBatch,
    PaperBaseline,
    PaperBaselineConfig,
    build_paper_baseline,
    load_paper_baseline,
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
from .runner import ExperimentRunner, ExperimentRunnerResult
from .study import Study, StudyResult

__all__ = [
    "BaselineFitResult",
    "BaselineScoreBatch",
    "BestResultMonitor",
    "EpisodeCalibrationResult",
    "EpisodeMaximumCalibrator",
    "Experiment",
    "ExperimentResult",
    "ExperimentRunner",
    "ExperimentRunnerResult",
    "MonitoringScoreScaler",
    "MonitorDecision",
    "PaperBaseline",
    "PaperBaselineConfig",
    "PaperProtocolConfig",
    "PaperProtocolExperiment",
    "PaperProtocolResult",
    "ResolvedPaperProtocolConfig",
    "StaticThreshold",
    "Study",
    "StudyResult",
    "build_paper_baseline",
    "load_paper_baseline",
    "resolve_paper_protocol_config",
]
