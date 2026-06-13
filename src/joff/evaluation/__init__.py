"""Evaluation public API."""

from joff.core.registry import EVALUATOR_REGISTRY

from .classification import ClassificationEvaluator, ClassificationReport
from .fault_detection import FaultDetectionEvaluator, FaultDetectionReport, reconstruction_scores
from .koopman import KoopmanContributionEvaluator, KoopmanContributionReport
from .metrics import MetricReport, ReconstructionEvaluator, RegressionEvaluator

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
    "ClassificationEvaluator",
    "ClassificationReport",
    "FaultDetectionEvaluator",
    "FaultDetectionReport",
    "KoopmanContributionEvaluator",
    "KoopmanContributionReport",
    "MetricReport",
    "ReconstructionEvaluator",
    "RegressionEvaluator",
    "reconstruction_scores",
]
