"""P9 full-normal 归因校准与集合值结构化隔离编排。

文件用途：
    在已冻结 detection calibration 之后，只用独立正常 attribution-calibration episode
    重放完整 observation/oracle 流程，并冻结 episode family-wise normal attribution
    分位 ``q_attr``。
主要职责：
    定义归因校准受控状态、``FullNormalCalibrator``、隔离候选集和受控报告语义；本文件
    当前不实现 full nonlinear oracle 后端或最终论文统计汇总。
关键输入与输出：
    输入为完整 ``DeployedObservation`` episode、normal ``ExplanationFamily``、冻结
    ``OuterExplanationOracle`` 及 detection/attribution 分区身份；输出为逐 episode
    maximum、有限秩、``q_attr``、候选 explanation 集及受控隔离报告。
依赖与副作用：
    依赖 Python 标准库以及 P6--P9 评估领域对象；不读取原始数据或网络，不运行训练，
    不修改 monitor/oracle，也不使用故障 episode。
重要约束：
    attribution calibration 必须与 detection calibration 来源和 episode 均不重用；
    每条观测必须属于真实 attribution stage 并绑定同一冻结 detection calibration；
    ``beta < 1/(n_attr+1)`` 时分位为正无穷，Normal 不可排除且非正常 singleton 禁用。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from .explanations import DeployedObservation, ExplanationFamily
from .oracle import OracleEvaluation, OuterExplanationOracle
from .protected_reference import MonitorStage


class AttributionCalibrationStatus(str, Enum):
    """full-normal attribution calibration 的受控结果状态。

    参数：
        value: ``ready``、``insufficient_resolution`` 或 ``incomplete_evidence``。
    返回：
        对应枚举成员。
    异常：
        未知字符串由 ``Enum`` 抛出 ``ValueError``。
    副作用：
        无。
    """

    READY = "ready"
    INSUFFICIENT_RESOLUTION = "insufficient_resolution"
    INCOMPLETE_EVIDENCE = "incomplete_evidence"


class IsolationOutcome(str, Enum):
    """集合值隔离报告允许出现的受控语义。

    参数：
        value: Normal-compatible、Nonunique、Out-of-model、singleton 或 Uncertified。
    返回：
        对应枚举成员。
    异常：
        未知字符串由 ``Enum`` 抛出 ``ValueError``。
    副作用：
        无。
    """

    NORMAL_COMPATIBLE = "Normal-compatible"
    NONUNIQUE = "Nonunique"
    OUT_OF_MODEL = "Out-of-model"
    SINGLETON = "singleton"
    UNCERTIFIED = "Uncertified"


_ISOLATION_RISK_STATEMENT = (
    "The episode guarantee controls family-wise Normal exclusion at beta; "
    "fault-class coverage remains conditional on declared signature and nuisance coverage."
)


@dataclass(frozen=True)
class FullNormalCalibrator:
    """冻结完整 normal attribution observation 的 episode-maximum 分位。

    参数：
        stage: 必须为 ``ATTRIBUTION_CALIBRATION``。
        beta/detection_alpha: normal 归因误排预算与已冻结 detection 风险预算，满足
            ``0 < beta < detection_alpha < 1``。
        detection_quantile/detection_calibration_hash: 已先冻结的有限 ``q_det`` 与身份。
        detection_source_hash/detection_episode_ids: detection calibration 的独立来源和
            episode 身份，用于拒绝归因数据复用。
        attribution_source_hash: 本次仅正常归因分区的来源身份。
        normal_family/normal_family_hash: nuisance-only Normal explanation 及内容身份。
        oracle/oracle_hash: 在所有 selectable isolation time 重放的冻结 outer oracle。
        episode_definition_hash/exchangeability_assumption_hash: 有限 episode 与风险前提。
        episode_ids/episode_observations/episode_scores/episode_maxima: 完整正常观测、逐时刻
            normal score 及其 episode maximum。
        rank/quantile/status/reason: 有限秩结果；分辨率不足或证据不完整时使用 ``+inf``。
    返回：
        ``fit`` 产生可审计校准；属性 ``nonnormal_singleton_enabled`` 给出全局分辨率门禁。
    异常：
        stage、来源、episode、哈希、风险、完整观测或派生分位不一致时抛出
        ``TypeError``/``ValueError``。
    副作用：
        无；构造和拟合只重放内存中的冻结线性 oracle，不读取故障数据。
    """

    stage: MonitorStage
    beta: float
    detection_alpha: float
    detection_quantile: float
    detection_calibration_hash: str
    detection_source_hash: str
    detection_episode_ids: tuple[str, ...]
    attribution_source_hash: str
    normal_family: ExplanationFamily
    normal_family_hash: str
    oracle: OuterExplanationOracle
    oracle_hash: str
    episode_definition_hash: str
    exchangeability_assumption_hash: str
    episode_ids: tuple[str, ...]
    episode_observations: tuple[tuple[DeployedObservation, ...], ...]
    episode_scores: tuple[tuple[float, ...], ...]
    episode_maxima: tuple[float, ...]
    rank: int
    quantile: float
    status: AttributionCalibrationStatus
    reason: str | None
    strict_exceedance: bool = True

    def __post_init__(self) -> None:
        """重放完整 normal score，并验证有限秩、分区和失败关闭状态。"""

        if self.stage is not MonitorStage.ATTRIBUTION_CALIBRATION:
            raise ValueError("Full-normal calibration requires attribution calibration.")
        normalized_beta = _open_unit_interval(self.beta, name="Attribution beta")
        normalized_alpha = _open_unit_interval(
            self.detection_alpha,
            name="Detection alpha",
        )
        if normalized_beta >= normalized_alpha:
            raise ValueError("Attribution beta must be strictly smaller than detection alpha.")
        object.__setattr__(self, "beta", normalized_beta)
        object.__setattr__(self, "detection_alpha", normalized_alpha)
        if (
            isinstance(self.detection_quantile, bool)
            or not isinstance(self.detection_quantile, (int, float))
            or not math.isfinite(float(self.detection_quantile))
            or float(self.detection_quantile) < 0.0
        ):
            raise ValueError("Frozen detection quantile must be finite and non-negative.")
        object.__setattr__(
            self,
            "detection_quantile",
            float(self.detection_quantile),
        )
        if not all(
            _is_sha256(value)
            for value in (
                self.detection_calibration_hash,
                self.detection_source_hash,
                self.attribution_source_hash,
                self.normal_family_hash,
                self.oracle_hash,
                self.episode_definition_hash,
                self.exchangeability_assumption_hash,
            )
        ):
            raise ValueError("Full-normal calibration identities must be SHA-256.")
        if self.detection_source_hash == self.attribution_source_hash:
            raise ValueError(
                "Detection and attribution calibration sources must be independent."
            )
        if (
            not isinstance(self.detection_episode_ids, tuple)
            or not self.detection_episode_ids
            or any(
                not isinstance(episode_id, str) or not episode_id
                for episode_id in self.detection_episode_ids
            )
            or len(set(self.detection_episode_ids)) != len(self.detection_episode_ids)
            or self.detection_episode_ids != tuple(sorted(self.detection_episode_ids))
        ):
            raise ValueError("Detection calibration episode ids must be unique and sorted.")
        if not isinstance(self.normal_family, ExplanationFamily) or not self.normal_family.normal:
            raise ValueError("Full-normal calibration requires the Normal explanation family.")
        if self.normal_family_hash != self.normal_family.content_hash:
            raise ValueError("Stored Normal explanation identity does not match its family.")
        if not isinstance(self.oracle, OuterExplanationOracle):
            raise TypeError("Full-normal calibration requires an OuterExplanationOracle.")
        if self.oracle_hash != self.oracle.content_hash:
            raise ValueError("Stored attribution oracle identity does not match its snapshot.")
        if (
            not isinstance(self.episode_ids, tuple)
            or not self.episode_ids
            or len(set(self.episode_ids)) != len(self.episode_ids)
            or self.episode_ids != tuple(sorted(self.episode_ids))
            or not isinstance(self.episode_observations, tuple)
            or not isinstance(self.episode_scores, tuple)
            or not isinstance(self.episode_maxima, tuple)
            or not (
                len(self.episode_ids)
                == len(self.episode_observations)
                == len(self.episode_scores)
                == len(self.episode_maxima)
            )
        ):
            raise ValueError(
                "Attribution episode ids, observations, scores, and maxima must align."
            )
        if set(self.episode_ids) & set(self.detection_episode_ids):
            raise ValueError(
                "Detection and attribution calibration episode ids must not overlap."
            )

        incomplete_reasons: list[str] = []
        for episode_id, observations, scores, stored_maximum in zip(
            self.episode_ids,
            self.episode_observations,
            self.episode_scores,
            self.episode_maxima,
            strict=True,
        ):
            if (
                not isinstance(episode_id, str)
                or not episode_id
                or not isinstance(observations, tuple)
                or not observations
                or not all(
                    isinstance(observation, DeployedObservation)
                    for observation in observations
                )
                or not isinstance(scores, tuple)
                or len(scores) != len(observations)
            ):
                raise ValueError(
                    "Every attribution episode requires aligned complete observations and scores."
                )
            recomputed_scores: list[float] = []
            for observation in observations:
                if (
                    observation.monitor_state.episode_id != episode_id
                    or observation.monitor_state.stage
                    is not MonitorStage.ATTRIBUTION_CALIBRATION
                ):
                    raise ValueError(
                        "Attribution observation episode and stage must match its partition."
                    )
                if (
                    observation.detection_calibration_hash
                    != self.detection_calibration_hash
                ):
                    raise ValueError(
                        "Attribution observation must use the frozen detection calibration."
                    )
                evaluation = self.oracle.evaluate(
                    observation,
                    self.normal_family,
                    radius=float("inf"),
                )
                if not evaluation.certified or evaluation.unresolved_cell_ids:
                    incomplete_reasons.append(
                        f"episode {episode_id!r} has unresolved normal oracle evidence"
                    )
                recomputed_scores.append(evaluation.outer_score)
            normalized_scores = tuple(
                _nonnegative_finite_score(value, name="Stored attribution score")
                for value in scores
            )
            if tuple(recomputed_scores) != normalized_scores:
                raise ValueError(
                    "Attribution scores must be derived from their complete observations."
                )
            computed_maximum = max(recomputed_scores)
            if (
                isinstance(stored_maximum, bool)
                or not isinstance(stored_maximum, (int, float))
                or float(stored_maximum) != computed_maximum
            ):
                raise ValueError(
                    "Attribution episode maximum must be derived from its stored scores."
                )

        expected_rank = math.ceil(
            (len(self.episode_maxima) + 1) * (1.0 - self.beta)
        )
        if type(self.rank) is not int or self.rank != expected_rank:
            raise ValueError("Attribution rank must follow the finite-sample conformal rule.")
        if (
            not isinstance(self.status, AttributionCalibrationStatus)
            or self.strict_exceedance is not True
        ):
            raise ValueError("Attribution status/strict-exceedance contract is invalid.")
        if self.status is AttributionCalibrationStatus.READY:
            if (
                self.rank > len(self.episode_maxima)
                or incomplete_reasons
                or not math.isfinite(self.quantile)
                or self.reason is not None
            ):
                raise ValueError("READY attribution calibration needs a finite rank quantile.")
            expected_quantile = sorted(self.episode_maxima)[self.rank - 1]
            if self.quantile != expected_quantile:
                raise ValueError(
                    "Attribution quantile must equal its episode order statistic."
                )
        elif self.status is AttributionCalibrationStatus.INSUFFICIENT_RESOLUTION:
            if (
                self.rank != len(self.episode_maxima) + 1
                or incomplete_reasons
                or self.quantile != float("inf")
                or not self.reason
            ):
                raise ValueError(
                    "Insufficient attribution resolution requires complete evidence and +inf."
                )
        elif (
            not incomplete_reasons
            or self.quantile != float("inf")
            or not self.reason
        ):
            raise ValueError(
                "Incomplete attribution evidence must disable exclusion with +inf."
            )

    @property
    def risk_resolution(self) -> float:
        """返回 ``n_attr`` 可实现的最小 episode 风险 ``1/(n_attr+1)``。

        参数：
            无。
        返回：
            严格正浮点值。
        异常：
            无。
        副作用：
            无。
        """

        return 1.0 / (len(self.episode_maxima) + 1)

    @property
    def nonnormal_singleton_enabled(self) -> bool:
        """返回有限样本分辨率是否允许排除 Normal 并形成非正常 singleton。

        参数：
            无。
        返回：
            仅 ``READY`` 且 ``q_attr`` 有限时返回 ``True``。
        异常：
            无。
        副作用：
            无。
        """

        return (
            self.status is AttributionCalibrationStatus.READY
            and math.isfinite(self.quantile)
        )

    @classmethod
    def fit(
        cls,
        episodes: Mapping[str, Sequence[DeployedObservation]],
        *,
        normal_family: ExplanationFamily,
        oracle: OuterExplanationOracle,
        stage: MonitorStage | str,
        beta: float,
        detection_alpha: float,
        detection_quantile: float,
        detection_calibration_hash: str,
        detection_source_hash: str,
        detection_episode_ids: Sequence[str],
        attribution_source_hash: str,
        episode_definition_hash: str,
        exchangeability_assumption_hash: str,
    ) -> "FullNormalCalibrator":
        """重放完整 normal oracle，取 episode maximum 并应用有限秩规则。

        参数：
            episodes: ``episode_id -> DeployedObservation`` 序列；序列覆盖所有 selectable
                isolation time，且观测必须来自 attribution calibration stage。
            normal_family/oracle: 冻结 nuisance-only Normal explanation 与 outer oracle。
            stage/beta: 固定归因 stage 与 episode family-wise normal 风险预算。
            detection_alpha/detection_quantile/detection_calibration_hash: 已先冻结 detection
                calibration 的风险、有限分位和身份。
            detection_source_hash/detection_episode_ids/attribution_source_hash: 两次校准的
                来源和 episode 身份；两类都不得复用。
            episode_definition_hash/exchangeability_assumption_hash: 有限 episode 和风险前提。
        返回：
            READY、INSUFFICIENT_RESOLUTION 或 INCOMPLETE_EVIDENCE 校准对象。
        异常：
            容器、stage、身份、观测或风险非法时抛出 ``TypeError``/``ValueError``。
        副作用：
            无；只调用冻结 oracle 的确定性 ``evaluate``。
        """

        normalized_stage = MonitorStage.parse(stage)
        if normalized_stage is not MonitorStage.ATTRIBUTION_CALIBRATION:
            raise ValueError(
                "Full-normal fit may only use the attribution calibration stage."
            )
        if not isinstance(episodes, Mapping) or not episodes:
            raise TypeError("Attribution calibration episodes must be a non-empty mapping.")
        if not isinstance(normal_family, ExplanationFamily) or not normal_family.normal:
            raise ValueError("Full-normal fit requires the Normal explanation family.")
        if not isinstance(oracle, OuterExplanationOracle):
            raise TypeError("Full-normal fit requires an OuterExplanationOracle.")
        normalized_beta = _open_unit_interval(beta, name="Attribution beta")
        normalized_alpha = _open_unit_interval(
            detection_alpha,
            name="Detection alpha",
        )
        if normalized_beta >= normalized_alpha:
            raise ValueError("Attribution beta must be strictly smaller than detection alpha.")
        normalized_detection_ids = _sorted_unique_ids(
            detection_episode_ids,
            name="Detection calibration episode ids",
        )
        episode_ids = tuple(sorted(episodes))
        observations_by_episode: list[tuple[DeployedObservation, ...]] = []
        scores_by_episode: list[tuple[float, ...]] = []
        maxima: list[float] = []
        incomplete_reasons: list[str] = []
        for episode_id in episode_ids:
            if not isinstance(episode_id, str) or not episode_id:
                raise ValueError("Attribution episode ids must be non-empty strings.")
            observations = tuple(episodes[episode_id])
            if not observations or not all(
                isinstance(observation, DeployedObservation)
                for observation in observations
            ):
                raise TypeError(
                    "Every attribution episode must contain DeployedObservation values."
                )
            scores: list[float] = []
            for observation in observations:
                evaluation = oracle.evaluate(
                    observation,
                    normal_family,
                    radius=float("inf"),
                )
                if not evaluation.certified or evaluation.unresolved_cell_ids:
                    incomplete_reasons.append(
                        f"episode {episode_id!r} has unresolved normal oracle evidence"
                    )
                scores.append(evaluation.outer_score)
            observations_by_episode.append(observations)
            scores_by_episode.append(tuple(scores))
            maxima.append(max(scores))

        rank = math.ceil((len(maxima) + 1) * (1.0 - normalized_beta))
        if incomplete_reasons:
            status = AttributionCalibrationStatus.INCOMPLETE_EVIDENCE
            quantile = float("inf")
            reason = "; ".join(dict.fromkeys(incomplete_reasons))
        elif rank == len(maxima) + 1:
            status = AttributionCalibrationStatus.INSUFFICIENT_RESOLUTION
            quantile = float("inf")
            reason = (
                f"Requested beta {normalized_beta:.17g} is below finite resolution "
                f"{1.0 / (len(maxima) + 1):.17g} for {len(maxima)} episodes."
            )
        else:
            status = AttributionCalibrationStatus.READY
            quantile = sorted(maxima)[rank - 1]
            reason = None
        return cls(
            stage=normalized_stage,
            beta=normalized_beta,
            detection_alpha=normalized_alpha,
            detection_quantile=detection_quantile,
            detection_calibration_hash=detection_calibration_hash,
            detection_source_hash=detection_source_hash,
            detection_episode_ids=normalized_detection_ids,
            attribution_source_hash=attribution_source_hash,
            normal_family=normal_family,
            normal_family_hash=normal_family.content_hash,
            oracle=oracle,
            oracle_hash=oracle.content_hash,
            episode_definition_hash=episode_definition_hash,
            exchangeability_assumption_hash=exchangeability_assumption_hash,
            episode_ids=episode_ids,
            episode_observations=tuple(observations_by_episode),
            episode_scores=tuple(scores_by_episode),
            episode_maxima=tuple(maxima),
            rank=rank,
            quantile=float(quantile),
            status=status,
            reason=reason,
        )


@dataclass(frozen=True)
class IsolationCandidateSet:
    """一个完整部署观测对应的保守 explanation 候选集。

    参数：
        observation_hash/oracle_hash/detection_calibration_hash: 完整观测和冻结判决身份。
        observation/normal_calibration: 允许构造器重放 detection gate、``q_attr`` 和全部
            oracle 查询的权威快照。
        normal_family_id: nuisance-only Normal explanation 的稳定身份。
        candidate_families: 右闭 outer 查询后保留的 explanation；若保留 Normal 则固定首位。
        evaluations: Normal 与全部 fault family 的 oracle 结果，而不只保存最终候选。
        detection_normal_compatible: 当前观测是否仍在冻结 detection 接受域内。
        nonnormal_singleton_enabled: attribution 有限样本分辨率是否允许排除 Normal。
        declared_families: 查询时冻结的完整 Normal/fault family 字典；Normal 固定首位，
            fault family 按 family_id 排序。
        declared_family_hashes: oracle 冻结字典内全部 Normal/fault family 的内容哈希；用于
            验证 evaluations 没有遗漏竞争解释。
    返回：
        ``evaluate`` 组合 nested normal set 与 fault outer family 后产生不可变候选集。
    异常：
        身份、候选顺序、重复 family 或派生门禁不一致时抛出 ``TypeError``/``ValueError``。
    副作用：
        无。
    """

    observation_hash: str
    oracle_hash: str
    detection_calibration_hash: str
    observation: DeployedObservation
    normal_calibration: FullNormalCalibrator
    normal_family_id: str
    candidate_families: tuple[ExplanationFamily, ...]
    evaluations: tuple[OracleEvaluation, ...]
    detection_normal_compatible: bool
    nonnormal_singleton_enabled: bool
    declared_families: tuple[ExplanationFamily, ...]
    declared_family_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        """验证候选和全部 oracle 结果都绑定同一完整观测。"""

        if not all(
            _is_sha256(value)
            for value in (
                self.observation_hash,
                self.oracle_hash,
                self.detection_calibration_hash,
            )
        ):
            raise ValueError("Isolation candidate identities must be SHA-256.")
        if not isinstance(self.observation, DeployedObservation):
            raise TypeError("Isolation candidate requires its complete observation snapshot.")
        if not isinstance(self.normal_calibration, FullNormalCalibrator):
            raise TypeError("Isolation candidate requires its full-normal calibration snapshot.")
        if (
            self.observation_hash != self.observation.content_hash
            or self.oracle_hash != self.normal_calibration.oracle_hash
            or self.detection_calibration_hash
            != self.normal_calibration.detection_calibration_hash
            or self.observation.detection_calibration_hash
            != self.detection_calibration_hash
        ):
            raise ValueError(
                "Isolation candidate snapshots must match their frozen identities."
            )
        if not isinstance(self.normal_family_id, str) or not self.normal_family_id:
            raise ValueError("Isolation candidate Normal family id must be non-empty.")
        if self.normal_family_id != self.normal_calibration.normal_family.family_id:
            raise ValueError(
                "Isolation candidate Normal family must match the frozen calibration."
            )
        if (
            not isinstance(self.candidate_families, tuple)
            or not all(
                isinstance(family, ExplanationFamily)
                for family in self.candidate_families
            )
            or len({family.family_id for family in self.candidate_families})
            != len(self.candidate_families)
        ):
            raise ValueError("Isolation candidate families must be unique.")
        candidate_ids = tuple(family.family_id for family in self.candidate_families)
        if self.normal_family_id in candidate_ids and candidate_ids[0] != self.normal_family_id:
            raise ValueError("Normal explanation must be first when it remains a candidate.")
        if (
            not isinstance(self.declared_families, tuple)
            or not self.declared_families
            or not all(
                isinstance(family, ExplanationFamily)
                for family in self.declared_families
            )
            or self.declared_families[0].family_id != self.normal_family_id
            or not self.declared_families[0].normal
            or any(family.normal for family in self.declared_families[1:])
            or tuple(
                family.family_id for family in self.declared_families[1:]
            )
            != tuple(
                sorted(family.family_id for family in self.declared_families[1:])
            )
            or len({family.family_id for family in self.declared_families})
            != len(self.declared_families)
            or len({family.content_hash for family in self.declared_families})
            != len(self.declared_families)
        ):
            raise ValueError(
                "Isolation declared families must contain Normal first and sorted unique faults."
            )
        declared_by_id = {
            family.family_id: family for family in self.declared_families
        }
        if (
            self.declared_families[0] != self.normal_calibration.normal_family
            or set(self.declared_family_hashes)
            != set(self.normal_calibration.oracle.family_hashes)
        ):
            raise ValueError(
                "Isolation declared families must match the frozen calibration oracle."
            )
        if (
            not isinstance(self.evaluations, tuple)
            or not self.evaluations
            or not all(
                isinstance(evaluation, OracleEvaluation)
                for evaluation in self.evaluations
            )
            or any(
                evaluation.observation_hash != self.observation_hash
                for evaluation in self.evaluations
            )
            or len({evaluation.family_id for evaluation in self.evaluations})
            != len(self.evaluations)
            or {evaluation.family_id for evaluation in self.evaluations}
            != set(declared_by_id)
            or any(
                evaluation.oracle_hash != self.oracle_hash
                for evaluation in self.evaluations
            )
        ):
            raise ValueError(
                "Isolation candidate evaluations must cover one observation and frozen oracle."
            )
        evaluation_by_id = {
            evaluation.family_id: evaluation for evaluation in self.evaluations
        }
        normal_evaluation = evaluation_by_id.get(self.normal_family_id)
        if normal_evaluation is None:
            raise ValueError(
                "Isolation evaluations must include the frozen Normal explanation."
            )
        for family_id, family in declared_by_id.items():
            evaluation = evaluation_by_id[family_id]
            if evaluation.family_hash != family.content_hash:
                raise ValueError(
                    "Isolation evaluation family identity must match the frozen dictionary."
                )
            if not family.normal and evaluation.radius != family.radius:
                raise ValueError(
                    "Isolation fault evaluation must use its frozen family radius."
                )
        expected_evaluations = tuple(
            self.normal_calibration.oracle.evaluate(
                self.observation,
                family,
                radius=(
                    self.normal_calibration.quantile
                    if family.normal
                    else family.radius
                ),
            )
            for family in self.declared_families
        )
        if self.evaluations != expected_evaluations:
            raise ValueError(
                "Isolation evaluations must replay the frozen calibration and oracle."
            )
        for family in self.candidate_families:
            candidate_evaluation = evaluation_by_id.get(family.family_id)
            if (
                candidate_evaluation is None
                or declared_by_id.get(family.family_id) != family
                or candidate_evaluation.family_hash != family.content_hash
                or not candidate_evaluation.feasible
            ):
                raise ValueError(
                    "Every isolation candidate must be derived from a feasible oracle result."
                )
        candidate_id_set = set(candidate_ids)
        expected_fault_candidates = {
            evaluation.family_id
            for evaluation in self.evaluations
            if evaluation.family_id != self.normal_family_id
            and evaluation.feasible
        }
        if candidate_id_set - {self.normal_family_id} != expected_fault_candidates:
            raise ValueError(
                "Isolation fault candidates must include every feasible frozen family."
            )
        if (
            type(self.detection_normal_compatible) is not bool
            or type(self.nonnormal_singleton_enabled) is not bool
        ):
            raise TypeError("Isolation candidate gates must be strict booleans.")
        expected_detection_normal_compatible = (
            self.observation.detection_excess
            <= self.normal_calibration.detection_quantile
        )
        if (
            self.detection_normal_compatible
            is not expected_detection_normal_compatible
            or self.nonnormal_singleton_enabled
            is not self.normal_calibration.nonnormal_singleton_enabled
        ):
            raise ValueError(
                "Isolation gates must replay the frozen calibration and observation."
            )
        normal_expected = (
            self.detection_normal_compatible
            or normal_evaluation.feasible
            or not self.nonnormal_singleton_enabled
        )
        if (self.normal_family_id in candidate_id_set) is not normal_expected:
            raise ValueError(
                "Isolation Normal candidate must follow the nested calibrated acceptance set."
            )
        if (
            not isinstance(self.declared_family_hashes, tuple)
            or not self.declared_family_hashes
            or self.declared_family_hashes
            != tuple(sorted(set(self.declared_family_hashes)))
            or not all(_is_sha256(value) for value in self.declared_family_hashes)
            or {evaluation.family_hash for evaluation in self.evaluations}
            != set(self.declared_family_hashes)
            or {family.content_hash for family in self.declared_families}
            != set(self.declared_family_hashes)
        ):
            raise ValueError(
                "Isolation evaluations must cover the complete frozen family dictionary."
            )
        if not self.nonnormal_singleton_enabled and self.normal_family_id not in candidate_ids:
            raise ValueError(
                "Disabled nonnormal singleton gate must preserve the Normal explanation."
            )

    @property
    def candidate_family_ids(self) -> tuple[str, ...]:
        """返回稳定候选 explanation ids。

        参数：
            无。
        返回：
            Normal 在前、其余按 family_id 排序的元组。
        异常：
            无。
        副作用：
            无。
        """

        return tuple(family.family_id for family in self.candidate_families)

    @classmethod
    def evaluate(
        cls,
        observation: DeployedObservation,
        *,
        normal_calibration: FullNormalCalibrator,
        fault_families: Sequence[ExplanationFamily],
        oracle: OuterExplanationOracle,
    ) -> "IsolationCandidateSet":
        """组合 nested Normal 与所有声明 fault outer family。

        参数：
            observation: 冻结 normal/fault test 的完整部署观测。
            normal_calibration: 已冻结 ``q_det`` 和 ``q_attr`` 的 full-normal 校准。
            fault_families: 预声明非正常 explanation family；不得临时从测试数据增删。
            oracle: 与归因校准完全相同的冻结 outer oracle。
        返回：
            包含所有右闭可行解释的 ``IsolationCandidateSet``；未决 cell 已由 oracle
            保守保留。
        异常：
            calibration/oracle/观测身份、family 类型或重复身份非法时抛出
            ``TypeError``/``ValueError``。
        副作用：
            无；不调整半径、分位或 family dictionary。
        """

        if not isinstance(observation, DeployedObservation):
            raise TypeError("Isolation candidate evaluation requires a DeployedObservation.")
        if not isinstance(normal_calibration, FullNormalCalibrator):
            raise TypeError("Isolation candidate evaluation requires FullNormalCalibrator.")
        if not isinstance(oracle, OuterExplanationOracle):
            raise TypeError("Isolation candidate evaluation requires an OuterExplanationOracle.")
        if oracle.content_hash != normal_calibration.oracle_hash:
            raise ValueError("Isolation oracle must equal the frozen attribution oracle.")
        if (
            observation.detection_calibration_hash
            != normal_calibration.detection_calibration_hash
        ):
            raise ValueError(
                "Isolation observation must use the frozen detection calibration."
            )
        if isinstance(fault_families, (str, bytes)) or not isinstance(
            fault_families,
            Sequence,
        ):
            raise TypeError("Fault explanation families must be a sequence.")
        normalized_faults = tuple(sorted(fault_families, key=lambda item: item.family_id))
        if (
            not normalized_faults
            or not all(
                isinstance(family, ExplanationFamily) and not family.normal
                for family in normalized_faults
            )
            or len({family.family_id for family in normalized_faults})
            != len(normalized_faults)
            or any(
                family.family_id == normal_calibration.normal_family.family_id
                for family in normalized_faults
            )
        ):
            raise ValueError("Fault explanation families must be nonnormal and unique.")
        expected_fault_hashes = set(oracle.family_hashes) - {
            normal_calibration.normal_family_hash
        }
        actual_fault_hashes = {family.content_hash for family in normalized_faults}
        if (
            normal_calibration.normal_family_hash not in oracle.family_hashes
            or actual_fault_hashes != expected_fault_hashes
        ):
            raise ValueError(
                "Fault explanation families must equal the complete frozen fault dictionary."
            )

        normal_evaluation = oracle.evaluate(
            observation,
            normal_calibration.normal_family,
            radius=normal_calibration.quantile,
        )
        detection_normal_compatible = (
            observation.detection_excess
            <= normal_calibration.detection_quantile
        )
        normal_feasible = (
            detection_normal_compatible
            or normal_evaluation.feasible
            or not normal_calibration.nonnormal_singleton_enabled
        )
        fault_evaluations = tuple(
            oracle.evaluate(
                observation,
                family,
                radius=family.radius,
            )
            for family in normalized_faults
        )
        candidates: list[ExplanationFamily] = []
        if normal_feasible:
            candidates.append(normal_calibration.normal_family)
        candidates.extend(
            family
            for family, evaluation in zip(
                normalized_faults,
                fault_evaluations,
                strict=True,
            )
            if evaluation.feasible
        )
        return cls(
            observation_hash=observation.content_hash,
            oracle_hash=oracle.content_hash,
            detection_calibration_hash=normal_calibration.detection_calibration_hash,
            observation=observation,
            normal_calibration=normal_calibration,
            normal_family_id=normal_calibration.normal_family.family_id,
            candidate_families=tuple(candidates),
            evaluations=(normal_evaluation, *fault_evaluations),
            detection_normal_compatible=detection_normal_compatible,
            nonnormal_singleton_enabled=normal_calibration.nonnormal_singleton_enabled,
            declared_families=(
                normal_calibration.normal_family,
                *normalized_faults,
            ),
            declared_family_hashes=oracle.family_hashes,
        )


def _derive_isolation_report_fields(
    candidates: IsolationCandidateSet,
) -> tuple[
    IsolationOutcome,
    str,
    str | None,
    str,
    tuple[str, ...],
    bool,
]:
    """从自校验候选快照唯一派生隔离报告字段。

    参数：
        candidates: 已重放完整观测、校准和 oracle 的 ``IsolationCandidateSet``。
    返回：
        依次为 outcome、display label、reported family、Normal family、候选 ids 和认证状态。
    异常：
        候选对象类型非法时抛出 ``TypeError``；其余不变量已由候选构造器验证。
    副作用：
        无；只读取冻结候选、观测和物理证据。
    """

    if not isinstance(candidates, IsolationCandidateSet):
        raise TypeError("Isolation report requires an IsolationCandidateSet.")
    candidate_ids = candidates.candidate_family_ids
    evaluations_certified = all(
        evaluation.certified and not evaluation.unresolved_cell_ids
        for evaluation in candidates.evaluations
    )
    if candidates.normal_family_id in candidate_ids:
        return (
            IsolationOutcome.NORMAL_COMPATIBLE,
            IsolationOutcome.NORMAL_COMPATIBLE.value,
            None,
            candidates.normal_family_id,
            candidate_ids,
            evaluations_certified,
        )
    if not candidate_ids:
        return (
            IsolationOutcome.OUT_OF_MODEL,
            IsolationOutcome.OUT_OF_MODEL.value,
            None,
            candidates.normal_family_id,
            (),
            evaluations_certified,
        )
    if len(candidate_ids) > 1:
        return (
            IsolationOutcome.NONUNIQUE,
            IsolationOutcome.NONUNIQUE.value,
            None,
            candidates.normal_family_id,
            candidate_ids,
            evaluations_certified,
        )

    (family,) = candidates.candidate_families
    singleton_certified = (
        candidates.nonnormal_singleton_enabled
        and evaluations_certified
        and (
            not family.physical
            or candidates.observation.certifies_physical_family(family)
        )
    )
    if not singleton_certified:
        return (
            IsolationOutcome.UNCERTIFIED,
            IsolationOutcome.UNCERTIFIED.value,
            None,
            candidates.normal_family_id,
            candidate_ids,
            False,
        )
    return (
        IsolationOutcome.SINGLETON,
        family.label if family.physical else family.equivalence_label or family.label,
        family.family_id,
        candidates.normal_family_id,
        candidate_ids,
        True,
    )


@dataclass(frozen=True)
class IsolationReport:
    """把候选 explanation 集翻译为受控且不冒充后验概率的报告。

    参数：
        candidate_set: 报告的权威候选快照；其余字段必须可从该对象唯一重算。
        outcome/display_label/reported_family_id: 集合语义、读者标签和可选 singleton 身份。
        normal_family_id: 用于验证 Normal-compatible 与其他候选语义的冻结 Normal 身份。
        candidate_family_ids: 完整候选集合，不能只保存最终展示字符串。
        certified: 当前输出所需 oracle/operator/物理证据是否完整。
        risk_statement: episode 风险的受控说明，不解释为 PPV、FDR 或后验置信度。
    返回：
        ``from_candidate_set`` 生成不可变报告。
    异常：
        结果枚举、候选或 singleton 字段不一致时抛出 ``TypeError``/``ValueError``。
    副作用：
        无。
    """

    candidate_set: IsolationCandidateSet
    outcome: IsolationOutcome
    display_label: str
    reported_family_id: str | None
    normal_family_id: str
    candidate_family_ids: tuple[str, ...]
    certified: bool
    risk_statement: str

    def __post_init__(self) -> None:
        """验证展示语义没有丢弃候选集或伪造 singleton。"""

        if not isinstance(self.candidate_set, IsolationCandidateSet):
            raise TypeError("Isolation report requires its candidate-set snapshot.")
        if not isinstance(self.outcome, IsolationOutcome):
            raise TypeError("Isolation report outcome must use IsolationOutcome.")
        if not isinstance(self.display_label, str) or not self.display_label:
            raise ValueError("Isolation report display label must be non-empty.")
        if not isinstance(self.normal_family_id, str) or not self.normal_family_id:
            raise ValueError("Isolation report Normal family id must be non-empty.")
        if (
            not isinstance(self.candidate_family_ids, tuple)
            or len(set(self.candidate_family_ids)) != len(self.candidate_family_ids)
            or any(
                not isinstance(family_id, str) or not family_id
                for family_id in self.candidate_family_ids
            )
        ):
            raise ValueError("Isolation report candidate ids must be unique.")
        if type(self.certified) is not bool:
            raise TypeError("Isolation report certified flag must be a strict boolean.")
        if not isinstance(self.risk_statement, str) or not self.risk_statement:
            raise ValueError("Isolation report risk statement must be non-empty.")
        if self.risk_statement != _ISOLATION_RISK_STATEMENT:
            raise ValueError(
                "Isolation report must not claim posterior or classification risk."
            )
        expected_fields = _derive_isolation_report_fields(self.candidate_set)
        actual_fields = (
            self.outcome,
            self.display_label,
            self.reported_family_id,
            self.normal_family_id,
            self.candidate_family_ids,
            self.certified,
        )
        if actual_fields != expected_fields:
            raise ValueError(
                "Isolation report fields must follow candidate-set semantics."
            )
        if self.outcome is IsolationOutcome.SINGLETON:
            if (
                len(self.candidate_family_ids) != 1
                or self.reported_family_id != self.candidate_family_ids[0]
            ):
                raise ValueError("Singleton report must name its only candidate.")
        elif self.reported_family_id is not None:
            raise ValueError("Only a singleton report may name one reported family.")

    @classmethod
    def from_candidate_set(
        cls,
        candidates: IsolationCandidateSet,
        *,
        observation: DeployedObservation,
    ) -> "IsolationReport":
        """按集合基数、Normal 包含关系和认证边界产生报告。

        参数：
            candidates: 已完成所有 explanation 查询的候选集。
            observation: 与候选集哈希一致的完整观测，用于物理 singleton operator 门禁。
        返回：
            Normal-compatible、Nonunique、Out-of-model、equivalence/physical singleton
            或 Uncertified 报告。
        异常：
            对象类型或观测身份不一致时抛出 ``TypeError``/``ValueError``。
        副作用：
            无。
        """

        if not isinstance(candidates, IsolationCandidateSet):
            raise TypeError("Isolation report requires an IsolationCandidateSet.")
        if not isinstance(observation, DeployedObservation):
            raise TypeError("Isolation report requires a DeployedObservation.")
        if observation.content_hash != candidates.observation_hash:
            raise ValueError("Isolation report observation does not match its candidate set.")
        (
            outcome,
            display_label,
            reported_family_id,
            normal_family_id,
            candidate_family_ids,
            certified,
        ) = _derive_isolation_report_fields(candidates)
        return cls(
            candidate_set=candidates,
            outcome=outcome,
            display_label=display_label,
            reported_family_id=reported_family_id,
            normal_family_id=normal_family_id,
            candidate_family_ids=candidate_family_ids,
            certified=certified,
            risk_statement=_ISOLATION_RISK_STATEMENT,
        )


def _open_unit_interval(value: object, *, name: str) -> float:
    """校验并返回严格位于 ``(0,1)`` 的浮点数。

    参数：
        value: 风险预算候选值；bool 不按整数接受。
        name: 错误消息使用的字段名。
    返回：
        规范化 Python ``float``。
    异常：
        非数值时抛出 ``TypeError``；非有限或不在开区间时抛出 ``ValueError``。
    副作用：
        无。
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    normalized = float(value)
    if not 0.0 < normalized < 1.0 or not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite and strictly between zero and one.")
    return normalized


def _nonnegative_finite_score(value: object, *, name: str) -> float:
    """校验并返回非负有限 score。

    参数：
        value: normal attribution score 候选值。
        name: 错误消息使用的字段名。
    返回：
        规范化 Python ``float``。
    异常：
        非数值时抛出 ``TypeError``；负值或非有限值时抛出 ``ValueError``。
    副作用：
        无。
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    normalized = float(value)
    if normalized < 0.0 or not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite and non-negative.")
    return normalized


def _sorted_unique_ids(values: object, *, name: str) -> tuple[str, ...]:
    """把非空字符串序列规范化为稳定排序且唯一的 episode ids。

    参数：
        values: 待冻结 episode identity 序列；字符串本身不视为序列容器。
        name: 错误消息使用的字段名。
    返回：
        排序后的非空字符串元组。
    异常：
        容器非法时抛出 ``TypeError``；空、重复或含非法 id 时抛出 ``ValueError``。
    副作用：
        无；不修改调用方序列。
    """

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of strings.")
    normalized = tuple(sorted(values))
    if (
        not normalized
        or any(not isinstance(value, str) or not value for value in normalized)
        or len(set(normalized)) != len(normalized)
    ):
        raise ValueError(f"{name} must be non-empty and unique.")
    return normalized


def _is_sha256(value: object) -> bool:
    """判断对象是否为 64 位小写十六进制 SHA-256。

    参数：
        value: 待验证身份对象。
    返回：
        仅严格匹配 64 位小写十六进制字符串时返回 ``True``。
    异常：
        无。
    副作用：
        无。
    """

    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
