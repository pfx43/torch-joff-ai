"""P6 堆叠算子装配、认证状态和共享不确定元素的公开行为测试。

文件用途：
    验证线性特例的 ``G_nu/G_0``、输入响应和 exact stacked propagation，并锁定名义数值
    与认证 enclosure 的 fail-closed 分界。
主要职责：
    只通过 ``NominalJVPAssembler``、``OperatorBundle`` 和认证 provider 协议观察行为；
    不测试 P7 后滤波、P8 动态阈值或 P9 物理隔离实现。
关键输入与输出：
    输入为可手算的二维时变线性 Jacobian、局部输入响应和小型 provider；输出为堆叠矩阵、
    状态、联合 enclosure 与可序列化审计字段。
依赖与副作用：
    依赖 PyTorch 和 Joff evaluation 公共接口；不读写文件、不访问网络。
重要约束：
    segment integral 的解析恒等式不等于在线数值已认证；没有完整共享证据时，
    ``certified`` 必须为假，安全排除入口必须失败。
"""

from __future__ import annotations

from dataclasses import replace
import json
from typing import Any, cast

import pytest
import torch

from joff.artifacts import ArtifactStore
from joff.core.factory import build_model
from joff.evaluation import (
    CertifiedEnclosureProvider,
    JacobianSemantics,
    MonitorStage,
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


_TEST_ASSEMBLY_BUDGET = OperatorAssemblyBudget(
    max_workspace_elements=100_000,
    max_persisted_elements=100_000,
)


def _assembler(
    *,
    budget: OperatorAssemblyBudget = _TEST_ASSEMBLY_BUDGET,
) -> NominalJVPAssembler:
    """用测试显式资源预算构造无跨调用状态的 P6 装配器。"""

    return NominalJVPAssembler(resource_budget=budget)


def test_linear_segment_jacobians_recover_exact_stacked_operators() -> None:
    """二维时变线性系统精确恢复单位块下三角算子和手算递推。"""

    jacobians = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 2.0]],
            [[2.0, 0.0], [0.0, 3.0]],
            [[3.0, 0.0], [0.0, 4.0]],
        ],
        dtype=torch.float64,
    )
    input_jacobians = torch.tensor(
        [
            [[1.0], [2.0]],
            [[3.0], [4.0]],
            [[5.0], [6.0]],
        ],
        dtype=torch.float64,
    )
    path = OperatorPath(
        monitor_identity="linear-fixture",
        episode_id="episode-linear",
        stage=MonitorStage.ESTIMATE,
        start_raw_index=0,
        raw_indices=(1, 2, 3),
    )

    bundle = _assembler().assemble(
        transition_jacobians=jacobians,
        semantics=JacobianSemantics.SEGMENT_AVERAGED_EXACT,
        path=path,
        input_jacobians=input_jacobians,
    )

    identity = torch.eye(2, dtype=torch.float64)
    zero = torch.zeros((2, 2), dtype=torch.float64)
    expected_g_nu = torch.cat(
        (
            torch.cat((identity, zero, zero), dim=1),
            torch.cat((jacobians[1], identity, zero), dim=1),
            torch.cat((jacobians[2] @ jacobians[1], jacobians[2], identity), dim=1),
        ),
        dim=0,
    )
    expected_g_0 = torch.cat(
        (
            jacobians[0],
            jacobians[1] @ jacobians[0],
            jacobians[2] @ jacobians[1] @ jacobians[0],
        ),
        dim=0,
    )
    g_nu = torch.tensor(bundle.g_nu, dtype=torch.float64)
    g_0 = torch.tensor(bundle.g_0, dtype=torch.float64)
    input_response = torch.tensor(bundle.input_response, dtype=torch.float64)

    assert bundle.status is OperatorStatus.NOMINAL
    assert not bundle.certified
    assert torch.equal(g_nu, expected_g_nu)
    assert torch.equal(g_0, expected_g_0)
    assert torch.equal(
        input_response,
        expected_g_nu @ torch.block_diag(*input_jacobians),
    )
    assert torch.allclose(
        g_nu @ torch.linalg.inv(g_nu),
        torch.eye(6, dtype=torch.float64),
        atol=1e-14,
        rtol=0.0,
    )

    forcing = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float64)
    e_1 = forcing[:2]
    e_2 = jacobians[1] @ e_1 + forcing[2:4]
    e_3 = jacobians[2] @ e_2 + forcing[4:6]
    assert torch.equal(
        g_nu @ forcing,
        torch.cat((e_1, e_2, e_3)),
    )


class _FixtureEnclosureProvider(CertifiedEnclosureProvider):
    """测试用 provider，可切换共享、完整和 verified 三项认证条件。"""

    def __init__(
        self,
        *,
        shared: bool = True,
        complete: bool = True,
        verified: bool = True,
    ) -> None:
        self.shared = shared
        self.complete = complete
        self.verified = verified

    def enclose(
        self,
        request: OperatorCertificationRequest,
    ) -> OperatorEnclosure | None:
        """用一个共享标量系数同步生成全部命名算子的仿射 enclosure。"""

        names = request.required_operator_names
        if not self.complete:
            names = names[:-1]
        images = tuple(
            OperatorAffineImage(
                operator_name=name,
                center=request.nominal(name),
                generators=(
                    tuple(
                        tuple(0.01 for _ in row)
                        for row in request.nominal(name)
                    ),
                ),
            )
            for name in names
        )
        return OperatorEnclosure(
            images=images,
            error_radius=1.0,
            norm=OperatorNorm.SPECTRAL_L2,
            shared_uncertainty_id=(
                request.shared_uncertainty_id
                if self.shared
                else f"{request.shared_uncertainty_id}:independent"
            ),
            source="verified-interval-fixture",
            certificate_id="fixture-certificate",
            verified_remainder=self.verified,
        )


def _operator_fixture():
    """构造认证测试共用的两步二维算子输入。"""

    transitions = torch.tensor(
        [
            [[0.8, 0.1], [0.0, 0.9]],
            [[0.7, 0.0], [0.2, 0.6]],
        ],
        dtype=torch.float64,
    )
    inputs = torch.ones((2, 2, 1), dtype=torch.float64)
    process = torch.tensor(
        [
            [[1.0], [0.0]],
            [[0.0], [1.0]],
        ],
        dtype=torch.float64,
    )
    sensors = {
        "channel-0": torch.arange(8, dtype=torch.float64).reshape(4, 2),
        "channel-1": torch.eye(4, dtype=torch.float64),
    }
    path = OperatorPath(
        monitor_identity="certification-fixture",
        episode_id="episode-certified",
        stage=MonitorStage.ESTIMATE,
        start_raw_index=10,
        raw_indices=(11, 12),
    )
    return transitions, inputs, process, sensors, path


def test_certification_requires_complete_enclosures_with_one_shared_error_element() -> None:
    """完整且共享 operator-error 的 verified enclosure 才能升级为 certified。"""

    transitions, inputs, process, sensors, path = _operator_fixture()
    bundle = _assembler().assemble(
        transition_jacobians=transitions,
        semantics=JacobianSemantics.NOMINAL_POINTWISE,
        path=path,
        input_jacobians=inputs,
        sensor_jvps=sensors,
        process_prior=process,
        enclosure_provider=_FixtureEnclosureProvider(),
    )

    assert bundle.status is OperatorStatus.CERTIFIED
    assert bundle.certified
    required = (
        "g_nu",
        "g_0",
        "input_response",
        "process_prior",
        "sensor:channel-0",
        "sensor:channel-1",
    )
    enclosure = bundle.require_certified(*required)
    assert enclosure is bundle.enclosure
    assert enclosure.operator_names == required
    assert enclosure.shared_uncertainty_id == bundle.shared_uncertainty_id
    assert bundle.to_dict()["status"] == "certified"
    assert bundle.to_dict()["certified"] is True


@pytest.mark.parametrize(
    "provider",
    [
        _FixtureEnclosureProvider(shared=False),
        _FixtureEnclosureProvider(complete=False),
        _FixtureEnclosureProvider(verified=False),
    ],
)
def test_incomplete_or_independent_enclosures_cannot_authorize_safe_exclusion(
    provider: CertifiedEnclosureProvider,
) -> None:
    """缺项、独立误差或未验证 remainder 都不能授权任何安全排除消费者。"""

    transitions, inputs, process, sensors, path = _operator_fixture()
    bundle = _assembler().assemble(
        transition_jacobians=transitions,
        semantics=JacobianSemantics.NOMINAL_POINTWISE,
        path=path,
        input_jacobians=inputs,
        sensor_jvps=sensors,
        process_prior=process,
        enclosure_provider=provider,
    )

    assert bundle.status is OperatorStatus.UNAVAILABLE
    assert not bundle.certified
    with pytest.raises(UncertifiedOperatorError):
        bundle.require_certified("g_nu")


def test_nominal_bundle_cannot_be_relabelled_certified_without_evidence() -> None:
    """公开不可变对象也必须拒绝只改枚举、不补联合 enclosure 的伪认证。"""

    transitions, inputs, process, sensors, path = _operator_fixture()
    bundle = _assembler().assemble(
        transition_jacobians=transitions,
        semantics=JacobianSemantics.NOMINAL_POINTWISE,
        path=path,
        input_jacobians=inputs,
        sensor_jvps=sensors,
        process_prior=process,
    )

    with pytest.raises(ValueError, match="CERTIFIED"):
        replace(
            bundle,
            status=OperatorStatus.CERTIFIED,
            status_reason="manually relabelled",
        )


def test_nonlinear_segment_average_satisfies_exact_stacked_identity() -> None:
    """解析线段平均 Jacobian 对任意注入保持非线性残差递推恒等式。"""

    def transition(z: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                z[0].square() + 0.5 * z[1],
                z[0] * z[1] + 0.2 * z[0],
            )
        )

    def segment_average(
        protected: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        midpoint = protected + 0.5 * residual
        return torch.tensor(
            [
                [2.0 * midpoint[0], 0.5],
                [midpoint[1] + 0.2, midpoint[0]],
            ],
            dtype=torch.float64,
        )

    protected = torch.tensor([0.2, -0.1], dtype=torch.float64)
    data = protected.clone()
    forcings = (
        torch.tensor([0.1, -0.05], dtype=torch.float64),
        torch.tensor([-0.02, 0.08], dtype=torch.float64),
        torch.tensor([0.04, 0.03], dtype=torch.float64),
    )
    jacobians = []
    residuals = []
    for forcing in forcings:
        residual = data - protected
        jacobians.append(segment_average(protected, residual))
        protected = transition(protected)
        data = transition(data) + forcing
        residuals.append(data - protected)

    path = OperatorPath(
        monitor_identity="nonlinear-exact-fixture",
        episode_id="episode-nonlinear",
        stage=MonitorStage.ESTIMATE,
        start_raw_index=20,
        raw_indices=(21, 22, 23),
    )
    bundle = _assembler().assemble(
        transition_jacobians=torch.stack(jacobians),
        semantics=JacobianSemantics.SEGMENT_AVERAGED_EXACT,
        path=path,
    )

    actual = torch.tensor(bundle.g_nu, dtype=torch.float64) @ torch.cat(forcings)
    assert torch.allclose(
        actual,
        torch.cat(residuals),
        atol=1e-14,
        rtol=1e-14,
    )


def test_p4_rollout_jacobians_replay_one_nominal_operator_bundle() -> None:
    """P4 点值 Jacobian 进入 P6 时保持 nominal 语义并确定性重放 hash。"""

    torch.manual_seed(17)
    model = build_model(
        {
            "type": "protected_koopman_ts",
            "control_dim": 1,
            "measurement_dim": 1,
            "exogenous_dim": 1,
            "history_length": 2,
            "latent_dim": 2,
            "context_dim": 2,
            "max_rollout": 2,
            "horizon_seed": 3,
            "attention": {
                "embed_dim": 4,
                "num_heads": 1,
                "dropout": 0.0,
            },
            "channel_mask": {
                "all_pass_probability": 1.0,
                "single_channel_probability": 0.0,
                "independent_drop_probability": 0.0,
                "seed": 5,
            },
            "fuzzy": {
                "rule_count": 2,
                "premise_dim": 2,
                "premise_hidden_dim": 3,
                "metric_eigenvalue_min": 0.1,
                "metric_eigenvalue_max": 2.0,
                "spectral_cap": 1.1,
            },
            "loss": {
                "horizon_weights": [1.0, 1.0],
                "latent_weight": 1.0,
                "output_weight": 1.0,
                "decoding_weight": 0.5,
                "variance_weight": 0.1,
                "rule_balance_weight": 0.1,
                "jacobian_product_weight": 0.1,
                "minimum_latent_std": 0.1,
                "maximum_jacobian_product_norm": 2.0,
            },
        }
    )
    model.eval()
    rollout = model.rollout(
        past_u=torch.tensor([[[0.1], [0.2]]]),
        past_y=torch.tensor([[[0.3], [0.4]]]),
        past_xi=torch.tensor([[[0.5], [0.6]]]),
        future_u=torch.tensor([[[0.7], [0.8]]]),
        future_xi=torch.tensor([[[0.9], [1.0]]]),
    )
    path = OperatorPath(
        monitor_identity="p4-checkpoint-fixture",
        episode_id="episode-p4",
        stage=MonitorStage.ESTIMATE,
        start_raw_index=30,
        raw_indices=(31, 32),
    )
    assembler = _assembler()

    first = assembler.assemble(
        transition_jacobians=rollout["jacobian_z"][0],
        semantics=JacobianSemantics.NOMINAL_POINTWISE,
        path=path,
        input_jacobians=rollout["jacobian_u"][0],
    )
    replay = assembler.assemble(
        transition_jacobians=rollout["jacobian_z"][0],
        semantics=JacobianSemantics.NOMINAL_POINTWISE,
        path=path,
        input_jacobians=rollout["jacobian_u"][0],
    )

    assert first.status is OperatorStatus.NOMINAL
    assert len(first.g_nu) == 4
    assert len(first.g_0) == 4
    assert first.content_hash == replay.content_hash


def test_joint_enclosure_uses_one_coefficient_across_all_operator_images() -> None:
    """相反的算子 image 必须由同一系数同步变化，不能各自挑选最有利误差。"""

    enclosure = OperatorEnclosure(
        images=(
            OperatorAffineImage(
                operator_name="left",
                center=((0.0,),),
                generators=(((1.0,),),),
            ),
            OperatorAffineImage(
                operator_name="right",
                center=((0.0,),),
                generators=(((-1.0,),),),
            ),
        ),
        error_radius=1.0,
        norm=OperatorNorm.ELEMENTWISE_INF,
        shared_uncertainty_id="joint-fixture",
        source="analytic-fixture",
        certificate_id="joint-certificate",
        verified_remainder=True,
    )

    assert enclosure.support_upper(
        {
            "left": ((1.0,),),
            "right": ((1.0,),),
        }
    ) == pytest.approx(0.0)


def test_enclosure_rejects_non_boolean_verified_remainder() -> None:
    """认证余项字段只接受真正的 bool，字符串和整数都不能升级证据。"""

    image = OperatorAffineImage(
        operator_name="g_nu",
        center=((1.0,),),
        generators=(((0.0,),),),
    )
    with pytest.raises(ValueError, match="verified_remainder"):
        OperatorEnclosure(
            images=(image,),
            error_radius=0.0,
            norm=OperatorNorm.SPECTRAL_L2,
            shared_uncertainty_id="strict-bool-fixture",
            source="invalid-fixture",
            certificate_id="invalid-certificate",
            verified_remainder=cast(Any, "false"),
        )


@pytest.mark.parametrize("invalid_radius", [False, "0"])
def test_enclosure_rejects_coerced_certification_radius(invalid_radius: Any) -> None:
    """provider 直接构造时也不能把 bool/字符串半径伪装成零误差认证。"""

    image = OperatorAffineImage(
        operator_name="g_nu",
        center=((1.0,),),
        generators=(((0.0,),),),
    )
    with pytest.raises((TypeError, ValueError), match="error_radius"):
        OperatorEnclosure(
            images=(image,),
            error_radius=invalid_radius,
            norm=OperatorNorm.SPECTRAL_L2,
            shared_uncertainty_id="strict-radius-fixture",
            source="invalid-fixture",
            certificate_id="invalid-certificate",
            verified_remainder=True,
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("monitor_identity", 7),
        ("episode_id", 8),
        ("start_raw_index", True),
        ("raw_indices", (1.9,)),
    ],
)
def test_operator_path_rejects_identity_and_index_type_coercion(
    field: str,
    invalid_value: Any,
) -> None:
    """公共路径构造器必须与严格 JSON 恢复使用相同类型规则。"""

    values: dict[str, Any] = {
        "monitor_identity": "strict-path-fixture",
        "episode_id": "episode-strict-path",
        "stage": MonitorStage.ESTIMATE,
        "start_raw_index": 0,
        "raw_indices": (1,),
    }
    values[field] = invalid_value

    with pytest.raises((TypeError, ValueError)):
        OperatorPath(**values)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("operator_name", 1),
        ("shared_uncertainty_id", 2),
        ("source", 3),
        ("certificate_id", 4),
    ],
)
def test_operator_evidence_rejects_non_string_identities(
    field: str,
    invalid_value: Any,
) -> None:
    """联合证据各身份字段必须在直接构造时就是非空字符串。"""

    image_values: dict[str, Any] = {
        "operator_name": "g_nu",
        "center": ((1.0,),),
        "generators": (((0.0,),),),
    }
    if field == "operator_name":
        image_values[field] = invalid_value
        with pytest.raises((TypeError, ValueError)):
            OperatorAffineImage(**image_values)
        return

    image = OperatorAffineImage(**image_values)
    enclosure_values: dict[str, Any] = {
        "images": (image,),
        "error_radius": 0.0,
        "norm": OperatorNorm.SPECTRAL_L2,
        "shared_uncertainty_id": "strict-evidence-fixture",
        "source": "strict-fixture",
        "certificate_id": "strict-certificate",
        "verified_remainder": True,
    }
    enclosure_values[field] = invalid_value
    with pytest.raises((TypeError, ValueError)):
        OperatorEnclosure(**enclosure_values)


def test_assembly_rejects_output_beyond_explicit_resource_budget() -> None:
    """装配器在构造稠密 G_nu 前按可审计元素预算拒绝过大的窗口。"""

    budget = OperatorAssemblyBudget(
        max_workspace_elements=20,
        max_persisted_elements=20,
    )
    path = OperatorPath(
        monitor_identity="budget-fixture",
        episode_id="episode-budget",
        stage=MonitorStage.ESTIMATE,
        start_raw_index=0,
        raw_indices=(1, 2, 3),
    )

    with pytest.raises(ValueError, match="resource budget"):
        _assembler(budget=budget).assemble(
            transition_jacobians=torch.eye(2, dtype=torch.float64).repeat(3, 1, 1),
            semantics=JacobianSemantics.NOMINAL_POINTWISE,
            path=path,
        )


def test_certification_status_survives_json_artifact_round_trip(tmp_path) -> None:
    """认证状态、联合 enclosure 和公共身份经真实 JSON 产物往返后保持不变。"""

    transitions, inputs, process, sensors, path = _operator_fixture()
    bundle = _assembler().assemble(
        transition_jacobians=transitions,
        semantics=JacobianSemantics.NOMINAL_POINTWISE,
        path=path,
        input_jacobians=inputs,
        sensor_jvps=sensors,
        process_prior=process,
        enclosure_provider=_FixtureEnclosureProvider(),
    )
    store = ArtifactStore(tmp_path, "p6-artifact-round-trip")

    artifact_path = store.save_json("operator-report.json", bundle.to_dict())
    restored = OperatorBundle.from_dict(
        json.loads(artifact_path.read_text(encoding="utf-8"))
    )

    assert restored.status is OperatorStatus.CERTIFIED
    assert restored.status_reason == bundle.status_reason
    assert restored.shared_uncertainty_id == bundle.shared_uncertainty_id
    assert restored.enclosure is not None
    assert bundle.enclosure is not None
    assert restored.enclosure.to_dict() == bundle.enclosure.to_dict()
    assert restored.content_hash == bundle.content_hash

    tampered = bundle.to_dict()
    tampered["persisted_elements"] = 1
    with pytest.raises(ValueError, match="persisted_elements"):
        OperatorBundle.from_dict(tampered)
