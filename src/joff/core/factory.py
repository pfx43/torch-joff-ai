"""Joff 注册对象的公共构建工厂。

文件用途：
    把严格配置或普通 mapping 解析为显式注册的模型与评估器，隔离配置校验、按需导入和
    注册表查找错误。
主要职责：
    提供模型/评估器注册函数；在第一次构建时导入内置模块；按模型声明的
    ``config_model`` 选择专用严格 Pydantic 配置，同时保持通用 ``ModelConfig`` 兼容。
关键输入与输出：
    输入为注册键、严格配置或含 ``type`` 的 mapping；输出为已构造模型/评估器实例。
依赖与副作用：
    首次构建会导入 ``joff.models`` 或 ``joff.evaluation`` 并更新进程内注册表；不读写
    数据、运行目录或网络。
重要约束：
    所有模型配置都必须继承 ``StrictConfig`` 并拒绝未知字段；构建只允许显式注册类，
    禁止任意动态导入或 ``eval``。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from .config import ModelConfig, StrictConfig
from .errors import ConfigError
from .registry import EVALUATOR_REGISTRY, MODEL_REGISTRY


def register_model(
    key: str, model_cls: type | None = None, *, aliases: tuple[str, ...] = (), replace: bool = False
):
    """注册公开模型类。

    参数：
        key/model_cls: 稳定注册键与模型类；省略模型类时可作装饰器。
        aliases: 兼容别名。
        replace: 是否显式替换已有注册。
    返回：
        已注册类或注册装饰器。
    异常：
        重复键/别名且未允许替换时由注册表抛出 ``RegistryError``。
    副作用：
        修改进程内全局模型注册表。
    """

    return MODEL_REGISTRY.register(key, model_cls, aliases=aliases, replace=replace)


def register_evaluator(
    key: str,
    evaluator_cls: type | None = None,
    *,
    aliases: tuple[str, ...] = (),
    replace: bool = False,
):
    """注册公开评估器类。

    参数、返回、异常与副作用：
        语义与 ``register_model`` 相同，但目标是全局评估器注册表。
    """

    return EVALUATOR_REGISTRY.register(key, evaluator_cls, aliases=aliases, replace=replace)


def build_model(spec: StrictConfig | Mapping[str, Any]) -> Any:
    """从严格配置或 mapping 构造显式注册模型。

    参数：
        spec: 通用/模型专用 ``StrictConfig``，或至少可解析出 ``type`` 的 mapping。
    返回：
        注册模型实例。
    异常：
        未知模型类型由注册表抛出 ``RegistryError``；配置字段、嵌套类型或约束非法时包装
        为含合法键列表的 ``ConfigError``。
    副作用：
        首次调用按需导入内置模型并完成注册；构造模型会初始化其参数。
    """

    _ensure_builtin_models_registered()
    raw = spec.model_dump(mode="python") if isinstance(spec, StrictConfig) else dict(spec)
    model_type = raw.get("type", "mlp")
    if not isinstance(model_type, str):
        raise ConfigError(
            "Invalid model config. Model 'type' must be a string. "
            f"Current input was: {model_type!r}."
        )
    model_cls = MODEL_REGISTRY.get(model_type)
    config_model = getattr(model_cls, "config_model", ModelConfig)
    if not isinstance(config_model, type) or not issubclass(config_model, StrictConfig):
        raise ConfigError(
            f"Registered model {model_type!r} exposes an invalid config_model. "
            "Legal config models must inherit StrictConfig."
        )
    if isinstance(spec, config_model):
        config = spec
    else:
        try:
            config = config_model.model_validate(raw)
        except ValidationError as exc:
            legal = ", ".join(config_model.model_fields)
            raise ConfigError(
                f"Invalid model config. Legal model keys are: {legal}. "
                f"Current input was: {raw!r}. Details: {exc}"
            ) from exc
    return model_cls(config)


def build_evaluator(spec: str | Mapping[str, Any]) -> Any:
    """从字符串或 mapping 构造显式注册评估器。

    参数：
        spec: 注册键，或带可选 ``type`` 和构造关键字的 mapping。
    返回：
        已构造评估器。
    异常：
        未知类型由注册表抛出 ``RegistryError``；构造参数错误由评估器自身抛出。
    副作用：
        首次调用按需导入内置评估器并完成注册。
    """

    _ensure_builtin_evaluators_registered()
    if isinstance(spec, str):
        evaluator_type = spec
        kwargs: dict[str, Any] = {}
    else:
        raw = dict(spec)
        evaluator_type = str(raw.pop("type", "regression"))
        kwargs = raw
    evaluator_cls = EVALUATOR_REGISTRY.get(evaluator_type)
    return evaluator_cls(**kwargs)


def _ensure_builtin_models_registered() -> None:
    # Importing this module performs explicit registry registration. This happens only when
    # a user builds a model, not during core module import.
    import joff.models  # noqa: F401


def _ensure_builtin_evaluators_registered() -> None:
    import joff.evaluation  # noqa: F401
