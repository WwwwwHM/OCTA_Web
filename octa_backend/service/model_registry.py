"""
OCTA图像分割平台 - 模型注册表模块（Model Registry）

本模块提供分割模型的统一注册与查询能力，支持“即插即用”扩展：
- 新模型只需调用 register_model() 完成注册；
- 后端启动阶段可调用 load_registered_models() 自动构建全部已注册模型；
- 业务层通过 get_model()/list_models() 获取模型元信息。

核心设计目标：
1. 统一模型元数据结构，降低新模型接入复杂度；
2. 强类型与显式异常，减少运行时隐式错误；
3. 与 PyTorch 2.0+ 兼容，便于后续模型升级。
"""

from __future__ import annotations

import inspect
import logging
import warnings
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn


# ==================== 日志配置 ====================
logger = logging.getLogger(__name__)


# ==================== 全局注册表与实例缓存 ====================
# 注册表：模型名称 -> 模型定义信息
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {}

# 已加载模型缓存：模型名称 -> 已实例化模型
LOADED_MODELS: Dict[str, nn.Module] = {}


# ==================== 自定义异常 ====================
class ModelRegistryError(Exception):
    """模型注册表基础异常。"""


class ModelRegistryTypeError(ModelRegistryError, TypeError):
    """模型注册入参类型错误异常。"""


class ModelFactorySignatureError(ModelRegistryError):
    """模型构造函数签名不符合要求（应为无参）异常。"""


# ==================== 内部校验工具函数 ====================
def _validate_callable(name: str, func: Any) -> None:
    """
    校验对象是否为可调用对象。

    Args:
        name: 参数名称（用于构造错误信息）。
        func: 待校验对象。

    Raises:
        ModelRegistryTypeError: 当对象不是可调用对象时抛出。
    """
    if not callable(func):
        raise ModelRegistryTypeError(
            f"参数{name}必须是可调用对象(callable)，当前类型: {type(func).__name__}"
        )


def _validate_creator_signature(creator: Callable[[], nn.Module]) -> None:
    """
    校验creator是否支持无参调用。

    说明：
    - 按需求，create函数应为“无参构建函数”；
    - 本校验允许“所有参数都有默认值”的可调用对象通过。

    Args:
        creator: 模型构造函数。

    Raises:
        ModelFactorySignatureError: 当creator不能无参调用时抛出。
    """
    try:
        signature = inspect.signature(creator)
    except (TypeError, ValueError):
        logger.debug("creator签名无法静态解析，跳过签名校验。")
        return

    for parameter in signature.parameters.values():
        required_positional = (
            parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            and parameter.default is inspect.Parameter.empty
        )
        required_keyword_only = (
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            and parameter.default is inspect.Parameter.empty
        )
        if required_positional or required_keyword_only:
            raise ModelFactorySignatureError(
                "creator必须支持无参调用（无必填参数）。"
            )


# ==================== 核心接口函数 ====================
def register_model(
    model_name: str,
    creator: Callable[[], nn.Module],
    preprocess: Callable[[Any], torch.Tensor],
    postprocess: Callable[[torch.Tensor], Any],
    config: Dict[str, Any],
) -> None:
    """
    注册模型到全局MODEL_REGISTRY。

    注册信息结构：
    - create: 构建模型函数（无参）
    - preprocess: 预处理函数（输入文件对象 -> 模型输入Tensor）
    - postprocess: 后处理函数（模型输出Tensor -> 二值掩码矩阵）
    - config: 模型配置字典（如input_size/num_classes/weight_path）

    Args:
        model_name: 模型名称，例如"unet"。
        creator: 无参模型构建函数，返回PyTorch模型实例。
        preprocess: 模型输入预处理函数。
        postprocess: 模型输出后处理函数。
        config: 模型配置字典。

    Raises:
        ModelRegistryTypeError: 入参类型错误时抛出。
        ModelFactorySignatureError: creator不支持无参调用时抛出。
    """
    # ==================== 步骤1：基础类型校验 ====================
    if not isinstance(model_name, str) or not model_name.strip():
        raise ModelRegistryTypeError("参数model_name必须是非空字符串。")

    _validate_callable("creator", creator)
    _validate_callable("preprocess", preprocess)
    _validate_callable("postprocess", postprocess)

    if not isinstance(config, dict):
        raise ModelRegistryTypeError(
            f"参数config必须是dict，当前类型: {type(config).__name__}"
        )

    # ==================== 步骤2：creator签名校验（支持无参调用） ====================
    _validate_creator_signature(creator)

    # ==================== 步骤3：标准化名称并写入注册表 ====================
    normalized_name = model_name.strip().lower()

    if normalized_name in MODEL_REGISTRY:
        warnings.warn(
            f"模型'{normalized_name}'已存在，将执行覆盖注册。",
            category=UserWarning,
            stacklevel=2,
        )
        logger.warning("检测到重复注册，已覆盖模型: %s", normalized_name)

    MODEL_REGISTRY[normalized_name] = {
        "create": creator,
        "preprocess": preprocess,
        "postprocess": postprocess,
        "config": config,
    }

    logger.info("模型注册成功: %s", normalized_name)


def get_model(model_name: str) -> Optional[Dict[str, Any]]:
    """
    根据模型名称查询注册信息。

    Args:
        model_name: 模型名称。

    Returns:
        注册信息字典；不存在时返回None。

    Raises:
        ModelRegistryTypeError: model_name类型错误时抛出。
    """
    if not isinstance(model_name, str) or not model_name.strip():
        raise ModelRegistryTypeError("参数model_name必须是非空字符串。")

    normalized_name = model_name.strip().lower()
    model_info = MODEL_REGISTRY.get(normalized_name)

    if model_info is None:
        logger.warning("未找到模型注册信息: %s", normalized_name)
        return None

    return model_info


def list_models() -> List[str]:
    """
    返回所有已注册模型名称列表。

    Returns:
        已注册模型名列表（按字母升序）。
    """
    return sorted(MODEL_REGISTRY.keys())


# ==================== 启动期自动加载（扩展能力） ====================
def load_registered_models(force_reload: bool = False) -> Dict[str, nn.Module]:
    """
    构建并缓存所有已注册模型实例。

    典型用法：
    在FastAPI启动事件中调用，实现“后端启动即自动加载注册模型”。

    Args:
        force_reload: 是否强制重建已加载模型。

    Returns:
        当前已加载模型缓存字典。

    Raises:
        ModelRegistryError: 当某个模型构建失败时抛出。
    """
    for model_name, model_meta in MODEL_REGISTRY.items():
        if not force_reload and model_name in LOADED_MODELS:
            logger.debug("模型已在缓存中，跳过加载: %s", model_name)
            continue

        try:
            model_instance = model_meta["create"]()
            if not isinstance(model_instance, nn.Module):
                raise ModelRegistryTypeError(
                    f"模型'{model_name}'的creator返回值必须是nn.Module实例，"
                    f"当前类型: {type(model_instance).__name__}"
                )

            model_instance.eval()
            LOADED_MODELS[model_name] = model_instance
            logger.info("模型加载完成: %s", model_name)
        except Exception as exc:
            logger.exception("模型加载失败: %s", model_name)
            raise ModelRegistryError(
                f"模型'{model_name}'自动加载失败: {exc}"
            ) from exc

    return LOADED_MODELS


__all__ = [
    "MODEL_REGISTRY",
    "LOADED_MODELS",
    "ModelRegistryError",
    "ModelRegistryTypeError",
    "ModelFactorySignatureError",
    "register_model",
    "get_model",
    "list_models",
    "load_registered_models",
]
