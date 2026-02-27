"""通用分割推理路由（基于模型注册表）。

接口：POST /api/v1/seg/predict

功能：
- 按model_name选择已注册模型；
- 支持通过weight_id覆盖模型默认权重路径；
- 输出二值mask的base64编码，便于前端直接展示。
"""

from __future__ import annotations

import base64
import importlib
import importlib.util
import inspect
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from config.config import WEIGHT_UPLOAD_ROOT
from dao.file_dao import FileDAO
from service.model_registry import get_model, list_models

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/seg", tags=["分割推理"])

_ALLOWED_IMG_EXT = {"png", "jpg", "jpeg", "bmp", "tiff", "tif"}


def _validate_image(upload: UploadFile) -> None:
    """校验上传图片格式。

    Raises:
        HTTPException: 当格式不支持或文件为空时。
    """
    suffix = (upload.filename or "").split(".")[-1].lower()
    if suffix not in _ALLOWED_IMG_EXT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"图片格式不支持: {suffix}，仅允许 .png/.jpg/.bmp/.tiff",
        )


def _build_error_response(code: int, msg: str, http_status: int) -> JSONResponse:
    """构建统一错误响应。"""
    return JSONResponse(
        status_code=http_status,
        content={"code": code, "msg": msg, "data": None},
    )


def _import_model_module_if_needed(model_name: str) -> None:
    """尝试按约定路径动态导入模型模块，以触发自动注册。"""
    module_name = f"models.{model_name}.model"
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        logger.debug("标准导入失败，尝试文件路径导入 model=%s, module=%s, err=%s", model_name, module_name, exc)

        try:
            model_file = Path(__file__).resolve().parents[1] / "models" / model_name / "model.py"
            if not model_file.exists():
                logger.debug("模型文件不存在，跳过文件导入 model=%s, file=%s", model_name, model_file)
                return

            safe_module_name = f"autoload_{model_name}_model"
            spec = importlib.util.spec_from_file_location(safe_module_name, model_file)
            if spec is None or spec.loader is None:
                logger.debug("无法构建模块spec model=%s, file=%s", model_name, model_file)
                return

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            logger.info("模型文件路径导入成功 model=%s, file=%s", model_name, model_file)
        except Exception as file_exc:
            logger.debug("文件路径导入失败 model=%s, err=%s", model_name, file_exc)


def _resolve_weight_path(weight_id: Optional[str]) -> Optional[Path]:
    """根据weight_id解析权重路径。

    Args:
        weight_id: 权重ID，可为空。为空时返回None。

    Returns:
        解析后的权重路径；为空则返回None。

    Raises:
        HTTPException: 当weight_id存在但无法解析时抛出。
    """
    if not weight_id:
        return None

    # 1) 若weight_id本身就是文件路径
    direct_path = Path(weight_id)
    if direct_path.exists() and direct_path.is_file():
        return direct_path.resolve()

    # 1.1) 若weight_id是纯文件名，优先在常见目录中检索
    # 支持示例：attention_unet_transformer.pth
    candidate_name = Path(weight_id).name
    if candidate_name.lower().endswith((".pth", ".pt")):
        backend_root = Path(__file__).resolve().parents[1]
        model_weight_candidates = sorted((backend_root / "models").glob(f"**/weights/{candidate_name}"))
        if model_weight_candidates:
            return model_weight_candidates[0].resolve()

        upload_weight_candidates = sorted(Path(WEIGHT_UPLOAD_ROOT).glob(f"**/{candidate_name}"))
        if upload_weight_candidates:
            return upload_weight_candidates[0].resolve()

    # 2) 从数据库元信息解析
    candidates = FileDAO.get_file_list(file_type="weight")
    for item in candidates:
        file_path = item.get("file_path")
        if file_path and Path(file_path).parent.name == weight_id:
            path = Path(file_path)
            if path.exists():
                return path.resolve()

    # 3) 回退：按权重目录查找
    candidate_dir = Path(WEIGHT_UPLOAD_ROOT) / weight_id
    if candidate_dir.exists():
        for file_path in sorted(candidate_dir.glob("*")):
            if file_path.is_file():
                return file_path.resolve()

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"weight_id不存在: {weight_id}",
    )


def _create_model_instance(
    model_meta: Dict[str, Any],
    resolved_weight_path: Optional[Path],
) -> tuple[torch.nn.Module, torch.device]:
    """创建模型实例并返回模型与设备。"""
    creator = model_meta["create"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        signature = inspect.signature(creator)
    except (TypeError, ValueError):
        signature = None

    try:
        if resolved_weight_path is not None:
            if signature and "weight_path" in signature.parameters:
                model = creator(weight_path=str(resolved_weight_path))
            else:
                model = creator()
                logger.warning(
                    "当前create函数不支持weight_path参数，忽略weight_id覆盖。"
                )
        else:
            model = creator()

        if not isinstance(model, torch.nn.Module):
            raise TypeError("create_model返回值不是torch.nn.Module实例")

        model = model.to(device)
        model.eval()
        return model, device
    except Exception as exc:
        raise RuntimeError(f"模型加载失败: {exc}") from exc


def _mask_to_base64(mask_array: np.ndarray) -> str:
    """将二维mask矩阵编码为PNG base64字符串。"""
    if not isinstance(mask_array, np.ndarray):
        raise ValueError("postprocess返回结果必须为numpy.ndarray")
    if mask_array.ndim != 2:
        raise ValueError(f"mask维度必须为2，当前为{mask_array.ndim}")

    normalized = (mask_array > 0).astype(np.uint8) * 255
    image = Image.fromarray(normalized, mode="L")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@router.get(
    "/models",
    summary="获取已注册模型列表",
    description="返回当前后端已注册的分割模型及其关键配置（name/input_size/num_classes）。",
)
async def get_registered_models() -> JSONResponse:
    """获取已注册模型列表。"""
    try:
        model_names = list_models()
        model_items = []
        for model_name in model_names:
            model_meta = get_model(model_name)
            if model_meta is None:
                continue

            model_config = model_meta.get("config", {})
            model_items.append(
                {
                    "name": model_name,
                    "input_size": model_config.get("input_size"),
                    "num_classes": model_config.get("num_classes"),
                }
            )

        return JSONResponse(
            {
                "code": 200,
                "msg": "获取模型列表成功",
                "data": model_items,
            }
        )
    except Exception as exc:
        logger.exception("[模型列表] 获取失败")
        return _build_error_response(
            code=500,
            msg=f"获取模型列表失败: {exc}",
            http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.post(
    "/predict",
    summary="通用分割推理接口",
    description=(
        "基于模型注册表执行分割推理：按model_name获取模型，"
        "可通过weight_id覆盖权重路径，返回mask的base64结果。"
    ),
)
async def predict(
    model_name: str = Form(..., description="模型名称（如unet）"),
    weight_id: Optional[str] = Form(None, description="可选权重ID，用于覆盖模型默认weight_path"),
    image_file: UploadFile = File(..., description="上传的OCTA灰度图像（png/jpg/bmp/tiff）"),
) -> JSONResponse:
    """通用分割模型推理接口。

    Args:
        model_name: 要使用的注册模型名称。
        weight_id: 可选权重ID。
        image_file: 上传图像文件。

    Returns:
        统一响应结构。
    """
    logger.info("[分割预测] 收到请求 model=%s, file=%s", model_name, image_file.filename)

    try:
        # ==================== 步骤1：参数与模型校验 ====================
        if not model_name or not model_name.strip():
            return _build_error_response(
                code=400,
                msg="参数错误: model_name不能为空",
                http_status=status.HTTP_400_BAD_REQUEST,
            )

        normalized_model_name = model_name.strip().lower()
        model_info = get_model(normalized_model_name)
        if model_info is None:
            _import_model_module_if_needed(normalized_model_name)
            model_info = get_model(normalized_model_name)

        if model_info is None:
            return _build_error_response(
                code=400,
                msg="模型未注册",
                http_status=status.HTTP_400_BAD_REQUEST,
            )

        # ==================== 步骤2：图片格式校验 ====================
        _validate_image(image_file)
        logger.info("[分割预测] ✓ 步骤2完成：图片格式校验通过")

        # ==================== 步骤3：解析权重路径（可选覆盖） ====================
        resolved_weight_path = _resolve_weight_path(weight_id)
        if resolved_weight_path:
            logger.info("[分割预测] ✓ 步骤3完成：权重覆盖路径=%s", resolved_weight_path)
        else:
            logger.info("[分割预测] ✓ 步骤3完成：使用模型默认权重")

        # ==================== 步骤4：创建模型 ====================
        try:
            model, device = _create_model_instance(model_info, resolved_weight_path)
            logger.info("[分割预测] ✓ 步骤4完成：模型加载成功 device=%s", device)
        except Exception as exc:
            logger.error("[分割预测] ✗ 步骤4失败：模型加载失败 %s", exc)
            return _build_error_response(
                code=500,
                msg=f"模型加载失败: {exc}",
                http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # ==================== 步骤5：图像预处理 ====================
        try:
            preprocess_fn = model_info["preprocess"]
            input_tensor = preprocess_fn(image_file)
            if not isinstance(input_tensor, torch.Tensor):
                raise TypeError("preprocess返回值不是torch.Tensor")
            input_tensor = input_tensor.to(device)
            logger.info(
                "[分割预测] ✓ 步骤5完成：图像预处理成功 shape=%s",
                input_tensor.shape,
            )
        except Exception as exc:
            logger.error("[分割预测] ✗ 步骤5失败：图像预处理失败 %s", exc)
            return _build_error_response(
                code=400,
                msg=f"图像预处理失败: {exc}",
                http_status=status.HTTP_400_BAD_REQUEST,
            )

        # ==================== 步骤6：模型推理 ====================
        try:
            start_time = time.perf_counter()
            with torch.no_grad():
                output_logits = model(input_tensor)
            infer_time = time.perf_counter() - start_time
            logger.info(
                "[分割预测] ✓ 步骤6完成：模型推理成功 output_shape=%s, time=%.4fs",
                output_logits.shape,
                infer_time,
            )
        except Exception as exc:
            logger.error("[分割预测] ✗ 步骤6失败：模型推理失败 %s", exc)
            return _build_error_response(
                code=500,
                msg=f"模型推理失败: {exc}",
                http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # ==================== 步骤7：后处理与Base64编码 ====================
        try:
            postprocess_fn = model_info["postprocess"]
            mask_array = postprocess_fn(output_logits)
            mask_base64 = _mask_to_base64(mask_array)
            logger.info(
                "[分割预测] ✓ 步骤7完成：后处理成功 mask_shape=%s, base64_len=%d",
                np.asarray(mask_array).shape,
                len(mask_base64),
            )
        except Exception as exc:
            logger.error("[分割预测] ✗ 步骤7失败：后处理失败 %s", exc)
            return _build_error_response(
                code=500,
                msg=f"后处理失败: {exc}",
                http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # ==================== 步骤8：返回响应 ====================
        response_data = {
            "mask_base64": mask_base64,
            "model_name": normalized_model_name,
            "device": str(device),
            "infer_time": round(infer_time, 4),
            "weight_id": weight_id,
        }

        logger.info(
            "[分割预测] ✓ 全部完成 model=%s, file=%s, device=%s, time=%.4fs",
            normalized_model_name,
            image_file.filename,
            device,
            infer_time,
        )

        return JSONResponse(
            {
                "code": 200,
                "msg": "推理成功",
                "data": response_data,
            }
        )

    except HTTPException as exc:
        logger.warning(
            "[分割预测] 参数/资源异常 model=%s, file=%s, status=%s, detail=%s",
            model_name,
            image_file.filename,
            exc.status_code,
            exc.detail,
        )
        return _build_error_response(
            code=exc.status_code,
            msg=str(exc.detail),
            http_status=exc.status_code,
        )
    except Exception as exc:
        logger.exception(
            "[分割预测] ✗ 未知错误 model=%s, file=%s",
            model_name,
            image_file.filename,
        )
        return _build_error_response(
            code=500,
            msg=f"服务器内部错误: {exc}",
            http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
