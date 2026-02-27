"""
OCTA血管分割数据处理模块

功能：
1. preprocess_image: 读取UploadFile，转灰度，缩放，归一化，返回[1,1,H,W]张量
2. postprocess_mask: 将logits通过sigmoid+threshold二值化，还原原尺寸，返回numpy数组和base64字符串
3. 参数完全从config.py导入，禁止硬编码

作者：OCTA Web项目组
日期：2026-01-28
"""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from fastapi import UploadFile
from PIL import Image

# 从config导入所有预处理/后处理参数
from config.config import (
    IMAGE_SIZE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    BINARY_THRESHOLD,
    MASK_OUTPUT_FORMAT,
)

logger = logging.getLogger(__name__)


def preprocess_image(
    image_file: UploadFile,
    input_size: Tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE),
    mean: float = NORMALIZE_MEAN,
    std: float = NORMALIZE_STD,
) -> torch.Tensor:
    """预处理上传的图片文件，返回可直接输入模型的张量。

    Args:
        image_file: FastAPI上传的文件对象，支持png/jpg/bmp格式。
        input_size: 目标尺寸(width, height)，从config导入。
        mean: 归一化均值，从config导入。
        std: 归一化标准差，从config导入。

    Returns:
        torch.Tensor: 预处理后的张量，shape为[1, 1, H, W]。

    Raises:
        ValueError: 当图片读取失败、格式不支持或尺寸转换失败时。

    处理流程：
        1. 读取UploadFile内容
        2. 转换为灰度图(convert('L'))
        3. 保持比例缩放至input_size（填充黑边或裁剪）
        4. 归一化：(pixel/255 - mean) / std
        5. 转torch张量并添加batch维度
    """

    try:
        # 读取文件内容
        contents = image_file.file.read()
        image = Image.open(BytesIO(contents))
    except Exception as exc:
        msg = f"图片读取失败: {exc}"
        logger.error("[预处理] %s", msg)
        raise ValueError(msg) from exc

    try:
        # 转为单通道灰度图
        image = image.convert("L")
    except Exception as exc:
        msg = f"灰度转换失败: {exc}"
        logger.error("[预处理] %s", msg)
        raise ValueError(msg) from exc

    try:
        # 保持比例缩放（填充黑边）
        w, h = image.size
        target_w, target_h = input_size

        # 计算缩放比例
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # 缩放图像
        image_resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # 创建黑色画布并居中粘贴
        canvas = Image.new("L", input_size, 0)
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        canvas.paste(image_resized, (offset_x, offset_y))

    except Exception as exc:
        msg = f"尺寸缩放失败: {exc}"
        logger.error("[预处理] %s", msg)
        raise ValueError(msg) from exc

    try:
        # 转numpy并归一化
        arr = np.array(canvas, dtype=np.float32)
        normalized = (arr / 255.0 - mean) / std

        # 转torch张量并添加维度 [H,W] -> [1,1,H,W]
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)

        logger.info("[预处理] ✓ 完成: shape=%s", tensor.shape)
        return tensor

    except Exception as exc:
        msg = f"张量转换失败: {exc}"
        logger.error("[预处理] %s", msg)
        raise ValueError(msg) from exc


def postprocess_mask(
    mask_logits: torch.Tensor,
    original_size: Tuple[int, int],
    threshold: float = BINARY_THRESHOLD,
) -> Tuple[np.ndarray, str]:
    """后处理模型输出logits，返回二值化mask数组和base64字符串。

    Args:
        mask_logits: 模型输出张量，shape为[1, 1, H, W]。
        original_size: 原始图片尺寸(width, height)。
        threshold: 二值化阈值，从config导入。

    Returns:
        Tuple[np.ndarray, str]:
            - mask_array: [H, W] uint8数组，值为0或255。
            - base64_str: PNG格式的base64编码字符串。

    Raises:
        ValueError: 当尺寸还原或base64编码失败时。

    处理流程：
        1. 移除batch/channel维度
        2. sigmoid激活
        3. threshold二值化
        4. 还原至original_size
        5. 转uint8 (0/255)
        6. 编码为base64
    """

    try:
        # 移除维度并转CPU
        mask = mask_logits.squeeze().detach().cpu()

        # sigmoid激活
        mask = torch.sigmoid(mask)

        # 转numpy
        mask_np = mask.numpy()

        # 二值化
        binary = (mask_np > threshold).astype(np.uint8) * 255

        # 还原原始尺寸
        mask_pil = Image.fromarray(binary, mode="L")
        mask_resized = mask_pil.resize(original_size, Image.Resampling.NEAREST)
        mask_array = np.array(mask_resized, dtype=np.uint8)

    except Exception as exc:
        msg = f"mask后处理失败: {exc}"
        logger.error("[后处理] %s", msg)
        raise ValueError(msg) from exc

    try:
        # 转base64
        buffer = BytesIO()
        Image.fromarray(mask_array, mode="L").save(buffer, format="PNG")
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        logger.info("[后处理] ✓ 完成: mask shape=%s, base64 len=%d", mask_array.shape, len(base64_str))
        return mask_array, base64_str

    except Exception as exc:
        msg = f"base64编码失败: {exc}"
        logger.error("[后处理] %s", msg)
        raise ValueError(msg) from exc


# ==================== 向后兼容的类包装器 ====================


class DataProcessor:
    """保留旧接口的兼容性包装类。"""

    def preprocess(self, image_path: Path, device: str = "cpu") -> Tuple[torch.Tensor, Tuple[int, int]]:
        """基于路径的预处理（向后兼容）。"""
        img = Image.open(image_path)
        original_size = img.size

        # 创建临时UploadFile模拟
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        class FakeUpload:
            file = buffer

        tensor = preprocess_image(FakeUpload())
        return tensor.to(device), original_size

    def postprocess(
        self,
        output_tensor: torch.Tensor,
        original_size: Tuple[int, int],
        threshold: float = BINARY_THRESHOLD,
    ) -> np.ndarray:
        """后处理（向后兼容）。"""
        mask_array, _ = postprocess_mask(output_tensor, original_size, threshold)
        return mask_array

    def mask_to_base64(self, mask_array: np.ndarray) -> str:
        """mask转base64（向后兼容）。"""
        buffer = BytesIO()
        Image.fromarray(mask_array, mode="L").save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def save_mask(self, mask_array: np.ndarray, output_path: Path) -> bool:
        """保存mask到本地。"""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask_array, mode="L").save(output_path, format="PNG")
            logger.info("[保存mask] ✓ %s", output_path)
            return True
        except Exception as exc:
            logger.error("[保存mask] ✗ %s", exc)
            return False


_processor = None


def get_processor() -> DataProcessor:
    """获取全局处理器实例（向后兼容）。"""
    global _processor
    if _processor is None:
        _processor = DataProcessor()
    return _processor
