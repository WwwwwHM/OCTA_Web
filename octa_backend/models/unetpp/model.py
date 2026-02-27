"""
任意分割模型即插即用适配模板（UNet++示例）

模板固定模块：
1. 模型类定义（可替换为UNet++/DeepLabV3等）
2. create_model()：统一模型创建接口
3. preprocess()：统一输入预处理接口
4. postprocess()：统一输出后处理接口
5. config：统一模型配置字段
6. register_model()：统一注册逻辑

新增模型时，只需按“# 需适配修改”标记替换核心逻辑，
无需改动函数名和输入输出格式。
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import UploadFile
from PIL import Image

from service.model_registry import register_model


logger = logging.getLogger(__name__)


# ==================== 统一配置（固定字段） ====================
config: Dict[str, Any] = {
    "name": "unetpp",  # 需适配修改：模型名称（如 deeplabv3）
    "input_size": (256, 256),  # 需适配修改：输入尺寸
    "num_classes": 1,  # 需适配修改：类别数
    "weight_path": "./weights/unetpp.pth",  # 需适配修改：默认权重路径
}


# ==================== 自定义异常 ====================
class ModelTemplateError(Exception):
    """模板基础异常。"""


class ModelCreateError(ModelTemplateError):
    """模型创建/加载异常。"""


class PreprocessError(ModelTemplateError):
    """图像预处理异常。"""


class PostprocessError(ModelTemplateError):
    """结果后处理异常。"""


# ==================== 模型结构定义（UNet++ 示例） ====================
class ConvBlock(nn.Module):
    """基础卷积块。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetPlusPlus(nn.Module):
    """
    UNet++ 简化示例实现。

    说明：
    - 该结构用于演示“即插即用”接入流程；
    - 若需替换为其他模型，仅保留类名与输出logits形状语义一致即可。
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1) -> None:
        super().__init__()

        # 需适配修改：根据目标模型替换网络结构
        features = [32, 64, 128, 256]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc0 = ConvBlock(in_channels, features[0])
        self.enc1 = ConvBlock(features[0], features[1])
        self.enc2 = ConvBlock(features[1], features[2])
        self.enc3 = ConvBlock(features[2], features[3])

        self.up3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)

        self.dec2 = ConvBlock(features[2] * 2, features[2])
        self.dec1 = ConvBlock(features[1] * 2, features[1])
        self.dec0 = ConvBlock(features[0] * 2, features[0])

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 编码
        x0 = self.enc0(x)
        x1 = self.enc1(self.pool(x0))
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))

        # 解码（UNet++风格的简化跳连）
        d2 = self.up3(x3)
        if d2.shape[-2:] != x2.shape[-2:]:
            d2 = F.interpolate(d2, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([x2, d2], dim=1))

        d1 = self.up2(d2)
        if d1.shape[-2:] != x1.shape[-2:]:
            d1 = F.interpolate(d1, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([x1, d1], dim=1))

        d0 = self.up1(d1)
        if d0.shape[-2:] != x0.shape[-2:]:
            d0 = F.interpolate(d0, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        d0 = self.dec0(torch.cat([x0, d0], dim=1))

        logits = self.final_conv(d0)
        return logits


# ==================== 固定接口：create_model ====================
def create_model(weight_path: Optional[str] = None) -> nn.Module:
    """
    创建模型实例并自动加载权重（统一接口）。

    Args:
        weight_path: 可选权重路径；为空时使用config['weight_path']。

    Returns:
        已切换到eval模式的模型实例。

    Raises:
        ModelCreateError: 当模型构建或权重加载失败时抛出。
    """
    # 需适配修改：如需特定设备策略，可在此扩展
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_weight_path = weight_path or str(config["weight_path"])

    # 需适配修改：切换为你的模型类（如DeepLabV3）
    model = UNetPlusPlus(in_channels=1, out_channels=config["num_classes"]).to(device)

    resolved_path = Path(target_weight_path)
    if not resolved_path.is_absolute():
        resolved_path = (Path(__file__).resolve().parent / resolved_path).resolve()

    if not resolved_path.exists():
        raise ModelCreateError(f"权重文件不存在: {resolved_path}")

    try:
        checkpoint = torch.load(str(resolved_path), map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

        if not isinstance(state_dict, dict):
            raise ModelCreateError("权重格式无效：未找到state_dict。")

        model.load_state_dict(state_dict, strict=False)
        model.eval()
        logger.info("模型加载成功 model=%s, device=%s, weight=%s", config["name"], device, resolved_path)
        return model
    except Exception as exc:
        raise ModelCreateError(f"模型创建/权重加载失败: {exc}") from exc


# ==================== 固定接口：preprocess ====================
def preprocess(image_file: UploadFile) -> torch.Tensor:
    """
    统一预处理接口：UploadFile -> PyTorch张量。

    Args:
        image_file: FastAPI上传文件对象。

    Returns:
        形状为[1, 1, H, W]的float32张量。

    Raises:
        PreprocessError: 当图像解析或转换失败时抛出。
    """
    try:
        if image_file is None:
            raise PreprocessError("image_file不能为空。")

        content = image_file.file.read()
        if not content:
            raise PreprocessError("上传文件内容为空。")

        image = Image.open(BytesIO(content)).convert("L")

        # 需适配修改：根据新模型调整输入尺寸
        image = image.resize(config["input_size"], Image.BILINEAR)

        image_np = np.asarray(image, dtype=np.float32) / 255.0

        # 需适配修改：根据新模型训练配置调整归一化参数
        mean, std = 0.5, 0.5
        image_np = (image_np - mean) / std

        tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0).float()
        return tensor
    except PreprocessError:
        raise
    except Exception as exc:
        raise PreprocessError(f"预处理失败: {exc}") from exc


# ==================== 固定接口：postprocess ====================
def postprocess(logits: torch.Tensor) -> np.ndarray:
    """
    统一后处理接口：模型输出 -> 二维mask矩阵。

    Args:
        logits: 模型输出张量（通常为logits）。

    Returns:
        二维二值mask（numpy.uint8，元素为0或1）。

    Raises:
        PostprocessError: 当输入无效或处理失败时抛出。
    """
    try:
        if not isinstance(logits, torch.Tensor):
            raise PostprocessError("logits必须是torch.Tensor。")

        # 需适配修改：如模型输出已是概率图，可去掉sigmoid
        probs = torch.sigmoid(logits)

        # 需适配修改：根据任务调整阈值
        threshold = 0.5
        binary = (probs >= threshold).to(torch.uint8)

        if binary.dim() == 4:
            mask = binary[0, 0]
        elif binary.dim() == 3:
            mask = binary[0]
        elif binary.dim() == 2:
            mask = binary
        else:
            raise PostprocessError(f"不支持的输出维度: {binary.dim()}。")

        return mask.detach().cpu().numpy().astype(np.uint8)
    except PostprocessError:
        raise
    except Exception as exc:
        raise PostprocessError(f"后处理失败: {exc}") from exc


# ==================== 固定模块：模型注册逻辑 ====================
register_model(
    model_name=config["name"],
    creator=create_model,
    preprocess=preprocess,
    postprocess=postprocess,
    config=config,
)


__all__ = [
    "UNetPlusPlus",
    "config",
    "create_model",
    "preprocess",
    "postprocess",
    "ModelTemplateError",
    "ModelCreateError",
    "PreprocessError",
    "PostprocessError",
]
