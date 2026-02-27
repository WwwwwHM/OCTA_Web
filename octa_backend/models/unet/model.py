"""
OCTA图像分割平台 - U-Net即插即用模板模块

本模块提供一个可注册到模型注册表的U-Net分割模板实现，包含：
1. U-Net网络结构（编码器、解码器、CAM模块）；
2. create_model()：按配置加载权重并返回eval模式模型；
3. preprocess()：UploadFile/文件对象 -> 单通道256x256归一化Tensor；
4. postprocess()：logits -> sigmoid -> 阈值二值化 -> 2D numpy mask；
5. config：模型元配置；
6. register_model()：模块导入时自动注册。

兼容性：
- PyTorch 2.0+
- CPU/GPU自动切换（基于torch.cuda.is_available）
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
from PIL import Image

from service.model_registry import register_model


# ==================== 日志配置 ====================
logger = logging.getLogger(__name__)


# ==================== 模型配置 ====================
config: Dict[str, Any] = {
    "name": "unet",
    "input_size": (256, 256),
    "num_classes": 1,
    "weight_path": "./weights/unet.pth",
}


# ==================== 自定义异常 ====================
class UNetTemplateError(Exception):
    """U-Net模板基础异常。"""


class WeightLoadError(UNetTemplateError):
    """模型权重加载异常。"""


class PreprocessError(UNetTemplateError):
    """图像预处理异常。"""


class PostprocessError(UNetTemplateError):
    """结果后处理异常。"""


# ==================== CAM模块定义 ====================
class CAMBlock(nn.Module):
    """
    Channel Attention Module（通道注意力模块）。

    Args:
        channels: 输入特征通道数。
        reduction: 通道压缩比例，默认16。
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced_channels = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False),
        )
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_attention = self.mlp(self.avg_pool(x))
        max_attention = self.mlp(self.max_pool(x))
        attention = self.activation(avg_attention + max_attention)
        return x * attention


# ==================== 基础卷积块 ====================
class DoubleConv(nn.Module):
    """
    U-Net双卷积块。

    Args:
        in_channels: 输入通道数。
        out_channels: 输出通道数。
    """

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


# ==================== U-Net模型定义（含CAM） ====================
class UNet(nn.Module):
    """
    U-Net分割模型（输入1通道，输出1通道）。

    架构：
    - 编码器：DoubleConv + MaxPool
    - 瓶颈层：DoubleConv + CAM
    - 解码器：TransposeConv + Skip Connection + DoubleConv + CAM

    Args:
        in_channels: 输入图像通道数，默认1。
        out_channels: 输出掩码通道数，默认1。
        features: 每层特征通道列表。
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: Optional[Tuple[int, ...]] = None,
    ) -> None:
        super().__init__()
        if features is None:
            features = (64, 128, 256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_blocks = nn.ModuleList()
        self.decoder_ups = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.decoder_cams = nn.ModuleList()

        current_channels = in_channels
        for feature in features:
            self.encoder_blocks.append(DoubleConv(current_channels, feature))
            current_channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.bottleneck_cam = CAMBlock(features[-1] * 2)

        decoder_in_channels = features[-1] * 2
        for feature in reversed(features):
            self.decoder_ups.append(
                nn.ConvTranspose2d(decoder_in_channels, feature, kernel_size=2, stride=2)
            )
            self.decoder_blocks.append(DoubleConv(feature * 2, feature))
            self.decoder_cams.append(CAMBlock(feature))
            decoder_in_channels = feature

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        for encoder in self.encoder_blocks:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        x = self.bottleneck_cam(x)

        skip_connections = skip_connections[::-1]

        for idx, (up, decoder, cam) in enumerate(
            zip(self.decoder_ups, self.decoder_blocks, self.decoder_cams)
        ):
            x = up(x)
            skip = skip_connections[idx]

            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            x = torch.cat((skip, x), dim=1)
            x = decoder(x)
            x = cam(x)

        logits = self.final_conv(x)
        return logits


# ==================== 工具函数 ====================
def _resolve_device() -> torch.device:
    """自动解析运行设备（优先GPU）。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_weight_path(weight_path: str) -> Path:
    """
    解析权重文件路径。

    优先级：
    1. 调用方传入的绝对/相对路径；
    2. 相对于当前模块目录的路径。

    Args:
        weight_path: 权重路径字符串。

    Returns:
        解析后的Path对象。
    """
    raw = Path(weight_path)
    if raw.is_absolute():
        return raw
    return (Path(__file__).resolve().parent / raw).resolve()


# ==================== 即插即用核心函数 ====================
def create_model(weight_path: Optional[str] = None) -> nn.Module:
    """
    创建U-Net模型并加载权重，返回eval模式实例。

    Args:
        weight_path: 权重文件路径；为空时使用config中的默认路径。

    Returns:
        已加载权重并切换到eval模式的模型实例。

    Raises:
        WeightLoadError: 权重不存在、格式错误或加载失败时抛出。
    """
    selected_weight_path = weight_path or str(config["weight_path"])
    device = _resolve_device()

    model = UNet(in_channels=1, out_channels=1)
    model.to(device)

    resolved_path = _resolve_weight_path(selected_weight_path)
    if not resolved_path.exists():
        raise WeightLoadError(f"U-Net权重文件不存在: {resolved_path}")

    try:
        checkpoint = torch.load(str(resolved_path), map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        if not isinstance(state_dict, dict):
            raise WeightLoadError(
                "权重文件格式无效：未找到可用state_dict。"
            )

        model.load_state_dict(state_dict)
        model.eval()
        logger.info("U-Net模型权重加载成功: %s, device=%s", resolved_path, device)
        return model
    except Exception as exc:
        raise WeightLoadError(
            f"U-Net权重加载失败，请检查文件与模型结构是否匹配: {resolved_path}, 错误: {exc}"
        ) from exc


def preprocess(file_obj: Any) -> torch.Tensor:
    """
    预处理输入图像并返回模型输入张量。

    处理流程：
    1. 读取UploadFile/文件对象；
    2. 转单通道灰度图；
    3. 缩放到256x256；
    4. 归一化（mean=0.5, std=0.5）；
    5. 添加batch与channel维度，返回形状[1, 1, 256, 256]。

    Args:
        file_obj: FastAPI UploadFile对象或具备read()能力的文件对象。

    Returns:
        预处理后的PyTorch张量。

    Raises:
        PreprocessError: 图像读取或预处理失败时抛出。
    """
    try:
        if hasattr(file_obj, "file") and file_obj.file is not None:
            content = file_obj.file.read()
        elif hasattr(file_obj, "read"):
            content = file_obj.read()
        else:
            raise PreprocessError("输入对象不支持读取，请传入UploadFile或文件对象。")

        if not isinstance(content, (bytes, bytearray)) or len(content) == 0:
            raise PreprocessError("读取到的图像内容为空或格式非法。")

        image = Image.open(BytesIO(content)).convert("L")
        image = image.resize(config["input_size"], Image.BILINEAR)

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_np = (image_np - 0.5) / 0.5

        tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(dtype=torch.float32, device=_resolve_device())
        return tensor
    except PreprocessError:
        raise
    except Exception as exc:
        raise PreprocessError(f"图像预处理失败: {exc}") from exc


def postprocess(logits: torch.Tensor) -> np.ndarray:
    """
    对模型输出执行后处理，生成二维二值mask。

    处理流程：
    1. 对logits执行sigmoid；
    2. 使用0.5阈值二值化；
    3. 返回二维numpy数组（0/1）。

    Args:
        logits: 模型输出logits张量，支持形状[1,1,H,W]、[1,H,W]或[H,W]。

    Returns:
        二维mask矩阵（numpy.ndarray, dtype=uint8）。

    Raises:
        PostprocessError: 输出张量形状非法或处理失败时抛出。
    """
    try:
        if not isinstance(logits, torch.Tensor):
            raise PostprocessError("postprocess输入必须是torch.Tensor。")

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            binary = (probs >= 0.5).to(dtype=torch.uint8)

            if binary.dim() == 4:
                mask = binary[0, 0]
            elif binary.dim() == 3:
                mask = binary[0]
            elif binary.dim() == 2:
                mask = binary
            else:
                raise PostprocessError(
                    f"不支持的输出维度: {binary.dim()}，期望2/3/4维。"
                )

            return mask.detach().cpu().numpy().astype(np.uint8)
    except PostprocessError:
        raise
    except Exception as exc:
        raise PostprocessError(f"模型输出后处理失败: {exc}") from exc


# ==================== 注册到模型注册表 ====================
register_model(
    model_name=config["name"],
    creator=create_model,
    preprocess=preprocess,
    postprocess=postprocess,
    config=config,
)


__all__ = [
    "UNet",
    "CAMBlock",
    "config",
    "create_model",
    "preprocess",
    "postprocess",
    "UNetTemplateError",
    "WeightLoadError",
    "PreprocessError",
    "PostprocessError",
]
