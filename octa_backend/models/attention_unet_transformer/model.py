"""
OCTA图像分割平台 - Attention U-Net Transformer 即插即用模型

本模块集成以下可选结构：
1. ResidualBlock（残差卷积块）
2. 门控注意力（Gated Attention）
3. 空间注意力（Spatial Attention）
4. PDE注意力（PDE Attention）
5. ASPP瓶颈增强
6. Edge Branch边缘分支（可选返回）

并通过 register_model 自动注册到后端模型注册表。
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from service.model_registry import register_model


logger = logging.getLogger(__name__)


_RUNTIME_IN_CHANNELS: int = 3


config: Dict[str, Any] = {
    "name": "attention_unet_transformer",
    "input_size": (256, 256),
    "num_classes": 1,
    "weight_path": "./weights/unet+transformer+Res+GatedAT_20260225_1900.pth",
    "model_kwargs": {
        "features": (64, 128, 256, 512),
        "trans_dim": 1024,
        "use_residual": False,
        "use_gated_attention": False,
        "use_spatial_attention": False,
        "use_pde_attention": False,
        "use_aspp": False,
        "use_edge_branch": False,
        "return_edge": False,
    },
    "postprocess": {
        "threshold": 0.5,
        "smooth_kernel": 1,
        "min_component_area": 0,
        "min_hole_area": 0,
    },
}


class ModelCreateError(Exception):
    """模型创建/权重加载异常。"""


class PreprocessError(Exception):
    """图像预处理异常。"""


class PostprocessError(Exception):
    """输出后处理异常。"""


def _infer_in_channels_from_state_dict(state_dict: Dict[str, Any]) -> int:
    """从权重字典推断模型输入通道数。"""
    candidate_keys = (
        "downs.0.conv1.weight",
        "downs.0.block.0.weight",
    )
    for key in candidate_keys:
        weight = state_dict.get(key)
        if isinstance(weight, torch.Tensor) and weight.ndim == 4:
            inferred = int(weight.shape[1])
            if inferred > 0:
                return inferred
    return 3


def _adapt_legacy_state_dict_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """兼容旧版注意力模块命名（attentions -> gated_attentions）。"""
    adapted: Dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("attentions."):
            new_key = new_key.replace("attentions.", "gated_attentions.", 1)
        adapted[new_key] = value
    return adapted


def _infer_model_kwargs_from_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """从权重键名推断训练时模型开关，和离线predict.py保持一致。"""
    inferred = {
        "use_residual": True,
        "use_gated_attention": True,
        "use_spatial_attention": False,
        "use_pde_attention": False,
        "use_aspp": False,
        "use_edge_branch": False,
        "return_edge": False,
    }

    if any(key.startswith("downs.0.conv1") for key in state_dict.keys()):
        inferred["use_residual"] = True
    elif any(key.startswith("downs.0.block.0") or key.startswith("downs.0.0") for key in state_dict.keys()):
        inferred["use_residual"] = False

    inferred["use_gated_attention"] = any(
        key.startswith("gated_attentions.") or key.startswith("attentions.")
        for key in state_dict.keys()
    )
    inferred["use_spatial_attention"] = any(key.startswith("spatial_attentions.") for key in state_dict.keys())
    inferred["use_pde_attention"] = any(key.startswith("pde_attentions.") for key in state_dict.keys())
    inferred["use_aspp"] = any(key.startswith("aspp.") for key in state_dict.keys())
    inferred["use_edge_branch"] = any(key.startswith("edge_branch.") for key in state_dict.keys())

    return inferred


def _remove_small_components(mask: np.ndarray, min_area: int, target_value: int) -> np.ndarray:
    """移除指定像素值的小连通域（8邻域）。"""
    if min_area <= 0:
        return mask

    height, width = mask.shape
    visited = np.zeros((height, width), dtype=bool)
    output = mask.copy()

    for row in range(height):
        for col in range(width):
            if visited[row, col] or output[row, col] != target_value:
                continue

            stack = [(row, col)]
            visited[row, col] = True
            component_pixels = [(row, col)]

            while stack:
                current_row, current_col = stack.pop()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        next_row = current_row + dr
                        next_col = current_col + dc
                        if not (0 <= next_row < height and 0 <= next_col < width):
                            continue
                        if visited[next_row, next_col]:
                            continue
                        if output[next_row, next_col] != target_value:
                            continue

                        visited[next_row, next_col] = True
                        stack.append((next_row, next_col))
                        component_pixels.append((next_row, next_col))

            if len(component_pixels) < min_area:
                fill_value = 1 - target_value
                for pixel_row, pixel_col in component_pixels:
                    output[pixel_row, pixel_col] = fill_value

    return output


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        return out


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3_6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.conv3_12 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.conv3_18 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.out_conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv3_6(x)
        x3 = self.conv3_12(x)
        x4 = self.conv3_18(x)
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)
        return self.out_conv(x_cat)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.conv(torch.cat([avg_out, max_out], dim=1))
        attn = self.sigmoid(attn)
        return x * attn


class PDEAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.grad_x = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.grad_y = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.weight_conv = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = self.grad_x(x)
        gy = self.grad_y(x)
        grad = torch.cat([gx, gy], dim=1)
        attn = self.sigmoid(self.weight_conv(grad))
        return x * attn


class EdgeBranch(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        edge = self.conv(x)
        return F.interpolate(edge, size=target_size, mode="bilinear", align_corners=False)


class AttentionBlock(nn.Module):
    def __init__(self, f_g: int, f_l: int, f_int: int) -> None:
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, 1),
            nn.BatchNorm2d(f_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, 1),
            nn.BatchNorm2d(f_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[-2:] != x1.shape[-2:]:
            g1 = F.interpolate(g1, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class SimpleTransformerEncoder(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        x_flat = x.flatten(2).permute(2, 0, 1)
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out
        x_norm2 = self.norm2(x_flat)
        x_flat = x_flat + self.mlp(x_norm2)
        return x_flat.permute(1, 2, 0).reshape(batch, channels, height, width)


class UNetTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: Sequence[int] = (64, 128, 256, 512),
        trans_dim: int = 1024,
        use_residual: bool = False,
        use_gated_attention: bool = True,
        use_spatial_attention: bool = False,
        use_pde_attention: bool = False,
        use_aspp: bool = False,
        use_edge_branch: bool = False,
        return_edge: bool = False,
    ) -> None:
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.gated_attentions = nn.ModuleList()
        self.spatial_attentions = nn.ModuleList()
        self.pde_attentions = nn.ModuleList()

        self.use_residual = use_residual
        self.use_gated_attention = use_gated_attention
        self.use_spatial_attention = use_spatial_attention
        self.use_pde_attention = use_pde_attention
        self.use_aspp = use_aspp
        self.use_edge_branch = use_edge_branch
        self.return_edge = return_edge

        def make_block(in_ch: int, out_ch: int) -> nn.Module:
            return ResidualBlock(in_ch, out_ch) if self.use_residual else ConvBlock(in_ch, out_ch)

        current_in = in_channels
        for feature in features:
            self.downs.append(make_block(current_in, feature))
            current_in = feature

        self.bottleneck_conv = nn.Conv2d(features[-1], trans_dim, kernel_size=1)
        self.transformer = SimpleTransformerEncoder(trans_dim)
        bottleneck_out_channels = features[-1] * 2
        self.bottleneck_deconv = nn.Conv2d(trans_dim, bottleneck_out_channels, kernel_size=1)

        self.aspp = ASPP(bottleneck_out_channels, bottleneck_out_channels) if self.use_aspp else nn.Identity()
        self.edge_branch = EdgeBranch(bottleneck_out_channels) if self.use_edge_branch else None

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(make_block(feature * 2, feature))

            if self.use_gated_attention:
                self.gated_attentions.append(AttentionBlock(f_g=feature, f_l=feature, f_int=max(feature // 2, 1)))
            if self.use_spatial_attention:
                self.spatial_attentions.append(SpatialAttention())
            if self.use_pde_attention:
                self.pde_attentions.append(PDEAttention(feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        input_size = x.shape[2:]
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.bottleneck_conv(x)
        x = self.transformer(x)
        x = self.bottleneck_deconv(x)
        x = self.aspp(x)

        edge_out = self.edge_branch(x, target_size=input_size) if self.edge_branch is not None else None

        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape[-2:] != skip_connection.shape[-2:]:
                x = F.interpolate(x, size=skip_connection.shape[-2:], mode="bilinear", align_corners=False)

            refined_skip = skip_connection
            if self.use_gated_attention:
                refined_skip = self.gated_attentions[idx // 2](g=x, x=refined_skip)
            if self.use_spatial_attention:
                refined_skip = self.spatial_attentions[idx // 2](refined_skip)
            if self.use_pde_attention:
                refined_skip = self.pde_attentions[idx // 2](refined_skip)

            if x.shape[-2:] != refined_skip.shape[-2:]:
                x = F.interpolate(x, size=refined_skip.shape[-2:], mode="bilinear", align_corners=False)

            x = torch.cat((refined_skip, x), dim=1)
            x = self.ups[idx + 1](x)

        seg_out = self.final_conv(x)
        if self.return_edge and edge_out is not None:
            return seg_out, edge_out
        return seg_out


def create_model(weight_path: Optional[str] = None) -> nn.Module:
    """创建模型并按需加载权重。"""
    global _RUNTIME_IN_CHANNELS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = dict(config.get("model_kwargs", {}))

    target_path = Path(weight_path or str(config["weight_path"]))
    if not target_path.is_absolute():
        target_path = (Path(__file__).resolve().parent / target_path).resolve()

    if not target_path.exists():
        _RUNTIME_IN_CHANNELS = int(kwargs.pop("in_channels", 3))
        model = UNetTransformer(
            in_channels=_RUNTIME_IN_CHANNELS,
            out_channels=config["num_classes"],
            **kwargs,
        ).to(device)
        logger.warning("权重文件不存在，使用随机初始化参数继续运行: %s", target_path)
        model.eval()
        return model

    try:
        checkpoint = torch.load(str(target_path), map_location=device)

        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint
        else:
            state_dict = checkpoint

        if not isinstance(state_dict, dict):
            raise ModelCreateError("权重格式无效：未找到可用state_dict。")

        state_dict = _adapt_legacy_state_dict_keys(state_dict)

        _RUNTIME_IN_CHANNELS = _infer_in_channels_from_state_dict(state_dict)

        inferred_kwargs = _infer_model_kwargs_from_state_dict(state_dict)
        runtime_kwargs = dict(kwargs)
        runtime_kwargs.update(inferred_kwargs)
        runtime_kwargs.pop("in_channels", None)
        model = UNetTransformer(
            in_channels=_RUNTIME_IN_CHANNELS,
            out_channels=config["num_classes"],
            **runtime_kwargs,
        ).to(device)

        model.load_state_dict(state_dict, strict=True)
        model.eval()
        logger.info(
            "UNetTransformer加载完成: %s, in_channels=%d, inferred=%s",
            target_path,
            _RUNTIME_IN_CHANNELS,
            inferred_kwargs,
        )
        return model
    except Exception as exc:
        raise ModelCreateError(f"模型创建/权重加载失败: {exc}") from exc


def preprocess(file_obj: Any) -> torch.Tensor:
    """将输入图像预处理为模型张量。"""
    try:
        if hasattr(file_obj, "file") and file_obj.file is not None:
            content = file_obj.file.read()
        elif hasattr(file_obj, "read"):
            content = file_obj.read()
        else:
            raise PreprocessError("输入对象不支持读取，请传入UploadFile或文件对象。")

        if not isinstance(content, (bytes, bytearray)) or len(content) == 0:
            raise PreprocessError("读取到的图像内容为空或格式非法。")

        input_mode = "RGB" if _RUNTIME_IN_CHANNELS == 3 else "L"
        image = Image.open(BytesIO(content)).convert(input_mode)
        image = image.resize(config["input_size"], Image.BILINEAR)

        image_np = np.asarray(image, dtype=np.float32) / 255.0

        if _RUNTIME_IN_CHANNELS == 3:
            if image_np.ndim == 2:
                image_np = np.stack([image_np, image_np, image_np], axis=-1)
            tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float()
        else:
            if image_np.ndim == 3:
                image_np = image_np[..., 0]
            tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0).float()

        return tensor
    except PreprocessError:
        raise
    except Exception as exc:
        raise PreprocessError(f"图像预处理失败: {exc}") from exc


def postprocess(logits: torch.Tensor | Tuple[torch.Tensor, torch.Tensor]) -> np.ndarray:
    """将模型输出转换为二维二值mask。"""
    try:
        if isinstance(logits, tuple):
            logits = logits[0]

        if not isinstance(logits, torch.Tensor):
            raise PostprocessError("logits必须是torch.Tensor或(seg, edge)元组。")

        with torch.no_grad():
            post_cfg = config.get("postprocess", {})
            threshold = float(post_cfg.get("threshold", 0.5))
            smooth_kernel = int(post_cfg.get("smooth_kernel", 3))
            min_component_area = int(post_cfg.get("min_component_area", 0))
            min_hole_area = int(post_cfg.get("min_hole_area", 0))

            probs = torch.sigmoid(logits)

            if smooth_kernel >= 3 and smooth_kernel % 2 == 1:
                probs = F.avg_pool2d(
                    probs,
                    kernel_size=smooth_kernel,
                    stride=1,
                    padding=smooth_kernel // 2,
                )

            binary = (probs >= threshold).to(dtype=torch.uint8)

            if binary.dim() == 4:
                mask = binary[0, 0]
            elif binary.dim() == 3:
                mask = binary[0]
            elif binary.dim() == 2:
                mask = binary
            else:
                raise PostprocessError(f"不支持的输出维度: {binary.dim()}。")

            mask_np = mask.detach().cpu().numpy().astype(np.uint8)

            if min_component_area > 0:
                mask_np = _remove_small_components(mask_np, min_component_area, target_value=1)

            if min_hole_area > 0:
                mask_np = _remove_small_components(mask_np, min_hole_area, target_value=0)

            return mask_np
    except PostprocessError:
        raise
    except Exception as exc:
        raise PostprocessError(f"后处理失败: {exc}") from exc


register_model(
    model_name=config["name"],
    creator=create_model,
    preprocess=preprocess,
    postprocess=postprocess,
    config=config,
)


__all__ = [
    "UNetTransformer",
    "config",
    "create_model",
    "preprocess",
    "postprocess",
]
