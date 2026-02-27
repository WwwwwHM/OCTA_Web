"""
Model loading utilities for FastAPI + PyTorch.

Key capabilities
- Auto-select device (prefer CUDA, fallback CPU)
- Load state_dict from weight path into a provided model class
- Enforce state_dict-only loading (no full-model load) to avoid env conflicts
- Eval mode + no-grad setup
- Simple path-based cache to avoid repeated initializations
- FastAPI-friendly exceptions with descriptive messages
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple, Type

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Path -> instantiated model cache
_MODEL_CACHE: Dict[Path, nn.Module] = {}


class ModelLoadError(Exception):
    """Raised when a model or weight fails to load."""


def get_device() -> torch.device:
    """Auto-detect CUDA; fallback CPU.

    Returns:
        torch.device selected device.
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("[模型加载] 使用GPU: %s", torch.cuda.get_device_name(0))
        return device

    device = torch.device("cpu")
    logger.info("[模型加载] GPU不可用，使用CPU")
    return device


def _extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    """Extract a state_dict from common checkpoint layouts."""

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "net", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]  # type: ignore[return-value]
        return checkpoint  # type: ignore[return-value]

    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()

    raise ValueError("权重文件不是有效的 state_dict 格式")


def load_model_by_weight_path(
    weight_path: str,
    model_cls: Type[nn.Module],
) -> Tuple[nn.Module, torch.device]:
    """Load model weights into an instance of model_cls.

    Args:
        weight_path: Path to the .pth/.pt file.
        model_cls: Model class to instantiate.

    Returns:
        (model, device) where model is loaded and in eval mode.

    Raises:
        ModelLoadError: on any failure (file missing, load error, key mismatch).
    """

    path = Path(weight_path)

    if path in _MODEL_CACHE:
        cached = _MODEL_CACHE[path]
        device = cached.device if hasattr(cached, "device") else get_device()
        logger.info("[模型加载] 命中缓存: %s", path)
        return cached, device

    if not path.exists():
        msg = f"权重文件不存在: {path}"
        logger.error("[模型加载] %s", msg)
        raise ModelLoadError(msg)

    device = get_device()

    try:
        checkpoint = torch.load(path, map_location=device)
    except Exception as exc:
        msg = f"加载权重失败: {type(exc).__name__} @ {path}: {exc}"
        logger.error("[模型加载] %s", msg)
        raise ModelLoadError(msg) from exc

    try:
        state = _extract_state_dict(checkpoint)
    except Exception as exc:
        msg = f"state_dict提取失败: {type(exc).__name__} @ {path}: {exc}"
        logger.error("[模型加载] %s", msg)
        raise ModelLoadError(msg) from exc

    try:
        model = model_cls()
    except Exception as exc:
        msg = f"模型实例化失败: {type(exc).__name__}: {exc}"
        logger.error("[模型加载] %s", msg)
        raise ModelLoadError(msg) from exc

    try:
        missing, unexpected = model.load_state_dict(state, strict=True)
    except Exception as exc:
        msg = f"state_dict加载失败: {type(exc).__name__}: {exc}"
        logger.error("[模型加载] %s", msg)
        raise ModelLoadError(msg) from exc

    if missing:
        preview = ", ".join(missing[:10])
        msg = f"权重缺少关键参数，共{len(missing)}个，示例: {preview}"
        logger.error("[模型加载] %s", msg)
        raise ModelLoadError(msg)

    if unexpected:
        preview = ", ".join(unexpected[:10])
        msg = f"权重包含多余参数，共{len(unexpected)}个，示例: {preview}"
        logger.error("[模型加载] %s", msg)
        raise ModelLoadError(msg)

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # 缓存
    _MODEL_CACHE[path] = model

    logger.info("[模型加载] ✓ 成功: %s | device=%s", path.name, device)
    return model, device


def clear_model_cache() -> None:
    """Clear in-memory model cache (for testing or hot-reload)."""

    _MODEL_CACHE.clear()
"""
模型权重加载模块

Fix: 平台优化 - 安全的模型加载与设备适配
功能：
1. 根据weight_id读取权重文件
2. 自动适配设备（GPU/CPU）
3. 安全加载模型（带异常捕获）
4. 设置model.eval()模式
5. 确保推理无梯度计算

作者：OCTA Web项目组
日期：2026-01-27
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple
import logging

# 配置日志
logger = logging.getLogger(__name__)


class ModelLoader:
    """模型加载器"""
    
    def __init__(self, device: str = 'auto'):
        """
        初始化加载器
        
        Args:
            device: 设备类型（'auto'/'cuda'/'cpu'）
        """
        self.device = self._get_device(device)
        logger.info(f"[模型加载器] 初始化完成，设备: {self.device}")
    
    def _get_device(self, device: str) -> torch.device:
        """
        获取推理设备
        
        Args:
            device: 设备类型
        
        Returns:
            torch.device对象
        """
        if device == 'auto':
            if torch.cuda.is_available():
                device_name = 'cuda'
                logger.info("[设备检测] ✓ GPU可用，优先使用CUDA")
            else:
                device_name = 'cpu'
                logger.info("[设备检测] ⚠ GPU不可用，使用CPU")
        else:
            device_name = device
        
        return torch.device(device_name)
    
    def load_model(
        self,
        model: nn.Module,
        weight_path: Path,
        strict: bool = False
    ) -> Tuple[bool, Optional[str], Optional[nn.Module]]:
        """
        加载模型权重
        
        Args:
            model: 模型实例（未加载权重）
            weight_path: 权重文件路径
            strict: 是否严格匹配keys（默认False，允许部分加载）
        
        Returns:
            (success, error_msg, loaded_model)
            - success: 是否加载成功
            - error_msg: 错误信息（成功时为None）
            - loaded_model: 加载后的模型（失败时为None）
        """
        try:
            logger.info(f"[模型加载] 开始加载权重: {weight_path}")
            
            # 步骤1：检查文件存在性
            if not weight_path.exists():
                error_msg = f"权重文件不存在: {weight_path}"
                logger.error(f"[模型加载] ✗ {error_msg}")
                return False, error_msg, None
            
            # 步骤2：加载checkpoint（强制使用当前设备）
            try:
                checkpoint = torch.load(
                    weight_path,
                    map_location=self.device,
                    weights_only=False  # 允许加载完整checkpoint
                )
            except Exception as e:
                error_msg = f"无法读取权重文件: {str(e)}"
                logger.error(f"[模型加载] ✗ {error_msg}")
                return False, error_msg, None
            
            # 步骤3：提取state_dict
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    logger.info("[模型加载] 使用 'state_dict' 键")
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    logger.info("[模型加载] 使用 'model_state_dict' 键")
                else:
                    state_dict = checkpoint
                    logger.info("[模型加载] 直接使用checkpoint作为state_dict")
            else:
                state_dict = checkpoint
                logger.info("[模型加载] checkpoint直接为state_dict格式")
            
            # 步骤4：加载state_dict到模型
            try:
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
                
                if missing_keys:
                    logger.warning(f"[模型加载] ⚠ 缺少keys: {len(missing_keys)}个")
                if unexpected_keys:
                    logger.warning(f"[模型加载] ⚠ 多余keys: {len(unexpected_keys)}个")
                
            except Exception as e:
                error_msg = f"加载state_dict失败: {str(e)}"
                logger.error(f"[模型加载] ✗ {error_msg}")
                return False, error_msg, None
            
            # 步骤5：移动模型到目标设备
            model = model.to(self.device)
            
            # 步骤6：设置为评估模式
            model.eval()
            
            # 步骤7：禁用梯度计算（节省内存）
            for param in model.parameters():
                param.requires_grad = False
            
            logger.info(f"[模型加载] ✓ 权重加载成功，设备: {self.device}")
            return True, None, model
            
        except Exception as e:
            error_msg = f"模型加载异常: {str(e)}"
            logger.error(f"[模型加载] ✗ {error_msg}")
            return False, error_msg, None
    
    def get_device_info(self) -> dict:
        """
        获取设备信息
        
        Returns:
            设备信息字典
        """
        info = {
            'device': str(self.device),
            'cuda_available': torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            info['cuda_device_count'] = torch.cuda.device_count()
            info['cuda_device_name'] = torch.cuda.get_device_name(0)
            info['cuda_memory_allocated'] = torch.cuda.memory_allocated(0) / 1024**2  # MB
        
        return info


# 全局加载器实例
_loader = None

def get_loader(device: str = 'auto') -> ModelLoader:
    """获取全局加载器实例"""
    global _loader
    if _loader is None:
        _loader = ModelLoader(device=device)
    return _loader
