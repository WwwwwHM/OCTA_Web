"""
Weight validation utilities for FastAPI uploads.

This module provides two core validators:
- validate_weight_format: checks filename extension (.pth/.pt)
- validate_weight_content: checks state_dict compatibility against a model class

The design targets U-Net variants used in this project (with CAM and multi-scale
fusion blocks). All errors are expressed as user-friendly messages that can be
returned by FastAPI handlers with HTTP 400.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
from fastapi import UploadFile

logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".pth", ".pt"}


def validate_weight_format(upload: UploadFile) -> Tuple[bool, str]:
    """Validate uploaded weight filename extension.

    Args:
        upload: Incoming FastAPI UploadFile.

    Returns:
        (is_valid, msg): True when extension is supported; otherwise False with
        a descriptive message.
    """

    filename = upload.filename or ""
    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED_EXTS:
        return False, "文件格式不支持，仅允许 .pth 或 .pt"

    return True, "格式校验通过"


def _extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    """Extract a state_dict from common checkpoint layouts.

    Supports raw state_dict or wrapped forms containing keys like state_dict,
    model_state_dict, net, or model.
    """

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "net", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]  # type: ignore[return-value]
        # Fallback: assume the whole dict is the state_dict
        return checkpoint  # type: ignore[return-value]

    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()

    raise ValueError("权重文件不是有效的PyTorch state_dict 格式")


def validate_weight_content(weight_path: str, model_cls: Type[nn.Module]) -> Tuple[bool, str]:
    """Validate that a weight file matches the given model class.

    Args:
        weight_path: Path to the weight file on disk.
        model_cls: The nn.Module class used to generate the reference key list.

    Returns:
        (is_valid, msg): True when keys match; otherwise False with a detailed
        message describing missing keys.
    """

    path = Path(weight_path)
    if not path.exists():
        return False, f"权重文件不存在: {path}"

    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:  # torch.load failure
        logger.error("[权重校验] torch.load 失败: %s", exc)
        return False, f"无法加载权重文件，可能已损坏：{exc}"

    try:
        loaded_state = _extract_state_dict(checkpoint)
    except Exception as exc:
        logger.error("[权重校验] state_dict 提取失败: %s", exc)
        return False, f"权重文件不是有效的 state_dict：{exc}"

    if not isinstance(loaded_state, dict):
        return False, "权重内容无效：缺少 state_dict"

    # Reference keys from fresh model instance
    try:
        model = model_cls()
    except Exception as exc:
        logger.error("[权重校验] 模型实例化失败: %s", exc)
        return False, f"模型实例化失败：{exc}"

    expected_keys = set(model.state_dict().keys())
    loaded_keys = set(loaded_state.keys())

    missing = sorted(expected_keys - loaded_keys)
    if missing:
        preview = ", ".join(missing[:10])
        return False, f"权重缺少关键参数，共{len(missing)}个，示例: {preview}"

    return True, "权重内容与模型匹配"


class WeightValidator:
    """Composite validator combining format/size/state_dict checks."""

    def __init__(self, max_size_mb: int = 200):
        self.max_size_mb = max_size_mb
        self.max_size_bytes = max_size_mb * 1024 * 1024

    def validate_file(
        self,
        file_path: Path,
        model_type: str = "unet",
        model_cls: Optional[Type[nn.Module]] = None,
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, object]]]:
        """Full validation pipeline used by upload handlers.

        Returns (is_valid, error_msg, metadata).
        """

        # Format check
        if file_path.suffix.lower() not in SUPPORTED_EXTS:
            return False, "文件格式不支持，仅允许 .pth 或 .pt", None

        if not file_path.exists():
            return False, f"文件不存在: {file_path}", None

        # Size check
        size_bytes = file_path.stat().st_size
        if size_bytes == 0:
            return False, "文件为空", None
        if size_bytes > self.max_size_bytes:
            size_mb = size_bytes / 1024 / 1024
            return False, f"文件过大: {size_mb:.2f}MB，超过限制 {self.max_size_mb}MB", None

        # Choose model class
        if model_cls is None and model_type == "unet":
            try:
                from models.unet import UNetUnderfittingFix

                model_cls = UNetUnderfittingFix
            except Exception as exc:
                logger.warning("[权重校验] 导入默认U-Net模型失败: %s", exc)

        # Content check (optional when model class is available)
        if model_cls is not None:
            ok, msg = validate_weight_content(str(file_path), model_cls)
            if not ok:
                return False, msg, None

        # Build metadata snapshot
        metadata: Dict[str, object] = {
            "file_size_mb": round(size_bytes / 1024 / 1024, 4),
            "file_name": file_path.name,
        }

        try:
            checkpoint = torch.load(file_path, map_location="cpu")
            state = _extract_state_dict(checkpoint)
            metadata["total_keys"] = len(state)
            metadata["sample_keys"] = list(state.keys())[:10]
        except Exception:
            # Non-fatal for metadata
            pass

        logger.info("[权重校验] ✓ 校验通过: %s", file_path.name)
        return True, None, metadata


_validator: Optional[WeightValidator] = None


def get_validator(max_size_mb: int = 200) -> WeightValidator:
    """Singleton accessor used by routers."""

    global _validator
    if _validator is None:
        _validator = WeightValidator(max_size_mb=max_size_mb)
    return _validator
"""
权重文件校验模块

Fix: 平台优化 - 权重上传前的严格校验
功能：
1. 文件格式校验（仅.pth/.pt）
2. 文件大小校验（≤200MB）
3. State_dict完整性校验（与本地U-Net模型key匹配）
4. 返回明确的错误信息

作者：OCTA Web项目组
日期：2026-01-27
"""

import torch
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import logging

# 配置日志
logger = logging.getLogger(__name__)


class WeightValidator:
    """权重文件校验器"""
    
    # U-Net模型必需的关键state_dict keys（部分）
    # 完整列表应包含所有层的权重和偏置
    REQUIRED_UNET_KEYS = [
        # 编码器关键层
        'enc1.conv1.weight', 'enc1.bn1.weight', 'enc1.bn1.bias',
        'enc2.conv1.weight', 'enc2.bn1.weight',
        'enc3.conv1.weight', 'enc3.bn1.weight',
        'enc4.conv1.weight', 'enc4.bn1.weight',
        # 瓶颈层
        'bottleneck.conv1.weight', 'bottleneck.bn1.weight',
        # 解码器关键层
        'dec1.conv1.weight', 'dec1.bn1.weight',
        'dec2.conv1.weight', 'dec2.bn1.weight',
        'dec3.conv1.weight', 'dec3.bn1.weight',
        'dec4.conv1.weight', 'dec4.bn1.weight',
        # 输出层
        'final_conv.weight', 'final_conv.bias'
    ]
    
    def __init__(self, max_size_mb: int = 200):
        """
        初始化校验器
        
        Args:
            max_size_mb: 最大文件大小（MB），默认200MB
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_size_mb = max_size_mb
    
    def validate_file(self, file_path: Path, model_type: str = 'unet') -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        完整校验权重文件
        
        Args:
            file_path: 权重文件路径
            model_type: 模型类型（unet/rs_unet3_plus）
        
        Returns:
            (is_valid, error_msg, metadata)
            - is_valid: 是否通过校验
            - error_msg: 错误信息（通过时为None）
            - metadata: 权重元数据（通过时返回）
        """
        try:
            # 步骤1：文件格式校验
            is_valid, error_msg = self._validate_format(file_path)
            if not is_valid:
                return False, error_msg, None
            
            # 步骤2：文件大小校验
            is_valid, error_msg = self._validate_size(file_path)
            if not is_valid:
                return False, error_msg, None
            
            # 步骤3：State_dict完整性校验
            is_valid, error_msg, metadata = self._validate_state_dict(file_path, model_type)
            if not is_valid:
                return False, error_msg, None
            
            logger.info(f"[权重校验] ✓ 权重文件校验通过: {file_path.name}")
            return True, None, metadata
            
        except Exception as e:
            error_msg = f"权重校验异常: {str(e)}"
            logger.error(f"[权重校验] ✗ {error_msg}")
            return False, error_msg, None
    
    def _validate_format(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """
        校验文件格式
        
        Args:
            file_path: 文件路径
        
        Returns:
            (is_valid, error_msg)
        """
        valid_extensions = ['.pth', '.pt']
        file_ext = file_path.suffix.lower()
        
        if file_ext not in valid_extensions:
            return False, f"文件格式不支持：{file_ext}，仅支持 .pth 或 .pt 格式"
        
        if not file_path.exists():
            return False, f"文件不存在：{file_path}"
        
        return True, None
    
    def _validate_size(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """
        校验文件大小
        
        Args:
            file_path: 文件路径
        
        Returns:
            (is_valid, error_msg)
        """
        file_size = file_path.stat().st_size
        
        if file_size > self.max_size_bytes:
            size_mb = file_size / 1024 / 1024
            return False, f"文件过大：{size_mb:.2f}MB，超过限制 {self.max_size_mb}MB"
        
        if file_size == 0:
            return False, "文件为空"
        
        return True, None
    
    def _validate_state_dict(self, file_path: Path, model_type: str) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        校验State_dict完整性
        
        Args:
            file_path: 文件路径
            model_type: 模型类型
        
        Returns:
            (is_valid, error_msg, metadata)
        """
        try:
            # 加载权重文件（仅CPU，避免GPU占用）
            checkpoint = torch.load(file_path, map_location='cpu')
            
            # 提取state_dict
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # 获取所有keys
            weight_keys = list(state_dict.keys())
            
            # 校验必需的keys（根据模型类型）
            if model_type == 'unet':
                missing_keys = [key for key in self.REQUIRED_UNET_KEYS if key not in weight_keys]
                
                if missing_keys:
                    return False, f"权重文件缺少必需的层: {', '.join(missing_keys[:5])}等{len(missing_keys)}个", None
            
            # 提取元数据
            metadata = {
                'total_keys': len(weight_keys),
                'sample_keys': weight_keys[:10],  # 前10个key作为样本
                'total_params': sum(p.numel() for p in state_dict.values()),
                'file_size_mb': file_path.stat().st_size / 1024 / 1024
            }
            
            # 检查是否有epoch/loss等训练信息
            if isinstance(checkpoint, dict):
                if 'epoch' in checkpoint:
                    metadata['epoch'] = checkpoint['epoch']
                if 'loss' in checkpoint or 'best_loss' in checkpoint:
                    metadata['loss'] = checkpoint.get('loss') or checkpoint.get('best_loss')
            
            return True, None, metadata
            
        except Exception as e:
            return False, f"无法加载权重文件: {str(e)}", None


# 全局校验器实例
_validator = None

def get_validator(max_size_mb: int = 200) -> WeightValidator:
    """获取全局校验器实例"""
    global _validator
    if _validator is None:
        _validator = WeightValidator(max_size_mb=max_size_mb)
    return _validator
