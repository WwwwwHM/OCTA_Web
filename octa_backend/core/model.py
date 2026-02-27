"""
模型定义模块

Fix: 平台优化 - 聚焦预测功能，复用验证过的U-Net架构
功能：
1. 提供统一的模型创建接口
2. 复用models/unet.py中的UNetUnderfittingFix（Dice≥0.75）
3. 支持灰度输入（1通道）+ 灰度输出（1通道）

作者：OCTA Web项目组
日期：2026-01-27
"""

import logging
from models.unet import UNetUnderfittingFix

logger = logging.getLogger(__name__)


def create_model(in_channels: int = 1, out_channels: int = 1) -> UNetUnderfittingFix:
    """
    创建U-Net模型实例（未加载权重）
    
    Args:
        in_channels: 输入通道数（OCTA灰度图=1）
        out_channels: 输出通道数（分割掩码=1）
    
    Returns:
        UNetUnderfittingFix模型实例
    
    说明：
        - 使用UNetUnderfittingFix架构（已验证，Dice≥0.75）
        - 包含通道注意力、多尺度融合、Dropout正则化
        - 模型参数约40M+，适合医学影像分割
    """
    try:
        model = UNetUnderfittingFix(in_channels=in_channels, out_channels=out_channels)
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"[模型创建] ✓ UNetUnderfittingFix实例化成功")
        logger.info(f"  输入通道: {in_channels}, 输出通道: {out_channels}")
        logger.info(f"  参数总数: {param_count:,}")
        return model
    except Exception as e:
        logger.error(f"[模型创建] ✗ 失败: {str(e)}")
        raise


def get_model_info() -> dict:
    """
    获取模型信息
    
    Returns:
        模型配置信息字典
    """
    return {
        'architecture': 'UNetUnderfittingFix',
        'input_channels': 1,
        'output_channels': 1,
        'image_size': 256,
        'activation': 'sigmoid',
        'threshold': 0.5,
        'description': '改进版U-Net，包含通道注意力+多尺度融合+Dropout，适用于OCTA血管分割'
    }
