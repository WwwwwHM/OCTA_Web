"""
OCTA图像分割平台 - 服务层模块

本模块封装所有业务服务逻辑，包括：
- ModelService: AI模型调用服务（图像分割、预处理、后处理）
- [未来可扩展其他服务]

作者：OCTA Web项目组
日期：2026年1月14日
"""

from .model_service import ModelService

__all__ = ['ModelService']
