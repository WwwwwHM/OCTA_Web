"""OCTA图像分割平台 - 工具模块

本模块包含所有工具类，负责文件处理、配置管理等横切关注点。

模块导出：
  - FileUtils: 文件处理工具类

作者：OCTA Web项目组
日期：2026年1月14日
"""

from .file_utils import FileUtils

__all__ = ['FileUtils']
