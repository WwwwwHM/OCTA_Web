"""
OCTA图像分割平台 - 控制层模块

控制层（Controller Layer）是分层架构的中间层，负责：
1. 处理HTTP请求和响应
2. 数据验证和格式转换
3. 业务逻辑编排
4. 异常处理和错误返回

导出内容：
- ImageController：处理OCTA图像分割和历史记录的控制器类
"""

from .image_controller import ImageController

__all__ = ['ImageController']
