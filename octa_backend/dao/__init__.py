"""OCTA图像分割平台 - 数据访问对象（DAO）层

本模块包含所有数据层的操作，专门负责与SQLite数据库的交互。
遵循DAO设计模式，完全隔离数据库逻辑与业务逻辑。

模块导出：
  - ImageDAO: 图像数据访问对象，提供CRUD操作

作者：OCTA Web项目组
日期：2026年1月14日
"""

from .image_dao import ImageDAO

__all__ = ['ImageDAO']
