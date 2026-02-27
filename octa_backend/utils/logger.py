"""
日志配置模块

Fix: 平台优化 - 统一日志管理
功能：
1. 配置全局日志记录器
2. 支持文件和控制台双输出
3. 日志文件自动轮转（按大小）
4. 核心操作日志记录（权重上传/删除、推理请求）

作者：OCTA Web项目组
日期：2026-01-27
"""

import logging
import logging.handlers
from pathlib import Path
from config.config import (
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_FILE,
    LOG_MAX_SIZE_MB,
    LOG_BACKUP_COUNT
)


def setup_logging():
    """
    设置全局日志配置
    
    配置内容：
    1. 日志级别：从配置文件读取（默认INFO）
    2. 日志格式：时间戳 - 模块名 - 级别 - 消息
    3. 输出目标：文件（轮转）+ 控制台
    4. 文件轮转：按大小轮转，保留多个备份
    """
    # 创建日志目录
    log_path = Path(LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOG_LEVEL.upper()))
    
    # 创建格式化器
    formatter = logging.Formatter(LOG_FORMAT)
    
    # 文件处理器（自动轮转）
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_SIZE_MB * 1024 * 1024,
        backupCount=LOG_BACKUP_COUNT,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # 记录启动信息
    logging.info("=" * 50)
    logging.info("OCTA后端日志系统已启动")
    logging.info(f"日志级别: {LOG_LEVEL}")
    logging.info(f"日志文件: {LOG_FILE}")
    logging.info(f"文件轮转: {LOG_MAX_SIZE_MB}MB × {LOG_BACKUP_COUNT}个备份")
    logging.info("=" * 50)


# 自动初始化（导入时执行）
setup_logging()
