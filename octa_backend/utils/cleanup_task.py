"""
临时文件定时清理任务

Fix: 平台优化 - 自动清理未使用的文件（24小时）
功能：
1. 定时扫描上传目录和结果目录
2. 删除24小时未访问的文件
3. 使用APScheduler后台调度
4. 支持启用/禁用开关

作者：OCTA Web项目组
日期：2026-01-27
"""

import os
import time
import logging
from pathlib import Path
from typing import List
from apscheduler.schedulers.background import BackgroundScheduler

from config.config import (
    ENABLE_AUTO_CLEANUP,
    CLEANUP_INTERVAL_SECONDS,
    FILE_EXPIRY_SECONDS,
    CLEANUP_DIRS,
    CLEANUP_CRON,
    OFFICIAL_WEIGHT_PATH,
)

logger = logging.getLogger(__name__)


class FileCleanupTask:
    """文件清理任务"""
    
    def __init__(self):
        """初始化清理任务"""
        self.scheduler = BackgroundScheduler()
        self.enabled = ENABLE_AUTO_CLEANUP
        self._official_weight = Path(OFFICIAL_WEIGHT_PATH).resolve()
        self._official_dir = self._official_weight.parent
        logger.info(f"[清理任务] 初始化，状态: {'启用' if self.enabled else '禁用'}")
    
    def cleanup_expired_files(self):
        """执行文件清理"""
        if not self.enabled:
            return
        
        logger.info("[清理任务] 开始执行定时清理...")
        current_time = time.time()
        total_deleted = 0
        total_size_freed = 0
        
        for dir_path in CLEANUP_DIRS:
            directory = Path(dir_path)
            if not directory.exists():
                continue
            
            logger.debug(f"[清理任务] 扫描目录: {directory}")
            
            for file_path in directory.rglob("*"):
                if not file_path.is_file():
                    continue

                # 官方权重固定保留，避免被误删
                resolved_path = file_path.resolve()
                if resolved_path == self._official_weight or self._official_dir in resolved_path.parents:
                    continue
                
                try:
                    # 获取最后访问时间
                    last_access_time = file_path.stat().st_atime
                    file_age = current_time - last_access_time
                    
                    # 检查是否过期
                    if file_age > FILE_EXPIRY_SECONDS:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        total_deleted += 1
                        total_size_freed += file_size
                        logger.debug(f"[清理任务] ✓ 删除过期文件: {file_path.name} (已存在{file_age/3600:.1f}小时)")
                
                except Exception as e:
                    logger.warning(f"[清理任务] ✗ 删除文件失败: {file_path.name}, 错误: {str(e)}")
        
        if total_deleted > 0:
            logger.info(f"[清理任务] ✓ 完成，删除 {total_deleted} 个文件，释放 {total_size_freed/1024/1024:.2f}MB 空间")
        else:
            logger.info("[清理任务] ✓ 完成，无过期文件")
    
    def start(self):
        """启动清理任务"""
        if not self.enabled:
            logger.info("[清理任务] 已禁用，不启动调度器")
            return
        
        # 添加定时任务（优先使用cron配置，回退到interval）
        if CLEANUP_CRON:
            self.scheduler.add_job(
                self.cleanup_expired_files,
                'cron',
                id='file_cleanup',
                name='临时文件清理任务',
                **CLEANUP_CRON,
            )
            logger.info(f"[清理任务] ✓ 已启动，cron: {CLEANUP_CRON}")
        else:
            self.scheduler.add_job(
                self.cleanup_expired_files,
                'interval',
                seconds=CLEANUP_INTERVAL_SECONDS,
                id='file_cleanup',
                name='临时文件清理任务'
            )
            logger.info(f"[清理任务] ✓ 已启动，间隔: {CLEANUP_INTERVAL_SECONDS}秒 ({CLEANUP_INTERVAL_SECONDS/3600:.1f}小时)")
        
        # 启动调度器
        self.scheduler.start()
        logger.info(f"[清理任务] 文件过期时间: {FILE_EXPIRY_SECONDS}秒 ({FILE_EXPIRY_SECONDS/3600:.1f}小时)")
    
    def stop(self):
        """停止清理任务"""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("[清理任务] ✓ 已停止")
    
    def run_now(self):
        """立即执行一次清理"""
        logger.info("[清理任务] 手动触发清理...")
        self.cleanup_expired_files()


# 全局清理任务实例
_cleanup_task = None

def get_cleanup_task() -> FileCleanupTask:
    """获取全局清理任务实例"""
    global _cleanup_task
    if _cleanup_task is None:
        _cleanup_task = FileCleanupTask()
    return _cleanup_task
