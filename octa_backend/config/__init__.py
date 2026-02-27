"""
配置管理模块

导出所有配置常量，供其他模块使用。

使用示例：
    from config import DB_PATH, UPLOAD_DIR, MODEL_WEIGHT_PATH
    
    # 使用配置
    database = Database(DB_PATH)
    upload_path = Path(UPLOAD_DIR) / filename
"""

from .config import (
    # 数据库配置
    DB_PATH,
    DB_TABLE_NAME,
    
    # 文件存储配置
    UPLOAD_DIR,
    RESULT_DIR,
    MAX_FILE_SIZE,
    ALLOWED_FORMATS,
    FILE_NAME_PREFIX,
    
    # 模型配置
    MODEL_DIR,
    UNET_WEIGHT_PATH,
    DEFAULT_MODEL_TYPE,
    IMAGE_TARGET_SIZE,
    MODEL_DEVICE,
    
    # 服务配置
    SERVER_HOST,
    SERVER_PORT,
    RELOAD_MODE,
    
    # 跨域配置
    CORS_ORIGINS,
    CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_METHODS,
    CORS_ALLOW_HEADERS,
    
    # 扩展配置
    MYSQL_HOST,
    MYSQL_PORT,
    MYSQL_USER,
    MYSQL_PASSWORD,
    MYSQL_DATABASE,
    
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
    REDIS_PASSWORD,
    
    LOG_LEVEL,
    LOG_FILE,
    LOG_MAX_SIZE,
    LOG_BACKUP_COUNT,
    
    USE_GPU,
    GPU_DEVICE_ID,
    
    BATCH_SIZE,
    NUM_WORKERS,
    MIXED_PRECISION,
    
    RATE_LIMIT_ENABLED,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW,
    
    AUTO_CLEANUP_ENABLED,
    CLEANUP_DAYS,
    CLEANUP_SCHEDULE,
    
    # 工具函数
    validate_config,
    print_config,
)

__all__ = [
    # 数据库配置
    'DB_PATH',
    'DB_TABLE_NAME',
    
    # 文件存储配置
    'UPLOAD_DIR',
    'RESULT_DIR',
    'MAX_FILE_SIZE',
    'ALLOWED_FORMATS',
    'FILE_NAME_PREFIX',
    
    # 模型配置
    'MODEL_DIR',
    'UNET_WEIGHT_PATH',
    'DEFAULT_MODEL_TYPE',
    'IMAGE_TARGET_SIZE',
    'MODEL_DEVICE',
    
    # 服务配置
    'SERVER_HOST',
    'SERVER_PORT',
    'RELOAD_MODE',
    
    # 跨域配置
    'CORS_ORIGINS',
    'CORS_ALLOW_CREDENTIALS',
    'CORS_ALLOW_METHODS',
    'CORS_ALLOW_HEADERS',
    
    # 扩展配置
    'MYSQL_HOST',
    'MYSQL_PORT',
    'MYSQL_USER',
    'MYSQL_PASSWORD',
    'MYSQL_DATABASE',
    
    'REDIS_HOST',
    'REDIS_PORT',
    'REDIS_DB',
    'REDIS_PASSWORD',
    
    'LOG_LEVEL',
    'LOG_FILE',
    'LOG_MAX_SIZE',
    'LOG_BACKUP_COUNT',
    
    'USE_GPU',
    'GPU_DEVICE_ID',
    
    'BATCH_SIZE',
    'NUM_WORKERS',
    'MIXED_PRECISION',
    
    'RATE_LIMIT_ENABLED',
    'RATE_LIMIT_REQUESTS',
    'RATE_LIMIT_WINDOW',
    
    'AUTO_CLEANUP_ENABLED',
    'CLEANUP_DAYS',
    'CLEANUP_SCHEDULE',
    
    # 工具函数
    'validate_config',
    'print_config',
]
