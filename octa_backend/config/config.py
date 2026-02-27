"""
OCTA图像分割平台 - 统一配置管理

本模块集中管理项目所有配置常量，避免硬编码分散在各处代码中。
所有配置项都有详细注释说明用途、单位和默认值。

设计原则：
1. 单一数据源（Single Source of Truth）：所有配置统一管理
2. 解耦硬编码：代码中不再出现魔法数字和硬编码路径
3. 易于维护：修改配置只需修改此文件
4. 扩展友好：预留扩展配置字段，便于功能升级

使用示例：
    from config import DB_PATH, UPLOAD_DIR, MAX_FILE_SIZE
    
    # 使用数据库配置
    database = Database(DB_PATH)
    
    # 使用文件存储配置
    upload_path = Path(UPLOAD_DIR) / filename
    
    # 使用文件大小限制
    if file_size > MAX_FILE_SIZE:
        raise ValueError("文件过大")

作者：OCTA Web项目组
日期：2026年1月14日
"""


# ==================== 数据库配置 ====================

# SQLite数据库文件路径
# 用途：存储分割历史记录（上传文件、结果文件、时间戳等）
# 默认位置：项目根目录下的octa.db文件
# 注意：生产环境建议使用绝对路径或环境变量
DB_PATH = "./octa.db"

# 数据库表名
# 用途：存储图像分割历史记录的表名
# 结构：id、filename、upload_time、model_type、original_path、result_path
DB_TABLE_NAME = "images"


# ==================== 文件存储配置 ====================

# 上传文件存储目录
# 用途：保存用户上传的原始OCTA图像
# 格式：PNG/JPG/JPEG格式的医学影像
# 命名规则：使用UUID避免文件名冲突
UPLOAD_DIR = "./uploads"

# 分割结果存储目录
# 用途：保存AI模型分割后的结果图像
# 格式：8位灰度PNG图像（0-255值范围）
# 命名规则：原文件名_seg.png
RESULT_DIR = "./results"

# 文件上传大小限制
# 单位：字节（bytes）
# 当前值：10MB = 10 * 1024 * 1024 bytes
# 用途：防止超大文件占用服务器存储和带宽
# 建议：医学影像通常为256x256或512x512，1-5MB已足够
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# 允许上传的文件格式
# 用途：限制文件类型，防止非图像文件上传
# 格式：小写文件扩展名列表
# 注意：后端会进一步验证文件的MIME类型（Content-Type）
ALLOWED_FORMATS = ["png", "jpg", "jpeg"]

# 文件名前缀
# 用途：为生成的文件名添加项目标识前缀
# 示例：octa_abc123def456.png
# 可选：如不需要可设为空字符串""
FILE_NAME_PREFIX = "octa_"

# ==================== 权重上传/管理配置 ====================
# Fix: 平台优化 - 放弃训练模块，聚焦预测功能
# 权重上传根目录（按weight_id隔离存储）
WEIGHT_UPLOAD_ROOT = "./static/uploads/weight"
# 权重文件允许的最大体积（200MB）
WEIGHT_MAX_SIZE = 200 * 1024 * 1024
# 允许上传的权重格式
WEIGHT_ALLOWED_FORMATS = ["pth", "pt"]
# 官方预置权重（无需上传即可演示）
OFFICIAL_WEIGHT_PATH = "./static/uploads/weight/official/unet_best_dice0.78.pth"
DEFAULT_WEIGHT_ID = "official"

# ==================== 核心模块配置 ====================
# Fix: 平台优化 - 数据处理/模型加载核心参数

# ==================== 模型参数配置 ====================
# 用途：定义U-Net模型的输入输出通道数和图像尺寸

# U-Net模型输入通道数
# 值：1（灰度图像，单通道）
# 说明：OCTA血管图像为灰度医学影像，RGB需转为灰度
U_NET_IN_CHANNELS = 1

# U-Net模型输出通道数
# 值：1（二分类分割，单通道mask）
# 说明：输出为血管（前景）/背景的二值掩码
U_NET_OUT_CHANNELS = 1

# 输入图像统一尺寸（宽度, 高度）
# 值：(256, 256)像素
# 说明：所有输入图像会被resize到此尺寸，输出mask会恢复到原始尺寸
INPUT_SIZE = (256, 256)

# 掩码二值化阈值
# 值：0.5（sigmoid输出的中间值）
# 说明：模型输出概率>0.5视为血管，≤0.5视为背景
MASK_THRESHOLD = 0.5

# ==================== 预处理参数配置 ====================
# 用途：图像归一化参数（与本地U-Net训练脚本一致，禁止修改）

# 归一化均值
# 值：0.5
# 公式：normalized = (pixel/255 - MEAN) / STD
# 说明：将[0,255]像素值归一化到[-1, 1]范围
MEAN = 0.5

# 归一化标准差
# 值：0.5
# 说明：配合MEAN=0.5实现归一化
STD = 0.5

# 【兼容性保留】旧参数名（供现有代码使用）
IMAGE_SIZE = 256  # 输入图像统一尺寸（单个维度）
NORMALIZE_MEAN = MEAN  # 归一化均值（别名）
NORMALIZE_STD = STD  # 归一化标准差（别名）
BINARY_THRESHOLD = MASK_THRESHOLD  # 二值化阈值（别名）
MASK_OUTPUT_FORMAT = "uint8"  # 掩码输出格式（0/255）

# ==================== 权重配置 ====================
# 用途：权重文件存储和格式管理

# 权重文件存储根路径
# 值："./weights"
# 说明：所有用户上传和训练生成的权重文件统一存储在此目录
WEIGHT_SAVE_PATH = "./weights"

# 支持的权重文件格式
# 值：[".pth", ".pt"]
# 说明：仅允许PyTorch标准权重格式，拒绝其他扩展名
SUPPORTED_WEIGHT_FORMATS = [".pth", ".pt"]

# ==================== 模型加载参数 ====================
# 用途：设备管理和加载策略

# 设备优先级
# 值："cuda"（优先使用GPU）
# 说明：系统自动检测，若无CUDA则回退到CPU
DEVICE_PRIORITY = "cuda"

# CPU设备标识
# 值："cpu"
# 说明：无GPU或强制使用CPU时的设备名称
CPU_DEVICE = "cpu"

# 【兼容性保留】旧参数
AUTO_DEVICE = True  # 自动选择设备（GPU优先，无GPU则CPU）
DEFAULT_DEVICE = "auto"  # 默认设备类型
MODEL_EVAL_MODE = True  # 推理时强制eval模式
DISABLE_GRADIENTS = True  # 推理时禁用梯度计算

# 权重校验参数
WEIGHT_STRICT_LOADING = False  # state_dict加载是否严格匹配（False允许部分加载）

# ==================== 模型配置 ====================

# 模型代码目录
# 用途：存放U-Net、RS-Unet3+等模型定义文件
# 包含：unet.py、rs_unet3_plus.py（模型实现）、weights/（模型权重）
MODEL_DIR = "./models"

# U-Net预训练权重文件路径
# 用途：加载预训练的U-Net模型权重
# 格式：PyTorch .pth文件
# 注意：权重文件不存在时，模型使用随机初始化（仅用于测试）


# ==================== 临时文件清理配置 ====================
# Fix: 平台优化 - 定时清理未使用的文件

# 启用自动清理任务
ENABLE_AUTO_CLEANUP = True
# 清理调度：默认每天凌晨2:00执行（使用cron）；如需使用时间间隔，可保留CLEANUP_INTERVAL_SECONDS作为回退
CLEANUP_CRON = {"hour": 2, "minute": 0}
# 清理间隔（秒），仅在未设置CLEANUP_CRON时使用
CLEANUP_INTERVAL_SECONDS = 3600
# 文件过期时间（秒），默认24小时未访问则删除
FILE_EXPIRY_SECONDS = 24 * 3600
# 需要清理的目录列表（包含权重/上传/结果）
CLEANUP_DIRS = [UPLOAD_DIR, RESULT_DIR, WEIGHT_UPLOAD_ROOT]
UNET_WEIGHT_PATH = "./models/weights/unet_octa.pth"

# ==================== 权重存储隔离配置（按模型架构分类）====================
# 用途：防止不同模型的权重混淆，便于管理和部署
# 结构：
#   models/
#   ├── weights_unet/          ← U-Net训练权重
#   │   ├── trained_unet_20260120.pth
#   │   └── trained_unet_best.pth
#   ├── weights_rs_unet3_plus/ ← RS-Unet3+训练权重
#   │   ├── rs_unet3p_epoch10.pth
#   │   └── rs_unet3p_best.pth
#   └── weights/               ← 通用权重（向后兼容）
#       └── unet_octa.pth

import os

# U-Net训练权重存储目录
# 用途：存放U-Net训练生成的.pth文件
# 自动创建：训练开始时自动创建此目录
# 命名规范：unet_YYYYMMDD_HHMMSS.pth（如：unet_20260120_143052.pth）
UNET_WEIGHT_DIR = os.path.join(MODEL_DIR, "weights_unet")

# RS-Unet3+训练权重存储目录
# 用途：存放RS-Unet3+训练生成的.pth文件
# 自动创建：训练开始时自动创建此目录
# 命名规范：rs_unet3_plus_YYYYMMDD_HHMMSS.pth（如：rs_unet3_plus_20260120_143052.pth）
RS_UNET3_PLUS_WEIGHT_DIR = os.path.join(MODEL_DIR, "weights_rs_unet3_plus")

# 权重文件名前缀映射（用于自动生成文件名）
# 格式：{model_type: prefix}
WEIGHT_PREFIX_MAP = {
    "unet": "unet",
    "rs_unet3_plus": "rs_unet3_plus"
}

# 权重目录映射（用于获取模型对应的权重目录）
# 格式：{model_type: directory}
WEIGHT_DIR_MAP = {
    "unet": UNET_WEIGHT_DIR,
    "rs_unet3_plus": RS_UNET3_PLUS_WEIGHT_DIR
}

# 默认使用的模型类型
# 可选值：'unet'（推荐）、'rs_unet3_plus'（OCTA专用）
# 用途：指定默认的图像分割模型
# 推荐：U-Net在通用医学图像分割任务中表现更优，RS-Unet3+专为OCTA优化
DEFAULT_MODEL_TYPE = "unet"

# 图像预处理目标尺寸
# 格式：(宽度, 高度) 元组，单位为像素
# 用途：所有输入图像统一调整到此尺寸再输入模型
# 原因：神经网络需要固定尺寸的输入
# 注意：输出掩码会自动恢复到原始图像尺寸
IMAGE_TARGET_SIZE = (256, 256)

# 模型运行设备
# 可选值：'cpu'（推荐，无GPU环境）、'cuda'（NVIDIA GPU）、'mps'（Apple Silicon）
# 用途：指定模型推理和训练时使用的计算设备
# 重要：当前固定为'cpu'，适配医学影像服务器通常无GPU的情况
# 若需GPU加速，需先安装CUDA版本PyTorch，然后修改为'cuda'或'mps'
MODEL_DEVICE = "cuda"


# ==================== 服务配置 ====================

# 后端服务监听主机
# 可选值：
#   - "127.0.0.1"：仅本地访问（开发环境推荐）
#   - "0.0.0.0"：允许外部访问（生产环境）
#   - "localhost"：等同于127.0.0.1
SERVER_HOST = "127.0.0.1"

# 后端服务监听端口
# 默认值：8000（FastAPI推荐端口）
# 注意：
#   - 修改端口后需同步更新前端API请求地址
#   - 确保端口未被其他程序占用
#   - Linux下使用1024以下端口需要root权限
SERVER_PORT = 8000

# 开发模式热重载
# 可选值：True（开发模式，代码修改后自动重载）、False（生产模式）
# 用途：开发时快速迭代，生产时关闭避免意外重载
# 建议：仅在开发环境设为True
RELOAD_MODE = True


# ==================== 跨域配置（CORS）====================

# 允许跨域请求的前端地址列表
# 用途：解决浏览器的同源策略（Same-Origin Policy）限制
# 格式：完整的URL列表（包括协议、主机、端口）
# 注意：
#   - 开发环境：列出所有前端开发服务器地址
#   - 生产环境：仅列出实际部署的前端域名
#   - 安全警告：不要使用["*"]（允许所有来源），存在安全风险
CORS_ORIGINS = [
    "http://127.0.0.1:5173",  # Vite开发服务器默认地址
    "http://localhost:5173",  # 本地开发地址（备用）
    # 生产环境示例：
    # "https://octa.example.com",
    # "https://www.octa.example.com",
]

# 是否允许跨域请求携带凭证（cookies、authorization等）
# 可选值：True（允许）、False（不允许）
# 用途：用于cookie-based认证或需要发送敏感信息的场景
CORS_ALLOW_CREDENTIALS = True

# 允许的HTTP请求方法
# 可选值：["GET", "POST"]（显式列表）、["*"]（允许所有）
# 当前值：["*"]允许所有方法（GET、POST、PUT、DELETE、OPTIONS等）
CORS_ALLOW_METHODS = ["*"]

# 允许的HTTP请求头
# 可选值：["Content-Type", "Authorization"]（显式列表）、["*"]（允许所有）
# 当前值：["*"]允许所有请求头
CORS_ALLOW_HEADERS = ["*"]


# ==================== 日志配置 ====================
# 用途：统一管理日志输出、格式和存储

# 日志存储路径
# 值："./logs"
# 说明：所有日志文件统一存储在此目录
LOG_SAVE_PATH = "./logs"

# 日志级别
# 值："INFO"
# 可选值：DEBUG（详细调试）/INFO（一般信息）/WARNING（警告）/ERROR（错误）/CRITICAL（严重错误）
# 说明：INFO级别适合生产环境，DEBUG级别适合开发调试
LOG_LEVEL = "INFO"

# 日志格式
# 值："%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# 输出示例：2026-01-28 14:30:45 - octa_backend - INFO - 模型加载成功
# 格式说明：
#   %(asctime)s - 时间戳
#   %(name)s - 日志记录器名称
#   %(levelname)s - 日志级别
#   %(message)s - 日志消息内容
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# 【兼容性保留】旧参数名
LOG_FILE = "./logs/octa_backend.log"  # 日志文件完整路径
LOG_MAX_SIZE_MB = 10  # 日志文件最大大小（MB）
LOG_BACKUP_COUNT = 5  # 日志文件备份数量


# ==================== 扩展配置（预留）====================
# ==================== 错误码配置 ====================
# 用途：统一管理API响应状态码

# 成功状态码
# 值：200
# 说明：请求成功处理并返回结果
SUCCESS_CODE = 200

# 格式错误状态码
# 值：400（Bad Request）
# 说明：上传文件格式不符合要求（非.pth/.pt或非图像格式）
FORMAT_ERROR_CODE = 400

# 权重校验错误状态码
# 值：400（Bad Request）
# 说明：权重文件内容校验失败（state_dict缺失或键不匹配）
WEIGHT_VALID_ERROR_CODE = 400

# 模型加载错误状态码
# 值：500（Internal Server Error）
# 说明：模型加载失败（权重损坏、设备不可用等）
MODEL_LOAD_ERROR_CODE = 500

# 推理错误状态码
# 值：500（Internal Server Error）
# 说明：模型推理过程中发生异常（OOM、计算错误等）
INFERENCE_ERROR_CODE = 500

# Redis缓存配置（预留）
# 用途：缓存分割结果，提高重复请求响应速度
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_PASSWORD = None  # 生产环境使用环境变量

# MySQL数据库配置（预留）
# 用途：业务数据持久化（当前默认未启用）
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_USER = "root"
MYSQL_PASSWORD = ""
MYSQL_DATABASE = "octa_db"        # MySQL数据库名称

# 日志配置（预留）
# 用途：统一管理日志输出格式和级别
LOG_LEVEL = "INFO"                # 日志级别：DEBUG/INFO/WARNING/ERROR/CRITICAL
LOG_FILE = "./logs/octa.log"      # 日志文件路径
LOG_MAX_SIZE = 10 * 1024 * 1024   # 单个日志文件最大10MB
LOG_BACKUP_COUNT = 5              # 保留最近5个日志文件

# GPU配置（预留）
# 用途：当服务器有GPU时，可启用GPU加速推理
USE_GPU = True                 # 是否启用GPU加速
GPU_DEVICE_ID = 0                 # GPU设备ID（多GPU环境）

# 模型推理优化配置（预留）
# 用途：优化模型推理性能
BATCH_SIZE = 1                    # 批处理大小（当前单张图像推理）
NUM_WORKERS = 4                   # 数据加载线程数
MIXED_PRECISION = False           # 是否启用混合精度推理（FP16）

# API限流配置（预留）
# 用途：防止恶意请求耗尽服务器资源
RATE_LIMIT_ENABLED = False        # 是否启用限流
RATE_LIMIT_REQUESTS = 60          # 每分钟最大请求数
RATE_LIMIT_WINDOW = 60            # 限流时间窗口（秒）

# 文件清理配置（预留）
# 用途：自动清理过期的上传文件和结果文件
AUTO_CLEANUP_ENABLED = False      # 是否启用自动清理
CLEANUP_DAYS = 7                  # 保留最近N天的文件
CLEANUP_SCHEDULE = "0 2 * * *"    # 清理任务执行时间（Cron表达式：每天凌晨2点）


# ==================== 配置验证函数 ====================

def validate_config():
    """
    验证配置项的合法性
    
    检查项：
    1. 文件路径是否为字符串
    2. 文件大小限制是否为正整数
    3. 端口号是否在合法范围内
    4. 图像尺寸是否为正整数元组
    
    Raises:
        ValueError: 配置项不合法时抛出异常
    """
    # 验证文件大小限制
    if not isinstance(MAX_FILE_SIZE, int) or MAX_FILE_SIZE <= 0:
        raise ValueError(f"MAX_FILE_SIZE必须为正整数，当前值: {MAX_FILE_SIZE}")
    
    # 验证端口号
    if not isinstance(SERVER_PORT, int) or not (1 <= SERVER_PORT <= 65535):
        raise ValueError(f"SERVER_PORT必须在1-65535范围内，当前值: {SERVER_PORT}")
    
    # 验证图像尺寸
    if not isinstance(IMAGE_TARGET_SIZE, tuple) or len(IMAGE_TARGET_SIZE) != 2:
        raise ValueError(f"IMAGE_TARGET_SIZE必须为(width, height)元组，当前值: {IMAGE_TARGET_SIZE}")
    
    if not all(isinstance(size, int) and size > 0 for size in IMAGE_TARGET_SIZE):
        raise ValueError(f"IMAGE_TARGET_SIZE的宽度和高度必须为正整数，当前值: {IMAGE_TARGET_SIZE}")
    
    # 验证允许的文件格式
    if not isinstance(ALLOWED_FORMATS, list) or len(ALLOWED_FORMATS) == 0:
        raise ValueError(f"ALLOWED_FORMATS必须为非空列表，当前值: {ALLOWED_FORMATS}")
    
    # 验证模型设备
    if MODEL_DEVICE not in ['cpu', 'cuda', 'mps']:
        raise ValueError(f"MODEL_DEVICE必须为'cpu'、'cuda'或'mps'，当前值: {MODEL_DEVICE}")
    
    print("[INFO] 配置验证通过")


def print_config():
    """
    打印当前生效的配置项（便于调试）
    
    用途：启动服务时显示配置，方便排查问题
    """
    print("=" * 70)
    print("OCTA后端配置信息".center(70))
    print("=" * 70)
    
    print("\n【数据库配置】")
    print(f"  数据库路径: {DB_PATH}")
    print(f"  数据表名: {DB_TABLE_NAME}")
    
    print("\n【文件存储配置】")
    print(f"  上传目录: {UPLOAD_DIR}")
    print(f"  结果目录: {RESULT_DIR}")
    print(f"  文件大小限制: {MAX_FILE_SIZE / (1024*1024):.1f} MB")
    print(f"  允许格式: {', '.join(ALLOWED_FORMATS)}")
    print(f"  文件名前缀: '{FILE_NAME_PREFIX}'")
    
    print("\n【模型配置】")
    print(f"  模型目录: {MODEL_DIR}")
    print(f"  U-Net权重: {UNET_WEIGHT_PATH}")
    print(f"  默认模型: {DEFAULT_MODEL_TYPE}")
    print(f"  目标尺寸: {IMAGE_TARGET_SIZE[0]}x{IMAGE_TARGET_SIZE[1]}px")
    print(f"  运行设备: {MODEL_DEVICE}")
    
    print("\n【服务配置】")
    print(f"  监听地址: {SERVER_HOST}:{SERVER_PORT}")
    print(f"  热重载: {'✓ 开启（开发模式）' if RELOAD_MODE else '✗ 关闭（生产模式）'}")
    
    print("\n【跨域配置】")
    print(f"  允许来源: {len(CORS_ORIGINS)}个前端地址")
    for origin in CORS_ORIGINS:
        print(f"    - {origin}")
    print(f"  允许凭证: {'✓ 允许' if CORS_ALLOW_CREDENTIALS else '✗ 不允许'}")
    
    print("\n" + "=" * 70)


# ==================== 模块测试代码 ====================

if __name__ == '__main__':
    """
    测试配置模块
    
    运行方式：
        python -m config.config
    """
    print("配置模块测试")
    print("-" * 70)
    
    # 测试1：配置验证
    print("\n[测试1] 配置验证...")
    try:
        validate_config()
        print("✓ 配置验证通过")
    except ValueError as e:
        print(f"✗ 配置验证失败: {e}")
    
    # 测试2：打印配置
    print("\n[测试2] 打印配置...")
    print_config()
    
    # 测试3：配置访问
    print("\n[测试3] 配置访问...")
    print(f"✓ 数据库路径: {DB_PATH}")
    print(f"✓ 上传目录: {UPLOAD_DIR}")
    print(f"✓ 文件大小限制: {MAX_FILE_SIZE} bytes")
    print(f"✓ U-Net权重: {UNET_WEIGHT_PATH}")
    print(f"✓ 服务地址: {SERVER_HOST}:{SERVER_PORT}")
    
    print("\n" + "=" * 70)
    print("✅ 配置模块测试完成！")
    print("=" * 70)
