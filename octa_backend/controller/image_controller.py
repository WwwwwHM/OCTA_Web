"""
OCTA图像分割控制层 - ImageController

分层架构设计：
├── FastAPI路由层（main.py）
│   └── 调用 ↓
├── 控制层（ImageController）← 本文件
│   ├── 请求处理、数据验证、业务编排
│   └── 调用 ↓
├── 模型层（models/unet.py）
│   ├── 图像预处理、模型推理、结果后处理
│   └── 调用 ↓
└── 数据层
    └── 数据库操作、文件I/O

本类职责：
1. 处理HTTP请求的参数验证和格式转换
2. 编排业务逻辑流程（文件上传→模型推理→数据库保存）
3. 处理异常情况并返回标准HTTP响应
4. 提供文件查询和历史记录查询接口

作者：OCTA Web项目组
日期：2024
"""

import os
import sqlite3
import uuid
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

from fastapi import File, UploadFile, Form, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

from models.unet import segment_octa_image
from service.model_service import ModelService
# from service.weight_service import WeightService  # 已废弃，使用router/weight_router.py
from utils.file_utils import FileUtils
from dao.file_dao import FileDAO

# 导入配置（所有常量来自config.py，确保配置集中管理）
from config import (
    DB_PATH,           # 数据库文件路径
    UPLOAD_DIR,        # 上传目录
    RESULT_DIR,        # 结果目录
    ALLOWED_FORMATS,   # 允许的文件格式列表
    MAX_FILE_SIZE,     # 最大文件大小（字节）
    DEFAULT_MODEL_TYPE # 默认模型类型
)


# ==================== 常量配置 ====================

# 文件存储目录（从config加载）
UPLOAD_DIR = Path(UPLOAD_DIR)
RESULTS_DIR = Path(RESULT_DIR)

# 数据库配置（从config加载）
DB_PATH = Path(DB_PATH)

# 支持的文件格式（从config加载并转换为扩展名和MIME类型）
ALLOWED_EXTENSIONS = [f'.{fmt}' for fmt in ALLOWED_FORMATS]
ALLOWED_MIME_TYPES = ['image/png', 'image/x-png', 'image/jpeg', 'image/x-jpeg', 'image/jpg']


# ==================== OCTA图像分割控制器 ====================

class ImageController:
    """
    OCTA图像分割控制器
    
    职责：
    - 处理图像上传、验证、分割、保存的完整流程
    - 管理分割历史记录（数据库CRUD操作）
    - 处理所有异常情况并返回标准HTTP响应
    
    设计模式：
    - 所有方法为类方法（@classmethod）或静态方法（@staticmethod）
    - 实现单一职责原则（每个方法对应一个接口）
    - 异常处理遵循FastAPI规范
    
    接口映射关系：
    ├── GET /  → test_service()
    ├── POST /segment-octa/  → segment_octa()
    ├── GET /images/{filename}  → get_uploaded_image()
    ├── GET /results/{filename}  → get_result_image()
    ├── GET /history/  → get_all_history()
    ├── GET /history/{id}  → get_history_by_id()
    └── DELETE /history/{id}  → delete_history_by_id()
    """
    
    # ==================== 初始化方法 ====================
    
    @staticmethod
    def init_database() -> bool:
        """
        初始化SQLite数据库和表结构
        
        功能：
        - 创建octa.db数据库文件（如果不存在）
        - 创建images表用于记录分割历史
        - 表结构包含：id、filename、upload_time、model_type、original_path、result_path
        
        返回：
        - True：初始化成功
        - False：初始化失败
        
        此方法应在应用启动时调用，通常在main.py中进行
        """
        try:
            # ==================== 创建目录 ====================
            # 确保上传和结果目录存在
            UPLOAD_DIR.mkdir(exist_ok=True)
            RESULTS_DIR.mkdir(exist_ok=True)
            
            # ==================== 创建数据库表 ====================
            conn = sqlite3.connect(str(DB_PATH), timeout=10, check_same_thread=False)
            cursor = conn.cursor()
            
            # 定义images表结构
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE NOT NULL,
                upload_time TEXT NOT NULL,
                model_type TEXT NOT NULL,
                original_path TEXT NOT NULL,
                result_path TEXT NOT NULL
            )
            """
            
            cursor.execute(create_table_sql)
            conn.commit()
            
            print(f"[INFO] 数据库初始化成功: {DB_PATH}")
            return True
            
        except Exception as e:
            print(f"[ERROR] 数据库初始化失败: {e}")
            return False
        finally:
            try:
                if conn:
                    conn.close()
            except:
                pass
    
    # ==================== API接口方法 ====================
    
    @staticmethod
    def test_service() -> Dict:
        """
        【后端健康检查】测试服务是否正常运行
        
        功能：返回后端服务状态，用于前端验证后端连接
        
        返回：
        - Dict：{"message": "OCTA后端服务运行正常"}
        
        对应API接口：
        - GET /
        
        使用场景：
        - 前端启动时验证后端连接
        - 用于health check和监控
        """
        return {"message": "OCTA后端服务运行正常"}
    
    @classmethod
    async def segment_octa(
        cls,
        file: UploadFile = File(..., description="上传的PNG/JPG/JPEG格式图像文件"),
        model_type: str = Form(
            DEFAULT_MODEL_TYPE,
            description="模型类型：'unet'、'fcn' 或 'rs_unet3_plus'"
        ),
        weight_path: str = Form(None, description="模型权重路径（兼容参数，可选）"),
        weight_id: str = Form(None, description="模型权重ID（推荐，可选）")
    ) -> JSONResponse:
        """
        【核心接口】OCTA图像分割端点
        
        功能：接收OCTA图像，调用U-Net/FCN模型进行血管分割
        
        处理流程：
        1. 文件格式验证（PNG/JPG/JPEG + MIME类型）
        2. 生成唯一文件名并保存上传文件
        3. 验证保存的文件是否为有效图像
        4. 调用模型进行图像分割
        5. 验证分割结果
        6. 将记录插入数据库
        7. 返回成功响应
        
        参数：
        - file：上传的图像文件（支持PNG/JPG/JPEG格式）
        - model_type：模型类型，可选值：DEFAULT_MODEL_TYPE（默认，来自config.py）、'fcn' 或 'rs_unet3_plus'
        
        返回：
        - JSONResponse：包含以下字段
          - success：bool，是否成功
          - message：str，处理结果消息
          - original_filename：str，原始文件名
          - saved_filename：str，保存后的唯一文件名
          - result_filename：str，分割结果文件名
          - image_url：str，原图访问URL
          - result_url：str，分割结果访问URL
          - model_type：str，使用的模型类型
          - record_id：int，数据库记录ID
        
        异常处理：
        - 400：文件格式不正确或模型类型无效
        - 500：模型分割失败或数据库错误
        
        对应API接口：
        - POST /segment-octa/
        
        前端对接：
        - Content-Type: multipart/form-data
        - 字段：file（文件）、model_type（表单字段）
        """
        try:
            # ========== 1. 文件格式验证 ==========
            # 检查文件格式和MIME类型
            if not cls._validate_image_file(file):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="仅支持PNG/JPG/JPEG格式的OCTA图像"
                )
            
            # ========== 2. 模型类型验证 ==========
            model_type = model_type.lower()
            if model_type in ['rs-unet3+', 'rs-unet3_plus']:
                model_type = 'rs_unet3_plus'

            if model_type not in ['unet', 'fcn', 'rs_unet3_plus']:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="不支持的模型类型: {0}，仅支持 'unet'、'fcn' 或 'rs_unet3_plus'".format(model_type)
                )
            
            # ========== 3. 生成唯一文件名并保存 ==========
            unique_filename = cls._generate_unique_filename(file.filename)
            upload_path = UPLOAD_DIR / unique_filename
            
            # 读取并保存文件
            file_content = await file.read()
            with open(upload_path, "wb") as f:
                f.write(file_content)
            
            print(f"[INFO] 文件已保存: {upload_path}")

            # ========== 3.1 写入文件管理表 ==========
            try:
                file_size_mb = round(len(file_content) / 1024 / 1024, 4)
                FileDAO.add_file_record(
                    file_name=unique_filename,
                    file_path=str(upload_path),
                    file_type='image',
                    related_model=None,
                    file_size=file_size_mb
                )
            except Exception as e:
                # 不中断主流程，记录日志
                print(f"[WARNING] 文件记录写入失败: {e}")
            
            # ========== 4. 验证保存的文件是否为有效图像 ==========
            try:
                with Image.open(upload_path) as img:
                    img.verify()
                with Image.open(upload_path) as img:
                    img.load()
            except Exception as e:
                if upload_path.exists():
                    upload_path.unlink()
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="仅支持PNG/JPG/JPEG格式的OCTA图像"
                )
            
            # ========== 5. 调用模型进行图像分割 ==========
            print(f"[INFO] 开始图像分割，模型类型: {model_type}")
            
            # 生成结果文件名（与 unet.py 的 segment_octa_image() 保持一致）
            result_filename = f"{Path(unique_filename).stem}_seg.png"
            result_path = RESULTS_DIR / result_filename
            
            # 调用分割函数
            try:
                # 如果提供了权重路径，使用指定权重；否则使用默认权重
                # 权重优先级：weight_id > weight_path > 官方预置
                if weight_id:
                    model_path_to_use = WeightService.resolve_weight_path(weight_id, model_type)
                    print(f"[INFO] 使用 weight_id={weight_id} 映射的权重: {model_path_to_use}")
                elif weight_path:
                    model_path_to_use = weight_path
                    print(f"[INFO] 使用指定权重路径: {weight_path}")
                else:
                    model_path_to_use = WeightService.resolve_weight_path(None, model_type)
                    print(f"[INFO] 使用官方预置权重: {model_path_to_use}")

                if model_type == 'rs_unet3_plus':
                    actual_result_path = ModelService.segment_image(
                        image_path=str(upload_path),
                        model_type=model_type,
                        output_path=str(result_path),
                        weight_path=model_path_to_use
                    )
                else:
                    actual_result_path = segment_octa_image(
                        image_path=str(upload_path),
                        model_type=model_type,
                        model_path=model_path_to_use,
                        output_path=str(result_path),
                        device='auto'
                    )
            except Exception as seg_error:
                import traceback
                error_msg = f"模型分割失败: {str(seg_error)}"
                print(f"[ERROR] {error_msg}")
                print(f"[ERROR] 完整堆栈:")
                traceback.print_exc()
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=error_msg
                )
            
            # ========== 6. 检查分割是否成功 ==========
            if Path(actual_result_path) == upload_path:
                error_msg = "模型分割失败：模型可能未训练或加载失败，请检查模型文件是否正确"
                print(f"[ERROR] {error_msg}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=error_msg
                )
            
            # ========== 7. 验证结果文件是否存在 ==========
            result_path_obj = Path(actual_result_path)
            if not result_path_obj.exists():
                error_msg = f"分割结果文件未生成，预期路径: {str(result_path_obj)}"
                print(f"[ERROR] {error_msg}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=error_msg
                )
            
            # ========== 8. 插入数据库记录 ==========
            print(f"[INFO] 正在插入分割记录到数据库...")
            
            record_id = cls._insert_record(
                filename=unique_filename,
                model_type=model_type,
                original_path=str(upload_path),
                result_path=str(result_path_obj)
            )
            
            if record_id is None:
                print(f"[WARNING] 分割成功，但数据库记录失败")
            else:
                print(f"[SUCCESS] 分割记录已保存，ID: {record_id}")
            
            # ========== 9. 返回成功响应 ==========
            result_filename_only = result_path_obj.name
            
            return JSONResponse(
                content={
                    "success": True,
                    "message": "图像分割完成",
                    "original_filename": file.filename,
                    "saved_filename": unique_filename,
                    "result_filename": result_filename_only,
                    "image_url": f"/images/{unique_filename}",
                    "result_url": f"/results/{result_filename_only}",
                    "model_type": model_type,
                    "record_id": record_id
                },
                status_code=status.HTTP_200_OK
            )
            
        except HTTPException:
            raise
        except Exception as e:
            error_msg = f"模型分割失败: {str(e)}"
            print(f"[ERROR] 图像分割失败: {e}")
            import traceback
            traceback.print_exc()
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_msg
            )
    
    @staticmethod
    def get_uploaded_image(filename: str) -> FileResponse:
        """
        获取上传的原始图像
        
        功能：根据文件名返回uploads目录中的原始图像文件
        
        参数：
        - filename：图像文件名（由segment_octa接口返回的saved_filename）
        
        返回：
        - FileResponse：图像文件响应，自动设置Content-Type
        
        支持格式：PNG、JPG/JPEG（自动识别并设置MIME类型）
        
        异常处理：
        - 404：文件不存在
        - 400：文件格式不支持
        
        对应API接口：
        - GET /images/{filename}
        
        使用场景：
        - 前端显示上传的原始图像
        - 用于图像对比展示
        """
        try:
            file_path = UPLOAD_DIR / filename
            
            # 检查文件是否存在
            if not file_path.exists():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"图像文件不存在: {filename}"
                )
            
            # 验证文件格式
            file_ext = file_path.suffix.lower()
            if file_ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="仅支持PNG/JPG/JPEG格式的OCTA图像"
                )
            
            # 根据扩展名设置Content-Type
            content_type_map = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg'
            }
            content_type = content_type_map.get(file_ext, 'image/jpeg')
            
            return FileResponse(
                path=str(file_path),
                media_type=content_type,
                filename=filename
            )
            
        except HTTPException:
            raise
        except Exception as e:
            print(f"[ERROR] 获取图像时发生错误: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"获取图像时发生错误: {str(e)}"
            )
    
    @staticmethod
    def get_result_image(filename: str) -> FileResponse:
        """
        获取分割结果图像
        
        功能：根据文件名返回results目录中的分割结果图像文件
        
        参数：
        - filename：结果文件名（由segment_octa接口返回的result_filename）
        
        返回：
        - FileResponse：PNG格式的分割结果图像响应
        
        文件格式：PNG（8位灰度图，0-255值范围）
        
        异常处理：
        - 404：文件不存在
        - 400：文件格式错误（必须为PNG）
        
        对应API接口：
        - GET /results/{filename}
        
        使用场景：
        - 前端显示分割结果
        - 用于结果对比展示
        - 用户下载分割结果
        """
        try:
            file_path = RESULTS_DIR / filename
            
            # 检查文件是否存在
            if not file_path.exists():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"分割结果文件不存在: {filename}"
                )
            
            # 验证文件格式（结果必须为PNG）
            if file_path.suffix.lower() != '.png':
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="仅支持PNG格式的OCTA图像"
                )
            
            return FileResponse(
                path=str(file_path),
                media_type="image/png",
                filename=filename
            )
            
        except HTTPException:
            raise
        except Exception as e:
            print(f"[ERROR] 获取分割结果时发生错误: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"获取分割结果时发生错误: {str(e)}"
            )
    
    @classmethod
    def get_all_history(cls) -> JSONResponse:
        """
        【历史查询】获取所有分割历史记录
        
        功能：从SQLite数据库查询所有OCTA分割历史，按上传时间倒序排列
        
        返回：
        - JSONResponse：历史记录列表，每条记录包含：
          - id：int，记录ID
          - filename：str，保存的文件名（UUID格式）
          - upload_time：str，上传时间（YYYY-MM-DD HH:MM:SS）
          - model_type：str，使用的模型类型（unet/fcn）
          - original_path：str，原始图像保存路径
          - result_path：str，分割结果保存路径
        
        异常处理：
        - 返回空列表[]，如果查询失败则打印日志
        
        对应API接口：
        - GET /history/
        
        使用场景：
        - 前端显示分割历史列表
        - 统计分析和数据展示
        """
        try:
            records = cls._get_all_records()
            
            print(f"[INFO] 成功返回 {len(records)} 条分割历史记录")
            
            return JSONResponse(
                content=records,
                status_code=status.HTTP_200_OK
            )
            
        except Exception as e:
            print(f"[ERROR] 获取分割历史记录时发生错误: {e}")
            import traceback
            traceback.print_exc()
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"获取分割历史记录时发生错误: {str(e)}"
            )
    
    @classmethod
    def get_history_by_id(cls, record_id: int) -> JSONResponse:
        """
        【历史详情】获取单条分割历史记录的详情
        
        功能：根据记录ID查询并返回特定的分割历史详情
        
        参数：
        - record_id：int，要查询的记录ID（数据库主键）
        
        返回：
        - JSONResponse：单条记录的详情对象，包含：
          - id：int，记录ID
          - filename：str，保存的文件名
          - upload_time：str，上传时间
          - model_type：str，模型类型
          - original_path：str，原始图像路径
          - result_path：str，分割结果路径
        
        异常处理：
        - 400：记录ID无效（不是正整数）
        - 404：记录不存在
        - 500：数据库操作异常
        
        对应API接口：
        - GET /history/{record_id}
        
        使用场景：
        - 前端查看特定的历史记录详情
        - 获取对应的原图和结果路径
        """
        try:
            # 参数验证
            if record_id <= 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="记录ID必须为正整数"
                )
            
            # 查询记录
            record = cls._get_record_by_id(record_id)
            
            # 检查记录是否存在
            if record is None:
                print(f"[WARNING] 记录ID {record_id} 不存在")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"未找到ID为 {record_id} 的分割记录"
                )
            
            print(f"[SUCCESS] 成功返回ID为 {record_id} 的分割历史记录")
            
            return JSONResponse(
                content=record,
                status_code=status.HTTP_200_OK
            )
            
        except HTTPException:
            raise
        except Exception as e:
            print(f"[ERROR] 获取分割历史记录详情时发生错误: {e}")
            import traceback
            traceback.print_exc()
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"获取分割历史记录详情时发生错误: {str(e)}"
            )
    
    @classmethod
    def delete_history_by_id(cls, record_id: int) -> JSONResponse:
        """
        【历史删除】删除单条分割历史记录
        
        功能：根据记录ID删除特定的OCTA图像分割记录
        
        参数：
        - record_id：int，要删除的记录ID
        
        返回：
        - JSONResponse：删除结果响应，包含：
          - success：bool，删除是否成功
          - message：str，删除结果消息
          - deleted_id：int，被删除的记录ID
        
        注意：
        - 删除后该记录不可恢复
        - 仅删除数据库记录，不删除对应的图像文件
        
        异常处理：
        - 400：记录ID无效
        - 404：记录不存在
        - 500：数据库操作异常
        
        对应API接口：
        - DELETE /history/{record_id}
        
        使用场景：
        - 前端清理历史记录
        - 管理员删除不需要的分割结果
        """
        try:
            # 参数验证
            if record_id <= 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="记录ID必须为正整数"
                )
            
            # 检查记录是否存在
            record = cls._get_record_by_id(record_id)
            
            if record is None:
                print(f"[WARNING] 尝试删除不存在的记录，ID: {record_id}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"未找到ID为 {record_id} 的分割记录"
                )
            
            # 执行删除操作
            conn = None
            try:
                conn = sqlite3.connect(str(DB_PATH), timeout=10, check_same_thread=False)
                cursor = conn.cursor()
                
                delete_sql = "DELETE FROM images WHERE id = ?"
                cursor.execute(delete_sql, (record_id,))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count == 0:
                    print(f"[WARNING] 删除记录失败，没有受影响的行，ID: {record_id}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="删除失败：记录可能已被删除"
                    )
                
                print(f"[SUCCESS] 成功删除分割记录，ID: {record_id}")
                print(f"[INFO] 删除的文件名: {record.get('filename')}")
                
                return JSONResponse(
                    content={
                        "success": True,
                        "message": "分割记录已删除",
                        "deleted_id": record_id
                    },
                    status_code=status.HTTP_200_OK
                )
                
            except sqlite3.OperationalError as op_error:
                print(f"[ERROR] 删除记录时数据库操作错误: {op_error}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"删除失败: 数据库操作异常"
                )
            except Exception as e:
                print(f"[ERROR] 删除记录时发生错误: {e}")
                import traceback
                traceback.print_exc()
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"删除失败: {str(e)}"
                )
            finally:
                try:
                    if conn:
                        conn.close()
                except Exception as close_error:
                    print(f"[WARNING] 关闭数据库连接时出错: {close_error}")
        
        except HTTPException:
            raise
        except Exception as e:
            print(f"[ERROR] 删除分割历史记录时发生错误: {e}")
            import traceback
            traceback.print_exc()
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"删除分割历史记录时发生错误: {str(e)}"
            )
    
    # ==================== 私有辅助方法 ====================
    
    @staticmethod
    def _generate_unique_filename(original_filename: str) -> str:
        """
        【文件保存管理】生成唯一文件名
        
        功能：使用UUID为每个上传文件生成唯一标识，防止文件覆盖冲突
        
        参数：
        - original_filename：str，原始上传的文件名
        
        返回：
        - str，唯一文件名（UUID格式 + 原扩展名）
        
        示例：
        - input：'image.png' → output：'a1b2c3d4-e5f6-4g7h-8i9j-k0l1m2n3o4p5.png'
        
        核心逻辑：
        1. 提取文件扩展名（.png/.jpg/.jpeg）
        2. 生成UUID作为文件名主体
        3. 拼接扩展名形成新文件名
        """
        file_ext = Path(original_filename).suffix
        unique_id = str(uuid.uuid4())
        return f"{unique_id}{file_ext}"
    
    @staticmethod
    def _validate_image_file(file: UploadFile) -> bool:
        """
        【文件上传校验】验证上传的文件是否为支持的医学影像格式
        
        功能：检查文件扩展名与MIME类型，确保接受PNG/JPG/JPEG格式
        
        参数：
        - file：UploadFile，FastAPI上传文件对象
        
        返回：
        - bool，True表示支持的格式，False表示不支持或校验错误
        
        校验规则：
        1. 检查文件名后缀（支持.png/.jpg/.jpeg，不区分大小写）
        2. 验证Content-Type MIME类型
        
        支持的MIME类型：
        - image/png、image/x-png（PNG）
        - image/jpeg、image/x-jpeg、image/jpg（JPG/JPEG）
        """
        # 检查文件名是否存在
        if not file.filename:
            return False
        
        # 检查文件扩展名
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            return False
        
        # 检查Content-Type（MIME类型）
        if file.content_type not in ALLOWED_MIME_TYPES:
            return False
        
        return True
    
    @staticmethod
    def _insert_record(
        filename: str,
        model_type: str,
        original_path: str,
        result_path: str
    ) -> Optional[int]:
        """
        【数据库操作】插入分割记录
        
        功能：将图像分割的历史记录保存到SQLite数据库
        
        参数：
        - filename：str，图像文件名（UUID格式）
        - model_type：str，使用的模型类型（'unet'或'fcn'）
        - original_path：str，原始图像的保存路径
        - result_path：str，分割结果的保存路径
        
        返回：
        - int，插入记录的ID
        - None，如果插入失败
        
        数据库表结构：
        - id：自增主键
        - filename：唯一文件名
        - upload_time：当前时间（YYYY-MM-DD HH:MM:SS）
        - model_type：模型类型
        - original_path：原始图像路径
        - result_path：分割结果路径
        """
        conn = None
        try:
            conn = sqlite3.connect(str(DB_PATH), timeout=10, check_same_thread=False)
            cursor = conn.cursor()
            
            # 获取当前时间
            upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 执行INSERT
            insert_sql = """
            INSERT INTO images (filename, upload_time, model_type, original_path, result_path)
            VALUES (?, ?, ?, ?, ?)
            """
            
            cursor.execute(insert_sql, (filename, upload_time, model_type, original_path, result_path))
            conn.commit()
            
            record_id = cursor.lastrowid
            print(f"[SUCCESS] 记录已插入数据库，ID: {record_id}")
            
            return record_id
            
        except sqlite3.IntegrityError as integrity_error:
            print(f"[ERROR] 数据完整性错误（文件名可能重复）: {integrity_error}")
            return None
        except sqlite3.OperationalError as op_error:
            print(f"[ERROR] 数据库操作错误: {op_error}")
            return None
        except Exception as e:
            print(f"[ERROR] 插入记录时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            try:
                if conn:
                    conn.close()
            except Exception as close_error:
                print(f"[WARNING] 关闭数据库连接时出错: {close_error}")
    
    @staticmethod
    def _get_all_records() -> List[Dict]:
        """
        【数据库查询】查询所有分割历史记录
        
        功能：从SQLite数据库查询所有OCTA分割历史，按上传时间倒序排列
        
        返回：
        - List[Dict]，包含所有记录的字典列表，每个字典包含：
          - id：int，记录ID
          - filename：str，文件名
          - upload_time：str，上传时间
          - model_type：str，模型类型
          - original_path：str，原始图像路径
          - result_path：str，分割结果路径
        - []，如果查询失败
        
        排序规则：
        - 按upload_time DESC倒序（最新的在前）
        """
        conn = None
        try:
            conn = sqlite3.connect(str(DB_PATH), timeout=10, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            select_sql = """
            SELECT id, filename, upload_time, model_type, original_path, result_path
            FROM images
            ORDER BY upload_time DESC
            """
            
            cursor.execute(select_sql)
            rows = cursor.fetchall()
            
            records = [dict(row) for row in rows]
            print(f"[INFO] 成功查询 {len(records)} 条历史记录")
            
            return records
            
        except sqlite3.OperationalError as op_error:
            print(f"[ERROR] 查询历史记录时数据库操作错误: {op_error}")
            return []
        except Exception as e:
            print(f"[ERROR] 查询所有记录时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return []
        finally:
            try:
                if conn:
                    conn.close()
            except Exception as close_error:
                print(f"[WARNING] 关闭数据库连接时出错: {close_error}")
    
    @staticmethod
    def _get_record_by_id(record_id: int) -> Optional[Dict]:
        """
        【数据库查询】根据ID查询单条分割记录
        
        功能：查询指定ID的OCTA分割历史记录详情
        
        参数：
        - record_id：int，记录的主键ID
        
        返回：
        - Dict，记录详情对象，包含：
          - id：int，记录ID
          - filename：str，文件名
          - upload_time：str，上传时间
          - model_type：str，模型类型
          - original_path：str，原始图像路径
          - result_path：str，分割结果路径
        - None，如果记录不存在或查询失败
        """
        conn = None
        try:
            conn = sqlite3.connect(str(DB_PATH), timeout=10, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            select_sql = """
            SELECT id, filename, upload_time, model_type, original_path, result_path
            FROM images
            WHERE id = ?
            """
            
            cursor.execute(select_sql, (record_id,))
            row = cursor.fetchone()
            
            if row is None:
                print(f"[WARNING] 未找到ID为 {record_id} 的记录")
                return None
            
            record = dict(row)
            print(f"[INFO] 成功查询ID为 {record_id} 的历史记录")
            
            return record
            
        except sqlite3.OperationalError as op_error:
            print(f"[ERROR] 查询记录时数据库操作错误: {op_error}")
            return None
        except Exception as e:
            print(f"[ERROR] 查询单条记录时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            try:
                if conn:
                    conn.close()
            except Exception as close_error:
                print(f"[WARNING] 关闭数据库连接时出错: {close_error}")
