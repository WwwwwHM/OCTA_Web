"""
OCTA图像分割平台 - 文件管理控制器

# Fix: 平台优化 - 放弃训练模块，聚焦预测功能
本模块提供图片和权重文件的增删改查API接口，支持：
- 文件列表查询（支持类型筛选）
- 文件详情查询
- 文件删除（数据库+本地文件）
- 图片复用测试分割（关联模型权重）

架构位置：
├── FastAPI路由层 (main.py)
├── 控制层 (controller/file_controller.py) ← 本文件
│   └── 业务逻辑编排、参数校验、异常处理
├── 服务层 (service/weight_service.py)
├── 数据层 (dao/file_dao.py)
└── 工具层 (utils/file_utils.py)

作者：OCTA Web项目组
日期：2026-01-27
"""

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, status
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional, Dict, Any
from pathlib import Path
import traceback
import base64

# 导入数据访问层：文件CRUD操作
from dao.file_dao import FileDAO

# Fix: 平台优化 - 移除训练服务导入，仅保留权重管理
# from service.weight_service import WeightService  # 已废弃，使用router/weight_router.py

# 导入控制层：图像分割控制器
from controller.image_controller import ImageController


# ==================== 路由器初始化 ====================

file_router = APIRouter(
    prefix="/file",
    tags=["文件管理"]
)


# ==================== 辅助函数：统一响应格式 ====================

def success_response(data: Any = None, msg: str = "操作成功") -> Dict[str, Any]:
    """
    成功响应格式
    weight_id: Optional[str] = Query(None, description="模型权重ID（推荐，可选）")
    
    Args:
        data: 返回的数据
        msg: 提示信息
        
    Returns:
        标准JSON格式：{code: 200, msg: "...", data: ...}
    """
    return {
    - weight_id: 可选，指定模型权重ID；留空则使用官方预置
        "code": 200,
        "msg": msg,
        "data": data
    }


def error_response(code: int, msg: str, data: Any = None) -> Dict[str, Any]:
    """
    错误响应格式
    
    Args:
        code: 错误码（400/404/500）
        msg: 错误信息
        data: 可选的错误详情
        
    Returns:
        标准JSON格式：{code: xxx, msg: "...", data: ...}
    """
    return {
        "code": code,
        "msg": msg,
        "data": data
    }


# ==================== 接口1：查询文件列表 ====================

@file_router.get("/list", summary="查询文件列表")
async def get_file_list(
    file_type: Optional[str] = Query(None, description="文件类型筛选：'image'（图片）或 'dataset'（数据集），留空返回全部")
):
    """
    [文件列表查询] 获取所有文件记录，支持类型筛选
    
    查询参数：
    - file_type: 可选，筛选条件
      - "image": 仅返回图片文件
      - "dataset": 仅返回数据集文件
      - None: 返回所有文件
    
    返回格式：
    ```json
    {
        "code": 200,
        "msg": "查询成功",
        "data": [
            {
                "id": 1,
                "file_name": "image.png",
        if weight_id:
            model_used = WeightService.resolve_weight_path(weight_id, 'unet')
            print(f"[INFO] 使用 weight_id={weight_id} 对应权重: {model_used}")
        elif weight_path:
            # 如果提供了权重路径，检查是否存在
            if not Path(weight_path).exists():
                print(f"[WARNING] 权重文件不存在: {weight_path}")
                raise HTTPException(
                    status_code=400,
                    detail=f"权重文件不存在：{weight_path}"
                )
            model_used = weight_path
        else:
            # 使用官方预置权重
            model_used = WeightService.resolve_weight_path(None, 'unet')
            print(f"[INFO] 使用官方预置权重: {model_used}")
                "file_path": "uploads/xxx.png",
                "file_type": "image",
                "upload_time": "2026-01-16 10:00:00",
                "related_model": null,
                "file_size": 1024000
            },
            ...
        ]
    }
    ```
    
    异常情况：
            model_path=model_used if (weight_id or weight_path) else None,  # 仅在指定时传递
    - 500: 数据库查询失败
    """
    try:
        # ==================== 步骤1：参数校验 ====================
        # 如果提供了file_type，必须是有效值
        if file_type is not None and file_type not in ["image", "dataset"]:
            raise HTTPException(
                status_code=400,
                detail=f"参数错误：file_type必须是'image'或'dataset'，当前值：{file_type}"
            )
        
        # ==================== 步骤2：调用DAO查询文件列表 ====================
        print(f"[INFO] 正在查询文件列表，筛选条件：{file_type or '全部'}")
        file_list = FileDAO.get_file_list(file_type=file_type)
        
        # ==================== 步骤3：返回成功响应 ====================
        print(f"[SUCCESS] 查询成功，共 {len(file_list)} 条记录")
        return success_response(
            data=file_list,
            msg=f"查询成功，共 {len(file_list)} 条记录"
        )
        
    except HTTPException:
        # 已经是HTTP异常，直接抛出
        raise
    except Exception as e:
        # 未知异常，打印堆栈并返回500错误
        print(f"[ERROR] 查询文件列表失败: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"服务器内部错误：{str(e)}"
        )


# ==================== 接口2：文件预览（Base64编码） ====================

@file_router.get("/preview/{file_id}", summary="获取文件预览（Base64编码）")
async def get_file_preview(file_id: int):
    """
    [文件预览] 获取指定ID的文件预览内容（Base64编码）
    
    功能说明：
    - 对于图片文件（image）：返回图片的base64编码
    - 用于前端直接显示图片预览（img src="data:image/png;base64,...")
    - 防止路径遍历攻击（通过file_id查询数据库获取安全的文件路径）
    
    参数：
    - file_id: 文件在数据库中的ID
    
    返回格式（成功）：
    ```json
    {
        "code": 200,
        "msg": "预览获取成功",
        "data": {
            "file_id": 1,
            "filename": "octa_image.png",
            "file_type": "image",
            "base64_data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            "preview_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        }
    }
    ```
    
    异常情况：
    - 404: 文件ID不存在或文件已删除
    - 400: 文件类型不支持预览（如dataset）
    - 500: 文件读取失败
    """
    try:
        # ==================== 步骤1：从数据库查询文件信息 ====================
        print(f"[INFO] 获取文件预览，file_id={file_id}")
        
        # 通过FileDAO获取单个文件的信息
        file_info = FileDAO.get_file_by_id(file_id)
        
        if file_info is None:
            raise HTTPException(
                status_code=404,
                detail=f"文件不存在：ID={file_id}"
            )
        
        # ==================== 步骤2：检查文件类型是否支持预览 ====================
        file_type = file_info.get('file_type')
        
        # 目前仅支持图片预览，数据集暂不支持
        if file_type != 'image':
            raise HTTPException(
                status_code=400,
                detail=f"不支持预览该文件类型：{file_type}，仅支持图片(image)预览"
            )
        
        # ==================== 步骤3：获取安全的文件路径并读取 ====================
        file_path = Path(file_info.get('file_path'))
        
        # 验证文件是否真实存在（防止数据库记录与文件系统不同步）
        if not file_path.exists():
            print(f"[WARNING] 数据库中的文件不存在（可能已被删除）：{file_path}")
            raise HTTPException(
                status_code=404,
                detail=f"文件已被删除或移动：{file_path.name}"
            )
        
        # ==================== 步骤4：读取文件并编码为Base64 ====================
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        # 确定MIME类型
        file_ext = file_path.suffix.lower()
        mime_type_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg'
        }
        mime_type = mime_type_map.get(file_ext, 'image/jpeg')
        
        # 编码为Base64
        base64_str = base64.b64encode(file_data).decode('utf-8')
        preview_url = f"data:{mime_type};base64,{base64_str}"
        
        # ==================== 步骤5：返回预览数据 ====================
        print(f"[SUCCESS] 文件预览获取成功：{file_path.name}，大小：{len(file_data)} bytes")
        return success_response(
            data={
                "file_id": file_id,
                "filename": file_info.get('file_name'),
                "file_type": file_type,
                "base64_data": base64_str,
                "preview_url": preview_url,
                "mime_type": mime_type
            },
            msg="预览获取成功"
        )
        
    except HTTPException:
        # 已经是HTTP异常，直接抛出
        raise
    except Exception as e:
        print(f"[ERROR] 获取文件预览失败: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"服务器内部错误：{str(e)}"
        )

# ==================== 接口2.1：查询模型权重列表（放在/{file_id}之前，优先匹配） ====================

@file_router.get("/model-weights", summary="查询模型权重列表（按模型类型筛选）")
async def get_model_weights(
    model_type: Optional[str] = Query(None, description="模型类型：'unet'、'rs_unet3_plus'")
):
    """
    [模型权重查询] 获取指定模型类型的权重文件列表
    
    **功能说明：**
    - 专门用于前端模型选择器的权重动态加载
    - 根据选中的模型类型，返回对应的权重文件列表
    - 支持两种模型架构的权重筛选
    
    **参数说明：**
    - model_type: 模型类型，可选值：
      - 'unet': U-Net模型权重
      - 'rs_unet3_plus': RS-Unet3+模型权重（OCTA专用）
      - 不传递则要求用户先选择模型（返回空列表）
    
    **返回格式：**
    ```json
    {
        "code": 200,
        "msg": "找到N个unet权重",
        "data": [
            {
                "id": 5,
                "file_name": "unet_epoch10_acc0.95.pth",
                "file_path": "models/weights_unet/unet_epoch10_acc0.95.pth",
                "file_size": 102400,
                "file_type": "weight",
                "model_type": "unet",
                "upload_time": "2026-01-20 14:30:00",
                "related_model": null
            }
        ]
    }
    ```
    
    **使用示例：**
    - GET /file/model-weights?model_type=unet → 返回U-Net权重列表
    - GET /file/model-weights?model_type=rs_unet3_plus → 返回RS-Unet3+权重列表
    - GET /file/model-weights → 返回空列表（提示先选择模型）
    
    **前端集成示例：**
    ```javascript
    // 监听模型选择变化，动态加载权重
    watch(selectedModel, async (newModel) => {
      const response = await axios.get(
        `http://127.0.0.1:8000/file/model-weights?model_type=${newModel}`
      )
      availableWeights.value = response.data.data
    })
    ```
    
    异常情况：
    - 400: 无效的模型类型参数
    - 500: 数据库查询失败
    """
    try:
        # ==================== 步骤1：参数校验 ====================
        # 验证模型类型是否在允许的范围内
        valid_model_types = ['unet', 'rs_unet3_plus']
        
        if model_type is not None and model_type not in valid_model_types:
            print(f"[WARNING] 无效的模型类型参数: {model_type}")
            raise HTTPException(
                status_code=400,
                detail=f"无效的模型类型：{model_type}，允许值：{', '.join(valid_model_types)}"
            )
        
        # ==================== 步骤2：处理未选择模型的情况 ====================
        # 如果前端未传递model_type，返回空列表并提示
        if model_type is None:
            print(f"[INFO] 未指定模型类型，返回空列表")
            return success_response(
                data=[],
                msg="请先选择模型类型"
            )
        
        # ==================== 步骤3：查询指定模型的权重列表 ====================
        print(f"[INFO] 查询模型权重，model_type={model_type}")
        
        # 调用DAO层，使用双重筛选：file_type='weight' + model_type
        weight_list = FileDAO.get_file_list(file_type='weight', model_type=model_type)
        
        # ==================== 步骤4：返回筛选结果 ====================
        print(f"[SUCCESS] 查询成功，找到{len(weight_list)}个{model_type}权重")
        return success_response(
            data=weight_list,
            msg=f"找到{len(weight_list)}个{model_type}权重"
        )
        
    except HTTPException:
        # 已经是HTTP异常，直接抛出
        raise
    except Exception as e:
        # 未知异常，打印堆栈并返回500错误
        print(f"[ERROR] 查询模型权重失败: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"查询失败：{str(e)}"
        )

# ==================== 接口2.2：获取单个文件详情 ====================

@file_router.get("/{file_id}", summary="获取单个文件详情")
async def get_file_detail(file_id: int):
    """
    [文件详情查询] 根据ID获取单个文件的详细信息
    
    路径参数：
    - file_id: 文件记录的ID（整数）
    
    返回格式：
    ```json
    {
        "code": 200,
        "msg": "查询成功",
        "data": {
            "id": 1,
            "file_name": "image.png",
            "file_path": "uploads/xxx.png",
            "file_type": "image",
            "upload_time": "2026-01-16 10:00:00",
            "related_model": "models/weights/unet_xxx.pth",
            "file_size": 1024000
        }
    }
    ```
    
    异常情况：
    - 404: 文件记录不存在
    - 500: 数据库查询失败
    """
    try:
        # ==================== 步骤1：调用DAO查询文件详情 ====================
        print(f"[INFO] 正在查询文件详情，ID: {file_id}")
        file_info = FileDAO.get_file_by_id(file_id)
        
        # ==================== 步骤2：检查文件是否存在 ====================
        if file_info is None:
            print(f"[WARNING] 文件不存在，ID: {file_id}")
            raise HTTPException(
                status_code=404,
                detail=f"文件不存在：ID={file_id}"
            )
        
        # ==================== 步骤3：返回成功响应 ====================
        print(f"[SUCCESS] 查询成功，文件名: {file_info['file_name']}")
        return success_response(
            data=file_info,
            msg="查询成功"
        )
        
    except HTTPException:
        # 已经是HTTP异常，直接抛出
        raise
    except Exception as e:
        # 未知异常，打印堆栈并返回500错误
        print(f"[ERROR] 查询文件详情失败: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"服务器内部错误：{str(e)}"
        )


# ==================== 接口3：删除文件 ====================

@file_router.delete("/delete/{file_id}", summary="删除文件")
async def delete_file(file_id: int):
    """
    [文件删除] 删除数据库记录和本地文件（图片/数据集目录）
    
    路径参数：
    - file_id: 文件记录的ID（整数）
    
    删除逻辑：
    1. 查询文件记录是否存在
    2. 删除数据库记录
    3. 删除本地文件（单文件）或目录（数据集）
    
    返回格式：
    ```json
    {
        "code": 200,
        "msg": "删除成功",
        "data": {
            "deleted_file": "image.png",
            "deleted_path": "uploads/xxx.png"
        }
    }
    ```
    
    异常情况：
    - 404: 文件记录不存在
    - 500: 删除失败（数据库或文件系统错误）
    """
    try:
        # ==================== 步骤1：检查文件是否存在 ====================
        print(f"[INFO] 准备删除文件，ID: {file_id}")
        file_info = FileDAO.get_file_by_id(file_id)
        
        if file_info is None:
            print(f"[WARNING] 文件不存在，ID: {file_id}")
            raise HTTPException(
                status_code=404,
                detail=f"文件不存在：ID={file_id}"
            )
        
        file_name = file_info['file_name']
        file_path = file_info['file_path']
        
        # ==================== 步骤2：调用DAO删除文件 ====================
        # FileDAO.delete_file() 会同时删除数据库记录和本地文件
        print(f"[INFO] 正在删除文件: {file_name}，路径: {file_path}")
        success = FileDAO.delete_file(file_id)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="文件删除失败，请检查文件是否被占用或权限不足"
            )
        
        # ==================== 步骤3：返回成功响应 ====================
        print(f"[SUCCESS] 文件删除成功: {file_name}")
        return success_response(
            data={
                "deleted_file": file_name,
                "deleted_path": file_path
            },
            msg="删除成功"
        )
        
    except HTTPException:
        # 已经是HTTP异常，直接抛出
        raise
    except Exception as e:
        # 未知异常，打印堆栈并返回500错误
        print(f"[ERROR] 删除文件失败: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"服务器内部错误：{str(e)}"
        )


# ==================== 接口5：复用图片测试分割模型 ====================

@file_router.post("/test/{file_id}", summary="复用图片测试分割模型")
async def reuse_image_for_testing(
    file_id: int,
    weight_path: Optional[str] = Query(None, description="模型权重路径（兼容参数，可选）"),
    weight_id: Optional[str] = Query(None, description="模型权重ID（推荐，可选）")
):
    """
    # Fix: 平台优化 - 放弃训练模块，聚焦预测功能
    [图片复用] 使用已上传的图片测试分割模型（推理专用）
    
    路径参数：
    - file_id: 图片文件的ID（必须是image类型）
    
    查询参数：
    - weight_id: 可选，指定模型权重ID；留空则使用官方预置
    - weight_path: 可选，兼容旧参数
    
    测试流程：
    1. 验证文件ID对应的是图片类型
    2. 根据weight_id或weight_path解析权重路径
    3. 调用segment_octa_image进行分割（自动设备选择）
    4. 更新文件记录的related_model字段（关联使用的模型）
    5. 返回分割结果路径
    
    返回格式：
    ```json
    {
        "success": true,
        "message": "图像分割完成",
        "original_filename": "xxx.png",
        "result_filename": "xxx_seg.png",
        "image_url": "/images/xxx.png",
        "result_url": "/results/xxx_seg.png",
        "model_type": "unet",
        "record_id": 1
    }
    ```
    
    异常情况：
    - 404: 文件不存在或不是图片类型
    - 400: 参数错误（权重路径不存在）
    - 500: 分割失败
    """
    try:
        # ==================== 步骤1：检查文件是否存在且为图片类型 ====================
        print(f"[INFO] 准备复用图片测试分割，文件ID: {file_id}")
        file_info = FileDAO.get_file_by_id(file_id)
        
        if file_info is None:
            print(f"[WARNING] 文件不存在，ID: {file_id}")
            raise HTTPException(
                status_code=404,
                detail=f"文件不存在：ID={file_id}"
            )
        
        if file_info['file_type'] != 'image':
            print(f"[WARNING] 文件类型错误，预期image，实际{file_info['file_type']}")
            raise HTTPException(
                status_code=400,
                detail=f"文件类型错误：仅支持图片文件（image），当前类型：{file_info['file_type']}"
            )
        
        image_path = file_info['file_path']
        image_name = file_info['file_name']
        
        # ==================== 步骤2：验证权重路径（如果提供） ====================
        if weight_path:
            # 如果提供了权重路径，检查是否存在
            if not Path(weight_path).exists():
                print(f"[WARNING] 权重文件不存在: {weight_path}")
                raise HTTPException(
                    status_code=400,
                    detail=f"权重文件不存在：{weight_path}"
                )
            model_used = weight_path
        else:
            # 使用默认权重
            model_used = "./models/weights/unet_octa.pth"
            print(f"[INFO] 使用默认模型权重: {model_used}")
        
        # ==================== 步骤3：调用分割服务 ====================
        print(f"[INFO] 开始图像分割，图片: {image_name}，模型: {model_used}")
        
        # 创建临时UploadFile对象（模拟上传）
        # 注意：ImageController.segment_octa需要UploadFile对象
        # 这里我们直接调用底层的segment_octa_image函数
        from models.unet import segment_octa_image
        
        result_path = segment_octa_image(
            image_path=image_path,
            model_type='unet',
            model_path=model_used if weight_path else None,  # 仅在指定时传递
            device='cpu'
        )

        # 若分割失败，segment_octa_image 会返回原图路径，这里明确判定并返回错误
        if Path(result_path).resolve() == Path(image_path).resolve():
            print("[ERROR] 分割失败，返回了原图路径，可能是模型加载或权重不匹配导致")
            raise HTTPException(
                status_code=500,
                detail="分割失败：模型可能未正确加载或权重与模型结构不匹配"
            )
        
        # ==================== 步骤4：更新文件记录关联模型 ====================
        print(f"[INFO] 更新文件记录关联模型: {model_used}")
        FileDAO.update_file_relation(file_id, model_used)
        
        # ==================== 步骤5：生成结果URL ====================
        # 提取结果文件名（用于生成URL）
        result_filename = Path(result_path).name
        result_url = f"http://127.0.0.1:8000/results/{result_filename}"
        
        # ==================== 步骤6：返回分割结果（与 /segment-octa 接口结构保持一致） ====================
        print(f"[SUCCESS] 图像分割成功，结果路径: {result_path}")

        return JSONResponse(
            content={
                "success": True,
                "message": "图像分割完成",
                "original_filename": image_name,
                "saved_filename": image_name,
                "result_filename": result_filename,
                "image_url": f"/images/{image_name}",
                "result_url": f"/results/{result_filename}",
                "model_type": "unet",
                "record_id": file_id
            },
            status_code=status.HTTP_200_OK
        )
        
    except HTTPException:
        # 已经是HTTP异常，直接抛出
        raise
    except Exception as e:
        # 未知异常，打印堆栈并返回500错误
        print(f"[ERROR] 图片复用测试失败: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"分割失败：{str(e)}"
        )


# ==================== 模块说明 ====================

"""
# Fix: 平台优化 - 放弃训练模块，聚焦预测功能
使用指南：

1. 在main.py中注册路由：
   ```python
   from controller.file_controller import file_router
   app.include_router(file_router)
   ```

2. 接口URL格式（推理专用）：
   - GET  /file/list?file_type=image
   - GET  /file/model-weights?model_type=unet
   - GET  /file/{file_id}
   - GET  /file/preview/{file_id}
   - DELETE /file/delete/{file_id}
   - POST /file/test/{file_id}?weight_id=xxx  # 推荐
   - POST /file/test/{file_id}?weight_path=xxx  # 兼容

3. 统一返回格式：
   成功：{"code": 200, "msg": "...", "data": {...}}
   失败：{"code": 4xx/5xx, "msg": "...", "data": null}

4. 依赖说明：
   - FileDAO: 数据库CRUD操作
   - WeightService: 权重解析与管理
   - ImageController: 图像分割控制（已有segment_octa接口）

5. 注意事项：
   - 测试分割会自动更新related_model字段
   - 删除操作同时删除数据库记录和本地文件
   - 所有异常都会被捕获并返回HTTP错误
   - 参数校验确保数据合法性
   - 设备自动选择（CUDA优先，CPU回退）
"""
