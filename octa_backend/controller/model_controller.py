"""
OCTA图像分割平台 - 模型管理控制器（简化版）

本模块提供模型权重文件管理API接口，所有业务逻辑由 WeightService 处理。

架构设计：
  - Controller 职责：路由管理、参数验证、HTTP响应
  - Service 职责：业务逻辑、文件操作、数据处理
  
这样的分层设计带来的好处：
  1. 代码简洁：Controller 只需调用 Service 方法
  2. 易于维护：业务逻辑修改只需改 Service
  3. 易于测试：Service 可独立进行单元测试
  4. 代码复用：多个 Controller 可共享 Service

作者：OCTA Web项目组
日期：2026-01-27
"""

from fastapi import APIRouter, HTTPException

from service.model_service import WeightService

# ==================== 路由器初始化 ====================

model_router = APIRouter(
    prefix="/model",
    tags=["模型管理"]
)


# ==================== 接口1：列出可用的模型权重 ====================

@model_router.get("/weights", summary="获取可用模型权重列表")
async def list_model_weights():
    """
    [模型权重列表] 获取所有可用的模型权重文件
    
    功能：扫描 models/weights/ 目录，返回所有 .pth 权重文件信息
    
    返回格式：
    ```json
    {
        "code": 200,
        "msg": "查询成功",
        "data": {
            "weights": [
                {
                    "name": "unet_octa.pth",
                    "path": "models/weights/unet_octa.pth",
                    "size": 45.2,
                    "modified_time": 1705392000.0,
                    "is_default": true
                }
            ],
            "count": 1,
            "default_weight": "models/weights/unet_octa.pth"
        }
    }
    ```
    """
    # Fix: 调用 Service 层处理所有业务逻辑
    result = WeightService.list_weights()
    
    if result["code"] != 200:
        raise HTTPException(
            status_code=result["code"],
            detail=result["msg"]
        )
    
    return result


# ==================== 接口2：获取默认权重信息 ====================

@model_router.get("/default-weight", summary="获取默认模型权重")
async def get_default_weight():
    """
    [默认权重] 获取默认使用的模型权重信息
    
    功能：返回 unet_octa.pth 的详细信息，如果不存在则返回最新的权重
    """
    # Fix: 调用 Service 层处理所有业务逻辑
    result = WeightService.get_default_weight()
    
    if result["code"] != 200:
        raise HTTPException(
            status_code=result["code"],
            detail=result["msg"]
        )
    
    return result


# ==================== 接口3：切换默认模型权重 ====================

@model_router.post("/use-weight", summary="切换使用的模型权重")
async def use_model_weight(weight_path: str):
    """
    [模型切换] 切换使用的模型权重文件
    
    参数：
    - weight_path: 权重文件相对路径（如 "models/weights/unet_octa.pth"）
    
    返回格式：
    ```json
    {
        "code": 200,
        "msg": "模型切换成功",
        "data": {
            "current_weight": "models/weights/trained_unet_20260116140819.pth",
            "name": "trained_unet_20260116140819.pth",
            "size": 43.8
        }
    }
    ```
    """
    # Fix: 参数验证
    if not weight_path:
        raise HTTPException(
            status_code=400,
            detail="weight_path 参数不能为空"
        )
    
    # Fix: 调用 Service 层处理权重切换
    result = WeightService.use_weight(weight_path)
    
    if result["code"] != 200:
        raise HTTPException(
            status_code=result["code"],
            detail=result["msg"]
        )
    
    return result


# ==================== 接口4：获取当前使用的权重 ====================

@model_router.get("/current-weight", summary="获取当前使用的模型权重")
async def get_current_weight():
    """
    [当前权重] 获取当前使用的模型权重路径
    
    返回格式：
    ```json
    {
        "code": 200,
        "msg": "查询成功",
        "data": {
            "current_weight": "models/weights/unet_octa.pth"
        }
    }
    ```
    """
    # Fix: 调用 Service 层获取当前权重
    current_weight = WeightService.get_current_weight()
    
    return {
        "code": 200,
        "msg": "查询成功",
        "data": {
            "current_weight": current_weight
        }
    }


# ==================== 使用说明 ====================

"""
架构说明：

本模块采用 MVC 分层架构：
  - Model：无（数据由 Service 操作）
  - View：FastAPI 路由 + JSON 响应
  - Controller：本文件，处理 HTTP 请求和响应
  - Service：WeightService 类（service/model_service.py）
  
控制器职责：
  1. 接收 HTTP 请求
  2. 验证请求参数
  3. 调用 Service 层方法
  4. 返回 HTTP 响应
  
Service 职责：
  1. 权重文件扫描
  2. 权重信息获取
  3. 权重文件验证
  4. 权重切换管理

接口列表：
  - GET /model/weights - 获取所有可用权重列表
  - GET /model/default-weight - 获取默认权重
  - GET /model/current-weight - 获取当前权重
  - POST /model/use-weight - 切换权重

前端调用示例：
  ```javascript
  // 获取权重列表
  const resp = await axios.get('http://127.0.0.1:8000/model/weights')
  
  // 切换权重
  const switchResp = await axios.post(
    'http://127.0.0.1:8000/model/use-weight',
    null,
    { params: { weight_path: 'models/weights/unet_octa.pth' } }
  )
  ```

权重文件命名规范：
  - 默认权重：unet_octa.pth
  - 训练生成：trained_unet_YYYYMMDDHHMMSS.pth
  - 自定义权重：任意 .pth 文件
"""
