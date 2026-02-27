"""
权重管理控制器

# Fix: 平台优化 - 放弃训练模块，聚焦预测功能

路由前缀：/api/v1/weight
- POST /upload   : 上传并校验权重（.pth/.pt，≤200MB），返回 weight_id
- GET  /list     : 查询权重列表（可按model_type过滤）
- DELETE /delete/{weight_id} : 删除权重
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, status
from fastapi.responses import JSONResponse
from typing import Optional

# from service.weight_service import WeightService  # 已废弃，使用router/weight_router.py

weight_router = APIRouter(prefix="/api/v1/weight", tags=["权重管理"])


@weight_router.post("/upload")
async def upload_weight(
    file: UploadFile = File(..., description="上传的权重文件（.pth/.pt）"),
    model_type: str = Form("unet", description="模型类型：unet（默认）")
):
    """上传权重并做格式/体积/state_dict校验。"""
    try:
        result = WeightService.save_weight(file, model_type=model_type.lower())
        return JSONResponse({
            "code": 200,
            "msg": "权重上传成功",
            "data": result
        })
    except HTTPException:
        raise
    except Exception as exc:  # 兜底异常
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@weight_router.get("/list")
async def list_weights(model_type: Optional[str] = Query(None, description="模型类型筛选")):
    """查询已上传权重列表。"""
    weights = WeightService.list_weights(model_type=model_type.lower() if model_type else None)
    return JSONResponse({
        "code": 200,
        "msg": "查询成功",
        "data": [
            {
                "id": item.get("id"),
                "weight_id": (item.get("file_path") or "").split("/")[-2] if item.get("file_path") else None,
                "file_name": item.get("file_name"),
                "file_size": item.get("file_size"),
                "model_type": item.get("model_type"),
                "upload_time": item.get("upload_time"),
                "file_path": item.get("file_path"),
            }
            for item in weights
        ]
    })


@weight_router.delete("/delete/{weight_id}")
async def delete_weight(weight_id: str):
    """删除指定weight_id的权重文件与记录。"""
    WeightService.delete_weight(weight_id)
    return JSONResponse({
        "code": 200,
        "msg": "删除成功",
        "data": {"weight_id": weight_id}
    })
