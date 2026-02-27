"""
权重管理路由

实现接口：
- POST   /api/v1/weight/upload  上传+校验权重
- GET    /api/v1/weight/list    查询权重列表（含官方权重）
- DELETE /api/v1/weight/delete/{weight_id} 删除权重（禁止删除官方）

设计要点：
- 使用 core.weight_validator 进行格式/大小/state_dict 校验
- 权重文件落盘至 static/uploads/weight/{weight_id}/{filename}
- 记录写入 file_management 表（file_type='weight'）
- 响应统一格式：{"code": int, "msg": str, "data": {...}}
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, status
from fastapi.responses import JSONResponse

from config.config import (
    WEIGHT_UPLOAD_ROOT,
    WEIGHT_ALLOWED_FORMATS,
    WEIGHT_MAX_SIZE,
    OFFICIAL_WEIGHT_PATH,
)
from core.weight_validator import get_validator
from dao.file_dao import FileDAO

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/weight", tags=["权重管理"])

# 允许的扩展名（小写）
_ALLOWED_EXT = {ext.lower() for ext in WEIGHT_ALLOWED_FORMATS}


def _ensure_dirs() -> Path:
    root = Path(WEIGHT_UPLOAD_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    Path(OFFICIAL_WEIGHT_PATH).parent.mkdir(parents=True, exist_ok=True)
    return root


def _validate_upload(upload: UploadFile) -> bytes:
    suffix = upload.filename.split(".")[-1].lower()
    if suffix not in _ALLOWED_EXT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"权重格式错误，仅支持 {', '.join(_ALLOWED_EXT)}",
        )
    content = upload.file.read()
    size = len(content)
    if size > WEIGHT_MAX_SIZE:
        size_mb = size / 1024 / 1024
        max_mb = WEIGHT_MAX_SIZE / 1024 / 1024
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"权重文件过大 ({size_mb:.2f}MB)，最大支持{max_mb:.2f}MB",
        )
    if size == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="权重文件为空")
    return content


def _add_db_record(file_name: str, file_path: str, file_size_mb: float, model_type: str) -> Optional[int]:
    return FileDAO.add_file_record(
        file_name=file_name,
        file_path=file_path,
        file_type="weight",
        file_size=file_size_mb,
        model_type=model_type,
    )


@router.post("/upload")
async def upload_weight(
    weight_file: UploadFile = File(..., description="上传的权重文件（.pth/.pt）"),
    model_type: str = "unet",
    user_id: Optional[str] = Query(None, description="用户ID，可选")
):
    """上传并校验权重，返回 weight_id。"""
    try:
        root = _ensure_dirs()
        content = _validate_upload(weight_file)
        weight_id = uuid.uuid4().hex
        target_dir = root / weight_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / weight_file.filename

        with open(target_path, "wb") as f:
            f.write(content)

        # 权重校验
        validator = get_validator(max_size_mb=WEIGHT_MAX_SIZE // 1024 // 1024)
        is_valid, error_msg, metadata = validator.validate_file(target_path, model_type=model_type.lower())
        if not is_valid:
            logger.warning(f"[权重上传] 校验失败 weight_id={weight_id}, file={weight_file.filename}, error={error_msg}")
            target_path.unlink(missing_ok=True)
            if not any(target_dir.iterdir()):
                target_dir.rmdir()
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"权重校验失败: {error_msg}")

        size_mb = len(content) / 1024 / 1024
        file_id = _add_db_record(weight_file.filename, str(target_path), size_mb, model_type.lower())
        upload_time = datetime.utcnow().isoformat()

        logger.info(f"[权重上传] 成功 weight_id={weight_id}, file={weight_file.filename}, size_mb={size_mb:.3f}, user_id={user_id}")

        return JSONResponse({
            "code": 200,
            "msg": "权重上传成功",
            "data": {
                "weight_id": weight_id,
                "file_name": weight_file.filename,
                "file_size_mb": round(size_mb, 4),
                "upload_time": upload_time,
                "file_id": file_id,
                "metadata": metadata,
            }
        })
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"[权重上传] 存储失败 file={weight_file.filename}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"存储失败: {exc}")


@router.get("/list")
async def list_weights(
    user_id: Optional[str] = Query(None, description="用户ID，可选（当前未存储用户维度，仅返回全部）"),
    model_type: Optional[str] = Query(None, description="模型类型，可选")
):
    """查询权重列表，包含官方权重。"""
    _ensure_dirs()
    weights = FileDAO.get_file_list(file_type="weight", model_type=model_type.lower() if model_type else None)

    items = []

    # 官方权重
    official_path = Path(OFFICIAL_WEIGHT_PATH)
    if official_path.exists():
        items.append({
            "weight_id": "official",
            "file_name": official_path.name,
            "file_size_mb": round(official_path.stat().st_size / 1024 / 1024, 4),
            "upload_time": None,
            "is_official": 1,
            "file_path": str(official_path),
        })

    for item in weights:
        path = item.get("file_path")
        weight_id = Path(path).parent.name if path else None
        items.append({
            "weight_id": weight_id,
            "file_name": item.get("file_name"),
            "file_size_mb": item.get("file_size"),
            "upload_time": item.get("upload_time"),
            "is_official": 0,
            "file_path": path,
            "model_type": item.get("model_type"),
            "id": item.get("id"),
        })

    return JSONResponse({
        "code": 200,
        "msg": "查询成功",
        "data": {
            "weights": items,
            "total": len(items)
        }
    })


@router.delete("/delete/{weight_id}")
async def delete_weight(weight_id: str):
    """删除指定权重，官方权重禁止删除。"""
    _ensure_dirs()
    if weight_id == "official":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="禁止删除官方权重")

    target_dir = Path(WEIGHT_UPLOAD_ROOT) / weight_id
    if not target_dir.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="权重不存在")

    # 删除文件夹内容
    try:
        for child in target_dir.glob("**/*"):
            if child.is_file():
                child.unlink(missing_ok=True)
        target_dir.rmdir()
    except Exception as exc:
        logger.exception(f"[权重删除] 删除失败 weight_id={weight_id}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"删除失败: {exc}")

    # 删除数据库记录
    records = FileDAO.get_file_list(file_type="weight")
    for item in records:
        path = item.get("file_path") or ""
        if weight_id in Path(path).parts:
            try:
                FileDAO.delete_file(item.get("id"))
            except Exception:
                pass

    logger.info(f"[权重删除] 成功 weight_id={weight_id}")

    return JSONResponse({
        "code": 200,
        "msg": "权重删除成功",
        "data": {"weight_id": weight_id}
    })
