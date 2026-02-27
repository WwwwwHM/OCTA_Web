"""
OCTA图像分割平台 - FastAPI后端主程序

核心能力：
1. 启动时自动扫描models目录并导入所有子目录下的model.py；
2. 触发模型注册（即插即用）；
3. 注册通用分割推理接口；
4. 输出已注册模型列表；
5. 提供统一日志与CORS配置。
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from pathlib import Path
from typing import List

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.config import (
    CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_HEADERS,
    CORS_ALLOW_METHODS,
    CORS_ORIGINS,
)
from router.seg_router import router as seg_router
from service import model_registry


# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ==================== 路径配置 ====================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


def _import_module_from_file(model_file: Path) -> str:
    """
    按文件路径安全导入模块，避免与同名.py文件冲突。

    Args:
        model_file: models下的model.py文件路径。

    Returns:
        实际加载的模块名。
    """
    relative_path = model_file.relative_to(BASE_DIR).with_suffix("")
    default_module_name = ".".join(relative_path.parts)
    safe_module_name = f"autoload_{model_file.parent.name}_model"

    spec = importlib.util.spec_from_file_location(safe_module_name, model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法构建模块spec: {model_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return default_module_name


def auto_import_model_modules() -> List[str]:
    """
    自动导入models目录下所有子目录中的model.py。

    Returns:
        成功导入的模块名列表。
    """
    imported_modules: List[str] = []

    if not MODELS_DIR.exists():
        logger.warning("models目录不存在: %s", MODELS_DIR)
        return imported_modules

    for model_file in sorted(MODELS_DIR.rglob("model.py")):
        if model_file.parent == MODELS_DIR:
            continue

        try:
            module_name = _import_module_from_file(model_file)
            imported_modules.append(module_name)
            logger.info("模型模块导入成功: %s", module_name)
        except Exception as exc:
            logger.exception("模型模块导入失败: %s, 错误: %s", model_file, exc)

    return imported_modules


# ==================== FastAPI应用 ====================
app = FastAPI(
    title="OCTA图像分割API",
    description="支持即插即用模型注册的通用分割后端服务",
    version="2.0.0",
)

# CORS跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
)

# 注册分割推理路由
app.include_router(seg_router)


@app.get("/", tags=["基础接口"])
async def root() -> dict:
    """后端健康检查接口。"""
    return {"code": 200, "msg": "服务运行中", "data": {"service": "OCTA FastAPI"}}


@app.on_event("startup")
async def startup_event() -> None:
    """应用启动事件：自动导入模型并打印注册结果。"""
    logger.info("后端启动中，开始扫描模型目录: %s", MODELS_DIR)

    imported = auto_import_model_modules()
    logger.info("模型模块扫描完成，成功导入数量: %d", len(imported))

    registered_models = model_registry.list_models()
    logger.info("当前已注册模型列表: %s", registered_models)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
