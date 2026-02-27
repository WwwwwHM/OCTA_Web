# 🔧 后端集成验证指南

## 概述

前端已全面支持 RS-Unet3+ 模型的选择和参数配置。本指南帮助你验证和更新后端以完全支持这些新功能。

---

## ✅ 第一步：验证分割接口支持

### 检查 `/segment-octa/` 接口

**文件位置**：`octa_backend/main.py`

**查找代码**：
```python
@app.post("/segment-octa/", tags=["图像分割"])
async def segment_octa(
    file: UploadFile = File(..., description="上传的PNG/JPG/JPEG格式图像文件"),
    model_type: str = Form("unet", description="模型类型：'unet' 或 'fcn'"),
    weight_path: str = Form(None, description="模型权重路径（可选）")
):
    """[核心接口] OCTA图像分割端点"""
    return await ImageController.segment_octa(file, model_type, weight_path)
```

**验证清单**：
- [x] 接口存在 `/segment-octa/`
- [x] 接受 `model_type` 参数
- [ ] `model_type` 文档说明是否需要更新？

**如需更新**：
```python
@app.post("/segment-octa/", tags=["图像分割"])
async def segment_octa(
    file: UploadFile = File(..., description="上传的PNG/JPG/JPEG格式图像文件"),
    model_type: str = Form("unet", description="模型类型：'unet'、'fcn' 或 'rs_unet3_plus'"),
    weight_path: str = Form(None, description="模型权重路径（可选）")
):
    """[核心接口] OCTA图像分割端点 - 支持U-Net、FCN、RS-Unet3+"""
    return await ImageController.segment_octa(file, model_type, weight_path)
```

---

## ✅ 第二步：验证训练接口

### 检查 `/train/upload-dataset` 接口

**文件位置**：`octa_backend/main.py` 或 `octa_backend/controller/train_controller.py`

**当前接口可能看起来像**：
```python
@app.post("/train/upload-dataset")
async def upload_dataset(
    file: UploadFile = File(...),
    epochs: int = Form(10),
    lr: float = Form(0.001),
    batch_size: int = Form(4)
):
    # 处理逻辑
```

**需要更新为**：
```python
@app.post("/train/upload-dataset")
async def upload_dataset(
    file: UploadFile = File(..., description="训练数据集 ZIP 包"),
    model_arch: str = Form("unet", description="模型架构：'unet'、'fcn' 或 'rs_unet3_plus'"),
    epochs: int = Form(10, description="训练轮数"),
    lr: float = Form(0.001, description="学习率"),
    weight_decay: float = Form(0.0001, description="权重衰减"),
    batch_size: int = Form(4, description="批次大小")
):
    """[核心接口] 模型训练端点 - 支持多种模型架构"""
    return await TrainController.upload_dataset(
        file=file,
        model_arch=model_arch,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size
    )
```

---

## ✅ 第三步：更新 ImageController（分割）

**文件位置**：`octa_backend/controller/image_controller.py`

**检查 `segment_octa` 方法**：

```python
@staticmethod
async def segment_octa(file: UploadFile, model_type: str, weight_path: Optional[str]):
    """处理OCTA图像分割"""
    # 当前逻辑可能只支持 'unet' 和 'fcn'
    # 需要添加对 'rs_unet3_plus' 的支持
    
    if model_type == 'unet':
        # 调用 U-Net 分割
        pass
    elif model_type == 'fcn':
        # 调用 FCN 分割
        pass
    elif model_type == 'rs_unet3_plus':
        # 调用 RS-Unet3+ 分割 ← 需要实现
        from service.infer_rs_unet3_plus import infer_rs_unet3_plus
        result_path = infer_rs_unet3_plus(image_path, weight_path)
        return {"code": 200, "result_url": result_path}
    else:
        return {"code": 400, "msg": f"不支持的模型类型: {model_type}"}
```

**需要添加的导入**：
```python
from service.infer_rs_unet3_plus import infer_rs_unet3_plus
```

---

## ✅ 第四步：更新 TrainController（训练）

**文件位置**：`octa_backend/controller/train_controller.py`

**更新 `upload_dataset` 方法**：

```python
@staticmethod
async def upload_dataset(
    file: UploadFile,
    model_arch: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int
):
    """处理模型训练"""
    
    # 1. 保存上传的文件
    dataset_path = await save_upload_file(file)
    
    # 2. 根据模型架构路由到相应的训练器
    if model_arch == 'unet':
        from service.train_service import train_model
        result = train_model(
            dataset_path=dataset_path,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            model_type='unet'
        )
    elif model_arch == 'rs_unet3_plus':
        from service.train_rs_unet3_plus import train_rs_unet3_plus
        result = train_rs_unet3_plus(
            dataset_path=dataset_path,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size
        )
    elif model_arch == 'fcn':
        from service.train_service import train_model
        result = train_model(
            dataset_path=dataset_path,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            model_type='fcn'
        )
    else:
        return {"code": 400, "msg": f"不支持的模型架构: {model_arch}"}
    
    # 3. 返回训练结果
    return {"code": 200, "data": result}
```

**需要添加的导入**：
```python
from service.train_rs_unet3_plus import train_rs_unet3_plus
```

---

## 🔍 验证步骤

### 步骤 1：代码审查
```bash
# 检查分割接口
grep -n "async def segment_octa" octa_backend/main.py
grep -n "rs_unet3_plus" octa_backend/main.py

# 检查训练接口  
grep -n "async def upload_dataset" octa_backend/main.py
grep -n "model_arch" octa_backend/main.py
```

### 步骤 2：启动后端服务
```bash
cd octa_backend
# 激活虚拟环境
..\octa_env\Scripts\activate  # Windows
# source ../octa_env/bin/activate  # Linux/Mac

# 启动服务
python main.py
# 或
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### 步骤 3：测试分割接口（RS-Unet3+）
```bash
# 使用 curl 测试
curl -X POST http://127.0.0.1:8000/segment-octa/ \
  -F "file=@test_octa_image.png" \
  -F "model_type=rs_unet3_plus" \
  -F "weight_path=./models/weights/rs_unet3_plus.pth"

# 或使用 Python requests
import requests
response = requests.post(
    'http://127.0.0.1:8000/segment-octa/',
    files={'file': open('test_image.png', 'rb')},
    data={
        'model_type': 'rs_unet3_plus',
        'weight_path': './models/weights/rs_unet3_plus.pth'
    }
)
print(response.json())
```

**期望返回**：
```json
{
  "code": 200,
  "result_url": "/results/image_seg.png",
  "msg": "分割成功"
}
```

### 步骤 4：测试训练接口（RS-Unet3+）
```bash
# 准备测试数据集（ZIP格式，包含 images/ 和 masks/ 目录）
# 然后运行测试

curl -X POST http://127.0.0.1:8000/train/upload-dataset \
  -F "file=@dataset.zip" \
  -F "model_arch=rs_unet3_plus" \
  -F "epochs=10" \
  -F "lr=0.0001" \
  -F "weight_decay=0.0001" \
  -F "batch_size=4"
```

**期望返回**：
```json
{
  "code": 200,
  "data": {
    "model_path": "/results/rs_unet3_plus_model.pth",
    "dice_score": 0.85,
    "iou_score": 0.75,
    "train_losses": [...],
    "val_losses": [...]
  }
}
```

---

## 🐛 常见问题和解决方案

### 问题 1：模块导入错误
```
ModuleNotFoundError: No module named 'service.train_rs_unet3_plus'
```

**解决**：
1. 检查 `octa_backend/service/train_rs_unet3_plus.py` 是否存在
2. 检查 `__init__.py` 文件是否存在于 service 目录
3. 确保导入路径正确

---

### 问题 2：模型加载失败
```
[ERROR] 模型权重加载失败: ...
```

**解决**：
1. 检查权重文件路径是否正确
2. 确保权重文件存在于 `./models/weights/` 目录
3. 验证权重文件格式（PyTorch .pth）

---

### 问题 3：CORS 跨域错误
```
Access to XMLHttpRequest ... has been blocked by CORS policy
```

**解决**：
1. 检查 `main.py` 的 CORS 配置
2. 确保前端 URL 在 `allow_origins` 列表中
3. 重启后端服务

```python
# main.py 中的 CORS 配置
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

### 问题 4：参数不被识别
```
422 Unprocessable Entity: ... extra fields not permitted
```

**解决**：
1. 检查表单参数名称是否与后端一致
2. 确保参数类型匹配（int vs float）
3. 检查 Pydantic 模型定义（如果使用）

---

## 📝 参考实现示例

### RS-Unet3+ 推理函数签名
```python
# octa_backend/service/infer_rs_unet3_plus.py

def infer_rs_unet3_plus(
    image_path: str,
    weight_path: Optional[str] = None,
    device: str = 'cpu'
) -> str:
    """
    RS-Unet3+ 图像分割推理
    
    Args:
        image_path: 输入图像路径
        weight_path: 模型权重文件路径
        device: 推理设备 ('cpu' 或 'cuda')
    
    Returns:
        分割结果图像路径
    """
    # 实现细节...
    return result_image_path
```

### RS-Unet3+ 训练函数签名
```python
# octa_backend/service/train_rs_unet3_plus.py

def train_rs_unet3_plus(
    dataset_path: str,
    epochs: int = 200,
    lr: float = 0.0001,
    weight_decay: float = 0.0001,
    batch_size: int = 4,
    device: str = 'cpu'
) -> dict:
    """
    RS-Unet3+ 模型训练
    
    Args:
        dataset_path: 训练数据集路径（ZIP文件）
        epochs: 训练轮数
        lr: 学习率
        weight_decay: 权重衰减
        batch_size: 批次大小
        device: 训练设备
    
    Returns:
        {
            'model_path': str,
            'dice_score': float,
            'iou_score': float,
            'train_losses': list,
            'val_losses': list,
            ...
        }
    """
    # 实现细节...
    return result_dict
```

---

## ✨ 最佳实践

### 1. 错误处理
```python
try:
    # 调用模型推理或训练
    result = infer_rs_unet3_plus(image_path, weight_path)
except FileNotFoundError:
    return {"code": 404, "msg": "权重文件不存在"}
except RuntimeError as e:
    return {"code": 500, "msg": f"推理失败: {str(e)}"}
```

### 2. 日志记录
```python
import logging
logger = logging.getLogger(__name__)

logger.info(f"开始 RS-Unet3+ 推理: {image_path}")
logger.debug(f"使用权重: {weight_path}")
logger.error(f"推理失败: {e}")
```

### 3. 输入验证
```python
def validate_model_arch(model_arch: str) -> bool:
    """验证模型架构"""
    valid_archs = ['unet', 'fcn', 'rs_unet3_plus']
    return model_arch in valid_archs

if not validate_model_arch(model_arch):
    return {"code": 400, "msg": f"无效的模型架构: {model_arch}"}
```

---

## 📚 相关文件清单

| 文件 | 说明 | 优先级 |
|-----|------|-------|
| `octa_backend/main.py` | API 路由定义 | 🔴 必须 |
| `octa_backend/controller/image_controller.py` | 分割业务逻辑 | 🔴 必须 |
| `octa_backend/controller/train_controller.py` | 训练业务逻辑 | 🔴 必须 |
| `octa_backend/service/infer_rs_unet3_plus.py` | RS-Unet3+ 推理 | 🟠 已有 |
| `octa_backend/service/train_rs_unet3_plus.py` | RS-Unet3+ 训练 | 🟠 已有 |
| `octa_backend/models/rs_unet3_plus.py` | RS-Unet3+ 模型定义 | 🟠 已有 |

---

## 🚀 完成清单

- [ ] 确认分割接口支持 'rs_unet3_plus' 参数
- [ ] 更新分割接口文档注释
- [ ] 添加 `model_arch` 参数到训练接口
- [ ] 添加 `weight_decay` 参数到训练接口
- [ ] 更新 ImageController 的 `segment_octa` 方法
- [ ] 更新 TrainController 的 `upload_dataset` 方法
- [ ] 导入 RS-Unet3+ 推理和训练模块
- [ ] 测试分割接口（RS-Unet3+）
- [ ] 测试训练接口（RS-Unet3+）
- [ ] 验证错误处理
- [ ] 验证日志记录
- [ ] 前后端集成测试

---

**文档版本**：v1.0  
**最后更新**：2026年1月17日  
**作者**：GitHub Copilot AI
