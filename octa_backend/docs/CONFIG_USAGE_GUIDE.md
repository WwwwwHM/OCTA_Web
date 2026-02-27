# OCTA平台配置文件使用指南

## 📋 概述

`config/config.py` 是OCTA血管分割平台的**全局配置中心**，集中管理所有模块的配置参数。

**设计原则：**
- ✅ **单一数据源**：所有配置统一管理，避免硬编码分散
- ✅ **解耦硬编码**：代码中不再出现魔法数字和硬编码路径
- ✅ **易于维护**：修改配置只需修改此文件
- ✅ **扩展友好**：预留扩展配置字段

---

## 🎯 核心配置分类

### 1. 模型参数配置

```python
# 模型输入输出通道数
U_NET_IN_CHANNELS = 1      # 灰度图像输入（单通道）
U_NET_OUT_CHANNELS = 1     # 二分类掩码输出（单通道）

# 图像尺寸
INPUT_SIZE = (256, 256)    # 统一尺寸（宽, 高）

# 二值化阈值
MASK_THRESHOLD = 0.5       # sigmoid输出>0.5为血管
```

**使用示例：**
```python
from config.config import U_NET_IN_CHANNELS, U_NET_OUT_CHANNELS, INPUT_SIZE

# 创建模型
model = UNetUnderfittingFix(
    in_channels=U_NET_IN_CHANNELS,
    out_channels=U_NET_OUT_CHANNELS
)

# 图像预处理
resized_img = img.resize(INPUT_SIZE)
```

---

### 2. 预处理参数配置

```python
# 归一化参数（与训练脚本一致，禁止修改）
MEAN = 0.5                 # 归一化均值
STD = 0.5                  # 归一化标准差
```

**使用示例：**
```python
from config.config import MEAN, STD

# 图像归一化
normalized = (pixel_array / 255.0 - MEAN) / STD
```

**⚠️ 重要提示：**
- MEAN和STD必须与本地训练脚本**完全一致**
- 任意修改会导致模型精度严重下降

---

### 3. 权重配置

```python
# 权重存储路径
WEIGHT_SAVE_PATH = "./weights"

# 支持的权重格式
SUPPORTED_WEIGHT_FORMATS = [".pth", ".pt"]
```

**使用示例：**
```python
from config.config import WEIGHT_SAVE_PATH, SUPPORTED_WEIGHT_FORMATS
from pathlib import Path

# 保存权重
weight_path = Path(WEIGHT_SAVE_PATH) / "my_model.pth"
torch.save(model.state_dict(), weight_path)

# 校验格式
if weight_path.suffix not in SUPPORTED_WEIGHT_FORMATS:
    raise ValueError("不支持的权重格式")
```

---

### 4. 错误码配置

```python
# HTTP状态码
SUCCESS_CODE = 200               # 成功
FORMAT_ERROR_CODE = 400          # 格式错误
WEIGHT_VALID_ERROR_CODE = 400    # 权重校验失败
MODEL_LOAD_ERROR_CODE = 500      # 模型加载失败
INFERENCE_ERROR_CODE = 500       # 推理失败
```

**使用示例：**
```python
from fastapi import HTTPException
from config.config import FORMAT_ERROR_CODE, MODEL_LOAD_ERROR_CODE

# 格式校验失败
if file.suffix not in SUPPORTED_FORMATS:
    raise HTTPException(
        status_code=FORMAT_ERROR_CODE,
        detail="文件格式不支持"
    )

# 模型加载失败
try:
    model = load_model(path)
except Exception as e:
    raise HTTPException(
        status_code=MODEL_LOAD_ERROR_CODE,
        detail=f"模型加载失败: {e}"
    )
```

---

### 5. 日志配置

```python
# 日志存储
LOG_SAVE_PATH = "./logs"

# 日志级别
LOG_LEVEL = "INFO"  # DEBUG/INFO/WARNING/ERROR/CRITICAL

# 日志格式
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

**使用示例：**
```python
import logging
from config.config import LOG_LEVEL, LOG_FORMAT, LOG_SAVE_PATH
from pathlib import Path

# 配置日志
Path(LOG_SAVE_PATH).mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(f"{LOG_SAVE_PATH}/app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("应用启动")
```

---

### 6. 设备配置

```python
# 设备优先级
DEVICE_PRIORITY = "cuda"   # 优先使用GPU
CPU_DEVICE = "cpu"         # CPU设备标识
```

**使用示例：**
```python
import torch
from config.config import DEVICE_PRIORITY, CPU_DEVICE

# 自动选择设备
def get_device():
    if DEVICE_PRIORITY == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device(CPU_DEVICE)

device = get_device()
model.to(device)
```

---

## 📦 模块导入示例

### weight_validator.py 使用配置

```python
from config.config import (
    SUPPORTED_WEIGHT_FORMATS,  # 权重格式校验
    WEIGHT_VALID_ERROR_CODE,   # 校验失败状态码
    U_NET_IN_CHANNELS,         # 模型参数
    U_NET_OUT_CHANNELS
)

def validate_weight_format(upload: UploadFile):
    suffix = Path(upload.filename).suffix
    if suffix not in SUPPORTED_WEIGHT_FORMATS:
        raise HTTPException(
            status_code=WEIGHT_VALID_ERROR_CODE,
            detail=f"权重格式不支持: {suffix}"
        )
```

### model_loader.py 使用配置

```python
from config.config import (
    DEVICE_PRIORITY,           # 设备优先级
    CPU_DEVICE,                # CPU设备
    MODEL_LOAD_ERROR_CODE,     # 加载失败状态码
    LOG_LEVEL                  # 日志级别
)

def load_model_by_weight_path(weight_path, model_cls):
    device = get_device()
    try:
        model = model_cls()
        checkpoint = torch.load(weight_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device).eval()
        return model, device
    except Exception as e:
        raise HTTPException(
            status_code=MODEL_LOAD_ERROR_CODE,
            detail=f"模型加载失败: {e}"
        )
```

### data_process.py 使用配置

```python
from config.config import (
    INPUT_SIZE,                # 输入尺寸
    MEAN,                      # 归一化均值
    STD,                       # 归一化标准差
    MASK_THRESHOLD             # 二值化阈值
)

def preprocess_image(upload_file: UploadFile):
    img = Image.open(upload_file.file).convert('L')
    img = img.resize(INPUT_SIZE)
    array = np.array(img, dtype=np.float32) / 255.0
    normalized = (array - MEAN) / STD
    return torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)

def postprocess_mask(logits, original_size):
    mask = torch.sigmoid(logits).squeeze().cpu().numpy()
    binary = (mask > MASK_THRESHOLD).astype(np.uint8) * 255
    resized = Image.fromarray(binary).resize(original_size, Image.NEAREST)
    return np.array(resized)
```

### seg_router.py 使用配置

```python
from config.config import (
    SUCCESS_CODE,              # 成功状态码
    INFERENCE_ERROR_CODE,      # 推理失败状态码
    LOG_FORMAT,                # 日志格式
    LOG_LEVEL                  # 日志级别
)

@router.post("/predict")
async def predict(image_file: UploadFile, weight_id: str):
    try:
        # ... 推理逻辑 ...
        return JSONResponse({
            "code": SUCCESS_CODE,
            "msg": "推理成功",
            "data": {"mask_base64": result}
        })
    except Exception as e:
        logger.error(f"推理失败: {e}")
        raise HTTPException(
            status_code=INFERENCE_ERROR_CODE,
            detail=f"推理失败: {e}"
        )
```

---

## 🔧 配置修改指南

### 修改日志级别（开发调试）

```python
# config.py
LOG_LEVEL = "DEBUG"  # 改为DEBUG查看详细日志
```

### 修改设备优先级（强制CPU）

```python
# config.py
DEVICE_PRIORITY = "cpu"  # 强制使用CPU
```

### 修改阈值（调优分割效果）

```python
# config.py
MASK_THRESHOLD = 0.3  # 降低阈值，提高灵敏度（更多血管）
MASK_THRESHOLD = 0.7  # 提高阈值，降低噪声（更少误检）
```

### 添加新的权重格式

```python
# config.py
SUPPORTED_WEIGHT_FORMATS = [".pth", ".pt", ".ckpt"]  # 添加.ckpt
```

---

## ⚠️ 注意事项

### 1. 禁止修改的参数

| 参数 | 原因 |
|-----|------|
| `MEAN`, `STD` | 必须与训练脚本一致，否则模型精度严重下降 |
| `U_NET_IN_CHANNELS`, `U_NET_OUT_CHANNELS` | 模型架构固定，修改会导致权重加载失败 |

### 2. 修改后必须重启服务

所有配置修改后需重启FastAPI服务：

```bash
# 停止服务（Ctrl+C）
# 重新启动
cd octa_backend
python main.py
```

### 3. 生产环境建议

```python
# 生产环境推荐配置
LOG_LEVEL = "WARNING"  # 减少日志量
RELOAD_MODE = False    # 关闭热重载
DEVICE_PRIORITY = "cuda"  # 使用GPU加速
```

---

## 📊 配置依赖关系

```
config.py (全局配置中心)
    ├── weight_validator.py
    │   ├── SUPPORTED_WEIGHT_FORMATS
    │   ├── WEIGHT_VALID_ERROR_CODE
    │   └── U_NET_IN_CHANNELS, U_NET_OUT_CHANNELS
    │
    ├── model_loader.py
    │   ├── DEVICE_PRIORITY
    │   ├── CPU_DEVICE
    │   └── MODEL_LOAD_ERROR_CODE
    │
    ├── data_process.py
    │   ├── INPUT_SIZE
    │   ├── MEAN, STD
    │   └── MASK_THRESHOLD
    │
    └── seg_router.py
        ├── SUCCESS_CODE
        ├── INFERENCE_ERROR_CODE
        └── LOG_FORMAT, LOG_LEVEL
```

---

## 🚀 快速开始

1. **导入配置**
   ```python
   from config.config import INPUT_SIZE, MEAN, STD
   ```

2. **使用配置**
   ```python
   img = img.resize(INPUT_SIZE)
   normalized = (array - MEAN) / STD
   ```

3. **修改配置**
   - 编辑 `config/config.py`
   - 重启服务
   - 验证生效

---

**最后更新**：2026-01-28  
**维护者**：OCTA Web项目组
