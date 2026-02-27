# OCTA图像分割平台 - 后端API文档

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**OCTA（Optical Coherence Tomography Angiography）图像分割平台**的后端服务，基于FastAPI框架，提供AI驱动的医学影像分割、模型训练、文件管理等功能。

## ✨ 核心特性

- 🎯 **双模型支持**：U-Net（经典）+ RS-Unet3+（前沿），适配不同场景
- 🧠 **在线训练**：上传数据集即可训练自定义模型，支持损失曲线可视化
- 📁 **智能文件管理**：统一管理图像/数据集/模型权重，支持复用训练/测试
- 🔧 **配置化设计**：所有常量统一在 `config/config.py` 管理，易于维护
- 🏗️ **分层架构**：Controller → Service → DAO 清晰分层，代码可读性高
- 🌐 **跨域支持**：预配置CORS，前后端分离开发友好

## 🚀 快速开始

### 方式1：使用启动脚本（推荐）

**Windows:**
```bash
双击 start_server.bat
```

**Linux/Mac:**
```bash
chmod +x start_server.sh
./start_server.sh
```

### 方式2：手动启动

#### 1. 激活虚拟环境

**Windows:**
```powershell
..\octa_env\Scripts\activate
```

**Linux/Mac:**
```bash
source ../octa_env/bin/activate
```

#### 2. 安装依赖

```bash
pip install -r requirements.txt
```

#### 3. 验证环境（可选）

```bash
python check_backend.py
```

#### 4. 启动服务

```bash
# 方式1：直接运行（开发环境）
python main.py

# 方式2：使用uvicorn命令
uvicorn main:app --host 127.0.0.1 --port 8000 --reload

# 方式3：生产环境（多进程）
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

服务启动后，访问以下地址：
- **Swagger文档**：http://127.0.0.1:8000/docs
- **ReDoc文档**：http://127.0.0.1:8000/redoc
- **健康检查**：http://127.0.0.1:8000/

## 📚 详细文档

- **[启动指南](./START_GUIDE.md)** - 详细的启动步骤和Swagger测试方法
- **[常见问题](./TROUBLESHOOTING.md)** - 跨域、模型加载等问题的解决方案
- **[项目结构](./PROJECT_STRUCTURE.md)** - 项目文件结构说明
- **[配置管理](./config/config.py)** - 所有配置常量统一管理
- **[快速参考](./QUICK_REFERENCE.md)** - 常用命令和API速查
- **[数据库指南](./DATABASE_USAGE_GUIDE.md)** - SQLite数据库操作说明
- **[DAO完整指南](./DAO_COMPLETE_GUIDE.md)** - 数据访问层详细文档
- **[📖 毕设论文写作指南](./docs/THESIS_WRITING_GUIDE.md)** - 学位论文各章节写作规范和示例

## 🎯 核心功能模块

## 🎯 核心功能模块

### 1. 图像分割（Image Segmentation）

**端点**：`POST /segment-octa/`

上传 OCTA 图像，使用预训练模型进行血管分割，自动记录到文件管理表。

**请求参数：**
- `file`（必选）：图像文件（multipart/form-data），支持 PNG/JPG/JPEG
- `model_type`（可选）：模型类型，默认 `unet`（也支持 `rs_unet3_plus`）
- `weight_path`（可选）：自定义模型权重路径（默认使用预训练权重）

**响应示例：**
```json
{
  "success": true,
  "message": "图像分割完成",
  "original_filename": "input.png",
  "saved_filename": "octa_a1b2c3...7890.png",
  "result_filename": "octa_a1b2c3...7890_seg.png",
  "image_url": "/images/octa_a1b2c3...7890.png",
  "result_url": "/results/octa_a1b2c3...7890_seg.png",
  "model_type": "unet",
  "record_id": 12
}
```

### 2. 模型训练（Model Training）

**端点**：`POST /train/upload-dataset`

上传包含 `images/` 和 `masks/` 文件夹的数据集压缩包，触发模型训练。

**请求参数：**
- `file`（必选）：ZIP/RAR/7Z 压缩包
- `model_arch`（可选）：模型架构，`unet`（默认）或 `rs_unet3_plus`
- `epochs`（可选）：训练轮数，默认 300
- `lr`（可选）：学习率，默认 1e-4
- `batch_size`（可选）：批次大小，默认 4
- `weight_decay`（可选）：权重衰减，默认 0
- `attention_weight`（可选）：注意力权重，默认 0.8（RS-Unet3+ 专用）
- `deep_supervision`（可选）：深度监督，默认 true（RS-Unet3+ 专用）
- `loss_function`（可选）：损失函数，默认 `Lovasz-Softmax`（RS-Unet3+ 专用）

**响应示例：**
```json
{
  "success": true,
  "message": "模型训练完成",
  "model_path": "models/weights_unet/trained_unet_20260127_143022.pth",
  "loss_curve_path": "results/loss_curve_20260127_143022.png",
  "dataset_id": 5,
  "training_stats": {
    "final_loss": 0.0234,
    "epochs": 300,
    "best_epoch": 287
  }
}
```

### 3. 文件管理（File Management）

统一管理上传的图像、数据集、模型权重，支持列表查询、详情查看、删除、复用等操作。

**接口列表：**
- `GET /file/list?file_type=image|dataset|weight` - 文件列表（可按类型筛选）
- `GET /file/detail/{file_id}` - 文件详情
- `DELETE /file/delete/{file_id}` - 删除文件（数据库记录+本地文件）
- `POST /file/reuse/{file_id}?epochs=100&lr=0.0001` - 复用数据集重新训练
- `POST /file/test/{file_id}?weight_path=xxx.pth` - 复用图像测试分割

**响应格式：**
```json
{
  "code": 200,
  "msg": "操作成功",
  "data": {
    "files": [
      {
        "id": 1,
        "file_name": "dataset_20260127.zip",
        "file_type": "dataset",
        "file_path": "uploads/dataset_20260127.zip",
        "file_size": 5242880,
        "upload_time": "2026-01-27 14:30:22"
      }
    ]
  }
}
```

### 4. 模型权重管理（Model Weight Management）

**端点**：`GET /model/weights`

列出所有可用的模型权重文件，包括预训练权重和用户训练的权重。

**响应示例：**
```json
{
  "code": 200,
  "msg": "查询成功",
  "data": {
    "weights": [
      {
        "name": "unet_octa.pth",
        "path": "models/weights/unet_octa.pth",
        "size": 117.5,
        "size_unit": "MB",
        "modified_time": "2026-01-20 10:30:00"
      },
      {
        "name": "trained_unet_20260127_143022.pth",
        "path": "models/weights_unet/trained_unet_20260127_143022.pth",
        "size": 115.2,
        "size_unit": "MB",
        "modified_time": "2026-01-27 14:35:15"
      }
    ],
    "total": 2
  }
}
```

### 5. 历史记录（History Records）

**端点**：
- `GET /history/` - 获取所有分割历史记录
- `GET /history/{record_id}` - 获取单条记录详情
- `DELETE /history/{record_id}` - 删除历史记录

### 6. 静态文件访问（Static Files）

- `GET /images/{filename}` - 获取上传的原始图像
- `GET /results/{filename}` - 获取分割结果图像

## 🏗️ 项目架构

### 分层设计（MVC + Service）

```
octa_backend/
├── main.py                      # FastAPI应用入口，路由注册
├── config/
│   └── config.py                # 统一配置管理（数据库、文件、模型、训练）
├── controller/                  # 控制层（HTTP接口）
│   ├── image_controller.py      # 图像分割接口
│   ├── train_controller.py      # 模型训练接口
│   ├── file_controller.py       # 文件管理接口
│   └── model_controller.py      # 模型权重管理接口
├── service/                     # 业务逻辑层
│   ├── model_service.py         # 模型加载/推理服务
│   ├── train_service.py         # 训练流程编排服务
│   └── weight_service.py        # 权重文件管理服务
├── dao/                         # 数据访问层
│   ├── image_dao.py             # 图像历史记录 DAO
│   └── file_dao.py              # 文件管理表 DAO
├── models/                      # 模型定义
│   ├── unet.py                  # U-Net 模型（经典架构）
│   ├── rs_unet3_plus.py         # RS-Unet3+ 模型（前沿架构）
│   ├── losses.py                # 损失函数（Dice、Lovasz、Focal等）
│   ├── dataset_underfitting_fix.py  # 数据增强策略
│   ├── weights/                 # 预训练权重
│   ├── weights_unet/            # U-Net训练权重（隔离存储）
│   └── weights_rs_unet3_plus/   # RS-Unet3+训练权重（隔离存储）
├── utils/                       # 工具类
│   └── file_utils.py            # 文件操作工具
├── uploads/                     # 上传文件存储（自动创建）
├── results/                     # 分割结果存储（自动创建）
├── octa.db                      # SQLite数据库（自动创建）
└── requirements.txt             # Python依赖包
```

### 架构优势

1. **配置化**：所有常量（路径、大小限制、超参数）统一在 `config.py` 管理
2. **分层清晰**：Controller（路由）→ Service（业务）→ DAO（数据）→ Model（模型）
3. **易于扩展**：新增模型/功能只需添加对应模块，不影响现有代码
4. **代码复用**：Service 层可被多个 Controller 调用
5. **易于测试**：每层独立，单元测试简单

## ⚙️ 配置管理

所有配置常量统一在 `config/config.py` 管理，便于维护和扩展。

### 核心配置项

#### 数据库配置
```python
DB_PATH = "./octa.db"              # SQLite数据库路径
DB_TABLE_NAME = "images"           # 图像历史记录表名
```

#### 文件存储配置
```python
UPLOAD_DIR = "./uploads"           # 上传文件目录
RESULT_DIR = "./results"           # 分割结果目录
MAX_FILE_SIZE = 10 * 1024 * 1024   # 最大文件10MB
ALLOWED_FORMATS = ["png", "jpg", "jpeg"]  # 允许的图像格式
```

#### 模型配置
```python
MODEL_DIR = "./models"             # 模型代码目录
UNET_WEIGHT_PATH = "./models/weights/unet_octa.pth"  # U-Net预训练权重
# 权重隔离存储
WEIGHTS_UNET_DIR = "./models/weights_unet"          # U-Net训练权重
WEIGHTS_RS_UNET3_PLUS_DIR = "./models/weights_rs_unet3_plus"  # RS-Unet3+训练权重
```

#### 训练配置
```python
DEFAULT_EPOCHS = 300               # 默认训练轮数
DEFAULT_LEARNING_RATE = 1e-4       # 默认学习率
DEFAULT_BATCH_SIZE = 4             # 默认批次大小
```

#### CORS配置
```python
CORS_ORIGINS = [
    "http://127.0.0.1:5173",       # 前端开发地址
    "http://localhost:5173"
]
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = ["*"]
CORS_ALLOW_HEADERS = ["*"]
```

#### 服务器配置
```python
SERVER_HOST = "127.0.0.1"          # 服务器地址
SERVER_PORT = 8000                 # 服务器端口
RELOAD_MODE = True                 # 热重载（开发环境）
```

### 修改配置

修改配置只需编辑 `config/config.py`，无需修改业务代码：

```python
# 示例：增大文件上传限制
MAX_FILE_SIZE = 50 * 1024 * 1024  # 改为50MB

# 示例：添加新的前端地址
CORS_ORIGINS = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://192.168.1.100:5173"  # 新增局域网地址
]
```

## 🗄️ 数据库设计

### images 表（图像历史记录）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键，自增 |
| filename | TEXT | 原始文件名 |
| upload_time | TEXT | 上传时间（ISO格式） |
| model_type | TEXT | 使用的模型类型（unet/rs_unet3_plus） |
| original_path | TEXT | 原图保存路径 |
| result_path | TEXT | 分割结果路径 |

### file_management 表（文件管理）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键，自增 |
| file_name | TEXT | 文件名 |
| file_type | TEXT | 文件类型（image/dataset/weight） |
| file_path | TEXT | 文件路径 |
| file_size | INTEGER | 文件大小（字节） |
| upload_time | TEXT | 上传时间（ISO格式） |
| model_type | TEXT | 模型类型（用于dataset/weight类型） |
| extra_info | TEXT | 扩展信息（JSON格式） |

## 📋 注意事项

### 1. 文件格式要求

**图像文件**：
- 支持格式：PNG、JPG、JPEG
- 推荐尺寸：256×256 或 512×512
- 最大大小：10MB
- 注意：JPEG 格式会自动转换为 RGB，透明通道会丢失

**数据集压缩包**：
- 支持格式：ZIP、RAR、7Z
- 最大大小：50MB
- 必须包含：`images/` 和 `masks/` 两个文件夹
- 文件命名：images 和 masks 中的文件名必须一一对应
- 示例结构：
  ```
  dataset.zip
  ├── images/
  │   ├── img001.png
  │   ├── img002.png
  │   └── ...
  └── masks/
      ├── img001.png
      ├── img002.png
      └── ...
  ```

### 2. 模型选择建议

| 模型 | 优势 | 适用场景 | 训练时间 |
|------|------|----------|---------|
| **U-Net** | 经典稳定、训练快 | 通用场景、快速验证 | 短 |
| **RS-Unet3+** | 精度高、细节好 | 高质量分割、细小血管 | 长 |

### 3. 训练超参数建议

- **epochs**：300（充分学习）
- **learning_rate**：1e-4（稳定收敛）
- **batch_size**：4（CPU友好）或 8（GPU推荐）
- **weight_decay**：0（避免梯度消失）

### 4. 设备要求

- **CPU模式**：默认启用，无需GPU
- **内存要求**：至少 4GB RAM
- **存储空间**：预留 1GB 用于模型权重和数据集

### 5. 文件管理机制

- 所有上传文件自动记录到 `file_management` 表
- 删除操作会同时删除数据库记录和本地文件
- UUID 命名避免文件冲突
- 权重文件按模型架构隔离存储（`weights_unet/` 和 `weights_rs_unet3_plus/`）

## 🚨 错误处理

所有接口都包含完善的错误处理机制，返回标准化的错误信息：

### HTTP状态码

| 状态码 | 说明 | 常见原因 |
|--------|------|---------|
| 200 | 成功 | 请求正常处理 |
| 400 | 请求错误 | 文件格式错误、参数缺失 |
| 404 | 未找到 | 文件不存在、记录不存在 |
| 413 | 文件过大 | 超过文件大小限制 |
| 500 | 服务器错误 | 内部错误、模型加载失败 |

### 错误响应格式

**标准错误响应（Controller）：**
```json
{
  "success": false,
  "message": "文件格式不正确，仅支持 PNG/JPG/JPEG",
  "error": "详细的错误堆栈信息（开发模式）"
}
```

**统一错误响应（File/Model API）：**
```json
{
  "code": 400,
  "msg": "文件不存在",
  "data": null
}
```

### 常见错误及解决方案

#### 1. CORS 跨域错误
```
Access to XMLHttpRequest has been blocked by CORS policy
```
**解决**：检查 `config/config.py` 中的 `CORS_ORIGINS` 是否包含前端地址。

#### 2. 模型加载失败
```
[WARNING] 权重文件不存在: models/weights/unet_octa.pth
```
**解决**：下载预训练权重放到指定路径，或使用随机初始化模型（仅测试用）。

#### 3. 数据集格式错误
```
解压后未找到 images/ 或 masks/ 文件夹
```
**解决**：确保压缩包根目录包含 `images/` 和 `masks/` 两个文件夹。

#### 4. 端口被占用
```
ERROR: [Errno 48] Address already in use
```
**解决**：修改 `config/config.py` 中的 `SERVER_PORT`，或关闭占用端口的进程。

详细排错指南：[TROUBLESHOOTING.md](./TROUBLESHOOTING.md)

## 🧪 测试指南

### 1. 验证后端环境

```bash
# 检查依赖安装、数据库连接、目录权限
python check_backend.py
```

### 2. Swagger 交互式测试

访问 http://127.0.0.1:8000/docs，可以：
- 查看所有 API 接口文档
- 在线测试接口（无需前端）
- 查看请求/响应示例

### 3. 单元测试（可选）

```bash
# 测试数据库 DAO 层
python test_file_dao.py

# 测试数据增强
python test_data_pipeline.py

# 测试损失函数
python test_loss_function.py

# 测试训练循环
python test_training_loop.py
```

## 📦 依赖包说明

```txt
# Web 框架
fastapi>=0.104.0          # 现代化的Python Web框架
uvicorn[standard]>=0.24.0 # ASGI服务器

# 图像处理
Pillow>=10.0.0            # Python图像库
numpy>=1.24.0             # 数值计算库
albumentations>=1.3.0     # 强数据增强库

# 深度学习
torch>=2.0.0              # PyTorch深度学习框架
torchvision>=0.15.0       # PyTorch视觉库

# 压缩包支持
py7zr>=0.21.0             # 7Z格式支持
rarfile>=4.2              # RAR格式支持

# 工具
python-multipart>=0.0.6   # 文件上传支持
requests>=2.31.0          # HTTP请求库
```

## 🔄 版本历史

### v1.2.0（2026-01-27）
- ✨ 新增 RS-Unet3+ 模型支持
- ✨ 新增模型权重管理接口
- ✨ 新增配置统一管理（config.py）
- 🔧 优化训练流程（欠拟合修复、损失函数优化）
- 🔧 权重文件按模型架构隔离存储
- 📝 更新完整项目文档

### v1.1.0（2026-01-15）
- ✨ 新增在线模型训练功能
- ✨ 新增文件管理系统
- ✨ 新增数据集复用训练
- 🔧 优化分层架构（Controller/Service/DAO）
- 📝 新增数据库使用指南

### v1.0.0（2026-01-12）
- 🎉 初始版本发布
- ✅ U-Net 图像分割功能
- ✅ FastAPI 后端框架
- ✅ SQLite 数据库集成
- ✅ CORS 跨域支持

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

**开发流程：**
1. Fork 本仓库
2. 创建特性分支（`git checkout -b feature/AmazingFeature`）
3. 提交更改（`git commit -m 'Add some AmazingFeature'`）
4. 推送到分支（`git push origin feature/AmazingFeature`）
5. 提交 Pull Request

**代码规范：**
- 遵循 PEP 8 Python 代码规范
- 使用有意义的变量和函数名
- 添加必要的注释和文档字符串
- 更新相关文档

## 📞 联系方式

- 📧 Email: octa-dev@example.com
- 📝 Issue Tracker: [GitHub Issues](https://github.com/your-repo/issues)
- 📖 文档中心: [Documentation](./DOCUMENTATION_INDEX.md)

## 📄 许可证

本项目采用 MIT 许可证，详见 [LICENSE](../LICENSE) 文件。

---

**最后更新**：2026年1月27日  
**维护团队**：OCTA Web 项目组
