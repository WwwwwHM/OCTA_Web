# OCTA后端项目结构说明

## 📁 标准项目结构

```
octa_backend/
│
├── main.py                    # FastAPI主程序入口（必需）
│
├── models/                    # 模型目录
│   ├── __init__.py           # Python包初始化文件（必需）
│   ├── unet.py               # U-Net模型实现（必需）
│   └── weights/              # 模型权重文件目录（可选）
│       └── .gitkeep          # 保持目录在git中（可选）
│
├── uploads/                  # 上传文件保存目录（自动创建）
│   └── .gitkeep              # 保持目录在git中（可选）
│
├── results/                  # 分割结果保存目录（自动创建）
│   └── .gitkeep              # 保持目录在git中（可选）
│
├── check_backend.py          # 环境验证脚本（推荐）
│
├── requirements.txt          # Python依赖包列表（必需）
│
├── start_server.bat          # Windows启动脚本（可选）
├── start_server.sh            # Linux/Mac启动脚本（可选）
│
├── START_GUIDE.md            # 启动指南文档（推荐）
├── TROUBLESHOOTING.md         # 常见问题解决方案（推荐）
├── PROJECT_STRUCTURE.md       # 本文件（可选）
└── README.md                  # 项目说明文档（推荐）
```

## 📝 文件说明

### 核心文件

#### `main.py`
- **作用**：FastAPI应用主入口
- **必需**：是
- **说明**：包含所有API接口定义、CORS配置、错误处理等

#### `models/unet.py`
- **作用**：U-Net模型实现
- **必需**：是
- **说明**：包含模型定义、加载函数、分割函数等

#### `models/__init__.py`
- **作用**：使models成为Python包
- **必需**：是
- **说明**：导出主要函数和类，方便导入

### 目录说明

#### `models/weights/`
- **作用**：存放预训练模型权重文件
- **必需**：否（如果使用未训练模型）
- **文件格式**：`.pth` 或 `.pth.tar`
- **说明**：将预训练模型文件放在此目录，在`main.py`中指定路径

#### `uploads/`
- **作用**：保存用户上传的原始图像
- **创建方式**：自动创建（在`main.py`启动时）
- **说明**：上传的文件会以UUID命名保存，避免文件名冲突

#### `results/`
- **作用**：保存图像分割结果
- **创建方式**：自动创建（在`main.py`启动时）
- **说明**：分割结果以`原文件名_segmented.png`格式保存

### 辅助文件

#### `check_backend.py`
- **作用**：验证后端环境配置
- **功能**：
  - 检查虚拟环境
  - 验证依赖包
  - 检查目录结构
  - 测试API接口
- **使用方法**：`python check_backend.py`

#### `requirements.txt`
- **作用**：列出所有Python依赖包
- **安装方法**：`pip install -r requirements.txt`
- **必需包**：
  - fastapi：Web框架
  - uvicorn：ASGI服务器
  - pillow：图像处理
  - numpy：数值计算
  - torch：深度学习框架
  - torchvision：计算机视觉工具
  - python-multipart：文件上传支持
  - requests：API测试

#### `start_server.bat` / `start_server.sh`
- **作用**：一键启动后端服务
- **功能**：
  - 自动激活虚拟环境
  - 检查并安装依赖
  - 创建必要目录
  - 启动服务
- **使用方法**：
  - Windows: 双击 `start_server.bat`
  - Linux/Mac: `bash start_server.sh` 或 `chmod +x start_server.sh && ./start_server.sh`

### 文档文件

#### `START_GUIDE.md`
- **作用**：详细的启动指南
- **内容**：快速开始、启动步骤、Swagger测试方法等

#### `TROUBLESHOOTING.md`
- **作用**：常见问题解决方案
- **内容**：跨域、模型加载、文件路径等问题的解决方法

#### `README.md`
- **作用**：项目总体说明
- **内容**：项目介绍、API说明、使用示例等

## 🔧 目录创建说明

### 自动创建的目录

以下目录会在`main.py`启动时自动创建（如果不存在）：

```python
# 在main.py中（约第50行）
UPLOAD_DIR = Path("./uploads")
RESULTS_DIR = Path("./results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
```

### 手动创建的目录

以下目录需要手动创建（如果不存在）：

```bash
# 创建模型权重目录
mkdir models/weights

# 或者使用Python
python -c "from pathlib import Path; Path('models/weights').mkdir(parents=True, exist_ok=True)"
```

## ✅ 验证项目结构

运行验证脚本检查项目结构：

```bash
python check_backend.py
```

验证脚本会检查：
- ✅ `models/` 目录存在
- ✅ `models/weights/` 目录存在
- ✅ `uploads/` 目录存在（不存在会自动创建）
- ✅ `results/` 目录存在（不存在会自动创建）

## 📦 Git配置建议

如果使用Git，建议在`.gitignore`中添加：

```gitignore
# Python
__pycache__/
*.py[cod]
*.so
.Python
*.egg-info/
dist/
build/

# 虚拟环境
octa_env/
venv/
env/

# 上传和结果文件
uploads/*
!uploads/.gitkeep
results/*
!results/.gitkeep

# 模型权重（通常很大，不提交）
models/weights/*.pth
models/weights/*.pth.tar

# IDE
.vscode/
.idea/
*.swp
*.swo

# 日志
*.log
```

## 🎯 快速检查清单

在开始使用前，确保：

- [ ] `main.py` 文件存在
- [ ] `models/unet.py` 文件存在
- [ ] `models/__init__.py` 文件存在
- [ ] `requirements.txt` 文件存在
- [ ] 虚拟环境已激活
- [ ] 依赖包已安装（`pip install -r requirements.txt`）
- [ ] 运行 `python check_backend.py` 通过所有检查

---

**最后更新：2024年**
