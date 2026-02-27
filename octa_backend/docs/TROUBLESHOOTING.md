# OCTA后端常见问题解决方案

本文档提供了OCTA后端服务常见问题的解决方案，适合小白用户参考。

## 🔴 问题1：跨域报错（CORS Error）

### 错误信息
```
Access to XMLHttpRequest at 'http://127.0.0.1:8000/segment-octa/' 
from origin 'http://127.0.0.1:5173' has been blocked by CORS policy
```

### 原因
前端和后端运行在不同的端口，浏览器阻止了跨域请求。

### 解决方案

#### 方案1：检查后端CORS配置（推荐）

1. 打开 `main.py` 文件
2. 找到CORS配置部分（约第39-48行）
3. 确保包含前端地址：

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",  # 确保这个地址正确
        "http://localhost:5173",  # 备用地址
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

4. 如果前端运行在其他端口，添加对应地址：
```python
allow_origins=[
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:3000",  # 如果前端在3000端口
    # 添加其他需要的地址
],
```

5. 重启后端服务

#### 方案2：临时允许所有来源（仅用于开发）

⚠️ **警告：仅用于本地开发，不要在生产环境使用！**

```python
allow_origins=["*"],  # 允许所有来源
```

### 验证
重启后端后，前端应该可以正常发送请求。

---

## 🔴 问题2：模型加载失败

### 错误信息
```
[ERROR] 模型加载过程中发生错误: ...
[WARNING] 图像分割失败，模型可能未训练或加载失败
```

### 原因
1. 模型权重文件不存在
2. 模型权重文件格式不正确
3. PyTorch版本不兼容

### 解决方案

#### 方案1：使用未训练模型（默认行为）

当前代码已经处理了这种情况：
- 如果模型权重不存在，会使用随机初始化的模型
- 分割结果可能不准确，但不会报错
- 这是**正常现象**，用于测试接口功能

#### 方案2：提供预训练模型权重

1. 将预训练模型文件（`.pth` 或 `.pth.tar`）放到 `models/weights/` 目录
2. 修改 `main.py` 中的模型路径：

```python
# 在 segment_octa 函数中（约第200行）
actual_result_path = segment_octa_image(
    image_path=str(upload_path),
    model_type=model_type,
    model_path='models/weights/unet_best.pth',  # 修改这里
    output_path=str(result_path),
    device='cpu'
)
```

3. 确保模型文件格式正确：
   - 文件扩展名：`.pth` 或 `.pth.tar`
   - 文件内容：PyTorch保存的state_dict

#### 方案3：检查PyTorch安装

```bash
# 检查PyTorch版本
python -c "import torch; print(torch.__version__)"

# 如果未安装或版本不对，重新安装
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 验证
运行验证脚本：
```bash
python check_backend.py
```

---

## 🔴 问题3：文件路径错误

### 错误信息
```
[ERROR] 输入图像不存在: ...
[ERROR] 文件路径错误
FileNotFoundError: ...
```

### 原因
1. 相对路径问题
2. 目录不存在
3. 文件权限问题

### 解决方案

#### 方案1：确保目录存在

后端启动时会自动创建 `uploads/` 和 `results/` 目录。如果不存在：

```bash
# 手动创建目录
mkdir uploads
mkdir results
mkdir models/weights
```

#### 方案2：使用绝对路径

如果相对路径有问题，可以修改 `main.py` 使用绝对路径：

```python
# 在文件开头（约第50行）
import os
BASE_DIR = Path(__file__).parent.absolute()  # 使用绝对路径
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
```

#### 方案3：检查文件权限

确保Python有读写权限：

**Windows:**
- 右键点击 `octa_backend` 文件夹
- 属性 → 安全 → 编辑
- 确保当前用户有完全控制权限

**Linux/Mac:**
```bash
chmod -R 755 octa_backend/
```

### 验证
运行验证脚本检查目录：
```bash
python check_backend.py
```

---

## 🔴 问题4：依赖包未安装

### 错误信息
```
ModuleNotFoundError: No module named 'fastapi'
ImportError: cannot import name '...'
```

### 解决方案

#### 步骤1：确认虚拟环境已激活

```bash
# Windows
..\octa_env\Scripts\activate

# Linux/Mac
source ../octa_env/bin/activate
```

激活后提示符前应该有 `(octa_env)`。

#### 步骤2：安装依赖

```bash
pip install -r requirements.txt
```

#### 步骤3：如果安装失败

**PyTorch安装问题（常见）：**

```bash
# CPU版本（推荐，适配无GPU环境）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 或者使用conda
conda install pytorch torchvision cpuonly -c pytorch
```

**其他包安装失败：**

```bash
# 升级pip
python -m pip install --upgrade pip

# 逐个安装
pip install fastapi
pip install uvicorn
pip install pillow
pip install numpy
```

### 验证
```bash
python check_backend.py
```

---

## 🔴 问题5：端口被占用

### 错误信息
```
ERROR:    [Errno 48] Address already in use
OSError: [WinError 10048] 通常每个套接字地址只允许使用一次
```

### 解决方案

#### 方案1：更改端口

修改启动命令：
```bash
uvicorn main:app --host 127.0.0.1 --port 8001  # 使用8001端口
```

同时需要修改前端请求地址。

#### 方案2：关闭占用端口的程序

**Windows:**
```bash
# 查找占用8000端口的进程
netstat -ano | findstr :8000

# 结束进程（替换PID为实际进程ID）
taskkill /PID <PID> /F
```

**Linux/Mac:**
```bash
# 查找占用8000端口的进程
lsof -i :8000

# 结束进程
kill -9 <PID>
```

### 验证
```bash
# 测试端口是否可用
curl http://127.0.0.1:8000/
```

---

## 🔴 问题6：文件上传失败

### 错误信息
```
400 Bad Request: 文件格式不正确
413 Request Entity Too Large: 文件太大
```

### 解决方案

#### 文件格式问题
- 确保上传的是PNG格式文件
- 检查文件扩展名是否为 `.png`
- 检查文件Content-Type是否为 `image/png`

#### 文件大小问题
- 建议文件大小不超过10MB
- 如果必须上传大文件，可以修改FastAPI配置：

```python
# 在main.py中（约第30行）
from fastapi import FastAPI
app = FastAPI(
    title="OCTA图像分割API",
    # 添加文件大小限制（50MB）
    max_request_size=50 * 1024 * 1024,
)
```

---

## 🔴 问题7：后端服务无法启动

### 错误信息
```
ImportError: cannot import name 'segment_octa_image' from 'models.unet'
```

### 解决方案

#### 检查导入路径

确保在 `octa_backend` 目录下运行：
```bash
cd octa_backend
python main.py
```

#### 检查模块结构

确保 `models/` 目录下有 `__init__.py` 文件：
```bash
# 如果不存在，创建它
touch models/__init__.py
```

#### 检查Python路径

```python
# 在main.py开头添加（用于调试）
import sys
print("Python路径:", sys.path)
print("当前工作目录:", os.getcwd())
```

---

## 📞 获取更多帮助

如果以上方案都无法解决问题：

1. **查看日志**：检查后端服务的完整错误日志
2. **运行验证脚本**：`python check_backend.py`
3. **检查环境**：确认Python版本、虚拟环境、依赖包版本
4. **查看文档**：参考 `README.md` 和 `START_GUIDE.md`

## 💡 调试技巧

### 启用详细日志

```python
# 在main.py中
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 测试单个组件

```python
# 测试模型加载
python -c "from models.unet import load_unet_model; model = load_unet_model('unet', None, 'cpu'); print('模型加载成功' if model else '模型加载失败')"
```

### 检查网络连接

```bash
# 测试后端是否可访问
curl http://127.0.0.1:8000/
```

---

**最后更新：2024年**
