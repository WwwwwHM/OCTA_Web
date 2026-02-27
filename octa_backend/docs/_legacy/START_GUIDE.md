# OCTA后端服务启动指南

## 📋 目录结构

```
octa_backend/
├── main.py                 # FastAPI主程序入口
├── check_backend.py        # 环境验证脚本
├── requirements.txt        # 依赖包列表
├── START_GUIDE.md          # 本启动指南
├── TROUBLESHOOTING.md      # 常见问题解决方案
├── models/                 # 模型目录
│   ├── __init__.py
│   ├── unet.py            # U-Net模型实现
│   └── weights/           # 模型权重文件目录（可选）
├── uploads/               # 上传文件保存目录（自动创建）
└── results/               # 分割结果保存目录（自动创建）
```

## 🚀 快速开始

### 步骤1：激活虚拟环境

**Windows:**
```bash
# 如果虚拟环境在项目根目录
..\octa_env\Scripts\activate

# 或者使用完整路径
D:\Code\OCTA_Web\octa_env\Scripts\activate
```

**Linux/Mac:**
```bash
source ../octa_env/bin/activate
```

激活成功后，命令行提示符前会显示 `(octa_env)`。

### 步骤2：安装依赖

```bash
pip install -r requirements.txt
```

### 步骤3：验证环境

运行验证脚本检查环境配置：

```bash
python check_backend.py
```

验证脚本会检查：
- ✅ 虚拟环境是否激活
- ✅ 依赖包是否安装
- ✅ 目录结构是否正确
- ✅ API接口是否正常（需要先启动服务）

### 步骤4：启动后端服务

**方式1：直接运行（推荐开发环境）**
```bash
python main.py
```

**方式2：使用uvicorn命令（推荐生产环境）**
```bash
# 开发模式（自动重载）
uvicorn main:app --host 127.0.0.1 --port 8000 --reload

# 生产模式
uvicorn main:app --host 0.0.0.0 --port 8000
```

启动成功后，你会看到类似输出：
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

## 📖 使用Swagger文档测试接口

### 访问Swagger UI

1. 启动后端服务后，在浏览器中打开：
   ```
   http://127.0.0.1:8000/docs
   ```

2. 你会看到交互式API文档界面，包含所有可用接口。

### 测试接口步骤

#### 1. 测试健康检查接口（GET /）

- 在Swagger UI中找到 `GET /` 接口
- 点击 "Try it out" 按钮
- 点击 "Execute" 执行请求
- 查看响应结果，应该返回服务状态信息

#### 2. 测试图像分割接口（POST /segment-octa/）

1. 在Swagger UI中找到 `POST /segment-octa/` 接口
2. 点击 "Try it out" 按钮
3. 在 `file` 参数中：
   - 点击 "Choose File" 按钮
   - 选择一个PNG格式的图像文件
4. 在 `model_type` 参数中：
   - 输入 `unet` 或 `fcn`（默认值：unet）
5. 点击 "Execute" 执行请求
6. 等待处理完成（可能需要几秒到几十秒）
7. 查看响应结果：
   - `success: true` 表示分割成功
   - `result_url` 是结果图像的访问路径
   - 如果 `success: false`，可能是模型未训练（这是正常的）

#### 3. 查看上传的原图（GET /images/{filename}）

1. 在Swagger UI中找到 `GET /images/{filename}` 接口
2. 点击 "Try it out"
3. 在 `filename` 参数中输入之前上传接口返回的 `saved_filename`
4. 点击 "Execute"
5. 浏览器会直接显示或下载图像

#### 4. 查看分割结果（GET /results/{filename}）

1. 在Swagger UI中找到 `GET /results/{filename}` 接口
2. 点击 "Try it out"
3. 在 `filename` 参数中输入之前分割接口返回的 `result_filename`
4. 点击 "Execute"
5. 浏览器会显示分割结果图像

## 🔧 常用命令

### 检查服务状态
```bash
# 测试健康检查接口
curl http://127.0.0.1:8000/
```

### 查看API文档
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

### 停止服务
在运行服务的终端中按 `Ctrl + C`

## ⚠️ 注意事项

1. **端口占用**：确保8000端口未被其他程序占用
2. **文件大小**：建议上传的图像文件不超过10MB
3. **模型权重**：当前使用未训练的模型，分割结果仅供参考
4. **跨域配置**：已配置允许 `http://127.0.0.1:5173` 的请求

## 🐛 遇到问题？

请查看 [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) 文件获取常见问题的解决方案。
