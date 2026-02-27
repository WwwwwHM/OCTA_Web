# OCTA血管分割平台 - 完整联调测试指南

## 📋 概述

本文档提供OCTA后端API的**完整联调测试流程**，适用于毕设答辩演示和功能验证。

---

## 🎯 测试目标

✅ 验证所有核心模块集成正确：
- `weight_validator.py` - 权重校验
- `model_loader.py` - 模型加载
- `data_process.py` - 数据处理
- `seg_router.py` - 分割预测API
- `config.py` - 全局配置

✅ 验证完整业务流程：
- 权重上传 → 权重校验 → 模型加载 → 图像预处理 → 推理 → 后处理 → 返回结果

✅ 验证API接口规范：
- 请求参数格式
- 响应数据格式
- 错误处理机制

---

## 🚀 快速开始（5分钟）

### 步骤1：启动后端服务

```bash
# 终端1：启动后端
cd octa_backend
python main.py
```

**预期输出：**
```
======================================================================
                     OCTA图像分割后端启动中...
======================================================================
[INFO] 配置来源: config/config.py
[INFO] 服务地址: 127.0.0.1:8000
[INFO] 热重载模式: 已禁用(生产)
[INFO] CORS允许源: 2 个前端地址
======================================================================
[SUCCESS] ✓ 文件管理表已就绪
[SUCCESS] ✓ 定时清理任务已启动
======================================================================

======================================================================
                         🚀 服务启动参数
======================================================================
  监听地址: 0.0.0.0:8000
  访问地址: http://127.0.0.1:8000 或 http://localhost:8000
  API文档: http://127.0.0.1:8000/docs (Swagger UI)
  热重载: 已禁用（稳定模式）
  日志级别: INFO
======================================================================

INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### 步骤2：准备测试数据

在 `octa_backend/` 目录下创建测试数据目录：

```bash
# 创建测试数据目录
mkdir test_data

# 准备测试文件（请提供真实的OCTA图像）
# test_data/test_image.png - 测试图像（256x256灰度PNG）
# models/weights/unet_octa.pth - 测试权重（可选）
```

**⚠️ 重要：** 请将您的测试图像放入 `test_data/test_image.png`

### 步骤3：运行测试脚本

```bash
# 终端2：运行测试
cd octa_backend
python test_seg_api.py
```

**预期输出：**
```
================================================================================
                    OCTA血管分割API完整联调测试
================================================================================

📋 测试环境信息
  后端地址: http://127.0.0.1:8000
  测试权重: ./models/weights/unet_octa.pth
  测试图像: ./test_data/test_image.png
  输出目录: test_results

================================================================================
                            [步骤 0] 后端服务健康检查
--------------------------------------------------------------------------------
✓ 服务状态: ok
✓ 服务信息: OCTA后端服务运行中

================================================================================
                      [步骤 1] 上传权重文件（可选，也可使用官方权重）
--------------------------------------------------------------------------------
ℹ️  跳过权重上传，使用官方预置权重

================================================================================
                          [步骤 2] 调用分割预测接口
--------------------------------------------------------------------------------
🔍 正在调用分割预测接口
  图像路径: ./test_data/test_image.png
  权重ID: official（官方权重）
✓ 预测成功
  总耗时: 1.23秒（含网络传输）
  服务器推理耗时: 0.8754秒
  推理设备: cuda
  使用权重: official

================================================================================
                        [步骤 3] 解码Base64掩码并保存
--------------------------------------------------------------------------------
💾 正在解码并保存掩码
  Base64长度: 87654 字符
  输出路径: test_results/api_result_mask.png
✓ 掩码保存成功
  图像尺寸: (256, 256)
  数组形状: (256, 256)
  数据类型: uint8
  值范围: [0, 255]

================================================================================
                      [步骤 4] 与本地推理结果对比（可选）
--------------------------------------------------------------------------------
⚖️  对比API结果与本地推理（可选）
  正在执行本地推理...
✓ 本地推理完成
  差异像素比例: 0.02%
  ✓ API结果与本地完全一致（差异<0.1%）

================================================================================
                         [步骤 5] 生成可视化对比图
--------------------------------------------------------------------------------
📊 生成可视化对比图
✓ 可视化对比图已保存: test_results/comparison.png

================================================================================
                             ✅ 联调测试完成
================================================================================

📊 测试结果总结:
  ✓ 健康检查: 通过
  ✓ 权重管理: 使用官方权重
  ✓ 分割预测: 成功
  ✓ 掩码解码: 成功
  ✓ 推理设备: cuda
  ✓ 推理耗时: 0.8754秒

📁 输出文件:
  - 分割掩码: test_results/api_result_mask.png
  - 对比图: test_results/comparison.png
  - 本地结果: test_results/local_result.png

💡 后续步骤:
  1. 查看输出文件验证分割效果
  2. 检查日志文件查看详细执行信息
  3. 使用浏览器访问 http://127.0.0.1:8000/docs 查看API文档
  4. 集成到前端进行端到端测试
```

---

## 📚 详细测试步骤

### 测试1：健康检查（Health Check）

**目的：** 确认后端服务正常运行

**接口：** `GET /`

**测试代码：**
```python
import requests

response = requests.get("http://127.0.0.1:8000/")
print(response.json())
# 输出: {"status": "ok", "message": "OCTA后端服务运行中"}
```

**验证点：**
- ✅ 返回状态码 200
- ✅ 响应包含 `status: "ok"`

---

### 测试2：权重上传（可选）

**目的：** 测试权重文件上传和校验

**接口：** `POST /api/v1/weight/upload`

**请求参数：**
- `file`: 权重文件（.pth/.pt格式）
- `description`: 权重描述（可选）

**测试代码：**
```python
import requests

with open("./models/weights/unet_octa.pth", "rb") as f:
    files = {"file": ("unet_octa.pth", f, "application/octet-stream")}
    data = {"description": "测试权重"}
    
    response = requests.post(
        "http://127.0.0.1:8000/api/v1/weight/upload",
        files=files,
        data=data
    )
    
    result = response.json()
    weight_id = result["data"]["weight_id"]
    print(f"权重ID: {weight_id}")
```

**响应格式：**
```json
{
  "code": 200,
  "msg": "权重上传成功",
  "data": {
    "weight_id": "abc123",
    "filename": "unet_octa.pth",
    "size": 45678901,
    "upload_time": "2026-01-28 14:30:45"
  }
}
```

**验证点：**
- ✅ 返回状态码 200
- ✅ 响应包含 `weight_id`
- ✅ 权重文件被正确保存到 `./static/uploads/weight/{weight_id}/`

---

### 测试3：分割预测（核心功能）

**目的：** 测试完整的分割推理流程

**接口：** `POST /api/v1/seg/predict`

**请求参数：**
- `image_file`: OCTA图像文件（PNG/JPG/BMP/TIFF格式）
- `weight_id`: 权重ID（可选，留空使用官方权重）

**测试代码：**
```python
import requests
import base64
from PIL import Image
from io import BytesIO

# 上传图像
with open("./test_data/test_image.png", "rb") as f:
    files = {"image_file": ("test.png", f, "image/png")}
    data = {"weight_id": "official"}  # 或使用上传的weight_id
    
    response = requests.post(
        "http://127.0.0.1:8000/api/v1/seg/predict",
        files=files,
        data=data
    )
    
    result = response.json()
    
    # 解码mask_base64
    mask_base64 = result["data"]["mask_base64"]
    mask_bytes = base64.b64decode(mask_base64)
    mask_image = Image.open(BytesIO(mask_bytes))
    
    # 保存结果
    mask_image.save("result_mask.png")
    print(f"推理耗时: {result['data']['infer_time']}秒")
    print(f"推理设备: {result['data']['device']}")
```

**响应格式：**
```json
{
  "code": 200,
  "msg": "推理成功",
  "data": {
    "mask_base64": "iVBORw0KGgoAAAANSUhEU...",
    "device": "cuda",
    "infer_time": 0.8754,
    "weight_id": "official"
  }
}
```

**验证点：**
- ✅ 返回状态码 200
- ✅ 响应包含 `mask_base64`（Base64编码的PNG图像）
- ✅ `infer_time` 在合理范围（CPU: 1-5秒，GPU: 0.1-1秒）
- ✅ 解码后的掩码尺寸与原图一致
- ✅ 掩码值范围为 [0, 255]

---

## 🔧 自定义测试配置

### 修改测试文件路径

编辑 `test_seg_api.py`：

```python
# 测试配置
TEST_WEIGHT_PATH = "./your_custom_weight.pth"  # 自定义权重路径
TEST_IMAGE_PATH = "./your_test_image.png"      # 自定义测试图像
```

### 测试上传权重

```python
# 启用权重上传测试
use_uploaded_weight = True  # 改为True
```

### 跳过本地对比

```python
# 如果没有本地模型，可以跳过此步骤
# 测试脚本会自动捕获ImportError并跳过
```

---

## 🐛 故障排查

### 问题1：后端服务无法启动

**错误信息：**
```
OSError: [WinError 10048] 通常每个套接字地址只允许使用一次
```

**解决方案：**
```bash
# 端口8000被占用，关闭占用进程或更换端口
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# 或修改config.py中的SERVER_PORT
```

### 问题2：测试图像不存在

**错误信息：**
```
❌ 测试图像不存在: ./test_data/test_image.png
```

**解决方案：**
```bash
# 创建测试数据目录并放入图像
mkdir test_data
# 将您的测试图像复制到此目录
copy /path/to/your/image.png test_data/test_image.png
```

### 问题3：权重文件不存在

**错误信息：**
```
⚠ 测试权重文件不存在: ./models/weights/unet_octa.pth
```

**解决方案：**
```python
# 方案1：使用官方权重（推荐）
use_uploaded_weight = False

# 方案2：提供自定义权重路径
TEST_WEIGHT_PATH = "./your_weight.pth"
```

### 问题4：CUDA内存不足

**错误信息：**
```
RuntimeError: CUDA out of memory
```

**解决方案：**
```python
# 修改config.py强制使用CPU
DEVICE_PRIORITY = "cpu"
```

### 问题5：网络连接失败

**错误信息：**
```
✗ 无法连接到后端服务
```

**解决方案：**
1. 确认后端服务已启动
2. 检查防火墙设置
3. 确认端口8000未被占用

---

## 📊 性能基准

### 预期性能指标

| 指标 | CPU模式 | GPU模式 |
|-----|--------|---------|
| 模型加载 | 1-3秒 | 0.5-1秒 |
| 单张推理（256×256） | 2-5秒 | 0.1-0.5秒 |
| 完整请求（含网络） | 3-8秒 | 1-3秒 |

### 性能优化建议

1. **使用GPU加速：** `DEVICE_PRIORITY = "cuda"`
2. **启用模型缓存：** 首次加载后，后续请求复用模型
3. **批处理推理：** 多张图像可批量处理（需修改API）

---

## 🎓 毕设答辩演示建议

### 演示流程（5分钟）

1. **展示后端启动**（30秒）
   - 运行 `python main.py`
   - 显示服务启动信息

2. **展示API文档**（30秒）
   - 打开 `http://127.0.0.1:8000/docs`
   - 展示Swagger UI界面

3. **运行联调测试**（2分钟）
   - 运行 `python test_seg_api.py`
   - 展示完整测试流程和输出

4. **展示测试结果**（1分钟）
   - 打开 `test_results/comparison.png`
   - 对比原图和分割结果

5. **代码讲解**（1分钟）
   - 展示核心模块代码
   - 说明模块间的调用关系

### 关键点强调

✅ **模块化设计：** 展示各模块独立性和接口清晰  
✅ **配置管理：** 展示 `config.py` 集中配置  
✅ **错误处理：** 展示各类异常的详细提示  
✅ **性能优化：** 展示GPU加速和模型缓存  
✅ **API规范：** 展示RESTful设计和响应格式

---

## 📁 文件清单

```
octa_backend/
├── main.py                      # ✅ FastAPI应用入口
├── test_seg_api.py              # ✅ 完整联调测试脚本
├── INTEGRATION_TEST_GUIDE.md    # ✅ 本文档
├── config/
│   └── config.py                # ✅ 全局配置
├── core/
│   ├── weight_validator.py      # ✅ 权重校验模块
│   ├── model_loader.py          # ✅ 模型加载模块
│   └── data_process.py          # ✅ 数据处理模块
├── router/
│   ├── weight_router.py         # ✅ 权重管理API
│   └── seg_router.py            # ✅ 分割预测API
├── test_data/
│   └── test_image.png           # 📝 需准备
└── test_results/
    ├── api_result_mask.png      # 自动生成
    ├── comparison.png           # 自动生成
    └── local_result.png         # 自动生成（可选）
```

---

## 🔗 相关文档

- **配置使用指南：** [CONFIG_USAGE_GUIDE.md](./CONFIG_USAGE_GUIDE.md)
- **API文档：** http://127.0.0.1:8000/docs（服务启动后访问）
- **故障排查：** [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)

---

**最后更新：** 2026-01-28  
**维护者：** OCTA Web项目组  
**适用版本：** v1.0.0
