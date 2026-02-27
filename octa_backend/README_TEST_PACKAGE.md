# OCTA血管分割平台 - 联调测试完整包

## 📦 文件清单

本次生成的联调测试文件包含：

### 1. 核心服务文件

| 文件 | 说明 | 状态 |
|-----|------|------|
| `main.py` | FastAPI应用入口，集成所有路由 | ✅ 已更新 |
| `config/config.py` | 全局配置中心（7大配置分组） | ✅ 已更新 |

### 2. 测试脚本

| 文件 | 说明 | 用途 |
|-----|------|------|
| `test_seg_api.py` | 完整联调测试脚本 | 测试所有API接口 |
| `run_test.bat` | Windows测试启动脚本 | 一键运行测试（Windows） |

### 3. 文档

| 文件 | 说明 |
|-----|------|
| `INTEGRATION_TEST_GUIDE.md` | 完整联调测试指南 |
| `CONFIG_USAGE_GUIDE.md` | 配置使用指南 |
| `README_TEST_PACKAGE.md` | 本文档 |

---

## 🚀 快速启动（3步）

### 步骤1：启动后端服务

**Windows:**
```bash
cd octa_backend
python main.py
```

**Linux/Mac:**
```bash
cd octa_backend
python3 main.py
```

### 步骤2：准备测试数据

```bash
# 创建测试图像（请提供您的OCTA测试图像）
# 文件路径: octa_backend/test_data/test_image.png
# 建议尺寸: 256x256
# 建议格式: PNG
```

### 步骤3：运行测试

**Windows:**
```bash
# 方式1：使用脚本（推荐）
cd octa_backend
run_test.bat

# 方式2：直接运行
cd octa_backend
python test_seg_api.py
```

**Linux/Mac:**
```bash
cd octa_backend
python3 test_seg_api.py
```

---

## 📋 测试内容

### ✅ 自动测试项

1. **健康检查** - 验证服务正常运行
2. **权重上传** - 测试权重文件上传和校验（可选）
3. **分割预测** - 测试完整推理流程
4. **结果解码** - 解码Base64掩码并保存
5. **本地对比** - 与本地模型对比一致性（可选）
6. **可视化** - 生成原图vs掩码对比图

### 📊 测试输出

测试完成后，在 `test_results/` 目录查看：

```
test_results/
├── api_result_mask.png    # API返回的分割掩码
├── comparison.png         # 原图vs掩码对比图
├── local_result.png       # 本地推理结果（可选）
└── diff_mask.png          # 差异图（可选）
```

---

## 🎯 核心模块验证

### 已集成的核心模块

| 模块 | 功能 | 测试覆盖 |
|-----|------|---------|
| `core/weight_validator.py` | 权重格式+内容校验 | ✅ 上传时自动校验 |
| `core/model_loader.py` | 设备自适应+模型缓存 | ✅ 每次预测调用 |
| `core/data_process.py` | 预处理+后处理+Base64 | ✅ 完整流程测试 |
| `router/seg_router.py` | 分割预测API | ✅ 主要测试接口 |
| `router/weight_router.py` | 权重管理API | ✅ 上传测试 |
| `config/config.py` | 全局配置 | ✅ 所有模块引用 |

### API接口列表

```
后端地址: http://127.0.0.1:8000

核心接口:
├── GET  /                      # 健康检查
├── POST /api/v1/weight/upload  # 权重上传
├── GET  /api/v1/weight/list    # 权重列表
└── POST /api/v1/seg/predict    # 分割预测 ⭐核心

文档:
└── GET  /docs                  # Swagger UI
```

---

## 🔧 自定义配置

### 修改测试文件路径

编辑 `test_seg_api.py`：

```python
# 第20-23行
TEST_WEIGHT_PATH = "./models/weights/unet_octa.pth"  # 测试权重
TEST_IMAGE_PATH = "./test_data/test_image.png"       # 测试图像
```

### 启用/禁用权重上传测试

```python
# 第65行
use_uploaded_weight = True   # True=测试上传, False=使用官方权重
```

### 修改服务地址

```python
# 第16行
BASE_URL = "http://127.0.0.1:8000"  # 后端地址
```

---

## 🐛 常见问题

### Q1: 后端服务无法启动

**症状：** `OSError: [WinError 10048] Address already in use`

**解决：**
```bash
# Windows: 查找并关闭占用端口的进程
netstat -ano | findstr :8000
taskkill /PID <进程ID> /F

# Linux: 
lsof -i :8000
kill -9 <进程ID>
```

### Q2: 测试图像不存在

**症状：** `❌ 测试图像不存在: ./test_data/test_image.png`

**解决：**
```bash
# 创建目录并放入测试图像
mkdir test_data
# 复制您的OCTA图像到此目录，重命名为test_image.png
```

### Q3: 依赖包未安装

**症状：** `ModuleNotFoundError: No module named 'requests'`

**解决：**
```bash
# 激活虚拟环境（如果有）
..\octa_env\Scripts\activate   # Windows
source ../octa_env/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### Q4: CUDA内存不足

**症状：** `RuntimeError: CUDA out of memory`

**解决：**
编辑 `config/config.py`：
```python
DEVICE_PRIORITY = "cpu"  # 强制使用CPU
```

---

## 📚 详细文档

### 查看完整测试指南

```bash
# 打开联调测试指南
notepad INTEGRATION_TEST_GUIDE.md  # Windows
cat INTEGRATION_TEST_GUIDE.md      # Linux/Mac
```

### 查看配置使用指南

```bash
# 打开配置指南
notepad CONFIG_USAGE_GUIDE.md  # Windows
cat CONFIG_USAGE_GUIDE.md      # Linux/Mac
```

### 查看API文档（Swagger UI）

启动后端后，浏览器访问：
```
http://127.0.0.1:8000/docs
```

---

## 🎓 毕设答辩演示建议

### 演示流程（5分钟）

1. **展示项目结构**（30秒）
   - 展示核心模块代码
   - 说明模块化设计

2. **启动后端服务**（30秒）
   ```bash
   python main.py
   ```
   - 展示服务启动日志
   - 强调配置集中管理

3. **运行联调测试**（2分钟）
   ```bash
   python test_seg_api.py
   ```
   - 展示完整测试流程
   - 强调8步骤流程清晰

4. **展示测试结果**（1分钟）
   - 打开 `test_results/comparison.png`
   - 对比原图和分割效果

5. **查看API文档**（1分钟）
   - 打开 `http://127.0.0.1:8000/docs`
   - 展示RESTful API设计

### 关键亮点

✅ **模块化设计** - 6大核心模块独立开发、集成  
✅ **配置管理** - config.py集中管理7大配置  
✅ **错误处理** - 完善的异常捕获和提示  
✅ **性能优化** - GPU自动检测、模型缓存  
✅ **API规范** - RESTful设计、统一响应格式  
✅ **完整测试** - 8步骤自动化测试流程

---

## 📊 预期测试结果

### 成功输出示例

```
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
```

### 性能基准

| 指标 | CPU模式 | GPU模式 |
|-----|--------|---------|
| 模型加载 | 1-3秒 | 0.5-1秒 |
| 单张推理 | 2-5秒 | 0.1-0.5秒 |
| 完整请求 | 3-8秒 | 1-3秒 |

---

## 🔗 相关资源

### 项目文档

- **主README:** `../README.md`
- **后端故障排查:** `TROUBLESHOOTING.md`
- **前端集成指南:** `../octa_frontend/README.md`

### API文档

- **Swagger UI:** http://127.0.0.1:8000/docs（服务启动后）
- **ReDoc:** http://127.0.0.1:8000/redoc（服务启动后）

### 代码仓库

```bash
# 查看Git提交历史
git log --oneline --graph

# 查看文件修改
git diff
```

---

## ✨ 核心代码亮点

### 1. 配置集中管理

```python
# config/config.py
U_NET_IN_CHANNELS = 1
U_NET_OUT_CHANNELS = 1
INPUT_SIZE = (256, 256)
MEAN = 0.5
STD = 0.5
DEVICE_PRIORITY = "cuda"
```

### 2. 设备自动检测

```python
# core/model_loader.py
def get_device():
    if DEVICE_PRIORITY == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
```

### 3. 完整的错误处理

```python
# router/seg_router.py
try:
    model, device = load_model_by_weight_path(...)
except ModelLoadError as e:
    raise HTTPException(status_code=500, detail=f"模型加载失败: {e}")
```

### 4. Base64编码返回

```python
# core/data_process.py
def postprocess_mask(logits, original_size):
    # 后处理 → Base64编码
    return mask_array, mask_base64
```

---

## 📞 技术支持

### 问题反馈

遇到问题请按以下顺序排查：

1. 查看 `INTEGRATION_TEST_GUIDE.md` 故障排查章节
2. 检查 `logs/octa_backend.log` 日志文件
3. 运行 `python test_seg_api.py` 查看详细错误
4. 查看 `TROUBLESHOOTING.md` 常见问题

### 联系方式

- **项目组：** OCTA Web项目组
- **日期：** 2026-01-28
- **版本：** v1.0.0

---

**🎉 祝您答辩成功！**
