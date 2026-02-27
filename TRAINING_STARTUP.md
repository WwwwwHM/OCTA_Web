# 🚀 OCTA 训练功能启动指南

> **本指南适合快速上手** - 只需 3 步即可开始模型训练

---

## ⚡ 30 秒快速启动

### Windows 用户

```bash
# 方案1：一键启动脚本（推荐）
cd octa_backend
start_server_cpu.bat
```

### 所有用户

```bash
# 方案2：两个终端分别启动

# 终端1：启动后端
cd octa_backend
python main.py

# 终端2：启动前端
cd octa_frontend
npm run dev
```

**完成后访问：** http://127.0.0.1:5173/train

---

## ✅ 系统验证

运行诊断脚本检查环境：

```bash
cd octa_backend
python diagnose.py
```

**预期输出（全部 ✓）：**
```
1️⃣  PyTorch 环境检查
✓ PyTorch 版本: 2.6.0
✓ CUDA 支持: True

2️⃣  配置文件检查
✓ MODEL_DEVICE 配置: cpu ✓

3️⃣  目录结构检查
✓ 所有必要目录已创建

4️⃣  模型权重检查
✓ 预训练权重存在: 118.50 MB

5️⃣  设备创建测试
✓ cpu: 成功
✓ cuda: 成功

6️⃣  训练模块检查
✓ train_service: 导入成功
✓ unet: 导入成功
✓ config: 导入成功
```

---

## 📊 训练快速测试

### 创建测试数据集（3 步）

**创建目录结构：**
```
test_dataset/
├── images/
│   ├── img1.png
│   ├── img2.png
│   └── ...
└── masks/
    ├── img1.png
    ├── img2.png
    └── ...
```

**Python 脚本生成（5 分钟）：**

```python
# 保存为 generate_test_data.py
from PIL import Image
import numpy as np
import os
from pathlib import Path

# 创建目录
Path("test_dataset/images").mkdir(parents=True, exist_ok=True)
Path("test_dataset/masks").mkdir(parents=True, exist_ok=True)

# 生成 10 张测试图像
for i in range(10):
    # 生成随机图像（256x256，RGB）
    img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    Image.fromarray(img).save(f"test_dataset/images/img{i:03d}.png")
    
    # 生成随机标注（256x256，灰度）
    mask = np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255
    Image.fromarray(mask).save(f"test_dataset/masks/img{i:03d}.png")

print("✓ 测试数据集已生成（10 张图像）")

# 压缩为 ZIP（可选）
import shutil
shutil.make_archive("test_dataset", "zip", ".", "test_dataset")
print("✓ 已创建 test_dataset.zip")
```

**运行生成脚本：**
```bash
python generate_test_data.py
```

### 开始训练

1. **打开训练页面：** http://127.0.0.1:5173/train

2. **拖拽或选择数据集：**
   - 上传 `test_dataset.zip`（或已压缩的任何数据集）

3. **配置参数：**
   ```
   训练轮数：5
   学习率：0.001
   批次大小：4
   验证分割：0.2
   ```

4. **点击"开始训练"**

5. **监控进度：**
   - 实时显示训练状态
   - 显示当前轮数和损失值
   - 完成后显示性能指标（Dice, IOU）
   - ECharts 曲线实时更新

**预期耗时：** 5-10 分钟（取决于数据量和 CPU 性能）

---

## 🎯 预期结果

### 成功指标
- ✓ 上传进度显示 0-100%
- ✓ 训练状态显示"训练进行中"
- ✓ 实时显示各轮次的损失值
- ✓ 完成后显示最终指标
- ✓ ECharts 曲线正确渲染

### 如果出现问题

| 症状 | 原因 | 解决方案 |
|------|------|--------|
| 后端连接失败 | 后端未运行 | 检查 http://127.0.0.1:8000 |
| 上传失败 | ZIP 格式错误 | 确保是有效的 ZIP 文件 |
| 训练很慢 | CPU 模式正常 | 可选：改用 GPU（见配置章节） |
| 曲线不显示 | 页面未加载 | 刷新 F5 或检查浏览器控制台 |

---

## ⚙️ 配置修改（可选）

### 使用 GPU 训练

编辑：`octa_backend/config/config.py` 第 107 行

```python
# 改为
MODEL_DEVICE = "cuda"  # 使用 GPU（需要 NVIDIA GPU 和 CUDA）
```

**优点：** 训练速度快 10 倍  
**要求：** 
- NVIDIA GPU（如 RTX 4050）
- CUDA 工具包
- cuDNN（可选）

**注意：** 系统会自动回退到 CPU 如果 GPU 不可用

### 修改其他参数

**服务器地址：**
```python
SERVER_HOST = "127.0.0.1"  # 第 126 行
SERVER_PORT = 8000          # 第 127 行
```

**前端跨域地址：**
```python
CORS_ORIGINS = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]  # 第 148 行
```

**数据库位置：**
```python
DB_PATH = "./octa.db"  # 第 35 行
```

**上传文件大小限制：**
```python
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB，第 65 行
```

---

## 📁 项目结构

```
OCTA_Web/
├── octa_backend/
│   ├── config/
│   │   └── config.py ⚙️ [所有配置在这里]
│   ├── service/
│   │   ├── train_service.py 🚂 [训练逻辑+Device Fallback]
│   │   └── model_service.py
│   ├── models/
│   │   └── unet.py 🧠 [U-Net架构]
│   ├── main.py 🔧 [后端主程序]
│   ├── start_server_cpu.bat 💨 [Windows一键启动]
│   ├── diagnose.py 🔍 [环境诊断工具]
│   └── requirements.txt 📦
│
├── octa_frontend/
│   ├── src/
│   │   ├── views/
│   │   │   ├── HomeView.vue 🏠
│   │   │   ├── TrainView.vue 🚂 [新增：训练页面]
│   │   │   └── HistoryView.vue 📋
│   │   ├── App.vue ✨ [已升级：Element Plus 菜单]
│   │   └── router/
│   │       └── index.js 🛣️
│   └── package.json 📦
│
└── QUICK_REFERENCE.md 📖 [快速参考]
```

---

## 🔄 工作流程

### 开发者工作流

```
1. 修改代码 (models/train_service.py)
      ↓
2. 启动后端 (python main.py)
      ↓
3. 启动前端 (npm run dev)
      ↓
4. 访问训练页面 (http://127.0.0.1:5173/train)
      ↓
5. 上传数据集测试
      ↓
6. 查看日志和曲线
      ↓
7. 修复问题，重复 1-6
```

### 使用者工作流

```
1. 准备数据集 (images + masks)
      ↓
2. 压缩为 ZIP
      ↓
3. 打开训练页面
      ↓
4. 拖拽上传 ZIP
      ↓
5. 配置参数
      ↓
6. 开始训练
      ↓
7. 等待完成，查看结果
```

---

## 💡 故障排查

### 错误1：后端启动失败

```
ModuleNotFoundError: No module named 'fastapi'
```

**解决：**
```bash
cd octa_backend
..\octa_env\Scripts\activate  # 激活虚拟环境
pip install -r requirements.txt
python main.py
```

### 错误2：前端无法连接

```
Failed to fetch: http://127.0.0.1:8000/train-start
```

**解决：**
1. 确认后端运行在 8000：`curl http://127.0.0.1:8000`
2. 检查防火墙是否阻止
3. 查看前端请求地址配置

### 错误3：训练立即失败

```
"detail": "训练失败：No such file or directory"
```

**解决：**
1. 检查 ZIP 文件有效性
2. 运行 `python diagnose.py` 检查目录
3. 查看后端控制台详细错误

### 错误4：ECharts 不显示

```
Chart is not a constructor
```

**解决：**
```bash
cd octa_frontend
npm install echarts
npm run dev
```

---

## 📞 技术支持

### 快速诊断

```bash
# 1. 检查环境
cd octa_backend && python diagnose.py

# 2. 查看后端日志
# 后端控制台输出（开始时显示）
[INFO] Configuration source: config/config.py
[INFO] Service address: 127.0.0.1:8000
[INFO] CORS allowed origins: 2 frontend addresses

# 3. 查看前端日志
# 浏览器 F12 → Console 标签

# 4. 测试 API
curl http://127.0.0.1:8000/docs  # Swagger UI
```

### 常见日志信息

| 日志 | 含义 | 处理 |
|------|------|------|
| `[INFO] 使用设备：cpu` | 正常（CPU 模式） | 无需操作 |
| `[WARNING] 设备不可用，回退到 cpu` | 自动回退 | 无需操作 |
| `[ERROR] 模型加载失败` | 权重文件问题 | 检查 ./models/weights/ |
| `[INFO] 训练已启动` | 训练开始 | 等待完成 |

---

## 📊 性能指标

### 预期性能

| 操作 | CPU | GPU |
|-----|-----|-----|
| 模型加载 | 1-2s | <1s |
| 数据处理（10张） | 2-3s | 1-2s |
| 1轮训练（10张） | 30-60s | 5-10s |
| 10轮训练 | 5-10分钟 | 30-60秒 |

### 优化建议

- 小数据集（<100张）：使用 CPU 即可
- 大数据集（>500张）：建议使用 GPU
- 内存不足：减少批次大小（4 → 2）
- 显存不足：改用 CPU 或减小模型

---

## 🎓 学习资源

| 资源 | 位置 | 用途 |
|------|------|------|
| 快速开始 | TRAINING_QUICK_START.md | 详细步骤 |
| 技术总结 | TRAINING_INTEGRATION_SUMMARY.md | 架构设计 |
| 完成报告 | TRAINING_COMPLETION_REPORT.md | 验收标准 |
| API 文档 | http://127.0.0.1:8000/docs | 接口参考 |
| 配置参考 | octa_backend/config/config.py | 所有参数 |

---

## ✨ 新增功能概览

### 📱 前端 (TrainView.vue)

- ✅ 拖拽上传数据集
- ✅ 实时上传进度显示
- ✅ 参数配置表单
- ✅ 训练进度实时显示
- ✅ ECharts 损失曲线
- ✅ 最终指标显示（Dice, IOU, Acc）

### 🔧 后端 (train_service.py)

- ✅ ZIP 解压和数据加载
- ✅ 图像预处理和增强
- ✅ U-Net 模型训练
- ✅ 实时指标计算
- ✅ 模型权重保存
- ✅ Device Fallback（GPU → CPU 自动回退）

### ⚙️ 配置管理 (config.py)

- ✅ 集中式配置管理
- ✅ 模型设备选择
- ✅ 数据库配置
- ✅ CORS 跨域配置
- ✅ 文件存储配置

### 🔍 诊断工具 (diagnose.py)

- ✅ 环境验证
- ✅ 设备检查
- ✅ 模块导入检查
- ✅ 目录验证
- ✅ 权重文件检查

---

## 🚀 下一步

1. **立即开始：** 按照"30秒快速启动"运行
2. **创建数据：** 使用提供的 Python 脚本生成测试数据
3. **开始训练：** 上传数据集，点击开始训练
4. **查看结果：** 监控损失曲线和指标
5. **调整参数：** 根据结果调整学习率等参数

---

**状态：** ✅ 生产就绪  
**最后更新：** 2026年1月16日  
**版本：** 1.0.0

现在就开始使用训练功能吧！🎉
