# OCTA后端优化 - 快速使用指南

## 🚀 快速开始（3步）

### 步骤1：安装新依赖

```bash
cd octa_backend
pip install APScheduler>=3.10.0
```

### 步骤2：启动后端

```bash
python main.py
```

**启动成功标志**：
```
======================================================================
                      OCTA图像分割后端启动中...
======================================================================
[INFO] 配置来源: config/config.py
[INFO] 服务地址: 127.0.0.1:8000
[SUCCESS] ✓ 文件管理表已就绪
[SUCCESS] ✓ 定时清理任务已启动
======================================================================
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 步骤3：测试接口

**健康检查**：
```bash
curl http://127.0.0.1:8000/
```

**响应**：
```json
{
  "status": "ok",
  "message": "OCTA Image Segmentation API is running"
}
```

---

## 📝 核心API使用

### 1. 权重上传

**请求**：
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/weight/upload" \
  -F "file=@unet_best.pth" \
  -F "model_type=unet"
```

**响应**：
```json
{
  "code": 200,
  "msg": "权重上传成功",
  "data": {
    "weight_id": "abc123def456",
    "file_id": 1,
    "file_name": "unet_best.pth",
    "file_size_mb": 45.67,
    "model_type": "unet",
    "metadata": {
      "total_params": 31042945,
      "total_keys": 234,
      "file_size_mb": 45.67
    }
  }
}
```

**校验失败示例**：
```json
{
  "detail": "权重校验失败: 权重文件缺少必需的层: enc1.conv1.weight, enc1.bn1.weight等5个"
}
```

---

### 2. 权重列表查询

**请求**：
```bash
curl "http://127.0.0.1:8000/api/v1/weight/list?model_type=unet"
```

**响应**：
```json
{
  "code": 200,
  "msg": "查询成功",
  "data": [
    {
      "weight_id": "abc123",
      "file_name": "unet_best.pth",
      "file_size_mb": 45.67,
      "model_type": "unet",
      "upload_time": "2026-01-27 10:30:00"
    }
  ]
}
```

---

### 3. 图像分割预测

**请求**：
```bash
curl -X POST "http://127.0.0.1:8000/segment-octa/" \
  -F "file=@image.png" \
  -F "model_type=unet" \
  -F "weight_id=abc123"  # 可选，默认使用官方权重
```

**响应**：
```json
{
  "code": 200,
  "msg": "分割成功",
  "data": {
    "mask_base64": "iVBORw0KGgoAAAANSUhEUgAAA...",
    "mask_url": "/results/image_seg.png",
    "inference_time": 0.125,
    "device": "cuda",
    "model_type": "unet",
    "weight_id": "abc123",
    "image_size": [512, 512]
  }
}
```

---

### 4. 权重删除

**请求**：
```bash
curl -X DELETE "http://127.0.0.1:8000/api/v1/weight/delete/abc123"
```

**响应**：
```json
{
  "code": 200,
  "msg": "权重删除成功"
}
```

---

## 🔍 日志查看

### 实时日志监控

**Linux/Mac**：
```bash
tail -f logs/octa_backend.log
```

**Windows PowerShell**：
```powershell
Get-Content logs\octa_backend.log -Wait -Tail 20
```

### 日志示例

```
2026-01-27 10:30:45 - core.weight_validator - INFO - [权重校验] ✓ 权重文件校验通过: unet_best.pth
2026-01-27 10:30:45 - service.weight_service - INFO - [权重上传] ✓ 成功，weight_id=abc123, file_id=1
2026-01-27 10:30:46 - core.model_loader - INFO - [模型加载] ✓ 权重加载成功，设备: cuda
2026-01-27 10:30:46 - service.prediction_service - INFO - [预测] ✓ 推理完成，耗时=0.125秒
2026-01-27 02:00:00 - utils.cleanup_task - INFO - [清理任务] ✓ 完成，删除 12 个文件，释放 45.67MB 空间
```

---

## 🛠️ 配置调整

### 修改日志级别

**文件**：`config/config.py`

```python
# 开发环境：详细日志
LOG_LEVEL = "DEBUG"

# 生产环境：关键日志
LOG_LEVEL = "INFO"

# 仅错误日志
LOG_LEVEL = "ERROR"
```

### 调整清理策略

**文件**：`config/config.py`

```python
# 禁用自动清理
ENABLE_AUTO_CLEANUP = False

# 调整清理间隔（6小时）
CLEANUP_INTERVAL_SECONDS = 6 * 3600

# 调整过期时间（48小时）
FILE_EXPIRY_SECONDS = 48 * 3600
```

### 修改权重大小限制

**文件**：`config/config.py`

```python
# 增加到500MB
WEIGHT_MAX_SIZE = 500 * 1024 * 1024
```

---

## 🔧 故障排查

### 问题1：权重校验失败

**错误信息**：
```
权重校验失败: 权重文件缺少必需的层: enc1.conv1.weight...
```

**解决方案**：
1. 确认权重文件是U-Net模型权重
2. 检查是否使用了不同架构的模型
3. 尝试重新训练或使用官方权重

---

### 问题2：CUDA不可用

**日志信息**：
```
[设备选择] ⚠ GPU不可用，使用CPU
```

**解决方案**：
1. 检查PyTorch CUDA安装：`python -c "import torch; print(torch.cuda.is_available())"`
2. 安装CUDA版本PyTorch：`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
3. CPU模式也可正常使用，仅速度较慢

---

### 问题3：模型加载失败

**错误信息**：
```
模型加载失败: 权重文件不存在
```

**解决方案**：
1. 检查权重文件路径：`config.OFFICIAL_WEIGHT_PATH`
2. 确认权重文件存在：
   ```bash
   ls -lh static/uploads/weight/official/unet_best_dice0.78.pth
   ```
3. 如无官方权重，上传自定义权重后使用weight_id

---

### 问题4：清理任务未启动

**日志信息**：
```
[WARNING] ⚠ 定时清理任务启动失败: ...
```

**解决方案**：
1. 检查APScheduler是否安装：`pip list | grep APScheduler`
2. 检查配置：`ENABLE_AUTO_CLEANUP = True`
3. 重启后端服务

---

## 📊 性能优化建议

### 1. GPU加速

**配置**：
```python
# config/config.py
MODEL_DEVICE = "cuda"  # 强制使用GPU
```

**性能对比**：
- GPU (CUDA)：0.05-0.15秒/张
- CPU：0.3-1.0秒/张

---

### 2. 模型缓存

**说明**：同一weight_id的多次预测会自动使用缓存，无需配置

**性能提升**：
- 首次加载：~2秒（加载权重）
- 后续预测：~0.1秒（使用缓存）

---

### 3. 批量预测

**建议**：如需处理大量图像，可考虑：
1. 使用多进程/多线程
2. 实现批量预测接口（一次传多张图）
3. 使用GPU批量推理（batch_size>1）

---

## 📁 重要文件路径

### 核心模块
- `core/weight_validator.py` - 权重校验
- `core/model_loader.py` - 模型加载
- `core/data_process.py` - 数据处理

### 服务层
- `service/prediction_service.py` - 预测服务
- `service/weight_service.py` - 权重管理

### 工具类
- `utils/logger.py` - 日志配置
- `utils/cleanup_task.py` - 定时清理

### 配置文件
- `config/config.py` - 统一配置管理

### 日志文件
- `logs/octa_backend.log` - 主日志文件
- `logs/octa_backend.log.1` - 备份1
- ...

### 权重存储
- `static/uploads/weight/official/` - 官方预置权重
- `static/uploads/weight/{weight_id}/` - 用户上传权重

---

## 🎓 进阶用法

### 1. Python脚本调用

```python
from pathlib import Path
from service.prediction_service import get_prediction_service

# 获取预测服务
service = get_prediction_service()

# 执行预测
result = service.predict(
    image_path=Path('uploads/test.png'),
    weight_id='abc123',  # 或None使用官方权重
    model_type='unet',
    save_result=True,
    output_dir=Path('results')
)

print(f"推理耗时: {result['inference_time']}秒")
print(f"运行设备: {result['device']}")
print(f"掩码已保存: {result['mask_path']}")
```

---

### 2. 自定义阈值

```python
from core.data_process import get_processor

processor = get_processor()
# 后处理时指定阈值
mask = processor.postprocess(output_tensor, original_size, threshold=0.3)
```

---

### 3. 手动触发清理

```python
from utils.cleanup_task import get_cleanup_task

cleanup = get_cleanup_task()
cleanup.run_now()  # 立即执行一次清理
```

---

## 📞 获取帮助

**文档**：
- 完整优化报告：`BACKEND_OPTIMIZATION_COMPLETE.md`
- API文档：http://127.0.0.1:8000/docs（Swagger UI）

**日志**：
- 主日志：`logs/octa_backend.log`
- 实时监控：`tail -f logs/octa_backend.log`

**配置**：
- 统一配置：`config/config.py`
- 所有参数都有详细注释

---

**版本**：v1.0  
**更新**：2026-01-27  
**维护**：OCTA Web项目组
