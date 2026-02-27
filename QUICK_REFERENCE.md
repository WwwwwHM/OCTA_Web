# 📖 快速参考 - unet.py 修改指南

## 🎯 三个核心函数速查

### 1. load_unet_model()
```python
def load_unet_model(model_type='unet', model_path=None, device='cpu'):
    """加载OCTA预训练模型"""
    # ✅ 固定权重路径：./models/weights/unet_octa.pth
    # ✅ 强制CPU模式
    # ✅ 权重不存在返回None
```

| 参数 | 说明 | 注意事项 |
|-----|-----|--------|
| `model_type` | 'unet'或'fcn' | 推荐'unet' |
| `model_path` | 权重路径参数 | 仅兼容性保留，实际固定 |
| `device` | 设备参数 | 仅兼容性保留，强制CPU |

**返回值：** 模型对象或None

---

### 2. postprocess_mask()
```python
def postprocess_mask(mask_tensor, original_size=None):
    """后处理分割掩码为8位灰度图"""
    # ✅ 张量维度处理
    # ✅ 值范围缩放 [0,1] -> [0,255]
    # ✅ uint8格式输出
```

| 参数 | 说明 | 格式 |
|-----|-----|------|
| `mask_tensor` | 模型输出 | (1,1,256,256) |
| `original_size` | 原始尺寸 | (width, height) |

**返回值：** numpy数组 (height, width), uint8, [0,255]

---

### 3. segment_octa_image()
```python
def segment_octa_image(image_path, model_type='unet', 
                      model_path=None, output_path=None, device='cpu'):
    """完整的OCTA图像分割流程"""
    # ✅ 8步完整流程
    # ✅ 容错返回原图
    # ✅ 详细日志输出
```

**8步流程：**
```
1. 输入文件验证 ──> 不存在返回原图
2. 模型加载 ──> 失败返回None → 返回原图
3. 图像预处理 ──> 失败返回原图
4. 获取原始尺寸 ──> 用于后处理
5. 模型推理 ──> CPU强制
6. 结果后处理 ──> 8位灰度图
7. 生成输出路径 ──> 自动或指定
8. 保存结果 ──> PNG格式
```

---

## 🔧 常见用法

### ✨ 最简单的方式（推荐）
```python
from models.unet import segment_octa_image

# 权重自动从 ./models/weights/unet_octa.pth 加载
result_path = segment_octa_image('uploads/image.png')

# 检查结果
if result_path.endswith('_segmented.png'):
    print("✓ 分割成功")
else:
    print("✗ 分割失败（返回原图）")
```

### 📋 完整参数方式
```python
result_path = segment_octa_image(
    image_path='uploads/image.png',
    model_type='unet',                    # 推荐unet
    model_path=None,                      # 固定路径，无需改动
    output_path='results/result.png',     # 可选，自动生成
    device='cpu'                          # 强制CPU，无需改动
)
```

### 🔍 手动加载模型
```python
from models.unet import load_unet_model
import torch

# 加载模型
model = load_unet_model('unet')

if model is not None:
    # 使用模型
    image_tensor = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(image_tensor)
else:
    print("权重文件不存在或加载失败")
```

---

## 📦 输入输出规范

### 输入要求
- **格式**：PNG图像
- **通道**：RGB（3通道）
- **尺寸**：任意（会被调整到256x256）
- **位深**：8位或更高

### 输出规范
- **格式**：PNG图像
- **通道**：灰度图（1通道）
- **尺寸**：与输入原始尺寸相同
- **位深**：8位（uint8）
- **范围**：0-255

---

## ⚠️ 错误处理

### 错误情况 → 返回值
```
权重文件不存在 → None（load_unet_model）
图像不存在 → 原图路径
预处理失败 → 原图路径
推理失败 → 原图路径
任何异常 → 原图路径（带详细日志）
```

### 日志标签
```
[INFO]    - 正常信息流
[WARNING] - 警告信息
[ERROR]   - 错误信息
[SUCCESS] - 成功完成
```

---

## 🎪 日志示例

### ✅ 成功的日志流
```
[INFO] 开始处理OCTA图像: uploads/img.png
[INFO] 正在加载OCTA预训练模型...
[INFO] 创建U-Net模型成功
[INFO] OCTA预训练权重路径: ./models/weights/unet_octa.pth
[INFO] 权重文件已加载到CPU内存
[SUCCESS] 成功加载OCTA预训练权重: ./models/weights/unet_octa.pth
[INFO] 模型已设置为评估模式（CPU）
[INFO] 模型参数总数: 28,257,281
[INFO] 正在预处理OCTA图像...
[INFO] 图像预处理完成，张量形状: torch.Size([1, 3, 256, 256])
[INFO] 原始图像尺寸: (512, 512)
[INFO] 正在进行OCTA分割模型推理...
[INFO] 模型推理完成，输出掩码形状: torch.Size([1, 1, 256, 256])
[INFO] 正在进行结果后处理...
[INFO] 后处理完成，掩码数据类型: uint8, 值范围: [0, 255]
[INFO] 输出文件路径: results/img_segmented.png
[SUCCESS] OCTA图像分割成功！
[INFO] 分割结果已保存: results/img_segmented.png
```

### ❌ 权重文件缺失的日志流
```
[INFO] 开始处理OCTA图像: uploads/img.png
[INFO] 正在加载OCTA预训练模型...
[INFO] 创建U-Net模型成功
[INFO] OCTA预训练权重路径: ./models/weights/unet_octa.pth
[WARNING] OCTA预训练权重文件不存在
[WARNING] 预期路径: D:\Code\OCTA_Web\octa_backend\models\weights\unet_octa.pth
[WARNING] 请将 unet_octa.pth 文件放入 ./models/weights/ 目录
[WARNING] 如无权重文件，将使用随机初始化模型进行测试
[WARNING] OCTA预训练权重加载失败
[WARNING] 请检查 ./models/weights/unet_octa.pth 是否存在
[WARNING] 返回原图路径以便调试
→ 返回原图路径
```

---

## 🔑 关键常数和路径

| 项目 | 值 |
|-----|-----|
| 权重路径 | `./models/weights/unet_octa.pth` |
| 输入大小 | 256×256 |
| 输入通道 | 3 (RGB) |
| 输出通道 | 1 (灰度) |
| 输出格式 | 8位灰度 (uint8) |
| 输出范围 | 0-255 |
| CPU强制 | torch.device('cpu') |
| Sigmoid激活 | [0,1] → 8位后 → [0,255] |

---

## 🧪 快速测试

### 测试1：验证导入
```bash
cd octa_backend
python -c "from models.unet import load_unet_model, segment_octa_image; print('✓ 导入成功')"
```

### 测试2：验证权重加载
```bash
python -c "from models.unet import load_unet_model; m = load_unet_model('unet'); print('✓ 权重加载成功' if m else '✗ 权重缺失')"
```

### 测试3：验证语法
```bash
python -m py_compile models/unet.py && echo "✓ 语法检查通过"
```

---

## 📞 故障排查快速指南

### 问题：权重文件不存在
**解决方案：**
1. 检查路径：`./models/weights/unet_octa.pth`
2. 权重文件应该存在（已验证 ✅）
3. 如果不存在，放入权重文件

### 问题：导入错误
**解决方案：**
1. 确保在 `octa_backend` 目录
2. 虚拟环境已激活
3. 依赖已安装：`pip install -r requirements.txt`

### 问题：推理失败
**解决方案：**
1. 查看详细日志
2. 检查输入图像格式（必须PNG）
3. 检查磁盘空间

---

## 📚 相关文档链接

- **完整修改报告**：[COMPLETION_REPORT.md](../COMPLETION_REPORT.md)
- **修改详情**：[MODIFICATION_SUMMARY.md](../MODIFICATION_SUMMARY.md)
- **AI开发指南**：[.github/copilot-instructions.md](../.github/copilot-instructions.md)
- **源代码**：[octa_backend/models/unet.py](../octa_backend/models/unet.py)

---

**最后更新**：2026年1月12日  
**状态**：✅ 生产就绪
