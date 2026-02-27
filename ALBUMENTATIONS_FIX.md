# 🔧 Albumentations参数修复说明

## ✅ 问题已修复

**错误原因：** `RandomResizedCrop` 的 `scale` 参数配置有问题
```
scale=(0.7, 1.3)  # ❌ 错误：超出了[0,1]范围
```

**修复方案：** 改用 `RandomScale` + `Resize` 组合
```python
A.OneOf([
    A.RandomScale(scale_limit=0.3, p=1.0),  # ✓ ±30%尺度
    A.Resize(height=256, width=256, p=1.0),
], p=0.8)
```

---

## 🚀 重新开始训练

修复已自动应用，现在可以：

1. **关闭后端**（如果还在运行）
   ```
   Ctrl+C
   ```

2. **重新启动后端**
   ```bash
   python main.py
   ```

3. **返回前端重新点击训练**

---

## ✨ 改进说明

| 变化 | 原来 | 现在 | 效果 |
|-----|-----|-----|------|
| **尺度变换** | RandomResizedCrop(scale=(0.7,1.3)) | RandomScale(±30%) | ✅ 参数合法 |
| **裁剪方式** | 中心点随机 | 随机缩放+固定Resize | ✅ 更稳定 |
| **内存占用** | 较高 | 较低 | ✅ 更高效 |

---

## 📝 其他检查

确保数据集结构正确：
```
dataset.zip
├── images/
│   ├── image1.png
│   ├── image2.jpg
│   └── ...
└── masks/
    ├── image1.png  (或 .jpg)
    ├── image2.png
    └── ...
```

文件名前缀必须一一对应（扩展名可不同）。

