# OCTA 损失函数快速参考

## 🎯 一句话总结

用更好的损失函数（Dice-BCE）替代了基础的 BCE Loss，让模型训练更稳定、效果更好。

---

## 📝 快速使用

### 方式 1：直接导入（最简单）

```python
from models.losses import DiceBCELoss

# 初始化损失函数（就这么简单！）
criterion = DiceBCELoss(alpha=0.5, smooth=1.0)

# 计算损失（和普通 Loss 一样用）
loss = criterion(model_output, target_mask)
```

### 方式 2：使用工厂函数（更灵活）

```python
from models.losses import create_loss_function

# 创建混合损失函数
criterion = create_loss_function("dice_bce", alpha=0.5)

# 或者只用 Dice Loss
criterion = create_loss_function("dice", smooth=1.0)

# 或者只用 BCE Loss
criterion = create_loss_function("bce")
```

---

## 🔧 参数配置

### DiceLoss 参数

```python
DiceLoss(
    smooth=1.0,    # 平滑因子，防止分母为 0
    sigmoid=True   # 是否对输入应用 Sigmoid（logits 需要）
)
```

### DiceBCELoss 参数

```python
DiceBCELoss(
    alpha=0.5,           # Dice Loss 权重，[0,1]
    smooth=1.0,          # 平滑因子
    pos_weight=None,     # BCE Loss 正样本权重（处理不平衡）
    sigmoid=True         # 是否对输入应用 Sigmoid
)
```

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `alpha` | 0.5 | 平衡两个损失（可调为0.3-0.7） |
| `smooth` | 1.0 | 保持默认即可 |
| `pos_weight` | None（或 2-100） | 处理前景特别稀少的情况 |

---

## 💡 常见场景配置

### 场景 1：标准 OCTA 图像分割（推荐）

```python
criterion = DiceBCELoss(alpha=0.5, smooth=1.0)
```

**原因：** 平衡稳定性和分割精度

### 场景 2：血管极度稀疏（前景 < 5%）

```python
criterion = DiceBCELoss(
    alpha=0.5,
    smooth=1.0,
    pos_weight=20.0  # 给予前景像素更大权重
)
```

**原因：** 处理严重的类别不平衡

### 场景 3：优先考虑分割精度

```python
criterion = DiceBCELoss(alpha=0.7, smooth=1.0)
```

**原因：** 更强调 Dice 系数优化

### 场景 4：优先考虑训练稳定性

```python
criterion = DiceBCELoss(alpha=0.3, smooth=1.0)
```

**原因：** 更强调 BCE Loss 的稳定梯度

---

## 📊 与其他损失函数的对比

| 损失函数 | 优点 | 缺点 | 何时使用 |
|---------|------|------|--------|
| **BCEWithLogitsLoss** | ✅ 稳定收敛 | ❌ 不处理不平衡 | 前景充足的场景 |
| **DiceLoss** | ✅ 优化分割指标 | ⚠️ 有时不稳定 | 不平衡严重的场景 |
| **DiceBCELoss** | ✅ 稳定 ✅ 优化指标 | ❌ 稍复杂 | ⭐ **推荐（最平衡）** |
| **Focal Loss** | ✅ 处理不平衡 | ❌ 需要调参 | 检测任务 |

---

## 🔍 如何判断效果

### 训练中应该看到

- ✅ 损失值稳定下降（不会大幅波动）
- ✅ Dice 系数逐步提高
- ✅ 验证损失也在下降（不严重过拟合）
- ✅ 分割结果逐步改善（从粗糙到清晰）

### 如果看到问题

| 现象 | 可能原因 | 解决方案 |
|------|--------|--------|
| 损失值不下降 | 学习率太低 | 增加学习率 |
| 损失值剧烈波动 | 学习率太高 | 降低学习率 |
| 仅 Dice 提高但 BCE 不变 | alpha 过大 | 降低 alpha（如 0.3） |
| 过拟合严重 | 数据不足或正则化不够 | 增加数据或 Dropout |

---

## 🧪 验证损失函数工作正常

```python
import torch
from models.losses import DiceBCELoss

# 创建损失函数
criterion = DiceBCELoss(alpha=0.5)

# 创建随机输入和目标
logits = torch.randn(2, 1, 256, 256)  # 模型输出（logits）
targets = torch.randint(0, 2, (2, 1, 256, 256)).float()  # 标签

# 计算损失
loss = criterion(logits, targets)

print(f"损失值: {loss.item():.4f}")
print(f"损失范围应该是 [0, 1]，实际是 {loss.item():.4f}")

# 验证反向传播
loss.backward()
print("✓ 反向传播成功")
```

预期输出：
```
损失值: 0.6234
✓ 反向传播成功
```

---

## 🚀 在训练脚本中使用

```python
# train_service.py 已经更新，自动使用 DiceBCELoss
# 无需额外配置，开箱即用！

# 如果需要自定义，在 train_service.py 中修改这一行：
criterion = DiceBCELoss(alpha=0.5, smooth=1.0)  # <-- 在这里调整参数
```

---

## 📚 更多信息

- **完整实现：** `octa_backend/models/losses.py`
- **集成位置：** `octa_backend/service/train_service.py` 第 133 行
- **测试代码：** `losses.py` 文件底部的测试部分

---

## 常见问题

**Q：DiceBCELoss 和 DiceLoss 有什么区别？**
A：DiceLoss 只优化 Dice 系数，容易不稳定。DiceBCELoss 混合了两个损失，更稳定且效果更好。

**Q：alpha 参数怎么选？**
A：0.5 是最平衡的选择。0.7 强调分割精度，0.3 强调训练稳定性。

**Q：pos_weight 什么时候用？**
A：当前景像素特别少（< 2%）时，可以设置 10-100 来增加前景权重。

**Q：是否需要修改其他代码？**
A：不需要！损失函数已经自动集成到训练服务中了。

---

**版本：** 1.0.0 | **更新日期：** 2026年1月16日 | **状态：** ✅ 就绪
