# Loss 卡在 0.6 的完整诊断与修复方案

## 🎯 问题描述
- 训练 loss 快速下降到 0.6 附近，之后停滞不前
- Val Dice 停留在 0.40 左右，无法提升
- Gradient Norm 在正常范围，不是梯度消失

**根本原因**很可能是以下几个中的一个或多个：
1. **数据质量问题**（掩码不是二值图像、前景比例过低等）
2. **学习率过小**（模型学习太慢，还没有充分收敛）
3. **Loss 函数权重不当**（BCE vs Dice 比例不合理）
4. **模型架构不适配**（模型太大、参数太多）

---

## 📋 快速诊断（5 分钟）

### 步骤 1：检查数据质量
```bash
cd octa_backend
..\octa_env\Scripts\python.exe diagnose_dataset.py "path/to/your/dataset"
```

**关键看这些输出：**
```
✓ 掩码是二值图像                          → 数据质量 ✓
⚠️ 掩码包含 N 个唯一值 (期望 2 个)       → ❌ 掩码不是二值
✓ 平均前景比例: 15.23% (正常)           → 数据 ✓
⚠️ 平均前景比例仅 0.50% (<2%)          → ❌ 严重类不平衡
```

---

### 步骤 2：查看训练日志

**关键信息 1：Loss 分量**（每个 epoch）
```
Train Loss: 0.6123 (BCE: 0.3000, Dice: 0.4271)
          ↑         ↑                 ↑
       总 loss      像素级分类         形状约束
```

**判断：**
- 如果 **Dice > 0.35** → 模型未学会分割形状，Loss 权重可能有问题
- 如果 **BCE > 0.3** → 像素级分类还不稳定，需要更多 epoch
- 如果 **两者都很高** → 掩码质量或学习率问题

**关键信息 2：梯度范数**（每个 step）
```
Epoch [5] Step 8 | encoder_conv1: 0.052314 | encoder_last: 0.003101 | 
decoder_first: 0.004522 | decoder_output: 0.002881 | 
Hint: ⚠️浅层/深层>10x，轻微消失
```

**判断：**
- 如果显示 **✓全层>1e-4** → 梯度正常，问题不在这
- 如果显示 **⚠️消失** → 梯度确实有问题，需要修改架构/激活

**关键信息 3：Loss 平台期诊断**（Epoch 11+ 每 10 个 epoch）
```
[诊断] Loss平台期检查:
  过去5个epoch loss: ['0.612534', '0.611893', '0.612105', '0.611567', '0.612001']
  方差: 8.23e-07, 5-epoch改进: 0.000533
  ⚠️ Loss已进入平台期
```

**判断：**
- 方差 < 1e-6 **且** 改进 < 0.0001 → Loss 确实停滞了
- 这时需要调整超参或检查数据

---

## 🔧 根据诊断结果修复

### **情况 1：数据质量有问题** 
```bash
诊断脚本输出:
❌ 掩码包含 256 个唯一值
❌ 平均前景比例仅 0.50%
```

**修复方案：**

#### 子问题 1a：掩码是灰度图，不是二值图像
```python
# 脚本修复：在数据加载前二值化
from PIL import Image
import numpy as np

# 批量修复所有掩码
import glob
for mask_file in glob.glob('masks/*.png'):
    mask = Image.open(mask_file)
    mask_arr = np.array(mask)
    # 二值化：>127 设为 255，<=127 设为 0
    mask_arr = (mask_arr > 127).astype(np.uint8) * 255
    Image.fromarray(mask_arr).save(mask_file)
    print(f"修复: {mask_file}")
```

#### 子问题 1b：前景比例过低（严重类不平衡）
**方案 A：在损失函数中加权**
```python
# 修改 train_service.py 中的 DiceBCELoss 初始化
# 计算前景比例
num_foreground = sum([(mask_arr > 127).sum() for mask_arr in all_masks])
num_background = sum([(mask_arr <= 127).sum() for mask_arr in all_masks])
pos_weight = num_background / (num_foreground + 1e-6)  # 例如 9.0

criterion = DiceBCELoss(pos_weight=pos_weight).to(device)
```

**方案 B：调整 Loss 权重**
```python
# 在 DiceBCELoss 中增加 Dice 权重（更多关注形状）
# 改为 0.2 * BCE + 0.8 * (1 - Dice)
self.dice_weight = 0.8  # 从 0.7 改为 0.8
```

---

### **情况 2：学习率过小**
```
观察到：
- 前 3 个 epoch Loss 从 0.8 降到 0.65
- 之后几乎不变，几乎是平线
```

**修复方案：增加学习率**

**修改 train_controller.py：**
```python
# 第 45 行左右，修改默认 lr
lr: float = Form(default=0.0005, description="学习率（推荐 0.0005-0.001）"),
```

或前端直接提交更大的学习率。

**修改 train_service.py（第 390 行左右）：**
```python
optimizer = optim.Adam(
    model.parameters(), 
    lr=0.0005,  # ← 从 0.0001 改为 0.0005
    betas=(0.9, 0.999),
    weight_decay=0
)

scheduler = optim.lr_scheduler.StepLR(
    optimizer, 
    step_size=20,  # ← 从 10 改为 20（给更多时间学习）
    gamma=0.8
)
```

**测试新学习率：**
```bash
# 用新参数重新训练
# 在前端输入 epochs=50, lr=0.0005
```

**预期结果：** Loss 应该继续下降，每个 epoch 改进 0.01-0.05

---

### **情况 3：Loss 权重不合理**
```
观察到：
Train Loss: 0.6123 (BCE: 0.2340, Dice: 0.4350)
         ↑ Dice 这个太高了！
```

**根本原因：** 当前权重是 `0.3*BCE + 0.7*Dice`，Dice 项权重太高

**修复方案：降低 Dice 权重**

**修改 train_service.py（第 193 行左右）：**
```python
# 在 DiceBCELoss 的 forward 方法中
# 当前: loss = 0.3 * bce_loss + 0.7 * (1 - dice_score)
# 改为:
bce_weight = 0.5  # 增加 BCE 权重
dice_weight = 0.5  # 降低 Dice 权重
loss = bce_weight * bce_loss + dice_weight * (1 - dice_score)
```

或者完全修改损失函数定义：
```python
# 查找 DiceBCELoss 类（大约 193 行）
class DiceBCELoss(nn.Module):
    def forward(self, pred, target):
        # ...
        # 改成:
        loss = 0.4 * bce_loss + 0.6 * (1 - dice_score)  # 更平衡
        return loss
```

---

### **情况 4：模型过大**
```
条件：
- 数据集 < 50 张图像
- Loss 卡在 0.6，Val Dice 只有 0.40
```

**诊断：** UNet_Transformer 有 31M 参数，对小数据集来说太大了

**修复方案：使用更小的原始 UNet**

**修改 train_service.py（第 371 行）：**
```python
# 改为：
model = UNet(in_channels=3, out_channels=1).to(device)  # 8.5M 参数，更小
# 而不是:
# model = UNet_Transformer(in_channels=3, out_channels=1).to(device)
```

**预期结果：** 训练速度更快，更容易收敛

---

## 📊 诊断矩阵（一键找到问题）

| Loss 卡在 | Dice 值 | BCE 值 | Dice Loss | 梯度 | 最可能原因 | 快速修复 |
|---------|--------|--------|----------|------|----------|--------|
| 0.60 | 0.40 | 0.30 | 高 | ✓正常 | Loss 权重不当 | 降低 Dice 权重到 0.5 |
| 0.65 | 0.35 | 0.50 | 低 | ✓正常 | BCE 未稳定 | 增加 lr 或 epoch |
| 0.70 | 0.42 | 0.60 | 中 | ⚠️轻微消失 | 学习率太小 | lr 0.0001→0.0005 |
| 0.55 | 0.43 | 0.20 | 高 | ✓正常 | 前景比例低 | 检查掩码，加权 |
| 0.80+ | 0.38 | 高 | 低 | ❌消失 | 梯度问题 | 改模型架构 |

---

## 🚀 推荐的调试优先级

### **第 1 优先级（最可能）：数据质量检查**
```bash
python diagnose_dataset.py your_dataset_path
```
- 检查结果是否有 ⚠️ 标记
- 如有，先修复数据

### **第 2 优先级：增加学习率和 epoch**
```python
# 试试更激进的超参
epochs = 50  # 从 10 改为 50
lr = 0.0005  # 从 0.0001 改为 0.0005
```

### **第 3 优先级：调整 Loss 权重**
```python
# 平衡 BCE 和 Dice
bce_weight = 0.5
dice_weight = 0.5
```

### **第 4 优先级：回退到更小的模型**
```python
# 如果数据集很小，用 UNet 而不是 UNet_Transformer
model = UNet(...)
```

---

## 📞 采集信息进行深度诊断

如果按上述步骤仍未解决，请运行以下命令并贴出结果：

```bash
# 1. 诊断数据集
python diagnose_dataset.py path/to/dataset

# 2. 在训练前 20 个 epoch 保存日志
# （在后端启动时会自动打印，复制粘贴前 20 个 epoch 的日志）

# 3. 检查梯度信息
# （观察日志中的 GradNorm 和 encoder_last 等值）
```

**贴出这些内容给我，能快速定位问题。**

---

**最后更新：2026年1月22日**
