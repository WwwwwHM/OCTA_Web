# 训练界面差异化 UI 实现

## 📋 概述

已成功为 U-Net 和 RS-Unet3+ 两个模型实现差异化的训练界面。U-Net 保持简洁，RS-Unet3+ 展示高级专业界面。

---

## ✨ 主要变更

### 1️⃣ **动态模型选择卡片**

替代了之前的 `<el-select>` 下拉框，改为视觉化的卡片选择器：

```vue
<!-- 模型选择区 -->
<div class="model-cards">
  <!-- U-Net 卡片 -->
  <div class="model-card active" @click="...">
    <div class="card-title">U-Net</div>
    <div class="card-badge">通用训练</div>
    <div class="card-desc">经典分割架构，适合通用任务</div>
  </div>

  <!-- RS-Unet3+ 卡片（带 Pro 徽章） -->
  <div class="model-card pro-card active" @click="...">
    <div class="pro-badge">
      <el-tag type="success" size="small">Pro</el-tag>
    </div>
    <div class="card-title">RS-Unet3+</div>
    <div class="card-badge">专用训练</div>
    <div class="card-desc">融合分割与注意力机制，OCTA微血管专用</div>
  </div>
</div>
```

**特点：**
- ✅ 鼠标悬停时蓝色高亮
- ✅ 选中时背景色变化
- ✅ RS-Unet3+ 卡片有绿色边框和 Pro 徽章
- ✅ 清晰的模型描述

---

### 2️⃣ **U-Net 简化界面**

显示基础参数，专为通用任务设计：

```vue
<el-form 
  v-if="trainParams.model_arch === 'unet'" 
  class="unet-form"
>
  <div class="form-section">
    <h4>基础参数配置</h4>
    <p>适用于通用医学图像分割任务，参数简洁易用</p>
    
    <!-- 参数 -->
    <el-form-item label="训练轮数：">
      <el-input-number v-model="trainParams.epochs" />
      <span class="param-hint">推荐: 50</span>
    </el-form-item>
    
    <el-form-item label="学习率：">
      <el-input-number v-model="trainParams.lr" />
      <span class="param-hint">推荐: 0.001</span>
    </el-form-item>
    
    <el-form-item label="批次大小：">
      <el-input-number v-model="trainParams.batch_size" />
      <span class="param-hint">推荐: 4</span>
    </el-form-item>
    
    <el-alert type="info">
      U-Net 是一个经典且稳定的分割架构...
    </el-alert>
  </div>
</el-form>
```

**参数列表：**
- Epochs（训练轮数）
- Learning Rate（学习率）
- Batch Size（批次大小）

---

### 3️⃣ **RS-Unet3+ 高级界面**

展示专业的 OCTA 优化参数：

```vue
<el-form 
  v-if="trainParams.model_arch === 'rs_unet3_plus'" 
  class="rs-unet-form"
>
  <!-- 核心参数 -->
  <div class="form-section">
    <div class="section-header">
      <div>
        <h4>核心参数</h4>
        <p>优化用于 OCTA 微血管分割，已自动配置最优值</p>
      </div>
      <el-tag type="success">自适应</el-tag>
    </div>
    <!-- Epochs, Learning Rate, Batch Size ... -->
  </div>

  <!-- 高级选项 -->
  <div class="form-section">
    <h4>高级选项</h4>
    <p>针对 OCTA 图像的特化参数</p>
    
    <!-- Attention Weight 滑块 -->
    <el-form-item label="注意力权重：">
      <el-slider
        v-model="trainParams.attention_weight"
        :min="0" :max="1" :step="0.1"
        show-stops
      ></el-slider>
      <span class="param-hint">
        调整 Split-Attention 机制强度（0=禁用，1=最强）
      </span>
    </el-form-item>

    <!-- Deep Supervision 开关 -->
    <el-form-item label="深度监督：">
      <el-switch 
        v-model="trainParams.deep_supervision"
        active-text="启用" inactive-text="禁用"
      ></el-switch>
      <span class="param-hint">
        在多个解码层应用监督信号，改进微血管分割精度
      </span>
    </el-form-item>

    <!-- Loss Function 只读 -->
    <el-form-item label="损失函数：">
      <el-input 
        v-model="trainParams.loss_function" 
        disabled
      ></el-input>
      <span class="param-hint">
        已锁定为 Lovasz-Softmax（最优，不可修改）
      </span>
    </el-form-item>
  </div>

  <el-alert type="success">
    RS-Unet3+ 专用优化：融合了 Split-Attention 机制...
  </el-alert>
</el-form>
```

**参数列表：**

| 参数 | 类型 | 说明 |
|-----|------|------|
| Epochs | Input | 推荐: 200 |
| Learning Rate | Input | 推荐: 1e-4 |
| Batch Size | Input | 推荐: 4 |
| **Attention Weight** | **Slider** | **0-1，控制注意力强度** |
| **Deep Supervision** | **Switch** | **启用/禁用深度监督** |
| **Loss Function** | **Read-only** | **锁定为 Lovasz-Softmax** |

---

## 🎨 样式设计

### 模型选择卡片样式

```css
.model-card {
  padding: 20px;
  border: 2px solid #dcdfe6;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.model-card:hover {
  border-color: #409eff;
  box-shadow: 0 2px 12px 0 rgba(64, 158, 255, 0.1);
}

.model-card.active {
  border-color: #409eff;
  background-color: #f0f9ff;
  box-shadow: 0 2px 12px 0 rgba(64, 158, 255, 0.15);
}

.model-card.pro-card {
  border-color: #67c23a;  /* 绿色边框 */
}

.model-card.pro-card.active {
  border-color: #67c23a;
  background-color: #f0f9ff;
  box-shadow: 0 2px 12px 0 rgba(103, 194, 58, 0.15);
}
```

### 参数表单样式

```css
.unet-form,
.rs-unet-form {
  background-color: white;
  padding: 20px;
  border-radius: 4px;
  border: 1px solid #ebeef5;
}

.form-section {
  margin-bottom: 25px;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 15px;
}

.param-hint {
  display: block;
  font-size: 12px;
  color: #909399;
  margin-top: 5px;
}
```

---

## 📝 脚本逻辑更新

### 响应式数据扩展

```javascript
const trainParams = reactive({
  model_arch: 'unet',
  epochs: 10,
  lr: 0.001,
  weight_decay: 0.0001,
  batch_size: 4,
  
  // RS-Unet3+ 特定参数
  attention_weight: 0.8,      // ✅ 新增
  deep_supervision: true,      // ✅ 新增
  loss_function: 'Lovasz-Softmax'  // ✅ 新增
})
```

### 模型切换自动配置

```javascript
const handleModelArchChange = (modelArch) => {
  if (modelArch === 'rs_unet3_plus') {
    trainParams.epochs = 200
    trainParams.lr = 0.0001
    trainParams.weight_decay = 0.0001
    trainParams.batch_size = 4
    
    // RS-Unet3+ 特定参数设置
    trainParams.attention_weight = 0.8      // ✅ 自动设置
    trainParams.deep_supervision = true     // ✅ 自动设置
    trainParams.loss_function = 'Lovasz-Softmax'  // ✅ 自动设置
  } else if (modelArch === 'unet') {
    trainParams.epochs = 50
    trainParams.lr = 0.001
    trainParams.weight_decay = 0.0001
    trainParams.batch_size = 4
  }
}
```

### 表单提交参数

```javascript
const startTraining = async () => {
  const formData = new FormData()
  formData.append('model_arch', trainParams.model_arch)
  formData.append('epochs', trainParams.epochs)
  formData.append('lr', trainParams.lr)
  formData.append('batch_size', trainParams.batch_size)
  
  // 如果是 RS-Unet3+ 模型，添加特定参数
  if (trainParams.model_arch === 'rs_unet3_plus') {
    formData.append('attention_weight', trainParams.attention_weight)
    formData.append('deep_supervision', trainParams.deep_supervision)
    formData.append('loss_function', trainParams.loss_function)
  }
  
  // 发送训练请求...
}
```

---

## 🔄 用户体验流程

### 切换到 U-Net

```
1. 用户点击 U-Net 卡片
2. 界面显示 U-Net 简化表单
3. 参数自动设置为推荐值（Epochs=50, LR=0.001）
4. 显示提示："U-Net 是一个经典且稳定的分割架构..."
5. 用户可修改基础参数后提交
```

### 切换到 RS-Unet3+

```
1. 用户点击 RS-Unet3+ Pro 卡片（带绿色边框和徽章）
2. 界面显示 RS-Unet3+ 高级表单
3. 参数自动设置为最优值：
   - Epochs: 200
   - Learning Rate: 0.0001
   - Batch Size: 4
   - Attention Weight: 0.8
   - Deep Supervision: ✅ 启用
   - Loss Function: Lovasz-Softmax（只读）
4. 显示警告："建议数据集包含 200+ 张高质量 OCTA 图像"
5. 用户可调整高级参数（如注意力权重）后提交
```

---

## 📊 建议的后端参数处理

后端应处理这些新参数：

```python
# 对于 RS-Unet3+ 训练
{
  "model_arch": "rs_unet3_plus",
  "epochs": 200,
  "lr": 0.0001,
  "batch_size": 4,
  "attention_weight": 0.8,      # ✅ 新参数
  "deep_supervision": True,      # ✅ 新参数
  "loss_function": "lovasz_softmax"  # ✅ 新参数
}

# 对于 U-Net 训练（不包含 RS-Unet3+ 特定参数）
{
  "model_arch": "unet",
  "epochs": 50,
  "lr": 0.001,
  "batch_size": 4
}
```

---

## ✅ 验证清单

- ✅ 前端构建成功（npm run build）
- ✅ 模型选择卡片实现并样式化
- ✅ U-Net 简化表单条件渲染
- ✅ RS-Unet3+ 高级表单条件渲染
- ✅ Pro 徽章和绿色边框显示
- ✅ 参数自动填充逻辑实现
- ✅ 表单提交参数包含新的 RS-Unet3+ 参数
- ✅ 响应式设计适配移动端

---

## 🚀 下一步建议

1. **后端更新**
   - 更新训练 API 以接收新的 RS-Unet3+ 参数
   - 在训练脚本中使用这些参数配置模型

2. **功能扩展**
   - 保存用户的参数偏好
   - 添加"使用预设"功能（快速切换推荐配置）
   - 添加参数验证和错误提示

3. **可视化增强**
   - 添加参数说明工具提示
   - 显示参数对性能的影响估计
   - 实时显示训练进度和模型性能

---

## 📁 修改文件

- `octa_frontend/src/views/TrainView.vue` - 完整的 UI 和逻辑更新

**修改时间：** 2026年1月20日  
**修改者：** GitHub Copilot AI  
**状态：** ✅ 完成并验证通过（前端构建成功）
