# OCTA前端 RS-Unet3+ 支持更新

## 📋 更新概述

对OCTA Web前端进行了全面升级，添加了 RS-Unet3+ 模型的完整支持，包括模型选择、权重过滤、参数自适应等功能。

**更新时间**：2026年1月17日  
**更新范围**：前端Vue3框架 (HomeView.vue, TrainView.vue)  
**兼容性**：保持向后兼容，现有U-Net和FCN模型继续正常工作

---

## ✨ 新增功能

### 1️⃣ **图像分割页面 (HomeView.vue)**

#### 模型选择器增强
```vue
<!-- 新增模型选项 -->
<el-option label="RS-Unet3+" value="rs_unet3_plus">
  <span>RS-Unet3+ <el-tag type="success" size="small">权重可用</el-tag></span>
</el-option>
```

- ✅ 模型下拉菜单添加 RS-Unet3+ 选项
- ✅ 条件标签显示权重可用性
- ✅ 模型切换时自动清空权重选择

#### 权重列表智能过滤
```javascript
// 新增函数：根据模型类型过滤权重
const filterWeightByModel = (modelArch) => {
  // 支持识别：
  // - RS-Unet3+: 'rs_unet3'、'rs-unet3'
  // - U-Net: 'unet'、'u-net' (排除RS变体)
  // - FCN: 'fcn'
  return availableWeights.value.filter(w => {
    // 权重名称/路径智能匹配
  })
}
```

- 🔍 权重列表自动按模型类型过滤
- 🚫 无匹配权重时禁用下拉框
- 📊 计算属性 `filteredWeights` 实时过滤
- ⚠️ 计算属性 `hasRS_Unet3PlusWeight` 检测RS权重

#### 用户交互反馈
```javascript
const handleModelChange = (newModel) => {
  // 自动消息提示
  if (newModel === 'rs_unet3_plus') {
    if (hasRS_Unet3PlusWeight.value) {
      ElMessage.info('已切换为 RS-Unet3+ 模型，推荐使用专用权重')
    } else {
      ElMessage.warning('暂无 RS-Unet3+ 权重，使用默认权重可能效果不佳')
    }
  }
}
```

- 📢 模型切换时自动弹出提示消息
- 🎯 根据权重可用性给出针对性建议
- 🔄 支持 U-Net、FCN、RS-Unet3+ 的个性化提示

---

### 2️⃣ **模型训练页面 (TrainView.vue)**

#### 模型架构选择器（新增）
```vue
<!-- 新增模型选择框 -->
<el-form-item label="模型架构：">
  <el-select 
    v-model="trainParams.model_arch" 
    @change="handleModelArchChange"
    style="width: 150px"
  >
    <el-option label="U-Net" value="unet"></el-option>
    <el-option label="RS-Unet3+" value="rs_unet3_plus">
      <span>RS-Unet3+ <el-tag type="success" size="small">推荐</el-tag></span>
    </el-option>
    <el-option label="FCN" value="fcn"></el-option>
  </el-select>
</el-form-item>
```

- 🎨 新增模型选择下拉框
- 🏷️ RS-Unet3+ 标记为"推荐"
- 📍 位于参数配置区顶部，优先级最高

#### 权重衰减参数（新增）
```javascript
trainParams = reactive({
  model_arch: 'unet',      // ✨ 新增
  epochs: 10,
  lr: 0.001,
  weight_decay: 0.0001,    // ✨ 新增（仅RS-Unet3+需要）
  batch_size: 4
})
```

- 🔧 添加 `weight_decay` 响应式参数
- 🎛️ 前端UI支持调整权重衰减值（0-1，精度4位）
- 📤 传递给后端 `/train/upload-dataset` 接口

#### 参数自适应系统
```javascript
const handleModelArchChange = (modelArch) => {
  if (modelArch === 'rs_unet3_plus') {
    // RS-Unet3+ 推荐配置
    trainParams.epochs = 200      // OCTA血管详细特征需更多轮数
    trainParams.lr = 0.0001        // 更小学习率，精细调参
    trainParams.weight_decay = 0.0001
    trainParams.batch_size = 4
  } else if (modelArch === 'unet') {
    // U-Net 标准配置
    trainParams.epochs = 50
    trainParams.lr = 0.001
    trainParams.weight_decay = 0.0001
    trainParams.batch_size = 4
  } else if (modelArch === 'fcn') {
    // FCN 配置
    trainParams.epochs = 30
    trainParams.lr = 0.0005
    trainParams.weight_decay = 0.0001
    trainParams.batch_size = 8
  }
}
```

- 🤖 选择RS-Unet3+自动配置最优参数
- ⚡ 每个模型架构有预设的最佳参数组合
- 🎯 参数设置基于医学影像分割的最佳实践

#### RS-Unet3+ 配置提示（新增）
```vue
<!-- 条件提示框 -->
<el-alert 
  v-if="trainParams.model_arch === 'rs_unet3_plus'"
  type="info"
>
  <strong>RS-Unet3+ 训练配置：</strong>
  已自动配置最优参数 - 训练轮数: 200，学习率: 0.0001，权重衰减: 0.0001。
  建议数据集包含至少 200+ 张OCTA图像以获得最佳效果。
</el-alert>
```

- 📝 动态提示框显示RS-Unet3+配置信息
- 💡 提供数据集规模建议（200+张）
- 🔄 参数变更时自动更新提示

#### 后端参数传递增强
```javascript
const formData = new FormData()
formData.append('file', selectedFile.value)
formData.append('model_arch', trainParams.model_arch)    // ✨ 新增
formData.append('epochs', trainParams.epochs)
formData.append('lr', trainParams.lr)
formData.append('weight_decay', trainParams.weight_decay) // ✨ 新增
formData.append('batch_size', trainParams.batch_size)
```

- 📤 新增 `model_arch` 参数传递给后端
- 🔗 新增 `weight_decay` 参数传递
- 📊 后端 `/train/upload-dataset` 接收完整参数

#### 损失曲线标题动态化
```javascript
const renderLossCurve = (trainLosses, valLosses) => {
  let modelName = 'U-Net'
  if (trainParams.model_arch === 'rs_unet3_plus') {
    modelName = 'RS-Unet3+'
  } else if (trainParams.model_arch === 'fcn') {
    modelName = 'FCN'
  }
  
  const option = {
    title: {
      text: `${modelName} 训练与验证损失曲线`,  // 动态标题
      // ...
    }
  }
}
```

- 📊 ECharts图表标题根据模型动态变化
- 🎨 不同模型显示对应的模型名称

---

## 🔄 技术架构

### 数据流向

```
用户选择模型
    ↓
权重列表自动过滤
    ↓
模型绑定参数自动调整（TrainView）
    ↓
参数传递到后端 API
    ↓
后端路由到对应的模型训练器
    ↓
返回训练结果和评估指标
```

### 权重识别规则

| 模型 | 识别关键词 | 示例 |
|-----|---------|------|
| RS-Unet3+ | `rs_unet3`、`rs-unet3` | `rs_unet3plus_v1.pth` |
| U-Net | `unet`、`u-net` (排除RS变体) | `unet_octa.pth` |
| FCN | `fcn` | `fcn_segmentation.pth` |

---

## 🚀 前后端集成

### 前端调用后端接口

#### 1. 分割接口 (HomeView.vue)
```javascript
// POST /segment-octa/
const response = await axios.post('http://127.0.0.1:8000/segment-octa/', formData, {
  headers: { 'Content-Type': 'multipart/form-data' }
})
// 参数：model_type, weight, file
```

#### 2. 训练接口 (TrainView.vue)
```javascript
// POST /train/upload-dataset
const response = await axios.post(
  'http://127.0.0.1:8000/train/upload-dataset',
  formData,
  { headers: { 'Content-Type': 'multipart/form-data' } }
)
// 参数：file, model_arch, epochs, lr, weight_decay, batch_size
```

### 后端需要的支持

**✅ 分割端点** (`/segment-octa/`)
- 支持 `model_type` 参数：'unet'、'fcn'、'rs_unet3_plus'
- 当前实现：已支持（见 main.py）

**⚠️ 训练端点** (`/train/upload-dataset`)
- 需要支持新参数：`model_arch`、`weight_decay`
- 需要路由到不同的模型训练器
- **状态**：需要更新

---

## 📝 修改文件清单

### HomeView.vue
| 修改项 | 说明 |
|-------|------|
| 导入 `computed` | Vue 3 响应式计算属性 |
| 模型下拉菜单 | 添加 RS-Unet3+ 选项和标签 |
| `filterWeightByModel()` 函数 | 新增权重过滤函数 |
| `filteredWeights` 计算属性 | 根据模型自动过滤权重 |
| `hasRS_Unet3PlusWeight` 计算属性 | 检测RS权重可用性 |
| `handleModelChange()` 函数 | 模型切换时的交互逻辑 |
| 权重下拉菜单 | 使用过滤后的权重列表、禁用状态、空状态提示 |

### TrainView.vue
| 修改项 | 说明 |
|-------|------|
| 响应式数据 `model_arch` | 新增模型架构选择变量 |
| 响应式数据 `weight_decay` | 新增权重衰减参数 |
| 参数表单 | 添加模型架构选择器、增加权重衰减输入框 |
| `handleModelArchChange()` 函数 | 模型变更时自动调整参数 |
| RS-Unet3+ 提示框 | 条件渲染的配置建议提示 |
| FormData 构建 | 新增 `model_arch` 和 `weight_decay` 参数 |
| 损失曲线渲染 | 图表标题根据模型动态变化 |

---

## ✔️ 验证清单

- [x] HomeView.vue 语法检查通过
- [x] TrainView.vue 语法检查通过
- [x] 权重过滤逻辑完整
- [x] 参数自适应系统完成
- [x] 模型切换消息提示实现
- [x] 后端参数传递就绪
- [ ] **待做**：启动前端开发服务器测试
- [ ] **待做**：检查后端是否支持新参数
- [ ] **待做**：端到端功能测试

---

## 🔧 使用指南

### 分割工作流
1. 选择模型：U-Net / FCN / RS-Unet3+
2. 权重列表自动过滤（仅显示对应模型的权重）
3. 选择权重文件
4. 上传OCTA图像
5. 点击"开始分割"

### 训练工作流
1. 选择模型架构：U-Net / RS-Unet3+ / FCN
2. 参数自动配置（也可手动调整）
3. 上传数据集ZIP包
4. 点击"开始训练"
5. 查看训练进度和结果

---

## 🎯 关键参数对比

| 参数 | U-Net | RS-Unet3+ | FCN |
|-----|-------|-----------|-----|
| 推荐轮数 | 50 | **200** | 30 |
| 学习率 | 0.001 | **0.0001** | 0.0005 |
| 权重衰减 | 0.0001 | 0.0001 | 0.0001 |
| 批次大小 | 4 | 4 | **8** |
| 最小数据集 | 50张 | 200张 | 50张 |
| 推荐使用场景 | 通用 | **OCTA血管** | 快速原型 |

---

## 💡 最佳实践

### 使用RS-Unet3+时的建议
1. **数据集规模**：至少200+张OCTA图像，1000+张为佳
2. **训练时间**：200轮训练预计2-4小时（CPU）
3. **硬件要求**：显存8GB+（推荐用GPU），或CPU内存16GB+
4. **数据预处理**：确保图像归一化和数据增强
5. **权重保存**：定期保存最佳模型权重

### 权重管理
- 训练完成后的权重保存在 `results/` 目录
- 权重文件需包含 'rs_unet3' 关键词以被正确识别
- 支持在 `/model/weights` 端点查看所有可用权重

---

## 📞 故障排除

### 权重列表为空
**原因**：后端 `/model/weights` 接口未返回数据  
**解决**：检查后端是否正确初始化权重列表

### 模型切换无反应
**原因**：计算属性未正确响应  
**解决**：查看浏览器控制台是否有错误信息

### 训练参数未自动调整
**原因**：`handleModelArchChange` 函数未被触发  
**解决**：确保模型选择器的 `@change` 事件绑定正确

### 后端返回参数错误
**原因**：后端未支持新参数 `model_arch`、`weight_decay`  
**解决**：更新后端训练接口实现

---

## 📚 相关文件

- **前端路由**：[src/router/index.js](octa_frontend/src/router/index.js)
- **主页分割**：[src/views/HomeView.vue](octa_frontend/src/views/HomeView.vue)
- **训练页面**：[src/views/TrainView.vue](octa_frontend/src/views/TrainView.vue)
- **后端API**：[octa_backend/main.py](octa_backend/main.py)
- **后端训练**：[octa_backend/service/train_rs_unet3_plus.py](octa_backend/service/train_rs_unet3_plus.py)

---

**文档版本**：v1.0  
**最后更新**：2026年1月17日  
**作者**：GitHub Copilot AI
