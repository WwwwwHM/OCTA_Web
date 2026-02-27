# OCTA 前端路由与导航系统指南

## 📋 概述

本文档说明OCTA图像分割平台前端的路由与导航系统，特别是RS-Unet3+模型的专用训练路由配置。

---

## 🗺️ 路由配置（`src/router/index.js`）

### 路由列表

| 路径 | 名称 | 组件 | 说明 | Meta配置 |
|------|------|------|------|---------|
| `/` | home | HomeView | 首页（图像分割） | `{ title: '首页' }` |
| `/history` | history | HistoryView (懒加载) | 历史记录 | `{ title: '历史记录' }` |
| `/train` | Train | TrainView | 通用训练页 | `{ title: '模型训练', subtitle: '通用训练（U-Net/FCN）' }` |
| `/train/rs-unet3-plus` | TrainRSUnet3Plus | TrainView | RS-Unet3+专用训练 | `{ title: 'RS-Unet3+训练', subtitle: 'OCTA专用训练（血管+FAZ）', icon: 'Science', modelArch: 'rs_unet3_plus' }` |
| `/file-manager` | FileManager | FileManager (懒加载) | 文件管理 | `{ title: '文件管理' }` |
| `/about` | about | AboutView (懒加载) | 关于 | `{ title: '关于' }` |

### 关键特性

#### 1. **RS-Unet3+ 专用路由**
```javascript
{
  path: '/train/rs-unet3-plus',
  name: 'TrainRSUnet3Plus',
  component: TrainView,  // 复用TrainView组件
  meta: { 
    title: 'RS-Unet3+训练',
    subtitle: 'OCTA专用训练（血管+FAZ）',
    icon: 'Science',
    modelArch: 'rs_unet3_plus'  // 关键：自动设置模型架构
  }
}
```

- **组件复用**：与通用训练页使用同一个 `TrainView.vue` 组件
- **自动配置**：通过 `meta.modelArch` 自动切换到 RS-Unet3+ 模型
- **最优参数**：进入页面时自动应用RS-Unet3+推荐参数（200轮，lr=0.0001）

#### 2. **路由meta的作用**

`meta.modelArch` 字段决定了TrainView组件的初始模型架构：
- `/train` → 无 `meta.modelArch` → 使用全局状态或默认unet
- `/train/rs-unet3-plus` → `meta.modelArch = 'rs_unet3_plus'` → 强制使用RS-Unet3+

---

## 🧭 导航栏配置（`src/App.vue`）

### 导航结构

```
首页 (/)
历史记录 (/history)
模型训练 (子菜单)
  ├─ 通用训练（U-Net/FCN） (/train)
  └─ RS-Unet3+专用训练 (/train/rs-unet3-plus)
文件管理 (/file-manager)
关于 (/about)
```

### 子菜单实现

```vue
<el-sub-menu index="train-menu">
  <template #title>
    <el-icon><VideoPlay /></el-icon>
    <span>模型训练</span>
  </template>
  <el-menu-item index="/train">
    <span>通用训练（U-Net/FCN）</span>
  </el-menu-item>
  <el-menu-item 
    index="/train/rs-unet3-plus"
    :disabled="!rsUnet3PlusAvailable"
  >
    <el-icon><Science /></el-icon>
    <span>RS-Unet3+专用训练</span>
    <el-tooltip 
      v-if="!rsUnet3PlusAvailable"
      content="后端未部署RS-Unet3+模型" 
      placement="right"
    >
      <el-icon class="disabled-hint"><WarningFilled /></el-icon>
    </el-tooltip>
  </el-menu-item>
</el-sub-menu>
```

### 关键特性

#### 1. **动态禁用状态**
- 通过 `rsUnet3PlusAvailable` 控制RS-Unet3+菜单项是否可用
- 禁用时显示警告图标和提示："后端未部署RS-Unet3+模型"

#### 2. **图标设计**
- 通用训练：无特殊图标（继承父菜单的 VideoPlay）
- RS-Unet3+：Science图标（科学烧杯），突出专业性

#### 3. **导航激活逻辑**
- Element Plus自动处理路由激活状态（通过 `router` 属性）
- 当前路由匹配时，对应菜单项高亮（蓝色）

---

## 🌐 全局状态管理（`src/composables/useGlobalState.js`）

### 设计理念

- **无Vuex/Pinia依赖**：使用Vue3 Composition API的响应式系统
- **轻量级**：仅管理必要的全局状态（模型架构、功能可用性）
- **组件间共享**：多个组件可读写同一状态

### API文档

#### 导出的函数

```javascript
import { useGlobalState } from '@/composables/useGlobalState'

const {
  globalModelArch,            // 只读ref：当前全局模型架构
  rsUnet3PlusAvailable,       // 只读ref：RS-Unet3+是否可用
  setGlobalModelArch,         // 函数：设置全局模型架构
  getGlobalModelArch,         // 函数：获取当前模型架构
  setRsUnet3PlusAvailable,    // 函数：设置RS-Unet3+可用性
  getRsUnet3PlusAvailable,    // 函数：获取RS-Unet3+可用性
  getModelDisplayName         // 函数：获取模型显示名称
} = useGlobalState()
```

#### 使用示例

**在App.vue中控制菜单禁用状态：**
```vue
<script setup>
import { useGlobalState } from '@/composables/useGlobalState'
const { rsUnet3PlusAvailable } = useGlobalState()
</script>

<template>
  <el-menu-item 
    index="/train/rs-unet3-plus"
    :disabled="!rsUnet3PlusAvailable"
  >
    RS-Unet3+专用训练
  </el-menu-item>
</template>
```

**在TrainView.vue中读写模型架构：**
```vue
<script setup>
import { useGlobalState } from '@/composables/useGlobalState'
const { setGlobalModelArch, getGlobalModelArch } = useGlobalState()

// 页面初始化时读取全局状态
onMounted(() => {
  const savedArch = getGlobalModelArch()
  trainParams.model_arch = savedArch
})

// 用户切换模型时同步到全局状态
watch(() => trainParams.model_arch, (newArch) => {
  setGlobalModelArch(newArch)
})
</script>
```

**在HomeView.vue中检查模型可用性：**
```vue
<script setup>
import { useGlobalState } from '@/composables/useGlobalState'
const { setRsUnet3PlusAvailable } = useGlobalState()

// 检查后端模型可用性（例如API检测）
async function checkBackendModels() {
  try {
    const response = await axios.get('http://127.0.0.1:8000/models/available')
    setRsUnet3PlusAvailable(response.data.rs_unet3_plus)
  } catch (error) {
    setRsUnet3PlusAvailable(false)
  }
}
</script>
```

---

## 🔄 工作流程

### 用户使用流程

#### 场景1：通用训练页（多模型选择）

1. 用户点击导航栏"模型训练"子菜单 → "通用训练（U-Net/FCN）"
2. 路由跳转到 `/train`
3. TrainView组件加载：
   - `route.meta.modelArch` 为 undefined
   - 从全局状态读取上次选择的模型架构（或默认unet）
   - 用户可手动切换下拉菜单（U-Net / RS-Unet3+ / FCN）
4. 用户上传数据集、配置参数、开始训练

#### 场景2：RS-Unet3+专用训练页

1. 用户点击导航栏"模型训练"子菜单 → "RS-Unet3+专用训练"
2. 路由跳转到 `/train/rs-unet3-plus`
3. TrainView组件加载：
   - 检测到 `route.meta.modelArch = 'rs_unet3_plus'`
   - 强制设置 `trainParams.model_arch = 'rs_unet3_plus'`
   - 自动应用RS-Unet3+最优参数（200轮，lr=0.0001）
   - 显示蓝色提示框："RS-Unet3+ 训练配置：已自动配置最优参数..."
4. 用户上传数据集、直接开始训练（参数已优化）

### 技术流程图

```
┌────────────────┐
│  用户点击菜单   │
└───────┬────────┘
        │
        ├─── /train ─────────────────────┐
        │                                 │
        │  ┌─────────────────────────┐   │
        │  │ TrainView.vue           │   │
        │  │ - 无 meta.modelArch     │   │
        │  │ - 使用全局状态或默认值  │   │
        │  │ - 用户可手动切换模型    │   │
        │  └─────────────────────────┘   │
        │                                 │
        └─── /train/rs-unet3-plus ───────┤
                                          │
           ┌─────────────────────────┐    │
           │ TrainView.vue           │    │
           │ - meta.modelArch存在    │    │
           │ - 强制设置RS-Unet3+     │    │
           │ - 自动应用最优参数      │    │
           └─────────────────────────┘    │
                                          │
                    ↓                     │
           ┌─────────────────────────┐    │
           │ useGlobalState          │    │
           │ - 同步模型架构          │    │
           │ - 跨组件状态共享        │    │
           └─────────────────────────┘    │
                                          │
                    ↓                     │
           ┌─────────────────────────┐    │
           │ App.vue 导航栏          │    │
           │ - 菜单激活状态更新      │    │
           │ - 禁用状态动态控制      │    │
           └─────────────────────────┘    │
                                          │
                    ↓                     │
           ┌─────────────────────────┐    │
           │ 后端API /train/...      │    │
           │ - 接收model_arch参数    │    │
           │ - 执行对应模型训练      │    │
           └─────────────────────────┘    │
                                          │
└─────────────────────────────────────────┘
```

---

## 🛠️ 开发指南

### 如何添加新的模型架构

假设要添加"FCN-Plus"模型：

#### 1. 修改路由配置（router/index.js）

```javascript
{
  path: '/train/fcn-plus',
  name: 'TrainFCNPlus',
  component: TrainView,
  meta: { 
    title: 'FCN-Plus训练',
    subtitle: '全卷积网络增强版',
    icon: 'Tools',
    modelArch: 'fcn_plus'
  }
}
```

#### 2. 修改导航栏（App.vue）

```vue
<el-sub-menu index="train-menu">
  <template #title>
    <el-icon><VideoPlay /></el-icon>
    <span>模型训练</span>
  </template>
  <el-menu-item index="/train">通用训练（U-Net/FCN）</el-menu-item>
  <el-menu-item index="/train/rs-unet3-plus">RS-Unet3+专用训练</el-menu-item>
  <el-menu-item index="/train/fcn-plus">FCN-Plus专用训练</el-menu-item>
</el-sub-menu>
```

#### 3. 修改TrainView.vue的下拉菜单

```vue
<el-select v-model="trainParams.model_arch">
  <el-option label="U-Net" value="unet"></el-option>
  <el-option label="RS-Unet3+" value="rs_unet3_plus"></el-option>
  <el-option label="FCN" value="fcn"></el-option>
  <el-option label="FCN-Plus" value="fcn_plus"></el-option>
</el-select>
```

#### 4. 添加模型专用参数配置

```javascript
const handleModelArchChange = (modelArch) => {
  // ...existing code...
  
  if (modelArch === 'fcn_plus') {
    trainParams.epochs = 100
    trainParams.lr = 0.0005
    trainParams.weight_decay = 0.00005
    trainParams.batch_size = 8
  }
}
```

### 如何动态控制菜单可用性

#### 1. 扩展全局状态（useGlobalState.js）

```javascript
const fcnPlusAvailable = ref(true)

function setFcnPlusAvailable(available) {
  fcnPlusAvailable.value = available
}

export function useGlobalState() {
  return {
    // ...existing...
    fcnPlusAvailable: readonly(fcnPlusAvailable),
    setFcnPlusAvailable,
  }
}
```

#### 2. 在App.vue中使用

```vue
<script setup>
const { fcnPlusAvailable } = useGlobalState()
</script>

<template>
  <el-menu-item 
    index="/train/fcn-plus"
    :disabled="!fcnPlusAvailable"
  >
    FCN-Plus专用训练
  </el-menu-item>
</template>
```

#### 3. 在应用启动时检测后端模型

```javascript
// main.js 或 App.vue mounted钩子
import { useGlobalState } from '@/composables/useGlobalState'
const { setRsUnet3PlusAvailable, setFcnPlusAvailable } = useGlobalState()

async function detectBackendModels() {
  try {
    const response = await axios.get('http://127.0.0.1:8000/models/available')
    setRsUnet3PlusAvailable(response.data.rs_unet3_plus)
    setFcnPlusAvailable(response.data.fcn_plus)
  } catch (error) {
    console.error('模型检测失败:', error)
  }
}

onMounted(() => {
  detectBackendModels()
})
```

---

## 🐛 常见问题

### Q1：为什么RS-Unet3+菜单项是灰色的？

**原因：** `rsUnet3PlusAvailable` 状态为 `false`，表示后端未部署该模型。

**解决方案：**
1. 检查后端是否实现了RS-Unet3+训练接口
2. 在前端手动设置可用性（开发测试）：
   ```javascript
   import { useGlobalState } from '@/composables/useGlobalState'
   const { setRsUnet3PlusAvailable } = useGlobalState()
   setRsUnet3PlusAvailable(true)  // 强制启用
   ```

### Q2：切换模型架构后参数没有自动更新？

**原因：** 可能是 `handleModelArchChange` 函数未被调用。

**检查点：**
1. 确保 `el-select` 绑定了 `@change="handleModelArchChange"`
2. 确认 `handleModelArchChange` 函数在正确的作用域内定义

### Q3：点击"RS-Unet3+专用训练"菜单后，下拉菜单还显示U-Net？

**原因：** TrainView.vue中下拉菜单应该根据 `route.meta.modelArch` 禁用或隐藏。

**建议实现：**
```vue
<el-form-item label="模型架构：" v-if="!route.meta?.modelArch">
  <!-- 仅在通用训练页显示下拉菜单 -->
  <el-select v-model="trainParams.model_arch">
    ...
  </el-select>
</el-form-item>

<el-alert v-else type="info">
  当前模型：{{ getModelDisplayName(trainParams.model_arch) }}
</el-alert>
```

### Q4：如何在HomeView.vue中获取当前使用的模型？

**方案：** 使用全局状态

```vue
<script setup>
import { useGlobalState } from '@/composables/useGlobalState'
const { globalModelArch, getModelDisplayName } = useGlobalState()
</script>

<template>
  <div>
    当前使用模型：{{ getModelDisplayName(globalModelArch) }}
  </div>
</template>
```

---

## 📝 最佳实践

### 1. **路由命名规范**
- 通用训练：`/train`
- 专用训练：`/train/{model-name}`（使用kebab-case）
- 避免使用驼峰或下划线

### 2. **Meta字段规范**
- `title`：页面标题（必填）
- `subtitle`：副标题（可选）
- `icon`：Element Plus图标名称（可选）
- `modelArch`：模型架构标识（专用训练页必填）

### 3. **全局状态使用原则**
- **只读暴露**：对外暴露 `readonly(ref)`，防止外部直接修改
- **函数修改**：通过专门的setter函数修改状态
- **命名一致**：状态名与函数名保持一致（如 `rsUnet3PlusAvailable` 对应 `setRsUnet3PlusAvailable`）

### 4. **导航菜单设计**
- **一级菜单**：简洁明了，最多6-7项
- **子菜单**：相关功能分组，最多3-4层
- **禁用提示**：必须提供Tooltip说明禁用原因

### 5. **组件复用策略**
- **通用组件**：使用props和路由meta区分行为
- **避免硬编码**：模型相关配置通过meta或props传递
- **保持灵活**：支持通过下拉菜单手动切换（通用页）和路由强制指定（专用页）

---

## 📚 相关文档

- **Vue Router官方文档**：https://router.vuejs.org/
- **Element Plus Menu组件**：https://element-plus.org/zh-CN/component/menu.html
- **Vue3 Composition API**：https://vuejs.org/api/composition-api-setup.html
- **OCTA后端API文档**：[octa_backend/README.md](../../octa_backend/README.md)

---

## 🔄 更新日志

| 日期 | 版本 | 修改内容 | 作者 |
|------|------|---------|------|
| 2026-01-17 | v1.0.0 | 初始版本，实现RS-Unet3+专用训练路由与导航 | GitHub Copilot AI |

---

**维护者**：OCTA Web项目组  
**最后更新**：2026-01-17
