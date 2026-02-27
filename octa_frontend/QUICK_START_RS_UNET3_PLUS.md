# RS-Unet3+ 路由与导航 - 快速使用指南

## 🎯 核心功能

为OCTA平台添加了**RS-Unet3+专用训练路由**，支持：
1. ✅ 专用训练路径 `/train/rs-unet3-plus`（自动配置最优参数）
2. ✅ 导航栏子菜单（通用训练 + RS-Unet3+专用训练）
3. ✅ 全局状态管理（模型架构跨组件共享）
4. ✅ 动态禁用控制（后端未部署时自动禁用菜单）

---

## 🚀 快速开始

### 用户使用方式

#### 场景1：我想用RS-Unet3+模型训练（推荐新手）

1. 打开OCTA平台 → 点击导航栏"模型训练"
2. 选择子菜单"**RS-Unet3+专用训练**"
3. 系统自动配置最优参数：
   - 训练轮数：200
   - 学习率：0.0001
   - 权重衰减：0.0001
   - 批次大小：4
4. 上传数据集ZIP包 → 点击"开始训练"

✅ **优势**：无需手动调参，一键开始训练

---

#### 场景2：我想自己选择模型和参数（高级用户）

1. 打开OCTA平台 → 点击导航栏"模型训练"
2. 选择子菜单"**通用训练（U-Net/FCN）**"
3. 手动选择模型架构：U-Net / RS-Unet3+ / FCN
4. 自定义训练参数（epochs、lr、weight_decay、batch_size）
5. 上传数据集ZIP包 → 点击"开始训练"

✅ **优势**：灵活调参，适合科研实验

---

### 开发者使用方式

#### 读取全局模型架构

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

#### 设置全局模型架构

```javascript
import { useGlobalState } from '@/composables/useGlobalState'

const { setGlobalModelArch } = useGlobalState()

// 用户在HomeView选择模型后同步全局状态
function handleModelChange(selectedModel) {
  setGlobalModelArch(selectedModel)
  ElMessage.success(`已切换为 ${getModelDisplayName(selectedModel)}`)
}
```

#### 控制菜单禁用状态

```vue
<script setup>
import { useGlobalState } from '@/composables/useGlobalState'

const { setRsUnet3PlusAvailable } = useGlobalState()

// 检测后端模型可用性（例如在App.vue mounted钩子）
async function checkBackendModels() {
  try {
    const response = await axios.get('http://127.0.0.1:8000/models/available')
    setRsUnet3PlusAvailable(response.data.rs_unet3_plus)
  } catch (error) {
    setRsUnet3PlusAvailable(false)  // 后端未启动时禁用
  }
}

onMounted(() => {
  checkBackendModels()
})
</script>
```

---

## 📂 文件结构

```
octa_frontend/
├── src/
│   ├── composables/
│   │   └── useGlobalState.js       # 全局状态管理（新增）
│   ├── router/
│   │   └── index.js                # 路由配置（已修改）
│   ├── views/
│   │   └── TrainView.vue           # 训练页面（已修改）
│   └── App.vue                     # 根组件（已修改）
├── ROUTER_NAVIGATION_GUIDE.md       # 完整使用文档（新增）
├── TESTING_CHECKLIST.md             # 测试清单（新增）
└── QUICK_START_RS_UNET3_PLUS.md     # 本文档（新增）
```

---

## 🔧 代码变更摘要

### 1. router/index.js - 添加RS-Unet3+专用路由

```diff
{
  path: '/train',
  name: 'Train',
  component: TrainView,
  meta: { 
    title: '模型训练',
+   subtitle: '通用训练（U-Net/FCN）'
  }
},
+{
+  path: '/train/rs-unet3-plus',
+  name: 'TrainRSUnet3Plus',
+  component: TrainView,
+  meta: { 
+    title: 'RS-Unet3+训练',
+    subtitle: 'OCTA专用训练（血管+FAZ）',
+    icon: 'Science',
+    modelArch: 'rs_unet3_plus'  # 关键：自动设置模型架构
+  }
+}
```

### 2. App.vue - 导航栏改为子菜单

```diff
- <el-menu-item index="/train">
-   <el-icon><VideoPlay /></el-icon>
-   <span>模型训练</span>
- </el-menu-item>

+ <el-sub-menu index="train-menu">
+   <template #title>
+     <el-icon><VideoPlay /></el-icon>
+     <span>模型训练</span>
+   </template>
+   <el-menu-item index="/train">
+     <span>通用训练（U-Net/FCN）</span>
+   </el-menu-item>
+   <el-menu-item 
+     index="/train/rs-unet3-plus"
+     :disabled="!rsUnet3PlusAvailable"
+   >
+     <el-icon><Science /></el-icon>
+     <span>RS-Unet3+专用训练</span>
+   </el-menu-item>
+ </el-sub-menu>
```

### 3. TrainView.vue - 支持路由meta自动配置

```diff
+ import { useRoute } from 'vue-router'
+ import { useGlobalState } from '@/composables/useGlobalState'

+ const route = useRoute()
+ const { setGlobalModelArch, getGlobalModelArch } = useGlobalState()

+ // 页面初始化时从路由meta读取模型架构
+ onMounted(() => {
+   const routeModelArch = route.meta?.modelArch
+   if (routeModelArch) {
+     trainParams.model_arch = routeModelArch
+     handleModelArchChange(routeModelArch)
+     ElMessage.success(`已进入 ${routeModelArch === 'rs_unet3_plus' ? 'RS-Unet3+' : 'U-Net'} 专用训练页`)
+   }
+ })
```

### 4. composables/useGlobalState.js - 全局状态管理（新文件）

```javascript
import { ref, readonly } from 'vue'

const globalModelArch = ref('unet')
const rsUnet3PlusAvailable = ref(true)

export function useGlobalState() {
  return {
    globalModelArch: readonly(globalModelArch),
    rsUnet3PlusAvailable: readonly(rsUnet3PlusAvailable),
    setGlobalModelArch(arch) {
      globalModelArch.value = arch
    },
    getGlobalModelArch() {
      return globalModelArch.value
    },
    // ...更多函数见完整文件
  }
}
```

---

## ❓ 常见问题

### Q1：为什么"RS-Unet3+专用训练"菜单是灰色的？

**A**：表示后端未部署RS-Unet3+模型。解决方案：
1. 确保后端实现了 `/train/upload-dataset` 接口并支持 `model_arch=rs_unet3_plus`
2. 开发测试时可手动启用：
   ```javascript
   import { useGlobalState } from '@/composables/useGlobalState'
   const { setRsUnet3PlusAvailable } = useGlobalState()
   setRsUnet3PlusAvailable(true)
   ```

### Q2：专用训练页和通用训练页有什么区别？

**对比表：**

| 特性 | 通用训练页（/train） | 专用训练页（/train/rs-unet3-plus） |
|------|---------------------|-----------------------------------|
| 模型选择 | 下拉菜单手动选择 | 自动锁定RS-Unet3+ |
| 参数配置 | 手动输入 | 自动设置推荐值 |
| 适合人群 | 高级用户、科研实验 | 新手用户、快速训练 |
| 灵活性 | 高（可随意调参） | 低（参数预设） |

### Q3：如何在HomeView中知道用户使用的是哪个模型？

**A**：使用全局状态：

```vue
<script setup>
import { useGlobalState } from '@/composables/useGlobalState'
const { globalModelArch, getModelDisplayName } = useGlobalState()
</script>

<template>
  <el-tag :type="globalModelArch === 'rs_unet3_plus' ? 'success' : 'info'">
    {{ getModelDisplayName(globalModelArch) }}
  </el-tag>
</template>
```

---

## 📚 延伸阅读

- **完整使用文档**：[ROUTER_NAVIGATION_GUIDE.md](./ROUTER_NAVIGATION_GUIDE.md)
- **测试清单**：[TESTING_CHECKLIST.md](./TESTING_CHECKLIST.md)
- **后端API文档**：[../octa_backend/README.md](../octa_backend/README.md)

---

## 🐛 报告问题

发现Bug或有功能建议？请提交Issue：

1. 标题格式：`[路由] 简短描述问题`
2. 内容包含：
   - 复现步骤
   - 预期行为
   - 实际行为
   - 浏览器版本
   - Console错误信息（如有）

---

**维护者**：OCTA Web项目组  
**最后更新**：2026-01-17  
**版本**：v1.0.0
