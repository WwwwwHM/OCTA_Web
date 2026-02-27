# RS-Unet3+ 路由与导航功能测试清单

## ✅ 测试环境

- **前端地址**：http://127.0.0.1:5173
- **后端地址**：http://127.0.0.1:8000
- **浏览器**：Chrome/Edge（推荐开发者工具开启）

---

## 📋 功能测试清单

### 1. 路由配置测试

#### 1.1 通用训练路由
- [ ] 访问 `http://127.0.0.1:5173/train`
  - [ ] 页面正常加载
  - [ ] 显示"模型架构"下拉菜单
  - [ ] 默认选中上次使用的模型（或unet）
  - [ ] 可手动切换模型（U-Net / RS-Unet3+ / FCN）

#### 1.2 RS-Unet3+专用路由
- [ ] 访问 `http://127.0.0.1:5173/train/rs-unet3-plus`
  - [ ] 页面正常加载
  - [ ] 自动选中RS-Unet3+模型
  - [ ] 显示蓝色提示框："RS-Unet3+ 训练配置：已自动配置最优参数..."
  - [ ] 训练参数自动设置为：
    - [ ] epochs = 200
    - [ ] lr = 0.0001
    - [ ] weight_decay = 0.0001
    - [ ] batch_size = 4
  - [ ] 成功消息弹出："已进入 RS-Unet3+ 专用训练页"

#### 1.3 路由跳转测试
- [ ] 从首页点击导航"模型训练"子菜单
  - [ ] 子菜单正确展开
  - [ ] 显示两个选项："通用训练（U-Net/FCN）"和"RS-Unet3+专用训练"
- [ ] 点击"通用训练（U-Net/FCN）"
  - [ ] 跳转到 `/train`
  - [ ] URL正确
  - [ ] 页面内容正确
- [ ] 点击"RS-Unet3+专用训练"
  - [ ] 跳转到 `/train/rs-unet3-plus`
  - [ ] URL正确
  - [ ] 页面内容正确

---

### 2. 导航栏测试

#### 2.1 菜单结构
- [ ] 打开应用首页
- [ ] 检查导航栏包含以下项目：
  - [ ] 首页
  - [ ] 历史记录
  - [ ] **模型训练（子菜单）**
    - [ ] 通用训练（U-Net/FCN）
    - [ ] RS-Unet3+专用训练（带Science图标）
  - [ ] 文件管理
  - [ ] 关于

#### 2.2 子菜单交互
- [ ] 鼠标悬停"模型训练"
  - [ ] 子菜单正确展开
  - [ ] 无延迟卡顿
- [ ] 点击"模型训练"标题
  - [ ] 子菜单展开/收起切换
- [ ] 子菜单项悬停效果
  - [ ] 背景变为浅蓝色（#ecf5ff）
  - [ ] 文字变为蓝色（#409EFF）

#### 2.3 激活状态
- [ ] 当前在 `/train` 路由
  - [ ] "通用训练（U-Net/FCN）"菜单项高亮（蓝色）
  - [ ] "模型训练"父菜单高亮
- [ ] 当前在 `/train/rs-unet3-plus` 路由
  - [ ] "RS-Unet3+专用训练"菜单项高亮（蓝色）
  - [ ] "模型训练"父菜单高亮

#### 2.4 禁用状态（如果rsUnet3PlusAvailable=false）
- [ ] 打开浏览器开发者工具Console
- [ ] 运行测试代码：
  ```javascript
  import { useGlobalState } from '@/composables/useGlobalState'
  const { setRsUnet3PlusAvailable } = useGlobalState()
  setRsUnet3PlusAvailable(false)
  ```
- [ ] 检查"RS-Unet3+专用训练"菜单项：
  - [ ] 变为灰色（禁用状态）
  - [ ] 鼠标悬停显示"后端未部署RS-Unet3+模型"提示
  - [ ] 显示黄色警告图标（WarningFilled）
  - [ ] 点击无反应（无法跳转）

---

### 3. 全局状态管理测试

#### 3.1 模型架构状态同步
- [ ] 在 `/train` 页面选择RS-Unet3+模型
- [ ] 跳转到首页（`/`）
- [ ] 返回 `/train` 页面
  - [ ] RS-Unet3+模型仍被选中（状态持久）
- [ ] 跳转到 `/train/rs-unet3-plus`
  - [ ] 自动设置为RS-Unet3+（覆盖之前的通用页选择）

#### 3.2 可用性状态管理
- [ ] 打开开发者工具Console
- [ ] 运行以下测试：
  ```javascript
  import { useGlobalState } from '@/composables/useGlobalState'
  const { 
    rsUnet3PlusAvailable, 
    setRsUnet3PlusAvailable,
    getRsUnet3PlusAvailable 
  } = useGlobalState()
  
  // 测试getter
  console.log('初始状态:', getRsUnet3PlusAvailable())  // 应为true
  
  // 测试setter
  setRsUnet3PlusAvailable(false)
  console.log('设置后:', getRsUnet3PlusAvailable())  // 应为false
  
  // 测试响应式
  console.log('响应式ref:', rsUnet3PlusAvailable.value)  // 应为false
  
  // 恢复
  setRsUnet3PlusAvailable(true)
  ```
- [ ] 确认所有输出符合预期

#### 3.3 显示名称函数测试
- [ ] 打开开发者工具Console
- [ ] 运行以下测试：
  ```javascript
  import { useGlobalState } from '@/composables/useGlobalState'
  const { getModelDisplayName } = useGlobalState()
  
  console.log(getModelDisplayName('unet'))  // 应输出"U-Net"
  console.log(getModelDisplayName('rs_unet3_plus'))  // 应输出"RS-Unet3+"
  console.log(getModelDisplayName('fcn'))  // 应输出"FCN"
  console.log(getModelDisplayName('unknown'))  // 应输出"unknown"
  ```
- [ ] 确认所有输出正确

---

### 4. TrainView.vue 组件测试

#### 4.1 通用训练页模式（/train）
- [ ] 访问 `/train`
- [ ] "模型架构"下拉菜单可见
- [ ] 可手动切换模型
- [ ] 切换到RS-Unet3+时：
  - [ ] 参数自动更新为推荐值
  - [ ] 显示蓝色提示框
  - [ ] 弹出成功消息
- [ ] 切换到U-Net时：
  - [ ] 参数自动更新为标准值
  - [ ] 蓝色提示框消失
- [ ] 切换到FCN时：
  - [ ] 参数自动更新

#### 4.2 专用训练页模式（/train/rs-unet3-plus）
- [ ] 访问 `/train/rs-unet3-plus`
- [ ] 模型架构强制锁定为RS-Unet3+
- [ ] 参数自动设置为推荐值
- [ ] 蓝色提示框显示
- [ ] **建议实现**（可选）：下拉菜单隐藏或禁用（防止用户切换）

#### 4.3 路由meta识别测试
- [ ] 打开开发者工具Console
- [ ] 在 `/train` 页面执行：
  ```javascript
  console.log('Route meta:', this.$route.meta.modelArch)  // 应为undefined
  ```
- [ ] 在 `/train/rs-unet3-plus` 页面执行：
  ```javascript
  console.log('Route meta:', this.$route.meta.modelArch)  // 应为'rs_unet3_plus'
  ```

#### 4.4 参数自动配置测试
- [ ] 访问 `/train/rs-unet3-plus`
- [ ] 检查表单参数：
  - [ ] epochs = 200
  - [ ] lr = 0.0001
  - [ ] weight_decay = 0.0001
  - [ ] batch_size = 4
- [ ] 手动修改参数
- [ ] 刷新页面（F5）
  - [ ] 参数恢复为推荐值（不保存用户修改）

---

### 5. 样式与响应式测试

#### 5.1 桌面端（>768px）
- [ ] 浏览器宽度调整到 1280px
- [ ] 导航栏横向排列
- [ ] 子菜单下拉显示
- [ ] 所有元素对齐正确

#### 5.2 移动端（<768px）
- [ ] 浏览器宽度调整到 375px
- [ ] 导航栏折叠或纵向排列
- [ ] 子菜单展开正常
- [ ] 按钮和表单元素不溢出

#### 5.3 子菜单样式
- [ ] 子菜单项左内边距 40px
- [ ] 子菜单项最小宽度 220px
- [ ] 禁用项透明度 0.5
- [ ] 禁用项鼠标指针为 `not-allowed`
- [ ] 悬停效果仅在非禁用项上生效

---

### 6. 错误处理测试

#### 6.1 后端未启动
- [ ] 关闭后端服务
- [ ] 访问 `/train/rs-unet3-plus`
- [ ] 尝试上传数据集
  - [ ] 显示错误提示："连接失败"或类似信息
  - [ ] 不会崩溃或白屏

#### 6.2 无效路由
- [ ] 访问 `http://127.0.0.1:5173/train/invalid-model`
  - [ ] 404页面或重定向到首页
  - [ ] 不会崩溃

#### 6.3 全局状态异常
- [ ] 打开开发者工具Console
- [ ] 手动破坏全局状态：
  ```javascript
  // 尝试直接修改只读ref（应失败）
  import { useGlobalState } from '@/composables/useGlobalState'
  const { rsUnet3PlusAvailable } = useGlobalState()
  rsUnet3PlusAvailable.value = false  // 应报错或无效
  ```
- [ ] 确认状态未被破坏

---

### 7. 浏览器兼容性测试

#### 7.1 Chrome（推荐）
- [ ] 所有功能正常
- [ ] 样式渲染正确
- [ ] 无Console错误

#### 7.2 Edge
- [ ] 所有功能正常
- [ ] 样式渲染正确
- [ ] 无Console错误

#### 7.3 Firefox（可选）
- [ ] 子菜单展开正常
- [ ] 路由跳转正常
- [ ] 样式兼容

---

## 🐛 已知问题

### 问题1：子菜单在移动端无法展开
**状态**：待修复  
**影响**：<768px屏幕无法访问RS-Unet3+专用训练  
**临时方案**：直接输入URL `/train/rs-unet3-plus`

### 问题2：刷新页面后全局状态丢失
**状态**：设计如此（无持久化）  
**影响**：刷新后恢复默认状态  
**改进方案**：可使用localStorage持久化

---

## 📊 测试结果记录

| 测试项 | 通过 | 失败 | 备注 |
|--------|------|------|------|
| 1. 路由配置 | ☐ | ☐ |  |
| 2. 导航栏 | ☐ | ☐ |  |
| 3. 全局状态 | ☐ | ☐ |  |
| 4. TrainView组件 | ☐ | ☐ |  |
| 5. 样式响应式 | ☐ | ☐ |  |
| 6. 错误处理 | ☐ | ☐ |  |
| 7. 浏览器兼容 | ☐ | ☐ |  |

---

## 🚀 快速测试命令

```bash
# 1. 启动前端（新终端）
cd octa_frontend
npm run dev

# 2. 启动后端（新终端）
cd octa_backend
..\octa_env\Scripts\activate
python main.py

# 3. 打开浏览器测试
# 访问: http://127.0.0.1:5173

# 4. 开发者工具测试（浏览器Console）
import { useGlobalState } from '@/composables/useGlobalState'
const { setRsUnet3PlusAvailable } = useGlobalState()
setRsUnet3PlusAvailable(false)  // 测试禁用状态
setRsUnet3PlusAvailable(true)   // 恢复启用
```

---

**测试人员**：_______________  
**测试日期**：2026-01-17  
**测试版本**：v1.0.0  
**通过率**：_____% （_____/28项）

