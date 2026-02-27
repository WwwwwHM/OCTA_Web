# HomeView.vue 优化 - 毕设演示指南

## 🎯 快速开始

### 启动应用
```bash
# 终端1：启动后端（确保在octa_backend目录）
cd octa_backend
start_server.bat  # Windows
# 或
python main.py

# 终端2：启动前端（确保在octa_frontend目录）
cd octa_frontend
npm run dev

# 浏览器打开
http://127.0.0.1:5173/
```

---

## 📸 功能演示流程

### 场景1：成功分割展示

#### 步骤1：上传图像
1. 打开首页，看到医疗蓝配色的专业界面
2. 拖拽PNG图像到上传区域，或点击选择文件
3. 选择完成后，下方即时显示缩略图预览（256×256px）
4. 显示文件名和文件大小（如 "2.5 MB"）

#### 步骤2：选择模型
5. 在"步骤2"中选择分割模型
   - U-Net（推荐）
   - FCN

#### 步骤3：执行分割
6. 点击 "🚀 开始图像分割" 按钮
7. 按钮变为 loading 状态，显示 "处理中..." 并禁用（防止重复提交）
8. 显示 ElMessage.loading 浮层：`"AI模型正在处理图像，请稍候..."`

#### 步骤4：查看结果
9. 分割完成后，下方显示分割结果卡片
10. **左右对比布局**：
    - 左侧：原始图像（256×256）
    - 中间：动画箭头 (→)
    - 右侧：分割结果（256×256）
11. 按钮恢复正常，可继续上传新图像

#### 步骤5：下载结果
12. 点击 "⬇️ 下载结果" 按钮
13. 自动下载分割结果PNG文件

---

### 场景2：文件大小校验

#### 上传超大文件演示
1. 选择超过10MB的PNG文件
2. 自动弹出 ElMessage.warning：
   ```
   ⚠️ 文件大小为 15.3 MB，超过10MB限制，请上传更小的文件
   ```
3. 文件列表和缩略图自动清空
4. 按钮禁用（因为没有有效文件）

---

### 场景3：响应式设计演示

#### 桌面版 (>768px)
- 缩略图：256×256px
- 结果对比：左右并排
- 卡片宽度：最大1000px
- 完整的医疗蓝UI

#### 平板版 (600-768px)
- 缩略图：200×200px
- 结果对比：竖直堆叠
- 卡片全宽
- 优化的间距

#### 手机版 (<600px)
- 缩略图：160×160px
- 结果对比：竖直堆叠
- 100%屏幕宽度
- 移动端友好的按钮大小

---

## 🎨 UI亮点展示

### 1. 医疗蓝配色系统
```
主色：#1677ff（医疗蓝）
  ↳ 用于：卡片边框、按钮、标题、输入框
  ↳ 效果：专业、可信、医学感

深色：#0050b3（强调）
  ↳ 用于：悬停状态、渐变底色
  ↳ 效果：交互反馈、视觉层级

背景：rgba(22, 119, 255, 0.02-0.08)
  ↳ 用于：背景色、容器背景
  ↳ 效果：微妙的蓝色背景，不抢眼
```

### 2. 卡片设计
```
元素：标题卡片、上传卡片、结果卡片

特点：
  ✓ 圆角：12px（增加现代感）
  ✓ 阴影：0 2px 12px rgba(22, 119, 255, 0.1)
  ✓ 边框：1px solid rgba(22, 119, 255, 0.08)
  ✓ 悬停：阴影增强 + 亮度提升
  ✓ 过渡：smooth 0.3s ease
```

### 3. 标题卡片渐变
```css
background: linear-gradient(135deg, #1677ff 0%, #0050b3 100%);
```
- 从医疗蓝渐变到深蓝
- 增加视觉深度和专业感

### 4. 上传区域交互
```
默认状态：
  ✓ 蓝色虚线边框（2px dashed）
  ✓ 轻微蓝色背景（rgba(22, 119, 255, 0.02)）

悬停状态：
  ✓ 边框变深（#0050b3）
  ✓ 背景变亮（rgba(22, 119, 255, 0.06)）
  ✓ 整体呈现强调感
```

### 5. 缩略图设计
```
上传后预览：
  ✓ 尺寸：256×256px
  ✓ 圆角：8px
  ✓ 边框：1px solid rgba(22, 119, 255, 0.2)
  ✓ 阴影：0 2px 8px rgba(0, 0, 0, 0.1)
  ✓ 背景：white
  ✓ 文字：文件名 + 文件大小（格式化）
```

### 6. 分割结果对比
```
布局：原图 → 分割结果

原图：
  ✓ 左侧，256×256px
  ✓ 标签："原始图像"

箭头：
  ✓ 中间，32px字体，医疗蓝
  ✓ 脉冲动画（2s循环）
  ✓ 强调分割转换过程

结果图：
  ✓ 右侧，256×256px
  ✓ 标签："分割结果"
  ✓ 悬停时：放大1.02倍 + 阴影增强
```

### 7. 脉冲动画
```css
@keyframes pulse {
  0%, 100% { opacity: 0.6; }
  50% { opacity: 1; }
}
```
- 使用：结果对比中的箭头
- 效果：流畅的脉冲呼吸动画
- 周期：2秒

### 8. 按钮设计
```
外观：
  ✓ 渐变背景：#1677ff → #0050b3
  ✓ 圆角：8px
  ✓ 字体权重：600
  ✓ 字母间距：0.5px

交互：
  ✓ 正常：主色渐变
  ✓ 悬停：阴影增强 + 上移2px
  ✓ Loading：转圈动画
  ✓ 禁用：透明度0.6 + 不可点击

文本变化：
  ✓ 空闲："🚀 开始图像分割"
  ✓ Loading："处理中..."
```

---

## 💻 代码亮点

### 1. 文件大小校验
```javascript
// 自动校验，超过10MB拒绝
const validateFileSize = (file) => {
  const MAX_SIZE = 10 * 1024 * 1024
  if (file.size > MAX_SIZE) {
    ElMessage.warning(
      `文件大小为 ${formatFileSize(file.size)}，超过10MB限制`
    )
    return false
  }
  return true
}
```

### 2. 文件大小格式化
```javascript
// 自动转换为 "2.5 MB" 格式
const formatFileSize = (bytes) => {
  if (!bytes) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i]
}
```

### 3. 实时缩略图预览
```javascript
// 使用FileReader生成Data URL
const reader = new FileReader()
reader.onload = (e) => {
  uploadedImageUrl.value = e.target.result  // 即时显示
}
reader.readAsDataURL(selectedFile)
```

### 4. Loading状态管理
```javascript
// 分割过程中禁用按钮
:disabled="!fileList.length || !selectedModel || isSegmentLoading"
:loading="isSegmentLoading"

// 流程
try {
  isSegmentLoading.value = true  // 开始
  const response = await axios.post(...)
} finally {
  isSegmentLoading.value = false  // 恢复
}
```

### 5. 响应式断点
```css
/* 桌面版 >768px */
.result-layout { flex-direction: row; gap: 20px; }
.thumbnail-img { width: 256px; height: 256px; }

/* 平板版 600-768px */
@media (max-width: 768px) {
  .result-layout { flex-direction: column; gap: 12px; }
  .thumbnail-img { width: 200px; height: 200px; }
}

/* 手机版 <600px */
@media (max-width: 480px) {
  .thumbnail-img { width: 160px; height: 160px; }
}
```

---

## 🎓 毕设展示建议

### 演示顺序
1. **介绍阶段**：展示系统整体架构和医疗蓝配色设计
2. **功能演示**：
   - 上传→预览→分割→对比→下载（完整流程）
   - 文件大小校验演示（拒绝超大文件）
   - 不同设备响应式演示（缩放浏览器或用手机）
3. **代码讲解**：
   - 文件校验逻辑（5分钟）
   - FileReader实时预览（3分钟）
   - 医疗蓝UI设计系统（5分钟）
   - 响应式设计断点（3分钟）

### 重点强调
- 🎨 **医学美学**：冷色系传达专业和可信度
- 🎯 **用户体验**：分步骤指引、即时反馈、智能校验
- 📱 **响应式设计**：从手机到桌面完美适配
- 🔧 **技术细节**：FileReader、Flexbox、Gradient等现代技术

### 时间分配建议
```
总时间：10分钟
├─ 介绍（1分钟）
├─ 功能演示（4分钟）
│  ├─ 完整上传→分割流程（2分钟）
│  ├─ 文件校验演示（1分钟）
│  └─ 响应式演示（1分钟）
├─ 代码讲解（4分钟）
│  ├─ UI设计系统（2分钟）
│  ├─ 交互逻辑（1分钟）
│  └─ 响应式实现（1分钟）
└─ Q&A（1分钟）
```

---

## ✨ 最终检查清单

在正式演示前，请检查：

- [ ] 后端服务已启动（http://127.0.0.1:8000/）
- [ ] 前端已启动（http://127.0.0.1:5173/）
- [ ] 医疗蓝色配色正确显示（#1677ff）
- [ ] 缩略图预览功能正常
- [ ] 文件大小校验正常（尝试上传>10MB文件）
- [ ] Loading状态和按钮禁用正常工作
- [ ] 分割结果左右对比显示正常
- [ ] 下载功能正常
- [ ] 响应式设计在不同屏幕尺寸都能正确显示

---

**优化完成日期**：2026年1月12日  
**优化级别**：毕设展示级别 ✨

