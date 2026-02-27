# OCTA 训练页面使用指南

## 📋 功能概述

TrainView.vue 提供了完整的 OCTA 图像分割模型训练功能，包括：
- ✅ 数据集 ZIP 包上传（拖拽或点击）
- ✅ 训练参数配置（轮数、学习率、批次大小）
- ✅ 实时训练状态显示
- ✅ 训练结果可视化（指标表格 + ECharts 损失曲线）
- ✅ 训练后模型切换功能

## 🚀 快速开始

### 1. 安装依赖（首次使用）

```bash
cd octa_frontend
npm install  # ECharts 已包含在 package.json 中
```

### 2. 启动前端服务

```bash
npm run dev
# 访问: http://127.0.0.1:5173/train
```

### 3. 启动后端服务

```bash
cd octa_backend
python main.py
# 后端运行在: http://127.0.0.1:8000
```

## 📦 数据集准备

### 数据集结构要求

训练数据集必须为 ZIP 格式，内部结构如下：

```
dataset.zip
├── images/          # 原始 OCTA 图像
│   ├── img001.png
│   ├── img002.jpg
│   └── img003.jpeg
└── masks/           # 对应的分割标注（二值图）
    ├── img001.png   # 文件名必须与 images/ 中对应
    ├── img002.png
    └── img003.png
```

### 注意事项

- ✅ ZIP 包最大支持 50MB
- ✅ 图像格式：PNG/JPG/JPEG
- ✅ 标注格式：PNG（二值图，0-黑色背景，255-白色前景）
- ✅ 文件名匹配：`images/img001.png` 对应 `masks/img001.png`
- ❌ 不允许其他文件夹或文件

### 示例数据集下载（测试用）

```bash
# 假设您有一个测试数据集
# 确保符合上述结构要求
```

## 🎛️ 训练参数说明

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| **训练轮数 (epochs)** | 10 | 1-50 | 训练迭代次数，越大效果越好但耗时更长 |
| **学习率 (lr)** | 0.001 | 0.0001-0.1 | 控制模型权重更新步长，过大会不收敛 |
| **批次大小 (batch_size)** | 4 | 1-16 | CPU 环境建议 2-4，GPU 可设置 8-16 |

### 推荐配置

**快速测试（5分钟内）：**
- 训练轮数：5
- 学习率：0.001
- 批次大小：4

**标准训练（10-20分钟）：**
- 训练轮数：10
- 学习率：0.001
- 批次大小：4

**高质量训练（30分钟以上）：**
- 训练轮数：30
- 学习率：0.0005
- 批次大小：2

## 📊 训练流程

### 1. 上传数据集

1. 点击或拖拽 ZIP 包到上传区
2. 系统会自动校验格式和大小
3. 显示 "文件选择成功，点击'开始训练'按钮"

### 2. 配置参数

根据需求调整训练参数（或使用默认值）

### 3. 开始训练

1. 点击 **"开始训练"** 按钮
2. 系统显示上传进度：0-50%
3. 数据集自动解压并开始训练
4. 训练过程显示 "训练进行中，请稍候..."
5. 完成后显示 ✅ "训练完成！"

### 4. 查看结果

训练完成后，页面会显示：

#### 评估指标表格
- **Dice系数**：分割重叠度（0-1，越接近1越好）
- **IOU系数**：交并比（0-1，越接近1越好）
- **训练轮数**、**学习率**、**批次大小**

#### 损失曲线（ECharts交互式图表）
- **蓝色线**：训练损失（Train Loss）
- **红色线**：验证损失（Val Loss）
- 理想情况：两条线同步下降且收敛
- 过拟合警告：训练损失下降但验证损失上升

#### 模型信息
- 模型文件路径
- 损失曲线图片链接
- 数据集 ID
- 训练设备（CPU/GPU）

### 5. 使用训练后的模型

点击 **"使用训练后的模型进行分割"** 按钮：
- 系统会切换为训练后的权重
- 返回分割页面进行推理测试

## 🔍 常见问题

### Q1: 上传失败，提示 "文件大小超过50MB限制"

**解决方案：**
- 减少数据集图像数量
- 压缩图像质量（保持 PNG 格式）
- 使用在线压缩工具

### Q2: 训练失败，提示 "数据集格式错误"

**解决方案：**
1. 检查 ZIP 包是否包含 `images/` 和 `masks/` 文件夹
2. 确认文件名是否一一对应
3. 检查是否有其他无关文件

### Q3: 损失曲线不显示

**解决方案：**
- 刷新页面重新查看
- 检查浏览器控制台是否有 ECharts 错误
- 确认训练确实完成（查看状态区）

### Q4: 训练速度慢

**原因：** CPU 环境训练较慢（正常现象）

**加速建议：**
- 减少训练轮数（5-10轮）
- 减小批次大小（2-4）
- 减少数据集图像数量

### Q5: 验证损失比训练损失高

**原因：** 可能存在轻微过拟合

**解决方案：**
- 增加数据集数量
- 减少训练轮数
- 降低学习率（0.0005）

## 🧪 测试流程

### 完整测试步骤

```bash
# 1. 启动后端
cd octa_backend
python main.py

# 2. 启动前端
cd octa_frontend
npm run dev

# 3. 访问训练页面
http://127.0.0.1:5173/train

# 4. 上传测试数据集
# 上传您准备的 dataset.zip

# 5. 配置参数（快速测试）
# 训练轮数: 5
# 学习率: 0.001
# 批次大小: 4

# 6. 点击"开始训练"

# 7. 等待训练完成（约5分钟）

# 8. 查看结果
# - 检查 Dice/IOU 指标
# - 查看损失曲线是否正常
# - 尝试切换模型
```

## 📝 技术细节

### 前端技术栈

- **Vue 3 Composition API**：响应式状态管理
- **Element Plus**：UI 组件库（上传、表格、进度条等）
- **ECharts 5**：损失曲线交互式可视化
- **Axios**：HTTP 请求库

### 后端接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/train/upload-dataset` | POST | 上传数据集并触发训练 |
| `/train/result/{dataset_id}` | GET | 查询训练结果 |
| `/train/loss-curve/{dataset_id}` | GET | 获取损失曲线图片 |
| `/train/dataset/{dataset_id}` | DELETE | 删除数据集 |

### 核心功能实现

```javascript
// 1. 文件上传与参数传递
const formData = new FormData()
formData.append('file', selectedFile.value)
formData.append('epochs', trainParams.epochs)
formData.append('lr', trainParams.lr)
formData.append('batch_size', trainParams.batch_size)

// 2. ECharts 损失曲线渲染
const option = {
  xAxis: { data: [1, 2, 3, ...] },
  series: [
    { name: '训练损失', data: trainLosses },
    { name: '验证损失', data: valLosses }
  ]
}
lossChart.setOption(option)

// 3. 响应式窗口自适应
window.addEventListener('resize', () => lossChart.resize())
```

## 🎨 页面截图（功能说明）

### 上传区
- 拖拽上传 ZIP 包
- 参数配置表单（轮数/学习率/批次大小）
- "开始训练" 按钮

### 状态区
- 上传进度条（0-50%）
- 训练状态提示（info/success/error）

### 结果区
- 指标表格（Dice/IOU）
- ECharts 损失曲线（可交互缩放、悬浮显示值）
- 模型信息详情
- 操作按钮（切换模型/重新训练）

## 🔗 相关文档

- [后端训练服务实现](../octa_backend/service/train_service.py)
- [后端训练接口](../octa_backend/controller/train_controller.py)
- [前端路由配置](./src/router/index.js)
- [Element Plus 文档](https://element-plus.org/zh-CN/)
- [ECharts 文档](https://echarts.apache.org/zh/index.html)

## 📧 支持与反馈

如遇到问题：
1. 检查浏览器控制台错误
2. 查看后端终端日志
3. 参考 [TROUBLESHOOTING.md](../octa_backend/TROUBLESHOOTING.md)

---

**最后更新：2026年1月15日**  
**版本：v1.0.0**  
**状态：✅ 生产就绪**
