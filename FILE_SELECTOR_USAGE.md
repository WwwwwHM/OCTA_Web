# 文件选择器功能使用说明

## 📋 功能概述

我们为OCTA图像分割平台添加了**文件选择器**功能,允许用户在进行图像分割和模型训练时,选择历史已上传的文件,而不必每次都重新上传。

## ✨ 主要特性

### 1. 图像分割页面 (HomeView.vue)
- ✅ 支持上传新图像和选择历史图像两种模式
- ✅ 历史图像自动预览(Base64编码)
- ✅ 显示文件详情(文件名、大小、上传时间)
- ✅ 无需重复上传相同图像

### 2. 模型训练页面 (TrainView.vue)
- ✅ 支持上传新数据集和选择历史数据集
- ✅ 显示已上传数据集列表
- ✅ 快速选择已验证的数据集
- ✅ 减少数据集传输时间

## 🔧 实现细节

### 后端API

#### 1. 文件列表API
**端点:** `GET /file/list?file_type=image|dataset`

**功能:** 获取已上传的文件列表

**参数:**
- `file_type` (string): 文件类型筛选 ("image" 或 "dataset")

**响应示例:**
```json
{
  "code": 200,
  "msg": "文件列表获取成功",
  "data": [
    {
      "id": 1,
      "file_name": "octa_sample.png",
      "file_path": "uploads/uuid123.png",
      "file_type": "image",
      "upload_time": "2026-01-20 14:30:00",
      "file_size": 2048576
    }
  ]
}
```

#### 2. 文件预览API (仅图像)
**端点:** `GET /file/preview/{file_id}`

**功能:** 获取图像文件的Base64编码预览

**响应示例:**
```json
{
  "code": 200,
  "msg": "预览获取成功",
  "data": {
    "file_id": 1,
    "filename": "octa_sample.png",
    "file_type": "image",
    "base64_data": "iVBORw0KGgoAAAANSUhEUg...",
    "preview_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg...",
    "mime_type": "image/png"
  }
}
```

#### 3. 历史文件训练API
**端点:** `POST /train/start-with-file/{file_id}`

**功能:** 使用历史数据集文件开始训练

**参数:**
- `file_id` (path): 数据集文件ID
- `model_arch` (form): 模型架构 ("unet" 或 "rs_unet3_plus")
- `epochs` (form): 训练轮数
- `lr` (form): 学习率
- 其他训练参数...

**响应示例:**
```json
{
  "code": 200,
  "msg": "训练完成",
  "data": {
    "dataset_id": "abc123",
    "model_path": "results/models/unet_abc123.pth",
    "loss_curve": "results/curves/loss_abc123.png"
  }
}
```

### 前端组件

#### FileSelector.vue
**位置:** `octa_frontend/src/components/FileSelector.vue`

**Props:**
- `fileType` (string, required): "image" 或 "dataset"
- `apiBaseUrl` (string): API基础URL,默认 "http://127.0.0.1:8000"

**Events:**
- `file-selected`: 文件选择完成时触发
  ```javascript
  { id: 1, file_name: "...", file_path: "...", file_size: 2048576 }
  ```
- `preview-loaded`: 预览加载完成时触发 (仅图像)
  ```javascript
  { id: 1, previewUrl: "data:image/png;base64,..." }
  ```

**Methods (通过ref访问):**
- `loadFileList()`: 手动刷新文件列表

## 📖 使用方法

### 1. 在分割页面选择历史图像

**步骤:**
1. 打开"图像分割"页面
2. 切换到"选择已上传文件"标签
3. 浏览历史图像列表
4. 点击"选择"按钮
5. 预览图像自动加载
6. 选择模型并点击"开始图像分割"

**代码示例 (HomeView.vue):**
```vue
<FileSelector
  ref="fileSelectorRef"
  file-type="image"
  @file-selected="handleHistoryFileSelected"
  @preview-loaded="handlePreviewLoaded"
>
  <template #upload-content>
    <!-- 上传新文件的UI -->
  </template>
</FileSelector>
```

### 2. 在训练页面选择历史数据集

**步骤:**
1. 打开"模型训练"页面
2. 切换到"选择已上传数据集"标签
3. 浏览历史数据集列表
4. 点击"选择"按钮
5. 配置训练参数
6. 点击"开始训练"

**代码示例 (TrainView.vue):**
```vue
<FileSelector
  ref="fileSelectorRef"
  file-type="dataset"
  @file-selected="handleHistoryFileSelected"
>
  <template #upload-content>
    <!-- 上传新数据集的UI -->
  </template>
</FileSelector>
```

## 🔒 安全性

### 路径遍历防护
- ✅ 使用文件ID代替路径,防止路径遍历攻击
- ✅ 后端验证文件类型和存在性
- ✅ 文件内容通过Base64编码安全传输

### 文件类型验证
- ✅ 前端限制文件选择扩展名
- ✅ 后端验证MIME类型
- ✅ 数据库记录文件类型字段

## ⚡ 性能优化

### 前端优化
- ✅ 懒加载:切换到"选择已上传文件"标签时才加载列表
- ✅ 预览缓存:Base64数据在组件内缓存
- ✅ 表格分页:大量文件时支持分页(Element Plus自带)

### 后端优化
- ✅ 文件复用:历史文件直接从磁盘读取,无需重新上传
- ✅ 数据库索引:file_type字段建立索引,快速筛选
- ✅ 流式传输:大文件使用FileResponse流式传输

## 🐛 故障排查

### 问题1:列表为空
**检查项:**
1. 数据库中是否有对应类型的文件记录
2. 查询参数 `file_type` 是否正确 ("image" 或 "dataset")
3. 后端日志是否有错误信息

**解决方案:**
```bash
# 检查数据库
sqlite3 octa_backend/octa.db "SELECT * FROM files WHERE file_type='image';"
```

### 问题2:预览加载失败
**检查项:**
1. 文件是否存在于磁盘 (`file_path` 对应的路径)
2. 文件格式是否支持 (仅支持PNG/JPG/JPEG)
3. 后端是否有文件读取权限

**解决方案:**
```bash
# 检查文件是否存在
ls -l uploads/your_file.png

# 检查文件权限
chmod 644 uploads/*.png
```

### 问题3:训练失败
**检查项:**
1. 数据集文件是否完整 (包含 images/ 和 masks/ 文件夹)
2. 压缩包格式是否支持 (ZIP/RAR/7Z)
3. 文件是否损坏

**解决方案:**
```bash
# 手动解压测试
unzip -l uploads/datasets/your_dataset.zip
```

## 📊 数据库结构

**表名:** `files`

**字段:**
| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | INTEGER | 主键 |
| file_name | TEXT | 原始文件名 |
| file_path | TEXT | 存储路径 |
| file_type | TEXT | 文件类型 ("image" 或 "dataset") |
| upload_time | TEXT | 上传时间 |
| file_size | INTEGER | 文件大小(字节) |

**索引:**
```sql
CREATE INDEX idx_file_type ON files(file_type);
CREATE INDEX idx_upload_time ON files(upload_time);
```

## 🚀 未来改进

### 短期计划
- [ ] 添加文件搜索功能 (按文件名搜索)
- [ ] 支持批量删除历史文件
- [ ] 添加文件标签/分类功能
- [ ] 显示文件使用次数统计

### 长期规划
- [ ] 支持文件夹视图
- [ ] 添加文件共享功能 (多用户)
- [ ] 实现文件版本管理
- [ ] 集成云存储 (OSS/S3)

## 📞 技术支持

如有问题,请查阅:
- 后端日志: `octa_backend/*.log`
- 前端控制台: 浏览器开发者工具
- 数据库查询: `sqlite3 octa_backend/octa.db`

---

**最后更新:** 2026-01-20  
**作者:** GitHub Copilot AI  
**版本:** 1.0
