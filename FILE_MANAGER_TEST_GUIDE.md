# 文件管理功能 - 快速验证指南

## ✅ 完成清单

### 后端实现（已完成）
- [x] `octa_backend/controller/file_controller.py` - 5个核心API接口
- [x] `octa_backend/dao/file_dao.py` - 数据库CRUD操作
- [x] 路由注册到 `main.py`（Line 46）
- [x] 接口验证通过：
  - GET /file/list - 查询文件列表
  - GET /file/detail/{file_id} - 查询文件详情
  - DELETE /file/delete/{file_id} - 删除文件
  - POST /file/reuse/{file_id} - 复用数据集训练
  - POST /file/test/{file_id} - 复用图片测试

### 前端实现（已完成）
- [x] `octa_frontend/src/views/FileManager.vue` - 文件管理页面
- [x] 路由注册到 `router/index.js`（/files路径）
- [x] 导航栏添加入口（App.vue）
- [x] 功能模块：
  - 筛选区：类型筛选（全部/图片/数据集）+ 刷新按钮
  - 列表区：el-table展示文件信息 + 操作按钮
  - 训练弹窗：输入epochs和lr参数
  - 测试弹窗：选择模型权重

---

## 🚀 快速测试流程

### 步骤1：启动后端服务

```bash
cd octa_backend
# Windows
start_server.bat

# 或手动启动
..\octa_env\Scripts\activate
python main.py
```

**验证**：访问 http://127.0.0.1:8000/docs
- 应该能看到"文件管理"标签下的5个接口

### 步骤2：启动前端服务

```bash
cd octa_frontend
npm run dev
```

**验证**：访问 http://127.0.0.1:5173
- 导航栏应该显示"文件管理"菜单项

### 步骤3：测试文件管理页面

#### 3.1 查看文件列表
1. 点击导航栏"文件管理"
2. 页面应该加载文件列表
3. 测试筛选功能：
   - 选择"图片" → 只显示image类型
   - 选择"数据集" → 只显示dataset类型
   - 选择"全部文件" → 显示所有文件
4. 点击"刷新列表"按钮 → 重新加载数据

#### 3.2 测试删除功能
1. 在文件列表中选择任意文件
2. 点击"删除"按钮
3. 确认弹窗 → 点击"确定"
4. 应该提示"删除成功"并刷新列表

#### 3.3 测试复用训练（数据集）
1. 先上传一个数据集文件（通过训练页面或其他方式）
2. 在文件列表中找到该数据集
3. 点击"重新训练"按钮
4. 在弹窗中输入参数：
   - 训练轮数：10
   - 学习率：0.0001
5. 点击"开始训练"
6. 等待训练完成（显示最终损失）
7. 查看"关联模型"列是否更新

#### 3.4 测试复用测试（图片）
1. 先上传一张图片（通过首页上传）
2. 在文件列表中找到该图片
3. 点击"测试分割"按钮
4. 在弹窗中选择模型权重（可留空使用默认）
5. 点击"开始分割"
6. 等待分割完成
7. 选择"查看结果" → 在新标签页打开分割结果

---

## 🔍 接口测试（Swagger UI）

访问 http://127.0.0.1:8000/docs 进行接口测试：

### 1. GET /file/list
```
参数：file_type=image（可选）
预期响应：
{
  "code": 200,
  "msg": "查询成功，共 X 条记录",
  "data": [...]
}
```

### 2. GET /file/detail/{file_id}
```
参数：file_id=1
预期响应：
{
  "code": 200,
  "msg": "查询成功",
  "data": {
    "id": 1,
    "file_name": "test.png",
    ...
  }
}
```

### 3. DELETE /file/delete/{file_id}
```
参数：file_id=1
预期响应：
{
  "code": 200,
  "msg": "删除成功",
  "data": {
    "deleted_file": "test.png",
    "deleted_path": "..."
  }
}
```

### 4. POST /file/reuse/{file_id}
```
参数：
- file_id=2（数据集ID）
- epochs=10
- lr=0.0001

预期响应：
{
  "code": 200,
  "msg": "训练成功",
  "data": {
    "epochs": 10,
    "final_loss": 0.123,
    "model_path": "models/weights/...",
    ...
  }
}
```

### 5. POST /file/test/{file_id}
```
参数：
- file_id=1（图片ID）
- weight_path=（可选）

预期响应：
{
  "code": 200,
  "msg": "分割成功",
  "data": {
    "original_image": "uploads/...",
    "result_image": "results/...",
    "result_url": "http://127.0.0.1:8000/results/..."
  }
}
```

---

## ⚠️ 常见问题

### 问题1：前端显示"网络错误"
**原因**：后端服务未启动或端口不正确
**解决**：
1. 检查后端是否运行在 http://127.0.0.1:8000
2. 检查 `FileManager.vue` 中的 `API_BASE_URL` 配置

### 问题2：文件列表为空
**原因**：数据库中没有文件记录
**解决**：
1. 先通过首页上传图片
2. 或通过训练页面上传数据集
3. 然后刷新文件管理页面

### 问题3：训练或测试失败
**原因**：
- 文件类型不匹配（图片调用训练，数据集调用测试）
- 模型权重文件不存在
- 数据集格式不正确

**解决**：
1. 检查文件类型是否正确
2. 确认模型权重文件存在于 `models/weights/` 目录
3. 查看后端日志获取详细错误信息

### 问题4：删除后文件仍存在
**原因**：文件被占用或权限不足
**解决**：
1. 关闭所有打开该文件的程序
2. 检查文件权限
3. 查看后端日志确认删除是否成功

---

## 📊 数据库表结构

文件记录存储在 `octa.db` 的 `file_management` 表：

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键，自增 |
| file_name | TEXT | 文件名 |
| file_path | TEXT | 文件路径 |
| file_type | TEXT | 类型（image/dataset） |
| upload_time | TEXT | 上传时间 |
| related_model | TEXT | 关联模型路径 |
| file_size | INTEGER | 文件大小（字节） |

---

## 🎯 下一步优化建议

1. **批量操作**：添加批量删除功能
2. **高级筛选**：按时间范围、文件大小筛选
3. **下载功能**：支持下载原图和结果图
4. **预览功能**：在弹窗中预览图片
5. **训练进度**：实时显示训练进度条
6. **模型管理**：自动扫描可用的模型权重文件
7. **导出功能**：导出文件列表为CSV/Excel
8. **分页功能**：文件较多时添加分页器

---

**文档生成时间**：2026-01-16  
**验证状态**：✅ 后端5个接口已验证 | ⏳ 前端页面待启动测试
