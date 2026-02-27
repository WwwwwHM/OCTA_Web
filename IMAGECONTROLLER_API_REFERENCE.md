# ImageController 类方法快速参考

## 📌 概览

`ImageController` 是OCTA图像分割平台的控制层核心类，所有API接口逻辑都在这里实现。

**位置**：`octa_backend/controller/image_controller.py`

---

## 🔧 初始化方法

### `ImageController.init_database()`

初始化SQLite数据库和表结构。

```python
# 用法
success = ImageController.init_database()  # 返回 True/False

# 功能
- 创建 uploads/ 和 results/ 目录
- 创建 octa.db 数据库文件
- 创建 images 表（记录分割历史）
```

**返回值**：`bool` - True表示初始化成功，False表示失败

---

## 📡 API接口方法

### 1. `ImageController.test_service()`

**对应API**：`GET /`

后端健康检查接口。

```python
# 返回值
{"message": "OCTA后端服务运行正常"}
```

---

### 2. `ImageController.segment_octa(file: UploadFile, model_type: str)`

**对应API**：`POST /segment-octa/`

核心分割接口。接收OCTA图像，调用模型分割，保存结果。

```python
# 参数
- file：UploadFile，上传的图像文件（PNG/JPG/JPEG）
- model_type：str，模型类型（'unet' 或 'fcn'）

# 返回值（成功时）
{
    "success": True,
    "message": "图像分割完成",
    "original_filename": "image.png",  # 原始文件名
    "saved_filename": "uuid-1234.png",  # 保存的唯一文件名
    "result_filename": "uuid-1234_segmented.png",  # 分割结果文件名
    "image_url": "/images/uuid-1234.png",  # 原图访问URL
    "result_url": "/results/uuid-1234_segmented.png",  # 结果访问URL
    "model_type": "unet",
    "record_id": 1  # 数据库记录ID
}

# 异常情况
- 400：文件格式不支持或模型类型无效
- 500：模型分割失败或数据库错误
```

---

### 3. `ImageController.get_uploaded_image(filename: str)`

**对应API**：`GET /images/{filename}`

获取上传的原始图像文件。

```python
# 参数
- filename：str，文件名（如 "uuid-1234.png"）

# 返回值
FileResponse - PNG/JPG/JPEG图像文件

# 异常情况
- 404：文件不存在
- 400：文件格式不支持
```

---

### 4. `ImageController.get_result_image(filename: str)`

**对应API**：`GET /results/{filename}`

获取分割结果图像文件。

```python
# 参数
- filename：str，结果文件名（如 "uuid-1234_segmented.png"）

# 返回值
FileResponse - PNG格式分割结果图像

# 异常情况
- 404：文件不存在
- 400：文件格式不是PNG
```

---

### 5. `ImageController.get_all_history()`

**对应API**：`GET /history/`

查询所有分割历史记录。

```python
# 参数
无

# 返回值
[
    {
        "id": 1,
        "filename": "uuid-1234.png",
        "upload_time": "2026-01-13 10:30:45",
        "model_type": "unet",
        "original_path": "./uploads/uuid-1234.png",
        "result_path": "./results/uuid-1234_segmented.png"
    },
    ...
]

# 排序
按 upload_time DESC（最新的在前）

# 异常情况
- 500：数据库查询错误
```

---

### 6. `ImageController.get_history_by_id(record_id: int)`

**对应API**：`GET /history/{record_id}`

查询单条分割历史记录。

```python
# 参数
- record_id：int，记录ID

# 返回值
{
    "id": 1,
    "filename": "uuid-1234.png",
    "upload_time": "2026-01-13 10:30:45",
    "model_type": "unet",
    "original_path": "./uploads/uuid-1234.png",
    "result_path": "./results/uuid-1234_segmented.png"
}

# 异常情况
- 400：record_id无效（不是正整数）
- 404：记录不存在
- 500：数据库查询错误
```

---

### 7. `ImageController.delete_history_by_id(record_id: int)`

**对应API**：`DELETE /history/{record_id}`

删除单条分割历史记录（仅删除数据库记录，不删除文件）。

```python
# 参数
- record_id：int，要删除的记录ID

# 返回值
{
    "success": True,
    "message": "分割记录已删除",
    "deleted_id": 1
}

# 异常情况
- 400：record_id无效
- 404：记录不存在
- 500：数据库删除错误
```

---

## 🔒 私有辅助方法

这些方法仅供`ImageController`内部使用，不应直接调用。

### `_generate_unique_filename(original_filename: str) -> str`

生成唯一的文件名（使用UUID）。

```python
# 例子
"image.png" → "a1b2c3d4-e5f6-4g7h-8i9j-k0l1m2n3o4p5.png"
```

---

### `_validate_image_file(file: UploadFile) -> bool`

验证上传的文件是否为支持的图像格式。

```python
# 支持的格式
✓ PNG：image/png, image/x-png
✓ JPG/JPEG：image/jpeg, image/x-jpeg, image/jpg

# 返回值
True - 支持的格式
False - 不支持的格式
```

---

### `_insert_record(...) -> Optional[int]`

将分割记录插入数据库。

```python
# 参数
- filename：str，文件名
- model_type：str，模型类型
- original_path：str，原图路径
- result_path：str，结果路径

# 返回值
record_id (int) - 插入的记录ID
None - 插入失败
```

---

### `_get_all_records() -> List[Dict]`

查询数据库中所有记录。

```python
# 返回值
[
    {"id": 1, "filename": "...", ...},
    {"id": 2, "filename": "...", ...},
    ...
]
```

---

### `_get_record_by_id(record_id: int) -> Optional[Dict]`

查询数据库中指定ID的记录。

```python
# 返回值
{"id": 1, "filename": "...", ...}  # 找到记录时
None  # 记录不存在时
```

---

## 📊 数据库表结构

### `images` 表

记录所有OCTA图像分割历史。

| 字段 | 类型 | 说明 |
|-----|-----|------|
| id | INTEGER | 主键，自动递增 |
| filename | TEXT | 唯一文件名（UUID格式） |
| upload_time | TEXT | 上传时间（格式：YYYY-MM-DD HH:MM:SS） |
| model_type | TEXT | 模型类型（'unet' 或 'fcn'） |
| original_path | TEXT | 原始图像保存路径 |
| result_path | TEXT | 分割结果保存路径 |

---

## 🎯 常见使用场景

### 场景1：上传并分割图像

```javascript
// 前端代码
const formData = new FormData()
formData.append('file', fileInput.files[0])
formData.append('model_type', 'unet')

const res = await axios.post('/segment-octa/', formData)
console.log(res.data.result_url)  // 分割结果URL
```

**后端流程**：
1. `main.py` 接收请求，转发给 `ImageController.segment_octa()`
2. `segment_octa()` 验证文件、保存文件、调用模型、保存结果、记录数据库
3. 返回包含result_url的JSON

---

### 场景2：查看分割历史

```javascript
// 前端代码
const res = await axios.get('/history/')
console.log(res.data)  // 所有历史记录数组
```

**后端流程**：
1. `main.py` 接收请求，转发给 `ImageController.get_all_history()`
2. `get_all_history()` 调用 `_get_all_records()` 查询数据库
3. 返回所有记录的JSON数组

---

### 场景3：显示特定历史记录的图像

```javascript
// 前端代码
const recordId = 1
const res = await axios.get(`/history/${recordId}`)
const { original_path, result_path } = res.data

// 显示原图和结果
const originalUrl = `/images/${res.data.filename}`
const resultUrl = `/results/${res.data.filename.replace('.png', '_segmented.png')}`
```

**后端流程**：
1. `ImageController.get_history_by_id()` 查询指定ID的记录
2. 前端根据返回的路径构建图像URL
3. 调用 `get_uploaded_image()` 和 `get_result_image()` 获取图像

---

## 🚨 错误处理

所有异常都会返回标准的HTTP异常，包含清晰的错误信息。

| 状态码 | 错误类型 | 常见原因 |
|--------|--------|--------|
| 400 | Bad Request | 文件格式不正确、参数无效 |
| 404 | Not Found | 文件/记录不存在 |
| 500 | Internal Server Error | 服务器处理错误 |

---

## 💡 最佳实践

### ✅ 推荐做法

```python
# 1. 让前端处理错误
try:
    const res = await axios.post('/segment-octa/', formData)
    // 处理成功情况
} catch (error) {
    // 处理错误（由后端返回详细信息）
    console.error(error.response.data.detail)
}

# 2. 使用record_id追踪历史
const recordId = res.data.record_id  // 保存此ID便于后续查询
```

### ❌ 避免做法

```python
# 不要
# 1. 直接访问文件系统路径（应该通过API）
# 2. 手动构建文件路径（应该使用API返回的URL）
# 3. 不处理API异常（应该捕捉并显示给用户）
```

---

## 📞 常见问题

**Q: 如何添加新的API接口？**

A: 在 `ImageController` 中添加新方法，然后在 `main.py` 中添加对应的FastAPI路由。

**Q: 修改业务逻辑需要改前端代码吗？**

A: 不需要！只要返回格式和接口路径不变，前端代码完全兼容。

**Q: 如何扩展到支持新的模型？**

A: 在 `segment_octa()` 方法中添加新的模型类型校验，在模型层中实现新模型即可。

**Q: 文件会一直保留在磁盘上吗？**

A: 是的，除非手动删除。建议定期清理老旧的uploads和results文件。

---

**最后更新**：2026年1月13日  
**版本**：1.0

