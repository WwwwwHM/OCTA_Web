# FileUtils工具类 - 完整指南

**版本**：Phase 9 | **状态**：✅ 完成 | **最后更新**：2026年1月14日

---

## 📚 目录

1. [快速开始](#快速开始)
2. [核心方法](#核心方法)
3. [使用场景](#使用场景)
4. [最佳实践](#最佳实践)
5. [常见问题](#常见问题)
6. [完整示例](#完整示例)

---

## 快速开始

### 导入方式

```python
# 方式1：导入FileUtils类
from utils import FileUtils

# 方式2：导入所有工具
from utils import FileUtils
```

### 最简单的用法

```python
# 验证文件格式
is_valid, msg = FileUtils.validate_file_format('image.png')

# 验证文件大小
is_valid, msg = FileUtils.validate_file_size(file_obj)

# 生成唯一文件名
unique_name = FileUtils.generate_unique_filename('photo.jpg')

# 创建目录
FileUtils.create_dir_if_not_exists('uploads/')

# 保存文件
success, msg = FileUtils.save_uploaded_file(file_obj, 'uploads/image.png')
```

---

## 核心方法

### 1️⃣ validate_file_format() - 文件格式校验

**功能**：验证文件是否为允许的格式（PNG/JPG/JPEG）

```python
def validate_file_format(
    filename: str,
    allow_formats: Optional[List[str]] = None
) -> Tuple[bool, str]:
```

**参数**：

| 参数 | 类型 | 说明 | 默认值 | 示例 |
|-----|------|------|--------|------|
| `filename` | str | 待验证的文件名 | 必需 | `'photo.jpg'` |
| `allow_formats` | List[str] | 允许的格式列表 | `['png', 'jpg', 'jpeg']` | `['png', 'gif']` |

**返回值**：`(是否有效, 提示信息)`

**示例**：

```python
# ✅ 有效的格式
is_valid, msg = FileUtils.validate_file_format('image.png')
# 返回: (True, "✓ 文件格式有效: PNG")

# ✅ 大小写不敏感
is_valid, msg = FileUtils.validate_file_format('image.JPG')
# 返回: (True, "✓ 文件格式有效: JPG")

# ❌ 无效的格式
is_valid, msg = FileUtils.validate_file_format('image.gif')
# 返回: (False, "✗ 不支持的文件格式: gif，仅支持: png, jpg, jpeg")

# ✅ 自定义允许格式
is_valid, msg = FileUtils.validate_file_format(
    'document.pdf',
    allow_formats=['pdf', 'doc', 'docx']
)
# 返回: (True, "✓ 文件格式有效: PDF")
```

**特点**：
- ✅ 大小写不敏感（.JPG == .jpg）
- ✅ 格式白名单机制
- ✅ 自定义格式列表支持

---

### 2️⃣ validate_file_size() - 文件大小校验

**功能**：验证文件大小是否超过限制

```python
def validate_file_size(
    file_obj,
    max_size: int = None
) -> Tuple[bool, str]:
```

**参数**：

| 参数 | 类型 | 说明 | 默认值 | 示例 |
|-----|------|------|--------|------|
| `file_obj` | object | 上传的文件对象 | 必需 | UploadFile / File |
| `max_size` | int | 最大允许大小（字节） | 10MB | `5*1024*1024` |

**单位参考**：
- 1 KB = 1,024 Bytes
- 1 MB = 1,048,576 Bytes
- 1 GB = 1,073,741,824 Bytes

**示例**：

```python
# ✅ 使用默认大小限制（10MB）
is_valid, msg = FileUtils.validate_file_size(file_obj)
# 返回: (True, "✓ 文件大小合法: 2.5 MB")

# ✅ 自定义大小限制（5MB）
is_valid, msg = FileUtils.validate_file_size(
    file_obj,
    max_size=5*1024*1024
)

# ❌ 文件超大
is_valid, msg = FileUtils.validate_file_size(file_obj, max_size=1*1024*1024)
# 返回: (False, "✗ 文件超大: 25.0 MB > 1.0 MB")
```

**支持的文件对象**：
- ✅ FastAPI的`UploadFile`
- ✅ Python标准文件对象
- ✅ 任何具有`.file`、`.seek()`、`.tell()`属性的对象

---

### 3️⃣ generate_unique_filename() - 唯一文件名生成

**功能**：生成UUID+原后缀的唯一文件名，避免覆盖

```python
def generate_unique_filename(
    original_filename: str
) -> str:
```

**参数**：

| 参数 | 类型 | 说明 | 示例 |
|-----|------|------|------|
| `original_filename` | str | 原始文件名 | `'photo.jpg'` |

**返回值**：唯一的文件名

**格式**：`img_{UUID}.{原扩展名}`

**示例**：

```python
# ✅ 保留扩展名
unique_name = FileUtils.generate_unique_filename('photo.jpg')
# 返回: 'img_abc123def456.jpg'

# ✅ 大小写转为小写
unique_name = FileUtils.generate_unique_filename('image.PNG')
# 返回: 'img_xyz789abc456.png'

# ✅ 无扩展名
unique_name = FileUtils.generate_unique_filename('README')
# 返回: 'img_def456xyz789'

# ✅ 多次调用生成不同的名称
name1 = FileUtils.generate_unique_filename('photo.jpg')
name2 = FileUtils.generate_unique_filename('photo.jpg')
assert name1 != name2  # 每次生成都不同
```

**特点**：
- ✅ UUID v4保证唯一性（碰撞率极低）
- ✅ 保留原始文件的扩展名
- ✅ 32字符十六进制UUID，避免文件覆盖

---

### 4️⃣ create_dir_if_not_exists() - 目录创建

**功能**：自动创建目录（包括所有父目录）

```python
def create_dir_if_not_exists(
    dir_path: str
) -> bool:
```

**参数**：

| 参数 | 类型 | 说明 | 示例 |
|-----|------|------|------|
| `dir_path` | str | 要创建的目录路径 | `'./uploads'` |

**返回值**：创建是否成功（True/False）

**示例**：

```python
# ✅ 创建单层目录
success = FileUtils.create_dir_if_not_exists('./uploads')

# ✅ 递归创建多层目录
success = FileUtils.create_dir_if_not_exists('./data/images/2026/01')

# ✅ 目录已存在（不会报错）
success = FileUtils.create_dir_if_not_exists('./uploads')
# 返回: True

# ❌ 权限不足
success = FileUtils.create_dir_if_not_exists('/root/protected')
# 返回: False，打印权限错误
```

**特点**：
- ✅ `parents=True`：递归创建所有父目录
- ✅ `exist_ok=True`：目录已存在不报错
- ✅ 完整的异常捕获（权限、磁盘等）

---

### 5️⃣ save_uploaded_file() - 文件保存

**功能**：保存上传的文件到指定路径

```python
def save_uploaded_file(
    file_obj,
    save_path: str
) -> Tuple[bool, str]:
```

**参数**：

| 参数 | 类型 | 说明 | 示例 |
|-----|------|------|------|
| `file_obj` | object | 上传的文件对象 | UploadFile |
| `save_path` | str | 保存的完整路径 | `'uploads/img.png'` |

**返回值**：`(保存是否成功, 提示信息)`

**示例**：

```python
# ✅ 基础保存
success, msg = FileUtils.save_uploaded_file(
    file_obj,
    'uploads/image.png'
)
if success:
    print(msg)  # ✓ 文件保存成功: uploads/image.png

# ✅ 完整工作流
from fastapi import UploadFile

async def handle_upload(file: UploadFile):
    # 步骤1：验证格式
    is_valid, msg = FileUtils.validate_file_format(file.filename)
    if not is_valid:
        return {"error": msg}
    
    # 步骤2：验证大小
    is_valid, msg = FileUtils.validate_file_size(file)
    if not is_valid:
        return {"error": msg}
    
    # 步骤3：生成唯一文件名
    unique_name = FileUtils.generate_unique_filename(file.filename)
    
    # 步骤4：保存文件
    save_path = f'uploads/{unique_name}'
    success, msg = FileUtils.save_uploaded_file(file, save_path)
    
    if success:
        return {"file_path": save_path}
    else:
        return {"error": msg}
```

**特点**：
- ✅ 自动创建保存目录
- ✅ 支持多种file_obj类型
- ✅ 完整的错误处理

---

## 使用场景

### 场景1：完整的文件上传流程

```python
from fastapi import UploadFile
from utils import FileUtils

async def upload_and_segment_image(file: UploadFile):
    """OCTA图像上传完整流程"""
    
    print(f"[INFO] 开始处理上传的文件: {file.filename}")
    
    # ==================== 步骤1：验证文件格式 ====================
    is_valid, error_msg = FileUtils.validate_file_format(file.filename)
    if not is_valid:
        return {"success": False, "error": error_msg}
    
    print(f"[✓] 文件格式验证通过")
    
    # ==================== 步骤2：验证文件大小 ====================
    is_valid, error_msg = FileUtils.validate_file_size(file)
    if not is_valid:
        return {"success": False, "error": error_msg}
    
    print(f"[✓] 文件大小验证通过")
    
    # ==================== 步骤3：生成唯一文件名 ====================
    unique_filename = FileUtils.generate_unique_filename(file.filename)
    print(f"[INFO] 生成唯一文件名: {unique_filename}")
    
    # ==================== 步骤4：确保存储目录存在 ====================
    upload_dir = './uploads'
    FileUtils.create_dir_if_not_exists(upload_dir)
    
    # ==================== 步骤5：保存文件 ====================
    save_path = f"{upload_dir}/{unique_filename}"
    success, save_msg = FileUtils.save_uploaded_file(file, save_path)
    
    if not success:
        return {"success": False, "error": save_msg}
    
    print(f"[SUCCESS] 文件保存成功: {save_path}")
    
    # ==================== 步骤6：后续处理（如图像分割） ====================
    # result_path = segment_octa_image(save_path)
    
    return {
        "success": True,
        "file_path": save_path,
        "filename": unique_filename
    }
```

### 场景2：批量处理多个文件

```python
from typing import List
from fastapi import UploadFile

async def batch_process_files(files: List[UploadFile]):
    """批量处理多个文件"""
    
    results = []
    
    for file in files:
        # 验证每个文件
        is_valid, msg = FileUtils.validate_file_format(file.filename)
        if not is_valid:
            results.append({"filename": file.filename, "status": "invalid", "error": msg})
            continue
        
        # 生成唯一名称
        unique_name = FileUtils.generate_unique_filename(file.filename)
        
        # 保存文件
        save_path = f'uploads/{unique_name}'
        success, msg = FileUtils.save_uploaded_file(file, save_path)
        
        results.append({
            "filename": file.filename,
            "unique_name": unique_name,
            "status": "success" if success else "failed",
            "message": msg
        })
    
    return results
```

### 场景3：不同的验证规则

```python
# 对于医学图像，使用更严格的限制
IMAGE_MAX_SIZE = 50 * 1024 * 1024  # 50MB

# 仅支持PNG格式
ALLOWED_FORMATS = ['png']

# 验证和保存
is_valid, msg = FileUtils.validate_file_format(
    filename='octa_scan.png',
    allow_formats=ALLOWED_FORMATS
)

if is_valid:
    is_valid, msg = FileUtils.validate_file_size(
        file_obj,
        max_size=IMAGE_MAX_SIZE
    )
```

---

## 最佳实践

### ✅ 推荐做法

1. **总是验证然后保存**
   ```python
   # 好的做法
   if validate_format and validate_size:
       save_file()
   ```

2. **使用唯一文件名避免覆盖**
   ```python
   # 好的做法
   unique_name = FileUtils.generate_unique_filename(original_name)
   save_path = f'uploads/{unique_name}'
   ```

3. **预先创建目录**
   ```python
   # 好的做法
   FileUtils.create_dir_if_not_exists('uploads/')
   FileUtils.create_dir_if_not_exists('results/')
   ```

4. **检查返回状态**
   ```python
   # 好的做法
   success, msg = FileUtils.save_uploaded_file(file, path)
   if not success:
       return {"error": msg}
   ```

### ❌ 不推荐做法

1. **不验证直接保存**
   ```python
   # 不好的做法
   save_file()  # 可能上传病毒文件
   ```

2. **直接使用原文件名**
   ```python
   # 不好的做法
   save_path = f'uploads/{file.filename}'  # 容易被覆盖
   ```

3. **忽略错误处理**
   ```python
   # 不好的做法
   FileUtils.save_uploaded_file(file, path)  # 不检查返回值
   ```

---

## 常见问题

### Q1：如何修改默认允许的文件格式？

**A**：在验证时传入`allow_formats`参数：

```python
is_valid, msg = FileUtils.validate_file_format(
    'document.pdf',
    allow_formats=['pdf', 'doc', 'docx']
)
```

### Q2：如何修改默认的最大文件大小？

**A**：在验证时传入`max_size`参数：

```python
is_valid, msg = FileUtils.validate_file_size(
    file_obj,
    max_size=100 * 1024 * 1024  # 100MB
)
```

### Q3：生成的文件名是否真的唯一？

**A**：是的，使用了UUID v4：
- UUID v4是128位随机数
- 碰撞概率极低（约1/5*10^36）
- 足以保证同一系统内的唯一性

### Q4：如何恢复原始文件名？

**A**：保存映射关系：

```python
# 保存到数据库
original_name = file.filename
unique_name = FileUtils.generate_unique_filename(original_name)
db.insert({
    'original_name': original_name,
    'unique_name': unique_name
})
```

### Q5：文件保存失败怎么办？

**A**：检查返回状态和错误信息：

```python
success, msg = FileUtils.save_uploaded_file(file, path)
if not success:
    print(f"保存失败: {msg}")
    # 可能原因：
    # - 权限不足
    # - 磁盘满
    # - 路径无效
```

---

## 完整示例

### 完整的ImageController集成示例

```python
from fastapi import UploadFile
from utils import FileUtils
from models import segment_octa_image

class ImageController:
    
    UPLOAD_DIR = './uploads'
    RESULTS_DIR = './results'
    
    @staticmethod
    async def segment_octa(file: UploadFile, model_type: str = 'unet'):
        """OCTA图像分割接口（使用FileUtils）"""
        
        try:
            # ==================== 步骤1：文件验证 ====================
            is_valid, error_msg = FileUtils.validate_file_format(file.filename)
            if not is_valid:
                return {"success": False, "error": f"格式错误: {error_msg}"}
            
            is_valid, error_msg = FileUtils.validate_file_size(file)
            if not is_valid:
                return {"success": False, "error": f"大小错误: {error_msg}"}
            
            # ==================== 步骤2：生成唯一文件名 ====================
            unique_filename = FileUtils.generate_unique_filename(file.filename)
            
            # ==================== 步骤3：确保目录存在 ====================
            FileUtils.create_dir_if_not_exists(ImageController.UPLOAD_DIR)
            FileUtils.create_dir_if_not_exists(ImageController.RESULTS_DIR)
            
            # ==================== 步骤4：保存原始图像 ====================
            upload_path = f"{ImageController.UPLOAD_DIR}/{unique_filename}"
            success, msg = FileUtils.save_uploaded_file(file, upload_path)
            if not success:
                return {"success": False, "error": f"保存失败: {msg}"}
            
            # ==================== 步骤5：执行分割 ====================
            result_path = segment_octa_image(upload_path, model_type)
            
            # ==================== 步骤6：返回结果 ====================
            return {
                "success": True,
                "upload_path": upload_path,
                "result_path": result_path,
                "filename": unique_filename
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
```

---

## 总结

FileUtils提供了5个核心方法，涵盖了文件处理的所有环节：

| 方法 | 功能 | 返回值 |
|-----|------|--------|
| `validate_file_format()` | 格式验证 | `(bool, str)` |
| `validate_file_size()` | 大小验证 | `(bool, str)` |
| `generate_unique_filename()` | 文件名生成 | `str` |
| `create_dir_if_not_exists()` | 目录创建 | `bool` |
| `save_uploaded_file()` | 文件保存 | `(bool, str)` |

通过使用FileUtils，代码变得更加清晰、可维护、可测试。

---

**文档版本**：1.0 | **最后更新**：2026年1月14日
