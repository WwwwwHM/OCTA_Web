# 文件管理DAO使用指南

## 📚 概述

`file_dao.py` 提供了完整的文件管理数据库CRUD操作，用于追踪上传的图片和数据集文件元信息。

---

## 🗄️ 数据库表结构

### file_management 表

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | INTEGER | PRIMARY KEY, AUTOINCREMENT | 记录ID |
| file_name | TEXT | NOT NULL | 原始文件名 |
| file_path | TEXT | NOT NULL | 本地存储路径 |
| file_type | TEXT | NOT NULL | 文件类型（'image' 或 'dataset'） |
| upload_time | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 上传时间 |
| related_model | TEXT | NULL | 关联模型权重路径 |
| file_size | REAL | NULL | 文件大小（MB） |

---

## 🚀 快速开始

### 1. 导入模块

```python
from dao.file_dao import FileDAO
```

### 2. 初始化数据库（自动执行）

模块导入时会自动创建表，无需手动调用：

```python
# 自动执行：FileDAO.create_file_table()
```

### 3. 添加文件记录

```python
# 添加图片文件
file_id = FileDAO.add_file_record(
    file_name='octa_001.png',
    file_path='uploads/images/octa_001.png',
    file_type='image',
    file_size=2.5
)

# 添加数据集文件
dataset_id = FileDAO.add_file_record(
    file_name='training_set.zip',
    file_path='uploads/datasets/training_set.zip',
    file_type='dataset',
    related_model='models/weights/unet_octa.pth',
    file_size=120.8
)
```

### 4. 查询文件

```python
# 查询所有文件
all_files = FileDAO.get_file_list()

# 查询所有图片
images = FileDAO.get_file_list(file_type='image')

# 查询所有数据集
datasets = FileDAO.get_file_list(file_type='dataset')

# 查询单个文件
file_info = FileDAO.get_file_by_id(1)
```

### 5. 更新文件关联

```python
# 训练完成后更新模型关联
success = FileDAO.update_file_relation(
    file_id=1,
    related_model='models/weights/unet_trained_20260116.pth'
)
```

### 6. 删除文件

```python
# 删除数据库记录 + 本地文件
success = FileDAO.delete_file(file_id=1)
```

---

## 📖 API详细说明

### create_file_table()

**功能：** 创建file_management表（若不存在）

**返回：** `bool` - 成功True，失败False

**示例：**
```python
if FileDAO.create_file_table():
    print("表创建成功")
```

---

### add_file_record()

**功能：** 添加文件记录到数据库

**参数：**
- `file_name` (str): 原始文件名，必填
- `file_path` (str): 文件存储路径，必填
- `file_type` (str): 文件类型（'image' 或 'dataset'），必填
- `related_model` (Optional[str]): 关联模型路径，可选
- `file_size` (Optional[float]): 文件大小（MB），可选

**返回：** `Optional[int]` - 成功返回记录ID，失败返回None

**示例：**
```python
# 最简示例
file_id = FileDAO.add_file_record(
    file_name='test.png',
    file_path='uploads/test.png',
    file_type='image'
)

# 完整示例
file_id = FileDAO.add_file_record(
    file_name='dataset.zip',
    file_path='uploads/datasets/dataset.zip',
    file_type='dataset',
    related_model='models/weights/unet.pth',
    file_size=50.5
)
```

---

### get_file_list()

**功能：** 查询文件列表，支持按类型筛选

**参数：**
- `file_type` (Optional[str]): 筛选类型（None=全部，'image'=图片，'dataset'=数据集）

**返回：** `List[Dict]` - 文件记录列表，每条记录为字典格式

**字典格式：**
```python
{
    'id': 1,
    'file_name': 'test.png',
    'file_path': 'uploads/test.png',
    'file_type': 'image',
    'upload_time': '2026-01-16 10:30:00',
    'related_model': None,
    'file_size': 2.5
}
```

**示例：**
```python
# 查询所有文件
all_files = FileDAO.get_file_list()
print(f"共有{len(all_files)}个文件")

# 查询所有图片
images = FileDAO.get_file_list(file_type='image')
for img in images:
    print(f"{img['file_name']}: {img['file_size']} MB")

# 查询所有数据集
datasets = FileDAO.get_file_list(file_type='dataset')
for ds in datasets:
    print(f"{ds['file_name']} -> {ds['related_model']}")
```

---

### get_file_by_id()

**功能：** 按ID查询单个文件信息

**参数：**
- `file_id` (int): 文件记录ID

**返回：** `Optional[Dict]` - 找到返回字典，未找到返回None

**示例：**
```python
file_info = FileDAO.get_file_by_id(1)

if file_info:
    print(f"文件名: {file_info['file_name']}")
    print(f"文件大小: {file_info['file_size']} MB")
    print(f"上传时间: {file_info['upload_time']}")
else:
    print("文件不存在")
```

---

### update_file_relation()

**功能：** 更新文件关联的模型权重

**参数：**
- `file_id` (int): 文件记录ID
- `related_model` (str): 模型权重文件路径

**返回：** `bool` - 成功True，失败False

**使用场景：** 训练完成后，将训练使用的数据集与生成的模型权重建立关联

**示例：**
```python
# 训练完成后关联模型
success = FileDAO.update_file_relation(
    file_id=1,
    related_model='models/weights/unet_trained_20260116.pth'
)

if success:
    print("模型关联更新成功")
```

---

### delete_file()

**功能：** 删除文件记录和本地文件

**参数：**
- `file_id` (int): 要删除的文件记录ID

**返回：** `bool` - 成功True，失败False

**执行步骤：**
1. 查询文件路径
2. 删除数据库记录
3. 删除本地文件或目录

**特性：**
- 自动区分文件和目录（目录使用shutil.rmtree递归删除）
- 路径存在性校验（避免FileNotFoundError）
- 数据库与文件系统同步

**示例：**
```python
# 删除单个文件
if FileDAO.delete_file(1):
    print("文件删除成功")
else:
    print("文件删除失败")
```

---

## 🎯 实际应用场景

### 场景1：文件上传处理

```python
from fastapi import UploadFile
from dao.file_dao import FileDAO
import os

async def handle_file_upload(file: UploadFile, file_type: str):
    """处理文件上传"""
    
    # 保存文件到本地
    save_path = f"uploads/{file_type}s/{file.filename}"
    with open(save_path, 'wb') as f:
        content = await file.read()
        f.write(content)
    
    # 计算文件大小（MB）
    file_size = len(content) / (1024 * 1024)
    
    # 添加数据库记录
    file_id = FileDAO.add_file_record(
        file_name=file.filename,
        file_path=save_path,
        file_type=file_type,
        file_size=file_size
    )
    
    return {
        'file_id': file_id,
        'file_name': file.filename,
        'file_size': file_size
    }
```

### 场景2：文件列表展示

```python
from dao.file_dao import FileDAO

def get_file_management_page():
    """获取文件管理页面数据"""
    
    # 获取所有文件
    all_files = FileDAO.get_file_list()
    
    # 统计信息
    total_count = len(all_files)
    image_count = len([f for f in all_files if f['file_type'] == 'image'])
    dataset_count = len([f for f in all_files if f['file_type'] == 'dataset'])
    total_size = sum(f['file_size'] or 0 for f in all_files)
    
    return {
        'files': all_files,
        'statistics': {
            'total_count': total_count,
            'image_count': image_count,
            'dataset_count': dataset_count,
            'total_size_mb': round(total_size, 2)
        }
    }
```

### 场景3：模型训练流程

```python
from dao.file_dao import FileDAO

def train_model_workflow(dataset_id: int):
    """模型训练流程"""
    
    # 1. 获取数据集信息
    dataset_info = FileDAO.get_file_by_id(dataset_id)
    if not dataset_info:
        raise ValueError(f"数据集不存在: {dataset_id}")
    
    dataset_path = dataset_info['file_path']
    
    # 2. 训练模型
    model = train_model(dataset_path)
    
    # 3. 保存模型权重
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'models/weights/unet_{timestamp}.pth'
    save_model(model, model_path)
    
    # 4. 更新数据集关联
    success = FileDAO.update_file_relation(dataset_id, model_path)
    
    if success:
        print(f"训练完成，模型保存至: {model_path}")
    
    return model_path
```

### 场景4：文件清理

```python
from dao.file_dao import FileDAO
from datetime import datetime, timedelta

def cleanup_old_files(days: int = 30):
    """清理超过指定天数的文件"""
    
    # 获取所有文件
    all_files = FileDAO.get_file_list()
    
    # 计算截止时间
    cutoff_date = datetime.now() - timedelta(days=days)
    
    deleted_count = 0
    for file in all_files:
        # 解析上传时间
        upload_time = datetime.strptime(file['upload_time'], '%Y-%m-%d %H:%M:%S')
        
        # 如果超过指定天数，删除
        if upload_time < cutoff_date:
            if FileDAO.delete_file(file['id']):
                deleted_count += 1
                print(f"已删除: {file['file_name']}")
    
    print(f"共清理{deleted_count}个过期文件")
    return deleted_count
```

---

## ⚠️ 注意事项

### 1. 文件类型验证

文件类型必须为 `'image'` 或 `'dataset'`，否则会被拒绝：

```python
# ✓ 正确
FileDAO.add_file_record(..., file_type='image')
FileDAO.add_file_record(..., file_type='dataset')

# ✗ 错误
FileDAO.add_file_record(..., file_type='video')  # 会返回None
```

### 2. 文件路径管理

建议使用相对路径，便于项目迁移：

```python
# ✓ 推荐：相对路径
file_path='uploads/images/test.png'

# ✗ 不推荐：绝对路径（不利于迁移）
file_path='D:/Code/OCTA_Web/uploads/images/test.png'
```

### 3. 文件删除同步

`delete_file()` 会同步删除数据库记录和本地文件：

```python
# 删除操作包括：
# 1. 删除数据库记录
# 2. 删除本地文件/目录

success = FileDAO.delete_file(1)
# 如果本地文件不存在，仍然会删除数据库记录
```

### 4. 异常处理

所有函数都有完善的异常处理，失败时返回False或None：

```python
# 添加记录失败返回None
file_id = FileDAO.add_file_record(...)
if not file_id:
    print("添加失败")

# 查询失败返回None或空列表
file_info = FileDAO.get_file_by_id(999)
if not file_info:
    print("文件不存在")

# 删除失败返回False
success = FileDAO.delete_file(999)
if not success:
    print("删除失败")
```

---

## 🔍 调试技巧

### 查看SQL日志

所有函数都会打印详细日志：

```
[SUCCESS] 文件记录添加成功
[INFO] 记录ID: 1
[INFO] 文件名: test.png
[INFO] 文件类型: image
[INFO] 文件大小: 2.5 MB
```

### 数据库路径

默认使用 `./octa.db`，可在 `config/config.py` 中修改：

```python
# config/config.py
DB_PATH = "./octa.db"
```

### 直接查询数据库

```bash
# 使用SQLite命令行工具
sqlite3 octa.db

# 查看表结构
.schema file_management

# 查询所有记录
SELECT * FROM file_management;

# 退出
.quit
```

---

## 📊 性能建议

### 1. 批量查询

查询所有文件后在内存中筛选，避免多次数据库查询：

```python
# ✓ 推荐：一次查询，内存筛选
all_files = FileDAO.get_file_list()
images = [f for f in all_files if f['file_type'] == 'image']
datasets = [f for f in all_files if f['file_type'] == 'dataset']

# ✗ 不推荐：多次数据库查询
images = FileDAO.get_file_list(file_type='image')
datasets = FileDAO.get_file_list(file_type='dataset')
```

### 2. 缓存文件列表

前端展示时可以缓存文件列表，减少数据库查询：

```python
from functools import lru_cache

@lru_cache(maxsize=1)
def get_cached_file_list():
    return FileDAO.get_file_list()

# 清除缓存（文件更新后）
get_cached_file_list.cache_clear()
```

---

## 🧪 测试

运行测试脚本验证功能：

```bash
cd octa_backend
python test_file_dao.py
```

测试内容：
- ✓ 数据库表创建
- ✓ 添加文件记录（图片和数据集）
- ✓ 查询所有文件
- ✓ 按类型筛选查询
- ✓ 按ID查询单个文件
- ✓ 更新文件关联模型
- ✓ 删除文件记录和本地文件
- ✓ 异常情况处理

---

## 📝 总结

**核心功能：**
- ✅ 完整的CRUD操作
- ✅ 文件类型分类管理
- ✅ 模型关联追踪
- ✅ 数据库与文件系统同步
- ✅ 完善的异常处理

**使用场景：**
- 文件上传管理
- 文件列表展示
- 模型训练追踪
- 文件清理维护

**技术特点：**
- 静态方法设计，无需实例化
- 所有操作自动关闭数据库连接
- 友好的错误提示和日志
- 支持文件和目录删除

---

**作者：** OCTA Web项目组  
**日期：** 2026年1月16日
