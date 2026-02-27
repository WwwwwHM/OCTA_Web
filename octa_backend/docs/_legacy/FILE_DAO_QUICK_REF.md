# 📋 文件管理DAO快速参考卡

## 🎯 核心功能（6个函数）

```python
from dao.file_dao import FileDAO
```

---

### 1️⃣ 创建表
```python
FileDAO.create_file_table()
# 返回: bool（自动执行，无需手动调用）
```

---

### 2️⃣ 添加文件
```python
file_id = FileDAO.add_file_record(
    file_name='test.png',          # 必填：文件名
    file_path='uploads/test.png',  # 必填：存储路径
    file_type='image',             # 必填：'image' 或 'dataset'
    related_model=None,            # 可选：关联模型路径
    file_size=None                 # 可选：文件大小（MB）
)
# 返回: int（记录ID）或 None（失败）
```

---

### 3️⃣ 查询列表
```python
# 查询所有文件
all_files = FileDAO.get_file_list()

# 查询图片文件
images = FileDAO.get_file_list(file_type='image')

# 查询数据集文件
datasets = FileDAO.get_file_list(file_type='dataset')

# 返回: List[Dict]（字典列表）或 []（失败/无记录）
```

---

### 4️⃣ 查询单个
```python
file_info = FileDAO.get_file_by_id(file_id=1)
# 返回: Dict（文件信息）或 None（不存在）
```

---

### 5️⃣ 更新关联
```python
success = FileDAO.update_file_relation(
    file_id=1,
    related_model='models/weights/unet.pth'
)
# 返回: bool
```

---

### 6️⃣ 删除文件
```python
success = FileDAO.delete_file(file_id=1)
# 删除数据库记录 + 本地文件/目录
# 返回: bool
```

---

## 📊 返回数据格式

### 文件记录字典
```python
{
    'id': 1,                                          # 记录ID
    'file_name': 'train_data.zip',                    # 文件名
    'file_path': 'uploads/datasets/train_data.zip',   # 存储路径
    'file_type': 'dataset',                           # 类型
    'upload_time': '2026-01-16 10:30:00',            # 上传时间
    'related_model': 'models/weights/unet.pth',      # 关联模型
    'file_size': 45.6                                # 大小（MB）
}
```

---

## ⚡ 常用代码片段

### 文件上传
```python
# 1. 保存文件到本地
save_path = f'uploads/{file_type}s/{filename}'
with open(save_path, 'wb') as f:
    f.write(content)

# 2. 添加数据库记录
file_id = FileDAO.add_file_record(
    file_name=filename,
    file_path=save_path,
    file_type=file_type,
    file_size=len(content) / (1024 * 1024)
)
```

### 展示文件列表
```python
files = FileDAO.get_file_list()
for file in files:
    print(f"{file['file_name']}: {file['file_size']} MB")
```

### 训练后关联模型
```python
# 训练完成
model_path = train_model(dataset_path)

# 更新关联
FileDAO.update_file_relation(dataset_id, model_path)
```

### 删除文件
```python
if FileDAO.delete_file(file_id):
    return {'message': '删除成功'}
else:
    return {'error': '删除失败'}
```

---

## ⚠️ 重要提示

| 项目 | 说明 |
|------|------|
| **文件类型** | 仅支持 `'image'` 和 `'dataset'` |
| **路径格式** | 推荐相对路径（便于迁移） |
| **删除操作** | 同步删除数据库记录和本地文件 |
| **异常处理** | 所有函数都有完善的异常捕获 |
| **数据库路径** | 配置在 `config/config.py` 的 `DB_PATH` |

---

## 🧪 快速测试

```bash
cd octa_backend
python test_file_dao.py
```

---

## 📚 详细文档

查看完整文档：[FILE_DAO_GUIDE.md](FILE_DAO_GUIDE.md)

---

**版本：** 1.0  
**日期：** 2026-01-16
