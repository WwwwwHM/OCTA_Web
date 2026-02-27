# ImageDAO × ImageController 集成指南

## 📌 概述

本文档说明如何将新创建的**ImageDAO**数据层集成到现有的**ImageController**控制层，实现完整的**三层分离架构**。

**集成目标**：
- ✅ 将ImageController中的SQL操作移到ImageDAO
- ✅ 保持API接口完全不变
- ✅ 减少ImageController代码复杂度
- ✅ 提高代码可维护性和可测试性

---

## 🔄 集成前后对比

### 集成前（现状）

```
ImageController（1420行）
├── 业务逻辑（segment_octa、get_all_history等）
├── 文件操作（_generate_unique_filename）
├── 数据验证（_validate_image_file）
└── ⚠️ 数据库SQL操作（应该分离）
    ├── _insert_record() - SQL INSERT
    ├── _get_all_records() - SQL SELECT
    └── _get_record_by_id() - SQL SELECT WHERE

+ 
ImageDAO（独立DAO层）
└── 专门的数据库操作类（新增，尚未集成）
```

### 集成后（目标）

```
ImageController（~ 1200行，精简~ 220行）
├── 业务逻辑（segment_octa、get_all_history等）
├── 文件操作（_generate_unique_filename）
├── 数据验证（_validate_image_file）
└── ✅ 调用ImageDAO进行数据库操作
    ├── ImageDAO.insert_record()
    ├── ImageDAO.get_all_records()
    └── ImageDAO.get_record_by_id()

+
ImageDAO（专门的数据层）
├── init_db()
├── insert_record()
├── get_all_records()
├── get_record_by_id()
└── delete_record_by_id()  ← 新增功能

=
完整的三层架构
├── 路由层（main.py）
├── 控制层（ImageController）
└── 数据层（ImageDAO）  ← 新增
```

---

## 📋 集成步骤

### 第1步：在ImageController中导入ImageDAO

**文件**：`octa_backend/controller/image_controller.py`

**操作**：在文件顶部找到导入部分，添加ImageDAO导入

```python
# 在现有导入之后添加
from dao import ImageDAO  # ← 新增此行
```

**现有导入示例**：
```python
from fastapi import HTTPException, UploadFile, File
from pathlib import Path
import uuid
import sqlite3
from datetime import datetime
from typing import Optional, Dict, List
import os

# ← 在此处添加
from dao import ImageDAO  # ← 新增
```

---

### 第2步：修改init_database()方法

**旧方法**（直接创建表）：
```python
@staticmethod
def init_database() -> bool:
    """初始化数据库和目录"""
    try:
        # ... 创建目录的代码 ...
        
        # ❌ 旧方式：直接执行SQL
        conn = sqlite3.connect(ImageController.DB_NAME)
        conn.execute(ImageController.CREATE_TABLE_SQL)
        conn.commit()
        conn.close()
        
        return True
    except Exception as e:
        print(f"[ERROR] 初始化失败: {e}")
        return False
```

**新方法**（使用ImageDAO）：
```python
@staticmethod
def init_database() -> bool:
    """初始化数据库和目录"""
    try:
        # ... 创建目录的代码 ...
        
        # ✅ 新方式：使用DAO初始化
        success = ImageDAO.init_db(ImageController.DB_NAME)
        if not success:
            return False
        
        return True
    except Exception as e:
        print(f"[ERROR] 初始化失败: {e}")
        return False
```

---

### 第3步：删除ImageController中的私有数据库方法

**删除以下方法**（因为功能已由ImageDAO提供）：

```python
# ❌ 删除这些方法，改用ImageDAO替代

@staticmethod
def _insert_record(filename: str, upload_time: str, model_type: str, 
                   original_path: str, result_path: str) -> Optional[int]:
    # 旧实现：直接SQL操作
    # ← 替换为 ImageDAO.insert_record()
    pass

@staticmethod
def _get_all_records() -> List[Dict]:
    # 旧实现：直接SQL操作
    # ← 替换为 ImageDAO.get_all_records()
    pass

@staticmethod
def _get_record_by_id(record_id: int) -> Optional[Dict]:
    # 旧实现：直接SQL操作
    # ← 替换为 ImageDAO.get_record_by_id()
    pass
```

---

### 第4步：修改segment_octa()方法

**旧代码**（直接调用_insert_record）：
```python
@classmethod
async def segment_octa(cls, file: UploadFile, model_type: str):
    # ... 文件验证、模型调用等逻辑 ...
    
    # ❌ 旧方式
    record_id = cls._insert_record(
        filename=filename,
        upload_time=datetime.now().isoformat(),
        model_type=model_type,
        original_path=str(upload_path),
        result_path=str(result_path)
    )
```

**新代码**（调用ImageDAO）：
```python
@classmethod
async def segment_octa(cls, file: UploadFile, model_type: str):
    # ... 文件验证、模型调用等逻辑 ...
    
    # ✅ 新方式
    record_id = ImageDAO.insert_record(
        filename=filename,
        upload_time=datetime.now().isoformat(),
        model_type=model_type,
        original_path=str(upload_path),
        result_path=str(result_path),
        db_path=cls.DB_NAME
    )
```

---

### 第5步：修改get_all_history()方法

**旧代码**：
```python
@classmethod
def get_all_history(cls) -> JSONResponse:
    try:
        # ❌ 旧方式
        records = cls._get_all_records()  # 调用私有方法
        
        if not records:
            return JSONResponse(
                status_code=404,
                content={"message": "暂无分割历史"}
            )
        
        return JSONResponse(
            status_code=200,
            content={"data": records}
        )
    except Exception as e:
        # 异常处理
        pass
```

**新代码**：
```python
@classmethod
def get_all_history(cls) -> JSONResponse:
    try:
        # ✅ 新方式
        records = ImageDAO.get_all_records(cls.DB_NAME)  # 调用DAO
        
        if not records:
            return JSONResponse(
                status_code=404,
                content={"message": "暂无分割历史"}
            )
        
        return JSONResponse(
            status_code=200,
            content={"data": records}
        )
    except Exception as e:
        # 异常处理
        pass
```

---

### 第6步：修改get_history_by_id()方法

**旧代码**：
```python
@classmethod
def get_history_by_id(cls, record_id: int) -> JSONResponse:
    try:
        # ❌ 旧方式
        record = cls._get_record_by_id(record_id)  # 调用私有方法
        
        if not record:
            raise HTTPException(
                status_code=404,
                detail="未找到该历史记录"
            )
        
        return JSONResponse(
            status_code=200,
            content={"data": record}
        )
    except Exception as e:
        # 异常处理
        pass
```

**新代码**：
```python
@classmethod
def get_history_by_id(cls, record_id: int) -> JSONResponse:
    try:
        # ✅ 新方式
        record = ImageDAO.get_record_by_id(record_id, cls.DB_NAME)  # 调用DAO
        
        if not record:
            raise HTTPException(
                status_code=404,
                detail="未找到该历史记录"
            )
        
        return JSONResponse(
            status_code=200,
            content={"data": record}
        )
    except Exception as e:
        # 异常处理
        pass
```

---

### 第7步：修改delete_history_by_id()方法

**旧代码**（如果有的话）：
```python
@classmethod
def delete_history_by_id(cls, record_id: int) -> JSONResponse:
    try:
        # ❌ 旧方式：直接SQL操作
        conn = sqlite3.connect(cls.DB_NAME)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM images WHERE id = ?", (record_id,))
        conn.commit()
        
        if cursor.rowcount == 0:
            raise HTTPException(
                status_code=404,
                detail="未找到该历史记录"
            )
        
        conn.close()
        # ...
    except Exception as e:
        pass
```

**新代码**：
```python
@classmethod
def delete_history_by_id(cls, record_id: int) -> JSONResponse:
    try:
        # ✅ 新方式
        success = ImageDAO.delete_record_by_id(record_id, cls.DB_NAME)  # 调用DAO
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="未找到该历史记录"
            )
        
        return JSONResponse(
            status_code=200,
            content={"message": "历史记录删除成功"}
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"删除失败: {str(e)}"
        )
```

---

## 🔍 集成验证清单

完成上述步骤后，检查以下项目：

### 代码检查

- [ ] ImageDAO已导入到ImageController
- [ ] `_insert_record()` 方法已删除
- [ ] `_get_all_records()` 方法已删除
- [ ] `_get_record_by_id()` 方法已删除
- [ ] 所有对这些方法的调用已改为调用ImageDAO
- [ ] `init_database()` 已改为调用`ImageDAO.init_db()`
- [ ] 所有SQL语句都已从ImageController中移除

### 功能测试

```python
# 测试脚本（在octa_backend目录下运行）

from controller import ImageController
from dao import ImageDAO
from datetime import datetime

# 1. 初始化
print("[测试] 初始化数据库...")
ImageController.init_database()

# 2. 插入记录
print("[测试] 插入记录...")
id1 = ImageDAO.insert_record(
    filename='test1.png',
    upload_time=datetime.now().isoformat(),
    model_type='unet',
    original_path='uploads/test1.png',
    result_path='results/test1_seg.png'
)
print(f"✓ 插入成功: ID={id1}")

# 3. 查询所有
print("[测试] 查询所有...")
records = ImageDAO.get_all_records()
print(f"✓ 查询成功: {len(records)}条记录")

# 4. 按ID查询
print("[测试] 按ID查询...")
record = ImageDAO.get_record_by_id(id1)
print(f"✓ 查询成功: {record['filename']}")

# 5. 删除
print("[测试] 删除...")
success = ImageDAO.delete_record_by_id(id1)
print(f"✓ 删除成功: {success}")

print("\n✅ 所有集成测试通过！")
```

### API测试

```bash
# 启动后端
python main.py

# 在另一个终端测试API
curl http://127.0.0.1:8000/history/
# 应该返回历史列表（使用ImageDAO获取）
```

---

## 📊 集成前后代码量对比

### ImageController代码量变化

| 部分 | 集成前 | 集成后 | 变化 |
|-----|-------|-------|------|
| 导入语句 | 8行 | 9行 | +1 |
| 常量定义 | 4行 | 4行 | ±0 |
| init_database() | 15行 | 10行 | -5 |
| segment_octa() | 150行 | 150行* | ±0* |
| get_all_history() | 30行 | 25行 | -5 |
| get_history_by_id() | 25行 | 20行 | -5 |
| delete_history_by_id() | 30行 | 25行 | -5 |
| 私有方法 | 200行 | 50行 | -150 |
| **总计** | **1420行** | **~1260行** | **-160行** |

*segment_octa()的插入部分会精简，因为调用ImageDAO只需1行代码

---

## 🎯 集成完成后的好处

### 1. 职责分离
```
┌─────────────────────────────────────────┐
│        ImageController (1260行)          │
│   ✓ 仅专注业务逻辑（分割、历史查询）     │
│   ✓ 不包含任何SQL代码                    │
│   ✓ 调用ImageDAO进行数据库操作          │
└─────────────────────────────────────────┘
                  │
                  ↓ 调用
┌─────────────────────────────────────────┐
│        ImageDAO (690行)                  │
│   ✓ 专注数据库操作                       │
│   ✓ 封装所有SQL逻辑                     │
│   ✓ 可独立测试                          │
└─────────────────────────────────────────┘
```

### 2. 易于维护
- 修改数据库结构只需改DAO
- 修改业务逻辑只需改Controller
- 两层互不影响

### 3. 易于测试
```python
# 可独立测试DAO
from dao import ImageDAO
def test_insert():
    id = ImageDAO.insert_record(...)
    assert id is not None

# 可独立测试Controller
from controller import ImageController
def test_segment_octa():
    result = ImageController.segment_octa(...)
    assert result['status'] == 'success'
```

### 4. 易于扩展
```python
# 想支持新的数据库？只需修改DAO
class ImageDAO:
    @staticmethod
    def init_db_postgresql(connection_string):
        # PostgreSQL实现
        pass

# Controller代码无需改动
```

---

## ⚠️ 注意事项

### 1. 数据库路径参数

ImageDAO的所有方法都接受可选的`db_path`参数：

```python
# 使用默认路径
ImageDAO.insert_record(filename='test.png', ...)

# 使用自定义路径
ImageDAO.insert_record(filename='test.png', ..., db_path='./data/octa.db')
```

在ImageController中，应该使用`cls.DB_NAME`常量：

```python
record_id = ImageDAO.insert_record(
    filename=filename,
    ...,
    db_path=cls.DB_NAME  # 使用Controller定义的常量
)
```

### 2. 返回值处理

不同的ImageDAO方法返回不同类型的值：

```python
# 插入：返回ID（int）或None
record_id = ImageDAO.insert_record(...)
if record_id:
    print(f"成功: {record_id}")
else:
    print("失败")

# 查询：返回列表（可能为空）
records = ImageDAO.get_all_records()
if records:
    for r in records:
        print(r)
else:
    print("无记录")

# 删除：返回bool
success = ImageDAO.delete_record_by_id(1)
if success:
    print("成功")
else:
    print("失败")
```

### 3. 异常处理

ImageDAO已处理所有SQLite异常，返回None/False。Controller可以直接根据返回值判断成功失败：

```python
# ✅ 推荐做法
record_id = ImageDAO.insert_record(...)
if record_id:
    # 成功
    pass
else:
    # 失败
    raise HTTPException(status_code=500, detail="插入失败")

# ❌ 不需要try-except
# ImageDAO已处理异常
```

---

## 🚀 集成完成后的下一步

1. **运行测试**
   ```bash
   python -m dao.image_dao  # DAO单元测试
   pytest tests/test_controller.py  # Controller集成测试
   ```

2. **启动后端**
   ```bash
   python main.py
   ```

3. **前端测试**
   - 上传图像进行分割
   - 查看历史记录
   - 删除历史记录

4. **性能监控**
   - 监控数据库查询时间
   - 检查连接是否正常关闭
   - 监控内存使用情况

---

## 📝 集成总结

| 步骤 | 描述 | 状态 |
|-----|------|------|
| 1 | 导入ImageDAO | [ ] |
| 2 | 修改init_database() | [ ] |
| 3 | 删除私有数据库方法 | [ ] |
| 4 | 修改segment_octa() | [ ] |
| 5 | 修改get_all_history() | [ ] |
| 6 | 修改get_history_by_id() | [ ] |
| 7 | 修改delete_history_by_id() | [ ] |
| ✅ | 验证所有功能 | [ ] |
| ✅ | 运行单元测试 | [ ] |
| ✅ | 启动后端测试 | [ ] |

---

**版本**：1.0  
**更新日期**：2026年1月14日  
**作者**：OCTA Web项目组

