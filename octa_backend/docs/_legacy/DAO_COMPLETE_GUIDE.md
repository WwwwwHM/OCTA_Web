# ImageDAO 数据访问对象 - 完整文档

## 📌 概述

**ImageDAO（Image Data Access Object）**是OCTA平台的**数据层**，专门负责与SQLite数据库的所有交互操作。

**设计目的**：
- 🔹 **数据库隔离**：所有SQL操作集中在DAO中，业务逻辑层无需接触数据库细节
- 🔹 **接口清晰**：提供统一的CRUD接口，隐藏SQL复杂性
- 🔹 **易于维护**：修改数据库结构只需改DAO，不影响其他层
- 🔹 **便于测试**：DAO可独立单元测试，不依赖FastAPI等框架

---

## 🏗️ 完整的OCTA架构（五层）

```
┌─────────────────────────────────────────┐
│       路由层（main.py - 130行）          │
│   负责：HTTP请求路由和响应转发           │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│    控制层（ImageController - 1420行）     │
│  负责：业务逻辑编排、数据验证、异常处理   │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│       数据层（ImageDAO - 新增）          │  ← 本次新增
│    负责：所有SQLite数据库操作（本文档）   │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│   业务层（models/ - UNet/FCN - 630行）   │
│  负责：图像处理、模型推理、结果后处理     │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  文件系统层（uploads/ + results/）       │
│         负责：文件I/O操作                 │
└─────────────────────────────────────────┘
```

---

## 📊 ImageDAO 类方法速览

| 方法 | 类型 | 功能 | 返回值 |
|-----|------|------|--------|
| `init_db()` | 静态方法 | 初始化数据库，创建表 | `bool` |
| `insert_record()` | 静态方法 | 插入分割记录 | `Optional[int]` |
| `get_all_records()` | 静态方法 | 查询所有记录（倒序） | `List[Dict]` |
| `get_record_by_id()` | 静态方法 | 按ID查询单条记录 | `Optional[Dict]` |
| `delete_record_by_id()` | 静态方法 | 按ID删除记录 | `bool` |

---

## 💾 数据库表结构

### images 表

```sql
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,          -- 自增主键
    filename TEXT UNIQUE NOT NULL,                 -- 文件名（唯一约束）
    upload_time TEXT NOT NULL,                     -- 上传时间（ISO 8601格式）
    model_type TEXT NOT NULL,                      -- 模型类型（'unet'/'fcn'）
    original_path TEXT NOT NULL,                   -- 原始图像路径
    result_path TEXT NOT NULL,                     -- 分割结果路径
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP -- 创建时间（数据库自动）
)
```

### 字段说明

| 字段 | 类型 | 约束 | 说明 |
|-----|------|------|------|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT | 记录的唯一标识符，自动递增 |
| `filename` | TEXT | UNIQUE NOT NULL | 上传的文件名，应为UUID格式（防重复） |
| `upload_time` | TEXT | NOT NULL | 上传时间，使用ISO 8601格式（2026-01-14T10:30:00） |
| `model_type` | TEXT | NOT NULL | 使用的分割模型（'unet' 或 'fcn'） |
| `original_path` | TEXT | NOT NULL | 原始图像在服务器上的相对路径 |
| `result_path` | TEXT | NOT NULL | 分割结果在服务器上的相对路径 |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 数据库记录创建时间（系统自动） |

### 索引说明

```sql
-- 主键索引（自动创建）
PRIMARY KEY (id)  -- 用于快速查询

-- UNIQUE索引（自动创建）
UNIQUE (filename)  -- 防止重复上传同一文件名

-- 推荐添加的索引（可扩展）
CREATE INDEX idx_upload_time ON images(upload_time DESC);  -- 加速排序查询
CREATE INDEX idx_model_type ON images(model_type);         -- 加速按模型查询
```

---

## 📖 完整使用指南

### 1️⃣ 初始化数据库

```python
from dao import ImageDAO

# 初始化数据库（创建表）
success = ImageDAO.init_db('./octa.db')

if success:
    print("✓ 数据库初始化成功")
else:
    print("✗ 数据库初始化失败")
```

**特点**：
- 自动创建目录（如果不存在）
- 自动创建数据库文件
- IF NOT EXISTS：重复调用不会出错

---

### 2️⃣ 插入记录

```python
from dao import ImageDAO
from datetime import datetime

record_id = ImageDAO.insert_record(
    filename='img_abc123def456.png',           # UUID文件名（必须唯一）
    upload_time=datetime.now().isoformat(),    # ISO 8601格式时间
    model_type='unet',                         # 'unet' 或 'fcn'
    original_path='uploads/img_abc123def456.png',
    result_path='results/img_abc123def456_seg.png'
)

if record_id:
    print(f"✓ 插入成功，记录ID: {record_id}")
else:
    print("✗ 插入失败")
```

**返回值说明**：
- 成功：返回新插入记录的ID（正整数）
- 失败：返回None
  - 原因1：文件名重复（UNIQUE约束冲突）
  - 原因2：数据库连接失败
  - 原因3：参数类型错误

---

### 3️⃣ 查询所有记录

```python
from dao import ImageDAO

records = ImageDAO.get_all_records('./octa.db')

print(f"找到 {len(records)} 条记录")
for record in records:
    print(f"  ID: {record['id']}")
    print(f"  文件: {record['filename']}")
    print(f"  时间: {record['upload_time']}")
    print(f"  模型: {record['model_type']}")
    print("  ---")
```

**返回值说明**：
- 成功：返回列表，每个元素是一个字典（包含id、filename等字段）
- 无记录：返回空列表[]
- 失败：返回空列表[]

**排序**：按upload_time倒序（最新的在前）

---

### 4️⃣ 按ID查询单条记录

```python
from dao import ImageDAO

record = ImageDAO.get_record_by_id(1, './octa.db')

if record:
    print(f"✓ 找到记录: {record['filename']}")
    print(f"  上传时间: {record['upload_time']}")
    print(f"  模型: {record['model_type']}")
    print(f"  原始路径: {record['original_path']}")
    print(f"  结果路径: {record['result_path']}")
else:
    print("✗ 未找到该记录")
```

**返回值说明**：
- 成功：返回字典（包含完整记录信息）
- 未找到：返回None
- 失败：返回None

**性能**：最快的查询方式（使用主键索引）

---

### 5️⃣ 删除记录

```python
from dao import ImageDAO

success = ImageDAO.delete_record_by_id(1, './octa.db')

if success:
    print("✓ 记录删除成功")
else:
    print("✗ 删除失败（可能是ID不存在）")
```

**返回值说明**：
- 成功：返回True
- 失败（ID不存在）：返回False
- 异常：返回False

**⚠️ 警告**：
- 删除操作不可逆，请谨慎使用
- 删除后应手动删除关联的文件（uploads/results目录中的文件）
- 建议先调用`get_record_by_id()`确认记录存在

---

## 🔄 与Controller的集成方式

### 原来的做法（已废弃）

在ImageController中直接写SQL代码：

```python
# 旧方式（在Controller中直接操作数据库）
class ImageController:
    @classmethod
    async def segment_octa(cls, file: UploadFile, model_type: str):
        # ... 业务逻辑 ...
        
        # 直接写SQL（不推荐）
        conn = sqlite3.connect('./octa.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO images ...")
        conn.commit()
        conn.close()
        # 问题：代码混乱，难以维护
```

### 新的做法（推荐）

在Controller中调用ImageDAO：

```python
from dao import ImageDAO

class ImageController:
    @classmethod
    async def segment_octa(cls, file: UploadFile, model_type: str):
        # ... 业务逻辑 ...
        
        # 使用DAO操作数据库（推荐）
        record_id = ImageDAO.insert_record(
            filename=filename,
            upload_time=datetime.now().isoformat(),
            model_type=model_type,
            original_path=str(upload_path),
            result_path=str(result_path)
        )
        
        if record_id:
            # 插入成功
            pass
        else:
            # 处理失败
            pass
```

**优势**：
- ✅ 业务逻辑清晰（Controller只关心业务）
- ✅ 数据库操作独立（DAO只关心数据库）
- ✅ 易于测试（DAO可单独测试）
- ✅ 易于维护（修改数据库不影响业务逻辑）

---

## 🚀 后续集成步骤

### 第1步：在ImageController中导入ImageDAO

```python
# 在 octa_backend/controller/image_controller.py 的顶部添加
from dao import ImageDAO
```

### 第2步：修改ImageController的数据库方法

将以下旧方法替换为对ImageDAO的调用：

**旧方法（直接数据库操作）**：
- `_insert_record()` → 改为调用 `ImageDAO.insert_record()`
- `_get_all_records()` → 改为调用 `ImageDAO.get_all_records()`
- `_get_record_by_id()` → 改为调用 `ImageDAO.get_record_by_id()`
- （新增）`_delete_record()` → 改为调用 `ImageDAO.delete_record_by_id()`

### 第3步：在main.py中初始化DAO

```python
# 在 octa_backend/main.py 的启动部分添加
print("初始化数据库层...")
ImageDAO.init_db('./octa.db')
print("✓ DAO初始化成功")
```

---

## 📋 异常处理说明

### 异常分类

| 异常类型 | 场景 | 处理方式 |
|---------|------|--------|
| `sqlite3.IntegrityError` | UNIQUE约束冲突（文件名重复） | 返回None，打印WARNING |
| `sqlite3.OperationalError` | 表不存在、磁盘满等 | 返回None/False，打印ERROR |
| `sqlite3.DatabaseError` | 数据库被锁定 | 返回None/False，打印ERROR |
| `sqlite3.Error` | 其他SQLite错误 | 返回None/False，打印ERROR |
| `Exception` | Python异常（参数错误等） | 返回None/False，打印ERROR+堆栈 |

### 日志级别

- `[INFO]` - 信息类日志（正常操作）
- `[SUCCESS]` - 成功类日志（操作完成）
- `[WARNING]` - 警告类日志（预期的异常情况）
- `[ERROR]` - 错误类日志（异常发生）

---

## 🧪 单元测试

### 运行DAO单元测试

```bash
# 在octa_backend目录下
python -m dao.image_dao
```

**测试覆盖**：
- ✅ init_db() - 初始化
- ✅ insert_record() - 插入
- ✅ get_all_records() - 查询所有
- ✅ get_record_by_id() - 按ID查询
- ✅ delete_record_by_id() - 按ID删除

**测试结果** 如果所有测试通过，会输出：
```
============================================================
ImageDAO 单元测试
============================================================
[✓] 所有测试通过！
============================================================
```

---

## 📊 性能特性

### 查询性能

| 操作 | 时间复杂度 | 说明 |
|-----|---------|------|
| `get_record_by_id()` | O(log n) | 使用主键索引，最快 |
| `get_all_records()` | O(n) | 全表扫描，需排序 |
| `insert_record()` | O(log n) | 有UNIQUE索引冲突检查 |
| `delete_record_by_id()` | O(log n) | 使用主键索引 |

### 数据库大小估算

假设每条记录平均占用500字节：
- 1000条记录 ≈ 500KB
- 10000条记录 ≈ 5MB
- 100000条记录 ≈ 50MB

对于中小型应用，SQLite绰绰有余。

---

## 🔐 安全特性

### SQL注入防护

所有查询都使用**参数化查询**（占位符）：

```python
# ✓ 安全的做法
sql = "SELECT * FROM images WHERE id = ?"
cursor.execute(sql, (record_id,))

# ✗ 危险的做法（不要这样做）
sql = f"SELECT * FROM images WHERE id = {record_id}"
cursor.execute(sql)
```

### 数据完整性保护

- ✅ UNIQUE约束防止重复文件名
- ✅ NOT NULL约束保证必填字段
- ✅ 主键约束保证ID唯一性
- ✅ 事务管理（commit/rollback）

---

## 🔄 版本兼容性

### 与原有ImageController的兼容性

当前ImageController中的数据库操作：
- ✅ `_insert_record()` → ImageDAO.insert_record() 完全兼容
- ✅ `_get_all_records()` → ImageDAO.get_all_records() 完全兼容
- ✅ `_get_record_by_id()` → ImageDAO.get_record_by_id() 完全兼容
- ✅ `_delete_record_by_id()` → ImageDAO.delete_record_by_id() 新增功能

### 前端兼容性

DAO是纯数据层操作，前端无需任何修改。

API接口保持完全不变：
- POST /segment-octa/
- GET /history/
- GET /history/{id}
- DELETE /history/{id}

---

## 🛠️ 常见问题

### Q1: 为什么要分离DAO层？

**A**: 遵循**单一职责原则**：
- Controller只负责业务逻辑
- DAO只负责数据库操作
- 这样修改数据库结构时，业务逻辑无需改动

### Q2: 为什么要用静态方法而不是实例方法？

**A**: 因为：
- DAO无需保持状态（无实例变量）
- 每次操作都创建新连接（连接池在大应用中单独实现）
- 静态方法更简洁，调用时无需实例化

### Q3: 数据库参数为什么可配置？

**A**: 支持：
- 不同环境使用不同数据库（开发/测试/生产）
- 单元测试用独立的测试数据库
- 多应用共享一个DAO类但使用不同数据库

### Q4: 如何扩展数据库表？

**A**: 修改步骤：
1. 更新`CREATE_TABLE_SQL`（添加新字段）
2. 所有方法的docstring中更新字段说明
3. 创建数据库迁移脚本（ALTER TABLE）
4. 运行单元测试验证

---

## 📚 扩展建议

### 短期扩展（1-2周）

```python
# 添加字段
class ImageDAO:
    @staticmethod
    def add_index(index_name: str, column: str, db_path: str = './octa.db'):
        """创建索引以加速查询"""
        pass
    
    @staticmethod
    def get_records_by_model(model_type: str, db_path: str = './octa.db'):
        """按模型类型查询"""
        pass
    
    @staticmethod
    def get_records_by_date_range(start_date: str, end_date: str, db_path: str = './octa.db'):
        """按时间范围查询"""
        pass
```

### 中期扩展（1个月）

- 添加数据备份功能
- 实现事务处理
- 添加数据库连接池
- 支持批量操作

### 长期扩展（3-6个月）

- 迁移到关系型数据库（PostgreSQL/MySQL）
- 实现缓存层（Redis）
- 添加完整的ORM框架（SQLAlchemy）

---

## 🎯 总结

**ImageDAO是OCTA平台的数据访问层**，负责：
- ✅ 所有SQLite操作（初始化、插入、查询、删除）
- ✅ 异常处理和日志记录
- ✅ 连接管理和资源释放
- ✅ 参数化查询防止SQL注入

**核心优势**：
- 🔹 **隔离性强** - 数据库逻辑完全独立
- 🔹 **易于测试** - DAO可单独单元测试
- 🔹 **易于维护** - 修改数据库不影响业务
- 🔹 **易于扩展** - 添加新功能只需扩展DAO

**立即使用**：
```python
from dao import ImageDAO

# 初始化
ImageDAO.init_db()

# CRUD操作
id = ImageDAO.insert_record(...)
records = ImageDAO.get_all_records()
record = ImageDAO.get_record_by_id(id)
ImageDAO.delete_record_by_id(id)
```

---

**版本**：1.0  
**更新日期**：2026年1月14日  
**作者**：OCTA Web项目组

