# SQLite 数据库查询参考

本文档提供了直接操作OCTA数据库的SQL查询示例，适用于毕设答辩演示。

## 打开数据库

### Windows
```bash
# 使用 sqlite3 命令行工具
sqlite3 octa_backend\octa.db

# 或使用 SQLite 图形界面工具（如 DB Browser for SQLite）
```

### Linux/Mac
```bash
sqlite3 octa_backend/octa.db
```

---

## 常用查询命令

### 1. 查看所有表
```sql
.tables
```

**输出示例**:
```
images
```

---

### 2. 查看 images 表结构
```sql
.schema images
```

**输出示例**:
```sql
CREATE TABLE images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT UNIQUE NOT NULL,
    upload_time TEXT NOT NULL,
    model_type TEXT NOT NULL,
    original_path TEXT NOT NULL,
    result_path TEXT NOT NULL
);
```

---

### 3. 查询所有记录
```sql
SELECT * FROM images;
```

**输出示例**:
```
1|a1b2c3d4-e5f6-7890.png|2026-01-12 14:30:25|unet|./uploads/a1b2c3d4-e5f6-7890.png|./results/a1b2c3d4-e5f6-7890_segmented.png
2|b2c3d4e5-f6a7-8901.png|2026-01-12 13:45:10|fcn|./uploads/b2c3d4e5-f6a7-8901.png|./results/b2c3d4e5-f6a7-8901_segmented.png
```

---

### 4. 统计记录数
```sql
SELECT COUNT(*) as 总记录数 FROM images;
```

**输出示例**:
```
总记录数
--------
42
```

---

### 5. 按时间倒序查看最新的10条记录
```sql
SELECT id, filename, upload_time, model_type 
FROM images 
ORDER BY upload_time DESC 
LIMIT 10;
```

**输出示例**:
```
1|a1b2c3d4-e5f6-7890.png|2026-01-12 14:30:25|unet
2|b2c3d4e5-f6a7-8901.png|2026-01-12 13:45:10|fcn
3|c3d4e5f6-a7b8-9012.png|2026-01-12 12:20:30|unet
```

---

### 6. 按模型类型统计
```sql
SELECT model_type, COUNT(*) as 数量 
FROM images 
GROUP BY model_type;
```

**输出示例**:
```
model_type|数量
-----------+----
unet|28
fcn|14
```

---

### 7. 查看特定日期的分割记录
```sql
SELECT id, filename, upload_time, model_type 
FROM images 
WHERE DATE(upload_time) = '2026-01-12'
ORDER BY upload_time DESC;
```

---

### 8. 查询特定ID的详细信息
```sql
SELECT * FROM images WHERE id = 1;
```

**输出示例**:
```
1|a1b2c3d4-e5f6-7890.png|2026-01-12 14:30:25|unet|./uploads/a1b2c3d4-e5f6-7890.png|./results/a1b2c3d4-e5f6-7890_segmented.png
```

---

### 9. 统计不同小时的分割数量
```sql
SELECT SUBSTR(upload_time, 1, 13) as 小时, COUNT(*) as 分割数
FROM images 
GROUP BY SUBSTR(upload_time, 1, 13)
ORDER BY 小时 DESC;
```

**输出示例**:
```
小时|分割数
-------+-----
2026-01-12 14|8
2026-01-12 13|5
2026-01-12 12|3
```

---

### 10. 导出全部数据为CSV
```sql
.mode csv
.output records.csv
SELECT * FROM images;
.output stdout
```

---

### 11. 格式化显示所有记录
```sql
.mode column
.headers on
.width 36 15 19 5 40 50
SELECT 
    id,
    filename,
    upload_time,
    model_type,
    original_path,
    result_path
FROM images
ORDER BY upload_time DESC;
```

**输出示例** (格式化表格):
```
id  filename                         upload_time      model original_path         result_path
--  -------------------------------- -----------      ----- ------------------- -------
1   a1b2c3d4-e5f6-7890.png          2026-01-12 14:30 unet  ./uploads/a1b2c...  ./results/a1b2c...
2   b2c3d4e5-f6a7-8901.png          2026-01-12 13:45 fcn   ./uploads/b2c3d...  ./results/b2c3d...
```

---

## 数据库维护命令

### 清空所有记录（仅保留表结构）
```sql
DELETE FROM images;
```

⚠️ **警告**: 此操作不可恢复！

---

### 删除特定ID的记录
```sql
DELETE FROM images WHERE id = 1;
```

---

### 重置自增ID计数器
在删除记录后，如果想重置ID计数器：

```sql
DELETE FROM sqlite_sequence WHERE name='images';
```

然后插入新数据时ID会从1开始。

---

### 检查数据库完整性
```sql
PRAGMA integrity_check;
```

**输出示例**:
```
ok
```

---

### 查看数据库文件信息
```sql
PRAGMA database_list;
```

---

### 优化数据库（释放空间）
```sql
VACUUM;
```

---

## 毕设答辩演示脚本

### 脚本 1: 完整的展示流程

```bash
#!/bin/bash
# 打开数据库
sqlite3 octa_backend/octa.db << EOF

echo "==== OCTA 数据库演示 ===="
echo ""

echo "1. 表结构信息"
.schema images
echo ""

echo "2. 当前记录总数"
SELECT COUNT(*) as 总数 FROM images;
echo ""

echo "3. 按模型统计"
SELECT model_type, COUNT(*) as 数量 FROM images GROUP BY model_type;
echo ""

echo "4. 最新10条记录"
.mode column
.headers on
SELECT id, filename, upload_time, model_type FROM images ORDER BY upload_time DESC LIMIT 10;
echo ""

echo "5. 时间分布"
SELECT SUBSTR(upload_time, 1, 13) as 小时, COUNT(*) as 数量 
FROM images 
GROUP BY SUBSTR(upload_time, 1, 13)
ORDER BY 小时 DESC
LIMIT 5;

EOF
```

---

### 脚本 2: 快速统计

```sql
-- 保存为 stats.sql
-- 运行: sqlite3 octa_backend/octa.db < stats.sql

.mode column
.headers on
.width 20 10

SELECT 
  '总记录数' as 统计项,
  COUNT(*) as 数值
FROM images
UNION ALL
SELECT 
  'U-Net 模型',
  COUNT(*) 
FROM images 
WHERE model_type = 'unet'
UNION ALL
SELECT 
  'FCN 模型',
  COUNT(*) 
FROM images 
WHERE model_type = 'fcn'
UNION ALL
SELECT 
  '今日分割数',
  COUNT(*) 
FROM images 
WHERE DATE(upload_time) = DATE('now');
```

---

## Python 脚本直接查询数据库

如果想在Python中直接查询数据库（不调用API）：

```python
import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path("octa_backend/octa.db")

# 连接数据库
conn = sqlite3.connect(str(DB_PATH))
conn.row_factory = sqlite3.Row  # 返回字典形式
cursor = conn.cursor()

# 查询所有记录
cursor.execute("SELECT * FROM images ORDER BY upload_time DESC")
records = cursor.fetchall()

# 显示结果
print(f"总记录数: {len(records)}")
for record in records[:5]:
    print(f"ID: {record['id']}, 时间: {record['upload_time']}, 模型: {record['model_type']}")

# 统计
cursor.execute("SELECT model_type, COUNT(*) FROM images GROUP BY model_type")
for model, count in cursor.fetchall():
    print(f"{model}: {count} 条")

conn.close()
```

---

## 答辩时的关键数据点

### 演示要点 1: 数据库自动初始化

**显示**:
```bash
sqlite3 octa_backend/octa.db ".schema images"
```

**说明**: 应用启动时自动创建 `octa.db` 和 `images` 表，6个字段完整对应需求。

---

### 演示要点 2: 数据持久化

**显示**:
```sql
SELECT COUNT(*) FROM images;
SELECT model_type, COUNT(*) FROM images GROUP BY model_type;
```

**说明**: 每次分割都会自动记录到数据库，包括文件名、时间、模型类型、路径等关键信息。

---

### 演示要点 3: API接口测试

**URL**:
```
GET http://127.0.0.1:8000/history/
GET http://127.0.0.1:8000/history/1
```

**说明**: 两个RESTful接口分别查询所有记录和单条记录，返回JSON格式，符合Web API规范。

---

### 演示要点 4: 异常处理

**显示数据库操作中的异常处理代码**:
```python
except sqlite3.IntegrityError:
    # 处理UNIQUE约束冲突
    return None
except sqlite3.OperationalError:
    # 处理数据库操作错误
    return None
except Exception as e:
    # 通用异常处理
    return None
finally:
    # 确保连接关闭
    conn.close()
```

**说明**: 完善的异常处理避免连接泄露，适应高并发异步环境。

---

## 常见问题回答

### Q: 数据库支持并发访问吗？
A: 是的，使用了 `check_same_thread=False` 支持FastAPI异步环境的并发访问。

### Q: 如何备份数据库？
A: 
```bash
# 简单备份（复制文件）
cp octa_backend/octa.db octa_backend/octa_backup.db

# 导出SQL
sqlite3 octa_backend/octa.db ".dump" > backup.sql
```

### Q: 如何恢复备份？
A:
```bash
# 从备份恢复
cp octa_backend/octa_backup.db octa_backend/octa.db

# 或从SQL恢复
sqlite3 octa_backend/octa.db < backup.sql
```

### Q: 数据库文件大小会很大吗？
A: 不会。SQLite很轻量级，每条记录约500字节，10000条记录仅需5MB。

### Q: 可以改为MySQL或PostgreSQL吗？
A: 可以，但SQLite无需额外服务器，最适合毕设项目。如需改为其他数据库，只需修改连接字符串和API层。

---

**最后更新**: 2026年1月12日  
**适用版本**: OCTA Backend v1.0+

