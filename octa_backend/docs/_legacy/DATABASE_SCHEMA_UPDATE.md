# 数据库架构更新 - model_type 字段集成指南

## ✅ 更新完成

已成功为 `file_management` 表添加 `model_type` 字段，用于区分不同模型的权重文件。

---

## 📋 更新内容

### 1️⃣ **数据库架构变更**

**新增字段：**
- `model_type` (TEXT): 模型类型标识，可为空
  - 值范围：`'unet'`、`'fcn'`、`'rs_unet3_plus'`
  - 用途：区分权重文件所属的模型类型
  - 向后兼容：老记录为 NULL，不影响现有功能

**表结构（file_management）：**
```sql
CREATE TABLE file_management (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_type TEXT NOT NULL,              -- 'image'、'dataset' 或 'weight'
    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    related_model TEXT,
    file_size REAL,
    model_type TEXT                        -- ✅ 新增字段
);
```

---

### 2️⃣ **DAO 函数更新**

#### **FileDAO.add_file_record()**

**函数签名：**
```python
@staticmethod
def add_file_record(
    file_name: str,
    file_path: str,
    file_type: str,
    related_model: Optional[str] = None,
    file_size: Optional[float] = None,
    model_type: Optional[str] = None        # ✅ 新增参数
) -> Optional[int]
```

**参数说明：**
- `model_type` (可选): 模型类型，仅权重文件（`file_type='weight'`）需要指定
  - 必须为 `'unet'`、`'fcn'` 或 `'rs_unet3_plus'`
  - 如果是权重文件但未指定，会返回错误

**示例：**
```python
# 添加 U-Net 权重文件
file_id = FileDAO.add_file_record(
    file_name='unet_best.pth',
    file_path='models/weights/unet_best.pth',
    file_type='weight',
    file_size=234.5,
    model_type='unet'                     # ✅ 指定模型类型
)

# 添加 RS-Unet3+ 权重文件
file_id = FileDAO.add_file_record(
    file_name='rs_unet3_plus_epoch_50.pth',
    file_path='models/weights/train_20260120/rs_unet3_plus_epoch_50.pth',
    file_type='weight',
    file_size=189.3,
    model_type='rs_unet3_plus'            # ✅ RS-Unet3+ 专用
)

# 添加图片文件（无需 model_type）
file_id = FileDAO.add_file_record(
    file_name='octa_001.png',
    file_path='uploads/images/octa_001.png',
    file_type='image',
    file_size=2.5
    # model_type 留空即可
)
```

---

#### **FileDAO.get_file_list()**

**函数签名：**
```python
@staticmethod
def get_file_list(
    file_type: Optional[str] = None,
    model_type: Optional[str] = None        # ✅ 新增参数
) -> List[Dict]
```

**参数说明：**
- `file_type` (可选): 文件类型筛选
  - `None`: 查询所有文件
  - `'image'`: 只查询图片文件
  - `'dataset'`: 只查询数据集文件
  - `'weight'`: 只查询权重文件
- `model_type` (可选): 模型类型筛选（仅当 `file_type='weight'` 时有效）
  - `None`: 查询所有权重文件
  - `'unet'`: 只查询 U-Net 权重
  - `'fcn'`: 只查询 FCN 权重
  - `'rs_unet3_plus'`: 只查询 RS-Unet3+ 权重

**示例：**
```python
# 查询所有权重文件
all_weights = FileDAO.get_file_list(file_type='weight')
print(f"共有 {len(all_weights)} 个权重文件")

# 查询 U-Net 权重文件
unet_weights = FileDAO.get_file_list(file_type='weight', model_type='unet')
for weight in unet_weights:
    print(f"U-Net 权重: {weight['file_name']} ({weight['file_size']} MB)")

# 查询 RS-Unet3+ 权重文件
rs_weights = FileDAO.get_file_list(file_type='weight', model_type='rs_unet3_plus')
for weight in rs_weights:
    print(f"RS-Unet3+ 权重: {weight['file_name']} ({weight['file_size']} MB)")

# 查询所有图片文件（model_type 忽略）
images = FileDAO.get_file_list(file_type='image')
```

---

### 3️⃣ **SQL 注入防护**

所有查询使用参数化查询（Prepared Statements），防止 SQL 注入攻击：

**✅ 安全的查询方式：**
```python
# 参数化查询（推荐）
cursor.execute(
    "SELECT * FROM file_management WHERE file_type = ? AND model_type = ?",
    [file_type, model_type]  # 参数通过列表传递
)
```

**❌ 不安全的查询方式：**
```python
# 字符串拼接（危险！）
cursor.execute(
    f"SELECT * FROM file_management WHERE file_type = '{file_type}'"
)
```

---

## 🔧 数据库迁移

### **自动迁移脚本**

已提供迁移脚本 `migrate_add_model_type.py`，自动检测并添加 `model_type` 字段。

**执行方式：**
```bash
cd octa_backend
python migrate_add_model_type.py
```

**脚本功能：**
1. ✅ 检查数据库文件是否存在
2. ✅ 检查 `file_management` 表是否存在
3. ✅ 检查 `model_type` 字段是否已存在
4. ✅ 如果不存在，执行 `ALTER TABLE` 添加字段
5. ✅ 验证字段添加成功
6. ✅ 显示完整表结构

**输出示例：**
```
======================================================================
数据库架构迁移：添加 model_type 字段
======================================================================
[INFO] 数据库路径: D:\Code\OCTA_Web\octa_backend\octa.db
[INFO] file_management 表已存在
[INFO] model_type 字段不存在，开始添加...
[SUCCESS] model_type 字段添加成功
[SUCCESS] 验证通过：model_type 字段已存在于数据库中

[INFO] 当前表结构（file_management）：
----------------------------------------------------------------------
列名                   类型         非空    默认值             主键
----------------------------------------------------------------------
id                   INTEGER    否     NULL            是
file_name            TEXT       是     NULL            否
file_path            TEXT       TEXT       是     NULL            否
file_type            TEXT       是     NULL            否
upload_time          TIMESTAMP  否     CURRENT_TIMESTAMP 否
related_model        TEXT       否     NULL            否
file_size            REAL       否     NULL            否
model_type           TEXT       否     NULL            否  ✅ 新增
----------------------------------------------------------------------

✅ 数据库迁移成功！
```

---

## 🎯 使用场景

### **场景1：训练完成后保存权重**

```python
# 训练完成，保存 RS-Unet3+ 权重到数据库
from dao.file_dao import FileDAO

weight_path = 'models/weights/train_20260120_153045/best_model.pth'
file_id = FileDAO.add_file_record(
    file_name='rs_unet3_plus_best.pth',
    file_path=weight_path,
    file_type='weight',
    file_size=189.3,
    model_type='rs_unet3_plus'  # ✅ 标记为 RS-Unet3+ 权重
)

if file_id:
    print(f"✓ 权重文件已保存到数据库（ID={file_id}）")
```

### **场景2：前端加载权重选择器**

```python
# 后端 API：获取特定模型的权重列表
from dao.file_dao import FileDAO
from fastapi import APIRouter

router = APIRouter()

@router.get("/api/weights/list")
async def get_weights(model_type: str = None):
    """
    获取权重文件列表
    
    参数：
        - model_type: 模型类型（unet/fcn/rs_unet3_plus），可选
    """
    if model_type:
        # 按模型类型筛选
        weights = FileDAO.get_file_list(file_type='weight', model_type=model_type)
    else:
        # 获取所有权重
        weights = FileDAO.get_file_list(file_type='weight')
    
    return {
        "status": "success",
        "model_type": model_type,
        "weights": weights
    }
```

### **场景3：HomeView.vue 动态加载权重**

```vue
<script setup>
import axios from 'axios'
import { ref, watch } from 'vue'

const selectedModel = ref('unet')
const availableWeights = ref([])

// 监听模型选择变化，自动加载对应权重列表
watch(selectedModel, async (newModel) => {
  try {
    // 调用后端 API，按模型类型筛选权重
    const response = await axios.get(
      `http://127.0.0.1:8000/api/weights/list?model_type=${newModel}`
    )
    
    if (response.data.status === 'success') {
      availableWeights.value = response.data.weights
      console.log(`✓ 加载 ${newModel} 权重 ${availableWeights.value.length} 个`)
    }
  } catch (error) {
    console.error('加载权重列表失败:', error)
  }
})
</script>
```

---

## ⚠️ 注意事项

### 1. **权重文件必须指定 model_type**

```python
# ❌ 错误：权重文件未指定 model_type
file_id = FileDAO.add_file_record(
    file_name='model.pth',
    file_path='models/weights/model.pth',
    file_type='weight'
    # 缺少 model_type 参数
)
# 输出：[ERROR] 权重文件必须指定model_type（'unet'、'fcn'或'rs_unet3_plus'）

# ✅ 正确：指定 model_type
file_id = FileDAO.add_file_record(
    file_name='model.pth',
    file_path='models/weights/model.pth',
    file_type='weight',
    model_type='unet'  # ✅ 必须指定
)
```

### 2. **model_type 值必须合法**

```python
# ❌ 错误：非法的 model_type 值
file_id = FileDAO.add_file_record(
    file_name='model.pth',
    file_path='models/weights/model.pth',
    file_type='weight',
    model_type='resnet50'  # ❌ 不支持
)
# 输出：[ERROR] model_type必须为'unet'、'fcn'或'rs_unet3_plus'，当前值: resnet50

# ✅ 正确：使用合法值
supported_models = ['unet', 'fcn', 'rs_unet3_plus']
```

### 3. **向后兼容性**

- ✅ 老记录的 `model_type` 为 NULL，不影响查询
- ✅ 查询时不指定 `model_type`，返回所有记录
- ✅ 非权重文件（image/dataset）无需指定 `model_type`

---

## 📚 相关文件

- **DAO 实现**: [dao/file_dao.py](d:\Code\OCTA_Web\octa_backend\dao\file_dao.py)
- **迁移脚本**: [migrate_add_model_type.py](d:\Code\OCTA_Web\octa_backend\migrate_add_model_type.py)
- **配置文件**: [config/config.py](d:\Code\OCTA_Web\octa_backend\config\config.py)

---

**文档版本**: 1.0.0  
**最后更新**: 2026-01-20  
**状态**: ✅ 生产就绪
