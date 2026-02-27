# /file/model-weights API端点实现文档

## 📋 概述

为支持前端动态加载不同模型架构的权重文件，新增 `/file/model-weights` GET端点，实现按模型类型筛选权重列表的功能。

**创建时间**: 2026-01-20  
**文件位置**: `octa_backend/controller/file_controller.py`  
**端点路径**: `GET /file/model-weights`

---

## 🎯 功能说明

### 核心功能
- **按模型类型筛选权重**：根据 `model_type` 查询参数返回对应模型的权重文件列表
- **参数验证**：确保 `model_type` 在允许值范围内（`unet`、`fcn`、`rs_unet3_plus`）
- **容错处理**：未传递参数时返回空列表并提示用户选择模型

### 支持的模型类型
| 模型类型 | 参数值 | 权重目录 |
|---------|--------|---------|
| U-Net | `unet` | `models/weights_unet/` |
| FCN | `fcn` | `models/weights_fcn/` |
| RS-Unet3+ | `rs_unet3_plus` | `models/weights_rs_unet3_plus/` |

---

## 📡 API规范

### 请求格式

```http
GET /file/model-weights?model_type=unet HTTP/1.1
Host: 127.0.0.1:8000
```

### 查询参数

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `model_type` | `string` | 否 | 模型类型：`unet`、`fcn`、`rs_unet3_plus` |

### 响应格式

#### 成功响应（200）

**示例1：有权重文件**
```json
{
  "code": 200,
  "msg": "找到3个unet权重",
  "data": [
    {
      "id": 5,
      "file_name": "unet_epoch10_acc0.95.pth",
      "file_path": "models/weights_unet/unet_epoch10_acc0.95.pth",
      "file_size": 102400,
      "file_type": "weight",
      "model_type": "unet",
      "upload_time": "2026-01-20 14:30:00",
      "related_model": null
    },
    {
      "id": 8,
      "file_name": "unet_best.pth",
      "file_path": "models/weights_unet/unet_best.pth",
      "file_size": 98765,
      "file_type": "weight",
      "model_type": "unet",
      "upload_time": "2026-01-20 15:00:00",
      "related_model": null
    }
  ]
}
```

**示例2：无权重文件**
```json
{
  "code": 200,
  "msg": "找到0个fcn权重",
  "data": []
}
```

**示例3：未指定模型类型**
```json
{
  "code": 200,
  "msg": "请先选择模型类型",
  "data": []
}
```

#### 错误响应（400）

**无效的模型类型**
```json
{
  "detail": "无效的模型类型：invalid_model，允许值：unet, fcn, rs_unet3_plus"
}
```

#### 错误响应（500）

**数据库查询失败**
```json
{
  "detail": "查询失败：数据库连接错误"
}
```

---

## 💻 实现细节

### 代码位置

- **文件**: `octa_backend/controller/file_controller.py`
- **函数**: `get_model_weights()`
- **行数**: 约 145-250（新增约105行）

### 核心逻辑

```python
@file_router.get("/model-weights", summary="查询模型权重列表（按模型类型筛选）")
async def get_model_weights(
    model_type: Optional[str] = Query(None, description="模型类型：'unet'、'fcn'、'rs_unet3_plus'")
):
    """处理流程：
    步骤1：参数校验 - 验证model_type是否在允许值范围
    步骤2：处理未选择情况 - 未传参数返回空列表
    步骤3：查询权重列表 - 调用DAO层双重筛选
    步骤4：返回结果 - 格式化响应
    """
    
    # 参数验证
    valid_model_types = ['unet', 'fcn', 'rs_unet3_plus']
    if model_type is not None and model_type not in valid_model_types:
        raise HTTPException(status_code=400, detail=f"无效的模型类型：{model_type}")
    
    # 未选择模型
    if model_type is None:
        return success_response(data=[], msg="请先选择模型类型")
    
    # 查询数据库（双重筛选：file_type='weight' + model_type）
    weight_list = FileDAO.get_file_list(file_type='weight', model_type=model_type)
    
    # 返回结果
    return success_response(
        data=weight_list,
        msg=f"找到{len(weight_list)}个{model_type}权重"
    )
```

### DAO层依赖

调用 `FileDAO.get_file_list()` 方法，传递两个参数：

```python
FileDAO.get_file_list(
    file_type='weight',      # 固定为'weight'类型
    model_type=model_type    # 传递用户选择的模型类型
)
```

**SQL查询逻辑**（位于 `dao/file_dao.py`）：
```sql
SELECT * FROM file_management 
WHERE file_type = ? AND model_type = ?
ORDER BY upload_time DESC
```

---

## 🧪 测试验证

### 测试脚本

已创建测试脚本：`octa_backend/test_model_weights_endpoint.py`

**运行方式**：
```bash
cd octa_backend
python test_model_weights_endpoint.py
```

### 测试用例

| 测试用例 | 请求参数 | 预期结果 |
|---------|---------|---------|
| 1. 无参数请求 | 无 | 返回空列表，提示"请先选择模型类型" |
| 2. U-Net权重查询 | `model_type=unet` | 返回所有U-Net权重列表 |
| 3. RS-Unet3+权重查询 | `model_type=rs_unet3_plus` | 返回所有RS-Unet3+权重列表 |
| 4. FCN权重查询 | `model_type=fcn` | 返回所有FCN权重列表 |
| 5. 无效模型类型 | `model_type=invalid_model` | 返回400错误 |
| 6. 空字符串 | `model_type=` | 返回400错误 |

### 手动测试

**使用 curl**:
```bash
# 查询U-Net权重
curl "http://127.0.0.1:8000/file/model-weights?model_type=unet"

# 查询RS-Unet3+权重
curl "http://127.0.0.1:8000/file/model-weights?model_type=rs_unet3_plus"

# 测试无参数情况
curl "http://127.0.0.1:8000/file/model-weights"
```

**使用浏览器**:
- 打开 `http://127.0.0.1:8000/docs`（Swagger UI）
- 找到 `/file/model-weights` 端点
- 点击 "Try it out" 测试不同参数组合

---

## 🌐 前端集成

### Vue3 集成示例

**HomeView.vue 动态加载权重**：

```vue
<script setup>
import { ref, watch } from 'vue'
import axios from 'axios'

// 状态定义
const selectedModel = ref('unet')  // 当前选择的模型
const availableWeights = ref([])   // 可用权重列表
const selectedWeight = ref('')     // 选中的权重

// 监听模型选择变化，自动加载对应权重
watch(selectedModel, async (newModel) => {
  try {
    const response = await axios.get(
      `http://127.0.0.1:8000/file/model-weights?model_type=${newModel}`
    )
    
    if (response.data.code === 200) {
      availableWeights.value = response.data.data
      
      // 如果有权重，默认选择第一个
      if (availableWeights.value.length > 0) {
        selectedWeight.value = availableWeights.value[0].file_path
      } else {
        selectedWeight.value = ''
        ElMessage.warning(`暂无${newModel}模型的权重文件`)
      }
    }
  } catch (error) {
    console.error('加载权重失败:', error)
    ElMessage.error('加载权重文件失败')
  }
})
</script>

<template>
  <!-- 模型选择器 -->
  <el-select v-model="selectedModel" placeholder="选择模型">
    <el-option label="U-Net" value="unet" />
    <el-option label="FCN" value="fcn" />
    <el-option label="RS-Unet3+" value="rs_unet3_plus" />
  </el-select>
  
  <!-- 权重选择器 -->
  <el-select 
    v-model="selectedWeight" 
    placeholder="选择权重文件"
    :disabled="availableWeights.length === 0"
  >
    <el-option 
      v-for="weight in availableWeights" 
      :key="weight.id"
      :label="weight.file_name"
      :value="weight.file_path"
    />
  </el-select>
</template>
```

### Axios 请求封装

```javascript
// api/weights.js
import axios from 'axios'

const BASE_URL = 'http://127.0.0.1:8000'

/**
 * 获取指定模型的权重列表
 * @param {string} modelType - 模型类型：'unet'、'fcn'、'rs_unet3_plus'
 * @returns {Promise<Array>} 权重文件列表
 */
export async function getModelWeights(modelType) {
  if (!modelType) {
    throw new Error('模型类型不能为空')
  }
  
  const response = await axios.get(
    `${BASE_URL}/file/model-weights`,
    { params: { model_type: modelType } }
  )
  
  if (response.data.code !== 200) {
    throw new Error(response.data.msg || '查询失败')
  }
  
  return response.data.data
}
```

---

## 🔗 依赖关系

### 数据库依赖

**表**: `file_management`  
**关键字段**:
- `file_type` TEXT - 文件类型（必须为'weight'）
- `model_type` TEXT - 模型类型（'unet'、'fcn'、'rs_unet3_plus'）

**数据示例**:
```sql
INSERT INTO file_management (
  file_name, file_path, file_type, model_type, upload_time
) VALUES (
  'unet_epoch10.pth', 
  'models/weights_unet/unet_epoch10.pth',
  'weight',
  'unet',
  '2026-01-20 14:30:00'
);
```

### DAO层依赖

**方法**: `FileDAO.get_file_list(file_type, model_type)`  
**文件**: `octa_backend/dao/file_dao.py`  
**功能**: 支持双参数筛选，使用参数化查询防止SQL注入

### 配置依赖

**文件**: `octa_backend/config/config.py`  
**常量**:
- `UNET_WEIGHT_DIR` - U-Net权重目录
- `FCN_WEIGHT_DIR` - FCN权重目录
- `RS_UNET3_PLUS_WEIGHT_DIR` - RS-Unet3+权重目录

---

## 📊 使用流程图

```
用户选择模型
    ↓
前端发起GET请求
    ↓
/file/model-weights端点接收
    ↓
参数验证（model_type）
    ↓
调用FileDAO.get_file_list()
    ↓
数据库查询（双重筛选）
    ↓
返回权重列表
    ↓
前端更新权重选择器
```

---

## ✅ 完成状态

### 已完成
- ✅ API端点实现（`/file/model-weights`）
- ✅ 参数验证逻辑（允许值：unet、fcn、rs_unet3_plus）
- ✅ DAO层集成（调用get_file_list双重筛选）
- ✅ 错误处理（400/500状态码）
- ✅ 详细注释（4步骤处理流程）
- ✅ 测试脚本（6个测试用例）
- ✅ 文档编写（API规范、集成示例）

### 验证通过
- ✅ Python语法验证（`python -m py_compile`）
- ✅ 代码结构完整（导入、函数定义、路由注册）
- ✅ 文档完备（docstring、注释、使用示例）

### 待测试
- ⏳ 端到端功能测试（需启动后端服务）
- ⏳ 前端集成测试（需前端调用）
- ⏳ 数据库数据验证（需上传权重文件）

---

## 🚀 后续步骤

### 1. 启动后端测试端点
```bash
cd octa_backend
python main.py
```

### 2. 运行测试脚本
```bash
python test_model_weights_endpoint.py
```

### 3. 前端集成
- 更新 `HomeView.vue` 添加权重选择器
- 实现模型切换时自动加载权重
- 添加权重选择器禁用/启用逻辑

### 4. 数据准备
- 上传不同模型的权重文件到文件管理系统
- 确保每个权重文件都正确设置了 `model_type` 字段
- 验证权重文件存储在正确的目录

---

## 📝 相关文档

- [数据库Schema更新文档](./DATABASE_SCHEMA_UPDATE.md) - model_type字段设计
- [权重隔离配置文档](./WEIGHT_ISOLATION_CONFIG.md) - 目录结构说明
- [文件管理DAO文档](./dao/file_dao.py) - get_file_list实现细节
- [前端集成文档](./RS_UNET3_PLUS_INTEGRATION.md) - RS-Unet3+完整集成方案

---

**文档版本**: v1.0  
**最后更新**: 2026-01-20  
**作者**: GitHub Copilot AI  
**状态**: ✅ 实现完成，待功能测试

