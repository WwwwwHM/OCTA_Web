# OCTA图像分割平台 - 完整开发指南

## 🚀 项目启动指南

### 前置条件

- ✅ Python 3.8+ （虚拟环境已配置在 `octa_env/`）
- ✅ Node.js 16+ （前端依赖已安装）
- ✅ 浏览器支持Vue 3和现代JavaScript

### 一键启动（推荐）

#### Windows用户

```bash
# 方式1：使用启动脚本
cd octa_backend
start_server.bat

# 方式2：手动启动（新开一个终端）
cd octa_backend
..\octa_env\Scripts\activate
python main.py
```

#### 前端启动（新开终端）

```bash
cd octa_frontend
npm run dev
```

### 验证启动成功

```bash
# 1. 后端检查
curl http://127.0.0.1:8000/
# 返回：{"message": "OCTA后端服务运行正常"}

# 2. 前端检查
# 打开浏览器访问 http://127.0.0.1:5173
# 应该看到OCTA图像分割平台UI
```

---

## 📁 项目结构

```
OCTA_Web/
├── octa_backend/                    # 后端目录
│   ├── controller/                  # ✨ 新增：控制层（分层架构）
│   │   ├── __init__.py
│   │   └── image_controller.py      # ImageController类（1420行）
│   ├── models/                      # 模型层
│   │   ├── __init__.py
│   │   └── unet.py                  # U-Net/FCN模型实现
│   ├── main.py                      # ✨ 精简：仅130行（路由层）
│   ├── requirements.txt             # Python依赖
│   ├── octa.db                      # SQLite数据库（启动时自动创建）
│   ├── uploads/                     # 上传的原始图像目录
│   ├── results/                     # 分割结果目录
│   └── ...其他文件
│
├── octa_frontend/                   # 前端目录
│   ├── src/
│   │   ├── views/
│   │   │   ├── HomeView.vue        # 主页（图像上传和对比展示）
│   │   │   ├── HistoryView.vue     # 历史记录页面
│   │   │   └── AboutView.vue       # 关于页面
│   │   ├── App.vue                 # 根组件
│   │   ├── main.js                 # 应用入口
│   │   └── router/                 # 路由配置
│   ├── index.html                  # HTML模板
│   ├── package.json                # Node.js依赖
│   └── vite.config.js              # Vite配置
│
├── octa_env/                        # Python虚拟环境（不修改）
│
├── MODIFICATION_SUMMARY.md          # 代码修改总结
├── CONTROLLER_REFACTOR_SUMMARY.md   # ✨ 控制层重构总结
├── IMAGECONTROLLER_API_REFERENCE.md # ✨ API参考手册
└── README.md                        # 项目说明

```

---

## 🏗️ 分层架构详解

### 架构图

```
┌─────────────────────────────────────────────┐
│         前端（Vue 3 + Element Plus）        │
│   http://127.0.0.1:5173                    │
└────────────────────┬────────────────────────┘
                     │ HTTP请求/响应
┌────────────────────▼────────────────────────┐
│        FastAPI路由层（main.py）             │
│   职责：请求转发 + CORS中间件               │
│   特点：130行代码，清晰的路由定义            │
├─────────────────────────────────────────────┤
│      控制层（ImageController）              │
│   职责：业务逻辑编排 + 数据验证 + 异常处理  │
│   特点：1420行，9个公开方法，5个私有方法    │
├─────────────────────────────────────────────┤
│     模型层（models/unet.py）                │
│   职责：图像预处理 + 模型推理 + 后处理      │
│   特点：630行，支持U-Net/FCN，CPU模式      │
├─────────────────────────────────────────────┤
│      数据层（SQLite + 文件系统）            │
│   职责：数据库操作 + 文件I/O                │
│   特点：images表记录分割历史                │
└─────────────────────────────────────────────┘
```

### 数据流向

```
1. 上传图像
   前端 → POST /segment-octa/ → main.py → ImageController.segment_octa()
   
2. 文件校验
   ImageController → _validate_image_file() → 检查PNG/JPG/JPEG
   
3. 文件保存
   ImageController → _generate_unique_filename() → UUID命名保存到uploads/
   
4. 模型推理
   ImageController → models.unet.segment_octa_image() → CPU模式推理
   
5. 结果保存
   模型 → 保存为PNG → results/uuid_segmented.png
   
6. 数据库记录
   ImageController → _insert_record() → 保存到SQLite
   
7. 返回响应
   ImageController → JSON响应 → 前端显示结果

```

---

## 🔌 API接口完整列表

### 基础接口

| HTTP方法 | 路由 | 控制器方法 | 功能 |
|---------|-----|----------|------|
| GET | `/` | `test_service()` | 后端健康检查 |

### 分割相关

| HTTP方法 | 路由 | 控制器方法 | 功能 |
|---------|-----|----------|------|
| POST | `/segment-octa/` | `segment_octa()` | 上传图像并分割 |
| GET | `/images/{filename}` | `get_uploaded_image()` | 获取原始图像 |
| GET | `/results/{filename}` | `get_result_image()` | 获取分割结果 |

### 历史记录相关

| HTTP方法 | 路由 | 控制器方法 | 功能 |
|---------|-----|----------|------|
| GET | `/history/` | `get_all_history()` | 查询所有历史 |
| GET | `/history/{id}` | `get_history_by_id()` | 查询单条历史 |
| DELETE | `/history/{id}` | `delete_history_by_id()` | 删除历史记录 |

### 详细说明

👉 详见 [IMAGECONTROLLER_API_REFERENCE.md](./IMAGECONTROLLER_API_REFERENCE.md)

---

## 💻 前端集成示例

### 上传图像并分割

```javascript
// HomeView.vue 中的示例
import axios from 'axios'

// 创建axios实例（后端地址）
const api = axios.create({
  baseURL: 'http://127.0.0.1:8000'
})

// 上传并分割
async function handleSegmentation() {
  const formData = new FormData()
  formData.append('file', imageFile.value)
  formData.append('model_type', selectedModel.value)
  
  try {
    const res = await api.post('/segment-octa/', formData)
    
    // 响应格式
    console.log({
      success: res.data.success,
      originalUrl: res.data.image_url,      // /images/uuid.png
      resultUrl: res.data.result_url,        // /results/uuid_segmented.png
      recordId: res.data.record_id           // 数据库记录ID
    })
    
    // 显示对比（原图 vs 分割结果）
    uploadedImageUrl.value = res.data.image_url
    resultImage.value = res.data.result_url
    
  } catch (error) {
    ElMessage.error(error.response?.data?.detail || '分割失败')
  }
}
```

### 查询分割历史

```javascript
// HistoryView.vue 中的示例

// 获取所有历史
async function loadHistory() {
  const res = await api.get('/history/')
  historyList.value = res.data  // 直接是数组
}

// 获取特定历史
async function loadHistoryDetail(recordId) {
  const res = await api.get(`/history/${recordId}`)
  const record = res.data
  
  // 使用record中的图像路径
  originalUrl.value = `/images/${record.filename}`
  resultUrl.value = record.result_path.replace('/results/', '/results/')
}

// 删除历史记录
async function deleteRecord(recordId) {
  await api.delete(`/history/${recordId}`)
  ElMessage.success('记录已删除')
  loadHistory()  // 刷新列表
}
```

---

## 🛠️ 常见开发任务

### 任务1：添加新的分割模型

**目标**：支持除U-Net/FCN外的新模型

**步骤**：

1. 在 `models/unet.py` 中实现新模型类

```python
class YourNewModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(YourNewModel, self).__init__()
        # 实现模型架构
    
    def forward(self, x):
        # 实现前向传播
        return output
```

2. 在 `segment_octa_image()` 函数中添加新模型支持

```python
def segment_octa_image(...):
    # ...
    model_type = model_type.lower()
    if model_type == 'yourmodel':
        model = YourNewModel()
    # ...
```

3. 前端不需要修改（model_type只是一个字符串参数）

---

### 任务2：修改分割结果后处理

**目标**：改变输出图像格式或应用后处理滤镜

**步骤**：

1. 找到 `models/unet.py` 中的 `postprocess_mask()` 函数

```python
def postprocess_mask(mask_tensor, original_size=None):
    # 这里修改处理逻辑
    mask = mask_tensor.squeeze().detach().cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    
    # 添加自定义后处理
    # 例如：应用高斯滤波
    from scipy.ndimage import gaussian_filter
    mask = gaussian_filter(mask, sigma=2)
    
    return mask
```

2. 重启后端，新的处理逻辑自动应用

---

### 任务3：添加上传限制

**目标**：限制上传文件大小或数量

**步骤**：

1. 在 `controller/image_controller.py` 的 `segment_octa()` 中添加检查

```python
@classmethod
async def segment_octa(cls, file: UploadFile, model_type: str):
    # 检查文件大小（例如：限制10MB）
    file_size = len(await file.read())
    await file.seek(0)  # 重置文件指针
    
    if file_size > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(
            status_code=400,
            detail="文件大小不能超过10MB"
        )
    
    # 继续原有逻辑
    # ...
```

---

### 任务4：添加用户认证

**目标**：为接口添加用户验证功能

**步骤**：

1. 在 `main.py` 中添加认证中间件或依赖注入

```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def verify_user(credentials = Depends(security)):
    # 验证用户Token
    if credentials.credentials != "valid_token":
        raise HTTPException(status_code=401, detail="认证失败")
    return credentials.credentials

@app.post("/segment-octa/")
async def segment_octa(
    file: UploadFile,
    model_type: str = Form("unet"),
    user = Depends(verify_user)  # 添加认证
):
    return await ImageController.segment_octa(file, model_type)
```

---

## 📊 数据库操作

### 查看历史记录

```bash
# 进入数据库
cd octa_backend
sqlite3 octa.db

# 查询所有记录
sqlite> SELECT * FROM images;

# 查询特定用户的分割
sqlite> SELECT * FROM images WHERE model_type='unet';

# 统计分割次数
sqlite> SELECT model_type, COUNT(*) as count FROM images GROUP BY model_type;

# 查看最新的分割
sqlite> SELECT * FROM images ORDER BY upload_time DESC LIMIT 5;
```

### 导出数据

```bash
# 导出为CSV
.mode csv
.output history.csv
SELECT * FROM images;
.quit
```

---

## 🐛 常见问题排查

### 问题1：跨域错误（CORS Error）

**错误信息**：`Access to XMLHttpRequest ... has been blocked by CORS policy`

**解决方案**：
1. 确认后端运行在 `127.0.0.1:8000`
2. 确认前端运行在 `127.0.0.1:5173`
3. 检查 `main.py` 中的 CORS 配置是否包含前端地址
4. 重启后端

### 问题2：模型分割失败

**错误信息**：`模型分割失败：模型可能未训练或加载失败`

**解决方案**：
1. 检查 `models/weights/unet_octa.pth` 是否存在
2. 如果不存在，使用随机初始化的模型（返回原图）
3. 放入正确的预训练权重文件

### 问题3：文件上传失败

**错误信息**：`仅支持PNG/JPG/JPEG格式的OCTA图像`

**解决方案**：
1. 确认上传的是PNG、JPG或JPEG格式
2. 检查文件的MIME类型是否正确
3. 使用支持的格式转换工具转换

### 问题4：数据库锁定

**错误信息**：`database is locked`

**解决方案**：
1. 确保只有一个后端实例在运行
2. 删除 `octa.db-wal` 和 `octa.db-shm` 文件
3. 重启后端

---

## 📈 性能优化建议

### 1. 缓存优化

在 `ImageController` 中添加缓存：

```python
from functools import lru_cache

@staticmethod
@lru_cache(maxsize=100)
def _get_cached_image_info(filename: str):
    # 缓存最近100个文件的信息
    return get_image_info(Path(filename))
```

### 2. 数据库连接池

当并发请求较多时，使用连接池：

```python
import sqlite3
from contextlib import contextmanager

class DBPool:
    def __init__(self, db_path, pool_size=5):
        self.db_path = db_path
        self.pool_size = pool_size
        # 实现连接池逻辑
```

### 3. 异步模型推理

对于大型模型，使用异步推理：

```python
import asyncio

async def async_segment(image_path):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, 
        segment_octa_image, 
        image_path
    )
    return result
```

---

## 📚 相关文档

| 文档 | 内容 |
|-----|------|
| [MODIFICATION_SUMMARY.md](./MODIFICATION_SUMMARY.md) | unet.py代码改进说明 |
| [CONTROLLER_REFACTOR_SUMMARY.md](./CONTROLLER_REFACTOR_SUMMARY.md) | 控制层重构详细说明 |
| [IMAGECONTROLLER_API_REFERENCE.md](./IMAGECONTROLLER_API_REFERENCE.md) | API接口完整参考 |
| [octa_backend/TROUBLESHOOTING.md](./octa_backend/TROUBLESHOOTING.md) | 后端故障排查指南 |

---

## 🎓 学习路径

### 初级：基本使用

1. 启动后端和前端
2. 上传PNG/JPG图像进行分割
3. 查看分割结果和历史记录

### 中级：代码理解

1. 阅读 `main.py`（路由层）- 10分钟
2. 阅读 `ImageController` 类的公开方法 - 30分钟
3. 了解数据流：上传 → 分割 → 保存 - 20分钟
4. 查看前端代码如何调用API - 15分钟

### 高级：扩展开发

1. 添加新的模型支持
2. 实现用户认证和权限控制
3. 优化性能（缓存、连接池、异步）
4. 部署到云服务（Azure、AWS等）

---

## 🚀 生产部署

### 基本步骤

1. **后端部署**

```bash
# 安装依赖
pip install -r requirements.txt

# 启动生产服务器（使用gunicorn）
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
```

2. **前端部署**

```bash
# 构建生产版本
npm run build

# 部署到Web服务器（nginx、Apache等）
```

3. **配置CORS**

修改 `main.py` 中的 `allow_origins`：

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # 生产域名
    # ...
)
```

---

**最后更新**：2026年1月13日  
**版本**：1.0  
**作者**：OCTA Web开发组

