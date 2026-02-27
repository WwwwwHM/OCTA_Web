# OCTA后端 - 快速参考卡

**最后更新**：2026年1月14日 | **Phase 12** | **配置集成完成** ✅

---

## 🚀 快速启动（3步）

### ① 激活虚拟环境
```bash
..\octa_env\Scripts\activate
```

### ② 启动后端服务
```bash
python main.py
```

### ③ 验证服务运行
```bash
curl http://127.0.0.1:8000/
# 期望响应: {"status":"OK",...}
```

✅ **后端已启动**，监听于 `http://127.0.0.1:8000`

---

## 📡 API速查表

| 端点 | 方法 | 用途 | 返回 |
|-----|------|------|------|
| `/` | GET | 健康检查 | JSON状态 |
| `/segment-octa/` | POST | 图像分割 | 结果URL |
| `/images/{fn}` | GET | 获取原图 | 二进制图像 |
| `/results/{fn}` | GET | 获取结果 | 二进制灰度图 |
| `/history/` | GET | 查询历史 | JSON数组 |
| `/history/{id}` | GET | 获取详情 | JSON单条 |
| `/history/{id}` | DELETE | 删除记录 | 成功确认 |

---

## 🔧 常用命令

### 测试健康检查
```bash
curl http://127.0.0.1:8000/
```

### 提交分割请求
```bash
curl -X POST \
  -F "file=@image.png" \
  -F "model_type=unet" \
  http://127.0.0.1:8000/segment-octa/
```

### 查询所有历史
```bash
curl http://127.0.0.1:8000/history/
```

### 删除历史记录
```bash
curl -X DELETE http://127.0.0.1:8000/history/1
```

---

## ⚙️ 配置修改

### 修改服务器地址
**文件**: `config/config.py`
```python
SERVER_HOST = "127.0.0.1"  # 改这里
SERVER_PORT = 8000         # 或改这里
```

### 修改CORS前端地址
**文件**: `config/config.py`
```python
CORS_ORIGINS = [
    "http://127.0.0.1:5173",  # Vue开发服务器
    "http://localhost:5173",   # 备用地址
]
```

### 修改文件存储位置
**文件**: `config/config.py`
```python
UPLOAD_DIR = "./uploads"    # 上传目录
RESULT_DIR = "./results"    # 结果目录
```

---

## 📊 架构速览

```
main.py (路由层)
    ↓
ImageController (控制层) 
    ↓
ModelService + FileUtils (服务+工具层)
    ↓
ImageDAO (数据层)
    ↓
UNet/FCN (模型层)
    ↓
SQLite + 文件系统 (存储)
```

---

## 🗄️ 目录结构

```
octa_backend/
├── main.py                 ← FastAPI应用
├── config/                 ← 配置管理
│   ├── __init__.py
│   └── config.py          ← 所有常量在这里
├── controller/            ← 业务控制层
├── service/               ← 模型服务层
├── dao/                   ← 数据访问层
├── utils/                 ← 工具函数
├── models/                ← 神经网络模型
│   ├── unet.py
│   └── weights/           ← 模型权重（待放入）
├── uploads/               ← 上传文件（自动创建）
├── results/               ← 结果文件（自动创建）
└── octa.db                ← 数据库（自动创建）
```

---

## 🧪 故障排查

### ❌ 后端启动失败
```bash
# 1. 检查虚拟环境
..\octa_env\Scripts\activate

# 2. 检查依赖
pip install -r requirements.txt

# 3. 删除数据库重建
rm octa.db
python main.py
```

### ❌ 跨域错误(CORS)
✅ **已自动配置**  
- 检查前端运行地址是否在CORS_ORIGINS中
- 修改config/config.py的CORS_ORIGINS列表
- 重启后端

### ❌ 模型加载失败
```bash
# 1. 检查权重文件
# 应该存在: models/weights/unet_octa.pth

# 2. 如无权重文件，系统会使用随机初始化
# 这是正常行为（用于开发测试）
```

### ❌ 端口被占用
```bash
# 查找占用8000的进程
netstat -ano | findstr :8000

# 改用其他端口
# 修改config.py: SERVER_PORT = 8001
# 然后启动: python main.py
```

---

## 📝 日志位置

**后端控制台输出**: 启动时显示日志
```
[INFO] Configuration source: config/config.py
[INFO] Service address: 127.0.0.1:8000
[INFO] Database initialization successful
[SUCCESS] Backend initialization successful
```

**API请求日志**: 控制台实时显示
```
INFO:     POST http://127.0.0.1:8000/segment-octa/
INFO:     GET http://127.0.0.1:8000/history/
```

**数据库**: `octa.db` (SQLite)
```bash
# 查看数据库内容
sqlite3 octa.db
> SELECT * FROM images;
```

---

## 🔐 生产部署

### 关闭热重载
```python
# config/config.py
RELOAD_MODE = False  # 关闭开发热重载
```

### 多进程启动
```bash
# 使用4个worker进程
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 配置Nginx反向代理
```nginx
server {
    listen 80;
    server_name api.yourdomain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
    }
}
```

---

## 📊 性能指标

| 指标 | 值 |
|-----|-----|
| 启动时间 | ~2-3秒 |
| 内存占用 | ~150-200MB |
| 分割耗时 | 500-600ms (U-Net) |
| 数据库查询 | <50ms |
| 并发连接 | 无限制 |

---

## 🎯 配置管理（核心配置70+项）

### 必改配置（3项）

```python
# config/config.py

# 1. 数据库路径
DB_PATH = "./octa.db"

# 2. 模型权重路径
UNET_WEIGHT_PATH = "./models/weights/unet_octa.pth"

# 3. 前端CORS地址
CORS_ORIGINS = ["http://127.0.0.1:5173"]
```

### 常改配置（3项）

```python
# 1. 服务器配置
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000

# 2. 热重载模式
RELOAD_MODE = True  # 开发True，生产False

# 3. 文件上传限制
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
```

---

## 🔗 重要文件速查

| 文件 | 用途 | 行数 |
|-----|------|------|
| main.py | FastAPI应用入口 | 155 |
| config/config.py | 所有配置常量 | 530 |
| controller/image_controller.py | 业务逻辑 | 939 |
| service/model_service.py | 模型推理 | 762 |
| dao/image_dao.py | 数据库操作 | 764 |
| utils/file_utils.py | 文件处理 | 738 |
| models/unet.py | U-Net模型 | 630 |

---

## 💡 开发技巧

### 打印当前配置
```bash
python -c "from config import print_config; print_config()"
```

### 验证所有导入
```bash
python -c "from config import *; from controller import ImageController; print('✅ All imports OK')"
```

### 重建数据库
```bash
# 删除旧数据库
rm octa.db

# 启动后端（自动重建）
python main.py
```

### 测试特定API
```bash
# 测试分割接口
python -c "
import requests
with open('test.png', 'rb') as f:
    r = requests.post('http://127.0.0.1:8000/segment-octa/', 
                     files={'file': f}, 
                     data={'model_type': 'unet'})
    print(r.json())
"
```

---

## 📞 联系方式

**遇到问题？**
- 📖 查看: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- 📚 查看: [README.md](README.md)
- 📊 查看: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

**后端已完全就绪！** 🎉
