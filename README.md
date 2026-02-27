# OCTA_Web

OCTA（Optical Coherence Tomography Angiography）图像分割平台，采用前后端分离架构：

- 后端：FastAPI + PyTorch（模型注册表 + 通用推理接口）
- 前端：Vue 3 + Vite + Element Plus（模型选择、图像预览、推理与管理页面）
- 数据：SQLite（文件/记录元数据）

## 当前项目能力（与代码一致）

### 1) 模型即插即用
- 后端启动时自动扫描并导入 `octa_backend/models/**/model.py`
- 模型通过 `register_model(...)` 注册到统一注册表
- 当前模型目录示例：`unet`、`unetpp`、`attention_unet_transformer`
- 实际可用模型以接口 `GET /api/v1/seg/models` 返回为准

### 2) 通用分割推理接口
- 路由：`POST /api/v1/seg/predict`
- 入参：`model_name`（必填）、`image_file`（必填）、`weight_id`（可选）
- 支持图像格式：`png/jpg/jpeg/bmp/tiff/tif`
- 返回：`mask_base64`、`infer_time`、`device` 等结构化字段

### 3) 权重管理模块（代码已提供）
- 路由文件：`octa_backend/router/weight_router.py`
- 提供上传/列表/删除能力（`/api/v1/weight/*`）
- 支持 `.pth/.pt`，默认上限 200MB（见 `config.py`）

### 4) 前端页面
- 首页：模型选择、上传预览、推理与结果展示
- 历史记录页：记录查询与统计
- 文件管理页：文件筛选/预览/删除
- 权重管理页：权重上传、下载、删除

## 项目结构

```text
OCTA_Web/
├─ octa_backend/               # FastAPI 后端
│  ├─ main.py                  # 应用入口（自动扫描模型并注册路由）
│  ├─ router/                  # 接口路由（seg/weight）
│  ├─ service/                 # 注册表、业务服务
│  ├─ dao/                     # SQLite 数据访问
│  ├─ models/                  # 各模型目录（每个目录含 model.py）
│  └─ config/config.py         # 统一配置
├─ octa_frontend/              # Vue3 前端
│  ├─ src/views/               # Home/History/FileManager/WeightManager
│  └─ package.json
└─ README.md
```

## 快速启动（Windows）

### 1) 启动后端

```powershell
cd octa_backend
..\octa_env\Scripts\activate
pip install -r requirements.txt
python main.py
```

后端默认：`http://127.0.0.1:8000`

- Swagger：`http://127.0.0.1:8000/docs`
- 健康检查：`GET /`

### 2) 启动前端

```powershell
cd octa_frontend
npm install
npm run dev
```

前端默认：`http://127.0.0.1:5173`

## API 速览

### 已启用（main.py 已挂载）

- `GET /`：服务健康检查
- `GET /api/v1/seg/models`：获取已注册模型列表
- `POST /api/v1/seg/predict`：执行分割推理

### 可扩展模块（已有实现）

- `octa_backend/router/weight_router.py`：权重上传/列表/删除接口
- 需要时可在 `main.py` 中挂载对应 router

## 关键配置

主要配置集中在 `octa_backend/config/config.py`：

- `CORS_ORIGINS`：默认允许 `5173` 前端地址
- `WEIGHT_UPLOAD_ROOT`、`WEIGHT_MAX_SIZE`：权重存储与大小限制
- `DB_PATH`：SQLite 路径（默认 `./octa.db`）

## 常见问题

- 跨域失败：检查 `CORS_ORIGINS` 是否包含当前前端端口，修改后重启后端。
- 模型列表为空：确认模型目录存在 `model.py` 且导入时成功执行 `register_model`。
- 推理报错“模型未注册”：确认 `model_name` 与 `GET /api/v1/seg/models` 返回项一致。

## GitHub 发布建议

- 发布前先阅读：`GITHUB_PUBLISH_CHECKLIST.md`
- 避免提交：虚拟环境、模型大权重、运行产物与敏感数据
- 已提供仓库级 `.gitignore`，建议保持启用
