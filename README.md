# OCTA图像分割Web平台设计与实现

## 1. 项目背景
OCTA（光学相干断层血管成像）用于无创观察视网膜血流情况，依赖分割算法识别血管结构。本平台提供端到端的Web服务，支持临床辅助诊断与科研数据处理，特点是：
- 浏览器即可完成上传、分割、预览与下载。
- 后端固定权重路径和CPU推理，适配无GPU环境。
- 失败容错：模型异常时返回原图，保障联调不中断。

## 2. 需求分析
核心功能：
- 上传：仅允许PNG，自动UUID重命名，校验格式与大小。
- 分割：调用后端模型（U-Net/FCN），输出8位灰度掩码。
- 历史记录：SQLite持久化上传时间、模型类型、原图与结果路径。
- 下载：结果PNG可直接下载/查看。

## 3. 技术选型
- 前端：Vue 3 + Element Plus + Axios（Vite构建）。
- 后端：FastAPI + SQLite，路径管理使用 pathlib，异常返回 HTTPException。
- AI模型：U-Net（主）、FCN（备选）；固定权重路径 ./models/weights/unet_octa.pth，CPU 推理。

## 4. 环境搭建步骤
### 4.1 后端（Windows）
```bash
cd octa_backend
..\octa_env\Scripts\activate
pip install -r requirements.txt
# 启动开发服务（默认 127.0.0.1:8000）
python main.py
# 或 uvicorn main:app --reload --port 8000
```

### 4.2 前端
```bash
cd octa_frontend
npm install
npm run dev  # 默认 http://127.0.0.1:5173
```

## 5. 核心功能实现
- 上传与校验：后端 validate_image_file() 仅接受 PNG，前端 el-upload 配合文件大小校验（10MB）。
- 分割流程：segment_octa_image() 预处理→推理→后处理，失败时回退原图路径以便联调。
- 路径管理：uploads/ 存原图，results/ 存掩码，静态访问 /images/{name} 与 /results/{name}。
- 历史记录：insert_record() 写入 SQLite，get_all_records()/get_record_by_id() 提供查询。
- 结果下载：前端生成临时 <a> 触发下载，后端 FileResponse 返回 PNG。

## 6. 功能演示
1) 打开前端 http://127.0.0.1:5173/ 。
2) 选择 PNG 文件（≤10MB），预览缩略图。
3) 选择模型（U-Net/FCN），点击“开始图像分割”。
4) 查看左右对比：左原图，右分割结果；可下载结果 PNG。
5) 在“历史记录”页查看分割记录并回溯文件。

## 7. 常见问题解决
- 跨域报错：确保后端 allow_origins 包含前端实际端口（默认 5173），修改后重启后端。
- 模型加载失败：确认 ./models/weights/unet_octa.pth 存在且可读；无权重时返回原图属预期。
- 文件路径错误：确保在 octa_backend 目录启动；uploads/ 与 results/ 会自动创建，如缺失手动创建。

## 8. GitHub 发布建议
- 发布前请先阅读仓库根目录清单：`GITHUB_PUBLISH_CHECKLIST.md`。
- 重点检查：虚拟环境、模型权重、运行产物与医学图像隐私数据，避免误上传。
- 若前端端口发生变化，发布说明中请同步标注后端 CORS 配置调整位置（`octa_backend/main.py`）。

---
如需演示脚本、技术细节与变更记录，可参阅 octa_frontend/HOMEVIEW_FILE_NAVIGATION.md 获取跳转指引。
