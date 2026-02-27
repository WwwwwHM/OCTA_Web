# OCTA图像分割平台 - AI开发助手指南

## 项目概览

OCTA (Optical Coherence Tomography Angiography，光学相干断层血管成像) 图像分割平台：前后端分离的医学图像处理Web应用。

**架构：**
- `octa_backend/` - FastAPI后端，提供图像分割REST API
- `octa_frontend/` - Vue 3 + Vite前端，提供图像上传和结果展示UI
- `octa_env/` - Python虚拟环境（已配置，勿修改）

## 开发环境启动

### 后端启动（必须先启动）
```bash
# 方式1：使用启动脚本（推荐）
cd octa_backend
start_server.bat  # Windows双击或命令行运行

# 方式2：手动启动
..\octa_env\Scripts\activate  # 激活虚拟环境
cd octa_backend
python main.py  # 或 uvicorn main:app --reload
```
后端运行在 `http://127.0.0.1:8000`，Swagger文档：`/docs`

### 前端启动
```bash
cd octa_frontend
npm install  # 首次运行
npm run dev
```
前端运行在 `http://127.0.0.1:5173`

### 验证环境
```bash
cd octa_backend
python check_backend.py  # 必须在后端运行后执行
```

## 关键架构决策

### 1. 文件管理策略
- **UUID命名**：所有上传文件使用UUID重命名（`generate_unique_filename()`），避免文件名冲突
- **目录结构**：`uploads/`存原图，`results/`存分割结果，自动创建于启动时
- **文件URL映射**：`/images/{filename}` → `uploads/`，`/results/{filename}` → `results/`

### 2. 模型加载机制（重要）
- **固定权重路径**：模型权重固定加载自 `./models/weights/unet_octa.pth`（不再使用参数化路径）
- **CPU强制模式**：所有模型推理强制使用CPU（`torch.device('cpu')`），适配无GPU环境
- **文件存在校验**：权重文件不存在时打印详细提示并返回None，由调用者处理
- **容错返回原图**：模型加载或推理失败时返回原图路径，不中断前后端联调
- **8位灰度输出**：分割掩码输出为8位灰度图[0,255]，可直接保存PNG

### 3. CORS配置
- **预定义白名单**：仅允许`http://127.0.0.1:5173`和`localhost:5173`（见[main.py](../octa_backend/main.py#L39-L48)）
- **修改前端端口时**：必须同步更新`main.py`的`allow_origins`列表，否则跨域失败

## 代码规范

### 后端规范
```python
# 所有公开函数必须包含详细docstring（Google风格）
async def segment_octa(
    file: UploadFile = File(..., description="上传的PNG图像文件"),
    model_type: str = Form("unet", description="模型类型")
):
    """
    OCTA图像分割接口
    
    流程：
    1. 验证文件格式
    2. 保存上传文件
    3. 调用分割模型
    4. 返回结果URL
    
    Args:
        file: PNG格式图像
        model_type: 'unet' 或 'fcn'
    
    Returns:
        JSON响应包含result_url等字段
    """
```

- **异常处理**：使用`HTTPException`返回明确的HTTP状态码和中文错误信息
- **Path对象**：文件路径使用`pathlib.Path`，不用字符串拼接
- **类型提示**：所有函数参数和返回值必须添加类型注解
- **步骤注释**：使用 `# ==================== 步骤N: 功能说明 ====================` 标记关键步骤，每步都有详细中文注释

### 前端规范
- **Composition API**：使用`<script setup>`语法（Vue 3推荐）
- **Element Plus**：UI组件库已全局注册，直接使用`<el-*>`组件
- **Axios**：API请求使用axios（已安装依赖），baseURL设为`http://127.0.0.1:8000`

## 常见任务

### 添加新的分割模型
1. 在`models/`创建模型文件（参考[unet.py](../octa_backend/models/unet.py)结构）
2. 实现`segment_octa_image()`接口函数
3. 在`main.py`的`model_type`验证中添加新模型名称
4. 更新API文档的`model_type`描述

### 修改图像处理逻辑
- 核心分割函数：[unet.py](../octa_backend/models/unet.py#L495-L630) `segment_octa_image()`
- 预处理：256x256裁剪/填充，归一化到[0,1]
- 后处理：Sigmoid阈值0.5二值化，保存为PNG
- **输出格式强制**：8位灰度图（uint8，范围[0,255]）

### 调试跨域问题
1. 检查前端实际运行端口（终端输出）
2. 确认`main.py`的`allow_origins`包含该端口
3. 重启后端（CORS配置仅在启动时加载）
4. 浏览器开发者工具查看Network标签的预检请求（OPTIONS）

### 权重文件管理
1. 将预训练权重文件放入 `./models/weights/unet_octa.pth`
2. 无需修改代码，路径已固定
3. 权重文件不存在时，系统打印警告并使用随机初始化模型（用于开发测试）
4. 权重格式支持：`state_dict`、`model_state_dict`或直接张量

## 关键文件索引

- **API入口**：[octa_backend/main.py](../octa_backend/main.py) - 所有接口定义
- **模型实现**：[octa_backend/models/unet.py](../octa_backend/models/unet.py) - U-Net架构和分割函数
- **依赖清单**：[octa_backend/requirements.txt](../octa_backend/requirements.txt) - PyTorch + FastAPI
- **前端路由**：[octa_frontend/src/router/index.js](../octa_frontend/src/router/index.js)
- **故障排查**：[octa_backend/TROUBLESHOOTING.md](../octa_backend/TROUBLESHOOTING.md) - CORS、模型加载等问题
- **修改记录**：[MODIFICATION_SUMMARY.md](../MODIFICATION_SUMMARY.md) - 最新代码改进总结

## 注意事项

⚠️ **虚拟环境**：所有后端操作必须在`octa_env`激活后执行，命令行显示`(octa_env)`前缀  
⚠️ **端口占用**：默认8000端口，若被占用需修改启动命令的`--port`参数  
⚠️ **PNG限制**：后端仅接受PNG格式，JPEG等其他格式会被拒绝（见[main.py](../octa_backend/main.py#L95-L107) `validate_image_file()`）  
⚠️ **模型警告正常**：未提供权重文件时出现"权重文件不存在"警告是预期行为，不影响接口测试  
⚠️ **强制CPU模式**：所有推理在CPU上进行，即使权重是GPU训练的也能正确加载  
⚠️ **固定权重路径**：权重文件位置已固定为 `./models/weights/unet_octa.pth`，无需参数配置
