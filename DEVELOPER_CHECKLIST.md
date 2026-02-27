# ✅ 开发者检查清单

> 确保系统正常运行的完整检查清单

---

## 🔍 环境检查

### 1. 虚拟环境验证

- [ ] 虚拟环境已创建：`octa_env/`
- [ ] 虚拟环境已激活：提示符显示 `(octa_env)`
- [ ] 激活命令：
  ```bash
  cd octa_backend
  ..\octa_env\Scripts\activate  # Windows
  source ../octa_env/bin/activate  # Linux/Mac
  ```

### 2. Python 版本检查

```bash
python --version
# 预期：Python 3.11.5 或更高
```

- [ ] Python 版本 ≥ 3.11

### 3. 依赖安装检查

```bash
cd octa_backend
pip list | grep -E "fastapi|torch|pillow"
```

- [ ] fastapi ≥ 0.104.0
- [ ] torch ≥ 2.0.0
- [ ] pillow ≥ 10.0.0
- [ ] numpy ≥ 1.24.0

### 4. 完整依赖验证

```bash
cd octa_backend
pip install -r requirements.txt
```

- [ ] 所有依赖已安装，无错误

---

## 🏗️ 后端检查

### 1. 目录结构验证

```bash
cd octa_backend
# 验证关键目录存在
ls models/weights/
ls uploads/
ls results/
ls train_results/
```

- [ ] models/weights/ 目录存在
- [ ] uploads/ 目录存在
- [ ] results/ 目录存在
- [ ] train_results/ 目录存在

### 2. 模型权重检查

```bash
ls -lh models/weights/unet_octa.pth
```

- [ ] unet_octa.pth 文件存在
- [ ] 文件大小 > 100MB（完整权重文件）

### 3. 配置文件检查

```bash
cat config/config.py | grep "MODEL_DEVICE"
# 预期输出：MODEL_DEVICE = "cpu"
```

- [ ] MODEL_DEVICE = "cpu" ✅ （已修复）
- [ ] SERVER_HOST = "127.0.0.1"
- [ ] SERVER_PORT = 8000
- [ ] CORS_ORIGINS 包含 "http://127.0.0.1:5173"

### 4. 后端启动测试

```bash
cd octa_backend
python main.py
```

**预期输出：**
```
[INFO] Configuration source: config/config.py
[INFO] Service address: 127.0.0.1:8000
[INFO] Hot reload mode: Enabled (development)
[INFO] CORS allowed origins: 2 frontend addresses
======================================================================
INFO:     Uvicorn running on http://127.0.0.1:8000
```

- [ ] 启动无错误
- [ ] 运行在 http://127.0.0.1:8000
- [ ] 热重载已启用

### 5. API 健康检查

```bash
# 在后端运行的情况下，新开终端
curl http://127.0.0.1:8000/

# 或访问 Swagger UI
# http://127.0.0.1:8000/docs
```

- [ ] 根接口可访问
- [ ] 返回服务状态信息
- [ ] Swagger UI 可加载

### 6. 诊断工具验证

```bash
cd octa_backend
python diagnose.py
```

**预期输出：全部 ✓**

- [ ] PyTorch 版本检查 ✓
- [ ] 配置文件检查 ✓
- [ ] 目录结构检查 ✓
- [ ] 模型权重检查 ✓
- [ ] 设备创建测试 ✓
- [ ] 训练模块检查 ✓

---

## 🎨 前端检查

### 1. 项目文件检查

```bash
cd octa_frontend
ls -la
```

- [ ] package.json 存在
- [ ] src/ 目录存在
- [ ] node_modules/ 已安装（或准备安装）

### 2. 依赖安装检查

```bash
cd octa_frontend
npm install
```

- [ ] 无错误完成
- [ ] node_modules/ 文件夹已生成

### 3. npm 包版本检查

```bash
npm list vue element-plus echarts
```

- [ ] vue ≥ 3.3.0
- [ ] element-plus ≥ 2.13.1
- [ ] echarts ≥ 5.5.1
- [ ] axios ≥ 1.6.0

### 4. 前端启动测试

```bash
cd octa_frontend
npm run dev
```

**预期输出：**
```
➜  Local:   http://127.0.0.1:5173/
```

- [ ] 启动无错误
- [ ] 运行在 http://127.0.0.1:5173
- [ ] 本地访问可用

### 5. 页面导航检查

访问 http://127.0.0.1:5173，依次点击：

- [ ] "首页" → HomeView 加载成功
- [ ] "模型训练" → TrainView 加载成功
- [ ] "历史记录" → HistoryView 加载成功
- [ ] "关于" → AboutView 加载成功

### 6. 前端 UI 检查

在 HomeView 页面：
- [ ] 图像上传区域显示
- [ ] 参数配置表单显示
- [ ] "开始分割" 按钮显示

在 TrainView 页面：
- [ ] 数据集上传区域显示
- [ ] 参数配置表单显示
- [ ] "开始训练" 按钮显示
- [ ] ECharts 图表区域显示

### 7. 浏览器控制台检查

访问前端应用，按 F12 打开开发者工具 → Console 标签

- [ ] 无红色错误（Error）
- [ ] 无关键警告（Warning）
- [ ] Network 标签显示对后端的请求

---

## 🔗 集成检查

### 1. CORS 跨域检查

后端和前端都运行时：
1. 在 HomeView 选择图像
2. 点击"开始分割"
3. 查看浏览器 Network 标签

- [ ] POST /segment-octa/ 请求成功（状态 200）
- [ ] 返回 JSON 响应
- [ ] 无 CORS 错误

### 2. 训练功能检查

在 TrainView 页面：
1. 上传测试数据集
2. 配置参数
3. 点击"开始训练"
4. 查看进度显示

- [ ] 上传进度显示 0-100%
- [ ] 训练状态显示"训练进行中"
- [ ] 实时显示损失值
- [ ] 完成后显示指标

### 3. 数据库检查

```bash
cd octa_backend
sqlite3 octa.db ".tables"
```

- [ ] 数据库文件存在：octa.db
- [ ] 数据表已创建：images, trainings（或类似表名）

---

## 📊 性能检查

### 1. 推理性能测试

准备一张 256x256 的 PNG 图像，上传到分割页面：

```bash
# 测试耗时（在后端日志或前端查看）
# 应该在 100-500ms 完成
```

- [ ] 推理耗时 < 500ms（CPU）
- [ ] 推理耗时 < 100ms（GPU，可选）

### 2. 训练性能测试

使用 10 张图像测试数据集运行 1 轮训练：

```bash
# 估计耗时：30-60 秒（CPU）
```

- [ ] 1 轮训练耗时合理
- [ ] 内存使用正常（不超过 4GB）
- [ ] 无 OOM 错误

### 3. 前端响应检查

- [ ] 页面加载 < 2 秒
- [ ] 导航切换流畅
- [ ] ECharts 实时更新

---

## 🛠️ 工具检查

### 1. 诊断脚本验证

```bash
cd octa_backend
python diagnose.py
```

- [ ] 脚本运行无错误
- [ ] 所有检查项通过 ✓
- [ ] 输出信息清晰

### 2. 启动脚本验证（Windows）

```bash
cd octa_backend
start_server_cpu.bat
```

- [ ] 脚本运行
- [ ] 虚拟环境自动激活
- [ ] 后端自动启动
- [ ] 可选：按任意键关闭

### 3. 日志输出验证

后端运行时查看控制台输出：

- [ ] [INFO] 日志清晰易读
- [ ] [WARNING] 日志有适当警告
- [ ] [ERROR] 日志准确描述错误
- [ ] 无乱码或格式错误

---

## 📝 文档完整性检查

### 1. 根目录文档

```bash
ls *.md
```

- [ ] README.md 存在
- [ ] QUICK_REFERENCE.md 存在
- [ ] PROJECT_STATUS.md 存在
- [ ] TRAINING_STARTUP.md 存在
- [ ] TRAINING_FIX_REPORT.md 存在
- [ ] TRAINING_INTEGRATION_SUMMARY.md 存在
- [ ] TRAINING_COMPLETION_REPORT.md 存在

### 2. 后端文档

```bash
cd octa_backend && ls *.md
```

- [ ] TROUBLESHOOTING.md 存在
- [ ] START_GUIDE.md 存在
- [ ] README.md 存在

### 3. 前端文档

```bash
cd octa_frontend && ls *.md
```

- [ ] TRAIN_PAGE_GUIDE.md 存在
- [ ] README.md 存在

---

## 🚀 完整功能测试

### 1. 分割功能测试

```bash
# 1. 访问 http://127.0.0.1:5173/
# 2. 选择本地 PNG 图像
# 3. 点击"开始分割"
# 4. 等待结果
```

**验证清单：**
- [ ] 图像成功上传
- [ ] 分割请求发送
- [ ] 返回分割结果
- [ ] 前后对比显示正常

### 2. 训练功能测试

```bash
# 1. 访问 http://127.0.0.1:5173/train
# 2. 创建并上传测试数据集（test_data.zip）
# 3. 配置参数：轮数=5，学习率=0.001
# 4. 点击"开始训练"
```

**验证清单：**
- [ ] ZIP 文件成功上传
- [ ] 训练进度显示
- [ ] 实时损失值更新
- [ ] ECharts 曲线正确渲染
- [ ] 完成后显示最终指标
- [ ] 模型权重已保存

### 3. 历史记录测试

```bash
# 1. 访问 http://127.0.0.1:5173/history
# 2. 查看历史记录列表
# 3. 删除一条记录
```

**验证清单：**
- [ ] 历史记录加载成功
- [ ] 显示时间戳和操作类型
- [ ] 删除功能正常
- [ ] 列表更新及时

---

## ⚙️ 配置调整检查

### 1. 如果需要使用 GPU

```bash
# 编辑 octa_backend/config/config.py
# 第 107 行改为：
MODEL_DEVICE = "cuda"

# 重启后端
python main.py
```

- [ ] 后端成功启动
- [ ] 诊断显示设备为 CUDA

### 2. 如果需要修改端口

```bash
# 编辑 octa_backend/config/config.py
SERVER_PORT = 9000  # 改为其他端口

# 同时编辑 octa_frontend/src/main.js
# 修改 axios baseURL：
// axios.defaults.baseURL = 'http://127.0.0.1:9000'
```

- [ ] 后端运行在新端口
- [ ] 前端请求正确路由

---

## 🐛 故障排查快速表

| 症状 | 检查 | 解决 |
|------|------|------|
| 后端无法启动 | 虚拟环境激活 | `..\octa_env\Scripts\activate` |
| 找不到模块 | 依赖是否安装 | `pip install -r requirements.txt` |
| 前端连接失败 | 后端是否运行 | `curl http://127.0.0.1:8000` |
| CORS 错误 | CORS 配置 | 检查 config.py 的 CORS_ORIGINS |
| 训练失败 | 运行诊断脚本 | `python diagnose.py` |
| 内存溢出 | 批次大小 | 减少 batch_size |

---

## 📋 发布前最终检查

在部署到生产环境前，完成以下检查：

- [ ] 所有测试通过
- [ ] 没有控制台错误
- [ ] 没有警告信息
- [ ] 数据库已备份
- [ ] 配置文件已审查
- [ ] 文档已更新
- [ ] 权限已设置
- [ ] 日志已配置
- [ ] 备份策略已制定
- [ ] 回滚方案已准备

---

## ✅ 完成验证

打印此表格并逐一检查：

```
日期：_____________
检查人：___________
结果：[ ] 全部通过  [ ] 有问题（详见备注）

备注：
_____________________________________________________________
_____________________________________________________________
_____________________________________________________________
```

---

**最后更新：** 2026年1月16日  
**版本：** 1.0.0

现在可以开始使用了！✨
