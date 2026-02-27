# ⚡ 前端训练失败 - 快速修复指南

## 🔴 问题症状

```
错误: Network Error / ERR_CONNECTION_RESET
原因: 后端没有启动或启动失败
```

---

## ✅ 解决步骤（3分钟搞定）

### 步骤1️⃣：检查虚拟环境

```bash
# Windows - 检查是否已激活虚拟环境
# 提示符前应该显示 (octa_env)

# 如果没有激活，运行：
cd octa_backend
..\octa_env\Scripts\activate
```

确认激活后，提示符应该变为：
```
(octa_env) D:\Code\OCTA_Web\octa_backend>
```

### 步骤2️⃣：安装缺失的依赖

```bash
pip install -r requirements.txt
```

特别是确保 albumentations 已安装：
```bash
pip install albumentations>=1.3.0
```

### 步骤3️⃣：启动后端服务

```bash
python main.py
```

或者运行诊断脚本先检查问题：
```bash
python quick_diagnose.py
```

---

## 🔍 诊断后端问题

### 方法1️⃣：运行快速诊断脚本

```bash
cd octa_backend
python quick_diagnose.py
```

脚本会检查：
- ✓ Python环境
- ✓ 所有依赖包
- ✓ 项目模块
- ✓ 后端启动

### 方法2️⃣：手动检查imports

```bash
python -c "from models.unet_underfitting_fix import UNetUnderfittingFix; print('✓ UNet OK')"
python -c "from models.loss_underfitting_fix import TripleHybridLoss; print('✓ Loss OK')"
python -c "from models.dataset_underfitting_fix import OCTADatasetWithAugmentation; print('✓ Dataset OK')"
```

### 方法3️⃣：查看后端日志

启动后端时，如果出错，会在控制台显示具体错误信息。

常见错误：
- `ModuleNotFoundError: No module named 'albumentations'` → 运行 `pip install albumentations>=1.3.0`
- `ImportError` → 运行 `pip install -r requirements.txt`
- `Port 8000 already in use` → 关闭其他占用8000端口的程序

---

## 🚀 验证后端正常运行

后端启动成功的标志：

```
[INFO] Service address: 127.0.0.1:8000
[INFO] Hot reload mode: Enabled (development)
[SUCCESS] File management table is ready
```

然后访问：
```
http://127.0.0.1:8000/docs
```

应该能看到Swagger API文档，包含 `/train/upload-dataset` 路由。

---

## 📋 完整启动流程

```bash
# 1. 进入后端目录
cd D:\Code\OCTA_Web\octa_backend

# 2. 激活虚拟环境（如果未激活）
..\octa_env\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 运行诊断（可选）
python quick_diagnose.py

# 5. 启动后端
python main.py

# 6. 在另一个终端启动前端
cd ../octa_frontend
npm run dev

# 7. 在浏览器访问
http://127.0.0.1:5173
```

---

## 💡 常见问题

### Q: 为什么会出现连接错误？
**A:** 后端没有在 http://127.0.0.1:8000 运行。检查：
1. 是否启动了 `python main.py`
2. 是否在 octa_backend 目录
3. 是否激活了虚拟环境

### Q: 如何检查后端是否运行？
**A:** 打开浏览器访问：
```
http://127.0.0.1:8000/
```
应该看到JSON响应。

### Q: 如何查看详细错误？
**A:** 后端启动时的控制台输出会显示错误信息。如果没有看到，在启动前加上日志：
```bash
python -u main.py 2>&1 | tee backend.log
```

### Q: albumentations 安装失败怎么办？
**A:** 如果 pip 速度慢，尝试：
```bash
pip install albumentations>=1.3.0 -i https://pypi.tsinghua.edu.cn/simple
```

---

## 🎯 一句话快速启动

```bash
cd octa_backend && ..\octa_env\Scripts\activate && pip install -r requirements.txt && python main.py
```

完成后，访问 http://127.0.0.1:5173 即可。

---

## 📞 如果仍然无法解决

1. **查看快速诊断脚本的输出** - 检查哪个模块导入失败
2. **手动运行Python导入检查** - 逐个检查依赖
3. **查看后端错误日志** - 控制台输出通常包含问题原因
4. **检查文件是否存在**：
   - `models/unet_underfitting_fix.py`
   - `models/loss_underfitting_fix.py`
   - `models/dataset_underfitting_fix.py`

---

**记住：** 后端必须在运行，前端才能发送请求！

